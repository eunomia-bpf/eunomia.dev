---
date: 2026-09-03
title: "When Should an eBPF Profiler Stop Trusting Its Own Diagnosis?"
description: "eBPF profilers can lose events, probes, or semantics while still emitting plausible diagnoses. This report develops evidence-aware abstention and adaptive collection."
tags:
  - Daily Report
  - eBPF
  - Observability
  - Profiling
  - Reliability
research_question: "How should an eBPF profiler detect when evidence loss or semantic drift makes a diagnosis unsupported, and decide whether to escalate collection or abstain?"
source_cutoff: 2026-09-03
status: daily-report
---

# When Should an eBPF Profiler Stop Trusting Its Own Diagnosis?

Imagine an always-on eBPF profiler watching a busy service. It sees scheduler delay, socket activity, a few application probes, and enough request context to report a likely root cause. During the incident, however, one event buffer saturates, a protocol parser stops seeing fields beyond its capture limit, and one optional probe fails to attach after a software update. The profiler can still produce a complete-looking explanation. The hard question is whether that explanation is still supported by the evidence it actually observed.

That problem is different from asking whether telemetry was dropped. A collector can know that 2% of records were lost and still have no idea whether the missing 2% contains the one transition needed to distinguish CPU contention from lock contention, or a successful response from a synthetic error.

<!-- more -->

Recent production evidence makes this distinction concrete. In August 2026, an [OpenTelemetry eBPF Instrumentation issue](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/issues/3067) showed a client span being force-finished with a synthesized HTTP 499 even though the call returned 200, because the instrumentation had not observed the response before `tcp_close`. Another [August issue](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/issues/2958) documents a 16 KiB BPF ABI and userspace-decoder limit that drops otherwise valid Go Auto SDK spans whose payload exceeds the bound. Earlier reports describe [trace-context breaks when `traceparent` lies beyond the eBPF capture buffer](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/issues/1381) and [GenAI traces disappearing when a configured capture buffer is smaller than the request](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/issues/2174).

These are not four independent studies. They come from one project and should be treated as one production evidence cluster. But they show several distinct ways an eBPF observer can remain alive while its evidence becomes incomplete: transport loss, bounded capture, parser limits, unsupported paths, and inference from missing events.

The Linux interfaces expose some of this loss. The [BPF ring-buffer documentation](https://docs.kernel.org/bpf/ringbuf.html) states that reservation fails when there is insufficient space and can also fail in NMI context when the reservation lock cannot be acquired. `bpf_ringbuf_query()` exposes useful snapshots, but the kernel documentation explicitly frames them as momentary values for debugging, reporting, and heuristics rather than stable truth. With perf buffers, [libbpf exposes a `lost_cb`](https://libbpf.readthedocs.io/en/latest/api.html) so userspace can learn that records were lost.

Those mechanisms answer a local question: did this transport path lose records? They do not answer the diagnostic question: is the conclusion I am about to report still justified?

This report argues that an eBPF profiler needs a **diagnosis validity contract** between the evidence it promises to observe and the conclusions it is allowed to emit. The runtime should track evidence deficits by diagnostic obligation, not only by global loss rate. When a required signal becomes unavailable, the profiler should either collect more evidence within a bounded budget or downgrade the conclusion to `unknown` instead of manufacturing certainty.

This is a deliberate continuation of the earlier report on [diagnostic-preserving eBPF telemetry compression](https://eunomia.dev/research/ebpf-diagnostic-telemetry-compression/). That report asks what a compact representation must preserve before high-rate data leaves the collector. Here the question starts later: **what should the collector do when the representation it planned to rely on is no longer being observed reliably?**

## A loss counter is not a diagnosis-validity test

Suppose a profiler supports three explanations for a slow request:

1. the task spent most of its delay runnable but unscheduled;
2. it slept on a futex or another synchronization primitive;
3. it waited for network or storage completion.

A collector might sample CPU stacks, trace scheduler transitions, trace selected futex paths, and correlate socket or I/O completion events. Now assume it loses 1% of scheduler records.

If the missing records are close to uniformly distributed, the profiler may still estimate runnable delay accurately enough. But if losses cluster during short bursts of scheduler activity, that same 1% can erase the transition separating runnable time from sleep. A global coverage value of 99% does not tell the investigator which case occurred.

The same problem appears when a probe is absent rather than overloaded. If a kernel version moves one code path away from the attached function, every record that still arrives can be perfectly delivered while the profiler systematically misses one class of events. Transport coverage is 100% for the surviving hook and semantic coverage is still incomplete.

Bounded payload inspection creates a third failure mode. A parser may receive every event but only the first N bytes of a protocol message. If the field that carries request identity, trace context, or status appears after N, the profiler does not have an unbiased sample of complete records. It has a workload-dependent blind spot.

The important unit is therefore not "percent of all telemetry retained." It is **whether each conclusion still has the evidence obligations required to distinguish it from plausible alternatives**.

## Current systems already adapt collection, but usually for a different objective

Adaptive observability is not new. [ViperProbe](https://ieeexplore.ieee.org/document/9335808/) introduced an eBPF-based microservice collection framework with dynamic sampling and workload-informed metric selection. More recently, [SysOM-AI](https://arxiv.org/abs/2603.29235) reports continuous cross-layer diagnosis for large AI training deployments using eBPF-based tracing together with CPU, GPU, and NCCL evidence; its abstract reports less than 0.4% overhead, deployment across more than 80,000 GPUs, and 94 confirmed production issues.

These systems demonstrate that collection policy can be selective and still useful. They also make the next question sharper. Adaptation is commonly driven by overhead, workload phase, anomaly detection, or a predefined diagnosis workflow. A profiler still needs to know whether the evidence available *right now* supports the particular diagnosis it is about to make.

That is closer to a runtime safety property than a sampling heuristic.

A useful design should separate three states:

- **supported:** all evidence obligations for this conclusion are currently satisfied within declared error bounds;
- **degraded:** some evidence is incomplete, but a bounded fallback can still distinguish the leading explanations;
- **unavailable:** the profiler cannot distinguish the leading explanations with the evidence it retained.

These states should not be a single opaque probability. If the collector says `degraded`, an operator should be able to inspect which obligation failed and what recovery action is being attempted.

## What must be tracked for an eBPF diagnosis

Consider a diagnosis rule that attributes request delay to scheduler queueing. Its evidence obligations might include:

```text
obligation scheduler_residency:
    hooks = sched_switch, sched_wakeup
    identity = task_generation
    required_coverage = complete transition pairs
    transport = ringbuf_generation_17

obligation blocking_alternative:
    hooks = futex_wait, selected_io_completion
    identity = task_generation
    required_coverage = at least one distinguishing edge

conclusion runqueue_delay:
    requires = scheduler_residency + blocking_alternative
```

The profiler does not need to export this exact syntax. It does need equivalent state somewhere in the runtime.

For each obligation, the collector can record:

- which BPF program and attach generation supplied the evidence;
- whether the expected hook is still attached and producing compatible records;
- eligible events versus successfully recorded events when that quantity is measurable;
- ring-buffer reservation failures, perf-buffer loss, map insert failures, and relevant evictions;
- schema or resource-semantics generation;
- collector restart epoch and any gap between epochs;
- bounded-capture or parser conditions that make a record semantically incomplete;
- which alternative explanations depend on the missing signal.

The last item is what ordinary telemetry health usually lacks. A lost futex event and a lost DNS event should not have the same effect on every diagnosis. Their importance depends on the question being answered.

## Why global confidence scores are a weak abstraction

It is tempting to combine all of these signals into one number such as `confidence=0.83`. That looks convenient for a dashboard and is difficult to interpret correctly.

A scalar hides why confidence changed. It also encourages invalid arithmetic across qualitatively different failures. Ten percent random sampling loss, one completely missing hook, a stale application schema, and a parser truncation boundary do not combine naturally into one calibrated probability.

A better first design is categorical and inspectable. Keep the obligation state, the failure mode, and the allowed conclusion explicit. Probabilities can be added later for evidence sources that have a real statistical model.

This matters because some losses are adversarial to the diagnosis even when nobody is attacking the system. Bursts fill buffers exactly when event rate changes. Bounded maps evict keys when cardinality rises. Short-lived processes disappear before periodic discovery attaches. Parser limits fail on large messages rather than a random sample. These are structured missing-data mechanisms.

## Where current work is still weak

### 1. Loss accounting is usually transport-local, not question-local

Linux perf buffers can report lost records, and a ring-buffer producer can count failed reservations. That is already useful operational evidence.

What is missing is the mapping from those failures to the conclusions they invalidate. A scheduler-loss counter does not say whether CPU utilization, run-queue residency, off-CPU attribution, or request causality remains usable. A global alert such as "1.2% events dropped" forces every downstream query to rediscover that relationship.

The missing artifact is a machine-readable dependency between evidence obligations and supported diagnoses. A discriminating experiment would inject loss into one event family at a time and measure whether the profiler disables only the conclusions that actually become ambiguous.

### 2. Probe presence is not the same as semantic coverage

An attached eBPF program can keep running while a new kernel path, application version, protocol shape, or compiler transformation moves the important state outside the observed boundary. The August 2026 OBI issues show that valid activity can exceed a fixed capture or transport bound without crashing the entire collector.

Current health checks often verify that the agent is alive, programs are loaded, and records are arriving. The missing capability is an explicit **coverage contract** that can fail even when those liveness checks pass.

A useful test should mutate kernel versions, application builds, message sizes, and code paths while preserving the same user-visible workload. The profiler should detect when the semantic preconditions behind one diagnosis no longer hold.

### 3. Adaptive collection rarely has an explicit recovery objective

Dynamic sampling and selective tracing can reduce overhead. An anomaly can also trigger more detailed observation. But "collect more" is underspecified.

If the current ambiguity is scheduler versus futex blocking, enabling extra filesystem probes wastes budget. If a stale schema invalidates resource identity, increasing sample rate does not repair the schema. If the decisive event already occurred, post-trigger tracing may be too late unless the collector kept a bounded pre-trigger exemplar.

The missing mechanism is recovery targeted at the failed evidence obligation. Evaluation should compare obligation-targeted escalation with generic high-fidelity mode under the same CPU, memory, and export budget.

### 4. Profilers need a principled abstention path

Observability tools are rewarded for producing answers. A missing answer looks less useful than a likely root cause, so implementations tend to continue with whatever evidence survives.

For automated remediation and AI-assisted operations, this is risky. An unsupported diagnosis can trigger a concrete action such as moving a workload, changing a limit, restarting a service, or rolling back code.

The missing property is explicit abstention: when the leading explanations cannot be distinguished, `unknown because sched_switch coverage was incomplete during generation 17` is a better systems output than a precise but unsupported root cause.

## Promising directions with academic and production value

### 1. Compile diagnosis obligations into an evidence-deficit ledger

**Gap.** Existing collectors can report low-level loss and health but do not connect it to the diagnostic questions that depend on each signal.

**Mechanism.** Extend the diagnostic contract from the earlier telemetry-compression report. For every supported conclusion, compile a small dependency graph from conclusions to evidence obligations. Each obligation carries hook generation, identity generation, loss accounting, schema version, and acceptable approximation mode. The userspace runtime maintains a ledger with states such as `supported`, `degraded`, and `unavailable`.

A BPF program does not need to evaluate the whole graph in kernel context. It only updates cheap local facts: successful observation counts, failed reservations, map pressure, generation IDs, and invariant violations. Userspace joins those facts with attach state and semantic metadata.

**Delta.** Loss callbacks and collector-health metrics expose transport state. Diagnostic contracts describe what a compact representation should preserve. The new mechanism connects those layers so a specific query can know whether its prerequisites still hold.

**Artifact.** A small contract language, libbpf-side accounting helpers, a userspace evidence ledger, and adapters for several existing eBPF profilers or tracing examples.

**Evaluation.** Use scheduler, networking, memory, and application-resource incidents with known root causes. Inject ring-buffer pressure, perf-buffer loss, map eviction, missing hooks, collector restarts, and schema mismatch. Compare a profiler with no health metadata, one with global loss counters, and one with the obligation ledger. Measure root-cause accuracy, false confident diagnoses, correct abstention, diagnosis availability, and overhead.

**Academic value.** The research question is whether observability completeness can be defined relative to a diagnosis rather than relative to a transport stream.

**Production value.** Operators and automated systems can tell which conclusions remain safe to use during partial telemetry failure instead of disabling the whole profiler or trusting every surviving metric.

**Failure condition.** If global loss and attach-health thresholds predict diagnosis failures just as well across diverse incidents, the dependency graph is unnecessary complexity.

### 2. Recover evidence by obligation, not by switching everything to verbose mode

**Gap.** A degraded collector often has only two modes: continue normally or enable much more tracing.

**Mechanism.** When an obligation becomes degraded, a userspace controller selects a bounded recovery action specific to that deficit. Possible actions include lowering a sampling divisor for one event family, enabling a fallback tracepoint, increasing raw exemplars for one entity generation, refreshing a resource-semantics manifest, or temporarily reserving more export budget for the affected stream.

The controller needs hysteresis and a hard budget. It should also know when recovery is impossible. If the decisive event has already passed and no exemplar exists, the correct transition is `unavailable`, not indefinite escalation.

**Delta.** ViperProbe and other adaptive collectors demonstrate dynamic collection. This direction makes the trigger and target query-specific: collection changes because a declared evidence obligation failed, and stops when that obligation is restored or declared unrecoverable.

**Artifact.** A userspace policy controller plus a small library of fallback probe plans. The same contract could drive libbpf skeleton variants or userspace eBPF runtimes such as [bpftime](https://eunomia.dev/bpftime/) when late dynamic attachment is useful.

**Evaluation.** Replay incidents while varying event rate and blind-spot type. Compare static low-overhead collection, always-high-fidelity tracing, anomaly-triggered verbose mode, and obligation-targeted recovery under equal CPU, map-memory, and export budgets. Measure time to restore a supported diagnosis, unnecessary probe work, missed root causes, and workload perturbation.

**Academic value.** This asks whether observability control can be formulated as a constrained recovery problem over missing evidence rather than a generic fidelity knob.

**Production value.** A fleet profiler can spend extra overhead only where current evidence is insufficient, preserving always-on operation while avoiding fleet-wide trace explosions.

**Failure condition.** If targeted recovery routinely costs as much as high-fidelity mode or reacts too late to preserve decisive evidence, the simpler static modes should win.

### 3. Build a counterexample benchmark for confident-but-wrong observability

**Gap.** Profilers are usually evaluated on overhead and on whether they diagnose known incidents when telemetry works as intended. That misses the failure mode where the profiler returns a plausible answer under partial evidence.

**Mechanism.** Build paired workloads with the same visible symptom but different true causes, then perturb the observer rather than only the application. Each case includes ground truth for the missing event or state transition. Faults should cover:

- ring-buffer reservation failure and perf-buffer loss;
- short process lifetimes and missed attachments;
- hook movement across kernel or application versions;
- bounded-map eviction and identity reuse;
- payloads beyond parser or transport capture limits;
- collector restart gaps;
- stale semantic manifests;
- non-random burst loss around the actual root-cause transition.

The benchmark scores not only correct root cause, but also whether the profiler knew when it lacked enough evidence to choose.

**Delta.** Existing observability benchmarks commonly compare runtime cost, trace volume, or diagnosis success. This benchmark makes **false confidence under observer degradation** the primary failure.

**Artifact.** Reproducible Linux workloads, fault injectors, ground-truth traces, and an evaluator for diagnosis accuracy, abstention calibration, recovery time, and resource cost.

**Evaluation.** Compare full tracing, fixed sampling, hand-written eBPF aggregation, loss-counter-aware diagnosis, and obligation-aware adaptive collection. Include an ablation without query-to-evidence dependencies to test whether the extra semantics matter.

**Academic value.** The benchmark turns a vague trust question into a measurable systems property: can a profiler distinguish an incorrect diagnosis from an unanswerable one under structured evidence loss?

**Production value.** Vendors and internal observability teams can regression-test upgrades, new kernels, and collection-budget changes against confident-but-wrong failure modes before fleet rollout.

**Failure condition.** If real incidents rarely change diagnosis under the injected evidence failures, then explicit validity control may not justify its operational complexity.

## The operational rule should be simple

An eBPF profiler should not ask only, "Did I receive enough telemetry?" It should ask, "Do I still have the evidence needed to distinguish this conclusion from the alternatives I claim to rule out?"

That rule changes several implementation choices. Loss metadata becomes part of query semantics. Attach and schema generations become diagnostic state rather than only deployment metadata. Adaptive collection gets a concrete recovery target. `Unknown` becomes a valid answer when the evidence contract is broken.

The point is not to make every profiler formally prove every diagnosis. The first useful system can be modest: declare a few high-value diagnoses, list their required evidence, track known deficits, and refuse to overstate what the surviving trace proves.

## What would change this conclusion?

This design assumes that structured evidence loss is common enough to alter real diagnoses and that low-level collector health is too coarse to predict those failures. If broad production traces show that simple global loss thresholds, attach-health checks, and static safety margins already identify nearly every unreliable diagnosis, the extra obligation graph is not worth maintaining.

The argument would also weaken if obligation-targeted recovery reacts too slowly. Many incidents have decisive transitions before the visible symptom. If bounded exemplars and fallback probes cannot preserve or recover those transitions, adaptive control should abstain more often rather than promise late repair.

The strongest evidence for this design would be a benchmark or deployment in which two profilers see the same degraded telemetry: both retain similar overhead and data volume, but the obligation-aware profiler produces fewer confident wrong diagnoses while remaining available on cases whose required evidence is still intact. If that result does not hold across scheduler, network, memory, and application-semantic incidents, diagnosis validity should remain a simpler collector-health problem.

## References

- Linux kernel documentation, [BPF ring buffer](https://docs.kernel.org/bpf/ringbuf.html).
- libbpf documentation, [`perf_buffer__new()` and record-loss callback](https://libbpf.readthedocs.io/en/latest/api.html).
- OpenTelemetry eBPF Instrumentation, [valid Go Auto SDK spans larger than 16 KiB can cross the current BPF transport bound](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/issues/2958), August 7, 2026.
- OpenTelemetry eBPF Instrumentation, [missing response observation can synthesize HTTP 499 for a request that returned 200](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/issues/3067), August 17, 2026.
- OpenTelemetry eBPF Instrumentation, [`traceparent` beyond the capture buffer can break trace context](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/issues/1381), February 28, 2026.
- M. H. Abbasi et al., [ViperProbe: Rethinking Microservice Observability with eBPF](https://ieeexplore.ieee.org/document/9335808/), IEEE CloudNet 2020.
- Yusheng Zheng et al., [SysOM-AI: Continuous Cross-Layer Performance Diagnosis for Production AI Training](https://arxiv.org/abs/2603.29235), 2026.
