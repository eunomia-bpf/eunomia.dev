---
date: 2026-08-20
title: "Was the GPU Kernel Slow, or Did It Just Start Late?"
description: "GPU launch latency can hide host scheduling, runtime work, queueing, dependencies, and device delay. This report proposes causal timing and ground-truth tests."
tags:
  - Daily Report
  - GPU Profiling
  - CUDA
  - Observability
research_question: "How can a profiler distinguish host launch delay, command-buffer queueing, dependency wait, and actual GPU kernel execution without inventing causality from one timeline gap?"
source_cutoff: 2026-08-20
status: daily-report
---

# Was the GPU Kernel Slow, or Did It Just Start Late?

A CUDA kernel launch can look simple on a timeline: a CPU thread calls a launch API, some time passes, the kernel starts on the GPU, and later it finishes. But if the call begins at 10.000 ms and the kernel does not start until 12.000 ms, the missing two milliseconds are not one thing.

The CPU thread may have been descheduled before it reached the launch. The runtime or driver may have spent time inside the API. The launch may have been written to a command buffer but not yet submitted. It may be ordered behind earlier stream work or a graph dependency. The GPU may already be busy. The command may be eligible but still wait for device resources. Those cases imply different fixes, yet a profiler can easily collapse them into one number called **launch latency** or **queue time**.

Current NVIDIA tooling already exposes much more than a single number. CUPTI API activity records include start and end timestamps, process and thread identity, and a correlation ID that matches the associated kernel activity. Current kernel activity records can also expose `queued` and `submitted` timestamps; these latency timestamps are optional rather than collected by default. Nsight Systems separates API time, queue time, and kernel time, and its documentation is unusually explicit about an important limitation: the reported queue time is measured from the end of the CUDA API call to kernel start even though the actual enqueue happens somewhere inside that API call. A kernel can even start before the launch API returns, in which case that simplified queue interval disappears.

That means the basic instrumentation is already strong. The unresolved problem is narrower: **can a profiler turn these timestamps and identities into a defensible explanation of why a particular kernel started late?**

This is not primarily an eBPF problem. Linux scheduling traces, including eBPF-based ones, can contribute host-side evidence, but the central mechanism is a GPU profiling and causal-attribution problem that also exists on systems using other host tracing mechanisms.

## What current GPU traces already tell us

CUPTI provides a useful chain from host API to device work. A runtime or driver API invocation has start/end timestamps and a `correlationId`; the corresponding kernel record carries the same correlation identity. For CUDA Graph launches, kernel activity records also expose graph and graph-node identities. CUPTI external correlation can map a higher-level runtime or application ID into CUDA correlation IDs, although the external-correlation stack is maintained per CPU thread and has to be managed by the client.

The kernel-side record can expose four especially useful moments:

1. `queued`: the command was written into a CUDA command buffer;
2. `submitted`: the command buffer containing the launch was submitted to the GPU;
3. `start`: kernel execution began;
4. `end`: kernel execution ended.

Those timestamps already support better questions than "how long did the kernel take?" For example, a long interval before `queued` points toward host/runtime work, while a long `queued -> submitted` interval is different from a long `submitted -> start` interval.

But these are still **events**, not a complete state machine. `submitted` does not necessarily mean "this kernel is dependency-ready and only the GPU scheduler is delaying it." CUDA stream ordering, events, graphs, synchronization, resource availability, and newer mechanisms such as Programmatic Dependent Launch all affect when execution may legally begin. The CUDA Programming Guide also notes that Programmatic Dependent Launch only creates an opportunity for overlap; concurrent execution is opportunistic rather than guaranteed.

Nsight Systems therefore makes a sensible practical compromise. Its CUDA kernel reports divide time into API, queue, and kernel phases, while warning that queue time is an approximation and is not inherently bad. A busy GPU can have substantial queue time precisely because useful work is already occupying it.

The missing layer is an explanation that keeps these distinctions intact instead of converting every pre-start gap into a single cause.

## The diagnostic failure: intervals are not causes

Suppose a model-serving process launches the same 80-microsecond kernel thousands of times. After a deployment, end-to-end latency rises by 1.5 ms. A trace shows that the kernel execution duration is still about 80 microseconds, but its start moves later.

There are at least four materially different diagnoses:

- **Host readiness delay.** The application or framework did not reach the CUDA API promptly because the launching CPU thread was descheduled, blocked, or doing other work.
- **Runtime or command-buffer delay.** The CUDA API itself, or the path from API entry to command-buffer queue/submission, became expensive.
- **Dependency delay.** The kernel was correctly ordered behind stream, event, graph, or programmatic dependencies. Calling this scheduler delay would be wrong.
- **Device availability delay.** The launch was submitted and eligible, but earlier GPU work or resource pressure prevented immediate execution.

A fifth case is even more awkward: the trace does not contain enough information to distinguish dependency delay from device availability. A trustworthy profiler should say **unknown at this boundary** rather than picking the most plausible-looking colored segment on the timeline.

This distinction matters operationally. Pinning a CPU thread cannot fix an intentional stream dependency. Rewriting a CUDA Graph cannot fix a host thread that wakes 2 ms late. Kernel optimization does not help when kernel duration is already stable. Increasing GPU occupancy can actually increase queue time while improving throughput.

The profiler therefore needs to answer a different question: **what state transition is evidenced, what transition is only inferred, and which possible causes remain observationally equivalent?**

## Where current work is still weak

### 1. There is no common launch-state contract for interpreting timestamps

CUPTI exposes the pieces needed for a richer timeline, but applications and profilers still have to decide what each interval means. `queued -> submitted` has a concrete command-buffer interpretation. `submitted -> start` is more ambiguous because stream dependencies, graph dependencies, device work, and resource availability can overlap conceptually.

The missing capability is not another timestamp. It is a machine-readable contract that says which state transition a timestamp proves, which preconditions are known, and which are unknown. The practical test is an injected-delay benchmark where each delay source is independently controlled. If a profiler labels the right interval but the wrong cause, the contract is insufficient.

### 2. Correlation does not automatically preserve high-level causality across host execution

CUPTI correlation IDs connect CUDA APIs to device activities very well. External correlation can connect those records to another API domain, but the client owns the mapping and the external-correlation stack is CPU-thread scoped. Modern runtimes may prepare work on one thread, hand it off, batch it into a graph, and launch from another.

The missing capability is a stable **launch identity across host handoffs and graph transformations**, not just a local API-to-kernel join. A profiler should be able to tell whether the thread that called CUDA late was itself late, or whether the application intentionally handed the work to it late. The test is a workload that moves launch preparation and submission across thread pools while preserving known logical request IDs.

### 3. Existing queue summaries do not establish dependency readiness

Nsight Systems correctly documents that its reported queue time is an API-end-to-kernel-start approximation and that queueing itself is not necessarily a problem. CUPTI's optional `queued` and `submitted` timestamps sharpen the command-buffer boundary, but neither timestamp alone proves the exact instant at which every dependency became satisfied.

The missing evidence is an explicit readiness boundary—or, when one cannot be observed, an uncertainty marker. The test is to compare traces against workloads with known stream/event/graph dependencies and independent GPU saturation. A useful profiler must avoid blaming the GPU scheduler for time that was legally required by dependencies.

## Promising directions with academic and production value

### Direction 1: A launch-state ledger with explicit unknown states

**Gap.** Current tools expose timestamps but do not provide one portable interpretation of the causal states between application intent and kernel execution.

**Mechanism.** Record one append-only ledger per logical launch with the strongest available evidence:

```text
request_ready? -> api_enter -> queued? -> submitted? -> dependency_ready? -> kernel_start -> kernel_end
```

The question marks are part of the schema. A field can be observed, inferred under a named rule, or unknown. Each transition carries the source that established it: runtime hook, CUPTI API activity, CUPTI latency timestamp, graph dependency, OS scheduler trace, or application correlation.

The ledger should not invent a `dependency_ready` timestamp when the runtime cannot expose one. Instead, it can bound the delay: for example, `submitted <= ready <= start`. A diagnosis can then say "1.2 ms is between submission and execution, but dependency readiness is unknown" rather than "1.2 ms GPU scheduler delay."

**Delta.** This is not a replacement for CUPTI or Nsight Systems. It is an interpretation layer above their activity records that preserves missing evidence as a first-class result.

**Artifact.** A small open trace schema plus a reference CUPTI collector that enables latency timestamps selectively and emits launch ledgers. On Linux, host scheduling evidence could come from perf, ftrace, eBPF, or Nsight OS-runtime/context-switch tracing; the schema should not depend on one collector.

**Evaluation.** Measure phase-bound accuracy and cause-classification precision on controlled launch delays. Compare against ordinary API/queue/kernel summaries and a full-trace baseline. Report collection overhead and the fraction of launches that remain honestly unresolved.

**Academic value.** The general question is how to represent causal uncertainty when multiple schedulers and dependency systems share one asynchronous timeline.

**Production value.** Operators get a diagnosis that points to the host, runtime, dependency graph, or device boundary without asking them to inspect several tools manually.

**Failure condition.** If existing CUPTI latency timestamps plus current Nsight reports already identify injected causes with equivalent accuracy and lower overhead, the extra ledger is not justified.

### Direction 2: A cross-domain launch identity that survives handoff and graph batching

**Gap.** CUDA correlation is precise near the CUDA API boundary, but high-level work may move across CPU threads or be transformed into CUDA Graph nodes before the final launch.

**Mechanism.** Give each logical GPU launch a versioned `launch_epoch` created when the application or framework first declares the work. Propagate that identity through host handoffs, then bind it to CUPTI external correlation, CUDA correlation IDs, and graph/node IDs when those become available. Record one-to-many relationships rather than assuming one request maps to one kernel.

The important property is **lineage, not naming**. If a graph replay launches the same graph node many times, the profiler needs both stable node identity and per-launch epoch. If one request fans out across multiple streams, the lineage should preserve that fan-out instead of flattening it into a thread-local stack.

**Delta.** Existing external correlation provides the hook; the proposed work defines propagation semantics across thread pools, graph construction/replay, and batched launches.

**Artifact.** Adapters for one framework plus a standalone correlation library and trace validator. A practical first target could be a CUDA application with a host thread pool and graph replay, because ground truth can be generated without modifying the GPU driver.

**Evaluation.** Inject host handoffs, graph replay, and batching; measure lost joins, incorrect joins, storage cost, and diagnosis accuracy. Compare thread-local correlation, CUDA-only correlation, and the launch-epoch design.

**Academic value.** This tests whether a small lineage contract can make heterogeneous host/device traces compositional across asynchronous runtime boundaries.

**Production value.** Framework teams can connect an end-user request to the exact delayed launches even when submission occurs on infrastructure threads.

**Failure condition.** If CUDA/graph correlation already preserves the required lineage for realistic frameworks without additional propagation, the launch epoch is redundant.

### Direction 3: A ground-truth benchmark for launch-delay attribution

**Gap.** Profilers can show detailed timelines without proving that their diagnosis of a pre-kernel gap is correct.

**Mechanism.** Build a benchmark that injects one delay source at a time and records the known cause independently of the profiler:

- host descheduling before API entry;
- deliberate CPU work inside the launch path;
- command-buffer batching or submission delay;
- explicit stream/event dependencies;
- CUDA Graph dependencies and replay;
- GPU saturation from an independent stream or process;
- kernels with different resource footprints;
- Programmatic Dependent Launch cases where overlap is permitted but not guaranteed.

Then combine causes to test ambiguity. The benchmark should include both latency-sensitive small kernels and throughput-oriented workloads where queueing is healthy.

**Delta.** Existing profiler benchmarks often ask whether timestamps are cheap or accurate. This benchmark asks whether the **explanation chosen from those timestamps** is correct.

**Artifact.** Reproducible CUDA workloads, injected ground-truth labels, CUPTI/Nsight export adapters, and a scorer for interval error, cause accuracy, unresolved-case calibration, and overhead.

**Evaluation.** The strongest baseline is the richest current CUPTI/Nsight trace configuration. A useful new design must reduce false causal diagnoses, not merely add more events. An ablation can remove host scheduling, queued/submitted timestamps, dependency metadata, or launch lineage to quantify which evidence actually matters.

**Academic value.** The benchmark makes heterogeneous scheduling attribution falsifiable rather than a timeline-visualization judgment.

**Production value.** Profiler vendors and framework teams can test whether a new "launch latency" diagnosis will send users toward the right subsystem.

**Failure condition.** If simple API/queue/kernel decomposition already classifies all injected cases reliably, there is no need for a richer causal model.

## What this changes for profiler design

The useful output is not a finer rainbow timeline. It is a smaller set of statements with stronger semantics:

- the application became ready late;
- the CUDA API path was expensive;
- the launch spent measurable time before submission;
- execution was ordered behind a known dependency;
- the launch was submitted but readiness is unknown;
- the kernel itself executed slowly.

A profiler should prefer one of those statements—or an explicit unresolved interval—over a generic "GPU launch latency" warning.

This complements two recent Daily Reports rather than repeating them. [The sampling-bias report](https://eunomia.dev/research/profiler-sampling-bias/) asks whether the measurements themselves are statistically trustworthy. [The asynchronous eBPF causal-profiler report](https://eunomia.dev/research/async-ebpf-causal-profiler/) asks how logical work crosses CPU-side asynchronous handoffs. This report asks a different question at the GPU boundary: once the host/device events are visible, what evidence is needed to explain **why execution started when it did**? The [page-level memory-attribution report](https://eunomia.dev/research/page-level-ebpf-memory-attribution/) uses the same broader principle: observed activity and causal ownership should not be treated as interchangeable.

## What would change this conclusion?

Three results would narrow or overturn the case for a richer launch-attribution layer.

First, if current CUPTI latency timestamps, graph identities, and Nsight Systems host scheduling traces already classify the injected benchmark causes with high precision at acceptable overhead, then the missing piece is documentation or UI rather than a new trace contract.

Second, if dependency readiness cannot be observed or bounded usefully for modern CUDA workloads, a profiler should stop short of fine-grained device-delay labels. The correct product may simply expose command-buffer timing plus an explicit unresolved dependency/device interval.

Third, if collecting `queued` and `submitted` timestamps or cross-domain lineage perturbs latency-sensitive workloads enough to change their scheduling behavior, the design should become hierarchical: collect cheap API/kernel correlation continuously and enable richer evidence only for selected launches or short diagnostic epochs.

The current evidence supports a cautious conclusion: GPU profilers already have many of the timestamps required to analyze launch delay, but **timestamps are not yet the same thing as a causal diagnosis**. The next useful step is to make that distinction testable.

## References

- NVIDIA, CUPTI Activity API, API activity records and kernel activity fields: <https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html>
- NVIDIA, `CUpti_ActivityAPI`: <https://docs.nvidia.com/cupti/api/structCUpti__ActivityAPI.html>
- NVIDIA, CUPTI usage guide, external correlation: <https://docs.nvidia.com/cupti/main/main.html>
- NVIDIA, CUPTI `CUpti_ActivityKernel9`, including optional `queued` and `submitted` latency timestamps: <https://docs.nvidia.com/cupti/13.0.0/api/structCUpti__ActivityKernel9.html>
- NVIDIA, Nsight Systems Post-Collection Analysis Guide, CUDA kernel launch/queue reports: <https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html>
- NVIDIA, CUDA Programming Guide, Programmatic Dependent Launch and Synchronization: <https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html>
