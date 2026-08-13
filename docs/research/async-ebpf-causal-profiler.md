---
date: 2026-08-13
title: "What Does an eBPF Profiler Need to Follow Asynchronous Work?"
description: "eBPF stack sampling finds where CPU time runs, but async handoffs break request lineage. We examine causal edges, sampling bias, and practical profiler design."
tags:
  - Daily Report
  - eBPF
  - Profiling
  - io_uring
  - Async Runtimes
research_question: "What additional runtime evidence does an eBPF profiler need to reconstruct causality across scheduler wakeups, work queues, io_uring, userspace async tasks, and application-defined resources without turning continuous profiling into full tracing?"
source_cutoff: 2026-08-13
status: daily-report
---

# What Does an eBPF Profiler Need to Follow Asynchronous Work?

A CPU profiler answers a familiar question: where was the machine spending time when it sampled execution? That question is useful, but it becomes incomplete once a request stops executing on the thread that received it.

Consider one request that parses input on thread A, submits an `io_uring` operation, wakes a runtime task, queues kernel work, resumes on thread B, and finally waits on an application-managed buffer pool. A conventional stack sample can show each active stack. It does not automatically say that these pieces belong to the same request, nor which wait caused which later execution. If the expensive work runs on a generic worker, the most visible stack may identify the worker implementation rather than the operation that created the work.

<!-- more -->

This is not evidence that stack sampling is obsolete. The [OpenTelemetry eBPF Profiler](https://github.com/open-telemetry/opentelemetry-ebpf-profiler) shows how much whole-system, cross-language stack information eBPF can collect non-intrusively at datacenter scale. OpenTelemetry's [Profiles specification](https://opentelemetry.io/docs/specs/otel/profiles/) also provides a clean model for sampled stacks and allows a sample to carry a trace/span link when such context is available.

The missing piece is different: **an asynchronous profiler needs a bounded causal handoff graph in addition to sampled stacks**. It should record selected enqueue, submit, wake, start, complete, and resume edges using identifiers that survive a change of thread. The profiler can then attach ordinary samples to the nearest known execution identity and propagate attribution across those edges. Where the kernel cannot see the semantic identity, the design needs a small runtime join contract rather than pretending an OS profiler can infer every Future, goroutine, request, or application resource from system calls alone.

The goal is deliberately smaller than universal tracing. Full traces can already preserve request context when an application is instrumented correctly. The research question is whether an eBPF profiler can recover enough causal structure for performance diagnosis while retaining the low, tunable overhead and deployment advantages that make continuous profiling attractive.

This report continues the [eBPF runtime contract](https://eunomia.dev/research/userspace-ebpf-runtime-contract/), [hook-composition contract](https://eunomia.dev/research/ebpf-hook-composition-contract/), and [stateful-upgrade report](https://eunomia.dev/research/stateful-ebpf-transactional-upgrade/). Those reports ask how eBPF extensions should attach, compose, and evolve. Here the same runtime view is applied to profiling: observations need identities and lifetime rules before they can be composed into a causal explanation.

## Stack sampling is strong at location and weak at handoff identity

Sampling is attractive because it is sparse. Instead of recording every call or every event, the profiler periodically observes a running context and estimates where execution time accumulates. OpenTelemetry defines a profile as stack traces plus associated values, and its eBPF profiler aims for continuous, whole-system collection without injecting agents into target processes.

That sparsity is also the boundary. A sample says what was active at one observation point. It does not contain the history that explains why that work became runnable.

The distinction becomes obvious in async runtimes. Tokio's [`spawn`](https://docs.rs/tokio/latest/tokio/task/fn.spawn.html) documentation states that a spawned task may execute on the current thread or may be sent to another thread. The `tracing` crate therefore provides [`Instrument`](https://docs.rs/tracing/latest/tracing/trait.Instrument.html), which attaches a span to a `Future` and enters that span every time the Future is polled. The explicit propagation mechanism is a useful clue: thread-local execution context is not a stable task identity once work can move.

Go exposes the same semantic boundary in another form. [`runtime/trace`](https://pkg.go.dev/runtime/trace) defines tasks as logical operations that may involve multiple goroutines and propagates task identity through `context.Context`. The runtime can preserve this relation because it knows about goroutines and task annotations. A generic kernel sampler does not.

So an async profiler must separate at least three concepts:

1. **Execution location**, represented by stacks, processes, threads, CPUs, and timestamps.
2. **Causal handoff**, represented by an event that transfers responsibility from one execution context to another.
3. **Semantic ownership**, represented by a request, task, future, resource, or operation that may not have a kernel-visible identifier.

Trying to encode all three in a stack alone loses information. Recording every event to recover the missing information defeats the point of profiling.

## Linux already exposes useful causal edges

The encouraging result is that several important handoffs already have stable-enough correlation fields at the kernel boundary. The profiler does not need to infer every transition from timing coincidence.

### Work queues expose the queued work object at both ends

Linux's [`workqueue` tracepoints](https://github.com/torvalds/linux/blob/master/include/trace/events/workqueue.h) expose `workqueue_queue_work` when work is queued and `workqueue_execute_start` / `workqueue_execute_end` around execution. The tracepoints carry the `work_struct` pointer and function pointer.

That gives a direct correlation candidate:

```text
producer thread
    |
    | queue(work_struct = W)
    v
work item W
    |
    | execute_start(W)
    v
worker thread + sampled stack
```

A stack-only profiler may charge the later CPU time to a generic worker. An edge-aware profiler can instead connect the worker execution to the producer that queued `W`.

The pointer is not a permanent globally unique ID. Kernel objects are reused. A practical collector therefore needs a generation or lifetime guard, for example queue sequence plus time-bounded state, rather than retaining pointer identity forever.

### io_uring exposes request, ring, and user correlation fields

Linux's [`io_uring` tracepoints](https://github.com/torvalds/linux/blob/master/include/trace/events/io_uring.h) are even more explicit. `io_uring_submit_req`, `io_uring_queue_async_work`, and `io_uring_complete` carry combinations of ring context, request pointer, opcode, and `user_data`. The UAPI and [liburing headers](https://github.com/axboe/liburing/blob/master/src/include/liburing/io_uring.h) define SQE `user_data` as data returned at completion, and liburing provides helpers that associate a pointer or 64-bit value with an SQE and retrieve it from the CQE.

This is almost a ready-made handoff key. A profiler can observe submission while the application thread still has useful stack/request context, remember a bounded `(ring, user_data, generation)` relation, and join completion or async worker execution later.

But `user_data` belongs to the application. It may be a pointer, counter, packed identifier, or reused value. The profiler cannot assume semantic uniqueness without checking lifetime. Multishot operations make this more important because one submission can generate multiple completions. The useful abstraction is not "user_data is a trace ID"; it is "the kernel carries an application-selected correlation token across an async I/O boundary."

### Scheduler wakeups expose who woke whom, but not why

The Linux [`sched` tracepoints](https://github.com/torvalds/linux/blob/master/include/trace/events/sched.h) provide another partial edge. `sched_waking` is documented as executing from the waking context and identifies the task being woken. `sched_switch` then identifies the tasks leaving and entering a CPU.

This allows a profiler to reconstruct a wakeup edge from current execution to a target task and later measure run-queue delay. It still does not explain the semantic reason for the wakeup. One request may wake a shared event loop that processes many requests. Scheduler causality is therefore useful evidence, but it cannot replace runtime or operation identity.

## Existing profile and trace formats can carry causality when it already exists

OpenTelemetry Profiles makes an important design choice: profile samples may contain a `Link` with trace and span IDs. That means a future eBPF causal profiler does not need to invent a competing storage format for application traces. When valid span context is already known at sample time, the sample should link to it.

The hard case is precisely when that context is absent or stops at one boundary. A closed-source service may have no tracing SDK. A kernel workqueue callback does not inherit a userspace span by magic. An `io_uring` completion may be processed by a shared thread after the submitting stack is gone. In these cases, a profile schema can store a link but cannot manufacture the link's meaning.

This suggests an interoperability rule: **use existing trace/span IDs as first-class semantic identities when present, and use eBPF handoff edges to extend attribution across boundaries that tracing does not cover**. The design should complement OpenTelemetry rather than fork it.

## Application-defined resources are a separate semantic boundary

OSDI 2026's [gigiprofiler](https://www.usenix.org/conference/osdi26/presentation/hu-yigong) is a useful counterpoint. It targets application-defined resources such as buffer pools and query caches that ordinary system metrics cannot understand. Its approach uses semantic inference plus static analysis to identify candidate resources and usage events, then tracks how requests interact with those resources at runtime. The evaluation reports diagnosis of all 15 studied issues across five applications and two additional MariaDB issues confirmed by developers.

The lesson for an eBPF profiler is not "add an LLM to profiling." It is that some performance state has no kernel-native name. If the question is "which request held this application buffer pool entry?", a syscall trace is insufficient unless the profiler has a join from an application resource identity to the request and execution context.

A general async profiler therefore needs a graded strategy:

- use kernel tracepoints when the kernel already exposes a correlation object;
- use uprobes or stable runtime hooks when a runtime has an observable task identity;
- accept a small explicit annotation or USDT-style join point for application-defined resources when no stable external observation exists;
- fall back to stack-only attribution and label the missing edge rather than inventing causality.

That last behavior matters. A profiler that says "unknown parent" is more useful than one that confidently joins two events because their timestamps are close.

## Sampling bias becomes harder when the profiler adapts to causal edges

Continuous profiling has an overhead budget. Recording every `sched_waking`, every workqueue event, every `io_uring` request, every runtime poll, and high-frequency stacks on a large host can approach tracing volume. A practical system will want selective or adaptive capture.

But selection changes the estimator.

OSDI 2026's [Blink](https://www.usenix.org/conference/osdi26/presentation/devsot) is a current warning that sampling errors are not always harmless variance. For flat workloads with many short-lived routines, the authors report that sampling profilers such as `perf` can suffer skid, shadow effects, and incomplete function coverage that produce systematic errors. Blink switches to lightweight instrumentation for that workload class and reports 99.999% accuracy at 1% overhead in its evaluated setting.

Async causal profiling adds another source of selection bias. Suppose a profiler increases sampling frequency only after it observes a long `io_uring` wait. If it later aggregates raw sample counts, slow requests are now more likely to be sampled than normal requests. The profiler may correctly find interesting examples while producing a biased estimate of how much total CPU or wait time each path consumes.

The design implication is simple but often omitted: **if capture probability changes based on observed behavior, record the inclusion probability with the sample or edge**. Then weighted estimators can at least attempt to recover population-level quantities. If the selection policy cannot expose a defensible probability, the result should be presented as targeted diagnostic evidence, not an unbiased profile.

## A bounded handoff graph is a better middle layer than a universal trace

A useful design can keep the data plane small by storing only the edges required to bridge execution identities.

One possible normalized edge is:

```text
HandoffEdge {
    source_execution
    target_kind
    target_id
    target_generation
    edge_kind          // wake, queue, submit, complete, spawn, poll ...
    timestamp
    expires_at
    sample_probability
}
```

`source_execution` can be a process/thread plus the currently known semantic context. `target_id` can be a work object, io_uring request token, runtime task, or application resource handle. `target_generation` prevents stale pointer reuse from silently merging lifetimes. The target becomes a new `source_execution` when execution resumes or a worker starts processing it.

The graph should be **bounded** in four ways:

- **time**, by expiring edges after the relevant operation lifetime;
- **scope**, by only instrumenting selected processes/cgroups or workloads;
- **edge type**, by enabling only the handoff families needed for the current diagnosis;
- **rate**, by sampling high-volume edges with an explicit probability or by switching to exact capture only for a bounded diagnostic interval.

This design preserves a key distinction from distributed tracing. A trace usually wants a durable end-to-end request history. The profiler only needs enough temporary lineage to attribute sparse samples and waits. Once aggregation is complete, most raw edges can disappear.

## Where current work is still weak

### Zero-instrumentation profiling still stops at semantic task boundaries

Whole-system eBPF profilers can unwind impressive mixed stacks without target modifications. They cannot automatically know that two polls on different Tokio workers belong to one Future, or that three goroutines cooperate on one application task, unless a stable runtime identity can be observed.

The missing element is a portable, minimal join contract for runtime task identity. The consequence today is a choice between low-overhead stack profiling with incomplete semantic lineage and richer tracing that requires runtime/application participation.

A decisive test is whether a small adapter can recover most request attribution while remaining substantially cheaper and easier to deploy than full tracing. If not, the adapter is just another tracing SDK with a different name.

### Kernel correlation identifiers have inconsistent lifetime semantics

`work_struct *`, io_uring request pointers, `user_data`, task PIDs, and runtime task IDs all have different reuse rules. Treating any one of them as a permanent causal ID creates false joins after reuse.

The missing element is an explicit identity model that combines object kind, owner scope, generation, and retirement condition. The consequence is subtle: a profiler may look complete while silently merging unrelated operations.

The falsifying experiment should aggressively reuse objects and identifiers under load. If a simple TTL and owner tuple eliminates false joins across real workloads, a more elaborate identity protocol is unnecessary.

### Profiles can link to traces, but there is no standard low-level handoff vocabulary

OpenTelemetry Profiles can link a sample to a span. Linux tracepoints expose queue and completion edges. What is still missing is a small cross-runtime vocabulary for saying that execution A submitted resource R, resource R resumed execution B, or execution B is waiting on application resource X.

Without that vocabulary, each profiler backend has to invent its own join logic and cannot easily distinguish "no causal edge" from "collector did not support this edge type."

The key experiment is portability. The same handoff schema should represent workqueue, io_uring, Tokio, Go, and an application-defined resource without flattening away their important lifetime differences.

### Adaptive causal capture can improve diagnosis while corrupting aggregates

A fixed-rate stack sampler has familiar statistical properties. A profiler that selectively increases edge or stack capture around suspicious waits changes which executions enter the dataset.

The missing element is probability-aware accounting and a clear separation between diagnostic evidence and population estimates. The consequence is that a system can find a real slow request but still report the wrong percentage of total cost.

The strongest evaluation is not another overhead graph. It is coverage calibration: when the profiler reports that a causal path accounts for 20% of CPU or blocked time, does a high-fidelity ground truth fall inside the reported confidence interval?

## Promising directions with academic and production value

### 1. Build a causal handoff ledger for kernel-visible async boundaries

**Gap.** Stack samples lose lineage when work crosses threads, worker pools, or asynchronous I/O boundaries even though Linux exposes correlation objects at several of those handoffs.

**Mechanism.** Attach eBPF programs to selected workqueue, io_uring, scheduler, and syscall tracepoints. Normalize queue/submit/wake and start/complete events into a bounded handoff graph. Join `(scope, object, generation)` rather than raw pointers alone. Attribute later stack samples and off-CPU intervals through the graph, then expire raw edges after aggregation.

**Delta.** Existing eBPF continuous profilers concentrate on stack collection and unwinding; full tracers record much richer event histories. The proposed layer records only cross-context edges needed to stitch sparse profile samples.

**Artifact.** A libbpf-based collector, a compact handoff schema, an online join engine, and a pprof/OpenTelemetry Profiles exporter that preserves ordinary profile output while attaching causal labels or trace links when available.

**Evaluation.** Build workloads that exercise synchronous calls, kernel workqueues, `io_uring`, scheduler wakeups, and mixed handoffs. Generate ground truth with controlled instrumentation. Measure edge precision/recall, sample-attribution accuracy, false joins under identifier reuse, memory state, event rate, and CPU overhead. Compare against stack-only profiling and full tracing.

**Academic value.** This tests whether causal reconstruction can be treated as a sparse graph problem between profiling and tracing, and identifies which kernel interfaces expose sufficient identity.

**Production value.** Operators could answer "which request or submitter caused this worker CPU time?" without enabling full application tracing across every service.

**Failure condition.** If exact edge capture approaches trace-level overhead, or if identifier reuse prevents accurate joins without invasive instrumentation, the middle layer loses its deployment advantage.

### 2. Make adaptive profiling probability-aware

**Gap.** Continuous capture must stay within an overhead budget, but increasing sample or edge rates around suspicious activity can bias aggregate cost estimates.

**Mechanism.** Use randomized base sampling and an adaptive controller that changes capture probability by scope or edge family. Record the effective inclusion probability on each sampled observation. Estimate CPU, wait, and handoff contributions with inverse-probability weighting and report uncertainty. Keep diagnostic-only observations separate when their selection probability is unknown.

**Delta.** Adaptive profilers commonly optimize where to spend the next sample. This direction makes the selection policy part of the measurement contract so changing the rate does not silently change the meaning of percentages.

**Artifact.** A sampling controller, probability metadata in the handoff schema, weighted aggregators, and a calibration harness with synthetic phase changes and real async services.

**Evaluation.** Compare fixed-frequency sampling, randomized sampling, naive adaptive sampling, probability-corrected adaptive sampling, and high-fidelity tracing/instrumentation. Measure error, confidence-interval coverage, detection latency, overhead, and variance across bursty and flat workloads. Include a Blink-like short-function workload to ensure the profiler recognizes cases where sampling itself is the wrong measurement technique.

**Academic value.** It connects profiler control policy to statistical estimability instead of evaluating adaptivity only by hotspot detection rate.

**Production value.** A profiler can spend more budget on emerging anomalies while keeping dashboards interpretable and preventing targeted capture from masquerading as an unbiased fleet-wide profile.

**Failure condition.** If inverse-probability weighting produces intolerable variance under the useful adaptive policies, the system should stop claiming population estimates and use adaptivity only for case finding.

### 3. Define a minimal runtime join contract for opaque tasks and resources

**Gap.** Kernel evidence cannot name every Future, goroutine task, request-local cache entry, or application-defined resource. Full tracing can carry those semantics, but requiring a tracing SDK everywhere gives up the zero-instrumentation advantage.

**Mechanism.** Define an optional join ABI with only a few operations: create semantic ID, hand off semantic ID to runtime/resource ID, activate/deactivate ID, and retire ID. Implement adapters using stable runtime hooks, uprobes, USDT probes, or tiny library calls. Import OpenTelemetry trace/span IDs when they already exist. Do not collect arbitrary application payloads.

**Delta.** Go `runtime/trace` tasks and Rust `tracing::Instrument` demonstrate that async semantic context can be propagated, while gigiprofiler demonstrates the value of request-to-resource attribution. The proposed contract is intentionally smaller than either a full language trace or resource-specific profiler.

**Artifact.** A versioned C ABI/schema plus reference adapters for Tokio and Go, an application-defined resource example, and an eBPF collector that can combine adapter identities with kernel handoff edges.

**Evaluation.** Run identical services in four modes: stack-only eBPF, kernel handoff graph, handoff graph plus join adapter, and full application tracing. Compare semantic attribution coverage, correctness, setup effort, event volume, overhead, and robustness across runtime versions.

**Academic value.** The experiment identifies the minimum semantic information that must cross the userspace/kernel boundary for causal profiling and where zero-instrumentation fundamentally ends.

**Production value.** Teams could keep whole-system profiling as the default and add small semantic adapters only to runtimes or resources where the missing identity materially blocks diagnosis.

**Failure condition.** If adapters require unstable internal symbols, frequent per-runtime maintenance, or nearly the same code changes and overhead as standard tracing, existing tracing instrumentation is the better interface.

## A concrete evaluation should test attribution, not only overhead

A new profiler can look excellent if the benchmark only reports CPU overhead and data volume. The central claim here is causal attribution, so the evaluation must include ground truth.

A useful matrix would contain:

| Workload | Handoff | Ground truth | Main failure to detect |
| --- | --- | --- | --- |
| synchronous RPC | same thread | instrumented request ID | false positive joins |
| workqueue microservice helper | `work_struct` | explicit queue ID | worker charged to wrong producer |
| async file server | `io_uring` | SQE/CQE operation ID | completion detached from submitter |
| Tokio service | Future moves workers | tracing span/task ID | thread attribution mistaken for task attribution |
| Go service | multiple goroutines per task | `runtime/trace` task | goroutine split loses logical operation |
| cache/buffer-pool workload | app-defined resource | explicit resource owner | kernel evidence cannot identify resource semantics |
| flat short-function workload | none | instrumentation | biased or incomplete stack sampling |

The primary metrics should be causal-edge precision/recall, end-to-end request attribution accuracy, cost-attribution error, false joins after ID reuse, and confidence-interval coverage for aggregate estimates. Overhead, memory footprint, event volume, and diagnosis latency are constraints, not substitutes for correctness.

The strongest baseline set is also heterogeneous: a stack-only eBPF profiler, OpenTelemetry traces where instrumentation exists, runtime-native traces for Go/Rust tasks, and a high-fidelity instrumented ground truth. If the new system only beats a deliberately weak stack-only baseline, the research result is not convincing.

## What would change this conclusion?

The bounded handoff graph is useful only if there is a real middle ground between stack sampling and full tracing.

Three results would weaken the thesis substantially.

First, if modern trace context is already available on nearly all CPU samples and follows kernel async work with low overhead, then adding a separate eBPF causal layer is unnecessary. OpenTelemetry's profile-to-span links would be enough.

Second, if kernel-visible identifiers prove too unstable to join accurately across realistic object reuse, and fixing them requires invasive runtime changes, the supposed zero-instrumentation advantage disappears.

Third, if exact handoff capture plus state maintenance costs about as much as a well-engineered trace while providing less semantic detail, operators should use tracing for causal questions and keep eBPF profiling focused on stacks.

The opposite result would be more interesting: if a small set of eBPF handoff probes plus optional runtime joins can attribute most async CPU and wait cost at continuous-profiling overhead, then "profile" and "trace" are not the only useful choices. There is a third layer: sparse causal evidence that explains why sampled work ran without retaining every event.

That is the mechanism worth building and measuring.

## Primary sources

- [OpenTelemetry eBPF Profiler](https://github.com/open-telemetry/opentelemetry-ebpf-profiler)
- [OpenTelemetry Profiles specification](https://opentelemetry.io/docs/specs/otel/profiles/)
- [Linux workqueue tracepoints](https://github.com/torvalds/linux/blob/master/include/trace/events/workqueue.h)
- [Linux io_uring tracepoints](https://github.com/torvalds/linux/blob/master/include/trace/events/io_uring.h)
- [liburing io_uring UAPI header](https://github.com/axboe/liburing/blob/master/src/include/liburing/io_uring.h)
- [Linux scheduler tracepoints](https://github.com/torvalds/linux/blob/master/include/trace/events/sched.h)
- [Tokio task spawn documentation](https://docs.rs/tokio/latest/tokio/task/fn.spawn.html)
- [Rust tracing Instrument documentation](https://docs.rs/tracing/latest/tracing/trait.Instrument.html)
- [Go runtime/trace package](https://pkg.go.dev/runtime/trace)
- [OSDI 2026: Diagnosing Performance Issues in Application-Defined Resources](https://www.usenix.org/conference/osdi26/presentation/hu-yigong)
- [OSDI 2026: When Sampling Lies](https://www.usenix.org/conference/osdi26/presentation/devsot)
