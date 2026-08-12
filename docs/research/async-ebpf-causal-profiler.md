---
date: 2026-08-12
title: "What Must an eBPF Profiler Track Beyond Threads?"
description: "Async runtimes move work across threads, queues, and resources. This report asks how eBPF profiling can recover causal paths without tracing everything."
tags:
  - Daily Report
  - eBPF
  - Profiling
  - Observability
  - Linux
research_question: "What causal identities and handoff edges must an eBPF-based asynchronous profiler capture across syscalls, io_uring, work queues, runtime tasks, and application-defined resources, and how can it do so without full tracing overhead or biased attribution?"
source_cutoff: 2026-08-12
status: daily-report
---

# What Must an eBPF Profiler Track Beyond Threads?

Imagine a request handler that looks cheap in a normal CPU profile. It parses a request, submits an `io_uring` read, yields, resumes as a runtime task on another worker thread, queues a kernel work item, touches an application-managed cache, and finally sends a response. The request spends little continuous time on any one thread.

A CPU flamegraph can show the functions that ran. An off-CPU profile can show where threads waited. A full trace can record many of the transitions. None of those automatically tells us that these pieces belong to one logical request, especially after the work leaves the original thread.

<!-- more -->

That is the core problem for asynchronous profiling. **Thread identity is no longer a sufficient causal identity.** Modern Linux already exposes useful handoff evidence. `io_uring` carries application-provided `user_data` from submission to completion. Workqueue tracepoints expose the same `work_struct *` when work is queued and when its callback starts. Async runtimes such as Tokio have their own task IDs and tracing spans. Application-defined resources such as buffer pools or query caches have identities that may exist only in program logic.

The pieces are individually observable, but they do not form one common causal graph by themselves.

The useful research question is therefore narrower than "build an async profiler with eBPF." The missing mechanism is a **typed causal-edge contract** that says which identity links two execution contexts, how long that identity is valid, how confident the link is, and what the profiler should report when an edge was not observed. A practical system should capture cheap synchronization and handoff edges more reliably than expensive stack context, then sample richer context under an explicit budget instead of pretending that an incomplete trace is complete.

This report continues the [eBPF runtime and extensibility series](https://eunomia.dev/research/userspace-ebpf-runtime-contract/). It also builds on Eunomia's [wall-clock eBPF profiler tutorial](https://eunomia.dev/tutorials/32-wallclock-profiler/), which combines on-CPU and off-CPU accounting. The step beyond wall-clock accounting is preserving causality when logical work changes execution context.

## CPU and off-CPU profiles stop being request histories

Continuous eBPF profiling is already practical. Parca's eBPF profiler automatically discovers processes and produces pprof profiles. Grafana's OpenTelemetry eBPF profiler collects system-wide CPU stack samples and ships them through the OpenTelemetry profiling pipeline. These systems are valuable because they provide low-friction, always-on visibility without application SDK instrumentation.

Their natural aggregation key is still close to a process, thread, stack, cgroup, or service label. That is exactly what a conventional profile needs. It is not necessarily what an asynchronous request needs.

Consider three simplified execution intervals:

```text
thread 17: request A -> submit async read -> yield
thread 23: request B -> CPU work -> yield
thread 31: completion of A -> resume task A -> queue work
```

A CPU profile attributes execution to stacks sampled on threads 17, 23, and 31. An off-CPU profile can attribute waiting intervals to the thread or stack that blocked. But request A is not a thread. Its causal path moved from one task to an asynchronous I/O request and later to another execution context.

This is not specific to Rust. Event loops, fibers, green threads, coroutine schedulers, work-stealing runtimes, user-level RPC systems, and callback-heavy C/C++ programs all separate logical work from the kernel thread currently executing it.

Tokio's own documentation makes this boundary explicit. A Tokio task has an opaque ID that uniquely identifies a currently running task, while the runtime can schedule many tasks across worker threads. Tokio's tracing guide also explains why ordinary thread-oriented logs are difficult in asynchronous systems: tasks are multiplexed on threads, so events from different logical flows become interleaved. The runtime can expose task semantics, but application tracing is still needed to get the richest causal context.

The consequence is simple: **a profiler that only samples thread stacks can be accurate about CPU execution and still be wrong about which logical operation paid for it.**

## Linux already exposes several strong causal handoff keys

The situation is not hopeless. Several important asynchronous mechanisms have explicit identities that can be observed at their handoff boundaries.

### `io_uring` carries a user-selected correlation value

The Linux `io_uring` UAPI defines a 64-bit `user_data` field in each submission queue entry. The corresponding completion queue entry returns the same value. Applications normally use it to identify which request completed.

That is unusually useful for profiling. If the profiler can observe submission and completion while preserving `user_data`, it has a natural edge:

```text
submitting logical context
        |
        | io_uring user_data = X
        v
   async request X
        |
        | completion user_data = X
        v
completion-handling context
```

Current kernel source also contains `io_uring` tracepoints around request submission and completion. The exact internal execution path is richer than one submit/complete pair: requests may complete inline, use task work, be punted to io-wq workers, be polled, form links, or emit multiple completions. That is why correlating only `sched_switch` events cannot reconstruct an `io_uring` request lifecycle.

Recent work makes the opportunity and the limitation concrete. The 2026 `uringscope` preprint reconstructs per-request `io_uring` flows with CO-RE eBPF and explicitly studies the fidelity/overhead tradeoff. It also points out that the relevant kernel tracepoint surface is not a stable ABI, so portability is part of the profiler problem rather than an implementation detail.

`io_uring` therefore gives us a good subsystem-local causal key, not yet a system-wide causal identity.

### Workqueues expose one object across queue and execution events

Linux workqueue tracepoints similarly expose a `struct work_struct *work` when work is queued, activated, starts executing, and finishes. That makes it possible to connect a producer context to a later worker callback even when a different kernel worker thread performs the work.

A profiler can model:

```text
producer context
    |
    | queue_work(work=W)
    v
work instance W
    |
    | execute_start(work=W)
    v
worker execution
```

The pointer is useful because the queue and execute events name the same object. It is not a permanent global identifier. Objects can be reused after their lifetime ends, events can be dropped, and a profiler that runs for days cannot treat a raw address as an eternal unique ID.

This immediately suggests a general rule for asynchronous profiling: **every correlation key needs an explicit lifetime or generation boundary.** A pointer, file descriptor, runtime task ID, request token, or application handle is only meaningful inside the scope where reuse is impossible or detectable.

### Thread scheduling remains useful, but as one edge type

Scheduler events still matter. They explain when a kernel task runs, blocks, wakes, migrates, or competes for CPU. Off-CPU profiling depends on exactly this evidence. The mistake is to promote the kernel task to the universal causal unit.

A better model treats scheduling as one family of edges among several:

- task runnable -> task running;
- syscall caller -> kernel operation;
- submitting task -> `io_uring` request;
- work producer -> workqueue item;
- runtime parent task -> spawned task;
- request -> application-defined resource event.

The final profile should be able to aggregate by thread when that is useful, but it should not require all causal paths to remain inside one thread.

## Runtime tasks and application resources cross the kernel visibility boundary

Pure eBPF observation has an important limit: the kernel sees execution and kernel objects, but it does not automatically know the semantic identity of every user-space task or resource.

Tokio provides an instructive example. It has task IDs, and its tracing ecosystem can expose structured task information. Those IDs are meaningful because the runtime defines them. The kernel scheduler sees the worker thread that polls a future, not the Rust future as a kernel task.

The same problem becomes stronger for application-defined resources. A database buffer pool page, an internal query cache entry, a model-serving KV-cache block, or a temporary compiler object may dominate performance while having no kernel object whose identity captures its semantic role.

The OSDI 2026 `gigiprofiler` work is strong evidence for this boundary. It targets performance problems caused by application-defined resources precisely because their semantics are not visible in ordinary system-level metrics. Its design combines semantic inference with static analysis, then instruments resource-use events and attributes them back to requests. It diagnosed 15 known issues across five applications and found two additional MariaDB issues confirmed by developers.

This is a useful counterexample to an overambitious "zero-instrumentation eBPF sees everything" thesis. It does not. A system-level profiler can discover many effects without application cooperation, but logical task IDs and resource semantics sometimes exist only above the kernel boundary.

The practical design should therefore be **hybrid, not purity-driven**:

1. use eBPF for kernel-visible handoffs, scheduling, syscalls, I/O, process boundaries, and low-level resources;
2. consume runtime-native task IDs when runtimes expose them;
3. allow small application adapters for resource identities that cannot be inferred safely;
4. keep every edge tagged with its source so downstream analysis knows which relationships were directly observed, inferred, or missing.

This is similar in spirit to the [userspace eBPF runtime contract](https://eunomia.dev/research/userspace-ebpf-runtime-contract/): portability comes from making host-specific capabilities explicit, not from pretending every backend exposes the same semantics.

## Full tracing is not the obvious answer

Once we admit that profiles need causal edges, the tempting solution is to trace every event and join everything offline. That works for some workloads and is the strongest baseline a proposed profiler should beat. It is not automatically suitable for always-on deployment.

Two recent OSDI 2026 systems illustrate opposite sides of the measurement problem.

`StriaTrace` observes that production LLM inference tracing can be prohibitively expensive. Its design traces key synchronization points and critical paths, then enables detailed tracing around abnormalities. The reported evaluation reduces tracing overhead by 97.8% relative to alternatives while retaining enough evidence to diagnose hundreds of anomalies across 19 root-cause classes.

`Blink` attacks a different problem: sampling itself can lie. In flat workloads with thousands of short-lived routines, sampling profilers can suffer from skid, shadow effects, and incomplete function coverage. The paper reports that these effects create systematic measurement error, not merely wider confidence intervals. Blink uses lightweight instrumentation and reports 99.999% accuracy at about 1% overhead for its target workloads.

Together they rule out a simplistic design choice:

- tracing every edge can be too expensive;
- sampling every context can be systematically biased;
- therefore "sample less" is not by itself an overhead solution.

An asynchronous causal profiler needs to distinguish **which events define graph structure** from **which events provide expensive context**.

A queue handoff may be rare and semantically decisive. Missing it can split one request into two unrelated profiles. A stack sample inside a long CPU phase can be statistically redundant because neighboring samples are similar. Those two event classes should not share the same sampling policy.

## A sparse typed causal graph is a better profiling object

The profiler does not need a complete event log. It needs enough evidence to reconstruct the parts of the causal graph that affect attribution.

One possible internal model is:

```text
Node {
    type: thread | runtime_task | io_request | work_item | resource | request
    id: source-specific identity
    generation: lifetime discriminator
}

Edge {
    type: submit | complete | queue | wake | spawn | resume | acquire | release
    src: Node
    dst: Node
    time: timestamp
    source: kernel | runtime | application | inferred
    confidence: observed | reconstructed | unknown
}

Sample {
    node: Node
    stack: optional stack context
    weight: measured or statistical weight
    inclusion_probability: optional sampling probability
}
```

The important design choice is that a missing edge is not silently replaced with a guessed parent. If a completion arrives after its submission record was evicted, the system should preserve an orphaned completion or an `unknown` parent. That makes uncertainty visible instead of fabricating a clean request flamegraph.

This graph does not need to remain in raw form. Once edges are joined, it can be compressed into profile stacks such as:

```text
request A
  -> tokio task 91
    -> io_uring read X
      -> completion
        -> tokio task 91
          -> cache shard 4
            -> function foo
```

The output can still be pprof-like and flamegraph-friendly. The difference is that stack ancestry now includes causal handoffs, not only call-stack ancestry.

This is close to what an async profiler should mean: **profile weights attached to a reconstructed causality tree or DAG, with explicit evidence quality.**

## Where current work is still weak

### Correlation identities are local to each subsystem

`io_uring` has `user_data`. Workqueues expose `work_struct *`. Runtime systems can expose task IDs. Networking, timers, futexes, block I/O, and user-level queues use other identities or no convenient identity at all.

The missing mechanism is a common representation that can say, "this identifier is unique only inside ring R until completion," or "this pointer identifies one work item until execute-end," and can map those scoped identities into a profiler-wide node namespace.

Without lifetime semantics, pointer or ID reuse creates false joins. Without source semantics, two 64-bit values from different subsystems can look comparable when they are not.

A decisive test is long-running stress with aggressive object reuse. If a proposed profiler cannot keep causal precision when request IDs, pointers, and runtime task IDs wrap or are reused, the identity model is not strong enough.

### Kernel observation cannot recover every logical task or resource

A pure eBPF profiler can observe the worker thread that polls a future, but the runtime defines the future's task identity. It can observe memory accesses and syscalls around a database, but an application-defined cache entry or buffer pool has semantics that may not correspond to a kernel resource.

The missing mechanism is a small adapter contract for runtime and application identities, with a safe fallback to `unknown` when no adapter exists.

The research question is not whether adapters can be added. They obviously can. The interesting question is how small the adapter can be while preserving the zero- or low-instrumentation advantage for the rest of the system. If every framework needs invasive tracing, the result has collapsed back into ordinary distributed tracing.

### The profiler lacks a budget model for graph edges versus context samples

Traditional profilers often choose a sample frequency. Tracers often choose which events to enable. An asynchronous profiler needs both decisions at once.

Missing a decisive handoff edge can destroy attribution for all later samples. Missing one of many similar CPU samples may barely change the aggregate. A uniform sampling probability therefore spends measurement budget poorly.

The missing mechanism is an explicit budget allocator that protects high-value causal edges and samples expensive context separately. It should also record inclusion probabilities or another defensible weighting model so that aggregate estimates remain interpretable.

The falsifier is empirical: if deterministic edge capture plus sampled context does not produce better attribution accuracy per unit overhead than full tracing, uniform sampling, or ordinary profiles, the extra machinery is not justified.

### There is no common ground-truth benchmark for asynchronous causal attribution

A generated flamegraph can look plausible even when its ancestry is wrong. That is dangerous because visual plausibility is not a correctness metric.

The missing artifact is a benchmark that knows the true parent-child relationships across several asynchronous mechanisms and can inject difficult cases such as work stealing, completion reordering, multishot I/O, object reuse, dropped events, nested runtime tasks, and application-resource contention.

The main metric should not only be runtime overhead. It should include causal-edge precision and recall, attribution error for wall time and resource cost, orphan rate, false-join rate, and error under event loss.

If ordinary OpenTelemetry/runtime tracing already provides the same attribution accuracy at similar deployment cost for the target workloads, an eBPF-centered causal profiler should admit that result rather than claim novelty from its data source.

## Promising directions with academic and production value

### A typed causal-edge substrate for eBPF profiling

**Gap.** Linux exposes useful asynchronous handoff identities, but each subsystem defines identity and lifetime differently. There is no common profiler contract across `io_uring`, workqueues, scheduler activity, runtime tasks, and application resources.

**Mechanism.** Build a small edge-normalization layer. Each adapter emits scoped node identities and typed handoff edges. A kernel adapter might convert `(ring, user_data, generation)` into an `io_request` node and `(work pointer, queue generation)` into a `work_item` node. Runtime adapters map a Tokio task ID or another runtime-native handle into a `runtime_task` node. Every edge records source, lifetime scope, timestamps, and whether it was directly observed or reconstructed.

The kernel side should keep only short-lived correlation state in bounded BPF maps. Userspace owns longer-term joining, generation management, eviction reporting, and graph compression. This avoids turning eBPF maps into an unbounded trace database.

**Delta.** The delta from a conventional trace is not another event schema. It is the explicit cross-subsystem identity and lifetime contract plus the rule that unknown ancestry remains unknown. The delta from ordinary pprof is causal ancestry that can cross thread boundaries.

**Artifact.** A libbpf or Aya-based collector with adapters for scheduler/syscalls, `io_uring`, workqueues, and one async runtime, plus a pprof-compatible exporter that can encode causal frames as synthetic profile labels or stack frames. The [eunomia-bpf developer tutorial](https://github.com/eunomia-bpf/bpf-developer-tutorial) could provide small reproducible eBPF probes for the kernel adapters.

**Evaluation.** Build controlled pipelines that move one request through increasing numbers of handoffs. Measure edge precision/recall, false joins under pointer/ID reuse, memory consumed per live causal node, event-loss behavior, and end-to-end attribution error. Compare against perf/eBPF CPU profiling, wall-clock on/off-CPU profiling, runtime-native tracing, and full event tracing.

**Academic value.** The research question is whether a small set of typed lifetime-aware edges is sufficient to reconstruct useful causality across heterogeneous asynchronous mechanisms.

**Production value.** Operators get request- or operation-oriented profiles without requiring every component to adopt one tracing SDK, while still seeing where evidence is incomplete.

**Failure condition.** If real applications require so many runtime-specific edge types that no compact contract emerges, the right abstraction is per-runtime tracing rather than a common causal profiler.

### A budgeted edge-aware profiler with honest statistical weights

**Gap.** Always-on full tracing can be expensive, while uniform stack sampling can miss short phases and causal handoffs or introduce systematic bias.

**Mechanism.** Separate capture into two planes. The **edge plane** prioritizes semantically decisive handoffs such as queue, submit, completion, wake, spawn, and resource-acquire events. The **context plane** samples stacks, arguments, resource counters, or payload metadata under a configurable budget. Expensive context can be sampled adaptively by causal node, phase, or anomaly signal.

When sampling is probabilistic, record enough information to estimate inclusion probability or to bound the class of valid aggregates. When a required edge was dropped because a buffer overflowed or correlation state was evicted, emit an explicit coverage break rather than joining across the gap.

A useful adaptive policy could increase context sampling when a causal path crosses many subsystems, accumulates unexpected delay, or reaches a rare error, while keeping ordinary paths cheap. StriaTrace's key-synchronization and anomaly-focused design is a strong production baseline for this idea, while Blink is a warning that sampling policy must be validated against systematic error rather than tuned only for overhead.

**Delta.** The novelty is treating causal edges and context samples as different statistical objects with different loss costs. Conventional sampling frequencies do not capture this asymmetry.

**Artifact.** An always-on profiler with a fixed CPU/event budget, configurable edge priorities, event-loss telemetry, and a query layer that reports both estimated cost and causal coverage.

**Evaluation.** Replay workloads under budgets from very small to near-full tracing. Compare causal attribution error, CPU overhead, event volume, tail-latency perturbation, and diagnostic success. Include flat workloads similar to Blink's failure mode so the profiler cannot win by choosing only easy heavy-hitter distributions.

**Academic value.** This becomes a measurement problem: what is the optimal allocation of a fixed observability budget between topology-defining events and value-estimating samples?

**Production value.** Teams can run the profiler continuously with a predictable cost and know when a diagnosis is based on incomplete causality instead of receiving a falsely precise flamegraph.

**Failure condition.** If the edge plane itself dominates overhead on high-event-rate workloads, or if no stable weighting scheme survives adaptive sampling, the design needs stronger aggregation at source or a narrower supported workload class.

### A ground-truth async causality benchmark

**Gap.** Existing profilers can be compared on CPU overhead and profile shape while still disagreeing about which logical operation caused the work.

**Mechanism.** Build a benchmark harness where every logical operation carries a hidden ground-truth ID through controlled asynchronous mechanisms. Include synchronous calls, `io_uring`, kernel workqueues, timers, futex handoffs, a Tokio or equivalent task runtime, and one application-defined resource such as a bounded cache or buffer pool. The ground-truth channel is used only by the evaluator, not by the profiler under test.

Fault modes should deliberately stress the identity model: work stealing, concurrent reuse of pools, delayed and out-of-order completion, multishot requests, dropped trace records, truncated buffers, nested task spawn, cancellation, and application-resource reuse.

**Delta.** The benchmark scores *causal attribution*, not merely whether expected trace events appeared. It can therefore compare a CPU sampler, an off-CPU profiler, full tracing, runtime/OpenTelemetry instrumentation, `uringscope`-style subsystem reconstruction, and a hybrid causal profiler under the same workloads.

**Artifact.** A reproducible Linux benchmark suite, reference traces, and metrics for edge precision/recall, false joins, orphan rate, attributed wall-time error, resource-cost error, overhead, and tail perturbation.

**Evaluation.** Run across kernel versions, CPU counts, runtime worker counts, queue depths, request fan-out, and controlled event-loss rates. Include both thread-affine workloads, where ordinary profiling should win on simplicity, and heavily asynchronous workloads, where causal reconstruction should show measurable benefit.

**Academic value.** The benchmark turns a vague claim about "understanding async systems" into a falsifiable profiling problem.

**Production value.** Tool builders can tell users which async mechanisms and loss regimes are supported rather than marketing a generic low-overhead profiler with unknown attribution accuracy.

**Failure condition.** If causal accuracy has little correlation with real diagnostic outcomes, then edge precision is the wrong target metric and the benchmark should be redesigned around operator decisions.

## What would change this conclusion?

The strongest counterexample is a workload whose performance is mostly thread-local. A CPU-bound worker pool, a synchronous service, or an application where request identity never leaves one thread may be diagnosed perfectly well by CPU plus off-CPU profiles. Such systems do not need a causal graph merely because the profiler can build one.

A second counterexample is cheap full tracing. If relevant synchronization and runtime events can be traced continuously with negligible overhead, bounded storage, and stable schemas, then a complex edge/context budget may be unnecessary. `uringscope` already shows that targeted request reconstruction can be practical for one subsystem; future kernels or tracing infrastructure could make broader full-fidelity capture much cheaper.

A third counterexample is instrumentation availability. If the runtime and application already emit high-quality OpenTelemetry or native causal spans, the eBPF layer may be most valuable for validating system effects and filling low-level gaps rather than reconstructing the primary request graph.

Finally, the proposed graph is only useful if it improves diagnosis. A prototype should be rejected if its causal flamegraphs are more visually sophisticated but do not reduce time-to-root-cause, improve attribution accuracy, or discover failures that simpler profiles miss.

The thesis is therefore bounded: **eBPF can provide a strong system-level edge substrate for asynchronous profiling, but a trustworthy profiler must combine subsystem-local identities, runtime/application semantics, explicit lifetime rules, and a measurement budget that treats missing causality as uncertainty rather than inventing a clean thread-shaped story.**

## References

- [Linux Kernel Tracepoint API: workqueue tracepoints](https://docs.kernel.org/core-api/tracepoint.html)
- [Linux `io_uring` UAPI: SQE and CQE `user_data`](https://github.com/torvalds/linux/blob/master/include/uapi/linux/io_uring.h)
- [Linux `io_uring` implementation and tracepoints](https://github.com/torvalds/linux/blob/master/io_uring/io_uring.c)
- [Tokio task IDs](https://docs.rs/tokio/latest/tokio/task/struct.Id.html)
- [Tokio: Getting started with Tracing](https://tokio.rs/tokio/topics/tracing)
- [Parca continuous profiling and eBPF profiler](https://github.com/parca-dev/parca)
- [Grafana Pyroscope: OpenTelemetry eBPF profiler](https://grafana.com/docs/pyroscope/latest/configure-client/opentelemetry/ebpf-profiler/)
- [uringscope: Portable, Low-Overhead Observability for io_uring](https://arxiv.org/abs/2606.15137)
- [Beyond Thread States: Diagnosing Performance Degradation with eBPF and Thread Dynamics](https://arxiv.org/abs/2605.25298)
- [When Sampling Lies: Trustworthy Performance Profiling for Flat Workloads with Blink](https://www.usenix.org/conference/osdi26/presentation/devsot)
- [StriaTrace: Efficient Tracing and Diagnosis for Online LLM Inference](https://www.usenix.org/conference/osdi26/presentation/wu-haonan)
- [Diagnosing Performance Issues in Application-Defined Resources](https://www.usenix.org/conference/osdi26/presentation/hu-yigong)
- [SysOM-AI: Continuous Cross-Layer Performance Diagnosis for Production AI Training](https://arxiv.org/abs/2603.29235)
