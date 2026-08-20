---
date: 2026-08-20
title: "Can a GPU Profiler Prove What Caused a Slow Kernel?"
description: "GPU traces can show a slow kernel without proving which host action caused it. This report develops a host-device causal contract for asynchronous profiling."
tags:
  - Daily Report
  - GPU
  - Profiling
  - CUDA
  - CUPTI
  - Causality
research_question: "How can a production GPU profiler preserve causal identity from host work through asynchronous CUDA streams and graphs to device activity, instead of inferring causes from timestamps alone?"
source_cutoff: 2026-08-20
status: daily-report
---

# Can a GPU Profiler Prove What Caused a Slow Kernel?

A service receives request R42, prepares a batch on one CPU thread, launches work into two CUDA streams, and returns to host code immediately. Fifty milliseconds later one GPU kernel starts late. A timeline shows a CPU scheduling gap, a memory copy, several kernel launches, and the delayed kernel close together.

Which host action caused the delay?

A visual timeline can suggest an answer, but asynchronous GPU execution makes time proximity weak evidence. CUDA calls can return before device work begins, commands in different streams can overlap, cross-stream events create explicit dependencies, graph launches reuse pre-built dependency structures, and the GPU scheduler may delay otherwise independent work because resources are busy. A nearby host event is not automatically the parent of a device event.

<!-- more -->

Modern tooling already exposes much more than a pair of timestamps. NVIDIA CUPTI assigns a correlation ID to a CUDA driver or runtime API invocation and carries the same ID into associated kernel, memcpy, and memset activity records. It also supports external correlation IDs, CUDA Graph `graphId` and `graphNodeId`, stream and context identifiers, dropped-record accounting, and optional kernel queued and submitted timestamps. The current CUDA Programming Guide describes streams as ordered work queues and events as explicit cross-stream dependency edges.

Those mechanisms are enough to connect many local pieces, but they do not by themselves answer a production root-cause question that crosses application work, host scheduling, CUDA API calls, stream dependencies, and GPU execution. The missing object is a causal identity that survives the whole path and remains explicit when evidence is incomplete.

This report argues for a **host-device causal contract**: represent GPU profiling as a typed dependency graph, carry stable work identity across the host/device boundary, use CUDA stream and graph semantics as ordering constraints, and mark unknown or lossy edges rather than filling them with timestamp guesses.

This is an adjacent-systems report, not an eBPF-centered report. eBPF is useful for observing host scheduling, processes, page faults, driver interactions, and closed-source application boundaries, but the causal mechanism also works with other host tracers. CUPTI and CUDA dependency semantics are equally central.

## CUDA already provides local correlation, but local is not end to end

CUPTI's correlation ID solves an important problem. When a CUDA API call launches a kernel or initiates a memory operation, the API activity and resulting GPU activity can carry the same correlation ID. That lets a profiler answer questions such as "which `cudaLaunchKernel` produced this kernel record?"

External correlation extends the chain upward. CUPTI can associate an external identifier with CUDA API activity, which is how tools can connect higher-level regions or APIs to the CUDA work emitted inside them. CUDA Graph activity adds graph and graph-node identities, and stream IDs preserve the execution queue in which work was submitted.

The repository already contains [CUPTI correlation examples](https://eunomia.dev/others/cupti-tutorial/cupti_correlation/) and [external-correlation examples](https://eunomia.dev/others/cupti-tutorial/cupti_external_correlation/). These demonstrate that the low-level join is implementable. The harder question begins one layer above them.

Suppose request R42 is handled by thread T1, which delegates preprocessing to T2. T2 is descheduled for 4 ms, then calls a framework function that eventually invokes CUDA on T3 through an executor. The CUDA API call launches a graph containing kernels on several streams. CUPTI can correlate API records to GPU records, but the profiler still needs to know that these records belong to R42 and that the 4 ms delay on T2 lies on the dependency path to the late kernel.

A process ID and a timestamp do not encode that relationship. A thread ID is also insufficient once work moves through executors, callbacks, queues, futures, CUDA Graphs, or library-internal worker threads. The same problem appears in distributed tracing: span timing is useful only after the system knows which spans are related.

The host/device boundary therefore needs a first-class work identity, not only a collection of clocks.

## Asynchrony turns timeline order into a partial order

The CUDA Programming Guide states two properties that matter for diagnosis:

1. operations in one stream execute in enqueue order;
2. operations in different streams may execute concurrently, while CUDA events and `cudaStreamWaitEvent()` can establish dependencies across streams.

This means a multi-stream execution is naturally a partial order. If kernel B waits on an event recorded after kernel A, then A's completion is a real predecessor of B. If kernels C and D are in independent streams with no dependency, their nearby timestamps do not prove that one delayed the other.

CUDA Graphs make this even more explicit. A graph is a dependency structure rather than merely a timeline, and CUPTI exposes graph and graph-node identifiers for traced activity. A profiler that flattens the graph into one timestamp-sorted list throws away information the runtime already knows.

The distinction matters for root cause. Consider three possible reasons why kernel K starts 6 ms later than expected:

- the host thread submitted K 6 ms late because it was descheduled;
- K was submitted on time but waited on an earlier stream dependency;
- K was ready but device resources were occupied by unrelated work.

All three can produce a similar visual gap before K. The corrective action is completely different. The first points to host scheduling or CPU contention, the second to dependency construction, and the third to device scheduling or interference.

CUPTI can optionally report when a kernel was queued into a command buffer and when the command buffer was submitted to the GPU. Those timestamps create useful phase boundaries, but they are still observations inside a causal model. They do not replace the model.

## Cross-layer systems show the value of correlation, but not a general causal contract

Recent production systems already demonstrate that cross-layer GPU diagnosis is worth doing.

[SysOM-AI](https://arxiv.org/abs/2603.29235) continuously combines CPU stack profiling, GPU kernel tracing, NCCL instrumentation, and eBPF-based tracing. Its reported production deployment spans more than 80,000 GPUs and uses layered differential diagnosis to narrow issues that would otherwise require manual cross-tool analysis.

[Host-Side Telemetry for Performance Diagnosis in Cloud and HPC GPU Infrastructure](https://arxiv.org/abs/2510.16946) similarly correlates host-side eBPF telemetry with GPU-internal events to diagnose shared-infrastructure causes such as NIC contention, PCIe pressure, and CPU interference.

These systems establish a practical point: GPU symptoms often originate outside the GPU. They do not make every host/device relationship self-describing. Differential diagnosis can identify that one rank or host differs from its peers without proving the exact parent edge for every delayed operation.

The site's earlier [GPU observability analysis](https://eunomia.dev/blog/2025/10/14/gpu-observability-challenges/) already argued that isolated CPU and GPU tools leave a cross-layer visibility gap. The narrower problem here is what a profiler must record so that a claimed causal chain is testable rather than reconstructed from visual coincidence.

## Where current work is still weak

### Correlation IDs stop at the boundary where application meaning begins

CUPTI correlation IDs are strong local join keys between CUDA API activity and the GPU work caused by that API invocation. They do not automatically identify the user request, training step, inference token, runtime task, workqueue item, or framework operation that caused the API invocation.

Production frameworks can add NVTX ranges, external correlation IDs, or their own task identifiers, but closed-source libraries and runtime-internal worker threads make coverage uneven. A general profiler therefore needs a way to preserve application work identity even when the execution path changes thread before reaching CUDA.

The material test is simple: create a workload where one logical request moves through several host queues before launching GPU work. If the profiler cannot recover the correct request-to-kernel mapping without relying on timestamp proximity, the identity contract is incomplete.

### Dependency semantics are richer than a timestamp-sorted trace

A timeline can order records by observed start time. CUDA streams, events, and graphs express dependency constraints that may disagree with a naive total order.

A profiler needs to distinguish "happened earlier" from "must complete before." Otherwise it can blame an earlier independent kernel for a later delay simply because the two overlap. The missing artifact is a dependency graph whose edges come from explicit runtime semantics when available and whose uncertain edges remain uncertain.

A discriminating test should run identical kernels with and without cross-stream event dependencies while preserving similar timestamps. A causal profiler should change its explanation when the dependency changes, while a timestamp-only baseline may not.

### Missing records should break a causal claim, not silently weaken it

CUPTI exposes dropped-record accounting, and some activity timestamps can be unknown. Activity delivery is buffered and asynchronous. Host tracers can also lose samples or fail to observe a library boundary.

If one required edge disappears, the profiler should say that the causal path is incomplete. Many trace pipelines instead keep rendering the surviving events, which makes a broken path look like a complete but sparse path.

This is a correctness problem, not only a telemetry-quality problem. An operator should be able to tell the difference between "no dependency existed" and "the record that would establish the dependency was lost."

### Current evaluations rarely score edge correctness directly

GPU profiling papers usually report overhead, diagnosis accuracy, or time to root cause. Those are useful outcomes, but they can hide a profiler that reaches the right label for the wrong causal chain.

A causal profiler needs ground truth for parent edges and critical-path attribution. Without it, the system can improve diagnosis accuracy through workload-specific heuristics while still producing misleading explanations when the workload changes.

## Promising directions with academic and production value

### 1. A generation-scoped host-device causal token

**Gap.** CUDA correlation IDs connect API calls to GPU activity, while host tracers connect processes and system events. Neither side guarantees a stable identity for logical work that crosses host queues, threads, CUDA streams, and graph launches.

**Mechanism.** Give each profiled work unit a 128-bit causal token plus a generation. The token starts at an application boundary when one exists, such as an RPC, training step, inference iteration, or explicit profiling region. When work moves through a host queue, executor, future, or callback, the profiler records a typed handoff edge from the old execution context to the new one.

At the CUDA boundary, the collector binds the current causal token to the CUDA API invocation. When CUPTI supplies a correlation ID, that local ID becomes another edge in the graph rather than the global identity. Kernel, memcpy, memset, stream, context, graph, and graph-node records inherit the token through the verified CUPTI correlation path.

Closed-source applications need a weaker path. eBPF uprobes or another userspace tracer can observe CUDA runtime/driver calls and associate them with the current process/thread work context. If no framework-level token is available, the profiler records a process-scoped root and marks the higher-level parent unknown instead of inventing one.

The generation prevents stale reuse. Stream handles, graph executables, contexts, and framework objects can be destroyed and recreated. A causal key should therefore include object lifetime or generation rather than treating a raw handle as globally unique forever.

**Delta.** CUPTI already provides API-to-GPU correlation and external correlation. The proposed contract makes those IDs one typed layer in a longer lifetime-aware causal namespace that also contains host task handoffs and explicit unknown roots.

**Artifact.** A collector and portable trace schema that joins host scheduler/process evidence with CUPTI activity. The first prototype can reuse the repository's [xpu-perf](https://github.com/eunomia-bpf/xpu-perf) and CUPTI examples rather than requiring a new kernel interface.

**Evaluation.** Build request and training-step workloads that deliberately hand work across several host threads before launching multi-stream CUDA work. Randomize thread pools and reuse stream/graph objects. Measure request-to-kernel parent-edge precision and recall, stale-ID collisions, unknown-edge rate, and overhead against CUPTI-only and timestamp-join baselines.

**Academic value.** The general question is how to maintain causal identity across runtimes whose local identifiers and lifetimes do not share one namespace.

**Production value.** Operators can ask which request, host task, or process state produced a problematic kernel without reconstructing the answer from several dashboards.

**Failure condition.** If framework and CUPTI correlation already recover nearly all parent edges across realistic asynchronous workloads, the extra token layer is unnecessary and the simpler local IDs should win.

### 2. A dependency-aware critical-path graph with explicit uncertainty

**Gap.** A correct parent identity still does not explain delay. The profiler must distinguish host enqueue delay, command-buffer submission delay, stream dependency wait, device execution, and unrelated overlap.

**Mechanism.** Build a partial-order graph rather than a globally sorted timeline. Add hard edges from same-stream order, CUDA events, CUDA Graph dependencies, API-to-activity correlation, and observed host handoffs. Add interval observations for API start/end, optional CUPTI queued/submitted timestamps, GPU start/end, synchronization waits, scheduler gaps, page faults, and relevant network or storage stalls.

Each edge carries an evidence class:

- `explicit`: runtime or API semantics establish the dependency;
- `observed`: a host handoff or system event establishes it;
- `inferred`: timing and state suggest the relation but do not prove it;
- `missing`: a required record was dropped or unavailable.

Critical-path attribution may use explicit and observed edges automatically. Inferred edges must carry confidence and should not silently become hard dependencies. Missing edges stop the graph from claiming a complete path.

The profiler can then decompose a late kernel into phases. A long API-to-queued interval points toward host or driver preparation. A long queued-to-submitted interval points toward command-buffer or driver delay. A long submitted-to-start interval with a dependency predecessor points toward stream/graph waiting; the same interval without such a predecessor may indicate device contention and needs deeper device evidence.

**Delta.** Existing timelines already display these timestamps and dependencies in various forms. The contribution is an explicit causal algebra that separates dependency proof from temporal coincidence and carries telemetry loss into the diagnosis result.

**Artifact.** A graph builder plus query interface: `why-late <kernel-id>` returns the critical predecessor chain, phase delays, evidence class for every edge, and the first unresolved boundary.

**Evaluation.** Inject one controlled delay at a time: CPU descheduling before launch, host page fault, delayed memcpy, cross-stream event wait, graph dependency, device resource contention, and unrelated concurrent work. Compare the returned critical path with injected ground truth. Measure edge accuracy, root-cause top-1/top-3 accuracy, false causal edges, unresolved-path recall, and query latency.

**Academic value.** The work tests whether a partial-order representation with evidence classes produces more reliable causal explanations than timestamp stitching.

**Production value.** A profiler can tell an engineer where the delay entered the pipeline and which layer owns the next investigation.

**Failure condition.** If timestamp-only correlation reaches the same causal accuracy across workloads with overlapping streams and injected confounders, the graph machinery adds little value.

### 3. A host-device causality benchmark with adversarial ambiguity

**Gap.** Diagnosis accuracy alone does not reveal whether the causal explanation is correct, and ordinary benchmarks often make the root cause visually obvious.

**Mechanism.** Build a benchmark whose generator knows the true dependency graph and can make several non-causes look temporally plausible. Each test emits a ground-truth work graph containing host tasks, CUDA API calls, stream operations, graph nodes, memory copies, kernels, synchronization points, and controlled interference.

The benchmark should include paired cases:

| Pair | Same visible symptom | Different true cause |
| --- | --- | --- |
| late launch vs device queueing | kernel starts 5 ms late | host thread stall vs GPU-side wait |
| dependency vs overlap | two kernels overlap in time | explicit event edge vs independent streams |
| slow copy vs blocked consumer | consumer kernel starts late | PCIe transfer vs unrelated concurrent copy |
| missing record vs no edge | parent appears absent | telemetry loss vs genuinely independent work |
| graph replay vs direct launch | same kernel names repeat | reused graph node vs fresh API invocation |

A runner can inject CPU scheduler delay, page faults, memory pressure, stream waits, PCIe contention, GPU occupancy pressure, and record loss at known points. The evaluation then scores the profiler's reconstructed graph, not just its final text label.

**Delta.** Production profiling systems demonstrate that cross-layer diagnosis can work. This benchmark isolates whether the explanation remains correct when timestamps are intentionally misleading and local IDs are reused.

**Artifact.** An open workload generator, ground-truth graph format, trace corpus, and evaluator. It should support CUPTI-only traces, host-only traces, combined traces, and synthetic loss so different profilers can be compared under the same conditions.

**Evaluation.** Baselines are CUPTI-only correlation, host-only eBPF or perf tracing, nearest-timestamp stitching, and the proposed typed causal graph. Primary metrics are parent-edge precision/recall, critical-path fidelity, root-cause accuracy, false-confidence rate under loss, and overhead. Report results separately for instrumented and closed-source workload modes.

**Academic value.** The benchmark makes causal fidelity a measurable property instead of an anecdotal debugging success.

**Production value.** Tool builders can decide which extra tracing layer earns its overhead and can regression-test explanations when CUDA, drivers, or frameworks change.

**Failure condition.** If simple CUPTI correlation plus timestamps already reconstructs the ground-truth graph reliably in the adversarial cases, the benchmark will show that stronger cross-layer instrumentation is unnecessary.

## eBPF is useful here, but it should not own the whole design

Host-side eBPF can observe scheduler delays, process and thread transitions, page faults, network activity, and selected userspace or driver boundaries without requiring application SDK instrumentation. The [async eBPF causal profiler report](https://eunomia.dev/research/async-ebpf-causal-profiler/) showed why thread identity is insufficient even before a GPU enters the path.

The GPU boundary adds another runtime with its own dependency model and identifiers. Trying to force every device relation into an eBPF event schema would lose useful CUDA semantics. Conversely, a CUPTI-only profiler cannot see many host causes outside the CUDA runtime.

The practical architecture is therefore federated: keep the native correlation mechanisms that are already strongest at each layer, then join them through a small typed causal contract. eBPF is one strong host evidence source. CUPTI is one strong CUDA evidence source. The contract should survive replacement of either collector.

This also separates the question from device-side extensibility. The `gpu_ext` work shows that host-side eBPF cannot observe every device-side event and motivates programmable hooks inside the GPU. That is valuable when the missing cause is inside kernel execution. The causal contract in this report addresses a different boundary: how to preserve and test the relationship among host work, asynchronous submission, dependency waiting, and device activity before deciding that deeper device instrumentation is necessary.

## The first prototype does not need a new GPU runtime

A useful first implementation can stay deliberately small:

1. collect CUPTI runtime/driver, kernel, memcpy, memset, synchronization, stream, and graph activity;
2. enable queued/submitted kernel timestamps where supported;
3. add one host work token at an application or framework boundary when available;
4. use uprobes or another host tracer for scheduler and system evidence around CUDA calls;
5. build a partial-order graph from explicit stream/event/graph semantics and correlation IDs;
6. inject known host, dependency, and device delays;
7. score reconstructed parent edges and critical paths against ground truth.

Only if this prototype finds an unobservable causal boundary should the work add a new kernel, driver, framework, or device hook.

That ordering keeps the research question honest. The missing mechanism is not "collect more GPU events." It is **preserve enough causal structure that a profiler can distinguish a proven dependency from a nearby event, and admit when the evidence does not support a complete explanation.**

## What would change this conclusion?

Three results would weaken the case for a new causal contract.

First, if current CUPTI correlation IDs, external correlation, CUDA Graph IDs, and ordinary framework tracing already reconstruct request-to-kernel critical paths with high accuracy across multi-threaded, multi-stream, and graph-heavy workloads, a new namespace would mostly duplicate existing mechanisms.

Second, if production diagnosis only needs cohort-level anomaly localization, as in many differential debugging systems, exact per-operation causal edges may not justify their collection and analysis cost. A cheaper cross-layer statistical profiler would be the better design.

Third, if the proposed benchmark cannot create cases where timestamp stitching reaches the wrong causal explanation while explicit dependency tracking succeeds, then the practical gap is smaller than this report claims.

The next step is therefore not to build a large unified GPU observability platform. It is to build the ground-truth workload and ask whether current local correlation mechanisms can reconstruct the causal graph. The answer determines whether stronger host-device identity and dependency tracking are worth shipping.

## References

- NVIDIA, [CUPTI Activity API usage and correlation](https://docs.nvidia.com/cupti/main/main.html)
- NVIDIA, [CUPTI Activity API](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html)
- NVIDIA, [CUDA Programming Guide: Asynchronous Execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)
- Linux kernel documentation, [Uprobe-based Event Tracing](https://docs.kernel.org/trace/uprobetracer.html)
- Zheng et al., [SysOM-AI: Continuous Cross-Layer Performance Diagnosis for Production AI Training](https://arxiv.org/abs/2603.29235)
- Darzi et al., [Host-Side Telemetry for Performance Diagnosis in Cloud and HPC GPU Infrastructure](https://arxiv.org/abs/2510.16946)
- Zheng et al., [gpu_ext: Extensible OS Policies for GPUs via eBPF](https://arxiv.org/abs/2512.12615)
