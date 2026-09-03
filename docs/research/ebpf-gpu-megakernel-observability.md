---
date: 2026-09-03
title: "Can eBPF Make a GPU Megakernel Observable?"
description: "GPU megakernels fuse operators into one persistent kernel, hiding logical task boundaries from kernel tracing. This report proposes eBPF semantic task hooks."
tags:
  - Daily Report
  - eBPF
  - GPU
  - Megakernel
  - Observability
  - Runtime Systems
research_question: "What runtime interface can preserve logical task identity and expose low-overhead, late-bound observability inside persistent GPU megakernels after conventional kernel boundaries disappear?"
source_cutoff: 2026-09-03
status: daily-report
---

# Can eBPF Make a GPU Megakernel Observable?

A GPU inference service is slow for a few requests, but its CUDA timeline looks almost empty: one long-running kernel occupies the device. There is no sequence of attention, GEMM, communication, and sampling kernels to line up with the request. Those operations still happen, but the runtime has fused them into tasks scheduled inside one persistent megakernel.

That is the point of a megakernel. It removes host launch gaps and coarse kernel boundaries so fine-grained work can overlap. It also removes a boundary that many GPU profilers use as their basic unit of explanation.

This report argues that megakernel observability should not try to reconstruct all logical work from program counters after the fact. **The compiler and in-kernel scheduler already know the task graph. They should expose a small, versioned semantic hook interface, and a device-side eBPF runtime should let operators attach verified monitors to those hooks without recompiling the application.** The monitor can then aggregate task latency, queueing, dependency stalls, and request attribution near the event, while ordinary CUPTI or hardware sampling remains available for instruction-level evidence.

<!-- more -->

This is not another argument for tracing every GPU instruction. The earlier report on [GPU instrumentation safety](https://eunomia.dev/research/gpu-instrumentation-safety-contract/) already asked how injected probes can preserve program meaning and declare perturbation. The report on [host/device causality](https://eunomia.dev/research/gpu-host-device-causality/) asked how asynchronous CPU and GPU events can be connected. Here the missing object is different: **what is the observable execution unit after a compiler deliberately removes kernel-per-operator boundaries?**

## Megakernels deliberately erase the unit that kernel timelines expose

The conventional GPU execution model gives tooling a useful, if imperfect, structure. A host thread launches kernels and memory operations into streams. A trace can record the launch, kernel instance, start and end timestamps, stream, and surrounding API calls. NVIDIA's current [CUPTI documentation](https://docs.nvidia.com/cupti/) describes Activity records for CUDA APIs, kernels, and memory operations, plus PC Sampling and SASS-level mechanisms for lower-level evidence.

Megakernels change that structure on purpose.

[MPK, presented at OSDI 2026](https://www.usenix.org/conference/osdi26/presentation/cheng), transforms multi-GPU model inference into a single persistent megakernel. Its compiler lowers a tensor program into an SM-level task graph, and its in-kernel runtime schedules those tasks across SMs. The paper reports up to 1.7x lower end-to-end latency than conventional kernel-per-operator serving baselines in its evaluation. The important observability fact is the architecture: many logical operators and communication steps now execute as tasks inside one kernel instance.

[Event Tensor](https://arxiv.org/abs/2604.13327) pushes the same direction further for dynamic workloads. It represents completion dependencies among tiled tasks as first-class event tensors, supports shape-dependent and data-dependent task graphs, and lowers them into persistent kernels with static or dynamic in-GPU scheduling. In a Mixture-of-Experts layer, for example, runtime routing decisions can determine which task tiles update or wait on which events.

This creates a useful inversion. The compiler has *more* semantic structure than before, because it explicitly manipulates tasks, events, dependencies, symbolic shapes, and scheduling transformations. The external kernel timeline has *less* semantic structure, because all of that work may appear under one persistent kernel.

The problem is therefore not that the logical boundaries disappeared everywhere. They moved from the CUDA launch interface into the compiler IR and the in-kernel scheduler.

## PC sampling can locate hot instructions, but it does not name the logical task

CUPTI's PC Sampling API can sample warp program counters and scheduler state, including stall reasons. That remains useful inside a persistent kernel. If one instruction region is repeatedly stalled on memory or execution dependencies, PC sampling can reveal it.

But a PC answers a different question from an application task identity.

The same device function can execute for many requests, decoding steps, experts, tiles, or communication epochs. A dynamic scheduler can send the same code to different SMs under different dependency conditions. Two samples at the same PC can therefore belong to different logical tasks and have different causes. Conversely, one logical operator may be decomposed into many task types and PCs.

A profiler can try to reconstruct semantics by combining debug metadata, compiler IR, addresses, and runtime state. That works when the compiler/runtime and profiler are integrated. It is a weak portable contract for late-bound production diagnosis because a binary address is not a stable statement such as:

```text
request_generation = 1842
decode_step = 37
logical_op = attention_qk
tile = (head=12, block=6)
task_generation = 991733
wait_reason = dependency_event_481
```

The missing capability is not another stack trace. It is a stable way to ask the in-kernel scheduler what logical work it is executing.

## Existing megakernel runtimes already prove that task-level profiling is possible

A strong alternative to an eBPF interface is simply: let each megakernel compiler implement its own profiler.

That is not hypothetical. The [Mirage MPK repository](https://github.com/mirage-project/mirage/tree/mpk) exposes a `--profiling` mode that visualizes the execution timeline of each task, and its persistent-kernel API includes a `profiler_tensor`. MPK already has the task graph and scheduler state, so a compiler-owned profiler can record meaningful internal events much more directly than a generic tool that only sees the kernel boundary.

This is important counterevidence against overengineering. A new runtime layer is only justified if it provides something materially different from a compiler-specific profiler.

The useful difference is **late-bound programmability**. Production operators often discover the question after deployment: show only tasks from one request generation, count dependency waits longer than 20 microseconds, correlate an expert-routing imbalance with one communication phase, or sample only when an SLO is already at risk. Recompiling or enabling a broad built-in trace can be too expensive, and a fixed profiler cannot anticipate every query.

The [gpu_ext](https://arxiv.org/abs/2512.12615) work provides a plausible execution mechanism. It exposes GPU-driver hooks and a device-side eBPF runtime capable of executing verified policy logic inside GPU kernels. That does not solve megakernel observability by itself. A verifier can make a monitor safer, but it cannot invent task semantics that the megakernel compiler never exposed. The interesting composition is therefore:

1. the compiler/runtime exports semantic scheduler hooks;
2. the device-side eBPF runtime provides safe late-bound programs at those hooks;
3. the monitor emits bounded summaries or selected evidence rather than exporting every task event.

This makes eBPF central to the mechanism rather than an optional host-side tracer.

## The semantic hook should be smaller than the compiler IR

Exporting the whole megakernel IR to every observability tool would couple tools to compiler internals. Exporting only raw PCs loses the logical structure. A useful interface sits between them.

A first semantic hook ABI could expose a small set of events:

- task becomes ready;
- task is assigned to a worker or SM;
- task starts and completes;
- task waits on or releases a dependency event;
- task participates in a communication operation;
- request or decode-step generation advances;
- scheduler queue crosses a declared pressure threshold.

The context should contain stable identifiers and bounded metadata, not arbitrary compiler objects. For example, a task-class ID can map to a side table that describes the source operator, generated device function, tensor region class, and expected dependency classes. A request-generation token can distinguish reused request slots. A task-generation number can distinguish repeated execution of the same logical tile.

The ABI also needs a capability boundary. A monitoring program may be allowed to read task metadata and update private maps but not change scheduling. A scheduling policy may be allowed to return a bounded priority hint but not modify tensor memory. Those program types should be separate rather than hiding control behind an observability hook.

This is where an eBPF-like model is useful. Program type, attach point, context schema, map visibility, helper set, and verifier rules can define a narrow contract that is smaller than the compiler implementation but richer than a program counter.

## Where current work is still weak

### Kernel-level correlation loses the semantic denominator inside a persistent kernel

Current GPU tracing can correlate host APIs, kernel instances, streams, source locations, PCs, and hardware samples. That evidence remains necessary. A persistent megakernel can still make one kernel instance contain thousands or millions of logical task executions.

Without a task denominator, a statement such as "35% of samples are in this instruction region" does not say whether one task class is universally slow, one request generation is pathological, or a scheduler repeatedly starves a small subset of ready tasks.

A useful test must keep the same megakernel binary while changing only the logical task schedule or request mapping. If a profiler reports the same diagnosis for cases with different ground-truth task bottlenecks, kernel/PC identity is insufficient for that question.

### Compiler-native profilers are semantic but not a portable late-bound interface

MPK demonstrates that the compiler can visualize task timelines. Event Tensor makes task and dependency events first-class in its compiler representation. These are strong baselines, not gaps to ignore.

The remaining question is whether an operator can load a small new query after deployment without rebuilding the megakernel, enabling a full trace, or teaching an external profiler every compiler's internal format. Today there is no common task-observability contract across megakernel runtimes.

The simplest answer may be a common export format rather than eBPF. Any eBPF proposal has to beat that baseline on query flexibility, overhead, or deployment safety.

### Dynamic task graphs make identity and coverage harder than static operator names

In dynamic MoE or continuous batching, task existence and dependencies can depend on runtime data. Reusing a static operator label is not enough. A useful record must identify the logical generation of the task and the dependency state that made it runnable.

Coverage is also dynamic. A monitor may sample one task class, throttle under load, or skip events because its telemetry budget is exhausted. The query result needs the eligible-event denominator and loss state. Otherwise a low count can mean either "this rarely happened" or "the monitor rarely observed it."

This report assumes the prior [instrumentation-safety contract](https://eunomia.dev/research/gpu-instrumentation-safety-contract/) for resource perturbation. Megakernel observability adds a different requirement: semantic coverage over a changing task graph.

### Cross-GPU task semantics can disappear behind one local scheduler view

MPK includes multi-GPU computation and communication inside the megakernel. A local task can block because remote work has not produced the expected data or event. A device-local profiler that records only local worker state can misclassify the wait as local scheduler imbalance.

A semantic interface therefore needs enough distributed identity to connect a task generation to communication or remote dependency generations, without exporting a full distributed trace for every tile. The right evidence budget is unresolved.

## Promising directions with academic and production value

### 1. A versioned semantic hook ABI for megakernel schedulers

**Gap.** Megakernel compilers know task identity and dependencies, but external tools either see a single kernel or depend on compiler-specific profiling formats.

**Mechanism.** During lowering, the compiler emits a compact task schema and stable hook descriptors for scheduler events. Each hook context exposes bounded fields such as task class, task generation, request/decode generation, dependency class, worker/SM identity, and communication generation. A device-side eBPF program type can attach to these hooks and read only declared fields. The compiler may change its internal IR freely as long as it preserves or versions the exported ABI.

The ABI should explicitly distinguish observation hooks from control hooks. An observability program can aggregate and emit evidence but cannot modify scheduler state. A later scheduling extension can use a separate program type with a much narrower return contract.

**Delta from related work.** MPK's profiler already uses compiler-owned task semantics, while gpu_ext already provides verified device-side eBPF execution. The proposed contribution is the contract between those two layers: a portable semantic attach surface for dynamically loaded monitors, not a new megakernel compiler or another binary instrumentation framework.

**Artifact.** A prototype MPK or Event Tensor backend that exports task schemas and invokes a gpu_ext-style eBPF hook on selected scheduler transitions, plus a host loader that can attach and replace monitoring programs without rebuilding the model engine.

**Evaluation.** Measure hook cost, register/resource delta, attach latency, task coverage, and diagnosis accuracy across static and dynamic task graphs. Compare compiler-native profiling, PC sampling, always-on task logging, and eBPF hooks under the same telemetry budget.

**Academic value.** The core question is whether compiler-created execution semantics can become a stable systems ABI after traditional kernel boundaries disappear.

**Production value.** An operator can ask a new targeted question on a running inference service instead of enabling an expensive universal trace or deploying a rebuilt engine.

**Failure condition.** If task schemas change so frequently that a stable ABI either leaks the full compiler IR or loses the information needed for diagnosis, a cross-runtime semantic hook is the wrong abstraction.

### 2. Coverage-carrying task aggregation inside the megakernel

**Gap.** Exporting every task transition from a megakernel can recreate the telemetry and synchronization overhead that fusion was meant to remove. Sampling without a denominator can produce confident but misleading summaries.

**Mechanism.** Run the first aggregation stage in the device-side eBPF monitor. Per task class or request generation, maintain bounded counters and sketches for ready-to-start delay, execution time, dependency-wait class, queue depth, and selected hardware sample correlation. Each aggregation epoch also records eligible events, observed events, throttled events, lost records, and the monitor/program generation.

The host receives compact summaries and only escalates to selected raw events when a predicate fires. A query such as "which task class explains the p99 decode-step gap?" can therefore use semantic aggregates first and request a narrow trace second.

**Delta from related work.** CUPTI provides kernel, PC, and hardware evidence; compiler-native profilers can provide task timelines. The new property is programmable semantic aggregation at the internal scheduler boundary with explicit observation coverage, rather than a fixed trace format.

**Artifact.** A small library of verified task monitors and a coverage-aware result format that can join task summaries with CUPTI PC or PM samples without claiming that every hardware sample has a logical task identity.

**Evaluation.** Sweep task-event rates from small models to highly tiled MoE workloads. Hold total telemetry bandwidth constant and compare raw task tracing, fixed-rate sampling, compiler-native profiling, and coverage-carrying eBPF aggregation on root-cause accuracy, bytes exported, device overhead, and false-confidence rate.

**Academic value.** This tests whether semantic observability can be treated as an explicit on-device evidence budget rather than an all-or-nothing trace switch.

**Production value.** Always-on monitoring can remain cheap enough for latency-sensitive inference while retaining a path to deeper evidence when a rare failure appears.

**Failure condition.** If a small fixed set of compiler-maintained counters preserves the same diagnosis quality across workloads, programmable aggregation is unnecessary complexity.

### 3. A counterexample benchmark for observability after kernel fusion

**Gap.** Megakernel evaluations primarily ask whether fusion improves performance. Observability tools are rarely tested on paired cases where the same persistent kernel has different internal bottlenecks but nearly identical kernel-level timelines.

**Mechanism.** Build paired executions from one task graph and binary. Inject controlled faults at the task scheduler: delay one task class, skew MoE routing, postpone one dependency notification, oversubscribe one worker queue, delay one communication generation, or repeatedly starve a ready tile. Keep the outer kernel launch and total runtime as similar as practical while changing the known internal cause.

The ground truth comes from the compiler task graph plus the injected scheduler event. Competing tools receive the same overhead or telemetry budget and must name the affected logical task class and cause. Include a conventional kernel-per-operator build and a CUDA Graph build to show when ordinary kernel boundaries are sufficient.

**Delta from related work.** This is not a benchmark of megakernel speed. It treats fusion itself as an observability transformation and asks when kernel-level or PC-level evidence stops preserving the diagnosis.

**Artifact.** A CUDA-first benchmark with MPK/Event Tensor style task graphs, fault injection, known task/dependency identities, and adapters for CUPTI, compiler-native profiling, and semantic eBPF hooks.

**Evaluation.** Primary metrics are cause-identification accuracy, task-attribution accuracy, false confidence, telemetry bytes, runtime perturbation, and time to attach a new query. The strongest baseline is the compiler's own profiler, not a deliberately weak trace.

**Academic value.** The benchmark makes "megakernels are harder to observe" a falsifiable statement and shows exactly which semantic boundaries matter.

**Production value.** Compiler and observability teams can regression-test new fusion strategies without silently destroying the evidence operators need during incidents.

**Failure condition.** If CUPTI PC sampling plus ordinary compiler debug metadata identifies the injected task-level cause as reliably as semantic hooks under the same budget, the extra eBPF layer is not justified.

## What would change this conclusion?

Three results would weaken the case for a semantic eBPF hook interface.

First, megakernel compilers may converge on built-in task profilers that are cheap, dynamically filterable, and stable enough for production. MPK already shows that task-level profiling can be native to the runtime. If those profilers accept late-bound predicates and export explicit coverage, a separate programmable monitor adds little.

Second, hardware and vendor tooling may expose richer in-kernel semantic ranges. CUPTI already offers kernel tracing, PC sampling, SASS metrics, and range profiling. If future hardware can associate those samples with compiler-defined task IDs at negligible cost, the right standard could be a hardware/compiler metadata channel rather than eBPF.

Third, the counterexample benchmark may show that PCs and compiler metadata are sufficient. If the same binary's internal bottlenecks can be diagnosed reliably without scheduler hooks, then adding a new device runtime boundary would make the system more complex without improving the answer.

The current evidence supports a narrower conclusion: **megakernel observability should follow the semantic task boundary that the compiler created, not the kernel boundary it intentionally removed.** A device-side eBPF runtime is interesting because it can make that boundary late-bound and programmable, but only if it beats compiler-native profiling on real diagnostic questions under a strict overhead and coverage budget.