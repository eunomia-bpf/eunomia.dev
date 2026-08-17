---
date: 2026-08-17
title: "Where Should eBPF Run in a Heterogeneous System?"
description: "eBPF can now run in kernels, userspace, SmartNICs, and GPUs. The missing problem is choosing placement without changing state, safety, or effect semantics."
tags:
  - Daily Report
  - eBPF
  - Heterogeneous Systems
  - SmartNIC
  - GPU
  - Runtime Systems
  - Offload
research_question: "What mechanism should choose whether an eBPF policy runs in the kernel, userspace, NIC/DPU, GPU-adjacent, or device-side runtime when several targets are possible, and how can state, authority, verifier assumptions, memory visibility, and observability remain correct if placement changes?"
source_cutoff: 2026-08-17
status: daily-report
---

# Where Should eBPF Run in a Heterogeneous System?

Suppose one service receives packets on a SmartNIC, parses requests in userspace, moves tensors to a GPU, and keeps part of its working set behind a CXL fabric. An operator wants one policy to reject malformed traffic, account for expensive memory movement, and react when a GPU-side queue becomes congested.

There are now several plausible places to execute BPF logic. A packet policy can run in the Linux kernel or on a programmable NIC. A hot application hook can run in a userspace runtime such as [bpftime](https://github.com/eunomia-bpf/bpftime). GPU policy can run at host-side driver hooks or inside a device-side runtime such as the one explored by [gpu_ext](https://arxiv.org/abs/2512.12615). Recent work such as [fabric_ext](https://arxiv.org/abs/2607.26335) goes further and lowers one policy across GPU, driver/runtime, DPU/NIC, and CXL-side targets.

The obvious rule is to put computation close to the event. That rule is incomplete.

A policy may observe a packet most cheaply on the NIC but depend on state owned by the host. A GPU-side policy may avoid a host round trip but lose access to helpers, maps, or authority that exist only in the kernel. Moving one program can also move the meaning of a pointer, the visibility of a map update, the timing of a side effect, and the evidence available after a failure.

<!-- more -->

This report argues that **heterogeneous eBPF needs an execution-placement layer distinct from both the BPF ISA and a single-backend runtime contract**. The first question is whether a target can execute a program with the semantics it requires. The next question is where an eligible program should execute, which target should own its state and effects, and how a placement change can happen without silently changing behavior.

The safest first design is not transparent runtime migration. It is a static or slowly changing planner that makes placement constraints explicit, chooses among valid targets, and records why. Dynamic migration should be added only if measurements show that workload phases make static placement materially suboptimal.

This is the sixth report in the current [eBPF Runtime, Extensibility, and Composition series](https://eunomia.dev/research/). The first report in the series asked what a [portable userspace eBPF runtime contract](https://eunomia.dev/research/userspace-ebpf-runtime-contract/) must expose. This report starts one layer above that contract: if several backends satisfy it, which backend should own execution now?

## Portable instructions do not imply portable execution environments

The BPF instruction set is becoming more portable than the environment around it. [RFC 9669](https://www.rfc-editor.org/rfc/rfc9669.html) defines the BPF ISA and named conformance groups. A runtime must support the base group and can advertise additional groups, which gives compilers and runtimes a way to discover instruction capabilities.

That is useful, but an instruction group does not say which events a target can observe, which context fields a program may read, which helpers it may call, which memory it can reach, or which effects it may cause.

Linux itself demonstrates this separation. The [eBPF verifier documentation](https://docs.kernel.org/bpf/verifier.html) describes program-type-specific context access through `is_valid_access()` and program-type-specific function availability through `get_func_proto()`. A socket program and a tracing program can execute the same ISA while receiving different context and helper surfaces. The host environment is part of the semantics.

Hardware offload narrows the point further. Current Linux [`kernel/bpf/offload.c`](https://github.com/torvalds/linux/blob/master/kernel/bpf/offload.c) binds hardware-offloaded programs and maps to registered network devices and device operations. The device-bound program initialization path accepts `SCHED_CLS` and `XDP` program types, not an arbitrary BPF program plus a generic heterogeneous target. Linux therefore has real offload machinery, but it is intentionally tied to a specific subsystem and attachment model.

A generic placement layer cannot begin with "this device speaks eBPF." It needs a richer answer:

| Question | ISA conformance can answer it? | Placement needs it? |
| --- | --- | --- |
| Can this target execute the required instructions? | yes | yes |
| Can it observe the required event and context? | no | yes |
| Can it call the required helpers or kfunc-like operations? | no | yes |
| Where does the policy state live? | no | yes |
| Which memory domains are directly visible? | no | yes |
| Can the program deny, mutate, schedule, or only observe? | no | yes |
| Who authorizes the effect? | no | yes |
| How much crossing cost does the placement introduce? | no | yes |

The previous runtime-contract report proposed making backend capabilities testable. Placement adds an optimization and ownership problem over those capabilities. A target can be compatible and still be the wrong place to run a particular policy.

## Existing systems already show why target choice matters

The case for heterogeneous placement is not hypothetical. Different systems have already made different placement choices because event locality changes what is practical.

### hXDP moves XDP execution onto an FPGA NIC

[hXDP](https://www.usenix.org/conference/osdi20/presentation/brunella) showed that real XDP programs could execute on an FPGA NIC rather than on the host CPU. Its design used an optimizing compiler, an extended BPF ISA, a soft processor, and FPGA support for XDP maps and helpers. In its OSDI 2020 evaluation, the prototype used about 15% of FPGA resources, matched the packet-processing throughput of a high-end CPU core, and reported 10x lower packet-forwarding latency.

The result is not evidence that every BPF program belongs on a NIC. It shows the benefit of specialization when the event, state path, and required execution environment can be reproduced near the packet stream. hXDP had to build that environment explicitly. The target was valuable because the policy and event source were a good fit.

### Userspace execution removes a different boundary

A userspace runtime solves another locality problem. [bpftime](https://github.com/eunomia-bpf/bpftime) can execute BPF programs around userspace functions and other application-local events without routing every event through a kernel probe path. Its attach, map, helper, verifier, and compatibility machinery makes it more than an instruction interpreter.

That placement has access to application-local control points that a NIC does not have, but it also changes what the program can safely assume. Kernel-only helpers and kernel object access do not automatically exist in a userspace backend. A lower event cost does not make the host process a semantic substitute for the kernel.

### gpu_ext places policy inside the GPU boundary

[gpu_ext](https://arxiv.org/abs/2512.12615) starts from a failure of host-only placement. Its paper argues that host-side eBPF cannot observe several device-side events that matter for GPU memory placement, scheduling, and observability. The design exposes GPU-driver hooks and adds a device-side eBPF runtime so verified policy code can execute in GPU kernels. The paper reports throughput improvements of up to 4.8x and tail-latency reductions of up to 2x on its evaluated workloads.

The important systems point is not the maximum speedup. It is that the placement changes the available event boundary. A host program can be perfectly safe and still arrive too late or observe too little to implement the intended policy.

Together, these systems make a simple rule impossible. Kernel, userspace, NIC, and GPU execution are not interchangeable implementations of one abstract machine. Each target creates a different combination of event locality, memory visibility, verifier assumptions, state cost, and effect authority.

## "Run near the data" is only half of the objective

Moving a program closer to its event can reduce crossings, but the event is only one input to a policy decision. A useful placement model has at least five forms of locality.

**Event locality** asks where the trigger first becomes visible. Packet arrival favors the NIC or network stack. GPU allocation and queue events may favor the driver or device. A userspace function call favors the process itself.

**State locality** asks where the maps, tables, counters, sketches, or learned policy state already live. A per-queue counter can be device-local. A tenant policy table shared by networking, storage, and GPU work may need a host authority or an explicit replica protocol.

**Memory locality** asks which address spaces and memory domains are directly readable. Host virtual addresses, device global memory, NIC SRAM, and CXL-attached memory do not have one uniform coherence model. A pointer that is valid for one target can be meaningless for another.

**Effect locality** asks where a decision can actually be enforced. Observing a packet on the host is less useful if the intended effect must happen before DMA. Observing a GPU queue from userspace is less useful if the policy needs to affect device scheduling at a finer boundary.

**Authority locality** asks which target is allowed to make that effect. A device-local policy should not quietly grant an operation that host security would deny. Conversely, a host controller that delegates a bounded scheduling choice should not have to handle every fast-path event synchronously.

These localities can point in different directions. Consider a tenant-aware GPU service with a shared quota table on the host. Queue events are device-local, but the quota policy is host-owned. Three placements are plausible:

1. keep the entire decision on the host and pay for event crossings;
2. copy the quota table to the device and accept a consistency protocol;
3. split the policy so the device enforces a locally valid budget while the host periodically changes the budget generation.

The third option is often more realistic than pretending that one target should own everything. A placement layer therefore needs to choose both **where code runs** and **where authoritative state remains**.

## The strongest adjacent design already spans several devices

A serious placement proposal has to account for [fabric_ext](https://arxiv.org/abs/2607.26335), not claim that cross-device eBPF compilation is unexplored.

fabric_ext targets GPU-CXL fabrics. Its central abstraction is a semantic movement graph that describes data movement using properties such as source and destination, bytes, stride, reuse, ordering, ownership, and transformations. The compiler lowers the graph into per-device eBPF programs, verifier obligations, consistency-classed BPF maps, and backend artifacts. It can place pieces of one policy at GPU hooks, driver/runtime hooks, DPU/NIC hooks, CXL switches, or near-memory targets.

That design is strong evidence that one source policy can be decomposed across heterogeneous execution sites while keeping state and verification in the compiler model. It also narrows the remaining gap.

The question in this report is broader in one dimension and deliberately less ambitious in another. It asks about **placement of arbitrary event-driven BPF policy across eligible targets**, not only data-movement transformations in a GPU-CXL fabric. At the same time, it does not assume that the system can automatically split any program. A first implementation can require explicit policy boundaries and choose among whole-program or developer-declared components.

The useful comparison is therefore:

- fabric_ext asks how a semantic data-movement graph should be lowered across a known fabric topology;
- this report asks what target contract, objective function, state ownership rule, and evidence are needed to choose execution placement for BPF policies whose events may come from networking, application code, GPU runtime, driver, or device.

If the fabric_ext abstraction turns out to generalize cleanly to those event classes, then a separate placement abstraction is unnecessary. That is a good falsifier, not a problem to hide.

## Placement changes can become semantic changes

The difficult part begins when a policy moves after deployment.

Imagine a program that increments a map counter and denies an operation after a threshold. On the host, the map update may be immediately visible to other host programs. On a NIC or GPU, the same logical map could be local, mirrored, cached, or accessed through a host round trip. Moving the program without specifying the map's ownership can change when the threshold becomes visible.

The same problem appears with time and ordering. A host-side event may be observed after a device has already accepted work. A device-side policy may act before host tracing records enough context to explain the decision. An asynchronous queue can reorder observations even when each target is internally correct.

This is why transparent migration is a dangerous first goal. A system should not advertise "move BPF anywhere" until it can define what must remain invariant.

A practical invariant set is smaller:

- each policy generation names the target that owns execution;
- each state object names its authoritative owner and permitted replicas;
- each target proves the required context, helper, memory, and effect capabilities before activation;
- a placement transition has a generation boundary, so old and new state are not mixed accidentally;
- observability records the placement and generation that produced each externally relevant decision.

This connects directly to the earlier report on [transactional eBPF upgrades](https://eunomia.dev/research/stateful-ebpf-transactional-upgrade/). Placement migration is an upgrade where the new generation also changes execution domain. The transition therefore needs at least as much care as replacing programs and maps on one host.

## Where current work is still weak

### There is no general placement objective for BPF policy

Existing systems tend to choose a target first and optimize inside that target. Linux hardware offload starts from XDP or classifier semantics on a network device. hXDP starts from FPGA NIC execution. gpu_ext starts from GPU policy. fabric_ext starts from a GPU-CXL movement graph.

What is missing is a common way to express that a policy has several valid homes and to compare them using the same constraints. "Closest to the data" does not capture shared-state traffic, authority boundaries, verifier differences, or the cost of losing observability.

A useful objective would not collapse everything into one weighted score. It should first reject semantically invalid placements, then compare valid placements on a small Pareto set such as event crossing cost, state crossing cost, latency, throughput, host load, and evidence completeness.

### Cross-target state has no common failure model

BPF maps are a familiar programming model, but a map on a GPU, NIC, userspace process, and Linux kernel does not automatically mean one coherent object. Replicating every update can destroy the performance gain that motivated offload. Keeping all state local can make a split policy incorrect.

The missing contract is state-specific rather than runtime-wide: which target is authoritative, which replicas may be stale, which updates commute, when a generation fence is required, and what happens when a device disappears during migration.

This is an area where the right answer may be deliberately limited. Many useful policies could be built from target-local counters plus a small host-authoritative configuration table. If that covers most real workloads, a general distributed-map system would be unnecessary complexity.

### Current evaluations rarely answer the placement question directly

An accelerator paper usually compares its chosen target with a host baseline. That establishes whether the accelerator is useful for its intended workload. It does not establish whether a different target would have been better, or which workload property predicts the winner.

A placement mechanism needs a benchmark in which the **same logical policy and expected effects** can run in more than one location. Without that, a planner can optimize its own cost model without evidence that it chooses the right system design.

### Observability becomes part of the placement contract

Offloading policy also offloads part of the explanation. If a device makes a deny, scheduling, memory-placement, or transformation decision, an operator needs enough evidence to identify the policy version, target, input class, and resulting effect.

Shipping every event back to the host defeats fast-path placement. Keeping only local device logs makes cross-target incidents hard to reconstruct. The placement problem therefore needs an evidence budget, not only an execution budget.

## Promising directions with academic and production value

### 1. A target manifest and placement planner

**Gap.** A runtime can report that it supports BPF, while leaving the planner to infer event support, helper availability, memory domains, authority, and crossing costs from documentation.

**Mechanism.** Give each target a machine-readable manifest. It advertises instruction conformance groups, supported event and attach classes, context versions, helper or kfunc-like capabilities, map/state classes, readable memory domains, permitted effects, authority constraints, and a measured cost model for event and state crossings. A policy separately declares required events, state, effects, and latency bounds. The planner first filters for semantic validity, then chooses among valid placements using measured costs.

The first version should place whole programs or explicitly declared components. Automatic arbitrary program partitioning is a later problem.

**Delta.** The earlier userspace-runtime contract answers whether one backend can satisfy required semantics. This planner compares several satisfying backends. Unlike fabric_ext's movement graph, the target manifest is intended to cover event-driven policy outside one fabric domain.

**Artifact.** A schema plus adapters for Linux kernel BPF, bpftime-style userspace execution, one NIC/FPGA target, and one GPU-side target; a small constraint solver; and a tool that explains why placements were rejected or selected.

**Evaluation.** Run policy workloads that have at least two legitimate placements: packet filtering/accounting, a hot userspace service hook, GPU memory or scheduling control, and one cross-device policy. Compare manual expert placement, always-host, always-near-event, and planner output. Measure semantic-invalid placements rejected, event and state boundary crossings, p50/p99 decision latency, throughput, host CPU cost, device utilization, and planner overhead. Compare with fabric_ext where a workload fits its movement-graph model.

**Academic value.** The experiment tests whether a compact target contract can predict the Pareto frontier across execution domains instead of encoding one accelerator's assumptions.

**Production value.** Operators get a pre-deployment answer to "where should this policy run here?" together with an explanation rather than a backend-specific feature checklist.

**Failure condition.** If useful target manifests become unrelated per-backend catalogs, or static expert placement matches the planner across workloads, the general planner adds little value.

### 2. Generation-scoped state ownership and migration

**Gap.** Moving execution can change map visibility and update ordering even when the program code remains unchanged.

**Mechanism.** Every state object declares an owner and one of a small number of consistency modes, for example target-local, host-authoritative, device-authoritative, or replicated read-mostly. A placement transition creates a new policy generation. The controller verifies the new target, transfers or reinitializes only state permitted by the declared mode, establishes a fence, then activates the new generation. Rollback retains the previous generation and its authoritative state until the new placement is committed.

This mechanism should avoid pretending that all maps are globally coherent. State that cannot be migrated safely simply makes a placement ineligible.

**Delta.** Transactional eBPF upgrade already needs generation boundaries for programs, links, and maps. Heterogeneous migration adds explicit memory-domain ownership and crossing semantics to that transition.

**Artifact.** A generation controller, a state-description schema, and state bridges for a small set of map classes shared between host and one device backend.

**Evaluation.** Use counters, sketches, quota tables, and read-mostly policy tables under network and GPU workloads. Inject device reset, controller crash, stale replica, and mid-transition failure. Measure lost or duplicated updates, stale-read rate, transition pause, transfer bandwidth, rollback correctness, and the performance difference between local, host-authoritative, and replicated state.

**Academic value.** The design asks which state semantics are both useful and implementable across heterogeneous BPF runtimes without building a general distributed shared-memory system.

**Production value.** A policy can move only when its state contract says the transition is safe, which turns migration failure from an implicit correctness risk into a rejected operation.

**Failure condition.** If most useful BPF state is naturally target-local or cheaply host-authoritative, cross-target migration may be too rare to justify a generic mechanism. In that case the planner should keep state immobile and move only stateless policy components.

### 3. A ground-truth placement and provenance benchmark

**Gap.** Current evaluations show that individual offload targets can work well, but there is little common evidence for when each target is the right one.

**Mechanism.** Build benchmark workloads around an event graph, state objects, expected externally visible effects, and a set of eligible targets. For every run, record the placement generation, boundary crossings, state movement, and the reason for each placement decision. A ground-truth checker validates the final effects independently of the runtime's own logs.

The benchmark should include both stable and phase-changing workloads. The latter matters because it can tell us whether runtime re-placement is worth implementing at all.

**Artifact.** A harness with reference policies for packet processing, userspace function control, GPU resource policy, and one cross-fabric data-movement case; adapters to the same target set used by the planner.

**Evaluation.** Compare throughput, p99 latency, host CPU, device utilization, transferred bytes, state staleness, failed semantic checks, and observability completeness across placements. Then test whether planner decisions predict the best valid placement and whether phase changes create enough benefit to justify switching generations.

**Academic value.** The benchmark turns "run near the data" from a slogan into a falsifiable systems hypothesis and exposes which workload features actually determine placement.

**Production value.** Teams gain a regression suite for target selection before they enable offload or migration in production.

**Failure condition.** If policies are so target-specific that the same logical effect cannot be expressed across at least two backends, a generic placement benchmark is the wrong abstraction. Target-specific evaluation should remain the norm.

## What I would build first

I would not start with a distributed eBPF scheduler that moves programs continuously among CPU, NIC, and GPU. That design combines too many unknowns: runtime compatibility, state transfer, cost prediction, target failure, and explanation.

A smaller first system would do three things:

1. describe target capabilities and policy requirements in a machine-readable form;
2. reject invalid placements and choose one placement before activation;
3. record placement generation, state owner, and measured crossing costs so the decision can be replayed later.

Then run the same workload in several placements. If the best target changes across stable workload phases and the benefit is larger than transition cost, add generation-scoped migration for the small set of state types that can move safely. If the winner is stable, keep placement static and spend the complexity budget elsewhere.

That sequencing matters. Heterogeneous eBPF already has enough evidence that specialized placement can deliver real performance and new control points. What it does not yet have is evidence that transparent dynamic movement is broadly necessary.

## What would change this conclusion?

The argument for a general placement layer becomes weaker under three results.

First, if target-specific systems cover almost all useful deployments and policies rarely have two semantically valid execution homes, then a generic planner has little to choose. Better backend-specific tooling would be enough.

Second, if fabric_ext's semantic movement graph or a similar existing abstraction can represent networking, userspace, GPU policy, device effects, and their state contracts without losing important semantics, then that abstraction should be generalized rather than creating a competing placement model.

Third, if cross-target state and evidence traffic dominate the cost in realistic workloads, then moving policy close to the event may save less than it costs. The right architecture would keep authority and state on the host and use devices only for narrow stateless predicates or pre-processing.

The current evidence supports a narrower conclusion: **eBPF does not have one best execution location. The missing mechanism is a testable placement decision over targets whose event, state, memory, authority, and verifier semantics differ.** Start with explicit static placement and measured evidence. Earn dynamic migration with benchmarks rather than assuming it.