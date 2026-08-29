---
date: 2026-08-29
title: "Can a GPU Runtime Place Memory Well With Only Page Faults?"
description: "GPU memory oversubscription turns page placement into a policy problem. This report asks what evidence a runtime needs before migrating or evicting UVM pages."
tags:
  - Daily Report
  - GPU
  - Unified Memory
  - Memory Management
  - Runtime
research_question: "What evidence should a GPU runtime expose and preserve so that memory migration, prefetch, eviction, and remote-access decisions remain explainable and testable under memory oversubscription?"
source_cutoff: 2026-08-29
status: daily-report
---

# Can a GPU Runtime Place Memory Well With Only Page Faults?

Suppose two GPU jobs together touch 120 GB of data on a GPU with 80 GB of HBM. Unified memory lets both jobs keep one virtual address space even though some pages must live in host DRAM. Eventually a kernel touches a page that is not in HBM, the GPU faults, and the runtime has to make room.

The fault answers one question: this page was needed and was not resident here. It does not answer the harder questions. Which resident page should leave? Is the faulting page worth migrating, or should the GPU access it remotely? Will the evicted page be needed again in 200 microseconds? Is the current access part of a short phase, a repeatedly reused tensor, or a task that is about to be descheduled?

Those questions decide whether oversubscription costs a few transfers or collapses into migration thrashing. Recent GPU memory systems increasingly answer them with more evidence than faults alone: sampled HBM access activity, compiler-derived object semantics, application-object phases, or future scheduling information. The common problem is no longer simply how to move a page. It is how the runtime knows enough to justify a placement decision.

<!-- more -->

This report argues for an **evidence-carrying GPU memory placement contract**. A runtime should preserve which observations caused a migration, prefetch, eviction, replication, or remote-access decision; how current those observations are; and which stronger application or scheduler intent the decision is trying to satisfy. That would make placement policies comparable across different observability mechanisms instead of treating every policy as a private driver heuristic.

This is an adjacent-systems report, not an eBPF-centered report. eBPF-like instrumentation could help collect host-side or runtime evidence, and [bpftime's GPU work](https://eunomia.dev/bpftime/documents/gpu/) is relevant to programmable instrumentation, but the core problem exists independently of eBPF.

## Unified memory already separates addressability from residency

CUDA Unified Memory gives CPU and GPU code a common addressable allocation while the runtime manages where backing pages reside. The current [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html) is explicit that `cudaMemAdviseSetPreferredLocation` is a performance hint, not a residency guarantee. `cudaMemPrefetchAsync` may move a region toward a specified processor, and later accesses or other hints may move it again.

The current [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html) exposes an especially useful warning about policy semantics. A preferred GPU location can override the driver's normal page-thrashing response. Without the preference, repeatedly bouncing pages may eventually be pinned in host memory; with a GPU preferred location, those pages can continue to migrate. The application has expressed intent, but that intent changes the driver's policy rather than creating a hard placement contract.

Linux Heterogeneous Memory Management makes the lower layer equally dynamic. The current [Linux HMM documentation](https://docs.kernel.org/mm/hmm.html) describes migration between system memory and device-private memory using `migrate_vma_*()`. The driver chooses which pages actually migrate, updates device mappings, and handles races where individual pages fail or lose the migration race. A virtual range can therefore remain valid while the physical placement of its pages changes over time.

AMD exposes a similar model. [HIP managed memory](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/unified_memory.html) can migrate pages between host and device when HMM and recoverable page faults are available. The exact capabilities depend on GPU architecture, kernel support, and XNACK configuration.

Across these interfaces, addressability is portable in a way residency is not. The application can often keep using the same pointer, while the runtime keeps making placement decisions underneath it.

## Recent systems keep adding evidence because faults are too weak

A page fault is excellent evidence of immediate demand. It is poor evidence of future reuse.

The 2026 ISCA paper [Observability-aided GPU Memory Oversubscription](https://www.csa.iisc.ac.in/~arkapravab/papers/ISCA26_ObservUVM.pdf) shows the consequence inside NVIDIA UVM. The CPU-side driver sees faults for pages outside HBM but normally cannot directly observe accesses to pages that are already resident in HBM. Its default eviction order is therefore based on migration recency, which can evict a page that is still actively used. The paper repurposes hardware access counters to sample activity in HBM-resident regions and builds ObservUVM, which moves eviction and prefetch policies into userspace. Across fourteen applications, its evaluated observability-aided policies report a 34% geometric-mean speedup over the UVM baseline.

That result is useful for a reason beyond the speedup. The extra information changes which page should be evicted. The fault stream itself did not contain that information.

Other systems add different evidence. [SUV](https://www.csa.iisc.ac.in/~arkapravab/papers/MICRO24_SUV.pdf), published at MICRO 2024, uses compiler-inferred access semantics plus runtime information to decide which objects or object regions deserve scarce HBM and when to prefetch them. [OASIS](https://yueqiwang42.github.io/assets/pdf/papers/OASIS_HPCA25.pdf), from HPCA 2025, observes that one placement policy is not best for every object or every phase of the same object in a multi-GPU program, and uses object-aware runtime behavior to choose among migration and duplication strategies.

The 2026 HPCA paper [ARIADNE](https://doi.org/10.1109/HPCA68181.2026.11408564) stays inside the UVM abstraction but derives a runtime sharing-degree signal, pipelines fault handling, and dynamically chooses between GPU memory and zero-copy placement. Its [artifact](https://zenodo.org/records/17830000) makes the driver-level implementation available for reproduction.

A different direction appears in [MSched](https://arxiv.org/abs/2512.24637). Instead of treating each fault as an isolated surprise, MSched couples GPU task scheduling with memory management. Kernel launch information and a known scheduling timeline let it predict working sets and prepare memory before a context switch. Its reported gains over demand paging become very large under the evaluated multi-task oversubscription workloads.

These systems disagree on the best source of knowledge, but they agree on the underlying diagnosis: immediate faults are not enough. Placement improves when the runtime knows something about resident-page activity, object meaning, execution phase, or future demand.

The site's earlier report on [page-level eBPF memory attribution](https://eunomia.dev/research/page-level-ebpf-memory-attribution/) asked how to connect memory cost back to the application object that caused it. The problem here is different. Even if attribution is perfect, the GPU memory manager still needs to decide what should remain in HBM next.

## Observability itself has a cost and a failure mode

More evidence is not automatically better. ObservUVM deliberately samples because tracking every HBM access at full bandwidth is impractical. NVIDIA [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) can record Unified Memory CPU and GPU page faults, but its documentation warns that collecting those fault traces can add up to 70% overhead in testing.

Static information has a different failure mode. Compiler-derived access patterns can be precise for code the compiler understands, but closed-source libraries, data-dependent indexing, and runtime phase changes can invalidate the prediction. Object-level policies improve semantics but need stable object identity across allocators and libraries. Scheduler knowledge is powerful in a controlled multitasking runtime but weaker when kernels arrive from independent processes or external services.

So the design problem is not "collect every signal." A production runtime needs to know which signal supports which decision, and when that signal is stale, sampled, missing, or contradicted by another source.

## Where current work is still weak

### Placement evidence has no common semantics

A GPU fault says that an address was demanded while non-resident. An access-counter notification says that a sampled region crossed an activity threshold. Static analysis predicts future accesses from code structure. Object-aware systems attach behavior to a logical allocation. A task scheduler can expose when a process is likely to run again.

All of these can influence the same eviction decision, but they do not share a common evidence model. That makes it difficult to compare policies or to combine them without hard-coding one system's assumptions into another.

A material test is to present several policies with the same workload and the same fixed observability budget, then ask whether they can explain each migration in terms of evidence available before the decision. If two policies report the same throughput but one repeatedly acts on stale or contradictory evidence, the difference should be visible.

### Placement hints do not expose whether the runtime satisfied the intent

CUDA intentionally defines memory advice as hints. That preserves implementation freedom, but it also leaves a control-plane gap for production systems. An application can request a preferred location or issue a prefetch, yet there is no portable service-level object saying "this 8 GB region should stay GPU-resident through this inference phase unless pressure exceeds X, and if that cannot be honored, report why."

The missing capability is not a stronger version of `cudaMemAdvise` for every application. It is an optional contract above vendor hints that records desired placement, allowed degradation, and the runtime's observed compliance.

The test is a workload with two classes of data, one latency-sensitive and one spillable, under controlled pressure. A contract-aware runtime should preserve the requested priority or explicitly report that it could not. A hint-only baseline may silently violate it.

### Policy evaluation rarely measures decision quality directly

Most GPU memory papers reasonably optimize application runtime, page-fault count, migration traffic, or throughput. Those metrics do not tell us whether a policy made good decisions for the right reason.

A policy can get lucky on one access pattern. It can also reduce faults by over-prefetching and spend extra PCIe bandwidth that becomes harmful when another workload shares the link. A shared benchmark needs ground truth for future reuse and a fixed evidence budget so it can measure decision regret, not only final runtime.

### Cross-vendor portability stops at the point where policy needs meaning

CUDA UVM, HIP managed memory, and Linux HMM all support forms of shared addressing and migration, but their hardware signals, fault behavior, coherence options, and policy hooks differ. A portable runtime cannot simply rename vendor events and assume they mean the same thing.

The useful abstraction is therefore not one universal eviction algorithm. It is a small vocabulary for evidence, placement intent, action, and uncertainty, with adapters that preserve vendor-specific facts rather than erasing them.

## Promising directions with academic and production value

### 1. Evidence-carrying placement decisions

**Gap.** Current memory managers can use faults, access sampling, object semantics, or scheduling predictions, but the reason for an individual placement action is usually buried inside one driver or policy implementation.

**Mechanism.** Give every managed virtual region a lifetime-scoped region ID and generation. Whenever the runtime changes placement, emit a compact decision record:

```text
region_generation
virtual_range
action = migrate | evict | prefetch | remote-map | replicate | keep
evidence = [fault, sampled_access, object_phase, kernel_prediction, schedule]
evidence_age
coverage_or_sampling_rate
pressure_state
policy_generation
confidence
```

The record does not claim that every signal is equally strong. A fault can be exact about one access while a sampled-access estimate is probabilistic. A scheduler prediction can expire. A compiler claim can identify a region but miss data-dependent accesses. The schema preserves those distinctions.

For high-frequency decisions, the driver can aggregate records by policy generation and retain exemplars for surprising or high-cost migrations. Operators can then ask why a page moved without enabling full memory-access tracing.

**Delta from related work.** ObservUVM separates mechanism from policy and supplies sampled access evidence. SUV, OASIS, and MSched each add richer semantics. The proposed layer standardizes the evidence-to-decision boundary rather than proposing another eviction heuristic.

**Artifact.** A trace schema, CUDA/HIP/HMM adapters, and a query tool that reconstructs `why-moved <region>` from placement decisions. The existing [bpftime GPU runtime work](https://github.com/eunomia-bpf/bpftime) could consume such records for experimentation, but the schema should remain independent of bpftime.

**Evaluation.** Run oversubscribed scientific kernels, graph workloads, DNN inference, LLM inference, and mixed multi-process workloads. Compare UVM fault-only traces, sampled-access policies, compiler/object-informed policies, and schedule-aware policies. Measure record overhead, unexplained decisions, stale-evidence rate, and whether postmortem queries identify the actual reason for migrations.

**Academic value.** This turns the boundary between GPU memory observability and placement policy into an explicit object that can be compared across mechanisms.

**Production value.** A runtime or cloud operator can distinguish "the application exceeded HBM" from "the policy evicted a still-hot region" without collecting every memory access.

**Failure condition.** If decision records do not improve diagnosis or policy comparison beyond existing fault and migration traces, the extra metadata is not worth retaining.

### 2. Placement intent with observable compliance

**Gap.** Existing advice APIs express useful preferences, but a higher-level scheduler cannot tell whether the runtime honored an application's phase-specific memory priority.

**Mechanism.** Add an optional runtime object above vendor-specific advice. A region generation can declare:

- preferred residence set, such as GPU 0 HBM or host DRAM;
- access modes that may use remote mapping instead of migration;
- a migration deadline or phase boundary;
- eviction priority relative to other region generations;
- maximum tolerated remote-access or migration budget;
- an expiry condition tied to a kernel, graph, request, or scheduler epoch.

The memory manager maps this intent onto CUDA advice, HIP advice, HMM migration, or its own policy. If the request cannot be satisfied because of capacity, topology, coherence, or another higher-priority region, the runtime records a degraded state and reason rather than pretending the hint succeeded.

The important detail is generation. A tensor buffer or virtual range may be reused for a different phase. Old placement intent must expire with the logical lifetime instead of sticking to an address forever.

**Delta from related work.** CUDA and HIP already expose placement and prefetch hints; MSched adds scheduling knowledge; OASIS attaches policy to objects and phases. The proposed contract makes the requested outcome and observed compliance explicit across these mechanisms.

**Artifact.** A userspace controller and small region-intent API implemented first on NVIDIA UVM, with a HIP/HMM compatibility prototype. The controller would expose both best-effort hints and a stronger "report degradation" mode without requiring a new correctness guarantee from existing hardware.

**Evaluation.** Use paired regions with different latency sensitivity under 110%, 150%, and 250% HBM subscription. Include multi-GPU peer access, CPU-GPU ping-pong, phase changes, and concurrent jobs. Measure intent satisfaction, migration bytes, remote-access bytes, tail kernel delay, throughput, and control overhead against plain UVM and static advice.

**Academic value.** The research question is whether memory placement can be treated as an explicit resource contract rather than an invisible side effect of faults.

**Production value.** Inference and training runtimes can express that a KV-cache working set, communication buffer, or imminent batch deserves HBM more than cold model state, while still receiving a truthful failure signal under pressure.

**Failure condition.** If simple static advice achieves the same tail latency and migration cost across phase-changing and multi-tenant workloads, the extra contract is unnecessary.

### 3. A counterexample benchmark for GPU placement evidence

**Gap.** Existing evaluations make performance differences visible but rarely force two policies to make a decision from deliberately ambiguous evidence.

**Mechanism.** Build workload pairs that preserve one observable signal while changing the correct placement action:

| Pair | Signal held similar | Hidden fact that changes the best action |
| --- | --- | --- |
| repeated faults | fault stream | one page is reused next, the other is dead |
| equal access counts | sampled hotness | one region's accesses are imminent, the other's are far in the future |
| same allocation size | object metadata | one phase touches 5%, another touches 95% |
| same current residency | HBM state | one task is about to be scheduled out |
| same preferred location | advice | one job has a strict latency budget, the other is spillable |

The harness knows future references and task order, so it can compute an oracle placement under the same HBM and PCIe constraints. It then reveals evidence incrementally: fault-only, sampled access, object/phase semantics, then scheduler knowledge.

**Delta from related work.** ARIADNE, ObservUVM, SUV, OASIS, and MSched demonstrate useful policy mechanisms. The benchmark isolates the marginal value of each evidence class and makes hidden assumptions fail on purpose.

**Artifact.** An open trace corpus, replayable oversubscription workloads, oracle solver, and evaluator. The harness should support NVIDIA UVM first and preserve a vendor-neutral trace format so HIP/HMM experiments can be added later.

**Evaluation.** Primary metrics are application runtime, fault stall time, migration and remote-access bytes, useful-prefetch ratio, decision regret versus the oracle, false-confidence rate, and observability overhead. Every policy receives the same memory, link, and evidence budget.

**Academic value.** The benchmark asks a general systems question: how much extra information is required before an online resource manager can make a materially better placement decision?

**Production value.** Runtime developers can choose whether an extra profiler, compiler analysis, or scheduler integration earns its complexity on their workload instead of adopting it because a paper reports a speedup elsewhere.

**Failure condition.** If additional evidence barely reduces regret or end-to-end cost over a fault-only policy, simpler UVM behavior should remain the default.

## What would change this conclusion?

The evidence today supports a narrow conclusion: GPU memory placement under oversubscription benefits from information that a demand fault alone cannot provide, and recent systems obtain that information from several incompatible layers. It does not prove that one universal placement policy or one universal signal should replace vendor UVM heuristics.

Three results would weaken the case for an evidence-carrying contract. First, a broad reproduction could show that modern default UVM already approaches the best specialized policies once current drivers and hardware are used. Second, sampled HBM observability, compiler semantics, and scheduler knowledge might improve disjoint workload classes so rarely that a common contract has little practical reuse. Third, the metadata and control path needed to expose decisions could cost more than the migrations it helps avoid.

The strongest next experiment is therefore not another isolated eviction heuristic. It is a fixed-budget comparison where the same placement controller receives progressively richer evidence and the benchmark measures how often that extra evidence changes a decision that matters.

## References

- NVIDIA. [CUDA Programming Guide: Unified Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html), CUDA 13.x documentation, accessed 2026-08-29.
- NVIDIA. [CUDA Runtime API: Memory Management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html), accessed 2026-08-29.
- NVIDIA. [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html), Unified Memory page-fault tracing, accessed 2026-08-29.
- Linux kernel documentation. [Heterogeneous Memory Management](https://docs.kernel.org/mm/hmm.html), accessed 2026-08-29.
- AMD. [HIP Unified Memory Management](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/unified_memory.html), accessed 2026-08-29.
- Pratheek B, Khushit Shah, Arkaprava Basu. [Observability-aided GPU Memory Oversubscription](https://www.csa.iisc.ac.in/~arkapravab/papers/ISCA26_ObservUVM.pdf). ISCA 2026.
- Hyunkyun Shin, Seongtae Bang, Hyungwon Park, Daehoon Kim. [ARIADNE: Adaptive UVM Management for Efficient GPU Memory Oversubscription](https://doi.org/10.1109/HPCA68181.2026.11408564). HPCA 2026. [Artifact](https://zenodo.org/records/17830000).
- Yueqi Wang et al. [OASIS: Object-Aware Page Management for Multi-GPU Systems](https://yueqiwang42.github.io/assets/pdf/papers/OASIS_HPCA25.pdf). HPCA 2025.
- Pratheek B, Guilherme Cox, Jan Vesely, Arkaprava Basu. [SUV: Static Analysis Guided Unified Virtual Memory](https://www.csa.iisc.ac.in/~arkapravab/papers/MICRO24_SUV.pdf). MICRO 2024.
- Weihang Shen, Yinqiu Chen, Rong Chen, Haibo Chen. [MSched: GPU Multitasking via Proactive Memory Scheduling](https://arxiv.org/abs/2512.24637). arXiv:2512.24637, 2026.
