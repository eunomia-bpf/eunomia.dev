---
date: 2026-08-31
title: "Can GPU Utilization Tell You Whether Another Workload Will Fit?"
description: "GPU utilization measures recent activity, not safe co-residency. This report derives an allocatability contract for admitting another workload without SLO risk."
tags:
  - Daily Report
  - GPU
  - Scheduling
  - Resource Allocation
  - Runtime
research_question: "When is GPU utilization sufficient to admit another workload, and what additional evidence is required to predict safe co-residency?"
source_cutoff: 2026-08-31
status: daily-report
---

# Can GPU Utilization Tell You Whether Another Workload Will Fit?

A GPU dashboard says a device is 35% utilized. Another inference service is waiting in a queue. The obvious scheduler decision is to put the second service on that GPU instead of opening another device.

That decision can be wrong in two opposite ways. The running workload may report low average utilization while occupying the registers, shared memory, workqueues, HBM capacity, or burst bandwidth that the incoming workload needs. Or the device may look highly utilized while still having a spatial partition, a complementary resource profile, or predictable idle phases that another workload can safely use.

The missing concept is not a better utilization percentage. It is **allocatability**: whether a particular incoming workload can be admitted onto a particular GPU, at this moment, under an explicit correctness and performance budget.

<!-- more -->

This report argues that utilization should remain an observability signal, not become an admission certificate. A practical GPU runtime should separate three questions:

1. **Does the incoming work physically fit?** Registers, shared memory/LDS, resident blocks or waves, device memory, SM partitions, workqueues, and other finite resources impose hard constraints.
2. **If it fits, how much interference is plausible?** DRAM, cache, interconnect, execution pipelines, power, and temporal overlap create soft contention that depends on both workloads.
3. **How strong is the evidence?** Interval averages, stale profiles, and missing counters should reduce confidence rather than silently become spare-capacity estimates.

The distinction is useful for cluster schedulers, model-serving runtimes, GPU serverless systems, and local multi-process workloads. It also advances a different boundary from the earlier Daily Reports on [GPU memory placement](https://eunomia.dev/research/gpu-memory-placement-evidence/) and [instrumentation safety](https://eunomia.dev/research/gpu-instrumentation-safety-contract/). Memory placement asks where data should live. Instrumentation safety asks whether the observer perturbs what it measures. Here the question is whether observed activity is enough evidence to admit another workload.

## A utilization percentage is an average of activity, not a resource inventory

NVIDIA's current [DCGM profiling documentation](https://docs.nvidia.com/datacenter/dcgm/latest/learn/modules/profiling.html) is unusually explicit about what its counters mean. `PROF_SM_ACTIVE` is the fraction of time at least one warp was active on an SM, averaged over all SMs and over the measurement interval. The same 20% value can arise from one fifth of the SMs being active for the whole interval or all SMs being active for one fifth of the interval. `PROF_SM_OCCUPANCY` separately measures resident warps relative to the hardware maximum. `PROF_DRAM_ACTIVE` measures memory-interface activity. DCGM also warns that a high value in one field does not by itself prove that a workload is compute- or memory-bound.

That is exactly the information loss that makes admission from one headline number unsafe. The scheduler wants to know whether a specific second workload can coexist. The metric only says how much activity the first workload produced after averaging over space and time.

Consider two devices that both report 40% SM activity during the last second:

- Device A ran a wide kernel on nearly every SM for 400 ms and was idle for 600 ms.
- Device B ran a narrow persistent kernel on 40% of the SMs for the full second.

Those devices have the same average activity but very different opportunities for another kernel. A latency-sensitive request arriving now may prefer the spatial slack on B, while a batch job that can wait for periodic idle windows may fit A. The percentage itself cannot tell us which execution shape produced it.

Occupancy is richer but still not an admission answer. DCGM defines occupancy as resident warps relative to the maximum and notes that higher occupancy is not inherently better. A memory-bandwidth-bound kernel and a compute-bound kernel can have the same occupancy while leaving very different contention surfaces for a colocated workload.

The production implication is simple: **utilization describes the work that happened; allocatability is a counterfactual about work that has not been admitted yet.** Turning one into the other requires a model of the incoming work and the resources it would share.

## Hard fit is discrete, and small resource changes can cross a boundary

GPU execution resources are not a continuous pool that can be inferred from “60% unused.” NVIDIA's current [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) explains that register use, threads per block, and allocation granularity determine how many blocks can be resident on an SM. Two block sizes using the same registers per thread can produce different occupancy because resource allocation happens at discrete boundaries.

AMD exposes the same phenomenon through different terminology. Current [ROCm workload-optimization guidance](https://rocm.docs.amd.com/en/docs-7.2.4/how-to/rocm-for-ai/inference-optimization/workload.html) derives occupancy from VGPR allocation, LDS allocation, and waves per workgroup. Its MI300X example shows how register allocation rounds to a hardware granularity and can abruptly reduce the number of resident waves. The newer [ROCm Compute Profiler documentation](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/docs-10.0.0/conceptual/rdna/wgp.html) likewise reports VGPRs and LDS as occupancy-limiting launch resources rather than as utilization percentages.

These are hard-feasibility facts. If a block cannot acquire enough registers or shared memory to become resident alongside existing work, recent SM activity is irrelevant. Likewise, enough free HBM bytes at one instant do not guarantee that a second model is safe if the first workload has an elastic working set or a memory-reclamation protocol that cannot meet the incoming service's latency bound.

This is why “free capacity” should be represented as a vector rather than one scalar. At minimum, a runtime may need to reason about:

```text
SM or CU partition / placement
resident block or wave capacity
register allocation class
shared memory / LDS per block
workqueue or connection resources
HBM reservation and reclaimability
DRAM / cache pressure sensitivity
PCIe / NVLink demand
power or thermal headroom
```

Not every backend exposes all of these directly, and some are contention signals rather than reservable resources. That is a reason to expose uncertainty and backend capability, not a reason to collapse the state into utilization.

## Even explicit SM partitions are not a complete concurrency guarantee

CUDA 13.2 makes the distinction between resource ownership and actual concurrency visible in the API. The current [CUDA Runtime execution-context documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION__CONTEXT.html) lets a program create Green Contexts from specified device resources, including disjoint SM partitions and workqueue configurations.

But NVIDIA explicitly states that kernels launched in Green Contexts with disjoint SM partitions are **not guaranteed** to run concurrently or make forward progress because other resources can introduce dependencies. Balanced non-overlapping workqueues improve the chance of avoiding interference, but stronger guarantees require more resource types than the interface currently exposes.

This is a useful design clue. Even when a scheduler has something stronger than utilization, such as a spatial SM allocation, one resource dimension does not define the whole sharing contract. An allocatability decision should therefore distinguish:

- **hard fit:** the runtime can prove that known reservable constraints are satisfiable;
- **interference risk:** shared, non-reservable, or temporally varying resources may still violate the performance budget;
- **unknown:** the backend cannot observe or reserve a required dimension strongly enough to justify a confident admission.

Treating `unknown` as a first-class outcome is important. Otherwise a scheduler gradually turns “the API does not expose this” into “there is no constraint.”

## Existing sharing systems already manage more than utilization

Production and research GPU-sharing systems repeatedly end up adding workload-specific control because aggregate activity alone is not enough.

[SIRIUS](https://www.usenix.org/conference/atc25/presentation/wang-jiali), published at USENIX ATC 2025, colocates inference and training by letting latency-sensitive inference use GPU resources while training consumes leftovers. The system does not simply look for a low-utilization GPU. It actively changes training memory consumption, performs explicit memory reclamation and handover, and uses SLO-aware reallocation to avoid thrashing.

[KRYPTON](https://www.usenix.org/conference/atc25/presentation/zhang-shulai), also at ATC 2025, argues for spatio-temporal GPU sharing with performance guarantees rather than fixed resource allocation alone. Its kernel-space interception layer exists partly because orchestration and isolation are properties of the sharing mechanism, not of a utilization counter.

The older but large-scale [AntMan](https://www.usenix.org/conference/osdi20/presentation/xiao) deployment at Alibaba co-designed the scheduler and training framework so memory and computation could be dynamically scaled before jobs consumed “spare” resources. Again, spare capacity becomes useful only after the runtime knows how one workload can yield resources to another.

A more recent preprint, [Roomie](https://arxiv.org/abs/2607.16784), makes a complementary point for model serving. It argues that aggregate resource profiles miss temporal kernel overlap and instead predicts pairwise interference from per-kernel resource configurations with an occupancy-based analytical model. Its reported evaluation shows fewer SLO violations than the compared systems while keeping comparable goodput. The exact design may not generalize outside model serving, but the direction is important: admission needs the incoming workload's execution shape, not just the resident workload's average utilization.

These systems do not imply that one universal allocator can replace workload-aware scheduling. They imply something narrower: utilization alone is too lossy to serve as the correctness boundary for sharing.

## Where current work is still weak

### Feasibility and interference are often collapsed into one score

A scheduler may estimate a “GPU load” score from utilization, memory use, and perhaps occupancy. That is convenient for ranking devices, but it mixes two different failure modes.

A hard-feasibility failure means the incoming work cannot obtain the required resources or isolation boundary. A soft-interference failure means it can execute but violates a latency, throughput, or slowdown budget after sharing begins. These should produce different explanations and different recovery actions.

The discriminating experiment is simple: construct workload pairs where the second kernel is physically admissible but strongly contends for DRAM, and pairs where recent utilization is low but register/shared-memory or partition constraints prevent useful co-residency. A single scalar should fail to distinguish the two; a useful runtime contract should not.

### The scheduler usually knows too little about the incoming workload

Telemetry is mostly about the workload already running. Admission is about a workload that has not started. Without a declared or learned resource envelope for the incoming task, even perfect measurement of the resident workload leaves the scheduler solving only half the problem.

Model-serving systems can profile known kernels offline. General GPU runtimes have a harder problem: JIT-generated kernels, dynamic shapes, CUDA Graph variants, and user-defined extensions can change resource demand. A practical system needs both a static or profile-derived envelope and a way to mark it stale when the binary, shape family, or phase changes.

### Average counters lose the overlap structure that determines contention

An interval-average 40% DRAM-active value can represent continuous moderate traffic or short saturation bursts. The incoming workload experiences those cases differently. The same problem applies to SM activity, NVLink traffic, and power.

A scheduler should therefore record enough phase or burst information to answer an admission question, not merely increase sampling frequency everywhere. Higher-frequency averages can still miss causal overlap while increasing monitoring cost.

### Failure after admission is not always fed back into the capacity model

Many systems react to an SLO violation by moving or throttling a workload, but the failed pair is valuable evidence. If a runtime admitted workload B beside A under a given resource state and immediately observed a 3× latency increase, that result should update the future allocatability estimate for the same kernel/shape/resource regime.

Without that loop, the system can repeatedly make the same “low utilization means spare capacity” mistake.

## Promising directions with academic and production value

### 1. An allocatability certificate instead of a utilization threshold

**Gap.** The scheduler has activity counters but no machine-readable object that says why a particular incoming workload fits, which resource limits the placement, or what evidence remains unknown.

**Mechanism.** Give each incoming workload a versioned resource envelope and evaluate it against a current device-resource snapshot. The envelope can combine static kernel attributes, offline profiles, memory reservations, runtime phase information, and declared SLOs. The result is an **allocatability certificate**:

```text
workload_version
hard_fit: yes | no | unknown
limiting_resource
resource_reservations
shared_contention_dimensions
evidence_timestamp
profile_coverage
predicted_slowdown_range
confidence
expiry_or_revalidation_trigger
```

The certificate deliberately separates `hard_fit` from predicted slowdown. CUDA Green Context resource descriptors or MIG-style partitions can strengthen the hard side where available; kernel attributes and occupancy calculations can reject impossible placements. DCGM or ROCm counters then inform the soft side without pretending to be reservations.

**Delta from related work.** Existing GPU schedulers often keep placement scores or model-specific profiles internally. The proposed artifact makes the admission proof, missing evidence, and limiting resource portable and inspectable across runtime backends.

**Artifact.** A small runtime service plus CUDA and ROCm adapters. A CLI such as `gpu-fit <device> <workload-profile>` returns admit/reject/unknown together with the exact limiting constraint rather than only a score.

**Evaluation.** Build workload pairs that sweep register pressure, shared memory/LDS, HBM footprint, DRAM intensity, tensor/FP pipelines, and temporal burstiness. Compare utilization-only, utilization-plus-memory, occupancy-based, and certificate-based admission. Measure false admits, false rejects, decision latency, achieved throughput, and SLO violations.

**Academic value.** The systems question is whether GPU co-residency can expose a useful proof boundary between discrete feasibility and statistical interference.

**Production value.** Operators get an explainable reason for why a nominally idle GPU cannot take another service, and schedulers can distinguish “try another device” from “collect more evidence.”

**Failure condition.** If a simple utilization-plus-memory threshold predicts safe admission as accurately as the richer certificate across architectures and workload classes, the extra contract is unnecessary.

### 2. Two-stage admission with a bounded interference probe

**Gap.** Hard resource checks cannot prove that two admissible workloads will meet a latency or throughput budget when they share DRAM, cache, interconnect, execution pipelines, or power.

**Mechanism.** Make admission two-stage. Stage one rejects impossible placements from hard constraints. Stage two estimates interference only for candidates that fit. It can combine a historical kernel-pair model with a short bounded canary run on a reserved slice, low-priority stream, or otherwise controlled execution window.

The canary is not allowed to become an unbounded benchmark. It has an explicit probe budget and rollback rule:

```text
max_probe_time
max_requests_exposed
max_slowdown_on_resident_workload
required_confidence
rollback_on_slo_violation
```

If the runtime cannot obtain enough evidence inside that budget, the answer remains `unknown` and the scheduler falls back to isolation or a different device.

**Delta from related work.** Roomie predicts model-serving interference from kernel configurations; SIRIUS and AntMan actively reshape colocated work. The proposed mechanism is a generic admission boundary: prove hard fit first, then spend a bounded amount of runtime evidence only on the uncertain interference dimensions.

**Artifact.** A pluggable interference estimator with a conservative canary executor and a persistent pairwise evidence cache keyed by kernel or workload version, shape family, device type, and resource regime.

**Evaluation.** Compare no probe, fixed offline profiling, analytical prediction, canary-only admission, and hybrid admission under phase changes and unseen workload pairs. Measure false-admit rate, SLO impact caused by probing, adaptation time, and how quickly repeated pairs become decidable without a new probe.

**Academic value.** This turns online profiling into an admission-control experiment with an explicit risk budget rather than an always-on measurement stream.

**Production value.** Unknown workload pairs can be tested conservatively instead of being rejected forever or admitted optimistically from a low utilization number.

**Failure condition.** If bounded canaries perturb resident work enough to erase their predictive value, or if interference changes faster than cached evidence can be reused, online probing is the wrong mechanism.

### 3. A counterexample benchmark for “spare GPU” claims

**Gap.** Utilization-based schedulers are often evaluated on average throughput or cluster utilization. Those metrics do not reveal whether the admission signal itself is trustworthy.

**Mechanism.** Build paired scenarios with the same headline metric but different ground-truth allocatability:

| Same dashboard signal | Hidden difference | Ground-truth question |
| --- | --- | --- |
| 40% SM activity | spatially narrow persistent work vs temporally bursty full-GPU work | can a latency-sensitive kernel start immediately? |
| 50% occupancy | register-limited vs DRAM-saturating resident kernel | does the new kernel violate its slowdown budget? |
| 40% HBM used | reclaimable training buffers vs pinned/non-evictable working set | can the model reserve memory within the deadline? |
| disjoint SM sets | shared workqueue or other dependency | do both workloads make progress concurrently? |
| equal DRAM average | smooth traffic vs short saturation bursts | is p99 latency preserved during overlap? |

The benchmark records a ground-truth answer from controlled co-runs and asks each scheduler to admit or reject before seeing that answer. The primary metrics are false admission, false rejection, p99 slowdown violation, time-to-admission, and explanation accuracy for the limiting resource.

**Delta from related work.** Existing colocation systems demonstrate that specific sharing mechanisms improve utilization or SLO compliance. This benchmark evaluates the *evidence used to claim spare capacity* across systems, including deliberately adversarial pairs where average telemetry is ambiguous.

**Artifact.** A CUDA-first suite with corresponding ROCm cases where practical, workload-profile manifests, reproducible phase schedules, and adapters for DCGM, ROCm Compute Profiler, MPS/Green Contexts, and higher-level schedulers.

**Evaluation.** The key ablation progressively adds information: GPU utilization only; utilization plus memory; plus occupancy; plus static kernel resources; plus phase information; plus pairwise interference evidence. The result should show which evidence actually reduces false decisions and where additional telemetry stops helping.

**Academic value.** It makes “GPU has spare capacity” a falsifiable systems claim rather than a dashboard interpretation.

**Production value.** Scheduler teams can regression-test a new GPU architecture, driver, or sharing mode before trusting the same admission thresholds used on older hardware.

**Failure condition.** If the adversarial pairs do not change admission outcomes under realistic schedulers, or if a small fixed set of counters fully separates them, the benchmark should narrow to that smaller evidence contract.

## What would change this conclusion?

The strongest alternative is that operators do not need a new runtime abstraction at all. Perhaps a carefully chosen vector of existing metrics, such as SM activity, occupancy, DRAM activity, free HBM, plus static kernel resource attributes, already predicts safe co-residency well enough. Roomie's recent result is some evidence in that direction for model serving: an occupancy-based analytical model can be effective when the workload family and kernel profiles are known.

That alternative should be tested directly. If a simple, vendor-supported metric vector predicts hard fit and SLO compliance across CUDA and ROCm devices, changing phases, unseen kernel pairs, and different sharing modes with low false-admit rates, an explicit certificate and bounded canary layer would mostly add plumbing.

The conclusion also weakens if GPU platforms expose stronger resource reservations. CUDA's current Green Context documentation already acknowledges that disjoint SMs alone do not guarantee concurrency because other resources remain shared. If future APIs make the relevant resources reservable with strong forward-progress and performance isolation guarantees, much of allocatability can become a direct resource-allocation query instead of an inference problem.

Until then, “35% utilized” is useful operational telemetry, but it is not an answer to “will this next workload fit?” A scheduler should say what fits, what may interfere, and what it still does not know.
