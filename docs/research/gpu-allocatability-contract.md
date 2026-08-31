---
date: 2026-08-31
title: "Can GPU Utilization Tell a Runtime Whether Another Kernel Will Fit?"
description: "GPU utilization describes recent activity, not whether a new kernel can co-reside safely. This report develops a candidate-aware allocatability contract."
tags:
  - Daily Report
  - GPU
  - Runtime
  - Scheduling
  - Observability
research_question: "What evidence should a GPU runtime use to decide whether a specific new kernel or task can safely co-reside with work already running?"
source_cutoff: 2026-08-31
status: daily-report
---

# Can GPU Utilization Tell a Runtime Whether Another Kernel Will Fit?

A GPU is showing 45% utilization, so a scheduler places another latency-sensitive kernel on it. The number looks comfortable: more than half of the device appears idle. Yet the new kernel starts late, or slows the existing job badly, or a synchronization-heavy workload stops making progress.

The problem is that "how busy was this GPU?" and "can this particular work fit now?" are different questions. Utilization is usually a retrospective measurement over a time window. Admission is a prospective decision about a candidate whose register use, shared memory, block shape, synchronization behavior, memory demand, and scheduling constraints may be very different from the work already running.

This distinction matters as GPUs become shared runtime targets rather than single-job accelerators. MPS, CUDA Green Contexts, MIG, inference multiplexing, communication/computation overlap, and cluster schedulers all create cases where a runtime needs to decide whether another piece of work can make progress without destroying an existing service-level objective.

<!-- more -->

This report argues that a shared GPU runtime should expose a **candidate-aware allocatability contract** instead of treating one utilization percentage as free capacity. The contract does not need to predict exact performance. It needs to answer a narrower and more useful question: given the candidate's resource footprint and the device's current resource state, which co-residency guarantees are justified, which are only probabilistic, and which cannot be made at all?

The question is distinct from the earlier report on [GPU memory placement](https://eunomia.dev/research/gpu-memory-placement-evidence/), which asks what evidence justifies migrating or evicting pages under oversubscription. It is also distinct from the report on [GPU instrumentation safety](https://eunomia.dev/research/gpu-instrumentation-safety-contract/), which asks whether an observer changes the kernel it measures. Here the decision comes before execution: whether another workload should be admitted onto the same physical GPU resources in the first place.

## Utilization measures activity, not a reservation

NVIDIA's current Fleet Intelligence documentation describes `DCGM_FI_PROF_SM_ACTIVE` as the ratio of cycles in which an SM has at least one warp assigned during the sampling window. `DCGM_FI_PROF_SM_OCCUPANCY` reports resident warps relative to the theoretical maximum, and `DCGM_FI_PROF_DRAM_ACTIVE` reports how often the device-memory interface was active. These are useful profiling signals because they tell an operator what kind of work consumed the device recently.

They do not by themselves reserve resources for the next kernel.

The distinction becomes obvious with time averaging. A measured SM activity of 20% can come from broad work that uses most SMs for one fifth of a sampling interval, or from narrower work that continuously occupies a smaller portion of the machine. Both histories can produce a similar aggregate number while presenting very different opportunities to launch another kernel at a specific moment.

Occupancy adds another dimension but still does not make admission candidate-independent. NVIDIA's current [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) explains that residency depends on finite registers and shared memory as well as block and warp limits. Register allocation is granular enough that two kernels using the same registers per thread can reach different occupancy depending on block size. The [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/pdf/cuda-programming-guide.pdf) likewise exposes per-SM limits for registers, shared memory, resident threads, blocks, and other execution resources, and occupancy calculation requires the resource use of the kernel being considered.

That last point changes the scheduler's question. "This GPU has 35% average occupancy" is a device observation. "Kernel B can place two blocks per SM beside kernel A without exhausting registers or shared memory" is a candidate-conditioned feasibility statement.

## CUDA now exposes the difference directly through resource partitioning

The current [CUDA Green Contexts documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/green-contexts.html) gives a concrete production example. Kernel A can occupy all available SM resources and delay later latency-sensitive kernel B. A Green Context can reserve a specific set of SMs for B, guaranteeing that A cannot consume those SM resources even when A could otherwise expand across the device.

The same documentation contrasts this with MPS. By default, MPS clients contend for available SM resources. An active-thread-percentage limit restricts how much of the GPU an MPS client may use, but the actual SMs can vary over time. Green Contexts instead bind a context to specific provisioned SMs. Starting with CUDA 13.1, MPS can also use static partitioning.

This is stronger than a utilization heuristic. The runtime turns capacity from an inferred percentage into an explicit resource boundary.

The guarantee still has scope. The Green Context documentation says that reserved SMs allow the latency-sensitive kernel to start without waiting for the other kernel to release SMs, barring other resource constraints. A scheduler therefore still has to reason about more than one scalar. Registers, shared memory, cluster scheduling requirements, memory bandwidth, copy engines, and synchronization can determine whether nominal SM headroom is useful headroom.

## Progress guarantees make bad admission decisions more than a performance bug

For ordinary independent kernels, a poor co-location decision may show up as latency or throughput loss. Synchronization-heavy GPU programs can make the failure sharper.

The current [NVSHMEM documentation](https://docs.nvidia.com/nvshmem/api/latest/using.html) states that when multiple processing elements share one GPU under MPS, full synchronization and collective support requires the sum of their active-thread percentages to remain at or below 100%. When the sum exceeds 100%, CUDA cannot guarantee that all PEs assigned to that GPU can run simultaneously, which can lead to deadlock in point-to-point synchronization or collective operations.

This is an important counterexample to the idea that utilization alone describes spare capacity. A runtime may see historically low activity and still need an explicit simultaneous-progress condition before admitting another participant in a collective.

Recent work on computation/communication overlap reaches the same boundary from performance rather than correctness. The June 2026 preprint [Resource-aware Computation-Communication Overlap for multi-GPU ML Workloads](https://arxiv.org/abs/2606.09200) deliberately shapes compute-kernel occupancy by adding shared-memory pressure, leaving enough on-chip resources for communication kernels to make progress. Across A40, A100, H100, and MI250X systems, the authors report up to 25.5% lower total execution time. The useful mechanism is not "use the GPU harder." It is "leave the right resources available for the other work."

Earlier production systems already showed why schedulers want to exploit this headroom. [AntMan](https://www.usenix.org/conference/osdi20/presentation/xiao) co-locates deep-learning jobs and dynamically scales memory and compute resources, reporting 42% higher GPU-memory utilization and 34% higher computation-unit utilization in Alibaba's multi-tenant cluster without compromising its fairness target. The open question is how a modern runtime should represent the admission evidence itself, especially now that hardware and software expose more partitioning choices.

## The missing abstraction is candidate-aware headroom

The practical mistake is to model free GPU capacity as one device-level number. A better model starts with a candidate and asks whether the currently available resources can satisfy that candidate's execution requirements.

For a kernel launch, the candidate description can include:

- registers per thread and the allocation granularity that matters on the target architecture;
- static and dynamic shared memory per block;
- threads, warps, blocks, and cluster or cooperative-launch requirements;
- expected HBM footprint and whether the task requires memory to remain resident;
- expected DRAM, copy-engine, or interconnect pressure when these are material;
- stream priority or partitioning context;
- whether forward progress depends on peer kernels or processing elements being resident at the same time.

The supply side is also structured. It may include specific SM partitions, current resident-resource headroom, memory reservations, Green Context or MIG ownership, MPS limits, and a freshness bound on sampled telemetry.

The result should not be a fake precise probability. A useful admission interface can return a small set of explicit outcomes:

```text
GUARANTEED_FIT       reserved resources satisfy the candidate contract
CONDITIONAL_FIT      feasible under stated bandwidth/interference assumptions
BEST_EFFORT          telemetry suggests headroom but no progress guarantee exists
REQUIRES_REPARTITION resources exist only after preemption or partition change
NO_FIT               a hard resource or progress constraint is violated
UNKNOWN              evidence is stale or the backend cannot establish the property
```

This turns a scheduler decision into something that can be inspected after an incident. If a latency-sensitive kernel was admitted as `BEST_EFFORT` because the runtime only had time-averaged utilization, the operator can see that the guarantee was weak. If it was admitted as `GUARANTEED_FIT` against an explicit SM partition and still missed its SLO, the investigation should move to bandwidth, synchronization, launch latency, or another assumption in the contract.

## Where current work is still weak

### Telemetry and admission are usually separate interfaces

GPU monitoring exposes activity, occupancy, memory, clocks, and throttling. Runtime APIs expose kernel resource use and partitioning controls. Schedulers often join these two worlds with local heuristics such as "admit below 60% utilization."

What is missing is a machine-readable admission record that connects a specific candidate's requirements to the resource evidence used for the decision. Without that record, it is hard to tell whether a bad placement came from stale telemetry, an incorrect resource model, unmodeled interference, or a scheduler policy mistake.

A material experiment would replay candidate launches against identical average-utilization traces but different spatial and temporal resource layouts. A useful admission contract should distinguish cases that a scalar threshold treats as equivalent.

### Time averages hide spatial and phase fragmentation

SM activity and occupancy are valuable because they summarize a large device cheaply. The same compression removes information a co-residency decision may need. A burst that occupies nearly all SMs for short periods and a persistent narrow kernel can produce similar averages while offering different launch opportunities and tail latency for a second kernel.

The missing evidence is not necessarily per-cycle tracing. A runtime could retain coarse per-partition or per-epoch headroom, plus the age and variance of that evidence. The experiment should compare how much admission accuracy improves as the representation moves from one aggregate percentage to a small headroom distribution.

### Resource feasibility and interference are often collapsed into one prediction

Registers, shared memory, and partition ownership can establish a hard feasibility boundary. DRAM contention, cache interference, power limits, and scheduler interactions often affect slowdown continuously rather than as a yes/no condition. Mixing both into one opaque score makes failures difficult to diagnose.

A better interface should separate hard admission constraints from performance-risk estimates. The runtime can prove that the candidate has enough resident resources while still stating that its bandwidth-interference risk is high. If a learned performance model is used, it belongs in the conditional part of the decision rather than replacing hard resource checks.

### Progress-sensitive workloads need stronger guarantees than independent kernels

NVSHMEM's MPS guidance shows that simultaneous residency can be required for correctness and forward progress, not merely for speed. General GPU schedulers rarely carry this requirement as a first-class property of a candidate workload.

A useful test should include collectives, producer/consumer kernels, persistent kernels, and other workloads that can block while waiting for peers. The admission mechanism should reject or repartition a placement when it cannot guarantee the required participants can make progress together.

## Promising directions with academic and production value

### 1. A candidate-conditioned allocatability certificate

**Gap.** Existing utilization telemetry describes the device, while admission depends on both device state and the exact kernel or task being placed.

**Mechanism.** Add a runtime query that takes a candidate resource manifest and returns an allocatability certificate. The manifest contains the candidate's registers, shared memory, block and cluster shape, memory residency needs, partition constraints, and progress dependencies. The backend combines this with current resource ownership and a timestamped headroom snapshot.

The certificate records the admitted partition, hard constraints checked, assumptions left conditional, evidence age, and one of the explicit outcomes above. A CUDA backend can use compiler/runtime resource metadata, Green Context or MPS partition state, occupancy calculations, and selected DCGM/CUPTI observations. Other GPU backends can implement the same public contract using native resource descriptors.

**Delta.** CUDA occupancy APIs estimate how one kernel maps onto an SM, and Green Contexts provide explicit SM partitions. The proposed layer connects those mechanisms to a multi-workload admission decision and preserves the evidence as an inspectable result.

**Artifact.** A small runtime library and scheduler plugin exposing `can_admit(candidate, device_state)`, plus a JSON certificate format and a replay tool for post-incident analysis.

**Evaluation.** Compare utilization thresholds, occupancy thresholds, the candidate-conditioned certificate, and an oracle built from controlled launch experiments. Workloads should sweep register pressure, shared memory, block shape, Green Context/MPS partitions, and memory pressure. Measure unsafe-admission rate, unnecessary rejection, start-latency error, throughput loss, and certificate-generation overhead.

**Academic value.** The general question is how much of heterogeneous-resource admission can be expressed as a portable, falsifiable contract rather than a scheduler-specific score.

**Production value.** Cluster and inference schedulers can make aggressive co-location decisions while retaining a reason that operators can inspect when latency or throughput changes.

**Failure condition.** If simple occupancy plus utilization thresholds achieve the same safe-admission precision and SLO outcomes across the same workloads, the certificate adds little value.

### 2. A temporal and spatial headroom ledger

**Gap.** A single average loses whether free capacity existed on specific partitions and whether it persisted long enough for a new kernel to use it.

**Mechanism.** Maintain a compact headroom ledger over short epochs. Instead of recording every hardware event, the runtime stores a bounded summary such as available SM partitions, minimum and percentile resident-resource headroom, DRAM/copy-engine pressure ranges, and the age of each observation. Phase transitions create new epochs when resource shape changes materially.

The ledger is not a new profiler trace. Its output is deliberately tailored to admission queries: "for the last 10 ms, at least this much headroom was continuously available on these resources, with this uncertainty." A candidate certificate can then ask whether its required horizon overlaps a suitable epoch.

**Delta.** DCGM and related tooling summarize activity for performance analysis. The proposed representation preserves just enough temporal and spatial structure to make prospective admission decisions without keeping full traces.

**Artifact.** A headroom summarizer fed by runtime launch metadata and existing device telemetry, plus synthetic and production replay traces that expose average-equivalent but allocatability-different states.

**Evaluation.** Construct paired workloads with the same average SM activity and occupancy but different burstiness, SM partitioning, resource fragmentation, and phase length. Compare admission precision and tail latency as the ledger budget is reduced from detailed epochs to one aggregate metric. Include an ablation that removes spatial identity while preserving the same number of bytes.

**Academic value.** This asks what minimum sufficient representation of recent device state supports reliable prospective scheduling.

**Production value.** A scheduler can reduce conservative idle capacity without paying for continuous fine-grained traces.

**Failure condition.** If spatial and temporal summaries do not improve admission decisions beyond cheap rolling averages at a fixed telemetry budget, the ledger is unnecessary.

### 3. An allocatability counterexample benchmark

**Gap.** GPU schedulers are often evaluated on aggregate utilization and job throughput, which can hide placements where the same headline utilization leads to opposite admission outcomes.

**Mechanism.** Build paired scenarios that intentionally preserve similar monitoring summaries while changing the resource property that controls co-residency:

| Pair | Similar headline metric | Different hidden condition |
| --- | --- | --- |
| bursty vs persistent work | average SM active | broad short bursts vs narrow continuous occupancy |
| low- vs high-register kernel | average utilization | resident-block headroom |
| shared-memory-light vs heavy | average occupancy | per-block shared-memory feasibility |
| independent vs collective peers | SM utilization | simultaneous-progress requirement |
| partitioned vs contended contexts | device utilization | guaranteed SM ownership |
| compute-light vs bandwidth-heavy | SM activity | DRAM/interconnect interference risk |

Each test asks one concrete question: should candidate B be admitted now under a stated SLO or progress requirement? The benchmark records an oracle from controlled execution, then scores each scheduler's admission decision rather than rewarding high utilization alone.

**Delta.** Systems such as AntMan show the value of co-location, and recent overlap work shows the value of deliberately reserving on-chip resources. This benchmark focuses on the decision boundary that precedes those mechanisms: whether the available evidence is enough to justify co-residency for this candidate.

**Artifact.** A CUDA-first suite with resource-swept kernels, Green Context/MPS variants, NVSHMEM progress tests, replayable telemetry summaries, and hooks for AMD or other runtimes.

**Evaluation.** Primary metrics are unsafe-admission rate, false rejection, SLO violation, progress failure, throughput, and evidence cost. The key comparison holds aggregate utilization nearly constant while changing the hidden resource constraint. An admission policy that cannot separate the pair has learned the wrong signal.

**Academic value.** The benchmark makes allocatability a measurable systems property and exposes where average utilization is information-theoretically insufficient for the decision.

**Production value.** Runtime and cluster teams can regression-test scheduler heuristics against new GPU architectures, drivers, partitioning modes, and workload mixes.

**Failure condition.** If the counterexamples occur only in artificial kernels and disappear across representative production workloads, allocatability can remain an offline tuning concern rather than a runtime abstraction.

## What would change this conclusion?

The argument assumes that GPU sharing will continue to combine workloads with materially different resource footprints and progress requirements. If production traces show that candidate kernels are homogeneous enough that low utilization almost always predicts safe co-residency, a candidate-aware contract would be more machinery than the scheduler needs.

The argument would also weaken if future hardware exposes a single authoritative admission primitive that already accounts for registers, shared memory, partition ownership, bandwidth, and progress dependencies. In that case the right abstraction would be to surface that primitive rather than reconstruct allocatability in software.

The strongest test is empirical. Hold telemetry cost constant and compare a simple utilization threshold with candidate-aware admission across deliberately difficult and representative workloads. If both policies produce the same unsafe-admission rate, tail latency, progress behavior, and throughput, keep the simple metric. If they diverge, the scheduler should stop treating utilization as capacity and start recording the evidence that actually justified the placement.