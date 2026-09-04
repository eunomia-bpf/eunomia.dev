---
date: 2026-09-04
title: "When Is a GPU Checkpoint Actually Safe to Restore?"
description: "GPU checkpoint/restore can save CPU and device state, yet a restorable image may cross application epochs. This report defines a consistent recovery cut."
tags:
  - Daily Report
  - GPU
  - Checkpoint Restore
  - Runtime Systems
  - Fault Tolerance
  - Distributed Systems
research_question: "Can transparent GPU checkpoint/restore guarantee an application-consistent recovery point across CPU state, GPU state, distributed communication, and externally visible effects?"
source_cutoff: 2026-09-04
status: daily-report
---

# When Is a GPU Checkpoint Actually Safe to Restore?

A cluster scheduler wants to preempt a CUDA service, free the GPU, and resume it later on another compatible device. The checkpoint tool reports success. Device memory is saved, CUDA objects can be reconstructed, and the CPU process can be restored. Yet the service may still return a duplicated result, repeat a collective, or resume from a request state that never existed as one coherent application instant.

The reason is a boundary mismatch. GPU checkpoint/restore has become a real systems primitive, but **a restorable GPU process image is not automatically an application-consistent recovery point**. CUDA state, CPU state, communication state, and externally visible effects can each be individually valid while belonging to different logical epochs.

This report argues for making the recovery cut explicit. A checkpoint should carry evidence that every state domain participating in one logical operation is either included before a common cut, excluded after it, or marked replayable. The scheduler can then distinguish "the bytes can be restored" from "the application can safely continue from here."

<!-- more -->

This question is different from the earlier report on [GPU membership changes](https://eunomia.dev/research/gpu-membership-generation-continuity/), which asks whether ranks converge on one state generation after communicator reconfiguration. It is also different from [GPU memory placement](https://eunomia.dev/research/gpu-memory-placement-evidence/), which asks whether a runtime has enough evidence to place data well. Here the membership and placement may stay fixed. The missing property is whether a checkpoint captures one semantically valid cut across state that lives on different sides of the CPU/GPU and process/external boundaries.

## A GPU checkpoint freezes CUDA state, not the whole application

NVIDIA now exposes checkpoint/restore directly in the CUDA Driver API. In CUDA 13.3.1, `cuCheckpointProcessLock()` moves a running CUDA process into a locked state and blocks further CUDA API calls. `cuCheckpointProcessCheckpoint()` then saves GPU memory into host memory and releases the underlying GPU references. Restore can remap checkpointed GPUs onto compatible GPUs of the same chip type when enough memory is available, after which the process returns to the locked state and can be unlocked.

The current [NVIDIA `cuda-checkpoint` utility](https://github.com/NVIDIA/cuda-checkpoint) makes the lifecycle concrete. Suspending CUDA locks state-changing driver calls, waits for already submitted GPU work and stream callbacks to finish, copies device memory to host storage managed by the driver, and releases GPU resources. Importantly, CPU threads are not suspended by this CUDA operation. NVIDIA therefore combines CUDA checkpointing with CRIU when a full Linux process checkpoint is needed.

That separation is useful. It means the GPU can be released without requiring every application to implement framework-specific save logic. It also exposes the consistency question: between the CUDA cut and the CPU-process cut, ordinary host code has its own state and may have effects outside the checkpointed process.

Consider a service that receives request `R`, launches a GPU update, records the result in host memory, and sends a reply. A checkpoint taken after the GPU update but before the reply is not inherently wrong. It is safe only if recovery knows whether the reply is still pending and can emit it once. A checkpoint taken after the reply reaches a remote peer but before local acknowledgement state is durable can be worse: restoring it may repeat an effect that the peer already observed.

CUDA cannot solve that alone because the remote peer is outside the CUDA process. CRIU cannot roll back a database, another service, or a remote collective participant merely because local sockets and process state were restored. The recovery boundary is larger than either mechanism's serialization boundary.

## System-level checkpointing is getting much better at reconstructing GPU state

This is not an argument that current checkpoint systems are primitive. Recent work closes several difficult reconstruction gaps.

[CRIUgpu](https://arxiv.org/abs/2502.16631) integrates GPU-aware checkpointing with CRIU and vendor mechanisms so CUDA and ROCm workloads can be checkpointed without steady-state API logging overhead. The work is important because it moves transparent GPU checkpointing closer to ordinary process checkpointing rather than requiring a framework-specific restart path.

[FlowGPU](https://doi.org/10.1007/978-3-032-35251-4_20), published online for Euro-Par 2026 in August, attacks correctness and performance problems in system-level GPU checkpoint/restore. Its per-task interception and ghost-process design separates GPU and non-GPU state only when checkpointing is needed. It preserves GPU virtual-address identity with CUDA VMM, replays operations that create opaque runtime objects, and coordinates distributed pauses. The paper explicitly handles an NCCL deadlock case in which one rank can be paused before a peer reaches the matching operation, using a bounded failed-pause path rather than leaving the job stuck.

Those mechanisms establish increasingly strong answers to "can this process and its GPU runtime state be reconstructed?" They do not by themselves define what application transaction the reconstructed state represents. A memory address can be identical after restore while the request associated with that memory has already committed an external effect.

The distinction is similar to filesystem crash consistency. Restoring every block to a syntactically valid value is weaker than proving that the recovered blocks correspond to one allowed transaction prefix.

## Persistent GPU runtimes make the semantic cut even more important

Persistent kernels move more control state onto the GPU. The previous Daily Report on [megakernel observability](https://eunomia.dev/research/ebpf-gpu-megakernel-observability/) showed that task identity, dependencies, and scheduling decisions can live inside one long-running kernel rather than at CUDA launch boundaries.

Checkpointing faces the same shift.

[Concordia](https://arxiv.org/abs/2606.23521) proposes checkpoint hooks inside a device-resident persistent kernel for long-running LLM inference. It registers GPU-resident state regions such as KV-cache blocks and scheduler state, uses JIT-compiled checkpoint handlers, and appends recovery records to CPU-visible memory. This is strong evidence that semantic recovery points can be exposed below framework code and near the state they protect.

It also sharpens the open problem. If a persistent runtime says "KV blocks through sequence 804 are committed," the host request queue, communication progress, output stream, and any external side effect need a compatible interpretation of sequence 804. Otherwise the device checkpoint is locally precise but globally ambiguous.

A useful checkpoint interface therefore needs a small amount of application meaning, not only more byte coverage.

## Where current work is still weak

### A successful restore does not state which logical effects are safe to replay

Most transparent checkpoint mechanisms define success in terms of reconstructing process, runtime, and memory state. That is necessary, but production recovery also needs an effect rule.

For each in-flight operation, recovery needs to know whether it is already committed, must be replayed, or must be discarded. Without that classification, the checkpoint manager cannot distinguish a harmless duplicate GPU computation from a duplicate network reply, a repeated storage mutation, or a repeated message to another rank.

This gap is easiest to expose with an adversarial service whose external receiver records monotonically increasing operation IDs. A checkpoint system can restore perfectly at the process level and still fail if recovery produces a missing or duplicated ID.

### Distributed quiescence is not the same as application quiescence

FlowGPU shows why pausing all ranks is a protocol rather than a `SIGSTOP` operation. It coordinates pauses and avoids a blocking NCCL pattern that can prevent the cut from completing. That is already stronger than independent per-process snapshots.

But all ranks being paused does not prove that their surrounding application state has one semantic epoch. A rank may have advanced optimizer state, consumed input, or published output that another state domain has not yet incorporated. A scheduler needs a condition such as "all operations through epoch 184 are committed; operations after 184 are either absent or replayable," not simply "all processes are currently stopped."

### Checkpoint coverage is usually implicit

A transparent tool often knows what it can serialize, but the consumer does not receive a machine-checkable statement of what was outside the cut. The current `cuda-checkpoint` project, for example, documents feature-specific support and restrictions that evolve across driver versions. A production scheduler should not have to infer from tool version and workload behavior whether IPC memory, unified-memory state, device-side runtime metadata, remote storage, or peer protocol state is relevant to this checkpoint.

The checkpoint artifact should say what it covers and which assumptions recovery depends on.

## Research directions worth building

### 1. Make every checkpoint carry a recovery-cut certificate

The first artifact is a small manifest generated by the checkpoint coordinator. It records the logical checkpoint epoch and the state domains that joined that cut:

```text
checkpoint = 91
application_epoch = 184
cpu_process = frozen@184
cuda_process = CHECKPOINTED@184
rank_set = {0,1,2,3}@184
persistent_kernel = quiescent@184
external_effect_fence = committed_through(183)
replayable = [request_7712]
coverage = [cpu, cuda, nccl, request_log]
unknown = []
```

The exact fields should be runtime-specific, but the contract is simple: a restore is eligible only when every required domain can prove a compatible cut or is explicitly handled by replay.

This can be layered over existing tools. CUDA provides the device-process state transition; CRIU provides Linux process state; FlowGPU-style interceptors provide GPU-object and distributed-pause evidence; a framework or persistent runtime can publish a semantic epoch. The certificate binds them instead of replacing them.

The academic question is whether a useful cross-domain consistency model can stay small enough for transparent systems. The production value is immediate: a cluster scheduler can reject unsafe migration or downgrade to a full restart when a certificate is incomplete.

### 2. Add semantic quiescence and effect fences, not application-specific checkpoint code everywhere

The second artifact is a narrow adapter interface for state that cannot be made consistent by freezing CUDA calls.

A persistent-kernel runtime might expose `prepare_checkpoint(epoch)` and report when all device tasks at or below the epoch are committed. A communication library can expose whether collectives before the epoch are complete. A serving runtime can append request/result IDs to a write-ahead log and delay external publication until the checkpoint coordinator has either fenced or classified the effect.

This is less invasive than teaching the checkpoint system the full application. Each subsystem exports only the transition needed to join a consistent cut: prepare, quiesce, commit, abort, and replay classification.

The design should prefer bounded failure. If one rank or external subsystem cannot reach the cut before a timeout, abort the checkpoint and resume the application, as FlowGPU already does for an incomplete distributed pause. A failed checkpoint is safer than a checkpoint whose consistency status is guessed.

### 3. Evaluate recovery correctness with effects, not only pause time and image size

Checkpoint papers naturally report runtime overhead, checkpoint size, pause duration, restore duration, and migration time. A consistency contract needs another benchmark dimension: whether the recovered execution is observationally equivalent to an allowed prefix plus replay.

A useful testbed would run deterministic workloads with ground-truth operation IDs across four state domains: CPU memory, GPU memory, NCCL or another peer protocol, and an external receiver such as a transactional log service. Fault injection chooses cuts around GPU completion, host callbacks, collective boundaries, allocator changes, and output publication.

For each restored run, the benchmark checks:

- missing committed operations;
- duplicated externally visible effects;
- state from incompatible epochs;
- stale or changed pointer identity;
- communication deadlock after restore;
- recovery attempts accepted despite unsupported coverage.

The primary metric is **invalid recovery rate under adversarial cut placement**, not raw checkpoint speed. A faster mechanism wins only after it produces the same valid recovery semantics.

## What would change this conclusion?

Three results would make a separate recovery-cut contract unnecessary.

First, a vendor or process-checkpoint interface could expand its atomic boundary to include CPU threads, GPU runtime state, multi-process communication, and all externally visible effects that matter to real applications. That would turn the larger application cut into the primitive itself. Current CUDA checkpointing intentionally composes with CPU checkpointing rather than claiming that boundary.

Second, production frameworks could converge on one application-level checkpoint protocol that already supplies durable operation epochs and exactly-once replay semantics while transparent GPU checkpointing only accelerates its implementation. In that world, the semantic contract belongs entirely above the GPU layer.

Third, experiments may show that transparent preemption and migration are used only at naturally quiescent framework boundaries where no ambiguous external effects exist. If adversarial cut placement cannot produce a semantic recovery failure beyond what current systems already detect, the proposed certificate adds complexity without buying correctness.

The current evidence points the other way. CUDA now has a real process checkpoint state machine, CRIUgpu and FlowGPU are making reconstruction practical, and persistent-kernel work is moving more valuable state below framework boundaries. **The next useful abstraction is not another way to copy GPU memory. It is a machine-checkable statement of what logical application cut that memory belongs to.**

## References

- NVIDIA. [CUDA Driver API: CUDA Checkpointing](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CHECKPOINT.html), CUDA 13.3.1 documentation, accessed 2026-09-04.
- NVIDIA. [`cuda-checkpoint`](https://github.com/NVIDIA/cuda-checkpoint), checkpoint/restore utility and current driver feature notes, accessed 2026-09-04.
- Radostin Stoyanov et al. [CRIUgpu: Transparent Checkpointing of GPU-Accelerated Workloads](https://arxiv.org/abs/2502.16631), 2025.
- Zehua Yang et al. [FlowGPU: Transparent and Efficient GPU Checkpointing and Restore](https://doi.org/10.1007/978-3-032-35251-4_20), Euro-Par 2026, first online 2026-08-15.
- Yuhang Gan et al. [Concordia: JIT-Compiled Persistent-Kernel Checkpointing for Fault-Tolerant LLM Inference](https://arxiv.org/abs/2606.23521), 2026.
