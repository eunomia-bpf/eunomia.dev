---
title: Daily Report
description: "Technical systems reports that compare primary evidence, identify unresolved research and production gaps, and propose testable directions with academic and operational value."
---

# Daily Report

Eunomia Daily Report examines concrete systems questions, compares primary evidence, identifies mechanisms that current research or production practice still cannot explain well, and develops ideas that could become useful research systems or deployable engineering mechanisms.

## Current reports

### [Can eBPF Compress Telemetry Without Losing the Diagnosis?](https://eunomia.dev/research/ebpf-diagnostic-telemetry-compression/)

Always-on eBPF telemetry can become too expensive to export continuously, while counters and histograms can erase the context needed for later diagnosis. This report develops diagnostic contracts, state-transition exemplars, coverage-carrying summaries, and an equal-budget benchmark that scores diagnosis retention instead of compression ratio alone.

### [Can eBPF Understand Application-Defined Resources?](https://eunomia.dev/research/ebpf-application-resource-semantics/)

Internal pools, queues, caches, and credits can determine performance while remaining invisible as OS resources. This report develops a versioned resource-semantics manifest compiled into eBPF attachments, runtime confidence loss for stale contracts, and a mutation benchmark that tests semantic correctness across software upgrades.

### [Was the GPU Kernel Slow, or Did It Just Start Late?](https://eunomia.dev/research/gpu-kernel-launch-latency/)

A late CUDA kernel start can come from host scheduling, runtime work, command-buffer queueing, dependencies, or device availability even when kernel execution itself is unchanged. This report develops an explicit launch-state ledger, cross-domain launch lineage, and a ground-truth benchmark for deciding which delay cause the trace actually proves.

### [Can a GPU Profiler Prove What Caused a Slow Kernel?](https://eunomia.dev/research/gpu-host-device-causality/)

Asynchronous CUDA traces can show host calls, streams, graph nodes, and GPU kernels without proving which earlier action caused a delay. This report develops generation-scoped host-device causal identity, a dependency-aware critical-path graph with explicit unknown edges, and a ground-truth benchmark that makes timestamp-only explanations fail visibly.

### [When Does Profiler Sampling Become Biased?](https://eunomia.dev/research/profiler-sampling-bias/)

Sampling percentages can be systematically wrong when a sampler phase-locks with periodic work, skids past the event that caused a sample, or repeatedly misses short-lived code. This report develops an explicit sampling-schedule contract with aliasing diagnostics, replicated profile epochs with rank uncertainty, and uncertainty-triggered selective instrumentation under a fixed overhead budget.

### [Can eBPF Attribute Memory to the Pages That Actually Matter?](https://eunomia.dev/research/page-level-ebpf-memory-attribution/)

Allocation stacks, RSS, page hotness, reclaim, migration, and hardware memory samples describe different parts of memory cost. This report develops a lifetime-aware provenance chain from application allocations to virtual-region generations and page activity, access-weighted attribution with explicit confidence, and a ground-truth benchmark for deciding when page-level lineage is worth its overhead.

### [Where Should eBPF Run in a Heterogeneous System?](https://eunomia.dev/research/heterogeneous-ebpf-execution-placement/)

Kernel, userspace, SmartNIC, and GPU-side runtimes can all be valid homes for eBPF logic, but they do not expose the same events, state, memory, authority, or verifier environment. This report develops a placement-aware target manifest, generation-scoped state ownership, and a ground-truth benchmark for choosing execution location without silently changing policy semantics.

### [How Far Can eBPF Programmability Move Into io_uring?](https://eunomia.dev/research/io-uring-bpf-programmability/)

Current Linux has both per-opcode io_uring BPF request filtering and an eBPF `struct_ops` execution path. This report separates the cBPF admission gate from the eBPF ring-loop control surface, then asks how restrictions, LSM authority, policy generations, provenance, and resource accounting should compose as io_uring absorbs FUSE, zero-copy networking, ublk, and other registered I/O resources.

### [What Must an eBPF Profiler Track Beyond Threads?](https://eunomia.dev/research/async-ebpf-causal-profiler/)

Async work can leave one thread through `io_uring`, workqueues, runtime tasks, and application-defined resources, so CPU and off-CPU stacks can lose logical attribution even when the samples themselves are accurate. This report develops a typed causal-edge model, a budget that treats topology edges differently from context samples, and a ground-truth benchmark for cross-thread attribution.

### [Can a Stateful eBPF Application Upgrade Atomically?](https://eunomia.dev/research/stateful-ebpf-transactional-upgrade/)

One BPF link can replace one program cleanly, but real stateful applications also span maps, pinned objects, multiple hooks, and userspace controllers. This report separates simple state reuse from semantic state migration and develops generation-gated activation, BTF-aware migration, and crash-consistent recovery as testable upgrade mechanisms.

### [eBPF Hook Composition: Sharing One Hook Safely](https://eunomia.dev/research/ebpf-hook-composition-contract/)

Linux, libxdp, and TCX already let multiple eBPF programs share execution points, but ordering alone does not define how mutations, shared state, competing outcomes, and updates compose. This report compares existing multi-program semantics and research on isolation and bytecode dependencies, then proposes typed composition manifests, explicit outcome algebras, and versioned hook generations.

### [What Is Missing Before Userspace eBPF Becomes a Real Runtime?](https://eunomia.dev/research/userspace-ebpf-runtime-contract/)

A BPF VM can execute the instruction set without defining how programs attach, which capabilities they receive, who owns state, or how extensions are revoked and accounted for. This report compares Linux eBPF, uBPF, bpftime, and eBPF for Windows, then proposes a machine-readable runtime contract, capability-aware attach handles, and per-extension resource accounting.

### [When Several AI Agents Work at Once, Who Makes Sure the Final Result Is Right?](https://eunomia.dev/research/parallel-agent-effect-serializability/)

Worktrees, sandboxes, and parallel tool calls can isolate workers while still producing a wrong combined outcome. This report uses code changes, shared budgets, approvals, and irreversible actions to explain why parallel agents need one validation and commit step before their effects become real. It also identifies missing benchmarks and effect contracts, then proposes an agent transaction layer, a semantic-conflict benchmark, and adaptive concurrency control.

### [What Should an AI Agent Trace Keep? Observability Under a Fixed Evidence Budget](https://eunomia.dev/research/agent-trace-evidence-budget/)

AI agent traces can generate hundreds of system events around each model call while still omitting decisive state, authority, or provenance. This report develops an evidence-portfolio architecture and then identifies open problems in evidence utility, portable schemas, unbiased adaptive capture, and equal-budget evaluation.
