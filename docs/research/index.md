---
title: Daily Report
description: "Technical systems reports that compare primary evidence, identify unresolved research and production gaps, and propose testable directions with academic and operational value."
---

# Daily Report

Eunomia Daily Report examines concrete systems questions, compares primary evidence, identifies mechanisms that current research or production practice still cannot explain well, and develops ideas that could become useful research systems or deployable engineering mechanisms.

## Current reports

### [How Should a Profiler Discover Application-Defined Resources?](https://eunomia.dev/research/application-defined-resource-profiling/)

Application-defined resources such as buffer pools, caches, and internal queues can dominate performance while remaining invisible to system resource metrics. This report develops a versioned resource-semantics manifest, runtime validation with explicit confidence loss, and a ground-truth benchmark for comparing declared, inferred, and dynamically observed resource models.

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
