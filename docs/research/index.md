---
title: Daily Report
description: "Technical systems reports that compare primary evidence, identify unresolved research and production gaps, and propose testable directions with academic and operational value."
---

# Daily Report

Eunomia Daily Report examines concrete systems questions, compares primary evidence, identifies mechanisms that current research or production practice still cannot explain well, and develops ideas that could become useful research systems or deployable engineering mechanisms.

## Current reports

### [eBPF Hook Composition: Sharing One Hook Safely](https://eunomia.dev/research/ebpf-hook-composition-contract/)

Linux, libxdp, and TCX already let multiple eBPF programs share execution points, but ordering alone does not define how mutations, shared state, competing outcomes, and updates compose. This report compares existing multi-program semantics and research on isolation and bytecode dependencies, then proposes typed composition manifests, explicit outcome algebras, and versioned hook generations.

### [What Is Missing Before Userspace eBPF Becomes a Real Runtime?](https://eunomia.dev/research/userspace-ebpf-runtime-contract/)

A BPF VM can execute the instruction set without defining how programs attach, which capabilities they receive, who owns state, or how extensions are revoked and accounted for. This report compares Linux eBPF, uBPF, bpftime, and eBPF for Windows, then proposes a machine-readable runtime contract, capability-aware attach handles, and per-extension resource accounting.

### [When Several AI Agents Work at Once, Who Makes Sure the Final Result Is Right?](https://eunomia.dev/research/parallel-agent-effect-serializability/)

Worktrees, sandboxes, and parallel tool calls can isolate workers while still producing a wrong combined outcome. This report uses code changes, shared budgets, approvals, and irreversible actions to explain why parallel agents need one validation and commit step before their effects become real. It also identifies missing benchmarks and effect contracts, then proposes an agent transaction layer, a semantic-conflict benchmark, and adaptive concurrency control.

### [What Should an AI Agent Trace Keep? Observability Under a Fixed Evidence Budget](https://eunomia.dev/research/agent-trace-evidence-budget/)

AI agent traces can generate hundreds of system events around each model call while still omitting decisive state, authority, or provenance. This report develops an evidence-portfolio architecture and then identifies open problems in evidence utility, portable schemas, unbiased adaptive capture, and equal-budget evaluation.
