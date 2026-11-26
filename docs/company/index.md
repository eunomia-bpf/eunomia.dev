---
title: Eunomia Labs, Inc.
description: Eunomia Labs, Inc. builds the runtime for recursive self-improvement. It profiles what AI systems do, gives them safe places to change themselves, and keeps every change inside limits the system sets.
---

# Eunomia Labs, Inc.

Eunomia Labs, Inc. builds the runtime for recursive self-improvement (RSI). AI agents now modify their own code, tools, and infrastructure, evaluate the result, and carry what works into the next round. That loop runs on real systems: CPUs, GPUs, files, networks, and the agents themselves. Our runtime gives it three things from the system side: an accurate profile of what ran, safe points where a change can land, and limits that every change respects.

The company operates eunomia.dev, develops the projects in the [eunomia-bpf](https://github.com/eunomia-bpf) GitHub organization, and is based in San Francisco, California.

## One runtime, three layers

eBPF connects the three layers. The same model of small, verified programs loaded at run time observes, extends, and enforces across the kernel, userspace, and GPUs.

### Profile: see what actually ran

- [AgentSight](../agentsight/index.md) records what AI agents do on a machine. It links prompts, model calls, and tool decisions to the processes, files, and network traffic they cause, with zero instrumentation of the agent.
- [bpftime GPU support](../bpftime/documents/gpu.md) runs eBPF programs inside CUDA and ROCm kernels, so profiling reaches the code running on NVIDIA and AMD GPUs.

### Change: safe places for improvements to land

- [bpftime](../bpftime/index.md) is a userspace eBPF runtime. Applications gain uprobe and syscall extension points that load eBPF programs at run time, using the same toolchains as kernel eBPF.
- [gpu_ext](https://github.com/eunomia-bpf/gpu_ext) turns GPU scheduling and memory management into programmable eBPF policies.
- [SchedCP](../GPTtrace/schedcp.md) lets AI agents profile a workload, then synthesize and deploy Linux scheduler policies through sched-ext.

### Bound: limits every change respects

- [ActPlane](../actplane/index.md) enforces the policies developers already write, such as rules in `AGENTS.md` and `CLAUDE.md`, at the operating-system level across process trees, files, and network effects, and reports each violation back to the agent.
- [AgentCgroup](https://github.com/eunomia-bpf/agentcgroup) connects an agent's task structure to Linux resource control.

## Research

Each layer comes from published systems research. bpftime appeared at OSDI 2025 as "Extending Applications Safely and Efficiently", AgentSight at PACMI 2025, and SchedCP at MLforSystem 2025. The [papers page](../papers/README.md) lists the full set with PDFs, artifacts, and explainers.

## Open source

All code is developed in the open at [github.com/eunomia-bpf](https://github.com/eunomia-bpf), and each repository states its license. We also maintain the [eunomia-bpf](../eunomia-bpf/index.md) and [wasm-bpf](../wasm-bpf/index.md) toolchains, the [eBPF tutorials](../tutorials/index.md), and tutorials on [CUDA](../others/cuda-tutorial/README.md), [CUPTI](../others/cupti-tutorial/README.md), and [NVBit](../others/nvbit-tutorial/README.md). Issues and [GitHub Discussions](https://github.com/orgs/eunomia-bpf/discussions) are the main collaboration channels, and the [activity reports](../reports/index.md) record the organization's public activity every month.

## Working with us

Eunomia Labs supports teams that adopt these projects in production, from evaluation to integration. [Products](../products/index.md) describes the product lines and enterprise support. To reach us directly, email [yusheng@eunomia.dev](mailto:yusheng@eunomia.dev).

## Company details

| Item | Detail |
|---|---|
| Legal name | Eunomia Labs, Inc. |
| Location | San Francisco, California, United States |
| Website | [eunomia.dev](https://eunomia.dev) |
| Open source | [github.com/eunomia-bpf](https://github.com/eunomia-bpf) |
| Contact | [yusheng@eunomia.dev](mailto:yusheng@eunomia.dev) |
