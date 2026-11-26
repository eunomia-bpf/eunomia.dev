---
title: Eunomia Labs, Inc.
description: Eunomia Labs, Inc. builds and maintains the Eunomia open-source projects for eBPF runtimes, GPU observability, and AI agent infrastructure.
---

# Eunomia Labs, Inc.

The company operates eunomia.dev and develops the projects in the [eunomia-bpf](https://github.com/eunomia-bpf) GitHub organization. Our work centers on eBPF runtime infrastructure: extending applications safely, observing and extending GPU workloads, and giving AI agents a system layer for observability and policy enforcement.

The company is based in San Francisco, California.

## What we build

| Project | What it does |
|---|---|
| [bpftime](../bpftime/index.md) | A userspace eBPF runtime for uprobes, syscalls, and application extensions, compatible with kernel eBPF toolchains. Its [GPU support](../bpftime/documents/gpu.md) runs eBPF programs inside CUDA and ROCm kernels on NVIDIA and AMD GPUs. |
| [gpu_ext](https://github.com/eunomia-bpf/gpu_ext) | Extensible OS policies for GPU scheduling and memory management, written as eBPF programs. |
| [AgentSight](../agentsight/index.md) | System-level observability for AI agents using eBPF, with zero instrumentation of the agent itself. |
| [ActPlane](../actplane/index.md) | OS-level safety, security, and governance controls for AI agents, with policy-as-code for process trees, files, network effects, and workflow gates. |
| [eunomia-bpf](../eunomia-bpf/index.md) and [wasm-bpf](../wasm-bpf/index.md) | Toolchains for building, packaging, and distributing eBPF programs, including as WebAssembly modules. |
| [Tutorials](../tutorials/index.md) | The bpf-developer-tutorial series, plus [CUDA](../others/cuda-tutorial/README.md), [CUPTI](../others/cupti-tutorial/README.md), and [NVBit](../others/nvbit-tutorial/README.md) tutorials for GPU programming, profiling, and instrumentation. |

## Research

Our projects grow out of systems research. bpftime appeared at OSDI 2025 as "Extending Applications Safely and Efficiently", AgentSight at PACMI 2025, and SchedCP at MLforSystem 2025. The [papers page](../papers/README.md) lists the full set with PDFs, artifacts, and explainers.

## Open source

All code is developed in the open at [github.com/eunomia-bpf](https://github.com/eunomia-bpf), and each repository states its license. Issues and [GitHub Discussions](https://github.com/orgs/eunomia-bpf/discussions) are the main collaboration channels, and the [activity reports](../reports/index.md) record the organization's public activity every month.

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
