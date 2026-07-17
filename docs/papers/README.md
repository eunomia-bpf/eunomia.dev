---
title: Papers and Research
description: Peer-reviewed papers, preprints, artifacts, and technical explanations from the eunomia-bpf community across eBPF, userspace runtimes, GPU systems, and AI agent infrastructure.
keywords: eunomia-bpf papers, eBPF research, bpftime, AI agent systems, GPU eBPF, systems research
author: eunomia-bpf community
---

# Papers and Research

This page is the canonical index for research produced by or directly connected to the eunomia-bpf community. Each entry links the paper, public artifact when available, and the corresponding technical blog. “Blog pending” marks research that still needs a developer-oriented explanation.

## 2026

| Paper | Artifact | Technical blog | Status |
|---|---|---|---|
| [Characterizing and Bridging the Diagnostic Gap in eBPF Verifier Rejections](https://arxiv.org/abs/2607.02748) | [bpfix](https://github.com/eunomia-bpf/bpfix) | Blog pending | arXiv preprint |
| [ActPlane: Programmable OS-Level Policy Enforcement for Agent Harnesses](https://arxiv.org/abs/2606.25189) | [ActPlane](https://github.com/eunomia-bpf/ActPlane) | [Read the blog](../blog/posts/actplane.md) | arXiv preprint, v2 |
| [Kops: Safely Extending the eBPF Compilation Pipeline with Native Operations](https://arxiv.org/abs/2606.24213) | [Kops artifact](https://github.com/eunomia-bpf/bpf-benchmark) | Blog pending | arXiv preprint |
| [ACRFence: Preventing Semantic Rollback Attacks in Agent Checkpoint-Restore](https://arxiv.org/abs/2603.20625) | Paper artifact described in the paper | [Read the blog](../blog/posts/agent-check-restore-safety.md) | arXiv preprint |
| [NCCLbpf: Verified, Composable Policy Execution for GPU Collective Communication](https://arxiv.org/abs/2603.11438) | [nccl-eBPF](https://github.com/eunomia-bpf/nccl-eBPF) | Blog pending | arXiv preprint |
| [AgentCgroup: Understanding and Controlling OS Resources of AI Agents](https://arxiv.org/abs/2602.09345) | [AgentCgroup](https://github.com/eunomia-bpf/agentcgroup) | [Read the blog](../blog/posts/agentcgroup-characterization.md) | arXiv preprint, v2 |

## 2025

| Paper | Artifact | Technical blog | Status |
|---|---|---|---|
| [gpu_ext: Extensible OS Policies for GPUs via eBPF](https://arxiv.org/abs/2512.12615) | [gpu_ext](https://github.com/eunomia-bpf/gpu_ext) | [GPU verification background](../blog/posts/gpu-bug-taxonomy.md), paper explainer pending | arXiv preprint, v2 |
| [Towards Agentic OS: An LLM Agent Framework for Linux Schedulers](https://arxiv.org/abs/2509.01245) | [SchedCP](https://github.com/eunomia-bpf/schedcp) | [Read the blog](../blog/posts/schedcp-agentic-os.md) | MLforSystem 2025 |
| [AgentSight: System-Level Observability for AI Agents Using eBPF](https://arxiv.org/abs/2508.02736) | [AgentSight](https://github.com/eunomia-bpf/agentsight) | [Read the blog](../blog/posts/agentsight_paper.md) | PACMI 2025 |
| [ChainIO: Bridging Disk and Network Domains with eBPF](https://doi.org/10.1145/3748355.3748371) | [ChainIO](https://github.com/eunomia-bpf/ChainIO) | Blog pending | eBPF@SIGCOMM 2025 |
| [Extending Applications Safely and Efficiently](https://www.usenix.org/conference/osdi25/presentation/zheng-yusheng) | [bpftime](https://github.com/eunomia-bpf/bpftime) | [bpftime background](../blog/posts/bpftime.md), OSDI paper explainer pending | OSDI 2025 |

## Earlier Work

| Paper | Artifact | Technical blog | Status |
|---|---|---|---|
| [Code-Survey: An LLM-Driven Methodology for Analyzing Large-Scale Codebases](https://arxiv.org/abs/2410.01837) | [code-survey](https://github.com/eunomia-bpf/code-survey) | [Read the blog](../blog/posts/code-survey.md) | arXiv preprint |
| [Wasm-bpf: Streamlining eBPF Deployment in Cloud Environments with WebAssembly](https://arxiv.org/abs/2408.04856) | [wasm-bpf](https://github.com/eunomia-bpf/wasm-bpf) | [Read the blog](../blog/posts/wasm-bpf.md) | arXiv preprint |
| [KEN: Kernel Extensions using Natural Language](https://arxiv.org/abs/2312.05531) | [KEN](https://github.com/eunomia-bpf/KEN) | [Read the blog](../blog/posts/kgent.md) | eBPF@SIGCOMM 2024 |

## Keeping This Index Current

A scheduled repository audit runs every week. It compares this index with recent arXiv records, checks paper and artifact links, verifies English and Chinese coverage, and opens or updates one GitHub issue when it finds drift. Maintainers can also run the same audit manually before publishing a paper-related change.
