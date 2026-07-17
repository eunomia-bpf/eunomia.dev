---
title: 论文与研究
description: eunomia-bpf 社区在 eBPF、用户态运行时、GPU systems 和 AI Agent infrastructure 方向发表的论文、预印本、artifact 与技术解读。
keywords: eunomia-bpf 论文, eBPF 研究, bpftime, AI Agent 系统, GPU eBPF, 系统研究
author: eunomia-bpf community
---

# 论文与研究

本页是 eunomia-bpf 社区研究成果的 canonical index，收录社区产出或直接相关的论文。每条记录包含论文、公开 artifact 和对应的技术 Blog。“待补 Blog”表示这项研究仍缺少面向开发者的独立解读。

## 2026

| 论文 | Artifact | 技术 Blog | 状态 |
|---|---|---|---|
| [Characterizing and Bridging the Diagnostic Gap in eBPF Verifier Rejections](https://arxiv.org/abs/2607.02748) | [bpfix](https://github.com/eunomia-bpf/bpfix) | 待补 Blog | arXiv preprint |
| [ActPlane: Programmable OS-Level Policy Enforcement for Agent Harnesses](https://arxiv.org/abs/2606.25189) | [ActPlane](https://github.com/eunomia-bpf/ActPlane) | [阅读 Blog](../blog/posts/actplane.zh.md) | arXiv preprint，v2 |
| [Kops: Safely Extending the eBPF Compilation Pipeline with Native Operations](https://arxiv.org/abs/2606.24213) | [Kops artifact](https://github.com/eunomia-bpf/bpf-benchmark) | 待补 Blog | arXiv preprint |
| [ACRFence: Preventing Semantic Rollback Attacks in Agent Checkpoint-Restore](https://arxiv.org/abs/2603.20625) | 论文中描述的 artifact | [阅读 Blog](../blog/posts/agent-check-restore-safety.zh.md) | arXiv preprint |
| [NCCLbpf: Verified, Composable Policy Execution for GPU Collective Communication](https://arxiv.org/abs/2603.11438) | [nccl-eBPF](https://github.com/eunomia-bpf/nccl-eBPF) | 待补 Blog | arXiv preprint |
| [AgentCgroup: Understanding and Controlling OS Resources of AI Agents](https://arxiv.org/abs/2602.09345) | [AgentCgroup](https://github.com/eunomia-bpf/agentcgroup) | [阅读 Blog](../blog/posts/agentcgroup-characterization.zh.md) | arXiv preprint，v2 |

## 2025

| 论文 | Artifact | 技术 Blog | 状态 |
|---|---|---|---|
| [gpu_ext: Extensible OS Policies for GPUs via eBPF](https://arxiv.org/abs/2512.12615) | [gpu_ext](https://github.com/eunomia-bpf/gpu_ext) | GPU verification 背景文章与论文解读待补 | arXiv preprint，v2 |
| [Towards Agentic OS: An LLM Agent Framework for Linux Schedulers](https://arxiv.org/abs/2509.01245) | [SchedCP](https://github.com/eunomia-bpf/schedcp) | [阅读 Blog](../blog/posts/schedcp-agentic-os.zh.md) | MLforSystem 2025 |
| [AgentSight: System-Level Observability for AI Agents Using eBPF](https://arxiv.org/abs/2508.02736) | [AgentSight](https://github.com/eunomia-bpf/agentsight) | [阅读 Blog](../blog/posts/agentsight_paper.zh.md) | PACMI 2025 |
| [ChainIO: Bridging Disk and Network Domains with eBPF](https://doi.org/10.1145/3748355.3748371) | [ChainIO](https://github.com/eunomia-bpf/ChainIO) | 待补 Blog | eBPF@SIGCOMM 2025 |
| [Extending Applications Safely and Efficiently](https://www.usenix.org/conference/osdi25/presentation/zheng-yusheng) | [bpftime](https://github.com/eunomia-bpf/bpftime) | [bpftime 背景](../blog/posts/bpftime.zh.md)，OSDI 论文解读待补 | OSDI 2025 |

## 早期工作

| 论文 | Artifact | 技术 Blog | 状态 |
|---|---|---|---|
| [Code-Survey: An LLM-Driven Methodology for Analyzing Large-Scale Codebases](https://arxiv.org/abs/2410.01837) | [code-survey](https://github.com/eunomia-bpf/code-survey) | [阅读 Blog](../blog/posts/code-survey.md) | arXiv preprint |
| [Wasm-bpf: Streamlining eBPF Deployment in Cloud Environments with WebAssembly](https://arxiv.org/abs/2408.04856) | [wasm-bpf](https://github.com/eunomia-bpf/wasm-bpf) | [阅读 Blog](../blog/posts/wasm-bpf.zh.md) | arXiv preprint |
| [KEN: Kernel Extensions using Natural Language](https://arxiv.org/abs/2312.05531) | [KEN](https://github.com/eunomia-bpf/KEN) | [阅读 Blog](../blog/posts/kgent.zh.md) | eBPF@SIGCOMM 2024 |

## 如何保持索引更新

仓库每周运行一次定时审计，把本页与近期 arXiv 记录比较，同时检查论文和 artifact 链接、中英文覆盖以及 Blog 状态。发现 drift 后，自动任务会创建或更新同一个 GitHub issue。Maintainer 在发布论文相关变更前，也可以手动运行同一项审计。
