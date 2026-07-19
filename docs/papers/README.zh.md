---
title: 论文与研究
description: eunomia-bpf 社区的全文研究库（每篇论文含 PDF 和纯文本），覆盖 eBPF、用户态运行时、GPU 系统和 AI Agent 基础设施。
keywords: eunomia-bpf 论文, eBPF 研究, bpftime, AI Agent 系统, GPU eBPF, 系统研究
author: eunomia-bpf community
---

# 论文与研究

本页是 eunomia-bpf 社区研究成果的正本库。每篇论文附有：已发表或 arXiv 版本链接、本地 PDF 副本、可供人和 LLM 工具直接阅读的纯文本提取、公开 artifact，以及面向开发者的技术 Blog 解读（如已撰写）。收录政策：只有已发表论文或公开 arXiv 版本会进入本库；处于评审中的工作不会出现在此处。元数据与 arXiv 版本锁定记录在本目录的 `registry.yaml` 中。

## 研究脉络

以下论文形成一条连贯的研究主线，可以从四条相互交织的线索来理解。

### 起点：把 eBPF 变成可用的扩展基础设施（2022 至 2024）

eunomia-bpf 工具链与 bpf-developer-tutorial 共同建立了 eBPF 编程的教育基础，但分发与编写仍是突出难题。Wasm-bpf（[arXiv 2408.04856](https://arxiv.org/abs/2408.04856)）解决分发问题：把 eBPF 程序打包进 WebAssembly 模块，使其像普通云工作负载一样可以分发和实例化。KEN（[arXiv 2312.05531](https://arxiv.org/abs/2312.05531)，eBPF@SIGCOMM 2024）解决编写问题：从自然语言意图生成通过内核 verifier 验证的扩展程序，是 LLM 与 eBPF 结合的首次演示。Code-Survey（[arXiv 2410.01837](https://arxiv.org/abs/2410.01837)）随后将 LLM 的分析能力转向内核自身，提供了一种自动化方法来大规模分析内核 eBPF 子系统的演化历史。

这三篇论文从互补的方向降低了门槛：打包与可移植性、生成与验证、大规模代码理解。

### 把扩展运行时带出内核

内核 eBPF 功能强大，但约束在内核边界之内。bpftime 最初以 arXiv 预印本（[2311.07923](https://arxiv.org/abs/2311.07923)）的形式描述了一个用户态 uprobe 和 syscall 扩展运行时，完全兼容内核 eBPF 工具链；随后成熟为 OSDI 2025 论文 "Extending Applications Safely and Efficiently"，证明同样的扩展模型可以在内核之外以有竞争力的性能运行。MVVM（[arXiv 2410.15894](https://arxiv.org/abs/2410.15894)）补充了 Wasm checkpoint-restore 和 live migration 的能力，使 agent 部署可以跨异构节点进行。

扩展流水线自身的健壮性也成为独立的关注点。Kops（[arXiv 2606.24213](https://arxiv.org/abs/2606.24213)）在不破坏安全保证的前提下向 eBPF 编译流水线引入原生操作（native operations）。bpfix（[arXiv 2607.02748](https://arxiv.org/abs/2607.02748)）系统性地刻画了 verifier 拒绝程序时产生的诊断鸿沟，这是首个研究开发者为何难以理解拒绝信息、工具如何弥合这一差距的工作。

CET-disassembly（[arXiv 2506.09426](https://arxiv.org/abs/2506.09426)）处于另一层面：利用 CET（control-flow enforcement technology）元数据实现健全（sound）且精确的静态二进制反汇编，为在无法进行源码级 eBPF 挂载的闭源软件上做插桩提供了二进制层面的基础。

### 伸进 GPU 与 IO 路径

扩展哲学在用户态得到验证后，自然的问题是它能否触及 CPU 和 syscall 视野之外的硬件域。gpu_ext（[arXiv 2512.12615](https://arxiv.org/abs/2512.12615)）给出了 GPU 方向的回答：将 eBPF 风格的可扩展 OS 策略附加到 GPU 调度和内存管理。NCCLbpf（[arXiv 2603.11438](https://arxiv.org/abs/2603.11438)）进一步深入 GPU 集合通信，在 NCCL 操作上组合经过验证的策略。ChainIO（[DOI 10.1145/3748355.3748371](https://doi.org/10.1145/3748355.3748371)，eBPF@SIGCOMM 2025）将同一原则应用于磁盘与网络 IO 域的桥接。

三篇论文共同表明，扩展模型可以从 CPU 约束的系统调用推广到现代基础设施真正消耗时间的硬件路径。

### AI agent 时代：观测与执行

当前的研究重心回扣前两条线索的铺垫：内核级基础设施正是使深度 agent 可观测性和策略执行成为可能的关键，且无需修改 agent 代码。

SchedCP（[arXiv 2509.01245](https://arxiv.org/abs/2509.01245)，MLforSystem 2025）展示了 LLM agent 可以安全驱动 Linux 调度器，将 agent 定位为自主运维者而非被动工具。AgentSight（[arXiv 2508.02736](https://arxiv.org/abs/2508.02736)，PACMI 2025）利用 eBPF 提供 AI agent 的系统级可观测性，在 SDK 层和 HTTP 层之下捕获行为。AgentCgroup（[arXiv 2602.09345](https://arxiv.org/abs/2602.09345)）刻画并控制 agent 消耗的 OS 资源，揭示 agent 工作负载与传统服务器进程的差异。

ACRFence（[arXiv 2603.20625](https://arxiv.org/abs/2603.20625)）识别了新的攻击面：agent checkpoint-restore 中的语义回滚攻击，恢复先前状态可能悄然撤销安全关键决策。论文提出的防御机制在保留 checkpoint-restore 实用性的同时阻止回滚利用。

ActPlane（[arXiv 2606.25189](https://arxiv.org/abs/2606.25189)）将上述线索综合为对开发者已写下的策略（policy）做 OS 级强制执行。一项对 64 个仓库中 84 份 CLAUDE.md/AGENTS.md 文件的实证研究发现了 2116 条与策略相关的语句；其中 64% 是策略（其余为描述性内容），83% 的策略是系统可观测的。ActPlane 在运行时执行这些策略，危险命令拒绝率达到 75.8%，而最佳基线为 53.7%（基于 190 条轨迹和 38 条规则），端到端开销仅 1.9%。

## 完整列表

### 2026

| 论文 | PDF / 文本 | 开源实现 | 技术 Blog | 状态 |
|---|---|---|---|---|
| [Characterizing and Bridging the Diagnostic Gap in eBPF Verifier Rejections](https://arxiv.org/abs/2607.02748) | [PDF](bpfix.pdf) · [txt](bpfix.txt) | [bpfix](https://github.com/eunomia-bpf/bpfix) | 待补 Blog | arXiv preprint |
| [ActPlane: Programmable OS-Level Policy Enforcement for Agent Harnesses](https://arxiv.org/abs/2606.25189) | [PDF](actplane.pdf) · [txt](actplane.txt) | [ActPlane](https://github.com/eunomia-bpf/ActPlane) | [阅读 Blog](../blog/posts/actplane.zh.md) | arXiv preprint，v2 |
| [Kops: Safely Extending the eBPF Compilation Pipeline with Native Operations](https://arxiv.org/abs/2606.24213) | [PDF](kops.pdf) · [txt](kops.txt) | [Kops artifact](https://github.com/eunomia-bpf/bpf-benchmark) | 待补 Blog | arXiv preprint |
| [ACRFence: Preventing Semantic Rollback Attacks in Agent Checkpoint-Restore](https://arxiv.org/abs/2603.20625) | [PDF](acrfence.pdf) · [txt](acrfence.txt) | 见论文描述 | [阅读 Blog](../blog/posts/agent-check-restore-safety.zh.md) | arXiv preprint |
| [NCCLbpf: Verified, Composable Policy Execution for GPU Collective Communication](https://arxiv.org/abs/2603.11438) | [PDF](ncclbpf.pdf) · [txt](ncclbpf.txt) | [nccl-eBPF](https://github.com/eunomia-bpf/nccl-eBPF) | 待补 Blog | arXiv preprint |
| [AgentCgroup: Understanding and Controlling OS Resources of AI Agents](https://arxiv.org/abs/2602.09345) | [PDF](agentcgroup.pdf) · [txt](agentcgroup.txt) | [AgentCgroup](https://github.com/eunomia-bpf/agentcgroup) | [阅读 Blog](../blog/posts/agentcgroup-characterization.zh.md) | arXiv preprint，v2 |

### 2025

| 论文 | PDF / 文本 | 开源实现 | 技术 Blog | 状态 |
|---|---|---|---|---|
| [gpu_ext: Extensible OS Policies for GPUs via eBPF](https://arxiv.org/abs/2512.12615) | [PDF](gpu-ext.pdf) · [txt](gpu-ext.txt) | [gpu_ext](https://github.com/eunomia-bpf/gpu_ext) | GPU 验证背景文章与论文解读待补 | arXiv preprint，v2 |
| [Towards Agentic OS: An LLM Agent Framework for Linux Schedulers](https://arxiv.org/abs/2509.01245) | [PDF](schedcp.pdf) · [txt](schedcp.txt) | [SchedCP](https://github.com/eunomia-bpf/schedcp) | [阅读 Blog](../blog/posts/schedcp-agentic-os.zh.md) | MLforSystem 2025 |
| [AgentSight: System-Level Observability for AI Agents Using eBPF](https://arxiv.org/abs/2508.02736) | [PDF](agentsight.pdf) · [txt](agentsight.txt) | [AgentSight](https://github.com/eunomia-bpf/agentsight) | [阅读 Blog](../blog/posts/agentsight_paper.zh.md) | PACMI 2025 |
| [ChainIO: Bridging Disk and Network Domains with eBPF](https://doi.org/10.1145/3748355.3748371) | [ACM DL](https://doi.org/10.1145/3748355.3748371) | [ChainIO](https://github.com/eunomia-bpf/ChainIO) | 待补 Blog | eBPF@SIGCOMM 2025 |
| [Extending Applications Safely and Efficiently](https://www.usenix.org/conference/osdi25/presentation/zheng-yusheng) | [PDF](bpftime-osdi25.pdf) · [txt](bpftime-osdi25.txt) | [bpftime](https://github.com/eunomia-bpf/bpftime) | [bpftime 背景](../blog/posts/bpftime.zh.md)，OSDI 论文解读待补 | OSDI 2025 |
| [Exploiting Control-flow Enforcement Technology for Sound and Precise Static Binary Disassembly](https://arxiv.org/abs/2506.09426) | [PDF](cet-disassembly.pdf) · [txt](cet-disassembly.txt) | 无 | 待补 Blog | arXiv preprint |

### 早期工作

| 论文 | PDF / 文本 | 开源实现 | 技术 Blog | 状态 |
|---|---|---|---|---|
| [MVVM: Deploy Your AI Agents Securely, Efficiently, Everywhere](https://arxiv.org/abs/2410.15894) | [PDF](mvvm.pdf) · [txt](mvvm.txt) | [MVVM](https://github.com/Multi-V-VM/MVVM) | 待补 Blog | arXiv preprint |
| [Code-Survey: An LLM-Driven Methodology for Analyzing Large-Scale Codebases](https://arxiv.org/abs/2410.01837) | [PDF](code-survey.pdf) · [txt](code-survey.txt) | [code-survey](https://github.com/eunomia-bpf/code-survey) | [阅读 Blog](../blog/posts/code-survey.md) | arXiv preprint |
| [Wasm-bpf: Streamlining eBPF Deployment in Cloud Environments with WebAssembly](https://arxiv.org/abs/2408.04856) | [PDF](wasm-bpf.pdf) · [txt](wasm-bpf.txt) | [wasm-bpf](https://github.com/eunomia-bpf/wasm-bpf) | [阅读 Blog](../blog/posts/wasm-bpf.zh.md) | arXiv preprint |
| [KEN: Kernel Extensions using Natural Language](https://arxiv.org/abs/2312.05531) | [PDF](ken.pdf) · [txt](ken.txt) | [KEN](https://github.com/eunomia-bpf/KEN) | [阅读 Blog](../blog/posts/kgent.zh.md) | eBPF@SIGCOMM 2024 |
| [bpftime: userspace eBPF Runtime for Uprobe, Syscall and Kernel-User Interactions](https://arxiv.org/abs/2311.07923) | [PDF](bpftime-arxiv.pdf) · [txt](bpftime-arxiv.txt) | [bpftime](https://github.com/eunomia-bpf/bpftime) | [bpftime 背景](../blog/posts/bpftime.zh.md) | arXiv preprint；已被 OSDI 2025 论文取代 |

## 阅读笔记

- [osdi20-brunella.md](osdi20-brunella.md)：hXDP（OSDI 2020）的中英文阅读笔记与全文，该论文研究 FPGA NIC 上的高效软件包处理，属外部工作。
- [uXDP camera-ready text](uXDP__Frictionless_XDP_Deployments_in_Userspace___Camera_Ready.txt)：作为参考材料保留在本库中。

## 维护方式

仓库每周运行一次审计，将本索引和 `registry.yaml` 与 arXiv 进行比对，检查新版本、新论文和链接失效。发现偏差后审计会开一个 GitHub issue。当论文收到新的 arXiv 版本时，需要刷新本地 PDF 和纯文本副本，更新 `registry.yaml` 中的版本锁定，并重新验证站点中引用的所有数字。新论文遵循 `.github/PAPER_PUBLICATION_CHECKLIST.md` 中的流程。
