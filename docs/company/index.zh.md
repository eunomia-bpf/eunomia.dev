---
title: Eunomia Labs, Inc.
description: Eunomia Labs, Inc. 构建面向递归自我改进的运行时：观测 AI 系统实际做了什么，给它们安全的修改落点，并让每一次修改都留在系统设定的边界内。
---

# Eunomia Labs, Inc.

Eunomia Labs, Inc. 构建面向递归自我改进（recursive self-improvement，RSI）的运行时。今天的 AI agent 已经会修改自己的代码、工具和运行基础设施，评估修改结果，再把有效的改进带入下一轮。这个循环跑在真实系统上：CPU、GPU、文件、网络，以及 agent 本身。我们的运行时从系统一侧为它提供三样东西：准确记录实际运行情况的 profile，可以安全落地修改的扩展点，以及每次修改都要遵守的边界。

公司运营 eunomia.dev，开发 GitHub 组织 [eunomia-bpf](https://github.com/eunomia-bpf) 下的项目，位于美国加州旧金山。

## 一个运行时，三层能力

三层能力由 eBPF 串起来：同一套在运行时加载、经过验证的小程序模型，在内核、用户态和 GPU 上完成观测、扩展和执行控制。

### 观测：看清实际运行了什么

- [AgentSight](../agentsight/index.md) 记录 AI agent 在机器上的实际行为，把提示词、模型调用和工具决策关联到它们引发的进程、文件和网络流量，对 agent 本身零插桩。
- [bpftime 的 GPU 支持](../bpftime/documents/gpu.md)让 eBPF 程序在 CUDA 和 ROCm kernel 内部运行，profiling 可以深入到 NVIDIA 和 AMD GPU 上执行的代码。

### 修改：给改进安全的落点

- [bpftime](../bpftime/index.md) 是用户态 eBPF 运行时，为应用提供 uprobe 和系统调用扩展点，在运行时加载 eBPF 程序，沿用内核 eBPF 的工具链。
- [gpu_ext](https://github.com/eunomia-bpf/gpu_ext) 把 GPU 调度和显存管理变成可编程的 eBPF 策略。
- [SchedCP](../GPTtrace/schedcp.md) 让 AI agent 先分析负载，再通过 sched-ext 合成并部署 Linux 调度策略。

### 约束：每次修改都要遵守的边界

- [ActPlane](../actplane/index.md) 把开发者已经写下的策略（例如 `AGENTS.md`、`CLAUDE.md` 里的规则）在操作系统层面执行，覆盖进程树、文件和网络行为，并把每次违规的原因反馈给 agent。
- [AgentCgroup](https://github.com/eunomia-bpf/agentcgroup) 把 agent 的任务结构接入 Linux 资源控制。

## 研究

每一层都来自已发表的系统研究。bpftime 以 "Extending Applications Safely and Efficiently" 发表在 OSDI 2025，AgentSight 发表在 PACMI 2025，SchedCP 发表在 MLforSystem 2025。[论文页](../papers/README.md)列出了全部论文、PDF、代码和解读文章。

## 开源

所有代码都在 [github.com/eunomia-bpf](https://github.com/eunomia-bpf) 公开开发，每个仓库写明了各自的许可证。我们同时维护 [eunomia-bpf](../eunomia-bpf/index.md) 和 [wasm-bpf](../wasm-bpf/index.md) 工具链、[eBPF 教程](../tutorials/index.md)，以及 [CUDA](../others/cuda-tutorial/README.md)、[CUPTI](../others/cupti-tutorial/README.md) 和 [NVBit](../others/nvbit-tutorial/README.md) 教程。Issue 和 [GitHub Discussions](https://github.com/orgs/eunomia-bpf/discussions) 是主要的协作渠道，[活动报告](../reports/index.md)每月记录组织的公开动态。

## 合作

Eunomia Labs 支持在生产环境中采用这些项目的团队，从评估一直到集成。[产品页](../products/index.md)介绍了各条产品线和企业支持。也可以直接发邮件到 [yusheng@eunomia.dev](mailto:yusheng@eunomia.dev)。

## 公司信息

| 项目 | 内容 |
|---|---|
| 法定名称 | Eunomia Labs, Inc. |
| 所在地 | 美国加州旧金山 |
| 网站 | [eunomia.dev](https://eunomia.dev) |
| 开源 | [github.com/eunomia-bpf](https://github.com/eunomia-bpf) |
| 联系方式 | [yusheng@eunomia.dev](mailto:yusheng@eunomia.dev) |
