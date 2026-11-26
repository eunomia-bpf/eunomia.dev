---
title: Eunomia Labs, Inc.
description: Eunomia Labs, Inc. 开发并维护 Eunomia 的开源项目，方向是 eBPF 运行时、GPU 可观测性和 AI agent 基础设施。
---

# Eunomia Labs, Inc.

公司运营 eunomia.dev，并开发 GitHub 组织 [eunomia-bpf](https://github.com/eunomia-bpf) 下的各个项目。我们的工作围绕 eBPF 运行时基础设施展开：安全地扩展应用程序，观测和扩展 GPU 上的负载，并为 AI agent 提供一层系统级的可观测性和策略执行。

公司位于美国加州旧金山。

## 我们在做什么

| 项目 | 作用 |
|---|---|
| [bpftime](../bpftime/index.md) | 用户态 eBPF 运行时，支持 uprobe、系统调用和应用扩展，兼容内核 eBPF 工具链。它的 [GPU 支持](../bpftime/documents/gpu.md)可以让 eBPF 程序在 NVIDIA 和 AMD GPU 的 CUDA、ROCm kernel 内部运行。 |
| [gpu_ext](https://github.com/eunomia-bpf/gpu_ext) | 用 eBPF 程序编写可扩展的 OS 策略，用于 GPU 调度和显存管理。 |
| [AgentSight](../agentsight/index.md) | 基于 eBPF 的 AI agent 系统级可观测性，对 agent 本身零插桩。 |
| [ActPlane](../actplane/index.md) | 面向 AI agent 的 OS 级安全与治理控制，用策略即代码约束进程树、文件、网络行为和工作流关卡。 |
| [eunomia-bpf](../eunomia-bpf/index.md) 和 [wasm-bpf](../wasm-bpf/index.md) | 构建、打包和分发 eBPF 程序的工具链，支持以 WebAssembly 模块的形式分发。 |
| [教程](../tutorials/index.md) | bpf-developer-tutorial 系列教程，以及面向 GPU 编程、性能分析和插桩的 [CUDA](../others/cuda-tutorial/README.md)、[CUPTI](../others/cupti-tutorial/README.md) 和 [NVBit](../others/nvbit-tutorial/README.md) 教程。 |

## 研究

这些项目都来自系统研究。bpftime 以 "Extending Applications Safely and Efficiently" 发表在 OSDI 2025，AgentSight 发表在 PACMI 2025，SchedCP 发表在 MLforSystem 2025。[论文页](../papers/README.md)列出了全部论文、PDF、代码和解读文章。

## 开源

所有代码都在 [github.com/eunomia-bpf](https://github.com/eunomia-bpf) 公开开发，每个仓库写明了各自的许可证。Issue 和 [GitHub Discussions](https://github.com/orgs/eunomia-bpf/discussions) 是主要的协作渠道，[活动报告](../reports/index.md)每月记录组织的公开动态。

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
