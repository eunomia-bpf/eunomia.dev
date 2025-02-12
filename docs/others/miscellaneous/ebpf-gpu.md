# 使用 eBPF 对 AI/ML 工作负载进行追踪和性能分析 - 操作系统大赛赛题

## 项目描述

随着人工智能（AI）技术的快速发展，以及 LLM 模型变得日益复杂，对计算资源的需求也在持续增长。在这种背景下，有效监控和优化AI 工作负载的性能成为了一个重要议题。当模型训练一个超参数组合可能需要数十万美元时，投资于加速这些工作可能会节省大量成本。成本指标可以是美元数额、兆瓦足迹和二氧化碳排放量。提高效率的关键在于可观测性和分析，而很大程度上可观测性是技术、工具、数据集和仪表盘的综合体，它使我们能够首先评估基础设施在不同粒度水平上实现的性能水平，然后指导我们进行投资。

本项目的目的是开发一种基于eBPF（Extended Berkeley Packet Filter）的系统，专门用于AI/ML工作负载的追踪和性能分析。eBPF是Linux内核的一项技术，能够安全地运行内核级的沙箱程序，提供了高效、可编程、低开销的性能分析能力。

项目的灵感来源于Meta（前身为Facebook）的AI观测基础设施，以及在 eBPF submit 2023 中的演讲。Meta 开发了一套多层次的系统，包括底层的硬件遥测和监控、高级性能内省工具、以及大规模性能分析平台。其中，使用 eBPF 进行 GPU 性能分析的概念特别引人注目，因为它提供了一种有效的方法来监控和优化 AI 工作负载，特别是在 GPU 加速计算领域。和传统操作系统的性能分析相比，GPU 的分析涉及到更多的硬件和软件层次，从用户态的运行时库分析到硬件的追踪单元，因此需要更多的工具和技术。

本项目将结合这些先进的方法和技术，开发一个系统，用于实时追踪和分析AI/ML工作负载的性能。该系统将具备以下特点：

1. **高效的GPU性能分析**：利用eBPF进行GPU事件（如CUDA内核启动、同步事件和内存事件）的追踪和分析。
2. **全面的性能监控**：结合CPU和GPU的性能数据，提供一个全面的性能视图。
3. **低开销的数据收集**：使用eBPF技术确保性能数据收集的开销最小化，同时不影响应用程序的性能。
4. **自动化追踪和分析**：自动化收集多个主机和分析器的数据，提供一个整合的性能分析视图。
5. **易于集成和使用**：设计易于集成到现有AI/ML开发环境中的工具，且用户友好。

此外，项目可以参考 Dynolog 系统、GPU性能指标（如FLOPs/秒）、以及使用 DCGM 和 CUPTI 等工具进行 FLOPs 估算的方法。目标是创建一个能够为AI/ML研究和开发社区提供强大支持的开源工具。

## 预期目标

- 完成一个主要基于 eBPF 的 AI/ML 工作负载追踪和性能分析工具。
- 该工具能够提供深入的性能分析，帮助 AI 开发人员识别瓶颈并进行效能优化。
- 如果可以的话，提供可视化的分析结果和报告。

## 特征

项目的特征包括：

- 能够完整地、以较低的开销追踪 AI/ML 工作负载，包括 CPU 和 GPU 相关的性能数据。
- 可以对追踪结果进行深度分析，帮助性能工程师快速定位性能瓶颈并进行效能优化。
- 支持实时追踪和离线分析，满足不同场景的需求。
- 使用内核态或用户态的 eBPF 运行时，接入多种分析框架，如 Kineto, PyTorch Profiler 和 Strobelight，以及基于 eBPF 的技术，搭建可编程的性能分析平台，实现端到端的性能剖析。

## 已有参考资料

- [GPU Profiling with BPF at Meta - Riham Selim](https://www.youtube.com/watch?v=5xAghByteYc)
- [SYSTEM@SCALE: AI OBSERVABILITY](https://atscaleconference.com/systemscale-ai-observability/)
- [eBPF — a new Swiss army knife in the system](https://medium.com/@chivier.humber_15513/ebpf-a-new-swiss-army-knife-in-the-system-2d6421c8d39)
- [bpftime](https://github.com/eunomia-bpf/bpftime)

## 赛题分类

【code:405】2.4.5 系统调试/支撑库的设计

## 参赛要求

- 以小组为单位参赛，最多三人一个小组，且小组成员是来自同一所高校的本科生或研究生
- 允许学生参加大赛的多个不同题目，最终自己选择一个题目参与评奖
- 请遵循“2024全国大学生操作系统比赛”的章程和技术方案要求

## 难度

高等

## License

GPL-3.0 License

## 所属赛道

2024全国大学生操作系统比赛的“OS功能挑战”赛道

## 项目导师

- 姓名：郑昱笙
- 单位：eunomia-bpf 开源社区
- github ID：[https://github.com/yunwei37](https://github.com/yunwei37)
- email：[team@eunomia.dev](mailto:team@eunomia.dev) and [yunwei356@gmail.com](mailto:yunwei356@gmail.com)
