# 基于内核态/用户态 eBPF 实现高性能用户态文件系统功能 - 操作系统大赛赛题

## 项目描述

Fuse 是一个用户态文件系统框架，它允许开发人员在用户态实现文件系统。然而，Fuse 的性能一直是一个问题，特别是在大量文件操作、大量访问元数据等的情况下。eBPF是Linux内核的新特性，方便用户在内核里安全运行自定义的逻辑，eBPF在网络、安全、可观测性方面已经有了很多应用，也有了一些将 eBPF 与文件系统或 Fuse 相结合的尝试，例如：

- Extfuse 论文和 GitHub 仓库：<https://github.com/extfuse/extfuse>
- Fuse-BPF: <https://lwn.net/Articles/915717/>
- XRP: <https://www.usenix.org/conference/osdi22/presentation/zhong>

可以进一步探索如何使用内核 eBPF 的机制，进一步提高 Fuse 的性能，或者也可以进一步探索如何使用用户态 eBPF 的机制来进行协同优化：

[bpftime](https://github.com/eunomia-bpf/bpftime) 是一个用户空间的 eBPF 运行时，可让现有的 eBPF 应用程序直接在非特权用户空间中运行，使用相同的库和工具链，并且获取到追踪分析的结果。它为 eBPF 提供 Uprobe 和 Syscall tracepoint 等追踪点，并且允许在用户态直接修改库函数或者系统调用的执行流程，不需要手动的代码插桩或进程重启，可以实现对于源代码和编译流程的无侵入式分析和扩展。

bpftime 可能有助于：

- 减少系统调用的开销并提高性能
- 为 fuse 启用缓存而无需依赖新版本内核或修改内核代码
- 根据性能 profile 的数据，动态调整文件系统策略
- 为 fuse 添加更多复杂策略和策略

## 预期目标

分析现有的 Fuse 和 eBPF 文件系统的性能瓶颈，结合 eBPF 设计并实现一个高性能的用户态文件系统功能

## 特征

项目的特征可能可以包括：

1. **用户空间和内核空间之间的协同优化**：利用 `bpftime` 预处理用户空间中的文件系统操作，如缓存和元数据查询，从而最小化系统调用开销。
2. **用户空间中的内核绕过机制**：使用 eBPF 为用户空间中的文件系统开发一个内核 bypass 机制，消除对用户应用程序进行侵入式更改的需要。
3. **动态策略调整**：实现一个系统，动态收集性能数据并实时调整 Fuse eBPF 策略。
4. **针对特定工作负载的定制**：使开发者能够为多种应用场景定制 eBPF 程序，针对不同的工作负载进行优化。

（以上并非强制）

## 已有参考资料

- Extfuse 论文和 GitHub 仓库：<https://github.com/extfuse/extfuse>
- Fuse-BPF: <https://lwn.net/Articles/915717/>
- XRP: <https://www.usenix.org/conference/osdi22/presentation/zhong>
- [bpftime](https://github.com/eunomia-bpf/bpftime)

## 赛题分类

【code:303】2.3.3 文件系统

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
