# 将用户态 eBPF 扩展到 MacOS、Windows、FreeBSD 等更多平台 - 操作系统大赛赛题

## 项目描述

bpftime 是一个用户空间的 eBPF 运行时，可让现有的 eBPF 应用程序直接在非特权用户空间中运行，使用相同的库和工具链，并且获取到追踪分析的结果。它为 eBPF 提供 Uprobe 和 Syscall tracepoint 等追踪点，比内核 uprobe 的开销降低约10倍，不需要手动的代码插桩或进程重启，可以实现对于源代码和编译流程的无侵入式分析。它也可以和 DPDK 等结合，在用户态网络中实现 XDP 的功能，并和内核 XDP 兼容。运行时在用户空间共享内存中支持进程间的 eBPF maps，也兼容内核的 eBPF maps，可以与内核的 eBPF 基础设施无缝操作。它还包含多种架构的高性能 eBPF LLVM JIT/AOT 编译器。

bpftime 现在可以让 bpftrace 和 bcc 等工具，以及一些商业 eBPF 的可观测性组件在用户态运行，而不需要内核 eBPF 支持，也不需要 root 权限，这也为其他非 Linux 系统、低版本内核、非特权容器环境下使用 eBPF 进行追踪分析提供了更多可能性。目前我们主要在高版本 Linux 上进行了测试，但我们希望能够把它移植到其他平台，例如 FreeBSD，Windows，MacOS 等等，或者一些嵌入式场景，让更多的平台支持 eBPF 生态。

## 预期目标

1. 在 FreeBSD，Windows，MacOS 或其他系统平台上移植 bpftime，扩展其跨平台的能力，并让其他平台的用户能够利用 eBPF 的强大功能在他们的开发和生产环境中。
2. 让 `bpftime` 与其他 OS兼容，确保核心功能和能力在该平台上的运行。bpftime 的核心二进制插桩、动态库诸如等能力仅依赖共享内存、动态库，其他可能的扩展包含和内核 eBPF 兼容、ptrace 注入运行中的进程等，这些能力可以根据不同平台的特性进行选择性的实现。
3. 目标是在不同平台上直接运行 bcc 和 bpftrace 等工具，或者其他商业 eBPF 的可观测性组件，而不需要修改对应的内核。

## 特征

项目的特征包括：

- 兼容性和集成：让 `bpftime` 与其他 OS 兼容，确保核心功能和能力在该平台上的运行。
- 运行对应的工作负载：让 `bpftime` 在其他平台上运行 bcc 和 bpftrace 等工具，或者其他商业 eBPF 的可观测性组件，获取对应的输出结果，而不需要修改对应的内核。

## 已有参考资料

- [Linux Plumbers 23 演讲：bpftime: Fast uprobes with user space BPF runtime](https://lpc.events/event/17/contributions/1639/)
- [bpftime](https://github.com/eunomia-bpf/bpftime)
- [Arxiv: bpftime: Fast uprobes with user space BPF runtime](https://arxiv.org/abs/2311.07923)
- [Issue: Try to run eBPF with bpftime on MacOS](https://github.com/eunomia-bpf/bpftime/issues/145)

## 赛题分类

【code:405】2.4.5 系统调试/支撑库的设计

## 参赛要求

- 以小组为单位参赛，最多三人一个小组，且小组成员是来自同一所高校的本科生或研究生
- 允许学生参加大赛的多个不同题目，最终自己选择一个题目参与评奖
- 请遵循“2024全国大学生操作系统比赛”的章程和技术方案要求

## 难度

中等

## License

GPL-3.0 License

## 所属赛道

2024全国大学生操作系统比赛的“OS功能挑战”赛道

## 项目导师

- 姓名：郑昱笙
- 单位：eunomia-bpf 开源社区
- github ID：[https://github.com/yunwei37](https://github.com/yunwei37)
- email：[team@eunomia.dev](mailto:team@eunomia.dev) and [yunwei356@gmail.com](mailto:yunwei356@gmail.com)
