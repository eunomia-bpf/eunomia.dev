# 实现内核与用户态共同工作的 eBPF 安全框架 - 操作系统大赛赛题

## 项目描述

bpftime 是一个用户空间的 eBPF 运行时，可让现有的 eBPF 应用程序直接在非特权用户空间中运行，使用相同的库和工具链，并且获取到追踪分析的结果。它为 eBPF 提供 Uprobe 和 Syscall tracepoint 等追踪点，比内核 uprobe 的开销降低约10倍，不需要手动的代码插桩或进程重启，可以实现对于源代码和编译流程的无侵入式分析。它也可以和 DPDK 等结合，在用户态网络中实现 XDP 的功能，并和内核 XDP 兼容。运行时在用户空间共享内存中支持进程间的 eBPF maps，也兼容内核的 eBPF maps，可以与内核的 eBPF 基础设施无缝操作。它还包含多种架构的高性能 eBPF LLVM JIT/AOT 编译器。

LSM (Linux Security Modules) 是一种在Linux内核中实现的安全框架，它提供了一种机制，允许各种安全策略模块插入到内核中，增强系统的安全性。LSM 旨在为Linux操作系统提供一个抽象层，以支持多种不同的安全策略，而不需要改变内核的核心代码。这种设计允许系统管理员或发行版选择适合其安全需求的安全模型，如SELinux、AppArmor、Smack等。

LSM 可以用来做什么？

1. 访问控制：LSM 最常见的用途是实现强制访问控制（MAC）策略，这与传统的基于所有者的访问控制（DAC）不同。MAC 可以细粒度地控制进程对文件、网络端口、进程间通信等资源的访问。
2. 日志和审计：LSM 可以用来记录和审计系统上的敏感操作，提供详细的日志信息，帮助检测和防范潜在的安全威胁。
3. 沙箱和隔离：通过限制程序的行为和它们可以访问的资源，LSM 可以实现应用程序的沙箱化，从而减少恶意软件或漏洞利用的风险。
4. 强化内核和用户空间的安全：LSM 允许实现额外的安全检查和限制，用于强化内核自身的安全，以及运行在用户空间的应用程序的安全。
5. 限制特权操作：LSM 可以限制即使是拥有root权限的进程所能执行的操作，从而减少系统管理员错误配置或者拥有root权限的恶意软件的潜在危害。

借助 bpftime，我们可以在用户空间运行 eBPF 程序，与内核兼容，并且可以和内核的 eBPF 协同工作来进行防御。我们有没有可能将 eBPF 的安全机制和特性进一步扩展到用户态，让用户态的 eBPF 和内核态的 eBPF 协同工作，来实现更强大、更灵活的安全策略和防御能力呢？让我们把这种机制叫做 USM（Userspace Security Modules or Union Security Modules）。

## 预期目标

1. 框架设计与实现：在 bpftime 中设计并实现一个灵活的安全框架 USM，该框架能够让用户态的 eBPF USM 程序与内核态的 eBPF LSM 程序兼容并且协同工作。这包括定义框架的架构、API 设计、通信机制以及安全策略的实施方法，预期能提供一个统一的安全防御体系。这可能包括对特定系统调用的拦截、动态调整安全策略等功能。
2. 尝试探索可能的安全场景，使用对应的机制对当前存在的安全问题进行拦截和防御。例如，对于某些应用层协议的安全漏洞，通过用户态 eBPF 进行拦截和分析可能是更好的选择，而不是在内核中进行处理；对于文件等系统资源的访问控制，内核中的 LSM 可以提供更好的安全性。

## 特征

项目的特征包括：

- 开源代码并且尽可能和上游社区一起合作
- 需要寻找对应的 CVE 漏洞或者安全问题，进行分析、搭建运行时和设计防御策略
- 需要有合适的方式来测试和验证安全策略的有效性

## 已有参考资料

- [Linux Plumbers 23 演讲：bpftime: Fast uprobes with user space BPF runtime](https://lpc.events/event/17/contributions/1639/)
- [bpftime](https://github.com/eunomia-bpf/bpftime)
- <https://docs.kernel.org/bpf/prog_lsm.html>
- <https://blog.cloudflare.com/live-patch-security-vulnerabilities-with-ebpf-lsm>
- <https://github.com/eunomia-bpf/bpftime/issues/148>

## 赛题分类

【code:503】2.5.3 安全应用（加密等）

## 参赛要求

- 以小组为单位参赛，最多三人一个小组，且小组成员是来自同一所高校的本科生或研究生
- 允许学生参加大赛的多个不同题目，最终自己选择一个题目参与评奖
- 请遵循“2024全国大学生操作系统比赛”的章程和技术方案要求

## 难度

高级

## License

GPL-3.0 License

## 所属赛道

2024全国大学生操作系统比赛的“OS功能挑战”赛道

## 项目导师

- 姓名：郑昱笙
- 单位：eunomia-bpf 开源社区
- github ID：[https://github.com/yunwei37](https://github.com/yunwei37)
- email：[team@eunomia.dev](mailto:team@eunomia.dev) and [yunwei356@gmail.com](mailto:yunwei356@gmail.com)
