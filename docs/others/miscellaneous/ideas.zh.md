# 开源活动的可能想法 - eunomia-bpf 2024

这是一些开源活动的可能想法，如 GSOC(Google Summer of Code) 或 OSPP(开源软件供应链点亮计划)。我们的项目设计适应于不同水平的专业知识，从学生到更高级的开发者。

这也是我们项目路线图的一部分，如果你不参加这些活动，你也可以帮助或合作这些想法！需要帮助？请在[电子邮件列表](mailto:team@eunomia.dev)或在[Discord 频道](https://discord.gg/jvM73AFdB8)联系。

## 目录

- [未来可能的想法](#未来可能的想法)
  - [目录](#目录)
  - [bpftime](#bpftime)
  - [把 bpftime 移植到 macOS](#把-bpftime-移植到-macos)
    - [实现在 macOS 上启用 eBPF 的目标](#实现在-macos-上启用-ebpf-的目标)
    - [预期的成果](#预期的成果)
    - [需要的技能](#需要的技能)
    - [参考及问题](#参考及问题)
  - [为轻量级容器提供用户空间的 eBPF 的 AOT 编译](#为轻量级容器提供用户空间的-ebpf-的-aot-编译)
    - [概述](#概述)
    - [目标和任务](#目标和任务)
    - [必要的技能](#必要的技能)
    - [预期的成果](#预期的成果-1)
    - [额外的资源](#额外的资源)
  - [为 bpftime 添加 Fuzzer 并改进兼容性](#为-bpftime-添加-fuzzer-并改进兼容性)
    - [项目概述](#项目概述)
    - [时间和难度](#时间和难度)
    - [导师](#导师)
    - [目标](#目标)
    - [预期的成果](#预期的成果-2)
    - [要求和技能](#要求和技能)
    - [参考和问题](#参考和问题)
  - [bpftime + fuse：用户空间的 eBPF 支持用户空间的文件系统](#bpftime--fuse用户空间的-ebpf-支持用户空间的文件系统)
    - [目标](#目标-1)
    - [预期的成果](#预期的成果-3)
    - [必要的技能](#必要的技能-1)
    - [资源](#资源)

## bpftime

一个用户空间的 eBPF 运行时，可让现有的 eBPF 应用程序在非特权用户空间中运行，使用相同的库和工具链。它为 eBPF 提供 Uprobe 和 Syscall 追踪点，比内核 uprobe 的性能提高很多，而且不需要手动的代码插桩或进程重启。运行时在用户空间共享内存中支持进程间的 eBPF 映射，也兼容内核的 eBPF 映射，可以与内核的 eBPF 基础设施无缝操作。它还包含多种架构的高性能 LLVM JIT，外加一个针对 x86 的轻量级 JIT 和一个解释器。

欲了解更多详情，请参见：

- <https://eunomia.dev/bpftime>
- [https://github.com/eunomia-bpf/bpftime](https://github.com/eunomia-bpf/bpftime)

## 把 bpftime 移植到 macOS

由于 bpftime 可以在用户空间中运行并且不需要内核 eBPF，那为什么不让 macOS 也可以使用 eBPF 呢？

这个项目的目标是把 `bpftime` 移植到 macOS，扩展其跨平台的能力，并让 macOS 用户能够利用 `eBPF` 的强大功能在他们的开发和生产环境中。有了 bpftime，现在你可能能在 macOS 上运行 bcc 和 bpftrace 工具！

- 时间：~175 小时
- 难度等级：中等
- 导师：Yusheng Zheng (<mailto:yunwei356@gmail.com>) 和 Yuxi Huang (<Yuxi4096@gmail.com>)

### 实现在 macOS 上启用 eBPF 的目标

1. **兼容性和集成**：让 `bpftime` 与 macOS 兼容，确保核心功能和能力在该平台上的运行。
2. **性能优化**：调整 `bpftime` 在 macOS 上的性能，专注于优化适用于 macOS 架构的 LLVM JIT 和轻量级 JIT 的性能。
3. **与 macOS 生态系统无缝集成**：确保 `bpftime` 能与 macOS 环境平滑集成，为 macOS eBPF 用户提供一种原生和高效的开发体验。
4. **文档和教程**：开发专为 macOS 用户设计的文档和教程，方便他们在该平台上容易地采用和使用 `bpftime`。

### 预期的成果

- 一个功能正常的 `bpftime` 在 macOS 上的移植，核心功能运行正常。
- 您应该能够在 MacOS 上运行 bpftrace 和 bcc 工具。
- 使用 `bpftime` 在 macOS 上的文档和指南。

### 需要的技能

- 精通 C/C++ 和系统编程。
- 熟悉 macOS 开发环境和工具。
- 对 eBPF 及其应用的理解。

### 参考及问题

- 问题和一些初始讨论：<https://github.com/eunomia-bpf/bpftime/issues>
- 以前的一些努力：[在 arm 上启用 bpftime](https://github.com/eunomia-bpf/bpftime/pull/151)

## 为轻量级容器提供用户空间的 eBPF 的 AOT 编译

### 概述

在云原生应用、物联网和嵌入式系统的发展世界中，对有效、安全、注意资源的计算方案的需求正在增加。我们的项目关注的是开发一个用户空间的 eBPF (扩展的伯克利数据包过滤器) 与 AOT (先于时间) 编译。这个项目的目标是创建一个轻量级的、事件驱动的计算模型，以满足嵌入式和资源受限环境的独特需求。

相比于其他方案， eBPF AOT可以带来的主要区别是，它可以帮助构建一个可验证和安全的运行时环境，并可以足够轻量和高效，以便在嵌入式设备上运行。

持续和难度等级

- 预期持续时间：~175 小时
- 难度等级：中等
- 导师：Tong Yu (<yt.xyxx@gmail.com>), Yusheng Zheng (<mailto:yunwei356@gmail.com>)

bpftime 已经有了一个 AOT 编译器，我们需要进行更多的工作以使其能在嵌入式设备上运行或者作为插件运行。

### 目标和任务

1. **开发用户空间 eBPF AOT 编译**：AOT 编译器应该能够很好地与 helpers、ufuncs maps 等 eBPF 的其它特性一起工作。目前有一个 AOT 编译器的 POC，但是不完整，需要更多的工作。

你可以选择以下目标中的一个或两个来工作：

1. **集成到 FaaS 容器中**：把这项技术无缝地集成到 Function-as-a-Service（FaaS）轻量级容器中，增强启动速度和运营效率。
2. **实现插件系统**：设计一个系统，允许 eBPF 程序作为插件嵌入到其它应用中，提供动态、优化的功能。
3. **在嵌入式设备上运行 AOT ebpf**：让 AOT eBPF 能在嵌入式设备上运行，比如 Raspberry Pi 和其他 IoT 设备。

### 必要的技能

- C/C++ 和系统层次编程的技能。
- 基本理解容器技术和 FaaS 架构。
- 熟悉 eBPF 的概念和应用。
- 对物联网、云原生和嵌入式系统有兴趣。

### 预期的成果

- 具有 AOT 编译能力的函数 eBPF 运行时。
- 在 FaaS 轻量级容器中集成的实践演示。
- 一个可以让 eBPF 程序作为插件嵌入到各种应用的插件系统。
- 在嵌入式设备上运行的 AOT eBPF。

### 额外的资源

1. bpftime 的 AOT 示例
<https://github.com/eunomia-bpf/bpftime/blob/master/.github/workflows/test-aot-cli.yml>
2. vm 的 API。 <https://github.com/eunomia-bpf/bpftime/tree/master/vm/include>
3. 编译它作为一个独立的库
<https://github.com/eunomia-bpf/bpftime/tree/master/vm/llvm-jit>

如果你想为微控制器增加映射支持，我认为你可以写一个 c 实现，编译它并链接到 bpftime AOT 产品。稍后我们会提供一个例子。

## 为 bpftime 添加 Fuzzer 并改进兼容性

### 项目概述

`bpftime` 项目，因其创新的用户空间 eBPF 运行时而知名，正在寻求通过整合 Fuzzer 来增强其健壮性和可靠性。该项目的目标是为 `bpftime` 开发并集成一个专用的 Fuzzer，使用像 [Google's Buzzer](https://github.com/google/buzzer) 这样的工具。Fuzzer 将系统地测试 `bpftime`，以揭示任何可能的 bug、内存泄漏或漏洞，从而确保更安全和稳定的运行时环境。

你还需要在 CI 中启用 Fuzzer。

### 时间和难度

- **时间承诺**：~90小时
- **难度级别**：容易

### 导师

- Tong Yu ([yt.xyxx@gmail.com](mailto:yt.xyxx@gmail.com))
- Yusheng Zheng ([yunwei356@gmail.com](mailto:yunwei356@gmail.com))

### 目标

1. **开发和集成 Fuzzer**：设计和开发一个可以无缝集成到 `bpftime` 的 Fuzzer。或者你可以使用现有的 eBPF Fuzzers。
2. **测试和调试**：使用 Fuzzer 来查找并报告在 `bpftime` 用户空间 eBPF 运行时中的 bug、内存泄漏或漏洞。
3. **文档**：创建一个文档，解释 Fuzzer 在 `bpftime` 环境中的实现和使用。
4. **实施反馈**：积极合并 `bpftime` 社区的反馈，以改进和加强 Fuzzer。

### 预期的成果

- 在 `bpftime` 环境中完全集成的 Fuzzer。
- `bpftime` 中侦测和解决的 bug 和漏洞数量的增加。
- 对未来的贡献者使用和改进 Fuzzer 的文档和指南

### 要求和技能

- C/C++ 和系统编程的技能。
- 对软件测试方法的了解，特别是 Fuzz 测试。
- 有使用 Google 的 Buzzer 等 Fuzzer 的经验的人将大受欢迎。
- 对 eBPF 和它的生态系统的基本了解。

### 参考和问题

- 在 `bpftime` 中需要 Fuzzer 的初始讨论：[GitHub Issue](https://github.com/eunomia-bpf/bpftime/issues/163)
- Google buzzer：<https://github.com/google/buzzer>

## bpftime + fuse：用户空间的 eBPF 支持用户空间的文件系统

在现代操作系统中，`fuse` （用户空间的文件系统）已成为流行的选择，使得开发者不修改内核代码就可以在用户空间中创建文件系统。然而，系统调用的代价仍然存在。这就是 `bpftime` 可以发挥作用的地方。

bpftime 可能有助于：

- 减少系统调用的开销，提高性能
- 在不修改内核的情况下为 fuse 启用缓存
- 根据性能数据动态调整文件系统策略
- 为 fuse 添加更多的策略和策略

你可以和我们一起探索更多的可能性：

- 时间：~350 小时
- 难度等级：困难
- 导师：Yiwei Yang (<yyang363@ucsc.edu>) Yusheng Zheng (<mailto:yunwei356@gmail.com>)

### 目标

1. **用户空间和内核空间之间的协同优化**：利用 `bpftime` 在用户空间中预处理文件系统操作，如缓存和元数据查询，从而最小化系统调用的开销。
2. **用户空间的内核绕过机制**：使用 eBPF 在用户空间开发一个文件系统的内核绕过机制，可能消除了对用户应用进行侵入性更改的需要。
3. **动态策略调整**：在 `bpftime` 中实施一个系统，动态地收集性能数据并实时调整操作策略。
4. **为特定工作负载的定制**：使开发者能够为多种应用场景量身定制 eBPF 程序，以优化各种工作负载。

### 预期的成果

- 展示 `bpftime` 和用户空间文件系统协同的概念验证实现。
- 减少在用户空间进行文件操作的系统调用开销。
- 一个可以根据性能数据动态调整文件系统策略的框架。
- 文档或论文

### 必要的技能

- 精通 C/C++ 和系统级编程。
- 熟悉文件系统的概念以及用户空间-内核空间的交互。
- 基础理解 eBPF 及其在现代操作系统中的应用。
- 对 `fuse` 或类似技术有经验者优先。

### 资源

- Extfuse 论文和 GitHub 仓库: <https://github.com/extfuse/extfuse>
- <https://lwn.net/Articles/915717/>
