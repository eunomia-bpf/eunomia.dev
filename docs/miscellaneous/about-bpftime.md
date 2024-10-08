# test

## 标题



## 项目描述

> 尽量详细地描述该项目。 例如：

各高校的计算机专业普遍都开设OS课，由于培养目标的差异，不同高校OS课程的要求是不同的；即使是同一所大学的学生，对OS课程&实验的需求也会有所不同，甚至有些学生还会通过参加操作系统比赛来学习掌握操作系统。

为此，每个高校的老师和同学都相应的需求，结合自己的实际情况，选择或设计实现适合自己OS实验，以及参加操作系统比赛。老师和同学设计的实验也可能会适合情况相近的高校OS课程，学生写的参赛项目指导书会帮助更多的学生参加比赛，提升系统能力。

本项目的目的鼓励各个高校的师生设计适合自己的OS实验教程和竞赛指导等，帮助并共享给大家。 这个项目的特点是：从不同的维度（使用OS、分析OS、实现OS、扩展OS、硬件特征、应用需求….）设计操作系统相关的实验内容和比赛项目指导教程等，使得学生可以从不同维度来更好地理解和掌握OS。



## 预期目标

> 项目预期达到何种目标？ 例如：

- 设计实现操作系统课核心算法的演示实验或工具；
- 设计不同难度和不同实验环境的操作系统教学实验；

## 特征

> 描述项目的特征。 例如：

- 从不同的维度（使用OS、扩展OS、实现OS、分析OS、硬件特征、应用需求….）设计操作系统相关的实验内容和比赛项目指导教程
- 文档、代码、问题、答疑交互过程都开放和开源的
- 支持各种硬件
    - 基于目前量产的处理器和开发板，比如：
        - D1哪吒开发板（基于平头哥C906 RV64 CPU）
        - K210开发板（基于K210处理器 RV64 CPU）
        - K510开发板（基于K510处理器 RV64 CPU）
        - StarFive开发板（基于U740 RV64 CPU）
        - SiFive Unmatch开发板（基于U740/U540 RV64 CPU）
        - 树莓派开发板（基于ARM处理器）

## 已有参考资料

> 列出有利于开展项目的已有参考文献、开源代码等； 例如：

- [xv6](https://github.com/mit-pdos/xv6-riscv-fall19)
- [ChCore](https://gitee.com/ipads-lab/chcore-lab-v2)
- [pke](https://gitee.com/hustos/riscv-pke)

## 赛题分类

> 请参考[2024全国大学生操作系统比赛的“OS功能挑战”赛道-赛题分类](https://docs.qq.com/doc/DS2FvdWVoYVBXR0Ni)，对赛题进行分类。

例如：2.6 教学支撑大类 -->  2.6.1 教学实验设计

或者：601

## 参赛要求

> 对参赛队的要求。 例如：

- 以小组为单位参赛，最多三人一个小组，且小组成员是来自同一所高校的本科生或研究生
- 允许学生参加大赛的多个不同题目，最终自己选择一个题目参与评奖
- 请遵循“2024全国大学生操作系统比赛”的章程和技术方案要求

## 难度

> 难度等级，选择“初等”、“中等”，或者"高等"。

## License

> 某种开源协议，推荐GPL和CC开源协议。例如：

GPL-3.0 License

## 所属赛道

2024全国大学生操作系统比赛的“OS功能挑战”赛道

## 项目导师

> 项目的导师联络信息，包括姓名、单位、（github id）、email信息； 例如：

- 姓名：郑昱笙
- 单位：eunomia-bpf 开源社区
- github ID：[https://github.com/xyongcn](https://github.com/xyongcn)
- email：[xyong@tsinghua.edu.cn](mailto:xyong@tsinghua.edu.cn)


1. 请介绍一下 eBPF 项目的标志事件和未来计划

我目前自己运营维护了一个小的 eBPF 开源社区，总共有 4k+ 的 star，几十个贡献者和几个项目，具体可以参考官网：<https://eunomia.dev> github 组织：<https://github.com/eunomia-bpf>。

我们一开始做了一些 eBPF 的开发工具，也做了一些将 Wasm 和 eBPF 结合起来，使用 GPT 生成 eBPF 代码的尝试。最近我们在做一个用户态的 eBPF runtime，和内核 eBPF 兼容并且，可以和内核 eBPF 协同工作，也可以独立在内核 eBPF 不可用的环境下运行。我们现在已经可以脱离 Linux 内核的支持，跑一些 bpftrace 或者 bcc 工具，可以在一些场景下达到和内核 eBPF 一样的效果，并且性能、可扩展性好很多。

实际上，我最初开始接触 eBPF 就是从操作系统大赛开始的，感谢各位老师和评委的鼓励，以及 PLCT 实验室的支持，让我能够把当时一些不成熟的 idea 继续做下去，完善成开源项目，并且慢慢拓展出一整个开源社区。我也在云栖大会、Kubecon、Linux Plumbers 等会议上做过一些关于这些开源项目的分享，希望能够让更多的人了解 eBPF，也能够让更多的朋友参与到 eBPF 的开发中来。

对于我们项目的愿景而言，我希望先回顾一下内核 eBPF 的愿景：

内核的 eBPF 的核心愿景是希望在内核里面允许一个能带来更多创新机会的、可编程的使用方式，不用修改内核，同时也保证安全性；也从内核的版本发布周期中解放出来，让更多的人能参与到内核的开发中来，让内核的功能更加灵活和高效。

我们也有些类似，我们希望能把 eBPF 引入用户态，来扩展从内核到用户态程序的整个软件堆栈，探索更多创新的用例、有更好的性能、更好的可扩展性和移植性的同时还保证和当前 eBPF 生态的兼容。我希望它能真正发展成为一个有生命力的开源项目，并且能为 eBPF 生态带来更多的可能性。接下来，我们会进一步增强它的稳定性和可用性，在用户态提供更多和内核兼容或者探索性的追踪、扩展方式，比如 uprobe，usdt，syscall tracepoint 等等追踪事件，XDP、socket 等网络扩展；同时我们也在尝试把它移植到其他平台，例如 FreeBSD，Windows，MacOS 等等，或者一些嵌入式场景，让更多的平台支持 eBPF 生态。

1. 您参与开源社区的动机或初心是什么？

从最初的想法来说，网络提供了一个非常好的平台，而个人的作品只有得到传播、协作和使用，才能够真正发挥出它的价值，我不希望我们大赛提交的作品仅仅只是用于获取奖项和奖金，而是希望它能够真正的帮助到其他人。从功利的角度来说，开源可以扩大个人的知名度与影响力，可以认识许多可能会对未来职业生涯带来帮助的朋友，也可以帮助自己更好的学习和提升；同时，很多开源项目也可能进一步孵化成为商业项目，带来直接或者间接的经济收益。

1. 您从参与开源项目中得到的收益是什么？

开源可能不能直接带来很多的经济收益，但它是一个扩大影响力、在真实世界中参与软件开发、设计、推广的一个很好的方式。我觉得最大的收益是能有机会真正在一个领域中深入探索和参与，和一个小团队一起做一些有意义的事情，并且在对应领域中得到一些认可。

bpftime 是一个高性能的用户态 eBPF 运行时框架，旨在提升系统的可观测性和扩展性。它突破了传统内核 eBPF 的限制，通过绕过内核，直接在用户态运行，提供更灵活的架构与更广泛的平台支持。bpftime 支持 Uprobe、USDT、Syscall hook、XDP 等多种事件源，并利用先进的 LLVM 编译器实现高效的即时（JIT）和提前（AOT）编译。 

该项目不仅在性能上显著提升，能够加快系统跟踪与网络应用的执行速度，还具备跨平台兼容性，使得 eBPF 能在内核权限受限或不可用的环境中正常运行。此外，bpftime 允许用户轻松进行原型开发和新功能的探索，而无需更改底层操作系统或工具链。通过其模块化设计，开发者可以快速集成新事件类型或扩展功能，推动创新。作为一个用户态的 eBPF 运行时与通用扩展框架，bpftime 为构建高效、灵活的系统观察工具提供了新的可能性，促进了技术社区的协作与创新。 

## bpftime intro

目前，基于 eBPF 的可观测性与网络优化方案已经成为许多云服务商的重要基础设施。然而，线上环境中依然存在大量低版本内核，无法原生支持 eBPF，或者在非特权容器环境中，无法访问内核的 eBPF 运行时。这使得传统的内核 eBPF 解决方案在这些环境中无法发挥作用。bpftime 针对这一痛点，为一家大型云服务商提供了跨平台的可观测性与安全工具。通过绕过内核，bpftime 在无需大规模操作系统升级的情况下，成功在多个老旧服务器上部署了用户态的性能监控方案，能高效获取未加密的 TLS 流量、各种语言运行时的性能指标、业务相关逻辑等观测信息。这为企业解决了难以升级内核版本的技术障碍。目前，该公司正在小规模试点 bpftime，预计在未来大规模部署后，有望节约数十万美元的系统升级成本，并极大提升其服务的可用性和安全性。 

### how bpftime can help: example

一家海外的 eBPF 可观测性与网络解决方案供应商，将 bpftime 集成到了他们的自动化测试流水线和持续集成（CI）系统中，用于替代之前依赖虚拟机的测试方案。在传统方案中，虚拟机的使用不仅带来了额外的开销，还增加了测试环境的复杂性，因为每个虚拟机都需要完整的操作系统实例来运行和模拟 eBPF 程序，这导致了大量的 CPU 占用和内存消耗。 

通过采用 bpftime，这家公司得以在用户态环境下直接运行和测试 eBPF 程序，完全绕过了虚拟机层。bpftime 提供了一个高效的用户态 eBPF 运行时框架，不需要虚拟化技术的介入。由于测试在用户态进行，eBPF 程序的执行速度更快，测试反馈更及时，从而提升了整体开发效率。通过这种集成，bpftime 帮助该公司节约了大量的 CPU 时间和性能资源，同时将测试周期缩短了 30% 以上。这种优化不仅加速了产品的迭代开发进程，也提升了整体的测试自动化水平，使得他们能够更快速、更频繁地发布新版本和功能更新。 

 

有一家专注于网络安全的公司将 bpftime 集成到其安全产品中，显著提高了对恶意代码注入攻击的防护能力。传统安全工具依赖静态规则或签名库，难以应对复杂的零日漏洞攻击。而 bpftime 通过用户态 eBPF 运行时，实时监控程序行为，动态检测并阻止可疑的内存操作或异常函数调用，从而有效防御代码注入攻击。此外，bpftime 还具备快速应用热补丁的能力，能够在不影响系统正常运行的情况下，实时修复漏洞。这种基于行为分析的防护机制，使该公司的安全产品在面对未知威胁时具备了更高的灵活性和防护效率。 

 

 

目前，许多网络运营商依赖 DPDK（Data Plane Development Kit）来实现高性能的数据包处理和转发。DPDK 作为一种内核旁路方案，将数据包处理移至用户态，带来了显著的吞吐量和性能提升。然而，这种架构也限制了使用基于内核网络协议栈的 eBPF 工具，难以灵活地进行网络管理和加速操作，导致网络服务提供商无法充分利用 eBPF 的灵活性和可扩展性。 

为应对这一挑战，bpftime 提供了用户态 XDP 支持，开辟了新的解决方案。一家网络运营商正在内部试点部署 bpftime，利用其高效的用户态 XDP 技术提升数据包处理效率。bpftime 允许直接在用户态运行 eBPF 程序，在 DPDK 中实现与内核 eBPF 类似的网络加速功能，同时减少了数据传输过程中的延迟。这项技术有望显著提升用户体验，并减少对基于 DPDK 方案的网络架构的改造和重构需求。预计在未来大规模部署后，bpftime 将帮助该运营商节省数十万美元的运营成本。 

 

数家软件开发公司使用 bpftime 实现用户态的错误注入和测试需求，这是因为内核 eBPF 不支持修改用户态函数的执行流程，尤其在使用 uprobe 时，无法灵活地改变用户态程序的行为。而 bpftime 则通过其用户态 eBPF 运行时，支持在用户态直接进行函数的追踪与修改，允许开发者在不中断或重启程序的情况下，动态地注入错误或调整函数执行逻辑。这种能力大大提升了测试的灵活性，帮助企业实现更精细的错误注入测试和更快速的故障排查，从而提高了整体的开发和运维效率。 

 

### 行业难题 

 

eBPF 是一种强大的技术，广泛应用于系统监控、网络性能优化和安全管理中。然而，eBPF 主要依赖于 Linux 内核的运行时环境，这为很多企业带来了挑战，特别是在使用低版本内核或非特权容器环境时，内核 eBPF 无法充分发挥其作用。这种局限性限制了 eBPF 在某些关键场景中的应用，也阻碍了企业提升系统的可观测性和安全性的能力。 

bpftime 的出现很好地解决了这一行业难题。作为一个高性能的用户态 eBPF 运行时框架，bpftime 通过绕过内核，直接在用户态运行 eBPF 程序，提供了一个更加灵活、兼容性更强的解决方案。无论是在老旧的操作系统上，还是在内核权限受限的环境中，bpftime 都能够正常运行，为企业带来更广泛的应用场景。这一技术突破极大扩展了 eBPF 的使用范围，特别是在云服务、网络安全、性能调优等领域，为企业节省了大量的系统升级成本和硬件开销。 

通过 bpftime，企业不仅可以轻松应对系统升级的技术障碍，还能够在不依赖内核权限的情况下，获取到精准的可观测性数据，从而更好地优化系统性能和进行安全防护。bpftime 的模块化设计使得它能够快速集成新的功能和事件类型，推动创新并降低企业开发和运维成本。例如，一些公司已通过 bpftime 在用户态实现了高效的错误注入测试和实时漏洞检测，这为他们提供了更灵活的开发环境，缩短了测试周期并提高了故障排查效率。 

此外，bpftime 还为网络运营商带来了全新的网络加速解决方案。它通过用户态的 XDP 支持，实现了与内核 eBPF 类似的网络加速能力，并在 DPDK 环境下成功试点部署，显著提升了数据包处理的效率。这样的创新不仅提升了网络服务的质量，还降低了企业对内核网络协议栈的依赖，帮助他们节省了大量的运营成本。 

 

### 社会影响

 

bpftime 在技术普及和社会影响方面的贡献显著，尤其是在提升技术社区的协作和帮助中小企业降低技术门槛方面，发挥了重要作用。通过 bpftime 的用户态 eBPF 运行时框架，企业和开发者无需再依赖内核权限即可实现高效的系统监控与安全管理。这一技术突破让许多中小型企业，尤其是资源有限、无法进行大规模系统升级的公司，能够轻松部署先进的性能监控和网络优化工具。通过活跃的开源社区，开发者可以共同贡献代码、分享经验、探索新的应用场景。bpftime 的灵活架构不仅允许开发者快速进行原型设计和功能实验，还为研究人员提供了一个强大的工具，用于创新系统架构和性能优化方案。这种模块化设计极大缩短了开发周期，推动了技术创新的加速。此外，bpftime 还推动了 eBPF 技术在不同行业的普及，带动了更多开发者和企业加入到开源社区中，共同推动这一技术的发展。 

  

在技术传播方面，bpftime 社区在全球范围内取得了显著成效。我们团队主导的 eBPF 教程项目（https://github.com/eunomia-bpf/bpf-developer-tutorial）已经成为 Github 上 star 数最多的 eBPF 教程资源，帮助了全球众多开发者入门和精通 eBPF 技术。这一教程不仅在 Github 上获得了广泛关注，还通过多个技术平台发布，累积了上百万的浏览量，显现出强大的社会影响力。 

 

### 奖项与宣传

 

bpftime 项目自发布以来，获得了开源社区的广泛关注与认可，并在多个技术会议，如 Linux Plumbers（Linux 内核领域的顶级会议）、OSS Summit Europe、eBPF summit、第二届 eBPF 开发者大会中进行了技术分享，并参加了 GSOC 和 OSPP 2024 的暑期活动。项目的开源贡献获得了技术社区的高度评价，持续推动了 eBPF 技术的发展与普及。 

 

- Linux Plumbers: https://lpc.events/event/17/contributions/1639/ 
- Hack news: https://news.ycombinator.com/item?id=38268958 
- eBPF summit: https://www.youtube.com/watch?v=YZbCBaTTkeE 
- 第二届 eBPF 开发者大会: https://www.bilibili.com/video/BV1vJ4m1H7Mh 
- GSOC： https://summerofcode.withgoogle.com/programs/2024/projects/eBI5l9e7 