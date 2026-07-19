# hXDP: Efficient Software Packet Processing on FPGA NICs

# hXDP: 在 FPGA 网卡上高效的软件包处理

**Authors:** Marco Spaziani Brunella, Giacomo Belocchi, Marco Bonola, Salvatore Pontarelli, Giuseppe Siracusano, Giuseppe Bianchi, Aniello Cammarano, Alessandro Palumbo, Luca Petrucci, Roberto Bifulco

**作者：** Marco Spaziani Brunella, Giacomo Belocchi, Marco Bonola, Salvatore Pontarelli, Giuseppe Siracusano, Giuseppe Bianchi, Aniello Cammarano, Alessandro Palumbo, Luca Petrucci, Roberto Bifulco

**Affiliations:** Axbryd, CNIT, University of Rome Tor Vergata, NEC Laboratories Europe

**机构：** Axbryd、CNIT、罗马第二大学、NEC 欧洲实验室

**Conference:** 14th USENIX Symposium on Operating Systems Design and Implementation (OSDI '20)

**会议：** 第14届 USENIX 操作系统设计与实现研讨会 (OSDI '20)

---

## Abstract

## 摘要

FPGA accelerators on the NIC enable the offloading of expensive packet processing tasks from the CPU. However, FPGAs have limited resources that may need to be shared among diverse applications, and programming them is difficult.

NIC 上的 FPGA 加速器能够将昂贵的数据包处理任务从 CPU 上卸载。然而，FPGA 资源有限，可能需要在不同应用之间共享，并且对其编程也很困难。

We present a solution to run Linux's eXpress Data Path programs written in eBPF on FPGAs, using only a fraction of the available hardware resources while matching the performance of high-end CPUs.

我们提出了一种在 FPGA 上运行用 eBPF 编写的 Linux eXpress Data Path 程序的解决方案，仅使用一小部分可用硬件资源，同时达到高端 CPU 的性能。

The iterative execution model of eBPF is not a good fit for FPGA accelerators. Nonetheless, we show that many of the instructions of an eBPF program can be compressed, parallelized or completely removed, when targeting a purpose-built FPGA executor, thereby significantly improving performance.

eBPF 的迭代执行模型并不适合 FPGA 加速器。尽管如此，我们展示了当针对专用构建的 FPGA 执行器时，eBPF 程序的许多指令可以被压缩、并行化或完全移除，从而显著提高性能。

We leverage that to design hXDP, which includes (i) an optimizing-compiler that parallelizes and translates eBPF bytecode to an extended eBPF Instruction-set Architecture defined by us; a (ii) soft-CPU to execute such instructions on FPGA; and (iii) an FPGA-based infrastructure to provide XDP's maps and helper functions as defined within the Linux kernel.

我们利用这一点设计了 hXDP，它包括：(i) 一个优化编译器，将 eBPF 字节码并行化并转换为我们定义的扩展 eBPF 指令集架构；(ii) 一个在 FPGA 上执行这些指令的软 CPU；以及 (iii) 一个基于 FPGA 的基础设施，提供 Linux 内核中定义的 XDP 映射和辅助函数。

We implement hXDP on an FPGA NIC and evaluate it running real-world unmodified eBPF programs. Our implementation is clocked at 156.25MHz, uses about 15% of the FPGA resources, and can run dynamically loaded programs. Despite these modest requirements, it achieves the packet processing throughput of a high-end CPU core and provides a 10x lower packet forwarding latency.

我们在 FPGA 网卡上实现了 hXDP，并使用真实世界未修改的 eBPF 程序对其进行评估。我们的实现时钟频率为 156.25MHz，使用约 15% 的 FPGA 资源，并且可以运行动态加载的程序。尽管这些要求很低，它仍达到了高端 CPU 核心的数据包处理吞吐量，并提供了 10 倍更低的数据包转发延迟。

---

## 1. Introduction

## 1. 引言

FPGA-based NICs have recently emerged as a valid option to offload CPUs from packet processing tasks, due to their good performance and re-programmability.

基于 FPGA 的网卡最近成为从 CPU 卸载数据包处理任务的有效选择，因为它们具有良好的性能和可重编程性。

Compared to other NIC-based accelerators, such as network processing ASICs [8] or many-core System-on-Chip SmartNICs [40], FPGA NICs provide the additional benefit of supporting diverse accelerators for a wider set of applications [42], thanks to their embedded hardware re-programmability.

与其他基于网卡的加速器（如网络处理 ASIC [8] 或多核片上系统 SmartNIC [40]）相比，FPGA 网卡由于其嵌入式硬件可重编程性，为更广泛的应用集提供支持多样化加速器的额外优势 [42]。

Notably, Microsoft has been advocating for the introduction of FPGA NICs, because of their ability to use the FPGAs also for tasks such as machine learning [13, 14].

值得注意的是，微软一直在倡导引入 FPGA 网卡，因为它们能够将 FPGA 也用于机器学习等任务 [13, 14]。

FPGA NICs play another important role in 5G telecommunication networks, where they are used for the acceleration of radio access network functions [11, 28, 39, 58].

FPGA 网卡在 5G 电信网络中也发挥着重要作用，它们被用于加速无线接入网络功能 [11, 28, 39, 58]。

In these deployments, the FPGAs could host multiple functions to provide higher levels of infrastructure consolidation, since physical space availability may be limited.

在这些部署中，由于物理空间可用性可能有限，FPGA 可以承载多个功能以提供更高级别的基础设施整合。

For instance, this is the case in smart cities [55], 5G local deployments, e.g., in factories [44,47], and for edge computing in general [6,30].

例如，在智能城市 [55]、5G 本地部署（例如在工厂中 [44,47]）以及边缘计算领域普遍存在这种情况 [6,30]。

Nonetheless, programming FPGAs is difficult, often requiring the establishment of a dedicated team composed of hardware specialists [18], which interacts with software and operating system developers to integrate the offloading solution with the system.

然而，FPGA 编程很困难，通常需要建立一个由硬件专家组成的专门团队 [18]，该团队与软件和操作系统开发人员交互，以将卸载解决方案集成到系统中。

Furthermore, previous work that simplifies network functions programming on FPGAs assumes that a large share of the FPGA is dedicated to packet processing [1, 45, 56], reducing the ability to share the FPGA with other accelerators.

此外，简化 FPGA 上网络功能编程的先前工作假设 FPGA 的很大一部分专用于数据包处理 [1, 45, 56]，从而降低了与其他加速器共享 FPGA 的能力。

In this paper, our goal is to provide a more general and easy-to-use solution to program packet processing on FPGA NICs, using little FPGA resources, while seamlessly integrating with existing operating systems.

在本文中，我们的目标是提供一个更通用且易于使用的解决方案来在 FPGA 网卡上编程数据包处理，使用少量 FPGA 资源，同时与现有操作系统无缝集成。

We build towards this goal by presenting hXDP, a set of technologies that enables the efficient execution of the Linux's eXpress Data Path (XDP) [27] on FPGA.

我们通过呈现 hXDP 来朝着这个目标努力，hXDP 是一套能够在 FPGA 上高效执行 Linux eXpress Data Path (XDP) [27] 的技术。

XDP leverages the eBPF technology to provide secure programmable packet processing within the Linux kernel, and it is widely used by the Linux's community in productive environments.

XDP 利用 eBPF 技术在 Linux 内核中提供安全的可编程数据包处理，并在生产环境中被 Linux 社区广泛使用。

hXDP provides full XDP support, allowing users to dynamically load and run their unmodified XDP programs on the FPGA.

hXDP 提供完整的 XDP 支持，允许用户在 FPGA 上动态加载和运行未修改的 XDP 程序。

The eBPF technology is originally designed for sequential execution on a high-performance RISC-like register machine, which makes it challenging to run XDP programs effectively on FPGA.

eBPF 技术最初是为在高性能类 RISC 寄存器机器上顺序执行而设计的，这使得在 FPGA 上有效运行 XDP 程序具有挑战性。

That is, eBPF is designed for server CPUs with high clock frequency and the ability to execute many of the sequential eBPF instructions per second.

也就是说，eBPF 是为具有高时钟频率并能够每秒执行许多顺序 eBPF 指令的服务器 CPU 设计的。

Instead, FPGAs favor a widely parallel execution model with clock frequencies that are 5-10x lower than those of high-end CPUs.

相反，FPGA 偏向于广泛并行的执行模型，时钟频率比高端 CPU 低 5-10 倍。

As such, a straightforward implementation of the eBPF iterative execution model on FPGA is likely to provide low packet forwarding performance.

因此，在 FPGA 上直接实现 eBPF 迭代执行模型可能会提供较低的数据包转发性能。

Furthermore, the hXDP design should implement arbitrary XDP programs while using little hardware resources, in order to keep FPGA's resources free for other accelerators.

此外，hXDP 设计应该在使用少量硬件资源的同时实现任意 XDP 程序，以便为其他加速器保留 FPGA 资源。

We address the challenge performing a detailed analysis of the eBPF Instruction Set Architecture (ISA) and of the existing XDP programs, to reveal and take advantage of opportunities for optimization.

我们通过对 eBPF 指令集架构 (ISA) 和现有 XDP 程序进行详细分析来应对这一挑战，以揭示和利用优化机会。

First, we identify eBPF instructions that can be safely removed, when not running in the Linux kernel context.

首先，我们识别出在不在 Linux 内核上下文中运行时可以安全删除的 eBPF 指令。

For instance, we remove data boundary checks and variable zero-ing instructions by providing targeted hardware support.

例如，我们通过提供有针对性的硬件支持来删除数据边界检查和变量清零指令。

Second, we define extensions to the eBPF ISA to introduce 3-operand instructions, new 6B load/store instructions and a new parametrized program exit instruction.

其次，我们定义了 eBPF ISA 的扩展，引入了 3 操作数指令、新的 6B 加载/存储指令和新的参数化程序退出指令。

Finally, we leverage eBPF instruction-level parallelism, performing a static analysis of the programs at compile time, which allows us to execute several eBPF instructions in parallel.

最后，我们利用 eBPF 指令级并行性，在编译时对程序进行静态分析，这使我们能够并行执行多个 eBPF 指令。

We design hXDP to implement these optimizations, and to take full advantage of the on-NIC execution environment, e.g., avoiding unnecessary PCIe transfers.

我们设计 hXDP 来实现这些优化，并充分利用网卡上的执行环境，例如避免不必要的 PCIe 传输。

Our design includes: (i) a compiler to translate XDP programs' bytecode to the extended hXDP ISA; (ii) a self-contained FPGA IP Core module that implements the extended ISA alongside several other low-level optimizations; (iii) and the toolchain required to dynamically load and interact with XDP programs running on the FPGA NIC.

我们的设计包括：(i) 一个将 XDP 程序字节码转换为扩展 hXDP ISA 的编译器；(ii) 一个独立的 FPGA IP 核模块，实现扩展 ISA 以及其他几个低级优化；(iii) 以及动态加载和与运行在 FPGA 网卡上的 XDP 程序交互所需的工具链。

To evaluate hXDP we provide an open source implementation for the NetFPGA [60].

为了评估 hXDP，我们为 NetFPGA [60] 提供了一个开源实现。

We test our implementation using the XDP example programs provided by the Linux source code, and using two real-world applications: a simple stateful firewall; and Facebook's Katran load balancer.

我们使用 Linux 源代码提供的 XDP 示例程序以及两个真实世界的应用程序来测试我们的实现：一个简单的有状态防火墙；以及 Facebook 的 Katran 负载均衡器。

hXDP can match the packet forwarding throughput of a multi-GHz server CPU core, while providing a much lower forwarding latency.

hXDP 可以匹配多 GHz 服务器 CPU 核心的数据包转发吞吐量，同时提供更低的转发延迟。

This is achieved despite the low clock frequency of our prototype (156MHz) and using less than 15% of the FPGA resources.

尽管我们的原型时钟频率较低（156MHz）并且使用不到 15% 的 FPGA 资源，但仍实现了这一点。

In summary, we contribute:

总之，我们的贡献包括：

• the design of hXDP including: the hardware design; the companion compiler; and the software toolchain;

• hXDP 的设计，包括：硬件设计；配套编译器；以及软件工具链；

• the implementation of a hXDP IP core for the NetFPGA

• 为 NetFPGA 实现的 hXDP IP 核

• a comprehensive evaluation of hXDP when running real-world use cases, comparing it with an x86 Linux server.

• 在运行真实世界用例时对 hXDP 进行全面评估，并将其与 x86 Linux 服务器进行比较。

• a microbenchmark-based comparison of the hXDP implementation with a Netronome NFP4000 SmartNIC, which provides partial eBPF offloading support.

• 基于微基准测试的 hXDP 实现与 Netronome NFP4000 SmartNIC 的比较，后者提供部分 eBPF 卸载支持。

---

## 2. Concept and Overview

## 2. 概念和概述

In this section we discuss hXDP goals, scope and requirements, we provide background information about XDP, and finally we present an overview of the hXDP design.

在本节中，我们讨论 hXDP 的目标、范围和需求，提供有关 XDP 的背景信息，最后呈现 hXDP 设计的概述。

### 2.1 Goals and Requirements

### 2.1 目标和需求

**Goals** Our main goal is to provide the ability to run XDP programs efficiently on FPGA NICs, while using little FPGA's hardware resources (See Figure 1).

**目标** 我们的主要目标是提供在 FPGA 网卡上高效运行 XDP 程序的能力，同时使用少量 FPGA 硬件资源（见图 1）。

A little use of the FPGA resources is especially important, since it enables extra consolidation by packing different application-specific accelerators on the same FPGA.

少量使用 FPGA 资源尤为重要，因为它通过在同一 FPGA 上打包不同的应用程序特定加速器来实现额外的整合。

The choice of supporting XDP is instead motivated by a twofold benefit brought by the technology: it readily enables NIC offloading for already deployed XDP programs; it provides an on-NIC programming model that is already familiar to a large community of Linux programmers.

选择支持 XDP 的动机是该技术带来的双重好处：它可以轻松地为已部署的 XDP 程序启用网卡卸载；它提供了一个网卡上编程模型，这对大量 Linux 程序员来说已经很熟悉。

Enabling such a wider access to the technology is important since many of the mentioned edge deployments are not necessarily handled by hyperscale companies.

实现对该技术的更广泛访问很重要，因为许多提到的边缘部署不一定由超大规模公司处理。

Thus, the companies developing and deploying applications may not have resources to invest in highly specialized and diverse professional teams of developers, while still needing some level of customization to achieve challenging service quality and performance levels.

因此，开发和部署应用程序的公司可能没有资源投资于高度专业化和多样化的专业开发团队，但仍然需要一定程度的定制来实现具有挑战性的服务质量和性能水平。

In this sense, hXDP provides a familiar programming model that does not require developers to learn new programming paradigms, such as those introduced by devices that support P4 [7] or FlowBlaze [45].

从这个意义上说，hXDP 提供了一个熟悉的编程模型，不需要开发人员学习新的编程范式，例如支持 P4 [7] 或 FlowBlaze [45] 的设备所引入的范式。

**Non-Goals** Unlike previous work targeting FPGA NICs [1, 45, 56], hXDP does not assume the FPGA to be dedicated to network processing tasks.

**非目标** 与针对 FPGA 网卡的先前工作 [1, 45, 56] 不同，hXDP 不假设 FPGA 专用于网络处理任务。

Because of that, hXDP adopts an iterative processing model, which is in stark contrast with the pipelined processing model supported by previous work.

因此，hXDP 采用迭代处理模型，这与先前工作支持的流水线处理模型形成鲜明对比。

The iterative model requires a fixed amount of resources, no matter the complexity of the program being implemented.

迭代模型需要固定数量的资源，无论实现的程序复杂性如何。

Instead, in the pipeline model the resource requirement is dependent on the implemented program complexity, since programs are effectively "unrolled" in the FPGA.

相反，在流水线模型中，资源需求取决于实现的程序复杂性，因为程序实际上在 FPGA 中被"展开"。

In fact, hXDP provides dynamic runtime loading of XDP programs, whereas solutions like P4->NetFPGA [56] or FlowBlaze need to often load a new FPGA bitstream when changing application.

事实上，hXDP 提供 XDP 程序的动态运行时加载，而像 P4->NetFPGA [56] 或 FlowBlaze 这样的解决方案在更改应用程序时通常需要加载新的 FPGA 位流。

As such, hXDP is not designed to be faster at processing packets than those designs. Instead, hXDP aims at freeing precious CPU resources, which can then be dedicated to workloads that cannot run elsewhere, while providing similar or better performance than the CPU.

因此，hXDP 的设计并不是要比这些设计更快地处理数据包。相反，hXDP 旨在释放宝贵的 CPU 资源，然后可以将其专用于无法在其他地方运行的工作负载，同时提供与 CPU 相似或更好的性能。

**Requirements** Given the above discussion, we can derive three high-level requirements for hXDP:

**需求** 根据上述讨论，我们可以推导出 hXDP 的三个高级需求：

1. it should execute unmodified compiled XDP programs, and support the XDP frameworks' toolchain, e.g., dynamic program loading and userspace access to maps;

1. 它应该执行未修改的编译 XDP 程序，并支持 XDP 框架的工具链，例如，动态程序加载和用户空间对映射的访问；

2. it should provide packet processing performance at least comparable to that of a high-end CPU core;

2. 它应该提供至少与高端 CPU 核心相当的数据包处理性能；

3. it should require a small amount of the FPGA's hardware resources.

3. 它应该需要少量的 FPGA 硬件资源。

Before presenting a more detailed description of the hXDP concept, we now give a brief background about XDP.

在更详细地描述 hXDP 概念之前，我们现在简要介绍一下 XDP 的背景。

### 2.2 XDP Primer

### 2.2 XDP 入门

XDP allows programmers to inject programs at the NIC driver level, so that such programs are executed before a network packet is passed to the Linux's network stack.

XDP 允许程序员在网卡驱动程序级别注入程序，以便在将网络数据包传递到 Linux 网络堆栈之前执行这些程序。

This provides an opportunity to perform custom packet processing at a very early stage of the packet handling, limiting overheads and thus providing high-performance.

这提供了在数据包处理的非常早期阶段执行自定义数据包处理的机会，限制了开销，从而提供了高性能。

At the same time, XDP allows programmers to leverage the Linux's kernel, e.g., selecting a subset of packets that should be processed by its network stack, which helps with compatibility and ease of development.

同时，XDP 允许程序员利用 Linux 内核，例如，选择应由其网络堆栈处理的数据包子集，这有助于兼容性和开发便利性。

XDP is part of the Linux kernel since release 4.18, and it is widely used in production environments [4, 17, 54].

XDP 自 4.18 版本以来一直是 Linux 内核的一部分，并在生产环境中被广泛使用 [4, 17, 54]。

In most of these use cases, e.g., load balancing [17] and packet filtering [4], a majority of the received network packets is processed entirely within XDP.

在这些用例中的大多数，例如负载均衡 [17] 和数据包过滤 [4]，大多数接收到的网络数据包完全在 XDP 中处理。

The production deployments of XDP have also pushed developers to optimize and minimize the XDP overheads, which now appear to be mainly related to the Linux driver model, as thoroughly discussed in [27].

XDP 的生产部署也推动开发人员优化和最小化 XDP 开销，这些开销现在似乎主要与 Linux 驱动程序模型有关，正如 [27] 中详细讨论的那样。

XDP programs are based on the Linux's eBPF technology.

XDP 程序基于 Linux 的 eBPF 技术。

eBPF provides an in-kernel virtual machine for the sandboxed execution of small programs within the kernel context.

eBPF 提供了一个内核虚拟机，用于在内核上下文中沙箱执行小程序。

An overview of the eBPF architecture and workflow is provided in Figure 2.

图 2 提供了 eBPF 架构和工作流程的概述。

In its current version, the eBPF virtual machine has 11 64b registers: r0 holds the return value from in-kernel functions and programs, r1 − r5 are used to store arguments that are passed to in-kernel functions, r6 − r9 are registers that are preserved during function calls and r10 stores the frame pointer to access the stack.

在当前版本中，eBPF 虚拟机有 11 个 64 位寄存器：r0 保存内核函数和程序的返回值，r1 − r5 用于存储传递给内核函数的参数，r6 − r9 是在函数调用期间保留的寄存器，r10 存储访问堆栈的帧指针。

The eBPF virtual machine has a well-defined ISA composed of more than 100 fixed length instructions (64b).

eBPF 虚拟机具有一个定义明确的 ISA，由 100 多条固定长度指令（64 位）组成。

The instructions give access to different functional units, such as ALU32, ALU64, branch and memory.

这些指令可以访问不同的功能单元，例如 ALU32、ALU64、分支和内存。

Programmers usually write an eBPF program using the C language with some restrictions, which simplify the static verification of the program.

程序员通常使用带有一些限制的 C 语言编写 eBPF 程序，这简化了程序的静态验证。

Examples of restrictions include forbidden unbounded cycles, limited stack size, lack of dynamic memory allocation, etc.

限制的例子包括禁止无界循环、有限的堆栈大小、缺乏动态内存分配等。

To overcome some of these limitations, eBPF programs can use helper functions that implement some common operations, such as checksum computations, and provide access to protected operations, e.g., reading certain kernel memory areas.

为了克服其中一些限制，eBPF 程序可以使用辅助函数来实现一些常见操作，例如校验和计算，并提供对受保护操作的访问，例如读取某些内核内存区域。

eBPF programs can also access kernel memory areas called maps, i.e., kernel memory locations that essentially resemble tables.

eBPF 程序还可以访问称为映射的内核内存区域，即本质上类似于表的内核内存位置。

Maps are declared and configured at compile time to implement different data structures, specifying the type, size and an ID.

映射在编译时声明和配置以实现不同的数据结构，指定类型、大小和 ID。

For instance, eBPF programs can use maps to implement arrays and hash tables.

例如，eBPF 程序可以使用映射来实现数组和哈希表。

An eBPF program can interact with map's locations by means of pointer deference, for un-structured data access, or by invoking specific helper functions for structured data access, e.g., a lookup on a map configured as a hash table.

eBPF 程序可以通过指针解引用与映射的位置交互，用于非结构化数据访问，或者通过调用特定的辅助函数进行结构化数据访问，例如，在配置为哈希表的映射上进行查找。

Maps are especially important since they are the only mean to keep state across program executions, and to share information with other eBPF programs and with programs running in user space.

映射特别重要，因为它们是在程序执行之间保持状态以及与其他 eBPF 程序和用户空间中运行的程序共享信息的唯一手段。

In fact, a map can be accessed using its ID by any other running eBPF program and by the control application running in user space.

事实上，任何其他正在运行的 eBPF 程序和在用户空间中运行的控制应用程序都可以使用其 ID 访问映射。

User space programs can load eBPF programs and read/write maps either using the libbf library or frontends such as the BCC toolstack.

用户空间程序可以使用 libbpf 库或诸如 BCC 工具栈之类的前端来加载 eBPF 程序并读/写映射。

XDP programs are compiled using LLVM or GCC, and the generated ELF object file is loaded trough the bpf syscall, specifying the XDP hook.

XDP 程序使用 LLVM 或 GCC 编译，生成的 ELF 目标文件通过 bpf 系统调用加载，指定 XDP 钩子。

Before the actual loading of a program, the kernel verifier checks if it is safe, then the program is attached to the hook, at the network driver level.

在实际加载程序之前，内核验证器检查它是否安全，然后将程序附加到网络驱动程序级别的钩子。

Whenever the network driver receives a packet, it triggers the execution of the registered programs, which starts from a clean context.

每当网络驱动程序接收到数据包时，它就会触发已注册程序的执行，从干净的上下文开始。

### 2.3 Challenges

### 2.3 挑战

To grasp an intuitive understanding of the design challenge involved in supporting XDP on FPGA, we now consider the example of an XDP program that implements a simple stateful firewall for checking the establishment of bi-directional TCP or UDP flows, and to drop flows initiated from an external location.

为了直观地理解在 FPGA 上支持 XDP 所涉及的设计挑战，我们现在考虑一个 XDP 程序的例子，该程序实现一个简单的有状态防火墙，用于检查双向 TCP 或 UDP 流的建立，并丢弃从外部位置发起的流。

We will use this function as a running example throughout the paper, since despite its simplicity, it is a realistic and widely deployed function.

我们将在整篇论文中使用这个函数作为运行示例，因为尽管它很简单，但它是一个现实且广泛部署的功能。

The simple firewall first performs a parsing of the Ethernet, IP and Transport protocol headers to extract the flow's 5-tuple (IP addresses, port numbers, protocol).

简单防火墙首先解析以太网、IP 和传输协议头以提取流的 5 元组（IP 地址、端口号、协议）。

Then, depending on the input port of the packet (i.e., external or internal) it either looks up an entry in a hashmap, or creates it.

然后，根据数据包的输入端口（即外部或内部），它要么在哈希映射中查找条目，要么创建它。

The hashmap key is created using an absolute ordering of the 5 tuple values, so that the two directions of the flow will map to the same hash.

哈希映射键使用 5 元组值的绝对排序创建，以便流的两个方向将映射到相同的哈希。

Finally, the function forwards the packet if the input port is internal or if the hashmap lookup retrieved an entry, otherwise the packet is dropped.

最后，如果输入端口是内部的或者哈希映射查找检索到了条目，则该函数转发数据包，否则丢弃数据包。

A C program describing this simple firewall function is compiled to 71 eBPF instructions.

描述这个简单防火墙功能的 C 程序被编译成 71 条 eBPF 指令。

We can build a rough idea of the potential best-case speed of this function running on an FPGA-based eBPF executor, assuming that each eBPF instruction requires 1 clock cycle to be executed, that clock cycles are not spent for any other operation, and that the FPGA has a 156MHz clock rate, which is common in FPGA NICs [60].

我们可以粗略地了解在基于 FPGA 的 eBPF 执行器上运行此功能的潜在最佳速度，假设每条 eBPF 指令需要 1 个时钟周期才能执行，时钟周期不花费在任何其他操作上，并且 FPGA 的时钟频率为 156MHz，这在 FPGA 网卡中很常见 [60]。

In such a case, a naive FPGA implementation that implements the sequential eBPF executor would provide a maximum throughput of 2.8 Million packets per second (Mpps).

在这种情况下，实现顺序 eBPF 执行器的朴素 FPGA 实现将提供每秒 280 万个数据包（Mpps）的最大吞吐量。

Notice that this is a very optimistic upper-bound performance, which does not take into account other, often unavoidable, potential sources of overhead, such as memory access, queue management, etc.

请注意，这是一个非常乐观的上限性能，没有考虑其他通常不可避免的潜在开销来源，例如内存访问、队列管理等。

For comparison, when running on a single core of a high-end server CPU clocked at 3.7GHz, and including also operating system overhead and the PCIe transfer costs, the XDP simple firewall program achieves a throughput of 7.4 Million packets per second (Mpps).

相比之下，当在时钟频率为 3.7GHz 的高端服务器 CPU 的单个核心上运行时，并且还包括操作系统开销和 PCIe 传输成本，XDP 简单防火墙程序实现了每秒 740 万个数据包（Mpps）的吞吐量。

Since it is often undesired or not possible to increase the FPGA clock rate, e.g., due to power constraints, in the lack of other solutions the FPGA-based executor would be 2-3x slower than the CPU core.

由于通常不希望或不可能提高 FPGA 时钟频率，例如由于功率限制，在缺乏其他解决方案的情况下，基于 FPGA 的执行器将比 CPU 核心慢 2-3 倍。

Furthermore, existing solutions to speed-up sequential code execution, e.g., superscalar architectures, are too expensive in terms of hardware resources to be adopted in this case.

此外，现有的加速顺序代码执行的解决方案，例如超标量架构，在硬件资源方面过于昂贵，无法在这种情况下采用。

In fact, in a superscalar architecture the speed-up is achieved leveraging instruction-level parallelism at runtime.

实际上，在超标量架构中，加速是通过在运行时利用指令级并行性来实现的。

However, the complexity of the hardware required to do so grows exponentially with the number of instructions being checked for parallel execution.

然而，执行此操作所需的硬件复杂性随着被检查以进行并行执行的指令数量呈指数增长。

This rules out re-using general purpose soft-core designs, such as those based on RISC-V [22, 25].

这排除了重用通用软核设计的可能性，例如基于 RISC-V [22, 25] 的设计。

### 2.4 hXDP Overview

### 2.4 hXDP 概述

hXDP addresses the outlined challenge by taking a software-hardware co-design approach.

hXDP 通过采用软硬件协同设计方法来应对上述挑战。

In particular, hXDP provides both a compiler and the corresponding hardware module.

特别是，hXDP 提供编译器和相应的硬件模块。

The compiler takes advantage of eBPF ISA optimization opportunities, leveraging hXDP's hardware module features that are introduced to simplify the exploitation of such opportunities.

编译器利用 eBPF ISA 优化机会，利用 hXDP 的硬件模块功能，这些功能被引入以简化对这些机会的利用。

Effectively, we design a new ISA that extends the eBPF ISA, specifically targeting the execution of XDP programs.

实际上，我们设计了一个新的 ISA，扩展了 eBPF ISA，专门针对 XDP 程序的执行。

The compiler optimizations perform transformations at the eBPF instruction level: remove unnecessary instructions; replace instructions with newly defined more concise instructions; and parallelize instructions execution.

编译器优化在 eBPF 指令级别执行转换：删除不必要的指令；用新定义的更简洁的指令替换指令；并行化指令执行。

All the optimizations are performed at compile-time, moving most of the complexity to the software compiler, thereby reducing the target hardware complexity.

所有优化都在编译时执行，将大部分复杂性转移到软件编译器，从而降低目标硬件的复杂性。

We describe the optimizations and the compiler in Section 3.

我们在第 3 节中描述优化和编译器。

Accordingly, the hXDP hardware module implements an infrastructure to run up to 4 instructions in parallel, implementing a Very Long Instruction Word (VLIW) soft-processor.

相应地，hXDP 硬件模块实现了一个基础设施，可以并行运行多达 4 条指令，实现一个超长指令字（VLIW）软处理器。

The VLIW soft-processor does not provide any runtime program optimization, e.g., branch prediction, instruction re-ordering, etc.

VLIW 软处理器不提供任何运行时程序优化，例如分支预测、指令重新排序等。

We rely entirely on the compiler to optimize XDP programs for high-performance execution, thereby freeing the hardware module of complex mechanisms that would use more hardware resources.

我们完全依赖编译器来优化 XDP 程序以实现高性能执行，从而使硬件模块免于使用更多硬件资源的复杂机制。

We describe the hXDP hardware design in Section 4.

我们在第 4 节中描述 hXDP 硬件设计。

Ultimately, the hXDP hardware component is deployed as a self-contained IP core module to the FPGA.

最终，hXDP 硬件组件作为独立的 IP 核模块部署到 FPGA。

The module can be interfaced with other processing modules if needed, or just placed as a bump-in-the-wire between the NIC's port and its PCIe driver towards the host system.

该模块可以根据需要与其他处理模块连接，或者只是作为网卡端口与其朝向主机系统的 PCIe 驱动程序之间的线路中继。

The hXDP software toolchain, which includes the compiler, provides all the machinery to use hXDP within a Linux operating system.

hXDP 软件工具链包括编译器，提供了在 Linux 操作系统中使用 hXDP 的所有机制。

From a programmer perspective, a compiled eBPF program could be therefore interchangeably executed in-kernel or on the FPGA, as shown in Figure 2.

从程序员的角度来看，因此编译的 eBPF 程序可以在内核中或在 FPGA 上交替执行，如图 2 所示。

---

## 3. hXDP Compiler

## 3. hXDP 编译器

In this section we describe the hXDP instruction-level optimizations, and the compiler design to implement them.

在本节中，我们描述 hXDP 指令级优化以及实现它们的编译器设计。

### 3.1 Instructions Reduction

### 3.1 指令削减

The eBPF technology is designed to enable execution within the Linux kernel, for which it requires programs to include a number of extra instructions, which are then checked by the kernel's verifier.

eBPF 技术旨在在 Linux 内核中启用执行，为此它要求程序包含许多额外的指令，然后由内核的验证器进行检查。

When targeting a dedicated eBPF executor implemented on FPGA, most such instructions could be safely removed, or they can be replaced by cheaper embedded hardware checks.

当针对在 FPGA 上实现的专用 eBPF 执行器时，大多数此类指令可以安全地删除，或者可以用更便宜的嵌入式硬件检查替换。

Two relevant examples are instructions for memory boundary checks and memory zero-ing.

两个相关的例子是用于内存边界检查和内存清零的指令。

Boundary checks are required by the eBPF verifier to ensure that programs only read valid memory locations, whenever a pointer operation is involved.

eBPF 验证器需要边界检查以确保程序仅在涉及指针操作时读取有效的内存位置。

For instance, this is relevant for accessing the socket buffer containing the packet data, during parsing.

例如，这与在解析期间访问包含数据包数据的套接字缓冲区有关。

Here, a required check is to verify that the packet is large enough to host the expected packet header.

在这里，所需的检查是验证数据包足够大以容纳预期的数据包头。

As shown in Figure 3, a single check like this may cost 3 instructions, and it is likely that such checks are repeated multiple times.

如图 3 所示，像这样的单个检查可能花费 3 条指令，并且很可能这些检查会重复多次。

In the simple firewall case, for instance, there are three such checks for the Ethernet, IP and L4 headers.

例如，在简单防火墙的情况下，对于以太网、IP 和 L4 头有三个这样的检查。

In hXDP we can safely remove these instructions, implementing the check directly in hardware.

在 hXDP 中，我们可以安全地删除这些指令，直接在硬件中实现检查。

Zero-ing is the process of setting a newly created variable to zero, and it is a common operation performed by programmers both for safety and for ensuring correct execution of their programs.

清零是将新创建的变量设置为零的过程，这是程序员为了安全和确保其程序正确执行而执行的常见操作。

A dedicated FPGA executor can provide hard guarantees that all relevant memory areas are zero-ed at program start, therefore making the explicit zero-ing of variables during initialization redundant.

专用的 FPGA 执行器可以提供硬保证，即在程序启动时所有相关内存区域都被清零，因此在初始化期间显式清零变量是多余的。

In the simple firewall function zero-ing requires 4 instructions, as shown in Figure 3.

在简单防火墙功能中，清零需要 4 条指令，如图 3 所示。

### 3.2 ISA Extension

### 3.2 ISA 扩展

To effectively reduce the number of instructions we define an ISA that enables a more concise description of the program.

为了有效地减少指令数量，我们定义了一个 ISA，使程序能够更简洁地描述。

Here, there are two factors at play to our advantage.

在这里，有两个因素对我们有利。

First, we can extend the ISA without accounting for constraints related to the need to support efficient Just-In-Time compilation.

首先，我们可以扩展 ISA，而无需考虑与支持高效即时编译需求相关的约束。

Second, our eBPF programs are part of XDP applications, and as such we can expect packet processing as the main program task.

其次，我们的 eBPF 程序是 XDP 应用程序的一部分，因此我们可以期望数据包处理是主要的程序任务。

Leveraging these two facts we define a new ISA that changes in three main ways the original eBPF ISA.

利用这两个事实，我们定义了一个新的 ISA，它在三个主要方面改变了原始 eBPF ISA。

**Operands number.** The first significant change has to deal with the inclusion of three-operand operations, in place of eBPF's two-operand ones.

**操作数数量。** 第一个重大变化是包含三操作数操作，以取代 eBPF 的双操作数操作。

Here, we believe that the eBPF's ISA selection of two-operand operations was mainly dictated by the assumption that an x86 ISA would be the final compilation target.

在这里，我们认为 eBPF 的 ISA 选择双操作数操作主要是由 x86 ISA 将是最终编译目标的假设所决定的。

Instead, using three-operand instructions allows us to implement an operation that would normally need two instructions with just a single instruction, as shown in Figure 4.

相反，使用三操作数指令允许我们用单条指令实现通常需要两条指令的操作，如图 4 所示。

**Load/store size.** The eBPF ISA includes byte-aligned memory load/store operations, with sizes of 1B, 2B, 4B and 8B.

**加载/存储大小。** eBPF ISA 包括字节对齐的内存加载/存储操作，大小为 1B、2B、4B 和 8B。

While these instructions are effective for most cases, we noticed that during packet processing the use of 6B load/store can reduce the number of instructions in common cases.

虽然这些指令在大多数情况下都很有效，但我们注意到在数据包处理期间使用 6B 加载/存储可以在常见情况下减少指令数量。

In fact, 6B is the size of an Ethernet MAC address, which is a commonly accessed field both to check the packet destination or to set a new one.

实际上，6B 是以太网 MAC 地址的大小，这是一个常用的访问字段，用于检查数据包目的地或设置新的目的地。

Extending the eBPF ISA with 6B load/store instructions often halves the required instructions.

使用 6B 加载/存储指令扩展 eBPF ISA 通常可以将所需指令减半。

**Parametrized exit.** The end of an eBPF program is marked by the exit instruction.

**参数化退出。** eBPF 程序的结束由退出指令标记。

In XDP, programs set the r0 to a value corresponding to the desired forwarding action (e.g., DROP, TX, etc), then, when a program exits the framework checks the r0 register to finally perform the forwarding action (see listing 4).

在 XDP 中，程序将 r0 设置为与所需转发操作相对应的值（例如，DROP、TX 等），然后，当程序退出时，框架检查 r0 寄存器以最终执行转发操作（参见清单 4）。

While this extension of the ISA only saves one (runtime) instruction per program, as we will see in Section 4, it will also enable more significant hardware optimizations.

虽然 ISA 的这种扩展每个程序只节省一条（运行时）指令，但正如我们将在第 4 节中看到的，它还将实现更重要的硬件优化。

### 3.3 Instruction Parallelism

### 3.3 指令并行性

Finally, we explore the opportunity to perform parallel processing of an eBPF program's instructions.

最后，我们探索并行处理 eBPF 程序指令的机会。

Here, it is important to notice that high-end superscalar CPUs are usually capable to execute multiple instructions in parallel, using a number of complex mechanisms such as speculative execution or out-of-order execution.

在这里，重要的是要注意高端超标量 CPU 通常能够并行执行多个指令，使用许多复杂的机制，例如推测执行或乱序执行。

However, on FPGAs the introduction of such mechanisms could incur significant hardware resources overheads.

然而，在 FPGA 上引入此类机制可能会产生大量硬件资源开销。

Therefore, we perform only a static analysis of the instruction-level parallelism of eBPF programs.

因此，我们仅对 eBPF 程序的指令级并行性进行静态分析。

To determine if two or more instructions can be parallelized, the three Bernstein conditions have to be checked [3].

为了确定两个或多个指令是否可以并行化，必须检查三个 Bernstein 条件 [3]。

Simplifying the discussion to the case of two instructions P1, P2:

将讨论简化为两个指令 P1、P2 的情况：

I1 ∩ O2 = ∅; O1 ∩ I2 = ∅; O2 ∩ O1 = ∅;

I1 ∩ O2 = ∅; O1 ∩ I2 = ∅; O2 ∩ O1 = ∅;

Where I1, I2 are the instructions' input sets (e.g. source operands and memory locations) and O1, O2 are their output sets.

其中 I1、I2 是指令的输入集（例如源操作数和内存位置），O1、O2 是它们的输出集。

The first two conditions imply that if any of the two instructions depends on the results of the computation of the other, those two instructions cannot be executed in parallel.

前两个条件意味着，如果两条指令中的任何一条依赖于另一条的计算结果，则这两条指令不能并行执行。

The last condition implies that if both instructions are storing the results on the same location, again they cannot be parallelized.

最后一个条件意味着，如果两条指令都将结果存储在同一位置，则同样不能并行化它们。

Verifying the Bernstein conditions and parallelizing instructions requires the design of a suitable compiler, which we describe next.

验证 Bernstein 条件和并行化指令需要设计合适的编译器，我们接下来将对此进行描述。

### 3.4 Compiler Design

### 3.4 编译器设计

We design a custom compiler to implement the optimizations outlined in this section and to transform XDP programs into a schedule of parallel instructions that can run with hXDP.

我们设计了一个定制编译器来实现本节中概述的优化，并将 XDP 程序转换为可以使用 hXDP 运行的并行指令调度。

The schedule can be visualized as a virtually infinite set of rows, each with multiple available spots, which need to be filled with instructions.

调度可以可视化为一个虚拟无限的行集，每行有多个可用位置，需要用指令填充。

The number of spots corresponds to the number of execution lanes of the target executor.

位置数量对应于目标执行器的执行通道数量。

The final objective of the compiler is to fit the given XDP program's instructions in the smallest number of rows.

编译器的最终目标是将给定 XDP 程序的指令放入最少的行数中。

To do so, the compiler performs five steps.

为此，编译器执行五个步骤。

**Control Flow Graph construction** First, the compiler performs a forward scan of the eBPF bytecode to identify the program's basic blocks, i.e., sequences of instructions that are always executed together.

**控制流图构建** 首先，编译器对 eBPF 字节码执行前向扫描以识别程序的基本块，即始终一起执行的指令序列。

The compiler identifies the first and last instructions of a block, and the control flow between blocks, by looking at branching instructions and jump destinations.

编译器通过查看分支指令和跳转目标来识别块的第一条和最后一条指令以及块之间的控制流。

With this information it can finally build the Control Flow Graph (CFG), which represents the basic blocks as nodes and the control flow as directed edges connecting them.

有了这些信息，它最终可以构建控制流图（CFG），它将基本块表示为节点，将控制流表示为连接它们的有向边。

**Peephole optimizations** Second, for each basic block the compiler performs the removal of unnecessary instructions (cf. Section 3.1), and the substitution of groups of eBPF instructions with an equivalent instruction of our extended ISA (cf. Section 3.2).

**窥孔优化** 其次，对于每个基本块，编译器执行不必要指令的删除（参见第 3.1 节），以及用我们扩展 ISA 的等效指令替换 eBPF 指令组（参见第 3.2 节）。

**Data Flow dependencies** Third, the compiler discovers Data Flow dependencies.

**数据流依赖** 第三，编译器发现数据流依赖。

This is done by implementing an iterative algorithm to analyze the CFG.

这是通过实现迭代算法来分析 CFG 来完成的。

The algorithm analyzes each block, building a data structure containing the block's input, output, defined, and used symbols.

该算法分析每个块，构建一个包含块的输入、输出、已定义和已使用符号的数据结构。

Here, a symbol is any distinct data value defined (and used) by the program.

在这里，符号是程序定义（和使用）的任何不同的数据值。

Once each block has its associated set of symbols, the compiler can use the CFG to compute data flow dependencies between instructions.

一旦每个块都有其关联的符号集，编译器就可以使用 CFG 来计算指令之间的数据流依赖关系。

This information is captured in per-instruction data dependency graphs (DDG).

此信息在每条指令的数据依赖图（DDG）中捕获。

**Instruction scheduling** Fourth, the compiler uses the CFG and the learned DDGs to define an instruction schedule that meets the first two Bernstein conditions.

**指令调度** 第四，编译器使用 CFG 和学习到的 DDG 来定义满足前两个 Bernstein 条件的指令调度。

Here, the compiler takes as input the maximum number of parallel instructions the target hardware can execute, and potential hardware constraints it needs to account for.

在这里，编译器将目标硬件可以执行的最大并行指令数以及需要考虑的潜在硬件约束作为输入。

For example, as we will see in Section 4, the hXDP executor has 4 parallel execution lanes, but helper function calls cannot be parallelized.

例如，正如我们将在第 4 节中看到的，hXDP 执行器有 4 个并行执行通道，但辅助函数调用不能并行化。

**Physical register assignment** Finally, in the last step the compiler assigns physical registers to the program's symbols.

**物理寄存器分配** 最后，在最后一步中，编译器将物理寄存器分配给程序的符号。

First, the compilers assigns registers that have a precise semantic, such as r0 for the exit code, r1-r5 for helper function argument passing, and r10 for the frame pointer.

首先，编译器分配具有精确语义的寄存器，例如 r0 用于退出代码，r1-r5 用于辅助函数参数传递，r10 用于帧指针。

After these fixed assignment, the compiler checks if for every row also the third Bernstein condition is met, otherwise it renames the registers of one of the conflicting instructions, and propagates the renaming on the following dependant instructions.

在这些固定分配之后，编译器检查每一行是否也满足第三个 Bernstein 条件，否则它重命名其中一个冲突指令的寄存器，并在后续依赖指令上传播重命名。

---

## 4. Hardware Module

## 4. 硬件模块

We design hXDP as an independent IP core, which can be added to a larger FPGA design as needed.

我们将 hXDP 设计为独立的 IP 核，可以根据需要添加到更大的 FPGA 设计中。

Our IP core comprises the elements to execute all the XDP functional blocks on the NIC, including helper functions and maps.

我们的 IP 核包含在网卡上执行所有 XDP 功能块的元素，包括辅助函数和映射。

This enables the execution of a program entirely on the FPGA NIC and therefore it avoids as much as possible PCIe transfers.

这使得程序可以完全在 FPGA 网卡上执行，因此尽可能避免 PCIe 传输。

### 4.1 Architecture and Components

### 4.1 架构和组件

The hXDP hardware design includes five components (see Figure 5): the Programmable Input Queue (PIQ); the Active Packet Selector (APS); the Sephirot processing core; the Helper Functions Module (HF); and the Memory Maps Module (MM).

hXDP 硬件设计包括五个组件（见图 5）：可编程输入队列（PIQ）；主动数据包选择器（APS）；Sephirot 处理核心；辅助函数模块（HF）；以及内存映射模块（MM）。

All the modules work in the same clock frequency domain.

所有模块在同一时钟频率域中工作。

Incoming data is received by the PIQ.

传入数据由 PIQ 接收。

The APS reads a new packet from the PIQ into its internal packet buffer.

APS 将新数据包从 PIQ 读入其内部数据包缓冲区。

In doing so, the APS provides a byte-aligned access to the packet data through a data bus, which Sephirot uses to selectively read/write the packet content.

在这样做时，APS 通过数据总线提供对数据包数据的字节对齐访问，Sephirot 使用该数据总线选择性地读/写数据包内容。

When the APS makes a packet available to the Sephirot core, the execution of a loaded eBPF program starts.

当 APS 向 Sephirot 核心提供数据包时，加载的 eBPF 程序的执行开始。

Instructions are entirely executed within Sephirot, using 4 parallel execution lanes, unless they call a helper function or read/write to maps.

指令完全在 Sephirot 内执行，使用 4 个并行执行通道，除非它们调用辅助函数或读/写映射。

In such cases, the corresponding modules are accessed using the helper bus and the data bus, respectively.

在这种情况下，使用辅助总线和数据总线分别访问相应的模块。

We detail each components next.

我们接下来详细介绍每个组件。

#### 4.1.3 Sephirot

#### 4.1.3 Sephirot

Sephirot is a VLIW processor with 4 parallel lanes that execute eBPF instructions.

Sephirot 是一个具有 4 个并行通道的 VLIW 处理器，用于执行 eBPF 指令。

Sephirot is designed as a pipeline of four stages: instruction fetch (IF); instruction decode (ID); instruction execute (IE); and commit.

Sephirot 设计为四级流水线：指令获取（IF）；指令解码（ID）；指令执行（IE）；以及提交。

A program is stored in a dedicated instruction memory, from which Sephirot fetches the instructions in order.

程序存储在专用指令存储器中，Sephirot 从中按顺序获取指令。

The processor has another dedicated memory area to implement the program's stack, which is 512B in size, and 11 64b registers stored in the register file.

处理器有另一个专用内存区域来实现程序的堆栈，大小为 512B，以及存储在寄存器文件中的 11 个 64 位寄存器。

These memory and register locations match one-to-one the eBPF virtual machine specification.

这些内存和寄存器位置与 eBPF 虚拟机规范一一对应。

Sephirot begins execution when the APS has a new packet ready for processing, and it gives the processor start signal.

当 APS 有新数据包准备处理时，Sephirot 开始执行，并向处理器发出启动信号。

On processor start (IF stage) a VLIW instruction is read and the 4 extended eBPF instructions that compose it are statically assigned to their respective execution lanes.

在处理器启动时（IF 阶段），读取一个 VLIW 指令，组成它的 4 条扩展 eBPF 指令被静态分配到它们各自的执行通道。

In this stage, the operands of the instructions are pre-fetched from the register file.

在此阶段，从寄存器文件中预取指令的操作数。

The remaining 3 pipeline stages are performed in parallel by the four execution lanes.

其余 3 个流水线阶段由四个执行通道并行执行。

During ID, memory locations are pre-fetched, if any of the eBPF instructions is a load, while at the IE stage the relevant sub-unit are activated, using the relevant pre-fetched values.

在 ID 期间，如果任何 eBPF 指令是加载，则预取内存位置，而在 IE 阶段，使用相关的预取值激活相关的子单元。

The sub-units are the Arithmetic and Logic Unit (ALU), the Memory Access Unit and the Control Unit.

子单元是算术逻辑单元（ALU）、内存访问单元和控制单元。

The ALU implements all the operations described by the eBPF ISA, with the notable difference that it is capable of performing operations on three operands.

ALU 实现 eBPF ISA 描述的所有操作，值得注意的区别是它能够对三个操作数执行操作。

The memory access unit abstracts the access to the different memory areas, i.e., the stack, the packet data stored in the APS, and the maps memory.

内存访问单元抽象了对不同内存区域的访问，即堆栈、存储在 APS 中的数据包数据以及映射内存。

The control unit provides the logic to modify the program counter, e.g., to perform a jump, and to invoke helper functions.

控制单元提供修改程序计数器的逻辑，例如执行跳转和调用辅助函数。

Finally, during the commit stage the results of the IE phase are stored back to the register file, or to one of the memory areas.

最后，在提交阶段，IE 阶段的结果存储回寄存器文件或内存区域之一。

Sephirot terminates execution when it finds an exit instruction, in which case it signals to the APS the packet forwarding decision.

当 Sephirot 找到退出指令时终止执行，在这种情况下，它向 APS 发出数据包转发决策的信号。

### 4.3 Implementation

### 4.3 实现

We prototyped hXDP using the NetFPGA [60], a board embedding 4 10Gb ports and a Xilinx Virtex7 FPGA.

我们使用 NetFPGA [60] 对 hXDP 进行了原型设计，该板嵌入了 4 个 10Gb 端口和一个 Xilinx Virtex7 FPGA。

The hXDP implementation uses a frame size of 32B and is clocked at 156.25MHz.

hXDP 实现使用 32B 的帧大小，时钟频率为 156.25MHz。

Both settings come from the standard configuration of the NetFPGA reference NIC design.

这两个设置都来自 NetFPGA 参考网卡设计的标准配置。

The hXDP FPGA IP core takes 9.91% of the FPGA logic resources, 2.09% of the register resources and 3.4% of the FPGA's available BRAM.

hXDP FPGA IP 核占用 9.91% 的 FPGA 逻辑资源、2.09% 的寄存器资源和 3.4% 的 FPGA 可用 BRAM。

The considered BRAM memory does not account for the variable amount of memory required to implement maps.

考虑的 BRAM 内存不包括实现映射所需的可变内存量。

A per-component breakdown of the required resources is presented in Table 1, where for reference we show also the resources needed to implement a map with 64 rows of 64B each.

表 1 中提供了所需资源的每个组件细分，作为参考，我们还显示了实现具有 64 行每行 64B 的映射所需的资源。

As expected, the APS and Sephirot are the components that need more logic resources, since they are the most complex ones.

正如预期的那样，APS 和 Sephirot 是需要更多逻辑资源的组件，因为它们是最复杂的。

Interestingly, even somewhat complex helper functions, e.g., a helper function to implement a hashmap lookup (HF Map Access), have just a minor contribution in terms of required logic, which confirms that including them in the hardware design comes at little cost while providing good performance benefits, as we will see in Section 5.

有趣的是，即使是某些复杂的辅助函数，例如实现哈希映射查找的辅助函数（HF Map Access），在所需逻辑方面也只有很小的贡献，这证实了将它们包含在硬件设计中成本很低，同时提供良好的性能优势，正如我们将在第 5 节中看到的。

When including the NetFPGA's reference NIC design, i.e., to build a fully functional FPGA-based NIC, the overall occupation of resources grows to 18.53%, 7.3% and 14.63% for logic, registers and BRAM, respectively.

当包括 NetFPGA 的参考网卡设计时，即构建一个完全功能的基于 FPGA 的网卡，资源的总体占用分别增长到逻辑的 18.53%、寄存器的 7.3% 和 BRAM 的 14.63%。

This is a relatively low occupation level, which enables the use of the largest share of the FPGA for other accelerators.

这是一个相对较低的占用水平，这使得 FPGA 的最大份额可用于其他加速器。

---

## 5. Evaluation

## 5. 评估

We use a selection of the Linux's XDP example applications and two real world applications to perform the hXDP evaluation.

我们使用 Linux XDP 示例应用程序的选择和两个真实世界的应用程序来执行 hXDP 评估。

The Linux examples are described in Table 2.

Linux 示例在表 2 中描述。

The real-world applications are the simple firewall we used as running example, and the Facebook's Katran server load balancer [17].

真实世界的应用程序是我们用作运行示例的简单防火墙，以及 Facebook 的 Katran 服务器负载均衡器 [17]。

Katran is a high performance software load balancer that translates virtual addresses to actual server addresses using a weighted scheduling policy, and providing per-flow consistency.

Katran 是一个高性能软件负载均衡器，使用加权调度策略将虚拟地址转换为实际服务器地址，并提供每流一致性。

Furthermore, Katran collects several flow metrics, and performs IPinIP packet encapsulation.

此外，Katran 收集多个流指标，并执行 IPinIP 数据包封装。

Using these applications, we perform an evaluation of the impact of the compiler optimizations on the programs' number of instructions, and the achieved level of parallelism.

使用这些应用程序，我们评估编译器优化对程序指令数量的影响以及实现的并行度水平。

Then, we evaluate the performance of our NetFPGA implementation.

然后，我们评估我们的 NetFPGA 实现的性能。

In addition, we run a large set of micro-benchmarks to highlight features and limitations of hXDP.

此外，我们运行了大量微基准测试来突出 hXDP 的特性和局限性。

We use the microbenchmarks also to compare the hXDP prototype performance with a Netronome NFP4000 SmartNIC.

我们还使用微基准测试将 hXDP 原型性能与 Netronome NFP4000 SmartNIC 进行比较。

Although the two devices target different deployment scenarios, this can provide further insights on the effect of the hXDP design choices.

尽管这两个设备针对不同的部署场景，但这可以进一步了解 hXDP 设计选择的影响。

### 5.2 Hardware Performance

### 5.2 硬件性能

We compare hXDP with XDP running on a server machine, and with the XDP offloading implementation provided by a SoC-based Netronome NFP 4000 SmartNIC.

我们将 hXDP 与在服务器机器上运行的 XDP 以及基于 SoC 的 Netronome NFP 4000 SmartNIC 提供的 XDP 卸载实现进行比较。

The NFP4000 has 60 programmable network processing cores (called microengines), clocked at 800MHz.

NFP4000 有 60 个可编程网络处理核心（称为微引擎），时钟频率为 800MHz。

The server machine is equipped with an Intel Xeon E5-1630 v3 @3.70GHz, an Intel XL710 40GbE NIC, and running Linux v.5.6.4 with the i40e Intel NIC drivers.

服务器机器配备了 Intel Xeon E5-1630 v3 @3.70GHz、Intel XL710 40GbE 网卡，运行 Linux v.5.6.4 和 i40e Intel 网卡驱动程序。

During the tests we use different CPU frequencies, i.e., 1.2GHz, 2.1GHz and 3.7GHz, to cover a larger spectrum of deployment scenarios.

在测试期间，我们使用不同的 CPU 频率，即 1.2GHz、2.1GHz 和 3.7GHz，以涵盖更大范围的部署场景。

In fact, many deployments favor CPUs with lower frequencies and a higher number of cores [24].

实际上，许多部署偏好具有较低频率和更多核心数量的 CPU [24]。

We use a DPDK packet generator to perform throughput and latency measurements.

我们使用 DPDK 数据包生成器来执行吞吐量和延迟测量。

#### 5.2.1 Applications Performance

#### 5.2.1 应用性能

**Simple firewall** In Section 2 we mentioned that an optimistic upper-bound for the hardware performance would have been 2.8Mpps.

**简单防火墙** 在第 2 节中，我们提到硬件性能的乐观上限应该是 2.8Mpps。

When using hXDP with all the compiler and hardware optimizations described in this paper, the same application achieves a throughput of 6.53Mpps, as shown in Figure 10.

当使用本文描述的所有编译器和硬件优化的 hXDP 时，相同的应用程序实现了 6.53Mpps 的吞吐量，如图 10 所示。

This is only 12% slower than the same application running on a powerful x86 CPU core clocked at 3.7GHz, and 55% faster than the same CPU core clocked at 2.1GHz.

这仅比在时钟频率为 3.7GHz 的强大 x86 CPU 核心上运行的相同应用程序慢 12%，比时钟频率为 2.1GHz 的相同 CPU 核心快 55%。

In terms of latency, hXDP provides about 10x lower packet processing latency, for all packet sizes (see Figure 11).

在延迟方面，hXDP 为所有数据包大小提供了约 10 倍更低的数据包处理延迟（见图 11）。

This is the case since hXDP avoids crossing the PCIe bus and has no software-related overheads.

这是因为 hXDP 避免穿越 PCIe 总线并且没有与软件相关的开销。

**Katran** When measuring Katran we find that hXDP is instead 38% slower than the x86 core at 3.7GHz, and only 8% faster than the same core clocked at 2.1GHz.

**Katran** 在测量 Katran 时，我们发现 hXDP 比 3.7GHz 的 x86 核心慢 38%，仅比时钟频率为 2.1GHz 的相同核心快 8%。

The reason for this relatively worse hXDP performance is the overall program length.

hXDP 性能相对较差的原因是整体程序长度。

Katran's program has many instructions, as such executors with a very high clock frequency are advantaged, since they can run more instructions per second.

Katran 的程序有许多指令，因此具有非常高时钟频率的执行器具有优势，因为它们每秒可以运行更多指令。

However, notice the clock frequencies of the CPUs deployed at Facebook's datacenters [24] have frequencies close to 2.1GHz, favoring many-core deployments in place of high-frequency ones.

然而，请注意部署在 Facebook 数据中心的 CPU 的时钟频率 [24] 接近 2.1GHz，偏向于多核部署而不是高频部署。

hXDP clocked at 156MHz is still capable of outperforming a CPU core clocked at that frequency.

时钟频率为 156MHz 的 hXDP 仍然能够超越该频率的 CPU 核心。

### 5.3 Discussion

### 5.3 讨论

**Suitable applications** hXDP can run XDP programs with no modifications, however, the results presented in this section show that hXDP is especially suitable for programs that can process packets entirely on the NIC, and which are no more than a few 10s of VLIW instructions long.

**适用应用程序** hXDP 可以运行未修改的 XDP 程序，然而，本节中呈现的结果表明，hXDP 特别适合可以完全在网卡上处理数据包的程序，并且不超过几十条 VLIW 指令。

This is a common observation made also for other offloading solutions [26].

这也是其他卸载解决方案的常见观察 [26]。

**FPGA Sharing** At the same time, hXDP succeeds in using little FPGA resources, leaving space for other accelerators.

**FPGA 共享** 同时，hXDP 成功地使用了很少的 FPGA 资源，为其他加速器留出了空间。

For instance, we could co-locate on the same FPGA several instances of the VLDA accelerator design for neural networks presented in [12].

例如，我们可以在同一个 FPGA 上共同定位 [12] 中提出的神经网络 VLDA 加速器设计的多个实例。

Here, one important note is about the use of memory resources (BRAM).

在这里，一个重要的注意事项是关于内存资源（BRAM）的使用。

Some XDP programs may need larger map memories.

一些 XDP 程序可能需要更大的映射内存。

It should be clear that the memory area dedicated to maps reduces the memory resources available to other accelerators on the FPGA.

应该清楚的是，专用于映射的内存区域减少了 FPGA 上其他加速器可用的内存资源。

As such, the memory requirements of XDP programs, which are anyway known at compile time, is another important factor to consider when taking program offloading decisions.

因此，XDP 程序的内存需求（无论如何在编译时已知）是在做出程序卸载决策时要考虑的另一个重要因素。

---

## 6. Future Work

## 6. 未来工作

While the hXDP performance results are already good to run real-world applications, e.g., Katran, we identified a number of optimization options, as well as avenues for future research.

虽然 hXDP 性能结果已经足以运行真实世界的应用程序，例如 Katran，但我们确定了许多优化选项以及未来研究的途径。

**Compiler** First, our compiler can be improved.

**编译器** 首先，我们的编译器可以改进。

For instance, we were able to hand-optimize the simple firewall instructions and run it at 7.1Mpps on hXDP.

例如，我们能够手动优化简单防火墙指令，并在 hXDP 上以 7.1Mpps 运行它。

This is almost a 10% improvement over the result presented in Section 5.

这比第 5 节中呈现的结果提高了近 10%。

The applied optimizations had to do with a better organization of the memory accesses, and we believe they could be automated by a smarter compiler.

应用的优化与内存访问的更好组织有关，我们相信它们可以通过更智能的编译器自动化。

**Hardware parser** Second, XDP programs often have large sections dedicated to packet parsing.

**硬件解析器** 其次，XDP 程序通常有大量专用于数据包解析的部分。

Identifying them and providing a dedicated programmable parser in hardware [23] may significantly reduce the number of instructions executed by hXDP.

识别它们并在硬件中提供专用的可编程解析器 [23] 可能会显著减少 hXDP 执行的指令数量。

However, it is still unclear what would be the best strategy to implement the parser on FPGA and integrate it with hXDP, and the related performance and hardware resources usage trade offs.

然而，目前还不清楚在 FPGA 上实现解析器并将其与 hXDP 集成的最佳策略是什么，以及相关的性能和硬件资源使用权衡。

**Multi-core and memory** Third, while in this paper we focused on a single processing core, hXDP can be extended to support two or more Sephirot cores.

**多核和内存** 第三，虽然在本文中我们专注于单个处理核心，但 hXDP 可以扩展以支持两个或更多 Sephirot 核心。

This would effectively trade off more FPGA resources for higher forwarding performance.

这将有效地用更多的 FPGA 资源换取更高的转发性能。

For instance, we were able to test an implementation with two cores, and two lanes each, with little effort.

例如，我们能够轻松测试具有两个核心、每个核心两个通道的实现。

This was the case since the two cores shared a common memory area and therefore there were no significant data consistency issues to handle.

这是因为两个核心共享一个公共内存区域，因此没有重大的数据一致性问题需要处理。

Extending to more cores (lanes) would instead require the design of a more complex memory access system.

扩展到更多核心（通道）将需要设计更复杂的内存访问系统。

Related to this, another interesting extension to our current design would be the support for larger DRAM or HBM memories, to store large memory maps.

与此相关，我们当前设计的另一个有趣的扩展是支持更大的 DRAM 或 HBM 内存，以存储大型内存映射。

---

## 8. Conclusion

## 8. 结论

This paper presented the design and implementation of hXDP, a system to run Linux's XDP programs on FPGA NICs.

本文介绍了 hXDP 的设计和实现，这是一个在 FPGA 网卡上运行 Linux XDP 程序的系统。

hXDP can run unmodified XDP programs on FPGA matching the performance of a high-end x86 CPU core clocked at more than 2GHz.

hXDP 可以在 FPGA 上运行未修改的 XDP 程序，性能匹配时钟频率超过 2GHz 的高端 x86 CPU 核心。

Designing and implementing hXDP required a significant research and engineering effort, which involved the design of a processor and its compiler, and while we believe that the performance results for a design running at 156MHz are already remarkable, we also identified several areas for future improvements.

设计和实现 hXDP 需要大量的研究和工程努力，其中涉及处理器及其编译器的设计，虽然我们认为运行在 156MHz 的设计的性能结果已经非常出色，但我们也确定了几个未来改进的领域。

In fact, we consider hXDP a starting point and a tool to design future interfaces between operating systems/applications and network interface cards/accelerators.

实际上，我们认为 hXDP 是一个起点和一个工具，用于设计操作系统/应用程序与网络接口卡/加速器之间的未来接口。

To foster work in this direction, we make our implementations available to the research community.

为了促进这个方向的工作，我们向研究社区提供我们的实现。

---

## Acknowledgments

## 致谢

We would like to thank the anonymous reviewers and our shepherd Costin Raiciu for their extensive and valuable feedback and comments, which have substantially improved the content and presentation of this paper.

我们要感谢匿名审稿人和我们的牧羊人 Costin Raiciu 提供的广泛而宝贵的反馈和评论，这些反馈和评论大大改善了本文的内容和呈现。

The research leading to these results has received funding from the ECSEL Joint Undertaking in collaboration with the European Union's H2020 Framework Programme (H2020/2014-2020) and National Authorities, under grant agreement n. 876967 (Project "BRAINE").

这些结果的研究获得了 ECSEL 联合事业与欧盟 H2020 框架计划（H2020/2014-2020）和国家机构合作的资助，资助协议编号为 876967（项目"BRAINE"）。

---

## 参考文献

本论文引用了大量相关工作，包括：

- FPGA 网卡编程相关：P4-NetFPGA、FlowBlaze、ClickNP 等
- eBPF 和 XDP 技术文献
- 网络加速和智能网卡相关研究
- FPGA 设计和优化技术

完整的参考文献列表请参见原始论文。

---

**论文源代码和工件**

源代码、示例和复现本文结果的说明可在以下地址获得：https://github.com/axbryd/hXDP-Artifacts

**开源实现**

hXDP 的开源实现可在 GitHub 上获得：https://github.com/axbryd

