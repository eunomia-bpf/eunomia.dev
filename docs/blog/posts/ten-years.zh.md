---
date: 2024-08-11
---

# eBPF的过去、现在与未来及其革新系统的道路

> 这篇博客文章主要参考了 Alexei Starovoitov 在 BPFConf 2024 上的演讲“《为未来十年现代化BPF》”。

想象一下，你的电脑核心操作中有一把瑞士军刀——可以让你窥探数据流动的内部、实时调整进程，并且随时监控一切。这正是[eBPF](https://en.wikipedia.org/wiki/EBPF)（扩展伯克利数据包过滤器）所提供的功能。在过去的十年里，eBPF已经从一个简单的数据包过滤工具转变为网络、可观测性和安全领域的强大利器。那么，eBPF的未来又会如何呢？让我们一起回顾它的历程，探讨未来十年的发展方向，并讨论前方的挑战与机遇。这将帮助我们理解eBPF如何塑造现代系统的未来。
<!-- more -->

## 十年回顾：eBPF迄今为止的历程

### eBPF如何开启可编程网络？

2014年，网络领域面临着严峻的限制。传统的网络栈非常僵化，难以实现针对特定需求的数据包处理逻辑。就在这时，[eBPF](https://isovalent.com/blog/post/ebpf-documentary-creation-story/)出现了——它允许开发者编写小型程序，每当网络数据包到达时，直接在内核中运行。这一创新带来了对数据处理更大的控制权，提升了性能和灵活性，同时避免了笨重的网络驱动程序的麻烦。

借助eBPF，开发者能够创建符合特定网络需求的解决方案，为更高效的数据处理和创新的网络应用铺平了道路。这标志着可编程网络的诞生，在这种网络中，定制化与性能可以完美结合。

### 是什么让XDP成为高速网络的游戏规则改变者？

随着eBPF在网络领域的普及，一个大难题浮现出来：Linux内核中的`sk_buff`结构引入了过多的开销，使得实现10 Gbps等高速变得困难。尽管用户空间的网络解决方案能够达到这些速度，但在内核中运行的eBPF却难以跟上。

突破出现在[eXpress Data Path (XDP)](https://www.iovisor.org/technology/xdp)。通过在网络接口卡（NIC）驱动程序内直接运行eBPF程序，XDP显著减少了处理开销。这使得数据包处理速度大大加快，让之前无法实现的高速网络应用成为可能。

像**Katran**和**Cilium**这样的工具利用XDP提供了闪电般快速的网络解决方案，展示了eBPF轻松处理高吞吐量数据的能力。XDP让eBPF成为需要顶级网络性能环境中的可行选项，巩固了其在现代网络中的地位。

### BTF和CO-RE如何让追踪更智能？

随着eBPF扩展到追踪和可观测性领域，开发者遇到了新的挑战：不同内核版本中的内核数据结构各异。这种不一致性意味着BPF程序必须包含特定的内核头文件，并为每个系统重新编译，增加了部署和维护的复杂性。

于是，[BPF Type Format (BTF)](https://docs.kernel.org/bpf/btf.html)和[Compile Once - Run Everywhere (CO-RE)](https://nakryiko.com/posts/bpf-portability-and-co-re/)应运而生。BTF为内核二进制文件（vmlinux）添加了类型信息，使BPF程序能够理解内核数据结构，而无需为每个版本重新编译。CO-RE由[libbpf](https://libbpf.readthedocs.io/en/latest/libbpf_overview.html)支持，使BPF程序可以编译一次，在任何系统上运行，加载时动态适应不同的内核版本。

这些进展使得追踪工具更加健壮和可移植，减轻了开发者的维护负担，并鼓励更广泛地采用基于eBPF的可观测性解决方案。开发者现在可以在各种环境中部署追踪工具，而无需担心内核版本不匹配，大大提高了生产力和可靠性。

### Skeleton和全局变量如何简化eBPF开发？

用C编写eBPF程序通常意味着要处理全局变量，这些变量从用户空间管理起来比较棘手。这些变量存在于`.data`和`.bss`部分，使得用户空间应用程序和BPF程序之间的交互变得笨拙且容易出错。

由BTF和libbpf驱动的Skeleton生成的引入改变了这一局面。Skeleton允许开发者生成类型安全的代码，桥接用户空间和BPF程序之间的差距。不再需要与不透明的全局变量苦苦挣扎！相反，开发者可以以结构化和安全的方式与BPF变量交互，显著简化了开发过程。

这不仅减少了错误，还加快了功能丰富的eBPF应用程序的创建。此外，像[GPTtrace](https://github.com/eunomia-bpf/GPTtrace)这样的工具利用大型语言模型（LLMs）进一步简化了eBPF程序的开发，降低了缺乏深厚操作系统专业知识的开发者的门槛。Skeleton和AI驱动工具的结合使eBPF开发比以往任何时候都更加便捷和高效。

### 从无循环到强大迭代器：eBPF的控制流如何演变？

早期的eBPF程序在控制流能力方面受到限制——不支持循环，以保持程序简单且可验证。虽然这种方法确保了安全性，但也限制了开发者的能力，限制了eBPF应用程序的复杂性。

多年来，eBPF逐步引入了更先进的循环机制：

- **2014年:** 不支持循环，保持简单和安全。
- **2019年:** 引入有限循环，允许固定次数的循环。
- **2021年:** 增加了[`bpf_loop()`](https://docs.ebpf.io/linux/helper-function/bpf_loop/)辅助函数，为循环构造提供了更多灵活性。
- **2023年:** 实现了开放编码的[迭代器](https://lwn.net/Articles/926041/)，提供了更强大和高效的循环机制。
- **2024年（计划）:** 引入[`cond_break`](https://lwn.net/Articles/964641/)，允许基于特定条件跳出循环。

这些增强功能使开发者能够编写更复杂和高效的eBPF程序。随着对循环和先进迭代器的支持，eBPF可以处理复杂的数据处理任务，并直接在内核中执行实时分析。控制流能力的这一演变为eBPF的潜力开启了新的可能性，使其成为开发者更加多用途的工具。

---

## 塑造未来：eBPF的下一步是什么？

展望未来，eBPF不断发展，推出前沿功能和增强功能，承诺彻底改变我们与系统内部交互的方式。让我们探讨一些即将到来的令人兴奋的发展，以及它们带来的机遇和挑战。

### kfuncs如何使内核接口更灵活？

传统上，BPF助手函数具有固定的用户空间API（UAPI）和硬编码的ID，限制了eBPF的灵活性和可扩展性。**kfunc**机制的引入改变了这种动态。[Kfuncs](https://docs.kernel.org/bpf/kfuncs.html)允许内核模块为BPF定义自己的助手函数，提供了更灵活和可扩展的接口。

这意味着开发者可以在无需等待内核更新的情况下扩展eBPF的功能。定制的助手函数可以根据特定需求进行调整，促进创新并实现以前无法实现的新用例。欲了解更多详情，请参见[关于kfuncs的教程](https://eunomia.dev/tutorials/43-kfuncs/)。

通过允许内核模块定义自己的助手函数，kfuncs使eBPF生态系统更加适应性强，对新兴需求做出响应，确保eBPF在快速变化的技术环境中保持相关性和强大功能。

### 什么是Struct-Ops，它们如何增强eBPF？

由于缺乏稳定的接口，为内核子系统添加新的eBPF附加类型一直是一个挑战。**struct-ops**机制通过允许一组BPF程序作为稳定内核API（如TCP拥塞控制）的回调来解决这个问题。

这为eBPF与各种内核子系统的深度集成打开了大门，例如：

- **调度器:** 创建自定义的[eBPF任务调度策略](https://www.kernel.org/doc/html/next/scheduler/sched-ext.html)以优化CPU使用。
- **HID（人机接口设备）:** 开发独特的[eBPF输入设备处理机制](https://docs.kernel.org/hid/hid-bpf.html)。
- **FUSE（用户空间文件系统）:** 实现灵活高效的[eBPF fuse解决方案](https://lpc.events/event/16/contributions/1339/attachments/945/1861/LPC2022%20Fuse-bpf.pdf)。
- **排队纪律:** 更有效地管理网络流量，减少延迟，提高吞吐量。

[Struct-ops](https://docs.ebpf.io/linux/program-type/BPF_PROG_TYPE_STRUCT_OPS/)使eBPF能够增强这些子系统的性能和灵活性，使其成为在各种内核级定制中多用途的工具。通过提供稳定的接口，struct-ops简化了集成过程，鼓励eBPF在系统管理和优化中更广泛的采用和创新应用。

### bpf_arena如何增强eBPF中的数据结构？

随着eBPF用例的扩展，对更复杂的数据结构（如树和图）的需求日益增长。**bpf_arena**的引入通过提供BPF与用户空间之间的共享内存空间来解决这一问题。[bpf_arena](https://lwn.net/Articles/961594/)允许开发者直接在eBPF程序中实现复杂的算法和数据结构。

借助bpf_arena，开发者可以处理更复杂的数据处理任务，优化内存使用，并改善访问模式。这一增强为eBPF支持需要强大数据管理能力的高级应用铺平了道路。详细功能在[eBPF文档](https://docs.ebpf.io/linux/kfuncs/bpf_arena_free_pages/)中有所阐述。

通过促进复杂数据结构的创建，bpf_arena显著扩大了eBPF的能力范围，使其能够在内核中实现更高级的分析、监控和优化任务。

### 为什么BPF库对丰富的生态系统很重要？

由于依赖管理问题，跨BPF程序共享代码历来是一项挑战。受Rust和Python等语言的启发，eBPF的未来在于强大的库支持。通过将库作为源代码分发，开发者可以简化依赖关系并鼓励代码重用。

这种方法培养了一个由社区驱动的生态系统，开发者可以在彼此的工作基础上进行构建，减少重复并加快开发。丰富的库生态系统将使创建功能丰富的eBPF应用程序变得更容易，推动更广泛的采用和创新。

强大的BPF库提供了开发者可以利用的标准化工具和函数，提升了生产力并确保了不同eBPF项目之间的一致性。这种集体努力不仅加快了开发速度，还提高了eBPF应用程序的整体质量和可靠性。

### 任意锁将如何改善eBPF中的并发性？

eBPF中现有的锁机制，如`bpf_spin_lock()`，功能有限且容易发生死锁。这限制了更复杂的并发BPF应用程序的发展。提议的解决方案是一个支持多重锁并防止死锁的新锁系统。

此升级将允许在BPF程序中使用更复杂的并发模式，使开发者能够构建更可靠和高效的应用程序。通过更好的并发支持，eBPF可以处理更苛刻的任务而不影响系统稳定性。了解更多信息，请参见[LWN.net](https://lwn.net/Articles/779120/)和[eBPF文档](https://docs.ebpf.io/linux/concepts/concurrency/)。

改进的并发机制将提升eBPF应用程序的性能和可扩展性，使其更适合需要多个操作同时运行且互不干扰的高性能环境。

### 拥抱图灵完备对eBPF意味着什么？

eBPF已经是[图灵完备](https://isovalent.com/blog/post/ebpf-yes-its-turing-complete/)，这意味着只要有足够的资源，它可以执行任何计算。然而，为了充分利用这一潜力，需要额外的功能，如跳转表和间接goto指令。这些增强将使eBPF程序内的控制流更加动态和灵活。

通过这些改进，eBPF可以在内核中支持更强大和灵活的编程模型。这将推动eBPF的能力边界，为开发者开辟新的可能性。完全拥抱图灵完备将使eBPF能够处理更复杂的算法和过程，使其成为系统编程和优化中更加不可或缺的工具。

---

## 让eBPF更强大：指令集和寄存器

### 发展BPF指令集（ISA）将如何改善eBPF？

eBPF中的某些操作仍然笨拙或低效。增强指令集可以在性能和易用性方面带来显著的改进。提议的增强包括：

- **间接调用:** 引入新的操作码以简化和加快函数调用。
- **位操作:** 添加用于常见位操作的指令，如查找和计数位，可以优化频繁任务。

这些补充将使eBPF程序更高效、更易于编写，扩展其可用性和性能。详细规格请参见[内核文档](https://docs.kernel.org/bpf/standardization/instruction-set.html)。

通过优化指令集，eBPF变得更加强大和多功能，使开发者能够编写更优化和功能丰富的程序，而不增加不必要的复杂性。

### 可以对eBPF寄存器进行哪些优化？

不同的架构提供不同数量的寄存器，eBPF有时在使用它们时效率不高。潜在的改进包括：

- **虚拟寄存器:** 抽象硬件限制以最大化效率。
- **寄存器溢出/填充:** 优化寄存器的使用和管理以防止瓶颈。
- **更多硬件寄存器:** 允许编译器在有额外寄存器时利用它们。

更好的寄存器管理意味着更快、更高效的eBPF程序，提升整体性能，使eBPF成为开发者更强大的工具。优化寄存器使用对于确保eBPF能够处理日益复杂的任务而不遇到资源限制至关重要。

### eBPF如何处理更多函数参数？

目前，由于寄存器限制，eBPF函数在传递参数方面仅限于五个。为克服这一限制，提出了两种解决方案：

- **额外寄存器:** 在可能的情况下利用更多寄存器来传递额外参数。
- **堆栈空间:** 通过堆栈传递额外参数，仔细管理性能和安全性。

这些解决方案将为函数调用提供更多灵活性，允许编写更复杂和功能更强大的eBPF程序。更多信息，请参见[指令集](https://www.kernel.org/doc/html/v5.17/bpf/instruction-set.html)和[Stack Overflow](https://stackoverflow.com/questions/70905815/how-to-read-all-parameters-from-a-function-ebpf)。

增强处理更多函数参数的能力将使开发者能够编写更全面和功能丰富的eBPF程序，扩大内核中可以高效管理的应用程序范围。

---

## 雄心勃勃的目标：将内核编译为BPF ISA

想象一下，如果Linux内核的关键部分可以编译为BPF指令集。这一愿景将彻底改变内核开发和分析，带来若干令人兴奋的好处：

- **增强分析:** 通过BPF的灵活可编程性，监控和验证内核行为变得更容易。
- **灵活性:** 无需完全重新编译即可快速适应和更新内核组件。

这个雄心勃勃的目标展望了一个更动态和适应性强的内核，由eBPF的强大和灵活性驱动。它可能导致更高效的内核开发周期和更具弹性的操作系统整体。相关讨论可以在[LWN.net](https://lwn.net/Articles/975830/)和[IETF关于BPF ISA的草案](https://www.ietf.org/archive/id/draft-ietf-bpf-isa-04.html)中找到。

将内核编译为BPF ISA将允许开发者以eBPF已经提供的相同的便捷性和灵活性编写和部署内核模块，简化开发并增强系统可靠性。

---

## 内存管理升级：动态堆栈及更多

### 我们如何突破512字节堆栈限制？

目前，eBPF程序面临严格的512字节堆栈限制，这限制了它们的复杂性和它们可以执行的计算类型。为克服这一限制，引入`alloca()`将允许在eBPF程序中进行动态内存分配。

通过`alloca()`，堆栈可以根据需要增长，使更复杂的函数和数据结构成为可能。这一增强将允许开发者创建更复杂和功能丰富的eBPF程序，扩展可能应用的范围。详细信息请参见[bpf_arena_alloc_pages](https://lwn.net/Articles/961594/)。

突破堆栈限制将使开发者能够在eBPF程序中实现更复杂的逻辑并处理更大的数据集，增强它们的能力和应用。

---

## 更安全的程序：可取消的eBPF脚本

### 我们如何确保eBPF程序保持安全和高效？

长时间运行的eBPF程序可能消耗大量CPU资源，可能导致系统不稳定。为了解决这个问题，提出了安全取消这些程序的新机制。

实施超时机制将自动终止运行过长时间的程序，而看门狗可以监控和管理程序执行。此外，提供安全的取消点确保可以在不引起系统问题的情况下停止程序。

这些保护措施将使eBPF程序更加可靠和稳定，即使在处理复杂任务时，也能确保系统保持响应和安全。通过引入这些安全机制，eBPF可以在系统稳定性至关重要的关键环境中更有信心地使用。

---

## 扩展可观测性到用户空间

### eBPF如何使用户空间监控更容易？

观察用户空间应用程序中发生的事情本质上比监控内核更复杂。多样的编程语言和运行时环境增加了挑战。然而，eBPF正在发展以弥合这一差距。

创新如Fast Uprobes提供了高效的用户空间探针，对性能影响最小。用户空间静态定义的追踪（USDT）允许应用程序定义自己的追踪点，提供更细粒度的监控。此外，针对C++、Python和Java等语言的特定语言栈遍历器可以解释它们特定的栈帧，提供有意义的追踪信息。

这些进展使得对用户空间应用程序的监控更加全面和详细，为开发者和系统管理员提供了更好的洞察和调试能力。通过将可观测性扩展到用户空间，eBPF确保系统性能的每个方面都可以被仔细跟踪和优化。

---

## 重新思考限制：100万指令限制

### 我们应该放宽eBPF中的100万指令限制吗？

目前，为确保eBPF程序可验证并正确终止，其指令数量被限制为100万条。虽然这一保护措施维护了系统稳定性，但也限制了eBPF程序所能实现的复杂性。

目前正在讨论对于能够展示前向验证进展的程序是否应该放宽这一限制。平衡对更复杂程序的需求与系统安全和性能至关重要。如果成功实施，这一变化可能允许更复杂的eBPF应用程序，扩展其用途而不牺牲安全性或稳定性。更多见解，请参见[LWN.net](https://lwn.net/Articles/975830/)。

放宽指令限制可能为eBPF开启新的可能性，使其能够处理更广泛和复杂的任务，同时仍保持必要的保护措施以保护系统完整性。

---

## 模块化eBPF：独立的内核模块

### 将eBPF作为独立内核模块有什么好处？

关于将eBPF作为独立的内核模块进行了大量讨论。想象一下能够在无需更新整个内核的情况下更新eBPF子系统。这一愿景通过将BPF作为独立的内核模块变为现实。

这种模块化方法提供了几个优势：

- **更快的更新:** 新功能和修复可以更快地推出，而无需等待完整的内核发布。
- **减少依赖:** 开发者和用户不必等待内核更新即可利用最新的eBPF功能。
- **增加灵活性:** 实验和创新可以在不受内核发布周期限制的情况下进行。

这一转变将导致一个更敏捷和响应迅速的eBPF生态系统，跟上快速的技术进步和开发者需求。通过将eBPF与内核解耦，更新和改进可以更高效地部署，提升整体用户体验和系统性能。

---

## 将eBPF扩展到其他平台

### 将eBPF扩展到其他平台

除了Linux，eBPF的能力正在扩展到其他平台，拓宽了其在不同环境中的影响力和实用性。

此外，[eBPF for Windows](https://github.com/microsoft/ebpf-for-windows)将eBPF的能力扩展到Linux之外，使开发者能够在Windows系统上利用eBPF的强大功能。这种跨平台支持为在异构环境中工作的开发者开辟了新途径，使他们能够无论操作系统如何都能应用eBPF的优势。

此外，[bpftime](https://github.com/eunomia-bpf/bpftime)提供了一个用户空间运行时，克服了内核空间的限制，释放了eBPF应用程序更多的潜力。通过启用如bcc-tools或bpftrace等eBPF应用程序的用户空间执行，bpftime提供了更大的灵活性和实验性，使eBPF适用于更广泛的用例和开发者。

将eBPF扩展到其他平台确保其强大的功能可供更广泛的用户使用，促进创新并提升不同操作系统的系统性能。

---

## eBPF的下一步是什么？

eBPF已经解决了追踪、可观测性和可编程网络中的主要挑战。但旅程并未止步于此。未来还有更多令人兴奋的可能性：

### bpf-lsm如何将eBPF扩展到安全领域？

**bpf-lsm** ([BPF Linux安全模块](https://docs.kernel.org/bpf/prog_lsm.html))允许eBPF实施定制的安全策略。这意味着开发者可以根据特定需求定制安全措施，利用eBPF的力量实时监控和控制系统行为。借助bpf-lsm，eBPF可以在增强系统安全性方面发挥关键作用，提供更细粒度和动态的保护机制。

通过eBPF直接将安全集成到内核，bpf-lsm提供了一种灵活而强大的方式来实施和管理安全策略，使系统在面对威胁和漏洞时更加有弹性。

### eBPF能否优化调度以获得更好的性能？

将eBPF应用于任务和数据包调度可以带来更好的性能和资源管理。对于任务调度，eBPF可以创建自定义调度策略，根据特定的工作负载优化CPU使用。对于数据包调度，eBPF可以更有效地管理网络流量，减少延迟，提高吞吐量。这些优化将导致更高效、更响应的系统，能够轻松处理多样化的工作负载。有关更多详情，请参见[sched-ext/scx](https://github.com/sched-ext/scx)仓库。

使用eBPF优化调度确保系统资源得到更有效的利用，提升整体性能和用户体验，特别是在具有多样化和苛刻工作负载的环境中。

---

## eBPF的指导原则

在eBPF演变的核心有三个基本原则：

1. **创新:** 持续推动内核和用户空间编程的可能性边界。
2. **使他人能够创新:** 提供工具和框架，使开发者能够构建新的令人兴奋的解决方案。
3. **挑战可能性:** 打破现有的限制，重新定义操作系统的功能。

这些原则确保eBPF保持在前沿工具的位置，通过促进创造力和克服挑战推动计算未来。通过坚持这些指导原则，eBPF不断演变和适应，保持其在系统开发和优化前沿的位置。

---

## 结论

回顾过去，显而易见，eBPF已经走过了漫长的路程——从可编程网络的谦逊起步，到成为追踪、可观测性等领域的强大工具。未来更加光明，前方有令人兴奋的功能，承诺使eBPF更加强大、灵活和用户友好。

这些发展，以及像[GPTtrace](https://github.com/eunomia-bpf/GPTtrace)、[eBPF for Windows](https://github.com/microsoft/ebpf-for-windows)和[bpftime](https://github.com/eunomia-bpf/bpftime)这样的用户空间eBPF运行时，克服了内核空间的限制，释放了eBPF应用程序更多的潜力。通过启用如bcc-tools或bpftrace等eBPF应用程序的用户空间执行，bpftime提供了更大的灵活性和实验性，使eBPF适用于更广泛的用例和开发者。此外，我们还通过[code-survey](https://github.com/eunomia-bpf/code-survey)等项目利用大型语言模型更好地理解内核中的eBPF代码，增强了我们分析和优化eBPF程序的能力。

随着我们进入下一个十年，eBPF准备好迎接新的挑战并开启新的可能性。无论你是希望优化应用程序的开发者，追求更好性能的系统管理员，还是渴望探索最新创新的技术爱好者，eBPF都有值得你关注的内容。

欲了解更多详情和有趣的主题，请访问[BPFConf 2024](http://oldvger.kernel.org/bpfconf2024.html)。

---

## 参考文献

1. [维基百科](https://en.wikipedia.org/wiki/EBPF) - eBPF历程的概述及其在内核中安全运行程序的能力。
2. [Isovalent博客](https://isovalent.com/blog/post/ebpf-documentary-creation-story/) - 记录了eBPF创建背后的故事及其对技术行业的影响。
3. [IO Visor](https://www.iovisor.org/technology/xdp) - XDP允许在内核中直接高效处理数据包，提供显著的性能提升。
4. [Tigera](https://www.tigera.io/learn/guides/ebpf/ebpf-xdp/) - 一个快速数据包处理框架，说明了将XDP与eBPF应用程序集成的优势。
5. [BPF类型格式](https://docs.kernel.org/bpf/btf.html) - BTF提供了增强BPF应用程序可验证性和可移植性的基本类型信息。
6. [libbpf文档](https://libbpf.readthedocs.io/en/latest/libbpf_overview.html) - Skeleton文件简化了用户空间与BPF程序之间的交互，优化了全局变量的管理。
7. [GitHub讨论](https://github.com/cilium/ebpf/discussions/943) - 讨论了如何在eBPF应用程序中访问和管理全局变量。
8. [Speaker Deck](https://speakerdeck.com/f1ko/ebpf-vienna-bpf-evolution-of-a-loop) - 分析了eBPF中控制流的功能和验证。
9. [内核文档](https://docs.kernel.org/bpf/kfuncs.html) - 提供了增强BPF应用程序灵活性和可扩展性的内核函数的见解。
10. [Eunomia博客](https://eunomia.dev/tutorials/43-kfuncs/) - 描述了自定义kfuncs如何实现内核函数和eBPF程序之间更强大的交互。
11. [eBPF文档](https://docs.ebpf.io/linux/program-type/BPF_PROG_TYPE_STRUCT_OPS/) - 解释了struct-ops如何提升性能，并允许BPF程序与内核子系统之间更复杂的接口。
12. [LWN.net](https://lwn.net/Articles/961594/) - 讨论了bpf_arena作为支持BPF程序与用户空间共享自定义数据结构的内存区域。
13. [eBPF文档](https://docs.ebpf.io/linux/kfuncs/bpf_arena_free_pages/) - 详细说明了bpf_arena在管理复杂数据结构方面的能力。
14. [Red Hat开发者](https://developers.redhat.com/articles/2023/10/19/ebpf-application-development-beyond-basics) - 讨论了libbpf在简化交互和增强程序开发中的作用。
15. [LWN.net](https://lwn.net/Articles/779120/) - 解释了任意锁在管理eBPF并发性、增强过程完整性方面的重要性。
16. [eBPF文档](https://docs.ebpf.io/linux/concepts/concurrency/) - 提供了eBPF程序中并发管理技术的全面概述。
17. [Isovalent](https://isovalent.com/blog/post/ebpf-yes-its-turing-complete/) - 确认eBPF是图灵完备的，能够解决任何可计算问题，潜在应用涉及多个领域。
18. [内核文档](https://docs.kernel.org/bpf/standardization/instruction-set.html) - 概述了BPF指令集的规格和历史背景，以及最近为了更好性能而进行的增强。
19. [Stack Overflow](https://stackoverflow.com/questions/70905815/how-to-read-all-parameters-from-a-function-ebpf) - 提供了有关如何在eBPF函数中访问和管理参数的见解。
20. [标准化BPF ISA - LWN.net](https://lwn.net/Articles/975830/) - 讨论了将内核模块编译为使用BPF ISA的更广泛影响及其带来的好处。
21. [IETF关于BPF ISA的草案](https://www.ietf.org/archive/id/draft-ietf-bpf-isa-04.html) - 审查了围绕BPF ISA的细节及其未来增强的路线图。
23. [bpf_arena_alloc_pages](https://lwn.net/Articles/961941/) - 详细介绍了在eBPF程序中引入`alloca()`进行动态内存分配。
24. [sched-ext/scx](https://github.com/sched-ext/scx) - 使用eBPF进行任务和数据包调度优化的代码库。

这套资源代表了截至2024年围绕eBPF技术的进展、挑战和影响的详细综合，展示了其不断发展的能力和在系统性能、安全性和可观测性等各个领域的应用潜力。
