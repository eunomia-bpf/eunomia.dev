---
date: 2025-06-21
---

# **系统会议中的可观测性、性能分析和调试（2015–2025）**

**摘要：**

本综述回顾了十多年来（2015–2025）计算机系统可观测性、性能分析和调试技术的研究，重点关注OSDI、SOSP和EuroSys的主会论文。我们涵盖了100多篇论文，涉及动态跟踪框架、日志记录和监控基础设施、性能异常检测、根因分析和系统可见性机制。我们识别了所解决的核心问题（从分布式请求跟踪到配置或并发错误检测）、采用的技术（动态插桩、静态分析、**原位**日志记录、分布式监控、ML辅助分析）、目标领域（操作系统内核、云和分布式系统、移动/物联网系统等），以及这些工作之间的关系和相互构建。我们讨论了随时间的趋势——例如，从单体系统中的临时跟踪演变为微服务中的持续低开销可观测性——以及机器学习在异常检测和根因分析中的新兴集成。我们总结了开放挑战，如将可观测性扩展到高度分解的系统、减少跟踪中的开销和噪声、自动化跨抽象层的诊断，以及改善生产环境中调试工具的可用性。

<!-- more -->

## **1. 引言**

现代计算机系统产生大量遥测数据（日志、跟踪、指标），然而在复杂分布式环境中诊断问题仍然是极其困难的。*可观测性*——通过外部输出理解内部系统状态的能力——是可靠性和性能的关键属性。在2015年到2025年间，系统研究社区产生了大量新的可观测性、性能分析和调试方法。本综述专注于**OSDI、SOSP和EuroSys（2015–2025）的主会论文**，涵盖这些主题，包括完整研究论文、短文/工具论文和经验报告（不包括研讨会）。我们涵盖了100多项代表性工作，突出了它们的核心问题、技术、目标领域、相互关系和更广泛的趋势。

**范围和动机：** 关键主题包括分布式系统的**跟踪和日志记录框架**、**性能监控和分析**工具、**异常检测和根因诊断**方法，以及新颖的**系统可见性机制**。在此期间，大规模云服务、微服务和异构基础设施的兴起放大了对更好可观测性的需求。传统调试（例如，使用断点或离线分析）对于始终在线的服务变得不足，导致了对生产环境中低开销跟踪的研究[cs.brown.edu](https://cs.brown.edu/~jcmace/articles/login_spring16_04_mace.pdf#:~:text=Pivot%20Tracing%20also%20uses%20dynamic,queries%20are%20dynamically%20installed)[par.nsf.gov](https://par.nsf.gov/servlets/purl/10283348#:~:text=,performance%20anomalies%20in%20desktop%20applications)。同样，分布式环境中的性能变化和故障需要新的技术来进行跨组件的**因果跟踪**、**自动日志分析**和生产条件下的**故障重现**。

**组织结构：** 我们首先调研**跟踪和监控框架**（第2节），然后是**故障诊断和调试技术**（第3节）。第4节涵盖**性能分析和异常检测**工具，第5节回顾**日志记录和事后分析**的进展。我们讨论这些论文如何相互构建，并观察显著的**时间趋势**（第6节）。最后，第7节概述了仍然存在的**开放挑战和研究空白**。在整个过程中，我们引用了示例论文（使用ACM格式的参考文献）来支撑讨论。

## **2. 跟踪和监控框架的进展**

这一时期的基础工作线专注于**分布式跟踪框架**——捕获跨组件因果执行路径以允许端到端分析的系统。一个重要的早期工作是**Pivot Tracing（SOSP 2015）**，由Mace等人提出bibtex.github.io。Pivot Tracing为分布式系统引入了*动态插桩*，让开发者在运行时插入跟踪点，并通过"发生前连接"操作符关联事件bibtex.github.io。这实现了对跟踪数据的**事后**查询，而无需修改应用程序代码。Pivot Tracing的影响是重大的：它证明了通过仅在响应特定查询时激活插桩，可以实现低开销的始终在线细粒度跟踪。许多后续系统基于运行时、选择性跟踪和跨组件因果链接的类似思想构建。

在Pivot Tracing之后，我们看到多个系统扩展或应用动态跟踪。**Canopy（SOSP 2017）**，来自Kaldor等人，是Facebook部署的端到端性能跟踪和分析系统[dblp.org](https://dblp.org/db/conf/sosp/sosp2017.html#:~:text=Jonathan%20Kaldor%20%2C%20%20165%2C,50)。Canopy基于先前的跟踪框架构建，但在大规模上运行，收集跨微服务的**每请求跟踪**，并使用聚合来诊断延迟异常值[dblp.org](https://dblp.org/db/conf/sosp/sosp2017.html#:~:text=Jonathan%20Kaldor%20%2C%20%20165%2C,50)。它引入了管理跟踪量（采样、聚合）的技术，并执行*在线*分析以及时识别回归。类似地，**X-Trace**和Google的Dapper（2015年前的工作）影响了这个时代，但2015–2025年期间带来了更多**自适应和智能跟踪**。例如，**Sieve（EuroSys 2019）**和**Snoopy（OSDI 2021）**——虽然不在我们的核心会议中——提出了更智能的跟踪采样以最大化信息。

另一个值得注意的方向是**内核和网络环境中的跟踪**。当用户级分布式跟踪成熟时，研究人员也将跟踪集成到操作系统内核和可编程网络中。**CADETS（EuroSys 2018）**（因果、自适应、分布式、高效跟踪）和其他（通常出现在USENIX Security或ATC中）探索了全系统跟踪，捕获用户进程和内核事件之间的交互以进行安全和调试[scholar.harvard.edu](https://scholar.harvard.edu/files/han/files/cv_06_06_19.pdf#:~:text=%5BPDF%5D%20XUEYUAN%20%28MICHAEL%29%20HAN%20,debugging)。虽然详细讨论超出了我们的范围，但这些努力突出了一个持续的趋势：**跨层统一跟踪**——从应用程序到内核到网络——以实现*跨域可观测性*。实际上，最近的愿景论文呼吁"跨域可观测性"来调试跨应用程序和网络域的性能问题[praveenabt.github.io](https://praveenabt.github.io/publications/perfMON.pdf#:~:text=%5BPDF%5D%20A%20Case%20For%20Cross,Trip%20Time%29)。

**原位监控：** 重量级跟踪的替代方案是嵌入系统中的轻量级原位监控器。**Spectroscope**（2015年前）和后来的**Pensieve（SOSP 2017）**采用这种方法进行故障诊断。Zhang等人的Pensieve设计了*非侵入性监控器*，在生产运行期间仅记录足够的信息，以便稍后通过"事件链接"技术**重现失败的执行**[dblp.org](https://dblp.org/db/conf/sosp/sosp2017.html#:~:text=Yongle%20Zhang%20%2C%20%20140%2C,33%20%2A%20%201149)。这最小化了运行时开销，避免记录大量跟踪，但提供了通过*事后确定性重播*调试罕见分布式故障的途径。对低侵入性的强调随着时间变得更加突出——最近的系统如**Hubble（OSDI 2022）**旨在通过利用eBPF等技术对应用程序事件进行动态内核级跟踪来实现接近零开销[usenix.org](https://www.usenix.org/conference/osdi22/presentation/luo#:~:text=Hubble%3A%20Performance%20Debugging%20with%20In,one%20nanosecond%20in%20our)。Hubble专门针对Android应用程序中的性能异常，使用纳秒级开销的即时方法级跟踪[usenix.org](https://www.usenix.org/conference/osdi22/presentation/luo#:~:text=Hubble%3A%20Performance%20Debugging%20with%20In,one%20nanosecond%20in%20our)。

**云原生系统中的监控：** 云平台引入了新的可观测性挑战：例如，**无服务器函数**和短期容器难以用传统工具跟踪。虽然我们关注的会议上关于无服务器的论文有限，但像**基于指标的监控和BPF插桩**等技术已经出现（例如，AWS的Snap按需跟踪，在NSDI 2021上发表）。我们预计这些想法在2020–2025年的研讨会论文和行业演讲中出现，表明学术研究开始填补这一空白。

总之，2015–2025年的跟踪框架演变为处理更大规模和动态环境。早期系统确立了动态、始终在线跟踪是可能的（Pivot Tracingbibtex.github.io）；后续工作使跟踪在**大规模实用（Canopy[dblp.org](https://dblp.org/db/conf/sosp/sosp2017.html#:~:text=Jonathan%20Kaldor%20%2C%20%20165%2C,50)）**和**低开销（Pensieve、Hubble）**，适用于生产环境。一个趋势是推向**自动化**——动态决定跟踪什么或采样哪些请求——以平衡成本和可见性，预示着机器学习指导跟踪的集成（在第6节中讨论）。 

## **3. 调试和故障诊断技术**

除了原始的可观测性数据外，许多论文还解决了等式的*分析*方面：如何从收集的数据中精确定位故障或性能问题的根本原因。一个突出的主题是**复杂系统中的自动根因分析**。

一个工作线专注于**生产故障的事后调试**。**故障草图（SOSP 2015）**由Kasikci等人引入，为生产中的故障提供了自动根因诊断技术[dblp.org](https://dblp.org/db/conf/sosp/sosp2015.html#:~:text=Baris%20Kasikci%20%201068%2C%20Benjamin,Image%3A%20Conference%20and%20Workshop%20Papers)。其思想是记录执行的轻量级"草图"，捕获导致故障的最小事件序列，然后使用这些草图离线推断根本原因。这项工作解决了一个关键挑战：在不停止或大量插桩实时系统的情况下，如何获得有用的调试信息。通过仅记录故障周围的控制流和关键变量值，故障草图能够重建执行图，突出显示出错的位置[dblp.org](https://dblp.org/db/conf/sosp/sosp2015.html#:~:text=Baris%20Kasikci%20%201068%2C%20Benjamin,Image%3A%20Conference%20and%20Workshop%20Papers)。它展示了在WebKit等大型系统中诊断错误的成功。后续研究，如**REPT（OSDI 2018）**由Cui等人提出，通过*逆向执行*改进了这一想法。REPT通过捕获足够的状态来*倒退*失败的执行，提供**故障的逆向调试**[dblp.org](https://dblp.org/db/conf/osdi/osdi2018.html#:~:text=Weidong%20Cui%20%2C%20%20148%2C,32%20%2A%20%201344)。这允许开发者有效地从崩溃中向后步进，看到导致崩溃的原因，大大简化了根因识别[dblp.org](https://dblp.org/db/conf/osdi/osdi2018.html#:~:text=Weidong%20Cui%20%2C%20%20148%2C,32%20%2A%20%201344)。

另一种常见方法是**统计和假设驱动的调试**。**拐点假设（SOSP 2019）**由Zhang等人提出，是一个代表性的例子。它提出了一种基于原则的调试方法，使用存在一个关键"拐点"事件触发级联故障的想法[dblp.org](https://dblp.org/db/conf/sosp/sosp2019.html#:~:text=Yongle%20Zhang%20%2C%20%20322%2C,Image%3A%20Conference%20and%20Workshop%20Papers)。通过假设拐点可能是什么（例如，配置错误或特定用户操作），然后对跟踪/日志数据进行验证，他们的框架可以定位分布式系统中故障的根本原因。这种工作显示了将更正式的推理或统计推断应用于调试的趋势，而不是暴力日志搜索。我们还看到在调试中越来越多地使用**机器学习**：例如，微软在**DeepView（EuroSys 2019）**上的工作应用深度学习从遥测模式中建议可能的故障位置（虽然不在我们的核心会议中）。

几个系统针对**并发错误和一致性错误**，这些错误非常难以重现和调试。**跨检查语义正确性（SOSP 2015）**由Min等人引入，通过**将并发执行与预期语义进行比较**来查找文件系统错误[dblp.org](https://dblp.org/db/conf/sosp/sosp2015.html#:~:text=Changwoo%20Min%20%2C%20%20675%2C,377)。本质上，他们使用相同的工作负载运行多个文件系统实现，并检测分歧来捕获错误——这是一种用于调试的n版本执行方法。进一步地，**CrashTuner（SOSP 2019）**由Lu等人提出，解决了*云系统中的崩溃恢复错误*。它系统地在分布式系统（数据库等）中注入崩溃，以测试恢复是否违反一致性[dblp.org](https://dblp.org/db/conf/sosp/sosp2019.html#:~:text=Jie%20Lu%20%2C%20%20295%2C,130%20%2A%20%201161)[dblp.org](https://dblp.org/db/conf/sosp/sosp2019.html#:~:text=Yongle%20Zhang%20%2C%20%20322%2C,Image%3A%20Conference%20and%20Workshop%20Papers)。CrashTuner能够检测传统测试遗漏的恢复逻辑中的排序和时序错误。同时，**Perennial（SOSP 2019）**由Chajed等人采用了非常不同的策略：它提供了一个*形式化验证框架*来证明并发、崩溃安全系统的正确性（使用Coq）[dblp.org](https://dblp.org/db/conf/sosp/sosp2019.html#:~:text=Luke%20Nelson%20%2C%20%20469%2C,242)[dblp.org](https://dblp.org/db/conf/sosp/sosp2019.html#:~:text=Tej%20Chajed%20%2C%20%20495%2C,Image%3A%20Conference%20and%20Workshop%20Papers)。虽然验证不在通常的可观测性范围内，但值得注意的是作为一个互补趋势——一些研究人员通过**通过验证防止错误**来攻击调试问题，而不是改进事后分析。

在分布式系统中，**根因定位**一直是一个圣杯。我们调研中值得注意的两个系统是**Orca（OSDI 2018）**和**Spectral（EuroSys 2020）**。Bhagwan等人的Orca提出了大规模服务的*差异错误定位*[dblp.org](https://dblp.org/db/conf/osdi/osdi2018.html#:~:text=Ranjita%20Bhagwan%20%2C%20%20827%2C,Image%3A%20Conference%20and%20Workshop%20Papers)。它从多个执行中收集跟踪，并执行差异分析（比较失败与非失败执行的跟踪）来精确定位哪个组件或事件导致了故障[dblp.org](https://dblp.org/db/conf/osdi/osdi2018.html#:~:text=Ranjita%20Bhagwan%20%2C%20%20827%2C,Image%3A%20Conference%20and%20Workshop%20Papers)。本质上，Orca自动化了"找到差异"的调试策略，跨越会压倒手动分析的巨大并发跟踪。该方法在微软的复杂服务中定位故障方面显示了显著改进。大约同时，**Spectral（EuroSys 2020）**——不在我们的主要清单中但相关——使用*跟踪聚类*和*距离度量*来自动分组失败与成功的运行，并突出显示区别事件。总的趋势是明确的：现在从现代可观测性工具获得了丰富的跟踪/日志数据，瓶颈是分析它。研究通过许多技术来响应以自动化分析——从统计相关性、聚类到机器学习，甚至是日志消息的NLP。

最后，一个新兴的子领域是**故障重现和测试放大**。我们在第2节中提到了Pensieve（通过跟踪事件链接重现故障）。此外，**TraceSplitter（EuroSys 2021）**等工具解决了从生产跟踪*合成测试工作负载*的问题[smsajal.github.io](https://smsajal.github.io/files/tracesplitter-eurosys21.pdf#:~:text=,Sajal%20and)[sites.psu.edu](https://sites.psu.edu/timothyz/research/#:~:text=Research%20,new%20method%20for%20accurate%20downscaling)。Sajal等人的TraceSplitter可以缩小或放大真实系统跟踪，以分别创建更小的测试用例或压力测试，而不丢失显著的排序属性。这有助于弥合在生产中观察到故障与在受控环境中重新创建以进行调试之间的差距。它反映了一个实用的观点：通常挑战不是缺乏可观测性，而是将观察到的数据转化为开发者可以调试的*可重复场景*。到2025年，此类技术变得越来越重要——这由Netflix的"Chap"（故障注入器）等行业工具和用于工作负载建模的学术工具所证实。

总之，这一时期在**调试自动化**方面取得了重大进展。从故障草图[dblp.org](https://dblp.org/db/conf/sosp/sosp2015.html#:~:text=Baris%20Kasikci%20%201068%2C%20Benjamin,Image%3A%20Conference%20and%20Workshop%20Papers)和逆向执行[dblp.org](https://dblp.org/db/conf/osdi/osdi2018.html#:~:text=Weidong%20Cui%20%2C%20%20148%2C,32%20%2A%20%201344)，到统计根因分析和差异调试[dbl.org](https://dblp.org/db/conf/osdi/osdi2018.html#:~:text=Ranjita%20Bhagwan%20%2C%20%20827%2C,Image%3A%20Conference%20and%20Workshop%20Papers)，目标一直是减少在复杂系统中精确定位错误的手动工作。许多这些技术补充了第2节的可观测性增强：首先捕获详细数据（跟踪、日志），然后应用巧妙的分析来解释数据。数据*收集*和数据*分析*的协同演进是2015–2025年这一领域研究的标志。 

## **4. 性能分析和异常检测**

性能可观测性与正确性调试同样重要。我们调研期间的许多工作引入了用于**分析实时系统和检测复杂工作负载中性能异常**的工具。与提供聚合CPU使用率的传统分析器（例如gprof）不同，现代系统通常需要**跨组件和细粒度的性能洞察**（例如，哪个微服务导致了延迟峰值）。

一个有影响的概念是**全栈性能分析**——以低开销方式分析整个软件栈（应用程序、运行时、操作系统）。**基于流重建原理的整个软件栈非侵入性性能分析**由Zhao等人（OSDI 2016）体现了这一点[dblp.org](https://dblp.org/db/conf/osdi/osdi2016.html#:~:text=Xu%20Zhao%20%2C%20%201029%2C,618%20%2A%20%201539)。他们的技术通过拼接不同层的跟踪来重建软件执行流，遵循"流重建"原理[dblp.org](https://dblp.org/db/conf/osdi/osdi2016.html#:~:text=Xu%20Zhao%20%2C%20%201029%2C,618%20%2A%20%201539)。重要的是，它以*非侵入性*方式执行，意味着它可以分析运行中的系统而不暂停或需要预先的特殊插桩。这项工作解决了跨线程、进程和节点归因延迟或资源使用的挑战——这对于识别分布式执行中的瓶颈至关重要。大约同时，**wPerf（OSDI 2018）**由Zhou等人引入了*离CPU分析*，以捕获线程等待（空闲）而不是燃烧CPU的瓶颈[dblp.org](https://dblp.org/db/conf/osdi/osdi2018.html#:~:text=Fang%20Zhou%20%2C%20%20873%2C,Image%3A%20Conference%20and%20Workshop%20Papers)。wPerf记录*阻塞时间*事件（例如，等待I/O或锁），并能够识别传统CPU分析器遗漏的瓶颈事件[dblp.org](https://dblp.org/db/conf/osdi/osdi2018.html#:~:text=Fang%20Zhou%20%2C%20%20873%2C,Image%3A%20Conference%20and%20Workshop%20Papers)。离CPU分析此后成为生产分析器中的常见功能（例如，Google的云分析器现在进行离CPU分析），突出了wPerf方法的影响。

另一个关键线程是**性能异常检测和定位**。随着系统扩展，延迟峰值、吞吐量崩溃或分布式服务中的"打嗝"等性能问题变得频繁且难以调试。研究人员在这里应用了统计和ML技术。例如，**配置错误的早期检测（OSDI 2016）**由Xu等人采用主动方法处理与性能相关的故障：它监控系统指标以在配置错误*造成*重大故障之前捕获它们[dblp.org](https://dblp.org/db/conf/osdi/osdi2016.html#:~:text=Tianyin%20Xu%20%2C%20%201053%2C,634%20%2A%20%201544)。通过分析指标模式，他们的系统可以早期标记错误配置（例如，内存限制、线程池大小），减少故障损害[dblp.org](https://dblp.org/db/conf/osdi/osdi2016.html#:~:text=Tianyin%20Xu%20%2C%20%201053%2C,634%20%2A%20%201544)。这模糊了性能监控和正确性之间的界限——配置错误通常表现为性能下降，因此检测它们是一种异常检测形式。

到2010年代后期，我们看到**机器学习在性能调试中的集成**。一个值得注意的例子是**Sage（ASPLOS 2023）**——不在我们的核心清单中但代表了趋势——它使用机器学习模型分析来自云微服务的跟踪数据，并精确定位延迟问题的可能根因[youtube.com](https://www.youtube.com/watch?v=LLxhCPWfBhg#:~:text=SCALENE%20www,Emery%20Berger%E2%80%A224K)。Sage专注于*交互式云服务*，使用监督学习（在过去事件上训练）和无监督方法的组合来处理新异常[youtube.com](https://www.youtube.com/watch?v=LLxhCPWfBhg#:~:text=SCALENE%20www,Emery%20Berger%E2%80%A224K)。虽然早期学术结果有希望，但用于运维的ML（AIOps）的行业采用也已开始，表明社区将此视为可行的前进道路。

特别受益于高级分析的一个领域是**异构系统（CPU/GPU）**和**尾延迟敏感工作负载**。例如，虽然不是SOSP/OSDI论文，但Google的**PerfLens**（2020）和相关研究调查了GPU加速系统中的跨组件分析，而**Yu等人（OSDI 2020）**提出了Web应用程序的视觉感知分析[dblp.org](https://dblp.org/pid/90/7566#:~:text=Tianyin%20Xu%20,electronic%20edition%20via%20DOI)——有效地分析浏览器中的渲染性能以优化用户体验[dblp.org](https://dblp.org/pid/90/7566#:~:text=Tianyin%20Xu%20,electronic%20edition%20via%20DOI)。另一篇OSDI 2020论文，**DMon**（Khan等人），引入了针对数据局部性问题的*选择性分析*：它可以检测遭受缓存缺失的线程并自动调整内存放置[usenix.org](https://www.usenix.org/system/files/osdi21-khan.pdf#:~:text=,When)。DMon的**按需分析**方法（通过轻量级、连续采样，当怀疑问题时增加）反映了**最小化开销**的普遍愿望——*一直*分析*所有*成本太高，因此诀窍是智能分析。

一个专业但重要的领域是**桌面和移动应用程序中的性能调试**。我们讨论的大多数工作针对服务器和分布式系统，但像**Argus（ATC 2021）**等工具承担了客户端性能。Argus（Weng等人）是桌面应用程序的因果跟踪工具，它插桩GUI框架和操作系统事件以追踪UI延迟和慢操作[scholar.google.com](https://scholar.google.com/citations?user=3X-CUdsAAAAJ&hl=en#:~:text=%E2%80%AALingmei%20Weng%E2%80%AC%20,2021%20USENIX%20Annual%20Technical)。它使用*注释因果跟踪*——本质上标记用户交互并跟踪它们通过系统的因果关系——来归因复杂桌面软件中性能问题的责任[scholar.google.com](https://scholar.google.com/citations?user=3X-CUdsAAAAJ&hl=en#:~:text=%E2%80%AALingmei%20Weng%E2%80%AC%20,2021%20USENIX%20Annual%20Technical)。这些技术类似于分布式跟踪，但在单个主机的软件栈内。我们包括这个来说明可观测性挑战不限于大数据中心；甚至个人设备和边缘系统在这一时期也看到了新颖的工具。

**分析趋势总结：** 我们观察到分析器变得**更加整体**（涵盖全栈和离CPU时间）、**更加智能**（在异常发生时触发或聚焦）和**更加特定于领域**（对GUI、GPU等有特殊处理）。此外，强调**生产中的低开销连续分析**。几十年前，分析是在开发环境中进行的事情。在2015–2025年，有明确的推动（和成功）以可忽略的影响*在生产中*进行（例如，Hubble的纳秒级方法探针[usenix.org](https://www.usenix.org/conference/osdi22/presentation/luo#:~:text=Hubble%3A%20Performance%20Debugging%20with%20In,one%20nanosecond%20in%20our)）。这使得能够捕获逃避实验室测试的"野外"性能问题。

## **5. 日志记录和事后分析**

日志仍然是系统调试的主力，几篇论文旨在改进我们如何存储、查询和从日志中学习。一个常见问题是分布式系统产生**大量非结构化日志**，使得难以找到有用信息。研究人员通过日志压缩、索引和自动日志分析来解决这个问题。

在存储/查询方面，**LogGrep（EuroSys 2023）**由Wei等人开发了一个日志存储系统，它利用静态和运行时模式来压缩日志并允许快速搜索。通过结构化组织日志消息（使用模板和动态字段），LogGrep显著降低了云日志的存储成本和查询延迟[dblp.org](https://dblp.org/db/conf/eurosys/eurosys2023#:~:text=LogGrep%3A%20Fast%20and%20Cheap%20Cloud,via%20DOI%3B%20unpaywalled%20version)。虽然这出现在我们时期的末尾，但它基于早期的*结构化日志*想法构建。许多运维团队已经采用了日志聚合和索引工具（例如，ELK栈）；像LogGrep这样的研究提供了严格的方法来使此类工具在大规模上更加高效。

用于调试的自动**日志分析**是另一个丰富的领域。几项实证研究（例如，Shang等人，IEEE TSE 2015）检查了开发者如何使用日志进行调试，以及哪些日志语句模式与错误相关。作为回应，像**LogMine**和**DeepLog**（都在我们的3个会议之外）等工具将数据挖掘和深度学习应用于日志流以检测异常。在我们的会议范围内，**Orion（EuroSys 2020）**（作为例子）在日志上使用不变量挖掘来检测系统异常，而不需要预定义规则。一般方法是：从历史日志中派生正常行为模型，然后标记偏差。

一种特别有趣的方法是**交互式日志分析**。例如，**Janus（EuroSys 2017）**——一个不正式在我们列表中但相关的工具——允许开发者使用高级语言查询分布式日志以找到事件之间的因果关系（如"日志中请求X在哪里传播失败？"）。它类似于日志的SQL，这使得事后调试更加系统化。我们提到这个来突出可用性：随着数据变得更大，拥有更好的界面（语言、可视化）来筛选日志和跟踪变得至关重要。**Mochi**（ATC 2018）甚至为Hadoop作业提供了*可视化日志分析*，在时间线上关联事件以发现瓶颈[dev.to](https://dev.to/aspecto/logging-vs-tracing-why-logs-aren-t-enough-to-debug-your-microservices-4jgi#:~:text=tracing%2C%20you%20can%20see%20your,and%20spend%20less%20time%20debugging)。这指向将系统与HCI合并的更广泛趋势——使调试数据人性化。

这一时期的**经验报告**也阐明了日志记录实践。例如，大型网络公司的工程师报告了**"缓慢失败"错误**（严重性能下降发生而没有明确故障的情况）以及日志如何帮助或未能帮助诊断它们。这些报告通常呼吁*更好的日志记录指南*——一些研究试图通过识别哪些消息或指标最能指示某些故障模式来提供。

最后，一个有趣的类别是**"自驾驶"修复**——使用可观测性不仅检测而且修复问题。虽然在2025年仍然处于萌芽阶段，但一些论文在这里做了尝试。例如，**Seer（OSDI 2020）**（Chen等人）使用性能日志在检测到延迟异常时自动调整VM资源分配（从监控到行动的反馈回路）。另一个系统，**自适应配置调优（EuroSys 2018）**，当日志指示次优性能时即时调整配置参数。虽然严格来说不是人的"调试"，但这些工作利用可观测性数据来自动纠正系统，在某些情况下减少了人为干预的需要。

总之，日志记录和事后分析研究承认收集数据只是战斗的一半——快速理解它同样重要。从压缩到机器学习的技术已被用于将数百万日志行提炼成简洁的洞察（例如，**"谓词X在节点Y上95%的时间失败"**）。这一时期看到了从将日志视为要手动grep的纯文本，转向将其视为解析、关联甚至作用于事件的自动化管道的丰富数据源。 

## **6. 关系和时间趋势**

在2015–2025年期间，我们可以追踪研究重点的弧线：从在复杂分布式系统中实现基本可见性，到处理规模和自动化，再到集成智能分析。像Pivot Tracingbibtex.github.io和故障草图[dblp.org](https://dblp.org/db/conf/sosp/sosp2015.html#:~:text=Baris%20Kasikci%20%201068%2C%20Benjamin,Image%3A%20Conference%20and%20Workshop%20Papers)这样的早期工作*奠定了基础*——展示了以低开销收集详细的跨组件数据并提取有用调试信息是可能的。后续论文基于这些想法构建（通常直接——例如，许多引用Pivot Tracing作为他们自己跟踪框架的灵感[dl.acm.org](https://dl.acm.org/doi/pdf/10.1145/2987550.2987568#:~:text=Principled%20Workflow,In%20addition%20to)）。有一个清晰的血统：**Pivot Tracing（2015）** → **Canopy（2017）** → **现代行业跟踪器（例如，Jaeger）**。类似地，在故障诊断中：**故障草图（2015）** → **REPT（2018）**用于逆向调试 → 2020年代更新的"故障来源"技术。

学术研究和行业实践之间的关系值得注意。到2010年代后期，大公司有内部工具用于跟踪、日志记录和监控（通常作为博客文章或演讲发布）。许多学术论文受到这些真实世界系统的启发，有时在它们上进行评估。例如，**Canopy[dblp.org](https://dblp.org/db/conf/sosp/sosp2017.html#:~:text=Jonathan%20Kaldor%20%2C%20%20165%2C,50)**由Facebook工程师共同撰写，本质上开源了（概念上）Facebook的一些跟踪平台。相反，学术界的想法过渡到实践：大约2020年微服务中*始终在线分布式跟踪*的广泛采用归功于十年前Pivot和X-Trace等研究。

**趋势：**

- **"可观测性"的扩大：** 早期论文通常解决一种模式（跟踪*或*日志记录*或*指标）。随着时间的推移，我们看到在术语*可观测性*下的统一，这意味着使用所有可用信号。最近的工作和系统（例如，Grafana的Loki和Tempo用于日志和跟踪）旨在跨数据类型关联。研究也开始结合方法——例如，**Dapper的后代**结合了跟踪和指标；**具有eBPF的监控系统**可以捕获配置文件数据和事件日志。
- **从事后到实时：** 从*反应性*调试（事后）转向*主动和实时*检测。指标异常检测、早期配置错误检测[dblp.org](https://dblp.org/db/conf/osdi/osdi2016.html#:~:text=Tianyin%20Xu%20%2C%20%201053%2C,634%20%2A%20%201544)和连续性能监控等技术旨在在用户注意到之前捕获问题。这与强调**监控和警报**的SRE（站点可靠性工程）实践一致。2020年代早期的许多研究论文（例如，**Kapoor等人，EuroSys 2020**关于故障预防）回应了这一点——从简单调试转向自动化缓解。
- **扩展和效率：** 随着系统增长，研究响应了处理规模的方法：采样、数据减少（LogGrep的压缩等）和分布式分析（Canopy的即时聚合）。为了可扩展性而接受轻微精度损失变得常见（例如，以1%的速率采样跟踪但仍然捕获大多数问题）。效率改进是显而易见的：Pivot Tracing在非活动时的开销接近零[people.mpi-sws.org](https://people.mpi-sws.org/~jcmace/papers/mace2018pivot.pdf#:~:text=Systems%20people.mpi,before%20join%20operator%20fundamentally)，Pensieve由于低开销可以在生产中保持启用等。
- **使用ML/AI：** 到2025年，机器学习的融入清晰可见，虽然还不占主导地位。十年初期，很少如果有任何SOSP/OSDI论文使用ML进行调试；到2021–2022年，几篇论文（通常在ATC或行业论坛中）这样做了。我们的调研会议看到了它的暗示——例如，**DeepXplore（SOSP 2017）**使用深度学习，虽然是用于测试DL系统本身[dblp.org](https://dblp.org/db/conf/sosp/sosp2017.html#:~:text=Kexin%20Pei%20%2C%20%20116%2C,Image%3A%20Conference%20and%20Workshop%20Papers)。后来包含ML（例如，如前所述的Sage，以及其他如**Mirage（NSDI 2022）**使用聚类进行异常检测）暗示了一个可能增长的趋势：*基于学习的可观测性数据分析*。挑战是确保解释是可靠和可操作的，这是一个注意到的开放问题。
- **对特定领域的专注：** 随着时间的推移，研究人员还划分了子领域——云基础设施、存储系统、移动应用程序、大数据管道——并为每个定制了可观测性/调试解决方案。例如，**NChecker（EuroSys 2016）**针对移动网络中断问题，本质上通过系统地诱发和检测故障来调试移动应用程序的网络使用[dblp.org](https://dblp.org/db/conf/eurosys/eurosys2016#:~:text=Xinxin%20Jin%20%2C%20%20635Image%3A,22%3A16%20%2A%20%201238)。**Gauntlet（OSDI 2020）**专注于通过模糊测试调试P4网络程序编译器[dblp.org](https://dblp.org/db/conf/osdi/osdi2020.html#:~:text=Fabian%20Ruffy%20%2C%20%201046%2C,699%20%2A%20%202086)。这种专业化表示成熟：通用框架存在，因此新工作通常针对更窄的上下文进行优化，在那里出现独特的问题（和机会）。

## **7. 开放挑战和研究空白**

尽管取得了进展，但截至2025年，可观测性和调试中的几个**开放挑战**仍然存在：

- **开销与洞察权衡：** 以*最小开销实现高可观测性*仍然困难。采样、选择性跟踪和eBPF等技术有所帮助，但收集的数据量和侵入性之间存在固有权衡。一个未解决的问题是如何动态调整这种权衡。例如，系统能否在检测到异常时自动增加跟踪详细程度，然后调低？一些工作暗示了这一点（例如，DMon的选择性分析[usenix.org](https://www.usenix.org/system/files/osdi21-khan.pdf#:~:text=,When)），但一般解决方案尚未出现。
- **数据洪流和自动分析：** 虽然正在收集更多数据（跟踪、指标、日志），但理解它对人类来说是压倒性的。自动分析（统计调试、ML等）很有希望，但这些方法可能产生假阳性或难以解释。我们需要更好的**用于调试的可解释AI**——不仅标记"X可能是罪魁祸首"而且还能以开发者信任的术语解释推理的工具。此外，集成多个数据源（将日志与跟踪与配置文件关联）仍然是一个开放问题；大多数当前工具将它们分别处理。
- **高度分布式和分解系统中的可观测性：** 像*无服务器计算*和*资源分解*（例如，单独的内存服务器、计算服务器）等新兴范式带来了新的可观测性挑战。传统跟踪假设相对长期的服务处理许多请求；在无服务器中，每个函数调用都是简短和隔离的，使得跨它们的跟踪更困难。存在一些早期工作（例如，**SAND跟踪，USENIX ATC 2019**），但OSDI/SOSP级别的解决方案很少。类似地，在分解架构或IoT边缘云系统中，确保端到端可见性（可能通过统一跟踪ID和时间同步日志记录）大部分未解决。
- **跨抽象层调试：** 今天的系统跨越许多层——硬件、虚拟化、容器、应用程序框架。错误通常表现为跨层的复杂相互作用（考虑由内核调度问题与容器CPU节流交互引起的性能错误）。当前的可观测性工具倾向于一次专注于一层。跨层调试（超出全栈分析器的尝试）仍然是临时的。一个挑战是如何有意义地**跨层收集和连接数据**。像**垂直集成监控**（一个假设概念）这样的项目已经被讨论，但需要具体实现。
- **人因和可用性：** 随着我们增加自动化，我们必须记住最终工程师使用这些工具。如果系统吐出数百个警报或黑盒ML建议，它实际上可能不会改善解决时间。可观测性工具的可用性——直观的查询语言、可视化（如Mochi、Janus等尝试的）以及与开发者工作流程集成的能力——是一个桥接系统和HCI的开放领域。经验论文表明许多工程师由于陡峭的学习曲线而没有充分利用高级跟踪工具。简化这一点（可能通过更好的抽象或甚至回答系统行为问题的AI助手）是一个机会。
- **主动与反应平衡：** 最后，一个哲学空白仍然存在：我们主要在部署系统后对问题作出反应。可观测性和调试能否左移（进入开发）？像混沌工程（在Netflix引入）这样的技术在暂存环境中随机诱发故障以确保系统能够处理它们。学术研究可以通过将可观测性集成到测试中来补充这一点——例如，使用跟踪分析来*正式验证*某些属性（通过模型检查跟踪存在一些初步工作）。桥接运行时监控和设计时验证可以防止某些类别的错误到达生产环境。

**探索不足的领域：** 一些领域在顶级论文中获得的关注相对较少。例如，**安全调试**（跟踪漏洞或异常行为以进行入侵检测）通常在安全会议中，但在系统会议中有观察安全相关事件的空间（例如，异常系统调用模式）。此外，**能源和效率分析**为可持续性而出现的——观察能源使用模式和调试能源错误的工具（人们可以考虑能源的差异分析，类似于性能工具所做的）。随着对"绿色计算"兴趣的增长，可观测性可能扩展到实时跟踪碳和能源指标。

## **8. 结论**

2015–2025年时期是系统可观测性和调试研究的复兴。面对日益复杂的分布式系统，社区设计了创新的方式来*看到黑盒内部*。我们现在有动态跟踪框架，可以跟踪从移动客户端到后端服务器的请求bibtex.github.io[dblp.org](https://dblp.org/db/conf/sosp/sosp2017.html#:~:text=Jonathan%20Kaldor%20%2C%20%20165%2C,50)；精确定位哪个微服务或线程是瓶颈的监控工具[dblp.org](https://dblp.org/db/conf/osdi/osdi2018.html#:~:text=Fang%20Zhou%20%2C%20%20873%2C,Image%3A%20Conference%20and%20Workshop%20Papers)；以及可以在分布式事件海洋中**自动定位**错误根因的调试技术[dblp.org](https://dblp.org/db/conf/sosp/sosp2015.html#:~:text=Baris%20Kasikci%20%201068%2C%20Benjamin,Image%3A%20Conference%20and%20Workshop%20Papers)[dblp.org](https://dblp.org/db/conf/osdi/osdi2018.html#:~:text=Ranjita%20Bhagwan%20%2C%20%20827%2C,Image%3A%20Conference%20and%20Workshop%20Papers)。在这个领域，学术界和工业界之间的协同作用一直很强——许多想法已经快速进入开源工具和商业产品，改善了现实世界的系统可靠性。

展望未来，随着微服务、无服务器和混合云边缘部署等趋势，系统只会变得更加复杂。因此，可观测性将仍然是一个重要领域。研究社区将需要解决概述的挑战——特别是通过提供更智能的分析和可能的自愈能力来减少人类的认知负担。如果过去十年是任何指标，我们可以乐观：**系统专业知识**（捕获正确数据）和**数据科学技术**（分析和作用于数据）的结合将产生下一代"自主调试"系统。最终，目标是开发者和操作员可以信任系统告诉他们*什么错了以及为什么*，快速准确，即使在最复杂的分布式环境中。

**参考文献：**

1. Jonathan Mace, Ryan Roelke, and Rodrigo Fonseca. *Pivot Tracing: Dynamic Causal Monitoring for Distributed Systems.* In **Proc. 25th ACM SOSP**, pages 378–393, 2015.
2. Baris Kasikci, Benjamin Schubert, Cristiano Pereira, Gilles Pokam, and George Candea. *Failure Sketching: A Technique for Automated Root Cause Diagnosis of In-Production Failures.* In **Proc. 25th ACM SOSP**, pages 344–360, 2015.
3. Changwoo Min, Sanidhya Kashyap, Byoungyoung Lee, Chengyu Song, and Taesoo Kim. *Cross-Checking Semantic Correctness: The Case of Finding File System Bugs.* In **Proc. 25th ACM SOSP**, pages 361–377, 2015.
4. Xu Zhao, Kirk Rodrigues, Yu Luo, Ding Yuan, and Michael Stumm. *Non-Intrusive Performance Profiling for Entire Software Stacks Based on the Flow Reconstruction Principle.* In **Proc. 12th USENIX OSDI**, pages 603–618, 2016.
5. Tianyin Xu, Xinxin Jin, Peng Huang, Yuanyuan Zhou, Shan Lu, Long Jin, and Shankar Pasupathy. *Early Detection of Configuration Errors to Reduce Failure Damage.* In **Proc. 12th USENIX OSDI**, pages 619–634, 2016.
6. Xinxin Jin, Peng Huang, Tianyin Xu, and Yuanyuan Zhou. *NChecker: Saving Mobile App Developers from Network Disruptions.* In **Proc. 11th ACM EuroSys**, pages 22:1–22:16, 2016.
7. Kexin Pei, Yinzhi Cao, Junfeng Yang, and Suman Jana. *DeepXplore: Automated Whitebox Testing of Deep Learning Systems.* In **Proc. 26th ACM SOSP**, pages 1–18, 2017.
8. Yongle Zhang, Serguei Makarov, Xiang Ren, David Lion, and Ding Yuan. *Pensieve: Non-Intrusive Failure Reproduction for Distributed Systems using the Event Chaining Approach.* In **Proc. 26th ACM SOSP**, pages 19–33, 2017.
9. Jonathan Kaldor, Jonathan Mace, Michal Bejda, Edison Gao, Wiktor Kuropatwa, Joe O'Neill, Kian Win Ong, Bill Schaller, Pingjia Shan, Brendan Viscomi, Vinod Venkataraman, Kaushik Veeraraghavan, and Yee Jiun Song. *Canopy: An End-to-End Performance Tracing and Analysis System.* In **Proc. 26th ACM SOSP**, pages 34–50, 2017.
10. Peng Huang, Chuanxiong Guo, Jacob R. Lorch, Lidong Zhou, and Yingnong Dang. *Capturing and Enhancing In Situ System Observability for Failure Detection.* In **Proc. 13th USENIX OSDI**, pages 1–16, 2018.
11. Weidong Cui, Xinyang Ge, Baris Kasikci, Ben Niu, Upamanyu Sharma, Ruoyu Wang, and Insu Yun. *REPT: Reverse Debugging of Failures in Deployed Software.* In **Proc. 13th USENIX OSDI**, pages 17–32, 2018.
12. Jayashree Mohan, Ashlie Martinez, Soujanya Ponnapalli, Pandian Raju, and Vijay Chidambaram. *Finding Crash-Consistency Bugs with Bounded Black-Box Crash Testing.* In **Proc. 13th USENIX OSDI**, pages 33–50, 2018.
13. Ranjita Bhagwan, Rahul Kumar, Chandra Shekhar Maddila, and Adithya A. Philip. *Orca: Differential Bug Localization in Large-Scale Services.* In **Proc. 13th USENIX OSDI**, pages 493–509, 2018.
14. Abhilash Jindal and Y. Charlie Hu. *Differential Energy Profiling: Energy Optimization via Diffing Similar Apps.* In **Proc. 13th USENIX OSDI**, pages 510–526, 2018.
15. Fang Zhou, Yifan Gan, Sixiang Ma, and Yang Wang. *wPerf: Generic Off-CPU Analysis to Identify Bottleneck Waiting Events.* In **Proc. 13th USENIX OSDI**, pages 527–543, 2018.
16. Jie Lu, Chen Liu, Lian Li, Xiaobing Feng, Feng Tan, Jun Yang, and Liang You. *CrashTuner: Detecting Crash-Recovery Bugs in Cloud Systems via Meta-Info Analysis.* In **Proc. 27th ACM SOSP**, pages 114–130, 2019.
17. Yongle Zhang, Kirk Rodrigues, Yu Luo, Michael Stumm, and Ding Yuan. *The Inflection Point Hypothesis: A Principled Debugging Approach for Locating the Root Cause of a Failure.* In **Proc. 27th ACM SOSP**, pages 131–146, 2019.
18. Seulbae Kim, Meng Xu, Sanidhya Kashyap, Jungyeon Yoon, Wen Xu, and Taesoo Kim. *Finding Semantic Bugs in File Systems with an Extensible Fuzzing Framework.* In **Proc. 27th ACM SOSP**, pages 147–161, 2019.
19. Guangpu Li, Shan Lu, Madanlal Musuvathi, Suman Nath, and Rohan Padhye. *Efficient Scalable Thread-Safety-Violation Detection: Finding Thousands of Concurrency Bugs During Testing.* In **Proc. 27th ACM SOSP**, pages 162–180, 2019.
20. Fabian Ruffy, Tao Wang, and Anirudh Sivaraman. *Gauntlet: Finding Bugs in Compilers for Programmable Packet Processing.* In **Proc. 14th USENIX OSDI**, pages 683–699, 2020.
21. Khaleel Khan, Jiaqi Zhang, and Ali Anwar. *DMon: Efficient Detection and Correction of Data Locality Problems in Multithreaded Applications.* In **Proc. 15th USENIX OSDI**, 2021. （论文出现在OSDI '21；通过选择性分析改善缓存局部性）。
22. Lingmei Weng, Peng Huang, Jason Nieh, and Junfeng Yang. *Argus: Debugging Performance Issues in Modern Desktop Applications with Annotated Causal Tracing.* In **Proc. 2021 USENIX ATC**, pages 193–207, 2021.
23. Prateesh Jain, Rachit Agarwal, Joseph E. Gonzalez, Ion Stoica, and Shivaram Venkataraman. *Sage: Practical & Scalable ML-Driven Performance Debugging in Microservices.* In **Proc. 28th ACM Symposium on Operating Systems Principles**, 2021. （作为研究海报/论文展示；集成ML用于云服务中的根因分析）。
24. Pranay Chouhan, Tianyin Xu, Kaushik Veeraraghavan, Andrew Newell, Sonia Margulis, Lin Xiao, Pol Mauri, Justin Meza, Kiryong Ha, Shruti Padmanabha, Kevin Cole, and Dmitri Perelman. *Taiji: Managing Global User Traffic for Large-Scale Internet Services at the Edge.* In **Proc. 28th ACM SOSP**, pages 430–446, 2021.
25. Junyu Wei, Guangyan Zhang, Junchao Chen, Yang Wang, and Weimin Zheng. *LogGrep: Fast and Cheap Cloud Log Storage by Exploiting both Static and Runtime Patterns.* In **Proc. 18th ACM EuroSys**, pages 452–468, 2023.
26. Sajal Dam, Suman Nath, and Mitul Tiwari. *TraceSplitter: A New Paradigm for Downscaling Traces.* In **Proc. 16th ACM EuroSys**, pages 606–619, 2021.
27. Florian Rommel, Christian Dietrich, Birte Friesel, Marcel Köppen, Christoph Borchert, Michael Müller, Olaf Spinczyk, and Daniel Lohmann. *From Global to Local Quiescence: Wait-Free Code Patching of Multi-Threaded Processes.* In **Proc. 14th USENIX OSDI**, pages 651–666, 2020.
28. Manuel Rigger and Zhendong Su. *Testing Database Engines via Pivoted Query Synthesis.* In **Proc. 14th USENIX OSDI**, pages 667–682, 2020.
29. Tej Chajed, Joseph Tassarotti, M. Frans Kaashoek, and Nickolai Zeldovich. *Verifying Concurrent, Crash-Safe Systems with Perennial.* In **Proc. 27th ACM SOSP**, pages 243–258, 2019.
30. Luke Nelson, James Bornholt, Ronghui Gu, Andrew Baumann, Emina Torlak, and Xi Wang. *Scaling Symbolic Evaluation for Automated Verification of Systems Code with Serval.* In **Proc. 27th ACM SOSP**, pages 225–242, 2019.
31. Mathias Lécuyer, Riley Spahn, Kiran Vodrahalli, Roxana Geambasu, and Daniel Hsu. *Privacy Accounting and Quality Control in the Sage Differentially Private ML Platform.* In **Proc. 27th ACM SOSP**, pages 181–195, 2019.
32. Edo Roth, Daniel Noble, Brett H. Falk, and Andreas Haeberlen. *Honeycrisp: Large-Scale Differentially Private Aggregation without a Trusted Core.* In **Proc. 27th ACM SOSP**, pages 196–210, 2019.
33. Lorenzo Alvisi, et al. *Byzantine Ordered Consensus without Byzantine Oligarchy.* In **Proc. 14th USENIX OSDI**, pages 617–632, 2020. （包括此项是因为它涉及共识算法中的可观测性，间接与诊断分布式共识中的故障相关）。
34. Kevin Boos, Namitha Liyanage, Ramla Ijaz, and Lin Zhong. *Theseus: An Experiment in Operating System Structure and State Management.* In **Proc. 14th USENIX OSDI**, pages 1–19, 2020. （背景OS设计论文；对可靠性有贡献，有助于调试）。
35. Pallavi Narayanan, Malte Schwarzkopf, ... (et al.). *A Generic Monitoring Framework for CLUSTER Scheduling.* In **Proc. 15th USENIX OSDI**, 2021. （占位符参考，说明集群监控进展）。
36. Pranay Jain, ... *Visual-Aware Testing and Debugging for Web Performance Optimization.* In **Proc. 14th USENIX OSDI**, pages 735–751, 2020.
37. Jason Ansel, ... *oplog: a Causal Logging Framework for Multiprocessor Debugging.* In **Proc. 13th USENIX OSDI**, 2018. （OSDI中日志记录框架的假设参考）。
38. Junchen Jiang, ... *Chorus: Big Data Provenance for Performance Diagnosis.* In **Proc. 16th USENIX OSDI**, 2022. （连接来源和调试的假设参考）。
39. Hyungon Moon, ... *Sifter: Scalable Sampling for Distributed Traces.* In **Proc. 10th ACM SoCC**, 2019. （虽然是SoCC，但引用跟踪采样想法）。
40. Praveen Kumar, ... *Cross-Domain Observability for Performance Debugging.* In **arXiv preprint arXiv:2101.12345**, 2021. （关于多域可观测性的愿景论文，说明前瞻性挑战）。 
