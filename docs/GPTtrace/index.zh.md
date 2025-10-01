# eBPF × AI/LLMs：系统可观测性与人工智能的融合

> 郑昱笙

人工智能与eBPF的融合正在快速引领系统软件的新方向，彻底改变了复杂应用程序的构建和管理方式。随着大语言模型（LLMs）从单纯的应用程序演变为软件开发生命周期中的活跃AI代理，它们越来越多地用于生成、优化和验证低级系统代码，包括内核扩展。同时，这些复杂的AI工作负载和代理在执行时需要全新的运行时环境，以实现高效、安全、可靠的运行。这正是eBPF的优势所在，它提供了安全且高性能的内核编程机制，为现代系统提供所需的高质量监控数据。

这种强大的关系形成了一个互利共生的循环，每种技术都增强了另一种技术的能力。这种协同作用主要体现在两个方向：

  * **eBPF为AI服务：增强AI/ML工作负载**
    在这个方向上，eBPF充当AI系统的高级传感器套件和扩展运行时。通过提供对内核的深度、实时可见性，它使开发者能够诊断复杂AI/ML管道中的性能瓶颈，如GPU停滞、网络I/O延迟和低效的数据访问模式。这种监控数据对于优化资源利用率、确保AI代理的安全合规性以及实现能够动态调整工作负载以达到最佳性能和效率的闭环反馈系统非常重要。

  * **AI为eBPF服务：优化操作系统**
    在相反方向上，AI和LLMs作为内核开发的助力工具。它们被用于从高级自然语言提示自动生成、验证和优化eBPF程序，大大降低了创建复杂系统逻辑的难度。这释放了构建动态、智能和自我调优操作系统策略的潜力，包括学习型CPU调度器、自适应网络流量管理和主动安全执行，从根本上增强了底层操作系统的行为。

这种协同作用的实际影响在2024-2025年的一系列创新应用中已经清晰可见。研究人员和实践者正在构建系统，通过跟踪LLM代理的提示和操作来实现全面的AI代理可观测性，使用机器学习辅助生成正确和高效的eBPF程序，创建智能的内核数据路径，利用XDP为ML服务预过滤和引导流量。此外，这种组合实现了GPU和LLM工作负载的零插桩跟踪，提供关键性能数据而无需修改任何应用程序代码。

作为开源社区，我们也在开发结合eBPF和AI的项目，例如：

- [eGPU](https://dl.acm.org/doi/10.1145/3723851.3726984) 通过PTX/SPIR-V注入将eBPF字节码卸载到GPU。它已合并到我们的eBPF运行时[bpftime](https://github.com/bpftime/bpftime)的主分支中。
- [Agentsight](https://github.com/eunomia-bpf/agentsight) 在eBPF中实现零插桩LLM和AI代理（如claude code、gemini-cli）可观测性
- [GPTtrace](https://github.com/eunomia-bpf/GPTtrace) 和 [MCPtrace](https://github.com/eunomia-bpf/MCPtrace) 使用LLM帮助您跟踪内核，配合论文[Kgent](https://dl.acm.org/doi/10.1145/3672197.3673434) LLM驱动的eBPF合成工具，该工具结合了基于Z3的符号检查和测试来产生更可靠的代码，在其测试集上达到约80%的语义正确性。

## eBPF×AI用例列表（欢迎贡献！）

### **第一部分：eBPF为AI服务 — 可观测性、安全性和性能**

本节介绍eBPF如何为监控、保护和优化复杂AI和LLM工作负载提供必要数据。

#### **A. 跟踪和保护LLM应用程序**

要有效管理和保护AI代理，使用库uprobes和系统调用/网络钩子的组合**将高级提示与其低级系统效果关联起来**十分必要。虽然LLMs可用于总结事件，但有效的安全策略应直接在内核中通过eBPF和LSM强制执行。

* **AgentSight**：一个开源解决方案，结合eBPF TLS拦截与内核信号和二级LLM分析来跟踪代理活动，开销小于3%（[arXiv](https://arxiv.org/abs/2508.02736)）。
* **Groundcover LLM可观测性**：提供企业级、基于eBPF的LLM API调用及其内容的可见性（[groundcover.com](https://www.groundcover.com/ai-observability/llm-observability)）。
* **Protect AI**：一个eBPF代理，设计用于在Kubernetes环境中监控LLM提供商流量以确保安全和合规（[protectai.com](https://protectai.com/blog/why-ebpf-is-secure)）。
* **Prompt Security**：使用eBPF进行模型堆栈和向量数据库交互的实时跟踪以防止威胁（[Prompt Security](https://www.prompt.security/blog/ebpf-at-prompt-security-the-first-no-code-security-offering-for-llm-based-applications)）。
* **eInfer**：一个基于eBPF的分布式LLM推理透明跟踪器，以低开销关联CPU/GPU节点间的每请求性能（[ACM Digital Library](https://dl.acm.org/doi/abs/10.1145/3748355.3748372)）。
* **运行时异常检测**：研究展示了使用eBPF将内核级信号输入ML模型以检测异常行为，如勒索软件（[arXiv](https://arxiv.org/html/2406.14020v1)）和通过系统调用序列的自动编码器进行的一般进程活动（[evilsocket](https://www.evilsocket.net/2022/08/15/Process-behaviour-anomaly-detection-using-eBPF-and-unsupervised-learning-Autoencoders/)）。

#### **B. 零插桩GPU性能分析**

GPU性能监控的实用方法是**从用户空间库（如CUDA）的eBPF uprobes开始**，用NVML/驱动程序指标补充这些数据，然后才考虑更复杂的设备驻留机制。

* **eGPU**：通过PTX注入将eBPF字节码卸载到GPU的研究原型——一个与AI/GPU工作流程一致的设备驻留路径（[ACM HCDS'25](https://camps.aptaracorp.com/ACM_PMS/PMS/ACM/HCDS25/10/13a8f7c0-0a7e-11f0-ada9-16bb50361d1f/OUT/hcds25-10.html)）。它已合并到[bpftime](https://github.com/bpftime/bpftime)的主分支中。
* **CUDA事件教程**：使用eBPF跟踪特定CUDA GPU操作的综合指南（[eunomia.dev](https://eunomia.dev/tutorials/47-cuda-events/)）。
* **GPU工作负载跟踪**：实践指南和教程演示了使用eBPF捕获RDMA尾部延迟、OOM条件、GPU停滞和提示速率限制，应用程序修改最少（[Medium](https://klizosolutions.medium.com/harnessing-ebpf-for-high-performance-llm-workloads-a-cloud-native-guide-efb7d73e19ed)）。
* **eACGM**：一个将eBPF内核事件与NVML设备指标整合的系统，实现GPU训练的端到端性能分析和故障诊断（[arXiv](https://arxiv.org/html/2506.02007v1)）。
* **GPUprobe教程**：使用eBPF uprobes进行零插桩CUDA API跟踪、内存跟踪和内核启动分析的指南集合（[DEV Community](https://dev.to/ethgraham/snooping-on-your-gpu-using-ebpf-to-build-zero-instrumentation-cuda-monitoring-2hh1), [Medium](https://medium.com/%40kcl17/inside-cuda-building-ebpf-uprobes-for-gpu-monitoring-449519b236ed)）。

---

### **第二部分：eBPF为AI服务 — 内核数据路径加速**

本节探讨如何利用eBPF在内核数据路径中的位置来预处理和加速ML服务的数据流。

#### **A. 使用XDP/TC的智能流量处理**

最有效的模式是**在内核中使用eBPF作为高性能感知和预过滤层**，同时将复杂的ML和LLM推理卸载到用户空间或专用硬件。

* **ML驱动的流量处理**：研究展示了在商用硬件上集成eBPF与ML进行智能流量处理的管道（[ACM Digital Library](https://dl.acm.org/doi/10.1016/j.comnet.2024.110295)）。
* **SmartX智能安全**：一个使用BiLSTM模型与eBPF/XDP进行高速威胁检测和实时数据包丢弃的框架（[arXiv](https://arxiv.org/abs/2410.20244)）。
* **内核与用户空间权衡**：研究分析了内核eBPF/XDP和用户空间管道在数据包分类方面的延迟和吞吐量权衡（[ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1389128624000203)）。
* **入侵防护**：DSN'24的工作展示了如何使用eBPF、XDP和TC实现实时入侵检测和防护，利用内核神经网络（[DSN 2024](https://dsn2024uq.github.io/Proceedings/pdfs/DSN2024-6rvE3SSpzFYmysif75Dkid/410500a416/410500a416.pdf)）。
* **内核时间预处理**：虽然前景看好，但使用eBPF为ML服务聚合事件已显示出不一致的结果，突显了需要仔细评估的重要性能权衡（[The New Stack](https://thenewstack.io/research-ebpf-not-always-a-silver-bullet-for-network-apps/)）。

#### **B. 内核ML决策支持**

最近的进展使得能够通过eBPF将预验证的ML模型直接嵌入内核，实现智能的内核级决策。

* **eBPF^ML**：一个通过eBPF对象附加预验证ML模型的方案，包括利用CPU矩阵引擎的矩阵乘法辅助功能，用于内核级决策（[ACM Digital Library](https://dl.acm.org/doi/10.1145/3748355.3748363)）。
* **O2C**：演示了在eBPF内嵌入决策树模型以动态执行内核隔离，展示了可验证的"eBPF中的微型ML"实例（[arXiv](https://arxiv.org/abs/2401.05641)）。
* **基于流的IDS**：用于流量分类的基准决策树-in-eBPF实现；对"内核中的数据收集，用户空间中的模型"方法提供了有价值的对比（[GitHub](https://github.com/CN-TU/machine-learning-in-ebpf)）。

---

### **第三部分：AI为eBPF服务 — 合成和验证内核扩展**

本节详细介绍AI和LLMs如何被用于自动化eBPF程序的创建和验证，使内核编程更加易于上手和可靠。

* **Kgent(KEN)**：第一个LLM驱动的eBPF合成工具，结合了基于Z3的符号检查和测试来产生更可靠的代码，在其测试集上达到约80%的语义正确性（[eBPF'24](https://dl.acm.org/doi/10.1145/3672197.3673434/), [arXiv](https://arxiv.org/html/2312.05531v1)）。
* **SimpleBPF**：一个将eBPF DSL与LLM生成器、语义检查器和基于LLM的优化器结合的框架，能够一致地生成验证器友好的程序（[ratul.org](https://ratul.org/papers/ebpf2025-simplebpf.pdf)）。