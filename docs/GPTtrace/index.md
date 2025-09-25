# eBPF × AI/LLMs: The Convergence of System Observability and Artificial Intelligence

> Yusheng Zheng

The convergence of Artificial Intelligence and eBPF is rapidly defining the next frontier in system software, creating a paradigm shift in both how we build and manage complex applications. As Large Language Models (LLMs) evolve from mere applications into active AI Agent participants in the software development lifecycle, they are increasingly used to generate, optimize, and verify low-level systems code, including kernel extensions. At the same time, as these sophisticated AI workloads and agents execute, they demand an unprecedented level of runtime context to be operated efficiently, securely, and reliably. This is where eBPF excels, offering a safe and performant mechanism to program the kernel and provide the exact high-fidelity telemetry that modern systems require.

This powerful relationship is a symbiotic loop, where each technology enhances the capabilities of the other. The synergy flows in two primary directions:

  * **eBPF for AI: Enhancing AI/ML Workloads**
    In this direction, eBPF acts as an advanced sensor suite and extension runtime for AI systems. By providing deep, real-time visibility into the kernel, it allows developers to diagnose performance bottlenecks in complex AI/ML pipelines, such as GPU stalls, network I/O latency, and inefficient data access patterns. This telemetry is crucial for optimizing resource utilization, ensuring security compliance for AI agents, and enabling closed-loop feedback systems that can dynamically tune workloads for maximum performance and efficiency.

  * **AI for eBPF: Optimizing the Operating System**
    In the reverse direction, AI and LLMs serve as a force multiplier for kernel development. They are used to automatically generate, verify, and optimize eBPF programs from high-level natural language prompts, dramatically lowering the barrier for creating sophisticated system logic. This unlocks the potential to build dynamic, intelligent, and self-tuning OS policies, including learned CPU schedulers, adaptive network traffic management, and proactive security enforcement, fundamentally enhancing the behavior of the underlying operating system.

The practical impact of this synergy is already evident across a range of cutting-edge applications from 2024–2025. Researchers and practitioners are building systems for comprehensive AI agent observability by tracing LLM Agent prompts and operations, using machine learning to assist in the generation of correct and efficient eBPF programs, and creating intelligent in-kernel data paths with XDP to pre-filter and steer traffic for ML services. Furthermore, this combination enables zero-instrumentation tracing of GPU and LLM workloads, providing critical performance data without requiring any application code changes.

As an open-source community, we are also working on projects combining eBPF and AI, such as:

- [eGPU](https://dl.acm.org/doi/10.1145/3723851.3726984) Offload eBPF bytecode onto GPUs via PTX/SPIR-V injection. It's merged into the main branch of our eBPF runtime [bpftime](https://github.com/bpftime/bpftime).
- [Agentsight](https://github.com/eunomia-bpf/agentsight) Zero instrucment LLM and AI agent (e.g. claude code, gemini-cli) observability in eBPF
- [GPTtrace](https://github.com/eunomia-bpf/GPTtrace) and [MCPtrace](https://github.com/eunomia-bpf/MCPtrace) using LLM to help you trace your kernel, with the paper [Kgent](https://dl.acm.org/doi/10.1145/3672197.3673434) LLM-powered eBPF synthesis tool that incorporates a Z3-based symbolic checks and tests to produce more reliable code, achieving ~80% semantic correctness on its test sets.

## Awesome Lists of eBPF×AI Use Cases (Welcome contribution!)

### **Part 1: eBPF for AI — Observability, Security, and Performance**

This section covers how eBPF provides the essential data for monitoring, securing, and optimizing complex AI and LLM workloads.

#### **A. Tracing and Securing LLM Applications**

To effectively manage and secure AI agents, it's crucial to **stitch high-level prompts to their low-level system effects** using a combination of library uprobes and syscall/network hooks. While LLMs can be used to summarize incidents, hard security policies should be enforced directly in the kernel with eBPF and LSM.

* **AgentSight**: An open-source solution that combines eBPF TLS interception with kernel signals and secondary LLM analysis to trace agent activity with less than 3% overhead ([arXiv](https://arxiv.org/abs/2508.02736)).
* **Groundcover LLM Observability**: Provides enterprise-grade, eBPF-based visibility into LLM API calls and their content ([groundcover.com](https://www.groundcover.com/ai-observability/llm-observability)).
* **Protect AI**: An eBPF agent designed to monitor LLM provider traffic within Kubernetes environments for security and compliance ([protectai.com](https://protectai.com/blog/why-ebpf-is-secure)).
* **Prompt Security**: Uses eBPF for real-time tracing of the model stack and vector database interactions to prevent threats ([Prompt Security](https://www.prompt.security/blog/ebpf-at-prompt-security-the-first-no-code-security-offering-for-llm-based-applications)).
* **eInfer**: An eBPF-based, transparent tracer for distributed LLM inference that correlates per-request performance across CPU/GPU nodes with low overhead ([ACM Digital Library](https://dl.acm.org/doi/abs/10.1145/3748355.3748372)).
* **Runtime Anomaly Detection**: Research demonstrates using eBPF to feed kernel-level signals into ML models for detecting anomalous behavior, such as in ransomware ([arXiv](https://arxiv.org/html/2406.14020v1)) and general process activity via autoencoders on syscall sequences ([evilsocket](https://www.evilsocket.net/2022/08/15/Process-behaviour-anomaly-detection-using-eBPF-and-unsupervised-learning-Autoencoders/)).

#### **B. Zero-Instrumentation GPU Performance Analysis**

A practical approach to GPU performance monitoring is to **start with eBPF uprobes on user-space libraries like CUDA**, supplement this data with NVML/driver metrics, and only then consider more complex device-resident mechanisms.

* **eGPU**: Research prototype that offloads eBPF bytecode onto GPUs via PTX injection—a device-resident path that aligns with AI/GPU workflows ([ACM HCDS'25](https://camps.aptaracorp.com/ACM_PMS/PMS/ACM/HCDS25/10/13a8f7c0-0a7e-11f0-ada9-16bb50361d1f/OUT/hcds25-10.html)). It's merged into the main branch of [bpftime](https://github.com/bpftime/bpftime).
* **CUDA Events Tutorial**: A comprehensive guide for tracing specific CUDA GPU operations using eBPF ([eunomia.dev](https://eunomia.dev/tutorials/47-cuda-events/)).
* **GPU Workload Tracing**: Practitioner guides and tutorials demonstrate using eBPF to catch RDMA tail latencies, OOM conditions, GPU stalls, and prompt rate-limits with minimal application changes ([Medium](https://klizosolutions.medium.com/harnessing-ebpf-for-high-performance-llm-workloads-a-cloud-native-guide-efb7d73e19ed)).
* **eACGM**: A system that merges eBPF kernel events with NVML device metrics to enable end-to-end performance analysis and fault diagnosis for GPU training ([arXiv](https://arxiv.org/html/2506.02007v1)).
* **GPUprobe Tutorials**: A collection of guides on using eBPF uprobes for zero-instrumentation CUDA API tracing, memory tracking, and kernel launch profiling ([DEV Community](https://dev.to/ethgraham/snooping-on-your-gpu-using-ebpf-to-build-zero-instrumentation-cuda-monitoring-2hh1), [Medium](https://medium.com/%40kcl17/inside-cuda-building-ebpf-uprobes-for-gpu-monitoring-449519b236ed)).
* **CUDA Events Tutorial**: A comprehensive guide for tracing specific CUDA GPU operations using eBPF ([eunomia.dev](https://eunomia.dev/tutorials/47-cuda-events/)).
* **eGPU**: Research prototype that offloads eBPF bytecode onto GPUs via PTX injection—a device-resident path that aligns with AI/GPU workflows ([ACM HCDS'25](https://camps.aptaracorp.com/ACM_PMS/PMS/ACM/HCDS25/10/13a8f7c0-0a7e-11f0-ada9-16bb50361d1f/OUT/hcds25-10.html)). It's 
---

### **Part 2: eBPF for AI — In-Kernel Data Path Acceleration**

This section explores how eBPF's position in the kernel's data path can be leveraged to pre-process and accelerate data flows for ML services.

#### **A. Intelligent Traffic Processing with XDP/TC**

The most effective pattern is to **use eBPF as a high-performance sensing and pre-filtering layer** in the kernel, while offloading heavy ML and LLM inference to user space or dedicated hardware.

* **ML-Powered Traffic Processing**: Research demonstrates pipelines that integrate eBPF with ML on commodity hardware for intelligent traffic processing ([ACM Digital Library](https://dl.acm.org/doi/10.1016/j.comnet.2024.110295)).
* **SmartX Intelligent Security**: A framework that uses a BiLSTM model with eBPF/XDP for high-speed threat detection and real-time packet dropping ([arXiv](https://arxiv.org/abs/2410.20244)).
* **In-Kernel vs. User-Space Trade-offs**: Studies have analyzed the latency and throughput trade-offs between in-kernel eBPF/XDP and user-space pipelines for packet classification ([ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1389128624000203)).
* **Intrusion Prevention**: Work from DSN'24 shows how eBPF, XDP, and TC can be used to implement real-time intrusion detection and prevention with in-kernel neural networks ([DSN 2024](https://dsn2024uq.github.io/Proceedings/pdfs/DSN2024-6rvE3SSpzFYmysif75Dkid/410500a416/410500a416.pdf)).
* **Kernel-Time Pre-processing**: While promising, using eBPF to aggregate events for ML services has shown mixed results, highlighting important performance trade-offs that must be measured carefully ([The New Stack](https://thenewstack.io/research-ebpf-not-always-a-silver-bullet-for-network-apps/)).

#### **B. In-Kernel ML Decision Support**

Recent advances enable embedding pre-verified ML models directly in the kernel via eBPF, allowing for intelligent kernel-time decisions.

* **eBPF^ML**: A proposal to attach pre-verified ML models via eBPF objects, including matrix-multiply helpers leveraging CPU matrix engines, for kernel-time decisions ([ACM Digital Library](https://dl.acm.org/doi/10.1145/3748355.3748363)).
* **O2C**: Demonstrates embedding a decision-tree model inside eBPF to enforce kernel compartmentalization on-the-fly, showing what "tiny ML in eBPF" looks like when verifiable ([arXiv](https://arxiv.org/abs/2401.05641)).
* **Flow-based IDS**: Baseline decision-tree-in-eBPF implementation for flow classification; useful foil for "sketch-in-kernel, model-in-user-space" approaches ([GitHub](https://github.com/CN-TU/machine-learning-in-ebpf)).
---

### **Part 3: AI for eBPF — Synthesizing and Verifying Kernel Extensions**

This section details how AI and LLMs are being used to automate the creation and validation of eBPF programs, making kernel programming more accessible and reliable.

* **Kgent(KEN)**: The first LLM-powered eBPF synthesis tool, that incorporates a Z3-based symbolic checks and tests to produce more reliable code, achieving ~80% semantic correctness on its test sets. ([eBPF'24](https://dl.acm.org/doi/10.1145/3672197.3673434/), [arXiv](https://arxiv.org/html/2312.05531v1)).
* **SimpleBPF**: A framework that couples an eBPF DSL with an LLM generator, a semantic checker, and an LLM-based optimizer to consistently emit verifier-friendly programs ([ratul.org](https://ratul.org/papers/ebpf2025-simplebpf.pdf)).
