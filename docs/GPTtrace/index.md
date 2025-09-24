# eBPF × AI/LLMs

## 1) Why this combination matters

* **eBPF** gives safe, low-overhead hooks at stable OS boundaries (syscalls, kprobes/uprobes, TCP, XDP/TC, cgroup, LSM, sched\_ext, etc.), yielding per-event, per-flow, per-process signals with kernel-time timestamps.
* **AI/LLMs** need trustworthy runtime context: agent intent vs. actual effects; network/IO stalls; GPU/accelerator behavior; policy enforcement; anomaly and threat detection.
* The synergy shows up in two directions:
  **(A) eBPF → AI**: high-fidelity telemetry feeding ML/LLM analysis or closed-loop control.
  **(B) AI/LLM → eBPF**: LLMs synthesize/verify/optimize eBPF programs and OS policies.

Representative problem statements from 2024–2025 work: agent/LLM observability across TLS and processes; ML-assisted eBPF program generation and correctness; ML-in-the-kernel data paths (XDP) to pre-filter and steer traffic; GPU/LLM workload tracing without app changes. ([arXiv][1])

---

## 2) A taxonomy of eBPF×AI use cases

### A. Observability & security for LLM apps/agents

1. **Agent/LLM boundary tracing**: Correlate LLM prompts/responses with kernel/process activity; surface prompt-injection, runaway loops, secret exfil, and coordination bottlenecks.
   – AgentSight: eBPF TLS interception + kernel signals + secondary LLM analysis; open-source; <3% overhead reported. ([arXiv][2])
   – Industry: Groundcover “LLM observability”, Protect AI’s eBPF-based LLM traffic monitoring, Prompt Security’s eBPF runtime for GenAI apps. ([groundcover.com][3])

2. **Runtime AI security**: Use eBPF signals for threat detection and prevention with ML backends.
   – Ransomware and behavior analytics using kernel-level features. ([arXiv][4])
   – Unsup. anomaly detection from syscall sequences (autoencoders). ([evilsocket][5])

3. **Zero-instr GPU/LLM tracing**: Uprobes on CUDA APIs and kernel launch paths; tie GPU events to LLM workloads without changing app code. ([DEV Community][6])

### B. ML inside the data path (pre-filtering, steering, rate-limits)

1. **XDP/TC with ML decisioning**: Combine ultra-fast packet hooks with ML models (often user-space inference, in-kernel feature extraction).
   – Commodity hardware pipeline integrating eBPF with ML for traffic processing. ([ACM Digital Library][7])
   – BiLSTM/XDP threat detection (SmartX Intelligent Sec). ([arXiv][8])
   – Studies comparing eBPF/XDP vs. user-space pipelines for classification latency/throughput. ([ScienceDirect][9])
   – Practical guides for XDP/TC roles in intrusion prevention. ([DSN 2024][10])

2. **Kernel-time pre-processing for ML services**: Reduce data movement by shaping/aggregating events before they hit user-space ML pipelines; mixed results, trade-offs highlighted in recent measurements. ([The New Stack][11])

### C. eBPF signals for performance modeling of AI/LLM stacks

* **GPU/AI performance correlation**: Merge eBPF kernel events with NVML device metrics for end-to-end analysis (training/fault diagnosis). ([arXiv][12])
* **LLM workload SRE**: Articles/practitioner guides show eBPF catching RDMA tail latencies, OOMs, GPU stalls, prompt rate-limits, with near-zero app changes. ([Medium][13])

### D. AI/LLMs to write and verify eBPF

* **Natural-language → eBPF**: KEN, Kgent: prompt-to-program pipelines with symbolic execution/constraints to satisfy the verifier, showing much higher semantic correctness than LLM-only synthesis. ([arXiv][14])
* **LLM-assisted DSLs & correctness**: SimpleBPF couples an eBPF DSL, LLM generator, semantic checker (Z3), and LLM optimizer to consistently emit verifier-friendly programs. (eBPF’25). ([ratul.org][15])
* **Verifier research** enabling richer safety/correctness guarantees (context for LLM pipelines): VEP two-stage proof checking; OSDI’24 state-embedding validator for the verifier. ([USENIX][16])

### E. Learning OS policies with eBPF hooks (sched\_ext, storage, prefetch)

* **sched\_ext** allows loading custom CPU schedulers via eBPF; several tutorials and adopters discuss non-trivial scheduling policies (potential for RL/learning-based scheduling). ([free5gc.org][17])
* **FetchBPF** (ATC’24): pluggable kernel prefetching policies via eBPF—fertile ground for data-driven/learned policies trained offline and enforced online. ([USENIX][18])

### Observability & security for LLM/agents

* **AgentSight (2025)**: LLM agent ops observability via eBPF+TLS+LLM correlation; detects prompt-injection, reasoning loops. ([arXiv][2])
* **Groundcover LLM Observability (2025)**: eBPF-based visibility into LLM API calls and content. ([groundcover.com][3])
* **Protect AI (2025)**: eBPF agent that watches LLM provider traffic in K8s for prompt/response scanning. ([protectai.com][20])
* **Prompt Security (2024)**: eBPF to trace model stack + vector DB interactions; real-time prevention. ([Prompt Security][21])

### GPU/AI workload tracing without app changes

* **GPUprobe tutorials & writeups (2024–2025)**: eBPF uprobes for CUDA API tracing, memory tracking, kernel launch profiling. ([DEV Community][6])
* **eACGM (2025)**: eBPF tracing + NVML metrics for GPU training analysis/fault diagnosis. ([arXiv][12])

### XDP/TC + ML pipelines

* **Commodity ML traffic processing with eBPF (2024, Computer Networks)**: integrates ML with fast in-kernel processing. ([ACM Digital Library][7])
* **SmartX Intelligent Sec (2024)**: eBPF/XDP + BiLSTM for threat detection, real-time packet dropping efficacy reported. ([arXiv][8])
* **Practicality of in-kernel vs user-space classification (2024)**: latency/throughput trade-offs measured. ([ScienceDirect][9])
* **Intrusion prevention via XDP and TC (DSN’24 paper context)**. ([DSN 2024][10])

### AI/LLMs to synthesize/verify eBPF

* **KEN (2023)**: NL→eBPF with analysis; \~80% semantically correct on their tests, surpassing naive LLM generation. ([arXiv][14])
* **Kgent (2024)**: LLM-powered eBPF synthesis with symbolic checks and tests; practical blog + tech report. ([Eunomia][22])
* **SimpleBPF (eBPF’25)**: DSL + LLM generator + Z3 semantic checker + LLM optimizer. ([ratul.org][15])
* **Verifier tooling**: VEP (NSDI’25) two-stage proof checker; OSDI’24 state-embedding validates verifier correctness. ([USENIX][16])

### Behavior modeling & anomaly detection with eBPF features

* **Process-behavior anomaly detection** (autoencoders on syscall sequences). ([evilsocket][5])
* **Ransomware detection**: two-phase ML with kernel-level sensing. ([arXiv][4])

### OS policy learning hooks

* **sched\_ext** examples in C/Rust and practitioner guides—ripe for RL/black-box optimization for LLM serving placement/QPS-latency tradeoffs. ([free5gc.org][17])
* **FetchBPF** pluggable prefetch policies—template for learned kernel policies. ([USENIX][18])

## 8) Open problems & research directions

1. **AgentOps causality & provenance**
   Design principled causality models linking prompts, tools, subprocesses, sockets, and files — with proofs of completeness/false-positive bounds. (AgentSight hints at this; formalization is open.) ([arXiv][1])

2. **LLM-in-the-loop kernel policies**
   With **sched\_ext** and frameworks like **FetchBPF**, explore learned prefetching, NUMA placement, admission control for inference QPS under tail-latency SLOs. Tight loops must still live outside the kernel; eBPF enforces, user space learns. ([free5gc.org][17])

3. **Verifier-aware code synthesis**
   Push SimpleBPF/KEN further: counter-example guided LLM refinement, proof-carrying eBPF, and dataflow certificates that the in-kernel checker can validate quickly. (VEP points the way.) ([ratul.org][15])

4. **GPU/accelerator-resident hooks**
   Today we uprobe userspace CUDA APIs; the next step is **device-context attach points** with safety guarantees akin to eBPF (ongoing community discussions). Tie device events to LLM pipelines for closed-loop optimization. ([Medium][23])

5. **Privacy-preserving LLM observability**
   Differentially private sketches in eBPF; on-hook redaction; encrypted analytics where only policy predicates leak. Industry content inspection tools suggest demand; academic work is thin. ([protectai.com][20])

6. **Benchmarking the stack**
   We need shared suites that include LLM agent tasks, network traffic (encrypted), GPU workloads, and attack traces, with eBPF hooks standardized for reproducibility. Current datasets/testbeds touch parts of this. ([arXiv][25])

* A **standardized schema** for agent/LLM observability records at the kernel boundary (think OpenTelemetry for eBPF hooks + agent semantics). (Industry pages hint at bespoke schemas.) ([groundcover.com][3])
* A **public, privacy-scrubbed dataset** combining LLM prompts+effects with kernel/network/GPU traces for method comparisons. (Current testbeds focus on traffic.) ([arXiv][25])
* **Verifier-aware LLMs** shipped with proof artifacts as first-class outputs (beyond paper prototypes). (SimpleBPF/VEP are steps.) ([ratul.org][15])
* Clear **guidelines for TLS inspection** boundaries for AI safety vs. user privacy, and reference implementations that enforce them at the hook. (Industry blogs raise the need.) ([Palo Alto Networks][24])

---

## 12) TL;DR guidance for practitioners

* Use eBPF as a **sensing + pre-filter layer**; keep heavy ML/LLM in user space.
* For agent/LLM observability, **stitch prompts↔effects** via library uprobes + syscall/net hooks; use an LLM to summarize and classify incidents, but keep **hard policies** in eBPF/LSM. ([arXiv][2])
* If you want AI to **write eBPF**, don’t ship raw LLM output; use a DSL + SMT checks + tests; integrate VEP-style checking. ([ratul.org][15])
* Treat GPU tracing with **uprobes first**; add NVML/driver metrics; only then consider device-resident mechanisms. ([DEV Community][6])

---

If you want, I can:

* turn this into a LaTeX survey (ACM format) with a taxonomy figure and a two-page related-work table,
* add a minimal **reference pipeline** (BPF + Python + NVML + OpenTelemetry exporter),
* or produce a **threat model checklist** for LLM agents with eBPF hooks.

[1]: https://arxiv.org/html/2508.02736v1 "System-Level Observability for AI Agents Using eBPF"
[2]: https://arxiv.org/abs/2508.02736 "AgentSight: System-Level Observability for AI Agents Using eBPF"
[3]: https://www.groundcover.com/ai-observability/llm-observability "groundcover LLM Observability"
[4]: https://arxiv.org/html/2406.14020v1 "Leveraging eBPF and AI for Ransomware Nose Out"
[5]: https://www.evilsocket.net/2022/08/15/Process-behaviour-anomaly-detection-using-eBPF-and-unsupervised-learning-Autoencoders/ "Process Behaviour Anomaly Detection Using eBPF and ..."
[6]: https://dev.to/ethgraham/snooping-on-your-gpu-using-ebpf-to-build-zero-instrumentation-cuda-monitoring-2hh1 "Using eBPF to Build Zero-instrumentation CUDA Monitoring"
[7]: https://dl.acm.org/doi/10.1016/j.comnet.2024.110295 "Machine learning-powered traffic processing in commodity ..."
[8]: https://arxiv.org/abs/2410.20244 "SmartX Intelligent Sec: A Security Framework Based on Machine Learning and eBPF/XDP"
[9]: https://www.sciencedirect.com/science/article/pii/S1389128624000203 "Practicality of in-kernel/user-space packet processing ..."
[10]: https://dsn2024uq.github.io/Proceedings/pdfs/DSN2024-6rvE3SSpzFYmysif75Dkid/410500a416/410500a416.pdf "Real-Time Intrusion Detection and Prevention with Neural ..."
[11]: https://thenewstack.io/research-ebpf-not-always-a-silver-bullet-for-network-apps/ "Research: eBPF Can Actually Slow Your Applications"
[12]: https://arxiv.org/html/2506.02007v1 "eACGM: Non-instrumented Performance Tracing and ..."
[13]: https://klizosolutions.medium.com/harnessing-ebpf-for-high-performance-llm-workloads-a-cloud-native-guide-efb7d73e19ed "Harnessing eBPF for High‑Performance LLM Workloads"
[14]: https://arxiv.org/html/2312.05531v1 "KEN: Kernel Extensions using Natural Language"
[15]: https://ratul.org/papers/ebpf2025-simplebpf.pdf "Offloading the Tedious Task of Writing eBPF Programs"
[16]: https://www.usenix.org/system/files/nsdi25-wu-xiwei.pdf "VEP: A Two-stage Verification Toolchain for Full eBPF ..."
[17]: https://free5gc.org/blog/20250509/20250509/ "Hands-On with sched_ext: Building Custom eBPF CPU ..."
[18]: https://www.usenix.org/system/files/atc24-cao.pdf "Customizable Prefetching Policies in Linux with eBPF"
[19]: https://middleware.io/blog/ebpf-observability/ "The Ultimate Guide to eBPF Observability - Middleware.io"
[20]: https://protectai.com/blog/why-ebpf-is-secure "Why eBPF is Secure: A Look at the Future Technology ..."
[21]: https://www.prompt.security/blog/ebpf-at-prompt-security-the-first-no-code-security-offering-for-llm-based-applications "eBPF at Prompt Security: The first no-code security offering ..."
[22]: https://eunomia.dev/en/blogs/kgent/ "Simplifying Kernel Programming: The LLM-Powered eBPF Tool"
[23]: https://medium.com/%40kcl17/inside-cuda-building-ebpf-uprobes-for-gpu-monitoring-449519b236ed "Inside CUDA: Building eBPF uprobes for GPU Monitoring"
[24]: https://www.paloaltonetworks.com/blog/network-security/beginners-guide-to-ai-security-with-ebpf/ "Beginner's Guide to AI Security with eBPF"
[25]: https://arxiv.org/abs/2410.18332 "Advancing Network Security: A Comprehensive Testbed and Dataset for Machine Learning-Based Intrusion Detection"
[26]: https://eunomia.dev/tutorials/47-cuda-events/ "eBPF Tutorial: Tracing CUDA GPU Operations - eunomia"
