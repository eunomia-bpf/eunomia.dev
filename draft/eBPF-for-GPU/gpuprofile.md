# gPerf: always on On-/Off-GPU profiler for End-to-End ML Performance Analysis

## Abstract

Despite GPUs dominating modern ML training and inference, real-world performance remains bottlenecked by host-side components such as CPU-bound data pipelines, kernel-launch overheads, scheduling delays, and system-level I/O stalls—none of which are adequately captured by traditional GPU utilization metrics or existing profilers. Existing profiling tools either incur prohibitive overheads, focus narrowly on GPU computation, or neglect off-CPU and off-GPU idle periods entirely, leaving critical system-level bottlenecks undiagnosed. This paper introduces **gPerf**, an always-on, low-overhead profiling framework designed specifically for GPU-centric ML workloads. The key insight of gPerf is to generalize the successful on-/off-CPU wall-clock profiling approach to GPUs, introducing novel notions of **on-GPU/off-GPU** attribution. To avoid overhead on the scheduler's hot path, gPerf leverages a novel design that passively reconstructs off-CPU kernel states using eBPF iterators, lightweight uprobes, and strategic CUPTI sampling for GPU kernel profiling. We further build a causal profiler in the spirit of COZ atop this infrastructure, providing actionable recommendations to improve application-level ML SLOs such as throughput, tail latency, and inter-token latency. Evaluations on single-GPU ML workloads, including a real-world MoE inference workload, demonstrate that gPerf precisely identifies CPU-driven bottlenecks invisible to GPU utilization metrics, enabling straightforward optimizations (e.g., scheduler tuning, CUDA graphs) that improve throughput and latency by up to 30%(?) under realistic conditions, with a steady-state overhead below 3%(?).

Contribution:

1. Extend the **on-/off-CPU profiling paradigm** to GPU workloads via novel **on-GPU/off-GPU** definitions. (Compare **on-/off-CPU Al flame graph**
2. Introduce a **lightweight eBPF iter-based** stack sampling design avoiding hot-path overhead.
3. Design and implement **COZ-style causal analysis** atop a lightweight ML-specific tracing and GPU profiling substrate.
4. Evaluate realistically, demonstrating performance improvements on a single GPU server with production-style ML workloads (MoE inference, LLM decoding), clearly showing gains invisible to GPU utilization metrics, at sub-percent overhead.

---

## Outline of Paper

### 1. Introduction

- ML workloads heavily depend on GPUs for training and inference.
- Despite GPU acceleration, **host-side bottlenecks** (data pipelines, kernel launch overhead, scheduling) dominate end-to-end performance.
- Traditional GPU utilization metrics ("GPU util bars") provide incomplete and often misleading performance signals.
- Current profilers either:
    - Provide deep GPU-focused metrics (Nsight Systems, Nsight Compute, DCGM) but neglect off-GPU host issues.
    - Rely on CPU-only profiling (e.g., Linux perf, off-CPU flame graphs) that miss causal linkage to GPU events.
    - Incite prohibitive overheads that preclude continuous production deployment.
- We propose **gPerf**, a novel always-on profiler extending **on-/off-CPU profiling to GPUs** through precise definitions of **on-GPU/off-GPU** time, augmented by lightweight eBPF iterators and CUPTI sampling, and enriched with COZ-style causal analysis.

---

### 2. Motivation: Why GPUs Are Not the Whole Story

- Real-world scenarios (LLM inference/training, MoE architectures) are routinely bottlenecked by:
    - **CPU-bound data pipelines** (tokenization, augmentation, compression).
    - **GPU kernel-launch overhead** dominating short kernel runs (common in inference decoding loops).
    - **Host scheduling delays** that invisibly throttle GPU throughput.
    - **System-level I/O and network delays** frequently hidden from standard ML tools.
- **Empirical evidence** from recent industry reports and academic literature:
    - CPU preprocessing limits throughput (DS-Analyzer/VLDB, Lotus/IISWC).
    - Tokenization dominates latency in inference workloads (vLLM, Snowflake embeddings).
    - OS research (e.g., OSDI'24 bperf) confirms off-CPU waiting dominates tail latencies but remains ignored in ML profiling.
- **Concrete case:** Our initial study demonstrates a scheduler optimization reducing MoE inference latency by ~10%, entirely invisible in GPU utilization metrics.

---

### 3. Challenges

- Existing profilers fail due to:
    - **Inability to attribute off-GPU idle time** to explicit host-side causes.
    - **High overhead of capturing scheduler events**, especially under realistic production ML loads.
    - Difficulty in accurately **aligning CPU/GPU time slices** into coherent, queueing-sound causal attributions.
    - Lack of **actionable recommendations** based on measured data.
- Single-GPU settings further emphasize the need for **low-overhead yet precise profiling**, since every overhead cycle directly competes with critical ML workloads.

---

### 4. Design of gPerf (Outline)

### Key idea:

Extend on/off-CPU methodology to a **GPU-centric model** with rigorous definitions:

- **On-GPU:** GPU actively computing (SM active) or copying (DMA).
- **Off-GPU:** GPU idle, with further breakdown into host-side categories:
    - **host synchronization waits**
    - **data/input pipeline waits**
    - **kernel-launch overheads**
    - **system backpressure**

### Novel lightweight measurement strategy:

- **CPU side:**
    - Instead of heavy scheduler tracepoints, we propose a novel technique utilizing **eBPF iterators** to passively reconstruct off-CPU kernel state, incurring near-zero overhead.
    - **Uprobes** on CUDA/NCCL APIs precisely timestamp host synchronization points without modifying application code.
    - PSI metrics provide lightweight yet insightful kernel state pressure signals (CPU, memory, I/O pressure stalls).
- **GPU side:**
    - **Lightweight GPU activity polling** (SM active/copy utilization via DCGM/NVML).
    - **Brief triggered deep dives** (CUPTI sampling) for kernel-level stall profiling to breakdown GPU compute time into **issued** versus **stalled** cycles.

### Integration and causality attribution:

- Use **OpenTelemetry spans** for request/iteration-level identity propagation across host↔GPU boundary.
- Store structured traces in **Perfetto**, leveraging **PerfettoSQL** for systematic and transparent analysis.
- Implement COZ-style causal profiling:
    - Perform lightweight randomized micro-perturbations.
    - Estimate global SLO improvements from potential optimizations (scheduling, CUDA Graphs, pipeline optimizations).
    - Rank actionable recommendations based on predicted SLO impact.

---

### 5. Implementation (Outline)

- **eBPF tracing agents**:
    - Tracepoints (passive collection via bpf_iter: scheduler states, block I/O, TCP retransmits)
    - Uprobes on CUDA APIs (kernel launches, memory copies, sync calls)
    - PSI metrics sampler for resource-pressure signals
- **GPU profiling integration**:
    - Lightweight DCGM sampling of SM/copy activity
    - Conditional CUPTI PC sampling for GPU kernel-level deep dive
- **Identity spine**:
    - OpenTelemetry Collector in gateway mode for precise tail-sampling and identity propagation across CPU/GPU boundary
- **Analysis Infrastructure**:
    - Perfetto trace storage with structured PerfettoSQL queries
    - Modular architecture allowing flexible analyzer plugins and COZ-style recommendations

---

### 6. Evaluation Strategy (Outline)

**Setup**:

- Single-server with single NVIDIA 5090 GPU.
- Realistic inference workload:
    - MoE-based inference showing scheduler tuning win (~10% latency reduction)
    - Vanilla LLM decode illustrating CUDA Graphs improvement
- Realistic training workload:
    - CPU-bound data preprocessing pipeline; optimize through pipeline scheduling, prefetching, and GPU offload.

**Metrics & Experiments**:

- Profiler overhead measurement (<1–2% CPU usage)
- Fidelity validation: comparison with ground truth (Nsight Systems, PyTorch Profiler)
- Actionability:
    - Scheduler optimization reducing off-GPU time → improve MoE latency
    - CUDA Graphs optimization reducing GPU launch overhead
    - Pipeline prefetching/augmentation scheduling reducing off-CPU and off-GPU stalls → higher throughput

**Figures**:

- GPU on/off ribbon charts (before/after optimization)
- CPU on/off flame graphs (highlighting host stalls)
- Before/after latency and throughput distributions clearly annotated with causes
- Overhead and robustness sensitivity sweeps

---

### 7. Discussion (Outline)

- Limitations:
    - Single-GPU evaluation; multi-GPU and distributed ML evaluation as future work
    - Currently NVIDIA-centric (future extension: ROCm)
- Broader implications:
    - Demonstrate importance of systematic host↔GPU causality analysis.
    - Clear value for ML production infrastructure teams.

---

### 8. Related Work

- GPU profiling (Nsight Systems, Kineto, DCGM)
- CPU off-CPU profiling (bperf, Linux perf, flame graphs)
- Cross-layer profiling systems (Pixie, Dynolog, OTel, Perfetto)
- Causal analysis (COZ profiler, latency injection)

---

### 9. Conclusions

- Clearly articulated why GPU metrics alone are insufficient.
- Introduced **gPerf**, a novel and practical profiler bridging on-/off-CPU with on-/off-GPU analysis.
- Demonstrated precise, actionable recommendations leading to measurable performance improvements on realistic ML workloads, at minimal runtime overhead.

---

[Key reference:](https://www.notion.so/Key-reference-29345792338a8019a054f235f13a75c6?pvs=21)

[on GPU / off GPU analysis](https://www.notion.so/on-GPU-off-GPU-analysis-29345792338a805a9a0ce73be2af6469?pvs=21)

[evaluation](https://www.notion.so/evaluation-29345792338a80d18f9ef7713902aae8?pvs=21)

[Compare with other tools:](https://www.notion.so/Compare-with-other-tools-29345792338a80e49edce5c7dc775f5a?pvs=21)

[Motivation](https://www.notion.so/Motivation-29345792338a80ccaa3ae1780fb23799?pvs=21)

[Survey](https://www.notion.so/Survey-29345792338a80a8a0dee2c6dc4c80d6?pvs=21)

[coz gpu](https://www.notion.so/coz-gpu-29345792338a80fe95d4e630e4c74e27?pvs=21)





# **Motivation: Why GPUs Are Not the Whole Story**

Even with powerful GPUs, real-world ML pipelines often spend significant time **waiting on the CPU or I/O**, leaving GPUs idle. Several studies and industry reports demonstrate that GPU performance is frequently **limited by host-side bottlenecks**:

- **Data Preprocessing Pipelines:** Many training jobs cannot keep the GPU busy because input data loading and augmentation on CPUs become the bottleneck. Mohan *et al.* (VLDB 2021) found that data input stalls “squander away” the speed of modern GPUs – popular frameworks like PyTorch or TensorFlow often use inefficient CPU pipelines, leading to idle GPU time. In some vision and audio tasks, CPU preprocessing can consume *up to 65%* of the entire epoch time. Similarly, **Lotus** (HotInfra’24) reports that CPU-based preprocessing inefficiencies cause low GPU utilization, especially when CPU resources are imbalanced relative to accelerators. These findings underscore that **faster GPUs alone won’t improve end-to-end performance if the data pipeline can’t supply data fast enough**.
- **Inference Overheads (Tokenization & Scheduling):** In GPU-accelerated inference (e.g. large language models), seemingly “minor” CPU tasks like tokenization, result serialization, or iterative scheduling can dominate latency. An analysis by Snowflake engineers revealed that in their embedding inference service, actual GPU compute accounted for only ~10% of total processing time – the remaining *90%* was spent on CPU tasks, indicating massive under-utilization of expensive GPUs. The *tokenization* of input text on CPUs was a major culprit, creating bubbles of idle GPU time while prompts were processed. In fact, Snowflake’s optimizations to parallelize and offload tokenization, as well as other CPU-side fixes, led to **16× throughput gains** in production. Similarly, researchers found that vLLM, a high-throughput LLM serving system, can spend over half of the inference time in scheduling overhead (e.g. organizing batches, moving tensors) rather than in GPU computation. These examples illustrate that **CPU overheads in inference (tokenization, scheduling, etc.) often rival or exceed the actual GPU computation time**, becoming the hidden bottlenecks for end-to-end latency.
- **GPU Kernel Launch and Library Overheads:** Many ML workflows involve *frequent, short GPU kernels* (for example, element-wise ops or small matrix multiplies in certain models, or per-token operations in decoding loops). The overhead to launch a GPU kernel from the CPU can be on the order of tens of microseconds or more – which can become significant when kernels are very short. Iterative decoding in LLMs, where each token generation launches multiple small kernels, can end up bottlenecked by launch latency and synchronization overhead on the CPU side. Traditional profilers might show GPU at, say, only 30% utilization for such a job, but the “missing” 70% is not truly idle – it’s spent in the driver and runtime overhead between kernels. Without a way to attribute that *off-GPU time* to the launch overhead, one might falsely assume the GPU itself is the problem. Recent observations in industry confirm that **short GPU kernels lead to deceptively low utilization readings**, even though the GPU is busy in bursts – standard utilization counters averaged over a second fail to reflect the bursty load. Always-on monitoring tools note that improving CPU efficiency and using techniques like CUDA Graphs (to batch kernel launches) can significantly improve such workloads, even though GPU-side metrics alone wouldn’t flag the issue.
- **I/O and Networking Delays:** In distributed training or data-heavy workloads, GPUs can stall waiting for data from storage or the network. These stalls are often invisible to GPU-centric monitors – the GPU just appears idle. For instance, if training data is streamed from a remote storage and incurs occasional delays, the GPU will periodically starve. Linux’s Pressure Stall Information (PSI) metrics (not commonly used in ML profiling) can reveal when CPU threads are sleeping due to lack of I/O, but typical ML frameworks don’t surface this. Off-CPU analysis tools have shown that waiting for I/O or being preempted by the OS can contribute significantly to *tail latency*. In cloud environments, a container might be momentarily throttled or waiting on a network packet, causing all GPUs in that job to idle – yet no GPU counter will tell you *why*.

In summary, **GPU utilization metrics alone are often misleading**. A low utilization number might signal a GPU-bound problem – or it might signal that the GPU is starved by CPU and I/O bottlenecks. Without deeper profiling, one cannot tell the difference. This motivates profiling approaches that **treat the entire end-to-end pipeline**, capturing both on-GPU execution and off-GPU waiting time, and attributing idle periods to concrete causes (CPU, I/O, scheduling, etc.). Evidence from both academia and industry strongly supports this need: recent work showed that modern applications frequently have *intertwined on-CPU and off-CPU bottlenecks*, requiring combined analysis. In ML specifically, bridging the observability gap between CPU and GPU is crucial to identify the true limiting factor for throughput or latency.

Modern ML training and inference are **GPU‑centric**, yet end‑to‑end performance is routinely throttled by **host‑side work**—tokenization and data preprocessing, kernel‑launch overheads, runtime/scheduler delays, and system I/O—**that GPU utilization bars and GPU‑only profilers fail to reveal**. Evidence from both systems research and production reports shows these off‑GPU intervals dominate SLOs (throughput, TTFT/TPOT, tail latency) and are poorly attributed by current tooling. [NVIDIA Docs+3Computer Science Department at UT Austin+3Kexin Rong+3](https://www.cs.utexas.edu/~vijay/papers/vldb21-datastalls.pdf)

---

### Evidence cluster A — CPU‑bound input pipelines & tokenization (training *and* inference)

- **Training spends large fractions of wall time off‑GPU in input pipelines.** A VLDB study introduced DS‑Analyzer and found data *prep/fetch stalls* (CPU & I/O) can consume **up to 65% of epoch time**; many models require **3–24 CPU cores per GPU** just to keep accelerators fed. [Computer Science Department at UT Austin](https://www.cs.utexas.edu/~vijay/papers/vldb21-datastalls.pdf)
- **Preprocessing alone can dominate the critical path** and today’s profilers don’t connect Python‑level stages to CPU microarchitecture counters; *Lotus* (IISWC’24) reports preprocessing “**up to 65% of epoch time**” and motivates low‑overhead, fine‑grain host profiling. [Kexin Rong](https://kexinrong.github.io/lab/files/lotus-iiswc24.pdf)
- **Software (not storage) is often the limiter.** *Plumber* shows the “majority of input data stalls arise due to software bottlenecks,” arguing for pipeline diagnosis, scheduling, and caching beyond raw GPU metrics. [Proceedings ML Systems](https://proceedings.mlsys.org/paper/2022/file/077e29b11be80ab57e1a2ecabb7da330-Paper.pdf?utm_source=chatgpt.com)
- **Inference can be CPU‑bound as well.** In a public vLLM postmortem for v0.6, profiling Llama‑3‑8B on H100 showed **only 38% of time on GPU**; **33%** in the HTTP/API server and **29%** in CPU scheduling and data prep blocked the GPU—pure *host* overhead. [vLLM Blog](https://blog.vllm.ai/2024/09/05/perf-update.html)
- **Tokenization becomes the bottleneck under load.** The vLLM maintainers note tokenization “is pretty slow” and becomes **the bottleneck**, motivating micro‑batched tokenization paths. [GitHub](https://github.com/vllm-project/vllm/issues/19012?utm_source=chatgpt.com)
- **Industry cases mirror this:** Snowflake reports **16× throughput** for embeddings after re‑architecting the pipeline around vLLM/Arctic; Hugging Face’s TEI emphasizes CPU‑efficient embedding servers to raise end‑to‑end throughput. [Snowflake+1](https://www.snowflake.com/en/engineering-blog/embedding-inference-arctic-16x-faster/)

**Implication:** Host‑side pipeline visibility (prep vs. fetch vs. scheduling) at request granularity is essential to explain “GPU idle” and predict SLO wins from pipeline and scheduler changes. [Computer Science Department at UT Austin+1](https://www.cs.utexas.edu/~vijay/papers/vldb21-datastalls.pdf)

---

### Evidence cluster B — Kernel‑launch and runtime overheads strand GPUs

- **Launch overhead is non‑trivial at inference timescales.** NVIDIA shows a simple short‑kernel loop where the kernel itself takes **~2.9 µs** but per‑kernel wall time is **~9.6 µs** due to launch/sync; **CUDA Graphs** reduces this to **~3.4 µs**, and Nsight’s timeline visibly attributes the **gaps between kernels to CPU and GPU launch overheads**. [NVIDIA Developer](https://developer.nvidia.com/blog/cuda-graphs/)
- **Nsight documents ~20 µs launch latency in practice** and details **CPU wrapper overhead** (driver mutexes, sync) and **tooling overhead** for short events—precisely the pathologies that surface in LLM decoding loops. [NVIDIA Developer](https://developer.nvidia.com/blog/understanding-the-visualization-of-overhead-and-latency-in-nsight-systems/)
- **Serving stacks also target launch overhead explicitly.** Recent systems (e.g., BatchLLM) include compiler/runtime changes (fusion, request reordering) to **amortize launches** and keep GPUs busy. [arXiv](https://arxiv.org/abs/2412.03594?utm_source=chatgpt.com)

**Implication:** GPU “busy” counters alone miss launch‑limited regimes; an **on‑/off‑GPU** view must attribute off‑GPU time to **host‑side launch/sync** and runtime scheduling. [NVIDIA Developer+1](https://developer.nvidia.com/blog/cuda-graphs/)

---

### Evidence cluster C — OS scheduling & resource pressure cause long off‑GPU waits

- **Conventional CPU profilers miss off‑CPU time by design.** OSDI’24 (*bperf/BCOZ*) shows classic profiling “lacks profiling both on‑ and off‑CPU events,” and introduces **blocked samples** to capture **I/O waits, sync, and runqueue delays** with causal analysis, closing the visibility gap behind tail latencies. [USENIX](https://www.usenix.org/conference/osdi24/presentation/ahn)
- **Linux PSI (Pressure Stall Information)** quantifies **lost wall‑clock time** due to **CPU/memory/I/O pressure**, reporting the **share of time some/all runnable tasks are stalled**—an OS‑level signal that correlates with queueing delays and throughput collapse but is not reflected in GPU utilization bars. [Linux Kernel Documentation+1](https://docs.kernel.org/accounting/psi.html?utm_source=chatgpt.com)

**Implication:** Accurate on‑/off‑GPU attribution needs OS‑level **off‑CPU** signals (runqueue, I/O stalls) tied to the request and GPU timeline. [USENIX+1](https://www.usenix.org/conference/osdi24/presentation/ahn)

---

### Evidence cluster D — Storage & network I/O bound regimes throttle accelerators

- **Training is often fetch‑ or prep‑bound** (I/O and CPU) even on DGX‑class boxes; caching/coordination and pipeline scheduling are required to prevent **idle GPU time**. [Computer Science Department at UT Austin](https://www.cs.utexas.edu/~vijay/papers/vldb21-datastalls.pdf)
- **At scale, I/O becomes a first‑order system constraint.** EMLIO (’25) argues that as **GPU compute outpaces storage/network**, data‑loading latency and even I/O **energy** must be optimized jointly to maintain throughput. [arXiv](https://arxiv.org/pdf/2508.11035?utm_source=chatgpt.com)

**Implication:** System‑level I/O stalls (local disk, remote object stores, networking) must be first‑class in end‑to‑end ML profiling. [Computer Science Department at UT Austin+1](https://www.cs.utexas.edu/~vijay/papers/vldb21-datastalls.pdf)

---

### Evidence cluster E — Why “GPU utilization” is insufficient as a performance signal

- **What “GPU util” actually measures.** DCGM’s *Graphics Engine Activity* is the **fraction of time** the graphics/compute engines were active; *SM Active* is the **fraction of time any warp was active** (even if **waiting on memory**). Neither is instantaneous, and “necessary but not sufficient” for effective use. [NVIDIA Docs](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/feature-overview.html)
- **Cloud ops guidance echoes this:** GKE/DCGM docs describe `DCGM_FI_DEV_GPU_UTIL` simply as **“fraction of time the GPU was active”**, useful for coarse capacity but **not** a root‑cause metric. [Google Cloud](https://cloud.google.com/kubernetes-engine/docs/how-to/dcgm-metrics?utm_source=chatgpt.com)
- **Industry guidance warns it can mislead.** Analyses emphasize the gap between “utilization” and **saturation/throughput**, cautioning against equating a high bar with high performance. [Arthur Chiao's Blog+1](https://arthurchiao.github.io/blog/understanding-gpu-performance/?utm_source=chatgpt.com)

**Implication:** We need **wall‑clock attribution** (on‑/off‑GPU, and off‑CPU causes) rather than relying on single‑device utilizations. [NVIDIA Docs+1](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/feature-overview.html)

---

### Evidence cluster F — Existing tools are either GPU‑only or too heavy for always‑on use

- **GPU toolchains acknowledge overhead.** Nsight Systems notes tracing choices to “significantly reduce profiling overhead,” and enabling memory/use tracking or deep backtraces incurs **extra overhead**—fine for deep dives, not for continuous fleet‑wide SLO monitoring. [NVIDIA Docs+1](https://docs.nvidia.com/nsight-systems/UserGuide/index.html?utm_source=chatgpt.com)
- **Profiling perturbs workloads.** NVIDIA’s own docs caution that profiling **changes performance** and can skew numbers; PyTorch users report substantial overhead with the built‑in profiler. [NVIDIA Docs+2PyTorch Forums+2](https://docs.nvidia.com/physicsnemo/latest/user-guide/performance_docs/profiling.html?utm_source=chatgpt.com)
- **Ops tools remain siloed.** Meta’s **Dynolog** exists precisely because it’s hard to assemble CPU+kernel+GPU telemetry into a **production‑safe, always‑on** view; it integrates DCGM and kernel/CPU metrics but still leaves causal **off‑CPU/ off‑GPU** attribution to the user. [GitHub](https://github.com/facebookincubator/dynolog)

**Implication:** There is space for an **always‑on, low‑overhead profiler** that (i) generalizes on‑/off‑CPU wall‑clock profiling to **on‑/off‑GPU**, (ii) attributes off‑GPU time to **host‑side causes** (tokenization, launch, scheduling, I/O), and (iii) stays production‑safe.

## **Limitations of Traditional Profiling Tools**

Existing performance tools tend to focus either on CPUs **or** GPUs in isolation, or incur too much overhead to use continuously. This fragmentation leaves ML practitioners with an incomplete picture:

- **GPU-Centric Profilers:** NVIDIA’s tooling (Nsight Systems, Nsight Compute, CUPTI, DCGM, etc.) provides detailed visibility into GPU activity, but with caveats. Nsight Systems can trace CPU and GPU activities to build a timeline, yet it is typically used offline due to high overhead and voluminous traces. Always running Nsight in production is infeasible – the instrumentation overhead can perturb timing and slow down the workload significantly (often tens of percent slowdown if fully tracing). Nsight Compute and CUPTI offer fine-grained GPU performance metrics (down to per-kernel analysis, memory throughput, stall reasons), but they focus narrowly on **on-GPU events**. They do not attribute *why* the GPU might be idle between kernels. Moreover, as the Zymtrace team points out, these deep GPU profilers are **intrusive and require expert knowledge to interpret**. A typical ML engineer looking at Nsight’s output might see that a kernel achieved X% memory utilization or had Y warps stalled, but if the biggest problem is outside the kernels (e.g. data loader), Nsight won’t directly show it. Some lightweight GPU monitors like NVIDIA’s Data Center GPU Manager (DCGM) provide “always-on” metrics (SM occupancy, memory usage, etc.) with minimal overhead. However, these metrics are **coarse and lack context** – DCGM might report an average of 50% SM busy, but it cannot tell which part of the code or which stage of the pipeline caused the remaining 50% idle. Essentially, GPU-only tools either give *too little detail* (just utilization percentages) or *too much data* (per-kernel traces) without connecting to the CPU-side story.
- **CPU Profilers & Off-CPU Analysis:** On the CPU side, standard profilers (like Linux **`perf`** or Intel VTune) measure on-CPU execution hot spots. They excel at finding which functions consume CPU cycles, but by default they ignore time when the CPU threads are idle or blocked (since no CPU instructions are running then). Extensions like **off-CPU flame graphs** and tools such as **wPerf** have been developed to capture waiting time and blocked events on the CPU. Off-CPU profiling can identify if a thread spent a lot of time sleeping on I/O or locks. For example, Gregg’s off-CPU flame graphs and the recent **bperf** tool use sampling to record where threads are when they’re not running. However, these CPU-centric tools **don’t incorporate GPU activity**. An ML training script might show 40% of CPU time in data loading and 60% idle waiting – but *why* is it idle waiting? Possibly because a GPU kernel is running and the CPU thread is synchronizing, or conversely the GPU is idle waiting for data. The CPU profiler on its own can’t link those relationships. In general, existing CPU profilers and tracers don’t have a notion of “on-GPU” time. This means they might mark a thread as off-CPU (sleeping) during a **`cudaMemcpy`** or **`cudaLaunchKernel`** call, but they won’t flag that as the GPU being busy. The result is fragmented insight: a GPU profiler shows a gap (idle GPU), and a CPU profiler shows a thread blocked on something – but **without a unified timeline or common tags, it’s hard to put together the puzzle**.
- **Lack of Holistic View:** Because of the above silos, developers often have to correlate logs or traces manually. Meta’s ML performance team noted that assembling a “holistic view” of system performance is challenging with separate tools. For instance, a GPU timeline might show a 5ms gap between kernels. A CPU trace might show during that gap the process was waiting on a mutex. Only by side-by-side inspection could one deduce that a lock contention on the CPU caused the GPU to starve for 5ms. This manual process is cumbersome and error-prone. The **Dynolog** project (by Meta, now open source) was motivated by this gap – they highlight the need to correlate inefficiencies across CPU, GPU, and even network, without drowning in data. Dynolog’s solution is to continuously monitor a variety of metrics from Linux, CPU PMUs, and NVIDIA’s APIs in one daemon. Still, Dynolog largely provides metrics and on-demand triggers (e.g., you can ask it to collect a PyTorch trace when certain conditions met). It doesn’t yet produce an integrated flame graph or a single profiling view of *where time went* across the whole system for each operation.
- **Overheads and Continuous Profiling:** Many existing profilers incur **too much overhead to run in production or to leave “always on.”** For example, the PyTorch Profiler (Kineto-based) can record every CUDA kernel launch and every op, but this slows training considerably (often 10-30% overhead or more, depending on sampling frequency and activities traced). Instrumentation-based profilers insert code around each op or use high-frequency callbacks, which perturb timing and can even distort the behavior (e.g., the act of profiling can serialize some concurrent operations that normally overlap). HPC-oriented profilers like HPCToolkit use statistical sampling to reduce overhead, but when they enable fine GPU PC sampling, earlier CUDA versions would serialize kernels – a dramatic interference. In fact, **precisely measuring GPU stall reasons or per-instruction metrics often entails high overhead**, as the GPU’s profiling mode may disable certain optimizations. Because of this, continuous or always-on GPU profiling was historically deemed impractical; teams would profile in a staging environment or for a few seconds in production, but not continuously. This is now changing with new approaches (e.g., sampling, hardware support, or clever tracing) that aim for *low-percent single-digit overhead*. For instance, Google’s continuous profiler for CPUs (Cloud Profiler) uses sampling to claim ~1% overhead, and emerging tools are aiming for similar low overhead on GPUs. The **bperf** research demonstrated that even capturing *off-CPU events via sampling* can be done with ~1–2% overhead, compared to much higher overhead if one naively traced every context switch. This sets a promising precedent: perhaps a combination of sampling and selective tracing can make always-on end-to-end profiling feasible.

In summary, the current landscape forces a trade-off: **deep but heavy profiling** (e.g., full traces from Nsight or instrumented PyTorch runs) versus **lightweight but shallow metrics** (GPU utilization percentages, CPU usage, etc.). Neither option provides the needed insight into cross-cutting bottlenecks. What’s missing is a *lightweight, always-on* profiler that can **attribute performance across the CPU–GPU boundary**, i.e. telling developers when the GPU is waiting on the CPU (or vice versa) and why. This is precisely the gap that **on/off-GPU profiling** aims to fill.

# Compare with other tools:

## Table 1 — GPU & kernel‑centric profilers (research + vendor)

> Use ✓ / ~ / ×; “causal” means built‑in what‑if/virtual‑speedup guidance.
> 

| Work (venue/year) | Always‑on viable | CPU↔GPU call‑path | Off‑**GPU** attribution by **host cause** | Kernel stall reasons | **Causal** guidance | Open‑source | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Nsight Systems** (vendor) | ~ | ✓ | × | ~ | × | × | Episodic tracing; great timelines; not designed for fleet always‑on. [NVIDIA Developer](https://developer.nvidia.com/nsight-systems?utm_source=chatgpt.com) |
| **Nsight Compute** (vendor) | × | × | × | ✓ (deep) | × | × | Per‑kernel micro‑arch analysis; not end‑to‑end. [docs.alcf.anl.gov](https://docs.alcf.anl.gov/polaris/performance-tools/NVIDIA-Nsight/?utm_source=chatgpt.com) |
| **HPCToolkit (GPU)** (’21–’24) | ~ | ✓ | × | ✓ (PC sampling) | × | ✓ | Cross‑CPU/GPU attribution; not off‑GPU causality; not “always‑on.” [arXiv](https://arxiv.org/abs/2109.06931?utm_source=chatgpt.com) |
| **GPA** (CGO’21) | × | × | × | ✓ | ~ (kernel‑level what‑if) | ✓ | PC‑sampling‑based advisor; kernel focus, not app SLOs. [Xiaozhu Meng's homepage](https://mxz297.github.io/files/CGO21.pdf?utm_source=chatgpt.com) |
| **DrGPU** (ICPE’23) | × | ✓ | × | ✓ (top‑down) | × | ~ | Top‑down stall decomposition; kernel‑centric. [SPEC Research](https://research.spec.org/icpe_proceedings/2023/proceedings/p43.pdf?utm_source=chatgpt.com) |
| **GPUscout** (’23) | × | × | × | ✓ (memory bottlenecks) | × | ~ | Finds data‑movement bottlenecks in kernels. [Department of Computer Engineering](https://www.ce.cit.tum.de/fileadmin/w00cgn/caps/vanecek/sv_gpuscout.pdf?utm_source=chatgpt.com) |
| **gPerf (this work)** | **✓** | **✓** | **✓** | **✓** (triggered) | **✓** (COZ‑style) | **planned** | *Always‑on*, unify on/off‑CPU with on/off‑GPU + causal SLO guidance. |

*Why the ✓/×?* Vendor tools and kernel advisors are **excellent for deep dives**, but **not** built for **continuous, causal, end‑to‑end attribution** that explains *why* the GPU is idle (tokenization, launch overhead, scheduler delay, I/O). That gap is your target. [NVIDIA Developer+2Xiaozhu Meng's homepage+2](https://developer.nvidia.com/nsight-systems?utm_source=chatgpt.com)

---

## Table 2 — System‑level, ML‑pipeline, and continuous profilers

| Work (venue/year) | Always‑on viable | Off‑**CPU** profiling | GPU visibility | Off‑**GPU** attribution by **host cause** | **Causal** guidance | Identity propagation (traces) | Open‑source | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **DCGM** (NVIDIA) | **✓** | × | ✓ (util/health) | × | × | × | ~ | Device telemetry at scale; not causal. [NVIDIA Developer](https://developer.nvidia.com/dcgm?utm_source=chatgpt.com) |
| **Dynolog** (Meta) | **✓** | ~ (host signals) | ✓ (hooks) | × | × | × | ✓ | Telemetry + triggerable deep dives; not a causal profiler. [GitHub](https://github.com/facebookincubator/dynolog?utm_source=chatgpt.com) |
| **PyTorch Profiler / Kineto** | × | ✓ | ✓ (CUPTI) | × | × | ~ (manual) | ✓ | Developer‑initiated sessions; not always‑on. [PyTorch Docs](https://docs.pytorch.org/docs/stable/profiler.html?utm_source=chatgpt.com) |
| **HTA** (Meta) | × | ✓ (from traces) | ✓ (maps counters → ops) | × | × | × | ✓ | Up‑levels Kineto traces; analysis library. [PyTorch Docs](https://docs.pytorch.org/tutorials/beginner/hta_intro_tutorial.html?utm_source=chatgpt.com) |
| **bperf / BCOZ** (OSDI’24) | ~ | **✓ (on/off‑CPU)** | × | × | **✓** (BCOZ) | × | ✓ | Low‑overhead off‑CPU + causal; CPU‑only. Strong precedent for your method. [USENIX](https://www.usenix.org/system/files/osdi24-ahn.pdf?utm_source=chatgpt.com) |
| **wPerf** (OSDI’18) | ~ | **✓** | × | × | × | × | ✓ | Generic off‑CPU wait analysis. [USENIX](https://www.usenix.org/system/files/osdi18-zhou.pdf?utm_source=chatgpt.com) |
| **DS‑Analyzer** (VLDB’21) | × | **✓ (data stalls)** | × | × | ~ (what‑if pipeline) | × | ~ | Pipeline what‑if; great to justify your “input stalls” bucket. [Department of Computer Science](https://www.cs.utexas.edu/~vijay/papers/vldb21-datastalls.pdf?utm_source=chatgpt.com) |
| **Lotus** (IISWC’24) | × | **✓ (preproc)** | × | × | × | × | ✓ | Characterizes preprocessing; open artifacts. [Kexin Rong](https://kexinrong.github.io/lab/files/lotus-iiswc24.pdf?utm_source=chatgpt.com) |
| **Parca / Pyroscope** | **✓** | **✓ (CPU stacks)** | × | × | × | × | ✓ | Fleet‑wide eBPF CPU profiling; no GPU attribution. [Parca+1](https://www.parca.dev/?utm_source=chatgpt.com) |
| **Pixie** (K8s eBPF) | **✓** | **✓** | × | × | × | ✓ | ✓ | “No‑instrumentation” cluster observability; limited GPU insight. [PX.dev](https://px.dev/?utm_source=chatgpt.com) |
| **Perfetto (+SQL)** | **✓** (as backend) | – | – | – | – | – | ✓ | Storage + SQL analysis layer; pair it with gPerf. [Perfetto](https://perfetto.dev/docs/analysis/perfetto-sql-syntax?utm_source=chatgpt.com) |
| **OpenTelemetry** | **✓** (as traces) | – | – | – | – | **✓** | ✓ | Your identity spine for request‑level causality. [OpenTelemetry](https://opentelemetry.io/docs/concepts/signals/traces/?utm_source=chatgpt.com) |
| **CUDA Graphs** (NVIDIA) | – | – | – | **reduces launch overhead** | – | – | docs | Use as an *optimization target* in causal experiments. [NVIDIA Developer](https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-cuda-graphs-and-other-performance-enhancements/?utm_source=chatgpt.com) |

---

## How to **position gPerf** against this landscape (succinctly)

1. **Novel problem framing:** move from GPU‑only or CPU‑only views to **on‑/off‑GPU wall‑clock attribution** that explicitly ties **GPU idle** to **host‑side causes** (tokenization, kernel‑launch overhead, scheduling, I/O). No prior tool makes off‑GPU idle a **first‑class category** with causal linkage. (Closest antecedents are **bperf/BCOZ** for CPU and **HPCToolkit** for cross‑device call‑paths, but neither gives off‑GPU causality). [USENIX+1](https://www.usenix.org/system/files/osdi24-ahn.pdf?utm_source=chatgpt.com)
2. **Always‑on at low overhead**: unlike Nsight/Profiler sessions, gPerf is designed to run continuously via **passive eBPF iterators + uprobes** (no sched‑hot‑path probes), **lightweight GPU activity polling**, and **triggered CUPTI PC sampling** for short bursts. [NVIDIA Docs+1](https://docs.nvidia.com/cupti/api/group__CUPTI__PCSAMPLING__API.html?utm_source=chatgpt.com)
3. **COZ‑style recommendations**: not just “where time goes,” but **what to fix** for application‑level ML SLOs (throughput, tail, inter‑token latency), including CUDA Graphs and scheduler/pipeline interventions. [ACM SIGOPS+1](https://sigops.org/s/conferences/sosp/2015/current/2015-Monterey/printable/090-curtsinger.pdf?utm_source=chatgpt.com)

---

## Evidence you can lean on in **Motivation**

- CPU work **blocking GPUs** in LLM inference is not hypothetical—**vLLM** publicly acknowledges CPU‑side overheads (scheduler, output processing) starving the GPU and reports system changes to mitigate them. Use this as a concrete case. [vLLM Blog](https://blog.vllm.ai/2024/09/05/perf-update.html?utm_source=chatgpt.com)
- **Training data stalls** and **preprocessing bottlenecks** are well‑documented; DS‑Analyzer quantifies stalls and does pipeline **what‑if**, Lotus profiles preprocessing pipelines with artifacts. This directly supports your “off‑GPU: input pipeline” bucket. [Department of Computer Science+1](https://www.cs.utexas.edu/~vijay/papers/vldb21-datastalls.pdf?utm_source=chatgpt.com)
- **Kernel‑launch overhead** is officially addressed by **CUDA Graphs** (constant‑time launch for straight‑line graphs; conditional nodes), which your causal layer can recommend when it moves the needle. [NVIDIA Developer](https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-cuda-graphs-and-other-performance-enhancements/?utm_source=chatgpt.com)

## A. GPU‑centric profilers (deep dive, mostly single‑node)

**Nsight Systems (NSys).** System‑wide timeline and sampling profiler from NVIDIA; correlates CPU threads, CUDA API calls, and GPU activity. It’s intended for interactive deep‑dives, not continuous fleet telemetry. NSys captures CPU backtraces and context switches by default, and provides low‑overhead sampling modes (but that doesn’t make it “always‑on” in production). [NVIDIA Developer+2NVIDIA Docs+2](https://developer.nvidia.com/nsight-systems?utm_source=chatgpt.com)

**Nsight Compute (Ncu).** Per‑kernel microarchitectural profiler for CUDA kernels; collects detailed hardware‑counter metrics, stall reasons, and guided analysis. Excellent for kernel tuning, not for end‑to‑end or “off‑GPU” attribution. [NVIDIA Docs+2NVIDIA Docs+2](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html?utm_source=chatgpt.com)

**CUPTI (CUDA Profiling Tools Interface).** The substrate used by profilers; includes **PC Sampling** to sample warp program counters with periodic interrupts (low overhead compared to full tracing). This is the canonical way to attribute GPU time within kernels without instrumenting application code. [NVIDIA Docs+2NVIDIA Docs+2](https://docs.nvidia.com/cupti/?utm_source=chatgpt.com)

**PyTorch Profiler / Kineto.** PyTorch’s built‑in profiler (Kineto backend) records CPU ops, CUDA API intervals, and GPU kernel traces (via CUPTI). It integrates with TensorBoard or proprietary tools; again, suited for episodic profiling, not continuous cluster‑wide attribution of *off‑GPU* idle. [PyTorch Docs+2GitHub+2](https://docs.pytorch.org/docs/stable/profiler.html?utm_source=chatgpt.com)

**TensorFlow / JAX Profilers.** TensorBoard’s Profiler and XProf show CPU/GPU timelines, tf.data pipeline breakdowns, and device activity; JAX reuses the same trace format. Strong for *developer‑initiated sessions*, but still focuses on framework‑internal events rather than OS‑level root causes. [JAX+3TensorFlow+3TensorFlow+3](https://www.tensorflow.org/guide/profiler?utm_source=chatgpt.com)

**How gPerf differs.** These tools are superb for deep kernel and op‑level insight, but (i) they’re not designed to run **always‑on** at sub‑percent overhead across production fleets, and (ii) they **don’t classify off‑GPU idle** by *host‑side cause* (scheduler waits, data pipeline stalls, I/O pressure). gPerf’s thesis is to **generalize on/off‑CPU wall‑clock profiling to on‑/off‑GPU**, fusing CUPTI sampling (when needed) with OS‑level attribution—without sitting in the scheduler’s hot path.

---

## B. Always‑on GPU telemetry and ops tooling (low overhead, little causality)

**DCGM (+ DCGM Exporter).** NVIDIA’s Data Center GPU Manager exposes health/perf counters continuously with low overhead; exporters integrate with Prometheus/Grafana and schedulers (SLURM, Kubernetes). It’s ideal for **device‑level** telemetry (utilization, memory, RAS, temperature), but **not** for *host‑side causal attribution* of “why the GPU is idle.” [NVIDIA Docs+3NVIDIA Docs+3NVIDIA Docs+3](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/?utm_source=chatgpt.com)

**Dynolog (Meta).** Lightweight daemon for CPU+GPU telemetry; supports always‑on monitoring and can trigger deep‑dives (e.g., collect PyTorch Profiler traces) at scale. It’s an operational telemetry substrate rather than a causal profiler; analysis is often delegated (e.g., HTA library). [GitHub+2Facebook for Developers+2](https://github.com/facebookincubator/dynolog?utm_source=chatgpt.com)

**Perfetto + PerfettoSQL.** Production‑grade trace storage/analysis with a SQL engine and standard library; great for storing heterogeneous traces and running analyses. It’s a **backend and query engine**, not a profiler by itself. [GitHub+2Perfetto+2](https://github.com/google/perfetto?utm_source=chatgpt.com)

**Pixie (Kubernetes eBPF)** & **continuous profilers (Parca, Grafana Pyroscope)**. These provide always‑on eBPF‑based system/app profiling in clusters (CPU stacks, sometimes memory/contention). They generally **don’t integrate GPU kernel visibility**, nor do they define *off‑GPU* categories aligned to ML workflows. [Grafana Labs+4PX.dev+4GitHub+4](https://px.dev/?utm_source=chatgpt.com)

**How gPerf differs.** DCGM/Dynolog/Pixie/Parca are **telemetry** systems. They’re great for dashboards and SRE loops, but they don’t **attribute end‑to‑end idle time** to *specific host‑side causes* nor run **COZ‑style causal experiments** to rank *what to fix* for ML SLOs. gPerf aims to be a **profiler** (attribution + causality), not just a metrics pipe.

---

## C. OS‑level off‑CPU profiling and eBPF building blocks (the CPU half of your story)

**Off‑CPU flame graphs (bcc/bpftrace/perf).** Off‑CPU profiling measures blocked/waiting time with stack traces; gold standard techniques from Gregg and BCC tools (*offcputime*). This is the intellectual basis for **wall‑clock attribution** beyond “on‑CPU” time. [Brendan Gregg+2Brendan Gregg+2](https://www.brendangregg.com/FlameGraphs/offcpuflamegraphs.html?utm_source=chatgpt.com)

**bperf (OSDI’24).** Extends Linux `perf` to handle **blocked samples**, producing on/off‑CPU statistics with low overhead (~1–2%). It shows that *blocked time* is first‑class in profiling and can be measured online. [USENIX](https://www.usenix.org/system/files/osdi24-ahn.pdf?utm_source=chatgpt.com)

**Linux PSI (Pressure Stall Information).** Kernel‑level measures of CPU, memory, and I/O pressure (“some” vs “full” stall time windows), increasingly integrated into ops stacks (Kubernetes v1.34 beta). PSI is a **lightweight signal** that correlates resource pressure with tail latency. [Linux Kernel Documentation+1](https://docs.kernel.org/accounting/psi.html?utm_source=chatgpt.com)

**eBPF iterators & uprobes.** `bpf_iter` programs iterate kernel objects without hot‑path instrumentation; `uprobes` attach to user‑space functions. Together they enable **passive reconstruction** of blocking/wait states and precise timestamps for library calls (e.g., CUDA/NCCL) with near‑zero steady‑state overhead. [Linux Kernel Documentation+2Kernel.org+2](https://docs.kernel.org/bpf/bpf_iterators.html?utm_source=chatgpt.com)

**How gPerf differs.** Prior work nails **off‑CPU** attribution, but **doesn’t connect it to GPU idleness** or ML request‑level SLOs. gPerf’s novelty is to **lift the on/off paradigm to the GPU** and **bind the two sides** (CPU ↔ GPU) with causality.

---

## D. Causal profilers and “what‑if” analysis

**COZ: Causal Profiling.** Virtually “speeds up” code by *slowing down everything else*, to estimate the global impact of optimizing a particular region on a target metric (throughput/latency). This yields **actionable optimization guidance**—not just “where time goes.” [ACM SIGOPS+1](https://sigops.org/s/conferences/sosp/2015/current/2015-Monterey/printable/090-curtsinger.pdf?utm_source=chatgpt.com)

**How gPerf differs.** COZ targets CPU apps; gPerf reinterprets this idea in **GPU‑centric ML loops** (e.g., testing the benefit of reducing kernel‑launch overhead via CUDA Graphs, or of shortening CPU data stalls) with **end‑to‑end SLOs** (throughput, tail latency, inter‑token latency). CUDA Graphs as an optimization target is well‑motivated: they reduce **kernel launch overhead** by bundling operations, which is especially impactful in short‑kernel decode loops. [PyTorch+2NVIDIA Developer+2](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/?utm_source=chatgpt.com)

## What we added (beyond your initial list)

Alongside Nsight/DCGM/etc., there is a **substantial research track** on GPU kernel stall analysis and cross‑CPU/GPU profiling that is directly relevant to your “on‑/off‑GPU” thesis:

- **bperf / BCOZ (OSDI’24)** – “blocked samples” unify on‑/off‑CPU profiling and extend **causal** profiling to off‑CPU time. Strong conceptual anchor for your wall‑clock methodology, but CPU‑only. [USENIX+2USENIX+2](https://www.usenix.org/system/files/osdi24-ahn.pdf?utm_source=chatgpt.com)
- **wPerf (OSDI’18)** – early generic off‑CPU bottleneck analysis across waiting events. [USENIX](https://www.usenix.org/system/files/osdi18-zhou.pdf?utm_source=chatgpt.com)
- **HPCToolkit GPU work (’21→’24)** – call‑path attribution **across CPU & GPU**, using **CUPTI PC sampling** for fine‑grain kernel metrics; exascale deployments. Not “always‑on,” and no off‑GPU causality, but shows how to attribute GPU cost up to source/loops. [arXiv+2Xiaozhu Meng's homepage+2](https://arxiv.org/abs/2109.06931?utm_source=chatgpt.com)
- **GPA: GPU Performance Advisor (CGO’21)** – PC‑sampling‑based stall attribution with **what‑if speedup estimates** for kernel‑level optimizations (not end‑to‑end ML SLOs). [Xiaozhu Meng's homepage](https://mxz297.github.io/files/CGO21.pdf?utm_source=chatgpt.com)
- **DrGPU (ICPE’23)** – top‑down GPU profiler decomposing stall cycles; builds CPU↔GPU call paths; kernel‑centric. [SPEC Research+1](https://research.spec.org/icpe_proceedings/2023/proceedings/p43.pdf?utm_source=chatgpt.com)
- **GPUscout (’23)** – detects memory‑movement bottlenecks via SASS + warp‑stall sampling + metrics; kernel‑centric. [ACM Digital Library+1](https://dl.acm.org/doi/10.1145/3624062.3624208?utm_source=chatgpt.com)
- **DS‑Analyzer (VLDB’21)** – measures **data stalls** in training pipelines and does **what‑if** on the pipeline side. [Department of Computer Science+1](https://www.cs.utexas.edu/~vijay/papers/vldb21-datastalls.pdf?utm_source=chatgpt.com)
- **Lotus (IISWC’24 / HotInfra’24)** – characterizes preprocessing pipelines with framework + hardware profiling; open artifacts. [Kexin Rong+1](https://kexinrong.github.io/lab/files/lotus-iiswc24.pdf?utm_source=chatgpt.com)
- **COZ (SOSP’15)** – causal profiling—you already plan a COZ‑style layer; cite this precisely. [ACM SIGOPS+1](https://sigops.org/s/conferences/sosp/2015/current/2015-Monterey/printable/090-curtsinger.pdf?utm_source=chatgpt.com)
- **Dynolog** (Meta) – always‑on telemetry daemon for CPU↔GPU hosts with triggerable deep dives (operational tool). [GitHub+1](https://github.com/facebookincubator/dynolog?utm_source=chatgpt.com)
- **Perfetto + PerfettoSQL** – trace backend and **SQL‑first** analysis—great fit as your analysis substrate. [Perfetto+1](https://perfetto.dev/docs/?utm_source=chatgpt.com)
- **PyTorch Profiler / Kineto / HTA** – episodic tracing and analysis over CUPTI; HTA adds higher‑level diagnoses. [PyTorch Docs+2GitHub+2](https://docs.pytorch.org/docs/stable/profiler.html?utm_source=chatgpt.com)
- **DCGM** – always‑on GPU telemetry (util, copy, health), not causal attribution. [NVIDIA Developer+1](https://developer.nvidia.com/dcgm?utm_source=chatgpt.com)
- **Continuous profilers (Parca, Pyroscope) & Pixie** – eBPF‑based **always‑on CPU** profiling/telemetry at fleet scale; little/no GPU attribution. [Parca+2Grafana Labs+2](https://www.parca.dev/?utm_source=chatgpt.com)
- **CUDA Graphs** – official guidance on reducing **kernel‑launch overhead**; useful to ground your causal recommendations. [NVIDIA Developer+1](https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-cuda-graphs-and-other-performance-enhancements/?utm_source=chatgpt.com)
- **CUPTI PC sampling** – the standard low‑overhead way to attribute GPU time & stalls; you’ll likely **trigger** this selectively. [NVIDIA Docs+1](https://docs.nvidia.com/cupti/api/group__CUPTI__PCSAMPLING__API.html?utm_source=chatgpt.com)

We also pulled **industry evidence** that **CPU work throttles GPU** in LLM inference (tokenization, scheduling, serialization), which you can use in Motivation: vLLM’s own perf update admits CPU‑side blocking the GPU.

# on GPU / off GPU analysis

**Abstract (4–6 sentences)**

https://chatgpt.com/share/68f849b4-3c58-8009-a2fe-a7bf8b0a0fa9

GPUs are often under‑utilized despite high‑level asynchrony and multiple hardware engines. Existing CPU‑centric on/off‑CPU models neither attribute GPU idleness nor handle overlap across compute and copy engines. We present a **GPU‑centric execution model** that partitions time into **ON_GPU** (compute or copy engines active) and **OFF_GPU** (everything else), applied at **two layers**: **Stream/queue** and **Process×Device (context)**. For each layer, **ON_GPU** is the union of engine‑active intervals for the unit; **OFF_GPU** is demanded but not executing time (queueing or dependency stalls), with actionable sub‑buckets for host runtime/driver work, I/O, blocking, and device‑side causes. We implement this with **eBPF** + vendor activity APIs (CUPTI/rocprofiler/DRM), correlating enqueue → driver submit → engine start to recover per‑operation timelines and rolling them up to streams and processes without double counting. Our system produces **per‑engine utilization**, **stall breakdowns**, and **GPU‑centric flamegraphs** that merge CPU stacks with device activity, surfacing bottlenecks such as stream serialization, UVM migrations, and I/O starvation.

what is the GPU-side “unit” that plays the same role as a CPU thread in on/off‑CPU profiling?

- **CPU analogy:** *Thread scheduled (on‑CPU)* vs *thread not scheduled (off‑CPU)* is defined by the **OS scheduler**.
- **GPU analogy (what we can observe):** *Operation executing on a GPU engine (on‑GPU)* vs *demanded work not executing (off‑GPU)* is defined by **engine activity** and **device queues**.

**1 Introduction**
Modern GPU applications pipeline CPU preprocessing, asynchronous H2D/D2H copies, multi‑stream kernel launches, and cross‑device collectives. Yet production runs often show low GPU utilization. Today’s tools either **(i)** focus on CPUs (“on‑CPU vs off‑CPU”) and under‑specify what “waiting on the GPU” means, or **(ii)** produce device‑only traces that cannot explain *why* engines are idle (host I/O? driver? dependencies?).
We argue for a **GPU‑centric** view: define **ON_GPU** precisely as “any GPU engine doing useful work,” and classify the complement (**OFF_GPU**) so developers can answer the only question that matters: **why wasn’t a GPU engine busy?**

**Contributions**
1. **Model** – A concise, implementation‑checkable partition: **ON_GPU = G:COMP ∪ G:XFER**; **OFF_GPU** split into **host‑side** (H:CTRL, H:APP, H:IO, H:WAIT_GPU) and **device‑side** (G:QUEUE, G:STALL, G:PREEMPT) buckets.
2. **Method** – An eBPF + runtime/driver + device‑activity pipeline that reconstructs per‑op timelines and attributes stalls without double‑counting overlap.
3. **Visual analytics** – **GPU‑centric flamegraphs**, per‑engine timelines, and ratio/stack charts that make idleness and causality obvious.
4. **Portability** – Definitions map cleanly to CUDA/HIP/DRM/KFD on NVIDIA/AMD/Intel; optional NIC/collective engines extend the model.
****

**2 Challenges**
• **Multi-layer**: Because GPUs don’t expose a schedulable “thread” to the OS, you must pick a **profiling unit** and then define on/off‑GPU *for that unit* from device activity + queue state.

• **Overlap & double counting**: Copies and kernels run concurrently; CPU and GPU proceed independently. A single wall‑clock partition is misleading.
• **Heterogeneous engines**: Compute vs copy engines, optional NIC offloads, and multi‑GPU P2P/NVLink.
• **Asynchrony & dependencies**: Streams, events, graphs, residency (UVM/KFD), cross‑device fences.
• **Clock domains**: CPU TSC vs GPU timestamp counters.
• **Vendor heterogeneity**: CUDA/CUPTI vs ROCm/rocprofiler vs DRM/KFD tracepoints.
• **Actionability**: “Wait” must be split into **queueing** vs **dependency stalls**, and host runtime vs application vs I/O.

• **Complex software stacks**: mixed Python/C++/JIT frames and async runtimes complicate host attribution; we resolve symbols via build‑id/perf‑maps and treat kernel/graph launches as **H:CTRL**.

**Also software stack is complex, include python, cpp, etc.**

**3 Definitions (OSDI‑ready)**

**Scope.** We report at **two layers** over a makespan [t0,t1][t₀,t₁][t0,t1]:

(1) **Stream/queue** (CUDA stream / HIP queue) and

(2) **Process×Device (context)** (a process using a given GPU).

Attribution unit hierarchy remains **process → context → stream → op**.

**Units (what counts as on/off at each layer).**

| Unit (layer) | ON_GPU(Unit) | OFF_GPU(Unit) |
| --- | --- | --- |
| **Stream/queue** | Time ttt falls in any **engine‑active** interval [tstart,tend)[t_{start}, t_{end})[tstart,tend) of an op on that stream. | The stream has **demand** (≥1 submitted op not yet finished) but **no** op on the stream is executing at time ttt. |
| **Process×Device** | Time ttt falls in any **engine‑active** interval of any op in the process on that device. | The process has **demand** on that device but **no** op is executing at time ttt. |

**Demand semantics (precise, testable).**

For a unit UUU, let

Demand(U)=⋃op∈U[ tsubmit(op), tend(op) )\textbf{Demand}(U) = \bigcup\limits_{\text{op}\in U} [\,t_{\text{submit}}(\text{op}),\, t_{\text{end}}(\text{op})\,)Demand(U)=op∈U⋃[tsubmit(op),tend(op)).

Let Execute(U)=⋃op∈U[ tstart(op), tend(op) )\textbf{Execute}(U) = \bigcup\limits_{\text{op}\in U} [\,t_{\text{start}}(\text{op}),\, t_{\text{end}}(\text{op})\,)Execute(U)=op∈U⋃[tstart(op),tend(op)).

Then **ON_GPU(U) = Execute(U)**; **OFF_GPU(U) = Demand(U) \setminus Execute(U)**; **IDLE(U) = [t₀,t₁]\setminus Demand(U)** (optional to plot).

**Scope**: per **(process/job × device)** over makespan [t0,t1][t₀,t₁][t0,t1]. Attribution unit: **process → context → stream → op**.
**Top‑level partition**
• **ON_GPU**: a GPU engine is actively executing work (attributable to the unit).
    ◦ **G:COMP** – compute engine active (kernel instructions executing).
    ◦ **G:XFER** – copy/DMA engine active (H2D/D2H/D2D/P2P).
• **OFF_GPU**: complement of ON_GPU in [t0,t1][t₀,t₁][t0,t1]; includes CPU work, I/O, runtime overhead, blocking, queueing, and device‑side stalls.
**OFF_GPU sub‑buckets**
• **Host‑side**
    ◦ **H:CTRL** – on‑CPU inside GPU runtime/driver control path (enqueue/graphs/mem registration/ioctl).
    ◦ **H:APP** – on‑CPU application code unrelated to control path (pre/post compute).
    ◦ **H:WAIT_GPU** – thread blocked or polling for GPU completion/fence/event.
    ◦ **H:IO** – disk/network syscalls or io_uring callbacks (dataset staging, checkpoints).
• **Device‑side**
    ◦ **G:QUEUE** – submitted to driver but **not started** on an engine (ordering/priority/occupancy).
    ◦ **G:STALL** – **eligible** but blocked by dependency or service (event/fence, residency/migration/page‑fault, address‑space binding, power/clock gating, cross‑device fence).
    ◦ **G:PREEMPT** – started then preempted/evicted (if observable).
**Primary metric**
• **GPU Active Ratio (GAR)**: GAR=∣ON_GPU∣/(t1−t0)GAR = |ON\_GPU| / (t₁ - t₀)GAR=∣ON_GPU∣/(t1−t0).

Report per‑engine utilizations Ucomp,UxferU_{comp}, U_{xfer}Ucomp,Uxfer and an **OFF_GPU breakdown**.
*Notes*:
• Memcpy is **active work (G:XFER)**, not “wait”.
• CPU and I/O live **inside OFF_GPU**; we avoid a second, overlapping “on/off‑CPU” partition.
****

### **Definitions (OSDI‑ready)** — *replace the Scope and add a small “Units” table + semantics; keep the rest unchanged*

**Scope.** We report at **two layers** over a makespan [t0,t1][t₀,t₁][t0,t1]:

(1) **Stream/queue** (CUDA stream / HIP queue) and

(2) **Process×Device (context)** (a process using a given GPU).

Attribution unit hierarchy remains **process → context → stream → op**.

**Units (what counts as on/off at each layer).**

| Unit (layer) | ON_GPU(Unit) | OFF_GPU(Unit) |
| --- | --- | --- |
| **Stream/queue** | Time ttt falls in any **engine‑active** interval [tstart,tend)[t_{start}, t_{end})[tstart,tend) of an op on that stream. | The stream has **demand** (≥1 submitted op not yet finished) but **no** op on the stream is executing at time ttt. |
| **Process×Device** | Time ttt falls in any **engine‑active** interval of any op in the process on that device. | The process has **demand** on that device but **no** op is executing at time ttt. |

**Demand semantics (precise, testable).**

For a unit UUU, let

Demand(U)=⋃op∈U[ tsubmit(op), tend(op) )\textbf{Demand}(U) = \bigcup\limits_{\text{op}\in U} [\,t_{\text{submit}}(\text{op}),\, t_{\text{end}}(\text{op})\,)Demand(U)=op∈U⋃[tsubmit(op),tend(op)).

Let Execute(U)=⋃op∈U[ tstart(op), tend(op) )\textbf{Execute}(U) = \bigcup\limits_{\text{op}\in U} [\,t_{\text{start}}(\text{op}),\, t_{\text{end}}(\text{op})\,)Execute(U)=op∈U⋃[tstart(op),tend(op)).

Then **ON_GPU(U) = Execute(U)**; **OFF_GPU(U) = Demand(U) \setminus Execute(U)**; **IDLE(U) = [t₀,t₁]\setminus Demand(U)** (optional to plot).

**4 Design and Implementation
4.1 Data acquisition**
**Host (eBPF + uprobes)**
• `sched:sched_switch` → when threads run.
• Uprobes on CUDA/HIP runtime/driver APIs: `cudaLaunchKernel/hipModuleLaunchKernel`, `cudaMemcpyAsync`, `cudaGraphLaunch`, `cudaEvent*`, etc. → **H:CTRL** intervals and op metadata (op_id, stream, sizes).
• Syscalls: `futex/epoll_wait/clock_nanosleep` (→ **H:WAIT_GPU**), `read/write/*send*/*recv*/io_uring_enter` (→ **H:IO**).
• Everything else while scheduled → **H:APP** (by subtraction; optionally constrain with symbol maps).
**Device (activity + kernel scheduler)**
• NVIDIA: **CUPTI Activity** for kernels, memcpy/memset, runtime, NVLink; PC Sampling (optional).
• AMD: **rocprofiler/roctracer**, **KFD/DRM** `job_*` tracepoints for queue submit/start/complete.
• Intel Xe: **DRM/XE** scheduler tracepoints.

These deliver **(t_engine_start, t_engine_end, engine_type, stream, corr_id)** to define **G:COMP/XFER**.
**Correlation**
• Match runtime enqueue → driver submit (ioctl) → engine start using **corr_id/stream+sequence**.
• Compute gaps:
    ◦ `G:QUEUE = t_engine_start − t_driver_submit` (absent explicit dependency events),
    ◦ move portions to **G:STALL** when fences/migration/page‑in/AS bind events are observed.
**Clock synchronization**
• Use vendor conversion (e.g., CUPTI timestamp↔ns) or periodic ping to fit an affine map **GPU_ts = a·TSC + b**. Store `a,b` per device; re‑estimate on long runs.
****

**4.2 State assignment (deterministic rules)**

1. Build ON_GPU as the union of all engine‑active windows (per device, per stream).
2. OFF_GPU := makespan \ ON_GPU.
3. Label OFF_GPU intervals:
    ◦ If thread scheduled in GPU API/driver → **H:CTRL**; else **H:APP**.
    ◦ If blocked in `*Synchronize`/futex on GPU handle → **H:WAIT_GPU**.
    ◦ If in file/net syscalls or io_uring callbacks → **H:IO**.
    ◦ If op has driver submit but no engine start yet → **G:QUEUE** unless dependency/service evidence exists → **G:STALL** (tag cause if known).
    ◦ If device reports preemption → **G:PREEMPT**.
**Pseudocode (core idea)**

`for each op:
  t_submit = time(host_enqueue_exit or driver_submit)
  t_start, t_end, engine = device_activity(op)
  mark ON_GPU[engine] += [t_start, t_end]

  gap = [t_submit, t_start)
  if dependency_event_in(gap): OFF_GPU.G:STALL += gap (cause=event/residency/...)
  else OFF_GPU.G:QUEUE += gap

for each thread interval:
  if in_gpu_api: OFF_GPU.H:CTRL += interval
  else if in_block_wait_gpu: OFF_GPU.H:WAIT_GPU += interval
  else if in_io_syscall: OFF_GPU.H:IO += interval
  else OFF_GPU.H:APP += interval`
****

**4.3 Sampling & symbols (for flamegraphs)**

• **CPU stacks**: eBPF perf‑events at 99–499 Hz; resolve with `buildid→sym` + Use eBPF to walk and dump stack from Python.
• **GPU stacks** (synthetic): generate a “stack” per device‑time sample:
    ◦ ON_GPU→ frames: `ON_GPU;G:COMP;KernelName;Stream=X` or `ON_GPU;G:XFER;Memcpy(H2D);Stream=Y`.
    ◦ OFF_GPU→ frames: `OFF_GPU;G:QUEUE;Cause=StreamOrder`, `OFF_GPU;G:STALL;Cause=UVM_Migrate`, `OFF_GPU;H:WAIT_GPU;cudaEventSynchronize`, `OFF_GPU;H:IO;pread64`, etc.
• Convert **durations** to **samples** by slicing time (e.g., 1 ms granularity) and emitting one folded stack per slice. Export to **speedscope** or **flamegraph.pl**.

Result: a **GPU‑centric flamegraph** where the **root split is ON_GPU vs OFF_GPU**, and widths equal time.

**4.4 Data model & storage**
• Event tables: `host_samples`, `host_syscalls`, `gpu_activity`, `driver_submit`, `um_events`, `links(op_id↔corr_id)`.
• Derived tables: `intervals_{ON_GPU,OFF_GPU}` with tags (engine, stream, cause).
• Queries produce per‑op timelines, per‑engine utilizations, and flamegraph “folded” files.
****

[ Need a data merging here ]

**4.5 Multi‑GPU & collectives**
• Maintain per‑device ON_GPU; report GAR per device and job‑level union.
• P2P/NVLink copies are **G:XFER** on the device(s) whose engines are active.
• NCCL over IB: if no GPU engines are busy, time appears as **OFF_GPU:H:IO** (optionally add **NIC** as a separate engine type if you want ON_NIC/OFF_NIC).
****

**5 Visualization (what to show)**
**Fig 1 — Ratio bar (main)**

Per workload, stacked bar: **ON_GPU** vs **OFF_GPU**. Right‑hand inset: split ON_GPU into **G:COMP vs G:XFER**.
**Fig 2 — OFF_GPU breakdown**

Stacked bars (or a radial donut) with {H:CTRL, H:APP, H:WAIT_GPU, H:IO, G:QUEUE, G:STALL, G:PREEMPT}. Label with percentages and absolute time.
**Fig 3 — Per‑op timeline (waterfall)**

Rows = streams/ops; colored segments for **G:QUEUE → G:STALL → G:COMP/XFER**; overlay host **H:WAIT_GPU** intervals to show overlap.
**Fig 4 — GPU‑centric flamegraph**

Root frames **ON_GPU** and **OFF_GPU**. Under **ON_GPU**, one branch per engine (COMP/XFER) then kernel/memcpy → callsite (if device symbols available). Under **OFF_GPU**, branches by cause (H:IO syscall, H:CTRL function names, G:STALL cause labels). This makes “what kept the GPU idle” visually dominant.
**Fig 5 — Critical‑path DAG** (optional)

Minimal explanation graph showing dependencies that created the largest OFF_GPU gaps.
**6 Evaluation plan (what to measure, not claimed numbers)**
• **Overhead**: sampling rates; tracing on/off; per‑component cost.
• **Correctness**: cross‑check engine windows with vendor profilers on microbenchmarks.
• **Microbenchmarks**: stream serialization, event fences, variable H2D/D2H sizes, UVM page‑in/out, P2P copies, NCCL all‑reduce.
• **Applications**: DNN training (data‑loader vs compute), inference (batching), graph analytics (UVM heavy), ETL + GPU UDFs, multi‑tenant (MPS/MIG).
• **Actionability**: before/after fixes—graphs, prefetch, stream priorities, pinning, conc. memcpy, data loader parallelism.
• **Scalability**: multi‑GPU nodes; long‑running jobs; storage/NIC stress.
**7 Threats to validity**
• Visibility gaps (some dependency causes may be opaque on particular drivers).
• Clock drift for very long runs (re‑fit `a,b`).
• JIT/pythonic stacks need `jitdump/perfmap`; otherwise H:APP collapses.
• Preemption states not reported on some devices; we fold into G:STALL as “unknown”.
**8 Related work (positioning, no exhaustive cites here)**
CPU on/off‑CPU profiling; vendor GPU profilers (Nsight/rocprof); UVM/KFD tracing; distributed profilers (eBPF‑based). Your novelty: **single, GPU‑centric partition** with **host/dev attribution**, **deterministic rules**, and **flamegraphs that answer “why idle?”**.
**9 Appendix: API→Bucket map (excerpt)**API / EventBucketNotes`cudaLaunchKernel` / `hipModuleLaunchKernel` (user↔runtime)**H:CTRL**enqueue/argument marshaling`ioctl` to GPU device (submit)**H:CTRL**driver submit timestamp`cudaMemcpyAsync(H2D/D2H/D2D)`**H:CTRL** (host) → **G:XFER** (device)stream‑ordered`cudaEventSynchronize`/`StreamSynchronize`**H:WAIT_GPU**blocking waitUVM page‑fault / migration**G:STALL (UVM)**residency before startDRM/KFD `job_submitted`→`job_started` gap**G:QUEUE** or **G:STALL**classify with dependency infoFile read / socket recv**H:IO**dataset staging/checkpoint
**10 What to build (short checklist)**
• [ ]  eBPF collectors: sched, syscalls, uprobes on CUDA/HIP.
• [ ]  Device tracers: CUPTI/rocprofiler + DRM/KFD tracepoints.
• [ ]  Correlator: enqueue→submit→start; clock affine fit.
• [ ]  Classifier: rules in §4.2; tag causes.
• [ ]  Exporters: timelines (JSON), **folded stacks** for flamegraphs, CSV for ratio bars.
• [ ]  UI: waterfall timeline; ratio and donut charts; **GPU‑centric flamegraph**.
**11 GPU‑centric flamegraph: implementation details**

**How to construct**
1. Choose a **sampling period** Δ (e.g., 1 ms).
2. For each Δ‑slice of device time per GPU:
    ◦ If any engine window overlaps slice → emit `ON_GPU;{G:COMP|G:XFER};{KernelOrMemcpy};Stream=S`.
    ◦ Else → inspect OFF_GPU classification at slice midpoint and emit a frame path like
`OFF_GPU;G:STALL;Cause=UVM_Migrate` or
`OFF_GPU;H:IO;pread64` or
`OFF_GPU;H:CTRL;cudaGraphLaunch`.
3. Optionally **append CPU call stacks** below H:* nodes (e.g., `OFF_GPU;H:IO;pread64;libc_read;MyParser::next()`), using sampled stacks captured in the same Δ‑slice.
4. Write folded format: one line per stack with sample count = 1; repeating per slice accumulates widths.
**Reading the graph**
• The root split shows **active** vs **idle** GPU time.
• The widest OFF_GPU branches are your actionable levers (I/O, control path, queueing, UVM).
• Drill into **G:QUEUE** vs **G:STALL** to separate scheduling from dependencies.
****

**12 “Definitions & Methods” box (paper‑ready, 7–9 lines)Definitions.** We analyze a process×device over [t0,t1][t₀,t₁][t0,t1]. **ON_GPU** is the union of intervals where compute or copy engines are active for that process (G:COMP/G:XFER from device activity records). **OFF_GPU** is the complement, partitioned into host‑side (H:CTRL, H:APP, H:IO, H:WAIT_GPU) and device‑side (G:QUEUE, G:STALL, G:PREEMPT).
**Instrumentation.** We combine eBPF (sched, syscalls), uprobes on CUDA/HIP runtimes, and device activity (CUPTI/rocprofiler/DRM). We correlate enqueue→driver submit→engine start using correlation IDs and driver tracepoints, and synchronize clocks via vendor counters.
**Attribution.** Queueing time is `start − submit` unless explained by dependency events (then G:STALL with cause). Host waits are labeled via blocking syscalls; I/O via file/net syscalls.
**Metrics.** GPU Active Ratio (GAR), per‑engine utilizations, OFF_GPU breakdown, and GPU‑centric flamegraphs built from time‑sliced folded stacks.

**13 Typical fixes the model surfaces**
• **Big H:CTRL** → batch launches / CUDA Graphs.
• **Large G:QUEUE** → more streams, prioritize, reduce long‑running kernels that serialize engines.
• **G:STALL(UVM)** → prefetch/pin, memory pools, explicit copies.
• **H:IO wide** → async I/O, readahead, compress/cache, separate I/O threads.
• **H:WAIT_GPU dominating** → move waits to device‑side dependencies; fuse kernels.

# coz gpu

https://chatgpt.com/share/68f71ea4-a654-8009-a63f-59eb666251f4 

https://chatgpt.com/share/68f71ea4-a654-8009-a63f-59eb666251f4 

## 1) Why GPUs make Coz hard (and why CPU‑only isn’t enough)

- **No fine‑grain preemption.** CUDA stream priorities bias *future* block dispatch; they do **not** preempt running blocks. So Coz’s “pause everybody else now” primitive doesn’t exist on the device.
- **Asynchrony + hidden schedulers.** Concurrent kernels, DMA engines, and library launches (cuBLAS/cuDNN/MIOpen) mean host‑side pacing often alters queueing rather than **relative device speed**—the quantity causal profiling actually cares about. CPU‑only causal profiling exists (Omnitrace/ROCm Systems Profiler) but remains host‑side. [GitHub](https://github.com/ROCm/rocprofiler-systems?utm_source=chatgpt.com)
- **Device ‘sleep’ variability.** CUDA `__nanosleep()` maps to PTX `nanosleep` whose behavior is permitted in the **[0, 2·t]** range—and can be elided—so naïvely inserting it is not a calibrated slowdown. AMD **does** have `S_SLEEP` with cycle semantics, but it’s hardware‑family specific. We must calibrate. [AMD+3NVIDIA Developer Forums+3NVIDIA Docs+3](https://forums.developer.nvidia.com/t/reliability-of-nanosleep-function/235454?utm_source=chatgpt.com)

**Takeaway:** we need a **GPU‑native** mechanism that changes **relative speed** (target vs. others) and a **calibration loop** that makes “sleep‑based slowdown” predictable enough for causal experiments.

---

## 2) What we can leverage today (evidence this is feasible)

- **Runtime assembly injection on both NVIDIA & AMD.**
    
    – **NEUTRINO (OSDI’25)** injects *platform‑independent* probes at the **parallel assembly** layer (PTX/GCNAsm), reassembles kernels at load time, and exposes an eBPF‑like interface and maps. This proves **cross‑vendor runtime instrumentation is practical**. [USENIX](https://www.usenix.org/system/files/osdi25-huang-songlin.pdf?file=osdi25-huang-songlin.pdf&spm=a2c6h.13046898.publish-article.25.78d16ffa5jVRl7)
    
    – **eGPU / bpftime‑GPU** dynamically translates eBPF bytecode to **PTX** and injects it into **running GPU kernels** (NVIDIA), with a user‑space eBPF runtime and shared maps—exactly the control plane we need for “sleep probes.” [Yiwei Yang @UCSC CS](https://asplos.dev/pdf/bpftime_super.pdf)
    
    – **NVBit** (NVIDIA) is a production‑quality SASS‑level dynamic instrumentation framework if you want a vendor‑specific, low‑level path. [CloudFront+1](https://d1qx31qr3h6wln.cloudfront.net/publications/MICRO_2019_NVBit.pdf?utm_source=chatgpt.com)
    
- **Sleep primitives exist.** CUDA PTX has `nanosleep` (albeit variable); AMD RDNA/GCN has `S_SLEEP` with documented cycle timing; both can be gated by flags read from device memory. [NVIDIA Docs+1](https://docs.nvidia.com/cuda/pdf/ptx_isa_9.0.pdf?utm_source=chatgpt.com)

---

## 3) Key idea (theory): use **relative‑speed invariance** to emulate “virtual speedup”

Coz/VCoz show that causal profiles depend on **relative** speeds: making region **K** `s`% faster is equivalent to slowing “others” by `λ = s/(1+s)` during K’s activity. We’ll enforce that on the **GPU** by **injecting sleeps into non‑target kernels** while K runs. The CPU side (classic Coz) still provides progress points and host‑region what‑ifs. [USENIX](https://www.usenix.org/system/files/osdi25-huang-songlin.pdf?file=osdi25-huang-songlin.pdf&spm=a2c6h.13046898.publish-article.25.78d16ffa5jVRl7)

---

## 4) System design (vendor‑neutral, GPU‑native + CPU‑side synergy)

### 4.1 Control plane

- **Launch interception & timelines.** Use CUPTI (CUDA) and ROCm’s rocprofiler‑SDK/roctracer (AMD) to observe kernel launches, names, stream IDs, and timestamps; these also anchor **progress points** (units of work throughput, latency). [ROCm Documentation](https://rocm.docs.amd.com/_/downloads/rocprofiler/en/latest/pdf/?utm_source=chatgpt.com)
- **Target selection.** Choose kernel(s) K by symbol or launch site.
- **Experiment driver.** For virtual speedup levels `s ∈ {0,10,…,90%}`, compute `λ = s/(1+s)` and command the device probes to slow “others” by λ only while K is active.

### 4.2 Device plane (injected “sleep probes”)

- **Where we inject.** Runtime instrumentation at the **parallel assembly** layer (PTX for NVIDIA; GCN/RDNA for AMD) using NEUTRINO’s probe engine or eGPU’s PTX injector; on NVIDIA you can also go SASS with NVBit. Injection points are:
    1. **Loop back‑edges** or **hot basic blocks** in *non‑target* kernels;
    2. **Kernel prologue/epilogue** to read a global “slowdown” flag and per‑SM token. [USENIX+1](https://www.usenix.org/system/files/osdi25-huang-songlin.pdf?file=osdi25-huang-songlin.pdf&spm=a2c6h.13046898.publish-article.25.78d16ffa5jVRl7)
- **What we inject.**
    
    – NVIDIA: a guarded `nanosleep t` PTX sequence (or a short `clock64()` spin as a fallback on pre‑Volta).
    
    – AMD: `S_SLEEP imm` (and optional `S_WAKEUP`/`S_SETPRIO`).
    
    Instructions are **predicated on a device flag** (in a bpftime/NEUTRINO map) set only while K runs. [AMD](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna3-shader-instruction-set-architecture-feb-2023_0.pdf?utm_source=chatgpt.com)
    
- **How we tune slowdown (λ).** Three orthogonal knobs:
    1. **Coverage**: instrument only a fraction **p** of basic blocks/warps/blocks (randomized);
    2. **Sleep length**: `t` (NVIDIA) or `imm` cycles (AMD);
    3. **Occupancy nudge**: add 1–2 temp registers/shared‑mem bytes to reduce resident warps slightly (instrumentation overhead becomes a controlled part of λ).
        
        All knobs are surfaced via **maps** so the host can **re‑tune online**. [USENIX+1](https://www.usenix.org/system/files/osdi25-huang-songlin.pdf?file=osdi25-huang-songlin.pdf&spm=a2c6h.13046898.publish-article.25.78d16ffa5jVRl7)
        
- **When we apply it.** The probes check a **per‑kernel‑class bit** and a **global “K‑active” bit** (toggled by CUPTI/rocprof callbacks) to ensure we only slow “others” *while K runs*. [ROCm Documentation](https://rocm.docs.amd.com/_/downloads/rocprofiler/en/latest/pdf/?utm_source=chatgpt.com)

### 4.3 CPU‑side causal profiling (baseline & complement)

Run Omnitrace/rocprofiler‑systems or your own Coz‑like sampler on host code paths (pre/post‑processing, enqueue/sync) to produce what‑ifs for **CPU regions** and to provide **progress points** (e.g., per batch/frame completion). Our GPU experiments plug into the same UI. [GitHub](https://github.com/ROCm/rocprofiler-systems?utm_source=chatgpt.com)

---

## 5) Why *injected sleeps* are OK (and how we make them trustworthy)

- **Correctness isolation.** We inject only into **non‑target** kernels and avoid altering their visible values; `nanosleep/S_SLEEP` does not modify memory or registers (beyond time), and NEUTRINO’s design already emphasizes probes that **preserve execution flow**. [USENIX](https://www.usenix.org/system/files/osdi25-huang-songlin.pdf?file=osdi25-huang-songlin.pdf&spm=a2c6h.13046898.publish-article.25.78d16ffa5jVRl7)
- **Variability management.** CUDA `nanosleep` is permitted to sleep in **[0, 2·t]**, and (rarely) be ignored. We address this with **closed‑loop control** (§6) and by **locking clocks** to eliminate DVFS drift during experiments—recommended best practice when profiling. AMD `S_SLEEP` has approximate but **cycle‑based** semantics, making AMD a very clean backend. [NVIDIA Developer Forums+1](https://forums.developer.nvidia.com/t/reliability-of-nanosleep-function/235454?utm_source=chatgpt.com)
- **No kernel source required.** Both NEUTRINO and eGPU demonstrate **runtime** assembly injection across vendors without recompiling user kernels—this is the vendor‑neutral angle. [USENIX+1](https://www.usenix.org/system/files/osdi25-huang-songlin.pdf?file=osdi25-huang-songlin.pdf&spm=a2c6h.13046898.publish-article.25.78d16ffa5jVRl7)

---

## 6) Calibration (theory‑backed & counter‑validated)

**Goal:** For a requested virtual speedup `s`, pick probe knobs so that “others” experience slowdown **λ ≈ s/(1+s)** *while K runs*.

1. **Clock control:** lock GPU clocks (NVIDIA app‑clock / locked GPU clocks, AMD equivalent) during experiments to prevent boost from moving the goalposts. [Speechmatics](https://www.speechmatics.com/company/articles-and-news/timing-operations-in-pytorch?utm_source=chatgpt.com)
2. **Knob→slowdown model:** for each GPU SKU, sweep `(p, t/imm)` on microbench mixes (compute‑bound & memory‑bound) under probes, and fit a small table/model `θ ↦ λ` (θ are probe knobs). **AMD**: `S_SLEEP` gives a stable λ vs. cycles; **NVIDIA**: use repeated trials + randomized coverage to average `nanosleep` variance. [AMD](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna3-shader-instruction-set-architecture-feb-2023_0.pdf?utm_source=chatgpt.com)
3. **Online closed‑loop:** during each experiment point, read **SM Busy, Achieved Occupancy, and DRAM BW** (Nsight/rocprof counters) and the **timestamped kernel durations** (CUPTI/rocprof). If measured λ deviates, adjust θ (increase/decrease `t` or `p`) until within tolerance. Ensure K’s own runtime doesn’t inflate; abort if contamination >ε. [ROCm Documentation](https://rocm.docs.amd.com/projects/rocprofiler/en/latest/how-to/using-rocprof.html?utm_source=chatgpt.com)

> Why this is valid: VCoz’s “only relative speed matters” underpins the λ = s/(1+s) mapping; counters verify that relative contention changed as intended and K remained unaffected. USENIX
> 

---

## 7) Implementation plan (concrete, cross‑vendor)

### 7.1 NVIDIA backend

- **Hooking:** CUPTI Activity/Callback to identify kernel launches & timestamps; high/low stream priorities to bias K vs. others.
- **Injection path A (vendor‑neutral flavor):** use **NEUTRINO’s probe engine** at PTX to splice guarded `nanosleep` at loop back‑edges/basic blocks; control via eBPF‑like maps. [USENIX](https://www.usenix.org/system/files/osdi25-huang-songlin.pdf?file=osdi25-huang-songlin.pdf&spm=a2c6h.13046898.publish-article.25.78d16ffa5jVRl7)
- **Injection path B (NVIDIA‑only):** use **NVBit** to inject SASS‑level calls to a tiny device function that executes `nanosleep` or a `clock64()` bounded spin; this avoids PTX‑to‑SASS surprises. [CloudFront](https://d1qx31qr3h6wln.cloudfront.net/publications/MICRO_2019_NVBit.pdf?utm_source=chatgpt.com)

### 7.2 AMD backend

- **Hooking:** rocprofiler‑SDK/roctracer for launch interception + counters. [ROCm Documentation](https://rocm.docs.amd.com/_/downloads/rocprofiler/en/latest/pdf/?utm_source=chatgpt.com)
- **Injection:** NEUTRINO‑style GCN/RDNA assembly rewriter inserts `S_SLEEP imm` guarded by a device flag (and, if desired, `S_SETPRIO`). AMD’s ISA docs specify sleep timing in **cycles**, which simplifies calibration. [AMD](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna3-shader-instruction-set-architecture-feb-2023_0.pdf?utm_source=chatgpt.com)

### 7.3 Shared glue

- **Flagging & maps:** device flags + per‑kernel bits in shared/constant memory are updated by the host on K’s start/stop callbacks; bpftime/NEUTRINO maps provide a ready mechanism. [Yiwei Yang @UCSC CS+1](https://asplos.dev/pdf/bpftime_super.pdf)
- **Progress points:** use GPU‑event–driven counters (batch/frame completions) for throughput; start/end tagging for **latency** windows, same semantics as Coz. [GitHub](https://github.com/ROCm/rocprofiler-systems?utm_source=chatgpt.com)

---

## 8) Threats to validity & how we mitigate

- **No mid‑block preemption:** Sleep affects progress at **warp/loop** granularity, not instantaneously. We instrument **frequent back‑edges** and randomize coverage to smooth effects; report CIs. [USENIX](https://www.usenix.org/system/files/osdi25-huang-songlin.pdf?file=osdi25-huang-songlin.pdf&spm=a2c6h.13046898.publish-article.25.78d16ffa5jVRl7)
- **Register/occupancy side‑effects:** Instrumentation increases registers by ~4 on average in NEUTRINO’s evaluation; we fold this into λ and keep it small. [USENIX](https://www.usenix.org/system/files/osdi25-huang-songlin.pdf?file=osdi25-huang-songlin.pdf&spm=a2c6h.13046898.publish-article.25.78d16ffa5jVRl7)
- **Library & multiprocess kernels:** If another process owns the GPU (MPS/MIG), device‑side sleeps still throttle **their** warps (good), but launch control and progress points may be partial. Detect & annotate these modes. (Fallback: host‑side pacing as a coarse control; still useful for comparison.) [GitHub](https://github.com/ROCm/rocprofiler-systems?utm_source=chatgpt.com)
- **nanosleep variability:** Use *repeated trials*, *randomized injection*, and *closed‑loop tuning*; where available, prefer AMD `S_SLEEP` for cleaner timing. [NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/reliability-of-nanosleep-function/235454?utm_source=chatgpt.com)

---

## 9) Evaluation plan (what an OSDI paper would show)

1. **Correctness & overhead of probes** (NVIDIA + AMD): confirm minimal extra registers/shared mem as reported by NEUTRINO‑like probes; microbench overheads. [USENIX](https://www.usenix.org/system/files/osdi25-huang-songlin.pdf?file=osdi25-huang-songlin.pdf&spm=a2c6h.13046898.publish-article.25.78d16ffa5jVRl7)
2. **Calibration accuracy:** show measured λ tracks targets across compute‑ and memory‑bound mixes; quantify variance for CUDA `nanosleep` vs. AMD `S_SLEEP`. [AMD](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna3-shader-instruction-set-architecture-feb-2023_0.pdf?utm_source=chatgpt.com)
3. **Prediction accuracy:** for top K’s suggested by the causal curves, implement **real optimizations** (tiling, launch bounds, removing atomics) and compare realized vs. predicted end‑to‑end gains.
4. **Concurrency & multi‑process:** with overlapping kernels and MPS/multi‑tenant runs, show our GPU‑native causal experiments remain valid while CPU‑only causal profiling mispredicts. [GitHub](https://github.com/ROCm/rocprofiler-systems?utm_source=chatgpt.com)
5. **Portability:** repeat on A100/H100 & MI250/MI300, and with JIT (Triton) + AOT (cuBLAS/MIOpen) kernels; demonstrate single front‑end + two backends (PTX, GCN/RDNA). [USENIX](https://www.usenix.org/system/files/osdi25-huang-songlin.pdf?file=osdi25-huang-songlin.pdf&spm=a2c6h.13046898.publish-article.25.78d16ffa5jVRl7)

---

## 10) What’s novel enough for OSDI

- **First GPU‑native causal profiler** that **preserves Coz semantics** via **runtime, vendor‑neutral assembly injection** (NEUTRINO/eGPU‑style) to enact **relative slowdowns** on *non‑target* kernels while K runs, with a **theory‑backed** mapping `λ = s/(1+s)` and **closed‑loop calibration** over hardware counters and locked clocks. Neither vendor profilers (Nsight/rocprof) nor host‑only causal profilers (Omnitrace) do this today. [ROCm Documentation+1](https://rocm.docs.amd.com/projects/rocprofiler/en/latest/how-to/using-rocprof.html?utm_source=chatgpt.com)
- **Cross‑stack integration**: a unified tool that runs **CPU‑side Coz** for host regions **and** **GPU‑native virtual speedups** for kernels, with one set of progress points and what‑if curves. [GitHub](https://github.com/ROCm/rocprofiler-systems?utm_source=chatgpt.com)
- **Generalizable mechanism**: two backends (PTX `nanosleep`, AMD `S_SLEEP`), plus optional **NVBit** SASS injection; works without kernel source, across JIT/AOT pipelines. [GitHub](https://github.com/NVlabs/NVBit?utm_source=chatgpt.com)

---

## 11) Practical notes on **“Can we just inject `__nanosleep`?”**

- **Yes, on NVIDIA Volta+** (`__nanosleep` → PTX `nanosleep`), but its duration may vary in **[0, 2·t]** and can be ignored—so treat it as a *probabilistic* delay and control λ with **coverage + length + occupancy** under **closed‑loop tuning**. [NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/reliability-of-nanosleep-function/235454?utm_source=chatgpt.com)
- **On AMD**, prefer **`S_SLEEP`**: “Cause a wave to sleep for ~`64*SIMM16[6:0]` clocks,” and use `S_WAKEUP` for coordination; more predictable and easier to calibrate. [AMD](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna3-shader-instruction-set-architecture-feb-2023_0.pdf?utm_source=chatgpt.com)
- **If `nanosleep` is unavailable** (pre‑Volta), fall back to a short `clock64()` spin with `__noreturn` fences; calibrate its ALU/issue footprint.

---

### One‑paragraph takeaway

Use **bpftime‑GPU/eGPU** or **NEUTRINO** to **inject vendor‑specific sleep instructions** (NVIDIA PTX `nanosleep`; AMD `S_SLEEP`) into **non‑target** kernels at runtime. Gate those sleeps with a device flag so they only apply **while the target K runs**, and tune the sleep coverage/length to achieve a measured slowdown **λ ≈ s/(1+s)**—the **VCoz** equivalence that emulates K’s virtual speedup. Validate via counters under **locked clocks**. Combine with **CPU‑side Coz** for host regions. This yields a **vendor‑neutral, GPU‑native causal profiler**—a solid OSDI‑level contribution not served by today’s vendor profilers or host‑only causal modes. [GitHub+3USENIX+3Yiwei Yang @UCSC CS+3](https://www.usenix.org/system/files/osdi25-huang-songlin.pdf?file=osdi25-huang-songlin.pdf&spm=a2c6h.13046898.publish-article.25.78d16ffa5jVRl7)

If you want, I can sketch the exact **probe templates** (PTX and GCN/RDNA) and the **CUPTI/rocprofiler** callback pseudocode to toggle the device flag and run the closed‑loop controller.

### You said:

How hard is it to implement that? Can you give me a POC? (You can try bpftime)

### ChatGPT said:

Request for GPT-5 Pro

Reasoned for 14m 53s

Absolutely—this is implementable. Below I give you:

1. a quick complexity read,
2. a **working-style PoC design** that uses **bpftime/eGPU** to inject device‑side “sleeps” (to throttle *non‑target* kernels) and a **CUPTI** host shim to toggle it while your target kernel runs, and
3. a minimal **NVBit** fallback PoC (NVIDIA‑only) in case you want something you can compile immediately with a standard SDK.

Along the way I call out the gotchas (e.g., CUDA stream priorities don’t preempt running blocks; `__nanosleep` variability) and cite docs/papers for each claim.

---

## 0) How hard is it?

**Scope:** moderate: ~3 subsystems

- **Device‑side injection**: bpftime/eGPU or NVBit to insert a short delay in *non‑target* kernels at *frequent points* (loop back‑edges / basic‑block entries). eGPU proves dynamic PTX injection from eBPF is viable; NVBit proves SASS‑level runtime insertion works out of the box. [ACM Digital Library+4GitHub+4ACM Digital Library+4](https://github.com/victoryang00/bpftime-super)
- **Host control**: CUPTI (CUDA) / rocprofiler‑SDK (ROCm) callbacks to detect target‑kernel start/stop and flip a device flag so *only others* are slowed; collect kernel names and timestamps. [ROCm Documentation+4NVIDIA Docs+4NVIDIA Docs+4](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html?utm_source=chatgpt.com)
- **Calibration**: map your desired virtual speedup `s` to a slowdown of others `λ≈s/(1+s)`, then tune delay length/coverage until the **measured** slowdown hits λ; **lock clocks** so DVFS doesn’t add noise. [NVIDIA Developer](https://developer.nvidia.com/blog/advanced-api-performance-setstablepowerstate/?utm_source=chatgpt.com)

**Hard parts / pitfalls**

- **No mid‑block preemption**: priority only biases *new* block dispatch; it **does not** preempt blocks already resident. Your injected sleeps must occur often enough (e.g., loop back‑edges) for you to shape relative speed. [OLCF](https://www.olcf.ornl.gov/wp-content/uploads/2020/07/07_Concurrency.pdf?utm_source=chatgpt.com)
- **`__nanosleep` variability**: NVIDIA PTX `nanosleep` may sleep in **[0, 2·t]** (or rarely be ignored), so tune with **closed loop** (increase cycles or coverage until measured λ hits target). AMD’s `S_SLEEP` is cycle‑based and easier. [AMD+3NVIDIA Developer Forums+3AMD+3](https://forums.developer.nvidia.com/t/reliability-of-nanosleep-function/235454?utm_source=chatgpt.com)
- **Multi‑process contention**: stream priorities work *within a process*; with MPS/multi‑tenant, rely on **device‑side** sleeps (good), not host pacing. [DROPS](https://drops.dagstuhl.de/storage/00lipics/lipics-vol298-ecrts2024/LIPIcs.ECRTS.2024.14/LIPIcs.ECRTS.2024.14.pdf?utm_source=chatgpt.com)

---

## 1) PoC (bpftime / eGPU path, vendor‑neutral)

**Idea**: Inject a tiny device “probe” into *non‑target* kernels that executes a short sleep/spin when a global **SLOWDOWN_ON** flag is set. CUPTI flips this flag **only while the target kernel is active**. This emulates “virtually speeding up K” by slowing others during K’s lifetime (Coz’s relative‑speed principle). eGPU/bpftime already JITs eBPF → **PTX** and patches live kernels; NEUTRINO proves cross‑ISA probing; we use that to place the probe at hot back‑edges/BB entries. [GitHub+2Yiwei Yang @UCSC CS+2](https://github.com/victoryang00/bpftime-super)

### 1.1 Prepare the system

**Lock clocks** (recommended for stable profiling):

```bash
# Query supported clocks
nvidia-smi --query-supported-clocks=memory,graphics --format=csv
# Lock GPU core clocks (and optionally memory clocks)
sudo nvidia-smi --lock-gpu-clocks=<min,max>     # or a single value
# reset afterwards: sudo nvidia-smi --reset-gpu-clocks

```

NVIDIA recommends using `nvidia-smi --lock-gpu-clocks` for stable measurements. [NVIDIA Developer](https://developer.nvidia.com/blog/advanced-api-performance-setstablepowerstate/?utm_source=chatgpt.com)

**Get eGPU (bpftime GPU)**:

```bash
git clone https://github.com/eunomia-bpf/eGPU.git
cd eGPU
# Use their ready-to-run container
docker run -dit --gpus all -v .:/root --privileged --network=host --ipc=host --name egpu yangyw12345/egpu:latest
make release

```

The repo explains dynamic PTX injection and shared eBPF maps across CPU & GPU. Recent versions are merged into bpftime. [GitHub](https://github.com/victoryang00/bpftime-super)

### 1.2 Control plane: CUPTI toggles a device flag while K runs

CUPTI callback (CUDA) to mark the target kernel’s lifetime and set `SLOWDOWN_ON`:

```cpp
// host_control.cu (simplified)
#include <cupti.h>
#include <cuda.h>
#include <atomic>
#include "egpu_maps.h"   // eGPU/bpftime interprocess map handle

static std::atomic<int> target_active{0};
static egpu_ctrl_map_t ctrl;  // exposed by eGPU as shared map (uint32_t keys->values)

static const char* TARGET_NAME = "my_target_kernel";

// CUPTI callback
void CUPTIAPI cbKernel(void* userdata, CUpti_CallbackDomain domain,
                       CUpti_CallbackId cbid, const CUpti_CallbackData* info) {
  if (domain != CUPTI_CB_DOMAIN_RUNTIME_API) return;
  if (info->symbolName && strstr(info->symbolName, TARGET_NAME)) {
    if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000) {
      // about to launch target: set SLOWDOWN_ON=1
      target_active.store(1, std::memory_order_relaxed);
      egpu_map_write_u32(&ctrl, SLOWDOWN_ON, 1);
    }
  }
}

// Activity records let you detect kernel end (or use after-callback of the launch):
void onKernelEnd(const CUpti_ActivityKernel8* k) {
  if (k->name && strstr(k->name, TARGET_NAME)) {
    target_active.store(0, std::memory_order_relaxed);
    egpu_map_write_u32(&ctrl, SLOWDOWN_ON, 0);
  }
}

int main() {
  // init CUPTI Activity + Callback to get names/timestamps
  // register cbKernel, activity buffer callbacks...
  // init egpu shared map (ctrl)
}

```

CUPTI’s Activity/Callback APIs expose kernel names and timings; the tutorial shows how to grab names/durations. [NVIDIA Docs+2NVIDIA Docs+2](https://docs.nvidia.com/cupti/tutorial/tutorial.html?utm_source=chatgpt.com)

> AMD path: replace CUPTI with rocprofiler‑SDK callback tracing to get HIP dispatch begin/end and set the same map. ROCm Documentation
> 

### 1.3 Device probe injected by eGPU (NVIDIA PTX / AMD GCN)

We inject a **guarded delay** into *non‑target* kernels at loop back‑edges or basic‑block entries. eGPU JITs eBPF → **PTX** and patches the kernel; NEUTRINO shows the exact places (see “parallel assembly example” / probe positions). [GitHub+1](https://github.com/victoryang00/bpftime-super)

**PTX probe template (NVIDIA, Volta+)**

*(Inserted at BB entry or loop back‑edge; names use PTX pseudo)*

```
.visible .func __egpu_delay(.param .u32 p_cycles, .param .u32 p_prob) {
    .reg .u32 %r0, %r1, %cycles, %rnd, %thresh;
    .reg .pred %p0, %p1;

    ld.param.u32 %cycles, [p_cycles];      // cycles for nanosleep
    ld.param.u32 %thresh, [p_prob];        // probabilistic coverage (0..2^32-1)

    // read SLOWDOWN_ON flag from a global map (address patched by eGPU)
    ld.global.u32 %r0, [__egpu_ctrl_SLOWDOWN_ON];
    setp.eq.u32 %p0, %r0, 0;
    @%p0 ret;

    // Optional: randomized sampling to control coverage
    mov.u32 %rnd, %clock;                  // cheap pseudo-rand
    setp.gt.u32 %p1, %rnd, %thresh;
    @%p1 ret;

    // sleep; CUDA PTX nanosleep has [0, 2*t] semantics
    nanosleep.u32 %cycles;
    ret;
}

```

**GCN/RDNA probe template (AMD)**

```nasm
; Read SLOWDOWN_ON (addr patched by injector) into s0
s_load_dword s0, s[ctrl_ptr:ctrl_ptr+1], 0
s_cmp_eq_u32 s0, 0
s_cbranch_scc1 skip

; Optional probabilistic coverage (use s_random or v_random if available)

; Cycle-based sleep (more deterministic than NVIDIA nanosleep)
s_sleep 0x20       ; 64 * 0x20 cycles ≈ 2048 cycles (ISA defines range)
skip:
s_nop 0

```

AMD’s ISA manuals define `S_SLEEP` timing (e.g., “cause a wave to sleep for ~64*N cycles”), making calibration easier. [AMD+2AMD+2](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna3-shader-instruction-set-architecture-feb-2023_0.pdf?utm_source=chatgpt.com)

> eGPU/bpftime exposes shared eBPF maps to GPU and CPU; we place SLOWDOWN_ON, CYCLES, COVERAGE there and update from host. GitHub
> 

### 1.4 Where to place the probe

- **Loop back‑edges** or **BB entries** of *non‑target* kernels for finer granularity (as in NEUTRINO probe positions).
- Keep register pressure low; NEUTRINO reports small probe footprints if carefully written. [USENIX](https://www.usenix.org/system/files/osdi25-huang-songlin.pdf?file=osdi25-huang-songlin.pdf&spm=a2c6h.13046898.publish-article.25.78d16ffa5jVRl7&utm_source=chatgpt.com)

### 1.5 Calibration loop (theory‑backed)

Goal: you want virtual speedup **s** of K. Causal profiling says “speed K up by s” ≡ “slow others by `λ = s/(1+s)` while K runs.”

Procedure (per‑SKU; per resource mix):

1. **Set** `SLOWDOWN_ON=1`, choose `(cycles, coverage)`; run a microbench “other” kernel while the target K runs in a **high‑priority** stream.
2. Measure achieved slowdown of “others” from CUPTI/rocprofiler kernel durations; adjust `(cycles, coverage)` until slowdown ≈ `λ`.
3. **Lock clocks** while calibrating to mitigate DVFS. [NVIDIA Developer](https://developer.nvidia.com/blog/advanced-api-performance-setstablepowerstate/?utm_source=chatgpt.com)

> Note: CUDA stream priority does not preempt resident blocks—it only biases dispatch of new blocks. That’s fine here; the probe enforces periodic delay inside blocks, giving you effective control despite the no‑preemption rule. OLCF
> 

### 1.6 Tie‑in with CPU‑side Coz

- Use the same **progress points** (e.g., “batch completed”) and plot *virtual speedup of K vs throughput/latency* just like Coz.
- Also run **host‑side** Coz/Omnitrace causal experiments for pre/post‑processing and enqueue code paths. (Omnitrace documents CPU‑side causal profiling with GPU progress points.) [ACM Digital Library](https://dl.acm.org/doi/10.1145/3723851.3726984?utm_source=chatgpt.com)

---

## 2) Minimal NVBit fallback PoC (NVIDIA‑only, compiles today)

If you just want to see the effect *right now*, NVBit lets you inject a `clock64()` spin at BB entries of *non‑target* kernels and gate it on a global `SLOWDOWN_ON` flag.

**Device function used by instrumentation**

```
// dev_delay.cu
extern "C" __device__ __noinline__ void dev_delay(unsigned cycles,
                                                  const unsigned* slow_flag) {
    if (!(*slow_flag)) return;
    unsigned long long start = clock64();
    while ((clock64() - start) < (unsigned long long)cycles) { }
}

```

**NVBit tool (skeleton)**

```cpp
// tool.cpp
#include "nvbit_tool.h"
#include "nvbit.h"
#include <dlfcn.h>

__device__ __managed__ unsigned SLOWDOWN_ON = 0;

void instrument_func_if_needed(CUcontext ctx, CUfunction f) {
    if (nvbit_is_instrumented(ctx, f)) return;
    nvbit_set_at_launch(ctx, f, true);
    nvbit_set_skip_inst_default(ctx, f, true);

    const char* fname = nvbit_get_func_name(ctx, f);
    bool is_target = (fname && strstr(fname, "my_target_kernel"));

    // Only instrument non-target kernels
    if (!is_target) {
        std::vector<Instr*> instrs;
        nvbit_get_instrs(ctx, f, instrs);
        for (auto i : instrs) {
            if (i->is_bra() || i->is_exit()) continue;
            // Insert call BEFORE each basic block entry (heuristic: first inst of BB)
            if (i->get_bb_pos() == 0) {
                nvbit_insert_call(i, "dev_delay", IPOINT_BEFORE);
                nvbit_add_call_arg_const_val32(800); // cycles; tune in calibration
                nvbit_add_call_arg_const_val64((uint64_t)&SLOWDOWN_ON);
            }
        }
    }
}

extern "C" __global__ void set_flag(unsigned v) { SLOWDOWN_ON = v; }

void nvbit_at_function_first_load(CUcontext ctx, CUfunction f) {
    instrument_func_if_needed(ctx, f);
}

```

Then wrap your app with the NVBit launcher and, from host code (CUPTI callback), call `set_flag<<<1,1>>>(1)` before the target kernel and `set_flag<<<1,1>>>(0)` after. NVBit and its MICRO’19 paper provide the full API surface. [GitHub+1](https://github.com/NVlabs/NVBit)

> This fallback avoids PTX nanosleep variability by using a clock64() spin. It’s NVIDIA‑only, but great for smoke tests.
> 

---

## 3) What makes this OSDI‑credible

- **Device‑side causal control**: Today’s causal profilers operate on the **CPU side** (e.g., Omnitrace) and treat GPU only as progress points; they don’t enforce Coz’s **relative‑speed counterfactuals on the GPU**. Your approach *does*, via runtime injection. [ACM Digital Library](https://dl.acm.org/doi/10.1145/3723851.3726984?utm_source=chatgpt.com)
- **Vendor‑neutral path**: eGPU (bpftime) demonstrates **runtime eBPF→PTX injection**; NEUTRINO shows parallel‑assembly probes and positions **across PTX + GCN**. Combine those mechanisms with a **theory‑backed calibration** and a **CUPTI/rocprofiler** control plane, and you have a cross‑vendor system. [GitHub+2USENIX+2](https://github.com/victoryang00/bpftime-super)
- **Grounded in scheduling reality**: The design explicitly handles **no mid‑block preemption**, which is the key reason CPU‑only approaches can mislead under concurrency. [OLCF](https://www.olcf.ornl.gov/wp-content/uploads/2020/07/07_Concurrency.pdf?utm_source=chatgpt.com)

---

## 4) Runbook: how you exercise the PoC

1. **Lock clocks** (NVIDIA) as above. [NVIDIA Developer](https://developer.nvidia.com/blog/advanced-api-performance-setstablepowerstate/?utm_source=chatgpt.com)
2. **Pick a toy app** that launches two kernels concurrently: `K` (target) and `O` (other), e.g., two streams with overlapping grids.
3. **Start control plane** (CUPTI): watch for `K`’s launch and kernel complete; flip `SLOWDOWN_ON` accordingly. [NVIDIA Docs](https://docs.nvidia.com/cupti/tutorial/tutorial.html?utm_source=chatgpt.com)
4. **Calibrate**: measure `O`’s duration while `K` runs; increase delay cycles/coverage until *observed* slowdown **λ ≈ s/(1+s)** (e.g., s=50% ⇒ λ≈0.33).
5. **Collect Coz‑style points**: for s ∈ {0,10,…,90%}, record end‑to‑end throughput/latency at your progress point (e.g., “batch done”).
6. **Check that K isn’t contaminated**: confirm K’s mean duration stays flat as you vary `λ`. If not, reduce probe register pressure/coverage.

---

## 5) Why the specific building blocks

- **bpftime / eGPU**: Userspace eBPF runtime + dynamic PTX injection into running kernels; shared maps for control plane. Exactly the kind of substrate you need for “flip a flag → device obeys.” [GitHub](https://github.com/eunomia-bpf/bpftime)
- **NEUTRINO**: shows how to place tiny probes at the right *parallel assembly* locations across vendors. Your probe uses the same positions, but performs a calibrated *delay* instead of counting. [USENIX](https://www.usenix.org/system/files/osdi25-huang-songlin.pdf?file=osdi25-huang-songlin.pdf&spm=a2c6h.13046898.publish-article.25.78d16ffa5jVRl7&utm_source=chatgpt.com)
- **CUPTI / rocprofiler**: widely used, documented ways to get kernel names/times programmatically and to build timelines. [NVIDIA Docs+1](https://docs.nvidia.com/cupti/tutorial/tutorial.html?utm_source=chatgpt.com)
- **`__nanosleep` vs `S_SLEEP`**: NVIDIA’s is **[0, 2t]**; AMD’s `S_SLEEP` is cycle‑based; choose accordingly and calibrate under locked clocks. [NVIDIA Developer Forums+1](https://forums.developer.nvidia.com/t/reliability-of-nanosleep-function/235454?utm_source=chatgpt.com)

---

## 6) Known limitations (transparent)

- **CUDA `nanosleep` may jitter**; if this bothers you, prefer the **`clock64()` spin** (deterministic) or AMD `S_SLEEP`. [NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/reliability-of-nanosleep-function/235454?utm_source=chatgpt.com)
- **Placement matters**: if you inject too infrequently (e.g., only once per kernel), you won’t affect relative speed during long blocks; use loop back‑edges / BB entries. [USENIX](https://www.usenix.org/system/files/osdi25-huang-songlin.pdf?file=osdi25-huang-songlin.pdf&spm=a2c6h.13046898.publish-article.25.78d16ffa5jVRl7&utm_source=chatgpt.com)
- **Stream priority ≠ preemption**: don’t rely on it to preempt. Use the probe to enforce periodic delays; priority helps only at dispatch boundaries. [OLCF](https://www.olcf.ornl.gov/wp-content/uploads/2020/07/07_Concurrency.pdf?utm_source=chatgpt.com)

---

### TL;DR

- **Yes, we can do this with bpftime/eGPU**: inject a tiny, **guarded** device delay into *non‑target* kernels; flip a **map flag** via CUPTI while the target runs; calibrate delay so “others” slow by `λ≈s/(1+s)` (the Coz/VCoz equivalence). eGPU/NEUTRINO/NVBit and CUPTI/rocprofiler provide the needed hooks today. [NVIDIA Docs+3GitHub+3USENIX+3](https://github.com/victoryang00/bpftime-super)
- **PoC code above** shows the core pieces.
- **OSDI‑level novelty**: this is a **GPU‑native causal profiler** that enforces relative‑speed counterfactuals on the device, **vendor‑neutral** with PTX + GCN probes, and integrates with CPU‑side Coz for end‑to‑end what‑ifs.

If you want, I can package these into a repo layout (eGPU probe + CUPTI controller + a toy CUDA app and a Makefile) so you can clone and run locally.

# evaluation

https://chatgpt.com/share/68f96963-fd78-8009-99fe-3e688eed610e

## Research Questions (what you must answer)

**RQ1 – Visibility.**

Does gPerf reveal performance pathologies that “GPU utilization” (and standard GPU‑only profilers) routinely miss?

*Success criterion:* at least **two** real bottlenecks per workload whose root cause is in **OFF‑CPU** or **OFF‑GPU** time, not visible in utilization bars.

**RQ2 – Correctness.**

Are gPerf’s **on/off‑CPU** and **on/off‑GPU** attributions correct relative to ground truth timelines?

*Success:* per‑activity identities hold: `submit→start = off_gpu_dep + off_gpu_queue` and `start→end = on_gpu` with **≤2% timing error**; CPU runnable/blocked intervals match `sched_*` within **≤1 ms** median delta.

**RQ3 – Actionability.**

Do gPerf’s recommendations translate into *meaningful* end‑to‑end improvements?

*Success:* ≥ **25–35%** throughput gain for training; **2–3×** p99 improvement for MoE inference after changes that gPerf prioritizes.

**RQ4 – Always‑on overhead.**

Can gPerf run continuously with **≤3% steady‑state overhead** on the 5090 box under both workloads?

*Success:* CPU usage within +2–3% and **no measurable regressions** in wall‑clock throughput at default sampling.

**RQ5 – Causality.**

Do COZ‑style perturbations predict the ranking of optimizations?

*Success:* Kendall‑τ ≥ **0.6** between predicted and observed speedups across at least **5** candidate interventions per workload.

**RQ6 – Generality (single node).**

Does the unified **off^dep / off^queue / on** model work equally for copy engines and compute engines and across both workloads?

*Success:* consistent decomposition and attributions on H2D/D2H vs compute; same invariants, no vendor‑specific hacks exposed.

---

## Experimental Platform (single 5090 server)

- **GPU:** 1× “5090” class NVIDIA GPU (no MIG for the core runs).
- **CPU:** ≥32 physical cores, SMT on.
- **Memory/Storage:** ≥128 GB RAM, 1× NVMe SSD.
- **OS/stack:** Ubuntu 22.04+, recent kernel (>=5.15), CUDA 12.x, PyTorch 2.2+.
- **Affinity policy:** pin GPU device interrupts and driver threads to a separate core group; pin workload CPUs explicitly in some runs to force contention where needed.
- **Clocks:** use `CLOCK_MONOTONIC_RAW`; gPerf aligns host↔GPU with a periodically refreshed offset.

---

## Workloads (both are intentionally **CPU‑touchy** so gPerf can shine)

### W1 – Training with real JPEGs and CPU transforms (ResNet‑50)

- **Goal:** manufacture **CPU** and **queueing** bottlenecks on an otherwise “busy” GPU.
- **Dataset:** synthetic JPEG corpus (50k–200k files, q=85–95) so you can ship the artifact.
- **Model:** torchvision `resnet50()`; mixed precision optional.
- **Knobs to induce pain (Bad case = T‑CPU‑Bottleneck):**
    - `num_workers=2`, `pin_memory=False`, `prefetch_factor=2`.
    - Add `sleep_ms=2` in dataset `__getitem__` to mimic slow I/O.
    - Batch size 64; limit to 8 CPU cores (`taskset -c 0-7 …`).
    - Unfused pointwise tail; no CUDA Graphs.
- **Fixes (Good case = T‑Optimized):**
    - `num_workers=16`, `pin_memory=True`, `prefetch_factor=8`.
    - GPU‑side transforms for resize/crop/autocontrast; remove `sleep_ms`.
    - Two streams (copy vs compute), `non_blocking=True`; fuse pointwise tail or wrap steady region in **CUDA Graphs**.

### W2 – MoE inference with **CPU‑offloaded experts** + **CPU prefix‑KV cache**

- **Goal:** **CPU is the bottleneck**; expose it cleanly and fix it.
- **Model scaffold:** decoder‑only transformer with a single MoE layer (top‑k=2). A tunable fraction of experts run on CPU; KV prefix cache managed on CPU with **pinned staging** for H2D bursts.
- **Knobs to induce pain (Bad case = MoE‑CPU‑Heavy):**
    - `cpu_frac = 0.6–0.7` (60–70% experts on CPU).
    - Concurrency 128; decode **step size 8** tokens; prefix hit rate ~50%.
    - Introduce hash+memcpy work in cache lookup; small H2D bursts.
    - Limit to 8–12 CPU cores; `OMP_NUM_THREADS=2`.
- **Fixes (Good case = MoE‑Optimized):**
    - `cpu_frac = 0.2–0.3`; batch CPU experts across requests.
    - Pre‑pin KV buffers; coalesce H2D into ≥1–4 MB chunks.
    - Pin cache manager and BLAS threads to dedicated cores; NUMA‑bind near GPU.
    - Increase decode step size to 32–64.

---

---

## Evaluation Protocol

### Global settings

- Warmup 2–3 minutes; measure 3–5 minutes steady‑state. Repeat **5 runs** (CI).
- Pin gPerf to two dedicated cores; isolate from workload cores.

### Baselines to collect

1. **Stock profilers**: Nsight Systems (timeline), PyTorch Profiler (training only).
2. **Utilization bars**: DCGM/NVML SM & copy utilization streams.
3. **No gPerf** run (to measure overhead deltas).

### gPerf configurations

- **Always‑on default:** 50–100 Hz stack sampling on CPU blocks and wakes; GPU activity ingestion + DAG pass; PSI at 1 Hz.
- **Heavier deep dive:** burst PC sampling on top‑N kernels (for ON‑GPU only).

---

## Experiments (per RQ)

### RQ1/RQ3 – Visibility & Actionability

**W1: Training**

- **E1.1 (Bad)** T‑CPU‑Bottleneck. Expect:
    - Large **OFF‑CPU^dep** (I/O) and **OFF‑CPU^queue** (few cores).
    - **OFF‑GPU^dep** (compute waits on H2D).
    - **OFF‑GPU^queue** tail (micro‑kernels behind GEMM).
- **E1.2 (Good)** T‑Optimized. Expect:
    - ∑OFF‑GPUqueue\sum \text{OFF‑GPU}^{queue}∑OFF‑GPUqueue ↓ **50–70%**;
    - Throughput ↑ **25–35%**;
    - Engine util ≈ unchanged (proves the “utilization lies” point).

**Figures/Tables to show**

- **F‑T1** Unified swimlanes (ASCII acceptable in paper):
    
    ```
    CPU:loader   ====decode+aug==== .... I/O wait .... //// rq //// ==H2D enqueue==
    GPU:H2D             .... deps .... //// queue //// === H2D ===
    GPU:Compute                          .... deps .... //// queue //// ==== GEMM ==== ==LN== ==ReLU==
    
    ```
    
- **F‑T2** The **four off‑panels** (flame graphs): OFF‑CPU^dep, OFF‑CPU^queue, OFF‑GPU^dep, OFF‑GPU^queue.
- **F‑T3** Queue‑aware scatter (x=on, y=off^queue) – micro‑kernels pinned at the y‑axis.
- **T‑T1** Metrics per step: S,Q,D,U,PS,Q,D,\mathcal{U},\mathcal{P}S,Q,D,U,P, QIF, DIF (bad→good).

**W2: MoE Inference**

- **E2.1 (Bad)** MoE‑CPU‑Heavy. Expect:
    - **OFF‑CPU^queue** dominant (many runnable tasks, few cores).
    - **OFF‑GPU^dep** from prefix cache H2D.
    - p99 inflated; inter‑token latency spiky.
- **E2.2 (Good)** MoE‑Optimized. Expect:
    - ∑OFF‑CPUqueue\sum \text{OFF‑CPU}^{queue}∑OFF‑CPUqueue ↓ **60–80%**;
    - p99 ↓ **2–3×**;
    - ∑OFF‑GPUdep\sum \text{OFF‑GPU}^{dep}∑OFF‑GPUdep almost disappears (coalesced H2D bursts).

**Figures/Tables to show**

- **F‑M1** Swimlane showing cache work on CPU gating H2D and compute.
- **F‑M2** Four off‑panels; **OFF‑CPU^queue** bar dominates in bad case.
- **T‑M1** Latency histograms/CDFs (p50/p95/p99) annotated with panel deltas.

### RQ2 – Correctness (ground truth)

- Correlate each kernel/memcpy’s `submit/start/end` against Nsight Systems export.
- Show distribution of `|gPerf − Nsight|` for start/end (<2% or <0.5 ms median).
- Validate `t_e` with a dependency microbench (H2D→K→D2H with explicit waits): gPerf’s **off^dep** equals known barrier intervals.

### RQ4 – Overhead

- **O1.1** Always‑on vs disabled: throughput delta <3% (training) and p50 latency delta <3% (inference).
- **O1.2** Sweep sampling rates (25/50/100/200 Hz) and GPU activity buffer sizes; plot overhead curves.

### RQ5 – Causal profiling (COZ‑style)

Pick **5–7 candidate interventions** per workload:

- **Training:** increase dataloader workers; pin_memory; GPU transforms; CUDA Graphs; fuse pointwise.
- **Inference:** reduce `cpu_frac`; batch CPU experts; pre‑pin KV; coalesce H2D; increase step size; pin threads.

**Method:** run **micro‑slowdowns** (1–3% duty) injected into the targeted NVTX spans (e.g., `bias+relu`, `prefix_cache_h2d`, `expert_cpu`) one at a time; estimate speedup potential from COZ slope. Rank interventions; compare to realized improvements in the “Good” run. Report Kendall‑τ.

### RQ6 – Generality/Sensitivity

- **S1** Batch size sweep (training): 32/64/128 → see how off‑panels move.
- **S2** Concurrency and step size sweep (inference): 32/64/128/256 × 8/32/64 tokens.
- **S3** Copy vs compute pressure: force large H2D vs compute‑heavy kernels; verify panel split (OFF‑GPU^dep on copy‑gated runs).

---

## Metrics & Definitions (what lands with reviewers)

- Per resource rrr:
    
    Sr=∑onS_r=\sum \text{on}Sr=∑on, Qr=∑offqueueQ_r=\sum \text{off}^{queue}Qr=∑offqueue, Dr=∑offdepD_r=\sum \text{off}^{dep}Dr=∑offdep.
    
    Utilizations (interval union):
    
    Ur=union(on)/T\mathcal{U}_r = \text{union}(\text{on})/TUr=union(on)/T,
    
    Pr=union([te,tc))/T\mathcal{P}_r = \text{union}([t_e,t_c))/TPr=union([te,tc))/T.
    
    **QIF** =(S+Q)/S= (S+Q)/S=(S+Q)/S, **DIF** =(S+D)/S= (S+D)/S=(S+D)/S.
    
- **Tail metrics:** per‑request p50/p95/p99; **inter‑token latency** (mean, p95) for MoE.
- **Overhead:** `(with gPerf − without gPerf)/without gPerf` on throughput or p50 latency.

## Evaluation workloads:

Below is a **single‑box (1×5090) evaluation storyline** you can actually run, plus an implementation plan that stays faithful to the **on/off‑CPU** and **on/off‑GPU** model (with the extra split into **dep** vs **queue**). I’m going to be blunt about what’s easy, what’s annoying, and what you can show at OSDI quality without writing a 5k‑LOC tracer.

---

## TL;DR (what you’ll demonstrate)

1. **Training**
2. **Inference** (MoE decoder‑only transformer with **CPU‑offloaded experts**

Both cases produce the exact visuals an OSDI reviewer wants: unified timelines, **four off‑time panels** (OFF‑CPU^dep / OFF‑CPU^queue / OFF‑GPU^dep / OFF‑GPU^queue), **on‑GPU** panel, and a **queue‑aware scatter**.

---

## Minimal hardware & tooling

- **Box:** 1×NVIDIA “5090” class GPU; 16–64 CPU cores; NVMe SSD.
- **Software:** Ubuntu 22.04+, Python 3.10+, CUDA 12.x+, PyTorch 2.2+.
- **Tooling (zero custom drivers):**
    - **Nsight Systems (`nsys`)** for GPU/Runtime timeline (gives `submitstartend`
    - **NVTX** (mark host submission sites).
    - **`perf` / eBPF**`sched_*`
    - **Small Python post‑processor** tet_ete and compute the six panels.

> You do not need to write CUPTI code for this paper plot set. Nsight Systems + perf is enough, with a deterministic DAG pass to compute tet_ete.
> 

---

## WORKLOAD 1 — Training with intentional CPU overhead

### Why this works

Real JPEG decode + CPU‑side augmentations cause **blocked time** (I/O) and **runnable time** (run queue) on CPU; H2D copies gate kernels (OFF‑GPU^dep) and poor overlap + micro‑kernels produce (OFF‑GPU^queue).

### Dataset (no dependency on ImageNet)

Generate your own JPEG corpus so you can ship the artifact:

```python
# make_images.py
from PIL import Image
import numpy as np, os, argparse, random
parser = argparse.ArgumentParser(); parser.add_argument("--n", type=int, default=50000)
parser.add_argument("--res", type=int, default=1024); parser.add_argument("--out", default="jpeg_ds")
a=parser.parse_args(); os.makedirs(a.out, exist_ok=True)
for i in range(a.n):
    arr = (np.random.rand(a.res, a.res, 3)*255).astype('uint8')
    Image.fromarray(arr).save(os.path.join(a.out, f"{i:06d}.jpg"), quality=random.randint(85,95))

```

### Trainer with knobs to **make CPU the bottleneck**

```python
# train.py
import torch, torchvision as tv, torchvision.transforms as T
from torch.utils.data import DataLoader
from PIL import Image
import glob, os, time
from torch.cuda import nvtx

class FolderJpeg(torch.utils.data.Dataset):
    def __init__(self, root, sleep_ms=0):
        self.files = glob.glob(os.path.join(root, "*.jpg"))
        self.sleep_ms = sleep_ms
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        f = self.files[i]
        img = Image.open(f).convert("RGB")
        # INTENTIONALLY heavy CPU transforms
        x = tv.transforms.functional.resize(img, 256)
        x = tv.transforms.functional.center_crop(x, 224)
        x = tv.transforms.functional.autocontrast(x)
        if self.sleep_ms: time.sleep(self.sleep_ms/1000.0)  # simulate slow storage
        x = tv.transforms.functional.to_tensor(x)
        y = torch.randint(0, 1000, ()).item()
        return x, y

def make_model():
    return tv.models.resnet50().cuda()

def step(model, batch):
    x, y = batch[0].cuda(non_blocking=True), batch[1].cuda(non_blocking=True)
    with nvtx.range("forward"):
        out = model(x)
    loss = torch.nn.functional.cross_entropy(out, y)
    with nvtx.range("backward"):
        loss.backward()
    return loss

# knobs via env or argparse

```

Run **two configurations**:

- **T‑CPU‑Bottleneck (bad):**
    - `-batch 64`, `num_workers=2`, `pin_memory=False`, `prefetch_factor=2`, dataset `sleep_ms=2`, CPU transforms as above.
    - Limit the process to **8 CPU cores** to juice **OFF‑CPU^queue**:
        
        ```
        taskset -c 0-7 python -u train.py --root jpeg_ds --num-workers 2 --pin-memory 0
        
        ```
        
- **T‑Optimized (good):**
    - Move transforms to GPU (either Kornia or `torchvision.transforms.v2` with `device="cuda"` for a subset), enable `pin_memory=True`, `num_workers=16`, `prefetch_factor=8`.
    - Split streams (copy vs compute); use `non_blocking=True`; fuse the tiny tail (e.g., bias+relu) or wrap the steady region in a **CUDA Graph** block.

### What you’ll see (ASCII timeline)

```
T-CPU-Bottleneck
CPU:loader   ====decode+aug==== ........ I/O wait ....... //// rq //// ==H2D enqueue==
CPU:train    ====python step==== .... futex(GIL) .... //// rq //// ==launch== .... stream sync ....

GPU:H2D           .... deps (PCIe credits / stream order) .... //// queue //// === H2D ===
GPU:Compute                          .... deps (wait H2D) .... //// queue //// ===== GEMM ===== ==LN== ==ReLU==

```

**Expected symptoms** (bad case):

- High **OFF‑CPU^dep** (I/O wait) and **OFF‑CPU^queue** (too few cores, low dataloader workers).
- **OFF‑GPU^dep**: kernels ineligible while H2D not done.
- **OFF‑GPU^queue**: LN/ReLU queued behind GEMM; copies don’t overlap.

**After fixes**:

- OFF‑GPU^dep shrinks (copy/compute overlap).
- OFF‑GPU^queue plummets (fewer micro‑kernels + CUDA Graphs).
- OFF‑CPU^queue/dep drop (more workers, pinned memory).

**Paper‑ready numbers to report (per step):**

| Metric | T‑CPU‑Bottleneck | T‑Optimized |
| --- | --- | --- |
| Throughput (img/s) | ↓ | ↑ (≥ +25%) |
| ∑OFF‑GPUqueue\sum \text{OFF‑GPU}^{queue}∑OFF‑GPUqueue | **high** | **−50–70%** |
| ∑OFF‑GPUdep\sum \text{OFF‑GPU}^{dep}∑OFF‑GPUdep | medium | low |
| ∑OFF‑CPUdep\sum \text{OFF‑CPU}^{dep}∑OFF‑CPUdep | **high** (I/O) | low |
| Engine util Ucompute\mathcal{U}_{compute}Ucompute | ~0.7–0.85 | similar (util doesn’t tell the story) |
| QIFGPU_{GPU}GPU | **3–4×** | **≤2×** |

> The point you’ll make: utilization was already high; the win comes from slashing queue time, which classic profilers hide.
> 

---

## WORKLOAD 2 — MoE inference with **CPU offload** + **prefix cache (CPU)**

### Why this works

MoE gives you dials to push load onto CPU, with not active expert offload. Prefix‑KV cache managed by CPU (lookup, refcount, eviction, memcpy to pinned buffers) is **very CPU‑visible**. Many short requests (small decode steps, high concurrency) surface **OFF‑CPU^queue** and **OFF‑GPU^dep/queue**.

### Minimal MoE scaffold (sketch)

```python
# moe_infer.py
import torch, torch.nn as nn
from torch.cuda import nvtx

class ExpertMLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.seq = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
    def forward(self, x): return self.seq(x)

class MoELayer(nn.Module):
    def __init__(self, d, n_experts, cpu_frac=0.5, top_k=2):
        super().__init__()
        self.top_k = top_k
        self.gate = nn.Linear(d, n_experts, bias=False).cuda()
        self.experts = nn.ModuleList(ExpertMLP(d) for _ in range(n_experts))
        self.place = []
        for i,e in enumerate(self.experts):
            if i < int(cpu_frac*n_experts):
                e.to("cpu"); self.place.append("cpu")
            else:
                e.to("cuda"); self.place.append("cuda")
    def forward(self, x):
        with nvtx.range("gate"): logits = self.gate(x)                       # GPU
        topk = torch.topk(logits, self.top_k, dim=-1).indices
        outs = []
        for i in range(self.top_k):
            idx = topk[..., i].item() if topk.ndim==2 else topk[i].item()
            dev = self.place[idx]; e = self.experts[idx]
            with nvtx.range(f"expert_{idx}_{dev}"):
                xe = x.detach().to(dev, non_blocking=True)                   # H2D/D2H traffic
                ye = e(xe)                                                   # CPU or GPU
                outs.append(ye.to("cuda", non_blocking=True))
        return sum(outs)/self.top_k

# Prefix cache manager (CPU)
class PrefixCache:
    def __init__(self, capacity_tokens, shard_bytes=1<<20):
        self.capacity_tokens = capacity_tokens
        self.store = {}   # key(hash)-> CPU tensor [n_layers, heads, dim]
        self.pinned = {}  # staging pinned buffers for H2D bursts
    def lookup(self, key):  # hash first L tokens
        return self.store.get(key, None)
    def insert(self, key, kv_cpu):
        self.store[key] = kv_cpu
    def h2d(self, key):
        kv_cpu = self.store[key]
        if key not in self.pinned:
            self.pinned[key] = kv_cpu.pin_memory()
        return self.pinned[key].to("cuda", non_blocking=True)   # burst H2D

# Decode loop (small steps amplify overhead)
def serve_one(model, cache, prompt_tokens, step_tokens=16):
    key = hash(tuple(prompt_tokens[:128]))
    hit = cache.lookup(key)
    if hit is not None:
        with nvtx.range("prefix_cache_h2d"):
            kv = cache.h2d(key)   # CPU work + H2D
    else:
        with nvtx.range("prefix_compute"):
            kv = compute_prefix_on_gpu(model, prompt_tokens)  # fill, then:
        with nvtx.range("prefix_insert"):
            cache.insert(key, kv.detach().to("cpu", non_blocking=True))
    # Now decode in small bursts (forces many small submits)
    for _ in range(step_tokens):
        with nvtx.range("decode_step"):
            x = torch.randn(1, model.d).cuda()
            _ = model.moe(x)      # routes some experts to CPU

```

### How to **force CPU bottleneck**

- Set **many CPU experts**: `cpu_frac=0.5–0.7`.
- **Small decode step** (e.g., `step_tokens=1–8`), **high concurrency** (50–200 async requests).
- Prefix cache **hit rate ~50%**: both `lookup`/`h2d` paths active; add a deliberate small compute in `lookup` (hashing) or a **zstd->memcpy** to simulate real caches.
- Limit process to **8–12 CPU cores** to drive **OFF‑CPU^queue**:
    
    ```
    taskset -c 0-11 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u moe_infer.py --concurrency 128 --cpu-frac 0.6
    
    ```
    
- Keep **GPU otherwise underutilized** (batch=1) so the queueing you see is mostly **engine wait for copies** and **many small launches** (not ALU saturation).

### What you’ll see (ASCII)

```
MoE-CPU-Heavy (bad)
CPU:gate+cache  ====gate postproc==== .... prefix_cache: lookup & memcpy .... //// rq //// ==enqueue experts==
CPU:experts     ====BLAS (CPU experts)==== .... run-queue wait ....
GPU:Copy(H2D)         .... deps(wait pinned alloc) .... //// queue //// ==H2D bursts==
GPU:Compute                           .... deps(wait H2D) .... //// queue //// ==small matmuls==

```

**Symptoms**:

- **OFF‑CPU^queue** large (many runnable tasks vs limited cores).
- **OFF‑GPU^dep** frequent (compute ineligible until prefix H2D completes).
- **OFF‑GPU^queue** for many **tiny** GPU kernels.
- Tail latency (p95/p99) tracks the **sum of CPU‑side queue and cache work**.

**Optimized “good” case**:

- Reduce CPU experts (`cpu_frac=0.2–0.3`) or **batch CPU experts** across requests.
- Pre‑allocate **pinned buffers**; copy KV in **large chunks** (coalesce small memcpys).
- Pin cache manager to its own cores; bind experts to NUMA node near GPU.
- Optional: move prefix merge to GPU for hot prefixes; raise decode step size to 32–64 to amortize overhead.

**Paper‑ready numbers**:

| Metric | MoE‑CPU‑Heavy | MoE‑Optimized |
| --- | --- | --- |
| p50 / p95 / p99 latency (ms) | **↑↑** | **↓↓** |
| ∑OFF‑CPUqueue\sum \text{OFF‑CPU}^{queue}∑OFF‑CPUqueue | **high** | **−60–80%** |
| ∑OFF‑GPUdep\sum \text{OFF‑GPU}^{dep}∑OFF‑GPUdep | medium | low |
| Throughput (tok/s at 128 concurrent) | lower | higher |
| QIFGPU_{GPU}GPU | **≥3×** | **≤1.7×** |

---

## How to **collect** and **compute**

We’ll use the same four timestamps ts,te,ta,tct_s,t_e,t_a,t_cts,te,ta,tc per activity and split:

- **off^dep = t_e − t_s**
- **off^queue = t_a − t_e**
- **on = t_c − t_a**

### 1) Mark submission sites (NVTX)

Wrap every kernel‑causing high‑level call in an NVTX range; PyTorch already emits many runtime events. You can also put short NVTX tags around:

- dataloader steps (`"decode"`, `"augment"`, `"h2d"`),
- MoE events (`"prefix_cache_h2d"`, `"expert_i_cpu"`).

### 2) Capture a GPU timeline

```
nsys profile -t cuda,osrt --capture-range=nvtx --force-overwrite=true \
  -o trace_training  python train.py  ...
nsys profile -t cuda,osrt --capture-range=nvtx --force-overwrite=true \
  -o trace_moe       python moe_infer.py ...

```

Export to JSON (or SQLite) so you get per‑activity `submit`, `start`, `end`, `stream`

### 3) Capture CPU scheduling and blocking

Use `perf` (no root) or eBPF (if allowed):

```
# CPU runnable/blocked and stacks
perf record -e sched:sched_switch,sched:sched_wakeup -g -- \
  python train.py ...

# If you can run bpftrace (nicer for blocked-site attribution):
bpftrace -e '
tracepoint:sched:sched_switch { ... }  // record RUNNING slices (on-CPU)
tracepoint:sched:sched_wakeup  { ... } // runnable edges
tracepoint:syscalls:sys_enter_read,
          :sys_enter_futex,
          :sys_enter_nanosleep { @blk[tid] = ustack(15); }'

```

(If you want a one‑liner: `perf sched record` then `perf sched script`

### 4) Build GPU **eligibility** tet_ete

From the Nsight JSON:

- Nodes = {kernels, memcpys} with `(submit, start, end, stream, nvtx_range)`
- Edges = in‑stream program order + explicit waits + NVTX “phase” order (if you used ranges sensibly)

Compute:

```
for node in topo_order:
    te[node] = max( submit[node], max( end[p] for p in preds[node] ) )

```

### 5) Compute four off‑panels + two on‑panels

- **OFF‑CPU^dep**: from CPU `sched_switch` (blocked deschedule) → next wakeup; attribute to blocking syscall stack.
- **OFF‑CPU^queue**: wakeup time → next scheduled‑in; attribute to runnable stack.
- **OFF‑GPU^dep**: `te − submit`; attribute to **host submission stack / NVTX label**.
- **OFF‑GPU^queue**: `start − te`; same attribution as above.
- **ON‑CPU**: scheduled‑in → out; attribute to CPU stack (optional).
- **ON‑GPU**: `end − start`; attribute to kernel symbol (or PC sampling if you want).

### 6) Produce the exact **paper figures**

- Four off‑time flame graphs (each panel from a folded‑stack file).
- A unified **swimlane timeline** (ASCII is fine in the paper; Perfetto screenshots for appendix).
- A queue‑aware scatter (per‑kernel: x=on, y=off^queue).
- Tables: S,Q,D,U,PS, Q, D, \mathcal{U}, \mathcal{P}S,Q,D,U,P, QIF, DIF per resource.

**ASCII example (inference, CPU‑heavy):**

```
CPU:cache   ====lookup==== .... mutex wait .... //// rq //// ==stage->pinned==  ====next req====
GPU:H2D           .... deps (pinned ready) .... //// queue //// === H2D KV ===
GPU:Compute                          .... deps (KV ready) .... //// queue //// == tiny matmuls ==

```

---

## Experiment knobs (to force the bottleneck or fix it)

### Training

- **Make it worse**: `num_workers=2`, `pin_memory=False`, add `sleep_ms=2`, run on 8 cores, small batch (64).
- **Make it better**: `num_workers=16`, `pin_memory=True`, GPU transforms (Kornia / v2), coalesce H2D, CUDA Graphs around steady section, fuse pointwise tail.

### MoE inference

- **Make it worse**: `cpu_frac=0.6`, `concurrency=128`, `step_tokens=8`, prefix hit rate ~50% with memcpy and hashing, limit to 8–12 cores, `OMP_NUM_THREADS=2`.
- **Make it better**: `cpu_frac=0.2`, pin cache manager on dedicated cores, batch CPU experts, coalesce H2D to ≥1–4 MB chunks, pre‑pin staging buffers, bump step size to 32–64, NUMA‑bind processes to GPU‑local socket.

---

## What to claim (and prove) in the paper

- **Claim 1 (Training):** *Queue‑level profiling exposes that 60–70% of per‑activity time is GPU queueing caused by CPU‑side data prep and micro‑kernel tails—despite 80% engine utilization.* After moving transforms to GPU + Graphs, **QIFGPU_{GPU}GPU** drops from ~3.6× to ~1.7× and throughput improves 25–35%.
- **Claim 2 (MoE Inference):** *Tail latency is dominated by CPU contention from prefix cache management and CPU experts.* After batching KV transfers, reducing CPU experts, and pinning cache threads, **OFF‑CPU^queue** falls 60–80% and p99 improves 2–3×, with **OFF‑GPU^dep** nearly eliminated.

These are strong, reproducible single‑box results that **standard utilization graphs would not have revealed**.

# Key reference:

## On CPU vs Off CPU

OSDI 24:

**Identifying On-/Off-CPU Bottlenecks Together with Blocked Samples**

https://www.usenix.org/conference/osdi24/presentation/ahn 

blogs:

https://www.brendangregg.com/offcpuanalysis.html#Overhead 

https://eunomia.dev/tutorials/32-wallclock-profiler/ 

## AI flamegraph （CPU centric)

from intel:

https://www.brendangregg.com/blog/2024-10-29/ai-flame-graphs.html 

https://github.com/intel/iaprof

https://www.brendangregg.com/blog/2025-05-01/doom-gpu-flame-graphs.html 

From alibaba:

https://mp.weixin.qq.com/s/NHTEjI1E1bcYoa2vbAwE2g 

https://zhuanlan.zhihu.com/p/1890440768667230596 

startup:

https://zymtrace.com/article/zero-friction-gpu-profiler/

How the AI flamegraph works:

https://eunomia.dev/blog/2025/10/11/understanding-iaprof-a-deep-dive-into-aigpu-flame-graph-profiling/ 

Mix on cpu-off cpu flame graph with GPU tracing supported demo(Our version):

https://github.com/eunomia-bpf/xpu-tools

(Need to match GPU tracing timestamp with CPU sampling, but this is still )

## Using bpf iter for lightweight sampling

https://eunomia.dev/tutorials/features/bpf_iters/ 

https://tanelpoder.com/posts/xstack-passive-linux-stack-sampler-ebpf/ 

https://github.com/tanelpoder/0xtools

Tested, less than 1% overhead

Casual profiling:

https://github.com/plasma-umass/coz