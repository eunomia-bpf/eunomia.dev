---
date: 2025-10-14
---
 
# The GPU Observability Gap: Why We Need eBPF on GPU devices

> Yusheng Zheng=, Tong Yu=, Yiwei Yang=

As a revolutionary technology that provides programmability in the kernel, eBPF has achieved tremendous success in CPU observability, networking, and security. However, for the increasingly important field of GPU computing, we also need a flexible and efficient means of observation. Currently, most GPU performance analysis tools are limited to observing from the CPU side through drivers/user-space APIs or vendor-specific performance analysis interfaces (like CUPTI), making it difficult to gain deep insights into the internal execution of the GPU. To address this, bpftime provides GPU support through its CUDA/SYCL attachment implementation, enabling eBPF programs to execute **within GPU kernels** on NVIDIA and AMD GPUs. This brings eBPF's programmability, observability, and customization capabilities to GPU computing workloads. By doing so, it enables real-time profiling, debugging, and runtime extension of GPU applications without source code modification, addressing a gap in the current observability landscape.

> **Note:** GPU support is still experimental. For questions or suggestions, [open an issue](https://github.com/eunomia-bpf/bpftime/issues) or [contact us](mailto:team@eunomia.dev).

<!-- more -->

## The Problem: GPU Observability Challenges

### The Core Challenge: An Opaque Execution Model

GPUs have become the dominant accelerators for machine learning, scientific computing, and high-performance computing workloads, but their SIMT (Single Instruction, Multiple Thread) execution model introduces significant observability and extensibility challenges. Modern GPUs organize thousands of threads into warps (typically 32 threads) that execute in lockstep on streaming multiprocessors (SMs), with kernels launched asynchronously from the host. These threads navigate complex multi-level memory hierarchies ranging from fast but limited per-thread registers, to shared memory/LDS within thread blocks, through L1/L2 caches, to slower but abundant device memory, while contending with limited preemption capabilities that make kernel execution difficult to interrupt, inspect, or extend. This architectural complexity creates rich performance characteristics including warp divergence, memory coalescing patterns, bank conflicts, and occupancy variations that directly impact throughput, yet these behaviors remain largely opaque to traditional observability tools. Understanding and optimizing issues like kernel stalls, memory bottlenecks, inefficient synchronization, or suboptimal SM utilization requires fine-grained visibility into the execution flow, memory access patterns, and inter-warp coordination happening deep inside the GPU, along with the ability to dynamically inject custom logic. These capabilities are what existing tooling struggles to provide in a flexible, programmable manner.

Existing GPU tracing and profiling tools fall into two categories, each with significant limitations. First, many tracing tools operate exclusively at the CPU-GPU boundary by intercepting CUDA/SYCL userspace library calls (e.g., via LD_PRELOAD hooks on libcuda.so) or instrumenting kernel drivers at the system call layer. While this approach captures host-side events like kernel launches, memory transfers, and API timing, it fundamentally treats the GPU as a black box, providing no visibility into what happens during kernel execution, no correlation with specific warp behaviors or memory stalls, and no ability to adaptively modify behavior based on runtime conditions inside the device. Second, GPU vendor-specific profilers (NVIDIA's CUPTI, Nsight Compute, Intel's GTPin, AMD's ROCProfiler, research tools like NVBit or Neutrino) do provide device-side instrumentation and can collect hardware performance counters, warp-level traces, or instruction-level metrics. However, these tools operate in isolated ecosystems disconnected from Linux kernel observability and extension stacks. They cannot correlate GPU events with CPU-side eBPF probes (kprobes, uprobes, tracepoints), require separate data collection pipelines and analysis workflows, some of them may impose substantial overhead (10-100x slowdowns for fine-grained instrumentation), and lack the dynamic programmability and control from the control plane that makes eBPF powerful for customizing what data to collect and how to process it in production environments, without recompilation or service interruption.


### The Timeline Visibility Gap: What Can and Cannot Be Observed

To illustrate this challenge, consider a common debugging scenario. "My CUDA application takes 500ms to complete, but I don't know where the time is spent. Is it memory transfers, kernel execution, or API overhead?" The answer depends critically on whether the application uses synchronous or asynchronous CUDA APIs, and reveals fundamental limitations in CPU-side observability.

In synchronous mode, CUDA API calls block until the GPU completes each operation, creating a tight coupling between CPU and GPU timelines. Consider a typical workflow of allocate device memory, transfer data to GPU (host-to-device), execute kernel, and wait for completion. CPU-side profilers can measure the wall-clock time of each blocking API call, providing useful high-level insights. For example, if `cudaMemcpy()` takes 200μs while `cudaDeviceSynchronize()` (which waits for kernel completion) takes only 115μs, a developer can quickly identify that data transfer dominates over computation, suggesting a PCIe bottleneck that might be addressed by using pinned memory, larger batch sizes, or asynchronous transfers. However, when the developer asks "The kernel sync takes 115μs, but why is my kernel slow? Is it launch overhead, memory stalls, warp divergence, or low SM utilization?", CPU-side tools hit a fundamental wall. The 115μs sync time is an opaque aggregate that conflates multiple hidden GPU-side phases including kernel launch overhead (~5μs to schedule work on SMs), actual kernel execution (~100μs of computation on streaming multiprocessors), and cleanup (~10μs to drain pipelines and release resources). Even with perfect timing of synchronous API calls, CPU-side tools cannot distinguish whether poor kernel performance stems from excessive launch overhead, compute inefficiency, memory access patterns causing stalls, or SM underutilization. These require visibility into warp-level execution, memory transaction statistics, and per-thread behavior that is only accessible from inside the GPU during kernel execution. We need a finer-grained, programmable, GPU-internal perspective to understand what is happening during kernel execution, not just when it starts and ends.

Modern CUDA applications increasingly use asynchronous APIs (`cudaMemcpyAsync()`, `cudaLaunchKernel()` with streams) to maximize hardware utilization by overlapping CPU work with GPU execution. This introduces temporal decoupling where API calls return immediately after enqueuing work to a stream, allowing the CPU to continue executing while the GPU processes operations sequentially in the background. This breaks the observability that CPU-side tools had in synchronous mode.

Consider the same workflow now executed asynchronously. The developer enqueues a host-to-device transfer (200μs), kernel execution (100μs), and device-to-host transfer (150μs), then continues CPU work before eventually calling `cudaStreamSynchronize()` to wait for all GPU operations to complete. From the CPU's perspective, all enqueue operations return in microseconds, and only the final sync point blocks, reporting a total of 456μs (1 + 200 + 5 + 100 + 150 μs of sequential GPU work).

```
CPU Timeline (what traditional tools see):
─────────────────────────────────────────────────────────────────────────────────
cudaMallocAsync() cudaMemcpyAsync() cudaLaunchKernel() cudaMemcpyAsync()      cudaStreamSync()
●─────●─────●─────●─────────────────────────────────────────────────────────────────●────
1μs  1μs  1μs  1μs         CPU continues doing other work...               456μs wait
(alloc)(H→D)(kernel)(D→H)                                                       (all done)

GPU Timeline (actual execution - sequential in stream):
─────────────────────────────────────────────────────────────────────────────────
◄─Alloc─►◄───────H→D DMA────────►◄─Launch─►◄────Kernel Exec────►◄────D→H DMA────►
│ ~1μs  │       200μs            │   5μs   │       100μs        │      150μs     │
┴────────┴────────────────────────┴─────────┴────────────────────┴────────────────┴─────
         ↑                                  ↑                                     ↑
    CPU already moved on              GPU still working                    Sync returns
```

In synchronous execution, measuring individual API call durations allowed developers to identify whether transfers or compute dominated. In asynchronous mode, this capability disappears entirely as all timing information is collapsed into a single 456μs aggregate at the sync point. The question "Is my bottleneck memory transfer or kernel execution?" becomes unanswerable from the CPU side. If the first transfer takes twice as long due to unpinned memory (400μs instead of 200μs), delaying all subsequent operations by 200μs, the developer only sees the total time increase from 456μs to 656μs with zero indication of which operation caused the delay, when it occurred, or whether it propagated to downstream operations.

Asynchronous execution not only hides the GPU-internal details that were already invisible in synchronous mode (warp divergence, memory stalls, SM utilization), but also eliminates the coarse-grained phase timing that CPU tools could previously provide. Developers lose the ability to even perform basic triage. They cannot determine whether to focus optimization efforts on memory transfers, kernel logic, or API usage patterns without either (1) reverting to slow synchronous execution for debugging (defeating the purpose of async), or (2) adding manual instrumentation that requires recompilation and provides only static measurement points.

Modern GPU applications like LLM serving further complicate this picture with advanced optimization techniques. Batching strategies combine multiple operations to maximize throughput like a pipeline, but make it harder to identify which individual operations are slow. Persistent kernels stay resident on the GPU processing multiple work batches, eliminating launch overhead but obscuring phase boundaries. Multi-stream execution with complex dependencies between streams creates intricate execution graphs where operations from different streams interleave unpredictably. Shared memory usage per thread block constrains occupancy and limits concurrent warp execution, creating subtle resource contention that varies based on kernel configuration. These optimizations significantly improve throughput but make the already-opaque async execution model even more difficult to observe and debug from the CPU side.

However, even if we had perfect GPU-internal visibility showing high memory stall cycles, we still couldn't determine the root cause: Is it warp divergence causing uncoalesced access? Host thread descheduling delaying async memory copies? PCIe congestion from concurrent RDMA operations? Or on-demand page migration latency from the kernel? Similarly, when observing SM underutilization, device-only metrics cannot distinguish whether the grid is too small, launches are serialized by a userspace mutex, or the driver throttled after an ECC error or power event.

The challenge becomes acute in production environments where tail latency spikes occur intermittently. Are they caused by GPU cache effects, cgroup throttling on the host producer thread, or interference from another container issuing large DMA transfers? Device-only tooling reports "what happened on the GPU," but not "why it happened now in this heterogeneous system." Without time-aligned correlation to userspace behavior (threads, syscalls, allocations), kernel events (page faults, scheduler context switches, block I/O), and driver decisions (stream dependency resolution, memory registration, power state transitions), engineers must iterate slowly: observe a GPU symptom → guess a host cause → rebuild with ad-hoc instrumentation → repeat. This feedback loop is expensive and often infeasible in latency-sensitive deployments. To address these challenges in production environments, the industry has started to adopt Continuous Profiling, which aims to provide "always-on" monitoring for online services without significant performance impact. However, existing GPU continuous profiling solutions often rely on sampling high-level metrics (like GPU utilization or power consumption), which cannot explain "why" a kernel is slow, or they depend on periodically running heavyweight vendor tools, which introduce unacceptable performance overhead. These solutions fail to strike a balance between low overhead, high fidelity, and fine-grained visibility, nor do they solve the fundamental problem of cross-layer correlation.

> **The key insight:** Effective GPU observability and extensibility requires a unified solution that spans multiple layers of the heterogeneous computing stack: from userspace applications making CUDA API calls, through OS kernel drivers managing device resources, down to device code executing on GPU hardware. Traditional tools are fragmented across these layers, providing isolated visibility at the CPU-GPU boundary or within GPU kernels alone, but lacking the cross-layer correlation needed to understand how decisions and events at one level impact performance and behavior at another.

#### Limitations of Existing Tools

Existing GPU tracing and profiling tools can be broadly categorized into three types, each with inherent limitations that prevent them from offering a complete, unified solution.

#### 1. CPU-GPU Boundary-Only Tracing Tools

Many tracing tools operate exclusively at the CPU-GPU boundary by intercepting CUDA/SYCL userspace library calls (e.g., via `LD_PRELOAD` hooks on `libcuda.so`) or instrumenting kernel drivers at the system call layer.

- **Advantages**: Can capture host-side events like kernel launches, memory transfers, and API timing.
- **Limitations**: This approach fundamentally treats the GPU as a "black box." It provides no visibility into what happens during kernel execution, cannot correlate performance issues with specific warp behaviors or memory stalls, and cannot adaptively modify behavior based on runtime conditions inside the device.

#### 2. Vendor-Specific, Heavyweight Profilers: The Case of Nsight

NVIDIA's Nsight suite attempts to address some cross-domain visibility challenges. **Nsight Systems** provides system-wide timelines showing CPU threads, CUDA API calls, kernel launches, and memory transfers in a unified view. **Nsight Compute** offers deep kernel-level microarchitectural metrics via CUPTI counters and replay-based analysis, with guided optimization rules. These tools can correlate launches with CPU thread scheduling and provide rich per-kernel stall reasons and memory metrics.

However, Nsight and similar vendor tools suffer from fundamental limitations compared to a unified eBPF approach.
- **Closed Event Model**: Nsight offers a closed event model with a fixed set of events and no arbitrary programmable logic at attach points—you cannot dynamically add custom instrumentation without recompiling applications or write filtering predicates like "only collect data when kernel execution exceeds 100ms."
- **Intrusive Profiling Sessions**: These tools require special profiling sessions that perturb workload behavior through counter multiplexing and replay mechanisms, making them unsuitable for always-on continuous telemetry in production, and causing replay-based collection to miss transient anomalies and rare events.
- **Lack of On-Device Filtering and Aggregation**: Nsight lacks in-situ filtering and aggregation, forcing all raw data to be exported then post-processed, which creates multi-GB traces from large applications with massive async pipelines and no programmable adaptive response to change sampling logic on-the-fly based on observed state.
- **Limited System Integration**: Nsight cannot attach dynamic probes to persistent kernels without restart, lacks integration with Linux eBPF infrastructure (kprobes/uprobes/tracepoints), and cannot share data structures (maps) across CPU and GPU instrumentation, making it extremely difficult to stitch causality chains like page fault (host) → delayed launch enqueue → warp stall spike.
- **Vendor Lock-in**: These are NVIDIA-only tools with no clear path to vendor-neutral deployment across AMD, Intel, or other accelerators in heterogeneous systems.

In practice, developers face iterative root cause analysis slowdowns with large traces, miss production issues that don't reproduce under profiling overhead, and cannot correlate GPU events with existing production observability stacks (perf, bpftrace, custom eBPF agents) without complex mode-switching to special "profiling sessions."

#### 3. Research Tools and Interfaces for Fine-Grained Analysis

When deeper visibility than Nsight is required, the industry has explored tools based on binary instrumentation or lower-level interfaces, such as NVIDIA CUPTI, NVBit, and NEUTRINO.

- **CUPTI (CUDA Profiling Tools Interface)**: As a mature interface, CUPTI is well-suited for obtaining kernel-level high-level metrics (like start/end times) and hardware performance counters. Projects like [xpu-perf](https://github.com/eunomia-bpf/xpu-perf) have demonstrated its effectiveness in correlating CPU-GPU data. However, when it comes to understanding "why" a kernel is slow, the high-level metrics provided by CUPTI are often insufficient.

- **Binary Instrumentation Tools (NVBit, NEUTRINO)**: These tools achieve fine-grained, instruction-level observability by instrumenting at the assembly or PTX (NVIDIA GPU's intermediate language) level. For instance, NEUTRINO, which emerged around the same time, uses assembly-level probes to gather data. However, it typically requires programming directly in assembly, which is not only complex but also lacks the safety and portability offered by eBPF. They are also often independent of CPU profilers, making it difficult to provide unified, cross-layer, multi-device visibility, correlate event logic, and handle clock drift, which can be very challenging. The process of correlating events may also involve multiple data copies, leading to additional performance overhead. Neutrino is not designed for always-on monitoring; it is also session-based, generating large amounts of information that await subsequent processing.

In summary, while existing tools are powerful in certain aspects, they are either too high-level or too cumbersome and isolated. Developers face a difficult trade-off between iteration speed, production safety, and the depth of problem diagnosis, and they cannot easily correlate GPU events with existing CPU observability stacks (like perf, bpftrace).

## The Solution: Extending eBPF to the GPU

To overcome the limitations of existing tools, we need a solution that can unify CPU and GPU observability while providing programmable, low-overhead monitoring. The eBPF technology and its implementation in the `bpftime` project make this possible.

### Why eBPF?

To understand why eBPF is the right tool for this challenge, it's helpful to look at its impact on the CPU world. eBPF (extended Berkeley Packet Filter) is a revolutionary technology in the Linux kernel that allows sandboxed programs to be dynamically loaded to extend kernel capabilities safely. On the CPU side, eBPF has become a cornerstone of modern observability, networking, and security due to its unique combination of programmability, safety, and performance. It enables developers to attach custom logic to thousands of hook points, collecting deep, customized telemetry with minimal overhead. The core idea behind `bpftime` is to bring this same transformative power to the traditionally opaque world of GPU computing.

By running eBPF programs natively inside GPU kernels, `bpftime` provides safe, programmable, unified observability and extensibility across the entire stack.
- **Unified Cross-Layer Observability**: The architecture treats CPU and GPU probes as peers in a unified control plane. Shared BPF maps and ring buffers enable direct data exchange, and dynamic instrumentation works without recompilation or restart. Integration with existing eBPF infrastructure (perf, bpftrace, custom agents) requires no mode-switching. Developers can simultaneously trace CPU-side CUDA API calls via uprobes, kernel driver interactions via kprobes, and GPU-side kernel execution via CUDA probes, all using the same eBPF toolchain and correlating events across the host-device boundary. Example questions now become answerable: "Did the CPU syscall delay at T+50μs cause the GPU kernel to stall at T+150μs?" or "Which CPU threads are launching the kernels that exhibit high warp divergence?" This cross-layer visibility enables root-cause analysis that spans the entire heterogeneous execution stack, from userspace application logic through kernel drivers to GPU hardware behavior, without leaving the production observability workflow.
- **Low-Overhead Production Monitoring**: Unlike session-based profilers, it enables always-on production monitoring with dynamic load/unload of probes and device-side predicate filtering to reduce overhead.
- **Restoring Asynchronous Visibility**: It recovers async-mode visibility with per-phase timestamps (H→D at T+200μs, kernel at T+206μs, D→H at T+456μs), exposes GPU-internal details with nanosecond-granularity telemetry for warp execution and memory patterns, and correlates CPU and GPU events without the heavyweight overhead of traditional separate profilers.

### `bpftime`: Running eBPF Natively on the GPU

**bpftime's approach** bridges this gap by extending eBPF's programmability and customization model directly into GPU execution contexts, enabling eBPF programs to run natively inside GPU kernels alongside application workloads. It employs a PTX injection technique, dynamically injecting eBPF programs into the intermediate assembly language (PTX) of NVIDIA GPUs, allowing for direct hooking of GPU threads.

This approach allows us to obtain extremely fine-grained, in-kernel runtime information that is difficult to access with high-level APIs like CUPTI. For example, with `bpftime`, we can:

-   **Trace memory access patterns of individual threads or thread blocks**: Understand exactly how memory access instructions are executed and identify issues like uncoalesced accesses.
-   **Observe the scheduling behavior of Streaming Multiprocessors (SMs)**: See how warps are scheduled and executed on the SMs.
-   **Analyze in-kernel control flow**: Identify the specific branches causing warp divergence and quantify their impact.

`bpftime`'s PTX injection is not intended to replace CUPTI but to complement its capabilities. When developers need to dive into the micro-level of GPU kernel execution to diagnose complex issues arising from thread behavior, memory access, or scheduling policies, `bpftime` fills the gap in fine-grained, programmable observability in the current toolchain. Compared to other binary instrumentation tools, `bpftime`'s eBPF-based solution offers several unique advantages:

1. **Enhanced Security**: The eBPF verifier provides a safe sandbox environment, preventing the execution of unsafe instructions.
2. **Convenient Cross-Layer Correlation**: eBPF on the GPU is essentially an extension of eBPF on the CPU, facilitating unified instrumentation and programming across CPU user-space, kernel-space, and GPU. This makes it easier to correlate events across the CPU, GPU, and network card, as well as between user-space and kernel-space.
3. **Simplified Programming Model**: eBPF uses a restricted C language for programming, abstracting away the complexities of low-level assembly and allowing developers to write powerful observation and analysis programs more efficiently.

| Tool Type | Representative Tools | Advantages | Limitations | Use Case |
|---|---|---|---|---|
| **Vendor Profilers** | NVIDIA Nsight, AMD ROCProfiler | Comprehensive hardware metrics, strong visualization | Heavyweight, requires special environment, cross-platform difficulty | Deep optimization during development |
| **High-Level APIs** | CUPTI, ROCTracer | Stable interface, supports tracing and sampling | Coarse granularity, cannot observe inside kernels | Application-level performance monitoring |
| **Binary Instrumentation**| NVBit, SASSI, NEUTRINO | Fine-grained observation, instruction-level control | High performance overhead (3-10×), limited programming | Offline deep analysis |
| **eBPF Extension** | bpftime | Unified CPU/GPU observation, low overhead, safe & programmable, cross-platform | Requires runtime support, features still evolving | Production real-time monitoring, cross-layer correlation analysis |

#### `bpftime`'s Architecture and Advantages

The system defines a comprehensive set of GPU-side attach points that mirror the flexibility of CPU-side kprobes/uprobes. Developers can instrument CUDA/SYCL device function entry and exit points (analogous to function probes), thread block lifecycle events (block begin/end), synchronization primitives (barriers, atomics), memory operations (loads, stores, transfers), and stream/event operations. eBPF programs written in restricted C are compiled through LLVM into device-native bytecode (PTX (Parallel Thread Execution) assembly for NVIDIA GPUs or SPIR-V for AMD/Intel) and dynamically injected into target kernels at runtime through binary instrumentation, without requiring source code modification or recompilation.

The runtime provides a full eBPF execution environment on the GPU including (1) a safety verifier to ensure bounded execution and memory safety in the SIMT context, (2) a rich set of GPU-aware helper functions for accessing thread/block/grid context, timing, synchronization, and formatted output, (3) specialized BPF map types that live in GPU memory for high-throughput per-thread data collection (GPU array maps) and event streaming (GPU ringbuf maps), and (4) a host-GPU communication protocol using shared memory and spinlocks for safely calling host-side helpers when needed.

This architecture enables not only collecting fine-grained telemetry (per-warp timing, memory access patterns, control flow divergence) at nanosecond granularity, but also adaptively modifying kernel behavior based on runtime conditions, building custom extensions and optimizations, and unifying GPU observability with existing CPU-side eBPF programs into a single analysis pipeline, all while maintaining production-ready overhead characteristics. This enables:

- **3-10x faster performance** than tools like NVBit for instrumentation
- **Vendor-neutral design** that works across NVIDIA, AMD and Intel GPUs
- **Unified observability and control** with Linux kernel eBPF programs (kprobes, uprobes)
- **Fine-grained profiling and runtime customization** at the warp or instruction level
- **Adaptive GPU kernel memory optimization** and programmable scheduling across SMs
- **Dynamic extensions** for GPU workloads without recompilation
- **Accelerated eBPF applications** by leveraging GPU compute power

The architecture is designed to achieve four core goals: (1) provide a unified eBPF-based interface that works seamlessly across userspace, kernel, multiple CPU and GPU contexts from different vendors, (2) enable dynamic, runtime instrumentation without requiring source code modification or recompilation, and (3) maintain safe and efficient execution within the constraints of GPU hardware and SIMT execution models. (4) Less dependency and easy to deploy, built on top of existing CUDA/SYCL/OpenGL runtimes without requiring custom kernel drivers, firmware modifications, or heavy-weight runtimes like record-and-replay systems.

But talking about architecture is one thing; seeing it in action is another. Just how fast is it? We'll dive deep into the performance benchmarks in our next blog post. Stay tuned!

## Architecture

### CUDA Attachment Pipeline

The GPU support is built as an instrumentation pipeline that dynamically injects eBPF programs into GPU kernels. The key components are:

1.  **CUDA/OpenCL Runtime Hooking**: Using `LD_PRELOAD`, `bpftime` intercepts calls to the CUDA/SYCL runtime library. This allows it to gain control over kernel launches and other GPU-related operations.
2.  **eBPF to PTX/SPIR-V JIT Compilation**: When a kernel is launched, `bpftime` takes the eBPF bytecode intended for a GPU probe and Just-In-Time (JIT) compiles it into the target GPU's instruction set architecture—PTX for NVIDIA or SPIR-V for AMD/Intel.
3.  **Binary Instrumentation and Injection**: The compiled eBPF code is injected into the target kernel's binary (e.g., PTX code) before it is loaded onto the GPU. This runtime modification allows eBPF programs to execute natively within the kernel's context.
4.  **Helper Function Trampoline**: `bpftime` provides a set of eBPF helper functions accessible from the GPU. These helpers are implemented as a trampoline to perform tasks like accessing maps, getting timestamps, or submitting data through ring buffers.
5.  **Shared Data Structures**: BPF maps and ring buffers are implemented over shared memory (pinned host memory) or device memory accessible by both the CPU and GPU, enabling efficient data exchange between the host and the device.

```
┌─────────────────────────────────────────────────────────────────┐
│                      Application Process                        │
│              (LD_PRELOAD)              JIT compile              │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     │
│  │     PTX/     │───▶│  bpftime     │────▶│  GPU Kernel  │     │
│  │    SPIR-V    │     │  runtime     │     │  with eBPF   │     │
│  └──────────────┘     └──────────────┘     └──────────────┘     │
│                     manage   │                      │           │
│                              ▼                      ▼           │
│                       ┌────────────────────────────────────┐    │
│                       │ Shared Maps                        │    │
│                       │  (Host-GPU)                        │    │
│                       └────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## Examples

We demonstrate GPU eBPF capabilities through several examples, bcc style tools:

### [kernelretsnoop](https://github.com/eunomia-bpf/bpftime/tree/master/example/gpu/kernelretsnoop) - Per-Thread Exit Timestamp Tracer

Attaches to CUDA kernel exits and records the exact nanosecond timestamp when each GPU thread completes execution. This reveals thread divergence, memory access patterns, and warp scheduling issues that are invisible to traditional profilers.

> **Note**: The `kprobe`/`kretprobe` naming convention used in these examples is a placeholder to maintain conceptual similarity with Linux kernel eBPF. In `bpftime`, these probes attach to device-side GPU kernel functions, not Linux kernel functions. This naming may be revised in the future to better reflect their scope.

**Use case**: You notice your kernel is slower than expected. `kernelretsnoop` reveals that thread 31 in each warp finishes 750ns later than threads 0-30, exposing a boundary condition causing divergence. You refactor to eliminate the branch, and all threads now complete within nanoseconds of each other.

```c
// eBPF program runs on GPU at kernel exit
SEC("kretprobe/_Z9vectorAddPKfS0_Pf")
int ret__cuda() {
    u64 tid_x, tid_y, tid_z;
    bpf_get_thread_idx(&tid_x, &tid_y, &tid_z);  // Which thread am I?
    u64 ts = bpf_get_globaltimer();               // When did I finish?

    // Write to ringbuffer for userspace analysis
    bpf_perf_event_output(ctx, &events, 0, &data, sizeof(data));
}
```

### [threadhist](https://github.com/eunomia-bpf/bpftime/tree/master/example/gpu/threadhist) - Thread Execution Count Histogram

Uses GPU array maps to count how many times each thread executes. Detects workload imbalance where some threads do far more work than others, wasting GPU compute capacity.

**Use case**: Your grid-stride loop processes 1M elements with 5 threads. You expect balanced work, but `threadhist` shows thread 4 executes only 75% as often as threads 0-3. The boundary elements divide unevenly, leaving thread 4 idle while others work. You adjust the distribution and achieve balanced execution.

```c
// eBPF program runs on GPU at kernel exit
SEC("kretprobe/_Z9vectorAddPKfS0_Pf")
int ret__cuda() {
    u64 tid_x, tid_y, tid_z;
    bpf_get_thread_idx(&tid_x, &tid_y, &tid_z);

    // Per-thread counter in GPU array map
    u64 *count = bpf_map_lookup_elem(&thread_counts, &tid_x);
    if (count) {
        __atomic_add_fetch(count, 1, __ATOMIC_SEQ_CST);  // Thread N executed once more
    }
}
```

### [launchlate](https://github.com/eunomia-bpf/bpftime/tree/master/example/gpu/launchlate) - Kernel Launch Latency Profiler

Measures the time between `cudaLaunchKernel()` on CPU and actual kernel execution on GPU. Reveals hidden queue delays, stream dependencies, and scheduling overhead that make fast kernels slow in production.

**Use case**: Your kernels execute in 100μs each, but users report 50ms latency. `launchlate` shows 200-500μs launch latency per kernel because each waits for the previous one and memory transfers to complete. Total time is 5ms, not 1ms. You switch to CUDA graphs, batching all launches, and latency drops to 1.2ms.

```c
BPF_MAP_DEF(BPF_MAP_TYPE_ARRAY, launch_time);

// CPU-side uprobe captures launch time
SEC("uprobe/app:cudaLaunchKernel")
int uprobe_launch(struct pt_regs *ctx) {
    u64 ts_cpu = bpf_ktime_get_ns();  // When did CPU request launch?
    bpf_map_update_elem(&launch_time, &key, &ts_cpu, BPF_ANY);
}

// GPU-side kprobe captures execution start
SEC("kprobe/_Z9vectorAddPKfS0_Pf")
int kprobe_exec() {
    u64 ts_gpu = bpf_get_globaltimer();  // When did GPU actually start?
    u64 *ts_cpu = bpf_map_lookup_elem(&launch_time, &key);

    u64 latency = ts_gpu - *ts_cpu;  // How long did kernel wait in queue?
    u32 bin = get_hist_bin(latency);
    // Update histogram...
}
```

### Other Examples

- **[cuda-counter](https://github.com/eunomia-bpf/bpftime/tree/master/example/gpu/cuda-counter)**: Basic probe/retprobe with timing measurements
- **[mem_trace](https://github.com/eunomia-bpf/bpftime/tree/master/example/gpu/mem_trace)**: Memory access pattern tracing and analysis
- **[directly_run_on_gpu](https://github.com/eunomia-bpf/bpftime/tree/master/example/gpu/directly_run_on_gpu)**: Run eBPF programs directly on GPU without attaching to kernels

### Key Components

1. **CUDA Runtime Hooking**: Intercepts CUDA API calls using Frida-based dynamic instrumentation.
2. **PTX/SPIR-V Modification**: Converts eBPF bytecode to PTX (Parallel Thread Execution) or SPIR-V assembly and injects it into GPU kernels.
3. **Helper Trampoline**: Provides GPU-accessible helper functions for map operations, timing, and context access.
4. **Host-GPU Communication**: Enables synchronous calls from GPU to host via pinned shared memory.

## References

1. [bpftime OSDI '25 Paper](https://www.usenix.org/conference/osdi25/presentation/zheng-yusheng)
2. [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/)
3. [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
4. [eBPF Documentation](https://ebpf.io/)
5. [eGPU: Extending eBPF Programmability and Observability to GPUs](https://dl.acm.org/doi/10.1145/3723851.3726984)
6. [NVBit: A Dynamic Binary Instrumentation Framework for NVIDIA GPUs](https://github.com/NVlabs/NVBit)
