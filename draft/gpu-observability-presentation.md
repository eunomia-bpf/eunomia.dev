---
marp: true
theme: default
paginate: true
---

# Extending eBPF to GPU Device Contexts

Yusheng Zheng, Tong Yu, Yiwei Yang

---

# The Problem: GPU Observability

GPUs have become dominant accelerators for ML, scientific computing, and HPC.

But their SIMT execution model introduces significant observability and extensibility challenges.

---

# GPU Architecture Complexity

Organization: Thousands of threads in warps (32 threads) executing in lockstep on streaming multiprocessors (SMs)

Memory hierarchy:
- Per-thread registers (fast, limited)
- Shared memory within thread blocks
- L1/L2 caches
- Device memory (slow, abundant)

Challenge: Limited preemption makes kernel execution difficult to interrupt or inspect

---

# Performance Challenges

Rich performance characteristics that impact throughput:
- Warp divergence
- Memory coalescing patterns
- Bank conflicts
- Occupancy variations

These behaviors remain largely opaque to traditional observability tools.

---

# Existing Tools: Two Categories

1. CPU-GPU Boundary Tracing
- Intercepts CUDA/ROCm library calls (LD_PRELOAD) or kernel drivers
- Captures kernel launches, memory transfers, API timing
- Limitation: Treats GPU as a black box—no visibility into kernel execution

---

# Existing Tools: GPU Profilers

2. GPU Vendor-Specific Profilers
- NVIDIA CUPTI, Nsight Compute
- Intel GTPin
- AMD ROCProfiler
- Research tools: NVBit, Neutrino

Limitations:
- Isolated ecosystems disconnected from Linux kernel observability
- Cannot correlate GPU events with CPU-side eBPF probes
- Substantial overhead (10-100x slowdowns)
- Lack dynamic programmability

---

# The Timeline Visibility Gap

Common debugging scenario:

"My CUDA application takes 500ms to complete, but I don't know where the time is spent. Is it memory transfers, kernel execution, or API overhead?"

The answer depends critically on whether the application uses synchronous or asynchronous CUDA APIs.

---

# Synchronous Execution Example

CPU Timeline (what traditional tools see):
- cudaMalloc() blocks ~1μs
- cudaMemcpy() blocks 200μs (H→D transfer)
- cudaLaunchKernel() returns immediately
- cudaDeviceSynchronize() blocks 115μs

GPU Timeline (hidden phases):
Alloc ~1μs → H→D DMA 200μs → Launch 5μs → Kernel Exec 100μs → Cleanup ~10μs

---

# What CPU-Side Tools Cannot See

When kernel sync takes 115μs, CPU-side tools cannot distinguish:
- Launch overhead (~5μs)
- Actual kernel execution (~100μs)
- Cleanup (~10μs)

Cannot determine root cause:
Excessive launch overhead? Warp divergence? Memory stalls? SM underutilization?

---

# Asynchronous Execution: Problem

Modern CUDA applications use async APIs to maximize hardware utilization by overlapping CPU work with GPU execution.

This introduces temporal decoupling where API calls return immediately after enqueuing work.

This breaks the observability that CPU-side tools had in synchronous mode.

---

# Async Example: What Happens

CPU perspective:
- All async calls return in ~1μs (enqueued)
- CPU continues other work
- cudaStreamSynchronize() blocks 456μs

GPU execution (sequential in stream):
Alloc 1μs → H→D DMA 200μs → Launch 5μs → Kernel 100μs → D→H DMA 150μs

---

# Async Visibility Problem

All timing information collapses into a single 456μs aggregate at the sync point.

"Is my bottleneck memory transfer or kernel execution?"

Unanswerable from CPU side. In sync mode you could measure individual API calls. In async mode, this capability disappears entirely.

---

# Modern GPU Optimizations Add Complexity

- Batching strategies: Combine operations, hide individual bottlenecks
- Persistent kernels: Obscure phase boundaries
- Multi-stream execution: Operations interleave unpredictably
- Shared memory constraints: Create subtle resource contention

Result: Improved throughput but even more opaque execution model

---

# The Cross-Layer Problem

Even with perfect GPU-internal visibility, we can't determine root causes:

High memory stall cycles - why?
- Warp divergence causing uncoalesced access?
- Host thread descheduling delaying async memory copies?
- PCIe congestion from concurrent RDMA operations?
- On-demand page migration latency?

---

# Production Correlation Challenge

SM underutilization - why?
- Grid too small?
- Launches serialized by userspace mutex?
- Driver throttled after ECC error?

Production tail latency spikes - why?
- GPU cache effects? cgroup throttling? Container DMA interference?

The gap: Device-only tooling reports "what" but not "why"

---

# The Key Insight

Effective GPU observability requires a unified solution spanning:
- Userspace applications (CUDA API calls)
- OS kernel drivers (resource management)
- Device code (GPU hardware execution)

Problem: Traditional tools are fragmented, providing isolated visibility without cross-layer correlation

---

# Current Tools: Nsight Systems/Compute

Nsight Systems: System-wide timelines with CPU threads, CUDA API calls, kernel launches, memory transfers

Nsight Compute: Deep kernel-level microarchitectural metrics via CUPTI counters, guided optimization rules

These can correlate launches with CPU thread scheduling and provide per-kernel stall reasons.

---

# Nsight Limitations vs. eBPF Approach

Closed event model: Fixed set of events, no arbitrary programmable logic at attach points

Profiling sessions: Require special sessions that perturb workload behavior, unsuitable for always-on production telemetry

No in-situ filtering: Export all raw data then post-process → multi-GB traces

---

# More Nsight Limitations

Poor system integration:
- Can't attach dynamic probes to persistent kernels without restart
- No integration with Linux eBPF infrastructure
- Can't share data structures (maps) across CPU and GPU instrumentation

NVIDIA-only: No vendor-neutral path for AMD, Intel, or heterogeneous systems

---

# Why eBPF?

eBPF (extended Berkeley Packet Filter): Revolutionary Linux kernel technology

Allows sandboxed programs to be dynamically loaded to extend kernel capabilities safely.

On CPU side: Cornerstone of modern observability, networking, and security

---

# eBPF's Power on CPU

Key advantages:
- Programmability: Attach custom logic to thousands of hook points
- Safety: Sandboxed execution with verification
- Performance: Collecting deep, customized telemetry with minimal overhead

bpftime's vision: Bring this same transformative power to the traditionally opaque world of GPU computing.

---

# The Solution: eBPF on GPU with bpftime

By running eBPF programs natively inside GPU kernels, bpftime provides:

- Safe, programmable, unified observability and extensibility across the entire stack
- Always-on production monitoring with dynamic load/unload of probes
- Device-side predicate filtering to reduce overhead

---

# Recovering Visibility

Async-mode visibility:
- Per-phase timestamps (H→D at T+200μs, kernel at T+206μs, D→H at T+456μs)

GPU-internal details:
- Nanosecond-granularity telemetry for warp execution and memory patterns

CPU-GPU correlation:
- Without heavyweight overhead of traditional separate profilers

---

# Unified Control Plane

CPU and GPU probes as peers in a unified control plane:
- Shared BPF maps and ring buffers enable direct data exchange
- Dynamic instrumentation without recompilation or restart
- Integration with existing eBPF infrastructure (perf, bpftrace, custom agents)

---

# Cross-Layer Tracing

Simultaneously trace:
- CPU-side CUDA API calls via uprobes
- Kernel driver interactions via kprobes
- GPU-side kernel execution via CUDA probes

All using the same eBPF toolchain, correlating events across the host-device boundary.

---

# Answerable Questions

Example questions now become answerable:

- "Did the CPU syscall delay at T+50μs cause the GPU kernel stall at T+150μs?"
- "Which CPU threads launch kernels with high warp divergence?"

Result: Root-cause analysis spanning the entire heterogeneous execution stack

---

# bpftime's Approach

Extends eBPF's programmability directly into GPU execution contexts, enabling eBPF programs to run natively inside GPU kernels.

Defines comprehensive GPU-side attach points mirroring CPU-side kprobes/uprobes flexibility.

---

# GPU Attach Points

Developers can instrument:
- CUDA/ROCm device function entry and exit points
- Thread block lifecycle events (begin/end)
- Synchronization primitives (barriers, atomics)
- Memory operations (loads, stores, transfers)
- Stream/event operations

---

# How It Works

1. eBPF programs written in restricted C
2. Compiled through LLVM into device-native bytecode (PTX for NVIDIA, SPIR-V for AMD/Intel)
3. Dynamically injected into target kernels at runtime through binary instrumentation
4. No source code modification or recompilation required

---

# Runtime Environment on GPU

Safety verifier: Bounded execution and memory safety in SIMT context

GPU-aware helpers: Thread/block/grid context, timing, synchronization

Specialized BPF maps: Array maps for per-thread data, ringbuf for event streaming

Host-GPU communication: Shared memory for calling host-side helpers

---

# What This Enables

- Fine-grained telemetry: Per-warp timing, memory patterns, control flow at nanosecond granularity
- Adaptive behavior: Modify kernels based on runtime conditions
- Custom extensions: Build optimizations without recompilation
- Unified observability: Integrate with existing CPU-side eBPF programs

Production-ready overhead characteristics maintained

---

# Performance & Features

- 3-10x faster than tools like NVBit for instrumentation
- Vendor-neutral design that works across NVIDIA, AMD and Intel GPUs
- Unified observability and control with Linux kernel eBPF programs
- Fine-grained profiling at the warp or instruction level
- Adaptive GPU kernel memory optimization and programmable scheduling
- Dynamic extensions without recompilation

---

# Architecture: Four Core Goals

1. Unified interface: Across userspace, kernel, multiple CPU/GPU contexts, vendor-neutral
2. Dynamic instrumentation: Runtime attachment without source modification or recompilation
3. Safe execution: Within GPU hardware and SIMT model constraints
4. Easy deployment: Built on existing CUDA/ROCm/OpenGL runtimes, no custom drivers

---

# CUDA Attachment Pipeline (1/2)

Key components:

1. CUDA Runtime Hooking: Using LD_PRELOAD, intercepts calls to CUDA/ROCm runtime
2. eBPF to PTX/SPIR-V JIT: Compiles eBPF bytecode into target GPU's instruction set
3. Binary Instrumentation: Injects compiled eBPF code into target kernel's binary at runtime

---

# CUDA Attachment Pipeline (2/2)

Key components (continued):

4. Helper Function Trampoline: Provides eBPF helper functions accessible from GPU (maps, timestamps, ring buffers)
5. Shared Data Structures: BPF maps and ring buffers over shared memory or device memory

---

# Architecture Diagram

```
[Application Process with LD_PRELOAD]
           |
    [PTX/SPIR-V] --> [bpftime runtime] --> [GPU Kernel with eBPF]
           |                                        |
           +----------> [Shared Maps] <------------+
                        (Host-GPU)
```

JIT compilation transforms eBPF bytecode to GPU-native code, injected at kernel launch time.

---

# Example 1: kernelretsnoop

Per-Thread Exit Timestamp Tracer

Attaches to CUDA kernel exits and records the exact nanosecond timestamp when each GPU thread completes execution.

What it reveals: Thread divergence, memory access patterns, warp scheduling issues invisible to traditional profilers.

---

# kernelretsnoop Use Case

Problem: Kernel is slower than expected.

kernelretsnoop reveals: Thread 31 in each warp finishes 750ns later than threads 0-30.

Root cause: Boundary condition causing divergence.

Fix: Refactor to eliminate the branch → all threads now complete within nanoseconds of each other.

---

# kernelretsnoop Code

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

---

# Example 2: threadhist

Thread Execution Count Histogram

Uses GPU array maps to count how many times each thread executes.

What it detects: Workload imbalance where some threads do far more work than others, wasting GPU compute capacity.

---

# threadhist Use Case

Scenario: Grid-stride loop processes 1M elements with 5 threads, expecting balanced work.

threadhist shows: Thread 4 executes only 75% as often as threads 0-3.

Root cause: Boundary elements divide unevenly, leaving thread 4 idle while others work.

Fix: Adjust distribution → balanced execution achieved.

---

# threadhist Code

```c
// eBPF program runs on GPU at kernel exit
SEC("kretprobe/_Z9vectorAddPKfS0_Pf")
int ret__cuda() {
    u64 tid_x, tid_y, tid_z;
    bpf_get_thread_idx(&tid_x, &tid_y, &tid_z);

    // Per-thread counter in GPU array map
    u64 *count = bpf_map_lookup_elem(&thread_counts, &tid_x);
    if (count) {
        __atomic_add_fetch(count, 1, __ATOMIC_SEQ_CST);
    }
}
```

---

# Example 3: launchlate

Kernel Launch Latency Profiler

Measures the time between `cudaLaunchKernel()` on CPU and actual kernel execution on GPU.

What it reveals: Hidden queue delays, stream dependencies, and scheduling overhead that make fast kernels slow in production.

---

# launchlate Use Case

Problem: Kernels execute in 100μs each, but users report 50ms latency

launchlate reveals: 200-500μs launch latency per kernel

Root cause: Each kernel waits for previous one and memory transfers (5ms total, not 1ms)

Fix: Switch to CUDA graphs → latency drops to 1.2ms

---

# launchlate Code

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
}
```

---

# Other Examples

- cuda-counter: Basic probe/retprobe with timing measurements
- mem_trace: Memory access pattern tracing and analysis
- directly_run_on_gpu: Run eBPF programs directly on GPU without attaching to kernels

All demonstrate bcc-style tools bringing familiar Linux observability patterns to GPUs.

---

# Key Components

1. CUDA Runtime Hooking: Intercepts CUDA API calls using Frida-based dynamic instrumentation
2. PTX Modification: Converts eBPF bytecode to PTX assembly and injects it into GPU kernels
3. Helper Trampoline: Provides GPU-accessible helper functions for map operations, timing, and context access
4. Host-GPU Communication: Enables synchronous calls from GPU to host via pinned shared memory

---

# Summary: The Problem

GPU computing lacks flexible, programmable observability:
- Traditional tools operate at CPU-GPU boundary or in isolated vendor ecosystems
- Asynchronous execution eliminates visibility
- Cross-layer correlation is manual and expensive
- Production issues require iterative guessing

---

# Summary: The Solution

bpftime brings eBPF to GPUs:
- Dynamic instrumentation inside GPU kernels
- Unified CPU-GPU observability with shared data structures
- Cross-layer event correlation spanning userspace, kernel, and device
- Production-ready: 3-10x faster, vendor-neutral, always-on monitoring

---

# Learn More

Project: https://github.com/eunomia-bpf/bpftime

Paper: bpftime OSDI '25 (USENIX)
https://www.usenix.org/conference/osdi25/presentation/zheng-yusheng

References:
- [eGPU: Extending eBPF to GPUs](https://dl.acm.org/doi/10.1145/3723851.3726984)
- [NVBit: Dynamic Binary Instrumentation](https://research.nvidia.com/publication/2016-08_nvbit)
- [eBPF Documentation](https://ebpf.io/)

---

# Questions?

Contact:
- GitHub: https://github.com/eunomia-bpf/bpftime
- Email: team@eunomia.dev

Note: GPU support is experimental. Feedback welcome!

Thank you!
