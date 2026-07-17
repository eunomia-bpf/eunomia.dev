# Title

Extending eBPF to GPU Device Contexts

# Abstract (3 paragraphs)

Widely used for ML workloads, GPUs are typically SIMT accelerators with threads in warps on SMs, organized into blocks, launched as kernels, using multi-level memory hierarchies (registers, shared/LDS, L2, device memory) and limited preemption. This complexity creates rich but challenging behavior patterns for observability and extension. Today, many tracing tools for GPU workloads sit at the CPU boundary (e.g., probes on CUDA userspace libraries or kernel drivers), which gives you host-side events, but treats the device as a black box: little visibility inside a running kernel, weak linkage to stalls or memory traffic, and no safe way to adapt behavior in-flight. Vendor profilers provide limited device-side visibility, but they are often siloed from eBPF pipelines, cannot interact with other eBPF programs in the kernel.

We propose offloading eBPF into GPU device contexts by defining GPU-side attach points (CUDA device function entry/exit, block begin/end, barrier/sync, memory ops, and stream ops) and compiling eBPF programs into device bytecode (PTX/SPIR-V), with verifier, helper, and map support for on-device execution. This approach can be 3-10x faster than NVBit, is not vendor-locked, and works with Linux kernel eBPF programs like kprobes and uprobes. This enables GPU extensions like fine-grained profiling at the warp or instruction level, adaptive GPU kernel memory optimization, and programmable scheduling across SMs with eBPF. It can also help accelerate some existing eBPF applications.

The goal of this talk is to explore the challenges and lessons learned from extending eBPF's programming model to GPUs.
