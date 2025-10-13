---
date: 2025-10-11
---

# Understanding iaprof: A Deep Dive into AI/GPU Flame Graph Profiling

*An exploration of Intel's innovative profiling tool that bridges the gap between CPU and GPU execution*

> **Project Link**: [github.com/intel/iaprof](https://github.com/intel/iaprof) - Intel's AI/GPU flame graph profiler

If you've ever tried to optimize a GPU-accelerated machine learning workload, you've likely encountered a frustrating problem: your code runs on the CPU, but the performance bottlenecks live on the GPU. Traditional profiling tools show you one world or the other, but never both together. This disconnect makes it nearly impossible to understand which lines of your Python or C++ code are responsible for expensive GPU operations.

Enter **iaprof**, Intel's solution to this observability challenge. At its core, iaprof is a profiling tool that generates what Intel calls "AI Flame Graphs" - interactive visualizations that seamlessly connect your application code to the GPU instructions it triggers, all while showing where performance is actually being lost.

## The Visibility Gap in GPU Computing

To appreciate what iaprof accomplishes, we need to understand the fundamental challenge of profiling GPU workloads. When you write code that uses GPUs for computation, there's an inherent disconnect between where you write code and where that code actually executes. Your application runs on the CPU, making calls to high-level frameworks like PyTorch or TensorFlow, which in turn call runtime libraries like Level Zero or CUDA, which eventually communicate with kernel drivers that manage the GPU hardware.

This deep software stack creates multiple layers of abstraction. By the time your matrix multiplication reaches the GPU, it's been transformed from Python through multiple runtime layers into GPU machine code - shader instructions executing on Execution Units (EUs) deep inside the silicon. When that operation runs slowly, traditional tools struggle to tell you why.

General-purpose profilers like `perf` or Intel VTune excel at showing CPU execution patterns. They can tell you exactly which CPU functions are consuming time, but they treat the GPU as an opaque black box. When your profile shows that most time is spent in `cudaLaunchKernel` or `zeCommandListAppendLaunchKernel`, you know you're waiting on the GPU, but not which specific GPU operations are slow or what those operations are actually doing.

GPU-specific profilers take the opposite approach. Tools like Intel GPA or `nvidia-smi` provide detailed GPU metrics - memory bandwidth utilization, compute unit occupancy, instruction throughput. These metrics are invaluable for understanding GPU performance, but they lack the crucial connection back to your source code. You might see that a particular shader has memory access stalls, but determining which line of your training loop launched that shader requires manual detective work, correlating timestamps across separate CPU and GPU timelines.

## How iaprof Bridges the Gap

iaprof's innovation is creating a single, unified view that connects application code to GPU hardware performance. The tool combines three sophisticated techniques to achieve this visibility.

First, it uses eBPF (extended Berkeley Packet Filter) to trace GPU driver operations at the kernel level. eBPF is a Linux kernel technology that allows safe, verified programs to run inside the kernel with minimal overhead. iaprof's eBPF programs hook into the GPU driver's execution paths, intercepting every GPU kernel launch and capturing the complete CPU call stack at the moment of launch. This happens transparently, without requiring any modifications to your application code or even recompilation.

Second, iaprof leverages Intel's Observability Architecture (OA) for hardware sampling. The OA is a hardware feature built into modern Intel GPUs that can collect performance counter samples directly from the Execution Units with minimal overhead - typically under 5%. These samples capture detailed information about what's happening at the hardware level: which instructions are executing, what types of stalls are occurring (memory access, control flow, synchronization), and how frequently each pattern appears.

Third, the tool uses the GPU debug API to retrieve shader binaries and execution metadata. This allows iaprof to disassemble GPU machine code and map hardware samples to specific instructions, complete with human-readable assembly syntax.

The magic happens when iaprof combines these three data streams. For every GPU instruction that shows up in hardware sampling, the tool can trace back through the debug information to identify the shader, through the driver trace to find the kernel launch, and through the CPU stack capture to determine which application functions triggered that launch. The result is a complete execution path from your Python script or C++ application all the way down to the specific multiply-add instruction stalling on memory access inside the GPU.

## Intel GPU Architecture Background

To understand how iaprof works, it's helpful to know the basics of Intel GPU architecture. Modern Intel GPUs are built around a hierarchy of parallel execution units designed for massive throughput.

At the lowest level are **Execution Units (EUs)**, the fundamental compute cores of the GPU. Each EU contains a SIMD (Single Instruction Multiple Data) processor that can execute the same instruction across multiple data elements simultaneously - typically 8 or 16 elements wide depending on the architecture. Think of an EU as analogous to a CPU core, but optimized for data-parallel workloads rather than sequential code.

Intel's use of SIMD differs from NVIDIA's SIMT (Single Instruction Multiple Thread) approach, though the distinction has blurred in modern architectures. In pure SIMD, all data lanes execute exactly the same instruction in lockstep with no per-lane control flow - if one lane needs to branch differently, all lanes must execute both paths with masking. NVIDIA's SIMT model, by contrast, treats each lane as an independent thread with its own instruction pointer and register file, allowing for more flexible divergence handling through hardware thread scheduling. Each NVIDIA "warp" contains 32 threads that preferentially execute together but can diverge when needed. However, Intel GPUs have evolved to include masked execution capabilities similar to SIMT, allowing individual SIMD lanes to be selectively enabled or disabled based on predicate masks. This hybrid approach gives Intel GPUs some SIMT-like flexibility while maintaining the efficiency of SIMD execution for uniform workloads. For profiling purposes, this means that control flow divergence can create performance artifacts where some EU lanes are active while others are masked off, wasting potential throughput.

EUs are organized into **subslices** (or **Xe-cores** in newer architectures), which group multiple EUs together with shared resources like a local L1 cache, texture sampling units, and load/store units. A subslice might contain 8-16 EUs working together. These subslices are further grouped into **slices**, which share larger caches and memory controllers.

When you launch a GPU kernel - whether it's a compute shader in SYCL, a CUDA kernel compiled for Intel, or a graphics shader in Vulkan - the GPU's thread dispatcher breaks your work into many parallel threads. These threads are grouped into **SIMD lanes** that execute together on an EU. The hardware automatically manages scheduling these threads across available EUs to maximize utilization.

The **Observability Architecture (OA)** is Intel's hardware performance monitoring system built into the GPU. It can sample the execution state of EUs at regular intervals, capturing which instruction each EU is executing and critically, what's preventing it from making progress. These "stall types" reveal whether an EU is actively computing, waiting for memory, blocked on control flow dependencies, or stalled on synchronization primitives.

This is where iaprof's power comes from: by sampling EU stall states and correlating them with specific shader instructions and CPU call stacks, it reveals exactly where your code is losing performance - not just that a kernel is slow, but which specific instructions within that kernel are causing the bottleneck and what hardware resource they're waiting on.

## The Architecture: Inside iaprof

Understanding how iaprof works requires looking at its multi-threaded architecture. The tool runs as a privileged user-space application (requiring root or `CAP_PERFMON` capabilities) that coordinates several concurrent collection threads, each gathering different pieces of the profiling puzzle.

At the heart of iaprof's approach is the combination of four key technologies working in concert. eBPF kernel tracing intercepts GPU driver calls to track kernel launches and memory mappings without requiring any code changes to the target application. Hardware sampling collects Execution Unit (EU) stall samples via Intel's Observability Architecture with less than 5% overhead, ensuring the profiler doesn't significantly perturb the workload being measured. The debug API retrieves shader binaries and detailed execution information enabling instruction-level analysis of what the GPU is actually doing. Finally, sophisticated symbol resolution translates raw memory addresses in both CPU and kernel stacks into human-readable symbols complete with file and line number information.

The end result is an interactive flame graph showing the complete execution path from application code - whether that's Python, C++, or another language - through runtime libraries like PyTorch, Level Zero, or Vulkan, down through the kernel driver, all the way to specific GPU shader instructions. Each frame in this visualization is annotated with hardware stall metrics broken down by type: memory access stalls, control flow stalls, synchronization stalls, and more.

This comprehensive visibility enables several critical use cases in modern GPU computing. Teams can optimize AI training and inference workloads by identifying exactly which parts of their neural network architecture are causing GPU performance bottlenecks. Graphics engineers can pinpoint bottlenecks in rendering pipelines and trace them back to the specific draw calls or shader code responsible. Machine learning engineers can understand and quantify the overhead introduced by framework layers, distinguishing between actual computation time and framework coordination overhead. Compiler teams can validate whether their GPU compiler optimizations are having the intended effect at the hardware level. And development teams can run performance regression testing to catch when code changes inadvertently introduce GPU performance issues.

Currently, iaprof supports Intel Data Center GPU Max (Ponte Vecchio), Intel Arc B-series (Battlemage), and other Xe2-based GPUs, with ongoing work to expand hardware compatibility.

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                           User Application                          │
│         (e.g., PyTorch, TensorFlow, SYCL, Vulkan, etc.)             │
└────────────────────────────┬────────────────────────────────────────┘
                             │ API calls
┌────────────────────────────▼────────────────────────────────────────┐
│                     Runtime Libraries                                │
│       (Level Zero, Vulkan, OpenCL, oneDNN, cuDNN, etc.)             │
└────────────────────────────┬────────────────────────────────────────┘
                             │ ioctl()
┌────────────────────────────▼────────────────────────────────────────┐
│                      Linux Kernel                                    │
│  ┌──────────────────┐  ┌─────────────────┐  ┌────────────────────┐ │
│  │  DRM Subsystem   │  │  i915/Xe Driver │  │  eBPF Subsystem    │ │
│  │                  │◄─┤  (GPU Driver)   │  │                    │ │
│  │  - Device mgmt   │  │  - Memory mgmt  │◄─┤  - iaprof BPF      │ │
│  │  - IOCTL routing │  │  - Execution    │  │    programs        │ │
│  │                  │  │  - Batch buffers│  │  - Ringbuffer      │ │
│  └──────────────────┘  └────────┬────────┘  └─────────┬──────────┘ │
│                                 │                      │             │
│                                 │ Commands             │ Events      │
│                                 ▼                      │             │
│  ┌────────────────────────────────────────────────────┼──────────┐ │
│  │                 Intel GPU Hardware                  │          │ │
│  │  ┌──────────┐  ┌──────────┐  ┌─────────────────┐  │          │ │
│  │  │ EU Array │  │ L1/L3/RAM│  │ OA (Observ.    │  │          │ │
│  │  │ (Compute)│  │ (Memory) │  │  Architecture) │◄─┘          │ │
│  │  │          │  │          │  │ - Sampling HW  │             │ │
│  │  │ Execution│  │ Data     │  │ - Perf counters│             │ │
│  │  └──────────┘  └──────────┘  └─────────────────┘             │ │
│  └────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────┬──────────────────────────────┘
                                       │
                                       │ 1. eBPF events (ringbuffer)
                                       │ 2. EU stall samples (debug API)
                                       │ 3. Shader uploads (debug API)
                                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                            iaprof                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │ BPF Collector│  │Debug Collect.│  │EUStall Coll. │             │
│  │              │  │              │  │              │             │
│  │ - Load BPF   │  │ - Shader bin │  │ - HW samples │             │
│  │ - Parse BB   │  │ - Context    │  │ - Stall attr.│             │
│  │ - Stacks     │  │   tracking   │  │   -ibution   │             │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘             │
│         │                 │                 │                       │
│         └────────┬────────┴────────┬────────┘                       │
│                  ▼                 ▼                                 │
│         ┌────────────────────────────────┐                          │
│         │    GPU Kernel Store (Tree)     │                          │
│         │  - Shader metadata & binaries  │                          │
│         │  - CPU/kernel stacks           │                          │
│         │  - Per-offset stall counts     │                          │
│         └───────────────┬────────────────┘                          │
│                         │                                            │
│                         ▼                                            │
│         ┌────────────────────────────────┐                          │
│         │       Stack Printer            │                          │
│         │  - Symbol resolution           │                          │
│         │  - Stack deduplication         │                          │
│         │  - Folded stack output         │                          │
│         └───────────────┬────────────────┘                          │
│                         │                                            │
└─────────────────────────┼────────────────────────────────────────────┘
                          │
                          ▼
                   Folded Stacks
  (CPU_stack;GPU_kernel;GPU_instruction stall_counts)
                          │
                          ▼
                   FlameGraph.pl
                          │
                          ▼
                  Interactive SVG
               (AI/GPU Flame Graph)
```

## Core Components

### 1. eBPF Subsystem (Kernel Space)

**Location**: `src/collectors/bpf/bpf/*.bpf.c`

**Purpose**: Trace GPU driver operations in real-time without modifying kernel source

#### Key eBPF Programs

iaprof uses several specialized eBPF programs, each targeting different aspects of GPU driver behavior. These programs work together to build a complete picture of what happens when your application submits work to the GPU.

**a) Execution Buffer Tracing** (`i915/execbuffer.bpf.c`, `xe/exec.bpf.c`)

The execution buffer tracer is the cornerstone of iaprof's CPU-to-GPU correlation. It hooks into the critical moment when the GPU driver receives a submission request from userspace. This hook uses fexit (function exit) probes on `i915_gem_do_execbuffer` for i915 driver or `xe_exec_ioctl` for Xe driver, ensuring it captures the complete context after the driver has processed the submission.

At this point, the tracer captures:
- Execution buffer ID
- VM and context IDs
- CPU user-space stack (via `bpf_get_stack` with `BPF_F_USER_STACK`)
- Kernel stack (via `bpf_get_stack`)
- Process metadata (PID, TID, CPU, timestamp, command name)
- Batch buffer location (GPU address, CPU mapping)

Sends: `struct execbuf_info` to ringbuffer

**b) Batch Buffer Parser** (`batchbuffer.bpf.c`)

The batch buffer parser represents one of iaprof's most sophisticated eBPF components. Batch buffers are the command streams that tell the GPU what to execute - they contain encoded instructions for loading shaders, configuring registers, and dispatching compute or graphics work. To correlate GPU execution with specific shaders, iaprof must parse these buffers to extract the addresses where shader code resides.

This parser runs in the same hook context as the execbuffer tracer, giving it access to the freshly submitted batch buffer. The parsing process:

```
1. Translates GPU batch buffer address to CPU address using gpu_cpu_map
2. Reads batch buffer from userspace memory (bpf_probe_read_user)
3. Parses GPU commands DWORD-by-DWORD:
   - Identifies command type from bits 31:29
   - Calculates command length (static or dynamic based on command)
   - Follows BATCH_BUFFER_START to chained buffers (up to 3 levels deep)
   - Extracts Kernel State Pointers (KSPs) from COMPUTE_WALKER and 3DSTATE commands
   - Extracts System Instruction Pointer (SIP) from STATE_SIP
4. If buffer incomplete (NOOPs encountered): Defers parsing with BPF timer
5. Emits KSPs and SIP to ringbuffer
```

Challenges:

Implementing this parser within eBPF presents several significant challenges. The eBPF verifier imposes strict constraints including maximum instruction counts and requirements that all loops must be provably bounded, making complex parsing logic difficult to express. The parser must also handle race conditions where the batch buffer may not yet be fully written by userspace when the execbuffer ioctl is called, requiring deferred parsing with timers. Additionally, nested batch buffers that chain to other buffers require careful state tracking to avoid infinite loops while ensuring all shader pointers are discovered.

**c) Memory Mapping Tracing** (`i915/mmap.bpf.c`, `xe/mmap.bpf.c`)

GPU buffers live in a complex memory space with multiple address translations. A single buffer might have a GPU virtual address (what the GPU sees), a handle (what userspace uses to reference it), and a CPU virtual address (what the application can read/write). The memory mapping tracer maintains these translation tables, which are essential for the batch buffer parser to read GPU command streams from userspace memory.

This tracer hooks:
- `i915_gem_mmap_offset_ioctl` (fexit) - Records fake offset
- `i915_gem_mmap` (fexit) - Captures CPU address, associates with handle
- `unmap_region` (fentry) - Cleans up mappings on munmap

Maintains: `gpu_cpu_map` (GPU addr → CPU addr) and `cpu_gpu_map` (reverse)

**d) VM Bind Tracing** (`i915/vm_bind.bpf.c`, `xe/vm_bind.bpf.c`)

Discrete GPUs (those with their own dedicated memory, separate from system RAM) use an additional layer of virtual memory management. Before the GPU can access a buffer, the buffer must be "bound" to a GPU virtual address within a specific virtual memory (VM) context. The VM bind tracer tracks these bindings, allowing iaprof to resolve GPU addresses back to the actual buffer contents.

This tracer hooks:
- `i915_gem_vm_bind_ioctl` / `xe_vm_bind` (fexit)
- `i915_gem_vm_unbind_ioctl` / `xe_vm_unbind` (fentry)

Purpose: For discrete GPUs, maps GPU virtual addresses to buffer handles and CPU addresses

**e) Context Tracking** (`i915/context.bpf.c`, `xe/context.bpf.c`)

GPU contexts are execution environments that encapsulate state like which VM is being used, what shaders are loaded, and various configuration settings. When an execution buffer is submitted, it's associated with a specific context. The context tracker maintains the mapping from context IDs to VM IDs, which is crucial for tying everything together - when we see a GPU address in a batch buffer, we need to know which VM context it belongs to in order to translate it correctly.

Tracks: Context ID → VM ID mapping

Needed: To resolve which VM an execbuffer belongs to

#### eBPF Maps

eBPF maps are shared data structures that act as the communication and storage layer for the eBPF programs. They allow programs to share data with each other and with userspace, and persist state across multiple invocations. iaprof uses several key maps:

- **`rb` (ringbuffer)**: Main output channel, 512 MB
- **`gpu_cpu_map`**: Hash table, GPU address → CPU address/size
- **`cpu_gpu_map`**: Reverse lookup
- **`deferred_parse`**: Batch buffers awaiting retry
- **`deferred_timers`**: BPF timer wrappers for deferred parsing
- **`bb_ksps`**: Per-CPU hash table of discovered KSPs
- **`context_create_wait_for_exec`**: Context → VM ID mappings

### 2. Userspace BPF Collector

**Location**: `src/collectors/bpf/bpf_collector.c`

**Responsibilities**:

**Initialization**:

- Loads compiled eBPF object file (`.bpf.o`)
- Resolves BTF relocations
- Attaches programs to kernel tracepoints
- Creates ringbuffer and maps file descriptors

**Event Loop** (runs in dedicated thread):

- `epoll_wait()` on ringbuffer file descriptor
- `ring_buffer__poll()` to consume events
- Dispatches to event handlers based on event type

**Event Handlers**:

- **EXECBUF**: Creates or updates shader in GPU kernel store
- **EXECBUF_END**: Marks batch buffer parsing complete
- **KSP**: Adds kernel pointer to shader
- **SIP**: Adds system routine pointer
- **UPROBE_*** (if enabled): User-space probe events

### 3. EU Stall Collector

**Location**: `src/collectors/eustall/eustall_collector.c`

**Purpose**: Collect and attribute hardware EU stall samples

The EU stall collector is where iaprof connects hardware performance data to the execution context captured by eBPF. Intel's Observability Architecture continuously samples the GPU's Execution Units, recording which instruction is executing and what type of stall (if any) is occurring at each sample point. The EU stall collector receives these samples and attributes them to specific shaders and instruction offsets, building up a statistical profile of where the GPU is spending its time and what's causing performance bottlenecks.

#### Data Flow

```
GPU Hardware Sampling
    ↓
OA / Debug Interface
    ↓
struct eustall_sample {
    uint32_t ip;      // GPU instruction pointer (shifted by 3)
    uint64_t active;  // Active execution count
    uint64_t control; // Control flow stall count
    uint64_t send;    // Memory access stall count
    // ... more stall types
}
    ↓
handle_eustall_sample()
    ├─→ addr = ip << 3
    ├─→ shader = acquire_containing_shader(addr)
    ├─→ if shader found:
    │   ├─→ offset = addr - shader->gpu_addr
    │   └─→ shader->stall_counts[offset] += sample counts
    └─→ else:
        └─→ add to deferred_eustall_waitlist
```

#### Stall Types

Understanding stall types is key to interpreting iaprof's output. Not all time on the GPU represents productive computation. The hardware distinguishes between several categories of execution state, from hardware sampling:

- **Active**: EU actively executing (not stalled)
- **Control**: Control flow dependencies (branches, predication)
- **Pipestall**: Pipeline resource stalls
- **Send**: Memory/sampler/URB send stalls
- **Dist_acc**: Distributed accumulator stalls
- **SBID**: Scoreboard ID stalls (dependency chains)
- **Sync**: Synchronization (barriers, fences)
- **Inst_fetch**: Instruction fetch stalls
- **TDR** (Xe only): Thread dispatch/retire stalls

#### Deferred Attribution

One challenge in correlating hardware samples with shaders is timing - hardware samples can arrive before iaprof has received the shader metadata from the debug collector. This happens because the GPU might start executing a shader while the debug information is still being processed by the kernel and transmitted to userspace. Rather than discarding these samples, iaprof uses a deferred attribution mechanism.

When EU stall arrives before corresponding shader metadata:

```c
struct deferred_eustall {
    struct eustall_sample sample;
    int satisfied;  // 0 = still waiting, 1 = attributed
};
```

These deferred samples are placed on a waitlist. A separate thread periodically retries attribution when new shaders arrive from the debug collector, eventually matching samples to their corresponding shaders. This ensures no performance data is lost, even when events arrive out of order.

### 4. Debug Collector

**Location**: `src/collectors/debug/debug_collector.c`

**Purpose**: Interface with GPU debug API for detailed execution information

The debug collector interfaces with Intel's GPU debug API to access information that's not available through standard driver interfaces. This includes the actual compiled shader binaries, detailed context information, and fine-grained execution control. Without the debug API, iaprof would only have GPU addresses - the debug API provides the actual machine code at those addresses, enabling instruction-level disassembly and analysis.

#### Capabilities

The debug collector provides several critical capabilities that enable deep GPU inspection. For shader binary collection, it receives shader upload events directly from the driver, copies the shader binary to the shader store, and enables instruction-level disassembly. This is essential for translating GPU instruction pointers into human-readable assembly code.

Context tracking is another key responsibility - the collector monitors context create and destroy events and tracks VM (virtual memory) associations. This information helps iaprof understand which memory space each shader operates in, crucial for correct address resolution.

For deterministic sampling scenarios, the debug collector can exercise execution control by sending SIGSTOP signals to GPU processes for synchronized sampling and triggering EU attention events. This allows for more precise profiling in controlled environments.

The implementation varies by platform: on i915 systems, it uses `PRELIM_DRM_I915_DEBUG_*` ioctls which require a patched kernel, while on Xe systems it uses `DRM_XE_EUDEBUG_*` ioctls that have mainline support starting from Linux 6.2.

### 5. OA Collector

**Location**: `src/collectors/oa/oa_collector.c`

**Purpose**: Initialize Observability Architecture hardware

The OA (Observability Architecture) collector is responsible for configuring Intel's hardware performance monitoring infrastructure. The OA is a dedicated hardware unit within Intel GPUs that can sample execution state with minimal performance impact. This collector sets up the OA hardware to specifically capture EU stall information, configuring sampling rates and which performance counters to monitor.

#### Initialization Sequence

```
1. Open DRM device
2. Write to OA control registers:
   - Enable OA unit
   - Configure sampling period
   - Select counter set (EU stalls)
   - Enable triggers
3. Start sampling
```

Interfaces with EU stall collector to provide sample stream.

### 6. GPU Kernel Store

**Location**: `src/stores/gpu_kernel.c`

The GPU kernel store is iaprof's central data repository - a thread-safe tree structure that maintains information about all discovered GPU shaders. Each shader entry contains not just the shader binary and GPU address, but also the CPU call stack that launched it and accumulated performance samples. This store acts as the meeting point where data from all the collectors comes together, enabling the correlation between CPU code, GPU shaders, and hardware performance metrics.

**Data Structure**:

```c
tree(uint64_t, shader_struct) shaders;  // Global, sorted by GPU address
pthread_rwlock_t shaders_lock;          // Protects tree operations

struct shader {
    pthread_mutex_t lock;                // Per-shader lock

    // Identity
    uint64_t gpu_addr;
    uint64_t size;
    enum shader_type type;               // SHADER, SYSTEM_ROUTINE, DEBUG_AREA

    // Attribution
    uint32_t pid;
    uint64_t proc_name_id, ustack_id, kstack_id;
    uint64_t symbol_id, filename_id;
    int linenum;

    // Binary
    unsigned char *binary;

    // Profiling data
    hash_table(uint64_t, offset_profile_struct) stall_counts;
};
```

**Thread Safety**:

The GPU kernel store uses a sophisticated two-level locking strategy to maximize concurrency while maintaining data consistency. At the first level, a reader-writer lock protects the tree structure itself, allowing multiple threads to search for shaders concurrently while ensuring exclusive access for structural modifications. At the second level, each individual shader has its own mutex that protects its fields, enabling different threads to update different shaders simultaneously without contention. This design ensures that the high-frequency operations like stall attribution don't block other collectors from accessing the store.

**Operations**:

- `acquire_containing_shader(addr)`: Range query, finds shader where `addr ∈ [gpu_addr, gpu_addr+size)`
- `acquire_or_create_shader(addr)`: Gets existing or allocates new
- `release_shader(shader)`: Unlocks shader mutex

### 7. Stack Printer

**Location**: `src/printers/stack/stack_printer.c`

**Purpose**: Transform collected data into folded stacks for flame graphs

The stack printer is the final transformation stage that converts iaprof's internal data structures into the "folded stack" text format that FlameGraph tools understand. This involves translating raw memory addresses from CPU call stacks into human-readable function names, file paths, and line numbers, then combining them with GPU shader names and instruction disassembly to create complete execution traces.

#### Symbol Resolution

Symbol resolution is the process of converting raw instruction pointer addresses into meaningful names. The stack printer handles both kernel and user-space symbols differently, using appropriate symbol sources for each.

**Kernel Symbols**:
```c
struct ksyms *ksyms = ksyms_load();  // Loads /proc/kallsyms
const struct ksym *sym = ksyms_search(ksyms, addr);
```

**User-Space Symbols**:
```c
struct syms_cache *cache = syms_cache_create();  // Per-PID cache
const struct sym *sym = syms_cache_get(cache, pid, addr);
// Parses ELF + DWARF to get function, file, line
```

#### Output Generation

```
For each shader:
  For each offset with stall counts:
    1. Resolve CPU user stack to symbols: main;worker;submit_kernel
    2. Append GPU kernel name: main;worker;submit_kernel;my_compute_kernel
    3. Disassemble GPU instruction at offset: mad(8) r10.0<1>:f ...
    4. Format stall counts: active=1234,send=567,control=89
    5. Output: main;worker;submit_kernel;my_compute_kernel;mad(8) active=1234,send=567
```

**Optimizations**:

Symbol resolution can be expensive when performed repeatedly for thousands of samples. The stack printer employs several optimization strategies to minimize this overhead. Stack string caching uses a hash table to map complete stacks to their folded string representations, ensuring that identical call stacks are only processed once regardless of how many samples they appear in. Symbol caching maintains per-PID symbol tables, so function addresses are resolved just once per process rather than for every sample. Additionally, batch symbol lookups group address resolution operations together, amortizing the cost of parsing debug information across multiple addresses.

## Execution Flow

Now that we understand the individual components, let's see how they work together during a typical profiling session. The execution flow shows how iaprof initializes, collects data, and produces output.

### Startup Sequence

When you run `iaprof record`, a carefully orchestrated initialization sequence sets up all the collectors and prepares the system for profiling. Each step must complete successfully before moving to the next, ensuring that all components are ready to capture data.

```
1. main(argc, argv)
     ↓
2. Parse command (record / flame / flamescope)
     ↓
3. record(argc, argv)
     ↓
4. check_permissions() - Verify root or CAP_PERFMON
     ↓
5. drm_open_device() - Open /dev/dri/card0
     ↓
6. drm_get_driver_name() - Detect i915 vs Xe
     ↓
7. Initialize collectors:
   a. init_bpf()
        ├─→ bpf_object__open_file("main.bpf.o")
        ├─→ bpf_object__load(obj)  // Loads into kernel
        └─→ attach tracepoints
   b. init_oa()
        └─→ Configure OA hardware registers
   c. init_debug()
        └─→ debug_attach(pid=0)  // Attach to all processes
   d. init_eustall()
        └─→ Initialize waitlists, start deferred thread
     ↓
8. Spawn collector threads:
   - bpf_collect_thread()
   - eustall_collect_thread()
   - debug_collect_thread()
   - eustall_deferred_attrib_thread()
     ↓
9. If command specified:
     fork() + exec() child process
   Else:
     Wait for SIGINT (Ctrl-C)
```

### Data Collection (Multi-threaded)

Once initialization completes, iaprof runs several threads concurrently, each collecting different types of data. These threads operate independently but coordinate through the shared GPU kernel store. This multi-threaded design allows iaprof to handle high event rates without dropping data.

**BPF Thread**:
```
while (!should_stop) {
    epoll_wait(rb_fd)  // Block until events
    ring_buffer__poll(rb, timeout)
        ↓ callback for each event
    handle_execbuf_event()
        ├─→ shader = acquire_or_create_shader(gpu_addr)
        ├─→ shader->pid = execbuf->pid
        ├─→ shader->ustack_id = store_stack(execbuf->ustack)
        └─→ release_shader(shader)
    handle_ksp_event()
        ├─→ shader = acquire_shader(ksp->gpu_addr)
        ├─→ shader->type = SHADER_TYPE_SHADER
        └─→ release_shader(shader)
}
```

**EU Stall Thread**:
```
while (!should_stop) {
    debug_read_eustall_sample(&sample)  // Block until sample
    handle_eustall_sample(&sample)
        ├─→ addr = sample.ip << 3
        ├─→ shader = acquire_containing_shader(addr)
        ├─→ if shader:
        │   ├─→ offset = addr - shader->gpu_addr
        │   ├─→ profile = shader->stall_counts[offset]
        │   ├─→ profile->active += sample.active
        │   └─→ ... (other stall types)
        └─→ else:
            └─→ add to deferred_eustall_waitlist
}
```

**Debug Thread**:
```
while (!should_stop) {
    debug_wait_event(&event)  // Block until event
    switch (event.type) {
    case SHADER_UPLOAD:
        shader = acquire_shader(event.addr)
        shader->binary = malloc(event.size)
        memcpy(shader->binary, event.data, event.size)
        release_shader(shader)
        signal_deferred_attrib_thread()  // Wake deferred thread
        break;
    case CONTEXT_CREATE:
        track_context_vm_mapping(event)
        break;
    }
}
```

**Deferred Attribution Thread**:
```
while (!should_stop) {
    pthread_cond_wait(&deferred_cond)  // Wait for signal
    for each deferred_sample in waitlist:
        shader = acquire_containing_shader(deferred_sample.addr)
        if shader:
            associate_sample(&deferred_sample, shader)
            deferred_sample.satisfied = 1
        release_shader(shader)
    remove satisfied samples from waitlist
}
```

### Shutdown and Output

When profiling ends (either because a profiled command exits or the user presses Ctrl-C), iaprof performs an orderly shutdown. This involves stopping all collector threads, processing any remaining deferred samples, walking through the entire shader store to generate output, and cleaning up resources.

```
SIGINT received
    ↓
collect_threads_should_stop = STOP_REQUESTED
    ↓
Join all collector threads
    ↓
Process remaining deferred samples (final retry)
    ↓
FOR_SHADER (shader in shaders tree):
    pthread_rwlock_rdlock(&shaders_lock)
    for each (offset, stall_profile) in shader->stall_counts:
        cpu_stack_str = resolve_stack(shader->ustack_id, shader->pid)
        kernel_name = get_kernel_name(shader)
        inst_str = disassemble(shader->binary, offset)
        stalls_str = format_stalls(stall_profile)
        printf("%s;%s;%s %s\n", cpu_stack_str, kernel_name, inst_str, stalls_str)
    pthread_rwlock_unlock(&shaders_lock)
    ↓
Cleanup:
    deinit_bpf()
    deinit_debug()
    deinit_eustall()
    free_profiles()
    ↓
Exit
```

## Key Design Decisions

Throughout iaprof's development, the team made several critical architectural choices that fundamentally shaped how the tool works. Understanding these decisions provides insight into the trade-offs involved in building a production-quality GPU profiler.

### 1. Why eBPF?

The decision to use eBPF for kernel-level tracing was fundamental to iaprof's design. Traditional approaches would require either a custom kernel module (difficult to maintain across kernel versions) or heavy instrumentation of the application (defeats the goal of transparent profiling). eBPF provides a middle ground that's both powerful and safe.

eBPF brings significant advantages: no kernel module is required, eliminating complex build and distribution issues; programs are verified by the kernel before loading, ensuring safety and preventing crashes; JIT compilation provides high performance approaching native code; and the comprehensive tracing capabilities allow hooking virtually any kernel function. However, there are trade-offs. The eBPF verifier imposes complexity constraints including instruction count limits and requirements for bounded loops, making sophisticated parsing logic challenging to implement. The system requires BTF (BPF Type Format) type information to be available in the kernel, and needs a relatively recent kernel version (5.8 or later) for the CO-RE (Compile Once, Run Everywhere) features that iaprof relies on.

### 2. Why In-Kernel Batch Buffer Parsing?

One of the most complex parts of iaprof is the batch buffer parser that runs inside the kernel via eBPF. An alternative would be to copy batch buffers to userspace and parse them there, which would be simpler to implement and debug. However, the team chose in-kernel parsing for important practical reasons.

**Alternative**: Parse in userspace after capturing buffer

**Chosen**: Parse in kernel with eBPF

**Rationale**:

Copying batch buffers to userspace would be expensive since these buffers can be megabytes in size, adding significant overhead to every GPU submission. More critically, the parsing needs to happen before the buffer is reused or unmapped by the driver, which could happen immediately after submission. The deferred parsing mechanism handles the case where buffers aren't yet complete when first intercepted, retrying later when the data is ready.

**Trade-off**: Higher kernel overhead, but more reliable

### 3. Why Deferred Attribution?

**Problem**: EU stall samples can arrive before shader metadata

**Solution**: Waitlist with retry mechanism

**Triggered by**:
- New shader arrival (debug collector signals)
- Periodic retry (every N seconds)
- Final cleanup (at shutdown)

### 4. Why Two-Level Locking?

**Alternative**: Global lock for entire shader tree

**Chosen**: RW-lock on tree + per-shader mutex

**Benefits**:

The two-level locking approach enables multiple readers to traverse the shader tree concurrently without blocking each other, which is crucial for lookup-heavy workloads. Stall attribution can proceed on one shader while other collectors work on different shaders, preventing bottlenecks. This fine-grained locking strategy significantly reduces contention compared to a global lock, allowing the profiler to scale efficiently with the number of shaders and collection threads.

### 5. Why Folded Stack Format?

**Alternative**: Custom binary format, JSON, protobuf

**Chosen**: Text-based folded stacks

**Rationale**:

The text-based folded stack format provides multiple advantages over binary alternatives. It's compatible with Brendan Gregg's existing FlameGraph ecosystem, allowing iaprof to leverage mature, well-tested visualization tools. The human-readable format makes debugging and validation straightforward - you can simply look at the output to verify correctness. Standard Unix tools like grep, sed, and awk can easily filter and transform the data for custom analyses. Despite being text-based, the format is remarkably compact due to its deduplication strategy where identical stack prefixes are represented once with semicolon-separated paths.

## Performance Characteristics

### Overhead

**i915 Driver**:
- **BPF Collection**: 10-20% (batch buffer parsing expensive)
- **EU Stall**: 2-5% (hardware sampling + attribution)
- **Debug**: 1-3% (event-driven)
- **Total**: 15-30% (workload-dependent)

**Xe Driver**:
- **BPF Collection**: 3-5% (simpler batch buffer handling)
- **EU Stall**: 2-5%
- **Debug**: 1-3%
- **Total**: 5-10% (generally lower)

### Scalability

**Number of Shaders**: O(log n) lookup, O(n) iteration

**Number of EU Stall Samples**: O(1) attribution (hash table)

**Stack Depth**: Linear in depth (but capped at 512)

**Multi-GPU**: Currently limited to first GPU found

### Memory Usage

**Typical**:
- Shader store: 100-500 MB (depends on number/size of shaders)
- eBPF ringbuffer: 512 MB (fixed)
- Symbol caches: 50-200 MB (per-PID symbol tables)
- **Total**: 1-2 GB

## Platform Support

### Supported Hardware

| Platform | Architecture | Driver | Status |
|----------|-------------|--------|--------|
| Ponte Vecchio (PVC) | Xe HPC | i915 (patched) | Fully supported |
| Battlemage (BMG) | Xe2 | Xe | Fully supported |
| Lunar Lake | Xe2 | Xe | Supported (iGPU) |
| Meteor Lake | Xe | Xe | Partial support |
| Older Gen9-Gen12 | Various | i915 | Limited support |

### Kernel Requirements

- **Linux 5.8+**: eBPF CO-RE support
- **BTF**: `/sys/kernel/btf/vmlinux` must exist
- **i915 debug**: Requires Intel backport kernel with `PRELIM_DRM_I915_DEBUG` patches
- **Xe debug**: Mainline support in Linux 6.2+, full support in 6.6+

## Future Enhancements

### Planned Features

1. **Multi-GPU Profiling**: Profile multiple GPUs simultaneously
2. **JSON Output**: Structured output for analysis tools
3. **Differential Profiling**: Compare two profile runs
4. **Continuous Profiling**: Lower overhead, always-on mode
5. **TUI**: Terminal UI for live profiling
6. **Remote Profiling**: Profile remote systems
7. **Container Support**: Profile containerized GPU workloads

### Research Directions

1. **ML-Based Bottleneck Detection**: Automatically identify performance issues
2. **Predictive Profiling**: Suggest optimizations based on patterns
3. **Cross-Platform Support**: NVIDIA, AMD GPU support
4. **Distributed Profiling**: Multi-node GPU cluster profiling

## References

### External Resources

- [FlameGraph](https://www.brendangregg.com/flamegraphs.html) - Original flame graph concept
- [AI Flame Graphs](https://www.brendangregg.com/blog/2024-10-29/ai-flame-graphs.html) - iaprof announcement blog
- [eBPF Documentation](https://ebpf.io/) - eBPF programming guide
- [Intel GPU Architecture](https://www.intel.com/content/www/us/en/developer/articles/technical/overview-of-the-compute-architecture-of-intel-processor-graphics-gen11.html) - Gen11 architecture (similar to PVC/Xe)

### Internal Documentation

- [`src/README.md`](src/README.md) - Source code overview
- [`src/collectors/bpf/README.md`](src/collectors/bpf/README.md) - BPF collector details
- [`src/stores/README.md`](src/stores/README.md) - GPU kernel store internals
- [`docs/README.pvc.md`](docs/README.pvc.md) - PVC-specific setup (if exists)
- [`docs/README.bmg.md`](docs/README.bmg.md) - Battlemage-specific setup (if exists)

## Flame Graph Generation Pipeline

The transformation from raw profiling data to interactive flame graphs involves multiple stages, each adding value and enabling different analyses. Understanding this pipeline helps explain why iaprof uses its particular data formats and how the tool chains together.

### Stage 1: Data Collection (record command)

The `iaprof record` command captures raw profiling data and outputs it in a structured text format. This intermediate format serves as a language between iaprof's internal representation and the flame graph visualization tools.

**Input**: GPU execution + hardware sampling
**Output**: Intermediate profile format (structured text)
**Location**: `src/commands/record.c`, collectors

#### Collected Data Per Sample

For each EU stall sample, the record command collects:

**Process Context**:

- PID, TID, CPU, timestamp
- Process name (16 chars)

**CPU Execution Context**:

- User-space stack (up to 512 frames)
- Kernel stack (up to 512 frames)

**GPU Execution Context**:

- GPU instruction pointer (IP)
- Shader GPU address
- Shader binary (from debug collector)

**Performance Metrics**:

- EU stall type breakdown (active, control, send, sync, etc.)
- Sample counts per stall type

#### Output Format

The record command outputs a custom structured format:

```
STRING;<id>;<string_value>
PROC_NAME;<proc_name_id>;<pid>
USTACK;<ustack_id>;<frame1>;<frame2>;...
KSTACK;<kstack_id>;<frame1>;<frame2>;...
GPU_FILE;<gpu_file_id>;<filename>
GPU_SYMBOL;<gpu_symbol_id>;<symbol_name>
INSN_TEXT;<insn_text_id>;<instruction_text>
EUSTALL;<proc_name_id>;<pid>;<ustack_id>;<kstack_id>;<gpu_file_id>;<gpu_symbol_id>;<insn_text_id>;<stall_type_id>;<offset>;<count>
```

**Key Features**:

The format incorporates several optimizations to reduce file size and improve processing efficiency. String interning ensures that repeated strings (like library names or function names) are stored only once, with subsequent references using compact numeric IDs. Stack deduplication means identical call stacks are assigned the same ID and only serialized once, dramatically reducing output size for workloads with repetitive execution patterns. The structured event system uses different event types (STRING for string definitions, EUSTALL for stall samples, etc.) to clearly delineate different kinds of profiling data.

**Example**:
```
STRING;1;python3.11
STRING;2;libpython3.11.so.1.0
STRING;3;PyEval_EvalFrameDefault
STRING;4;libze_loader.so.1
STRING;5;zeCommandListAppendLaunchKernel
PROC_NAME;1;12345
USTACK;100;0x7f8000001234;0x7f8000005678;0x7f800000abcd
KSTACK;200;0xffffffffc0123456;0xffffffffc0234567
GPU_FILE;10;my_kernel.cpp
GPU_SYMBOL;11;my_compute_kernel
INSN_TEXT;20;mad(8) r10.0<1>:f r2.0:f r3.0:f r4.0:f
EUSTALL;1;12345;100;200;10;11;20;1;0x40;1234
```

This represents:
- Process: python3.11 (PID 12345)
- User stack: 3 frames (resolved later)
- Kernel stack: 2 frames (resolved later)
- GPU kernel: my_compute_kernel in my_kernel.cpp
- Instruction: `mad(8) ...` at offset 0x40
- Stall count: 1234 samples

### Stage 2: Intermediate Processing (flame command)

**Input**: Record output (structured text)
**Output**: Folded stacks (semicolon-separated)
**Location**: `src/commands/flame.c`

#### Processing Steps

**Parse Structured Input**:

```c
while (getline(&line, &size, stdin)) {
    event = get_profile_event_func(line);
    switch (event) {
    case PROFILE_EVENT_STRING:
        store_string(id, value);
        break;
    case PROFILE_EVENT_EUSTALL:
        parse_eustall(&result);
        aggregate_sample(result);
        break;
    }
}
```

**Aggregate Samples**:

The flame command uses a hash table to deduplicate identical stack traces, combining their sample counts. The hash key consists of the complete stack fingerprint: process name, PID, user stack ID, kernel stack ID, GPU file, GPU symbol, instruction text, stall type, and instruction offset. The value is simply the accumulated count of how many times this exact stack was observed. This aggregation is crucial for reducing the data volume - a workload might generate millions of samples, but if they come from a few hot paths, they'll collapse down to just a few unique stacks with large counts.

```c
hash_table(eustall_result, uint64_t) flame_counts;

lookup = hash_table_get_val(flame_counts, result);
if (lookup) {
    *lookup += result.samp_count;
} else {
    hash_table_insert(flame_counts, result, result.samp_count);
}
```

**Resolve String IDs**:

Before outputting the final folded stacks, all numeric string IDs must be converted back to their actual string values. The flame command maintains a string table built from the STRING events encountered earlier in the input stream, allowing efficient lookup of any string by its ID.

**Format Folded Stacks**:
   ```c
   const char *flame_fmt = "%s;%u;%s%s-;%s_[G];%s_[G];%s_[g];%s_[g];0x%lx_[g]";

   printf(flame_fmt,
       proc_name,        // Process name
       pid,              // Process ID
       ustack,           // User stack (semicolon-separated)
       kstack,           // Kernel stack (semicolon-separated)
       gpu_file,         // GPU source file
       gpu_symbol,       // GPU kernel name
       insn_text,        // GPU instruction
       stall_type,       // Stall type
       offset);          // Instruction offset
   printf(";%lu\n", count);  // Sample count
   ```

#### Folded Stack Format

Format: `frame1;frame2;...;frameN count`

**Annotations** (for coloring):
- `_[G]`: GPU frame (uppercase G) - Major GPU components
- `_[g]`: GPU instruction frame (lowercase g) - Individual instructions
- `-`: Stack separator (between user and kernel)

**Example Output**:
```
python3.11;12345;PyEval_EvalFrameDefault;zeCommandListAppendLaunchKernel-;i915_gem_do_execbuffer_[G];my_kernel.cpp_[G];my_compute_kernel_[G];mad(8)_[g];active_[g];0x40_[g];1234
```

Breaking down this stack (bottom to top):

```
1. python3.11 - Process name
2. 12345 - PID
3. PyEval_EvalFrameDefault - Python interpreter frame
4. zeCommandListAppendLaunchKernel - Level Zero API call
5. - (dash) - User/kernel separator
6. i915_gem_do_execbuffer - Kernel driver function
7. my_kernel.cpp - GPU source file
8. my_compute_kernel - GPU kernel function
9. mad(8) - GPU instruction (multiply-add, 8-wide SIMD)
10. active - Stall type
11. 0x40 - Instruction offset
12. Count: 1234 samples
```

### Stage 3: Flame Graph Rendering (flamegraph.pl)

**Input**: Folded stacks
**Output**: Interactive SVG
**Location**: `deps/flamegraph/flamegraph.pl`

This is Brendan Gregg's FlameGraph toolkit, invoked as:
```bash
flamegraph.pl --colors=gpu --title="AI Flame Graph" < folded_stacks.txt > flame.svg
```

#### Rendering Process

**Parse Folded Stacks**:

```perl
while (<STDIN>) {
    chomp;
    if (/^(.*)\s+(\d+)$/) {
        $stack = $1;
        $count = $2;
        @frames = split /;/, $stack;
        accumulate(\@frames, $count);
    }
}
```

**Build Call Tree**:

From the folded stacks, FlameGraph.pl constructs a hierarchical tree structure where each node contains a function name, sample count, and list of children. As it processes each stack, it merges common prefixes - stacks that share the same initial call path are represented by the same nodes in the tree, with branches only where execution diverges. This tree structure is what enables the characteristic flame graph visualization where wide bases represent common entry points and narrower tops show specialized execution paths.

**Calculate Widths**:

The visual width of each frame in the flame graph is proportional to its sample count. The algorithm starts with the total image width (default 1200 pixels) and allocates space to each frame based on its percentage of total samples: `frame_width = (frame_count / total_samples) × image_width`. Frames narrower than a minimum threshold (default 0.1 pixels) are omitted entirely to avoid visual clutter and improve rendering performance.

**Assign Colors**:

Color assignment uses the `--colors=gpu` custom palette optimized for CPU/GPU profiling. The coloring is annotation-based: frames marked with `_[G]` (uppercase) representing major GPU components render in bright blue or cyan, while `_[g]` (lowercase) frames for individual GPU instructions use light blue, stack separator frames marked with `-` appear in gray, and all other frames (CPU code) use warm colors in the yellow/orange/red spectrum. Within each color family, the specific hue varies based on a hash of the function name, ensuring the same function always gets the same color across different profile runs for easy visual comparison.

**Generate SVG**:

The final step produces the interactive SVG visualization. Each frame becomes an SVG rectangle element with calculated x, y, width, and height attributes, filled with the assigned color. A text label containing the function name is overlaid on each rectangle. Embedded JavaScript provides rich interactivity: mouse-over displays a tooltip showing the full function name and percentage of total samples, clicking a frame zooms the view to show just that frame's subtree, Ctrl+F enables searching and highlighting frames by name, and a reset button returns to the full view.

#### SVG Structure

```xml
<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="..." >
  <!-- Title -->
  <text y="24">AI Flame Graph</text>

  <!-- Flame graph frames -->
  <g class="func_g">
    <title>python3.11 (1234 samples, 45%)</title>
    <rect x="0" y="100" width="540" height="16" fill="rgb(230,150,80)" />
    <text x="5" y="110">python3.11</text>
  </g>

  <g class="func_g">
    <title>PyEval_EvalFrameDefault (1000 samples, 36%)</title>
    <rect x="0" y="84" width="450" height="16" fill="rgb(240,160,90)" />
    <text x="5" y="94">PyEval_EvalFrameDefault</text>
  </g>

  <!-- GPU frames (blue) -->
  <g class="func_g">
    <title>my_compute_kernel (500 samples, 18%)</title>
    <rect x="0" y="36" width="225" height="16" fill="rgb(80,150,230)" />
    <text x="5" y="46">my_compute_kernel</text>
  </g>

  <!-- GPU instruction (light blue) -->
  <g class="func_g">
    <title>mad(8) r10 (200 samples, 7%)</title>
    <rect x="0" y="20" width="90" height="16" fill="rgb(150,200,250)" />
    <text x="5" y="30">mad(8) r10</text>
  </g>

  <!-- JavaScript for interactivity -->
  <script type="text/ecmascript">
    <![CDATA[
      function zoom(e) { ... }
      function search() { ... }
      function reset() { ... }
    ]]>
  </script>
</svg>
```
