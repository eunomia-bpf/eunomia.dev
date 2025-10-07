# iaprof: AI/GPU Flame Graph Profiler source code analysis
## Background and Motivation

### The GPU Performance Visibility Problem

Modern AI and GPU workloads present unique profiling challenges:

1. **Execution Disconnect**: Application code runs on the CPU while compute-intensive work executes on the GPU, creating a visibility gap between what developers write and where performance bottlenecks actually occur.

2. **Deep Stack Complexity**: A single GPU kernel launch involves multiple software layers:
   - High-level frameworks (PyTorch, TensorFlow, JAX)
   - Runtime libraries (Level Zero, Vulkan, OpenCL, CUDA)
   - Kernel drivers (i915, Xe)
   - GPU hardware

3. **Instruction-Level Bottlenecks**: GPU performance issues often manifest at specific shader instructions (memory access patterns, control flow, synchronization), not just at the kernel level.

4. **Hardware Sampling Overhead**: Traditional profilers either:
   - Use software instrumentation (high overhead, changes behavior)
   - Provide only GPU-side metrics (disconnected from CPU code)
   - Require code modifications (not practical for production workloads)

### Why Existing Tools Fall Short

**General-Purpose Profilers** (perf, VTune):
- Show CPU execution well but treat GPU as a black box
- Cannot attribute GPU stalls to specific CPU call paths
- Miss the correlation between framework code and GPU bottlenecks

**GPU-Specific Profilers** (Intel GPA, nvidia-smi, rocprof):
- Provide detailed GPU metrics but lack CPU context
- Don't show which application code triggered slow kernels
- Require manual correlation between CPU and GPU timelines

**Flame Graphs** (traditional):
- Excellent for CPU profiling (Brendan Gregg's innovation)
- Don't integrate GPU execution
- Can't show instruction-level GPU performance

### The iaprof Solution

iaprof bridges this gap by creating **AI Flame Graphs** - a unified visualization that:

✓ **Connects CPU to GPU**: Shows complete call stacks from Python/C++ application code down to specific GPU instructions
✓ **Uses Hardware Sampling**: Low-overhead EU (Execution Unit) stall sampling via Intel's Observability Architecture
✓ **Requires No Code Changes**: Uses eBPF kernel tracing to intercept GPU driver calls transparently
✓ **Instruction-Level Attribution**: Associates hardware stall samples with specific shader instructions and CPU call paths
✓ **Interactive Visualization**: Generates clickable, searchable flame graphs for rapid bottleneck identification

### Key Innovation: The Complete Stack

Traditional profilers show either CPU *or* GPU. iaprof shows the **complete execution path**:

```
Python Application
    ↓
PyTorch Framework
    ↓
Level Zero Runtime
    ↓
Xe/i915 Kernel Driver  ← eBPF tracing captures this
    ↓
GPU Hardware           ← OA sampling captures this
    ↓
Shader Instructions    ← Debug API captures binaries
    ↓
EU Stall Metrics       ← Hardware counters
```

All of this appears as a single, unified flame graph where you can:
- Click a GPU instruction to see which CPU function called it
- Search for a framework function to see which GPU code it triggered
- Identify the slowest shader instruction and trace back to source code

## Executive Summary

iaprof is a sophisticated GPU profiling tool that creates **AI Flame Graphs** - interactive visualizations linking CPU execution to GPU performance bottlenecks at the instruction level. It achieves this by combining:

1. **eBPF kernel tracing** - Intercepts GPU driver calls to track kernel launches and memory mappings without code changes
2. **Hardware sampling** - Collects EU (Execution Unit) stall samples via Intel's Observability Architecture with <5% overhead
3. **Debug API** - Retrieves shader binaries and detailed execution information for instruction-level analysis
4. **Symbol resolution** - Resolves CPU and kernel stacks to human-readable symbols with file/line information

The result is an interactive flame graph showing the complete execution path from application code (Python, C++, etc.) through runtime libraries (PyTorch, Level Zero, Vulkan) and kernel driver to specific GPU shader instructions, each annotated with hardware stall metrics (memory access, control flow, synchronization, etc.).

**Use Cases**:
- Optimize AI training/inference workloads
- Identify GPU bottlenecks in rendering pipelines
- Understand framework overhead in ML applications
- Validate GPU compiler optimizations
- Performance regression testing

**Supported Hardware**: Intel Data Center GPU Max (Ponte Vecchio), Intel Arc B-series (Battlemage), Xe2-based GPUs

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

**a) Execution Buffer Tracing** (`i915/execbuffer.bpf.c`, `xe/exec.bpf.c`)

Hooks: `i915_gem_do_execbuffer` (fexit) / `xe_exec_ioctl` (fexit)

Captures:
- Execution buffer ID
- VM and context IDs
- CPU user-space stack (via `bpf_get_stack` with `BPF_F_USER_STACK`)
- Kernel stack (via `bpf_get_stack`)
- Process metadata (PID, TID, CPU, timestamp, command name)
- Batch buffer location (GPU address, CPU mapping)

Sends: `struct execbuf_info` to ringbuffer

**b) Batch Buffer Parser** (`batchbuffer.bpf.c`)

Runs in: Same hook context as execbuffer

Process:
1. Translates GPU batch buffer address to CPU address using `gpu_cpu_map`
2. Reads batch buffer from userspace memory (`bpf_probe_read_user`)
3. Parses GPU commands DWORD-by-DWORD:
   - Identifies command type from bits 31:29
   - Calculates command length (static or dynamic based on command)
   - Follows `BATCH_BUFFER_START` to chained buffers (up to 3 levels deep)
   - Extracts Kernel State Pointers (KSPs) from `COMPUTE_WALKER` and 3DSTATE commands
   - Extracts System Instruction Pointer (SIP) from `STATE_SIP`
4. If buffer incomplete (NOOPs encountered): Defers parsing with BPF timer
5. Emits KSPs and SIP to ringbuffer

Challenges:
- eBPF verifier constraints (max instruction count, bounded loops)
- Buffer may not yet be written when execbuffer called
- Nested batch buffers require state tracking

**c) Memory Mapping Tracing** (`i915/mmap.bpf.c`, `xe/mmap.bpf.c`)

Hooks:
- `i915_gem_mmap_offset_ioctl` (fexit) - Records fake offset
- `i915_gem_mmap` (fexit) - Captures CPU address, associates with handle
- `unmap_region` (fentry) - Cleans up mappings on munmap

Maintains: `gpu_cpu_map` (GPU addr → CPU addr) and `cpu_gpu_map` (reverse)

**d) VM Bind Tracing** (`i915/vm_bind.bpf.c`, `xe/vm_bind.bpf.c`)

Hooks:
- `i915_gem_vm_bind_ioctl` / `xe_vm_bind` (fexit)
- `i915_gem_vm_unbind_ioctl` / `xe_vm_unbind` (fentry)

Purpose: For discrete GPUs, maps GPU virtual addresses to buffer handles and CPU addresses

**e) Context Tracking** (`i915/context.bpf.c`, `xe/context.bpf.c`)

Tracks: Context ID → VM ID mapping

Needed: To resolve which VM an execbuffer belongs to

#### eBPF Maps

Key BPF maps:

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

1. **Initialization**:
   - Loads compiled eBPF object file (`.bpf.o`)
   - Resolves BTF relocations
   - Attaches programs to kernel tracepoints
   - Creates ringbuffer and maps file descriptors

2. **Event Loop** (runs in dedicated thread):
   - `epoll_wait()` on ringbuffer file descriptor
   - `ring_buffer__poll()` to consume events
   - Dispatches to event handlers based on event type

3. **Event Handlers**:
   - **EXECBUF**: Creates or updates shader in GPU kernel store
   - **EXECBUF_END**: Marks batch buffer parsing complete
   - **KSP**: Adds kernel pointer to shader
   - **SIP**: Adds system routine pointer
   - **UPROBE_*** (if enabled): User-space probe events

### 3. EU Stall Collector

**Location**: `src/collectors/eustall/eustall_collector.c`

**Purpose**: Collect and attribute hardware EU stall samples

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

From hardware sampling, categorized as:

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

When EU stall arrives before corresponding shader metadata:

```c
struct deferred_eustall {
    struct eustall_sample sample;
    int satisfied;  // 0 = still waiting, 1 = attributed
};
```

Separate thread periodically retries deferred samples when new shaders arrive.

### 4. Debug Collector

**Location**: `src/collectors/debug/debug_collector.c`

**Purpose**: Interface with GPU debug API for detailed execution information

#### Capabilities

1. **Shader Binary Collection**:
   - Receives shader upload events from driver
   - Copies shader binary to shader store
   - Enables instruction-level disassembly

2. **Context Tracking**:
   - Monitors context create/destroy events
   - Tracks VM associations

3. **Execution Control** (for deterministic sampling):
   - Can SIGSTOP GPU processes for synchronized sampling
   - Triggers EU attention events

4. **Platform-Specific Interfaces**:
   - **i915**: Uses `PRELIM_DRM_I915_DEBUG_*` ioctls (requires patched kernel)
   - **Xe**: Uses `DRM_XE_EUDEBUG_*` ioctls (mainline support)

### 5. OA Collector

**Location**: `src/collectors/oa/oa_collector.c`

**Purpose**: Initialize Observability Architecture hardware

#### Initialization Sequence

1. Open DRM device
2. Write to OA control registers:
   - Enable OA unit
   - Configure sampling period
   - Select counter set (EU stalls)
   - Enable triggers
3. Start sampling

Interfaces with EU stall collector to provide sample stream.

### 6. GPU Kernel Store

**Location**: `src/stores/gpu_kernel.c`

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

Two-level locking:
1. RW-lock on tree (allows concurrent reads, exclusive writes)
2. Mutex per shader (protects shader fields)

**Operations**:

- `acquire_containing_shader(addr)`: Range query, finds shader where `addr ∈ [gpu_addr, gpu_addr+size)`
- `acquire_or_create_shader(addr)`: Gets existing or allocates new
- `release_shader(shader)`: Unlocks shader mutex

### 7. Stack Printer

**Location**: `src/printers/stack/stack_printer.c`

**Purpose**: Transform collected data into folded stacks for flame graphs

#### Symbol Resolution

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

For each shader:
  For each offset with stall counts:
    1. Resolve CPU user stack to symbols: `main;worker;submit_kernel`
    2. Append GPU kernel name: `main;worker;submit_kernel;my_compute_kernel`
    3. Disassemble GPU instruction at offset: `mad(8) r10.0<1>:f ...`
    4. Format stall counts: `active=1234,send=567,control=89`
    5. Output: `main;worker;submit_kernel;my_compute_kernel;mad(8) active=1234,send=567`

**Optimizations**:
- Stack string caching (hash table: stack → folded string)
- Symbol caching (per-PID symbol tables)
- Batch symbol lookups

## Execution Flow

### Startup Sequence

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

### 1. Why eBPF?

**Pros**:
- No kernel module required
- Safe (verified by kernel)
- High performance (JIT compiled)
- Comprehensive tracing capabilities

**Cons**:
- Complexity constraints (verifier limits)
- Requires BTF type information
- Requires recent kernel (5.8+)

### 2. Why In-Kernel Batch Buffer Parsing?

**Alternative**: Parse in userspace after capturing buffer

**Chosen**: Parse in kernel with eBPF

**Rationale**:
- Userspace copy would be expensive (buffers can be MB)
- Need to capture before buffer reused/unmapped
- Deferred parsing handles incomplete buffers

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
- Multiple readers can traverse concurrently
- Stall attribution doesn't block other operations
- Fine-grained locking reduces contention

### 5. Why Folded Stack Format?

**Alternative**: Custom binary format, JSON, protobuf

**Chosen**: Text-based folded stacks

**Rationale**:
- Compatible with existing FlameGraph tools
- Human-readable
- Easy to process with standard tools (grep, sed, awk)
- Compact (deduplication via semicolon syntax)

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

The transformation from raw profiling data to interactive flame graphs involves multiple stages, each adding value and enabling different analyses.

### Stage 1: Data Collection (record command)

**Input**: GPU execution + hardware sampling
**Output**: Intermediate profile format (structured text)
**Location**: `src/commands/record.c`, collectors

#### Collected Data Per Sample

For each EU stall sample, the record command collects:

1. **Process Context**:
   - PID, TID, CPU, timestamp
   - Process name (16 chars)

2. **CPU Execution Context**:
   - User-space stack (up to 512 frames)
   - Kernel stack (up to 512 frames)

3. **GPU Execution Context**:
   - GPU instruction pointer (IP)
   - Shader GPU address
   - Shader binary (from debug collector)

4. **Performance Metrics**:
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
- **String Interning**: Repeated strings stored once with ID references
- **Stack Deduplication**: Identical stacks share same ID
- **Structured Events**: Different event types (STRING, EUSTALL, etc.)

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

1. **Parse Structured Input**:
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

2. **Aggregate Samples**:
   - Uses hash table to deduplicate identical stacks
   - Key: `(proc_name, pid, ustack_id, kstack_id, gpu_file, gpu_symbol, insn_text, stall_type, offset)`
   - Value: Accumulated count
   ```c
   hash_table(eustall_result, uint64_t) flame_counts;

   lookup = hash_table_get_val(flame_counts, result);
   if (lookup) {
       *lookup += result.samp_count;
   } else {
       hash_table_insert(flame_counts, result, result.samp_count);
   }
   ```

3. **Resolve String IDs**:
   - Converts numeric IDs back to strings
   - Looks up in string table built from STRING events

4. **Format Folded Stacks**:
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
1. `python3.11` - Process name
2. `12345` - PID
3. `PyEval_EvalFrameDefault` - Python interpreter frame
4. `zeCommandListAppendLaunchKernel` - Level Zero API call
5. `-` - User/kernel separator
6. `i915_gem_do_execbuffer` - Kernel driver function
7. `my_kernel.cpp` - GPU source file
8. `my_compute_kernel` - GPU kernel function
9. `mad(8)` - GPU instruction (multiply-add, 8-wide SIMD)
10. `active` - Stall type
11. `0x40` - Instruction offset
12. Count: `1234` samples

### Stage 3: Flame Graph Rendering (flamegraph.pl)

**Input**: Folded stacks
**Output**: Interactive SVG
**Location**: `deps/flamegraph/flamegraph.pl`

This is Brendan Gregg's FlameGraph toolkit, invoked as:
```bash
flamegraph.pl --colors=gpu --title="AI Flame Graph" < folded_stacks.txt > flame.svg
```

#### Rendering Process

1. **Parse Folded Stacks**:
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

2. **Build Call Tree**:
   - Creates hierarchical tree structure
   - Each node: `{name, count, children}`
   - Merges common prefixes (same call path)

3. **Calculate Widths**:
   - Total width = image width (default 1200px)
   - Each frame width = (count / total_samples) × image_width
   - Frames < minimum width omitted (default 0.1px)

4. **Assign Colors**:
   - `--colors=gpu` uses custom GPU palette
   - Annotation-based coloring:
     - `_[G]`: Bright blue/cyan (major GPU components)
     - `_[g]`: Light blue (GPU instructions)
     - `-`: Gray (stack separators)
     - Others: Warm colors (yellow/orange/red)
   - Hue varies by function name hash (consistent across runs)

5. **Generate SVG**:
   - Rectangle per frame: `<rect x="..." y="..." width="..." height="16" fill="..." />`
   - Text label: `<text x="..." y="...">function_name</text>`
   - JavaScript for interactivity:
     - Mouse-over: Tooltip with function name + percentage
     - Click: Zoom to subtree
     - Ctrl+F: Search and highlight
     - Reset zoom button

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

### Complete End-to-End Example

**1. Run Profiling**:
```bash
sudo iaprof record > profile.txt
# ... run GPU workload ...
# Ctrl-C to stop
```

**2. Profile Output** (profile.txt, excerpt):
```
STRING;1;my_app
STRING;2;main
STRING;3;compute_kernel
STRING;4;libze_loader.so.1
STRING;5;zeCommandListAppendLaunchKernel
PROC_NAME;1;12345
USTACK;100;0x401234;0x4056ab;0x7f80001234
GPU_SYMBOL;10;compute_kernel
INSN_TEXT;20;send(8) null r20
EUSTALL;1;12345;100;0;0;10;20;0;0x80;5000
```

**3. Generate Folded Stacks**:
```bash
iaprof flame < profile.txt > folded.txt
```

**4. Folded Output** (folded.txt):
```
my_app;12345;main;zeCommandListAppendLaunchKernel-;compute_kernel_[G];send(8)_[g];0x80_[g];5000
```

**5. Generate Flame Graph**:
```bash
flamegraph.pl --colors=gpu --title="My App GPU Profile" < folded.txt > flame.svg
```

**6. View in Browser**:
```bash
firefox flame.svg
```

### Flame Graph Interpretation

#### Visual Encoding

- **X-axis**: Alphabetical ordering (NOT time!)
- **Y-axis**: Stack depth (bottom = entry point, top = sampled function)
- **Width**: Proportional to sample count (wider = more time/samples)
- **Color**:
  - Warm (yellow/orange/red): CPU code
  - Blue: GPU major components
  - Light blue: GPU instructions
  - Gray: Stack separators

#### Reading the Graph

**Identify Hotspots**:
- Wide towers = frequently sampled = hot code paths
- Look for wide boxes at the top = time spent in that function

**Trace Call Paths**:
- Bottom to top = caller to callee
- Follow a tower upward to see full call stack

**Compare Alternatives**:
- Wide towers for function A vs narrow for function B = A is slower
- Example: Three matrix multiply implementations side-by-side

**Find Optimization Targets**:
- Wide GPU instruction frames = stall-heavy instructions
- Wide CPU frames before GPU = submission overhead
- Many small boxes = fragmented execution

### Advanced Features

#### FlameScope (Time-Series Flame Graphs)

**Command**:
```bash
iaprof record --interval 100 > profile.txt
iaprof flamescope < profile.txt
```

**Output**: Multiple flame graphs, one per 100ms interval

**Use Cases**:
- Warm-up analysis (first intervals vs later)
- Phase detection (compute phase vs memory phase)
- Temporal patterns (periodic spikes)

#### Differential Flame Graphs

**Workflow**:
1. Profile baseline: `iaprof record > baseline.txt`
2. Apply optimization
3. Profile optimized: `iaprof record > optimized.txt`
4. Generate diff: `difffolded.pl baseline.txt optimized.txt | flamegraph.pl --negate > diff.svg`

**Colors**:
- Red: Increased in optimized version (regression)
- Blue: Decreased in optimized version (improvement)

#### Search and Filter

In the interactive SVG:
- **Ctrl+F**: Search for function/pattern
- **Click frame**: Zoom to subtree
- **Mouse-over**: See exact sample counts and percentages

### Customization Options

#### Color Palettes

```bash
# GPU-specific (default for iaprof)
flamegraph.pl --colors=gpu

# Memory access patterns
flamegraph.pl --colors=mem

# I/O operations
flamegraph.pl --colors=io

# Java-specific
flamegraph.pl --colors=java
```

#### Sizing

```bash
# Larger image
flamegraph.pl --width=2400 --height=24

# Filter small functions
flamegraph.pl --minwidth=0.5  # 0.5% of total time
```

#### Titles and Labels

```bash
flamegraph.pl \
  --title="PyTorch Training Profile" \
  --subtitle="Model: ResNet-50, Batch: 64" \
  --countname="stalls" \
  --nametype="Frame:"
```

### Performance Considerations

#### Flame Command

- **Aggregation**: O(n) where n = number of samples
- **Hash table**: O(1) average lookup
- **Memory**: ~100 bytes per unique stack
- Typical: 1M samples → 5-10 seconds, <1GB RAM

#### FlameGraph.pl

- **Parsing**: O(n) where n = folded stack lines
- **Tree building**: O(n × d) where d = avg stack depth
- **SVG generation**: O(m) where m = unique frames
- Typical: 100K unique stacks → 10-30 seconds, <500MB RAM

#### Optimization Tips

1. **Filter during collection**: Use `--quiet` to reduce overhead
2. **Interval-based**: Use `--interval` for time-series, reduces total data
3. **Post-process filtering**: `grep` folded stacks before flamegraph.pl
4. **Parallel processing**: Multiple `flamegraph.pl` invocations for different filters

### Troubleshooting

#### Missing Symbols

**Problem**: Flame graph shows hex addresses instead of function names

**Solutions**:
- Install debug symbols: `apt install libc6-dbg`
- Compile with frame pointers: `-fno-omit-frame-pointer`
- Check BTF available: `ls /sys/kernel/btf/vmlinux`

#### Incomplete Stacks

**Problem**: Stacks end prematurely (don't reach main)

**Solutions**:
- Increase stack depth: `sysctl kernel.perf_event_max_stack=512`
- Enable frame pointers in all libraries
- Check for frame pointer optimizations

#### Wrong Colors

**Problem**: GPU frames not colored blue

**Solutions**:
- Ensure `--colors=gpu` flag
- Check annotations in folded stacks (`_[G]`, `_[g]`)
- Verify flame command output format

## Glossary

- **eBPF**: Extended Berkeley Packet Filter - Linux kernel technology for safe, JIT-compiled programs
- **EU**: Execution Unit - GPU compute core
- **BTF**: BPF Type Format - Compact type information for eBPF
- **OA**: Observability Architecture - Intel GPU hardware sampling infrastructure
- **KSP**: Kernel State Pointer - GPU address of shader/kernel binary
- **SIP**: System Instruction Pointer - GPU address of system routine (exception handler)
- **Batch Buffer**: GPU command stream
- **Folded Stack**: Text format for flame graphs: `func1;func2;func3 count`
- **VM**: Virtual Memory - GPU virtual address space
- **DRM**: Direct Rendering Manager - Linux kernel graphics subsystem
- **String Interning**: Technique to store unique strings once, reference by ID
