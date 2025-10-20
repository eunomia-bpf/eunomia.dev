# GPU OS

Bg: we see many paper providing scheduling/memory management policy/other OS policies, like SOSP 25 LithOS

《
Are actually working as an intercept layer between OS kernel driver and usersapce library,

So we are thinking, can bpftime be something like “An extensible *GPU OS”?* 

It can provide fine-grain, corss layer GPU policy, like scheduler.

Some questions:

- There are a number of papers on Scheduling and Memory management as well?
- How does the Linux kernel currently support custom scheduling using eBPF for the CPU (is there inspiration that we should take from them?)
- Think and brainstorm: What does the wrold of the GPU stack look like? How does prior work fit into this picture? How would our work fit into this picture?
- Think and brainstorm: what does a GPU OS need to support? What are the abstractions that we want to provide to people?

SCX decouples **mechanism** (core runqueue operations, invariants) from **policy** (pick-next, placement, latency/throughput tradeoffs), enforced by the BPF verifier and a stable hook surface. This is exactly the recipe GPUs lack.

```c
SCX_OPS_DEFINE(simple_ops,
           .select_cpu  = (void *)simple_select_cpu,
           .enqueue   = (void *)simple_enqueue,
           .dispatch  = (void *)simple_dispatch,
           .running   = (void *)simple_running,
           .stopping  = (void *)simple_stopping,
           .enable   = (void *)simple_enable,
           .init   = (void *)simple_init,
           .exit   = (void *)simple_exit,
           .name   = "simple");
```

https://eunomia.dev/tutorials/44-scx-simple/

**Direct inspiration for GPUs:**

1. Define **stable hook points** around *submit, admit, dispatch, preempt, retire* of GPU kernels/graphs/copies.
2. Provide typed state via **maps** (per-tenant credits, per-kernel footprints, power/thermal ceilings).
3. Enforce safety with a **verifier** (no illegal MMIO, bounded loops, bounded metadata).
4. Allow **live swapping** of scheduling/memory/IO policies (“gBPF”).
5. Ship a **baseline policy** (like CFS): fair-share with SLA priorities; let sites replace it.
    
    (SCX shows this is socially and technically feasible in Linux.)

https://chatgpt.com/share/68f67566-de48-8009-8296-a99e52bc1e35

https://chatgpt.com/share/68f2dbaa-880c-8009-b770-d57faafe576f

## Prior work:

Per your request, I only add a **minimal** “OUR INSERTION POINT” note and keep the figure focused on the stack + prior work.

```
                               ┌─────────────────────────────────────────────────────────────────────┐
                               │                    CLUSTER / CONTROL PLANE                          │
                               │ Orchestrators: Kubernetes, Slurm                                    │
                               │ DL schedulers (cluster-level): [Gandiva], [Themis]                  │
                               └─────────────────────────────────────────────────────────────────────┘
                                                     │ submits jobs / allocates GPUs
                                                     ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                         ML/DATA/GRAPH FRAMEWORKS & SERVICES                                          │
│  PyTorch / TensorFlow / JAX / Triton / Inference servers (Triton-IS, TorchServe, etc.)                               │
│  Framework-level sharing: [Salus] vllm, Pie， ktransformer                                                           │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                     │ CUDA/ROCm/NCCL API calls (kernels, graphs, copies, collectives)
                                                     ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                             GPU RUNTIMES & LIBRARIES                                                 │
│  CUDA / ROCm runtimes; Streams/Graphs; NCCL collectives                                                              │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                     │ intercepted by user-space shims / driver plug-ins
                                                     ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   INTERCEPT LAYER (user-space / kernel-adjacent shims)                               │
│  [Orion, EuroSys’24]  – Dyn-linked library intercepts GPU ops; interference-aware scheduling                         │
│  [XSched, OSDI’25]   – Shim intercepts platform APIs; preemptible XQueue across diverse XPUs                         │
│  [LithOS, 2025]      – Lib-level & driver-adjacent control; TPC-aware scheduling; kernel atomization                 │
│  [REEF, OSDI’22]     – Host/device queue reset for μs-scale preemption; inference serving                            │
│                                                                                                                      │
│                                   ← OUR INSERTION POINT (thin interpose surface)                                     │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                     │ ioctl / UAPI / perf streams / telemetry
                                                     ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       HOST OS & VENDOR DRIVER INFRASTRUCTURE                                         │
│  Linux kernel + vendor GPU kernel driver                                                                             │
│    - Production sharing/partitioning:  MIG (spatial), MPS (temporal), vGPU/SR‑IOV (virtualization)                   │
│    - Remote/virt stacks: [rCUDA], [GVirtuS]                                                                          │
│    - Driver-level scheduling & OS mgmt: [TimeGraph], [Gdev]                                                          │
│    - OS services exported to GPU code: [GPUfs] (filesystem), [GPUnet] (sockets)                                      │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                     │ control messages / doorbells / perf counters
                                                     ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                            ON-DEVICE FIRMWARE / CONTROLLERS                                          │
│  GPU system processors (e.g., NVIDIA GSP), power/thermal mgmt, micro-schedulers                                      │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                     │ config registers / queues / memory mappings
                                                     ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                     GPU HARDWARE                                                     │
│  Compute Complex                                                                                                     │
│   ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│   │  SMs / TPCs / GPCs; warp schedulers; CTAs,  CLC api/gpu-graph-scheduler api                                  │   │                               │
│   │  Research here: [Kernelet] (kernel slicing), [NEEF], [XSched],                                               │   │
│   └──────────────────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                                      │
│  Memory Complex                                                                                                      │
│   ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│   │  HBM stacks, L2, MMU/TLBs, compression                                                                       │   │
│   │  Research here: [Mosaic] (multi page sizes), [MASK] (translation-aware), [Zorua] (virt),                     │   │
│   │                 [ETC] (oversubscription), [Buddy Compression] (capacity via compression)                     │   │
│   └──────────────────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                                      │
│  I/O & Interconnect                                                                                                  │
│   ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│   │  Copy/DMA engines, PCIe, NVLink/NVSwitch, GPUDirectIO; NCCL uses this                                        │   │
│   └──────────────────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                                      │
│  Cross-cutting Security                                                                                              │
│   ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│   │  TEEs / attestation / isolation: [Graviton]                                                                  │   │
│   └──────────────────────────────────────────────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

prior work:
  [Orion]  EuroSys’24, intercept library in user space
  [XSched] OSDI’25, preemptible command queues via shim across GPUs/NPUs/ASICs/FPGAs
  [LithOS] 2025, lib/driver-adjacent control (TPC scheduling, atomization)
  [REEF]   OSDI’22, μs-scale preemption via host/device/GPU reset

  [TimeGraph] driver-level real-time scheduling               [Gdev] OS-level GPU resource mgmt
  [GPUfs] filesystem to GPU code                              [GPUnet] sockets for GPU code
  [Gandiva], [Themis] cluster DL schedulers                   [Salus] framework-level sharing
  [Kernelet] kernel slicing                                   [Mosaic], [MASK], [Zorua], [ETC], [Buddy Compression] memory/VM
  [rCUDA], [GVirtuS] remote/virtual GPUs                      [Graviton] GPU TEEs / security

```

## Our picture:

- **gBPF‑H (host policy plane)**
    
    Lives in a small **GPU OS** between CUDA/ROCm and the vendor driver. Implements **admission**, **vSlice partitioning**, **MemPool quotas**, **I/O budgets**, **SLA logic**, and **observability governance**. Exposes verified, hot‑swappable policies (classic eBPF model).
    
- **gBPF‑D (device policy plane)**
    
    Lives **on offload eBPF to GPU** (e.g. a **persistent control kernel** on a tiny vTPC slice). Implements **CTA admission order**, **warp throttling**, **local DMA bursts**, **thermal reaction**, and **short‑horizon preempt/atomize**—*within quotas* pushed by gBPF‑H.
    

---

```c
                                   Applications (PyTorch/TF/JAX, services)
                                              │   CUDA/ROCm/NCCL API
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│     HOST / CPU DOMAIN — gBPF‑H  (Extensible GPU Control‑Plane inside a small Device OS)     │
│  ┌───────────────────────────────────────────────────────────────────────────────────────┐  │
│  │  Hooks: on_submit, on_admit, on_dispatch, on_preempt, on_retire                       │  │
│  │         on_mem_fault, on_reclaim, on_copy_enqueue, on_collective_phase, on_thermal    │  │
│  │  Policies (verified, hot‑swappable):                                                  │  │
│  │    • Admission / queueing / vSlice partitioning (per‑tenant weights, SLAs)            │  │
│  │    • MemPool quotas, oversub, page‑size policy, compression gating                    │  │
│  │    • IO budgets for DMA/NVLink, collective‑phase QoS                                  │  │
│  │    • Observability policy: sampling, flight recorders, reason codes                   │  │
│  │  Enforcement (trusted microkernel):                                                   │  │
│  │    • Can admit/deny/dispatch/atomize; set MIG/MPS configs; program MMU; set IO caps   │  │
│  │  State:                                                                               │  │
│  │    • Maps: TENANT_CREDITS, JOB_META, VSLICE_SET, MEMPOOL_SET, IO_BUDGETS, OBS_*       │  │
│  └─────────────▲─────────────────────────────────────────────────────────────────────────┘  │
│                │        Control/telemetry rings (BAR1/pinned)  •  Epoched quota updates     │
└────────────────┼────────────────────────────────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│   DEVICE DOMAIN — gBPF‑D  (Fine‑grain policy in eBPF, e.g. persistent control CTA)          │
│  ┌───────────────────────────────────────────────────────────────────────────────────────┐  │
│  │  Hooks (device‑local):                                                                │  │
│  │    • sm_tick, cta_admit, warp_throttle_tick, dma_tick, mem_event, therm_tick          │  │
│  │  Policy (verified, tiny ISA):                                                         │  │
│  │    • Within host quotas: choose next CTA, assign SM, gate warps, reschedule threads   │  │
│  │    • React to local stall/pressure (L1/L2/TLB/occupancy) in sub‑µs                    │  │
│  │  Enforcement:                                                                         │  │
│  │    • Control CTA issuance tokens; local engine pacing; cooperative preemption markers │  │
│  │  State (cache of host):                                                               │  │
│  │    • READ‑ONLY snapshots of quotas/weights; hot counters that roll up to host         │  │
│  └───────────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                 
                 

```

---

## 3) Timescales & division of labor

| Layer | Typical reaction time | Responsible decisions | Examples |
| --- | --- | --- | --- |
| **gBPF‑D (device)** | 50–500 ns (warp) • 0.5–5 µs (CTA/engine) | Pick next CTA/SM; gate warps; micro‑burst DMA; react to stall reasons; thermal throttle within quotas | Avoid HOL blocking; cut P99 spikes during collectives |
| **gBPF‑H (host)** | 5–100 µs (hook→decision) • 1–100 ms (global) | Admit/deny; set vSlice splits; move memory; set IO budgets; SLA policy; cross‑GPU coordination | Reduce tenant interference; reweight bursts; start flight recorder |

---

**Shared maps. e.g.**

- `TENANT_CREDITS{tenant → (weight, burst, vSlice_tokens)}`
- `MEMPOOL{tenant → (hard, soft, page_size_policy, compression)}` *(device sees ceilings only)*
- `IO_BUDGETS{(engine,tenant) → tokens_per_epoch}`
- `PLACEMENT{tenant → sm_mask/vTPC_set}`
- `OBS_CTRL{scope → sampling_level, budget}`

What we are:

1. An extensible *GPU policy runtime?*
2. An extensible *GPU OS?*

## Are CUDA/ROCm + driver some kinds of OS?

- **NVIDIA and AMD don’t call CUDA/ROCm a “library OS.”** They call them *runtimes/toolkits/platforms*.
- **However**, by the **OS literature’s definition** of a *library operating system* (libOS)—“the OS personality runs in the application’s address space as a library” (Drawbridge), with a minimal kernel beneath (Exokernel)—**CUDA/ROCm *behave like*** a libOS **for the GPU domain**: they implement application‑visible services (contexts, module loading/JIT, virtual memory, streams/priorities, sync) in user space on top of a kernel driver. That’s why the analogy is useful for our positioning. [Microsoft+1](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/asplos2011-drawbridge.pdf?utm_source=chatgpt.com)

Below I (1) show evidence that CUDA/ROCm provide OS‑like services as *library* code, (2) spell out the key differences from a “classical” libOS, and (3) summarize how different communities actually talk about them.

---

## 1) Evidence that CUDA/ROCm act like a *library OS* for the GPU “micro‑world”

**A. Process/Context management (naming, protection domain)**

CUDA’s **Driver API** owns **contexts** and explains how the **Runtime API** implicitly creates/uses *primary contexts* per device—exactly the kind of process/environment management a libOS would do on behalf of an app. [NVIDIA Docs+1](https://docs.nvidia.com/cuda/cuda-driver-api/driver-vs-runtime-api.html)

**B. Program loading and JIT linking (like an OS program loader)**

The Driver API exposes **Module Management**: `cuModuleLoad*`, `cuLink*` for JIT linking of PTX/CUBIN, function lookup, and lazy loading—again, the “loader” role a libOS would play. [NVIDIA Docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html)

**C. Virtual memory and unified memory (VM services)**

CUDA provides **virtual memory management** (`cuMemAddressReserve`, `cuMemMap`, shareable handles) and **Unified Memory** semantics (fault‑driven migration, HMM integration)—application‑visible memory policies implemented in the runtime/driver layer. [NVIDIA Docs+1](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html)

**D. Scheduling interface to applications (policy *hints*)**

CUDA and HIP expose **stream priorities** and concurrency control—users set priorities, the GPU scheduler treats them as **hints** (not hard guarantees). That’s the tell: the runtime exports scheduling semantics to apps, even if they’re not strict. [NVIDIA Docs+2NVIDIA Docs+2](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

**E. Discovery, initialization, sync, interop, graphs**

The Driver API’s table of contents makes it plain: **device management, stream/event management, execution control, graphs**, interop, and profiling control—all user‑space library services atop the kernel driver. That’s a libOS‑like surface for the GPU. [NVIDIA Docs](https://docs.nvidia.com/cuda/cuda-driver-api/index.html)

**Why this matches the libOS idea:** In exokernel/libOS designs, a **small kernel** securely multiplexes hardware, while **library OSes** implement higher‑level abstractions *in user space*; Drawbridge’s definition is the standard citation. Functionally, CUDA/ROCm implement the **GPU personality**—the app‑visible API and semantics—while the Linux kernel + vendor driver provide protection and low‑level multiplexing.

---

## Appendix: Term Explanations

**ROCm (Radeon Open Compute)**
AMD's open-source software platform for GPU computing, analogous to CUDA.

**NCCL (NVIDIA Collective Communications Library)**
Library for multi-GPU and multi-node collective communication primitives, optimized for NVIDIA GPUs.

**MIG (Multi-Instance GPU)**
NVIDIA technology that partitions a single GPU into multiple isolated GPU instances with separate memory and compute resources.

**MPS (Multi-Process Service)**
NVIDIA technology enabling temporal sharing of a GPU among multiple CUDA processes with reduced context-switching overhead.

**vGPU (Virtual GPU)**
Technology that allows multiple virtual machines to share physical GPU resources.

**SR-IOV (Single Root I/O Virtualization)**
PCI-SIG standard allowing a single physical device to present itself as multiple virtual devices.

**vSlice**
Virtual slice - a partitioned share of GPU resources (compute, memory, I/O) allocated to a tenant or workload.

**SM (Streaming Multiprocessor)**
The fundamental processing unit in NVIDIA GPUs containing CUDA cores, warp schedulers, and memory caches.

**TPC (Thread Processing Cluster)**
A collection of SMs grouped together with shared resources in NVIDIA GPU architecture.

**GPC (Graphics Processing Cluster)**
Higher-level grouping of TPCs in NVIDIA GPUs, containing multiple TPCs and shared resources.

**CTA (Cooperative Thread Array)**
CUDA's term for a thread block - a group of threads that execute the same kernel and can cooperate via shared memory.

**Warp**
A group of 32 threads in NVIDIA GPUs that execute in lockstep (SIMT - Single Instruction, Multiple Threads).

**HBM (High Bandwidth Memory)**
High-performance 3D-stacked memory technology used in modern GPUs, providing significantly higher bandwidth than DDR.

**NVLink**
NVIDIA's high-bandwidth, low-latency interconnect for GPU-to-GPU and GPU-to-CPU communication.

**NVSwitch**
NVIDIA's switch fabric enabling all-to-all GPU communication at full NVLink speed.

**GPUDirect Storage**
Technology enabling direct data path between GPU memory and storage, bypassing CPU memory.

**UAPI (User-space API)**
Kernel interfaces exposed to user-space programs.

**MMIO (Memory-Mapped I/O)**
Method of accessing hardware registers by mapping them into the processor's address space.

**BAR (Base Address Register)**
PCI configuration registers that define memory/I/O address ranges for device communication.

**GSP (GPU System Processor)**
NVIDIA's on-GPU processor that offloads GPU initialization and management tasks from the CPU driver.

**Runqueue**
Data structure containing processes/tasks ready to execute, managed by the scheduler.

**SIMT (Single Instruction, Multiple Threads)**
Execution model where multiple threads execute the same instruction on different data.

**Kernel (GPU context)**
A function that runs on the GPU, not to be confused with OS kernel.

**Stream (CUDA/ROCm)**
A sequence of operations that execute in order on the GPU.

**Graph (CUDA)**
A representation of GPU operations as a directed acyclic graph, allowing optimizations and reuse.
