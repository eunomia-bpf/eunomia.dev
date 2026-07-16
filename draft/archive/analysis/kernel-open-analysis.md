# NVIDIA Open GPU Kernel Modules - kernel-open/ Architecture Analysis

## Executive Summary

The `kernel-open/` directory contains the Linux kernel-side implementation of NVIDIA's open-source GPU driver stack. This is a sophisticated, multi-module architecture consisting of **208 C source files** and **246 header files**, totaling over **200,000 lines of code**. The architecture implements a layered approach with five primary kernel modules that provide comprehensive GPU management, memory virtualization, display support, and peer-to-peer communication capabilities.

**Version**: 580.95.05
**License**: Dual MIT/GPL
**Minimum Kernel**: Linux 4.15+
**Supported Architectures**: x86_64, arm64, riscv

---

## Architecture Overview

### Module Hierarchy and Dependencies

```
┌─────────────────────────────────────────────────────────────┐
│                     User Space Applications                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ├─────── ioctl interface
                              │
┌─────────────────────────────┴───────────────────────────────┐
│                    Kernel Space Modules                      │
├──────────────────────────────────────────────────────────────┤
│  nvidia.ko         ← Core GPU driver (38,762 LOC)           │
│     ├── nv-kernel.o_binary (proprietary core)               │
│     └── kernel interface layer (open source)                │
│                                                              │
│  nvidia-uvm.ko     ← Unified Virtual Memory (103,318 LOC)   │
│     └── Manages GPU memory, page faults, migrations         │
│                                                              │
│  nvidia-drm.ko     ← DRM/KMS driver (19 files)              │
│     └── Linux DRM subsystem integration                     │
│                                                              │
│  nvidia-modeset.ko ← Mode setting interface (2 files)       │
│     └── Display configuration and control                   │
│                                                              │
│  nvidia-peermem.ko ← Peer memory support (1 file)           │
│     └── RDMA and GPU-to-GPU DMA                             │
└──────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────┴───────────────────────────────┐
│                  Linux Kernel Subsystems                     │
│  • PCI/PCIe          • DMA/IOMMU        • Memory Management │
│  • DRM/KMS           • Power Management  • ACPI/Device Tree │
└──────────────────────────────────────────────────────────────┘
```

---

## 1. nvidia.ko - Core GPU Kernel Driver

**Location**: `kernel-open/nvidia/`
**Files**: 59 C files, ~38,762 lines
**Module**: nvidia.ko
**Purpose**: Primary kernel interface to NVIDIA GPU hardware

### 1.1 Core Architecture

The nvidia module implements a **hybrid architecture** combining:
- **Open-source kernel interface layer** (all files in nvidia/)
- **Proprietary binary core** (nv-kernel.o_binary)

The interface layer acts as a translation layer between:
1. Linux kernel APIs (which change frequently)
2. The stable ABI provided by nv-kernel.o_binary

### 1.2 Key Components and Files

#### Main Driver Core
- **nv.c** (159,862 lines) - Core driver initialization, device management, and control flow
  - Module initialization/cleanup: `nvidia_init_module()`, `nvidia_exit_module()`
  - Device probe/remove: `nvidia_probe()`, `nvidia_remove()`
  - File operations: `nvidia_open()`, `nvidia_close()`, `nvidia_ioctl()`, `nvidia_mmap()`
  - Interrupt handling: `nvidia_isr()`, `nvidia_isr_kthread()`
  - Power management: `nvidia_suspend()`, `nvidia_resume()`

#### Memory Management
- **nv-mmap.c** - Memory mapping implementation
  - Maps GPU memory into user space
  - Handles different memory types (framebuffer, registers, system memory)
  - Implements `vm_operations_struct` callbacks

- **nv-vm.c** - Virtual memory operations
  - Page table management
  - Memory access tracking
  - Address translation

- **nv-dma.c** (25,820 lines) - DMA subsystem integration
  - IOMMU support
  - Scatter-gather list management
  - DMA buffer allocation/mapping
  - SWIOTLB handling for bounce buffers

- **nv-dmabuf.c** (49,107 lines) - DMA-BUF framework integration
  - Cross-device buffer sharing
  - Import/export operations
  - Fence synchronization

#### PCI/PCIe Support
- **nv-pci.c** - PCI device interface
  - PCI device enumeration
  - Resource management (BAR mapping)
  - MSI/MSI-X interrupt setup
  - Power management (D0/D3 states)

- **os-pci.c** - PCI operations abstraction
  - Config space access
  - Resizable BAR support
  - ATS (Address Translation Services)
  - AER (Advanced Error Reporting)

- **nv-pci-table.h** - PCI device ID table
  - Lists all supported GPU device IDs

#### Platform Support
- **nv-platform.c** - Platform device support (Tegra SoCs)
  - Device tree parsing
  - Platform-specific initialization
  - Resource allocation

- **nv-acpi.c** (41,810 lines) - ACPI integration
  - ACPI method evaluation (_DSM, _ROM, etc.)
  - Power resource management
  - Hybrid graphics (Optimus) support
  - Backlight control

#### Specialized Subsystems

**Display and Backlight**:
- **nv-backlight.c** - LCD backlight control
- **nv-clk.c** (29,645 lines) - Clock management for display
- **nv-dsi-parse-panel-props.c** (32,501 lines) - DSI panel configuration

**Tegra-Specific**:
- **nv-bpmp.c** - Boot and Power Management Processor interface
- **nv-host1x.c** - Host1x integration (Tegra GPU command submission)
- **nv-gpio.c** - GPIO pin control
- **nv-ipc-soc.c** - Inter-processor communication

**NVLink and NVSwitch**:
- **nvlink_linux.c** - NVLink interconnect support
- **nvlink_caps.c** - NVLink capabilities management
- **linux_nvswitch.c** (61,971 lines) - NVSwitch fabric management
- **i2c_nvswitch.c** - I2C bus management for NVSwitch

**Cryptography (libspdm integration)**:
- **libspdm_*.c** (15 files) - SPDM (Security Protocol and Data Model) implementation
  - AES-GCM encryption
  - RSA/ECC public key crypto
  - SHA hashing
  - HMAC
  - X.509 certificate handling
  - Used for secure device attestation

**Peer-to-Peer**:
- **nv-p2p.c** - Peer-to-peer GPU memory access
  - Exposes GPU memory to third-party drivers
  - Used for RDMA, GPU Direct

**Debugging and Monitoring**:
- **nv-procfs.c** - /proc filesystem interface
  - Driver version info
  - GPU status
  - Registry parameters

- **nv-memdbg.c** - Memory debugging facilities
- **nv-report-err.c** - Error reporting infrastructure

**Utility**:
- **nv-caps.c** - Capability-based security model
- **nv-caps-imex.c** - Import/export channel capabilities
- **nv-i2c.c** - I2C bus implementation
- **nv-kthread-q.c** - Kernel thread queue for deferred work
- **nv-rsync.c** - Resource synchronization
- **nv-msi.c** - MSI/MSI-X interrupt management
- **nv-pat.c** - Page Attribute Table management (x86)
- **nv-usermap.c** - User memory mapping
- **nv-vtophys.c** - Virtual to physical address translation
- **os-interface.c** - OS abstraction layer
- **os-mlock.c** - Memory locking operations
- **os-registry.c** - Registry parameter interface
- **os-usermap.c** - User space memory operations

### 1.3 Key Data Structures

#### nv_state_t
Core per-device state structure (defined in common/inc/nv.h):
- GPU identification (PCI info, UUID)
- Device resources (register mappings, IRQs)
- Hardware capabilities
- Power state tracking
- Queue management

#### nv_linux_state_t
Linux-specific device state wrapper:
- Extends nv_state_t
- Linux-specific resources (character device, work queues)
- Locking primitives
- Reference counting

### 1.4 Module Initialization Flow

1. **nvidia_init_module()**
   - Register character device
   - Initialize global state
   - Register PCI driver
   - Create /proc entries
   - Initialize UVM interface

2. **nvidia_probe()** (per GPU)
   - Allocate device state
   - Map PCI resources
   - Setup interrupts
   - Initialize RM (Resource Manager) - calls into binary core
   - Register with DRM subsystem
   - Create device nodes

3. **Runtime Operations**
   - ioctl dispatch to RM core
   - Interrupt handling
   - Memory management
   - Power state transitions

### 1.5 Binary Interface

The module links against **nv-kernel.o_binary**, which contains:
- Resource Manager (RM) - main control logic
- Hardware abstraction layer
- GPU initialization and configuration
- Command submission and scheduling
- Proprietary algorithms

The open-source interface layer provides:
- OS primitive implementations (memory, locks, timers)
- Kernel API compatibility
- Platform-specific code

---

## 2. nvidia-uvm.ko - Unified Virtual Memory

**Location**: `kernel-open/nvidia-uvm/`
**Files**: 127 C files, ~103,318 lines
**Module**: nvidia-uvm.ko
**Purpose**: GPU memory virtualization and unified addressing

### 2.1 Overview

UVM provides a **unified virtual address space** spanning CPU and GPU memory, enabling:
- Automatic data migration between CPU and GPU
- On-demand page faulting
- Oversubscription (GPU memory can exceed physical capacity)
- Multi-GPU coherence
- Heterogeneous Memory Management (HMM) integration

### 2.2 Core Architecture

#### Virtual Address Space Management
- **uvm_va_space.c** - VA space lifecycle
  - Per-process GPU address space
  - Multi-GPU management
  - MM (memory management) integration

- **uvm_va_range.c** - Virtual address range management
  - VMA (Virtual Memory Area) tracking
  - Range tree for efficient lookups
  - Policy management (preferred location, accessed-by)

- **uvm_va_block.c** - VA block operations
  - 2MB granularity blocks
  - Page residency tracking
  - Migration state machine
  - Fault handling

#### Memory Management
- **uvm_mem.c** - Memory allocation
  - Kernel memory for driver metadata
  - GPU accessible memory
  - DMA mapping

- **uvm_mmu.c** - MMU management
  - Page table operations
  - TLB management
  - Address translation

- **uvm_migrate.c** - Page migration
  - CPU ↔ GPU migrations
  - GPU ↔ GPU migrations
  - Batched operations
  - Performance optimization

- **uvm_migrate_pageable.c** - Pageable memory migration
  - System memory handling
  - Swap support

#### GPU Page Fault Handling
- **uvm_gpu_replayable_faults.c** - Replayable fault handling
  - Fault buffer management
  - Fault coalescing
  - Replay mechanism
  - Thrashing detection

- **uvm_gpu_non_replayable_faults.c** - Non-replayable faults
  - Fatal error handling
  - Page poisoning

- **uvm_ats_faults.c** (33,966 lines) - ATS fault handling
  - Address Translation Services
  - IOMMU integration
  - SVA (Shared Virtual Addressing)

#### Channel Management
- **uvm_channel.c** (166,151 lines) - GPU command channels
  - Channel allocation/scheduling
  - Push buffer management
  - Semaphore operations
  - Work submission

- **uvm_push.c** - Command push infrastructure
  - GPU method encoding
  - Synchronization

#### GPU Architecture Support

Hardware-specific implementations for each GPU generation:
- **uvm_maxwell*.c** - Maxwell (GM10x)
- **uvm_pascal*.c** - Pascal (GP10x)
- **uvm_volta*.c** - Volta (GV100)
- **uvm_turing*.c** - Turing (TU10x)
- **uvm_ampere*.c** - Ampere (GA100)
- **uvm_ada*.c** - Ada Lovelace
- **uvm_hopper*.c** - Hopper (GH100)
- **uvm_blackwell*.c** - Blackwell (GB100)

Each implements:
- **_mmu.c** - MMU programming
- **_host.c** - Host/CPU interface
- **_fault_buffer.c** - Fault buffer format
- **_ce.c** - Copy Engine operations

#### HAL (Hardware Abstraction Layer)
- **uvm_hal.c** - HAL infrastructure
  - GPU-specific function tables
  - Runtime dispatch based on architecture
  - Virtualized operations

#### Performance Subsystems
- **uvm_perf_*.c** - Performance optimizations
  - Prefetching
  - Thrashing mitigation
  - Access counter sampling
  - Heuristics for migration

#### HMM Integration
- **uvm_hmm.c** - Heterogeneous Memory Management
  - Linux HMM subsystem integration
  - Seamless GPU integration with system memory
  - MMU notifier callbacks

#### ATS/SVA Support
- **uvm_ats.c** (4,561 lines) - Address Translation Services
- **uvm_ats_sva.c** (15,909 lines) - Shared Virtual Addressing
  - IOMMU integration
  - PASID (Process Address Space ID) management
  - Allows GPU to use CPU page tables directly

#### Testing and Verification
- **uvm_test.c** - Test infrastructure
- **uvm_test_*.c** - Component-specific tests
  - Channel tests
  - MMU tests
  - Migration tests
  - Fault handling tests

### 2.3 Key Data Structures

#### uvm_va_space_t
Per-process GPU virtual address space:
- VA range tree
- GPU registration
- MM struct reference
- Lock hierarchy

#### uvm_va_block_t
2MB-aligned memory region:
- Page residency bitmap (where each page lives)
- Processor mask (which GPUs have accessed)
- Migration tracking
- Fault handling state

#### uvm_gpu_t
Per-GPU state:
- Parent association (for multi-GPU)
- Channel manager
- Page tree (GPU page tables)
- Replayable fault buffer
- Access counters

#### uvm_channel_t
GPU command submission channel:
- Push buffer
- Semaphore pool
- Work tracking
- Error handling

### 2.4 Operational Flow

#### Page Fault Handling
1. GPU accesses unmapped/unmigrated page
2. GPU MMU generates fault
3. Fault written to replayable fault buffer
4. UVM interrupt handler queues work
5. Fault servicing:
   - Resolve VA range
   - Determine required operations (map, migrate, populate)
   - Execute copy engine operations if needed
   - Update page tables
   - Replay faulting accesses

#### Migration
1. Policy decision (fault-driven, prefetch, thrashing)
2. Acquire locks (VA block, GPU)
3. Allocate destination memory
4. Copy data (via GPU copy engine or CPU)
5. Update page tables on all GPUs
6. TLB invalidation
7. Release source memory if appropriate

### 2.5 Device Interface

**Character Device**: /dev/nvidia-uvm

**Main IOCTLs** (defined in uvm_linux_ioctl.h):
- UVM_INITIALIZE - Initialize UVM
- UVM_CREATE_RANGE - Create virtual address range
- UVM_MIGRATE - Explicit migration
- UVM_MAP_EXTERNAL_ALLOCATION - Map memory from nvidia.ko
- UVM_ENABLE_PEER_ACCESS - Enable GPU-to-GPU access
- UVM_SET_PREFERRED_LOCATION - Set migration hint
- UVM_SET_ACCESSED_BY - Set access pattern hint

---

## 3. nvidia-drm.ko - DRM/KMS Driver

**Location**: `kernel-open/nvidia-drm/`
**Files**: 19 C files
**Module**: nvidia-drm.ko
**Purpose**: Linux Direct Rendering Manager integration

### 3.1 Overview

Integrates NVIDIA GPUs with the Linux **DRM (Direct Rendering Manager)** subsystem, providing:
- KMS (Kernel Mode Setting) - display configuration
- DRM framebuffer management
- Atomic display updates
- GEM (Graphics Execution Manager) object management
- Synchronization primitives (fences)
- Integration with Wayland/X11 compositors

### 3.2 Core Components

#### Driver Core
- **nvidia-drm-drv.c** (70,065 lines) - DRM driver registration
  - DRM device initialization
  - Driver callbacks
  - Property management
  - Client interface

- **nvidia-drm.c** (2,006 lines) - Module entry point
  - Module initialization
  - Global state

- **nvidia-drm-linux.c** - Linux-specific glue

#### Display Subsystem (KMS)
- **nvidia-drm-modeset.c** (29,261 lines) - Mode setting implementation
  - Output configuration
  - Mode validation
  - Display state management

- **nvidia-drm-crtc.c** (118,258 lines) - CRTC (Display Controller) implementation
  - Scanout configuration
  - Atomic commit
  - Page flipping
  - VBlank handling
  - Cursor management

- **nvidia-drm-connector.c** (21,957 lines) - Display connector management
  - Hotplug detection
  - EDID parsing
  - Mode enumeration
  - Connector properties (HDR, color space)

- **nvidia-drm-encoder.c** (9,360 lines) - Encoder management
  - Output encoding (TMDS, DisplayPort, HDMI)

#### Memory Management
- **nvidia-drm-gem.c** (11,703 lines) - GEM object implementation
  - Buffer object lifecycle
  - CPU/GPU access
  - Reference counting

- **nvidia-drm-gem-nvkms-memory.c** (20,129 lines) - NVKMS memory objects
  - Memory allocated by nvidia-modeset
  - Scanout buffers

- **nvidia-drm-gem-user-memory.c** (7,437 lines) - User memory import
  - Pin user pages
  - Create GEM objects from userptr

- **nvidia-drm-gem-dma-buf.c** (7,977 lines) - DMA-BUF integration
  - Import external buffers
  - Export GEM objects as DMA-BUF

- **nvidia-drm-fb.c** (9,779 lines) - Framebuffer management
  - FB creation/destruction
  - Format handling

#### Synchronization
- **nvidia-drm-fence.c** (58,535 lines) - Fence/syncobj implementation
  - GPU work synchronization
  - Cross-device sync
  - Timeline semaphores
  - Implicit/explicit sync

#### Utilities
- **nvidia-drm-helper.c** (6,249 lines) - Helper functions
- **nvidia-drm-format.c** (6,889 lines) - Pixel format handling
- **nvidia-drm-utils.c** (7,425 lines) - Utility functions

### 3.3 Integration with nvidia-modeset.ko

nvidia-drm acts as a **thin wrapper** over nvidia-modeset:
- nvidia-modeset provides actual display control (via RM)
- nvidia-drm translates DRM API calls to NVKMS interface
- NVKMS = NVIDIA Kernel Mode Setting (internal API)

### 3.4 Key Features

#### Atomic Display Updates
- Implements DRM atomic API
- Allows glitch-free display reconfiguration
- Used by Wayland compositors

#### HDR Support
- HDR10 (PQ transfer function)
- HDR metadata
- Wide color gamut (BT.2100)

#### Multi-plane Support
- Hardware cursor plane
- Primary plane
- Overlay planes

#### Synchronization
- Implicit fences (attached to DMA-BUF)
- Explicit fences (sync_file, syncobj)
- In-fence (wait before scan out)
- Out-fence (signal when flip complete)

---

## 4. nvidia-modeset.ko - Mode Setting Interface

**Location**: `kernel-open/nvidia-modeset/`
**Files**: 2 C files
**Module**: nvidia-modeset.ko
**Purpose**: Display mode setting and configuration

### 4.1 Overview

Provides the **kernel interface to NVIDIA's mode setting implementation**:
- Thin shim between kernel and RM
- Links with proprietary binary (nvidia-modeset-kernel.o_binary)
- Handles display configuration, mode setting, scanout

### 4.2 Files

- **nvidia-modeset-linux.c** (57,447 lines) - Main implementation
  - Module initialization
  - NVKMS interface implementation
  - Callbacks from nvidia.ko
  - Callbacks to nvidia-drm.ko

- **nv-kthread-q.c** (11,520 lines) - Kernel thread queue
  - Deferred work processing
  - Display update serialization

### 4.3 Architecture

Similar to nvidia.ko, uses a **hybrid model**:
- Open source interface layer (nvidia-modeset-linux.c)
- Proprietary binary core (nvidia-modeset-kernel.o_binary)

The binary core contains:
- NVKMS (Kernel Mode Setting) implementation
- Display hardware abstraction
- Mode validation and timing generation
- Output routing and configuration

### 4.4 Integration Points

**From nvidia.ko**:
- GPU initialization callback
- Resource management
- Interrupt handling

**To nvidia-drm.ko**:
- Display event notification
- Configuration callbacks
- Buffer management

---

## 5. nvidia-peermem.ko - Peer Memory Support

**Location**: `kernel-open/nvidia-peermem/`
**Files**: 1 C file (22,891 lines)
**Module**: nvidia-peermem.ko
**Purpose**: RDMA integration for GPU memory

### 5.1 Overview

Enables **RDMA (Remote Direct Memory Access)** devices to access GPU memory directly:
- GPU Direct RDMA
- Peer-to-peer DMA between GPU and InfiniBand/RoCE NICs
- Zero-copy networking
- Used for HPC and distributed GPU computing

### 5.2 Implementation

**nvidia-peermem.c**:
- Implements peer_memory_client interface
- Registers with ib_core (InfiniBand core)
- Provides callbacks:
  - `acquire()` - Pin GPU memory
  - `get_pages()` - Get physical pages
  - `dma_map()` - Setup DMA mapping
  - `put_pages()` - Release pages
  - `release()` - Unpin memory

### 5.3 Use Cases

- GPU Direct RDMA for MPI applications
- Distributed machine learning (e.g., NCCL over InfiniBand)
- GPU-accelerated storage systems
- Direct network-to-GPU data transfer

---

## 6. common/ - Shared Interfaces

**Location**: `kernel-open/common/`
**Purpose**: Shared headers and definitions

### 6.1 Structure

```
common/
├── inc/            - Public headers
│   ├── nv.h       - Core definitions
│   ├── nv-linux.h - Linux kernel integration
│   ├── nv-ioctl.h - ioctl definitions
│   ├── nvstatus.h - Status codes
│   ├── nvlimits.h - Limits and constants
│   └── os/        - OS abstraction
└── shared/         - (if present) Shared with user-space
```

### 6.2 Key Headers

#### nv.h (55,083 lines)
Core data structure definitions:
- nv_state_t - per-GPU state
- nv_ioctl structures
- GPU capabilities
- Firmware definitions
- NUMA support

#### nv-linux.h
Linux kernel integration:
- Includes all necessary kernel headers
- Compatibility macros
- OS primitive abstractions
- Platform-specific code

#### nv-ioctl.h
ioctl interface definitions:
- ioctl command numbers
- Parameter structures
- Legacy API support

#### nvstatus.h
Comprehensive status code enumeration:
- Success/failure codes
- Detailed error conditions
- Used throughout driver stack

#### os/
OS abstraction layer headers:
- Memory management primitives
- Locking primitives
- Timer interfaces
- DMA operations

### 6.3 Cross-Module Interfaces

**nv_uvm_interface.h** - nvidia.ko ↔ nvidia-uvm.ko:
- GPU registration
- Memory type queries
- DMA mapping services
- Channel allocation

**nv_modeset_interface.h** - nvidia.ko ↔ nvidia-modeset.ko:
- Display subsystem callbacks
- Resource sharing
- Event notification

---

## 7. Build System

### 7.1 Overview

The build system uses Linux kernel's **Kbuild infrastructure** with sophisticated configuration testing.

### 7.2 Key Files

#### Makefile
Top-level makefile:
- Invokes kernel's Kbuild
- Determines kernel source/output paths
- Sets up compilation environment
- Handles module installation

#### Kbuild
Main Kbuild file:
- Includes per-module Kbuild files
- Defines global CFLAGS
- Manages conftest system
- Handles binary object linking

#### conftest.sh (195,621 bytes)
Massive configuration testing script:
- Tests kernel API availability
- Generates compatibility headers
- Checks for kernel configuration options
- Tests symbol exports
- Validates data structure layouts

**Purpose**: Abstract kernel version differences
- Linux kernel APIs change frequently
- conftest detects available features
- Generates conditional compilation macros

**Test Categories**:
1. **Function tests** - e.g., `set_memory_uc()` availability
2. **Type tests** - e.g., structure member existence
3. **Symbol tests** - e.g., exported symbol checks
4. **Generic tests** - e.g., Xen presence
5. **Header presence tests** - e.g., `linux/fence.h`

#### Per-Module Kbuild Files

**nvidia/nvidia.Kbuild**:
- Defines NVIDIA_SOURCES list
- Links nv-kernel.o_binary
- Registers conftests (~238 tests)
- Builds nv-interface.o

**nvidia-uvm/nvidia-uvm.Kbuild**:
- Defines NVIDIA_UVM_SOURCES
- Pure open source (no binary)
- Registers UVM-specific conftests

**nvidia-drm/nvidia-drm.Kbuild**:
- Conditional on DRM_AVAILABLE
- Links nvidia-drm-kernel.o_binary
- DRM subsystem integration

**nvidia-modeset/nvidia-modeset.Kbuild**:
- Links nvidia-modeset-kernel.o_binary
- Display subsystem integration

**nvidia-peermem/nvidia-peermem.Kbuild**:
- Conditional on InfiniBand support
- Pure open source

### 7.3 Build Phases

#### Phase 1: Configuration
1. conftest.sh runs
2. Generates conftest/*.h headers
3. Creates compatibility layer

#### Phase 2: Compilation
1. Kbuild compiles all .c files
2. Per-object CFLAGS applied
3. Creates .o files

#### Phase 3: Linking
1. Link interface .o files
2. Link binary .o_binary files
3. Create .ko modules
4. MODPOST stage (module verification)

### 7.4 Compilation Flags

**Common flags** (from Kbuild):
```
-D__KERNEL__ -DMODULE -DNVRM
-DNV_KERNEL_INTERFACE_LAYER
-DNV_VERSION_STRING="580.95.05"
-DNV_UVM_ENABLE
-Wall -Wno-cast-qual -Wno-format-extra-args
-fno-strict-aliasing
-ffreestanding
```

**Architecture-specific**:
- x86_64: `-mno-red-zone -mcmodel=kernel`
- arm64: `-mstrict-align -mgeneral-regs-only -march=armv8-a`

**Build types**:
- release: `-DNDEBUG`
- develop: `-DNDEBUG -DNV_MEM_LOGGER`
- debug: `-DDEBUG -g -DNV_MEM_LOGGER`

### 7.5 Conftest Examples

**Function test** (set_memory_uc):
```c
#if defined(NV_SET_MEMORY_UC_PRESENT)
    set_memory_uc(addr, pages);
#else
    set_pages_uc(virt_to_page(addr), pages);
#endif
```

**Type test** (vm_operations_struct changes):
```c
#if defined(NV_VM_OPS_FAULT_REMOVED)
    .fault = nv_drm_gem_fault,
#else
    .pfn_mkwrite = nv_drm_gem_pfn_mkwrite,
#endif
```

---

## 8. Module Interactions and Data Flow

### 8.1 Initialization Sequence

1. **nvidia.ko loads**
   - Initializes PCI subsystem
   - Creates /dev/nvidia* devices
   - Initializes RM (Resource Manager)

2. **nvidia-modeset.ko loads**
   - Registers with nvidia.ko
   - Initializes display subsystem

3. **nvidia-uvm.ko loads**
   - Registers with nvidia.ko
   - Creates /dev/nvidia-uvm
   - Initializes UVM subsystem

4. **nvidia-drm.ko loads**
   - Registers DRM driver
   - Connects to nvidia-modeset
   - Creates DRM devices

5. **nvidia-peermem.ko loads** (if InfiniBand present)
   - Registers peer memory client
   - Enables GPU Direct RDMA

### 8.2 GPU Memory Access Flow

**User Space → GPU Memory**:
1. Application opens /dev/nvidia0
2. ioctl(NV_ESC_ALLOC_MEMORY) - allocates GPU memory
3. mmap(/dev/nvidia0) - maps into user space
4. Application can read/write GPU memory

**UVM Managed Memory**:
1. Application opens /dev/nvidia-uvm
2. cudaMallocManaged() - creates unified address
3. UVM creates VA range
4. On CPU/GPU access:
   - Page fault occurs
   - UVM migrates page to accessing processor
   - Page mapped in faulting processor's page table
   - Access completes

### 8.3 Display Output Flow

**Mode Setting**:
1. Wayland compositor opens DRM device
2. DRM atomic commit:
   - nvidia-drm receives atomic state
   - Translates to NVKMS calls
   - nvidia-modeset programs hardware
   - RM configures display engine

**Frame Presentation**:
1. Client submits frame buffer (GEM object)
2. Attaches fence for synchronization
3. DRM atomic commit with new FB
4. nvidia-drm:
   - Waits for in-fence
   - Programs scanout from new FB
   - Signals out-fence on VBlank

### 8.4 Inter-GPU Communication

**Peer-to-Peer (P2P)**:
1. nvidia.ko enables P2P between GPUs
2. Maps one GPU's BAR into other GPU's address space
3. UVM enables peer access
4. GPUs can directly access each other's memory

**NVLink**:
1. Automatic discovery via nvlink_linux.c
2. High-bandwidth, low-latency interconnect
3. Transparent to UVM (looks like fast P2P)

### 8.5 Power Management

**Runtime PM**:
1. nvidia.ko implements PM callbacks
2. On idle: suspend GPU (D3hot state)
3. On access: resume GPU (D0 state)

**System Suspend/Resume**:
1. System PM triggers driver suspend
2. nvidia.ko: save state, power down GPU
3. nvidia-modeset: disable displays
4. nvidia-uvm: flush pending operations
5. On resume: reverse process

---

## 9. Hardware Abstraction

### 9.1 GPU Architecture Support

The driver supports multiple GPU architectures through **HAL (Hardware Abstraction Layer)**:

| Architecture | Codename | Release | Support Level |
|-------------|----------|---------|---------------|
| Maxwell | GM10x | 2014 | Full |
| Pascal | GP10x | 2016 | Full |
| Volta | GV100 | 2017 | Full |
| Turing | TU10x | 2018 | Full |
| Ampere | GA100 | 2020 | Full |
| Ada Lovelace | AD10x | 2022 | Full |
| Hopper | GH100 | 2022 | Full |
| Blackwell | GB100 | 2024 | Full |

### 9.2 Architecture-Specific Code

Each GPU architecture has specialized implementations in UVM:
- **MMU format** - Page table entry layout
- **Fault buffer** - Fault reporting format
- **Copy engine** - DMA methods
- **Host interface** - Doorbell registers

Example from UVM:
```
uvm_hopper_mmu.c     - Hopper page table management
uvm_hopper_host.c    - Host interface programming
uvm_hopper_ce.c      - Copy engine operations
uvm_hopper_fault_buffer.c - Fault buffer parsing
```

### 9.3 Binary Core Abstraction

The proprietary nv-kernel.o_binary contains:
- GPU-specific initialization sequences
- Hardware workarounds
- Power management sequences
- Display engine programming
- GPU scheduling algorithms
- Memory controller configuration

This allows NVIDIA to:
- Protect trade secrets
- Ship hardware-specific code without upstreaming delays
- Provide unified driver across architectures

---

## 10. Security and Isolation

### 10.1 Capability-Based Security

**nv-caps.c** implements a capability system:
- Device access controlled by capability files
- Prevents unauthorized GPU access
- Used for:
  - MIG (Multi-Instance GPU) partitions
  - Confidential Computing
  - Container isolation

### 10.2 Confidential Computing

**uvm_conf_computing.c** in UVM:
- GPU memory encryption
- Secure channels
- Attestation support
- Used for TEE (Trusted Execution Environment)

### 10.3 IOMMU Integration

**nv-dma.c** provides IOMMU support:
- DMA address translation
- Device isolation
- Protection against malicious DMA
- Required for virtualization security

---

## 11. Performance Optimizations

### 11.1 UVM Performance Features

**Access Counter Sampling**:
- Hardware tracks memory access patterns
- UVM uses counters for migration decisions
- Reduces unnecessary migrations

**Thrashing Detection**:
- Detects pathological access patterns
- Stops counterproductive migrations
- Maintains performance under stress

**Prefetching**:
- Predicts future memory accesses
- Migrates pages before faults occur
- Reduces latency

**Batched Operations**:
- Groups page table updates
- Amortizes TLB shootdown costs
- Improves migration throughput

### 11.2 Interrupt Coalescing

nvidia.ko implements interrupt coalescing:
- Reduces interrupt overhead
- Improves throughput for small operations
- Configurable thresholds

### 11.3 DMA Optimizations

- Scatter-gather to minimize copies
- IOMMU bypass where safe
- Large page support (2MB pages)
- Copy engine offload

---

## 12. Testing and Debugging

### 12.1 Test Infrastructure

**UVM Tests** (uvm_test*.c):
- Comprehensive test suite
- Tests all major subsystems
- Accessible via ioctl
- Used for validation and regression testing

**Test Categories**:
- Channel operations
- MMU programming
- Page migration
- Fault handling
- Multi-GPU coherence
- Performance counters

### 12.2 Debug Features

**Memory Debugging** (nv-memdbg.c):
- Tracks allocations
- Detects leaks
- Guards against corruption

**Logging**:
- Detailed trace points
- Configurable verbosity
- Per-subsystem control

**Procfs Interface**:
- Runtime status queries
- Configuration viewing
- Statistics reporting

---

## 13. Platform-Specific Support

### 13.1 Tegra/Jetson Support

Tegra SoC integration (ARM-based):
- **nv-platform.c** - Platform device support
- **nv-bpmp.c** - Boot and Power Management Processor
- **nv-host1x.c** - Command submission via Host1x
- **nv-dsi-parse-panel-props.c** - Display panel configuration
- Device tree parsing
- Clock management
- Power domain control

### 13.2 Virtualization Support

**GPU Virtualization**:
- vGPU support in nvidia.ko
- SR-IOV integration
- VFIO-based passthrough
- Mediated device framework (mdev)

### 13.3 Container Support

- cgroups integration
- Device isolation via capabilities
- MIG (Multi-Instance GPU) support
- Resource accounting

---

## 14. Future Directions and Active Development

### 14.1 Recent Additions

**Blackwell Support**:
- Latest GPU architecture (2024)
- New fault buffer format
- Enhanced copy engines
- Updated MMU features

**HMM Improvements**:
- Tighter integration with Linux MM
- Reduced complexity
- Better page migration policies

**ATS/SVA**:
- Address Translation Services
- Shared Virtual Addressing
- IOMMU integration
- Enables GPU to use CPU page tables

### 14.2 Open Source Transition

NVIDIA is gradually opening more code:
- UVM fully open source
- Kernel interface layers open
- Binary core remains proprietary (RM)
- Long-term: potential for full open source stack

---

## 15. Critical Implementation Details

### 15.1 Locking Hierarchy

Complex lock ordering to prevent deadlocks:

**nvidia.ko**:
1. nv_linux_devices_lock (global)
2. nv_state_t per-device locks
3. RM internal locks

**nvidia-uvm.ko**:
1. g_uvm_global lock
2. VA space lock
3. GPU lock
4. VA range lock
5. VA block lock

### 15.2 Error Handling

Robust error handling throughout:
- NV_STATUS return codes
- Error propagation
- Resource cleanup on failure
- Graceful degradation

### 15.3 Memory Reclaim

Integration with Linux memory management:
- Shrinker callback for memory pressure
- Eviction of idle GPU allocations
- Cooperation with OOM killer

### 15.4 GPU Reset and Recovery

Handling GPU hangs:
- Timeout detection
- Channel preemption
- Reset and recovery
- Minimal application impact

---

## 16. Documentation and Resources

### 16.1 Key Files for Understanding

**Start here**:
1. `nvidia/nv.c` - Main driver entry point
2. `nvidia-uvm/uvm.c` - UVM entry point
3. `common/inc/nv.h` - Core data structures
4. `Kbuild` - Build system overview

**For specific subsystems**:
- Memory management: `nvidia-uvm/uvm_va_block.c`
- Display: `nvidia-drm/nvidia-drm-crtc.c`
- DMA: `nvidia/nv-dma.c`
- Faults: `nvidia-uvm/uvm_gpu_replayable_faults.c`

### 16.2 Development Workflow

**Building**:
```bash
make -j$(nproc)
```

**Installing**:
```bash
sudo make modules_install
sudo depmod -a
```

**Testing**:
```bash
# Load modules
sudo modprobe nvidia
sudo modprobe nvidia-uvm

# Run tests (if available)
./run_tests.sh
```

### 16.3 Debugging Tips

**Kernel logs**:
```bash
dmesg | grep -i nvidia
journalctl -k | grep -i nvidia
```

**Module parameters**:
```bash
# Increase debug level
sudo modprobe nvidia NVreg_ResmanDebugLevel=0xffffffff
```

**UVM statistics**:
```bash
cat /sys/module/nvidia_uvm/parameters/*
```

---

## Summary of Critical Findings

### Architectural Highlights

1. **Hybrid Design**: Open-source interface layer wrapping proprietary core
   - Enables rapid kernel version support
   - Protects IP while providing open interfaces
   - Binary components: nv-kernel.o_binary, nvidia-modeset-kernel.o_binary, nvidia-drm-kernel.o_binary

2. **UVM: Standout Component**
   - **Fully open source** - No proprietary components
   - Largest module (103,318 LOC)
   - Sophisticated virtual memory subsystem
   - Rivals complexity of Linux kernel's MM subsystem
   - Implements unified addressing, automatic migration, and fault handling

3. **Comprehensive Hardware Support**
   - 8 GPU architectures (Maxwell through Blackwell)
   - HAL provides clean abstraction
   - Architecture-specific optimizations

4. **Advanced Features**
   - Unified Virtual Memory with automatic migration
   - GPU Direct RDMA for zero-copy networking
   - Atomic display updates for tear-free rendering
   - ATS/SVA for IOMMU integration
   - Confidential Computing support
   - Multi-Instance GPU (MIG) isolation

5. **Build System Sophistication**
   - conftest.sh: 195K+ script for kernel compatibility
   - Tests ~300+ kernel features
   - Enables single driver for kernels 4.15+
   - Handles API changes across 6+ years of kernel development

6. **Module Ecosystem**
   - 5 distinct kernel modules
   - Clear separation of concerns
   - Well-defined interfaces between modules
   - Integration with Linux subsystems (DRM, IOMMU, MM)

### Technical Debt and Challenges

1. **Binary Dependencies**: Core GPU management remains closed
2. **Complexity**: Over 200K lines of code with deep integration
3. **Maintenance Burden**: Supporting wide kernel version range
4. **Documentation**: Code-centric, limited high-level docs

### Future Outlook

The gradual open-sourcing (especially UVM) represents a significant shift in NVIDIA's strategy. The kernel-open codebase provides:
- Reference implementation for GPU memory management
- Insights into high-performance driver design
- Framework for community contributions
- Path toward fully open-source GPU stack

This driver stack represents **state-of-the-art GPU kernel integration**, balancing performance, features, compatibility, and (increasingly) openness.
