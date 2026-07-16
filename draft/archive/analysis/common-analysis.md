# NVIDIA Open GPU Kernel Modules - src/common/ Directory Analysis

## Executive Summary

The `src/common/` directory contains **1,391 source files** (235 .c files and 1,156 .h files) organized into 12 major subdirectories. This forms the foundational layer of shared libraries, hardware abstractions, and utilities that support the entire NVIDIA GPU kernel driver stack. The common libraries provide critical functionality including DisplayPort/HDMI protocol implementations, high-speed NVLink interconnect management, hardware reference definitions for multiple GPU architectures, floating-point mathematics, Unix-specific utilities, and microprocessor support libraries.

## Directory Structure Overview

```
src/common/
├── displayport/      - DisplayPort 1.2/1.4/2.0 MST/SST implementation (C++)
├── nvlink/          - NVLink high-speed interconnect library
├── nvswitch/        - NVSwitch fabric management
├── sdk/             - SDK headers and control interfaces
├── shared/          - Shared utilities and message queue library
├── softfloat/       - IEEE 754 floating-point arithmetic library
├── unix/            - Unix-specific utilities (3D, push buffers, headsurface)
├── uproc/           - Microprocessor support (ELF, DWARF, crash decoding)
├── modeset/         - Mode-setting utilities (HDMI packets, timing)
├── inc/             - Common headers and hardware reference definitions
├── src/             - Common source implementations (SHA-256, SMG)
└── README.md        - Basic directory information
```

---

## 1. DisplayPort Library (displayport/)

**Location:** `/src/common/displayport/`
**Files:** 41 files (C++ implementation)
**Architecture:** Object-oriented C++ with namespace `DisplayPort`

### Purpose
Implements a complete DisplayPort 1.2/1.4/2.0 protocol stack supporting both Single-Stream Transport (SST) and Multi-Stream Transport (MST) modes. This library handles display connectivity, link training, bandwidth management, EDID parsing, HDCP encryption, and Display Stream Compression (DSC).

### Key Components

#### Core Interfaces (inc/)
- **dp_connector.h** - Primary client interface defining the `Connector` class
  - Main entry point for DisplayPort functionality
  - Handles device enumeration, hotplug events, link configuration
  - Supports SST/MST switching, compound queries for bandwidth validation
  - Provides modeset notifications and power management

- **dp_device.h & dp_deviceimpl.h** - Device abstraction
  - Represents physical DisplayPort devices in the topology
  - EDID reading and parsing, DPCD access
  - DSC capability negotiation
  - HDCP availability checks
  - Power state management (D0/D3)

- **dp_group.h & dp_groupimpl.h** - Stream grouping
  - Groups multiple devices for simultaneous driving
  - Handles single-head-multi-stream (SHMS) configurations
  - HDCP encryption control per group

#### Link Management
- **dp_mainlink.h** - Hardware abstraction for physical link
  - Link training state machine (OFF -> SWCFG -> ACTIVE)
  - Lane count and link rate negotiation
  - AUX channel transactions
  - Pattern generation for compliance testing

- **dp_linkconfig.h** - Link configuration parameters
  - Bandwidth calculations
  - Enhanced framing, scrambling control
  - FEC (Forward Error Correction) support
  - Multi-stream configuration

- **dp_auxbus.h & dp_auxretry.h** - AUX channel communication
  - DPCD register access (read/write)
  - I2C-over-AUX for EDID/DDC
  - Automatic retry on DEFER responses
  - Transaction timeout handling

#### Topology Discovery
- **dp_address.h** - MST device addressing
  - Relative address encoding for MST branch devices
  - Port number management in topology tree

- **dp_discovery.h** - Device detection
  - Branch/sink device enumeration
  - GUID-based device tracking
  - Plug/unplug event processing

- **dp_messages.h** - MST sideband messaging
  - LINK_ADDRESS, ENUM_PATH_RESOURCES
  - ALLOCATE_PAYLOAD, CLEAR_PAYLOAD_ID_TABLE
  - Remote DPCD read/write
  - Connection status notification

#### Advanced Features
- **dp_watermark.h** - Display pipeline watermark calculation
  - Prevents underflow during bandwidth changes
  - Accounts for pixel clock, link bandwidth, color depth

- **dp_vrr.h** - Variable Refresh Rate (Adaptive Sync)
  - VESA Adaptive-Sync signaling
  - Monitor enablement via vendor-specific DPCD

- **dp_edid.h** - EDID parsing
  - CEA-861 extension block handling
  - Timing validation
  - Audio capability detection

- **dp_crc.h** - CRC generation for DSC and link validation

#### Implementation Files (src/)
While headers are visible, implementation (.cpp) files handle the actual protocol state machines, timing-sensitive operations, and hardware interactions.

### Data Flow
1. **Hotplug Detection** → `Connector::notifyLongPulse()` → Topology discovery
2. **Mode Validation** → `Connector::compoundQueryAttach()` → Bandwidth calculation
3. **Modeset** → `Connector::notifyAttachBegin()` → Link training → `notifyAttachEnd()`
4. **MST Stream Allocation** → Payload table programming → Virtual channel assignment

### Integration Points
- Calls into RM (Resource Manager) via `EvoInterface` for hardware programming
- Uses timer services for link training timeouts
- Provides event callbacks via `Connector::EventSink` for device add/remove

---

## 2. NVLink Library (nvlink/)

**Location:** `/src/common/nvlink/`
**Files:** 30+ files (C implementation)
**Architecture:** Modular library with device/link registration and state management

### Purpose
Manages NVIDIA's proprietary high-speed NVLink interconnect, enabling GPU-to-GPU, GPU-to-CPU, and GPU-to-switch communication. Supports bandwidths up to 900 GB/s (NVLink 4.0/5.0) with coherent memory access across devices.

### Key Components

#### Core Library (kernel/nvlink/)

**nvlink.h** - Primary API header
```c
// Core data structures
struct nvlink_device {
    NvU64 deviceId;           // Unique device identifier
    NvU64 type;               // GPU, NVSWITCH, IBMNPU, EBRIDGE, TEGRASHIM
    NVListRec link_list;      // List of links on this device
    char *deviceName;
    NvU8 *uuid;
    struct nvlink_pci_info pciInfo;
    NvBool enableALI;         // Adaptive Link Interface training
    NvU16 nodeId;             // Fabric node ID
    void *pDevInfo;           // Client private data
};

struct nvlink_link {
    NvU64 linkId;             // Unique link identifier
    struct nvlink_device *dev;
    NvU32 linkNumber;         // Per-device link number
    NvU32 state;              // NVLINK_LINKSTATE_*
    NvU32 tx_sublink_state;   // TX PHY state
    NvU32 rx_sublink_state;   // RX PHY state
    NvU32 version;            // NVLink version (1.0-5.0)
    NvBool master;            // Master/slave training role
    NvU64 localSid;           // Local system ID
    NvU64 remoteSid;          // Remote system ID
    NvU32 remoteLinkId;       // Connected link ID
    NvU32 seedData[7];        // Training seed values
    const struct nvlink_link_handlers *link_handlers;
};
```

**Link States** (state machine)
```
OFF → SWCFG (safe mode) → ACTIVE (high speed) → L2 (sleep)
       ↓
    DETECT → RESET → INITPHASE1 → INITNEGOTIATE → INITOPTIMIZE →
    INITTL → INITPHASE5 → ALI → ACTIVE_PENDING → HS (High Speed)
```

#### Core Operations (core/)

**nvlink_discovery.c** - Topology detection
- Link endpoint discovery via token exchange
- SID (System ID) and remote device type detection
- Builds intranode and internode connection tables

**nvlink_training.c** - Link training state machines
- `nvlink_lib_train_links_from_swcfg_to_active()` - Primary training API
- Implements INITPHASE1-5 sequence
- ALI (Adaptive Link Interface) for optimized signal integrity
- Seed data save/restore for fast re-training
- Parallel training of multiple links for low latency

**nvlink_initialize.c** - Link initialization
- `nvlink_lib_reinit_link_from_off_to_swcfg()` - Power up link to safe mode
- PHY calibration and receiver detect
- Common mode voltage setup

**nvlink_shutdown.c** - Link power management
- `nvlink_lib_powerdown_links_from_active_to_L2()` - Clean shutdown to sleep
- `nvlink_lib_powerdown_links_from_active_to_off()` - Force shutdown
- State save/restore for L2 transitions

**nvlink_link_mgmt.c** - Link lifecycle management
- Link registration/unregistration
- Master link assignment
- Link state queries

**nvlink_conn_mgmt.c** - Connection management
- Maintains intranode and internode connection lists
- Validates topology consistency
- Handles multi-node fabrics

#### Lock Management (nvlink_lock.c)
- Per-link locking for concurrent access
- Top-level library lock for global operations
- Prevents deadlocks during multi-link training

#### Interface Layer (interface/)
Entry points for driver integration:
- **nvlink_kern_registration_entry.c** - Device/link registration
- **nvlink_kern_discovery_entry.c** - Topology discovery APIs
- **nvlink_kern_initialize_entry.c** - Initialization entry points
- **nvlink_kern_training_entry.c** - Training APIs
- **nvlink_kern_shutdown_entry.c** - Shutdown APIs
- **nvlink_ioctl_entry.c** - IOCTL handling for userspace tools

#### Inband Messaging (inband/)
**nvlink_inband_msg.h** - Inband message protocol
- Used for remote link control in multi-node systems
- Message types: INIT, TRAINING, TOPOLOGY_EXCHANGE
- Transport-agnostic message queue interface

### Supported Devices
- **NVLINK_DEVICE_TYPE_GPU** (0x2) - NVIDIA GPUs
- **NVLINK_DEVICE_TYPE_NVSWITCH** (0x3) - NVSwitch fabric switches
- **NVLINK_DEVICE_TYPE_IBMNPU** (0x1) - IBM POWER9/10 NPU
- **NVLINK_DEVICE_TYPE_EBRIDGE** (0x0) - External bridges
- **NVLINK_DEVICE_TYPE_TEGRASHIM** (0x4) - Tegra SoC integration

### NVLink Versions
- **1.0** (Pascal) - 20 GB/s per link, 4 links/GPU
- **2.0** (Volta) - 25 GB/s per link, 6 links/GPU
- **2.2** (Turing) - 25 GB/s per link
- **3.0** (Ampere) - 50 GB/s per link, 12 links/GPU
- **3.1** (Hopper with NVSwitch) - Enhanced fabric support
- **4.0** (Hopper) - 100 GB/s per link
- **5.0** (Blackwell) - 150 GB/s per link (projected)

### Key Features
- **Coherent memory access** - Cache-coherent shared memory
- **Atomic operations** - Hardware atomics across NVLink
- **Error detection** - CRC, replay, error thresholds
- **Power management** - L0 (active), L1 (low power), L2 (sleep)
- **Multi-path routing** - Multiple links between devices for redundancy
- **Fabric management** - Up to 624 system links in large fabrics

---

## 3. NVSwitch Library (nvswitch/)

**Location:** `/src/common/nvswitch/`
**Files:** 100+ files
**Architecture:** HAL (Hardware Abstraction Layer) with per-chip implementations

### Purpose
Manages NVIDIA NVSwitch chips, which act as fabric switches enabling full NVLink connectivity between multiple GPUs. NVSwitch provides non-blocking crossbar switching with 18 ports per chip (NVSwitch 2.0/Hopper) and 64 ports (NVSwitch 3.0/Blackwell).

### Key Components

#### Interface (interface/)
- **export_nvswitch.h** - External API for driver integration
- **ctrl_dev_nvswitch.h** - Control commands for switch management
- **ioctl_common_nvswitch.h** - IOCTL interface definitions

#### InfoROM Support (kernel/inforom/)
Non-volatile storage on NVSwitch for telemetry and configuration:
- **inforom_nvswitch.c** - InfoROM core management
- **ifrnvlink_nvswitch.c** - NVLink error tracking
- **ifrecc_nvswitch.c** - ECC error logging
- **ifrbbx_nvswitch.c** - Blackbox error records
- **ifr oms_nvswitch.c** - OMS (Object Management System) data
- **inforom_nvl_v3_nvswitch.c** & **v4** - NVLink InfoROM versions

#### Falcon Microcontroller Support (kernel/flcn/)
NVSwitch contains embedded Falcon processors for management:
- **flcn_nvswitch.c** - Core Falcon management
- **flcnable_nvswitch.c** - Falcon enable/disable
- **flcnqueue_nvswitch.c** - Command/message queues
- **flcndmem_nvswitch.c** - DMEM access for Falcon
- **flcnrtosdebug_nvswitch.c** - RTOS debug facilities

**Version-specific implementations:**
- **v03/** - Falcon 3.0 (NVSwitch 1.0)
- **v04/** - Falcon 4.0 (NVSwitch 2.0)
- **v05/** - Falcon 5.0 (transition version)
- **v06/** - Falcon 6.0 (NVSwitch 3.0)

#### Queue Management
- **flcnqueue_fb_nvswitch.c** - Framebuffer-backed queues
- **flcnqueue_dmem_nvswitch.c** - DMEM-backed queues
- **flcnqueuerd_nvswitch.c** - Queue read operations

### NVSwitch Generations
1. **NVSwitch 1.0** (2018, Volta/V100)
   - 18 ports × 25 GB/s = 450 GB/s aggregate
   - Used in DGX-2 (16 GPUs)

2. **NVSwitch 2.0** (2020, Ampere/A100)
   - 18 ports × 50 GB/s = 900 GB/s aggregate
   - Used in DGX A100 (8 GPUs)

3. **NVSwitch 3.0** (2022, Hopper/H100)
   - 64 ports × 50 GB/s = 3.2 TB/s aggregate
   - Used in DGX H100 (8 GPUs with 4 NVSwitches)

### Key Functions
- **Crossbar switching** - Non-blocking path between any two GPUs
- **Multicast** - Efficient all-reduce operations for AI training
- **QoS (Quality of Service)** - Traffic prioritization
- **Error handling** - Port isolation, error containment
- **Telemetry** - Link statistics, error counters
- **In-band management** - Remote switch control via NVLink

---

## 4. SDK Headers (sdk/nvidia/inc/)

**Location:** `/src/common/sdk/nvidia/inc/`
**Files:** 700+ header files
**Architecture:** Hierarchical API definitions

### Purpose
Defines the public API surface for NVIDIA GPU programming, including control commands, allocation interfaces, object classes, and type definitions. This forms the contract between userspace applications and the kernel driver.

### Directory Structure

#### Core Type Definitions
- **nvtypes.h** - Fundamental types (NvU8, NvU16, NvU32, NvU64, NvBool, etc.)
- **nvstatus.h** - Status codes (NV_OK, NV_ERR_*)
- **nvgputypes.h** - GPU-specific types
- **nvfixedtypes.h** - Fixed-point types
- **nvlimits.h** - System limits (max devices, max channels, etc.)
- **nv_vgpu_types.h** - vGPU (virtualization) types

#### Control Interfaces (ctrl/)
Over 70 control interface directories organized by class ID:

**GPU Controls (ctrl2080/)**
- Memory management controls
- Clock and power controls
- Performance monitoring
- Thermal management
- GPU reset controls

**Display Controls (ctrl0073/)**
- **ctrl0073dp.h** - DisplayPort-specific controls
  - Link training control
  - MST topology management
  - DSC configuration
  - HDCP control
- **ctrl0073dfp.h** - Digital Flat Panel controls
- **ctrl0073system.h** - Display system controls

**Subdevice Controls (ctrl2080/)**
- Per-GPU controls for multi-GPU systems
- Bus (PCIe) configuration
- Memory configuration
- Clock domain management

**Other Key Control Classes:**
- **ctrl0000/** - Root (RM client) controls
- **ctrl0080/** - Device controls
- **ctrl5070/** - Display common controls
- **ctrla06f/** - Channel controls
- **ctrlc370/** - Display supervisor controls
- **ctrl83de/** - Debug controls
- **ctrl90e7/** - Memory fabric controls
- **ctrlb0cc/** - Profiler controls
- **ctrlc372/** - Window channel controls

#### Allocation Interfaces (alloc/)
Define parameters for allocating GPU objects:
- **alloc_channel.h** - Channel allocation (GPFIFO, DMA, compute)
- Class-specific allocation structures

#### Object Classes (class/)
Hardware and software object class definitions:
- Engine classes (compute, graphics, copy, etc.)
- Memory classes (video, system, peer, etc.)
- Display classes (cursor, base, overlay, etc.)

#### Operating System Interfaces
- **nvos.h** - OS function and IOCTL interfaces
  - NVOS status codes
  - IOCTL definitions
  - Memory mapping interfaces
  - Event handling

- **nv-kernel-interface-api.h** - Kernel abstraction layer
  - OS-agnostic memory operations
  - Lock/synchronization primitives
  - Time and delay functions

#### Miscellaneous Headers
- **nverror.h** - Error code definitions
- **nvmisc.h** - Miscellaneous utilities
- **nvi2c.h** - I2C bus interface
- **nvdisptypes.h** - Display types
- **mmu_fmt_types.h** - MMU format definitions
- **cpuopsys.h** - CPU operation abstractions
- **rs_access.h** - Resource Server access control

### Control Command Pattern
```c
// Typical control command structure
typedef struct NV2080_CTRL_GPU_GET_INFO_PARAMS {
    NvU32 gpuInfoListSize;
    struct {
        NvU32 index;
        NvU32 data;
    } gpuInfoList[NV2080_CTRL_GPU_INFO_MAX_LIST_SIZE];
} NV2080_CTRL_GPU_GET_INFO_PARAMS;

// Command ID encoding
#define NV2080_CTRL_CMD_GPU_GET_INFO \
    (0x20800102) // Class 2080, category 0x01, command 0x02
```

### Integration
- Userspace libraries (CUDA, OpenGL, Vulkan drivers) include these headers
- Kernel driver implements the control commands
- FINN (Framework for Interfacing NVIDIA to Nvidia) may generate marshaling code

---

## 5. Shared Utilities (shared/)

**Location:** `/src/common/shared/`
**Files:** 15+ files
**Architecture:** Standalone utility libraries

### Purpose
Provides low-level shared utilities used across the driver, including message queues, device identification, and compatibility helpers.

### Key Components

#### Message Queue Library (msgq/)

**msgq.h** - Lock-free inter-processor message queue
```c
// Queue handle
typedef void *msgqHandle;

// Configuration
#define MSGQ_MSG_SIZE_MIN 16
#define MSGQ_FLAGS_SWAP_RX 1  // Bidirectional communication

// Core API
unsigned msgqGetMetaSize(void);
int msgqInit(msgqHandle *pHandle, void *pBuffer);
int msgqTxCreate(msgqHandle handle, void *pBackingStore,
                 unsigned size, unsigned msgSize,
                 unsigned hdrAlign, unsigned entryAlign,
                 unsigned flags);
int msgqRxLink(msgqHandle handle, const void *pBackingStore,
               unsigned size, unsigned msgSize);
unsigned msgqTxGetFreeSpace(msgqHandle handle);
void *msgqTxGetWriteBuffer(msgqHandle handle, unsigned n);
int msgqTxSubmitBuffers(msgqHandle handle, unsigned n);
unsigned msgqRxGetReadAvailable(msgqHandle handle);
const void *msgqRxGetReadBuffer(msgqHandle handle, unsigned n);
int msgqRxMarkConsumed(msgqHandle handle, unsigned n);
```

**Features:**
- **Lock-free design** - No mutexes, suitable for real-time use
- **Bidirectional** - Separate TX and RX channels
- **Cache-coherent** - Configurable cache ops (flush/invalidate)
- **Zero-copy** - Direct buffer access
- **Backend abstraction** - Can use memory-mapped or register-based storage

**Use Cases:**
- GSP (GPU System Processor) communication
- Falcon microcontroller messaging
- Inter-processor communication in vGPU

**Implementation (msgq.c):**
- Circular buffer with read/write pointers
- Barrier operations for memory ordering
- Notification callbacks for peer wakeup

#### Device Identification

**nvdevid.h** - Device ID definitions
- PCI device IDs for all NVIDIA GPU families
- Chipset identification macros

**g_vgpu_chip_flags.h** - vGPU chip-specific flags
- Per-architecture virtualization capabilities

**g_vgpu_resman_specific.h** - vGPU resource manager specifics

#### NV Status Codes (nvstatus/)

**nvstatus.c** - Status code utilities
- Converts error codes to human-readable strings
- Status code validation
- Logging helpers

#### Compatibility (compat.h)
- Handles API changes between driver versions
- Deprecation warnings
- Forward/backward compatibility macros

#### Self-Hosted Detection (detect-self-hosted.h)
- Detects if running on native hardware or in a VM
- Influences driver behavior for virtualization

---

## 6. Softfloat Library (softfloat/)

**Location:** `/src/common/softfloat/`
**Files:** 80+ files
**License:** BSD (John R. Hauser's SoftFloat library)
**Architecture:** Architecture-specific optimizations

### Purpose
Provides software implementations of IEEE 754 floating-point arithmetic. Used when hardware floating-point is unavailable or when precise rounding control is required (e.g., in kernel mode).

### Supported Formats
- **float16** (f16) - Half precision (10-bit mantissa, 5-bit exponent)
- **float32** (f32) - Single precision (23-bit mantissa, 8-bit exponent)
- **float64** (f64) - Double precision (52-bit mantissa, 11-bit exponent)

### Operations

#### Arithmetic (source/)
- **Addition:** `f32_add.c`, `f64_add.c`
- **Subtraction:** `f32_sub.c`, `f64_sub.c`
- **Multiplication:** `f32_mul.c`, `f64_mul.c`
- **Division:** `f32_div.c`, `f64_div.c`
- **Fused Multiply-Add:** `f32_mulAdd.c`, `f64_mulAdd.c` (a×b+c with one rounding)
- **Square Root:** `f32_sqrt.c`, `f64_sqrt.c`
- **Remainder:** `f32_rem.c`, `f64_rem.c` (IEEE 754 remainder)

#### Conversions
- **Between formats:** `f16_to_f32.c`, `f32_to_f64.c`, `f64_to_f32.c`
- **Float to int:** `f32_to_i32.c`, `f32_to_i64.c`, `f64_to_i32.c`, `f64_to_i64.c`
- **Int to float:** `i32_to_f32.c`, `i64_to_f64.c`, `ui32_to_f32.c`, `ui64_to_f64.c`
- **Rounding modes:** `f32_to_i32_r_minMag.c` (round toward zero)

#### Comparisons
- **Equality:** `f32_eq.c`, `f64_eq.c`
- **Less than:** `f32_lt.c`, `f64_lt.c`
- **Less or equal:** `f32_le.c`, `f64_le.c`
- **Quiet comparisons:** `f32_lt_quiet.c` (no exception on quiet NaN)
- **Signaling:** `f32_eq_signaling.c` (signal on any NaN)

#### Utilities (s_*.c - internal subroutines)
- **s_addMagsF32.c / s_addMagsF64.c** - Add magnitudes (same sign)
- **s_subMagsF32.c / s_subMagsF64.c** - Subtract magnitudes (opposite signs)
- **s_mulAddF32.c / s_mulAddF64.c** - Fused multiply-add internals
- **s_roundPackToF32.c / s_roundPackToF64.c** - Rounding and packing
- **s_normRoundPackToF32.c** - Normalize, round, and pack
- **s_normSubnormalF32Sig.c** - Handle subnormal numbers
- **s_countLeadingZeros64.c** - Count leading zeros (for normalization)
- **s_mul64To128.c** - 64-bit × 64-bit = 128-bit multiplication
- **s_shiftRightJam128.c** - Shift with sticky bit (for rounding)

#### Rounding Modes (softfloat_state.c)
- **round_near_even** (default) - Round to nearest, ties to even
- **round_minMag** - Round toward zero (truncate)
- **round_min** - Round toward negative infinity (floor)
- **round_max** - Round toward positive infinity (ceiling)
- **round_near_maxMag** - Round to nearest, ties away from zero

#### Special Values
- **NaN handling:** `f32_isSignalingNaN.c` - Detect signaling NaN
- **Infinity** - Correctly handles +/-Inf in all operations
- **Denormals** - Full subnormal number support

### Architecture Optimizations (8086-SSE/)
- x86 SSE-optimized implementations
- Uses intrinsics for faster execution
- Falls back to portable C for other architectures

### Use Cases in Driver
- **GPU firmware** - Floating-point math in microcontroller code
- **Kernel space** - Calculations where hardware FPU is unavailable
- **Deterministic results** - Exact IEEE 754 compliance
- **Virtualization** - Emulated FPU for guest OSes

---

## 7. Unix Utilities (unix/)

**Location:** `/src/common/unix/`
**Files:** 100+ files
**Architecture:** Unix-specific (Linux, Solaris, FreeBSD)

### Purpose
Provides Unix-specific utilities for 3D rendering, command stream generation, headsurface composition, and compression. These are primarily used by the display driver and X11/Wayland compositing.

### Directory Structure

#### 7.1 NVIDIA 3D Library (nvidia-3d/)

**Interface (nvidia-3d/interface/):**
- **nvidia-3d.h** - Main 3D rendering interface
- **nvidia-3d-types.h** - 3D types (vertices, textures, shaders)
- **nvidia-3d-utils.h** - Utility functions
- **nvidia-3d-shaders.h** - Shader definitions
- **nvidia-3d-color-targets.h** - Render target formats
- **nvidia-3d-constant-buffers.h** - Uniform buffer definitions
- **nvidia-3d-shader-constants.h** - Shader constant layout

**Implementation (nvidia-3d/src/):**
Architecture-specific 3D engine programming:
- **nvidia-3d-fermi.c** (GF100+, 2010) - Fermi 3D class 0x9097
- **nvidia-3d-kepler.c** (GK100+, 2012) - Kepler 3D class 0xA097
- **nvidia-3d-maxwell.c** (GM100+, 2014) - Maxwell 3D class 0xB097
- **nvidia-3d-pascal.c** (GP100+, 2016) - Pascal 3D class 0xC097
- **nvidia-3d-volta.c** (GV100+, 2017) - Volta 3D class 0xC397
- **nvidia-3d-turing.c** (TU100+, 2018) - Turing 3D class 0xC597
- **nvidia-3d-hopper.c** (GH100+, 2022) - Hopper 3D class

**Core Files:**
- **nvidia-3d-init.c** - 3D engine initialization
- **nvidia-3d-core.c** - Core rendering operations
- **nvidia-3d-surface.c** - Surface management
- **nvidia-3d-vertex-arrays.c** - Vertex buffer handling

**Purpose:**
- Used by the display driver for desktop composition
- GPU-accelerated window rendering
- Console framebuffer operations
- Used by `nvidia-settings` for UI rendering

#### 7.2 Push Buffer Library (nvidia-push/)

**Purpose:** Generates command streams for NVIDIA GPUs

**Interface (nvidia-push/interface/):**
- Command stream generation macros
- Method encoding helpers

**Implementation (nvidia-push/src/):**
- **fermi/** - Fermi command stream (GF100+)
- **kepler/** - Kepler extensions (GK100+)
- **maxwell/** - Maxwell updates (GM100+)

**Usage:**
- Encodes GPU commands into push buffers
- Used by display driver for hardware programming
- Type-safe command submission

#### 7.3 Headsurface Library (nvidia-headsurface/)

**Purpose:** Manages display headsurface (final scanout buffer)

**Functionality:**
- Composites window surfaces into final framebuffer
- Handles overlays, cursors, and transformations
- Synchronizes with display scanout
- Used in X11 and Wayland compositing

#### 7.4 Common Unix Utilities (common/)

**Interface (common/inc/):**
- Unix-specific data structures
- Memory management helpers
- Synchronization primitives

**Implementation (common/src/):**
- Memory allocation wrappers
- File I/O helpers
- Process management

#### 7.5 XZ Compression (xzminidec/)

**Purpose:** Minimal XZ decompression library

**Files:**
- **xzminidec/inc/** - Decompression API
- **xzminidec/src/** - LZMA2 decompression implementation

**Usage:**
- Decompresses firmware blobs
- Compressed kernel modules
- On-the-fly decompression of data files

---

## 8. Microprocessor Support (uproc/)

**Location:** `/src/common/uproc/`
**Files:** 20+ files
**Architecture:** Embedded processor libraries

### Purpose
Provides libraries for communicating with embedded microprocessors in NVIDIA GPUs, including ELF/DWARF parsing, crash decoding, and logging facilities. Critical for debugging GPU firmware (GSP, Falcon, RISC-V).

### Directory Structure

#### libOS Interfaces (os/common/include/)

**libos_init_args.h** - libOS initialization
- Defines startup parameters for embedded OS
- Memory layout, heap configuration
- Interrupt vectors

**libos_status.h** - Status codes for libOS operations

**libos_log.h** - Logging interface
- Structured logging from firmware
- Ring buffer for log messages
- Severity levels (DEBUG, INFO, WARN, ERROR, FATAL)

**libos_printf_arg.h** - Printf argument parsing
- Extracts format strings and arguments from firmware logs
- Used for decoding binary log records

#### Crash Dump Analysis

**nv-crashcat.h / nv-crashcat-decoder.h** - Crash dump decoder
- Decodes GPU crash dumps from embedded processors
- Extracts register state, stack traces, memory contents

**libos_v2_crashcat.h / libos_v3_crashcat.h** - Version-specific crash formats
- libOS v2.x format (older GSP firmware)
- libOS v3.x format (current GSP firmware)

#### ELF/DWARF Support

**libelf.h / libelf.c** - ELF file parsing
```c
// Parses ELF headers, sections, symbols
// Used for loading firmware binaries
// Relocates code for different load addresses
```

**libdwarf.h / libdwarf.c** - DWARF debug information
```c
// Parses DWARF debug info in firmware ELF files
// Maps PC addresses to source lines
// Extracts function names, variables for crash dumps
```

#### Logging Decoder

**liblogdecode.h / liblogdecode.c** - Log decoder
```c
// Decodes binary log records from firmware
// Matches format string IDs to actual strings
// Expands variable arguments (ints, floats, strings)
// Produces human-readable logs
```

### libOS Versions

**os/libos-v3.1.0/** - Current version (Hopper+)
- Enhanced crash dump format
- Improved debug info
- RISC-V support (some Blackwell uControllers)

### Use Cases
1. **GSP-RM Firmware** (Hopper+)
   - GSP (GPU System Processor) runs a full firmware stack
   - libOS provides OS services (task scheduling, IPC, memory management)
   - Driver loads GSP firmware ELF, relocates, executes
   - liblogdecode converts binary GSP logs to text

2. **Falcon Microcontrollers**
   - Various Falcon processors (PMU, SEC2, NVDEC, etc.)
   - Each runs custom firmware
   - Crash dumps decoded with libdwarf

3. **Debug Tools**
   - `nvidia-bug-report.sh` uses these libraries
   - Extracts crash dumps from `/sys/kernel/debug/dri/*/gsp/`
   - Symbolizes stack traces

---

## 9. Modeset Utilities (modeset/)

**Location:** `/src/common/modeset/`
**Files:** 30+ files
**Architecture:** Display protocol utilities

### Purpose
Provides mode-setting utilities including HDMI packet generation and display timing calculations. These are used by the display driver for programming monitor timings and generating infoframes.

### Directory Structure

#### 9.1 HDMI Packet Library (hdmipacket/)

**Purpose:** Generates HDMI infoframes and packets per HDMI specification

**nvhdmipkt.h** - Main interface
```c
// Packet types
- AVI InfoFrame (video format, colorimetry, aspect ratio)
- Audio InfoFrame (channel count, sample rate)
- Vendor Specific InfoFrame (3D, 4K, HDR)
- GCP (General Control Packet) - deep color indication
- ISRC (International Standard Recording Code)
- ACP (Audio Content Protection)
- SPD (Source Product Descriptor)
- Dynamic Range and Mastering InfoFrame (HDR metadata)
```

**Architecture-specific implementations:**
Each file handles HDMI packet generation for specific GPU families:
- **nvhdmipkt_0073.c** - Common packet generation (base)
- **nvhdmipkt_9171.c** - Fermi (GF119, HDMI 1.4)
- **nvhdmipkt_9271.c** - Kepler (GK104, HDMI 1.4a)
- **nvhdmipkt_9471.c** - Maxwell (GM204, HDMI 2.0)
- **nvhdmipkt_9571.c** - Pascal (GP104, HDMI 2.0b)
- **nvhdmipkt_C371.c** - Volta (GV100, HDMI 2.0b)
- **nvhdmipkt_C671.c** - Turing (TU104, HDMI 2.0b)
- **nvhdmipkt_C771.c** - Ampere (GA102, HDMI 2.1)
- **nvhdmipkt_C871.c** - Ada Lovelace (AD102, HDMI 2.1a)
- **nvhdmipkt_C971.c** - Hopper (GH100, HDMI 2.1a)
- **nvhdmipkt_CA71.c** - (Future architecture)
- **nvhdmipkt_CB71.c** - Blackwell (GB100, HDMI 2.1b)
- **nvhdmipkt_CC71.c** - (Future architecture)

**nvhdmi_frlInterface.h** - HDMI FRL (Fixed Rate Link)
- HDMI 2.1 high-speed link training
- 48 Gbps bandwidth for 4K120, 8K60, 10K
- FRL rates: 3, 6, 8, 10, 12 Gbps per lane (4 lanes)

**nvhdmipkt_common.h** - Shared packet structures

**nvhdmipkt_internal.h** - Internal helpers

**nvhdmipkt_class.h** - Hardware class definitions

#### 9.2 Display Timing Library (timing/)

**Purpose:** Calculates and validates display timings

**nvt_util.c** - Timing utilities
```c
// Functions for:
- Calculating pixel clock from refresh rate
- Validating CVT (Coordinated Video Timings) modes
- Validating GTF (Generalized Timing Formula) modes
- Calculating blanking intervals
- Handling reduced blanking (CVT-RB, CVT-RB2)
- DSC (Display Stream Compression) timing adjustments
```

**nvt_dsc_pps.h** - DSC PPS (Picture Parameter Set)
- Encodes DSC parameters per VESA DSC standard
- PPS packet sent to display to configure decompressor
- Parameters: slice width/height, bpp, line buffer depth, etc.

### HDMI Features Supported
- **HDMI 1.4** - 4K30, 3D, ARC
- **HDMI 2.0** - 4K60, HDR (static), HDCP 2.2
- **HDMI 2.1** - 8K60, 4K120, VRR (Variable Refresh Rate), DSC, FRL

### Display Timing Standards
- **DMT** (VESA Discrete Monitor Timings)
- **CVT** (Coordinated Video Timings)
- **GTF** (Generalized Timing Formula)
- **CVT-RB** (Reduced Blanking for flat panels)
- **CVT-RB2** (Reduced Blanking v2 for even less overhead)

---

## 10. Common Headers (inc/)

**Location:** `/src/common/inc/`
**Files:** 600+ header files
**Architecture:** Hardware register definitions

### Purpose
Contains hardware reference documentation and register definitions for all NVIDIA GPU architectures. This is the "hardware manual" in code form, defining every MMIO register, bitfield, and hardware constant.

### Directory Structure

#### Software Reference (swref/published/)

**Architecture Families:**
- **kepler/** (GK100, 2012) - Kepler architecture
- **maxwell/** (GM100, 2014) - Maxwell architecture (gm107/, gm200/)
- **pascal/** (GP100, 2016) - Pascal architecture (gp100/, gp102/)
- **volta/** (GV100, 2017) - Volta architecture (gv100/, gv11b/)
- **turing/** (TU100, 2018) - Turing architecture (headers in ada/)
- **ampere/** (GA100, 2020) - Ampere architecture
- **ada/** (AD100, 2022) - Ada Lovelace architecture
- **hopper/** (GH100, 2022) - Hopper architecture (gh100/)
- **blackwell/** (GB100, 2024) - Blackwell architecture

**Tegra SoCs:**
- **t23x/t234/** - Tegra Orin (Xavier derivative)

**Infrastructure:**
- **br03/** - Bridge chip registers
- **br04/** - Bridge chip v4
- **pcie_switch/** - PCIe switch chips
- **nvswitch/** - NVSwitch fabric switch registers

**Display:**
- **disp/** - Display engine registers (common across architectures)

### Header File Contents

Each architecture directory contains headers like:
- **dev_\*.h** - Device register definitions
  - `dev_bus.h` - Bus interface (PCIe, NVLink)
  - `dev_fb.h` - Framebuffer controller
  - `dev_fifo.h` - Channel FIFO
  - `dev_gr.h` - Graphics engine
  - `dev_ce.h` - Copy engine
  - `dev_disp.h` - Display engine
  - `dev_mc.h` - Memory controller
  - `dev_mmu.h` - MMU/IOMMU
  - `dev_timer.h` - GPU timers
  - `dev_pri.h` - Privileged registers
  - `dev_xtl.h` - External interface

**Register Definition Format:**
```c
// Example from dev_fb.h
#define NV_PFB_PRI_MMU_CTRL                     0x00100200
#define NV_PFB_PRI_MMU_CTRL_ATOMIC_CAPABILITY   1:1
#define NV_PFB_PRI_MMU_CTRL_ATOMIC_CAPABILITY_ENABLED 0x00000001
#define NV_PFB_PRI_MMU_CTRL_ATOMIC_CAPABILITY_DISABLED 0x00000000

// Register at offset 0x00100200
// Bit 1 controls atomic capability
// Value 1 = enabled, 0 = disabled
```

#### Top-Level Headers

**nv_ref.h** - Common reference header
- Includes architecture-specific headers based on chip ID
- Provides abstraction macros

**nv_arch.h** - Architecture detection
- Macros for detecting GPU architecture
- `IS_KEPLER(pGpu)`, `IS_PASCAL(pGpu)`, etc.

### Usage
- Hardware register access in kernel driver
- Firmware references these for MMIO
- Debug tools for register dumping
- Validation tests use expected values

### Completeness
These headers are **publicly released** as part of the open-source driver. They represent NVIDIA's hardware documentation for GPU internals, previously only available under NDA.

---

## 11. Common Source (src/)

**Location:** `/src/common/src/`
**Files:** 2 files
**Architecture:** Standalone implementations

### Purpose
Contains common source implementations shared across the driver.

### Files

#### nvSha256.c - SHA-256 Implementation
```c
// Software implementation of SHA-256 cryptographic hash
// Used for:
- Firmware signature verification
- Secure boot chain of trust
- Config file integrity checks
- Random number generation (seeding)
```

**Features:**
- FIPS 180-4 compliant
- Operates on 512-bit blocks
- Produces 256-bit hash
- No external dependencies (can run in firmware)

#### nv_smg.c - SMG (Shared Memory Gateway)
```c
// Shared memory management for inter-processor communication
// Used for:
- GSP-RM communication (CPU ↔ GSP)
- Falcon message passing
- vGPU host-guest shared buffers
```

**Features:**
- Coherent shared memory regions
- Synchronization primitives
- Message passing semantics
- Cache management (flush/invalidate)

---

## 12. Hardware Abstraction Layers and Cross-Component Dependencies

### Layered Architecture

```
┌─────────────────────────────────────────────────────────┐
│          Display Driver / CUDA Runtime / Vulkan         │
│                    (Userspace)                          │
└─────────────────────────────────────────────────────────┘
                          ↓ ioctl
┌─────────────────────────────────────────────────────────┐
│            Resource Manager (nvidia.ko)                 │
│          (src/nvidia/ - not in common/)                 │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                 Common Libraries                        │
│  ┌──────────────────────────────────────────────────┐  │
│  │ DisplayPort   NVLink   NVSwitch   Modeset       │  │
│  │ (Protocol     (Interconnect)      (Timing/HDMI) │  │
│  │  Stack)                                          │  │
│  └──────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────┐  │
│  │ SDK Headers   Softfloat   MSGQ   Unix Utils     │  │
│  │ (API Defs)    (FP Math)   (IPC)  (3D/Push/XZ)   │  │
│  └──────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Uproc (libOS, ELF, DWARF, CrashCat)             │  │
│  │ Shared (nvstatus, nvdevid, compat)              │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│          Hardware (GPU, NVLink, NVSwitch)               │
│   inc/swref/published/* - Register Definitions          │
└─────────────────────────────────────────────────────────┘
```

### Key Dependencies

#### DisplayPort → SDK + Modeset + Shared
- Uses SDK control commands (`ctrl0073dp.h`) for hardware programming
- Uses modeset timing library for mode validation
- Uses shared utilities for device IDs

#### NVLink → SDK + MSGQ + Shared
- Uses SDK types (`nvgputypes.h`, `nvstatus.h`)
- Uses MSGQ for inband messaging
- Depends on PCI device detection from shared/

#### NVSwitch → NVLink + Uproc + SDK
- Built on NVLink library for link management
- Uses uproc libraries for Falcon firmware
- Uses SDK control interfaces

#### Display Driver → DisplayPort + Modeset + NVLink
- DisplayPort library for DP/MST
- Modeset library for HDMI and timing
- NVLink for multi-GPU display (Mosaic mode)

#### GSP-RM Firmware → Uproc + MSGQ + Softfloat
- GSP firmware uses libOS from uproc/
- MSGQ for CPU-GSP communication
- Softfloat for firmware math

---

## 13. Compilation and Build System Integration

### Compilation Units

The common libraries are compiled into the `nvidia.ko` kernel module:

```makefile
# Simplified build flow
nvidia.ko:
    - src/nvidia/*.c (core driver)
    - src/common/displayport/src/*.cpp (DisplayPort library)
    - src/common/nvlink/kernel/nvlink/*.c (NVLink library)
    - src/common/shared/msgq/*.c (Message queue)
    - src/common/softfloat/source/*.c (Softfloat)
    - src/common/unix/nvidia-3d/src/*.c (3D utilities)
    # ... etc
```

### Conditional Compilation

Many components have conditional compilation:
- **DisplayPort:** Only if `NV_BUILD_SUPPORTS_DISPLAYPORT`
- **NVLink:** Only on platforms with NVLink support
- **NVSwitch:** Only if `NV_BUILD_SUPPORTS_NVSWITCH`
- **Architecture-specific code:** Compiled based on supported GPU families

### Header Include Paths

```c
// Typical include order in driver code:
#include "nvtypes.h"              // src/common/sdk/nvidia/inc/
#include "nvstatus.h"             // src/common/sdk/nvidia/inc/
#include "nv-kernel-interface-api.h" // src/common/sdk/nvidia/inc/
#include "dev_fb.h"               // src/common/inc/swref/published/hopper/gh100/
#include "dp_connector.h"         // src/common/displayport/inc/
#include "nvlink.h"               // src/common/nvlink/interface/
```

---

## 14. Key Interfaces and APIs

### 14.1 DisplayPort Library API

```c++
// Create DisplayPort connector
DisplayPort::Connector *connector =
    DisplayPort::createConnector(mainLink, auxBus, timer, eventSink);

// Resume from suspend
Group *firmwareGroup = connector->resume(
    firmwareLinkHandsOff,
    firmwareDPActive,
    plugged,
    isUefiSystem,
    firmwareHead,
    bFirmwareLinkUseMultistream,
    bDisableVbiosScratchRegisterUpdate,
    bAllowMST
);

// Handle hotplug
connector->notifyLongPulse(statusConnected);

// Enumerate devices
for (Device *dev = connector->enumDevices(NULL);
     dev;
     dev = connector->enumDevices(dev)) {
    // Process device
    unsigned edidSize = dev->getEDIDSize();
    char *edid = new char[edidSize];
    dev->getEDID(edid, edidSize);
    // ...
}

// Create group and attach to head
Group *group = connector->newGroup();
group->insert(device1);
group->insert(device2); // MST configuration

// Validate mode
DpModesetParams params;
params.headIndex = 0;
params.modesetInfo.pixelClockHz = 148500000; // 1920x1080@60
params.modesetInfo.rasterWidth = 1920;
params.modesetInfo.rasterHeight = 1080;
// ...

connector->beginCompoundQuery();
bool possible = connector->compoundQueryAttach(
    group, &params, &dscParams, &errorStatus);
connector->endCompoundQuery();

// Perform modeset
if (possible) {
    connector->notifyAttachBegin(group, params);
    // Program GPU hardware
    connector->notifyAttachEnd(false);
}
```

### 14.2 NVLink Library API

```c
// Register device
nvlink_device device = {
    .deviceId = ...,
    .type = NVLINK_DEVICE_TYPE_GPU,
    .deviceName = "GPU0",
    .uuid = gpuUuid,
    .enableALI = NV_TRUE,
};
nvlink_lib_register_device(&device);

// Register link
nvlink_link link = {
    .dev = &device,
    .linkNumber = 0,
    .version = NVLINK_DEVICE_VERSION_40,
    .link_handlers = &gpu_link_handlers,
};
nvlink_lib_register_link(&device, &link);

// Discover topology
nvlink_conn_info conn_info;
nvlink_lib_discover_and_get_remote_conn_info(
    &link, &conn_info, 0, NV_FALSE);

// Train links
nvlink_link *links[] = {&link0, &link1, &link2};
nvlink_lib_train_links_from_swcfg_to_active(
    links, 3, NVLINK_STATE_CHANGE_SYNC);

// Check training result
if (link.state == NVLINK_LINKSTATE_HS) {
    // Link active at high speed
    // NVLink 4.0: 100 GB/s per link
}
```

### 14.3 Message Queue API

```c
// Allocate metadata
void *meta = malloc(msgqGetMetaSize());
msgqHandle tx, rx;

// Initialize TX queue
msgqInit(&tx, meta);
msgqSetNotification(tx, notifyGspWrite, NULL);
msgqTxCreate(tx, txBuffer, 4096, 64, 12, 6, 0);

// Initialize RX queue
msgqInit(&rx, meta);
msgqSetNotification(rx, notifyGspRead, NULL);
msgqRxLink(rx, rxBuffer, 4096, 64);

// Send message
unsigned freeSlots = msgqTxGetFreeSpace(tx);
if (freeSlots > 0) {
    void *msg = msgqTxGetWriteBuffer(tx, 0);
    memcpy(msg, &myData, sizeof(myData));
    msgqTxSubmitBuffers(tx, 1); // Notifies GSP
}

// Receive message
msgqRxSync(rx); // Invalidate cache
unsigned available = msgqRxGetReadAvailable(rx);
for (unsigned i = 0; i < available; i++) {
    const void *msg = msgqRxGetReadBuffer(rx, i);
    processMessage(msg);
}
msgqRxMarkConsumed(rx, available); // Notifies sender
```

### 14.4 HDMI Packet API

```c
// Create AVI InfoFrame
HDMI_AVI_INFOFRAME avi = {0};
avi.header.type = 0x82;
avi.header.version = 2;
avi.header.length = 13;
avi.videoIdCode = 16; // 1920x1080@60
avi.colorSpace = HDMI_COLORSPACE_RGB;
avi.activeFormatAspectRatio = HDMI_AFAR_SAME_AS_PICTURE;
avi.pictureAspectRatio = HDMI_PAR_16_9;
avi.colorimetry = HDMI_COLORIMETRY_ITU709;
avi.scanInfo = HDMI_SCAN_UNDERSCAN;

// Generate packet
NV_HDMI_PKT pkt;
nvHdmiPkt_GenAviInfoFrame(pGpu, pDev, &avi, &pkt);

// Send to hardware
nvHdmiPkt_Write(pGpu, pDev, &pkt);

// HDR metadata (HDMI 2.0a)
HDMI_DRM_INFOFRAME drm = {0};
drm.eotf = HDMI_EOTF_ST2084; // PQ (Perceptual Quantizer)
drm.staticMetadataType = 0;
drm.displayPrimaries[0].x = 0.708 * 50000; // Red X (BT.2020)
drm.displayPrimaries[0].y = 0.292 * 50000; // Red Y
// ... more primaries
drm.maxDisplayMasteringLuminance = 1000 * 10000; // 1000 cd/m²
drm.minDisplayMasteringLuminance = 0.0001 * 10000;
drm.maxContentLightLevel = 1000; // MaxCLL
drm.maxFrameAverageLightLevel = 400; // MaxFALL

nvHdmiPkt_GenDrmInfoFrame(pGpu, pDev, &drm, &pkt);
nvHdmiPkt_Write(pGpu, pDev, &pkt);
```

---

## 15. Performance and Optimization Considerations

### DisplayPort Library
- **Fast Link Training (FLT)** - Reuses training parameters to avoid full re-training
- **Parallel MST enumeration** - Discovers multiple branches concurrently
- **Bandwidth pre-calculation** - Compound queries avoid trial-and-error modesets
- **DSC dynamic toggling** - Enables/disables DSC without full re-train (DP 1.4a)

### NVLink Library
- **Low-latency training** - Parallel link training across all links
- **ALI (Adaptive Link Interface)** - Optimized signal integrity calibration (NVLink 3.0+)
- **Seed data caching** - Saves training results for fast resume from L2
- **CCI-managed links** - Offloads link management to hardware (Hopper+)

### Message Queue (MSGQ)
- **Lock-free** - Uses atomic read/write pointers, no mutexes
- **Zero-copy** - Direct buffer access, no memcpy
- **Batching** - Submit/consume multiple messages at once
- **Cache-optimized** - Aligned structures, explicit flush/invalidate

### Softfloat
- **Architecture-specific** - Uses SSE intrinsics on x86
- **Lazy normalization** - Defers normalization until final round
- **Fused operations** - FMA (fused multiply-add) with single rounding

### Unix Utilities
- **GPU-accelerated composition** - Uses 3D engine for window blending
- **Push buffer coalescing** - Batches commands to reduce overhead
- **XZ streaming** - Decompresses firmware on-the-fly without temporary files

---

## 16. Debugging and Diagnostics

### Debug Facilities

#### DisplayPort
- **dp_tracing.h** - Event tracing for link training, discovery, bandwidth
- Debug logs controlled by registry keys (Windows) / module parameters (Linux)
- Example: `DISPLAYPORT_LOG_LEVEL=4` enables verbose logging

#### NVLink
- **NVLINK_PRINT()** macro - Conditional compilation for debug prints
- Link state history in debugfs: `/sys/kernel/debug/dri/0/nvlink`
- Per-link error counters (CRC errors, replay, NACKs)

#### NVSwitch
- **InfoROM blackbox** - Records all errors with timestamps
- Telemetry via `nvidia-smi nvlink --status`
- Crash dump in `/sys/kernel/debug/nvswitch`

#### GSP-RM
- **liblogdecode** - Decodes binary logs to text
- `/sys/kernel/debug/dri/0/gsp/logs` - GSP log buffer
- Crash dumps include ELF core with full register state

### Common Debug Tools

**nvidia-bug-report.sh**
- Collects logs, register dumps, crash dumps
- Uses uproc libraries to decode firmware state
- Generates tarball for NVIDIA support

**nvidia-settings**
- GUI tool for configuration
- Uses SDK control commands to query/set parameters
- DisplayPort MST topology viewer

**nvidia-smi**
- Command-line tool for monitoring
- NVLink bandwidth counters
- PCIe bandwidth utilization
- Temperature, power, clocks

---

## 17. Security Considerations

### Secure Boot Chain
- **SHA-256 verification** (nvSha256.c) validates firmware signatures
- Boot ROM → BootLoader → GSP-RM → Falcon firmware
- Each stage verifies the next

### Firmware Isolation
- GSP-RM runs in privileged mode on GPU
- CPU driver communicates via MSGQ (limited interface)
- Prevents CPU tampering with critical GPU state

### HDCP Support
- DisplayPort library implements HDCP 1.x and 2.x
- Key management handled by hardware
- Encryption status exposed via `Group::hdcpGetEncrypted()`

### Vulnerability Mitigation
- No user-controlled buffer sizes in MSGQ (fixed at init)
- All DPCD/I2C reads validated for size
- Firmware ELF loading checks for overlapping sections

---

## 18. Future Directions and Evolution

### Hopper and Beyond (Current)
- **GSP-RM offload** - More driver logic moves to GPU firmware
- **Confidential Computing** - Encrypted GPU memory for VMs
- **NVLink 4.0/5.0** - Higher bandwidths (100-150 GB/s per link)
- **DisplayPort 2.1** - 80 Gbps bandwidth, UHBR rates
- **HDMI 2.1b** - Fixed Rate Link (FRL) at 48 Gbps

### Blackwell (2024-2025)
- **NVSwitch 3.0** - 64 ports for massive fabrics
- **RISC-V uControllers** - Some Falcon cores replaced with RISC-V
- **DisplayPort 2.1 UHBR20** - 20 Gbps per lane (80 Gbps total)
- **Enhanced DSC** - Display Stream Compression 1.2a

### Architecture Trends
- **More firmware offload** - libOS complexity increases
- **Hardware-managed coherence** - NVLink evolves toward full cache coherence
- **Unified memory** - CPU/GPU share page tables (Grace Hopper)
- **AI-optimized interconnects** - Custom topologies for LLM training

---

## 19. Conclusion

The `src/common/` directory is the **foundational layer** of the NVIDIA open GPU kernel driver, providing:

1. **Protocol Stacks** - Complete DisplayPort and HDMI implementations
2. **Interconnect Management** - NVLink and NVSwitch for multi-GPU systems
3. **Hardware Abstraction** - Register definitions for all GPU architectures
4. **Utility Libraries** - Floating-point math, message queues, compression
5. **Firmware Support** - ELF loading, crash decoding, logging for embedded processors
6. **Unix Integration** - 3D rendering, command streams, display composition

These libraries are **architecture-agnostic** where possible, with **HAL (Hardware Abstraction Layer)** patterns for chip-specific code. They enable the driver to support:
- **10+ GPU architectures** (Kepler to Blackwell)
- **Multiple display standards** (DP 1.2/1.4/2.0, HDMI 1.4/2.0/2.1)
- **High-speed interconnects** (NVLink 1.0-5.0, up to 900 GB/s)
- **Complex topologies** (MST displays, NVSwitch fabrics with 1000s of GPUs)

The open-source release of these libraries (MIT license) represents **decades of NVIDIA engineering**, previously only available under NDA. This enables:
- **Community contributions** to improve Linux GPU support
- **Academic research** into GPU architecture and interconnects
- **Custom hardware integration** (e.g., ARM servers with NVLink)
- **Bug fixes and optimizations** from the wider developer community

---

## 20. Summary of Most Important Common Libraries

### Critical Path Libraries

1. **DisplayPort Library (displayport/)** - 41 files, C++
   - **Role:** Complete DP 1.2/1.4/2.0 protocol stack for modern displays
   - **Importance:** Essential for all external monitor support (SST/MST)
   - **Key Features:** Link training, MST topology, DSC, HDCP, VRR

2. **NVLink Library (nvlink/)** - 30+ files, C
   - **Role:** Manages high-speed GPU-GPU interconnects (20-150 GB/s per link)
   - **Importance:** Critical for multi-GPU systems (DGX, HGX platforms)
   - **Key Features:** Link training, topology discovery, power management, ALI

3. **SDK Headers (sdk/nvidia/inc/)** - 700+ files
   - **Role:** Defines entire GPU driver API (types, controls, classes)
   - **Importance:** Contract between userspace and kernel, enables CUDA/Vulkan
   - **Key Features:** Control commands, allocation interfaces, status codes

4. **Hardware Reference (inc/swref/published/)** - 600+ files
   - **Role:** Register definitions for all GPU architectures
   - **Importance:** Hardware programming manual in code form
   - **Key Features:** MMIO registers, bitfields, constants for 10+ GPU families

5. **Message Queue (shared/msgq/)** - 2 files, lock-free IPC
   - **Role:** Inter-processor communication (CPU ↔ GSP, CPU ↔ Falcon)
   - **Importance:** Foundation for GSP-RM offload architecture (Hopper+)
   - **Key Features:** Zero-copy, lock-free, cache-coherent

### Supporting Libraries

6. **NVSwitch (nvswitch/)** - 100+ files
   - **Role:** Fabric switch management for large GPU clusters
   - **Importance:** Enables non-blocking multi-GPU connectivity (DGX SuperPOD)

7. **Modeset Utilities (modeset/)** - 30+ files
   - **Role:** HDMI packet generation and display timing calculations
   - **Importance:** Required for all HDMI displays (4K, 8K, HDR)

8. **Microprocessor Support (uproc/)** - 20+ files
   - **Role:** ELF loading, crash decoding, logging for GPU firmware
   - **Importance:** Enables debugging of GSP-RM and Falcon firmware

9. **Unix Utilities (unix/)** - 100+ files
   - **Role:** 3D rendering, command streams, composition, compression
   - **Importance:** Display driver and console functionality

10. **Softfloat (softfloat/)** - 80+ files
    - **Role:** IEEE 754 floating-point arithmetic in software
    - **Importance:** Firmware calculations, deterministic math

---

**Total:** 1,391 files forming the core of NVIDIA's open-source GPU driver stack.
