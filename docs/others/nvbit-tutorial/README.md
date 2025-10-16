# NVBit Tutorial: Comprehensive Guide to GPU Binary Instrumentation

> Tutorial Github repo: <https://github.com/eunomia-bpf/nvbit-tutorial>
>
> Official NVBit repo: <https://github.com/NVlabs/NVBit>

This repository provides a comprehensive, blog-style tutorial (Unofficial) for learning NVIDIA Binary Instrumentation Tool (NVBit). It offers detailed, step-by-step guidance with in-depth explanations of code to help you understand GPU binary instrumentation concepts and techniques.

NVBit is covered by the same End User License Agreement as that of the
NVIDIA CUDA Toolkit. By using NVBit you agree to End User License Agreement
described in the EULA.txt file.


## Table of Contents

- [Quick Start (5 Minutes)](#quick-start-5-minutes)
- [About This Tutorial Repository](#about-this-tutorial-repository)
- [Prerequisites](#prerequisites)
- [Introduction to NVBit](#introduction-to-nvbit)
- [Requirements](#requirements)
- [Building the Tools](#building-the-tools-and-test-applications)
- [Using NVBit Tools](#using-an-nvbit-tool)
- [Key Concepts](#key-concepts-covered-in-this-tutorial)
- [Creating Your Own Tools](#creating-your-own-tools)
- [FAQ](#frequently-asked-questions)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## Quick Start (5 Minutes)

Get started with NVBit in just a few minutes:

```bash
# Ensure CUDA toolkit is installed and nvdisasm is in PATH
export PATH=/usr/local/cuda/bin:$PATH

# Clone this repository (if you haven't already)
cd nvbit-tutorial

# Build all tools
cd tools && make && cd ..

# Build test applications
cd test-apps && make && cd ..

# Run your first instrumentation (instruction counting)
LD_PRELOAD=./tools/instr_count/instr_count.so ./test-apps/vectoradd/vectoradd

# You should see output like:
# kernel 0 - vecAdd(...) - #thread-blocks 98, kernel instructions 50077, total instructions 50077
```

**Next Steps:** Read the [instr_count tutorial](tools/instr_count/README.md), try [opcode_hist](tools/opcode_hist/README.md) to analyze instruction mix, or see [FAQ](#frequently-asked-questions) if you encounter issues.

## About This Tutorial Repository

This tutorial repository goes beyond basic examples with detailed blog-style documentation for each tool with comprehensive code explanations, step-by-step implementation guides showing how each tool is built, visual diagrams and examples to illustrate key concepts, best practices and performance considerations, and extension ideas for developing your own custom tools.

The repository contains the core NVBit library (`core/`) with the main library and header files, example tools (`tools/`) with practical instrumentation tools and detailed explanations, and test applications (`test-apps/`) with simple CUDA applications to demonstrate the tools.

Each tool in the `tools/` directory includes a comprehensive tutorial README that walks through the code line-by-line, explains the build process, and describes how to interpret the output.

## Prerequisites

### Required Software

**CUDA Toolkit** (>= 12.0) - Download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads). Verify installation: `nvcc --version`

**nvdisasm** (included with CUDA Toolkit)
```bash
# Add CUDA bin directory to PATH
export PATH=/usr/local/cuda/bin:$PATH
# Or for specific version:
export PATH=/usr/local/cuda-12.8/bin:$PATH

# Verify nvdisasm is accessible
which nvdisasm
```
**⚠️ IMPORTANT:** NVBit requires `nvdisasm` to be in your PATH. Without it, tools will fail to run.

**GCC** (>= 8.5.0 for x86_64; >= 8.5.0 for aarch64) - Verify: `gcc --version`

**Make** - Verify: `make --version`

### Supported Hardware

- **GPU Compute Capability:** SM 3.5 to SM 12.1
- **Architecture:** x86_64, aarch64 (ARM64)

### Environment Setup

Add these to your `~/.bashrc` or `~/.zshrc`:

```bash
# CUDA paths
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

Then reload: `source ~/.bashrc`

## Introduction to NVBit

NVBit (NVIDIA Binary Instrumentation Tool) is a research prototype of a dynamic
binary instrumentation library for NVIDIA GPUs.

NVBit provides a set of simple APIs that enable writing a variety of
instrumentation tools. Example of instrumentation tools are: dynamic
instruction counters, instruction tracers, memory reference tracers,
profiling tools, etc.

NVBit allows writing instrumentation tools (which we call **NVBit tools**)
that can inspect and modify the assembly code (SASS) of a GPU application
without requiring recompilation, thus dynamic. NVBit allows instrumentation
tools to inspect the SASS instructions of each function (\_\_global\_\_ or
\_\_device\_\_) as it is loaded for the first time in the GPU. During this
phase is possible to inject one or more instrumentation calls to arbitrary
device functions before (or after) a SASS instruction. It is also possible to
remove SASS instructions, although in this case NVBit does not guarantee that
the application will continue to work correctly.

NVBit tries to be as low overhead as possible, although any injection of
instrumentation function has an associated cost due to saving and restoring
application state before and after jumping to/from the instrumentation
function.

Because NVBit does not require application source code, any pre-compiled GPU
application should work regardless of which compiler (or version) has been
used (i.e. nvcc, pgicc, etc).

## Requirements

### Minimum Requirements

* **SM compute capability:** >= 3.5 && <= 12.1
* **Host CPU:** x86_64, aarch64
* **OS:** Linux
* **GCC version:** >= 8.5.0 for x86_64; >= 8.5.0 for aarch64
* **CUDA version:** >= 12.0
* **CUDA driver version:** <= 575.xx

### Supported Platform Range

| Component | Supported Range | Tested Version |
|-----------|----------------|----------------|
| **SM Architecture** | 3.5 - 12.1 | 12.0 (RTX 5090) |
| **CUDA Toolkit** | 12.0+ | 12.8 |
| **NVIDIA Driver** | 520.xx - 575.xx | 575.57.08 |
| **GCC** | 8.5.0 - 14.x | 14.2.0 |
| **Operating System** | Linux (Ubuntu 20.04+, RHEL 8+, etc.) | Ubuntu 24.04.3 LTS |

## Tested Platform

This repository has been successfully tested on the following configuration:

| Component | Version/Details |
|-----------|----------------|
| **GPU** | NVIDIA GeForce RTX 5090 (SM 12.0) |
| **CUDA Toolkit** | 12.8 (V12.8.93) |
| **NVIDIA Driver** | 575.57.08 |
| **NVBit Version** | 1.7.6 |
| **Operating System** | Ubuntu 24.04.3 LTS |
| **Kernel** | Linux 6.14.0-1007-intel |
| **GCC** | 14.2.0 |
| **Architecture** | x86_64 |

**Build Configuration:**
- Tools linked with g++ (not nvcc) to avoid CUDA 12.8+ device linking issues
- Test applications compiled with `-cudart shared` flag
- All 8 example tools compile successfully
- 7 out of 8 tools tested and working (mem_printf2 has known device function issue - see [FAQ](#frequently-asked-questions))

## Getting Started with NVBit

This repository uses **NVBit v1.7.6** which includes support for newer CUDA versions and SM architectures up to SM_120.

NVBit is provided in this repository with the `core` folder containing the main static library `libnvbit.a` and header files (including `nvbit.h` with all main NVBit APIs declarations), the `tools` folder with various source code examples and detailed tutorial documentation, and the `test-apps` folder with simple applications to test NVBit tools (like a vector addition program). After learning from these examples, you can copy and modify one to create your own tool.

## Building the Tools and Test Applications

### Building All Tools

```bash
cd tools
make
```

This compiles all 8 NVBit tools into shared libraries (`.so` files).

### Building Test Applications

```bash
cd test-apps
make
```

This compiles the test applications that you can use to try out the tools.

### Building Individual Tools

```bash
cd tools/instr_count
make
```

### Cleaning Build Artifacts

```bash
# Clean all tools
cd tools && make clean

# Clean test apps
cd test-apps && make clean
```

### Important Build Notes for CUDA 12.x+

**Why we use g++ instead of nvcc for linking:**

CUDA 12.8+ introduced stricter device code linking requirements. To avoid "undefined reference to device functions" errors, we:
- Compile device code with `nvcc -dc` (device code compilation)
- **Link the final shared library with g++** (not nvcc)
- Include necessary CUDA libraries explicitly: `-lcuda -lcudart_static -lpthread -ldl`

**Makefile Pattern:**
```makefile
# Compile device functions
$(NVCC) -dc inject_funcs.cu -o inject_funcs.o

# Link with g++ (NOT nvcc)
g++ -shared -o tool.so $(OBJECTS) -L$(CUDA_LIB) -lcuda -lcudart_static
```

The provided Makefiles handle this automatically, but if you create your own tool, follow this pattern.

**Test Application Compilation:**
- Use `-cudart shared` flag: `nvcc -cudart shared your_app.cu -o your_app`
- This ensures the CUDA runtime is dynamically linked

### Rebuilding After Changes

If you modify device code (`inject_funcs.cu`), you must rebuild the entire tool:

```bash
cd tools/your_tool
make clean
make
```

## Using an NVBit Tool

Before running an NVBit tool, make sure ```nvdisasm``` is in your PATH. In
Ubuntu distributions, this is typically done by adding `/usr/local/cuda/bin` or
`/usr/local/cuda-"version"/bin` to the PATH environment variable.

To use an NVBit tool, either LD_PRELOAD the tool before the application command:
```bash
LD_PRELOAD=./tools/instr_count/instr_count.so ./test-apps/vectoradd/vectoradd
```

Or use CUDA_INJECTION64_PATH:
```bash
CUDA_INJECTION64_PATH=./tools/instr_count/instr_count.so ./test-apps/vectoradd/vectoradd
```

**NOTE**: NVBit uses the same mechanism as nvprof, nsight system, and nsight compute,
thus they cannot be used together.

## Key Concepts Covered in This Tutorial

Throughout the tutorial, you'll learn important concepts in GPU binary instrumentation: SASS instruction analysis (understanding GPU assembly), function instrumentation (adding code to existing GPU functions), basic block analysis (working with control flow graphs), memory access tracking (capturing and analyzing memory patterns), efficient GPU-CPU communication, register manipulation (reading and writing GPU registers directly), instruction replacement (modifying GPU code behavior), and performance optimization (minimizing instrumentation overhead).

## Creating Your Own Tools

After working through the examples, you'll be ready to create your own custom instrumentation tools. The repository includes templates and guidance for tool structure (understanding host/device code organization), build systems (setting up Makefiles), common patterns (reusing code for frequently needed functionality), and debugging techniques (troubleshooting instrumentation issues).

## Frequently Asked Questions

### General Questions

**What is NVBit?**
NVBit is a research tool that lets you analyze and modify GPU code at the binary level without recompiling.

**Do I need source code?**
No! NVBit works on compiled CUDA binaries.

**Can I use NVBit with nvprof or Nsight?**
No. They use the same injection mechanism and cannot run simultaneously.

**Which GPUs are supported?**
SM 3.5 to SM 12.1 (Kepler to Blackwell architecture).

### Using the Tools

**Which tool should I start with?**
Start with `instr_count`, then try `opcode_hist`.

**The tool produces too much output. How do I reduce it?**
```bash
# Instrument only first kernel
KERNEL_BEGIN=0 KERNEL_END=1 LD_PRELOAD=./tools/instr_count/instr_count.so ./app

# Instrument only first 100 instructions
INSTR_END=100 LD_PRELOAD=./tools/instr_count/instr_count.so ./app
```

**My application runs 100x slower with instrumentation**
Normal for instruction-level tools. Use `instr_count_bb` for lower overhead, or instrument selectively.

**What's the overhead of each tool?**
- `instr_count_bb`: 2-5x
- `instr_count_cuda_graph`: 5-20x
- `instr_count`/`opcode_hist`: 20-100x
- `mem_trace`/`mov_replace`/`record_reg_vals`: 100-1000x

**Can I use these tools in production?**
Only `instr_count_bb` and `instr_count_cuda_graph` have low enough overhead.

### Specific Tool Questions

**mem_printf2 doesn't work - why?**
Known device function call issue. It's included as an educational example.

**What's the difference between warp-level and thread-level counting?**
- Warp-level (default): Counts 1 per warp (32 threads)
- Thread-level: Counts each thread separately (32x higher)
Set via `COUNT_WARP_LEVEL=0` for thread-level.

**How do I create my own tool?**
1. Copy existing tool directory (e.g., `instr_count`)
2. Modify host code and device code (inject_funcs.cu)
3. Update Makefile
4. Rebuild with `make`

## Troubleshooting

### Common Issues and Solutions

#### 1. "nvdisasm: command not found"

**Problem:** NVBit cannot find the `nvdisasm` tool.

**Solution:**
```bash
# Add CUDA bin to PATH
export PATH=/usr/local/cuda/bin:$PATH

# Verify it works
which nvdisasm
```

Add this to your `~/.bashrc` to make it permanent.

#### 2. "undefined reference to device functions" during linking

**Problem:** CUDA 12.8+ linking error when building tools.

**Solution:** Make sure you're using g++ for linking (not nvcc). Our Makefiles already handle this, but if you're creating a custom tool:
```makefile
# Use g++ for linking, NOT nvcc
g++ -shared -o mytool.so $(OBJECTS) -L$(CUDA_LIB) -lcuda -lcudart_static
```

#### 3. Tool loads but produces no output

**Possible Causes:** Check if nvdisasm is in PATH, verify the tool compiled successfully (check for `.so` file), or try with `TOOL_VERBOSE=1`:
```bash
TOOL_VERBOSE=1 LD_PRELOAD=./tools/instr_count/instr_count.so ./your_app
```

#### 4. "CUDA_ERROR_INVALID_VALUE" or segmentation fault

**Possible Causes:** GPU compute capability not supported (need SM 3.5+), driver version too old or too new (need <= 575.xx), or concurrent use with nvprof/nsight (NVBit cannot run with these tools).

**Solution:**
```bash
# Check GPU compute capability
nvidia-smi --query-gpu=compute_cap --format=csv

# Check driver version
nvidia-smi
```

#### 5. Build errors with "cannot find -lcuda"

**Solution:**
```bash
# Add CUDA library path
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

#### 6. Application crashes with instrumentation

**Debugging Steps:** Verify the app works without instrumentation (`./your_app`), test with the simplest tool (`LD_PRELOAD=./tools/instr_count/instr_count.so ./your_app`), or use selective instrumentation:
```bash
KERNEL_BEGIN=0 KERNEL_END=1 LD_PRELOAD=./tools/instr_count/instr_count.so ./your_app
```

### Getting More Help

Read tool-specific READMEs in `tools/TOOLNAME/README.md`, examine `core/nvbit.h` for API documentation, or report issues at the GitHub repository.

Before asking for help, try with `TOOL_VERBOSE=1` to get diagnostic output, verify your setup meets all requirements in [Prerequisites](#prerequisites), and test with provided test applications first.

## Contributing

We welcome contributions to improve the tutorial! If you find issues or have suggestions, open an issue describing the problem or enhancement, submit a pull request with your proposed changes, and follow the coding style of the existing examples.

## Further Resources

For more details on the NVBit APIs, see the comments in `core/nvbit.h`.

You may also find these resources helpful:
- [NVIDIA Developer Blog](https://developer.nvidia.com/blog)
- [CUDA Documentation](https://docs.nvidia.com/cuda/)
- [CUDA GPU Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html)

Happy learning!
