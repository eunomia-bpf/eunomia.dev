# NVBit Tutorial: Instruction Counting

> Github repo: <https://github.com/eunomia-bpf/nvbit-tutorial>

**TL;DR:** This tool counts how many GPU instructions your CUDA kernels execute. Perfect for learning NVBit basics.

**Quick Start:**
```bash
LD_PRELOAD=./tools/instr_count/instr_count.so ./test-apps/vectoradd/vectoradd
# Output: kernel 0 - vecAdd(...) - kernel instructions 50077
```

## Table of Contents

- [Overview](#overview)
- [What are Predicates?](#what-are-predicates)
- [Architecture Diagram](#architecture-diagram)
- [Code Structure](#code-structure)
- [How It Works: Host Side](#how-it-works-host-side)
- [How It Works: Device Side](#how-it-works-device-side)
- [Building the Tool](#building-the-tool)
- [Running the Tool](#running-the-tool)
- [Sample Output](#sample-output)
- [Common Issues](#common-issues)
- [Next Steps](#next-steps)

## Overview

The instruction counting tool demonstrates the fundamental workflow of NVBit instrumentation: intercept CUDA kernel launches, analyze and instrument kernel code before execution, collect data during kernel execution, and process and report results after kernel completion. This tutorial walks through the implementation in detail to help you understand how to build similar tools.

## What are Predicates?

Before diving into the code, understand a key GPU concept: **predicates**.

GPU instructions can be conditionally executed using predicates:
```
@P0 ADD R1, R2, R3    // Only executes if predicate P0 is true
@!P1 MOV R4, R5       // Only executes if predicate P1 is false
```

Think of predicates like if-statements at the instruction level. When a predicate is false, the instruction is "predicated off" - it doesn't execute, but the thread still processes it.

**Why this matters for counting:**
- By default, we count all instructions, even predicated-off ones
- Set `EXCLUDE_PRED_OFF=1` to count only instructions that actually execute
- The tool checks each instruction's predicate to determine if it should count

## Architecture Diagram

Here's how the instruction counter works:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         HOST (CPU)                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. Application launches kernel                                      │
│     │                                                                │
│     v                                                                │
│  2. NVBit intercepts launch (nvbit_at_cuda_event)                   │
│     │                                                                │
│     v                                                                │
│  3. Instrument kernel function (instrument_function_if_needed)       │
│     - Iterate through instructions                                   │
│     - Insert call to count_instrs before each instruction            │
│     - Pass predicate, counter pointer as arguments                   │
│     │                                                                │
│     v                                                                │
│  4. Enable instrumentation, reset counter                            │
│     │                                                                │
│     v                                                                │
│  5. Kernel executes on GPU  ─────────────────────┐                  │
│                                                   │                  │
└───────────────────────────────────────────────────┼──────────────────┘
                                                    │
┌───────────────────────────────────────────────────┼──────────────────┐
│                        DEVICE (GPU)               │                  │
├───────────────────────────────────────────────────┼──────────────────┤
│                                                   v                  │
│  For each instrumented instruction:                                  │
│     Original:  ADD R1, R2, R3                                       │
│                                                                      │
│     Becomes:   count_instrs(predicate, ...)  ◄── Injected           │
│                ADD R1, R2, R3                ◄── Original            │
│                                                                      │
│  count_instrs():                                                     │
│     - Check if thread's predicate is true                           │
│     - Use warp vote to collect all thread states                    │
│     - First thread in warp atomically updates counter               │
│     │                                                                │
│     v                                                                │
│  __managed__ counter updated (accessible from host)                  │
│                                                                      │
└───────────────────────────────────────────────────┬──────────────────┘
                                                    │
┌───────────────────────────────────────────────────┼──────────────────┐
│                         HOST (CPU)                │                  │
├───────────────────────────────────────────────────┼──────────────────┤
│                                                   v                  │
│  6. Kernel completes                                                 │
│     │                                                                │
│     v                                                                │
│  7. cudaDeviceSynchronize() - wait for GPU                          │
│     │                                                                │
│     v                                                                │
│  8. Read counter from managed memory                                 │
│     │                                                                │
│     v                                                                │
│  9. Print results: "kernel instructions 50077"                      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Key Points:** Host code runs on CPU and instruments the kernel before it executes. Device code (`count_instrs`) runs on GPU, called before each instruction. Managed memory (`__managed__ counter`) is shared between CPU and GPU. Warp-level operations ensure efficient counting (one atomic op per warp).

## Code Structure

The tool consists of two main source files:

- `instr_count.cu` – Host-side code that handles intercepting CUDA kernel launches, instrumenting the kernel code, and printing results after execution
- `inject_funcs.cu` – Device-side code that contains the `count_instrs` function (executes on the GPU) and logic to update the instruction counter

## How It Works: Host Side (instr_count.cu)

Let's examine the key components of the host-side code:

### 1. Global Variables

```cpp
/* kernel id counter, maintained in system memory */
uint32_t kernel_id = 0;

/* total instruction counter, maintained in system memory */
uint64_t tot_app_instrs = 0;

/* kernel instruction counter, updated by the GPU */
__managed__ uint64_t counter = 0;
```

The `counter` variable is declared as `__managed__`, meaning it's automatically managed by CUDA and accessible from both host and device code. This allows the GPU threads to update it atomically.

### 2. Initialization (nvbit_at_init)

```cpp
void nvbit_at_init() {
    /* force managed memory to be allocated on GPU */
    setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
    
    /* read environment variables for configuration */
    GET_VAR_INT(instr_begin_interval, "INSTR_BEGIN", 0, "...");
    GET_VAR_INT(instr_end_interval, "INSTR_END", UINT32_MAX, "...");
    // ... more environment variables
}
```

This function runs when the tool is loaded. It sets up configuration options from environment variables that control which instructions and kernels to instrument.

### 3. Instrumentation Function

```cpp
void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
    /* Get related functions (device functions called by the kernel) */
    std::vector<CUfunction> related_functions = nvbit_get_related_functions(ctx, func);
    
    /* add kernel itself to the related function vector */
    related_functions.push_back(func);
    
    /* iterate on functions */
    for (auto f : related_functions) {
        /* skip if already instrumented */
        if (!already_instrumented.insert(f).second) {
            continue;
        }
        
        /* Get instructions for this function */
        const std::vector<Instr *> &instrs = nvbit_get_instrs(ctx, f);
        
        /* Iterate through instructions */
        for (auto i : instrs) {
            /* Check if in target range */
            if (i->getIdx() >= instr_begin_interval && i->getIdx() < instr_end_interval) {
                /* Insert call to our device function */
                nvbit_insert_call(i, "count_instrs", IPOINT_BEFORE);
                
                /* Add arguments to the call */
                nvbit_add_call_arg_guard_pred_val(i);  // predicate value
                nvbit_add_call_arg_const_val32(i, count_warp_level);  // count mode
                nvbit_add_call_arg_const_val64(i, (uint64_t)&counter);  // counter pointer
            }
        }
    }
}
```

This is where the magic happens. For each instruction in the specified range, a call to our `count_instrs` device function is inserted before the instruction. Arguments are added to the call: the instruction's predicate value (to handle predicated instructions), whether to count at warp or thread level, and a pointer to our counter variable.

### 4. CUDA Event Callback

```cpp
void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char *name, void *params, CUresult *pStatus) {
    /* Handle kernel launches */
    if (cbid == API_CUDA_cuLaunchKernel || /* other launch APIs */) {
        /* Get the kernel function */
        CUfunction func = /* extracted from params */;
        
        if (!is_exit) {
            /* Before kernel launch: */
            pthread_mutex_lock(&mutex);
            instrument_function_if_needed(ctx, func);
            
            /* Enable/disable instrumentation based on kernel index */
            if (active_region) {
                nvbit_enable_instrumented(ctx, func, true);
            } else {
                nvbit_enable_instrumented(ctx, func, false);
            }
            
            counter = 0;  // Reset counter
        } else {
            /* After kernel completion: */
            CUDA_SAFECALL(cudaDeviceSynchronize());
            
            /* Print results */
            printf("kernel %d - %s - #thread-blocks %d, kernel instructions %ld, total instructions %ld\n",
                  kernel_id++, nvbit_get_func_name(ctx, func, mangled), num_ctas,
                  counter, tot_app_instrs);
                  
            tot_app_instrs += counter;
            pthread_mutex_unlock(&mutex);
        }
    }
}
```

This callback is triggered for every CUDA API call. We use it to intercept kernel launches, instrument the kernel before it runs, enable or disable our instrumentation, reset the counter before execution, and print results after the kernel completes.

## How It Works: Device Side (inject_funcs.cu)

The device function is concise but powerful:

```cpp
extern "C" __device__ __noinline__ void count_instrs(int predicate,
                                                    int count_warp_level,
                                                    uint64_t pcounter) {
    /* Calculate active threads in the warp */
    const int active_mask = __ballot_sync(__activemask(), 1);
    
    /* Compute which threads have true predicates */
    const int predicate_mask = __ballot_sync(__activemask(), predicate);
    
    /* Get the current thread's lane ID */
    const int laneid = get_laneid();
    
    /* Find the first active lane */
    const int first_laneid = __ffs(active_mask) - 1;
    
    /* Count active threads with true predicates */
    const int num_threads = __popc(predicate_mask);
    
    /* Only the first active thread updates the counter */
    if (first_laneid == laneid) {
        if (count_warp_level) {
            /* Count once per warp if any thread is active */
            if (num_threads > 0) {
                atomicAdd((unsigned long long*)pcounter, 1);
            }
        } else {
            /* Count once per active thread */
            atomicAdd((unsigned long long*)pcounter, num_threads);
        }
    }
}
```

We use warp-level voting functions (`__ballot_sync`, `__activemask()`) to determine which threads are active and have true predicates. Only one thread per warp performs the atomic update to avoid contention. Two counting modes are supported: warp-level counting (adds 1 per executed warp instruction) and thread-level counting (adds the number of active threads).

## Building the Tool

The Makefile includes all necessary steps to compile the tool:

1. Compile the host code with nvcc:
   ```
   $(NVCC) -dc -c -std=c++11 $(INCLUDES) ... instr_count.cu -o instr_count.o
   ```

2. Compile the device function with special flags to preserve it for instrumentation:
   ```
   $(NVCC) $(INCLUDES) $(MAXRREGCOUNT_FLAG) -Xptxas -astoolspatch --keep-device-functions ... inject_funcs.cu -o inject_funcs.o
   ```

3. Link everything into a shared library:
   ```
   $(NVCC) -arch=$(ARCH) $(NVCC_OPT) $(OBJECTS) $(LIBS) $(NVCC_PATH) -lcuda -lcudart_static -shared -o instr_count.so
   ```

The `--keep-device-functions` flag is crucial as it prevents optimization from removing our device function which needs to be called dynamically.

## Running the Tool

Use LD_PRELOAD to inject the tool into any CUDA application:

```bash
LD_PRELOAD=./tools/instr_count/instr_count.so ./your_cuda_application
```

### Environment Variables

Several environment variables control the tool's behavior:

- `INSTR_BEGIN`/`INSTR_END`: Specify which instruction range to instrument
- `START_GRID_NUM`/`END_GRID_NUM`: Limit which kernel launches to instrument
- `COUNT_WARP_LEVEL`: Set to 0 to count individual thread instructions instead of warp instructions
- `EXCLUDE_PRED_OFF`: Set to 1 to exclude predicated-off instructions
- `TOOL_VERBOSE`: Enable detailed output about instrumentation

Examples:

```bash
# Count only the first 100 instructions in each function
INSTR_END=100 LD_PRELOAD=./tools/instr_count/instr_count.so ./vectoradd

# Count only the second kernel launch
START_GRID_NUM=1 END_GRID_NUM=2 LD_PRELOAD=./tools/instr_count/instr_count.so ./vectoradd

# Count thread-level instructions instead of warp-level
COUNT_WARP_LEVEL=0 LD_PRELOAD=./tools/instr_count/instr_count.so ./vectoradd
```

## Sample Output

```
------------- NVBit (NVidia Binary Instrumentation Tool) Loaded --------------
[Environment variables and settings shown here]
----------------------------------------------------------------------------------------------------
kernel 0 - vecAdd(double*, double*, double*, int) - #thread-blocks 98, kernel instructions 50077, total instructions 50077
Final sum = 100000.000000; sum/n = 1.000000 (should be ~1)
```

The output shows:
- Kernel ID and name
- Number of thread blocks launched
- Total dynamic instructions executed in this kernel
- Cumulative instruction count across all kernels

## Creating Your Own Counting Tool

To create a similar tool for your own metrics, modify `inject_funcs.cu` to count a different metric, update the instrumentation in `instr_count.cu` to target specific instructions, and change how results are reported in the CUDA event callback. For example, you could count only memory operations, track instruction latency, or gather statistics on control flow.

## Common Issues

### High Overhead (20-100x slowdown)

**Expected behavior.** Instrumenting every instruction is expensive. Solutions: use `instr_count_bb` for 5-10x better performance, instrument selectively (`INSTR_END=100` for first 100 instructions only), or limit kernels (`KERNEL_BEGIN=0 KERNEL_END=1` for first kernel only).

### Counter Shows Zero

Check if `nvdisasm` is in PATH (`which nvdisasm`), if instrumentation is enabled (try `ACTIVE_FROM_START=1`), and run with `TOOL_VERBOSE=1` to see what's happening.

### Unexpectedly Low Count

Likely counting at warp-level (default). For thread-level count:
```bash
COUNT_WARP_LEVEL=0 LD_PRELOAD=./tools/instr_count/instr_count.so ./app
```

Thread count should be ~32x higher (32 threads per warp).

## Next Steps

After mastering instruction counting, explore the more advanced tools: `opcode_hist` (generate histograms of executed instructions), `instr_count_bb` (more efficient basic-block level instrumentation), and `mem_trace` (track memory access patterns). With NVBit, you can develop sophisticated GPU analysis tools without modifying application source code.
