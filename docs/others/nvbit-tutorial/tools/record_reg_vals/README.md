# NVBit Tutorial: Register Value Recording

> Github repo: <https://github.com/eunomia-bpf/nvbit-tutorial>

**TL;DR:** Captures register values during kernel execution. Perfect for debugging numerical errors (NaN/Inf), algorithm verification, and understanding data flow. Very high overhead.

**Quick Start:**
```bash
# Instrument specific instruction range
START_GRID_NUM=0 END_GRID_NUM=1 INSTR_BEGIN=10 INSTR_END=15 \
  env CUDA_INJECTION64_PATH=./tools/record_reg_vals/record_reg_vals.so ./test-apps/vectoradd/vectoradd
```

**Use Cases:**
- Finding where NaN/Infinity appears
- Verifying algorithm correctness
- Debugging divergence issues
- Understanding register data flow

## Overview

Understanding the dynamic values of registers during kernel execution is invaluable for debugging and performance analysis. The `record_reg_vals` tool instruments selected CUDA instructions to capture register values, collects register values from all threads in a warp, transfers this data efficiently from the GPU to the CPU, and prints or analyzes the register state information.

This capability allows developers to debug complex kernel issues by examining register values, verify data flow through computation stages, detect value patterns and anomalies, and understand divergence between threads in a warp.

## Code Structure

- `record_reg_vals.cu` – Host-side code that identifies and selects instructions to instrument, determines which registers to track, and processes and displays register values
- `inject_funcs.cu` – Device-side code that captures register values from all threads in a warp, packages data with execution context, and sends the data through a channel to the host
- `common.h` – Shared structure definition for the `reg_info_t` structure that holds the captured register data

## How It Works: Register Access with NVBit

NVBit provides special mechanisms to access register values during execution, which is not possible with standard CUDA programming. The core insight is that we can identify registers of interest (function parameters, local variables, etc.), insert instrumentation to capture their values at specific points, use warp-level operations to collect values from all threads, and transfer this data to the host for analysis.

## How It Works: Host Side (record_reg_vals.cu)

Let's examine the key elements of the host-side implementation:

### 1. Instruction Selection and Register Analysis

```cpp
void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
    // ... standard NVBit setup ...
    
    /* Iterate on instructions */
    for (auto instr : instrs) {
        // ... filter instructions based on criteria ...
        
        if (selected_for_instrumentation(instr)) {
            /* Create a list of registers to record */
            std::vector<int> reg_ids;
            
            /* Identify registers of interest */
            for (int i = 0; i < instr->getNumOperands(); i++) {
                const InstrType::operand_t* op = instr->getOperand(i);
                if (op->type == InstrType::OperandType::REG) {
                    reg_ids.push_back(op->u.reg.num);
                }
            }
            
            /* Insert instrumentation call */
            nvbit_insert_call(instr, "record_reg_val", IPOINT_BEFORE);
            
            /* Add predicate as argument */
            nvbit_add_call_arg_guard_pred_val(instr);
            
            /* Add opcode ID */
            nvbit_add_call_arg_const_val32(instr, opcode_id);
            
            /* Add channel pointer */
            nvbit_add_call_arg_const_val64(instr, (uint64_t)ctx_state->channel_dev);
            
            /* Add number of registers to record */
            nvbit_add_call_arg_const_val32(instr, reg_ids.size());
            
            /* Add each register as an argument */
            for (int reg_id : reg_ids) {
                nvbit_add_call_arg_reg_val(instr, reg_id);
            }
        }
    }
}
```

The key aspects of instruction selection and register analysis: we select instructions based on criteria (specific opcodes, instruction types, etc.), identify registers of interest in the instruction's operands, and insert a call to `record_reg_val` with the instruction's predicate for conditional execution, an opcode ID to identify the instruction type, a pointer to the communication channel, the number of registers to record, and the actual register values via `nvbit_add_call_arg_reg_val`.

### 2. Receiver Thread for Processing Register Data

```cpp
void* recv_thread_fun(void* args) {
    CUcontext ctx = (CUcontext)args;
    CTXstate* ctx_state = ctx_state_map[ctx];
    ChannelHost* ch_host = &ctx_state->channel_host;
    
    char* recv_buffer = (char*)malloc(CHANNEL_SIZE);
    
    while (!ctx_state->recv_thread_done) {
        /* Receive data from the channel */
        uint32_t num_recv_bytes = ch_host->recv(recv_buffer, CHANNEL_SIZE);
        
        if (num_recv_bytes > 0) {
            uint32_t num_processed_bytes = 0;
            while (num_processed_bytes < num_recv_bytes) {
                /* Cast received data to our structure */
                reg_info_t* ri = (reg_info_t*)&recv_buffer[num_processed_bytes];
                
                /* Print header with context information */
                printf("CTA %d,%d,%d - Warp %d - Opcode %s\n",
                       ri->cta_id_x, ri->cta_id_y, ri->cta_id_z,
                       ri->warp_id, id_to_opcode_map[ri->opcode_id]);
                
                /* Print register values for all threads */
                for (int i = 0; i < ri->num_regs; i++) {
                    printf("  Register %d: ", i);
                    for (int t = 0; t < 32; t++) {
                        printf("0x%08x ", ri->reg_vals[t][i]);
                    }
                    printf("\n");
                }
                
                num_processed_bytes += sizeof(reg_info_t);
            }
        }
    }
    
    free(recv_buffer);
    return NULL;
}
```

The receiver thread continuously polls the channel for new register data, processes received data as `reg_info_t` structures, prints context information (CTA coordinates, warp ID, opcode), formats and prints the register values for all 32 threads in the warp, and continues until signaled to stop.

## How It Works: Device Side (inject_funcs.cu)

The device-side code captures register values and sends them to the host:

```cpp
extern "C" __device__ __noinline__ void record_reg_val(int pred, int opcode_id,
                                                       uint64_t pchannel_dev,
                                                       int32_t num_regs...) {
    /* Skip if thread is predicated off */
    if (!pred) {
        return;
    }
    
    /* Get active threads in the warp */
    int active_mask = __ballot_sync(__activemask(), 1);
    const int laneid = get_laneid();
    const int first_laneid = __ffs(active_mask) - 1;
    
    /* Create register info record */
    reg_info_t ri;
    
    /* Fill in execution context */
    int4 cta = get_ctaid();
    ri.cta_id_x = cta.x;
    ri.cta_id_y = cta.y;
    ri.cta_id_z = cta.z;
    ri.warp_id = get_warpid();
    ri.opcode_id = opcode_id;
    ri.num_regs = num_regs;
    
    /* Process variable arguments (register values) */
    if (num_regs) {
        va_list vl;
        va_start(vl, num_regs);
        
        for (int i = 0; i < num_regs; i++) {
            uint32_t val = va_arg(vl, uint32_t);
            
            /* Collect register values from all threads using warp vote */
            for (int tid = 0; tid < 32; tid++) {
                ri.reg_vals[tid][i] = __shfl_sync(active_mask, val, tid);
            }
        }
        va_end(vl);
    }
    
    /* Only the first active thread pushes data to the channel */
    if (first_laneid == laneid) {
        ChannelDev* channel_dev = (ChannelDev*)pchannel_dev;
        channel_dev->push(&ri, sizeof(reg_info_t));
    }
}
```

Key aspects of this device function: we skip execution if the thread's predicate is false (predicate handling), use C's variadic function capability to accept a variable number of register values (variadic arguments), use `__shfl_sync` to gather each register value from all threads in the warp (warp-level collection), record CTA coordinates, warp ID, and opcode information (context capture), and ensure only one thread per warp pushes data to avoid duplicates (channel communication).

## The Register Information Structure (common.h)

The shared structure for transferring register data:

```cpp
typedef struct {
    int32_t cta_id_x;
    int32_t cta_id_y;
    int32_t cta_id_z;
    int32_t warp_id;
    int32_t opcode_id;
    int32_t num_regs;
    /* 32 lanes, each thread can store up to 8 register values */
    uint32_t reg_vals[32][8];
} reg_info_t;
```

This structure captures which thread block (CTA coordinates), which warp within the block, what instruction type (opcode_id), how many registers we're tracking, and the register values for all 32 threads in the warp (up to 8 registers per instruction).

## Building the Tool

The build process follows the standard pattern for NVBit tools:

1. Compile the host code:
   ```
   $(NVCC) -dc -c -std=c++11 $(INCLUDES) ... record_reg_vals.cu -o record_reg_vals.o
   ```

2. Compile the device function:
   ```
   $(NVCC) $(INCLUDES) $(MAXRREGCOUNT_FLAG) -Xptxas -astoolspatch --keep-device-functions ... inject_funcs.cu -o inject_funcs.o
   ```

3. Link into a shared library:
   ```
   $(NVCC) -arch=$(ARCH) $(NVCC_OPT) $(OBJECTS) $(LIBS) $(NVCC_PATH) -lcuda -lcudart_static -shared -o record_reg_vals.so
   ```

## Running the Tool

Launch your CUDA application with the tool preloaded:

```bash
env CUDA_INJECTION64_PATH=./tools/record_reg_vals/record_reg_vals.so ./test-apps/vectoradd/vectoradd
```

### Environment Variables

The tool supports these environment variables:

- `INSTR_BEGIN`/`INSTR_END`: Instruction range to instrument
- `TOOL_VERBOSE`: Enable verbose output

## Sample Output

Here's an example of what the output might look like:

```
CTA 0,0,0 - Warp 0 - Opcode IMAD
  Register 0: 0x00000000 0x00000001 0x00000002 0x00000003 0x00000004 0x00000005 0x00000006 0x00000007 0x00000008 0x00000009 0x0000000a 0x0000000b 0x0000000c 0x0000000d 0x0000000e 0x0000000f 0x00000010 0x00000011 0x00000012 0x00000013 0x00000014 0x00000015 0x00000016 0x00000017 0x00000018 0x00000019 0x0000001a 0x0000001b 0x0000001c 0x0000001d 0x0000001e 0x0000001f
  Register 1: 0x3f800000 0x3f800000 0x3f800000 0x3f800000 0x3f800000 0x3f800000 0x3f800000 0x3f800000 0x3f800000 0x3f800000 0x3f800000 0x3f800000 0x3f800000 0x3f800000 0x3f800000 0x3f800000 0x3f800000 0x3f800000 0x3f800000 0x3f800000 0x3f800000 0x3f800000 0x3f800000 0x3f800000 0x3f800000 0x3f800000 0x3f800000 0x3f800000 0x3f800000 0x3f800000 0x3f800000 0x3f800000
  Register 2: 0x3f800000 0x40000000 0x40400000 0x40800000 0x40a00000 0x40c00000 0x40e00000 0x41000000 0x41100000 0x41200000 0x41300000 0x41400000 0x41500000 0x41600000 0x41700000 0x41800000 0x41880000 0x41900000 0x41980000 0x41a00000 0x41a80000 0x41b00000 0x41b80000 0x41c00000 0x41c80000 0x41d00000 0x41d80000 0x41e00000 0x41e80000 0x41f00000 0x41f80000 0x42000000
```

Each output block shows:
- The CTA (thread block) coordinates and warp ID
- The opcode of the instrumented instruction
- For each tracked register, the values for all 32 threads in the warp

## Analyzing Register Values

The register value data can provide insights into many aspects of kernel execution:

### 1. Thread Divergence

Look for patterns where register values differ significantly between threads in the same warp. This can indicate divergent execution paths that reduce parallelism.

### 2. Data Patterns

Examine register values to identify patterns such as:
- Linear sequences (thread IDs, array indices)
- Constant values (uniform parameters)
- Mathematical progressions (computed values)

### 3. Value Correctness

Check if register values match expected results at different points in the computation. This is invaluable for debugging complex algorithms.

### 4. Register Utilization

Monitor which registers are used and how their values change over time to understand register pressure and utilization.

## Performance Considerations

Register value recording has significant overhead due to the additional function calls for instrumented instructions, the collection and communication of register values, and the processing and printing of the data. For large applications, consider limiting instrumentation to specific instructions of interest, filtering for particular register patterns, or sampling rather than capturing all register values.

## Extending the Tool

You can extend this tool in several ways: selective register tracking (instrument only instructions that use specific registers), value filtering (only report register values that meet certain criteria), statistical analysis (compute statistics on register values rather than printing raw data), visual representation (output register values in a format suitable for visualization), and divergence detection (automatically identify and report thread divergence).

## Advanced Application: Debugging Numerical Issues

A powerful application of register value recording is debugging numerical issues in scientific computing: NaN/Infinity detection (track when floating-point values become NaN or infinity), precision loss (identify where precision is lost in complex calculations), determinism checking (verify that results are consistent across multiple runs), and boundary condition analysis (detect when values approach problematic boundaries).

## Next Steps

After mastering register value recording, consider creating custom analysis tools for specific patterns or issues, combining with instruction mix analysis from `opcode_hist`, correlating register values with memory access patterns from `mem_trace`, or building a visualization tool to better understand register evolution. Register value recording provides a powerful window into the internal state of GPU kernels, enabling debugging and analysis techniques that would otherwise be impossible.
