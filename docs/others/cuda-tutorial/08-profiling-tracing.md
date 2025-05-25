# CUDA GPU Profiling and Tracing

This document provides a comprehensive guide to profiling and tracing CUDA applications to identify performance bottlenecks and optimize GPU code execution.

You can find the code in <https://github.com/eunomia-bpf/basic-cuda-tutorial>

## Table of Contents

1. [Introduction to GPU Profiling](#introduction-to-gpu-profiling)
2. [Profiling Tools](#profiling-tools)
3. [Key Performance Metrics](#key-performance-metrics)
4. [Profiling Methodology](#profiling-methodology)
5. [Common Performance Bottlenecks](#common-performance-bottlenecks)
6. [Tracing Techniques](#tracing-techniques)
7. [Example Application](#example-application)
8. [Best Practices](#best-practices)
9. [Further Reading](#further-reading)

## Introduction to GPU Profiling

GPU profiling is the process of measuring and analyzing the performance characteristics of GPU applications. It helps developers:

- Identify performance bottlenecks
- Optimize resource utilization
- Understand execution patterns
- Validate optimization decisions
- Ensure scalability across different hardware

Effective profiling is essential for high-performance CUDA applications as the complex nature of GPU architecture makes intuitive optimization insufficient.

## Profiling Tools

### NVIDIA Nsight Systems

Nsight Systems is a system-wide performance analysis tool that provides insights into CPU and GPU execution:

- **System-level tracing**: CPU, GPU, memory, and I/O activities
- **Timeline visualization**: Shows kernel execution, memory transfers, and CPU activity
- **API trace**: Captures CUDA API calls and their durations
- **Low overhead**: Suitable for production code profiling

### NVIDIA Nsight Compute

Nsight Compute is an interactive kernel profiler for CUDA applications:

- **Detailed kernel metrics**: SM utilization, memory throughput, instruction mix
- **Guided analysis**: Provides optimization recommendations
- **Roofline analysis**: Shows performance relative to hardware limits
- **Kernel comparison**: Compare kernels across runs or hardware platforms

### NVIDIA Visual Profiler and nvprof

Legacy tools (deprecated but still useful for older CUDA versions):

- **nvprof**: Command-line profiler with low overhead
- **Visual Profiler**: GUI-based analysis tool
- **CUDA profiling APIs**: Allow programmatic access to profiling data

### Other Tools

- **Compute Sanitizer**: Memory access checking and race detection
- **CUPTI**: CUDA Profiling Tools Interface for custom profilers
- **PyTorch/TensorFlow Profilers**: Framework-specific profiling for deep learning

## Key Performance Metrics

### Execution Metrics

1. **SM Occupancy**: Ratio of active warps to maximum possible warps
   - Higher values generally enable better latency hiding
   - Target: >50% for most applications

2. **Warp Execution Efficiency**: Percentage of threads active during execution
   - Lower values indicate branch divergence
   - Target: >80% for compute-bound kernels

3. **Instruction Throughput**:
   - Instructions per clock (IPC)
   - Arithmetic intensity (operations per byte)
   - Mix of instruction types

### Memory Metrics

1. **Memory Throughput**:
   - Global memory read/write bandwidth
   - Shared memory bandwidth
   - L1/L2 cache hit rates
   - Target: As close to peak hardware bandwidth as possible

2. **Memory Access Patterns**:
   - Load/store efficiency
   - Global memory coalescing rate
   - Shared memory bank conflicts

3. **Data Transfer**:
   - Host-device transfer bandwidth
   - PCIe utilization
   - NVLink utilization (if available)

### Compute Metrics

1. **Compute Utilization**:
   - SM activity
   - Tensor/RT core utilization (if used)
   - Instruction mix (FP32, FP64, INT, etc.)

2. **Compute Efficiency**:
   - Achieved vs. theoretical FLOPS
   - Resource limitations (compute vs. memory bound)
   - Roofline model position

## Profiling Methodology

A structured approach to profiling CUDA applications:

### 1. Initial Assessment

- Start with high-level system profiling (Nsight Systems)
- Identify time distribution between CPU, GPU, and data transfers
- Look for obvious bottlenecks like excessive synchronization or transfers

### 2. Kernel Analysis

- Profile individual kernels (Nsight Compute)
- Identify the most time-consuming kernels
- Collect key metrics for these kernels

### 3. Bottleneck Identification

- Determine if kernels are compute-bound or memory-bound
- Use the roofline model to understand performance limiters
- Check for specific inefficiencies (divergence, non-coalesced access)

### 4. Guided Optimization

- Address the most significant bottlenecks first
- Make one change at a time and measure impact
- Compare before/after profiles to validate improvements

### 5. Iterative Refinement

- Repeat the process for the next bottleneck
- Re-profile the entire application periodically
- Continue until performance goals are met

## Common Performance Bottlenecks

### Memory-Related Issues

1. **Non-coalesced Memory Access**:
   - Symptoms: Low global memory load/store efficiency
   - Solution: Reorganize data layout or access patterns

2. **Shared Memory Bank Conflicts**:
   - Symptoms: Low shared memory bandwidth
   - Solution: Adjust padding or access patterns

3. **Excessive Global Memory Access**:
   - Symptoms: High memory dependency
   - Solution: Increase data reuse through shared memory or registers

### Execution-Related Issues

1. **Warp Divergence**:
   - Symptoms: Low warp execution efficiency
   - Solution: Reorganize algorithms to minimize divergent paths

2. **Low Occupancy**:
   - Symptoms: SM occupancy below 50%
   - Solution: Reduce register/shared memory usage or adjust block size

3. **Kernel Launch Overhead**:
   - Symptoms: Many small, short-duration kernels
   - Solution: Kernel fusion or persistent kernels

### System-Level Issues

1. **Excessive Host-Device Transfers**:
   - Symptoms: High PCIe utilization, many transfer operations
   - Solution: Batch transfers, use pinned memory, or unified memory

2. **CPU-GPU Synchronization**:
   - Symptoms: GPU idle periods between kernels
   - Solution: Use CUDA streams, asynchronous operations

3. **Underutilized GPU Resources**:
   - Symptoms: Low overall GPU utilization
   - Solution: Concurrent kernels, streams, or increase problem size

## Tracing Techniques

Tracing provides a timeline view of application execution:

### CUDA Events

```cuda
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
myKernel<<<grid, block>>>(data);
cudaEventRecord(stop);
cudaEventSynchronize(stop);

float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
printf("Kernel execution time: %f ms\n", milliseconds);
```

### NVTX Markers and Ranges

NVIDIA Tools Extension (NVTX) allows custom annotations:

```cuda
#include <nvtx3/nvToolsExt.h>

// Mark an instantaneous event
nvtxMark("Interesting point");

// Begin a range
nvtxRangePushA("Data preparation");
// ... code ...
nvtxRangePop(); // End the range

// Colored range for better visibility
nvtxEventAttributes_t eventAttrib = {0};
eventAttrib.version = NVTX_VERSION;
eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
eventAttrib.colorType = NVTX_COLOR_ARGB;
eventAttrib.color = 0xFF00FF00; // Green
eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
eventAttrib.message.ascii = "Kernel Execution";
nvtxRangePushEx(&eventAttrib);
myKernel<<<grid, block>>>(data);
nvtxRangePop();
```

### Programmatic Profiling with CUPTI

CUDA Profiling Tools Interface (CUPTI) enables programmatic access to profiling data:

```c
// Simplified CUPTI usage example
#include <cupti.h>

void CUPTIAPI callbackHandler(void *userdata, CUpti_CallbackDomain domain,
                             CUpti_CallbackId cbid, const void *cbInfo) {
    // Handle callback
}

// Initialize CUPTI and register callbacks
CUpti_SubscriberHandle subscriber;
cuptiSubscribe(&subscriber, callbackHandler, NULL);
cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
                   CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020);
```

## Example Application

The accompanying `basic08.cu` demonstrates:

1. **Basic kernel timing**: Using CUDA events
2. **NVTX annotations**: Adding markers and ranges
3. **Memory transfer profiling**: Analyzing host-device transfers
4. **Kernel optimization**: Comparing different implementation strategies
5. **Interpreting profiling data**: Making optimization decisions

### Key Code Sections

Basic kernel timing:
```cuda
__global__ void computeKernel(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        // Perform computation
        float result = x * x + x + 1.0f;
        output[idx] = result;
    }
}

void timeKernel() {
    // Allocate memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, SIZE * sizeof(float));
    cudaMalloc(&d_output, SIZE * sizeof(float));
    
    // Initialize data
    float *h_input = new float[SIZE];
    for (int i = 0; i < SIZE; i++) h_input[i] = i;
    
    cudaMemcpy(d_input, h_input, SIZE * sizeof(float), cudaMemcpyHostToDevice);
    
    // Timing with events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warm-up run
    computeKernel<<<(SIZE + 255) / 256, 256>>>(d_input, d_output, SIZE);
    
    // Timed run
    cudaEventRecord(start);
    computeKernel<<<(SIZE + 255) / 256, 256>>>(d_input, d_output, SIZE);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time: %f ms\n", milliseconds);
    
    // Cleanup
    delete[] h_input;
    cudaFree(d_input);
    cudaFree(d_output);
}
```

## Best Practices

### Profiling Workflow

1. **Start with high-level profiling** before diving into details
2. **Establish baselines** for important kernels
3. **Profile regularly** during development, not just at the end
4. **Automate profiling** where possible for regression testing
5. **Compare across hardware** to ensure portability

### Tool Selection

1. **Nsight Systems** for system-level analysis and timeline
2. **Nsight Compute** for detailed kernel metrics
3. **NVTX markers** for custom annotations
4. **CUDA events** for lightweight timing measurements

### Optimization Approach

1. **Focus on hotspots**: Address the most time-consuming operations first
2. **Use roofline analysis**: Understand theoretical limits
3. **Balance efforts**: Don't over-optimize less critical sections
4. **Consider trade-offs**: Sometimes readability > minor performance gains
5. **Document insights**: Record profiling discoveries for future reference

## Further Reading

- [NVIDIA Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/)
- [NVIDIA Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [CUDA Profiling Tools Interface (CUPTI)](https://docs.nvidia.com/cuda/cupti/)
- [Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)
- [NVTX Documentation](https://nvidia.github.io/NVTX/doxygen/index.html)
- [CUDA C++ Best Practices Guide: Performance Metrics](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#performance-metrics)
- [Parallel Thread Execution (PTX) Documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) 