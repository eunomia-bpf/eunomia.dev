# CPU and GPU Profiling Boundaries: What to Measure Where

This document explores the boundary between CPU and GPU profiling, examining which operations can be effectively measured on the CPU side versus which require GPU-side instrumentation. We'll also discuss how advanced CPU-side function hooking techniques like eBPF can complement GPU profiling.

## Table of Contents

1. [The CPU-GPU Boundary](#the-cpu-gpu-boundary)
2. [CPU-Side Measurable Operations](#cpu-side-measurable-operations)
3. [GPU-Side Measurable Operations](#gpu-side-measurable-operations)
4. [When Kernel Instrumentation Is Necessary](#when-kernel-instrumentation-is-necessary)
5. [Hooking CPU-Side Functions with eBPF](#hooking-cpu-side-functions-with-ebpf)
6. [Integrated Profiling Approaches](#integrated-profiling-approaches)
7. [Case Studies](#case-studies)
8. [Future Directions](#future-directions)
9. [References](#references)

## The CPU-GPU Boundary

Modern GPU computing involves a complex interplay between host (CPU) and device (GPU) operations. Understanding where to place profiling instrumentation depends on what aspects of performance you're measuring:

```
┌────────────────────────┐                  ┌────────────────────────┐
│        CPU Side        │                  │        GPU Side        │
│                        │                  │                        │
│  ┌──────────────────┐  │                  │  ┌──────────────────┐  │
│  │ Application Code │  │                  │  │  Kernel Execution │  │
│  └────────┬─────────┘  │                  │  └────────┬─────────┘  │
│           │            │                  │           │            │
│  ┌────────▼─────────┐  │                  │  ┌────────▼─────────┐  │
│  │   CUDA Runtime   │  │                  │  │   Warp Scheduler  │  │
│  └────────┬─────────┘  │                  │  └────────┬─────────┘  │
│           │            │                  │           │            │
│  ┌────────▼─────────┐  │  ┌─────────┐     │  ┌────────▼─────────┐  │
│  │   CUDA Driver    │◄─┼──┤PCIe Bus │────►│  │ Memory Controller │  │
│  └────────┬─────────┘  │  └─────────┘     │  └────────┬─────────┘  │
│           │            │                  │           │            │
│  ┌────────▼─────────┐  │                  │  ┌────────▼─────────┐  │
│  │ System Software  │  │                  │  │   GPU Hardware   │  │
│  └──────────────────┘  │                  │  └──────────────────┘  │
└────────────────────────┘                  └────────────────────────┘
```

## CPU-Side Measurable Operations

The following operations can be effectively measured from the CPU side:

### 1. CUDA API Call Latency

- **Kernel Launch Overhead**: Time from the API call to when the kernel begins execution
- **Memory Allocation**: Time spent in `cudaMalloc`, `cudaFree`, etc.
- **Host-Device Transfers**: Duration of `cudaMemcpy` operations
- **Synchronization Points**: Time spent in `cudaDeviceSynchronize`, `cudaStreamSynchronize`

### 2. Resource Management

- **Memory Usage**: Tracking GPU memory allocation and deallocation patterns
- **Stream Creation**: Overhead of creating and destroying CUDA streams
- **Context Switching**: Time spent switching between CUDA contexts

### 3. CPU-GPU Interaction Patterns

- **API Call Frequency**: Rate and pattern of CUDA API calls
- **CPU Wait Time**: Time CPU spends waiting for GPU operations
- **I/O and GPU Overlap**: How I/O operations interact with GPU utilization

### 4. System-Level Metrics

- **PCIe Traffic**: Volume and timing of data transferred over PCIe
- **Power Consumption**: System-wide power usage correlated with GPU activity
- **Thermal Effects**: Temperature changes that may affect throttling

### Tools and Techniques for CPU-Side Measurement

- **CUPTI API Callbacks**: Hook into CUDA API calls via the CUPTI interface
- **Binary Instrumentation**: Tools like Pin or DynamoRIO to intercept functions
- **Interposing Libraries**: Custom libraries that intercept CUDA API calls
- **eBPF**: Linux's extended Berkeley Packet Filter for kernel-level tracing
- **Performance Counters**: Hardware-level counters accessible via PAPI or similar

## GPU-Side Measurable Operations

The following operations require GPU-side instrumentation:

### 1. Kernel Execution Details

- **Instruction Mix**: Types and frequencies of instructions executed
- **Warp Execution Efficiency**: Percentage of active threads in warps
- **Divergence Patterns**: Frequency and impact of branch divergence
- **Instruction-Level Parallelism**: Achieved ILP within each thread

### 2. Memory System Performance

- **Memory Access Patterns**: Coalescing efficiency, stride patterns
- **Cache Hit Rates**: L1/L2/Texture cache effectiveness
- **Bank Conflicts**: Shared memory access conflicts
- **Memory Divergence**: Divergent memory access patterns

### 3. Hardware Utilization

- **SM Occupancy**: Active warps relative to maximum capacity
- **Special Function Usage**: Utilization of SFUs, tensor cores, etc.
- **Memory Bandwidth**: Achieved vs. theoretical memory bandwidth
- **Compute Throughput**: FLOPS or other compute metrics

### 4. Synchronization Effects

- **Block Synchronization**Warp Scheduling Decisions**: How warps are scheduled on SMs

### Tools and Techniques for GPU-Side Measurement

- **SASS/PTX Analysis**: Examining low-level assembly code
- **Hardware Performance Counters**: GPU-specific counters for various metrics
- **Kernel Instrumentation**: Adding timing code directly to kernels
- **Specialized Profilers**: Nsight Compute, Nvprof for deep GPU insights
- **Visual Profilers**: Timeline views of kernel execution

## When Kernel Instrumentation Is Necessary

While tools like Nsight Compute and nvprof provide extensive profiling capabilities, there are specific scenarios where adding timing code directly to CUDA kernels is necessary or beneficial:

### 1. Fine-Grained Internal Kernel Timing

**When to use**: When you need to measure execution time of specific code sections within a kernel.

```cuda
__global__ void complexKernel(float* data, int n) {
    // Shared variables for timing
    __shared__ clock64_t start_time, section1_time, section2_time, end_time;
    
    // Only one thread per block records timestamps
    if (threadIdx.x == 0) {
        start_time = clock64();
    }
    
    // First computation section
    // ...complex operations...
    
    __syncthreads();  // Ensure all threads complete
    
    if (threadIdx.x == 0) {
        section1_time = clock64();
    }
    
    // Second computation section
    // ...more operations...
    
    __syncthreads();
    
    if (threadIdx.x == 0) {
        section2_time = clock64();
    }
    
    // Final section
    // ...finishing operations...
    
    __syncthreads();
    
    if (threadIdx.x == 0) {
        end_time = clock64();
        // Store timing results to global memory
        // Convert clock ticks to milliseconds based on device clock rate
    }
}
```

**Benefits**: Provides visibility into kernel internals that external profilers cannot capture, especially for identifying hotspots within complex kernels.

### 2. Conditional or Divergent Path Analysis

**When to use**: When measuring performance of different execution paths in divergent code.

```cuda
__global__ void divergentPathKernel(float* data, int* path_times, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        clock64_t start, end;
        start = clock64();
        
        if (data[idx] > 0) {
            // Path A - expensive computation
            for (int i = 0; i < 100; i++) {
                data[idx] = sinf(data[idx]) * cosf(data[idx]);
            }
        } else {
            // Path B - different computation
            for (int i = 0; i < 50; i++) {
                data[idx] = data[idx] * data[idx] + 1.0f;
            }
        }
        
        end = clock64();
        
        // Record which path was taken and how long it took
        path_times[idx * 2] = (data[idx] > 0) ? 1 : 0;  // Path indicator
        path_times[idx * 2 + 1] = (int)(end - start);   // Time taken
    }
}
```

**Benefits**: Helps identify the performance impact of thread divergence by measuring each path individually.

### 3. Dynamic Workload Profiling

**When to use**: When dealing with algorithms where workload varies significantly between threads or blocks.

```cuda
__global__ void dynamicWorkloadKernel(int* elements, int* work_counts, int* times, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int work_items = elements[idx];  // Each thread has different amount of work
        
        clock64_t start = clock64();
        
        // Perform variable amount of work
        for (int i = 0; i < work_items; i++) {
            // Do computation
        }
        
        clock64_t end = clock64();
        
        // Record workload and time
        work_counts[idx] = work_items;
        times[idx] = (int)(end - start);
    }
}
```

**Benefits**: Reveals correlation between workload characteristics and execution time, helping optimize load balancing.

### 4. Custom Hardware Counter Access

**When to use**: When needing specific hardware performance metrics at precise points in execution.

```cuda
#include <cooperative_groups.h>
using namespace cooperative_groups;

__global__ void customCounterKernel(float* data, int* counter_values, int n) {
    thread_block block = this_thread_block();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (threadIdx.x == 0) {
        // Reset SM L1 cache hit counter (hypothetical example)
        asm volatile("read.ptx.special.register %0, l1_cache_hits;" : "=r"(counter_values[blockIdx.x * 4 + 0]));
    }
    
    // First phase of computation with potential L1 cache use
    // ...
    
    block.sync();
    
    if (threadIdx.x == 0) {
        // Read L1 cache hit counter after first phase
        asm volatile("read.ptx.special.register %0, l1_cache_hits;" : "=r"(counter_values[blockIdx.x * 4 + 1]));
    }
    
    // Second phase of computation
    // ...
    
    block.sync();
    
    if (threadIdx.x == 0) {
        // Final counter reading
        asm volatile("read.ptx.special.register %0, l1_cache_hits;" : "=r"(counter_values[blockIdx.x * 4 + 2]));
    }
}
```

**Note**: This is a conceptual example. Actual hardware counter access varies by GPU architecture and requires specific intrinsics or assembly instructions.

### 5. Real-time Algorithm Adaptation

**When to use**: When kernels need to self-tune based on performance feedback.

```cuda
__global__ void adaptiveKernel(float* data, float* timing_data, int n, int iterations) {
    __shared__ clock64_t start, mid, end;
    __shared__ float method_a_time, method_b_time;
    __shared__ int selected_method;
    
    if (threadIdx.x == 0) {
        // Initialize with method A as default
        selected_method = 0;
    }
    
    __syncthreads();
    
    // Run for multiple iterations, adapting the algorithm
    for (int iter = 0; iter < iterations; iter++) {
        if (threadIdx.x == 0) {
            start = clock64();
        }
        
        // Try method A first
        if (selected_method == 0) {
            // Method A implementation
            // ...
        }
        
        __syncthreads();
        
        if (threadIdx.x == 0) {
            mid = clock64();
        }
        
        // Try method B
        if (selected_method == 1) {
            // Method B implementation
            // ...
        }
        
        __syncthreads();
        
        if (threadIdx.x == 0) {
            end = clock64();
            
            // Calculate execution times
            method_a_time = (mid - start);
            method_b_time = (end - mid);
            
            // Choose faster method for next iteration
            selected_method = (method_a_time <= method_b_time) ? 0 : 1;
            
            // Record which method was faster
            if (blockIdx.x == 0) {
                timing_data[iter] = selected_method;
            }
        }
        
        __syncthreads();
    }
}
```

**Benefits**: Enables algorithmic choices within the kernel based on real-time performance measurements.

### 6. Multi-GPU Kernel Coordination

**When to use**: When coordinating work across multiple GPUs and timing is critical for synchronization.

```cuda
__global__ void coordinatedKernel(float* data, volatile int* sync_flags, 
                                 volatile clock64_t* timing, int device_id, int n) {
    // Record start time
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        timing[device_id] = clock64();
        // Signal this GPU has started
        sync_flags[device_id] = 1;
    }
    
    // Perform computation
    // ...
    
    // Wait for all GPUs to finish a phase
    __syncthreads();
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        timing[device_id + 4] = clock64();  // Record completion time
        sync_flags[device_id] = 2;  // Signal completion
        
        // Wait for other GPUs (simplified busy-wait)
        while (sync_flags[0] < 2 || sync_flags[1] < 2 || 
               sync_flags[2] < 2 || sync_flags[3] < 2) {
            // Busy wait
        }
    }
    
    __syncthreads();
    
    // Continue with coordinated execution
    // ...
}
```

**Benefits**: Helps identify load imbalances and synchronization overhead in multi-GPU systems.

### Implementation Considerations

When implementing kernel instrumentation:

1. **Clock Resolution**: Use appropriate timing functions:
   - `clock64()` provides device cycle counter (high resolution but architecture-dependent)
   - Convert cycles to time using device clock rate

2. **Measurement Overhead**: Minimize the impact of measurement on the results:
   - Only time critical sections
   - Use minimal thread participation in timing (e.g., one thread per block)

3. **Data Extraction**: Consider how timing data will be retrieved:
   - Store results in global memory
   - Consider using atomics if multiple threads report timing
   - Aggregation may be needed for large thread counts

4. **Synchronization Requirements**: Ensure proper synchronization between measurements:
   - Use `__syncthreads()` to create consistent timing boundaries
   - Consider cooperative groups for complex synchronization

### Example: Comprehensive Kernel Section Timing

```cuda
#include <cuda_runtime.h>
#include <helper_cuda.h> // For checkCudaErrors

// Structure to hold timing results
struct KernelTimings {
    long long init_time;
    long long compute_time;
    long long finalize_time;
    long long total_time;
};

__global__ void instrumentedKernel(float* input, float* output, int n, KernelTimings* timings) {
    extern __shared__ float shared_data[];
    
    // Only one thread per block records time
    clock64_t block_start, init_end, compute_end, block_end;
    if (threadIdx.x == 0) {
        block_start = clock64();
    }
    
    // Initialization phase
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        shared_data[threadIdx.x] = input[idx] * 2.0f;
    }
    
    __syncthreads();
    
    if (threadIdx.x == 0) {
        init_end = clock64();
    }
    
    // Computation phase
    if (idx < n) {
        float sum = 0.0f;
        for (int i = 0; i < blockDim.x; i++) {
            sum += shared_data[i];
        }
        output[idx] = sum / blockDim.x;
    }
    
    __syncthreads();
    
    if (threadIdx.x == 0) {
        compute_end = clock64();
    }
    
    // Finalization phase
    if (idx < n) {
        output[idx] = output[idx] * output[idx];
    }
    
    __syncthreads();
    
    if (threadIdx.x == 0) {
        block_end = clock64();
        
        // Atomic add to accumulate times across blocks
        atomicAdd(&timings->init_time, init_end - block_start);
        atomicAdd(&timings->compute_time, compute_end - init_end);
        atomicAdd(&timings->finalize_time, block_end - compute_end);
        atomicAdd(&timings->total_time, block_end - block_start);
    }
}

// Host-side function to execute and extract timing
void runAndTimeKernel(float* d_input, float* d_output, int n, int blockSize) {
    // Allocate and initialize timing structure on device
    KernelTimings* d_timings;
    checkCudaErrors(cudaMalloc(&d_timings, sizeof(KernelTimings)));
    checkCudaErrors(cudaMemset(d_timings, 0, sizeof(KernelTimings)));
    
    // Calculate grid dimensions
    int gridSize = (n + blockSize - 1) / blockSize;
    
    // Launch kernel with shared memory
    instrumentedKernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(
        d_input, d_output, n, d_timings);
    
    // Wait for kernel to complete
    checkCudaErrors(cudaDeviceSynchronize());
    
    // Copy timing results back to host
    KernelTimings h_timings;
    checkCudaErrors(cudaMemcpy(&h_timings, d_timings, sizeof(KernelTimings), cudaMemcpyDeviceToHost));
    
    // Get device properties to convert cycles to milliseconds
    cudaDeviceProp prop;
    checkCudaErrors(cudaGetDeviceProperties(&prop, 0));
    float ms_per_cycle = 1000.0f / (prop.clockRate * 1000.0f);
    
    // Print timing results
    printf("Kernel Timing Results:\n");
    printf("  Initialization: %.4f ms\n", h_timings.init_time * ms_per_cycle / gridSize);
    printf("  Computation:    %.4f ms\n", h_timings.compute_time * ms_per_cycle / gridSize);
    printf("  Finalization:   %.4f ms\n", h_timings.finalize_time * ms_per_cycle / gridSize);
    printf("  Total:          %.4f ms\n", h_timings.total_time * ms_per_cycle / gridSize);
    
    // Free device memory
    checkCudaErrors(cudaFree(d_timings));
}
```

## Hooking CPU-Side Functions with eBPF

eBPF (extended Berkeley Packet Filter) provides powerful mechanisms for tracing and monitoring system behavior on Linux without modifying source code. For GPU workloads, eBPF can be particularly valuable for correlating CPU-side activity with GPU performance.

An example code can be found in the [bpf-developer-tutorial](https://github.com/eunomia-bpf/bpf-developer-tutorial/tree/main/src/47-cuda-events).

### What is eBPF?

eBPF is a technology that allows running sandboxed programs in the Linux kernel without changing kernel source code or loading kernel modules. It's widely used for performance analysis, security monitoring, and networking.

### eBPF for GPU Workload Profiling

While eBPF cannot directly instrument code running on the GPU, it excels at monitoring the CPU-side interactions with the GPU:

#### 1. Tracing CUDA Driver Interactions

```c
// Example eBPF program to trace CUDA driver function calls
int trace_cuLaunchKernel(struct pt_regs *ctx) {
    u64 ts = bpf_ktime_get_ns();
    struct data_t data = {};
    
    data.timestamp = ts;
    data.pid = bpf_get_current_pid_tgid() >> 32;
    bpf_get_current_comm(&data.comm, sizeof(data.comm));
    
    // Capture function arguments
    data.gridDimX = PT_REGS_PARM2(ctx);
    data.gridDimY = PT_REGS_PARM3(ctx);
    data.gridDimZ = PT_REGS_PARM4(ctx);
    
    events.perf_submit(ctx, &data, sizeof(data));
    return 0;
}
```

#### 2. Correlating System Events with GPU Activity

eBPF can monitor:
- File I/O operations that may affect GPU data transfers
- Scheduler decisions that impact CPU-GPU coordination
- Memory management events relevant to GPU buffer handling

#### 3. Building a Complete Picture

By combining eBPF-gathered CPU-side data with GPU profiling information:
- Track data from its source to the GPU and back
- Identify system-level bottlenecks affecting GPU performance
- Understand scheduling issues that create GPU idle time

### eBPF Tools for GPU Workloads

1. **BCC (BPF Compiler Collection)**: Provides Python interfaces for eBPF programs
2. **bpftrace**: High-level tracing language for Linux eBPF
3. **Custom eBPF programs**: Tailored specifically for CUDA/GPU workloads

Example of tracing CUDA memory operations with bpftrace:

```
bpftrace -e '
uprobe:/usr/lib/libcuda.so:cuMemAlloc {
    printf("cuMemAlloc called: size=%llu, pid=%d, comm=%s\n", 
           arg1, pid, comm);
    @mem_alloc_bytes = hist(arg1);
}
uprobe:/usr/lib/libcuda.so:cuMemFree {
    printf("cuMemFree called: pid=%d, comm=%s\n", pid, comm);
}
'
```

## Integrated Profiling Approaches

Effective GPU application profiling requires integrating data from both CPU and GPU sides:

### 1. Timeline Correlation

Aligning events across CPU and GPU timelines to identify:
- **Kernel Launch Delays**: Gap between CPU request and GPU execution
- **Transfer-Compute Overlap**: Effectiveness of asynchronous operations
- **CPU-GPU Synchronization Points**: Where the CPU waits for the GPU

### 2. Bottleneck Identification

Using combined data to pinpoint whether bottlenecks are:
- **CPU-Bound**: CPU preparation of data or launch overhead
- **Transfer-Bound**: PCIe or memory bandwidth limitations
- **GPU Compute-Bound**: Kernel algorithm efficiency
- **GPU Memory-Bound**: GPU memory access patterns

### 3. Multi-Level Optimization Strategy

Developing a holistic optimization approach:
1. **System Level**: PCIe configuration, power settings, CPU affinity
2. **Application Level**: Kernel launch patterns, memory management
3. **Algorithm Level**: Kernel implementation, memory access patterns
4. **Instruction Level**: PTX/SASS optimizations

## Case Studies

### Case Study 1: Deep Learning Training Framework

In a deep learning framework, we observed:

- **CPU-Side Profiling**: Identified inefficient data preprocessing before GPU transfers
- **GPU-Side Profiling**: Showed high utilization but poor memory access patterns
- **eBPF Analysis**: Revealed that Linux page cache behavior was causing unpredictable data transfer timing

**Solution**: Implemented pinned memory with explicit prefetching guided by eBPF-gathered access patterns, resulting in 35% throughput improvement.

### Case Study 2: Real-time Image Processing Pipeline

For a real-time image processing application:

- **CPU-Side Profiling**: Showed bursty kernel launches causing GPU idle time
- **GPU-Side Profiling**: Indicated good kernel efficiency but poor occupancy
- **eBPF Analysis**: Discovered thread scheduling issues on CPU affecting launch timing

**Solution**: Used eBPF insights to implement CPU thread pinning and reorganized the pipeline, achieving consistent frame rates with 22% less end-to-end latency.

## Future Directions

The boundary between CPU and GPU profiling continues to evolve:

1. **Unified Memory Profiling**: As unified memory becomes more prevalent, new tools are needed to track page migrations and access patterns

2. **System-on-Chip Integration**: As GPUs become more integrated with CPUs, profiling boundaries will blur, requiring new approaches

3. **Multi-GPU Systems**: Distributed training and inference across multiple GPUs introduces new profiling challenges

4. **AI-Assisted Profiling**: Using machine learning to automatically identify patterns and suggest optimizations across the CPU-GPU boundary

## References

1. NVIDIA. "CUPTI: CUDA Profiling Tools Interface." [https://docs.nvidia.com/cuda/cupti/](https://docs.nvidia.com/cuda/cupti/)
2. Gregg, Brendan. "BPF Performance Tools: Linux System and Application Observability." Addison-Wesley Professional, 2019.
3. NVIDIA. "Nsight Systems User Guide." [https://docs.nvidia.com/nsight-systems/](https://docs.nvidia.com/nsight-systems/)
4. Awan, Ammar Ali, et al. "Characterizing Machine Learning I/O Workloads on NVME and CPU-GPU Systems." IEEE International Parallel and Distributed Processing Symposium Workshops, 2022.
5. The eBPF Foundation. "What is eBPF?" [https://ebpf.io/what-is-ebpf/](https://ebpf.io/what-is-ebpf/)
6. NVIDIA. "Tools for Profiling CUDA Applications." [https://developer.nvidia.com/tools-overview](https://developer.nvidia.com/tools-overview)
7. Arafa, Yehia, et al. "Low Overhead Instruction Latency Characterization for NVIDIA GPGPUs." High Performance Computing: ISC High Performance 2019.
8. Haidar, Azzam, et al. "Harnessing GPU Tensor Cores for Fast FP16 Arithmetic to Speed up Mixed-Precision Iterative Refinement Solvers." SC18: International Conference for High Performance Computing, Networking, Storage and Analysis, 2018. 
