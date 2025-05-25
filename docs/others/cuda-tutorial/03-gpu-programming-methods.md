# CUDA Programming Methods Comparison: Matrix Multiplication

This example demonstrates and compares different approaches to GPU programming using CUDA, focusing on a matrix multiplication problem. By implementing the same algorithm using various techniques, we can understand the trade-offs between programming complexity, performance, and code maintainability.

You can find the code in <https://github.com/eunomia-bpf/basic-cuda-tutorial>

## Overview

Matrix multiplication is a classic problem that benefits greatly from parallelization. This example implements a simple matrix multiplication C = A × B using seven different approaches:

1. Standard CUDA C/C++
2. CUDA with Inline PTX Assembly
3. CUDA Unified Memory
4. CUDA Shared Memory
5. Thrust (high-level C++ abstraction)
6. CUDA Streams
7. CUDA Dynamic Parallelism

For each implementation, we measure and compare the execution time, verifying the correctness of results against a CPU implementation.

## Programming Methods Explained

### 1. Standard CUDA C/C++

The standard CUDA approach uses explicit memory management and a kernel function:

```cuda
__global__ void matrix_multiply_cuda(float *A, float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}
```

This implementation requires manual:
- Memory allocation on both host and device
- Data transfer between host and device
- Kernel launch configuration
- Synchronization
- Memory deallocation

**Advantages**:
- Direct control over memory management
- Good performance
- Familiar programming model

**Disadvantages**:
- Requires explicit memory management
- More verbose code
- Memory transfers can be a bottleneck

**Implementation Details**:
- Each thread computes one element of the output matrix
- Threads are organized in a 2D grid that maps directly to the output matrix dimensions
- The kernel has O(n) computational complexity per thread
- Global memory is used for all matrix elements

### 2. CUDA with Inline PTX Assembly

PTX (Parallel Thread Execution) is NVIDIA's low-level assembly language. We can embed PTX directly in CUDA C++ code:

```cuda
__device__ float multiply_add_ptx(float a, float b, float c) {
    float result;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(result) : "f"(a), "f"(b), "f"(c));
    return result;
}
```

This implementation uses a fused multiply-add (FMA) instruction which computes `a * b + c` in a single operation, potentially improving performance and numerical accuracy.

**Advantages**:
- Fine-grained control over specific operations
- Can leverage architecture-specific instructions
- Potentially better performance for critical sections
- Direct access to hardware features not exposed in CUDA C/C++

**Disadvantages**:
- Least portable across different architectures
- Most complex to write and maintain
- Requires deep knowledge of GPU architecture
- Architecture-specific optimizations may become obsolete with newer hardware

**Implementation Details**:
- Uses the `fma.rn.f32` PTX instruction for fused multiply-add
- The `.rn` suffix specifies round-to-nearest-even rounding mode
- The instruction executes in a single clock cycle on compatible hardware
- PTX assembly allows precise control over instruction selection and scheduling

### 3. CUDA Unified Memory

Unified Memory provides a single memory space accessible by both CPU and GPU:

```cuda
cudaMallocManaged(&u_A, matrix_size);
cudaMallocManaged(&u_B, matrix_size);
cudaMallocManaged(&u_C, matrix_size);
```

The kernel code remains the same as the standard CUDA version, but memory management is simplified.

**Advantages**:
- Simplified memory management
- No explicit data transfers
- Easier programming model
- Automatic page migration between CPU and GPU
- Enables larger-than-GPU-memory datasets

**Disadvantages**:
- Potentially lower performance due to automatic page migration
- Less control over data movement
- Performance depends heavily on the access pattern
- First-touch overhead and potential page faults

**Implementation Details**:
- Uses `cudaMallocManaged()` instead of separate `malloc()` and `cudaMalloc()`
- The CUDA runtime handles data transfers automatically
- Memory pages migrate on-demand between CPU and GPU
- The same pointer can be used on both host and device
- Prefetching hints can be used to optimize data movement (not shown in this example)

### 4. CUDA Shared Memory

Shared memory is a fast on-chip memory accessible by all threads in a block:

```cuda
__global__ void matrix_multiply_shared(float *A, float *B, float *C, int n) {
    __shared__ float shared_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shared_B[BLOCK_SIZE][BLOCK_SIZE];
    
    // Load tiles of matrices into shared memory
    // ...
    
    // Compute using the faster shared memory
    // ...
}
```

This implementation divides matrices into tiles that fit in shared memory, reducing global memory accesses.

**Advantages**:
- Much faster memory access (typically 100x faster than global memory)
- Reduced global memory bandwidth requirements
- Better performance for memory-bound applications
- Enables data reuse within thread blocks

**Disadvantages**:
- Limited size of shared memory (typically 48KB-64KB per SM)
- More complex programming model
- Requires careful management of tile sizes
- Potential for bank conflicts if not carefully designed

**Implementation Details**:
- Matrices are divided into tiles of size BLOCK_SIZE × BLOCK_SIZE
- Each thread block loads one tile from each input matrix into shared memory
- Threads within a block synchronize using `__syncthreads()`
- Each thread computes one element of the output tile
- The algorithm reduces global memory accesses by a factor of BLOCK_SIZE

### 5. Thrust High-Level Implementation

Thrust is a C++ template library for CUDA that provides high-level abstractions:

```cpp
void run_thrust_implementation(float *h_A, float *h_B, float *h_C, int n) {
    thrust::device_vector<float> d_A(h_A, h_A + n * n);
    thrust::device_vector<float> d_B(h_B, h_B + n * n);
    thrust::device_vector<float> d_C(n * n);
    
    // Create a 2D index space and transform
    // ...
}
```

**Advantages**:
- Most concise code
- High-level abstractions similar to STL
- Automatic memory management
- Highly reusable components
- Reduced development time and fewer bugs

**Disadvantages**:
- Less control over implementation details
- May be less performant for specific use cases
- Can be harder to debug
- Potential overhead from abstractions

**Implementation Details**:
- Uses `thrust::device_vector` for automatic GPU memory management
- Leverages `thrust::transform` algorithm with a custom functor
- Creates a 2D index space using `thrust::make_zip_iterator` and `thrust::counting_iterator`
- The functor implements the matrix multiplication for each element
- Memory transfers are handled automatically by the Thrust containers

### 6. CUDA Streams

CUDA streams enable concurrent operations on the GPU:

```cuda
void run_cuda_streams_implementation(float *h_A, float *h_B, float *h_C, int n) {
    // Create multiple CUDA streams
    const int numStreams = 4;
    cudaStream_t streams[numStreams];
    
    // Divide work among streams
    for (int i = 0; i < numStreams; i++) {
        // Asynchronous memory transfers and kernel launches
        cudaMemcpyAsync(..., streams[i]);
        matrix_multiply_cuda<<<grid, threads, 0, streams[i]>>>(...);
        cudaMemcpyAsync(..., streams[i]);
    }
    
    // Synchronize all streams
    for (int i = 0; i < numStreams; i++) {
        cudaStreamSynchronize(streams[i]);
    }
}
```

**Advantages**:
- Overlaps computation with data transfers
- Enables concurrent kernel execution
- Better utilization of GPU resources
- Can significantly improve overall throughput
- Good for processing independent data chunks

**Disadvantages**:
- More complex synchronization
- Harder to debug and reason about
- Potential for race conditions if not carefully designed
- Benefits depend on hardware capabilities

**Implementation Details**:
- Divides the matrix into horizontal strips, one per stream
- Each stream processes its strip independently
- Uses `cudaMemcpyAsync()` for non-blocking memory transfers
- Launches kernels in different streams with the stream parameter
- Each stream has its own command queue that executes independently
- Final synchronization ensures all operations complete

### 7. Dynamic Parallelism

Dynamic parallelism allows CUDA kernels to launch nested kernels:

```cuda
__global__ void matrix_multiply_dynamic_parent(float *A, float *B, float *C, int n) {
    // Calculate submatrix position based on thread index
    // ...
    
    // Launch a child kernel for this submatrix
    multiply_submatrix<<<dimGrid, dimBlock>>>(A, B, C, n, row_start, col_start, subsize);
}
```

**Advantages**:
- Enables adaptive algorithms that refine work dynamically
- Allows recursive problem decomposition
- Can handle irregular workloads efficiently
- Reduces CPU-GPU coordination overhead
- More natural expression of divide-and-conquer algorithms

**Disadvantages**:
- Additional overhead of kernel launch from device
- More complex resource management
- Requires newer GPU architectures (Compute Capability 3.5+)
- Potentially deeper call stacks and more register usage

**Implementation Details**:
- Parent kernel divides the matrix into large submatrices
- Each thread in the parent grid launches a child grid to process one submatrix
- Child kernels work on smaller, more manageable chunks
- Requires the `-rdc=true` compilation flag for relocatable device code
- Parent grid synchronizes automatically with all child grids at kernel end

## Memory Hierarchy and Access Patterns

Different memory types in CUDA have vastly different performance characteristics:

| Memory Type       | Access Speed | Scope              | Lifetime         | Caching    |
|-------------------|--------------|--------------------| -----------------|------------|
| Global Memory     | Slowest      | Host & All threads | Application      | L2 only    |
| Shared Memory     | Fast         | Thread block       | Kernel execution | On-chip    |
| Registers         | Fastest      | Single thread      | Thread lifetime  | On-chip    |
| Constant Memory   | Medium       | All threads (read) | Application      | Special cache |
| Texture Memory    | Medium       | All threads (read) | Application      | Special cache |
| Local Memory      | Slow         | Single thread      | Thread lifetime  | L2 only    |
| Unified Memory    | Variable     | Host & All threads | Application      | System managed |

Our implementations demonstrate different ways to leverage this memory hierarchy:

1. **Standard CUDA**: Uses global memory for all data
2. **PTX Assembly**: Same memory pattern as standard CUDA but with optimized instructions
3. **Unified Memory**: Uses automatically managed memory
4. **Shared Memory**: Explicitly caches data in on-chip shared memory
5. **Thrust**: Abstracts memory management through containers
6. **CUDA Streams**: Uses global memory with overlapped transfers
7. **Dynamic Parallelism**: Uses global memory with hierarchical access patterns

## Advanced Optimization Techniques

Beyond the methods shown in the examples, other optimization techniques include:

1. **Warp Shuffle Instructions**: Exchange data between threads in a warp without using shared memory
2. **Tensor Cores**: Specialized hardware for matrix operations (on newer GPUs)
3. **Persistent Threads**: Keep threads resident for multiple work items
4. **Register Blocking**: Store partial results in registers to reduce memory traffic
5. **Memory Coalescing**: Ensure aligned, contiguous memory access patterns
6. **Instruction-Level Parallelism**: Schedule independent operations to hide latency
7. **Loop Unrolling**: Reduce loop overhead and increase instruction-level parallelism
8. **Function Inlining**: Eliminate function call overhead
9. **Occupancy Optimization**: Balance resource usage to maximize active warps

## Choosing the Right Approach

Use this decision framework to select the appropriate implementation:

- **Standard CUDA** when you need a balance of control and readability
- **PTX Assembly** only for performance-critical sections that can benefit from specific instructions
- **Unified Memory** for easier development with modest performance requirements
- **Shared Memory** when memory access is a bottleneck and memory locality can be exploited
- **Thrust** for rapid development, especially when implementing standard algorithms
- **CUDA Streams** when you need to overlap computation and data transfers
- **Dynamic Parallelism** for problems that benefit from recursive decomposition or adaptive refinement

## Compiler Flags and Optimization Levels

Different compiler flags can significantly impact performance:

```makefile
# Basic compilation
nvcc -o basic_version file.cu

# Optimized compilation
nvcc -O3 -arch=sm_70 -o optimized_version file.cu

# With dynamic parallelism
nvcc -rdc=true -O3 -arch=sm_70 -o dynamic_parallel_version file.cu

# With fast math (may reduce precision)
nvcc -O3 -use_fast_math -arch=sm_70 -o fast_math_version file.cu
```

Our example uses `-O3` for high optimization and `-rdc=true` for dynamic parallelism.

## Building and Running

To compile the example:
```bash
make basic03
```

To run:
```bash
./basic03
```

The program will:
1. Generate random matrices
2. Compute the result on CPU for reference
3. Run each GPU implementation
4. Verify correctness of each implementation
5. Print timing information and speedup compared to CPU

## Profiling and Performance Analysis

To analyze the performance of these implementations more deeply, use NVIDIA's profiling tools:

```bash
# Basic profiling
nvprof ./basic03

# Detailed timeline
nvprof --export-profile timeline.nvvp ./basic03

# For newer versions, use Nsight Systems
nsys profile ./basic03
```

Key metrics to monitor:
- Global memory load/store throughput
- Shared memory load/store throughput
- Achieved occupancy
- SM efficiency
- Warp execution efficiency
- Memory bandwidth utilization

## Future Directions and Advanced Topics

The field of GPU computing continues to evolve. Some advanced topics to explore:

1. **Multi-GPU Programming**: Distributing work across multiple GPUs
2. **Heterogeneous Computing**: Combining CPU and GPU computation optimally
3. **Mixed Precision**: Using lower precision where appropriate for better performance
4. **Tensor Core Programming**: Leveraging specialized hardware for matrix operations
5. **Graph-based Execution**: Using CUDA Graphs for optimized workflow execution
6. **Cooperative Groups**: More flexible thread synchronization capabilities
7. **CUDA-aware MPI**: Direct GPU-to-GPU communication in distributed systems

## Further Reading

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [CUDA PTX ISA Documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
- [CUDA Unified Memory](https://developer.nvidia.com/blog/unified-memory-cuda-beginners/)
- [CUDA Shared Memory](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)
- [Thrust Documentation](https://docs.nvidia.com/cuda/thrust/index.html)
- [CUDA Streams and Concurrency](https://developer.nvidia.com/blog/gpu-pro-tip-cuda-7-streams-simplify-concurrency/)
- [CUDA Dynamic Parallelism](https://developer.nvidia.com/blog/introduction-cuda-dynamic-parallelism/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) 