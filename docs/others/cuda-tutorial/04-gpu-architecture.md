# Tutorial: Understanding GPU Architecture and Execution Model

**Time Required:** 60-75 minutes
**Difficulty:** Intermediate
**Prerequisites:** Completed Tutorials 01 and 02, basic understanding of computer architecture

By the end of this tutorial, you will understand how GPUs are physically organized, how threads are scheduled and executed, how the memory hierarchy works, and why certain coding patterns perform better than others. This knowledge is essential for writing high-performance CUDA code.

You can find the code at <https://github.com/eunomia-bpf/basic-cuda-tutorial>

## Why GPU Architecture Matters

You might wonder why you need to understand the hardware when writing CUDA code. After all, compilers are supposed to handle optimization for you. The reality is that GPU architecture is fundamentally different from CPU architecture, and writing efficient GPU code requires understanding these differences.

Consider this: on a CPU, you might write a loop that processes one element at a time, and the processor tries to speed it up through pipelining, branch prediction, and out-of-order execution. On a GPU, you have thousands of threads running simultaneously, and performance depends on how well these threads cooperate and access memory. Writing code that ignores the GPU's architecture is like trying to drive a motorcycle the same way you drive a car - technically possible, but you're missing the point entirely.

## The GPU Hardware Hierarchy

Let's start by examining what a modern GPU actually looks like. When you run the example, you'll see detailed information about your GPU:

```bash
make 04-gpu-architecture
./04-gpu-architecture
```

On an NVIDIA GeForce RTX 5090, you'll see output like this:

```
Device 0: NVIDIA GeForce RTX 5090
  Compute Capability: 12.0
  SMs (Multiprocessors): 170
  Warp Size: 32 threads
  Max Threads per SM: 1536
  Max Threads per Block: 1024
```

Let's break down what these numbers mean and why they matter.

### Streaming Multiprocessors: The Core Building Blocks

A GPU consists of many Streaming Multiprocessors (SMs). Each SM is like a small processor that can execute many threads simultaneously. The RTX 5090 has 170 SMs, meaning it can run 170 different groups of threads in parallel at any given moment.

Think of an SM as a factory floor with multiple assembly lines. Each assembly line (called a warp) has 32 workers (threads) who all perform the same operation at the same time. The SM schedules which assembly lines are active and feeds them work.

When you launch a kernel, the CUDA runtime distributes your thread blocks across the available SMs. If you launch 340 blocks and have 170 SMs, initially two blocks will be assigned to each SM. As blocks complete, new ones get scheduled. This is why having more blocks than SMs is generally good - it keeps all the hardware busy.

### The Thread Execution Model

Every thread in CUDA has a unique identity determined by its position in a three-dimensional grid:

```cuda
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
```

This hierarchy exists for a reason. Threads are organized into blocks, and blocks are organized into grids. This structure maps directly to how the hardware schedules and executes code.

Run the thread hierarchy demonstration. You'll see output like:

```
Thread ID: 0, Position: (0,0,0) in Block (0,0,0) of Grid (4,4,4)
Thread ID: 800, Position: (0,0,2) in Block (0,3,0) of Grid (4,4,4)
```

Each thread knows exactly where it is in this hierarchy. This is crucial because it allows threads to compute which data they should process without any centralized coordination. Thread 0 in block (0,0,0) processes different data than thread 0 in block (1,0,0).

### Understanding Warps: The Fundamental Execution Unit

Here's something that might surprise you: even though you program individual threads, the GPU doesn't execute them one at a time. Instead, it executes them in groups of 32 called warps.

All 32 threads in a warp execute the same instruction at the same time. This is called SIMT (Single Instruction, Multiple Threads). It's similar to SIMD (Single Instruction, Multiple Data) from CPUs, but more flexible because threads can diverge when necessary.

Look at the warp execution output:

```
Thread   0: Warp 10000, Lane  0
Thread  32: Warp 10001, Lane  0
Thread  64: Warp 10002, Lane  0
```

Threads 0-31 are in warp 10000, threads 32-63 are in warp 10001, and so on. Within each warp, threads have lane IDs from 0 to 31. This organization has profound implications for performance.

### Warp Divergence: When Threads Take Different Paths

What happens when threads in a warp need to execute different code? Consider this kernel:

```cuda
if (threadIdx.x < 16) {
    // First half of warp does this
    result = expensive_operation_a(data);
} else {
    // Second half of warp does this
    result = expensive_operation_b(data);
}
```

Since all threads in a warp must execute the same instruction, the hardware does something clever but inefficient: it executes both paths. First, threads 0-15 execute operation_a while threads 16-31 are masked off (inactive). Then threads 16-31 execute operation_b while threads 0-15 are masked off.

This means divergent code takes the time of both paths combined. The demonstration shows this:

```
Note on warp divergence:
  When threads within a warp take different paths (diverge),
  the warp executes both paths, masking threads as appropriate.
  This reduces efficiency.
```

To avoid warp divergence, try to keep threads in a warp executing the same code path. If you must have conditionals, group threads so that all 32 threads in a warp take the same branch.

## Memory Hierarchy: The Key to Performance

GPU performance is often limited not by computation but by memory access. Understanding the memory hierarchy is crucial for writing fast code.

### Global Memory: Large but Slow

When you use `cudaMalloc`, you're allocating global memory. This is the main GPU memory - large (32 GB on the RTX 5090) but with high latency (hundreds of clock cycles).

```
Total Global Memory: 31.36 GB
Memory Clock Rate: 14001 MHz
Memory Bus Width: 512 bits
Peak Memory Bandwidth: 1792.13 GB/s
```

That bandwidth number is theoretical peak. In practice, you'll achieve a fraction of it depending on your memory access patterns. This brings us to a critical concept: memory coalescing.

### Memory Coalescing: Why Access Patterns Matter

Modern GPUs load memory in large chunks (typically 128 bytes). When threads in a warp access adjacent memory locations, the hardware can combine (coalesce) these accesses into a single transaction.

Consider these two patterns:

**Coalesced access:**
```cuda
// Thread 0 accesses data[0], thread 1 accesses data[1], etc.
output[tid] = data[tid] * 2.0f;
```

**Non-coalesced access:**
```cuda
// Thread 0 accesses data[0], thread 1 accesses data[8], etc.
output[tid] = data[tid * 8 % n] * 2.0f;
```

The first pattern can be satisfied with one 128-byte memory transaction for the entire warp. The second pattern might require 32 separate transactions. The performance difference is dramatic.

Run the memory coalescing demonstration:

```
Memory access timing:
  Elapsed time: 0.024 ms
```

While this simple example may not show huge differences, in real applications, proper coalescing can improve performance by 5-10x.

### Shared Memory: Fast On-Chip Storage

Shared memory is a small (48 KB per block) but extremely fast memory space that all threads in a block can access. It's located on the SM itself, so access latency is much lower than global memory.

```
Shared Memory per Block: 48 KB
```

Shared memory is perfect for situations where threads need to cooperate. Here's the classic pattern:

```cuda
__shared__ float sharedData[256];

// Load from global to shared memory
sharedData[threadIdx.x] = globalData[globalIdx];

// Synchronize to ensure all threads have loaded their data
__syncthreads();

// Now all threads can read from shared memory quickly
result = sharedData[threadIdx.x] + sharedData[threadIdx.x + 1];
```

The `__syncthreads()` is crucial. It's a barrier that ensures all threads in the block reach this point before any continue. Without it, some threads might try to read data that hasn't been loaded yet.

The demonstration shows a simple stencil operation using shared memory:

```
First few output values (should be sums of adjacent input values):
  output[0] = 1.0
  output[1] = 3.0
  output[2] = 5.0
```

Each output is the sum of two adjacent inputs, computed efficiently using shared memory.

### L2 Cache and Register File

Between shared and global memory sits the L2 cache:

```
L2 Cache Size: 98304 KB
```

The L2 cache is automatic - you don't explicitly manage it. It caches frequently accessed global memory, helping to hide latency. Recent GPU architectures allow you to influence caching behavior through PTX instructions, as we saw in Tutorial 02.

The fastest memory of all is registers:

```
Registers per Block: 65536
```

Each thread has its own private registers. Variables you declare in kernel code typically go in registers. Register access has essentially zero latency, but each SM has a limited number of registers to share among all active threads.

This creates a tradeoff: using more registers per thread allows more complex computations but limits how many threads can be active simultaneously (occupancy).

## Occupancy: Keeping the GPU Busy

Occupancy refers to the ratio of active warps to the maximum number of warps an SM can support. Higher occupancy generally means better performance because it helps hide memory latency.

```
Max Threads per SM: 1536
```

With a warp size of 32, this means each SM can handle up to 48 warps (1536 / 32). If your kernel uses so many registers or shared memory that each SM can only run 24 warps, your occupancy is 50%.

The occupancy calculation considers three factors:

**Threads per block:** More threads per block generally better, but don't exceed hardware limits.

**Registers per thread:** Fewer is better for occupancy, but might hurt performance if you spill to local memory.

**Shared memory per block:** Less is better for occupancy, but shared memory enables important optimizations.

You can check occupancy with:

```bash
nvcc --ptxas-options=-v 04-gpu-architecture.cu
```

This shows register and shared memory usage. Use the CUDA Occupancy Calculator to find the sweet spot for your kernel.

## Practical Example: Optimized Matrix Multiplication

Let's see how understanding architecture improves real code. Here's a naive matrix multiplication:

```cuda
__global__ void matmul_naive(float *A, float *B, float *C, int n) {
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

This works, but has terrible performance. Each thread reads an entire row of A and an entire column of B from global memory. For a 1024x1024 matrix, that's 1024 global memory accesses per thread.

Now consider a shared memory version that tiles the computation. By loading data into shared memory and reusing it across threads, we dramatically reduce global memory traffic. The kernel in `04-gpu-architecture.cu` at line 126 demonstrates this optimization.

## Common Architecture-Related Pitfalls

**Launching too few blocks:** If you only launch as many blocks as you have SMs, you can't hide latency. Launch 2-4x more blocks than SMs.

**Ignoring alignment:** Memory accesses should be aligned to 128 bytes when possible. Unaligned accesses waste bandwidth.

**Thread block size not a multiple of 32:** If you launch 100 threads per block, you're wasting 28 thread slots in the last warp of each block. Use 96 or 128 instead.

**Excessive register usage:** Check with `nvcc --ptxas-options=-v` and reduce if occupancy is low.

**Unnecessary synchronization:** `__syncthreads()` is expensive. Only use it when threads actually need to coordinate.

## Challenge Exercises

1. **Measure coalescing impact:** Write two versions of a simple kernel - one with perfect coalescing and one with terrible coalescing. Measure the performance difference on your GPU.

2. **Optimize occupancy:** Take a kernel that uses many registers. Use shared memory or restructure the computation to reduce register pressure. Measure the impact on performance.

3. **Test warp divergence:** Write a kernel with varying amounts of divergence (none, 50%, 100%). Measure the performance impact and verify it matches the theory.

## Summary

GPU architecture is fundamentally different from CPU architecture. Understanding the hierarchy of SMs, warps, and threads is essential for writing efficient code. Memory access patterns often matter more than computation. Coalesced accesses use memory bandwidth efficiently, while scattered accesses waste it.

Shared memory enables cooperation between threads in a block, dramatically reducing global memory traffic. Warps execute in lockstep, so divergent code paths hurt performance. Occupancy affects how well the GPU can hide memory latency.

The key to GPU performance is keeping all the hardware busy. Launch enough blocks to saturate the SMs. Use memory access patterns that enable coalescing. Leverage shared memory for data reuse. Avoid warp divergence when possible.

## Next Steps

Continue to **Tutorial 05: Neural Network Forward Pass** to see how architectural understanding applies to deep learning. You'll implement a simple neural network on the GPU and learn how matrix multiplication, activation functions, and memory layouts affect performance in real ML workloads.

## Further Reading

- [CUDA C++ Programming Guide - Hardware Implementation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#hardware-implementation)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [GPU Architecture Whitepapers](https://www.nvidia.com/en-us/data-center/resources/gpu-architecture/)
- [CUDA Occupancy Calculator](https://docs.nvidia.com/cuda/cuda-occupancy-calculator/index.html)
