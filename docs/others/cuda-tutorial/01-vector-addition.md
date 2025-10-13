# Tutorial: Your First CUDA Program - Vector Addition

**Time Required:** 30-45 minutes
**Difficulty:** Beginner
**Prerequisites:** Basic C/C++ knowledge, NVIDIA GPU with CUDA support

By the end of this tutorial, you will understand how to write, compile, and run a complete CUDA program that performs parallel vector addition on the GPU. You'll learn the fundamental workflow of GPU programming and see real performance improvements over CPU-only code.

## Understanding the Challenge

Imagine you need to add two arrays of 50,000 numbers together. On a CPU, you would write a loop that processes one element at a time. This sequential approach works, but it's slow when dealing with large datasets. GPUs excel at this type of problem because they can process thousands of elements simultaneously.

Think of it like this: a CPU is like having one very fast worker, while a GPU is like having thousands of workers who can each handle a small piece of the problem at the same time. For simple, repetitive tasks like adding numbers, the GPU's massive parallelism wins.

## Getting Started

First, make sure you have the CUDA toolkit installed. Verify your installation by running:

```bash
nvcc --version
nvidia-smi
```

The first command shows your CUDA compiler version, and the second displays your GPU information. If both commands work, you're ready to begin.

Clone the tutorial repository and navigate to the first example:

```bash
git clone https://github.com/eunomia-bpf/basic-cuda-tutorial
cd basic-cuda-tutorial
```

## Building and Running Your First CUDA Program

Let's start by building and running the example to see it in action:

```bash
make 01-vector-addition
./01-vector-addition
```

You should see output similar to:

```
Vector addition of 50000 elements
CUDA kernel launch with 196 blocks of 256 threads
Test PASSED
Done
```

Now that it works, let's understand what's happening under the hood.

## The CUDA Programming Model

Open `01-vector-addition.cu` in your editor. Every CUDA program follows a similar pattern:

1. Allocate memory on both CPU (host) and GPU (device)
2. Copy input data from CPU to GPU
3. Launch a kernel (function) that runs on the GPU
4. Copy results back from GPU to CPU
5. Clean up all allocated memory

This workflow might seem verbose compared to regular C programming, but it's necessary because the CPU and GPU have separate memory spaces. Data must be explicitly moved between them.

## Dissecting the Kernel Function

The heart of any CUDA program is the kernel function. Here's our vector addition kernel:

```cuda
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}
```

The `__global__` keyword tells the CUDA compiler that this function runs on the GPU but can be called from CPU code. Notice how simple the actual computation is: just one addition. The magic is in how we calculate the index `i`.

### Understanding Thread Indexing

Every GPU thread needs to know which element of the array it should process. The formula `blockDim.x * blockIdx.x + threadIdx.x` computes a unique global index for each thread.

To visualize this, imagine we have 50,000 elements and we organize our GPU threads into blocks of 256 threads each. We would need 196 blocks (rounding up from 50000/256). Here's how the indexing works:

- Thread 0 in Block 0: index = 256 * 0 + 0 = 0
- Thread 5 in Block 0: index = 256 * 0 + 5 = 5
- Thread 0 in Block 1: index = 256 * 1 + 0 = 256
- Thread 100 in Block 2: index = 256 * 2 + 100 = 612

The boundary check `if (i < numElements)` is crucial because our last block might have threads that extend beyond the array size. Without this check, those threads would access invalid memory.

## Memory Management in CUDA

Look at how we allocate memory in the main function:

```cuda
// Host (CPU) memory
float *h_A = (float *)malloc(size);

// Device (GPU) memory
float *d_A = NULL;
cudaMalloc((void **)&d_A, size);
```

We use a naming convention where `h_` prefix indicates host memory and `d_` prefix indicates device memory. This helps prevent common mistakes like passing GPU pointers to CPU functions.

The `cudaMalloc` function works similarly to `malloc`, but it allocates memory in the GPU's global memory. This memory is accessible by all GPU threads but has higher latency than shared memory or registers.

## Data Transfer Between Host and Device

After allocating memory on both sides, we need to copy our input data to the GPU:

```cuda
cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
```

The `cudaMemcpy` function is synchronous, meaning the CPU waits until the transfer completes before continuing. The fourth parameter specifies the direction: `cudaMemcpyHostToDevice` for CPU to GPU, and `cudaMemcpyDeviceToHost` for GPU to CPU.

These memory transfers can be a performance bottleneck. As a rule of thumb, you want to minimize the number of transfers and maximize the amount of computation performed on the GPU between transfers.

## Launching the Kernel

The kernel launch is where the parallel magic happens:

```cuda
int threadsPerBlock = 256;
int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
```

The triple angle bracket syntax `<<<blocksPerGrid, threadsPerBlock>>>` is CUDA's way of specifying the execution configuration. We're telling the GPU to launch 196 blocks with 256 threads each, giving us 50,176 total threads (slightly more than our 50,000 elements).

Why 256 threads per block? This is a carefully chosen default that works well on most NVIDIA GPUs. Thread blocks are scheduled on streaming multiprocessors (SMs), and 256 threads per block typically provides good occupancy without using too many resources.

The ceiling division formula `(numElements + threadsPerBlock - 1) / threadsPerBlock` ensures we always have enough threads to cover all elements. For 50,000 elements and 256 threads per block, this gives us 196 blocks.

## Error Checking

CUDA kernel launches are asynchronous, meaning the CPU doesn't wait for the kernel to complete. Errors in the kernel don't show up immediately. That's why we check for launch errors:

```cuda
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    fprintf(stderr, "Failed to launch kernel: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
}
```

This catches configuration errors like requesting too many threads per block or running out of GPU memory. Always check for errors after kernel launches and CUDA API calls during development.

## Verification and Cleanup

After copying results back to the host, we verify the computation:

```cuda
for (int i = 0; i < numElements; ++i) {
    if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
        fprintf(stderr, "Result verification failed at element %d!\n", i);
        exit(EXIT_FAILURE);
    }
}
```

We use floating-point comparison with a small epsilon (1e-5) because floating-point arithmetic on GPUs might produce slightly different results than on CPUs due to different rounding modes and instruction execution order.

Finally, we free all allocated memory:

```cuda
cudaFree(d_A);  // Free GPU memory
free(h_A);      // Free CPU memory
```

Forgetting to free memory causes leaks. GPU memory is often more limited than system RAM, so leaks can cause problems quickly.

## Hands-On Exercise: Measuring Performance

Now let's add timing to see the actual speedup. We'll use CUDA events for precise GPU timing. Add this code before the kernel launch:

```cuda
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
```

And after the kernel launch:

```cuda
cudaEventRecord(stop);
cudaEventSynchronize(stop);

float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
printf("Kernel execution time: %.3f ms\n", milliseconds);

cudaEventDestroy(start);
cudaEventDestroy(stop);
```

Recompile and run. You should see the kernel completes in less than a millisecond. Now compare this to a CPU implementation by adding this function:

```cuda
void vectorAddCPU(const float *A, const float *B, float *C, int numElements) {
    for (int i = 0; i < numElements; i++) {
        C[i] = A[i] + B[i];
    }
}
```

Time it using standard C timing functions. On most systems, the GPU version will be significantly faster, even including the memory transfer overhead.

## Understanding Occupancy and Block Size

Let's experiment with the block size. The current code uses 256 threads per block. Try modifying it to use different values:

```cuda
int threadsPerBlock = 128;  // or 512, or 1024
```

Recompile and run each version with timing enabled. You'll notice that 128 threads per block is slower, while 512 or 1024 might be similar to 256. This is because of GPU occupancy.

Occupancy refers to how well you're utilizing the GPU's streaming multiprocessors. Each SM has a limited number of registers, shared memory, and warp slots. Using 256 threads per block (8 warps) typically achieves good occupancy on most GPUs without exhausting these resources.

To check your kernel's occupancy, compile with:

```bash
nvcc --ptxas-options=-v 01-vector-addition.cu -o 01-vector-addition
```

Look for the "registers" and "shared memory" usage in the output. You can use the CUDA Occupancy Calculator to determine optimal block sizes for your specific GPU and kernel.

## Memory Bandwidth Analysis

Vector addition is a memory-bound operation, meaning performance is limited by how fast we can read and write data, not by computation speed. Let's calculate the achieved memory bandwidth:

For 50,000 elements:
- We read two float arrays: 50,000 * 4 bytes * 2 = 400 KB
- We write one float array: 50,000 * 4 bytes = 200 KB
- Total memory traffic: 600 KB

If your kernel runs in 0.1 ms, the bandwidth is:
- 600 KB / 0.0001 s = 6 GB/s

Compare this to your GPU's theoretical bandwidth (check `nvidia-smi` or GPU specifications). For example, the RTX 5090 has about 1792 GB/s theoretical bandwidth. If you're achieving 100-200 GB/s, you're doing well for a simple kernel.

The gap between theoretical and achieved bandwidth comes from several factors: memory access patterns, cache behavior, and PCIe transfer overhead. We'll explore optimizations in later tutorials.

## Memory Coalescing

Our kernel has excellent memory access patterns. Adjacent threads (within a warp) access adjacent memory locations. This allows the GPU to coalesce multiple memory requests into a single wide transaction.

Try this experiment: modify the kernel to access memory in a strided pattern:

```cuda
int i = (blockDim.x * blockIdx.x + threadIdx.x) * 2;  // Skip every other element
if (i < numElements) {
    C[i] = A[i] + B[i];
}
```

You'll need to launch twice as many threads and adjust the bounds check. Time this version and compare. The non-coalesced access pattern will be significantly slower because each memory request requires a separate transaction.

## Debugging Your CUDA Code

When things go wrong, CUDA provides several debugging tools:

**cuda-memcheck:** Detects memory errors like out-of-bounds access
```bash
cuda-memcheck ./01-vector-addition
```

**cuda-gdb:** GPU debugger for step-through debugging
```bash
cuda-gdb ./01-vector-addition
```

**Compute Sanitizer:** Modern replacement for cuda-memcheck
```bash
compute-sanitizer ./01-vector-addition
```

Common mistakes to watch for:
- Passing host pointers to kernels (will cause segfault)
- Forgetting to copy data back from device
- Not checking CUDA error codes
- Accessing memory beyond array bounds
- Mismatched memory allocation and free calls

## Profiling Your Code

To understand what's happening on the GPU in detail, use NVIDIA's profiling tools:

**Nsight Systems** for timeline analysis:
```bash
nsys profile --stats=true ./01-vector-addition
```

This shows you exactly where time is spent: memory transfers, kernel execution, and CPU code.

**Nsight Compute** for kernel analysis:
```bash
ncu --set full ./01-vector-addition
```

This provides detailed metrics about memory bandwidth, SM utilization, and performance bottlenecks.

## Common Issues and Solutions

**Error: "CUDA driver version is insufficient"**
Your driver is too old for your CUDA toolkit version. Update your NVIDIA driver.

**Error: "out of memory"**
The GPU doesn't have enough memory. Reduce `numElements` or process data in batches.

**Error: "invalid device function"**
The kernel was compiled for a different GPU architecture. Check your Makefile's `-arch` flag matches your GPU's compute capability.

**Segmentation fault:**
Likely passing host pointers to kernel or accessing unallocated memory. Run with `cuda-memcheck` to diagnose.

**Results incorrect:**
Check your index calculation. Add bounds checking and verify with a smaller dataset first.

## Advanced Topics Preview

Now that you understand basic CUDA programming, future tutorials will cover:

**Unified Memory:** Simplifies memory management by using a single pointer for CPU and GPU
**Streams:** Overlap computation and data transfer for better performance
**Shared Memory:** Use fast on-chip memory for thread collaboration
**Atomic Operations:** Handle race conditions when multiple threads write to the same location
**Texture Memory:** Optimize for spatial locality in image processing

## Challenge Exercises

1. **Modify the kernel** to compute C[i] = A[i] * B[i] + C[i] (multiply-add operation)

2. **Implement error checking** for all CUDA API calls using a macro:
```cuda
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)
```

3. **Add comprehensive timing** that separately measures:
   - Host to Device transfer time
   - Kernel execution time
   - Device to Host transfer time
   - Total GPU time vs CPU implementation time

4. **Experiment with vector sizes:** Try 1K, 10K, 100K, 1M, 10M elements. Plot execution time vs size. At what point does GPU become faster than CPU?

5. **Implement unified memory version:** Replace explicit cudaMalloc/cudaMemcpy with cudaMallocManaged. Does performance change?

## Summary

You've now written your first CUDA program and understand the fundamental concepts:

The CPU and GPU have separate memory spaces requiring explicit data movement. Kernels execute in parallel across thousands of threads organized into blocks. Each thread computes a unique index to determine which data element to process. Proper error checking and verification are essential for reliable GPU code.

Vector addition is memory-bound, so performance depends on memory bandwidth rather than computation speed. Good memory access patterns (coalescing) are crucial for performance. The typical workflow is: allocate, copy to device, compute, copy to host, free.

These foundations will serve you throughout GPU programming. The same patterns apply whether you're adding vectors, training neural networks, or simulating physics.

## Next Steps

Continue to **Tutorial 02: PTX Assembly** to learn how to write low-level GPU code and understand what the compiler generates. You'll see the actual assembly instructions that execute on the GPU and learn when to use inline PTX for performance-critical code.

## Further Reading

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) - Official comprehensive guide
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) - Performance optimization techniques
- [GPU Architecture Whitepaper](https://www.nvidia.com/en-us/data-center/resources/gpu-architecture/) - Understanding the hardware
- [CUDA Samples](https://github.com/NVIDIA/cuda-samples) - More example code
