# Tutorial: CNN Convolution with Shared Memory Optimization

**Time Required:** 60-75 minutes
**Difficulty:** Intermediate to Advanced
**Prerequisites:** Completed Tutorials 04 and 05, understanding of convolutions

By the end of this tutorial, you will understand how convolutional layers work on the GPU, why shared memory optimization is crucial for performance, and how the architectural principles from Tutorial 04 translate to dramatic real-world speedups. You'll see a 30x performance improvement from a single optimization technique.

You can find the code at <https://github.com/eunomia-bpf/basic-cuda-tutorial>

## Why Convolutions Dominate Computer Vision

Convolutional Neural Networks (CNNs) have revolutionized computer vision. They power face recognition on your phone, object detection in self-driving cars, and medical image analysis. But CNNs are computationally expensive – a single forward pass through a modern CNN can require billions of multiply-add operations.

The convolution operation is the bottleneck. In a typical CNN like ResNet-50, over 95% of computation time is spent in convolutional layers. This makes convolution the most important operation to optimize when implementing CNNs on GPUs.

Unlike the fully connected layers from Tutorial 05, convolutions exploit spatial structure. Instead of connecting every input to every output, a convolution applies a small filter (kernel) that slides across the input. This sliding window operation processes local regions, which is perfect for images where nearby pixels are related.

## Running the Example

Let's start by seeing the optimization in action:

```bash
make 06-cnn-convolution
./06-cnn-convolution
```

You'll see output like this:

```
=== CNN Convolution with Shared Memory Optimization ===

Configuration:
  Input size: 28x28 with 1 channels
  Kernel size: 5x5 with 16 output channels
  Padding: 2, Stride: 1
  Output size after convolution: 28x28
  Output size after pooling: 14x14

Performance comparison:
  Direct Convolution: 0.223 ms
  Shared Memory Convolution: 0.008 ms
  Speedup: 29.68x

Layer timings (Shared Memory):
  Convolution: 0.008 ms
  ReLU Activation: 0.006 ms
  Max Pooling: 0.006 ms

Verification:
  Max difference between implementations: 0.000000e+00
  Results match: YES
```

The shared memory version is nearly 30 times faster than the direct implementation. Both produce identical results, but one leverages GPU architecture effectively while the other doesn't. This tutorial explains why.

## Understanding the Convolution Operation

A convolution applies a small filter (like 5×5) to every location in an image. Think of the filter as a pattern detector. As it slides across the image, it produces high outputs where the pattern matches and low outputs where it doesn't.

Mathematically, for each output position (x, y), we compute:

```
Output[y,x] = Σky Σkx Input[y+ky, x+kx] × Kernel[ky, kx]
```

This is a dot product between the kernel and a patch of the input. For a 28×28 output with a 5×5 kernel, we perform 28 × 28 × 5 × 5 = 19,600 dot products. Each dot product involves 25 multiply-adds. That's 490,000 operations for a single channel.

Real CNNs have multiple input channels (like RGB images with 3 channels) and multiple output channels (filters that detect different patterns). Our example uses 1 input channel and 16 output channels, multiplying the work by 16.

### Padding and Stride

Two parameters control how the convolution operates:

**Padding:** Adding zeros around the border. With padding=2, we add 2 rows/columns of zeros on each side. This allows the filter to process edge pixels and keeps the output the same size as the input.

**Stride:** How far the filter moves each step. Stride=1 means the filter moves one pixel at a time. Stride=2 would halve the output dimensions.

Our example uses padding=2 and stride=1, so the 28×28 input produces a 28×28 output.

## The Naive Implementation

Let's start with the straightforward implementation. Each thread computes one output element:

```cuda
__global__ void convolutionDirectKernel(
    float *input, float *kernels, float *output,
    int batchSize, int inputChannels, int inputSize,
    int kernelSize, int kernelCount, int outputSize,
    int padding, int stride)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z; // Output channel
    int b = threadIdx.z; // Batch index

    if (x >= outputSize || y >= outputSize || k >= kernelCount || b >= batchSize)
        return;

    float sum = 0.0f;

    for (int c = 0; c < inputChannels; c++) {
        for (int ky = 0; ky < kernelSize; ky++) {
            for (int kx = 0; kx < kernelSize; kx++) {
                int in_x = x * stride - padding + kx;
                int in_y = y * stride - padding + ky;

                if (in_x >= 0 && in_x < inputSize && in_y >= 0 && in_y < inputSize) {
                    float in_val = input[...];
                    float kernel_val = kernels[...];
                    sum += in_val * kernel_val;
                }
            }
        }
    }

    output[...] = sum;
}
```

Thread (x, y) for kernel k computes output position (x, y) for that filter. It reads a 5×5 patch from the input, multiplies element-wise with the kernel, and sums the results.

This works, but has terrible memory efficiency. Look at what happens when adjacent threads execute. Thread computing output (0, 0) reads input pixels (0,0) through (4,4). Thread computing output (0, 1) reads input pixels (0,1) through (4,5). These overlap! Pixels (0,1) through (0,4) are read twice from global memory.

For a 5×5 kernel, adjacent outputs share 4 out of 5 input columns. That's 80% redundancy. Every input pixel gets read from global memory approximately 5 times on average.

Running this gives 0.223 ms for our test case. Now let's see how shared memory eliminates this redundancy.

## Shared Memory Optimization: The Key Insight

The optimization insight is simple: adjacent threads need overlapping input data. Instead of each thread reading its own 5×5 patch from slow global memory, the entire block should cooperatively load a larger tile into fast shared memory once, then all threads read from there.

Consider a block of 8×8 threads computing 8×8 outputs. Without shared memory, we read 8×8×5×5 = 1,600 values from global memory (with lots of redundancy). With shared memory, we load a 12×12 tile once (8 + 5 - 1 = 12), then all threads read from shared memory. That's 144 global memory loads instead of 1,600 – an 11x reduction!

Here's the structure:

```cuda
__global__ void convolutionSharedKernel(...) {
    extern __shared__ float sharedData[];

    int tileSize = blockDim.x; // 8
    int tileSizeWithPadding = tileSize + kernelSize - 1; // 12

    // Phase 1: Cooperatively load input tile to shared memory
    for (int c = 0; c < inputChannels; c++) {
        // Load tileSizeWithPadding × tileSizeWithPadding elements
        // Each thread loads multiple elements to cover the tile
        ...
    }

    __syncthreads(); // Wait for all threads to finish loading

    // Phase 2: Compute convolution using shared memory
    float sum = 0.0f;
    for (int c = 0; c < inputChannels; c++) {
        for (int ky = 0; ky < kernelSize; ky++) {
            for (int kx = 0; kx < kernelSize; kx++) {
                float in_val = sharedData[...]; // Read from shared memory!
                float kernel_val = kernels[...];
                sum += in_val * kernel_val;
            }
        }
    }

    output[...] = sum;
}
```

The loading phase is tricky. We have 8×8 = 64 threads but need to load 12×12 = 144 elements. Each thread must load multiple elements. We use nested loops where each thread loads elements at positions (tx + dx, ty + dy) for various offsets.

The `__syncthreads()` barrier is crucial. It ensures all threads have finished loading before any start computing. Without it, some threads might try to read from shared memory before it's fully populated, getting garbage data.

## Memory Access Pattern Analysis

Let's quantify the improvement. For an 8×8 tile with a 5×5 kernel:

**Direct convolution:**
- Each of 64 threads reads 25 values from global memory
- Total global reads: 64 × 25 = 1,600
- Many of these are duplicates due to overlapping regions

**Shared memory convolution:**
- Cooperatively load 12×12 = 144 values to shared memory
- Total global reads: 144
- Each thread then reads 25 values from shared memory (fast!)

The reduction: 1,600 → 144 global memory accesses. That's an 11x reduction in memory traffic.

But wait – shared memory accesses aren't free. They're much faster than global memory (roughly 100x lower latency), but they can still have bank conflicts. Fortunately, our access pattern mostly avoids conflicts because threads access different elements when loading and nearby elements have sequential addresses.

The actual speedup (29.68x in our example) comes from:
1. Reduced global memory traffic (11x fewer loads)
2. Better cache utilization (fewer unique addresses)
3. Coalesced memory access during loading phase
4. Low shared memory bank conflicts

## Understanding the Implementation Details

The loading phase deserves a closer look:

```cuda
for (int dy = 0; dy < tileSizeWithPadding; dy += tileSize) {
    for (int dx = 0; dx < tileSizeWithPadding; dx += tileSize) {
        int in_y = in_y_base + ty + dy;
        int in_x = in_x_base + tx + dx;

        float value = 0.0f;
        if (in_y >= 0 && in_y < inputSize && in_x >= 0 && in_x < inputSize) {
            value = input[...];
        }

        if (ty + dy < tileSizeWithPadding && tx + dx < tileSizeWithPadding) {
            sharedInput[...] = value;
        }
    }
}
```

We iterate in increments of `tileSize` (8). In the first iteration (dy=0, dx=0), thread (tx, ty) loads element (tx, ty) of the tile. In the next iteration (dy=0, dx=8), the same thread loads element (tx, ty+8), and so on.

For a 12×12 tile with 8×8 threads, we need two passes in each dimension. The first pass covers positions 0-7, the second covers 8-11. Not all threads participate in the second pass since we only need 4 more elements in each dimension.

The bounds checking handles padding. When computing output near the image edge, our tile extends beyond the input boundaries. We load zeros for out-of-bounds positions, implementing zero-padding automatically.

## Activation and Pooling Layers

After convolution, CNNs typically apply an activation function. ReLU (Rectified Linear Unit) is standard:

```cuda
__global__ void reluActivationKernel(float *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}
```

This is applied element-wise and takes only 0.006 ms – much faster than convolution. It's memory-bound: the GPU spends more time reading and writing data than computing the max.

Pooling reduces spatial dimensions. Max pooling takes the maximum value in each window:

```cuda
__global__ void maxPoolingKernel(
    float *input, float *output,
    int batchSize, int channels, int inputSize,
    int poolSize, int outputSize, int stride)
{
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    int in_x_base = out_x * stride;
    int in_y_base = out_y * stride;

    float maxVal = -FLT_MAX;
    for (int dy = 0; dy < poolSize; dy++) {
        for (int dx = 0; dx < poolSize; dx++) {
            float value = input[...];
            maxVal = fmaxf(maxVal, value);
        }
    }

    output[...] = maxVal;
}
```

Our example uses 2×2 pooling with stride 2, reducing the 28×28 feature maps to 14×14. This halves the spatial dimensions, reducing computation in subsequent layers.

Pooling could also benefit from shared memory, but for 2×2 windows, the overhead isn't worthwhile. The performance is already good at 0.006 ms.

## Complete CNN Layer Pipeline

Putting it together, a typical CNN layer executes:

1. Convolution: 0.008 ms (with shared memory)
2. ReLU activation: 0.006 ms
3. Max pooling: 0.006 ms

Total: 0.020 ms per layer for processing 64 images. That's about 0.3 microseconds per image per layer. A 50-layer CNN could process images at ~60 microseconds per image on computation alone (memory transfer adds overhead).

Modern CNNs like ResNet have skip connections and batch normalization, adding complexity. But the convolution remains the bottleneck, making our optimization critical.

## Why Shared Memory Matters So Much

The 30x speedup might seem too good to be true. Let's understand what's happening at the hardware level.

From Tutorial 04, we know that global memory access takes hundreds of cycles. Shared memory access takes just a few cycles. When we read the same input pixel 5 times from global memory (direct method), we wait for memory 5 times. When we read it once from global memory into shared memory, then 5 times from shared memory (optimized method), we only wait for global memory once.

Additionally, the direct method has poor coalescing. When 32 threads in a warp each read their own 5×5 patch, they access scattered memory locations. The optimized method loads a contiguous tile, giving perfect coalescing during the load phase.

Finally, the direct method may exceed the GPU's cache capacity. The L2 cache on the RTX 5090 is 98 MB, but a batch of images can be hundreds of megabytes. The shared memory explicitly manages a working set that fits on-chip, guaranteeing fast access.

## Comparing to cuDNN

Our optimized implementation is good, but production code uses cuDNN. How do they compare?

**Our implementation:**
- Shared memory tiling: 0.008 ms
- Simple to understand
- Works for any kernel size

**cuDNN implementation:**
- Multiple algorithms (GEMM, Winograd, FFT)
- Algorithm selection based on problem size
- Estimated ~0.002 ms for our test case
- Highly optimized, architecture-specific

cuDNN might be 4x faster because it uses additional techniques:
- Winograd transformation for 3×3 and 5×5 kernels (fewer multiplications)
- Im2col + GEMM for leveraging highly tuned matrix multiplication
- Tensor Core utilization on newer GPUs
- Kernel fusion to combine convolution, bias, and activation

But our 30x speedup from shared memory alone captures most of the possible optimization. The remaining 4x requires algorithm-level changes, not just better memory management.

## Memory Bandwidth Analysis

Let's calculate our memory bandwidth. For the shared memory version:

**Convolution input read:** 64 × 1 × 28 × 28 × 4 bytes = 200 KB
**Kernel read:** 16 × 1 × 5 × 5 × 4 bytes = 1.6 KB
**Output write:** 64 × 16 × 28 × 28 × 4 bytes = 3.2 MB
**Total:** 3.4 MB

In 0.008 ms, that's 3.4 MB / 0.000008 s = 425 GB/s.

The RTX 5090 has 1792 GB/s theoretical bandwidth. We're achieving 24% of peak – much better than the 0.2% we saw in Tutorial 05's naive matrix multiplication!

The direct convolution achieves only 3.4 MB / 0.223 ms = 15 GB/s (0.8% of peak) because of redundant reads and poor coalescing.

## Practical Implications

When building CNN applications, these lessons apply:

**Use cuDNN for standard operations.** Our manual optimization is educational, but cuDNN is faster and handles edge cases we haven't covered.

**Understand why optimizations work.** If you hit a performance problem cuDNN can't solve (custom layers, novel architectures), knowing these principles lets you write efficient custom kernels.

**Profile before optimizing.** The 30x speedup here is huge, but it only matters if convolution is your bottleneck. Always profile to find the real bottleneck.

**Consider memory first.** Most GPU kernels are memory-bound. Reducing memory traffic usually helps more than optimizing computation.

## Challenge Exercises

1. **Measure bank conflicts:** Add a counter for shared memory bank conflicts using PTX assembly (from Tutorial 02). Does our access pattern avoid conflicts?

2. **Implement 3×3 convolution:** Modify the code for a 3×3 kernel. Does the speedup change? Why?

3. **Add channel fusion:** Modify the shared memory kernel to load all input channels at once. Does this improve performance?

4. **Experiment with tile sizes:** Try 4×4, 16×16, and 32×32 tiles. Plot performance vs tile size. What's optimal?

5. **Implement depthwise separable convolution:** This efficient variant separates spatial and channel-wise filtering. Implement it and compare to standard convolution.

## Summary

Convolutional layers are the computational heart of CNNs, making their optimization critical. The naive implementation repeatedly reads the same data from slow global memory due to overlapping receptive fields.

Shared memory tiling eliminates this redundancy by loading each input pixel once into fast on-chip memory. This dramatically reduces memory traffic and improves coalescing, achieving 30x speedup in our example.

The principles apply beyond convolution. Any sliding window operation benefits from tiling: correlation, morphological operations, temporal convolution in audio processing, and 3D convolution in video understanding.

Understanding these optimizations reveals why GPUs dominate deep learning. The massive parallelism handles billions of multiply-adds. The memory hierarchy enables reuse patterns that make these operations feasible. Without both, modern AI would be impractical.

## Next Steps

Continue to **Tutorial 07: Attention Mechanism** to learn how transformer networks process sequences on the GPU. You'll implement the self-attention operation that powers models like GPT and BERT, and see how memory access patterns differ from convolution.

## Further Reading

- [NVIDIA cuDNN Documentation](https://docs.nvidia.com/deeplearning/cudnn/developer-guide/)
- [Optimizing Parallel Reduction in CUDA](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)
- [Fast Algorithms for Convolutional Neural Networks](https://arxiv.org/abs/1509.09308) - Winograd convolution
- [Im2col and GEMM for Convolution](https://sahnimanas.github.io/post/anatomy-of-a-high-performance-convolution/)
