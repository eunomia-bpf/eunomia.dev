# Tutorial: Neural Network Forward Pass on GPU

**Time Required:** 45-60 minutes
**Difficulty:** Intermediate
**Prerequisites:** Completed Tutorials 01 and 04, basic understanding of neural networks

By the end of this tutorial, you will understand how to implement a complete neural network forward pass on the GPU. You'll learn why neural networks are perfect for GPU acceleration, how to implement matrix multiplications and activation functions efficiently, and how architectural understanding from Tutorial 04 applies to real machine learning workloads.

You can find the code at <https://github.com/eunomia-bpf/basic-cuda-tutorial>

## Why Neural Networks on GPUs

Neural networks have revolutionized artificial intelligence, powering everything from image recognition to language translation. But modern networks can have billions of parameters and require trillions of operations for a single forward pass. Without GPUs, training or even running these networks would be impossibly slow.

Consider what happens in a neural network: at each layer, you multiply an input matrix by a weight matrix, add biases, and apply an activation function. For a batch of 64 images (28x28 pixels each) passing through a layer with 128 neurons, you need to perform 64 × 784 × 128 = 6,422,528 multiply-add operations. On a CPU processing one operation at a time, this takes milliseconds. On a GPU with thousands of cores working in parallel, it takes microseconds.

The operations in neural networks are almost embarrassingly parallel. Each output neuron's computation is independent of the others. This makes neural networks an ideal fit for GPU architecture.

## Building a Simple Network

Let's build and run a complete neural network on the GPU. Our network will be simple but representative of real neural networks:

```bash
make 05-neural-network
./05-neural-network
```

You'll see output like this:

```
=== Neural Network Forward Pass Example ===

Network configuration:
  Input size: 784
  Hidden layer size: 128
  Output size: 10
  Batch size: 64

Forward pass completed in 0.328 ms

Example results (first 5 samples):
Sample 0 - True label: 3, Predicted: 1
  Probabilities: 0.0934 0.1879 0.0828 0.0640 0.0906 0.0844 0.1827 0.0417 0.0938 0.0788
Sample 1 - True label: 1, Predicted: 1
  Probabilities: 0.0955 0.1514 0.1014 0.0777 0.0916 0.0864 0.1401 0.0761 0.1023 0.0775

Batch accuracy: 1.56%
```

The network processed 64 images in 0.328 milliseconds. That's about 5 microseconds per image. The low accuracy is expected because we're using random weights – the network hasn't been trained. But the speed is what matters for this tutorial.

## Network Architecture

Our network has three layers:

**Input layer:** 784 neurons (representing a 28×28 pixel image, like MNIST handwritten digits)

**Hidden layer:** 128 neurons with ReLU activation

**Output layer:** 10 neurons with softmax activation (for classifying digits 0-9)

This architecture is small by modern standards, but it contains all the key components you'd find in much larger networks. Understanding how to implement this efficiently teaches you principles that scale to networks with millions of parameters.

The forward pass transforms input through these layers: input → linear transformation → ReLU → linear transformation → softmax → output probabilities. Each digit gets a probability score, and the highest score is the prediction.

## Matrix Multiplication: The Core Operation

At the heart of neural networks is matrix multiplication. When you pass data through a layer, you're computing Y = X × W + b, where X is your input, W is the weight matrix, and b is a bias vector.

Here's our matrix multiplication kernel:

```cuda
__global__ void matrixMultiplyKernel(float *A, float *B, float *C,
                                     int A_rows, int A_cols, int B_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < A_rows && col < B_cols) {
        float sum = 0.0f;
        for (int k = 0; k < A_cols; k++) {
            sum += A[row * A_cols + k] * B[k * B_cols + col];
        }
        C[row * B_cols + col] = sum;
    }
}
```

Each thread computes one element of the output matrix. For our hidden layer with batch size 64, we launch 64 × 128 = 8,192 threads to compute all outputs in parallel. Thread (row, col) computes the dot product of row `row` from A with column `col` from B.

This is not the fastest possible matrix multiplication. Production code would use cuBLAS, NVIDIA's optimized library that uses shared memory tiling, register blocking, and other advanced techniques. But our simple version is clear and already much faster than a CPU implementation.

The memory access pattern here is worth examining. Each thread reads an entire row of A and an entire column of B. Threads in the same block read the same row of A (good for caching) but access B in a strided pattern (not ideal). Tutorial 06 will show you how to optimize this using shared memory.

## Activation Functions: Adding Nonlinearity

After each linear transformation, we apply an activation function. Without activation functions, stacking multiple layers would be pointless – the composition of linear functions is just another linear function.

### ReLU: The Workhorse of Deep Learning

ReLU (Rectified Linear Unit) is elegantly simple: it outputs the input if positive, otherwise zero. Mathematically, ReLU(x) = max(0, x).

```cuda
__global__ void reluKernel(float *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}
```

Each thread processes one element independently. For our hidden layer (64 samples × 128 neurons = 8,192 elements), we can launch 8,192 threads that all execute in parallel. The operation is memory-bound: the GPU spends more time loading and storing data than computing the max.

ReLU is applied in-place, meaning we modify the data directly rather than creating a new array. This saves memory and bandwidth.

### Softmax: Producing Probabilities

The output layer uses softmax, which converts raw scores into probabilities that sum to 1. For a vector x, softmax(x_i) = exp(x_i) / Σ exp(x_j).

```cuda
__global__ void softmaxKernel(float *input, float *output, int batch_size, int num_classes) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < batch_size) {
        // Find max value for numerical stability
        float max_val = -FLT_MAX;
        for (int i = 0; i < num_classes; i++) {
            max_val = fmaxf(max_val, input[batch_idx * num_classes + i]);
        }

        // Compute exponentials and sum
        float sum = 0.0f;
        for (int i = 0; i < num_classes; i++) {
            output[batch_idx * num_classes + i] = expf(input[batch_idx * num_classes + i] - max_val);
            sum += output[batch_idx * num_classes + i];
        }

        // Normalize
        for (int i = 0; i < num_classes; i++) {
            output[batch_idx * num_classes + i] /= sum;
        }
    }
}
```

Each thread handles one sample in the batch. The thread reads 10 values (one per class), computes exponentials, sums them, and normalizes. This is more complex than ReLU because it requires coordination across all classes.

Notice the numerical stability trick: we subtract the maximum value before exponentiation. This prevents overflow when dealing with large values. Without this, exp(1000) would overflow to infinity, making the entire computation invalid.

Unlike ReLU, softmax doesn't parallelize across output classes – each sample is processed by a single thread. This is because we need to sum across all classes for normalization. Parallelizing within softmax would require synchronization overhead that isn't worthwhile for only 10 classes.

## Memory Layout and Data Flow

Understanding memory flow is crucial for neural network performance. Let's trace what happens to data as it moves through our network.

First, we initialize weights on the CPU using Xavier/Glorot initialization. This technique sets initial weights to have appropriate variance, which helps training converge:

```cuda
float weight1_scale = sqrtf(6.0f / (INPUT_SIZE + HIDDEN_SIZE));
for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; i++) {
    weights1[i] = (2.0f * (float)rand() / RAND_MAX - 1.0f) * weight1_scale;
}
```

Then we transfer these weights to GPU memory:

```cuda
cudaMemcpy(d_weights1, h_weights1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float),
           cudaMemcpyHostToDevice);
```

For our network, this transfers:
- Layer 1 weights: 784 × 128 × 4 bytes = 401 KB
- Layer 1 biases: 128 × 4 bytes = 512 bytes
- Layer 2 weights: 128 × 10 × 4 bytes = 5 KB
- Layer 2 biases: 10 × 4 bytes = 40 bytes

Total parameters: about 406 KB. Modern networks can have billions of parameters (gigabytes), but the principles are the same.

Once on the GPU, we also allocate memory for intermediate activations:

```cuda
cudaMalloc(&d_hidden_preact, BATCH_SIZE * HIDDEN_SIZE * sizeof(float));  // 64 × 128 × 4 = 32 KB
cudaMalloc(&d_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float));          // 64 × 10 × 4 = 2.5 KB
```

These activations are temporary – we only need them during the forward pass. In training, we'd also need to keep them for the backward pass.

## The Complete Forward Pass

Now let's put it all together. The forward pass executes as a sequence of kernel launches:

```cuda
// Layer 1: Input → Hidden
matrixMultiplyKernel<<<grid_mm2, block_mm>>>(d_input, d_weights1, d_hidden_preact,
                                            BATCH_SIZE, INPUT_SIZE, HIDDEN_SIZE);
addBiasKernel<<<grid_bias1, block_bias>>>(d_hidden_preact, d_bias1, BATCH_SIZE, HIDDEN_SIZE);
reluKernel<<<grid_act1, block_act>>>(d_hidden_preact, BATCH_SIZE * HIDDEN_SIZE);

// Copy activation for next layer
cudaMemcpy(d_hidden_output, d_hidden_preact, BATCH_SIZE * HIDDEN_SIZE * sizeof(float),
          cudaMemcpyDeviceToDevice);

// Layer 2: Hidden → Output
matrixMultiplyKernel<<<grid_mm1, block_mm>>>(d_hidden_output, d_weights2, d_output_preact,
                                            BATCH_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
addBiasKernel<<<grid_bias2, block_bias>>>(d_output_preact, d_bias2, BATCH_SIZE, OUTPUT_SIZE);

// Softmax activation
softmaxKernel<<<grid_pred, block_pred>>>(d_output_preact, d_output, BATCH_SIZE, OUTPUT_SIZE);
```

Each kernel launch is asynchronous. The CPU issues the launch command and immediately continues to the next line. The kernels execute on the GPU in order because they use the default CUDA stream.

The `cudaMemcpy` from device to device is unfortunate – we're copying data that's already on the GPU. A more efficient implementation would reuse buffers or fuse operations to avoid this copy. But for clarity, we keep the operations separate.

Timing the entire forward pass gives us 0.328 ms. Let's analyze this:

- Matrix multiply layer 1: Most expensive (64 × 784 × 128 operations)
- ReLU layer 1: Memory-bound, very fast
- Matrix multiply layer 2: Smaller (64 × 128 × 10 operations)
- Softmax: Minimal time (only 64 samples × 10 classes)

The matrix multiplications dominate. This is typical of neural networks – most time is spent in linear transformations.

## Batch Processing: Amortizing Overhead

Notice we process 64 images at once, not one at a time. This batch processing is crucial for GPU efficiency.

Each kernel launch has overhead – the CPU must communicate with the GPU, and the GPU must schedule threads. For a single image, this overhead might dominate the actual computation time. But when processing 64 images together, the overhead is amortized across all images.

Additionally, larger matrix dimensions make better use of the GPU. A 64 × 784 matrix multiplication can keep more streaming multiprocessors busy than a 1 × 784 multiplication.

Try modifying the batch size in the code and observing the effect on throughput:

- Batch size 1: ~0.15 ms per batch = 0.15 ms per image
- Batch size 64: ~0.33 ms per batch = 0.005 ms per image

The per-image time drops by 30x when batching. The larger batch doesn't take 64x longer because the GPU processes images in parallel.

There's a limit, of course. If you increase the batch size too much, you'll run out of GPU memory. The optimal batch size depends on your network architecture and GPU capacity.

## Memory Bandwidth Analysis

Let's calculate how much data we're moving. For the first layer:

**Input:** 64 × 784 × 4 bytes = 200 KB (read)
**Weights:** 784 × 128 × 4 bytes = 401 KB (read)
**Output:** 64 × 128 × 4 bytes = 32 KB (write)
**Total:** 633 KB

If the layer takes 0.2 ms, our bandwidth is 633 KB / 0.0002 s = 3.17 GB/s.

Compare this to the RTX 5090's theoretical 1792 GB/s bandwidth. We're achieving less than 0.2% of peak bandwidth. This is because our naive matrix multiplication doesn't optimize memory access patterns. Each thread reads data independently with poor cache reuse.

This is where libraries like cuBLAS excel. An optimized matrix multiplication uses shared memory to cache tiles of the input matrices, dramatically reducing global memory traffic. Tutorial 06 demonstrates these techniques for convolution operations.

## Weight Initialization Matters

Notice we use Xavier/Glorot initialization for weights:

```cuda
float weight1_scale = sqrtf(6.0f / (INPUT_SIZE + HIDDEN_SIZE));
weights1[i] = (2.0f * (float)rand() / RAND_MAX - 1.0f) * weight1_scale;
```

Why this specific formula? Neural network training involves gradients flowing backward through layers. If initial weights are too large, gradients explode. If too small, gradients vanish. Xavier initialization sets the scale to preserve gradient variance across layers.

The factor sqrt(6 / (n_in + n_out)) comes from analyzing variance propagation in networks with tanh activation. For ReLU networks, He initialization (sqrt(2 / n_in)) is theoretically better, but Xavier works reasonably well for both.

Even though our network isn't being trained, using proper initialization gives us reasonable initial predictions instead of completely random output.

## Numerical Stability in Softmax

Look again at the softmax implementation. Why do we subtract the maximum value?

```cuda
for (int i = 0; i < num_classes; i++) {
    output[i] = expf(input[i] - max_val);
    sum += output[i];
}
```

Consider what happens with raw inputs. If input[i] = 100, then exp(100) ≈ 2.7 × 10^43, which overflows float precision to infinity. The softmax would output NaN.

By subtracting the maximum, we ensure all inputs to exp() are ≤ 0. The largest value becomes exp(0) = 1, and all others are smaller. This makes overflow impossible while producing mathematically equivalent results (since softmax is translation-invariant).

These kinds of numerical tricks are essential when implementing neural networks from scratch. Libraries like PyTorch handle them automatically, but understanding them helps you debug when things go wrong.

## Comparing to cuDNN

Our implementation is educational, but production code would use cuDNN (CUDA Deep Neural Network library). Let's compare:

**Our implementation:**
- Simple matrix multiplication: ~0.3 ms for forward pass
- Memory bandwidth: ~3 GB/s (0.2% of peak)
- Code complexity: 200 lines

**cuDNN implementation:**
- Optimized convolutions and matrix multiplications: ~0.05 ms for forward pass
- Memory bandwidth: ~200 GB/s (11% of peak)
- Code complexity: 10 lines (calling library functions)

cuDNN is 6x faster because it uses:
- Tiled matrix multiplication with shared memory
- Tensor Core acceleration (on supported GPUs)
- Kernel fusion to combine operations
- Optimized memory layouts

But understanding our implementation helps you know what cuDNN is doing under the hood, which is valuable when optimizing performance or debugging issues.

## Challenge Exercises

1. **Measure layer timing:** Modify the code to time each layer separately using CUDA events. Which layer takes the most time? Does it match your expectation based on operation counts?

2. **Optimize memory:** The current code copies activations between layers. Modify it to reuse buffers and eliminate the `cudaMemcpy` device-to-device copy.

3. **Add dropout:** Implement a dropout layer that randomly sets activations to zero with probability p. Use `curand_kernel.h` to generate random numbers on the GPU.

4. **Batch size experiment:** Write a script that runs the forward pass with batch sizes 1, 2, 4, 8, 16, 32, 64, 128. Plot time per image vs batch size. Where does it plateau?

5. **Matrix multiplication optimization:** Implement a tiled matrix multiplication using shared memory (see Tutorial 04). Compare performance to the naive version.

## Summary

Neural networks are ideally suited for GPU acceleration because their operations are massively parallel. Matrix multiplications, the core operation, can be distributed across thousands of threads. Activation functions apply independently to each element.

Batch processing is essential for GPU efficiency. Processing multiple samples simultaneously amortizes kernel launch overhead and improves memory bandwidth utilization. Our simple network achieves 0.005 ms per image when batching 64 images together.

Memory bandwidth is often the bottleneck in neural networks. Naive implementations achieve only a small fraction of peak bandwidth. Optimized libraries like cuBLAS and cuDNN use shared memory tiling and other techniques to dramatically improve bandwidth utilization.

Understanding these fundamentals prepares you for more complex architectures. Convolutional networks, recurrent networks, and transformers all build on these same basic operations: matrix multiplications and element-wise activations.

## Next Steps

Continue to **Tutorial 06: CNN Convolution Operations** to learn how convolutional layers work on the GPU. You'll see how shared memory tiling optimizes the sliding window operations that make CNNs effective for image processing, and understand why convolutions are more memory-efficient than fully connected layers for spatial data.

## Further Reading

- [NVIDIA cuDNN Documentation](https://docs.nvidia.com/deeplearning/cudnn/developer-guide/)
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)
- [Efficient Processing of Deep Neural Networks](https://arxiv.org/abs/2002.03360)
- [Matrix Multiplication on CUDA](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)
