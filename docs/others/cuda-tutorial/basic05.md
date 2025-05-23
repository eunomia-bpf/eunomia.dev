# Neural Network Forward Pass on GPU with CUDA

This tutorial demonstrates how to implement a basic neural network forward pass on a GPU using CUDA. Neural networks are at the core of deep learning and have revolutionized fields like computer vision, natural language processing, and reinforcement learning. GPUs are particularly well-suited for neural network computation due to their ability to perform massive parallel computations.

You can find the code in <https://github.com/eunomia-bpf/basic-cuda-tutorial>

## Table of Contents

1. [Introduction to Neural Networks on GPU](#introduction-to-neural-networks-on-gpu)
2. [Network Architecture](#network-architecture)
3. [CUDA Implementation](#cuda-implementation)
   - [Matrix Multiplication](#matrix-multiplication)
   - [Activation Functions](#activation-functions)
   - [Memory Management](#memory-management)
   - [Forward Pass Workflow](#forward-pass-workflow)
4. [Performance Considerations](#performance-considerations)
5. [Further Improvements](#further-improvements)

## Introduction to Neural Networks on GPU

Neural networks consist of layers of neurons that transform input data through a series of mathematical operations. The two primary operations in neural networks are:

1. **Linear Transformations**: Matrix multiplications followed by bias additions
2. **Non-linear Activations**: Functions like ReLU, sigmoid, or tanh that introduce non-linearity

These operations are inherently parallel, making them perfect for GPU acceleration:

- Matrix multiplications can be distributed across thousands of GPU cores
- Activation functions can be applied independently to each element
- Batch processing allows multiple samples to be processed simultaneously

GPUs can provide 10-50x speedup for neural network inference compared to CPUs, making real-time applications possible.

## Network Architecture

Our example implements a simple feedforward neural network with:

- **Input Layer**: 784 neurons (representing a 28×28 image, like MNIST digits)
- **Hidden Layer**: 128 neurons with ReLU activation
- **Output Layer**: 10 neurons with softmax activation (for 10-class classification)

The network performs a forward pass on a batch of 64 samples simultaneously.

### Mathematical Operations

For each layer, the forward pass involves:

1. **Linear transformation**: `Y = X × W + b`
   - `X`: Input matrix (batch_size × input_features)
   - `W`: Weight matrix (input_features × output_features)
   - `b`: Bias vector (output_features)
   - `Y`: Output matrix (batch_size × output_features)

2. **Activation function**:
   - Hidden layer: `ReLU(x) = max(0, x)`
   - Output layer: `Softmax(x_i) = exp(x_i) / Σ exp(x_j)`

## CUDA Implementation

### Matrix Multiplication

Matrix multiplication is the most computationally intensive operation in neural networks. Our implementation uses a straightforward CUDA kernel:

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

This kernel assigns each thread to compute one element of the output matrix. For a batch size of 64 with 128 hidden neurons, we're computing 8,192 elements in parallel.

**Note**: This implementation focuses on clarity rather than maximum performance. Production systems would use optimized libraries like cuBLAS for matrix operations.

### Activation Functions

#### ReLU Activation

The ReLU function is applied element-wise and is highly parallelizable:

```cuda
__global__ void reluKernel(float *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}
```

#### Softmax Activation

Softmax is slightly more complex as it requires normalization across all output classes:

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

The softmax implementation includes numerical stability techniques by subtracting the maximum value before exponentiation to prevent overflow.

### Memory Management

Neural networks require careful memory management to efficiently handle:

1. **Network Parameters**: Weights and biases
2. **Activations**: Input, hidden layers, and output
3. **Temporary Buffers**: Pre-activation values and gradients (for training)

Our implementation follows these steps:

1. **Allocate host memory** for network parameters and initialize them
2. **Transfer parameters to GPU memory** using `cudaMemcpy`
3. **Allocate GPU memory** for intermediate activations
4. **Perform forward pass** entirely on the GPU
5. **Transfer results back** to host memory for evaluation

```cpp
// Allocate device memory for network parameters
float *d_weights1, *d_bias1, *d_weights2, *d_bias2;
cudaMalloc(&d_weights1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
// ...

// Allocate device memory for intermediate results
float *d_hidden_preact, *d_hidden_output, *d_output_preact, *d_output;
cudaMalloc(&d_hidden_preact, BATCH_SIZE * HIDDEN_SIZE * sizeof(float));
// ...
```

### Forward Pass Workflow

The forward pass combines all operations into a sequential workflow:

```cpp
// Forward pass: input -> hidden layer
matrixMultiplyKernel<<<grid_mm2, block_mm>>>(d_input, d_weights1, d_hidden_preact, 
                                            BATCH_SIZE, INPUT_SIZE, HIDDEN_SIZE);
addBiasKernel<<<grid_bias1, block_bias>>>(d_hidden_preact, d_bias1, BATCH_SIZE, HIDDEN_SIZE);
reluKernel<<<grid_act1, block_act>>>(d_hidden_preact, BATCH_SIZE * HIDDEN_SIZE);

// Copy hidden layer activation to output for next layer
cudaMemcpy(d_hidden_output, d_hidden_preact, BATCH_SIZE * HIDDEN_SIZE * sizeof(float),
          cudaMemcpyDeviceToDevice);

// Forward pass: hidden -> output layer
matrixMultiplyKernel<<<grid_mm1, block_mm>>>(d_hidden_output, d_weights2, d_output_preact,
                                            BATCH_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
addBiasKernel<<<grid_bias2, block_bias>>>(d_output_preact, d_bias2, BATCH_SIZE, OUTPUT_SIZE);

// Apply softmax activation
softmaxKernel<<<grid_pred, block_pred>>>(d_output_preact, d_output, BATCH_SIZE, OUTPUT_SIZE);
```

Each kernel is launched with an appropriate grid and block configuration to ensure all elements are processed efficiently.

## Performance Considerations

### Kernel Launch Overhead

Each kernel launch incurs overhead. For small networks, this overhead can be significant. Techniques to mitigate this include:

1. **Kernel Fusion**: Combining multiple operations into a single kernel
2. **Persistent Kernels**: Keeping kernels running and feeding them new work
3. **CUDA Graphs**: Creating a graph of operations that can be launched together

### Memory Bandwidth

Neural networks are often memory-bound rather than compute-bound. Strategies to optimize memory usage include:

1. **Coalesced Memory Access**: Ensuring threads in a warp access adjacent memory locations
2. **Shared Memory**: Using on-chip shared memory for frequently accessed data
3. **Memory Layout**: Organizing data for better memory access patterns (e.g., NHWC vs NCHW format)

### Batch Processing

Increasing batch size generally improves GPU utilization up to a point:

- Larger batches amortize kernel launch overhead
- Matrix operations become more efficient with larger dimensions
- Too large batches can exceed available memory

The optimal batch size depends on the specific GPU and network architecture.

## Further Improvements

This implementation can be enhanced in several ways:

1. **Use Optimized Libraries**:
   - Replace custom matrix multiplication with cuBLAS
   - Use cuDNN for standard neural network operations

2. **Memory Optimization**:
   - Implement in-place operations where possible
   - Use half-precision (FP16) for inference
   - Add memory pooling for dynamic networks

3. **Advanced Features**:
   - Implement backpropagation for training
   - Add convolutional layers and pooling
   - Support recurrent and transformer architectures

4. **Multi-GPU Support**:
   - Distribute computation across multiple GPUs
   - Implement model parallelism for large networks

## Conclusion

This tutorial demonstrates the fundamental techniques for implementing neural network inference on GPUs using CUDA. While our implementation prioritizes clarity over maximum performance, it illustrates the key concepts and operations required for neural network computation.

By leveraging the massive parallelism of GPUs, even this basic implementation can achieve significant speedups compared to CPU-only execution, highlighting why GPUs have become the standard hardware for deep learning applications.

## References

- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [NVIDIA cuDNN Documentation](https://docs.nvidia.com/deeplearning/cudnn/developer-guide/)
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)
- [Deep Learning Book](https://www.deeplearningbook.org/) 