# 使用CUDA在GPU上实现神经网络前向传播

本教程演示了如何使用CUDA在GPU上实现基本神经网络前向传播。神经网络是深度学习的核心，已经彻底改变了计算机视觉、自然语言处理和强化学习等领域。由于GPU能够执行大规模并行计算，它们特别适合神经网络计算。

您可以在 <https://github.com/eunomia-bpf/basic-cuda-tutorial> 找到代码

## 目录

1. [GPU上的神经网络简介](#gpu上的神经网络简介)
2. [网络架构](#网络架构)
3. [CUDA实现](#cuda实现)
   - [矩阵乘法](#矩阵乘法)
   - [激活函数](#激活函数)
   - [内存管理](#内存管理)
   - [前向传播工作流程](#前向传播工作流程)
4. [性能考虑因素](#性能考虑因素)
5. [进一步改进](#进一步改进)

## GPU上的神经网络简介

神经网络由通过一系列数学运算转换输入数据的神经元层组成。神经网络中的两个主要操作是：

1. **线性变换**：矩阵乘法后接偏置加法
2. **非线性激活**：像ReLU、sigmoid或tanh这样引入非线性的函数

这些操作本质上是并行的，使其非常适合GPU加速：

- 矩阵乘法可以分布在数千个GPU核心上
- 激活函数可以独立应用于每个元素
- 批处理允许同时处理多个样本

与CPU相比，GPU可以为神经网络推理提供10-50倍的加速，使实时应用成为可能。

## 网络架构

我们的示例实现了一个简单的前馈神经网络，包含：

- **输入层**：784个神经元（代表28×28的图像，如MNIST数字）
- **隐藏层**：128个神经元，使用ReLU激活函数
- **输出层**：10个神经元，使用softmax激活（用于10类分类）

该网络同时对64个样本的批次执行前向传播。

### 数学运算

对于每一层，前向传播包括：

1. **线性变换**：`Y = X × W + b`
   - `X`：输入矩阵（batch_size × input_features）
   - `W`：权重矩阵（input_features × output_features）
   - `b`：偏置向量（output_features）
   - `Y`：输出矩阵（batch_size × output_features）

2. **激活函数**：
   - 隐藏层：`ReLU(x) = max(0, x)`
   - 输出层：`Softmax(x_i) = exp(x_i) / Σ exp(x_j)`

## CUDA实现

### 矩阵乘法

矩阵乘法是神经网络中计算强度最高的操作。我们的实现使用了一个直接的CUDA内核：

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

这个内核让每个线程计算输出矩阵的一个元素。对于批量大小为64，隐藏神经元128个的情况，我们同时计算8,192个元素。

**注意**：这个实现注重清晰度而非最大性能。生产系统会使用优化库如cuBLAS进行矩阵运算。

### 激活函数

#### ReLU激活

ReLU函数按元素应用，高度可并行化：

```cuda
__global__ void reluKernel(float *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}
```

#### Softmax激活

Softmax稍微复杂一些，因为它需要在所有输出类别上进行归一化：

```cuda
__global__ void softmaxKernel(float *input, float *output, int batch_size, int num_classes) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size) {
        // 找到最大值以确保数值稳定性
        float max_val = -FLT_MAX;
        for (int i = 0; i < num_classes; i++) {
            max_val = fmaxf(max_val, input[batch_idx * num_classes + i]);
        }
        
        // 计算指数和总和
        float sum = 0.0f;
        for (int i = 0; i < num_classes; i++) {
            output[batch_idx * num_classes + i] = expf(input[batch_idx * num_classes + i] - max_val);
            sum += output[batch_idx * num_classes + i];
        }
        
        // 归一化
        for (int i = 0; i < num_classes; i++) {
            output[batch_idx * num_classes + i] /= sum;
        }
    }
}
```

Softmax实现包括数值稳定性技术，在指数运算前减去最大值以防止溢出。

### 内存管理

神经网络需要仔细的内存管理，以高效处理：

1. **网络参数**：权重和偏置
2. **激活值**：输入、隐藏层和输出
3. **临时缓冲区**：预激活值和梯度（用于训练）

我们的实现遵循这些步骤：

1. **分配主机内存**用于网络参数并初始化它们
2. **将参数传输到GPU内存**使用`cudaMemcpy`
3. **分配GPU内存**用于中间激活值
4. **执行前向传播**完全在GPU上进行
5. **将结果传回**主机内存进行评估

```cpp
// 为网络参数分配设备内存
float *d_weights1, *d_bias1, *d_weights2, *d_bias2;
cudaMalloc(&d_weights1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
// ...

// 为中间结果分配设备内存
float *d_hidden_preact, *d_hidden_output, *d_output_preact, *d_output;
cudaMalloc(&d_hidden_preact, BATCH_SIZE * HIDDEN_SIZE * sizeof(float));
// ...
```

### 前向传播工作流程

前向传播将所有操作组合成一个顺序工作流：

```cpp
// 前向传播：输入 -> 隐藏层
matrixMultiplyKernel<<<grid_mm2, block_mm>>>(d_input, d_weights1, d_hidden_preact, 
                                            BATCH_SIZE, INPUT_SIZE, HIDDEN_SIZE);
addBiasKernel<<<grid_bias1, block_bias>>>(d_hidden_preact, d_bias1, BATCH_SIZE, HIDDEN_SIZE);
reluKernel<<<grid_act1, block_act>>>(d_hidden_preact, BATCH_SIZE * HIDDEN_SIZE);

// 将隐藏层激活复制到输出，用于下一层
cudaMemcpy(d_hidden_output, d_hidden_preact, BATCH_SIZE * HIDDEN_SIZE * sizeof(float),
          cudaMemcpyDeviceToDevice);

// 前向传播：隐藏层 -> 输出层
matrixMultiplyKernel<<<grid_mm1, block_mm>>>(d_hidden_output, d_weights2, d_output_preact,
                                            BATCH_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
addBiasKernel<<<grid_bias2, block_bias>>>(d_output_preact, d_bias2, BATCH_SIZE, OUTPUT_SIZE);

// 应用softmax激活
softmaxKernel<<<grid_pred, block_pred>>>(d_output_preact, d_output, BATCH_SIZE, OUTPUT_SIZE);
```

每个内核都以适当的网格和块配置启动，以确保所有元素都被高效处理。

## 性能考虑因素

### 内核启动开销

每次内核启动都会产生开销。对于小型网络，这种开销可能很显著。减轻这种开销的技术包括：

1. **内核融合**：将多个操作合并到一个内核中
2. **持久内核**：保持内核运行并向其提供新工作
3. **CUDA图**：创建可一起启动的操作图

### 内存带宽

神经网络通常受内存限制而非计算限制。优化内存使用的策略包括：

1. **合并内存访问**：确保线程束中的线程访问相邻的内存位置
2. **共享内存**：为频繁访问的数据使用片上共享内存
3. **内存布局**：组织数据以获得更好的内存访问模式（例如NHWC与NCHW格式）

### 批处理

增加批量大小通常会提高GPU利用率，但有一定限度：

- 更大的批量可以分摊内核启动开销
- 更大维度的矩阵操作更有效
- 太大的批量可能超出可用内存

最佳批量大小取决于特定的GPU和网络架构。

## 进一步改进

这个实现可以通过多种方式增强：

1. **使用优化库**：
   - 用cuBLAS替换自定义矩阵乘法
   - 使用cuDNN进行标准神经网络操作

2. **内存优化**：
   - 尽可能实现原地操作
   - 使用半精度（FP16）进行推理
   - 为动态网络添加内存池

3. **高级功能**：
   - 实现反向传播进行训练
   - 添加卷积层和池化层
   - 支持循环和transformer架构

4. **多GPU支持**：
   - 在多个GPU上分布计算
   - 为大型网络实现模型并行

## 结论

本教程演示了使用CUDA在GPU上实现神经网络推理的基本技术。虽然我们的实现优先考虑清晰度而非最大性能，但它说明了神经网络计算所需的关键概念和操作。

通过利用GPU的大规模并行性，即使这个基本实现也可以与仅CPU执行相比获得显著的加速，强调了为什么GPU已成为深度学习应用的标准硬件。

## 参考资料

- [NVIDIA CUDA编程指南](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [NVIDIA cuDNN文档](https://docs.nvidia.com/deeplearning/cudnn/developer-guide/)
- [神经网络与深度学习](http://neuralnetworksanddeeplearning.com/)
- [深度学习书籍](https://www.deeplearningbook.org/) 