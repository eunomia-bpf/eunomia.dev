# 教程：GPU 上的神经网络前向传播

**所需时间：** 45-60 分钟
**难度：** 中级
**前置要求：** 完成教程 01 和 04，基本了解神经网络

完成本教程后，你将理解如何在 GPU 上实现完整的神经网络前向传播。你将学习为什么神经网络非常适合 GPU 加速、如何高效实现矩阵乘法和激活函数，以及如何将教程 04 中的架构理解应用于真实的机器学习工作负载。

代码仓库：<https://github.com/eunomia-bpf/basic-cuda-tutorial>

## 为什么在 GPU 上运行神经网络

神经网络彻底改变了人工智能，为从图像识别到语言翻译的一切提供动力。但现代网络可能有数十亿个参数，单次前向传播需要数万亿次操作。没有 GPU，训练甚至运行这些网络都会慢得令人无法接受。

考虑神经网络中发生的事情：在每一层，你将输入矩阵乘以权重矩阵，加上偏置,然后应用激活函数。对于一批 64 张图像（每张 28x28 像素）通过一个有 128 个神经元的层，你需要执行 64 × 784 × 128 = 6,422,528 次乘加操作。在一次处理一个操作的 CPU 上,这需要毫秒级时间。在有数千个核心并行工作的 GPU 上,这只需要微秒级时间。

神经网络中的操作几乎是令人尴尬地并行的。每个输出神经元的计算都独立于其他神经元。这使得神经网络非常适合 GPU 架构。

## 构建一个简单的网络

让我们在 GPU 上构建并运行一个完整的神经网络。我们的网络很简单，但代表了真实的神经网络：

```bash
make 05-neural-network
./05-neural-network
```

你会看到类似这样的输出：

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

网络在 0.328 毫秒内处理了 64 张图像。每张图像大约 5 微秒。低准确率是预期的，因为我们使用的是随机权重——网络还没有经过训练。但速度才是本教程关注的重点。

## 网络架构

我们的网络有三层：

**输入层：** 784 个神经元（代表 28×28 像素的图像，如 MNIST 手写数字）

**隐藏层：** 128 个神经元，使用 ReLU 激活

**输出层：** 10 个神经元，使用 softmax 激活（用于分类 0-9 的数字）

按现代标准，这个架构很小，但它包含了你在更大网络中能找到的所有关键组件。理解如何高效实现它，教会你可以扩展到有数百万参数的网络的原则。

前向传播通过这些层转换输入：输入 → 线性变换 → ReLU → 线性变换 → softmax → 输出概率。每个数字得到一个概率分数，最高分数就是预测结果。

## 矩阵乘法：核心操作

神经网络的核心是矩阵乘法。当你通过一层传递数据时，你在计算 Y = X × W + b，其中 X 是输入，W 是权重矩阵，b 是偏置向量。

这是我们的矩阵乘法内核：

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

每个线程计算输出矩阵的一个元素。对于批量大小为 64 的隐藏层，我们启动 64 × 128 = 8,192 个线程来并行计算所有输出。线程 (row, col) 计算 A 的第 `row` 行与 B 的第 `col` 列的点积。

这不是最快的矩阵乘法。生产代码会使用 cuBLAS，NVIDIA 的优化库，它使用共享内存分块、寄存器分块和其他高级技术。但我们的简单版本清晰明了，而且已经比 CPU 实现快得多。

这里的内存访问模式值得研究。每个线程读取 A 的整行和 B 的整列。同一块中的线程读取 A 的同一行（有利于缓存），但以跨步模式访问 B（不理想）。教程 06 将向你展示如何使用共享内存来优化这一点。

## 激活函数：增加非线性

在每次线性变换后，我们应用一个激活函数。没有激活函数，堆叠多层将毫无意义——线性函数的组合仍然只是另一个线性函数。

### ReLU：深度学习的主力

ReLU（Rectified Linear Unit，修正线性单元）优雅地简单：如果输入为正则输出输入，否则输出零。数学上，ReLU(x) = max(0, x)。

```cuda
__global__ void reluKernel(float *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}
```

每个线程独立处理一个元素。对于我们的隐藏层（64 个样本 × 128 个神经元 = 8,192 个元素），我们可以启动 8,192 个并行执行的线程。该操作是内存受限的：GPU 花在加载和存储数据上的时间比计算 max 的时间多。

ReLU 就地应用，意味着我们直接修改数据，而不是创建新数组。这节省了内存和带宽。

### Softmax：产生概率

输出层使用 softmax，它将原始分数转换为总和为 1 的概率。对于向量 x，softmax(x_i) = exp(x_i) / Σ exp(x_j)。

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

每个线程处理批次中的一个样本。线程读取 10 个值（每个类别一个），计算指数，求和，然后归一化。这比 ReLU 更复杂，因为它需要跨所有类别协调。

注意数值稳定性技巧：我们在指数运算前减去最大值。这可以防止处理大值时发生溢出。没有这个，exp(1000) 会溢出到无穷大，使整个计算无效。

与 ReLU 不同，softmax 不跨输出类别并行化——每个样本由单个线程处理。这是因为我们需要对所有类别求和以进行归一化。在 softmax 内部并行化需要同步开销，对于只有 10 个类别来说不值得。

## 内存布局和数据流

理解内存流对于神经网络性能至关重要。让我们跟踪数据在网络中移动时发生的事情。

首先，我们在 CPU 上使用 Xavier/Glorot 初始化来初始化权重。这种技术设置初始权重以具有适当的方差，这有助于训练收敛：

```cuda
float weight1_scale = sqrtf(6.0f / (INPUT_SIZE + HIDDEN_SIZE));
for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; i++) {
    weights1[i] = (2.0f * (float)rand() / RAND_MAX - 1.0f) * weight1_scale;
}
```

然后我们将这些权重传输到 GPU 内存：

```cuda
cudaMemcpy(d_weights1, h_weights1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float),
           cudaMemcpyHostToDevice);
```

对于我们的网络，这传输了：
- 第 1 层权重：784 × 128 × 4 字节 = 401 KB
- 第 1 层偏置：128 × 4 字节 = 512 字节
- 第 2 层权重：128 × 10 × 4 字节 = 5 KB
- 第 2 层偏置：10 × 4 字节 = 40 字节

总参数：约 406 KB。现代网络可能有数十亿个参数（千兆字节），但原理是相同的。

在 GPU 上后，我们还为中间激活分配内存：

```cuda
cudaMalloc(&d_hidden_preact, BATCH_SIZE * HIDDEN_SIZE * sizeof(float));  // 64 × 128 × 4 = 32 KB
cudaMalloc(&d_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float));          // 64 × 10 × 4 = 2.5 KB
```

这些激活是临时的——我们只在前向传播期间需要它们。在训练中，我们还需要保留它们以进行反向传播。

## 完整的前向传播

现在让我们把所有东西放在一起。前向传播作为一系列内核启动执行：

```cuda
// 第 1 层：输入 → 隐藏
matrixMultiplyKernel<<<grid_mm2, block_mm>>>(d_input, d_weights1, d_hidden_preact,
                                            BATCH_SIZE, INPUT_SIZE, HIDDEN_SIZE);
addBiasKernel<<<grid_bias1, block_bias>>>(d_hidden_preact, d_bias1, BATCH_SIZE, HIDDEN_SIZE);
reluKernel<<<grid_act1, block_act>>>(d_hidden_preact, BATCH_SIZE * HIDDEN_SIZE);

// 复制激活用于下一层
cudaMemcpy(d_hidden_output, d_hidden_preact, BATCH_SIZE * HIDDEN_SIZE * sizeof(float),
          cudaMemcpyDeviceToDevice);

// 第 2 层：隐藏 → 输出
matrixMultiplyKernel<<<grid_mm1, block_mm>>>(d_hidden_output, d_weights2, d_output_preact,
                                            BATCH_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
addBiasKernel<<<grid_bias2, block_bias>>>(d_output_preact, d_bias2, BATCH_SIZE, OUTPUT_SIZE);

// Softmax 激活
softmaxKernel<<<grid_pred, block_pred>>>(d_output_preact, d_output, BATCH_SIZE, OUTPUT_SIZE);
```

每个内核启动都是异步的。CPU 发出启动命令并立即继续到下一行。内核在 GPU 上按顺序执行，因为它们使用默认的 CUDA 流。

从设备到设备的 `cudaMemcpy` 是不幸的——我们正在复制已经在 GPU 上的数据。更有效的实现会重用缓冲区或融合操作以避免这次复制。但为了清晰起见，我们保持操作分离。

对整个前向传播计时给我们 0.328 毫秒。让我们分析一下：

- 第 1 层矩阵乘法：最昂贵（64 × 784 × 128 次操作）
- 第 1 层 ReLU：内存受限，非常快
- 第 2 层矩阵乘法：较小（64 × 128 × 10 次操作）
- Softmax：最少时间（只有 64 个样本 × 10 个类别）

矩阵乘法占主导地位。这是神经网络的典型情况——大部分时间花在线性变换上。

## 批处理：分摊开销

注意我们一次处理 64 张图像，而不是一次一张。这种批处理对 GPU 效率至关重要。

每次内核启动都有开销——CPU 必须与 GPU 通信，GPU 必须调度线程。对于单张图像，这种开销可能主导实际的计算时间。但是当一起处理 64 张图像时，开销分摊到所有图像上。

此外，更大的矩阵维度更好地利用 GPU。64 × 784 的矩阵乘法比 1 × 784 的乘法能让更多的流式多处理器保持忙碌。

尝试修改代码中的批量大小并观察对吞吐量的影响：

- 批量大小 1：每批约 0.15 毫秒 = 每张图像 0.15 毫秒
- 批量大小 64：每批约 0.33 毫秒 = 每张图像 0.005 毫秒

批处理时每张图像的时间下降了 30 倍。更大的批量不会花费 64 倍的时间，因为 GPU 并行处理图像。

当然有一个限制。如果批量大小增加太多，你会耗尽 GPU 内存。最佳批量大小取决于网络架构和 GPU 容量。

## 内存带宽分析

让我们计算我们移动了多少数据。对于第一层：

**输入：** 64 × 784 × 4 字节 = 200 KB（读取）
**权重：** 784 × 128 × 4 字节 = 401 KB（读取）
**输出：** 64 × 128 × 4 字节 = 32 KB（写入）
**总计：** 633 KB

如果该层需要 0.2 毫秒，我们的带宽是 633 KB / 0.0002 秒 = 3.17 GB/秒。

将此与 RTX 5090 的理论 1792 GB/秒带宽进行比较。我们达到的峰值带宽不到 0.2%。这是因为我们朴素的矩阵乘法没有优化内存访问模式。每个线程独立读取数据，缓存重用很差。

这就是像 cuBLAS 这样的库擅长的地方。优化的矩阵乘法使用共享内存来缓存输入矩阵的分块，大大减少全局内存流量。教程 06 展示了卷积操作的这些技术。

## 权重初始化很重要

注意我们对权重使用 Xavier/Glorot 初始化：

```cuda
float weight1_scale = sqrtf(6.0f / (INPUT_SIZE + HIDDEN_SIZE));
weights1[i] = (2.0f * (float)rand() / RAND_MAX - 1.0f) * weight1_scale;
```

为什么是这个特定的公式？神经网络训练涉及梯度向后流经层。如果初始权重太大，梯度爆炸。如果太小，梯度消失。Xavier 初始化设置缩放以保留跨层的梯度方差。

因子 sqrt(6 / (n_in + n_out)) 来自分析使用 tanh 激活的网络中的方差传播。对于 ReLU 网络，He 初始化（sqrt(2 / n_in)）在理论上更好，但 Xavier 对两者都相当有效。

即使我们的网络没有被训练，使用适当的初始化也给我们合理的初始预测，而不是完全随机的输出。

## Softmax 中的数值稳定性

再看看 softmax 实现。为什么我们要减去最大值？

```cuda
for (int i = 0; i < num_classes; i++) {
    output[i] = expf(input[i] - max_val);
    sum += output[i];
}
```

考虑原始输入会发生什么。如果 input[i] = 100，那么 exp(100) ≈ 2.7 × 10^43，这会溢出浮点精度到无穷大。softmax 会输出 NaN。

通过减去最大值，我们确保 exp() 的所有输入都 ≤ 0。最大值变为 exp(0) = 1，所有其他值都更小。这使得溢出不可能，同时产生数学上等价的结果（因为 softmax 是平移不变的）。

当从头实现神经网络时，这些数值技巧至关重要。像 PyTorch 这样的库会自动处理它们，但理解它们有助于你在出错时进行调试。

## 与 cuDNN 比较

我们的实现是教育性的，但生产代码会使用 cuDNN（CUDA 深度神经网络库）。让我们比较一下：

**我们的实现：**
- 简单矩阵乘法：前向传播约 0.3 毫秒
- 内存带宽：约 3 GB/秒（峰值的 0.2%）
- 代码复杂度：200 行

**cuDNN 实现：**
- 优化的卷积和矩阵乘法：前向传播约 0.05 毫秒
- 内存带宽：约 200 GB/秒（峰值的 11%）
- 代码复杂度：10 行（调用库函数）

cuDNN 快 6 倍是因为它使用：
- 使用共享内存的分块矩阵乘法
- Tensor Core 加速（在支持的 GPU 上）
- 内核融合以组合操作
- 优化的内存布局

但理解我们的实现有助于你知道 cuDNN 在底层做什么，这在优化性能或调试问题时很有价值。

## 挑战练习

1. **测量层计时：** 修改代码以使用 CUDA 事件分别为每一层计时。哪一层花费的时间最多？这是否符合你基于操作数的预期？

2. **优化内存：** 当前代码在层之间复制激活。修改它以重用缓冲区并消除 `cudaMemcpy` 设备到设备的复制。

3. **添加 dropout：** 实现一个 dropout 层，以概率 p 随机将激活设置为零。使用 `curand_kernel.h` 在 GPU 上生成随机数。

4. **批量大小实验：** 编写一个脚本，使用批量大小 1、2、4、8、16、32、64、128 运行前向传播。绘制每张图像时间 vs 批量大小的图表。它在哪里趋于平稳？

5. **矩阵乘法优化：** 使用共享内存实现分块矩阵乘法（见教程 04）。将性能与朴素版本进行比较。

## 总结

神经网络非常适合 GPU 加速，因为它们的操作是大规模并行的。矩阵乘法，核心操作，可以分布在数千个线程上。激活函数独立应用于每个元素。

批处理对 GPU 效率至关重要。同时处理多个样本可以分摊内核启动开销并提高内存带宽利用率。当将 64 张图像批处理在一起时，我们的简单网络每张图像只需 0.005 毫秒。

内存带宽通常是神经网络的瓶颈。朴素实现只能达到峰值带宽的一小部分。像 cuBLAS 和 cuDNN 这样的优化库使用共享内存分块和其他技术来显著提高带宽利用率。

理解这些基础知识为你准备更复杂的架构。卷积网络、循环网络和 transformers 都建立在这些相同的基本操作之上：矩阵乘法和逐元素激活。

## 下一步

继续学习**教程 06：CNN 卷积操作**，了解卷积层如何在 GPU 上工作。你将看到共享内存分块如何优化使 CNN 对图像处理有效的滑动窗口操作，并理解为什么对于空间数据，卷积比全连接层更节省内存。

## 延伸阅读

- [NVIDIA cuDNN 文档](https://docs.nvidia.com/deeplearning/cudnn/developer-guide/)
- [神经网络与深度学习](http://neuralnetworksanddeeplearning.com/)
- [深度神经网络的高效处理](https://arxiv.org/abs/2002.03360)
- [CUDA 上的矩阵乘法](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)
