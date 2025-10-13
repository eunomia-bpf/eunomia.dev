# 教程：共享内存优化的 CNN 卷积

**所需时间：** 60-75 分钟
**难度：** 中级到高级
**前置要求：** 完成教程 04 和 05，理解卷积操作

完成本教程后，你将理解卷积层如何在 GPU 上工作、为什么共享内存优化对性能至关重要，以及教程 04 中的架构原理如何转化为显著的实际加速。你将看到单个优化技术带来的 30 倍性能提升。

代码仓库：<https://github.com/eunomia-bpf/basic-cuda-tutorial>

## 为什么卷积主导计算机视觉

卷积神经网络（CNNs）彻底改变了计算机视觉。它们为手机上的人脸识别、自动驾驶汽车中的物体检测和医学图像分析提供动力。但 CNNs 计算量很大——通过现代 CNN 的单次前向传播可能需要数十亿次乘加操作。

卷积操作是瓶颈。在像 ResNet-50 这样的典型 CNN 中，超过 95% 的计算时间花在卷积层上。这使得卷积成为在 GPU 上实现 CNNs 时最重要的优化操作。

与教程 05 中的全连接层不同，卷积利用空间结构。卷积不是将每个输入连接到每个输出，而是应用在输入上滑动的小滤波器（核）。这种滑动窗口操作处理局部区域，非常适合相邻像素相关的图像。

## 运行示例

让我们从看到优化的实际效果开始：

```bash
make 06-cnn-convolution
./06-cnn-convolution
```

你会看到类似这样的输出：

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

共享内存版本比直接实现快近 30 倍。两者产生相同的结果，但一个有效利用了 GPU 架构，而另一个没有。本教程解释了原因。

## 理解卷积操作

卷积将一个小滤波器（如 5×5）应用到图像的每个位置。可以把滤波器想象成模式检测器。当它在图像上滑动时，在模式匹配的地方产生高输出，在不匹配的地方产生低输出。

数学上，对于每个输出位置 (x, y)，我们计算：

```
Output[y,x] = Σky Σkx Input[y+ky, x+kx] × Kernel[ky, kx]
```

这是核与输入块之间的点积。对于 28×28 的输出和 5×5 的核，我们执行 28 × 28 × 5 × 5 = 19,600 次点积。每次点积涉及 25 次乘加。对于单个通道，这是 490,000 次操作。

真实的 CNNs 有多个输入通道（如 RGB 图像有 3 个通道）和多个输出通道（检测不同模式的滤波器）。我们的例子使用 1 个输入通道和 16 个输出通道，将工作量乘以 16。

### 填充和步长

两个参数控制卷积如何操作：

**填充：** 在边界周围添加零。使用 padding=2，我们在每一侧添加 2 行/列的零。这允许滤波器处理边缘像素并保持输出与输入大小相同。

**步长：** 滤波器每次移动的距离。Stride=1 意味着滤波器一次移动一个像素。Stride=2 将使输出维度减半。

我们的例子使用 padding=2 和 stride=1，所以 28×28 的输入产生 28×28 的输出。

## 朴素实现

让我们从直接的实现开始。每个线程计算一个输出元素：

```cuda
__global__ void convolutionDirectKernel(
    float *input, float *kernels, float *output,
    int batchSize, int inputChannels, int inputSize,
    int kernelSize, int kernelCount, int outputSize,
    int padding, int stride)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z; // 输出通道
    int b = threadIdx.z; // 批次索引

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

核 k 的线程 (x, y) 计算该滤波器的输出位置 (x, y)。它从输入读取 5×5 块，与核逐元素相乘，并求和结果。

这能工作，但内存效率很差。看看相邻线程执行时发生什么。计算输出 (0, 0) 的线程读取输入像素 (0,0) 到 (4,4)。计算输出 (0, 1) 的线程读取输入像素 (0,1) 到 (4,5)。这些重叠了！像素 (0,1) 到 (0,4) 从全局内存读取了两次。

对于 5×5 的核，相邻输出共享 5 列中的 4 列。那是 80% 的冗余。平均而言，每个输入像素从全局内存读取大约 5 次。

运行这个给我们的测试用例 0.223 毫秒。现在让我们看看共享内存如何消除这种冗余。

## 共享内存优化：关键洞察

优化洞察很简单：相邻线程需要重叠的输入数据。不是让每个线程从慢速全局内存读取自己的 5×5 块，整个块应该协作地将更大的分块加载到快速共享内存一次，然后所有线程从那里读取。

考虑一个 8×8 线程的块计算 8×8 输出。没有共享内存，我们从全局内存读取 8×8×5×5 = 1,600 个值（有很多冗余）。使用共享内存，我们一次加载 12×12 分块（8 + 5 - 1 = 12），然后所有线程从共享内存读取。那是 144 次全局内存加载而不是 1,600 次——11 倍的减少！

这是结构：

```cuda
__global__ void convolutionSharedKernel(...) {
    extern __shared__ float sharedData[];

    int tileSize = blockDim.x; // 8
    int tileSizeWithPadding = tileSize + kernelSize - 1; // 12

    // 阶段 1：协作地将输入分块加载到共享内存
    for (int c = 0; c < inputChannels; c++) {
        // 加载 tileSizeWithPadding × tileSizeWithPadding 元素
        // 每个线程加载多个元素以覆盖分块
        ...
    }

    __syncthreads(); // 等待所有线程完成加载

    // 阶段 2：使用共享内存计算卷积
    float sum = 0.0f;
    for (int c = 0; c < inputChannels; c++) {
        for (int ky = 0; ky < kernelSize; ky++) {
            for (int kx = 0; kx < kernelSize; kx++) {
                float in_val = sharedData[...]; // 从共享内存读取！
                float kernel_val = kernels[...];
                sum += in_val * kernel_val;
            }
        }
    }

    output[...] = sum;
}
```

加载阶段很棘手。我们有 8×8 = 64 个线程，但需要加载 12×12 = 144 个元素。每个线程必须加载多个元素。我们使用嵌套循环，其中每个线程在位置 (tx + dx, ty + dy) 加载各种偏移的元素。

`__syncthreads()` 屏障至关重要。它确保所有线程在任何线程开始计算之前完成加载。没有它，一些线程可能会在共享内存完全填充之前尝试从中读取，获得垃圾数据。

## 内存访问模式分析

让我们量化改进。对于 8×8 分块和 5×5 核：

**直接卷积：**
- 64 个线程中的每一个从全局内存读取 25 个值
- 总全局读取：64 × 25 = 1,600
- 由于重叠区域，其中许多是重复的

**共享内存卷积：**
- 协作地加载 12×12 = 144 个值到共享内存
- 总全局读取：144
- 然后每个线程从共享内存读取 25 个值（快！）

减少：1,600 → 144 全局内存访问。那是 11 倍的减少。

但等等——共享内存访问不是免费的。它们比全局内存快得多（延迟大约低 100 倍），但仍然可能有 bank 冲突。幸运的是，我们的访问模式大多避免了冲突，因为线程在加载时访问不同的元素，相邻元素具有顺序地址。

实际加速（在我们的例子中是 29.68 倍）来自：
1. 减少的全局内存流量（11 倍更少的加载）
2. 更好的缓存利用率（更少的唯一地址）
3. 加载阶段的合并内存访问
4. 低共享内存 bank 冲突

## 理解实现细节

加载阶段值得仔细看看：

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

我们以 `tileSize`（8）的增量迭代。在第一次迭代（dy=0, dx=0）中，线程 (tx, ty) 加载分块的元素 (tx, ty)。在下一次迭代（dy=0, dx=8）中，同一线程加载元素 (tx, ty+8)，依此类推。

对于 12×12 分块和 8×8 线程，我们在每个维度需要两次遍历。第一次遍历覆盖位置 0-7，第二次覆盖 8-11。并非所有线程都参与第二次遍历，因为我们在每个维度只需要 4 个更多元素。

边界检查处理填充。当计算图像边缘附近的输出时，我们的分块扩展到输入边界之外。我们为越界位置加载零，自动实现零填充。

## 激活和池化层

在卷积之后，CNNs 通常应用激活函数。ReLU（修正线性单元）是标准的：

```cuda
__global__ void reluActivationKernel(float *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}
```

这是逐元素应用的，只需要 0.006 毫秒——比卷积快得多。它是内存受限的：GPU 花在读取和写入数据上的时间比计算最大值的时间多。

池化减少空间维度。最大池化在每个窗口中取最大值：

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

我们的例子使用 2×2 池化，步长为 2，将 28×28 特征图减少到 14×14。这使空间维度减半，减少后续层的计算。

池化也可以从共享内存中受益，但对于 2×2 窗口，开销不值得。性能在 0.006 毫秒时已经很好了。

## 完整的 CNN 层流水线

综合起来，典型的 CNN 层执行：

1. 卷积：0.008 毫秒（使用共享内存）
2. ReLU 激活：0.006 毫秒
3. 最大池化：0.006 毫秒

总计：处理 64 张图像每层 0.020 毫秒。每张图像每层约 0.3 微秒。50 层 CNN 可以在约 60 微秒内处理图像（仅计算，内存传输增加开销）。

像 ResNet 这样的现代 CNNs 有跳跃连接和批归一化，增加了复杂性。但卷积仍然是瓶颈，使我们的优化至关重要。

## 为什么共享内存如此重要

30 倍的加速可能看起来好得令人难以置信。让我们理解硬件层面发生的事情。

从教程 04 中，我们知道全局内存访问需要数百个周期。共享内存访问只需要几个周期。当我们从全局内存读取同一输入像素 5 次时（直接方法），我们等待内存 5 次。当我们从全局内存读取一次到共享内存，然后从共享内存读取 5 次时（优化方法），我们只等待全局内存一次。

此外，直接方法的合并很差。当 warp 中的 32 个线程各自读取自己的 5×5 块时，它们访问分散的内存位置。优化方法加载连续的分块，在加载阶段提供完美的合并。

最后，直接方法可能超过 GPU 的缓存容量。RTX 5090 上的 L2 缓存是 98 MB，但一批图像可能是数百兆字节。共享内存明确管理一个适合片上的工作集，保证快速访问。

## 与 cuDNN 比较

我们的优化实现很好，但生产代码使用 cuDNN。它们如何比较？

**我们的实现：**
- 共享内存分块：0.008 毫秒
- 易于理解
- 适用于任何核大小

**cuDNN 实现：**
- 多种算法（GEMM、Winograd、FFT）
- 基于问题大小的算法选择
- 我们的测试用例估计约 0.002 毫秒
- 高度优化，特定于架构

cuDNN 可能快 4 倍，因为它使用额外的技术：
- 3×3 和 5×5 核的 Winograd 变换（更少的乘法）
- Im2col + GEMM 用于利用高度调优的矩阵乘法
- 在较新的 GPU 上利用 Tensor Core
- 内核融合以组合卷积、偏置和激活

但仅共享内存就带来的 30 倍加速捕获了大部分可能的优化。剩下的 4 倍需要算法级别的变化，而不仅仅是更好的内存管理。

## 内存带宽分析

让我们计算我们的内存带宽。对于共享内存版本：

**卷积输入读取：** 64 × 1 × 28 × 28 × 4 字节 = 200 KB
**核读取：** 16 × 1 × 5 × 5 × 4 字节 = 1.6 KB
**输出写入：** 64 × 16 × 28 × 28 × 4 字节 = 3.2 MB
**总计：** 3.4 MB

在 0.008 毫秒中，那是 3.4 MB / 0.000008 秒 = 425 GB/秒。

RTX 5090 有 1792 GB/秒的理论带宽。我们达到了峰值的 24%——比教程 05 中朴素矩阵乘法的 0.2% 好得多！

直接卷积仅达到 3.4 MB / 0.223 毫秒 = 15 GB/秒（峰值的 0.8%），因为冗余读取和糟糕的合并。

## 实际影响

构建 CNN 应用时，这些经验适用：

**对标准操作使用 cuDNN。** 我们的手动优化是教育性的，但 cuDNN 更快，处理我们没有覆盖的边缘情况。

**理解优化为什么有效。** 如果你遇到 cuDNN 无法解决的性能问题（自定义层、新颖架构），了解这些原理让你可以编写高效的自定义内核。

**在优化前进行分析。** 这里的 30 倍加速很大，但只有当卷积是你的瓶颈时才重要。始终进行分析以找到真正的瓶颈。

**首先考虑内存。** 大多数 GPU 内核是内存受限的。减少内存流量通常比优化计算更有帮助。

## 挑战练习

1. **测量 bank 冲突：** 使用 PTX 汇编（来自教程 02）添加共享内存 bank 冲突的计数器。我们的访问模式避免冲突了吗？

2. **实现 3×3 卷积：** 修改代码用于 3×3 核。加速改变了吗？为什么？

3. **添加通道融合：** 修改共享内存内核以一次加载所有输入通道。这能提高性能吗？

4. **实验分块大小：** 尝试 4×4、16×16 和 32×32 分块。绘制性能 vs 分块大小。什么是最优的？

5. **实现深度可分离卷积：** 这种高效变体分离空间和通道过滤。实现它并与标准卷积比较。

## 总结

卷积层是 CNNs 的计算核心，使它们的优化至关重要。朴素实现由于重叠的感受野而反复从慢速全局内存读取相同数据。

共享内存分块通过将每个输入像素加载一次到快速片上内存来消除这种冗余。这显著减少了内存流量并改善了合并，在我们的例子中实现了 30 倍加速。

这些原理超越了卷积。任何滑动窗口操作都受益于分块：相关、形态学操作、音频处理中的时间卷积和视频理解中的 3D 卷积。

理解这些优化揭示了为什么 GPU 主导深度学习。大规模并行性处理数十亿次乘加。内存层次结构实现使这些操作可行的重用模式。没有两者，现代 AI 将是不切实际的。

## 下一步

继续学习**教程 07：注意力机制**，了解 transformer 网络如何在 GPU 上处理序列。你将实现驱动 GPT 和 BERT 等模型的自注意力操作，并看到内存访问模式与卷积有何不同。

## 延伸阅读

- [NVIDIA cuDNN 文档](https://docs.nvidia.com/deeplearning/cudnn/developer-guide/)
- [CUDA 中的优化并行归约](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)
- [卷积神经网络的快速算法](https://arxiv.org/abs/1509.09308) - Winograd 卷积
- [用于卷积的 Im2col 和 GEMM](https://sahnimanas.github.io/post/anatomy-of-a-high-performance-convolution/)
