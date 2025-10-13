# 教程：你的第一个CUDA程序 - 向量加法

**所需时间：** 30-45分钟
**难度：** 初学者
**前置要求：** 基础C/C++知识，支持CUDA的NVIDIA GPU

完成本教程后，你将理解如何编写、编译和运行一个完整的CUDA程序，在GPU上执行并行向量加法。你将学习GPU编程的基本工作流程，并看到相比CPU代码的实际性能提升。

## 理解挑战

想象一下，你需要将两个包含50,000个数字的数组相加。在CPU上，你会写一个循环，一次处理一个元素。这种顺序方法可以工作，但在处理大型数据集时很慢。GPU擅长处理这类问题，因为它们可以同时处理数千个元素。

这样想：CPU就像拥有一个非常快的工人，而GPU就像拥有数千个工人，每个工人都可以同时处理问题的一小部分。对于像加法这样简单、重复的任务，GPU的大规模并行性能获胜。

## 开始入门

首先，确保你已安装CUDA工具包。通过运行以下命令验证安装：

```bash
nvcc --version
nvidia-smi
```

第一个命令显示你的CUDA编译器版本，第二个显示你的GPU信息。如果两个命令都能正常工作，你就可以开始了。

克隆教程仓库并导航到第一个示例：

```bash
git clone https://github.com/eunomia-bpf/basic-cuda-tutorial
cd basic-cuda-tutorial
```

## 构建并运行你的第一个CUDA程序

让我们先构建并运行示例，看看它的实际效果：

```bash
make 01-vector-addition
./01-vector-addition
```

你应该看到类似以下的输出：

```
Vector addition of 50000 elements
CUDA kernel launch with 196 blocks of 256 threads
Test PASSED
Done
```

现在它能运行了，让我们理解底层发生了什么。

## CUDA编程模型

在编辑器中打开`01-vector-addition.cu`。每个CUDA程序都遵循类似的模式：

1. 在CPU（主机）和GPU（设备）上分配内存
2. 将输入数据从CPU复制到GPU
3. 启动在GPU上运行的内核（函数）
4. 将结果从GPU复制回CPU
5. 清理所有分配的内存

与常规C编程相比，这个工作流程可能看起来很冗长，但这是必要的，因为CPU和GPU有独立的内存空间。数据必须在它们之间显式移动。

## 剖析内核函数

任何CUDA程序的核心都是内核函数。这是我们的向量加法内核：

```cuda
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}
```

`__global__`关键字告诉CUDA编译器这个函数在GPU上运行，但可以从CPU代码调用。注意实际的计算有多简单：只是一个加法。神奇之处在于我们如何计算索引`i`。

### 理解线程索引

每个GPU线程都需要知道它应该处理数组的哪个元素。公式`blockDim.x * blockIdx.x + threadIdx.x`为每个线程计算一个唯一的全局索引。

为了可视化这一点，假设我们有50,000个元素，我们将GPU线程组织成每个块256个线程。我们需要196个块（从50000/256向上取整）。索引的工作方式如下：

- 块0中的线程0：索引 = 256 * 0 + 0 = 0
- 块0中的线程5：索引 = 256 * 0 + 5 = 5
- 块1中的线程0：索引 = 256 * 1 + 0 = 256
- 块2中的线程100：索引 = 256 * 2 + 100 = 612

边界检查`if (i < numElements)`至关重要，因为我们的最后一个块可能有超出数组大小的线程。没有这个检查，这些线程会访问无效内存。

## CUDA中的内存管理

看看我们在主函数中如何分配内存：

```cuda
// 主机（CPU）内存
float *h_A = (float *)malloc(size);

// 设备（GPU）内存
float *d_A = NULL;
cudaMalloc((void **)&d_A, size);
```

我们使用命名约定，其中`h_`前缀表示主机内存，`d_`前缀表示设备内存。这有助于防止常见错误，比如将GPU指针传递给CPU函数。

`cudaMalloc`函数的工作方式类似于`malloc`，但它在GPU的全局内存中分配内存。这个内存可以被所有GPU线程访问，但延迟比共享内存或寄存器更高。

## 主机和设备之间的数据传输

在两侧分配内存后，我们需要将输入数据复制到GPU：

```cuda
cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
```

`cudaMemcpy`函数是同步的，意味着CPU会等待传输完成后再继续。第四个参数指定方向：`cudaMemcpyHostToDevice`表示CPU到GPU，`cudaMemcpyDeviceToHost`表示GPU到CPU。

这些内存传输可能成为性能瓶颈。经验法则是，你希望最小化传输次数，并最大化在传输之间在GPU上执行的计算量。

## 启动内核

内核启动是并行魔法发生的地方：

```cuda
int threadsPerBlock = 256;
int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
```

三角括号语法`<<<blocksPerGrid, threadsPerBlock>>>`是CUDA指定执行配置的方式。我们告诉GPU启动196个块，每个块256个线程，总共给我们50,176个线程（略多于我们的50,000个元素）。

为什么是每个块256个线程？这是一个精心选择的默认值，在大多数NVIDIA GPU上都能很好地工作。线程块被调度到流式多处理器（SM）上，每个块256个线程通常在不使用太多资源的情况下提供良好的占用率。

向上取整除法公式`(numElements + threadsPerBlock - 1) / threadsPerBlock`确保我们总是有足够的线程来覆盖所有元素。对于50,000个元素和每个块256个线程，这给我们196个块。

## 错误检查

CUDA内核启动是异步的，意味着CPU不会等待内核完成。内核中的错误不会立即显示。这就是为什么我们检查启动错误：

```cuda
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    fprintf(stderr, "Failed to launch kernel: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
}
```

这捕获配置错误，如请求每个块太多线程或GPU内存不足。在开发期间，始终在内核启动和CUDA API调用后检查错误。

## 验证和清理

将结果复制回主机后，我们验证计算：

```cuda
for (int i = 0; i < numElements; ++i) {
    if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
        fprintf(stderr, "Result verification failed at element %d!\n", i);
        exit(EXIT_FAILURE);
    }
}
```

我们使用带有小epsilon（1e-5）的浮点比较，因为由于不同的舍入模式和指令执行顺序，GPU上的浮点运算可能产生与CPU上略有不同的结果。

最后，我们释放所有分配的内存：

```cuda
cudaFree(d_A);  // 释放GPU内存
free(h_A);      // 释放CPU内存
```

忘记释放内存会导致泄漏。GPU内存通常比系统RAM更有限，所以泄漏会很快引起问题。

## 动手练习：测量性能

现在让我们添加计时来看实际的加速效果。我们将使用CUDA事件进行精确的GPU计时。在内核启动前添加以下代码：

```cuda
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
```

在内核启动后：

```cuda
cudaEventRecord(stop);
cudaEventSynchronize(stop);

float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
printf("Kernel execution time: %.3f ms\n", milliseconds);

cudaEventDestroy(start);
cudaEventDestroy(stop);
```

重新编译并运行。你应该看到内核在不到一毫秒内完成。现在通过添加此函数与CPU实现进行比较：

```cuda
void vectorAddCPU(const float *A, const float *B, float *C, int numElements) {
    for (int i = 0; i < numElements; i++) {
        C[i] = A[i] + B[i];
    }
}
```

使用标准C计时函数对其计时。在大多数系统上，即使包括内存传输开销，GPU版本也会明显更快。

## 理解占用率和块大小

让我们实验一下块大小。当前代码使用每个块256个线程。尝试修改为使用不同的值：

```cuda
int threadsPerBlock = 128;  // 或512，或1024
```

重新编译并启用计时运行每个版本。你会注意到每个块128个线程更慢，而512或1024可能与256相似。这是因为GPU占用率。

占用率指的是你对GPU流式多处理器的利用程度。每个SM都有有限数量的寄存器、共享内存和warp槽。每个块使用256个线程（8个warp）通常在大多数GPU上实现良好的占用率，而不会耗尽这些资源。

要检查你的内核的占用率，使用以下命令编译：

```bash
nvcc --ptxas-options=-v 01-vector-addition.cu -o 01-vector-addition
```

在输出中查找"registers"和"shared memory"使用情况。你可以使用CUDA占用率计算器来确定特定GPU和内核的最佳块大小。

## 内存带宽分析

向量加法是内存受限操作，意味着性能受限于我们读写数据的速度，而不是计算速度。让我们计算实现的内存带宽：

对于50,000个元素：
- 我们读取两个float数组：50,000 * 4字节 * 2 = 400 KB
- 我们写入一个float数组：50,000 * 4字节 = 200 KB
- 总内存流量：600 KB

如果你的内核在0.1毫秒内运行，带宽是：
- 600 KB / 0.0001秒 = 6 GB/s

将此与你的GPU的理论带宽进行比较（检查`nvidia-smi`或GPU规格）。例如，RTX 5090的理论带宽约为1792 GB/s。如果你实现了100-200 GB/s，对于一个简单的内核来说已经做得很好了。

理论带宽和实现带宽之间的差距来自几个因素：内存访问模式、缓存行为和PCIe传输开销。我们将在后面的教程中探索优化。

## 内存合并访问

我们的内核具有出色的内存访问模式。相邻的线程（在warp内）访问相邻的内存位置。这允许GPU将多个内存请求合并为单个宽事务。

尝试这个实验：修改内核以跨步模式访问内存：

```cuda
int i = (blockDim.x * blockIdx.x + threadIdx.x) * 2;  // 跳过每个其他元素
if (i < numElements) {
    C[i] = A[i] + B[i];
}
```

你需要启动两倍的线程并调整边界检查。对这个版本计时并比较。非合并访问模式会明显更慢，因为每个内存请求都需要单独的事务。

## 调试你的CUDA代码

当出现问题时，CUDA提供了几个调试工具：

**cuda-memcheck：** 检测内存错误，如越界访问
```bash
cuda-memcheck ./01-vector-addition
```

**cuda-gdb：** GPU调试器，用于逐步调试
```bash
cuda-gdb ./01-vector-addition
```

**Compute Sanitizer：** cuda-memcheck的现代替代品
```bash
compute-sanitizer ./01-vector-addition
```

要注意的常见错误：
- 将主机指针传递给内核（会导致段错误）
- 忘记从设备复制数据回来
- 不检查CUDA错误代码
- 访问超出数组边界的内存
- 内存分配和释放调用不匹配

## 分析你的代码

要详细了解GPU上发生了什么，使用NVIDIA的分析工具：

**Nsight Systems** 用于时间线分析：
```bash
nsys profile --stats=true ./01-vector-addition
```

这准确显示时间花在哪里：内存传输、内核执行和CPU代码。

**Nsight Compute** 用于内核分析：
```bash
ncu --set full ./01-vector-addition
```

这提供了关于内存带宽、SM利用率和性能瓶颈的详细指标。

## 常见问题和解决方案

**错误："CUDA driver version is insufficient"**
你的驱动程序对于你的CUDA工具包版本来说太旧了。更新你的NVIDIA驱动程序。

**错误："out of memory"**
GPU没有足够的内存。减少`numElements`或分批处理数据。

**错误："invalid device function"**
内核是为不同的GPU架构编译的。检查你的Makefile中的`-arch`标志是否与你的GPU的计算能力匹配。

**段错误：**
可能是将主机指针传递给内核或访问未分配的内存。使用`cuda-memcheck`诊断。

**结果不正确：**
检查你的索引计算。添加边界检查并首先用较小的数据集验证。

## 高级主题预览

现在你理解了基本的CUDA编程，未来的教程将涵盖：

**统一内存：** 通过为CPU和GPU使用单个指针来简化内存管理
**流：** 重叠计算和数据传输以获得更好的性能
**共享内存：** 使用快速片上内存进行线程协作
**原子操作：** 处理多个线程写入同一位置时的竞争条件
**纹理内存：** 优化图像处理中的空间局部性

## 挑战练习

1. **修改内核**以计算C[i] = A[i] * B[i] + C[i]（乘加操作）

2. **实现错误检查**，为所有CUDA API调用使用宏：
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

3. **添加全面的计时**，分别测量：
   - 主机到设备传输时间
   - 内核执行时间
   - 设备到主机传输时间
   - 总GPU时间与CPU实现时间

4. **实验向量大小：** 尝试1K、10K、100K、1M、10M个元素。绘制执行时间与大小的关系。在什么时候GPU变得比CPU更快？

5. **实现统一内存版本：** 用cudaMallocManaged替换显式的cudaMalloc/cudaMemcpy。性能会改变吗？

## 总结

你现在已经编写了第一个CUDA程序并理解了基本概念：

CPU和GPU有独立的内存空间，需要显式的数据移动。内核在组织成块的数千个线程上并行执行。每个线程计算一个唯一索引以确定要处理哪个数据元素。适当的错误检查和验证对于可靠的GPU代码至关重要。

向量加法是内存受限的，所以性能取决于内存带宽而不是计算速度。良好的内存访问模式（合并）对性能至关重要。典型的工作流程是：分配、复制到设备、计算、复制到主机、释放。

这些基础将在整个GPU编程中为你服务。无论你是添加向量、训练神经网络还是模拟物理，相同的模式都适用。

## 下一步

继续学习**教程02：PTX汇编**，了解如何编写低级GPU代码并理解编译器生成的内容。你将看到在GPU上执行的实际汇编指令，并学习何时使用内联PTX进行性能关键代码。

## 进一步阅读

- [CUDA C++编程指南](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) - 官方综合指南
- [CUDA C++最佳实践指南](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) - 性能优化技术
- [GPU架构白皮书](https://www.nvidia.com/en-us/data-center/resources/gpu-architecture/) - 理解硬件
- [CUDA示例](https://github.com/NVIDIA/cuda-samples) - 更多示例代码
