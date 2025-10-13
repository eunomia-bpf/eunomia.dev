# 教程：理解 GPU 架构与执行模型

**所需时间：** 60-75 分钟
**难度：** 中级
**前置要求：** 完成教程 01 和 02，基本了解计算机架构

完成本教程后，你将理解 GPU 的物理组织结构、线程如何被调度和执行、内存层次结构如何工作，以及为什么某些编码模式能获得更好的性能。这些知识对于编写高性能 CUDA 代码至关重要。

代码仓库：<https://github.com/eunomia-bpf/basic-cuda-tutorial>

## 为什么 GPU 架构很重要

你可能会问，既然编译器应该为你处理优化，为什么还需要理解硬件？现实是，GPU 架构与 CPU 架构有着根本性的不同，编写高效的 GPU 代码需要理解这些差异。

想想这个场景：在 CPU 上，你可能会写一个循环逐个处理元素，处理器会通过流水线、分支预测和乱序执行来加速它。但在 GPU 上，你有数千个线程同时运行，性能取决于这些线程如何协作以及如何访问内存。如果编写代码时忽略 GPU 的架构特点，就像用驾驶汽车的方式骑摩托车一样——技术上可行，但完全没有发挥出真正的性能。

## GPU 硬件层次结构

让我们从检查现代 GPU 的实际结构开始。运行示例程序时，你会看到关于 GPU 的详细信息：

```bash
make 04-gpu-architecture
./04-gpu-architecture
```

在 NVIDIA GeForce RTX 5090 上，你会看到类似这样的输出：

```
Device 0: NVIDIA GeForce RTX 5090
  Compute Capability: 12.0
  SMs (Multiprocessors): 170
  Warp Size: 32 threads
  Max Threads per SM: 1536
  Max Threads per Block: 1024
```

让我们分解这些数字的含义以及为什么它们重要。

### 流式多处理器：核心构建模块

GPU 由许多流式多处理器（Streaming Multiprocessors，简称 SMs）组成。每个 SM 就像一个小型处理器，可以同时执行许多线程。RTX 5090 有 170 个 SMs，这意味着它可以在任何时刻并行运行 170 个不同的线程组。

可以把 SM 想象成一个有多条流水线的工厂车间。每条流水线（称为 warp）有 32 个工人（线程），他们同时执行相同的操作。SM 负责调度哪些流水线处于活动状态，并为它们提供工作。

当你启动一个内核时，CUDA 运行时会在可用的 SMs 上分配你的线程块。如果你启动 340 个块而有 170 个 SMs，最初每个 SM 会分配到两个块。当块完成后，新的块会被调度上去。这就是为什么拥有比 SMs 更多的块通常是好的——它可以保持所有硬件处于忙碌状态。

### 线程执行模型

CUDA 中的每个线程都有一个由其在三维网格中的位置决定的唯一标识：

```cuda
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
```

这个层次结构的存在是有原因的。线程被组织成块，块被组织成网格。这种结构直接映射到硬件调度和执行代码的方式。

运行线程层次结构演示。你会看到类似这样的输出：

```
Thread ID: 0, Position: (0,0,0) in Block (0,0,0) of Grid (4,4,4)
Thread ID: 800, Position: (0,0,2) in Block (0,3,0) of Grid (4,4,4)
```

每个线程都确切地知道自己在这个层次结构中的位置。这非常关键，因为它允许线程在没有任何集中协调的情况下计算应该处理哪些数据。块 (0,0,0) 中的线程 0 处理的数据与块 (1,0,0) 中的线程 0 不同。

### 理解 Warps：基本执行单元

这里有件事可能会让你惊讶：尽管你是以单个线程的方式编程，GPU 并不是逐个执行它们的。相反，它以 32 个线程为一组（称为 warp）来执行。

warp 中的所有 32 个线程同时执行相同的指令。这被称为 SIMT（Single Instruction, Multiple Threads，单指令多线程）。它类似于 CPU 的 SIMD（Single Instruction, Multiple Data），但更灵活，因为线程在必要时可以分化。

看看 warp 执行的输出：

```
Thread   0: Warp 10000, Lane  0
Thread  32: Warp 10001, Lane  0
Thread  64: Warp 10002, Lane  0
```

线程 0-31 在 warp 10000 中，线程 32-63 在 warp 10001 中，依此类推。在每个 warp 内，线程有从 0 到 31 的 lane ID。这种组织方式对性能有深远的影响。

### Warp 分化：当线程走不同的路径时

当 warp 中的线程需要执行不同的代码时会发生什么？考虑这个内核：

```cuda
if (threadIdx.x < 16) {
    // warp 的前半部分执行这个
    result = expensive_operation_a(data);
} else {
    // warp 的后半部分执行这个
    result = expensive_operation_b(data);
}
```

由于 warp 中的所有线程必须执行相同的指令，硬件会做一些巧妙但低效的事情：它执行两条路径。首先，线程 0-15 执行 operation_a，而线程 16-31 被屏蔽（不活跃）。然后线程 16-31 执行 operation_b，而线程 0-15 被屏蔽。

这意味着分化的代码需要两条路径加起来的时间。演示展示了这一点：

```
Note on warp divergence:
  When threads within a warp take different paths (diverge),
  the warp executes both paths, masking threads as appropriate.
  This reduces efficiency.
```

为了避免 warp 分化，尽量让 warp 中的线程执行相同的代码路径。如果必须有条件语句，对线程进行分组，使 warp 中的所有 32 个线程都采取相同的分支。

## 内存层次结构：性能的关键

GPU 性能通常不是受计算限制，而是受内存访问限制。理解内存层次结构对于编写快速代码至关重要。

### 全局内存：大但慢

当你使用 `cudaMalloc` 时，你分配的是全局内存。这是 GPU 的主内存——很大（RTX 5090 上有 32 GB），但延迟高（数百个时钟周期）。

```
Total Global Memory: 31.36 GB
Memory Clock Rate: 14001 MHz
Memory Bus Width: 512 bits
Peak Memory Bandwidth: 1792.13 GB/s
```

带宽数字是理论峰值。实际上，你能达到的只是它的一部分，具体取决于你的内存访问模式。这就引出了一个关键概念：内存合并。

### 内存合并：为什么访问模式很重要

现代 GPU 以大块加载内存（通常是 128 字节）。当 warp 中的线程访问相邻的内存位置时，硬件可以将这些访问合并（coalesce）成单个事务。

考虑这两种模式：

**合并访问：**
```cuda
// 线程 0 访问 data[0]，线程 1 访问 data[1]，等等
output[tid] = data[tid] * 2.0f;
```

**非合并访问：**
```cuda
// 线程 0 访问 data[0]，线程 1 访问 data[8]，等等
output[tid] = data[tid * 8 % n] * 2.0f;
```

第一种模式可以用一次 128 字节的内存事务满足整个 warp。第二种模式可能需要 32 次独立的事务。性能差异是巨大的。

运行内存合并演示：

```
Memory access timing:
  Elapsed time: 0.024 ms
```

虽然这个简单的例子可能不会显示巨大的差异，但在实际应用中，适当的合并可以将性能提高 5-10 倍。

### 共享内存：快速的片上存储

共享内存是一个小的（每个块 48 KB）但极快的内存空间，块中的所有线程都可以访问。它位于 SM 本身，因此访问延迟远低于全局内存。

```
Shared Memory per Block: 48 KB
```

共享内存非常适合线程需要协作的情况。以下是经典模式：

```cuda
__shared__ float sharedData[256];

// 从全局内存加载到共享内存
sharedData[threadIdx.x] = globalData[globalIdx];

// 同步以确保所有线程都已加载数据
__syncthreads();

// 现在所有线程都可以快速从共享内存读取
result = sharedData[threadIdx.x] + sharedData[threadIdx.x + 1];
```

`__syncthreads()` 至关重要。它是一个屏障，确保块中的所有线程都到达这个点后才继续。没有它，一些线程可能会尝试读取尚未加载的数据。

演示展示了使用共享内存的简单模板操作：

```
First few output values (should be sums of adjacent input values):
  output[0] = 1.0
  output[1] = 3.0
  output[2] = 5.0
```

每个输出是两个相邻输入的和，使用共享内存高效计算。

### L2 缓存和寄存器文件

在共享内存和全局内存之间是 L2 缓存：

```
L2 Cache Size: 98304 KB
```

L2 缓存是自动的——你不需要显式管理它。它缓存频繁访问的全局内存，帮助隐藏延迟。最近的 GPU 架构允许你通过 PTX 指令影响缓存行为，正如我们在教程 02 中看到的。

最快的内存是寄存器：

```
Registers per Block: 65536
```

每个线程都有自己的私有寄存器。你在内核代码中声明的变量通常存储在寄存器中。寄存器访问基本上没有延迟，但每个 SM 有限数量的寄存器要在所有活跃线程之间共享。

这产生了一个权衡：每个线程使用更多寄存器允许更复杂的计算，但限制了可以同时活跃的线程数量（占用率）。

## 占用率：保持 GPU 忙碌

占用率是指活跃 warps 与 SM 可以支持的最大 warps 数量的比率。更高的占用率通常意味着更好的性能，因为它有助于隐藏内存延迟。

```
Max Threads per SM: 1536
```

warp 大小为 32，这意味着每个 SM 最多可以处理 48 个 warps（1536 / 32）。如果你的内核使用如此多的寄存器或共享内存，以至于每个 SM 只能运行 24 个 warps，你的占用率就是 50%。

占用率计算考虑三个因素：

**每个块的线程数：** 更多线程通常更好，但不要超过硬件限制。

**每个线程的寄存器：** 越少越好，但如果溢出到本地内存可能会损害性能。

**每个块的共享内存：** 越少越好，但共享内存能实现重要的优化。

你可以用以下命令检查占用率：

```bash
nvcc --ptxas-options=-v 04-gpu-architecture.cu
```

这显示寄存器和共享内存使用情况。使用 CUDA 占用率计算器找到内核的最佳点。

## 实际示例：优化的矩阵乘法

让我们看看理解架构如何改进实际代码。这是一个朴素的矩阵乘法：

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

这能工作，但性能很糟糕。每个线程从全局内存读取 A 的整行和 B 的整列。对于 1024x1024 的矩阵，每个线程要进行 1024 次全局内存访问。

现在考虑一个使用共享内存的分块版本。通过将数据加载到共享内存并在线程间重用，我们大大减少了全局内存流量。`04-gpu-architecture.cu` 第 126 行的内核展示了这种优化。

## 常见的架构相关陷阱

**启动的块太少：** 如果你只启动与 SMs 数量相同的块，你无法隐藏延迟。启动比 SMs 多 2-4 倍的块。

**忽略对齐：** 内存访问应该尽可能对齐到 128 字节。未对齐的访问浪费带宽。

**线程块大小不是 32 的倍数：** 如果你每个块启动 100 个线程，你会在每个块的最后一个 warp 中浪费 28 个线程位置。使用 96 或 128。

**过度使用寄存器：** 用 `nvcc --ptxas-options=-v` 检查，如果占用率低就减少使用。

**不必要的同步：** `__syncthreads()` 很昂贵。只在线程真正需要协调时使用。

## 挑战练习

1. **测量合并的影响：** 编写一个简单内核的两个版本——一个完美合并，一个糟糕合并。在你的 GPU 上测量性能差异。

2. **优化占用率：** 采用一个使用许多寄存器的内核。使用共享内存或重构计算以减少寄存器压力。测量对性能的影响。

3. **测试 warp 分化：** 编写一个具有不同程度分化（无、50%、100%）的内核。测量性能影响并验证它与理论相符。

## 总结

GPU 架构与 CPU 架构有根本性的不同。理解 SMs、warps 和线程的层次结构对于编写高效代码至关重要。内存访问模式往往比计算更重要。合并访问高效利用内存带宽，而分散访问会浪费它。

共享内存使块中的线程能够协作，大大减少全局内存流量。Warps 以锁步执行，因此分化的代码路径会损害性能。占用率影响 GPU 隐藏内存延迟的能力。

GPU 性能的关键是保持所有硬件忙碌。启动足够的块来使 SMs 饱和。使用能够实现合并的内存访问模式。利用共享内存进行数据重用。尽可能避免 warp 分化。

## 下一步

继续学习**教程 05：神经网络前向传播**，看看架构理解如何应用于深度学习。你将在 GPU 上实现一个简单的神经网络，并学习矩阵乘法、激活函数和内存布局如何影响真实机器学习工作负载的性能。

## 延伸阅读

- [CUDA C++ 编程指南 - 硬件实现](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#hardware-implementation)
- [CUDA C++ 最佳实践指南](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [GPU 架构白皮书](https://www.nvidia.com/en-us/data-center/resources/gpu-architecture/)
- [CUDA 占用率计算器](https://docs.nvidia.com/cuda/cuda-occupancy-calculator/index.html)
