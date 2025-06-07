# basic-cuda-tutorial

您可以在 <https://github.com/eunomia-bpf/basic-cuda-tutorial> 找到代码

这是一个CUDA编程示例集合，用于学习使用NVIDIA CUDA进行GPU编程。

请确保在Makefile中将GPU架构`sm_61`更改为您自己的GPU架构。

## 示例和教程

- **01-vector-addition.cu** 和 [01-vector-addition.md](01-vector-addition.md)：通过向量加法示例介绍CUDA编程
- **02-ptx-assembly.cu** 和 [02-ptx-assembly.md](02-ptx-assembly.md)：使用向量乘法示例演示CUDA PTX内联汇编
- **03-gpu-programming-methods.cu** 和 [03-gpu-programming-methods.md](03-gpu-programming-methods.md)：全面比较GPU编程方法，包括CUDA、PTX、Thrust、统一内存、共享内存、CUDA流和动态并行，使用矩阵乘法作为示例
- **04-gpu-architecture.cu** 和 [04-gpu-architecture.md](04-gpu-architecture.md)：详细探索GPU组织层次结构，包括硬件架构、线程/块/网格结构、内存层次结构和执行模型
- **05-neural-network.cu** 和 [05-neural-network.md](05-neural-network.md)：在GPU上使用CUDA实现基本神经网络前向传播
- **06-cnn-convolution.cu** 和 [06-cnn-convolution.md](06-cnn-convolution.md)：用于CNN的GPU加速卷积操作，采用共享内存优化
- **07-attention-mechanism.cu** 和 [07-attention-mechanism.md](07-attention-mechanism.md)：Transformer模型注意力机制的CUDA实现
- **08-profiling-tracing.cu** 和 [08-profiling-tracing.md](08-profiling-tracing.md)：使用CUDA Events、NVTX和CUPTI进行CUDA应用程序的性能分析和跟踪，以优化性能
- **09-gpu-extension.cu** 和 [09-gpu-extension.md](09-gpu-extension.md)：GPU应用程序扩展机制，用于在不更改源代码的情况下修改行为，包括API拦截、内存管理、内核优化和错误恢复能力
- **10-cpu-gpu-profiling-boundaries.cu** 和 [10-cpu-gpu-profiling-boundaries.md](10-cpu-gpu-profiling-boundaries.md)：高级GPU内核插桩技术，演示了CUDA内核内的细粒度内部计时、分歧路径分析、动态工作负载分析和自适应算法选择
- **11-fine-grained-gpu-modifications.cu** 和 [11-fine-grained-gpu-modifications.md](11-fine-grained-gpu-modifications.md)：细粒度GPU代码定制，包括数据结构布局优化、warp级原语、内存访问模式、内核融合和动态执行路径选择
- **12-advanced-gpu-customizations.cu** 和 [12-advanced-gpu-customizations.md](12-advanced-gpu-customizations.md)：高级GPU定制技术，包括线程分歧缓解、寄存器使用优化、混合精度计算、用于负载平衡的持久线程和warp专用化模式
- **13-low-latency-gpu-packet-processing.cu** 和 [13-low-latency-gpu-packet-processing.md](13-low-latency-gpu-packet-processing.md)：GPU基础网络数据包处理的延迟最小化技术，包括固定内存、零拷贝内存、流水线、持久内核和CUDA图，用于实时网络应用

每个教程都包含全面的文档，解释在GPU上进行机器学习/人工智能工作负载时使用的概念、实现细节和优化技术。 