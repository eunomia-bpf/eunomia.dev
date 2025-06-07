# CUDA GPU组织层次结构

本文档提供了NVIDIA GPU架构和编程模型层次结构的全面概述，从硬件和软件两个角度进行介绍。

您可以在 <https://github.com/eunomia-bpf/basic-cuda-tutorial> 找到代码

## 目录

1. [硬件组织](#硬件组织)
2. [软件编程模型](#软件编程模型)
3. [内存层次结构](#内存层次结构)
4. [执行模型](#执行模型)
5. [性能考虑因素](#性能考虑因素)
6. [示例应用](#示例应用)

## 硬件组织

### GPU架构演变

NVIDIA GPU已经经历了多代架构的演变：

| 架构 | 示例GPU | 主要特性 |
|--------------|--------------|-------------|
| Tesla | GeForce 8/9/200系列 | 首批支持CUDA的GPU |
| Fermi | GeForce 400/500系列 | L1/L2缓存，改进的双精度 |
| Kepler | GeForce 600/700系列 | 动态并行，Hyper-Q |
| Maxwell | GeForce 900系列 | 改进的能效 |
| Pascal | GeForce 10系列，Tesla P100 | 统一内存改进，NVLink |
| Volta | Tesla V100 | Tensor核心，独立线程调度 |
| Turing | GeForce RTX 20系列 | RT核心，改进的Tensor核心 |
| Ampere | GeForce RTX 30系列，A100 | 第3代Tensor核心，稀疏性加速 |
| Hopper | H100 | 第4代Tensor核心，Transformer引擎 |
| Ada Lovelace | GeForce RTX 40系列 | RT改进，DLSS 3 |

### 硬件组件

一个现代NVIDIA GPU由以下部分组成：

1. **流式多处理器（SMs）**：基本计算单元
2. **Tensor核心**：专门用于矩阵运算（较新的GPU）
3. **RT核心**：专门用于光线追踪（RTX GPU）
4. **内存控制器**：与设备内存接口
5. **L2缓存**：在所有SM之间共享
6. **调度器**：管理线程块的执行

### 流式多处理器（SM）架构

每个SM包含：

- **CUDA核心**：整数和浮点算术单元
- **Tensor核心**：矩阵乘累加单元
- **线程束调度器**：管理线程执行
- **寄存器文件**：线程变量的超快存储
- **共享内存/L1缓存**：块中线程共享的快速内存
- **加载/存储单元**：处理内存操作
- **特殊功能单元（SFUs）**：计算超越函数（sin，cos等）
- **纹理单元**：专门用于纹理操作

![SM架构](https://developer.nvidia.com/blog/wp-content/uploads/2018/04/volta-architecture-768x756.png)
*示例SM架构（图表未包含，仅供参考）*

## 软件编程模型

CUDA程序以层次结构组织：

### 线程层次结构

1. **线程**：最小执行单元，运行程序实例
2. **线程束**：32个以锁步方式执行的线程组（SIMT）
3. **块**：可以通过共享内存协作的线程组
4. **网格**：执行相同内核的块的集合

```
网格
├── 块 (0,0)  块 (1,0)  块 (2,0)
├── 块 (0,1)  块 (1,1)  块 (2,1)
└── 块 (0,2)  块 (1,2)  块 (2,2)

块 (1,1)
├── 线程 (0,0)  线程 (1,0)  线程 (2,0)
├── 线程 (0,1)  线程 (1,1)  线程 (2,1)
└── 线程 (0,2)  线程 (1,2)  线程 (2,2)
```

### 线程索引

线程可以组织成1D、2D或3D排列。每个线程可以通过以下方式唯一标识：

```cuda
// 1D网格的1D块
int tid = blockIdx.x * blockDim.x + threadIdx.x;

// 2D网格的2D块
int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
int tid = tid_y * gridDim.x * blockDim.x + tid_x;

// 3D网格的3D块
int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
int tid_z = blockIdx.z * blockDim.z + threadIdx.z;
```

### 内核执行配置

内核以特定的网格和块配置启动：

```cuda
dim3 block(16, 16, 1);   // 每块16×16个线程
dim3 grid(N/16, N/16, 1); // 网格维度根据数据大小调整
myKernel<<<grid, block>>>(params...);
```

### 同步

- **块级**：`__syncthreads()`同步块中的所有线程
- **系统级**：`cudaDeviceSynchronize()`等待所有内核完成
- **流级**：`cudaStreamSynchronize(stream)`等待流中的操作
- **协作组**：更灵活的同步模式（较新的CUDA版本）

## 内存层次结构

GPU具有不同性能特性的复杂内存层次结构：

### 设备内存类型

1. **全局内存**
   - 最大容量（几GB）
   - 所有线程都可访问
   - 高延迟（数百个周期）
   - 用于主要数据存储
   - 带宽：根据GPU不同，约为500-2000 GB/s

2. **共享内存**
   - 小容量（在较新的GPU中每个SM最多164KB）
   - 块内的线程可访问
   - 低延迟（类似于L1缓存）
   - 用于线程间通信和数据重用
   - 组织成存储体以实现并行访问

3. **常量内存**
   - 小容量（每个设备64KB）
   - 对内核只读
   - 缓存和优化用于广播
   - 用于不变的参数

4. **纹理内存**
   - 缓存的只读内存
   - 针对2D/3D空间局部性优化
   - 硬件插值
   - 用于图像处理

5. **本地内存**
   - 每线程私有存储
   - 用于寄存器溢出
   - 实际上位于全局内存中
   - 自动变量数组通常存储在这里

6. **寄存器**
   - 最快的内存类型
   - 每线程私有存储
   - 每个线程数量有限
   - 用于线程本地变量

### 内存管理模型

1. **显式内存管理**
   ```cuda
   // 分配设备内存
   float *d_data;
   cudaMalloc(&d_data, size);
   
   // 将数据传输到设备
   cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
   
   // 启动内核
   kernel<<<grid, block>>>(d_data);
   
   // 将结果传回
   cudaMemcpy(h_result, d_data, size, cudaMemcpyDeviceToHost);
   
   // 释放设备内存
   cudaFree(d_data);
   ```

2. **统一内存**
   ```cuda
   // 分配统一内存
   float *data;
   cudaMallocManaged(&data, size);
   
   // 初始化数据（在主机上）
   for (int i = 0; i < N; i++) data[i] = i;
   
   // 启动内核（数据自动迁移）
   kernel<<<grid, block>>>(data);
   
   // 等待内核完成
   cudaDeviceSynchronize();
   
   // 访问结果（数据自动迁移回来）
   float result = data[0];
   
   // 释放统一内存
   cudaFree(data);
   ```

3. **零拷贝内存**
   ```cuda
   float *data;
   cudaHostAlloc(&data, size, cudaHostAllocMapped);
   
   float *d_data;
   cudaHostGetDevicePointer(&d_data, data, 0);
   
   kernel<<<grid, block>>>(d_data);
   ```

### 内存访问模式

1. **合并访问**：线程束中的线程访问连续内存
   ```cuda
   // 合并访问（高效）
   data[threadIdx.x] = value;
   ```

2. **跨步访问**：线程束中的线程以步长访问内存
   ```cuda
   // 跨步访问（低效）
   data[threadIdx.x * stride] = value;
   ```

3. **存储体冲突**：多个线程访问同一共享内存存储体
   ```cuda
   // 如果多个线程的threadIdx.x % 32相同，可能存在存储体冲突
   shared[threadIdx.x] = data[threadIdx.x];
   ```

## 执行模型

### SIMT执行

GPU使用单指令多线程（SIMT）执行方式，以32个线程（线程束）为组执行：

- 线程束中的所有线程执行相同的指令
- 分歧路径被序列化（线程束分歧）
- 对短条件段使用谓词技术

### 调度

1. **块调度**：
   - 根据资源将块分配给SM
   - 一旦分配，块在该SM上运行直到完成
   - 块之间不能相互通信

2. **线程束调度**：
   - 线程束是基本的调度单元
   - 硬件线程束调度器选择就绪的线程束执行
   - 通过线程束交错实现延迟隐藏

3. **指令级调度**：
   - 来自不同线程束的指令可以交错执行
   - 帮助隐藏内存和指令延迟

### 占用率

占用率是活动线程束与SM上最大可能线程束的比率：

- 受资源限制：寄存器、共享内存、块大小
- 更高的占用率通常改善延迟隐藏
- 与性能并非总是线性相关

影响占用率的因素：
- **每线程寄存器使用量**：更多寄存器 = 更少线程束
- **每块共享内存**：更多共享内存 = 更少块
- **块大小**：非常小的块会降低占用率

## 性能考虑因素

### 内存优化

1. **合并访问**：确保线程束中的线程访问连续内存
2. **共享内存**：用于块内重用的数据
3. **L1/纹理缓存**：利用于具有空间局部性的只读数据
4. **内存带宽**：通常是限制因素；最小化传输

### 执行优化

1. **占用率**：平衡资源使用以最大化活动线程束
2. **线程束分歧**：最小化线程束内的分歧路径
3. **指令混合**：平衡算术操作和内存访问
4. **内核融合**：将多个操作合并为一个内核以减少启动开销

### 常见优化技术

1. **分块**：将数据分成适合共享内存的瓦片
2. **循环展开**：减少循环开销
3. **预取**：在需要数据之前加载
4. **线程束洗牌**：在不使用共享内存的情况下交换线程束内线程之间的数据
5. **持久线程**：保持线程活动以处理多个工作项

## 示例应用

附带的`basic04.cu`演示了：

1. **硬件检查**：查询和显示设备属性
2. **线程层次结构**：可视化网格/块/线程结构
3. **内存类型**：使用全局、共享、常量、本地和寄存器内存
4. **内存访问模式**：演示合并与非合并访问
5. **线程束执行**：显示线程束ID、通道ID和分歧效果

### 关键代码部分

线程标识和层次结构：
```cuda
__global__ void threadHierarchyKernel() {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;
    
    // 打印线程位置
    printf("Thread (%d,%d,%d) in Block (%d,%d,%d)\n", tx, ty, tz, bx, by, bz);
}
```

共享内存使用：
```cuda
__global__ void sharedMemoryKernel(float *input, float *output) {
    __shared__ float sharedData[256];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int localId = threadIdx.x;
    
    // 将数据加载到共享内存
    sharedData[localId] = input[tid];
    
    // 同步
    __syncthreads();
    
    // 使用共享数据
    output[tid] = sharedData[localId];
}
```

## 进一步阅读

- [CUDA C++编程指南](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [CUDA最佳实践指南](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [专业CUDA C编程](https://www.amazon.com/Professional-CUDA-Programming-John-Cheng/dp/1118739329)
- [大规模并行处理器编程](https://www.amazon.com/Programming-Massively-Parallel-Processors-Hands/dp/0124159923) 