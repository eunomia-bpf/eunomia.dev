# CUDA 基础示例 - 向量加法解释

本文档提供了`basic01.cu`中向量加法CUDA示例的详细解释。

您可以在 <https://github.com/eunomia-bpf/basic-cuda-tutorial> 找到代码

## 前提条件

要运行此示例，您需要：
- 支持CUDA的NVIDIA GPU
- 已安装NVIDIA CUDA工具包
- 与您的CUDA版本兼容的C++编译器
- GNU Make（用于使用提供的Makefile进行构建）

## 构建和运行

1. 构建示例：
```bash
make
```

2. 运行程序：
```bash
./basic01
```

## 代码结构和解释

### 1. 头文件和包含
```cuda
#include <stdio.h>
#include <stdlib.h>
```
这些标准C头文件提供：
- `stdio.h`：输入/输出函数，如`printf`
- `stdlib.h`：内存管理函数，如`malloc`和`free`

### 2. CUDA内核函数
```cuda
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements)
```

- `__global__`：指定这是一个CUDA内核函数，它：
  - 在GPU上运行
  - 可以从CPU代码调用
  - 必须返回void
- 参数：
  - `float *A, *B`：GPU内存中的输入向量
  - `float *C`：GPU内存中的输出向量
  - `numElements`：向量的大小

内核内部：
```cuda
int i = blockDim.x * blockIdx.x + threadIdx.x;
```
这计算每个线程的唯一索引，其中：
- `threadIdx.x`：块内的线程索引（0到blockDim.x-1）
- `blockIdx.x`：网格内的块索引
- `blockDim.x`：每个块的线程数

### 3. 主函数组件

#### 3.1 内存分配
```cuda
// 主机内存分配
float *h_A = (float *)malloc(size);  // CPU内存

// 设备内存分配
float *d_A = NULL;
cudaMalloc((void **)&d_A, size);     // GPU内存
```

- 主机（CPU）内存使用标准C的`malloc`
- 设备（GPU）内存使用CUDA的`cudaMalloc`
- 'h_'前缀表示主机内存
- 'd_'前缀表示设备内存

#### 3.2 数据传输
```cuda
cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
```

`cudaMemcpy`参数：
1. 目标指针
2. 源指针
3. 字节大小
4. 传输方向：
   - `cudaMemcpyHostToDevice`：CPU到GPU
   - `cudaMemcpyDeviceToHost`：GPU到CPU

#### 3.3 内核启动配置
```cuda
int threadsPerBlock = 256;
int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
```

- `threadsPerBlock = 256`：常见大小，性能良好
- `blocksPerGrid`：计算以确保有足够的线程处理所有元素
- 公式`(numElements + threadsPerBlock - 1) / threadsPerBlock`向上取整除法
- 启动语法`<<<blocks, threads>>>`指定执行配置

#### 3.4 错误检查
```cuda
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    fprintf(stderr, "Failed to launch kernel: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
}
```

在内核启动和CUDA API调用后始终检查CUDA错误。

#### 3.5 结果验证
```cuda
for (int i = 0; i < numElements; ++i) {
    if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
        fprintf(stderr, "Result verification failed at element %d!\n", i);
        exit(EXIT_FAILURE);
    }
}
```

通过与CPU结果比较来验证GPU计算。

#### 3.6 清理
```cuda
// 释放GPU内存
cudaFree(d_A);
cudaFree(d_B);
cudaFree(d_C);

// 释放CPU内存
free(h_A);
free(h_B);
free(h_C);
```

始终释放分配的内存以防止内存泄漏。

## 性能考虑

1. **线程块大小**
   - 我们每个块使用256个线程
   - 这是一个在大多数GPU上都能良好工作的常见选择
   - 2的幂通常是高效的

2. **内存合并访问**
   - 相邻线程访问相邻内存位置
   - 这种模式实现高效内存访问

3. **错误检查**
   - 代码包含健壮的错误检查
   - 对调试和可靠性很重要

## 常见问题和调试

1. **CUDA安装**
   - 确保CUDA工具包正确安装
   - 检查`nvcc --version`能够正常工作
   - 使用`nvidia-smi`验证GPU兼容性

2. **编译错误**
   - 检查CUDA路径是否在系统PATH中
   - 验证GPU计算能力与Makefile中的`-arch`标志匹配

3. **运行时错误**
   - 内存不足：减小向量大小
   - 内核启动失败：检查GPU可用性
   - 结果不正确：验证索引计算

## 预期输出

成功运行时，您应该看到：
```
Vector addition of 50000 elements
CUDA kernel launch with 196 blocks of 256 threads
Test PASSED
Done
```

## 修改示例

要试验代码：

1. 更改向量大小（`numElements`）
2. 修改每个块的线程数
3. 添加时间测量
4. 尝试不同的数据类型
5. 实现其他向量操作

修改后记得处理错误并验证结果。 