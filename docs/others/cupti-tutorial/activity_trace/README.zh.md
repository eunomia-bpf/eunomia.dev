# CUPTI 活动跟踪教程

> GitHub 仓库和完整教程可在 <https://github.com/eunomia-bpf/cupti-tutorial> 获取。

## 简介

对CUDA应用程序进行性能分析对于理解其性能特征至关重要。CUPTI Activity API 提供了一种强大的方法来收集CUDA API调用和GPU活动的详细跟踪信息。本教程解释如何使用CUPTI收集和分析这些数据。

## 您将学到什么

- 如何初始化CUPTI Activity API
- 设置和管理活动记录缓冲区
- 处理来自多个源的活动记录
- 解释活动数据以进行优化

## 代码演练

### 1. 设置活动跟踪

活动跟踪系统的核心围绕缓冲区管理展开。CUPTI请求缓冲区来存储活动记录，并在这些缓冲区被填满时通知。

```cpp
// 缓冲区请求回调 - 当CUPTI需要新缓冲区时调用
static void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords)
{
  // 为CUPTI记录分配缓冲区
  *size = BUF_SIZE;
  *buffer = (uint8_t *)malloc(*size + ALIGN_SIZE);
  
  // 确保缓冲区正确对齐
  *buffer = ALIGN_BUFFER(*buffer, ALIGN_SIZE);
  *maxNumRecords = 0;
}
```

当CUPTI请求缓冲区来存储活动记录时，此函数分配内存。对齐对性能很重要。

### 2. 处理已完成的缓冲区

```cpp
// 缓冲区完成回调 - 当CUPTI填满缓冲区时调用
static void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize)
{
  CUpti_Activity *record = NULL;
  
  // 处理缓冲区中的所有记录
  CUptiResult status = CUPTI_SUCCESS;
  while (validSize > 0) {
    status = cuptiActivityGetNextRecord(buffer, validSize, &record);
    if (status == CUPTI_SUCCESS) {
      printActivity(record);
      validSize -= record->common.size;
      buffer += record->common.size;
    }
    else
      break;
  }
  
  free(buffer);
}
```

当CUPTI用活动数据填满缓冲区时，此回调处理每个记录，然后释放缓冲区。

### 3. 活动记录处理

`printActivity` 函数是分析的核心，解释不同类型的活动：

```cpp
static void printActivity(CUpti_Activity *record)
{
  switch (record->kind) {
  case CUPTI_ACTIVITY_KIND_DEVICE:
    // 打印设备信息
    ...
  case CUPTI_ACTIVITY_KIND_MEMCPY:
    // 打印内存复制详细信息
    ...
  case CUPTI_ACTIVITY_KIND_KERNEL:
    // 打印内核执行详细信息
    ...
    
  // 更多活动类型...
  }
}
```

每种活动类型提供不同的见解：
- 设备活动显示硬件能力
- 内存复制活动显示数据传输模式和时间
- 内核活动显示执行时间和参数

### 4. 初始化和清理

```cpp
void initTrace()
{
  // 为缓冲区管理注册回调
  CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));
  
  // 启用各种活动类型
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DEVICE));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONTEXT));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));
  // ... 更多活动类型 ...
  
  // 捕获时间戳以标准化时间
  CUPTI_CALL(cuptiGetTimestamp(&startTimestamp));
}

void finiTrace()
{
  // 刷新任何剩余数据
  CUPTI_CALL(cuptiActivityFlushAll(0));
}
```

初始化函数启用您想要监控的特定活动类型并注册回调。清理函数确保处理所有数据。

### 5. 测试内核

示例使用简单的向量加法内核（在 `vec.cu` 中）来生成要跟踪的活动：

```cpp
__global__ void vecAdd(const float *A, const float *B, float *C, int numElements)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < numElements)
    C[i] = A[i] + B[i];
}
```

## 运行示例

1. 构建示例：
   ```bash
   make
   ```

2. 运行活动跟踪：
   ```bash
   ./activity_trace
   ```

## 理解输出

输出显示活动的时间顺序跟踪：

```
设备 Device Name (0)，计算能力 7.0，全局内存（带宽 900 GB/s，大小 16000 MB），多处理器 80，时钟 1530 MHz
上下文 1，设备 0，计算 API CUDA，NULL 流 1
驱动器_API cuCtxCreate [ 10223 - 15637 ] 
内存复制 HtoD [ 22500 - 23012 ] 设备 0，上下文 1，流 7，相关性 1/1
内核 "vecAdd" [ 32058 - 35224 ] 设备 0，上下文 1，流 7，相关性 2
内存复制 DtoH [ 40388 - 41002 ] 设备 0，上下文 1，流 7，相关性 3/3
```

让我们解码这些信息：
1. **设备信息**：显示GPU能力
2. **上下文创建**：CUDA上下文初始化
3. **内存复制**：
   - `HtoD`（主机到设备）显示数据正在上传到GPU
   - `DtoH`（设备到主机）显示结果正在下载
4. **内核执行**：显示我们向量加法的执行时间

时间戳（方括号中）标准化为跟踪开始时的时间，使得容易看到操作的相对时序。

## 性能见解

使用这些跟踪数据，您可以：
- 识别内存传输中的瓶颈
- 确定内核执行效率
- 找到同步点及其影响
- 测量CUDA API调用的开销

## 下一步

- 尝试修改向量大小以查看它如何影响性能
- 启用其他活动类型以收集更详细的信息
- 比较您自己应用程序中不同GPU操作的时序
- 探索CUPTI的其他基于活动的示例以获得更高级的跟踪功能 