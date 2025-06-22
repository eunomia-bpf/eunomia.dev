# CUPTI 异步活动跟踪教程

> GitHub 仓库和完整教程可在 <https://github.com/eunomia-bpf/cupti-tutorial> 获取。

## 简介

如果不谨慎处理，GPU应用程序的性能分析可能会显著影响性能。本教程演示如何使用CUPTI的异步活动跟踪来收集性能数据，同时将对应用程序执行时间的影响降到最低。您将学习如何在应用程序继续全速运行的同时收集详细的GPU和API活动跟踪。

## 您将学到什么

- 如何使用CUPTI设置异步缓冲区处理
- 非阻塞性能数据收集技术
- 在独立线程中处理性能分析信息
- 最小化性能分析对性能的影响

## 异步 vs 同步跟踪

在深入代码之前，让我们了解为什么异步跟踪很重要：

| 同步跟踪 | 异步跟踪 |
|---------|----------|
| 在缓冲区处理期间阻塞应用程序 | 在缓冲区处理期间应用程序继续运行 |
| 实现简单 | 需要线程安全处理 |
| 性能影响较高 | 性能影响较低 |
| 适用于短期测试运行 | 适用于生产环境或长期运行的应用程序 |

## 代码演练

### 1. 设置异步缓冲区处理

异步跟踪的关键是配置CUPTI使用独立线程进行缓冲区管理：

```cpp
void initTrace()
{
  // 启用异步缓冲区处理
  CUpti_ActivityAttribute attr = CUPTI_ACTIVITY_ATTR_CALLBACKS;
  CUpti_BuffersCallbackRequestFunc bufferRequested = bufferRequestedCallback;
  CUpti_BuffersCallbackCompleteFunc bufferCompleted = bufferCompletedCallback;
  
  CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));
  
  // 启用您想要跟踪的活动类型
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DEVICE));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONTEXT));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  
  // 捕获时间戳以标准化时间
  CUPTI_CALL(cuptiGetTimestamp(&startTimestamp));
}
```

### 2. 异步缓冲区请求回调

缓冲区请求回调与同步版本类似，但现在必须是线程安全的：

```cpp
static void CUPTIAPI
bufferRequestedCallback(uint8_t **buffer, size_t *size, size_t *maxNumRecords)
{
  *size = BUF_SIZE;
  
  // 分配一个带有额外对齐空间的缓冲区
  *buffer = (uint8_t *)malloc(*size + ALIGN_SIZE);
  if (*buffer == NULL) {
    printf("错误：内存不足\n");
    exit(-1);
  }
  
  // 将缓冲区对齐到ALIGN_SIZE
  *buffer = ALIGN_BUFFER(*buffer, ALIGN_SIZE);
  
  // 对记录数量没有限制
  *maxNumRecords = 0;
}
```

### 3. 异步缓冲区完成回调

最重要的部分是缓冲区完成回调，它现在在独立线程中运行：

```cpp
static void CUPTIAPI
bufferCompletedCallback(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize)
{
  CUptiResult status;
  CUpti_Activity *record = NULL;
  
  // 此回调在独立线程中运行，因此线程安全性至关重要
  
  // 处理缓冲区中的所有记录
  do {
    status = cuptiActivityGetNextRecord(buffer, validSize, &record);
    if (status == CUPTI_SUCCESS) {
      // 处理记录 - 必须快速且非阻塞
      printActivity(record);
      
      // 移动到下一个记录
      validSize -= record->common.size;
      buffer += record->common.size;
    }
    else if (status != CUPTI_ERROR_MAX_LIMIT_REACHED) {
      // 处理任何错误
      CUPTI_CALL(status);
    }
  } while (status == CUPTI_SUCCESS);
  
  // 释放缓冲区 - 必须是bufferRequested返回的相同指针
  free(buffer);
}
```

**关于此回调的重要注意事项：**
- 它在与您的应用程序不同的线程中运行
- 它应该快速返回以避免减慢缓冲区处理速度
- 在此回调中访问的任何数据结构都必须是线程安全的
- 必须在此回调中释放缓冲区以避免内存泄漏

### 4. 记录处理

记录处理与同步版本类似：

```cpp
static void
printActivity(CUpti_Activity *record)
{
  switch (record->kind) {
  case CUPTI_ACTIVITY_KIND_DEVICE:
    {
      // 打印设备信息
      CUpti_ActivityDevice2 *device = (CUpti_ActivityDevice2 *)record;
      printf("设备 %s (%u)，计算能力 %u.%u，...\n",
             device->name, device->id,
             device->computeCapabilityMajor, device->computeCapabilityMinor);
      break;
    }
  case CUPTI_ACTIVITY_KIND_KERNEL:
    {
      // 打印内核信息
      CUpti_ActivityKernel4 *kernel = (CUpti_ActivityKernel4 *)record;
      printf("内核 \"%s\" [ %llu - %llu ] 设备 %u，上下文 %u，流 %u\n",
             kernel->name,
             (unsigned long long)(kernel->start - startTimestamp),
             (unsigned long long)(kernel->end - startTimestamp),
             kernel->deviceId, kernel->contextId, kernel->streamId);
      break;
    }
  // ... 其他活动类型 ...
  }
}
```

### 5. 测试内核和主函数

示例包含一个简单的向量加法内核来生成性能分析数据：

```cpp
__global__ void vecAdd(const float *A, const float *B, float *C, int numElements)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < numElements)
    C[i] = A[i] + B[i];
}

int main(int argc, char *argv[])
{
  // 初始化CUPTI跟踪
  initTrace();
  
  // 分配和初始化向量
  // ...
  
  // 启动内核
  dim3 threadsPerBlock(256);
  dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x);
  vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
  
  // 将结果复制回主机
  cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
  
  // 验证结果
  // ...
  
  // 完成跟踪 - 确保所有数据都被刷新
  finiTrace();
  
  return 0;
}
```

### 6. 完成跟踪

当应用程序完成时，确保刷新所有剩余的缓冲区：

```cpp
void finiTrace()
{
  // 刷新缓冲区中仍然存在的任何活动记录
  CUPTI_CALL(cuptiActivityFlushAll(0));
}
```

## 运行示例

1. 构建示例：
   ```bash
   make
   ```

2. 运行异步跟踪器：
   ```bash
   ./activity_trace_async
   ```

## 理解输出

输出显示所有GPU活动及其时间戳：

```
设备 Device Name (0)，计算能力 7.0，全局内存（带宽 900 GB/s，大小 16000 MB），多处理器 80，时钟 1530 MHz
上下文 1，设备 0，计算 API CUDA，NULL 流 1
驱动器_API cuCtxCreate [ 10223 - 15637 ] 
内存复制 HtoD [ 22500 - 23012 ] 设备 0，上下文 1，流 7，相关性 1/1
内核 "vecAdd" [ 32058 - 35224 ] 设备 0，上下文 1，流 7，相关性 2
内存复制 DtoH [ 40388 - 41002 ] 设备 0，上下文 1流 7，相关性 3/3
```

时间戳（方括号中）显示每个操作的开始和结束时间，标准化为跟踪开始时的时间。

## 性能考虑

异步跟踪提供几个性能优势：

1. **减少开销**：主应用程序线程继续运行，而活动记录在独立线程中处理
2. **降低延迟**：CUDA操作不会因等待缓冲区处理而被阻塞
3. **更好的可扩展性**：适用于长期运行的应用程序或生产监控

## 高级使用技巧

1. **缓冲区大小调优**：根据您的应用程序调整`BUF_SIZE`：
   - 较大的缓冲区减少回调频率但使用更多内存
   - 较小的缓冲区使用更少内存但可能增加回调开销

2. **选择性跟踪**：只启用您需要的活动类型：
   ```cpp
   // 如果您只关心内核，就启用这些
   cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL);
   cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
   ```

3. **线程安全**：实现您自己的版本时，确保缓冲区完成回调访问的任何数据结构都是线程安全的。

## 下一步

- 尝试启用不同的活动类型以查看应用程序的不同方面
- 修改代码以将记录存储在数据库或文件中以供后续分析
- 实现自定义可视化工具以帮助理解时间线数据
- 尝试在您自己的CUDA应用程序上运行异步跟踪器 