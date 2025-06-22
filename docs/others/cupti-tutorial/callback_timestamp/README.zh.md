# CUPTI 时间戳回调教程

> 完整的 GitHub 仓库和教程请访问 <https://github.com/eunomia-bpf/cupti-tutorial>。

## 简介

准确测量 CUDA 操作的执行时间对于性能优化至关重要。虽然基于 CPU 的计时可以给出近似结果，但 GPU 时间戳提供了操作在设备上实际执行时间的精确测量。本教程演示如何使用 CUPTI 回调来收集 CUDA 操作的 GPU 时间戳，为内存传输和内核执行提供准确的时序信息。

## 学习内容

- 如何为 CUDA 运行时 API 函数设置 CUPTI 回调
- 在操作开始和结束时收集精确的 GPU 时间戳
- 测量内存传输和内核的执行时间
- 分析完整 CUDA 工作流的性能

## 理解 GPU 时间戳

GPU 时间戳在几个重要方面与 CPU 时间戳不同：

1. 它们从 GPU 的角度测量时间
2. 它们提供纳秒级精度
3. 它们准确捕获操作在设备上执行的时间，而不仅仅是提交的时间
4. 它们对于理解 GPU 操作的真实性能特征至关重要

## 代码详解

### 1. 设置回调系统

首先，我们需要初始化 CUPTI 并注册我们的回调函数：

```cpp
int main(int argc, char *argv[])
{
    // 初始化 CUDA
    RUNTIME_API_CALL(cudaSetDevice(0));
    
    // 订阅 CUPTI 回调
    CUpti_SubscriberHandle subscriber;
    CUPTI_CALL(cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)getTimestampCallback, &traceData));
    
    // 启用 CUDA 运行时的回调
    CUPTI_CALL(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API));
    
    // 运行我们的测试工作负载
    runTest();
    
    // 取消订阅回调
    CUPTI_CALL(cuptiUnsubscribe(subscriber));
    
    return 0;
}
```

这设置了 CUPTI，在调用 CUDA 运行时 API 函数时调用我们的 `getTimestampCallback` 函数。

### 2. 回调函数

示例的核心是收集时间戳的回调函数：

```cpp
void CUPTIAPI getTimestampCallback(void *userdata, CUpti_CallbackDomain domain,
                                  CUpti_CallbackId cbid, const CUpti_CallbackData *cbInfo)
{
    RuntimeApiTrace_t *traceData = (RuntimeApiTrace_t*)userdata;
    
    // 只处理运行时 API 回调
    if (domain != CUPTI_CB_DOMAIN_RUNTIME_API) return;
    
    // 我们对内存传输和内核启动感兴趣
    if ((cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020) ||
        (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020) ||
        (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaDeviceSynchronize_v3020)) {
        
        if (cbInfo->callbackSite == CUPTI_API_ENTER) {
            // 在 API 调用开始时记录信息
            uint64_t timestamp;
            CUPTI_CALL(cuptiDeviceGetTimestamp(cbInfo->context, &timestamp));
            
            // 存储函数名
            traceData->functionName = cbInfo->functionName;
            traceData->startTimestamp = timestamp;
            
            // 对于内存传输，捕获大小和方向
            if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020) {
                cudaMemcpy_v3020_params *params = 
                    (cudaMemcpy_v3020_params*)cbInfo->functionParams;
                traceData->memcpyBytes = params->count;
                traceData->memcpyKind = params->kind;
            }
        }
        else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
            // 在 API 调用结束时记录信息
            uint64_t timestamp;
            CUPTI_CALL(cuptiDeviceGetTimestamp(cbInfo->context, &timestamp));
            
            // 计算持续时间
            traceData->gpuTime = timestamp - traceData->startTimestamp;
            
            // 打印时序信息
            printTimestampData(traceData);
        }
    }
}
```

此函数的关键方面：
1. 它过滤我们想要计时的特定 CUDA 函数
2. 在 API 进入时，它记录函数名、开始时间戳和任何相关参数
3. 在 API 退出时，它记录结束时间戳并计算持续时间
4. 对于内存传输，它捕获大小和方向

### 3. 测试工作负载

为了演示计时，我们运行一个简单的向量加法工作流：

```cpp
void runTest()
{
    int N = 50000;
    size_t size = N * sizeof(float);
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    
    // 分配主机内存
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);
    
    // 初始化主机数组
    for (int i = 0; i < N; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }
    
    // 分配设备内存
    RUNTIME_API_CALL(cudaMalloc((void**)&d_A, size));
    RUNTIME_API_CALL(cudaMalloc((void**)&d_B, size));
    RUNTIME_API_CALL(cudaMalloc((void**)&d_C, size));
    
    // 从主机传输数据到设备（这些将被计时）
    RUNTIME_API_CALL(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    RUNTIME_API_CALL(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
    
    // 启动内核（这将被计时）
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    
    // 将结果传输回主机（这将被计时）
    RUNTIME_API_CALL(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));
    
    // 同步以确保所有操作完成
    RUNTIME_API_CALL(cudaDeviceSynchronize());
    
    // 清理
    free(h_A);
    free(h_B);
    free(h_C);
    RUNTIME_API_CALL(cudaFree(d_A));
    RUNTIME_API_CALL(cudaFree(d_B));
    RUNTIME_API_CALL(cudaFree(d_C));
}
```

此工作流包括：
1. 两个主机到设备的内存传输
2. 一个内核执行
3. 一个设备到主机的内存传输
4. 一个设备同步

### 4. 显示时序结果

结果以格式化表格显示：

```cpp
void printTimestampData(RuntimeApiTrace_t *traceData)
{
    static bool headerPrinted = false;
    
    if (!headerPrinted) {
        printf("\n开始时间戳/GPU时间以纳秒报告\n\n");
        printf("%-15s %-24s %-15s %-8s %-10s\n", "名称", "开始时间", "GPU时间", "字节数", "类型");
        headerPrinted = true;
    }
    
    // 打印函数名和时序信息
    printf("%-15s %-24llu %-15llu", 
           traceData->functionName,
           (unsigned long long)traceData->startTimestamp,
           (unsigned long long)traceData->gpuTime);
    
    // 对于内存传输，打印大小和方向
    if (strcmp(traceData->functionName, "cudaMemcpy") == 0) {
        printf(" %-8lu %-10s", 
               (unsigned long)traceData->memcpyBytes,
               getMemcpyKindString((cudaMemcpyKind)traceData->memcpyKind));
    } else {
        printf(" %-8s %-10s", "不适用", "不适用");
    }
    
    printf("\n");
}
```

## 运行教程

1. 构建示例：
   ```bash
   make
   ```

2. 运行时间戳收集器：
   ```bash
   ./callback_timestamp
   ```

## 理解输出

示例产生类似的输出：

```
开始时间戳/GPU时间以纳秒报告

名称            开始时间                GPU时间        字节数   类型
cudaMemcpy      123456789012            5432           200000   主机到设备
cudaMemcpy      123456794444            5432           200000   主机到设备
VecAdd          123456799876            10864          不适用    不适用
cudaMemcpy      123456810740            5432           200000   设备到主机
cudaDeviceSync  123456816172            0              不适用    不适用
```

让我们分析这个输出：

1. **开始时间**：操作开始时的 GPU 时间戳（以纳秒为单位）
2. **GPU 时间**：GPU 上操作的持续时间（以纳秒为单位）
3. **字节数**：对于内存传输，传输的数据量
4. **类型**：对于内存传输，方向（主机到设备或设备到主机）

从这个输出我们可以看到：
- 内存传输每次大约需要 5.4 微秒
- 内核执行大约需要 10.8 微秒
- 整个工作流在大约 27 微秒内完成

## 性能洞察

这个时序数据揭示了 CUDA 性能的几个重要方面：

1. **内存传输开销**：内存传输可能是一个重要的瓶颈
2. **内核执行时间**：GPU 上的实际计算时间
3. **操作序列**：工作流中操作的顺序和时序

### 优化机会识别

基于时序数据，我们可以识别：

1. **内存带宽利用率**：
   ```cpp
   float bandwidth = (bytes / 1e9) / (time_seconds);
   printf("内存带宽：%.1f GB/s\n", bandwidth);
   ```

2. **计算密度**：
   ```cpp
   float computeRatio = kernelTime / totalTime;
   printf("计算时间比例：%.1f%%\n", computeRatio * 100);
   ```

3. **API 开销**：
   ```cpp
   float apiOverhead = (apiExitTime - apiEnterTime) - gpuTime;
   printf("API 开销：%.2f 微秒\n", apiOverhead / 1000.0);
   ```

## 高级时序分析

### 多流分析

对于复杂的应用程序，您可能需要分析多个 CUDA 流：

```cpp
struct StreamTimeline {
    int streamId;
    std::vector<OperationTiming> operations;
    
    void analyzeOverlap() {
        // 分析流之间的重叠
        for (size_t i = 0; i < operations.size() - 1; i++) {
            if (operations[i].endTime > operations[i+1].startTime) {
                printf("流 %d：操作重叠检测\n", streamId);
            }
        }
    }
};
```

### 内存传输模式

分析内存访问模式：

```cpp
class MemoryAnalyzer {
private:
    std::map<void*, size_t> allocationSizes;
    std::vector<MemoryTransfer> transfers;

public:
    void analyzePattern() {
        size_t totalHostToDevice = 0;
        size_t totalDeviceToHost = 0;
        
        for (const auto& transfer : transfers) {
            if (transfer.kind == cudaMemcpyHostToDevice) {
                totalHostToDevice += transfer.bytes;
            } else if (transfer.kind == cudaMemcpyDeviceToHost) {
                totalDeviceToHost += transfer.bytes;
            }
        }
        
        printf("总主机到设备传输：%zu MB\n", totalHostToDevice / (1024*1024));
        printf("总设备到主机传输：%zu MB\n", totalDeviceToHost / (1024*1024));
        printf("数据传输比率：%.2f\n", 
               (double)totalDeviceToHost / totalHostToDevice);
    }
};
```

## 最佳实践

### 时序精度

1. **使用 GPU 时间戳**：始终使用 `cuptiDeviceGetTimestamp` 而不是 CPU 时钟
2. **避免同步开销**：最小化不必要的同步点
3. **考虑时钟域**：GPU 和 CPU 时钟可能不同步

### 性能影响

1. **最小化回调开销**：只跟踪关键操作
2. **批量数据收集**：避免在回调中进行昂贵的操作
3. **后处理分析**：将详细分析推迟到执行后

### 数据验证

```cpp
bool validateTimingData(const TimingResult& result) {
    // 检查合理的时间范围
    if (result.duration < 0 || result.duration > MAX_REASONABLE_TIME) {
        return false;
    }
    
    // 验证内存传输速率
    if (result.isMemoryTransfer) {
        float bandwidth = calculateBandwidth(result);
        if (bandwidth > MAX_MEMORY_BANDWIDTH) {
            printf("警告：内存带宽超出预期：%.1f GB/s\n", bandwidth);
            return false;
        }
    }
    
    return true;
}
```

这个时间戳回调教程为精确测量和分析 CUDA 应用程序性能提供了强大的基础。 