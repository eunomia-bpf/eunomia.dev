# CUPTI 统一内存分析教程

> GitHub 仓库和完整教程可在 <https://github.com/eunomia-bpf/cupti-tutorial> 获取。

## 简介

CUDA 统一内存创建了一个可被 CPU 和 GPU 访问的单一内存空间，简化了异构计算中的内存管理。虽然这种抽象使编程更容易，但它引入了幕后的数据迁移，可能显著影响性能。本教程演示如何使用 CUPTI 分析统一内存操作，帮助您理解内存迁移模式并优化应用程序以获得更好的性能。

## 您将学到什么

- 如何跟踪和分析统一内存事件
- 监控 CPU 和 GPU 之间的页面错误和数据迁移
- 理解不同内存访问模式的性能影响
- 应用内存建议来优化数据放置
- 使用 CUPTI 深入了解统一内存行为

## 理解统一内存

统一内存提供了一个可以从 CPU 和 GPU 代码访问的单一指针。CUDA 运行时根据需要自动在主机和设备之间迁移数据。这个过程涉及：

1. **页面错误**：当 CPU 或 GPU 访问本地不存在的内存时
2. **数据迁移**：在主机和设备之间移动内存页面
3. **内存驻留**：跟踪内存页面当前所在的位置
4. **访问计数器**：监控内存访问模式

在 Pascal 及更新的 GPU 上，硬件页面错误启用细粒度迁移，而较老的 GPU 使用更粗粒度的方法。

## 代码演练

### 1. 为统一内存分析设置 CUPTI

首先，我们需要配置 CUPTI 来跟踪统一内存事件：

```cpp
void setupUnifiedMemoryProfiling()
{
    // 初始化 CUPTI
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMORY));
    
    // 为活动缓冲区注册回调
    CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));
    
    // 配置要跟踪的统一内存事件
    CUpti_ActivityUnifiedMemoryCounterConfig config[2];
    
    // 跟踪 CPU 页面错误
    config[0].scope = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_PROCESS_SINGLE_DEVICE;
    config[0].kind = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT;
    config[0].deviceId = 0;
    config[0].enable = 1;
    
    // 跟踪 GPU 页面错误
    config[1].scope = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_PROCESS_SINGLE_DEVICE;
    config[1].kind = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT_COUNT;
    config[1].deviceId = 0;
    config[1].enable = 1;
    
    // 配置计数器
    CUPTI_CALL(cuptiActivityConfigureUnifiedMemoryCounter(config, 2));
}
```

这段代码：
1. 启用统一内存事件的 CUPTI 活动跟踪
2. 设置回调来处理活动缓冲区
3. 配置 CPU 和 GPU 页面错误的特定计数器

### 2. 分配统一内存

接下来，我们使用统一内存 API 分配内存：

```cpp
void allocateUnifiedMemory(void **data, size_t size)
{
    // 分配可从 CPU 和 GPU 访问的统一内存
    RUNTIME_API_CALL(cudaMallocManaged(data, size));
    
    // 获取设备属性以检查硬件页面错误支持
    cudaDeviceProp prop;
    RUNTIME_API_CALL(cudaGetDeviceProperties(&prop, 0));
    
    // 检查 GPU 是否支持硬件页面错误
    bool hasPageFaultSupport = (prop.major >= 6);
    printf("GPU %s硬件页面错误支持\n", 
           hasPageFaultSupport ? "具有" : "不具有");
}
```

这个函数：
1. 使用 `cudaMallocManaged` 分配内存
2. 检查 GPU 是否支持硬件页面错误（Pascal 或更新）

### 3. 测试不同的访问模式

为了演示访问模式如何影响统一内存性能，我们实现几个测试用例：

```cpp
void testSequentialAccess(float *data, size_t size)
{
    printf("\n测试顺序访问（先 CPU 后 GPU）：\n");
    
    // 首先在 CPU 上访问数据
    for (size_t i = 0; i < size/sizeof(float); i++) {
        data[i] = i;
    }
    
    // 同步以确保 CPU 操作完成
    RUNTIME_API_CALL(cudaDeviceSynchronize());
    
    // 然后在 GPU 上访问
    vectorAdd<<<(size/sizeof(float) + 255)/256, 256>>>(data, data, data, size/sizeof(float));
    
    // 等待 GPU 完成
    RUNTIME_API_CALL(cudaDeviceSynchronize());
}

void testPrefetchedAccess(float *data, size_t size)
{
    printf("\n测试预取访问（使用 cudaMemPrefetchAsync）：\n");
    
    // 在 CPU 上初始化数据
    for (size_t i = 0; i < size/sizeof(float); i++) {
        data[i] = i;
    }
    
    // 在使用前将数据预取到 GPU
    RUNTIME_API_CALL(cudaMemPrefetchAsync(data, size, 0));
    
    // 在 GPU 上访问
    vectorAdd<<<(size/sizeof(float) + 255)/256, 256>>>(data, data, data, size/sizeof(float));
    
    // 等待 GPU 完成
    RUNTIME_API_CALL(cudaDeviceSynchronize());
}

void testConcurrentAccess(float *data, size_t size)
{
    printf("\n测试并发访问（CPU 和 GPU）：\n");
    
    // 使用内存建议来优化并发访问
    RUNTIME_API_CALL(cudaMemAdvise(data, size, cudaMemAdviseSetAccessedBy, 0));
    
    // 启动 GPU 内核
    vectorAdd<<<(size/sizeof(float) + 255)/256, 256>>>(data, data, data, size/sizeof(float));
    
    // 当 GPU 运行时，从 CPU 访问部分数据
    for (size_t i = 0; i < size/(2*sizeof(float)); i++) {
        data[i] = i;
    }
    
    // 等待所有操作完成
    RUNTIME_API_CALL(cudaDeviceSynchronize());
}
```

这些函数演示：
1. 先从 CPU 后从 GPU 的顺序访问
2. 在使用前将数据预取到 GPU
3. CPU 和 GPU 的并发访问

### 4. 处理统一内存事件

当 CUPTI 收集活动记录时，我们处理它们以提取统一内存信息：

```cpp
void processUnifiedMemoryActivity(CUpti_Activity *record)
{
    switch (record->kind) {
    case CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER:
    {
        CUpti_ActivityUnifiedMemoryCounter *umcRecord = 
            (CUpti_ActivityUnifiedMemoryCounter *)record;
        
        // 根据计数器类型处理
        switch (umcRecord->counterKind) {
        case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT:
            cpuPageFaults += umcRecord->value;
            printf("  CPU 页面错误：%llu\n", (unsigned long long)umcRecord->value);
            break;
            
        case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT_COUNT:
            gpuPageFaults += umcRecord->value;
            printf("  GPU 页面错误：%llu\n", (unsigned long long)umcRecord->value);
            break;
            
        case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD:
            bytesTransferredHtoD += umcRecord->value;
            printf("  主机到设备传输：%llu 字节\n", 
                   (unsigned long long)umcRecord->value);
            break;
            
        case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH:
            bytesTransferredDtoH += umcRecord->value;
            printf("  设备到主机传输：%llu 字节\n", 
                   (unsigned long long)umcRecord->value);
            break;
        }
        break;
    }
    }
}
```

这个函数：
1. 检查活动记录类型
2. 提取统一内存计数器信息
3. 跟踪页面错误和数据传输

## 性能分析

### 内存迁移模式

```cpp
class UnifiedMemoryAnalyzer {
private:
    uint64_t totalCPUFaults;
    uint64_t totalGPUFaults;
    uint64_t totalHtoDTransfer;
    uint64_t totalDtoHTransfer;

public:
    void analyzeMemoryBehavior() {
        printf("=== 统一内存分析 ===\n");
        
        // 分析页面错误模式
        if (totalCPUFaults > totalGPUFaults * 2) {
            printf("检测到 CPU 主导的访问模式\n");
            printf("建议：考虑 CPU 内存的数据驻留建议\n");
        } else if (totalGPUFaults > totalCPUFaults * 2) {
            printf("检测到 GPU 主导的访问模式\n");
            printf("建议：使用 cudaMemPrefetchAsync 预取到 GPU\n");
        } else {
            printf("检测到平衡的 CPU/GPU 访问模式\n");
            printf("建议：使用访问建议进行并发访问\n");
        }
        
        // 分析传输效率
        uint64_t totalTransfer = totalHtoDTransfer + totalDtoHTransfer;
        if (totalTransfer > 0) {
            double htodRatio = (double)totalHtoDTransfer / totalTransfer;
            printf("主机到设备传输比率：%.2f%%\n", htodRatio * 100);
            
            if (htodRatio > 0.8) {
                printf("高 HtoD 传输 - 数据主要从 CPU 流向 GPU\n");
            } else if (htodRatio < 0.2) {
                printf("高 DtoH 传输 - 数据主要从 GPU 流向 CPU\n");
            }
        }
    }
    
    void suggestOptimizations() {
        printf("\n=== 优化建议 ===\n");
        
        // 基于页面错误模式的建议
        if (totalCPUFaults + totalGPUFaults > 1000) {
            printf("1. 高页面错误计数检测\n");
            printf("   - 使用 cudaMemPrefetchAsync 减少运行时迁移\n");
            printf("   - 考虑数据局部性优化\n");
        }
        
        // 基于传输模式的建议
        if (totalHtoDTransfer > totalDtoHTransfer * 3) {
            printf("2. 主要是主机到设备的数据流\n");
            printf("   - 在 GPU 计算前预取数据\n");
            printf("   - 在 GPU 上保持数据更长时间\n");
        }
        
        if (totalDtoHTransfer > totalHtoDTransfer * 3) {
            printf("3. 主要是设备到主机的数据流\n");
            printf("   - 在 CPU 处理前预取数据\n");
            printf("   - 批量传输以减少开销\n");
        }
    }
};
```

### 内存建议优化

```cpp
void optimizeWithMemoryAdvice(void *data, size_t size) {
    // 分析访问模式并应用适当的建议
    int device = 0;
    
    // 设置首选位置
    RUNTIME_API_CALL(cudaMemAdvise(data, size, cudaMemAdviseSetPreferredLocation, device));
    
    // 设置访问者
    RUNTIME_API_CALL(cudaMemAdvise(data, size, cudaMemAdviseSetAccessedBy, device));
    RUNTIME_API_CALL(cudaMemAdvise(data, size, cudaMemAdviseSetAccessedBy, cudaCpuDeviceId));
    
    // 对于只读数据
    RUNTIME_API_CALL(cudaMemAdvise(data, size, cudaMemAdviseSetReadMostly, device));
    
    printf("应用内存建议以优化访问模式\n");
}
```

## 实际用例

### 数据流分析

```cpp
void analyzeDataFlow() {
    printf("=== 数据流分析 ===\n");
    
    // 跟踪随时间的内存活动
    std::vector<uint64_t> timePoints;
    std::vector<uint64_t> htodTransfers;
    std::vector<uint64_t> dtohTransfers;
    
    // 收集数据点...
    
    // 分析模式
    for (size_t i = 1; i < timePoints.size(); i++) {
        uint64_t htodRate = htodTransfers[i] - htodTransfers[i-1];
        uint64_t dtohRate = dtohTransfers[i] - dtohTransfers[i-1];
        uint64_t timeDiff = timePoints[i] - timePoints[i-1];
        
        if (timeDiff > 0) {
            printf("时间 %llu: HtoD 速率 = %llu MB/s, DtoH 速率 = %llu MB/s\n",
                   timePoints[i], 
                   htodRate * 1000 / timeDiff / (1024*1024),
                   dtohRate * 1000 / timeDiff / (1024*1024));
        }
    }
}
```

### 性能比较

```cpp
class PerformanceComparator {
public:
    void compareAccessPatterns() {
        printf("=== 访问模式性能比较 ===\n");
        
        // 测试不同的模式
        float *data;
        size_t size = 1024 * 1024 * sizeof(float);
        
        // 分配统一内存
        cudaMallocManaged(&data, size);
        
        // 基线：无优化
        auto start = std::chrono::high_resolution_clock::now();
        testSequentialAccess(data, size);
        auto end = std::chrono::high_resolution_clock::now();
        auto baselineDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        // 预取优化
        start = std::chrono::high_resolution_clock::now();
        testPrefetchedAccess(data, size);
        end = std::chrono::high_resolution_clock::now();
        auto prefetchDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        // 内存建议优化
        optimizeWithMemoryAdvice(data, size);
        start = std::chrono::high_resolution_clock::now();
        testConcurrentAccess(data, size);
        end = std::chrono::high_resolution_clock::now();
        auto adviseDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        // 打印结果
        printf("基线（无优化）：%ld μs\n", baselineDuration.count());
        printf("预取优化：%ld μs (%.2fx 改进)\n", 
               prefetchDuration.count(), 
               (double)baselineDuration.count() / prefetchDuration.count());
        printf("内存建议优化：%ld μs (%.2fx 改进)\n", 
               adviseDuration.count(),
               (double)baselineDuration.count() / adviseDuration.count());
        
        cudaFree(data);
    }
};
```

## 最佳实践

### 内存访问模式

1. **预取数据**：在需要前将数据移动到适当的位置
2. **使用内存建议**：指导运行时进行最佳数据放置
3. **批量访问**：减少页面错误的频率
4. **避免震荡**：防止数据在设备间频繁移动

### 调试技巧

```cpp
void debugUnifiedMemory() {
    // 启用详细的统一内存日志记录
    setenv("CUDA_LAUNCH_BLOCKING", "1", 1);
    
    // 检查设备能力
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    printf("设备：%s\n", prop.name);
    printf("计算能力：%d.%d\n", prop.major, prop.minor);
    printf("并发管理内存：%s\n", 
           prop.concurrentManagedAccess ? "支持" : "不支持");
    printf("页面错误支持：%s\n", 
           (prop.major >= 6) ? "支持" : "不支持");
}
```

### 性能监控

```cpp
class UnifiedMemoryMonitor {
private:
    std::map<std::string, uint64_t> metrics;

public:
    void trackMetric(const std::string& name, uint64_t value) {
        metrics[name] += value;
    }
    
    void printSummary() {
        printf("=== 统一内存摘要 ===\n");
        for (const auto& [name, value] : metrics) {
            printf("%s: %llu\n", name.c_str(), value);
        }
        
        // 计算效率指标
        uint64_t totalFaults = metrics["cpu_faults"] + metrics["gpu_faults"];
        uint64_t totalTransfer = metrics["htod_bytes"] + metrics["dtoh_bytes"];
        
        if (totalFaults > 0) {
            double avgTransferPerFault = (double)totalTransfer / totalFaults;
            printf("每页面错误平均传输：%.2f 字节\n", avgTransferPerFault);
        }
    }
};
```

统一内存分析提供了优化异构应用程序性能的强大洞察。通过理解数据迁移模式和应用适当的优化策略，您可以显著提高使用统一内存的 CUDA 应用程序的性能。 