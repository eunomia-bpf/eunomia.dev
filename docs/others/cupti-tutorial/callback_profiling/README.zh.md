# CUPTI 基于回调的性能分析教程

> GitHub 仓库和完整教程可在 <https://github.com/eunomia-bpf/cupti-tutorial> 获取。

## 简介

CUPTI基于回调的性能分析示例演示如何使用CUPTI的回调API实现全面的性能分析。这种方法允许您拦截CUDA运行时和驱动API调用，收集详细的性能指标，并在应用程序执行期间实时分析GPU活动模式。

## 您将学到什么

- 如何注册和处理用于性能分析的CUPTI回调
- 在CUDA API调用期间实现实时指标收集
- 理解回调时序和同步
- 收集API时序和GPU性能指标
- 使用回调构建非侵入式性能分析系统

## 理解基于回调的性能分析

基于回调的性能分析提供独特的优势：

1. **实时拦截**：在CUDA操作发生时监控它们
2. **API级粒度**：分析单个API调用及其参数
3. **最小开销**：高效的数据收集，无需修改应用程序
4. **灵活过滤**：选择要分析的操作
5. **全面覆盖**：访问运行时和驱动API层

## 关键概念

### 回调类型

CUPTI为不同的API域提供回调：
- **运行时API**：cudaMalloc、cudaMemcpy、cudaLaunchKernel等
- **驱动API**：cuMemAlloc、cuMemcpyHtoD、cuLaunchKernel等
- **资源API**：上下文和流的创建/销毁
- **同步API**：cudaDeviceSynchronize、cudaStreamSynchronize等

### 回调阶段
每个回调可以在两个阶段发生：
- **入口**：API调用执行之前
- **出口**：API调用完成之后

### 回调数据
回调提供访问：
- 函数名称和参数
- 线程和上下文信息
- 时序数据
- 返回值和错误代码

## 构建示例

### 先决条件

确保您有：
- 带CUPTI的CUDA工具包
- 支持C++11的C++编译器
- CUPTI开发头文件

### 构建过程

```bash
cd callback_profiling
make
```

这创建了演示基于回调的性能分析技术的`callback_profiling`可执行文件。

## 代码架构

### 主要组件

1. **回调注册**：为所需的API域设置回调
2. **数据收集**：收集时序和参数信息
3. **指标集成**：收集GPU性能指标
4. **输出生成**：格式化和呈现性能分析结果

### 核心实现

```cpp
// 回调函数签名
void CUPTIAPI callbackHandler(void *userdata, CUpti_CallbackDomain domain,
                             CUpti_CallbackId cbid, const CUpti_CallbackData *cbInfo)
{
    const char *funcName = cbInfo->functionName;
    
    if (cbInfo->callbackSite == CUPTI_API_ENTER) {
        // 入口：记录开始时间，记录参数
        recordAPIEntry(funcName, cbInfo->functionParams);
    }
    else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
        // 出口：记录结束时间，记录结果
        recordAPIExit(funcName, cbInfo->functionReturnValue);
    }
}
```

## 运行示例

### 基本执行

```bash
./callback_profiling
```

### 示例输出

```
=== CUPTI回调性能分析结果 ===

CUDA运行时API调用：
  cudaMalloc：3次调用，总计：145μs，平均：48.3μs
  cudaMemcpy：6次调用，总计：2.1ms，平均：350μs
  cudaLaunchKernel：100次调用，总计：5.2ms，平均：52μs
  cudaDeviceSynchronize：1次调用，总计：15.3ms，平均：15.3ms

CUDA驱动API调用：
  cuCtxCreate：1次调用，总计：125μs，平均：125μs
  cuModuleLoad：1次调用，总计：2.3ms，平均：2.3ms
  cuLaunchKernel：100次调用，总计：4.8ms，平均：48μs

性能指标：
  GPU利用率：78.5%
  内存带宽：245.2 GB/s
  缓存命中率：92.3%

总性能分析开销：0.8ms（总执行时间的0.5%）
```

## 详细分析功能

### API调用跟踪

示例跟踪每个API调用的全面信息：

1. **调用频率**：每个API被调用的次数
2. **时序统计**：最小、最大、平均和总执行时间
3. **参数分析**：内存大小、内核配置等
4. **错误跟踪**：失败的调用和错误代码

### 内存使用分析

```cpp
// 跟踪内存分配
void trackMemoryAllocation(size_t size, void* ptr) {
    totalAllocated += size;
    activeAllocations[ptr] = size;
    allocationHistory.push_back({getCurrentTime(), size, true});
}

// 跟踪内存释放
void trackMemoryDeallocation(void* ptr) {
    auto it = activeAllocations.find(ptr);
    if (it != activeAllocations.end()) {
        allocationHistory.push_back({getCurrentTime(), it->second, false});
        activeAllocations.erase(it);
    }
}
```

### 内核启动分析

```cpp
// 分析内核启动参数
void analyzeKernelLaunch(const dim3& gridDim, const dim3& blockDim, 
                        size_t sharedMem, cudaStream_t stream) {
    int totalThreads = gridDim.x * gridDim.y * gridDim.z * 
                      blockDim.x * blockDim.y * blockDim.z;
    
    kernelStats.totalLaunches++;
    kernelStats.totalThreads += totalThreads;
    kernelStats.sharedMemUsage += sharedMem;
    
    if (stream != 0) {
        kernelStats.asyncLaunches++;
    }
}
```

## 高级功能

### 选择性性能分析

为特定API类别启用性能分析：

```cpp
// 仅运行时API
CUPTI_CALL(cuptiEnableCallback(1, subscriber, 
           CUPTI_CB_DOMAIN_RUNTIME_API, 
           CUPTI_RUNTIME_TRACE_CBID_INVALID));

// 仅特定函数
CUPTI_CALL(cuptiEnableCallback(1, subscriber,
           CUPTI_CB_DOMAIN_RUNTIME_API,
           CUPTI_RUNTIME_TRACE_CBID_cudaMalloc_v3020));
```

### 性能指标集成

```cpp
// 在回调期间收集GPU指标
void collectMetrics(CUcontext context) {
    CUpti_EventGroup eventGroup;
    CUpti_EventID eventIds[NUM_EVENTS];
    
    // 设置事件收集
    CUPTI_CALL(cuptiEventGroupCreate(context, &eventGroup, 0));
    
    for (int i = 0; i < NUM_EVENTS; i++) {
        CUPTI_CALL(cuptiEventGroupAddEvent(eventGroup, eventIds[i]));
    }
    
    // 启用和读取事件
    CUPTI_CALL(cuptiEventGroupEnable(eventGroup));
    // ... 内核执行 ...
    
    uint64_t eventValues[NUM_EVENTS];
    CUPTI_CALL(cuptiEventGroupReadAllEvents(eventGroup, 
               CUPTI_EVENT_READ_FLAG_NONE,
               &bytesRead, eventValues, 
               &numEventIds, eventIds));
}
```

### 多线程分析

```cpp
// 线程安全的数据收集
class ThreadSafeProfiler {
private:
    std::mutex dataMutex;
    std::unordered_map<std::thread::id, ProfileData> threadData;
    
public:
    void recordAPICall(const std::string& apiName, uint64_t duration) {
        std::lock_guard<std::mutex> lock(dataMutex);
        auto threadId = std::this_thread::get_id();
        threadData[threadId].apiCalls[apiName].addSample(duration);
    }
};
```

## 实际应用

### 性能瓶颈检测

1. **API开销分析**：识别昂贵的CUDA API调用
2. **内存传输优化**：分析数据移动模式
3. **内核启动效率**：优化启动配置
4. **同步分析**：检测不必要的同步点

### 应用程序特征化

```cpp
// 生成应用程序配置文件
struct ApplicationProfile {
    double computeToMemoryRatio;
    double asyncUtilization;
    size_t peakMemoryUsage;
    int averageOccupancy;
    
    void generateReport() {
        std::cout << "计算/内存比率：" << computeToMemoryRatio << std::endl;
        std::cout << "异步利用率：" << asyncUtilization * 100 << "%" << std::endl;
        std::cout << "峰值内存使用：" << peakMemoryUsage / (1024*1024) << " MB" << std::endl;
        std::cout << "平均占用率：" << averageOccupancy << "%" << std::endl;
    }
};
```

### 实时监控

```cpp
// 实时性能仪表板
class LiveProfiler {
private:
    std::atomic<uint64_t> totalAPITime{0};
    std::atomic<uint64_t> totalKernelTime{0};
    std::atomic<size_t> memoryAllocated{0};
    
public:
    void updateDashboard() {
        while (profiling) {
            system("clear");
            std::cout << "=== 实时CUDA性能分析仪表板 ===" << std::endl;
            std::cout << "API时间：" << totalAPITime.load() / 1000 << "ms" << std::endl;
            std::cout << "内核时间：" << totalKernelTime.load() / 1000 << "ms" << std::endl;
            std::cout << "内存分配：" << memoryAllocated.load() / (1024*1024) << "MB" << std::endl;
            
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
};
```

## 与开发工作流程集成

### 自动化性能测试

```bash
# 性能回归测试
./callback_profiling --baseline > baseline_profile.txt
# ... 进行代码更改 ...
./callback_profiling --compare baseline_profile.txt > regression_report.txt
```

### 持续集成

```cpp
// CI友好的输出格式
void generateCIReport(const ProfileData& data) {
    json report;
    report["total_api_time"] = data.totalAPITime;
    report["memory_efficiency"] = data.memoryEfficiency;
    report["kernel_utilization"] = data.kernelUtilization;
    
    // 如果性能显著下降，则失败CI
    if (data.totalAPITime > PERFORMANCE_THRESHOLD) {
        std::exit(1);
    }
}
```

## 故障排除

### 常见问题

1. **回调未触发**：验证回调注册和域选择
2. **高开销**：减少回调频率或优化数据收集
3. **线程安全**：确保多线程应用程序中的适当同步
4. **内存泄漏**：检查回调数据结构的适当清理

### 调试技巧

1. **从简单回调开始**：在添加复杂分析之前从基本时序开始
2. **使用选择性性能分析**：专注于特定API以减少开销
3. **使用已知应用程序验证**：首先使用CUDA示例进行测试
4. **监控开销**：测量性能分析对应用程序性能的影响

## 下一步

- 扩展示例以分析应用程序的特定方面
- 将回调性能分析集成到您的开发和测试过程中
- 与其他CUPTI功能结合进行全面分析
- 为您的用例开发自定义指标和分析算法
- 为回调性能分析数据创建可视化工具 