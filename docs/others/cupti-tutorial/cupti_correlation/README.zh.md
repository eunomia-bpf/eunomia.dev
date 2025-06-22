# CUPTI API-GPU 活动关联教程

> GitHub 仓库和完整教程可在 <https://github.com/eunomia-bpf/cupti-tutorial> 获取。

## 简介

本示例演示了如何使用 CUPTI 关联 ID 将 CUDA API 调用与其对应的 GPU 活动关联起来。理解这种关系对于性能分析和调试至关重要。

## 您将学到的内容

- 如何将 CUDA API 调用与 GPU 活动关联
- 使用关联 ID 跟踪执行流程
- 构建用于分析的关联映射
- 通过关联识别性能瓶颈

## 关键概念

### 关联 ID

每个 CUDA API 调用都获得一个唯一的关联 ID，该 ID 将其链接到它生成的 GPU 活动：

```cpp
// 带有关联 ID 的 API 记录
CUpti_ActivityAPI *apiRecord = ...;
uint32_t correlationId = apiRecord->correlationId;

// 具有相同关联 ID 的 GPU 活动
CUpti_ActivityKernel9 *kernelRecord = ...;
uint32_t sameId = kernelRecord->correlationId;
```

### 示例架构

```cpp
// 用于存储关联记录的映射
static std::map<uint32_t, CUpti_Activity*> s_CorrelationMap;  // GPU 活动
static std::map<uint32_t, CUpti_Activity*> s_ConnectionMap;   // API 调用

void ProcessActivityRecord(CUpti_Activity* record) {
    switch (record->kind) {
        case CUPTI_ACTIVITY_KIND_API:
            // 存储 API 记录
            s_ConnectionMap[apiRecord->correlationId] = record;
            break;
        case CUPTI_ACTIVITY_KIND_KERNEL:
            // 存储 GPU 活动
            s_CorrelationMap[kernelRecord->correlationId] = record;
            break;
    }
}
```

## 示例演练

示例执行向量操作并将每个 API 调用与其 GPU 活动关联：

1. **内存分配**：`cudaMalloc` → GPU 内存分配
2. **内存传输**：`cudaMemcpyAsync` → DMA 传输活动
3. **内核启动**：`VectorAdd<<<>>>` → 内核执行活动
4. **同步**：`cudaStreamSynchronize` → GPU 空闲/同步活动

### 关联分析

```cpp
void PrintCorrelationInformation() {
    for (auto& pair : s_CorrelationMap) {
        uint32_t correlationId = pair.first;
        CUpti_Activity* gpuActivity = pair.second;
        
        // 查找对应的 API 记录
        auto apiIter = s_ConnectionMap.find(correlationId);
        if (apiIter != s_ConnectionMap.end()) {
            printf("关联 ID: %u\n", correlationId);
            PrintActivity(gpuActivity, stdout);
            PrintActivity(apiIter->second, stdout);
        }
    }
}
```

## 构建和运行

```bash
cd cupti_correlation
make
./cupti_correlation
```

## 示例输出

```
CUDA_API 和 GPU 活动关联 : 关联 1
CUPTI_ACTIVITY_KIND_MEMCPY : start=1000 end=1500 duration=500
CUPTI_ACTIVITY_KIND_API : start=950 end=1600 name=cudaMemcpyAsync

CUDA_API 和 GPU 活动关联 : 关联 2  
CUPTI_ACTIVITY_KIND_KERNEL : start=2000 end=2100 duration=100
CUPTI_ACTIVITY_KIND_API : start=1900 end=2200 name=cudaLaunchKernel
```

## 使用场景

- **性能分析**：识别 API 开销与 GPU 执行时间
- **调试**：追踪哪些 API 调用生成特定的 GPU 活动
- **优化**：寻找重叠操作的机会
- **性能分析**：构建完整的执行时间线

## 高级关联技术

### 多活动关联

一个 API 调用可能生成多个 GPU 活动：

```cpp
class AdvancedCorrelator {
private:
    std::map<uint32_t, std::vector<CUpti_Activity*>> multiActivityMap;
    std::map<uint32_t, CUpti_Activity*> apiMap;

public:
    void addActivity(CUpti_Activity* activity) {
        if (activity->kind == CUPTI_ACTIVITY_KIND_API) {
            CUpti_ActivityAPI* apiActivity = (CUpti_ActivityAPI*)activity;
            apiMap[apiActivity->correlationId] = activity;
        } else {
            // 假设所有其他活动都有关联 ID
            uint32_t correlationId = getCorrelationId(activity);
            multiActivityMap[correlationId].push_back(activity);
        }
    }
    
    void analyzeCorrelations() {
        for (const auto& pair : apiMap) {
            uint32_t correlationId = pair.first;
            CUpti_Activity* apiActivity = pair.second;
            
            auto activitiesIter = multiActivityMap.find(correlationId);
            if (activitiesIter != multiActivityMap.end()) {
                printf("API 调用 %s 生成了 %zu 个 GPU 活动\n",
                       getApiName(apiActivity), 
                       activitiesIter->second.size());
                
                for (CUpti_Activity* gpuActivity : activitiesIter->second) {
                    analyzeApiToGpuRelation(apiActivity, gpuActivity);
                }
            }
        }
    }
};
```

### 时间线重建

```cpp
class TimelineReconstructor {
private:
    struct CorrelatedEvent {
        uint32_t correlationId;
        uint64_t apiStart;
        uint64_t apiEnd;
        uint64_t gpuStart;
        uint64_t gpuEnd;
        std::string apiName;
        std::string activityType;
    };
    
    std::vector<CorrelatedEvent> timeline;

public:
    void buildTimeline() {
        // 根据关联 ID 匹配 API 和 GPU 活动
        for (const auto& correlation : correlations) {
            CorrelatedEvent event;
            event.correlationId = correlation.id;
            event.apiStart = correlation.apiActivity->start;
            event.apiEnd = correlation.apiActivity->end;
            event.gpuStart = correlation.gpuActivity->start;
            event.gpuEnd = correlation.gpuActivity->end;
            event.apiName = getApiName(correlation.apiActivity);
            event.activityType = getActivityType(correlation.gpuActivity);
            
            timeline.push_back(event);
        }
        
        // 按时间排序
        std::sort(timeline.begin(), timeline.end(),
                 [](const CorrelatedEvent& a, const CorrelatedEvent& b) {
                     return a.apiStart < b.apiStart;
                 });
    }
    
    void analyzeGaps() {
        for (size_t i = 0; i < timeline.size() - 1; i++) {
            const auto& current = timeline[i];
            const auto& next = timeline[i + 1];
            
            // 分析 API 开销
            uint64_t apiOverhead = current.gpuStart - current.apiStart;
            
            // 分析空闲时间
            uint64_t idleTime = next.apiStart - current.gpuEnd;
            
            printf("操作 %s: API 开销 %llu ns, 空闲时间 %llu ns\n",
                   current.apiName.c_str(), apiOverhead, idleTime);
        }
    }
};
```

### 性能瓶颈识别

```cpp
class BottleneckAnalyzer {
public:
    void analyzeBottlenecks(const std::vector<CorrelatedEvent>& timeline) {
        printf("\n=== 性能瓶颈分析 ===\n");
        
        for (const auto& event : timeline) {
            // 计算各种时间指标
            uint64_t apiDuration = event.apiEnd - event.apiStart;
            uint64_t gpuDuration = event.gpuEnd - event.gpuStart;
            uint64_t launchOverhead = event.gpuStart - event.apiStart;
            
            // 识别高开销操作
            if (launchOverhead > HIGH_OVERHEAD_THRESHOLD) {
                printf("高启动开销检测: %s (开销: %llu ns)\n",
                       event.apiName.c_str(), launchOverhead);
            }
            
            // 识别低效率操作
            double efficiency = (double)gpuDuration / apiDuration;
            if (efficiency < LOW_EFFICIENCY_THRESHOLD) {
                printf("低效率操作: %s (效率: %.2f%%)\n",
                       event.apiName.c_str(), efficiency * 100.0);
            }
            
            // 识别长时间运行的操作
            if (gpuDuration > LONG_DURATION_THRESHOLD) {
                printf("长时间运行操作: %s (持续时间: %llu ns)\n",
                       event.apiName.c_str(), gpuDuration);
            }
        }
    }
};
```

## 实际应用

### 自动化性能报告

```cpp
class PerformanceReporter {
public:
    void generateReport(const std::vector<CorrelatedEvent>& events) {
        // 计算统计信息
        uint64_t totalApiTime = 0;
        uint64_t totalGpuTime = 0;
        uint64_t totalOverhead = 0;
        
        for (const auto& event : events) {
            totalApiTime += (event.apiEnd - event.apiStart);
            totalGpuTime += (event.gpuEnd - event.gpuStart);
            totalOverhead += (event.gpuStart - event.apiStart);
        }
        
        printf("\n=== 性能报告 ===\n");
        printf("总 API 时间: %llu ns\n", totalApiTime);
        printf("总 GPU 时间: %llu ns\n", totalGpuTime);
        printf("总启动开销: %llu ns\n", totalOverhead);
        printf("GPU 利用率: %.2f%%\n", 
               (double)totalGpuTime / totalApiTime * 100.0);
        printf("开销比例: %.2f%%\n",
               (double)totalOverhead / totalApiTime * 100.0);
        
        // 按 API 类型分组分析
        analyzeByApiType(events);
    }
    
private:
    void analyzeByApiType(const std::vector<CorrelatedEvent>& events) {
        std::map<std::string, std::vector<CorrelatedEvent>> byApiType;
        
        for (const auto& event : events) {
            byApiType[event.apiName].push_back(event);
        }
        
        printf("\n=== 按 API 类型分析 ===\n");
        for (const auto& pair : byApiType) {
            const auto& apiEvents = pair.second;
            uint64_t totalTime = 0;
            
            for (const auto& event : apiEvents) {
                totalTime += (event.gpuEnd - event.gpuStart);
            }
            
            printf("%s: %zu 次调用, 总时间 %llu ns, 平均时间 %llu ns\n",
                   pair.first.c_str(), apiEvents.size(), totalTime,
                   totalTime / apiEvents.size());
        }
    }
};
```

## 故障排除

### 常见问题

1. **丢失关联**：
   ```cpp
   void validateCorrelations() {
       for (const auto& apiPair : apiMap) {
           uint32_t correlationId = apiPair.first;
           if (gpuActivityMap.find(correlationId) == gpuActivityMap.end()) {
               printf("警告: API 调用 %u 没有对应的 GPU 活动\n", correlationId);
           }
       }
   }
   ```

2. **时间戳不一致**：
   ```cpp
   void validateTimestamps(const CorrelatedEvent& event) {
       if (event.gpuStart < event.apiStart) {
           printf("警告: GPU 活动在 API 调用之前开始\n");
       }
       if (event.gpuEnd < event.gpuStart) {
           printf("错误: 无效的 GPU 活动时间戳\n");
       }
   }
   ```

## 下一步

- 将关联扩展到每个 API 调用包含多个 GPU 活动
- 添加时间分析以识别瓶颈
- 实现多流应用程序的关联
- 使用关联数据构建可视化工具

理解 API-GPU 关联是 CUDA 性能优化和调试的基础。本示例为构建复杂的性能分析和分析工具提供了基础。 