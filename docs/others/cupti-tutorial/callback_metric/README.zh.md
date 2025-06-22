# CUPTI 性能指标教程

> GitHub 仓库和完整教程可在 <https://github.com/eunomia-bpf/cupti-tutorial> 获取。

## 简介

理解 GPU 性能需要的不仅仅是原始事件计数——它需要有意义的指标来提供关于代码运行效率的见解。本教程演示了如何使用 CUPTI 回调在 CUDA 内核执行期间收集和计算性能指标，为您的应用程序的 GPU 利用率提供强大的见解。

## 您将学到的内容

- 如何在内核执行期间收集复杂的性能指标
- 处理需要多次遍历的指标的技术
- 将原始事件计数转换为有意义的性能指标
- 解释指标以识别优化机会

## 理解性能指标

CUDA 中的性能指标通过组合多个硬件事件提供高级见解。例如：

- **IPC (每周期指令数)**：测量计算效率
- **内存吞吐量**：测量内存带宽利用率
- **Warp 执行效率**：测量 warp 内线程利用率

这些指标比原始事件计数更直观，并直接关联到优化策略。

## 代码演练

### 1. 设置指标收集

首先，我们需要识别目标指标所需的事件：

```cpp
int main(int argc, char *argv[])
{
    // 默认为 "ipc"（每周期指令数）或使用命令行参数
    const char *metricName = "ipc";
    if (argc > 1) {
        metricName = argv[1];
    }
    
    // 初始化 CUDA
    RUNTIME_API_CALL(cudaSetDevice(0));
    
    // 获取设备属性
    CUdevice device = 0;
    DRIVER_API_CALL(cuDeviceGet(&device, 0));
    
    // 获取请求指标的指标 ID
    CUpti_MetricID metricId;
    CUPTI_CALL(cuptiMetricGetIdFromName(device, metricName, &metricId));
    
    // 确定此指标所需的事件
    uint32_t numEvents = 0;
    CUPTI_CALL(cuptiMetricGetNumEvents(metricId, &numEvents));
    
    // 为事件 ID 分配空间
    CUpti_EventID *eventIds = (CUpti_EventID *)malloc(numEvents * sizeof(CUpti_EventID));
    CUPTI_CALL(cuptiMetricEnumEvents(metricId, &numEvents, eventIds));
    
    // 确定收集所有事件需要多少次遍历
    MetricData_t metricData;
    metricData.device = device;
    metricData.eventIdArray = eventIds;
    metricData.numEvents = numEvents;
    metricData.eventValueArray = (uint64_t *)calloc(numEvents, sizeof(uint64_t));
    metricData.eventIdx = 0;
    
    // 为每次遍历创建事件组
    createEventGroups(&metricData);
    
    printf("为指标 %s 收集事件\n", metricName);
}
```

此代码：
1. 识别要收集的指标（默认为 "ipc"）
2. 获取该指标所需的事件列表
3. 设置用于保存事件值的数据结构
4. 创建用于收集事件的事件组

### 2. 创建事件组

某些事件由于硬件限制无法同时收集，因此我们需要将它们组织成兼容的组：

```cpp
void createEventGroups(MetricData_t *metricData)
{
    CUcontext context = NULL;
    DRIVER_API_CALL(cuCtxGetCurrent(&context));
    
    // 获取设备上的事件域数量
    uint32_t numDomains = 0;
    CUPTI_CALL(cuptiDeviceGetNumEventDomains(metricData->device, &numDomains));
    
    // 获取事件域
    CUpti_EventDomainID *domainIds = (CUpti_EventDomainID *)malloc(numDomains * sizeof(CUpti_EventDomainID));
    CUPTI_CALL(cuptiDeviceEnumEventDomains(metricData->device, &numDomains, domainIds));
    
    // 对于每个事件，找到其域和可用实例
    for (int i = 0; i < metricData->numEvents; i++) {
        CUpti_EventDomainID domainId;
        CUPTI_CALL(cuptiEventGetAttribute(metricData->eventIdArray[i], 
                                        CUPTI_EVENT_ATTR_DOMAIN, 
                                        &domainId));
        
        // 在我们的列表中找到此域
        int domainIndex = -1;
        for (int j = 0; j < numDomains; j++) {
            if (domainId == domainIds[j]) {
                domainIndex = j;
                break;
            }
        }
        
        // 如果这是新的事件组或事件无法放入当前组，
        // 创建新的事件组
        if (metricData->numEventGroups == 0 || 
            !canAddEventToGroup(metricData, metricData->eventIdArray[i])) {
            
            // 创建新的事件组
            CUpti_EventGroup eventGroup;
            CUPTI_CALL(cuptiEventGroupCreate(context, &eventGroup, 0));
            
            // 将事件添加到组
            CUPTI_CALL(cuptiEventGroupAddEvent(eventGroup, metricData->eventIdArray[i]));
            
            // 存储事件组
            metricData->eventGroups[metricData->numEventGroups++] = eventGroup;
        }
        else {
            // 将事件添加到现有组
            CUPTI_CALL(cuptiEventGroupAddEvent(
                metricData->eventGroups[metricData->numEventGroups-1], 
                metricData->eventIdArray[i]));
        }
    }
    
    free(domainIds);
}
```

此函数：
1. 获取设备上的所有事件域
2. 对于每个事件，确定其域
3. 创建与硬件限制兼容的事件组
4. 如果事件无法同时收集，可能创建多个组

### 3. 设置回调函数

现在我们注册一个在内核启动时被调用的回调：

```cpp
// 订阅 CUPTI 回调
CUpti_SubscriberHandle subscriber;
CUPTI_CALL(cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)getMetricValueCallback, &metricData));

// 为 CUDA 运行时启用回调
CUPTI_CALL(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API));
```

### 4. 回调函数

示例的核心是收集事件数据的回调函数：

```cpp
void CUPTIAPI getMetricValueCallback(void *userdata, CUpti_CallbackDomain domain,
                                    CUpti_CallbackId cbid, const CUpti_CallbackData *cbInfo)
{
    MetricData_t *metricData = (MetricData_t*)userdata;
    
    // 仅处理内核启动的运行时 API 回调
    if (domain != CUPTI_CB_DOMAIN_RUNTIME_API ||
        (cbid != CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020 &&
         cbid != CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000)) {
        return;
    }
    
    // 检查我们是进入还是退出函数
    if (cbInfo->callbackSite == CUPTI_API_ENTER) {
        // 我们即将启动内核
        
        // 为此遍历启用事件组
        for (int i = 0; i < metricData->numEventGroups; i++) {
            CUPTI_CALL(cuptiEventGroupEnable(metricData->eventGroups[i]));
        }
        
        // 设置收集模式为仅在内核执行期间收集
        CUpti_EventCollectionMode mode = CUPTI_EVENT_COLLECTION_MODE_KERNEL;
        for (int i = 0; i < metricData->numEventGroups; i++) {
            CUPTI_CALL(cuptiEventGroupSetAttribute(metricData->eventGroups[i],
                                                 CUPTI_EVENT_GROUP_ATTR_COLLECTION_MODE,
                                                 sizeof(mode), &mode));
        }
    }
    else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
        // 内核已完成
        
        // 确保所有工作完成
        RUNTIME_API_CALL(cudaDeviceSynchronize());
        
        // 从所有事件组读取事件值
        for (int i = 0; i < metricData->numEventGroups; i++) {
            uint32_t numInstances = 0;
            uint32_t numTotalInstances = 0;
            uint64_t *values = NULL;
            size_t valuesSize = 0;
            
            // 获取此组中的实例数
            CUPTI_CALL(cuptiEventGroupGetAttribute(metricData->eventGroups[i],
                                                 CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT,
                                                 &valueSize, &numInstances));
            
            // 获取此组中的事件数
            uint32_t numEvents = 0;
            CUPTI_CALL(cuptiEventGroupGetAttribute(metricData->eventGroups[i],
                                                 CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS,
                                                 &valueSize, &numEvents));
            
            // 分配值数组
            valuesSize = sizeof(uint64_t) * numInstances * numEvents;
            values = (uint64_t *)malloc(valuesSize);
            
            // 读取事件值
            CUPTI_CALL(cuptiEventGroupReadAllEvents(metricData->eventGroups[i],
                                                   CUPTI_EVENT_READ_FLAG_NONE,
                                                   &valuesSize,
                                                   values,
                                                   &numTotalInstances,
                                                   &numEvents));
            
            // 聚合所有实例的值
            for (int j = 0; j < numEvents; j++) {
                uint64_t sum = 0;
                for (int k = 0; k < numInstances; k++) {
                    sum += values[j * numInstances + k];
                }
                metricData->eventValueArray[metricData->eventIdx++] = sum;
            }
            
            free(values);
            CUPTI_CALL(cuptiEventGroupDisable(metricData->eventGroups[i]));
        }
    }
}
```

此回调函数：
1. 检测内核启动
2. 在内核开始前启用事件收集
3. 在内核完成后读取所有事件值
4. 聚合多个实例的事件计数

### 5. 计算指标值

收集所有事件后，我们计算最终指标：

```cpp
void calculateMetricValue(MetricData_t *metricData, const char *metricName) {
    CUpti_MetricID metricId;
    CUPTI_CALL(cuptiMetricGetIdFromName(metricData->device, metricName, &metricId));
    
    // 计算指标值
    CUpti_MetricValue metricValue;
    CUPTI_CALL(cuptiMetricGetValue(metricData->device,
                                  metricId,
                                  metricData->numEvents * sizeof(CUpti_EventID),
                                  metricData->eventIdArray,
                                  metricData->numEvents * sizeof(uint64_t),
                                  metricData->eventValueArray,
                                  0.0, // 持续时间（对于某些指标）
                                  &metricValue));
    
    // 根据指标类型显示值
    switch (metricValue.kind) {
        case CUPTI_METRIC_VALUE_KIND_DOUBLE:
            printf("%s = %.2f\n", metricName, metricValue.metricValueDouble);
            break;
        case CUPTI_METRIC_VALUE_KIND_UINT64:
            printf("%s = %llu\n", metricName, metricValue.metricValueUint64);
            break;
        case CUPTI_METRIC_VALUE_KIND_INT64:
            printf("%s = %lld\n", metricName, metricValue.metricValueInt64);
            break;
        case CUPTI_METRIC_VALUE_KIND_PERCENT:
            printf("%s = %.2f%%\n", metricName, metricValue.metricValuePercent);
            break;
        case CUPTI_METRIC_VALUE_KIND_THROUGHPUT:
            printf("%s = %llu bytes/sec\n", metricName, metricValue.metricValueThroughput);
            break;
        case CUPTI_METRIC_VALUE_KIND_UTILIZATION_LEVEL:
            printf("%s = Level %d\n", metricName, metricValue.metricValueUtilizationLevel);
            break;
    }
}
```

## 可用指标

CUPTI 提供多种有用的性能指标：

### 计算指标
```cpp
"ipc"                           // 每周期指令数
"achieved_occupancy"            // 实际占用率
"sm_efficiency"                 // 流多处理器效率
"warp_execution_efficiency"     // Warp 执行效率
```

### 内存指标
```cpp
"gld_efficiency"                // 全局加载效率
"gst_efficiency"                // 全局存储效率
"shared_efficiency"             // 共享内存效率
"l1_cache_global_hit_rate"      // L1 缓存命中率
"l2_l1_read_hit_rate"          // L2 对 L1 读取命中率
"dram_read_throughput"          // DRAM 读取吞吐量
"dram_write_throughput"         // DRAM 写入吞吐量
```

### 控制流指标
```cpp
"branch_efficiency"             // 分支效率
"divergent_branch"              // 发散分支
"warp_nonpred_execution_efficiency" // Warp 非预测执行效率
```

## 运行示例

1. 构建示例：
   ```bash
   make
   ```

2. 使用默认指标（ipc）运行：
   ```bash
   ./callback_metric
   ```

3. 指定不同的指标：
   ```bash
   ./callback_metric achieved_occupancy
   ./callback_metric gld_efficiency  
   ./callback_metric dram_read_throughput
   ```

## 示例输出

```
为指标 ipc 收集事件
向量加法内核执行完成
ipc = 0.85

为指标 achieved_occupancy 收集事件  
向量加法内核执行完成
achieved_occupancy = 62.50%

为指标 gld_efficiency 收集事件
向量加法内核执行完成
gld_efficiency = 100.00%
```

## 性能分析见解

### IPC（每周期指令数）分析

- **高 IPC (>0.8)**：内核计算密集，GPU 核心利用率高
- **中等 IPC (0.4-0.8)**：均衡的计算和内存操作
- **低 IPC (<0.4)**：可能是内存绑定或有大量分支发散

### 占用率分析

- **高占用率 (>75%)**：GPU 资源被充分利用
- **中等占用率 (50-75%)**：可能有资源限制（寄存器、共享内存）
- **低占用率 (<50%)**：可能需要优化线程块大小或资源使用

### 内存效率分析

- **高内存效率 (>90%)**：良好的内存合并访问
- **低内存效率 (<50%)**：需要优化内存访问模式

## 高级用法

### 多指标批量分析

```cpp
const char* metrics[] = {
    "ipc",
    "achieved_occupancy", 
    "gld_efficiency",
    "gst_efficiency",
    "shared_efficiency"
};

void analyzeMultipleMetrics(int numMetrics, const char* metricNames[]) {
    for (int i = 0; i < numMetrics; i++) {
        printf("\n分析指标: %s\n", metricNames[i]);
        // 重置事件收集
        resetMetricData(&metricData);
        // 为新指标设置事件
        setupMetricEvents(&metricData, metricNames[i]);
        // 运行内核并收集数据
        runKernelWithProfiling();
        // 计算并显示结果
        calculateMetricValue(&metricData, metricNames[i]);
    }
}
```

### 性能瓶颈识别

```cpp
void identifyBottlenecks(MetricResults* results) {
    // 分析计算绑定
    if (results->ipc > 0.7 && results->occupancy > 0.7) {
        printf("内核是计算绑定的\n");
    }
    
    // 分析内存绑定
    if (results->gld_efficiency < 0.5 || results->gst_efficiency < 0.5) {
        printf("检测到内存访问效率低下\n");
        if (results->l1_hit_rate < 0.5) {
            printf("考虑优化数据局部性\n");
        }
    }
    
    // 分析发散问题
    if (results->warp_efficiency < 0.8) {
        printf("检测到 warp 发散问题\n");
        printf("考虑减少分支发散\n");
    }
}
```

### 时间序列分析

```cpp
typedef struct PerformanceHistory_st {
    double timestamps[MAX_SAMPLES];
    double metricValues[MAX_SAMPLES];
    int sampleCount;
} PerformanceHistory;

void trackPerformanceOverTime(PerformanceHistory* history, double metric) {
    if (history->sampleCount < MAX_SAMPLES) {
        history->timestamps[history->sampleCount] = getCurrentTime();
        history->metricValues[history->sampleCount] = metric;
        history->sampleCount++;
    }
}

void analyzePerformanceTrends(PerformanceHistory* history) {
    // 计算趋势、方差等
    double avgMetric = calculateAverage(history->metricValues, history->sampleCount);
    double variance = calculateVariance(history->metricValues, history->sampleCount);
    
    printf("平均指标值: %.2f\n", avgMetric);
    printf("性能方差: %.2f\n", variance);
}
```

## 故障排除

### 常见问题

1. **指标不可用**：
   ```cpp
   CUptiResult result = cuptiMetricGetIdFromName(device, metricName, &metricId);
   if (result != CUPTI_SUCCESS) {
       printf("指标 '%s' 在此设备上不可用\n", metricName);
       listAvailableMetrics(device);
   }
   ```

2. **多遍历需求**：
   ```cpp
   if (metricData.numEventGroups > 1) {
       printf("此指标需要 %d 次遍历来收集所有事件\n", 
              metricData.numEventGroups);
   }
   ```

3. **内存不足**：
   ```cpp
   // 监控事件缓冲区大小
   size_t requiredSize = numEvents * numInstances * sizeof(uint64_t);
   if (requiredSize > MAX_BUFFER_SIZE) {
       printf("警告：需要大事件缓冲区 (%zu 字节)\n", requiredSize);
   }
   ```

## 性能考虑

### 多遍历开销

某些指标需要多次内核执行来收集所有必需的事件：

```cpp
void optimizeMultiPassCollection(MetricData_t* metricData) {
    // 检查是否可以减少遍历次数
    if (metricData->numEventGroups > 2) {
        printf("考虑使用需要较少事件的替代指标\n");
    }
    
    // 实现智能缓存以避免重复计算
    if (isMetricCached(metricName)) {
        return getCachedMetricValue(metricName);
    }
}
```

### 选择性分析

```cpp
// 仅对重要内核启用指标收集
bool shouldCollectMetrics(const char* kernelName) {
    // 实现启发式规则来决定是否分析
    return (strstr(kernelName, "Critical") != NULL) ||
           (getKernelImportance(kernelName) > IMPORTANCE_THRESHOLD);
}
```

## 总结

CUPTI 指标回调提供了对 GPU 性能的深入了解。通过收集和分析诸如 IPC、占用率和内存效率等高级指标，开发者可以：

- 识别性能瓶颈
- 指导优化策略
- 验证性能改进
- 理解硬件利用率模式

这种方法为 CUDA 应用程序优化提供了数据驱动的见解，实现更好的 GPU 性能和效率。 