# CUPTI 回调事件教程

> GitHub 仓库和完整教程可在 <https://github.com/eunomia-bpf/cupti-tutorial> 获取。

## 简介

跟踪 GPU 性能事件对于优化 CUDA 应用程序至关重要。本教程演示了如何使用 CUPTI 的回调机制在内核执行期间收集特定的性能事件。我们将专注于"已执行指令"指标，以展示您的内核运行了多少 GPU 指令。

## 您将学到的内容

- 如何为 CUDA 运行时 API 调用设置 CUPTI 回调函数
- 创建和管理用于收集性能数据的事件组
- 围绕内核启动收集事件数据的技术
- 解释事件值以进行性能分析

## 代码演练

### 1. 事件收集结构

示例设置了两个关键数据结构：

```cpp
// 存储事件组和事件 ID
typedef struct cupti_eventData_st {
  CUpti_EventGroup eventGroup;
  CUpti_EventID eventId;
} cupti_eventData;

// 存储回调收集的事件数据和值
typedef struct RuntimeApiTrace_st {
  cupti_eventData *eventData;
  uint64_t eventVal;
} RuntimeApiTrace_t;
```

这些结构在整个应用程序中维护事件状态和收集的值。

### 2. 回调函数

示例的核心是回调函数，当调用某些 CUDA 运行时 API 函数时会被调用：

```cpp
void CUPTIAPI getEventValueCallback(void *userdata, CUpti_CallbackDomain domain,
                                    CUpti_CallbackId cbid, const CUpti_CallbackData *cbInfo)
{
  // 仅处理内核启动的回调
  if ((cbid != CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020) &&
      (cbid != CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000)) {
    return;
  }

  RuntimeApiTrace_t *traceData = (RuntimeApiTrace_t*)userdata;
  
  // 进入 CUDA 运行时函数时（内核启动前）
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    // 同步设备以确保干净的事件收集
    cudaDeviceSynchronize();
    
    // 将收集模式设置为内核级别
    cuptiSetEventCollectionMode(cbInfo->context, CUPTI_EVENT_COLLECTION_MODE_KERNEL);
    
    // 启用事件组开始收集数据
    cuptiEventGroupEnable(traceData->eventData->eventGroup);
  }
  
  // 退出 CUDA 运行时函数时（内核完成后）
  if (cbInfo->callbackSite == CUPTI_API_EXIT) {
    // 确定发生了多少个事件实例
    uint32_t numInstances = 0;
    cuptiEventGroupGetAttribute(traceData->eventData->eventGroup, 
                               CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT, 
                               &valueSize, &numInstances);
    
    // 为事件值分配空间
    uint64_t *values = (uint64_t *) malloc(sizeof(uint64_t) * numInstances);
    
    // 确保内核完成
    cudaDeviceSynchronize();
    
    // 读取事件值
    cuptiEventGroupReadEvent(traceData->eventData->eventGroup, 
                            CUPTI_EVENT_READ_FLAG_NONE, 
                            traceData->eventData->eventId, 
                            &bytesRead, values);
    
    // 聚合所有实例的值
    traceData->eventVal = 0;
    for (i=0; i<numInstances; i++) {
      traceData->eventVal += values[i];
    }
    
    // 清理
    free(values);
    cuptiEventGroupDisable(traceData->eventData->eventGroup);
  }
}
```

该函数执行以下关键操作：
- 在 API 进入时（内核启动前）：启用事件收集
- 在 API 退出时（内核完成后）：读取事件值并禁用收集

### 3. 设置事件和回调

```cpp
int main(int argc, char *argv[])
{
  // 要跟踪的默认事件（已执行指令）
  const char *eventName = EVENT_NAME;
  
  // 解析设备和事件的命令行参数
  if (argc > 2)
    eventName = argv[2];
  
  // 设置 CUPTI 订阅者
  cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)getEventValueCallback, &trace);
  
  // 为内核启动启用回调
  cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API);
  
  // 查找事件并创建事件组
  cuptiEventGetIdFromName(device, eventName, &cuptiEvent.eventId);
  cuptiEventGroupCreate(context, &cuptiEvent.eventGroup, 0);
  cuptiEventGroupAddEvent(cuptiEvent.eventGroup, cuptiEvent.eventId);
  
  // 将事件数据存储在我们的跟踪结构中
  trace.eventData = &cuptiEvent;
  trace.eventVal = 0;
  
  // 运行向量加法内核
  // ...
  
  // 显示事件值
  displayEventVal(&trace, eventName);
  
  // 清理
  cuptiEventGroupDestroy(cuptiEvent.eventGroup);
  cuptiUnsubscribe(subscriber);
  
  return 0;
}
```

主函数设置事件收集系统，运行内核，并显示结果。

### 4. 测试内核

示例使用简单的向量加法内核来演示事件收集：

```cpp
__global__ void VecAdd(const int* A, const int* B, int* C, int N)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N)
    C[i] = A[i] + B[i];
}
```

## 运行示例

1. 构建示例：
   ```bash
   make
   ```

2. 使用默认参数运行（设备 0，跟踪 "inst_executed"）：
   ```bash
   ./callback_event
   ```

3. 您可以指定不同的设备或事件：
   ```bash
   ./callback_event [device_num] [event_name]
   ```

   例如：
   ```bash
   ./callback_event 0 inst_executed_global_loads
   ```

## 理解输出

示例产生类似于以下的输出：

```
CUDA Device Number: 0
CUDA Device Name: NVIDIA GeForce RTX 3080
Compute capability: 8.6
Event Name : inst_executed
Event Value : 2048000
```

这显示了：
1. 用于测试的设备
2. 计算能力（架构版本）
3. 被跟踪的事件（"inst_executed" = 已执行指令）
4. 内核执行的指令总数

## 性能见解

### 指令计数分析

**inst_executed** 事件提供了关于内核计算复杂性的见解：

- **高指令计数**：可能表示计算密集型内核
- **低指令计数**：可能表示内存绑定或简单操作
- **与理论值的比较**：帮助识别编译器优化机会

### 事件类型

CUPTI 支持多种有用的事件类型：

```cpp
// 内存相关事件
"global_load"           // 全局内存加载
"global_store"          // 全局内存存储
"shared_load"           // 共享内存加载
"shared_store"          // 共享内存存储

// 指令执行事件  
"inst_executed"         // 总执行指令
"inst_integer"          // 整数指令
"inst_fp_32"            // 32位浮点指令
"inst_fp_64"            // 64位浮点指令

// 控制流事件
"divergent_branch"      // 分支发散
"thread_inst_executed"  // 每线程指令
```

## 高级使用

### 多事件收集

您可以同时收集多个事件：

```cpp
typedef struct MultiEventData_st {
    CUpti_EventGroup eventGroup;
    CUpti_EventID eventIds[MAX_EVENTS];
    uint64_t eventValues[MAX_EVENTS];
    int numEvents;
} MultiEventData;

void setupMultipleEvents(MultiEventData* data, const char* eventNames[], int count) {
    cuptiEventGroupCreate(context, &data->eventGroup, 0);
    
    for (int i = 0; i < count; i++) {
        cuptiEventGetIdFromName(device, eventNames[i], &data->eventIds[i]);
        cuptiEventGroupAddEvent(data->eventGroup, data->eventIds[i]);
    }
    data->numEvents = count;
}
```

### 事件归一化

将原始计数转换为有意义的指标：

```cpp
void analyzePerformance(RuntimeApiTrace_t* trace, int numThreads) {
    double instructionsPerThread = (double)trace->eventVal / numThreads;
    double computationalIntensity = instructionsPerThread / memoryOperations;
    
    printf("Instructions per thread: %.2f\n", instructionsPerThread);
    printf("Computational intensity: %.2f\n", computationalIntensity);
}
```

### 时间关联

将事件数据与时间信息相结合：

```cpp
typedef struct TimedEventData_st {
    uint64_t eventVal;
    uint64_t startTime;
    uint64_t endTime;
    double duration;
    double eventRate;
} TimedEventData;

void calculateEventRate(TimedEventData* data) {
    data->duration = (data->endTime - data->startTime) * 1e-9; // 转换为秒
    data->eventRate = data->eventVal / data->duration; // 每秒事件数
}
```

## 故障排除

### 常见问题

1. **事件不可用**：
   ```cpp
   CUptiResult result = cuptiEventGetIdFromName(device, eventName, &eventId);
   if (result != CUPTI_SUCCESS) {
       printf("Event '%s' not available on this device\n", eventName);
       // 列出可用事件
       listAvailableEvents(device);
   }
   ```

2. **权限错误**：
   - 确保以管理员权限运行
   - 检查 GPU 是否支持性能监控

3. **计数器冲突**：
   ```cpp
   // 检查事件是否可以同时收集
   CUpti_EventGroupSets* eventGroupSets;
   cuptiEventGroupSetsCreate(context, sizeof(eventIds), eventIds, &eventGroupSets);
   
   if (eventGroupSets->numSets > 1) {
       printf("Events require multiple passes to collect\n");
   }
   ```

### 调试提示

- 使用 `cuptiGetResultString()` 获取详细的错误信息
- 在生产代码中实现事件可用性检查
- 考虑不同 GPU 架构的事件兼容性

## 性能考虑

### 开销分析

事件收集会产生一些开销：

- **回调开销**：每个 API 调用的额外处理
- **同步成本**：`cudaDeviceSynchronize()` 调用
- **内存开销**：事件值存储

### 优化技巧

1. **选择性监控**：
   ```cpp
   // 仅为感兴趣的内核启用回调
   if (strstr(cbInfo->functionName, "TargetKernel") != NULL) {
       // 启用事件收集
   }
   ```

2. **批量处理**：
   ```cpp
   // 批量收集多个启动的事件
   static int launchCount = 0;
   if (++launchCount % BATCH_SIZE == 0) {
       processCollectedEvents();
   }
   ```

3. **异步收集**：
   ```cpp
   // 使用单独的线程处理事件数据
   std::thread processingThread(processEventData, eventBuffer);
   processingThread.detach();
   ```

## 实际应用

### 代码优化工作流程

1. **基线测量**：使用 `inst_executed` 建立基线
2. **识别热点**：找到指令计数高的内核  
3. **优化目标**：专注于高影响的内核
4. **验证改进**：比较优化前后的指令计数

### 与其他指标的集成

将事件数据与其他性能指标结合：

```cpp
typedef struct PerformanceProfile_st {
    uint64_t instructions;
    uint64_t globalLoads;
    uint64_t globalStores;
    double kernelTime;
    double ipc;  // 每周期指令数
    double bandwidth; // 内存带宽利用率
} PerformanceProfile;
```

## 总结

CUPTI 回调事件收集为了解 GPU 性能提供了详细的见解。通过跟踪特定事件如已执行指令，开发者可以：

- 量化计算复杂性
- 识别优化机会
- 验证性能改进
- 了解硬件利用率

结合其他 CUPTI 功能，事件回调形成了一个强大的性能分析工具包，帮助优化 CUDA 应用程序以获得最佳性能。 