# CUPTI 多 GPU 事件分析教程

> 完整的 GitHub 仓库和教程请访问 <https://github.com/eunomia-bpf/cupti-tutorial>。

## 简介

许多高性能计算系统使用多个 GPU 来加速计算。在对这类系统进行性能分析时，您需要同时从所有 GPU 收集性能数据，而不影响它们的并行执行。本教程演示如何使用 CUPTI 并发地从多个 GPU 收集性能事件，同时保持它们的独立执行。

## 学习内容

- 如何在多个 GPU 上设置事件收集
- 管理多个 CUDA 上下文的技术
- 在不序列化的情况下跨 GPU 启动和分析内核的方法
- 正确同步并从所有设备收集结果

## 理解多 GPU 分析的挑战

在分析多 GPU 应用程序时，会出现几个挑战：

1. **上下文管理**：每个 GPU 需要自己的 CUDA 上下文
2. **避免序列化**：简单的分析方法可能导致 GPU 执行序列化
3. **资源协调**：事件组和资源必须按设备管理
4. **同步点**：需要适当的同步，但不能阻塞并行执行

本教程展示如何解决这些挑战，同时保持多 GPU 执行的性能优势。

## 代码详解

### 1. 检测可用的 GPU

首先，我们需要识别所有可用的 CUDA 设备：

```cpp
// 获取设备数量
int deviceCount = 0;
RUNTIME_API_CALL(cudaGetDeviceCount(&deviceCount));

// 此示例至少需要两个设备
if (deviceCount < 2) {
    printf("此示例至少需要两个支持 CUDA 的设备，但只找到了 %d 个设备\n", deviceCount);
    return 0;
}

printf("找到 %d 个设备\n", deviceCount);

// 获取设备名称
for (int i = 0; i < deviceCount; i++) {
    cudaDeviceProp prop;
    RUNTIME_API_CALL(cudaGetDeviceProperties(&prop, i));
    printf("CUDA 设备名称：%s\n", prop.name);
}
```

### 2. 设置上下文和事件收集

对于每个 GPU，我们创建一个单独的上下文并设置事件收集：

```cpp
// 在每个设备上创建上下文
CUcontext context[MAX_DEVICES];
CUpti_EventGroup eventGroup[MAX_DEVICES];
CUpti_EventID eventId[MAX_DEVICES];
const char *eventName = "inst_executed"; // 默认事件

// 如果提供了命令行参数，则使用它
if (argc > 1) {
    eventName = argv[1];
}

// 对于每个设备，创建上下文并设置事件收集
for (int i = 0; i < deviceCount; i++) {
    // 设置当前设备
    RUNTIME_API_CALL(cudaSetDevice(i));
    
    // 为此设备创建上下文
    DRIVER_API_CALL(cuCtxCreate(&context[i], 0, i));
    
    // 获取指定事件的事件 ID
    CUPTI_CALL(cuptiEventGetIdFromName(i, eventName, &eventId[i]));
    
    // 为此设备创建事件组
    CUPTI_CALL(cuptiEventGroupCreate(context[i], &eventGroup[i], 0));
    
    // 将事件添加到组中
    CUPTI_CALL(cuptiEventGroupAddEvent(eventGroup[i], eventId[i]));
    
    // 设置收集模式为内核级别
    CUPTI_CALL(cuptiEventGroupSetAttribute(eventGroup[i], 
                                          CUPTI_EVENT_GROUP_ATTR_COLLECTION_MODE,
                                          sizeof(CUpti_EventCollectionMode), 
                                          &kernel_mode));
    
    // 启用事件组
    CUPTI_CALL(cuptiEventGroupEnable(eventGroup[i]));
}
```

此代码的关键方面：
1. 我们为每个 GPU 创建单独的上下文
2. 我们在所有设备上设置相同事件的事件收集
3. 我们使用内核模式收集来专注于内核执行

### 3. 在所有 GPU 上启动内核

现在我们在所有 GPU 上启动内核，不等待每个完成：

```cpp
// 在每个设备上分配内存并启动内核，不在它们之间同步
int *d_data[MAX_DEVICES];
size_t dataSize = sizeof(int) * ITERATIONS;

for (int i = 0; i < deviceCount; i++) {
    // 设置当前设备和上下文
    RUNTIME_API_CALL(cudaSetDevice(i));
    DRIVER_API_CALL(cuCtxSetCurrent(context[i]));
    
    // 在此设备上分配内存
    RUNTIME_API_CALL(cudaMalloc((void **)&d_data[i], dataSize));
    
    // 在此设备上启动内核
    dim3 threads(256);
    dim3 blocks((ITERATIONS + threads.x - 1) / threads.x);
    
    dummyKernel<<<blocks, threads>>>(d_data[i], ITERATIONS);
}
```

这是示例的关键部分 - 我们在所有 GPU 上启动内核而不在启动之间同步，允许它们并发执行。

### 4. 同步并读取事件值

启动所有内核后，我们同步每个设备并读取事件值：

```cpp
// 同步所有设备并读取事件值
uint64_t eventValues[MAX_DEVICES];

for (int i = 0; i < deviceCount; i++) {
    // 设置当前设备和上下文
    RUNTIME_API_CALL(cudaSetDevice(i));
    DRIVER_API_CALL(cuCtxSetCurrent(context[i]));
    
    // 同步设备以确保内核完成
    RUNTIME_API_CALL(cudaDeviceSynchronize());
    
    // 读取事件值
    size_t valueSize = sizeof(uint64_t);
    CUPTI_CALL(cuptiEventGroupReadEvent(eventGroup[i],
                                      CUPTI_EVENT_READ_FLAG_NONE,
                                      eventId[i],
                                      &valueSize,
                                      &eventValues[i]));
    
    // 为此设备打印事件值
    printf("[%d] %s: %llu\n", i, eventName, (unsigned long long)eventValues[i]);
    
    // 清理
    RUNTIME_API_CALL(cudaFree(d_data[i]));
    CUPTI_CALL(cuptiEventGroupDisable(eventGroup[i]));
    CUPTI_CALL(cuptiEventGroupDestroy(eventGroup[i]));
    DRIVER_API_CALL(cuCtxDestroy(context[i]));
}
```

此代码的关键方面：
1. 我们在启动所有内核后单独同步每个设备
2. 我们使用每个设备特定的事件组读取事件值
3. 我们为每个设备清理资源

### 5. 测试内核

此示例中使用的内核是一个简单的虚拟内核，执行固定次数的迭代：

```cpp
__global__ void dummyKernel(int *data, int iterations)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < iterations) {
        // 做一些工作来生成事件
        int value = 0;
        for (int i = 0; i < 100; i++) {
            value += i;
        }
        data[idx] = value;
    }
}
```

此内核确保每个 GPU 都有足够的工作来生成可测量的事件计数。

## 运行教程

1. 构建示例：
   ```bash
   make
   ```

2. 使用默认事件（指令计数）运行：
   ```bash
   ./event_multi_gpu
   ```

3. 尝试不同的事件：
   ```bash
   ./event_multi_gpu branch
   ```

## 理解输出

运行示例时，您会看到类似的输出：

```
找到 2 个设备
CUDA 设备名称：NVIDIA GeForce RTX 3080
CUDA 设备名称：NVIDIA GeForce RTX 3070
[0] inst_executed: 4194304
[1] inst_executed: 4194304
```

这显示了：
1. 检测到的 CUDA 设备数量
2. 每个设备的名称
3. 从每个设备收集的事件值

在此示例中，两个 GPU 执行了相同数量的指令，因为它们运行了相同的内核。在实际应用程序中，您可能会根据工作负载分布和 GPU 能力看到不同的值。

## 性能考虑

### 非序列化执行

此方法的主要好处是非序列化执行：

```
// 传统方法（序列化）：
for (int i = 0; i < deviceCount; i++) {
    cudaSetDevice(i);
    launchKernel();
    collectEvents();  // 这强制在下一个 GPU 开始前同步
}

// 我们的方法（并行）：
for (int i = 0; i < deviceCount; i++) {
    cudaSetDevice(i);
    launchKernel();   // 在所有 GPU 上启动而不等待
}
for (int i = 0; i < deviceCount; i++) {
    cudaSetDevice(i);
    collectEvents();  // 在所有 GPU 开始工作后收集
}
```

### 工作负载平衡

在实际应用程序中，考虑：

1. **设备能力差异**：较新的 GPU 可能更快完成工作
2. **内存带宽变化**：不同设备可能有不同的内存性能
3. **热量限制**：设备可能由于热量限制而降频
4. **并发内核限制**：某些设备对并发内核数量有限制

## 高级技术

### 动态负载平衡

```cpp
// 根据设备能力分配工作
void distributeWork(int deviceCount, int totalWork, int* workPerDevice) {
    cudaDeviceProp props[MAX_DEVICES];
    int totalCompute = 0;
    
    // 获取每个设备的计算能力
    for (int i = 0; i < deviceCount; i++) {
        cudaGetDeviceProperties(&props[i], i);
        totalCompute += props[i].multiProcessorCount;
    }
    
    // 按比例分配工作
    for (int i = 0; i < deviceCount; i++) {
        workPerDevice[i] = (totalWork * props[i].multiProcessorCount) / totalCompute;
    }
}
```

### 事件相关性分析

```cpp
// 分析跨设备的事件模式
void analyzeEventCorrelation(uint64_t* eventValues, int deviceCount) {
    printf("\n事件相关性分析：\n");
    
    // 计算平均值和标准差
    uint64_t sum = 0;
    for (int i = 0; i < deviceCount; i++) {
        sum += eventValues[i];
    }
    
    double mean = (double)sum / deviceCount;
    double variance = 0;
    
    for (int i = 0; i < deviceCount; i++) {
        double diff = eventValues[i] - mean;
        variance += diff * diff;
    }
    
    double stddev = sqrt(variance / deviceCount);
    
    printf("平均事件计数：%.2f\n", mean);
    printf("标准差：%.2f\n", stddev);
    printf("变异系数：%.2f%%\n", (stddev / mean) * 100);
}
```

### 错误处理和恢复

```cpp
// 健壮的多 GPU 事件收集
bool collectEventsRobust(CUpti_EventGroup* eventGroups, int deviceCount) {
    bool success = true;
    
    for (int i = 0; i < deviceCount; i++) {
        CUptiResult result = cuptiEventGroupReadEvent(
            eventGroups[i], 
            CUPTI_EVENT_READ_FLAG_NONE,
            eventId[i], 
            &valueSize, 
            &eventValues[i]
        );
        
        if (result != CUPTI_SUCCESS) {
            printf("警告：设备 %d 事件读取失败\n", i);
            eventValues[i] = 0;  // 使用默认值
            success = false;
        }
    }
    
    return success;
}
```

## 故障排除

### 常见问题

1. **设备数量不足**：确保系统至少有两个 CUDA 设备
2. **内存不足**：大工作负载可能超出设备内存
3. **驱动兼容性**：确保 CUDA 驱动与 CUPTI 版本兼容
4. **权限问题**：某些事件可能需要管理员权限

### 调试技巧

```cpp
// 启用详细的错误报告
#define CHECK_CUPTI_ERROR(call) do { \
    CUptiResult _status = call; \
    if (_status != CUPTI_SUCCESS) { \
        const char *errstr; \
        cuptiGetResultString(_status, &errstr); \
        printf("CUPTI 错误：%s:%d: %s\n", __FILE__, __LINE__, errstr); \
        exit(-1); \
    } \
} while (0)
```

## 最佳实践

### 资源管理

1. **及时清理**：始终清理 CUPTI 资源以避免内存泄漏
2. **错误处理**：实现健壮的错误处理以处理设备故障
3. **资源池化**：对于重复的分析，重用上下文和事件组

### 性能优化

1. **批量操作**：将多个操作批处理以减少 API 调用开销
2. **异步操作**：尽可能使用异步 CUDA 操作
3. **内存预分配**：预分配缓冲区以避免运行时分配开销

这个多 GPU 事件分析教程为在复杂的多设备环境中实现高效的性能监控提供了坚实的基础。 