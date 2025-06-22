# CUPTI 事件采样教程

> GitHub 仓库和完整教程可在 <https://github.com/eunomia-bpf/cupti-tutorial> 获取。

## 简介

在对CUDA应用程序进行性能分析时，您通常需要在应用程序运行时监控性能指标。本教程演示如何使用CUPTI的事件采样功能在内核执行期间定期收集GPU性能数据，让您实时了解应用程序的行为。

## 您将学到什么

- 如何在NVIDIA GPU上设置连续事件采样
- 在内核运行时监控事件的技术
- 创建多线程性能分析系统
- 解释采样事件数据以进行性能分析

## 理解事件采样

与在内核执行结束时给出单个值的一次性事件收集不同，事件采样允许您：

1. 监控事件在内核执行期间如何随时间变化
2. 检测性能变化和异常
3. 将GPU活动与算法的特定阶段关联
4. 观察动态工作负载的影响

## 代码演练

### 1. 设置采样线程

此示例的核心是专用采样线程，在主线程运行计算时收集事件数据：

```cpp
static void *sampling_func(void *arg)
{
    SamplingInfo *info = (SamplingInfo *)arg;
    CUcontext context = info->context;
    CUdevice device = info->device;
    
    // 使此线程使用与主线程相同的CUDA上下文
    cuCtxSetCurrent(context);
    
    // 设置我们想要监控的事件
    CUpti_EventGroup eventGroup;
    CUpti_EventID eventId;
    
    // 获取指定事件的事件ID（默认为"inst_executed"）
    CUPTI_CALL(cuptiEventGetIdFromName(device, info->eventName, &eventId));
    
    // 为设备创建事件组
    CUPTI_CALL(cuptiEventGroupCreate(context, &eventGroup, 0));
    
    // 将事件添加到组中
    CUPTI_CALL(cuptiEventGroupAddEvent(eventGroup, eventId));
    
    // 设置连续收集模式（对执行期间采样至关重要）
    CUPTI_CALL(cuptiEventGroupSetAttribute(eventGroup, 
                                          CUPTI_EVENT_GROUP_ATTR_COLLECTION_MODE,
                                          sizeof(CUpti_EventCollectionMode), 
                                          &continuous));
    
    // 启用事件组
    CUPTI_CALL(cuptiEventGroupEnable(eventGroup));
    
    // 采样直到计算完成
    while (!info->terminate) {
        // 读取当前事件值
        size_t valueSize = sizeof(uint64_t);
        uint64_t eventValue = 0;
        
        CUPTI_CALL(cuptiEventGroupReadEvent(eventGroup,
                                           CUPTI_EVENT_READ_FLAG_NONE,
                                           eventId,
                                           &valueSize,
                                           &eventValue));
        
        // 打印当前值
        printf("%s: %llu\n", info->eventName, (unsigned long long)eventValue);
        
        // 等待再次采样
        millisleep(SAMPLE_PERIOD_MS);
    }
    
    // 清理
    CUPTI_CALL(cuptiEventGroupDisable(eventGroup));
    CUPTI_CALL(cuptiEventGroupDestroy(eventGroup));
    
    return NULL;
}
```

此代码的关键方面：

1. **上下文共享**：采样线程使用与主线程相同的CUDA上下文
2. **连续收集模式**：启用在内核运行时读取事件值
3. **定期采样**：以固定间隔读取事件值（默认50ms）
4. **非阻塞**：采样不会中断内核执行

### 2. 计算线程（主线程）

主线程运行我们想要分析的实际计算：

```cpp
int main(int argc, char *argv[])
{
    // 初始化CUDA并获取设备/上下文
    CUdevice device;
    CUcontext context;
    
    // 初始化CUDA驱动API
    DRIVER_API_CALL(cuInit(0));
    DRIVER_API_CALL(cuDeviceGet(&device, 0));
    DRIVER_API_CALL(cuCtxCreate(&context, 0, device));
    
    // 设置采样信息
    SamplingInfo samplingInfo;
    samplingInfo.device = device;
    samplingInfo.context = context;
    samplingInfo.terminate = 0;
    
    // 默认为"inst_executed"或使用命令行参数
    if (argc > 1) {
        samplingInfo.eventName = argv[1];
    } else {
        samplingInfo.eventName = "inst_executed";
    }
    
    // 创建并启动采样线程
    pthread_t sampling_thread;
    pthread_create(&sampling_thread, NULL, sampling_func, &samplingInfo);
    
    // 为向量加法分配内存
    float *d_A, *d_B, *d_C;
    float *h_A, *h_B, *h_C;
    size_t size = VECTOR_SIZE * sizeof(float);
    
    // 分配并初始化主机内存
    h_A = (float *)malloc(size);
    h_B = (float *)malloc(size);
    h_C = (float *)malloc(size);
    
    // 初始化向量
    for (int i = 0; i < VECTOR_SIZE; i++) {
        h_A[i] = i * 1.0f;
        h_B[i] = i * 2.0f;
    }
    
    // 分配设备内存
    RUNTIME_API_CALL(cudaMalloc((void **)&d_A, size));
    RUNTIME_API_CALL(cudaMalloc((void **)&d_B, size));
    RUNTIME_API_CALL(cudaMalloc((void **)&d_C, size));
    
    // 将主机内存复制到设备
    RUNTIME_API_CALL(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    RUNTIME_API_CALL(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
    
    // 多次启动内核以便有时间采样
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((VECTOR_SIZE + threadsPerBlock.x - 1) / threadsPerBlock.x);
    
    for (int i = 0; i < ITERATIONS; i++) {
        vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, VECTOR_SIZE);
    }
    
    // 确保所有内核都完成
    RUNTIME_API_CALL(cudaDeviceSynchronize());
    
    // 通知采样线程终止
    samplingInfo.terminate = 1;
    
    // 等待采样线程完成
    pthread_join(sampling_thread, NULL);
    
    // 清理并退出
    free(h_A);
    free(h_B);
    free(h_C);
    RUNTIME_API_CALL(cudaFree(d_A));
    RUNTIME_API_CALL(cudaFree(d_B));
    RUNTIME_API_CALL(cudaFree(d_C));
    
    DRIVER_API_CALL(cuCtxDestroy(context));
    
    return 0;
}
```

此代码的关键方面：

1. **长期运行的工作负载**：内核运行2000次以确保我们有足够时间收集样本
2. **线程协调**：主线程在计算完成时通知采样线程
3. **简单测试内核**：使用向量加法作为采样的测试用例

### 3. 向量加法内核

```cpp
__global__ void vecAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
        C[i] = A[i] + B[i];
}
```

这个简单的内核将两个向量相加。虽然不是计算密集型的，但重复运行为我们提供了一致的工作负载来监控。

## 运行教程

1. 构建示例：
   ```bash
   make
   ```

2. 使用默认事件（指令计数）运行：
   ```bash
   ./event_sampling
   ```

3. 尝试不同的事件：
   ```bash
   ./event_sampling branch
   ```

## 理解输出

使用默认的"inst_executed"事件运行时，您将看到如下输出：

```
inst_executed: 0
inst_executed: 25600000
inst_executed: 51200000
inst_executed: 76800000
inst_executed: 102400000
...
inst_executed: 4582400000
inst_executed: 4608000000
```

每行表示：
1. 被采样事件的名称
2. 采样时该事件的累积计数

在这种情况下，我们看到GPU执行的指令总数，随着我们的内核运行而稳步增加。规律的增量显示我们的工作负载在时间上执行得一致。

## 可采样的可用事件

不同的GPU支持不同的事件。您可能想要采样的一些常见事件包括：

- `inst_executed`：执行的指令
- `branch`：执行的分支指令
- `divergent_branch`：发散分支指令
- `active_cycles`：至少一个warp活跃的周期 