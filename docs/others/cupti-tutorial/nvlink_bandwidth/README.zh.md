# CUPTI NVLink 带宽监控教程

> 完整的 GitHub 仓库和教程请访问 <https://github.com/eunomia-bpf/cupti-tutorial>。

## 简介

在多 GPU 系统中，GPU 之间的通信带宽可能显著影响应用程序性能。NVLink 是 NVIDIA 的高速 GPU 互连技术，提供比传统 PCIe 连接高得多的带宽。本教程演示如何使用 CUPTI 检测 NVLink 连接并监控 GPU 之间的数据传输速率，帮助您优化多 GPU 应用程序。

## 学习内容

- 如何在多 GPU 系统中检测和识别 NVLink 连接
- 测量 NVLink 带宽利用率的技术
- 使用 CUPTI 指标实时监控 NVLink 流量
- 使用 NVLink 优化 GPU 之间的数据传输

## 理解 NVLink

NVLink 是一种高带宽互连技术，使 GPU 之间能够以远高于 PCIe 的速率直接通信：

- **带宽**：每个 NVLink 连接每个方向高达 25-50 GB/s（取决于 GPU 代数）
- **拓扑结构**：GPU 可以以各种配置连接（网格、混合立方网格等）
- **可扩展性**：多个 NVLink 连接可在 GPU 对之间使用以增加带宽
- **双向**：两个方向同时数据传输

## 代码详解

### 1. 检测 NVLink 连接

首先，我们需要识别哪些 GPU 通过 NVLink 连接：

```cpp
void detectNVLinkConnections()
{
    int deviceCount = 0;
    RUNTIME_API_CALL(cudaGetDeviceCount(&deviceCount));
    
    printf("在 %d 个 GPU 中检测 NVLink 连接...\n", deviceCount);
    
    // 创建矩阵跟踪连接
    int connections[MAX_DEVICES][MAX_DEVICES] = {0};
    
    // 对于每个设备对，检查它们是否通过 NVLink 连接
    for (int i = 0; i < deviceCount; i++) {
        for (int j = i + 1; j < deviceCount; j++) {
            // 获取这些 GPU 之间的 NVLink 数量
            int nvlinkStatus = 0;
            CUdevice device1, device2;
            
            DRIVER_API_CALL(cuDeviceGet(&device1, i));
            DRIVER_API_CALL(cuDeviceGet(&device2, j));
            
            // 检查设备是否通过 NVLink 连接
            DRIVER_API_CALL(cuDeviceGetP2PAttribute(&nvlinkStatus, 
                                                 CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED,
                                                 device1, device2));
            
            // 如果存在 NVLink，确定有多少个链接
            if (nvlinkStatus) {
                int numLinks = 0;
                DRIVER_API_CALL(cuDeviceGetP2PAttribute(&numLinks,
                                                     CU_DEVICE_P2P_ATTRIBUTE_CUDA_ARRAY_ACCESS_SUPPORTED,
                                                     device1, device2));
                
                connections[i][j] = numLinks;
                connections[j][i] = numLinks; // 矩阵是对称的
                
                printf("  GPU %d <-> GPU %d：%d 个 NVLink 连接\n", i, j, numLinks);
            }
        }
    }
}
```

此函数：
1. 枚举系统中的所有 CUDA 设备
2. 对于每对设备，检查它们是否通过 NVLink 连接
3. 确定每对之间的 NVLink 连接数量
4. 构建表示 NVLink 拓扑的连接矩阵

### 2. 为 NVLink 监控设置 CUPTI 指标

要监控 NVLink 带宽，我们需要设置适当的 CUPTI 指标：

```cpp
void setupNVLinkMetrics(NVLinkMetrics *metrics, int deviceId)
{
    CUdevice device;
    DRIVER_API_CALL(cuDeviceGet(&device, deviceId));
    
    // 获取 NVLink 传输和接收带宽的指标 ID
    CUPTI_CALL(cuptiMetricGetIdFromName(device, "nvlink_total_data_transmitted", 
                                      &metrics->transmitMetricId));
    CUPTI_CALL(cuptiMetricGetIdFromName(device, "nvlink_total_data_received",
                                      &metrics->receiveMetricId));
    
    // 为这些指标创建事件组
    CUcontext context;
    DRIVER_API_CALL(cuCtxCreate(&context, 0, device));
    metrics->context = context;
    
    // 获取传输指标所需的事件
    uint32_t numTransmitEvents = 0;
    CUPTI_CALL(cuptiMetricGetNumEvents(metrics->transmitMetricId, &numTransmitEvents));
    
    CUpti_EventID *transmitEvents = (CUpti_EventID *)malloc(numTransmitEvents * sizeof(CUpti_EventID));
    CUPTI_CALL(cuptiMetricEnumEvents(metrics->transmitMetricId, &numTransmitEvents, transmitEvents));
    
    // 为传输事件创建事件组
    CUPTI_CALL(cuptiEventGroupCreate(context, &metrics->transmitEventGroup, 0));
    
    // 将每个事件添加到组中
    for (uint32_t i = 0; i < numTransmitEvents; i++) {
        CUPTI_CALL(cuptiEventGroupAddEvent(metrics->transmitEventGroup, transmitEvents[i]));
    }
    
    // 类似地设置接收指标事件
    // ...
    
    // 启用事件组
    CUPTI_CALL(cuptiEventGroupEnable(metrics->transmitEventGroup));
    CUPTI_CALL(cuptiEventGroupEnable(metrics->receiveEventGroup));
    
    free(transmitEvents);
}
```

此函数：
1. 获取 NVLink 传输和接收带宽的指标 ID
2. 识别这些指标所需的事件
3. 为收集这些事件创建事件组
4. 启用事件组进行监控

### 3. 运行带宽测试

要测量 NVLink 带宽，我们将在连接的 GPU 之间执行内存传输：

```cpp
void runBandwidthTest(int srcDevice, int dstDevice, size_t dataSize, NVLinkMetrics *metrics)
{
    printf("传输 GPU %d -> GPU %d：\n", srcDevice, dstDevice);
    
    // 设置源设备
    RUNTIME_API_CALL(cudaSetDevice(srcDevice));
    
    // 在源设备上分配内存
    void *srcData;
    RUNTIME_API_CALL(cudaMalloc(&srcData, dataSize));
    
    // 初始化源数据
    RUNTIME_API_CALL(cudaMemset(srcData, 0xA5, dataSize));
    
    // 设置目标设备
    RUNTIME_API_CALL(cudaSetDevice(dstDevice));
    
    // 在目标设备上分配内存
    void *dstData;
    RUNTIME_API_CALL(cudaMalloc(&dstData, dataSize));
    
    // 启用对等访问
    RUNTIME_API_CALL(cudaDeviceEnablePeerAccess(srcDevice, 0));
    
    // 读取初始指标值
    uint64_t startTransmit = readNVLinkMetric(metrics, true);
    uint64_t startReceive = readNVLinkMetric(metrics, false);
    
    // 记录开始时间
    cudaEvent_t start, stop;
    RUNTIME_API_CALL(cudaEventCreate(&start));
    RUNTIME_API_CALL(cudaEventCreate(&stop));
    RUNTIME_API_CALL(cudaEventRecord(start));
    
    // 执行内存传输
    RUNTIME_API_CALL(cudaMemcpy(dstData, srcData, dataSize, cudaMemcpyDeviceToDevice));
    
    // 记录结束时间
    RUNTIME_API_CALL(cudaEventRecord(stop));
    RUNTIME_API_CALL(cudaEventSynchronize(stop));
    
    // 读取最终指标值
    uint64_t endTransmit = readNVLinkMetric(metrics, true);
    uint64_t endReceive = readNVLinkMetric(metrics, false);
    
    // 计算经过的时间
    float milliseconds = 0;
    RUNTIME_API_CALL(cudaEventElapsedTime(&milliseconds, start, stop));
    float seconds = milliseconds / 1000.0f;
    
    // 计算带宽
    float gbps = (dataSize / (1024.0f * 1024.0f * 1024.0f)) / seconds;
    
    // 计算 NVLink 利用率
    uint64_t transmittedBytes = endTransmit - startTransmit;
    uint64_t receivedBytes = endReceive - startReceive;
    
    printf("  数据大小：%.1f MB\n", dataSize / (1024.0f * 1024.0f));
    printf("  时间：%.3f 秒\n", seconds);
    printf("  带宽：%.1f GB/s\n", gbps);
    printf("  NVLink 指标：\n");
    printf("    传输：%.1f MB\n", transmittedBytes / (1024.0f * 1024.0f));
    printf("    接收：%.1f MB\n", receivedBytes / (1024.0f * 1024.0f));
    
    // 清理
    RUNTIME_API_CALL(cudaEventDestroy(start));
    RUNTIME_API_CALL(cudaEventDestroy(stop));
    RUNTIME_API_CALL(cudaFree(srcData));
    RUNTIME_API_CALL(cudaFree(dstData));
}
```

此函数：
1. 在源和目标 GPU 上分配内存
2. 启用 GPU 之间的对等访问
3. 记录起始 NVLink 指标值
4. 执行设备到设备的内存传输
5. 记录结束 NVLink 指标值
6. 计算并显示实现的带宽

### 4. 读取 NVLink 指标

要读取当前的 NVLink 指标：

```cpp
uint64_t readNVLinkMetric(NVLinkMetrics *metrics, bool isTransmit)
{
    CUpti_EventGroup group = isTransmit ? metrics->transmitEventGroup : metrics->receiveEventGroup;
    CUpti_MetricID metricId = isTransmit ? metrics->transmitMetricId : metrics->receiveMetricId;
    
    // 读取事件值
    size_t eventValueBufferSize = metrics->numEvents * sizeof(uint64_t);
    uint64_t *eventValues = (uint64_t *)malloc(eventValueBufferSize);
    
    // 对于组中的每个事件，读取其值
    CUpti_EventID *eventIds = metrics->eventIds;
    for (int i = 0; i < metrics->numEvents; i++) {
        size_t valueSize = sizeof(uint64_t);
        CUPTI_CALL(cuptiEventGroupReadEvent(group, CUPTI_EVENT_READ_FLAG_NONE, 
                                          eventIds[i], &valueSize, &eventValues[i]));
    }
    
    // 从事件值计算指标值
    double metricValue = 0.0;
    CUPTI_CALL(cuptiMetricGetValue(metrics->device, metricId,
                                 metrics->numEvents * sizeof(CUpti_EventID), eventIds,
                                 metrics->numEvents * sizeof(uint64_t), eventValues,
                                 0, &metricValue));
    
    free(eventValues);
    
    return (uint64_t)metricValue;
}
```

## 运行教程

1. 构建示例：
   ```bash
   make
   ```

2. 运行 NVLink 带宽测试：
   ```bash
   ./nvlink_bandwidth
   ```

## 理解输出

示例产生类似的输出：

```
在 4 个 GPU 中检测 NVLink 连接...
  GPU 0 <-> GPU 1：2 个 NVLink 连接
  GPU 2 <-> GPU 3：2 个 NVLink 连接
  GPU 0 <-> GPU 2：1 个 NVLink 连接
  GPU 1 <-> GPU 3：1 个 NVLink 连接

传输 GPU 0 -> GPU 1：
  数据大小：512.0 MB
  时间：0.023 秒
  带宽：22.3 GB/s
  NVLink 指标：
    传输：512.0 MB
    接收：512.0 MB

传输 GPU 0 -> GPU 2：
  数据大小：512.0 MB
  时间：0.041 秒
  带宽：12.5 GB/s
  NVLink 指标：
    传输：512.0 MB
    接收：512.0 MB
```

从此输出我们可以看到：
1. **拓扑结构**：GPU 0 和 1 有 2 个 NVLink 连接，提供更高的带宽
2. **性能差异**：更多连接的 GPU 对实现更高的带宽
3. **指标验证**：NVLink 指标确认了预期的数据传输量

## 高级 NVLink 分析

### 拓扑结构优化

```cpp
class NVLinkTopologyAnalyzer {
private:
    int deviceCount;
    int connections[MAX_DEVICES][MAX_DEVICES];
    
public:
    void findOptimalPath(int src, int dst) {
        // 使用 Dijkstra 算法找到最佳路径
        std::vector<int> path = findShortestPath(src, dst);
        
        printf("GPU %d 到 GPU %d 的最佳路径：", src, dst);
        for (int device : path) {
            printf("%d ", device);
        }
        printf("\n");
    }
    
    void analyzeBottlenecks() {
        // 识别拓扑中的瓶颈
        for (int i = 0; i < deviceCount; i++) {
            int linkCount = 0;
            for (int j = 0; j < deviceCount; j++) {
                linkCount += connections[i][j];
            }
            
            if (linkCount < 2) {
                printf("警告：GPU %d 的 NVLink 连接有限\n", i);
            }
        }
    }
};
```

### 动态带宽监控

```cpp
class BandwidthMonitor {
private:
    std::map<std::pair<int,int>, std::vector<float>> bandwidthHistory;
    
public:
    void recordBandwidth(int src, int dst, float bandwidth) {
        bandwidthHistory[{src, dst}].push_back(bandwidth);
    }
    
    void generateReport() {
        for (const auto& entry : bandwidthHistory) {
            int src = entry.first.first;
            int dst = entry.first.second;
            const auto& history = entry.second;
            
            float avgBandwidth = std::accumulate(history.begin(), history.end(), 0.0f) / history.size();
            float maxBandwidth = *std::max_element(history.begin(), history.end());
            float minBandwidth = *std::min_element(history.begin(), history.end());
            
            printf("GPU %d -> GPU %d：平均 %.1f GB/s，最大 %.1f GB/s，最小 %.1f GB/s\n",
                   src, dst, avgBandwidth, maxBandwidth, minBandwidth);
        }
    }
};
```

### 负载平衡优化

```cpp
class NVLinkLoadBalancer {
private:
    struct LinkUsage {
        int src, dst;
        float utilization;
        uint64_t lastUpdateTime;
    };
    
    std::vector<LinkUsage> linkUsages;
    
public:
    int selectOptimalTarget(int srcDevice, const std::vector<int>& candidates) {
        int bestTarget = candidates[0];
        float lowestUtilization = 1.0f;
        
        for (int candidate : candidates) {
            float utilization = getLinkUtilization(srcDevice, candidate);
            if (utilization < lowestUtilization) {
                lowestUtilization = utilization;
                bestTarget = candidate;
            }
        }
        
        return bestTarget;
    }
    
    void updateUtilization(int src, int dst, float newUtilization) {
        for (auto& usage : linkUsages) {
            if (usage.src == src && usage.dst == dst) {
                usage.utilization = newUtilization;
                usage.lastUpdateTime = getCurrentTimestamp();
                break;
            }
        }
    }
};
```

## 性能优化策略

### 数据分配策略

```cpp
void optimizeDataPlacement(void** dataPointers, size_t* dataSizes, int numArrays) {
    // 分析数据访问模式
    std::map<std::pair<int,int>, float> transferFrequency;
    
    // 基于 NVLink 拓扑优化数据放置
    for (int i = 0; i < numArrays; i++) {
        int optimalDevice = findOptimalDevice(i, transferFrequency);
        
        printf("将数组 %d 放置在 GPU %d 上\n", i, optimalDevice);
        cudaSetDevice(optimalDevice);
        cudaMalloc(&dataPointers[i], dataSizes[i]);
    }
}
```

### 批量传输优化

```cpp
class BatchTransferOptimizer {
public:
    void optimizeBatchSize(int src, int dst, size_t totalData) {
        // 测试不同的批量大小
        std::vector<size_t> batchSizes = {1<<20, 1<<22, 1<<24, 1<<26}; // 1MB 到 64MB
        
        float bestBandwidth = 0;
        size_t optimalBatchSize = batchSizes[0];
        
        for (size_t batchSize : batchSizes) {
            float bandwidth = measureBandwidth(src, dst, batchSize);
            printf("批量大小 %zu MB：带宽 %.1f GB/s\n", 
                   batchSize/(1024*1024), bandwidth);
            
            if (bandwidth > bestBandwidth) {
                bestBandwidth = bandwidth;
                optimalBatchSize = batchSize;
            }
        }
        
        printf("最佳批量大小：%zu MB（%.1f GB/s）\n", 
               optimalBatchSize/(1024*1024), bestBandwidth);
    }
};
```

## 故障排除

### 常见问题

1. **NVLink 检测失败**：确保您的 GPU 支持 NVLink 并且正确连接
2. **带宽低于预期**：检查对等访问是否启用，验证拓扑配置
3. **指标收集错误**：某些指标可能需要管理员权限或特定驱动版本

### 调试技巧

```cpp
void debugNVLinkIssues() {
    // 检查对等访问状态
    for (int i = 0; i < deviceCount; i++) {
        for (int j = 0; j < deviceCount; j++) {
            if (i != j) {
                int canAccess = 0;
                cudaDeviceCanAccessPeer(&canAccess, i, j);
                printf("GPU %d 可以访问 GPU %d：%s\n", 
                       i, j, canAccess ? "是" : "否");
            }
        }
    }
    
    // 验证 NVLink 状态
    for (int i = 0; i < deviceCount; i++) {
        cudaSetDevice(i);
        // 检查设备属性和 NVLink 状态
        printDeviceNVLinkInfo(i);
    }
}
```

## 最佳实践

### 性能监控

1. **基准测试**：建立不同工作负载的基准带宽
2. **持续监控**：在生产中监控 NVLink 利用率
3. **容量规划**：基于带宽需求规划 GPU 拓扑

### 应用程序设计

1. **数据局部性**：最小化跨 NVLink 的数据移动
2. **管道化**：重叠计算和通信
3. **负载平衡**：在 NVLink 连接之间分布工作负载

这个 NVLink 带宽监控教程为优化多 GPU 应用程序的互连性能提供了强大的工具和技术。 