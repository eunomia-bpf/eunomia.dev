# CUPTI 并发分析教程

> GitHub 仓库和完整教程可在 <https://github.com/eunomia-bpf/cupti-tutorial> 获取。

## 简介

CUPTI 并发分析示例演示了对使用多个流、设备和线程的复杂 CUDA 应用程序进行性能分析的高级技术。本教程展示了如何处理分析并发 GPU 操作的挑战，同时保持准确性并最小化开销。

## 您将学到的内容

- 如何分析具有多个 CUDA 流的应用程序
- 多设备分析和分析技术
- 理解 GPU 应用程序中的并发模式
- 在高吞吐量场景中管理分析开销
- 跨不同执行上下文关联活动

## 理解并发分析挑战

分析并发 CUDA 应用程序面临独特的挑战：

1. **重叠操作**：多个内核和内存传输同时执行
2. **多设备协调**：跨多个 GPU 同步分析
3. **线程安全**：处理来自多个 CPU 线程的分析数据
4. **上下文管理**：跨不同 CUDA 上下文跟踪活动
5. **时间线关联**：维护准确的时间关系

## 关键概念

### CUDA 中的并发模式

#### 基于流的并发
- 不同流上的多个操作
- 重叠内核执行和内存传输
- 异步 API 调用

#### 多设备并发
- 跨多个 GPU 并行执行
- 点对点内存传输
- 跨设备同步

#### 基于线程的并发
- 多个 CPU 线程进行 CUDA 调用
- 共享上下文和资源
- 线程本地分析数据

## 构建示例

### 先决条件

- 带有 CUPTI 的 CUDA 工具包
- 多 GPU 系统（推荐用于完整功能）
- C++11 兼容编译器

### 构建过程

```bash
cd concurrent_profiling
make
```

这将创建演示各种并发场景的 `concurrent_profiling` 可执行文件。

## 示例架构

### 测试场景

示例包含几种并发模式：

1. **单流顺序**：用于比较的基线
2. **多流并行**：并发内核执行
3. **多设备执行**：跨 GPU 工作负载分布
4. **混合工作负载**：计算和内存操作的组合

### 分析组件

```cpp
class ConcurrentProfiler {
private:
    std::vector<CUcontext> contexts;
    std::vector<std::thread> profileThreads;
    std::atomic<bool> profiling;
    ThreadSafeDataCollector collector;

public:
    void startProfiling();
    void profileDevice(int deviceId);
    void collectStreamMetrics(cudaStream_t stream);
    void generateConcurrencyReport();
};
```

## 运行示例

### 基本执行

```bash
./concurrent_profiling
```

### 示例输出

```
=== 并发分析分析 ===

设备 0 分析：
  总流数：4
  并发内核：8
  流利用率：85.3%
  
设备 1 分析：
  总流数：4
  并发内核：6
  流利用率：78.1%

并发指标：
  内核重叠比：0.73
  内存传输重叠：0.89
  跨设备带宽：28.5 GB/s
  
时间线分析：
  总执行时间：45.2ms
  顺序等效时间：124.7ms
  加速因子：2.76x
```

## 高级分析技术

### 流时间线分析

```cpp
class StreamProfiler {
private:
    struct StreamActivity {
        uint64_t startTime;
        uint64_t endTime;
        std::string activityType;
        size_t dataSize;
    };
    
    std::map<cudaStream_t, std::vector<StreamActivity>> streamTimelines;

public:
    void recordActivity(cudaStream_t stream, const std::string& type,
                       uint64_t start, uint64_t end, size_t size = 0) {
        streamTimelines[stream].push_back({start, end, type, size});
    }
    
    double calculateOverlapRatio() {
        // 分析时间线重叠
        uint64_t totalTime = 0;
        uint64_t overlappedTime = 0;
        
        // 复杂的重叠计算算法
        return static_cast<double>(overlappedTime) / totalTime;
    }
};
```

### 多设备协调

```cpp
class MultiDeviceProfiler {
private:
    std::vector<int> deviceIds;
    std::map<int, std::unique_ptr<DeviceProfiler>> deviceProfilers;

public:
    void initializeDevices() {
        int deviceCount;
        RUNTIME_API_CALL(cudaGetDeviceCount(&deviceCount));
        
        for (int i = 0; i < deviceCount; i++) {
            deviceIds.push_back(i);
            deviceProfilers[i] = std::make_unique<DeviceProfiler>(i);
        }
    }
    
    void profileAllDevices() {
        std::vector<std::thread> threads;
        
        for (int deviceId : deviceIds) {
            threads.emplace_back([this, deviceId]() {
                deviceProfilers[deviceId]->startProfiling();
            });
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
    }
};
```

### 线程安全数据收集

```cpp
class ThreadSafeCollector {
private:
    std::mutex dataMutex;
    std::condition_variable dataReady;
    std::queue<ProfilingData> dataQueue;
    std::atomic<bool> collecting;

public:
    void addData(const ProfilingData& data) {
        std::lock_guard<std::mutex> lock(dataMutex);
        dataQueue.push(data);
        dataReady.notify_one();
    }
    
    void processData() {
        std::unique_lock<std::mutex> lock(dataMutex);
        while (collecting) {
            dataReady.wait(lock, [this] { return !dataQueue.empty() || !collecting; });
            
            while (!dataQueue.empty()) {
                ProfilingData data = dataQueue.front();
                dataQueue.pop();
                lock.unlock();
                
                // 处理数据而不持有锁
                processProfilingData(data);
                
                lock.lock();
            }
        }
    }
};
```

## 并发模式分析

### 重叠检测

```cpp
class OverlapAnalyzer {
private:
    struct TimeInterval {
        uint64_t start;
        uint64_t end;
        std::string operation;
        int streamId;
    };
    
    std::vector<TimeInterval> activities;

public:
    void addActivity(const TimeInterval& activity) {
        activities.push_back(activity);
    }
    
    OverlapMetrics analyzeOverlap() {
        OverlapMetrics metrics;
        
        // 按开始时间排序活动
        std::sort(activities.begin(), activities.end(),
                 [](const TimeInterval& a, const TimeInterval& b) {
                     return a.start < b.start;
                 });
        
        // 计算重叠区间
        for (size_t i = 0; i < activities.size(); i++) {
            for (size_t j = i + 1; j < activities.size(); j++) {
                if (activitiesOverlap(activities[i], activities[j])) {
                    metrics.overlapCount++;
                    metrics.totalOverlapTime += calculateOverlapDuration(
                        activities[i], activities[j]);
                }
            }
        }
        
        return metrics;
    }
    
private:
    bool activitiesOverlap(const TimeInterval& a, const TimeInterval& b) {
        return !(a.end <= b.start || b.end <= a.start);
    }
    
    uint64_t calculateOverlapDuration(const TimeInterval& a, const TimeInterval& b) {
        uint64_t overlapStart = std::max(a.start, b.start);
        uint64_t overlapEnd = std::min(a.end, b.end);
        return overlapEnd - overlapStart;
    }
};
```

### 流利用率分析

```cpp
class StreamUtilizationAnalyzer {
private:
    struct StreamMetrics {
        uint64_t totalActiveTime;
        uint64_t totalTime;
        int activityCount;
        double averageUtilization;
    };
    
    std::map<cudaStream_t, StreamMetrics> streamMetrics;

public:
    void recordStreamActivity(cudaStream_t stream, uint64_t start, uint64_t end) {
        StreamMetrics& metrics = streamMetrics[stream];
        metrics.totalActiveTime += (end - start);
        metrics.activityCount++;
        
        // 更新总时间窗口
        if (metrics.totalTime == 0) {
            metrics.totalTime = end - start;
        } else {
            metrics.totalTime = std::max(metrics.totalTime, end - start);
        }
    }
    
    void generateUtilizationReport() {
        printf("\n=== 流利用率报告 ===\n");
        for (const auto& pair : streamMetrics) {
            const StreamMetrics& metrics = pair.second;
            double utilization = static_cast<double>(metrics.totalActiveTime) / 
                               metrics.totalTime * 100.0;
            
            printf("流 %p: %.2f%% 利用率, %d 个活动\n",
                   pair.first, utilization, metrics.activityCount);
        }
    }
};
```

## 性能瓶颈识别

### 关键路径分析

```cpp
class CriticalPathAnalyzer {
private:
    struct Dependency {
        std::string operation;
        uint64_t duration;
        std::vector<std::string> dependencies;
    };
    
    std::vector<Dependency> operations;

public:
    void addOperation(const std::string& name, uint64_t duration,
                     const std::vector<std::string>& deps) {
        operations.push_back({name, duration, deps});
    }
    
    std::vector<std::string> findCriticalPath() {
        // 实现关键路径算法
        std::map<std::string, uint64_t> earliestStart;
        std::map<std::string, uint64_t> earliestFinish;
        
        // 拓扑排序和关键路径计算
        std::vector<std::string> criticalPath;
        
        // 计算最早开始时间
        for (const auto& op : operations) {
            uint64_t maxDepFinish = 0;
            for (const auto& dep : op.dependencies) {
                maxDepFinish = std::max(maxDepFinish, earliestFinish[dep]);
            }
            earliestStart[op.operation] = maxDepFinish;
            earliestFinish[op.operation] = maxDepFinish + op.duration;
        }
        
        // 回溯找到关键路径
        // ... 实现关键路径回溯逻辑
        
        return criticalPath;
    }
};
```

### 资源争用检测

```cpp
class ResourceContentionDetector {
private:
    struct ResourceUsage {
        std::string resourceType;
        uint64_t startTime;
        uint64_t endTime;
        int deviceId;
        std::string operation;
    };
    
    std::vector<ResourceUsage> resourceUsages;

public:
    void recordResourceUsage(const std::string& resource, uint64_t start,
                           uint64_t end, int device, const std::string& op) {
        resourceUsages.push_back({resource, start, end, device, op});
    }
    
    std::vector<ResourceContention> detectContentions() {
        std::vector<ResourceContention> contentions;
        
        // 按资源类型分组
        std::map<std::string, std::vector<ResourceUsage>> byResource;
        for (const auto& usage : resourceUsages) {
            byResource[usage.resourceType].push_back(usage);
        }
        
        // 检测每种资源类型的争用
        for (const auto& pair : byResource) {
            const auto& usages = pair.second;
            
            for (size_t i = 0; i < usages.size(); i++) {
                for (size_t j = i + 1; j < usages.size(); j++) {
                    if (usagesOverlap(usages[i], usages[j])) {
                        contentions.push_back({
                            pair.first,
                            usages[i].operation,
                            usages[j].operation,
                            calculateOverlapTime(usages[i], usages[j])
                        });
                    }
                }
            }
        }
        
        return contentions;
    }
};
```

## 优化策略

### 动态负载平衡

```cpp
class DynamicLoadBalancer {
private:
    std::vector<int> deviceLoads;
    std::mutex loadMutex;

public:
    int selectOptimalDevice() {
        std::lock_guard<std::mutex> lock(loadMutex);
        
        // 找到负载最小的设备
        int minLoadDevice = 0;
        int minLoad = deviceLoads[0];
        
        for (size_t i = 1; i < deviceLoads.size(); i++) {
            if (deviceLoads[i] < minLoad) {
                minLoad = deviceLoads[i];
                minLoadDevice = i;
            }
        }
        
        // 增加选定设备的负载
        deviceLoads[minLoadDevice]++;
        
        return minLoadDevice;
    }
    
    void updateDeviceLoad(int device, int deltaLoad) {
        std::lock_guard<std::mutex> lock(loadMutex);
        deviceLoads[device] += deltaLoad;
    }
};
```

### 自适应流管理

```cpp
class AdaptiveStreamManager {
private:
    std::vector<cudaStream_t> availableStreams;
    std::queue<cudaStream_t> freeStreams;
    std::mutex streamMutex;
    
    StreamUtilizationAnalyzer utilAnalyzer;

public:
    cudaStream_t acquireStream() {
        std::lock_guard<std::mutex> lock(streamMutex);
        
        if (freeStreams.empty()) {
            // 创建新流如果需要且可行
            if (availableStreams.size() < MAX_STREAMS) {
                cudaStream_t newStream;
                RUNTIME_API_CALL(cudaStreamCreate(&newStream));
                availableStreams.push_back(newStream);
                return newStream;
            } else {
                // 等待流变为可用
                // 实现等待逻辑或返回null
                return nullptr;
            }
        }
        
        cudaStream_t stream = freeStreams.front();
        freeStreams.pop();
        return stream;
    }
    
    void releaseStream(cudaStream_t stream) {
        std::lock_guard<std::mutex> lock(streamMutex);
        freeStreams.push(stream);
    }
    
    void optimizeStreamCount() {
        // 基于利用率分析调整流数量
        auto utilizationData = utilAnalyzer.getUtilizationData();
        
        if (averageUtilization(utilizationData) < LOW_UTILIZATION_THRESHOLD) {
            // 减少流数量
            reduceStreamCount();
        } else if (averageUtilization(utilizationData) > HIGH_UTILIZATION_THRESHOLD) {
            // 增加流数量
            increaseStreamCount();
        }
    }
};
```

## 可视化和报告

### 并发时间线可视化

```cpp
class ConcurrencyVisualizer {
private:
    struct VisualEvent {
        uint64_t start;
        uint64_t end;
        std::string name;
        int streamId;
        int deviceId;
        std::string color;
    };
    
    std::vector<VisualEvent> events;

public:
    void addEvent(uint64_t start, uint64_t end, const std::string& name,
                 int stream, int device) {
        events.push_back({start, end, name, stream, device, 
                         generateColor(stream, device)});
    }
    
    void generateTimelineHTML(const std::string& filename) {
        std::ofstream file(filename);
        
        file << "<!DOCTYPE html>\n";
        file << "<html><head><title>CUDA 并发时间线</title>\n";
        file << "<script src=\"https://d3js.org/d3.v5.min.js\"></script>\n";
        file << "</head><body>\n";
        file << "<div id=\"timeline\"></div>\n";
        
        // 生成 D3.js 可视化代码
        generateD3Timeline(file);
        
        file << "</body></html>\n";
    }
    
private:
    void generateD3Timeline(std::ofstream& file) {
        file << "<script>\n";
        file << "const data = [\n";
        
        for (const auto& event : events) {
            file << "  {start: " << event.start 
                 << ", end: " << event.end
                 << ", name: \"" << event.name << "\""
                 << ", stream: " << event.streamId
                 << ", device: " << event.deviceId
                 << ", color: \"" << event.color << "\"},\n";
        }
        
        file << "];\n";
        
        // D3.js 时间线渲染代码
        file << "// D3.js 时间线可视化代码在这里\n";
        file << "</script>\n";
    }
};
```

## 总结

并发 CUDA 应用程序的分析需要专门的技术来处理多个同时执行的操作的复杂性。本教程演示的关键技术包括：

- **线程安全数据收集**：确保来自多个线程的分析数据的完整性
- **时间线分析**：理解操作如何重叠和交互
- **资源争用检测**：识别性能瓶颈
- **动态优化**：基于运行时分析调整执行策略

通过应用这些技术，开发者可以：
- 优化并发执行模式
- 提高 GPU 利用率
- 减少执行时间
- 实现更好的可扩展性

并发分析是掌握高性能 CUDA 应用程序的关键技能，特别是在现代多 GPU 系统中。 