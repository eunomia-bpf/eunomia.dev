# CUDA 内存追踪教程

> GitHub 仓库和完整教程可在 <https://github.com/eunomia-bpf/cupti-tutorial> 获取。

## 简介

CUDA 内存追踪示例演示了如何使用 CUPTI 的活动追踪功能跟踪和分析 CUDA 应用程序中的内存操作。本教程专注于内存管理、传输模式和通过详细追踪分析来优化内存使用。

## 您将学到的内容

- 如何追踪所有类型的 CUDA 内存操作
- 理解内存传输模式和瓶颈
- 分析内存分配和释放模式
- 检测内存泄漏和使用效率低下
- 优化内存带宽利用率

## 理解 CUDA 内存操作

CUDA 应用程序涉及各种类型的内存操作：

1. **内存分配**：cudaMalloc, cudaMallocPitch, cudaMallocManaged
2. **内存传输**：cudaMemcpy, cudaMemcpyAsync, 点对点传输
3. **内存映射**：cudaHostAlloc, cudaHostRegister
4. **统一内存**：cudaMallocManaged, 自动迁移
5. **内存释放**：cudaFree, cudaFreeHost

## 关键概念

### 内存域

#### 设备内存
- GPU 上的全局内存
- 纹理和表面内存
- 常量内存
- 本地内存（寄存器/共享内存）

#### 主机内存
- 可分页系统内存
- 固定（页锁定）内存
- 统一内存区域

#### 传输类型
- 主机到设备 (H2D)
- 设备到主机 (D2H)
- 设备到设备 (D2D)
- 点对点 (P2P)

## 构建示例

### 先决条件

- 带有 CUPTI 的 CUDA 工具包
- 具有多样内存操作的应用程序
- 足够的内存用于追踪缓冲区

### 构建过程

```bash
cd cuda_memory_trace
make
```

这将创建演示内存操作追踪的 `cuda_memory_trace` 可执行文件。

## 代码架构

### 内存活动追踪

```cpp
class MemoryTracer {
private:
    struct MemoryActivity {
        CUpti_ActivityKind kind;
        uint64_t start;
        uint64_t end;
        size_t bytes;
        CUdeviceptr srcPtr;
        CUdeviceptr dstPtr;
        int srcDevice;
        int dstDevice;
        cudaMemcpyKind copyKind;
    };
    
    std::vector<MemoryActivity> activities;
    std::map<CUdeviceptr, AllocationInfo> allocations;

public:
    void processActivity(CUpti_Activity* record);
    void analyzeMemoryPatterns();
    void generateMemoryReport();
};
```

### 内存分配追踪

```cpp
class AllocationTracker {
private:
    struct AllocationInfo {
        size_t size;
        uint64_t allocTime;
        uint64_t freeTime;
        bool isActive;
        std::string allocationType;
    };
    
    std::map<void*, AllocationInfo> hostAllocations;
    std::map<CUdeviceptr, AllocationInfo> deviceAllocations;
    size_t peakMemoryUsage;
    size_t currentMemoryUsage;

public:
    void recordAllocation(void* ptr, size_t size, const std::string& type, uint64_t timestamp);
    void recordDeallocation(void* ptr, uint64_t timestamp);
    void detectMemoryLeaks();
    void calculateMemoryStatistics();
};
```

## 运行示例

### 基本执行

```bash
./cuda_memory_trace
```

### 示例输出

```
=== CUDA 内存追踪分析 ===

内存分配摘要：
  总设备分配：1,024 MB
  总主机分配：512 MB
  峰值内存使用：1,536 MB
  活动分配：768 MB
  检测到的内存泄漏：0

内存传输分析：
  主机到设备：2,048 MB (平均: 8.5 GB/s)
  设备到主机：1,024 MB (平均: 7.2 GB/s)
  设备到设备：512 MB (平均: 450 GB/s)
  点对点：256 MB (平均: 28.5 GB/s)

传输模式：
  顺序传输：75.5%
  并发传输：24.5%
  最优合并：89.3%
  带宽效率：85.7%

内存热点：
  大传输 (>100MB)：12 次传输，总计 2.1 GB
  频繁小传输 (<1MB)：847 次传输，总计 45 MB
  冗余传输：23 次传输，总计 128 MB

性能问题：
  - 检测到未合并传输：18 个实例
  - 内存碎片：12.3% 开销
  - 默认流上的同步传输：156 个实例
```

## 高级内存分析

### 带宽分析

```cpp
class BandwidthAnalyzer {
private:
    struct TransferMetrics {
        double achievedBandwidth;
        double theoreticalBandwidth;
        double efficiency;
        size_t transferSize;
        cudaMemcpyKind direction;
    };
    
    std::vector<TransferMetrics> transferHistory;

public:
    void analyzeTransfer(const MemoryActivity& activity) {
        double duration = (activity.end - activity.start) * 1e-9; // 转换为秒
        double achievedBandwidth = activity.bytes / duration / 1e9; // GB/s
        
        TransferMetrics metrics;
        metrics.achievedBandwidth = achievedBandwidth;
        metrics.theoreticalBandwidth = getTheoreticalBandwidth(activity.copyKind);
        metrics.efficiency = achievedBandwidth / metrics.theoreticalBandwidth;
        metrics.transferSize = activity.bytes;
        metrics.direction = activity.copyKind;
        
        transferHistory.push_back(metrics);
    }
    
    void generateBandwidthReport() {
        std::map<cudaMemcpyKind, std::vector<double>> bandwidthByType;
        
        for (const auto& metrics : transferHistory) {
            bandwidthByType[metrics.direction].push_back(metrics.achievedBandwidth);
        }
        
        printf("\n=== 带宽分析报告 ===\n");
        for (const auto& pair : bandwidthByType) {
            const auto& bandwidths = pair.second;
            double avgBandwidth = calculateAverage(bandwidths);
            double maxBandwidth = *std::max_element(bandwidths.begin(), bandwidths.end());
            
            printf("%s:\n", getMemcpyKindString(pair.first));
            printf("  平均带宽: %.2f GB/s\n", avgBandwidth);
            printf("  峰值带宽: %.2f GB/s\n", maxBandwidth);
            printf("  传输次数: %zu\n", bandwidths.size());
        }
    }
};
```

### 内存访问模式分析

```cpp
class AccessPatternAnalyzer {
private:
    struct AccessPattern {
        CUdeviceptr baseAddress;
        size_t accessSize;
        uint64_t timestamp;
        bool isCoalesced;
        int stridePattern;
    };
    
    std::vector<AccessPattern> accessHistory;

public:
    void recordAccess(CUdeviceptr address, size_t size, uint64_t time) {
        AccessPattern pattern;
        pattern.baseAddress = address;
        pattern.accessSize = size;
        pattern.timestamp = time;
        pattern.isCoalesced = analyzeCoalescing(address, size);
        pattern.stridePattern = detectStridePattern(address);
        
        accessHistory.push_back(pattern);
    }
    
    void analyzeAccessPatterns() {
        int coalescedAccesses = 0;
        int totalAccesses = accessHistory.size();
        
        std::map<int, int> strideDistribution;
        
        for (const auto& access : accessHistory) {
            if (access.isCoalesced) {
                coalescedAccesses++;
            }
            strideDistribution[access.stridePattern]++;
        }
        
        double coalescingRatio = (double)coalescedAccesses / totalAccesses * 100.0;
        
        printf("\n=== 访问模式分析 ===\n");
        printf("合并访问比例: %.2f%%\n", coalescingRatio);
        printf("总访问次数: %d\n", totalAccesses);
        
        printf("\n步长分布:\n");
        for (const auto& pair : strideDistribution) {
            printf("  步长 %d: %d 次访问\n", pair.first, pair.second);
        }
    }
};
```

### 内存泄漏检测

```cpp
class MemoryLeakDetector {
private:
    std::map<void*, AllocationInfo> allocations;
    size_t totalAllocated;
    size_t totalFreed;

public:
    void recordAllocation(void* ptr, size_t size, const std::string& type) {
        AllocationInfo info;
        info.size = size;
        info.allocTime = getCurrentTime();
        info.allocationType = type;
        info.isActive = true;
        
        allocations[ptr] = info;
        totalAllocated += size;
    }
    
    void recordDeallocation(void* ptr) {
        auto it = allocations.find(ptr);
        if (it != allocations.end()) {
            it->second.freeTime = getCurrentTime();
            it->second.isActive = false;
            totalFreed += it->second.size;
        }
    }
    
    void detectLeaks() {
        printf("\n=== 内存泄漏检测 ===\n");
        
        size_t leakedMemory = 0;
        int leakCount = 0;
        
        for (const auto& pair : allocations) {
            if (pair.second.isActive) {
                leakedMemory += pair.second.size;
                leakCount++;
                
                printf("泄漏检测: %p (%zu 字节, %s)\n",
                       pair.first, pair.second.size, 
                       pair.second.allocationType.c_str());
            }
        }
        
        printf("总泄漏内存: %zu 字节\n", leakedMemory);
        printf("泄漏分配数: %d\n", leakCount);
        printf("总分配: %zu 字节\n", totalAllocated);
        printf("总释放: %zu 字节\n", totalFreed);
    }
};
```

## 性能优化建议

### 带宽优化

```cpp
class BandwidthOptimizer {
public:
    std::vector<std::string> analyzeAndSuggest(const std::vector<TransferMetrics>& metrics) {
        std::vector<std::string> suggestions;
        
        // 分析传输大小
        size_t smallTransfers = 0;
        size_t totalTransfers = metrics.size();
        
        for (const auto& metric : metrics) {
            if (metric.transferSize < SMALL_TRANSFER_THRESHOLD) {
                smallTransfers++;
            }
            
            if (metric.efficiency < LOW_EFFICIENCY_THRESHOLD) {
                suggestions.push_back("考虑批量处理小传输以提高效率");
            }
        }
        
        if (smallTransfers > totalTransfers * 0.3) {
            suggestions.push_back("检测到大量小传输，考虑合并传输");
        }
        
        return suggestions;
    }
};
```

### 内存使用优化

```cpp
class MemoryUsageOptimizer {
public:
    void analyzeUsagePatterns(const AllocationTracker& tracker) {
        printf("\n=== 内存使用优化建议 ===\n");
        
        // 分析分配寿命
        analyzeAllocationLifetime(tracker);
        
        // 检查内存池机会
        suggestMemoryPooling(tracker);
        
        // 分析碎片化
        analyzeFragmentation(tracker);
    }
    
private:
    void analyzeAllocationLifetime(const AllocationTracker& tracker) {
        // 分析分配的生存期，建议内存重用
        printf("建议: 实现内存池以重用短生存期分配\n");
    }
    
    void suggestMemoryPooling(const AllocationTracker& tracker) {
        // 识别可以池化的分配模式
        printf("建议: 对频繁的固定大小分配使用内存池\n");
    }
    
    void analyzeFragmentation(const AllocationTracker& tracker) {
        // 检测和报告内存碎片化
        printf("建议: 考虑内存对齐以减少碎片化\n");
    }
};
```

## 实际应用场景

### 实时监控

```cpp
class RealTimeMemoryMonitor {
private:
    std::thread monitorThread;
    std::atomic<bool> monitoring;
    std::chrono::milliseconds updateInterval;

public:
    void startMonitoring(std::chrono::milliseconds interval = std::chrono::milliseconds(100)) {
        monitoring = true;
        updateInterval = interval;
        
        monitorThread = std::thread([this]() {
            while (monitoring) {
                collectMemorySnapshot();
                std::this_thread::sleep_for(updateInterval);
            }
        });
    }
    
    void stopMonitoring() {
        monitoring = false;
        if (monitorThread.joinable()) {
            monitorThread.join();
        }
    }
    
private:
    void collectMemorySnapshot() {
        // 收集当前内存使用快照
        size_t freeMemory, totalMemory;
        cudaMemGetInfo(&freeMemory, &totalMemory);
        
        size_t usedMemory = totalMemory - freeMemory;
        double utilizationPercent = (double)usedMemory / totalMemory * 100.0;
        
        // 记录到时间序列数据
        recordMemoryUsage(usedMemory, utilizationPercent);
    }
};
```

## 故障排除

### 常见内存问题

1. **内存泄漏**：
   ```cpp
   void checkForLeaks() {
       MemoryLeakDetector detector;
       // 在应用程序结束时检查未释放的分配
       detector.detectLeaks();
   }
   ```

2. **内存访问冲突**：
   ```cpp
   void detectAccessViolations() {
       // 使用 CUDA 内存检查器检测无效访问
       // 需要编译时启用 -lineinfo 标志
   }
   ```

3. **带宽低效**：
   ```cpp
   void optimizeBandwidth() {
       BandwidthOptimizer optimizer;
       auto suggestions = optimizer.analyzeAndSuggest(transferMetrics);
       for (const auto& suggestion : suggestions) {
           printf("优化建议: %s\n", suggestion.c_str());
       }
   }
   ```

## 总结

CUDA 内存追踪为理解和优化 GPU 内存使用提供了强大的见解。通过全面追踪内存操作：

### 关键优势

- **可见性**：完整的内存操作视图
- **性能分析**：识别带宽瓶颈
- **泄漏检测**：防止内存资源浪费
- **优化指导**：数据驱动的改进建议

### 最佳实践

1. **监控内存使用模式**：识别优化机会
2. **优化传输大小**：平衡延迟和吞吐量
3. **使用异步传输**：提高并发性
4. **实现内存池**：减少分配开销
5. **定期检查泄漏**：维护应用程序健康

内存追踪是 CUDA 性能优化的基础工具，为开发高效 GPU 应用程序提供了必要的见解。 