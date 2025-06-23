# 性能监控采样教程

> GitHub 仓库和完整教程可在 <https://github.com/eunomia-bpf/cupti-tutorial> 获取。

## 简介

CUPTI 性能监控 (PM) 采样示例演示如何使用 CUPTI 的性能监控功能收集详细的 GPU 性能指标。本教程向您展示如何收集、分析和解释各种性能计数器和指标，这些指标提供对 GPU 利用率、内存带宽、指令吞吐量和其他关键性能指标的洞察。

## 您将学到什么

- 如何配置和收集 GPU 性能指标
- 理解不同类别的性能计数器
- 实现基于指标的性能分析
- 将性能指标与应用程序行为关联
- 使用性能数据进行优化指导

## 理解性能监控采样

PM 采样通过以下方式提供全面的性能洞察：

1. **硬件性能计数器**：低级 GPU 指标
2. **派生指标**：计算的性能指标
3. **实时监控**：连续性能跟踪
4. **多维分析**：同时多个指标
5. **性能关联**：将指标与应用程序阶段链接

## 关键性能指标

### 计算指标
- **sm_efficiency**：流式多处理器利用率
- **achieved_occupancy**：最大理论占用率的百分比
- **inst_per_warp**：每个线程束执行的指令数
- **ipc**：每时钟周期指令数
- **branch_efficiency**：非分歧分支的百分比

### 内存指标
- **dram_utilization**：设备内存利用率
- **tex_cache_hit_rate**：纹理缓存命中率
- **l2_cache_hit_rate**：L2 缓存命中率
- **global_hit_rate**：全局内存缓存命中率
- **shared_efficiency**：共享内存银行效率

### 吞吐量指标
- **gld_throughput**：全局加载吞吐量
- **gst_throughput**：全局存储吞吐量
- **tex_cache_throughput**：纹理缓存吞吐量
- **dram_read_throughput**：设备内存读取吞吐量
- **dram_write_throughput**：设备内存写入吞吐量

## 构建示例

### 先决条件

- 带 CUPTI 的 CUDA 工具包
- 支持性能计数器的 GPU
- 管理员权限（对于某些性能计数器）

### 构建过程

```bash
cd pm_sampling
make
```

这会创建用于性能监控的 `pm_sampling` 可执行文件。

## 运行示例

### 基本执行

```bash
./pm_sampling
```

### 示例输出

```
=== 性能监控采样结果 ===

内核：vectorAdd
性能指标分析：

计算效率：
  SM 效率：87.5%
  达到的占用率：0.73
  每线程束指令数：128.4
  IPC（每时钟指令数）：1.85
  分支效率：94.2%

内存性能：
  DRAM 利用率：45.8%
  L2 缓存命中率：78.9%
  全局内存命中率：82.3%
  纹理缓存命中率：N/A
  共享内存效率：89.4%

吞吐量指标：
  全局加载吞吐量：156.7 GB/s
  全局存储吞吐量：142.3 GB/s
  DRAM 读取吞吐量：89.5 GB/s
  DRAM 写入吞吐量：76.2 GB/s

性能分析：
  ✓ 良好的 SM 利用率 (87.5% > 80%)
  ⚠ 内存带宽未充分利用 (45.8% < 60%)
  ✓ 优秀的缓存性能 (78.9% L2 命中率)
  ✓ 最小分支分歧 (94.2% 效率)

优化建议：
  - 增加内存访问强度以更好地利用带宽
  - 考虑优化内存访问模式
  - 当前计算/内存平衡偏向计算绑定工作负载
```

## 代码架构

### 性能指标收集器

```cpp
class PerformanceMetricsCollector {
private:
    struct MetricDefinition {
        std::string name;
        CUpti_MetricID metricId;
        std::string category;
        std::string description;
        std::string unit;
    };
    
    std::vector<MetricDefinition> availableMetrics;
    std::vector<MetricDefinition> activeMetrics;
    CUpti_EventGroup eventGroup;
    std::map<std::string, double> collectedMetrics;

public:
    void initializeMetrics(CUcontext context, CUdevice device);
    void addMetric(const std::string& metricName);
    void addMetricCategory(const std::string& category);
    void startCollection();
    void stopCollection();
    void analyzeMetrics();
    void generateReport();
};
```

### 指标分析引擎

```cpp
class MetricAnalysisEngine {
private:
    struct MetricThresholds {
        double excellent;
        double good;
        double fair;
        double poor;
    };
    
    std::map<std::string, MetricThresholds> thresholds;
    std::map<std::string, double> metricValues;

public:
    void setThresholds(const std::string& metricName, 
                      double excellent, double good, double fair, double poor);
    
    std::string evaluateMetric(const std::string& metricName, double value) {
        auto it = thresholds.find(metricName);
        if (it == thresholds.end()) return "Unknown";
        
        const auto& threshold = it->second;
        if (value >= threshold.excellent) return "Excellent";
        else if (value >= threshold.good) return "Good";
        else if (value >= threshold.fair) return "Fair";
        else return "Poor";
    }
    
    void generateOptimizationSuggestions() {
        for (const auto& metric : metricValues) {
            std::string evaluation = evaluateMetric(metric.first, metric.second);
            
            if (evaluation == "Poor") {
                std::cout << "⚠ " << metric.first << " 需要优化" << std::endl;
                provideSuggestion(metric.first);
            } else if (evaluation == "Excellent") {
                std::cout << "✓ " << metric.first << " 表现优秀" << std::endl;
            }
        }
    }
};
```

## 实际应用

### 内存绑定分析

```cpp
void analyzeMemoryBound() {
    double dramUtil = getMetricValue("dram_utilization");
    double l2HitRate = getMetricValue("l2_cache_hit_rate");
    double memBandwidth = getMetricValue("gld_throughput") + getMetricValue("gst_throughput");
    
    if (dramUtil > 80.0) {
        std::cout << "检测到内存绑定内核" << std::endl;
        std::cout << "建议：" << std::endl;
        std::cout << "- 优化内存访问模式" << std::endl;
        std::cout << "- 使用共享内存减少全局内存访问" << std::endl;
        std::cout << "- 考虑内存合并" << std::endl;
    }
    
    if (l2HitRate < 60.0) {
        std::cout << "L2 缓存命中率低" << std::endl;
        std::cout << "建议：考虑数据重用模式优化" << std::endl;
    }
}
```

### 计算绑定分析

```cpp
void analyzeComputeBound() {
    double smEfficiency = getMetricValue("sm_efficiency");
    double occupancy = getMetricValue("achieved_occupancy");
    double ipc = getMetricValue("ipc");
    
    if (smEfficiency < 70.0) {
        std::cout << "SM 利用率低" << std::endl;
        std::cout << "建议：" << std::endl;
        std::cout << "- 增加并行度" << std::endl;
        std::cout << "- 检查负载平衡" << std::endl;
    }
    
    if (occupancy < 0.5) {
        std::cout << "占用率低" << std::endl;
        std::cout << "建议：" << std::endl;
        std::cout << "- 减少寄存器使用" << std::endl;
        std::cout << "- 减少共享内存使用" << std::endl;
        std::cout << "- 调整线程块大小" << std::endl;
    }
}
```

## 高级分析技术

### 多内核比较

```cpp
class MultiKernelAnalyzer {
private:
    std::map<std::string, std::map<std::string, double>> kernelMetrics;

public:
    void addKernelMetrics(const std::string& kernelName, 
                         const std::map<std::string, double>& metrics) {
        kernelMetrics[kernelName] = metrics;
    }
    
    void compareKernels() {
        std::cout << "=== 内核性能比较 ===" << std::endl;
        
        for (const auto& kernel : kernelMetrics) {
            std::cout << "\n内核：" << kernel.first << std::endl;
            
            double efficiency = kernel.second.at("sm_efficiency");
            double dramUtil = kernel.second.at("dram_utilization");
            
            std::cout << "  SM 效率：" << efficiency << "%" << std::endl;
            std::cout << "  DRAM 利用率：" << dramUtil << "%" << std::endl;
            
            // 提供内核特定的优化建议
            if (efficiency < 50.0) {
                std::cout << "  建议：优化计算密度" << std::endl;
            }
            if (dramUtil > 90.0) {
                std::cout << "  建议：优化内存访问" << std::endl;
            }
        }
    }
};
```

### 趋势分析

```cpp
class TrendAnalyzer {
private:
    std::vector<std::map<std::string, double>> historicalData;

public:
    void addSample(const std::map<std::string, double>& metrics) {
        historicalData.push_back(metrics);
    }
    
    void analyzeTrends() {
        if (historicalData.size() < 2) return;
        
        std::cout << "=== 性能趋势分析 ===" << std::endl;
        
        for (const auto& metric : historicalData.back()) {
            double current = metric.second;
            double previous = historicalData[historicalData.size()-2].at(metric.first);
            double change = ((current - previous) / previous) * 100;
            
            std::cout << metric.first << "：";
            if (change > 5.0) {
                std::cout << "↑ 改善 " << change << "%" << std::endl;
            } else if (change < -5.0) {
                std::cout << "↓ 降级 " << change << "%" << std::endl;
            } else {
                std::cout << "→ 稳定" << std::endl;
            }
        }
    }
};
```

## 最佳实践

### 指标选择

1. **专注于关键指标**：选择与您的优化目标相关的指标
2. **平衡覆盖面和开销**：更多指标意味着更高的采样开销
3. **使用分层方法**：从高级指标开始，然后深入详细指标

### 数据解释

1. **考虑上下文**：将指标与应用程序阶段关联
2. **查找模式**：识别指标之间的相关性
3. **验证假设**：使用多个指标确认性能瓶颈

### 优化工作流

```cpp
void optimizationWorkflow() {
    // 1. 基线测量
    auto baselineMetrics = collectMetrics();
    
    // 2. 识别瓶颈
    auto bottlenecks = identifyBottlenecks(baselineMetrics);
    
    // 3. 应用优化
    for (const auto& bottleneck : bottlenecks) {
        applyOptimization(bottleneck);
        
        // 4. 验证改进
        auto newMetrics = collectMetrics();
        validateImprovement(baselineMetrics, newMetrics);
    }
}
```

性能监控采样为 CUDA 应用程序优化提供了强大的数据驱动方法。通过系统地收集和分析性能指标，您可以做出明智的优化决策并跟踪改进进度。 