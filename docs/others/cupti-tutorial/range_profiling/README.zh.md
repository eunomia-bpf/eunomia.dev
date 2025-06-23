# CUPTI 范围性能分析教程

> GitHub 仓库和完整教程可在 <https://github.com/eunomia-bpf/cupti-tutorial> 获取。

## 简介

CUPTI范围性能分析示例演示如何在CUDA应用程序中使用自定义定义的范围实现有针对性的性能分析。这种技术允许您分析特定的代码段、算法阶段或功能组件，精确控制测量和分析的内容。

## 您将学到什么

- 如何定义和管理自定义性能分析范围
- 实现选择性性能测量
- 理解基于范围的指标收集
- 创建分层性能分析
- 构建有针对性的优化策略

## 理解范围性能分析

范围性能分析通过允许您执行以下操作提供focused性能分析：

1. **定义特定区域**：标记精确的代码段进行分析
2. **控制测量范围**：仅分析对您重要的内容
3. **减少开销**：通过针对特定区域最小化性能分析影响
4. **创建性能基线**：为特定函数建立指标
5. **启用比较分析**：比较相同功能的不同实现

## 关键概念

### 范围定义

范围由以下定义：
- **起始点**：测量区域的开始
- **结束点**：测量区域的结束
- **范围名称**：区域的描述性标识符
- **范围类别**：用于分组相关范围的分类
- **相关指标**：在范围内收集的性能计数器

### 范围类型

#### 函数级范围
分析整个函数或主要算法组件

#### 循环级范围
测量特定的迭代模式或计算循环

#### 阶段级范围
跟踪复杂算法的不同阶段

#### 条件范围
基于运行时条件分析代码路径

## 构建示例

### 先决条件

- 带CUPTI的CUDA工具包
- 支持C++11的C++编译器
- 用于增强范围可视化的NVTX库

### 构建过程

```bash
cd range_profiling
make
```

这创建了演示有针对性性能分析技术的`range_profiling`可执行文件。

## 代码架构

### 范围管理系统

```cpp
class RangeProfiler {
private:
    struct ProfileRange {
        std::string name;
        std::string category;
        uint64_t startTime;
        uint64_t endTime;
        bool isActive;
        std::map<std::string, uint64_t> metrics;
    };
    
    std::stack<ProfileRange*> activeRanges;
    std::vector<ProfileRange> completedRanges;
    std::map<std::string, CUpti_EventGroup> eventGroups;

public:
    void startRange(const std::string& name, const std::string& category = "default");
    void endRange();
    void addMetric(const std::string& metricName);
    void generateReport();
};
```

### RAII范围助手

```cpp
class ScopedProfileRange {
private:
    RangeProfiler& profiler;
    bool isValid;

public:
    ScopedProfileRange(RangeProfiler& prof, const std::string& name, const std::string& category = "default")
        : profiler(prof), isValid(true) {
        profiler.startRange(name, category);
    }
    
    ~ScopedProfileRange() {
        if (isValid) {
            profiler.endRange();
        }
    }
    
    // 移动语义
    ScopedProfileRange(ScopedProfileRange&& other) noexcept 
        : profiler(other.profiler), isValid(other.isValid) {
        other.isValid = false;
    }
};

// 便利宏
#define PROFILE_RANGE(profiler, name, category) \
    ScopedProfileRange _prof_range(profiler, name, category)
```

## 运行示例

### 基本执行

```bash
./range_profiling
```

### 示例输出

```
=== 范围性能分析结果 ===

范围："矩阵初始化"（类别：setup）
  持续时间：2.3ms
  执行的指令：1,245,678
  内存带宽：12.5 GB/s
  缓存命中率：94.2%

范围："矩阵乘法核心"（类别：compute）
  持续时间：45.7ms
  执行的指令：89,456,123
  FLOPS：2.1 TFLOPS
  内存带宽：385.2 GB/s
  计算利用率：87.3%

范围："结果验证"（类别：verification）
  持续时间：8.1ms
  执行的指令：3,876,234
  内存带宽：45.6 GB/s
  分支效率：96.8%

按类别的性能摘要：
  setup：2.3ms（4.1%）
  compute：45.7ms（81.2%）
  verification：8.1ms（14.4%）
  other：0.2ms（0.3%）

总执行时间：56.3ms
性能分析开销：0.8ms（1.4%）
```

## 高级范围功能

### 条件范围性能分析

```cpp
class ConditionalRangeProfiler {
private:
    std::map<std::string, bool> enabledCategories;
    std::map<std::string, int> rangeCounts;
    int maxRangeCount;

public:
    void setCategory(const std::string& category, bool enabled) {
        enabledCategories[category] = enabled;
    }
    
    void setMaxCount(const std::string& rangeName, int maxCount) {
        rangeCounts[rangeName] = maxCount;
    }
    
    bool shouldProfile(const std::string& name, const std::string& category) {
        // 检查类别过滤器
        auto catIt = enabledCategories.find(category);
        if (catIt != enabledCategories.end() && !catIt->second) {
            return false;
        }
        
        // 检查计数限制
        auto countIt = rangeCounts.find(name);
        if (countIt != rangeCounts.end() && countIt->second <= 0) {
            return false;
        }
        
        return true;
    }
    
    void recordRangeExecution(const std::string& name) {
        auto it = rangeCounts.find(name);
        if (it != rangeCounts.end()) {
            it->second--;
        }
    }
};

// 使用示例
void conditionallyProfiledFunction() {
    ConditionalRangeProfiler& condProf = ConditionalRangeProfiler::getInstance();
    
    if (condProf.shouldProfile("detailed_analysis", "debug")) {
        PROFILE_RANGE(profiler, "详细分析", "debug");
        // 详细性能分析代码
        performDetailedAnalysis();
        condProf.recordRangeExecution("detailed_analysis");
    } else {
        // 轻量级或无性能分析
        performStandardAnalysis();
    }
}
```

### 自定义指标集成

```cpp
class MetricIntegratedRangeProfiler {
private:
    struct CustomMetrics {
        std::vector<CUpti_EventID> events;
        std::vector<CUpti_MetricID> metrics;
        CUpti_EventGroup eventGroup;
        CUpti_MetricValueKind valueKind;
    };
    
    std::map<std::string, CustomMetrics> rangeMetrics;

public:
    void configureRangeMetrics(const std::string& rangeName, 
                              const std::vector<std::string>& eventNames,
                              const std::vector<std::string>& metricNames) {
        CustomMetrics metrics;
        
        // 设置事件
        for (const auto& eventName : eventNames) {
            CUpti_EventID eventId;
            CUPTI_CALL(cuptiEventGetIdFromName(device, eventName.c_str(), &eventId));
            metrics.events.push_back(eventId);
        }
        
        // 设置指标
        for (const auto& metricName : metricNames) {
            CUpti_MetricID metricId;
            CUPTI_CALL(cuptiMetricGetIdFromName(device, metricName.c_str(), &metricId));
            metrics.metrics.push_back(metricId);
        }
        
        // 创建事件组
        CUPTI_CALL(cuptiEventGroupCreate(context, &metrics.eventGroup, 0));
        for (auto eventId : metrics.events) {
            CUPTI_CALL(cuptiEventGroupAddEvent(metrics.eventGroup, eventId));
        }
        
        rangeMetrics[rangeName] = metrics;
    }
    
    void startRangeWithMetrics(const std::string& rangeName) {
        auto it = rangeMetrics.find(rangeName);
        if (it != rangeMetrics.end()) {
            CUPTI_CALL(cuptiEventGroupEnable(it->second.eventGroup));
        }
        
        startRange(rangeName);
    }
    
    void endRangeWithMetrics(const std::string& rangeName) {
        auto it = rangeMetrics.find(rangeName);
        if (it != rangeMetrics.end()) {
            // 读取事件值
            uint64_t eventValues[it->second.events.size()];
            size_t valueSize = sizeof(eventValues);
            
            CUPTI_CALL(cuptiEventGroupReadAllEvents(it->second.eventGroup,
                       CUPTI_EVENT_READ_FLAG_NONE,
                       &valueSize, eventValues,
                       nullptr, nullptr));
            
            // 计算指标
            for (size_t i = 0; i < it->second.metrics.size(); i++) {
                CUpti_MetricValue metricValue;
                CUPTI_CALL(cuptiMetricGetValue(device, it->second.metrics[i],
                           it->second.events.size(), it->second.events.data(),
                           eventValues, 0, &metricValue));
                
                // 存储指标值
                recordMetricValue(rangeName, it->second.metrics[i], metricValue);
            }
            
            CUPTI_CALL(cuptiEventGroupDisable(it->second.eventGroup));
        }
        
        endRange();
    }
};
```

### 统计范围分析

```cpp
class StatisticalRangeAnalyzer {
private:
    struct RangeStatistics {
        std::string name;
        std::string category;
        std::vector<double> durations;
        std::map<std::string, std::vector<double>> metricValues;
        
        double getMean() const {
            return std::accumulate(durations.begin(), durations.end(), 0.0) / durations.size();
        }
        
        double getStdDev() const {
            double mean = getMean();
            double sq_sum = 0.0;
            for (auto duration : durations) {
                sq_sum += (duration - mean) * (duration - mean);
            }
            return std::sqrt(sq_sum / durations.size());
        }
        
        double getMin() const {
            return *std::min_element(durations.begin(), durations.end());
        }
        
        double getMax() const {
            return *std::max_element(durations.begin(), durations.end());
        }
    };
    
    std::map<std::string, RangeStatistics> statistics;

public:
    void recordRangeExecution(const std::string& name, const std::string& category,
                             double duration, const std::map<std::string, double>& metrics) {
        auto& stats = statistics[name];
        stats.name = name;
        stats.category = category;
        stats.durations.push_back(duration);
        
        for (const auto& [metricName, value] : metrics) {
            stats.metricValues[metricName].push_back(value);
        }
    }
    
    void generateStatisticalReport() {
        std::cout << "=== 统计范围分析 ===" << std::endl;
        
        for (const auto& [rangeName, stats] : statistics) {
            std::cout << "\n范围：" << rangeName << "（类别：" << stats.category << "）" << std::endl;
            std::cout << "  执行次数：" << stats.durations.size() << std::endl;
            std::cout << "  持续时间 - 平均：" << stats.getMean() << "ms，"
                     << "标准差：" << stats.getStdDev() << "ms" << std::endl;
            std::cout << "  持续时间 - 最小：" << stats.getMin() << "ms，"
                     << "最大：" << stats.getMax() << "ms" << std::endl;
            
            // 检测性能异常
            detectAnomalies(stats);
        }
    }
    
private:
    void detectAnomalies(const RangeStatistics& stats) {
        double mean = stats.getMean();
        double stddev = stats.getStdDev();
        double threshold = 2.0; // 2个标准差
        
        for (size_t i = 0; i < stats.durations.size(); i++) {
            if (std::abs(stats.durations[i] - mean) > threshold * stddev) {
                std::cout << "  检测到异常：执行" << i 
                         << "耗时" << stats.durations[i] << "ms" << std::endl;
            }
        }
    }
};
```

## 实际应用

### 算法阶段分析

```cpp
void profileSortingAlgorithm(std::vector<int>& data) {
    RangeProfiler profiler;
    
    {
        PROFILE_RANGE(profiler, "数据准备", "setup");
        // 准备数据结构
        prepareDataStructures(data);
    }
    
    {
        PROFILE_RANGE(profiler, "分区阶段", "algorithm");
        // 分区数据
        auto pivot = partition(data);
    }
    
    {
        PROFILE_RANGE(profiler, "递归排序左", "algorithm");
        // 排序左分区
        if (leftPartition.size() > 1) {
            quickSort(leftPartition);
        }
    }
    
    {
        PROFILE_RANGE(profiler, "递归排序右", "algorithm");
        // 排序右分区
        if (rightPartition.size() > 1) {
            quickSort(rightPartition);
        }
    }
    
    {
        PROFILE_RANGE(profiler, "结果合并", "finalization");
        // 合并结果
        combinePartitions(data, leftPartition, rightPartition);
    }
    
    profiler.generateReport();
}
```

### GPU内核范围性能分析

```cpp
class KernelRangeProfiler {
public:
    void profileMultiKernelWorkflow() {
        RangeProfiler profiler;
        
        // 为不同内核类型配置指标
        profiler.configureRangeMetrics("memory_intensive", 
                                      {"dram_read_transactions", "dram_write_transactions"},
                                      {"dram_utilization", "achieved_occupancy"});
                                      
        profiler.configureRangeMetrics("compute_intensive",
                                      {"inst_executed", "inst_fp_32"},
                                      {"flop_count_sp", "sm_efficiency"});
        
        {
            PROFILE_RANGE(profiler, "内存密集型内核", "memory_intensive");
            launchMemoryKernel<<<grid, block>>>(data);
            cudaDeviceSynchronize();
        }
        
        {
            PROFILE_RANGE(profiler, "计算密集型内核", "compute_intensive");
            launchComputeKernel<<<grid, block>>>(data);
            cudaDeviceSynchronize();
        }
        
        profiler.generateReport();
    }
};
```

## 下一步

- 在您自己的应用程序中实现自定义范围
- 开发特定于域的性能分析范围
- 与其他CUPTI功能结合进行全面分析
- 创建性能分析数据的可视化工具
- 实现自动性能回归检测 