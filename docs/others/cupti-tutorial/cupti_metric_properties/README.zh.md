# CUPTI 指标属性教程

> GitHub 仓库和完整教程可在 <https://github.com/eunomia-bpf/cupti-tutorial> 获取。

## 简介

理解可用 GPU 指标的属性对于有效的性能分析至关重要。本示例演示了如何使用 CUPTI 的性能分析 API 查询指标属性，包括指标类型、收集方法、硬件单元和遍历要求。

## 您将学到的内容

- 如何查询可用的 GPU 指标及其属性
- 理解指标类型（计数器、比率、吞吐量）
- 确定收集方法（硬件与软件）
- 查找与指标关联的硬件单元
- 计算指标收集的遍历要求
- 使用指标子指标和汇总操作

## 关键概念

### 指标类型

- **计数器**：原始硬件计数器值
- **比率**：计数器之间的计算比率
- **吞吐量**：基于速率的指标（每单位时间的操作数）

### 收集方法

- **硬件**：直接硬件计数器收集
- **软件**：需要内核插桩
- **混合**：硬件和软件收集的组合

### 硬件单元

提供指标的不同 GPU 组件：
- **SM**：流多处理器
- **L1TEX**：L1 纹理缓存
- **L2**：L2 缓存
- **DRAM**：设备内存
- **SYS**：系统级指标

## 示例架构

### 指标评估器

```cpp
class MetricEvaluator {
private:
    NVPW_MetricsEvaluator* m_pNVPWMetricEvaluator;
    std::vector<uint8_t> m_scratchBuffer;

public:
    MetricEvaluator(const char* pChipName, uint8_t* pCounterAvailabilityImage) {
        // 初始化 NVPW 指标评估器
        NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params params = {};
        params.pChipName = pChipName;
        params.pCounterAvailabilityImage = pCounterAvailabilityImage;
        
        NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize(&params);
        m_scratchBuffer.resize(params.scratchBufferSize);
        
        // 初始化评估器
        NVPW_CUDA_MetricsEvaluator_Initialize_Params initParams = {};
        initParams.pChipName = pChipName;
        initParams.pScratchBuffer = m_scratchBuffer.data();
        initParams.scratchBufferSize = m_scratchBuffer.size();
        
        NVPW_CUDA_MetricsEvaluator_Initialize(&initParams);
        m_pNVPWMetricEvaluator = initParams.pMetricsEvaluator;
    }
};
```

### 指标详情结构

```cpp
struct MetricDetails {
    const char* name;           // 指标名称
    const char* description;    // 人类可读的描述
    const char* type;          // 计数器/比率/吞吐量
    const char* hwUnit;        // 硬件单元（SM、L2等）
    std::string collectionType; // 硬件/软件收集
    size_t numOfPasses;        // 收集所需的遍历数
    std::vector<std::string> submetrics; // 可用的子指标
};
```

## 示例演练

### 列出所有可用指标

```cpp
bool MetricEvaluator::ListAllMetrics(std::vector<MetricDetails>& metrics) {
    for (auto i = 0; i < NVPW_METRIC_TYPE__COUNT; ++i) {
        NVPW_MetricType metricType = static_cast<NVPW_MetricType>(i);
        
        // 获取此类型的指标名称
        NVPW_MetricsEvaluator_GetMetricNames_Params params = {};
        params.metricType = metricType;
        params.pMetricsEvaluator = m_pNVPWMetricEvaluator;
        
        NVPW_MetricsEvaluator_GetMetricNames(&params);
        
        // 处理每个指标
        for (size_t metricIndex = 0; metricIndex < params.numMetrics; ++metricIndex) {
            size_t nameIndex = params.pMetricNameBeginIndices[metricIndex];
            const char* metricName = &params.pMetricNames[nameIndex];
            
            MetricDetails metric = {};
            metric.name = metricName;
            
            // 获取详细属性
            GetMetricProperties(metric, metricType, metricIndex);
            metric.collectionType = GetMetricCollectionMethod(metricName);
            
            metrics.push_back(metric);
        }
    }
    return true;
}
```

### 查询指标属性

```cpp
bool MetricEvaluator::GetMetricProperties(MetricDetails& metric, 
                                         NVPW_MetricType metricType, 
                                         size_t metricIndex) {
    NVPW_HwUnit hwUnit = NVPW_HW_UNIT_INVALID;
    
    switch (metricType) {
        case NVPW_METRIC_TYPE_COUNTER:
        {
            NVPW_MetricsEvaluator_GetCounterProperties_Params params = {};
            params.pMetricsEvaluator = m_pNVPWMetricEvaluator;
            params.counterIndex = metricIndex;
            
            NVPW_MetricsEvaluator_GetCounterProperties(&params);
            metric.description = params.pDescription;
            hwUnit = (NVPW_HwUnit)params.hwUnit;
            break;
        }
        case NVPW_METRIC_TYPE_RATIO:
        {
            NVPW_MetricsEvaluator_GetRatioMetricProperties_Params params = {};
            params.pMetricsEvaluator = m_pNVPWMetricEvaluator;
            params.ratioMetricIndex = metricIndex;
            
            NVPW_MetricsEvaluator_GetRatioMetricProperties(&params);
            metric.description = params.pDescription;
            hwUnit = (NVPW_HwUnit)params.hwUnit;
            break;
        }
        case NVPW_METRIC_TYPE_THROUGHPUT:
        {
            NVPW_MetricsEvaluator_GetThroughputMetricProperties_Params params = {};
            params.pMetricsEvaluator = m_pNVPWMetricEvaluator;
            params.throughputMetricIndex = metricIndex;
            
            NVPW_MetricsEvaluator_GetThroughputMetricProperties(&params);
            metric.description = params.pDescription;
            hwUnit = (NVPW_HwUnit)params.hwUnit;
            break;
        }
    }
    
    // 将硬件单元转换为字符串
    NVPW_MetricsEvaluator_HwUnitToString_Params hwParams = {};
    hwParams.pMetricsEvaluator = m_pNVPWMetricEvaluator;
    hwParams.hwUnit = hwUnit;
    
    NVPW_MetricsEvaluator_HwUnitToString(&hwParams);
    metric.hwUnit = hwParams.pHwUnitName;
    metric.type = GetMetricTypeString(metricType);
    
    return true;
}
```

### 收集方法分析

```cpp
std::string MetricEvaluator::GetMetricCollectionMethod(std::string metricName) {
    std::vector<NVPA_RawMetricRequest> rawMetricRequests;
    
    if (GetRawMetricRequests(metricName, rawMetricRequests)) {
        bool hasHardware = false;
        bool hasSoftware = false;
        
        for (const auto& request : rawMetricRequests) {
            if (request.isolated) {
                hasSoftware = true;  // 隔离指标需要插桩
            } else {
                hasHardware = true;  // 非隔离可以使用硬件计数器
            }
        }
        
        if (hasHardware && hasSoftware) {
            return "混合 (HW + SW)";
        } else if (hasSoftware) {
            return "软件";
        } else {
            return "硬件";
        }
    }
    
    return "未知";
}
```

### 遍历要求计算

```cpp
class PassRequirementCalculator {
public:
    size_t CalculatePassRequirements(const std::vector<std::string>& metricNames) {
        std::vector<NVPA_RawMetricRequest> allRequests;
        
        // 收集所有指标的原始请求
        for (const auto& metricName : metricNames) {
            std::vector<NVPA_RawMetricRequest> requests;
            GetRawMetricRequests(metricName, requests);
            allRequests.insert(allRequests.end(), requests.begin(), requests.end());
        }
        
        // 分析硬件限制
        return AnalyzeHardwareConstraints(allRequests);
    }
    
private:
    size_t AnalyzeHardwareConstraints(const std::vector<NVPA_RawMetricRequest>& requests) {
        std::map<NVPW_HwUnit, std::set<std::string>> unitRequests;
        
        // 按硬件单元分组请求
        for (const auto& request : requests) {
            unitRequests[request.hwUnit].insert(request.counterName);
        }
        
        size_t maxPasses = 1;
        
        // 计算每个硬件单元所需的遍历数
        for (const auto& [hwUnit, counters] : unitRequests) {
            size_t unitCapacity = GetHardwareUnitCapacity(hwUnit);
            size_t requiredPasses = (counters.size() + unitCapacity - 1) / unitCapacity;
            maxPasses = std::max(maxPasses, requiredPasses);
        }
        
        return maxPasses;
    }
    
    size_t GetHardwareUnitCapacity(NVPW_HwUnit hwUnit) {
        // 基于硬件单元返回并发计数器容量
        switch (hwUnit) {
            case NVPW_HW_UNIT_SM: return 4;
            case NVPW_HW_UNIT_L1TEX: return 4;
            case NVPW_HW_UNIT_L2: return 4;
            case NVPW_HW_UNIT_DRAM: return 2;
            default: return 1;
        }
    }
};
```

## 运行示例

### 构建和执行

```bash
cd cupti_metric_properties
make
./cupti_metric_properties
```

### 示例输出

```
=== GPU 指标属性分析 ===

计数器指标:
  smsp__cycles_elapsed.avg - SM 平均已过周期数 [硬件, SM 单元]
  smsp__inst_executed.sum - SM 执行的指令总数 [硬件, SM 单元]
  l1tex__t_bytes.sum - L1TEX 总字节数 [硬件, L1TEX 单元]

比率指标:
  sm__throughput.avg.pct_of_peak_sustained_elapsed - SM 吞吐量百分比 [混合, SM 单元]
  achieved_occupancy - 实现的占用率 [软件, SM 单元]
  ipc - 每周期指令数 [混合, SM 单元]

吞吐量指标:
  dram__bytes.sum.per_second - DRAM 字节/秒 [硬件, DRAM 单元]
  smsp__inst_executed.sum.per_cycle_elapsed - 每周期执行的指令 [硬件, SM 单元]

遍历要求分析:
  单一指标收集: 1 遍历
  多指标收集 (5个): 2 遍历
  完整指标集 (50个): 8 遍历

硬件单元利用率:
  SM 单元: 75% 利用率
  L1TEX 单元: 50% 利用率
  L2 单元: 25% 利用率
  DRAM 单元: 100% 利用率
```

## 高级分析

### 指标兼容性分析

```cpp
class MetricCompatibilityAnalyzer {
public:
    bool AnalyzeCompatibility(const std::vector<std::string>& metricNames) {
        printf("\n=== 指标兼容性分析 ===\n");
        
        // 检查冲突的收集方法
        if (HasConflictingCollectionMethods(metricNames)) {
            printf("警告: 检测到冲突的收集方法\n");
            return false;
        }
        
        // 分析硬件资源冲突
        if (HasHardwareResourceConflicts(metricNames)) {
            printf("警告: 硬件资源冲突，需要多次遍历\n");
        }
        
        // 检查依赖性
        AnalyzeDependencies(metricNames);
        
        return true;
    }
    
private:
    bool HasConflictingCollectionMethods(const std::vector<std::string>& metricNames) {
        bool hasHardware = false;
        bool hasSoftware = false;
        
        for (const auto& metricName : metricNames) {
            std::string method = GetMetricCollectionMethod(metricName);
            if (method.find("硬件") != std::string::npos) {
                hasHardware = true;
            }
            if (method.find("软件") != std::string::npos) {
                hasSoftware = true;
            }
        }
        
        // 硬件和软件方法通常可以混合
        return false; // 简化实现
    }
    
    bool HasHardwareResourceConflicts(const std::vector<std::string>& metricNames) {
        std::map<NVPW_HwUnit, int> unitUsage;
        
        for (const auto& metricName : metricNames) {
            NVPW_HwUnit hwUnit = GetMetricHardwareUnit(metricName);
            unitUsage[hwUnit]++;
        }
        
        // 检查是否任何硬件单元超过容量
        for (const auto& [hwUnit, usage] : unitUsage) {
            if (usage > GetHardwareUnitCapacity(hwUnit)) {
                return true;
            }
        }
        
        return false;
    }
    
    void AnalyzeDependencies(const std::vector<std::string>& metricNames) {
        printf("指标依赖性分析:\n");
        
        for (const auto& metricName : metricNames) {
            auto dependencies = GetMetricDependencies(metricName);
            if (!dependencies.empty()) {
                printf("  %s 依赖于: ", metricName.c_str());
                for (const auto& dep : dependencies) {
                    printf("%s ", dep.c_str());
                }
                printf("\n");
            }
        }
    }
};
```

### 子指标分析

```cpp
class SubmetricAnalyzer {
public:
    void AnalyzeSubmetrics(const std::string& metricName) {
        printf("\n=== %s 子指标分析 ===\n", metricName.c_str());
        
        auto submetrics = GetAvailableSubmetrics(metricName);
        
        for (const auto& submetric : submetrics) {
            printf("子指标: %s\n", submetric.c_str());
            
            // 分析子指标属性
            SubmetricProperties props = GetSubmetricProperties(submetric);
            printf("  类型: %s\n", props.type.c_str());
            printf("  单元: %s\n", props.unit.c_str());
            printf("  描述: %s\n", props.description.c_str());
            
            // 分析汇总操作
            auto rollupOps = GetAvailableRollupOperations(submetric);
            printf("  可用汇总: ");
            for (const auto& op : rollupOps) {
                printf("%s ", op.c_str());
            }
            printf("\n");
        }
    }
    
private:
    std::vector<std::string> GetAvailableSubmetrics(const std::string& metricName) {
        std::vector<std::string> submetrics;
        
        // 查询可用的子指标
        NVPW_MetricsEvaluator_GetSupportedSubmetrics_Params params = {};
        params.pMetricsEvaluator = m_pNVPWMetricEvaluator;
        params.pMetricName = metricName.c_str();
        
        NVPW_MetricsEvaluator_GetSupportedSubmetrics(&params);
        
        for (size_t i = 0; i < params.numSubmetrics; i++) {
            size_t nameIndex = params.pSubmetricNameBeginIndices[i];
            submetrics.push_back(&params.pSubmetricNames[nameIndex]);
        }
        
        return submetrics;
    }
    
    std::vector<std::string> GetAvailableRollupOperations(const std::string& submetric) {
        // 常见的汇总操作
        return {"sum", "avg", "min", "max", "pct"};
    }
};
```

### 性能特征分析

```cpp
class PerformanceCharacterizer {
public:
    void CharacterizeMetrics(const std::vector<std::string>& metricNames) {
        printf("\n=== 指标性能特征 ===\n");
        
        for (const auto& metricName : metricNames) {
            MetricCharacteristics chars = AnalyzeMetricCharacteristics(metricName);
            
            printf("指标: %s\n", metricName.c_str());
            printf("  收集开销: %s\n", chars.overhead.c_str());
            printf("  准确性: %s\n", chars.accuracy.c_str());
            printf("  适用场景: %s\n", chars.useCases.c_str());
            printf("  限制: %s\n", chars.limitations.c_str());
        }
    }
    
private:
    struct MetricCharacteristics {
        std::string overhead;
        std::string accuracy;
        std::string useCases;
        std::string limitations;
    };
    
    MetricCharacteristics AnalyzeMetricCharacteristics(const std::string& metricName) {
        MetricCharacteristics chars;
        
        std::string collectionMethod = GetMetricCollectionMethod(metricName);
        
        if (collectionMethod.find("硬件") != std::string::npos) {
            chars.overhead = "低";
            chars.accuracy = "高";
            chars.useCases = "生产环境、连续监控";
            chars.limitations = "硬件计数器限制";
        } else if (collectionMethod.find("软件") != std::string::npos) {
            chars.overhead = "中到高";
            chars.accuracy = "高";
            chars.useCases = "开发调试、详细分析";
            chars.limitations = "需要内核插桩";
        } else {
            chars.overhead = "中";
            chars.accuracy = "高";
            chars.useCases = "混合工作负载分析";
            chars.limitations = "可能需要多次遍历";
        }
        
        return chars;
    }
};
```

## 故障排除

### 常见问题

1. **指标不可用**：
   ```cpp
   void ValidateMetricAvailability(const std::string& metricName) {
       auto availableMetrics = GetAvailableMetrics();
       if (std::find(availableMetrics.begin(), availableMetrics.end(), 
                     metricName) == availableMetrics.end()) {
           printf("错误: 指标 '%s' 在此设备上不可用\n", metricName.c_str());
           printf("可用指标:\n");
           for (const auto& metric : availableMetrics) {
               printf("  %s\n", metric.c_str());
           }
       }
   }
   ```

2. **过多遍历需求**：
   ```cpp
   void OptimizePassRequirements(std::vector<std::string>& metricNames) {
       PassRequirementCalculator calc;
       size_t passes = calc.CalculatePassRequirements(metricNames);
       
       if (passes > MAX_ACCEPTABLE_PASSES) {
           printf("警告: 需要 %zu 次遍历，考虑减少指标数量\n", passes);
           
           // 建议优化策略
           SuggestMetricOptimization(metricNames);
       }
   }
   ```

## 总结

CUPTI 指标属性提供了对 GPU 性能测量能力的深入了解。通过理解指标属性：

### 关键优势

- **明智的指标选择**：选择适合您用例的指标
- **优化的收集策略**：最小化遍历次数和开销
- **硬件感知分析**：理解硬件限制和能力
- **准确的性能解释**：正确理解指标含义

### 最佳实践

1. **评估指标属性**：在使用前了解指标特征
2. **平衡覆盖面和开销**：选择提供最佳洞察/成本比的指标
3. **考虑硬件约束**：规划多遍历收集策略
4. **验证兼容性**：确保指标组合可以有效收集

指标属性分析是构建高效、准确性能分析工具的基础。 