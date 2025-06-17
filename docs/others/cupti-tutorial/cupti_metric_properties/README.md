# CUPTI Metric Properties Tutorial

> The GitHub repo and complete tutorial is available at <https://github.com/eunomia-bpf/cupti-tutorial>.

## Introduction

Understanding the properties of available GPU metrics is crucial for effective performance analysis. This sample demonstrates how to query metric properties using CUPTI's profiling APIs, including metric types, collection methods, hardware units, and pass requirements.

## What You'll Learn

- How to query available GPU metrics and their properties
- Understanding metric types (counter, ratio, throughput)
- Determining collection methods (hardware vs software)
- Finding hardware units associated with metrics
- Calculating pass requirements for metric collection
- Working with metric submetrics and rollup operations

## Key Concepts

### Metric Types
- **Counter**: Raw hardware counter values
- **Ratio**: Calculated ratios between counters  
- **Throughput**: Rate-based metrics (operations per unit time)

### Collection Methods
- **Hardware**: Direct hardware counter collection
- **Software**: Requires kernel instrumentation
- **Mixed**: Combination of hardware and software collection

### Hardware Units
Different GPU components that provide metrics:
- **SM**: Streaming Multiprocessor
- **L1TEX**: L1 Texture Cache
- **L2**: L2 Cache
- **DRAM**: Device Memory
- **SYS**: System-level metrics

## Sample Architecture

### Metric Evaluator
```cpp
class MetricEvaluator {
private:
    NVPW_MetricsEvaluator* m_pNVPWMetricEvaluator;
    std::vector<uint8_t> m_scratchBuffer;

public:
    MetricEvaluator(const char* pChipName, uint8_t* pCounterAvailabilityImage) {
        // Initialize NVPW metric evaluator
        NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params params = {};
        params.pChipName = pChipName;
        params.pCounterAvailabilityImage = pCounterAvailabilityImage;
        
        NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize(&params);
        m_scratchBuffer.resize(params.scratchBufferSize);
        
        // Initialize the evaluator
        NVPW_CUDA_MetricsEvaluator_Initialize_Params initParams = {};
        initParams.pChipName = pChipName;
        initParams.pScratchBuffer = m_scratchBuffer.data();
        initParams.scratchBufferSize = m_scratchBuffer.size();
        
        NVPW_CUDA_MetricsEvaluator_Initialize(&initParams);
        m_pNVPWMetricEvaluator = initParams.pMetricsEvaluator;
    }
};
```

### Metric Details Structure
```cpp
struct MetricDetails {
    const char* name;           // Metric name
    const char* description;    // Human-readable description
    const char* type;          // Counter/Ratio/Throughput
    const char* hwUnit;        // Hardware unit (SM, L2, etc.)
    std::string collectionType; // Hardware/Software collection
    size_t numOfPasses;        // Passes required for collection
    std::vector<std::string> submetrics; // Available submetrics
};
```

## Sample Walkthrough

### Listing All Available Metrics
```cpp
bool MetricEvaluator::ListAllMetrics(std::vector<MetricDetails>& metrics) {
    for (auto i = 0; i < NVPW_METRIC_TYPE__COUNT; ++i) {
        NVPW_MetricType metricType = static_cast<NVPW_MetricType>(i);
        
        // Get metric names for this type
        NVPW_MetricsEvaluator_GetMetricNames_Params params = {};
        params.metricType = metricType;
        params.pMetricsEvaluator = m_pNVPWMetricEvaluator;
        
        NVPW_MetricsEvaluator_GetMetricNames(&params);
        
        // Process each metric
        for (size_t metricIndex = 0; metricIndex < params.numMetrics; ++metricIndex) {
            size_t nameIndex = params.pMetricNameBeginIndices[metricIndex];
            const char* metricName = &params.pMetricNames[nameIndex];
            
            MetricDetails metric = {};
            metric.name = metricName;
            
            // Get detailed properties
            GetMetricProperties(metric, metricType, metricIndex);
            metric.collectionType = GetMetricCollectionMethod(metricName);
            
            metrics.push_back(metric);
        }
    }
    return true;
}
```

### Querying Metric Properties
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
    
    // Convert hardware unit to string
    NVPW_MetricsEvaluator_HwUnitToString_Params hwParams = {};
    hwParams.pMetricsEvaluator = m_pNVPWMetricEvaluator;
    hwParams.hwUnit = hwUnit;
    
    NVPW_MetricsEvaluator_HwUnitToString(&hwParams);
    metric.hwUnit = hwParams.pHwUnitName;
    metric.type = GetMetricTypeString(metricType);
    
    return true;
}
```

### Collection Method Analysis
```cpp
std::string MetricEvaluator::GetMetricCollectionMethod(std::string metricName) {
    std::vector<NVPA_RawMetricRequest> rawMetricRequests;
    
    if (GetRawMetricRequests(metricName, rawMetricRequests)) {
        bool hasHardware = false;
        bool hasSoftware = false;
        
        for (const auto& request : rawMetricRequests) {
            if (request.isolated) {
                hasSoftware = true;  // Isolated metrics require instrumentation
            } else {
                hasHardware = true;  // Non-isolated can use hardware counters
            }
        }
        
        if (hasHardware && hasSoftware) {
            return "Mixed (HW + SW)";
        } else if (hasSoftware) {
            return "Software";
        } else {
            return "Hardware";
        }
    }
    
    return "Unknown";
}
```

### Pass Requirement Calculation
```cpp
class MetricConfig {
public:
    bool GetNumOfPasses(const std::vector<const char*>& metrics, 
                       MetricEvaluator* pMetricEvaluator, 
                       size_t& numOfPasses) {
        
        // Create configuration for the metric set
        NVPW_CUDA_MetricsConfig_Create_Params createParams = {};
        createParams.pChipName = mChipName.c_str();
        
        NVPW_CUDA_MetricsConfig_Create(&createParams);
        
        // Add each metric to the configuration
        for (const char* metricName : metrics) {
            NVPW_MetricsConfig_AddMetrics_Params addParams = {};
            addParams.pMetricsConfig = createParams.pMetricsConfig;
            addParams.pMetricNames = &metricName;
            addParams.numMetricNames = 1;
            
            NVPW_MetricsConfig_AddMetrics(&addParams);
        }
        
        // Generate configuration and get pass count
        NVPW_MetricsConfig_GenerateConfigImage_Params genParams = {};
        genParams.pMetricsConfig = createParams.pMetricsConfig;
        
        NVPW_MetricsConfig_GenerateConfigImage(&genParams);
        
        // Get number of passes required
        NVPW_CUDA_MetricsConfig_GetNumPasses_Params passParams = {};
        passParams.pConfig = genParams.pConfigImage;
        passParams.configImageSize = genParams.configImageSize;
        
        NVPW_CUDA_MetricsConfig_GetNumPasses(&passParams);
        numOfPasses = passParams.numPasses;
        
        return true;
    }
};
```

## Building and Running

```bash
cd cupti_metric_properties
make
./cupti_metric_properties [options]
```

### Command Line Options
- `--list-metrics`: List all available metrics
- `--metric <name>`: Query specific metric properties
- `--list-submetrics`: Include submetrics in output
- `--device <id>`: Target specific GPU device

## Sample Output

```
=== GPU Metric Properties ===

Metric: smsp__cycles_active
Type: Counter
Hardware Unit: SM
Description: Number of cycles the streaming multiprocessor was active
Collection: Hardware
Passes Required: 1
Submetrics: .avg, .max, .min, .sum

Metric: sm__throughput.avg.pct_of_peak_sustained_elapsed
Type: Throughput  
Hardware Unit: SM
Description: Average SM throughput as percentage of peak sustained
Collection: Hardware
Passes Required: 1
Submetrics: .per_second, .pct_of_peak_sustained_active

Metric: gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed
Type: Throughput
Hardware Unit: SYS
Description: GPU compute memory throughput percentage
Collection: Mixed (HW + SW)
Passes Required: 2
Submetrics: .per_second, .pct_of_peak_sustained_active, .peak_sustained
```

## Advanced Analysis

### Metric Categorization
```cpp
class MetricCategorizer {
public:
    void CategorizeMetrics(const std::vector<MetricDetails>& metrics) {
        std::map<std::string, std::vector<MetricDetails>> categories;
        
        for (const auto& metric : metrics) {
            std::string category = GetMetricCategory(metric.name);
            categories[category].push_back(metric);
        }
        
        PrintCategorizedMetrics(categories);
    }
    
private:
    std::string GetMetricCategory(const char* metricName) {
        std::string name(metricName);
        
        if (name.find("smsp__") == 0) return "Streaming Multiprocessor";
        if (name.find("l1tex__") == 0) return "L1 Texture Cache";
        if (name.find("l2__") == 0) return "L2 Cache";
        if (name.find("dram__") == 0) return "Device Memory";
        if (name.find("pcie__") == 0) return "PCIe";
        if (name.find("nvlink__") == 0) return "NVLink";
        if (name.find("gpu__") == 0) return "GPU-wide";
        
        return "Other";
    }
};
```

### Performance Impact Analysis
```cpp
class MetricImpactAnalyzer {
public:
    void AnalyzeCollectionImpact(const std::vector<MetricDetails>& metrics) {
        std::map<std::string, size_t> collectionTypeCounts;
        std::map<size_t, size_t> passCounts;
        
        for (const auto& metric : metrics) {
            collectionTypeCounts[metric.collectionType]++;
            passCounts[metric.numOfPasses]++;
        }
        
        printf("\n=== Collection Impact Analysis ===\n");
        printf("Collection Type Distribution:\n");
        for (const auto& [type, count] : collectionTypeCounts) {
            printf("  %s: %zu metrics\n", type.c_str(), count);
        }
        
        printf("\nPass Requirements:\n");
        for (const auto& [passes, count] : passCounts) {
            printf("  %zu pass(es): %zu metrics\n", passes, count);
        }
        
        AnalyzePerformanceImpact(collectionTypeCounts, passCounts);
    }
    
private:
    void AnalyzePerformanceImpact(const std::map<std::string, size_t>& types,
                                 const std::map<size_t, size_t>& passes) {
        printf("\nPerformance Impact Assessment:\n");
        
        // Analyze software metrics impact
        auto swIter = types.find("Software");
        if (swIter != types.end()) {
            printf("  Software metrics (%zu): High overhead due to instrumentation\n", 
                   swIter->second);
        }
        
        // Analyze multi-pass requirements
        size_t multiPassMetrics = 0;
        for (const auto& [passCount, metricCount] : passes) {
            if (passCount > 1) {
                multiPassMetrics += metricCount;
            }
        }
        
        if (multiPassMetrics > 0) {
            printf("  Multi-pass metrics (%zu): Requires application replay\n", 
                   multiPassMetrics);
        }
    }
};
```

## Real-World Applications

### Profiling Tool Integration
```cpp
class ProfilerMetricSelector {
public:
    std::vector<std::string> SelectOptimalMetrics(const std::string& analysisType) {
        std::vector<std::string> selectedMetrics;
        
        if (analysisType == "memory_analysis") {
            selectedMetrics = {
                "dram__throughput.avg.pct_of_peak_sustained_elapsed",
                "l1tex__t_sector_hit_rate.pct",
                "l2__t_sector_hit_rate.pct",
                "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"
            };
        } else if (analysisType == "compute_analysis") {
            selectedMetrics = {
                "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                "smsp__inst_executed.avg.per_cycle_active",
                "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
                "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum"
            };
        } else if (analysisType == "occupancy_analysis") {
            selectedMetrics = {
                "sm__warps_active.avg.pct_of_peak_sustained_active",
                "smsp__warps_eligible.avg.per_cycle_active",
                "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"
            };
        }
        
        return ValidateMetricCompatibility(selectedMetrics);
    }
    
private:
    std::vector<std::string> ValidateMetricCompatibility(
        const std::vector<std::string>& metrics) {
        
        // Check if metrics can be collected together
        std::vector<std::string> validatedMetrics;
        
        MetricConfig config(chipName.c_str(), counterAvailabilityImage.data());
        size_t numPasses;
        
        if (config.GetNumOfPasses(ConvertToCharArray(metrics), 
                                 &metricEvaluator, numPasses)) {
            if (numPasses <= maxAllowedPasses) {
                validatedMetrics = metrics;
            } else {
                // Split metrics into compatible groups
                validatedMetrics = SplitIntoCompatibleGroups(metrics);
            }
        }
        
        return validatedMetrics;
    }
};
```

### Dynamic Metric Selection
```cpp
class DynamicMetricSelector {
public:
    std::vector<std::string> SelectMetricsForKernel(const KernelCharacteristics& kernel) {
        std::vector<std::string> metrics;
        
        // Memory-bound kernels
        if (kernel.memoryIntensive) {
            metrics.insert(metrics.end(), {
                "dram__throughput.avg.pct_of_peak_sustained_elapsed",
                "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
                "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum"
            });
        }
        
        // Compute-bound kernels
        if (kernel.computeIntensive) {
            metrics.insert(metrics.end(), {
                "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
                "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum",
                "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum"
            });
        }
        
        // Control flow heavy kernels
        if (kernel.hasControlFlow) {
            metrics.insert(metrics.end(), {
                "smsp__thread_inst_executed_pred_on.avg.per_cycle_active",
                "smsp__warps_active.avg.pct_of_peak_sustained_active"
            });
        }
        
        return RemoveDuplicates(metrics);
    }
};
```

## Best Practices

### Efficient Metric Querying
```cpp
class EfficientMetricQuerier {
public:
    void QueryMetricsEfficiently() {
        // Cache metric evaluator for reuse
        static std::unique_ptr<MetricEvaluator> cachedEvaluator;
        
        if (!cachedEvaluator) {
            cachedEvaluator = std::make_unique<MetricEvaluator>(
                chipName.c_str(), counterAvailabilityImage.data());
        }
        
        // Batch metric queries
        std::vector<MetricDetails> allMetrics;
        cachedEvaluator->ListAllMetrics(allMetrics);
        
        // Pre-compute commonly used metric sets
        PrecomputeCommonMetricSets(allMetrics);
    }
    
private:
    void PrecomputeCommonMetricSets(const std::vector<MetricDetails>& allMetrics) {
        // Group by hardware unit for efficient selection
        std::map<std::string, std::vector<std::string>> hwUnitMetrics;
        
        for (const auto& metric : allMetrics) {
            hwUnitMetrics[metric.hwUnit].push_back(metric.name);
        }
        
        // Cache compatible metric combinations
        for (const auto& [hwUnit, metrics] : hwUnitMetrics) {
            CacheCompatibleCombinations(hwUnit, metrics);
        }
    }
};
```

## Use Cases

- **Profiler Development**: Build custom profiling tools with optimal metric selection
- **Performance Analysis**: Understand which metrics are available for specific hardware
- **Optimization Tools**: Select metrics based on application characteristics
- **Research**: Analyze GPU architecture capabilities through available metrics
- **Automated Profiling**: Dynamically select metrics based on workload patterns

## Next Steps

- Implement intelligent metric recommendation based on application analysis
- Build visualization tools for metric properties and relationships
- Develop metric selection algorithms for optimal profiling efficiency
- Create metric compatibility matrices for complex applications
- Integrate with automated performance analysis workflows

Understanding metric properties is essential for building effective GPU profiling and optimization tools. This sample provides the foundation for intelligent metric selection and analysis. 