# CUPTI Range Profiling Tutorial

## Introduction

The CUPTI Range Profiling sample demonstrates how to implement targeted performance analysis using custom-defined ranges within your CUDA applications. This technique allows you to profile specific code sections, algorithm phases, or functional components with precise control over what gets measured and analyzed.

## What You'll Learn

- How to define and manage custom profiling ranges
- Implementing selective performance measurement
- Understanding range-based metric collection
- Creating hierarchical performance analysis
- Building targeted optimization strategies

## Understanding Range Profiling

Range profiling provides focused performance analysis by allowing you to:

1. **Define Specific Regions**: Mark exact code sections for analysis
2. **Control Measurement Scope**: Profile only what matters to you
3. **Reduce Overhead**: Minimize profiling impact by targeting specific areas
4. **Create Performance Baselines**: Establish metrics for specific functions
5. **Enable Comparative Analysis**: Compare different implementations of the same functionality

## Key Concepts

### Range Definition

Ranges are defined by:
- **Start Point**: Beginning of the measurement region
- **End Point**: Conclusion of the measurement region  
- **Range Name**: Descriptive identifier for the region
- **Range Category**: Classification for grouping related ranges
- **Associated Metrics**: Performance counters to collect within the range

### Range Types

#### Function-Level Ranges
Profile entire functions or major algorithm components

#### Loop-Level Ranges  
Measure specific iteration patterns or computational loops

#### Phase-Level Ranges
Track distinct phases of complex algorithms

#### Conditional Ranges
Profile code paths based on runtime conditions

## Building the Sample

### Prerequisites

- CUDA Toolkit with CUPTI
- C++ compiler with C++11 support
- NVTX library for enhanced range visualization

### Build Process

```bash
cd range_profiling
make
```

This creates the `range_profiling` executable demonstrating targeted profiling techniques.

## Code Architecture

### Range Management System

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

### RAII Range Helper

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
    
    // Move semantics
    ScopedProfileRange(ScopedProfileRange&& other) noexcept 
        : profiler(other.profiler), isValid(other.isValid) {
        other.isValid = false;
    }
};

// Convenience macro
#define PROFILE_RANGE(profiler, name, category) \
    ScopedProfileRange _prof_range(profiler, name, category)
```

## Running the Sample

### Basic Execution

```bash
./range_profiling
```

### Sample Output

```
=== Range Profiling Results ===

Range: "Matrix Initialization" (Category: setup)
  Duration: 2.3ms
  Instructions Executed: 1,245,678
  Memory Bandwidth: 12.5 GB/s
  Cache Hit Rate: 94.2%

Range: "Matrix Multiplication Core" (Category: compute)
  Duration: 45.7ms
  Instructions Executed: 89,456,123
  FLOPS: 2.1 TFLOPS
  Memory Bandwidth: 385.2 GB/s
  Compute Utilization: 87.3%

Range: "Result Validation" (Category: verification)
  Duration: 8.1ms
  Instructions Executed: 3,876,234
  Memory Bandwidth: 45.6 GB/s
  Branch Efficiency: 96.8%

Performance Summary by Category:
  setup: 2.3ms (4.1%)
  compute: 45.7ms (81.2%)  
  verification: 8.1ms (14.4%)
  other: 0.2ms (0.3%)

Total Execution Time: 56.3ms
Profiling Overhead: 0.8ms (1.4%)
```

## Advanced Range Features

### Conditional Range Profiling

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
        // Check category filter
        auto catIt = enabledCategories.find(category);
        if (catIt != enabledCategories.end() && !catIt->second) {
            return false;
        }
        
        // Check count limit
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

// Usage example
void conditionallyProfiledFunction() {
    ConditionalRangeProfiler& condProf = ConditionalRangeProfiler::getInstance();
    
    if (condProf.shouldProfile("detailed_analysis", "debug")) {
        PROFILE_RANGE(profiler, "Detailed Analysis", "debug");
        // Detailed profiling code
        performDetailedAnalysis();
        condProf.recordRangeExecution("detailed_analysis");
    } else {
        // Lightweight or no profiling
        performStandardAnalysis();
    }
}
```

### Custom Metric Integration

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
        
        // Set up events
        for (const auto& eventName : eventNames) {
            CUpti_EventID eventId;
            CUPTI_CALL(cuptiEventGetIdFromName(device, eventName.c_str(), &eventId));
            metrics.events.push_back(eventId);
        }
        
        // Set up metrics
        for (const auto& metricName : metricNames) {
            CUpti_MetricID metricId;
            CUPTI_CALL(cuptiMetricGetIdFromName(device, metricName.c_str(), &metricId));
            metrics.metrics.push_back(metricId);
        }
        
        // Create event group
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
            // Read event values
            uint64_t eventValues[it->second.events.size()];
            size_t valueSize = sizeof(eventValues);
            
            CUPTI_CALL(cuptiEventGroupReadAllEvents(it->second.eventGroup,
                       CUPTI_EVENT_READ_FLAG_NONE,
                       &valueSize, eventValues,
                       nullptr, nullptr));
            
            // Calculate metrics
            for (size_t i = 0; i < it->second.metrics.size(); i++) {
                CUpti_MetricValue metricValue;
                CUPTI_CALL(cuptiMetricGetValue(device, it->second.metrics[i],
                           it->second.events.size(), it->second.events.data(),
                           eventValues, 0, &metricValue));
                
                // Store metric value
                recordMetricValue(rangeName, it->second.metrics[i], metricValue);
            }
            
            CUPTI_CALL(cuptiEventGroupDisable(it->second.eventGroup));
        }
        
        endRange();
    }
};
```

### Statistical Range Analysis

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
        std::cout << "=== Statistical Range Analysis ===" << std::endl;
        
        for (const auto& [rangeName, stats] : statistics) {
            std::cout << "\nRange: " << rangeName << " (Category: " << stats.category << ")" << std::endl;
            std::cout << "  Executions: " << stats.durations.size() << std::endl;
            std::cout << "  Duration - Mean: " << stats.getMean() << "ms, "
                     << "StdDev: " << stats.getStdDev() << "ms" << std::endl;
            std::cout << "  Duration - Min: " << stats.getMin() << "ms, "
                     << "Max: " << stats.getMax() << "ms" << std::endl;
            
            // Detect performance anomalies
            detectAnomalies(stats);
        }
    }
    
private:
    void detectAnomalies(const RangeStatistics& stats) {
        double mean = stats.getMean();
        double stddev = stats.getStdDev();
        double threshold = 2.0; // 2 standard deviations
        
        for (size_t i = 0; i < stats.durations.size(); i++) {
            if (std::abs(stats.durations[i] - mean) > threshold * stddev) {
                std::cout << "  Anomaly detected: Execution " << i 
                         << " took " << stats.durations[i] << "ms" << std::endl;
            }
        }
    }
};
```

## Real-World Applications

### Algorithm Phase Analysis

```cpp
void profileSortingAlgorithm(std::vector<int>& data) {
    RangeProfiler profiler;
    
    {
        PROFILE_RANGE(profiler, "Data Preparation", "setup");
        // Prepare data structures
        prepareDataStructures(data);
    }
    
    {
        PROFILE_RANGE(profiler, "Partitioning Phase", "algorithm");
        // Partition the data
        auto pivot = partition(data);
    }
    
    {
        PROFILE_RANGE(profiler, "Recursive Sort Left", "algorithm");
        // Sort left partition
        if (leftPartition.size() > 1) {
            quickSort(leftPartition);
        }
    }
    
    {
        PROFILE_RANGE(profiler, "Recursive Sort Right", "algorithm");
        // Sort right partition  
        if (rightPartition.size() > 1) {
            quickSort(rightPartition);
        }
    }
    
    {
        PROFILE_RANGE(profiler, "Result Combination", "finalization");
        // Combine results
        combinePartitions(data, leftPartition, rightPartition);
    }
    
    profiler.generateReport();
}
```

### GPU Kernel Range Profiling

```cpp
class KernelRangeProfiler {
public:
    void profileMultiKernelWorkflow() {
        RangeProfiler profiler;
        
        // Configure metrics for different kernel types
        profiler.configureRangeMetrics("memory_intensive", 
                                      {"dram_read_transactions", "dram_write_transactions"},
                                      {"dram_utilization", "achieved_occupancy"});
        
        profiler.configureRangeMetrics("compute_intensive",
                                      {"inst_executed", "inst_fp_32"},
                                      {"flop_count_sp", "sm_efficiency"});
        
        {
            PROFILE_RANGE(profiler, "Data Transfer In", "memory");
            cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
        }
        
        {
            profiler.startRangeWithMetrics("memory_intensive");
            // Launch memory-bound kernel
            memoryIntensiveKernel<<<grid, block>>>(d_input, d_temp);
            cudaDeviceSynchronize();
            profiler.endRangeWithMetrics("memory_intensive");
        }
        
        {
            profiler.startRangeWithMetrics("compute_intensive");
            // Launch compute-bound kernel
            computeIntensiveKernel<<<grid, block>>>(d_temp, d_output);
            cudaDeviceSynchronize();
            profiler.endRangeWithMetrics("compute_intensive");
        }
        
        {
            PROFILE_RANGE(profiler, "Data Transfer Out", "memory");
            cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
        }
        
        profiler.generateReport();
    }
};
```

### Iterative Algorithm Profiling

```cpp
class IterativeProfiler {
private:
    StatisticalRangeAnalyzer analyzer;
    int iterationCount;

public:
    void profileIterativeAlgorithm() {
        const int maxIterations = 1000;
        double tolerance = 1e-6;
        
        for (int iteration = 0; iteration < maxIterations; iteration++) {
            auto start = std::chrono::high_resolution_clock::now();
            
            // Profile individual iteration
            {
                PROFILE_RANGE(profiler, "Convergence Check", "validation");
                if (checkConvergence(tolerance)) {
                    std::cout << "Converged after " << iteration << " iterations" << std::endl;
                    break;
                }
            }
            
            {
                PROFILE_RANGE(profiler, "Update Step", "computation");
                performUpdateStep();
            }
            
            {
                PROFILE_RANGE(profiler, "Error Calculation", "analysis");
                double error = calculateError();
                recordIterationMetrics(iteration, error);
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            double duration = std::chrono::duration<double, std::milli>(end - start).count();
            
            // Record iteration statistics
            std::map<std::string, double> metrics = {{"error", getCurrentError()}};
            analyzer.recordRangeExecution("Full Iteration", "iteration", duration, metrics);
            
            // Detect performance degradation
            if (iteration > 10 && isPerformanceDegrading()) {
                std::cout << "Performance degradation detected at iteration " << iteration << std::endl;
            }
        }
        
        analyzer.generateStatisticalReport();
    }
};
```

## Optimization Insights from Range Profiling

### Performance Bottleneck Identification

```cpp
class BottleneckAnalyzer {
public:
    void analyzeRangePerformance(const std::vector<ProfileRange>& ranges) {
        // Find the slowest ranges
        std::vector<std::pair<std::string, double>> rangesByDuration;
        
        for (const auto& range : ranges) {
            rangesByDuration.push_back({range.name, range.getDuration()});
        }
        
        std::sort(rangesByDuration.begin(), rangesByDuration.end(),
                 [](const auto& a, const auto& b) { return a.second > b.second; });
        
        std::cout << "Performance Bottlenecks (Top 5):" << std::endl;
        for (int i = 0; i < std::min(5, (int)rangesByDuration.size()); i++) {
            const auto& [name, duration] = rangesByDuration[i];
            double percentage = duration / getTotalDuration(ranges) * 100;
            std::cout << "  " << i+1 << ". " << name << ": " << duration << "ms (" 
                     << percentage << "% of total)" << std::endl;
        }
    }
    
    void suggestOptimizations(const std::vector<ProfileRange>& ranges) {
        for (const auto& range : ranges) {
            if (range.getCategory() == "memory" && range.getBandwidth() < getExpectedBandwidth()) {
                std::cout << "Optimization suggestion for " << range.name 
                         << ": Consider memory coalescing or async transfers" << std::endl;
            }
            
            if (range.getCategory() == "compute" && range.getUtilization() < 0.8) {
                std::cout << "Optimization suggestion for " << range.name 
                         << ": Consider increasing occupancy or optimizing instruction mix" << std::endl;
            }
        }
    }
};
```

### Comparative Range Analysis

```cpp
class ComparativeAnalyzer {
public:
    void compareImplementations(const std::string& baselineFile, const std::string& optimizedFile) {
        auto baselineRanges = loadRangeData(baselineFile);
        auto optimizedRanges = loadRangeData(optimizedFile);
        
        std::cout << "=== Implementation Comparison ===" << std::endl;
        
        for (const auto& [rangeName, baselineRange] : baselineRanges) {
            auto it = optimizedRanges.find(rangeName);
            if (it != optimizedRanges.end()) {
                const auto& optimizedRange = it->second;
                
                double speedup = baselineRange.getDuration() / optimizedRange.getDuration();
                double improvement = (1.0 - (optimizedRange.getDuration() / baselineRange.getDuration())) * 100;
                
                std::cout << "Range: " << rangeName << std::endl;
                std::cout << "  Baseline: " << baselineRange.getDuration() << "ms" << std::endl;
                std::cout << "  Optimized: " << optimizedRange.getDuration() << "ms" << std::endl;
                std::cout << "  Speedup: " << speedup << "x (" << improvement << "% improvement)" << std::endl;
                
                if (speedup < 1.0) {
                    std::cout << "  WARNING: Performance regression detected!" << std::endl;
                }
            }
        }
    }
};
```

## Integration with Development Workflow

### Automated Range Profiling

```cpp
// Automated profiling for CI/CD
class AutomatedRangeProfiler {
public:
    bool performRegressionTest(const std::string& referenceFile) {
        RangeProfiler profiler;
        bool passed = true;
        
        // Run profiled application
        executeProfiledApplication(profiler);
        
        // Load reference performance data
        auto referenceData = loadReferenceData(referenceFile);
        
        // Compare against thresholds
        for (const auto& range : profiler.getCompletedRanges()) {
            auto it = referenceData.find(range.name);
            if (it != referenceData.end()) {
                double regressionThreshold = 1.1; // 10% slowdown threshold
                if (range.getDuration() > it->second * regressionThreshold) {
                    std::cout << "FAIL: Performance regression in " << range.name << std::endl;
                    passed = false;
                }
            }
        }
        
        return passed;
    }
};
```

## Next Steps

- Apply range profiling to identify specific bottlenecks in your applications
- Experiment with different granularities of range definition
- Integrate range profiling into your optimization workflow
- Develop custom metrics collection for your specific use cases
- Combine range profiling with other CUPTI features for comprehensive analysis 