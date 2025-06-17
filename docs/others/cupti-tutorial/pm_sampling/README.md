# Performance Monitor Sampling Tutorial

## Introduction

The CUPTI Performance Monitor (PM) Sampling sample demonstrates how to collect detailed GPU performance metrics using CUPTI's performance monitoring capabilities. This tutorial shows you how to gather, analyze, and interpret various performance counters and metrics that provide insights into GPU utilization, memory bandwidth, instruction throughput, and other critical performance indicators.

## What You'll Learn

- How to configure and collect GPU performance metrics
- Understanding different categories of performance counters
- Implementing metric-based performance analysis
- Correlating performance metrics with application behavior
- Using performance data for optimization guidance

## Understanding Performance Monitor Sampling

PM sampling provides comprehensive performance insights through:

1. **Hardware Performance Counters**: Low-level GPU metrics
2. **Derived Metrics**: Calculated performance indicators
3. **Real-time Monitoring**: Continuous performance tracking
4. **Multi-dimensional Analysis**: Multiple metrics simultaneously
5. **Performance Correlation**: Linking metrics to application phases

## Key Performance Metrics

### Compute Metrics
- **sm_efficiency**: Streaming Multiprocessor utilization
- **achieved_occupancy**: Percentage of maximum theoretical occupancy
- **inst_per_warp**: Instructions executed per warp
- **ipc**: Instructions per clock cycle
- **branch_efficiency**: Percentage of non-divergent branches

### Memory Metrics
- **dram_utilization**: Device memory utilization
- **tex_cache_hit_rate**: Texture cache hit rate
- **l2_cache_hit_rate**: L2 cache hit rate
- **global_hit_rate**: Global memory cache hit rate
- **shared_efficiency**: Shared memory bank efficiency

### Throughput Metrics
- **gld_throughput**: Global load throughput
- **gst_throughput**: Global store throughput
- **tex_cache_throughput**: Texture cache throughput
- **dram_read_throughput**: Device memory read throughput
- **dram_write_throughput**: Device memory write throughput

## Building the Sample

### Prerequisites

- CUDA Toolkit with CUPTI
- GPU with performance counter support
- Administrative privileges (for some performance counters)

### Build Process

```bash
cd pm_sampling
make
```

This creates the `pm_sampling` executable for performance monitoring.

## Running the Sample

### Basic Execution

```bash
./pm_sampling
```

### Sample Output

```
=== Performance Monitor Sampling Results ===

Kernel: vectorAdd
Performance Metrics Analysis:

Compute Efficiency:
  SM Efficiency: 87.5%
  Achieved Occupancy: 0.73
  Instructions per Warp: 128.4
  IPC (Instructions per Clock): 1.85
  Branch Efficiency: 94.2%

Memory Performance:
  DRAM Utilization: 45.8%
  L2 Cache Hit Rate: 78.9%
  Global Memory Hit Rate: 82.3%
  Texture Cache Hit Rate: N/A
  Shared Memory Efficiency: 89.4%

Throughput Metrics:
  Global Load Throughput: 156.7 GB/s
  Global Store Throughput: 142.3 GB/s
  DRAM Read Throughput: 89.5 GB/s
  DRAM Write Throughput: 76.2 GB/s

Performance Analysis:
  ✓ Good SM utilization (87.5% > 80%)
  ⚠ Memory bandwidth underutilized (45.8% < 60%)
  ✓ Excellent cache performance (78.9% L2 hit rate)
  ✓ Minimal branch divergence (94.2% efficiency)

Optimization Suggestions:
  - Increase memory access intensity to better utilize bandwidth
  - Consider memory access pattern optimization
  - Current compute/memory balance favors compute-bound workloads
```

## Code Architecture

### Performance Metrics Collector

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

void PerformanceMetricsCollector::initializeMetrics(CUcontext context, CUdevice device) {
    // Enumerate available metrics
    uint32_t numMetrics;
    CUPTI_CALL(cuptiDeviceGetNumMetrics(device, &numMetrics));
    
    CUpti_MetricID* metricIds = new CUpti_MetricID[numMetrics];
    CUPTI_CALL(cuptiDeviceEnumMetrics(device, metricIds, &numMetrics));
    
    for (uint32_t i = 0; i < numMetrics; i++) {
        MetricDefinition metric;
        metric.metricId = metricIds[i];
        
        // Get metric name
        size_t size;
        CUPTI_CALL(cuptiMetricGetAttribute(metricIds[i], CUPTI_METRIC_ATTR_NAME, &size, nullptr));
        metric.name.resize(size);
        CUPTI_CALL(cuptiMetricGetAttribute(metricIds[i], CUPTI_METRIC_ATTR_NAME, &size, &metric.name[0]));
        
        // Get metric category
        CUpti_MetricCategory category;
        size = sizeof(category);
        CUPTI_CALL(cuptiMetricGetAttribute(metricIds[i], CUPTI_METRIC_ATTR_CATEGORY, &size, &category));
        metric.category = getCategoryName(category);
        
        // Get metric description
        CUPTI_CALL(cuptiMetricGetAttribute(metricIds[i], CUPTI_METRIC_ATTR_SHORT_DESCRIPTION, &size, nullptr));
        metric.description.resize(size);
        CUPTI_CALL(cuptiMetricGetAttribute(metricIds[i], CUPTI_METRIC_ATTR_SHORT_DESCRIPTION, &size, &metric.description[0]));
        
        availableMetrics.push_back(metric);
    }
    
    delete[] metricIds;
    
    // Create event group for metric collection
    CUPTI_CALL(cuptiEventGroupCreate(context, &eventGroup, 0));
}
```

### Metric Analysis Engine

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
    void setThresholds(const std::string& metricName, double excellent, double good, double fair, double poor) {
        thresholds[metricName] = {excellent, good, fair, poor};
    }
    
    void analyzeMetric(const std::string& metricName, double value) {
        metricValues[metricName] = value;
        
        auto it = thresholds.find(metricName);
        if (it != thresholds.end()) {
            const auto& thresh = it->second;
            std::string assessment = assessPerformance(value, thresh);
            std::string suggestion = generateSuggestion(metricName, value, thresh);
            
            std::cout << metricName << ": " << value;
            if (!getMetricUnit(metricName).empty()) {
                std::cout << " " << getMetricUnit(metricName);
            }
            std::cout << " (" << assessment << ")" << std::endl;
            
            if (!suggestion.empty()) {
                std::cout << "  Suggestion: " << suggestion << std::endl;
            }
        }
    }
    
    void generateOverallAssessment() {
        // Analyze metric relationships and overall performance
        double computeScore = calculateComputeScore();
        double memoryScore = calculateMemoryScore();
        double efficiencyScore = calculateEfficiencyScore();
        
        std::cout << "\nOverall Performance Assessment:" << std::endl;
        std::cout << "  Compute Performance: " << getScoreRating(computeScore) << std::endl;
        std::cout << "  Memory Performance: " << getScoreRating(memoryScore) << std::endl;
        std::cout << "  Efficiency: " << getScoreRating(efficiencyScore) << std::endl;
        
        // Identify primary bottlenecks
        identifyBottlenecks();
    }

private:
    std::string assessPerformance(double value, const MetricThresholds& thresh) {
        if (value >= thresh.excellent) return "Excellent";
        if (value >= thresh.good) return "Good";
        if (value >= thresh.fair) return "Fair";
        return "Poor";
    }
    
    void identifyBottlenecks() {
        std::vector<std::pair<std::string, double>> bottlenecks;
        
        // Check for common bottleneck patterns
        if (getMetricValue("dram_utilization") < 60.0 && getMetricValue("sm_efficiency") > 80.0) {
            bottlenecks.push_back({"Memory Bandwidth", 0.8});
        }
        
        if (getMetricValue("achieved_occupancy") < 0.5) {
            bottlenecks.push_back({"Low Occupancy", 0.7});
        }
        
        if (getMetricValue("branch_efficiency") < 85.0) {
            bottlenecks.push_back({"Branch Divergence", 0.6});
        }
        
        if (!bottlenecks.empty()) {
            std::cout << "\nIdentified Bottlenecks:" << std::endl;
            for (const auto& [name, severity] : bottlenecks) {
                std::cout << "  " << name << " (severity: " << (severity * 100) << "%)" << std::endl;
            }
        }
    }
};
```

### Continuous Performance Monitoring

```cpp
class ContinuousPerformanceMonitor {
private:
    struct PerformanceSnapshot {
        uint64_t timestamp;
        std::map<std::string, double> metrics;
        std::string phase;
    };
    
    std::vector<PerformanceSnapshot> history;
    PerformanceMetricsCollector collector;
    std::thread monitorThread;
    std::atomic<bool> monitoring;
    uint32_t samplingIntervalMs;

public:
    ContinuousPerformanceMonitor() : monitoring(false), samplingIntervalMs(100) {}
    
    void startMonitoring() {
        monitoring = true;
        monitorThread = std::thread([this]() {
            performContinuousMonitoring();
        });
    }
    
    void stopMonitoring() {
        monitoring = false;
        if (monitorThread.joinable()) {
            monitorThread.join();
        }
    }
    
    void setPhase(const std::string& phaseName) {
        currentPhase = phaseName;
    }
    
    void analyzePerformanceTrends() {
        if (history.size() < 2) return;
        
        std::cout << "Performance Trend Analysis:" << std::endl;
        
        // Analyze trends for key metrics
        std::vector<std::string> keyMetrics = {
            "sm_efficiency", "dram_utilization", "achieved_occupancy"
        };
        
        for (const auto& metric : keyMetrics) {
            analyzeTrend(metric);
        }
        
        // Detect performance anomalies
        detectAnomalies();
    }

private:
    std::string currentPhase;
    
    void performContinuousMonitoring() {
        while (monitoring) {
            PerformanceSnapshot snapshot;
            snapshot.timestamp = getCurrentTimestamp();
            snapshot.phase = currentPhase;
            
            // Collect current metrics
            collector.startCollection();
            std::this_thread::sleep_for(std::chrono::milliseconds(10)); // Brief sampling
            collector.stopCollection();
            snapshot.metrics = collector.getMetrics();
            
            history.push_back(snapshot);
            
            // Limit history size
            if (history.size() > 1000) {
                history.erase(history.begin());
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(samplingIntervalMs));
        }
    }
    
    void analyzeTrend(const std::string& metricName) {
        if (history.size() < 10) return;
        
        std::vector<double> values;
        for (const auto& snapshot : history) {
            auto it = snapshot.metrics.find(metricName);
            if (it != snapshot.metrics.end()) {
                values.push_back(it->second);
            }
        }
        
        if (values.size() < 10) return;
        
        // Calculate trend (simple linear regression)
        double trend = calculateTrend(values);
        
        std::cout << "  " << metricName << " trend: ";
        if (trend > 0.01) {
            std::cout << "Improving (" << (trend * 100) << "%/sample)" << std::endl;
        } else if (trend < -0.01) {
            std::cout << "Degrading (" << (trend * 100) << "%/sample)" << std::endl;
        } else {
            std::cout << "Stable" << std::endl;
        }
    }
};
```

## Advanced Analysis Techniques

### Performance Metric Correlation

```cpp
class MetricCorrelationAnalyzer {
public:
    void analyzeCorrelations(const std::vector<PerformanceSnapshot>& history) {
        std::cout << "Metric Correlation Analysis:" << std::endl;
        
        // Common correlation patterns
        analyzeCorrelation(history, "sm_efficiency", "achieved_occupancy");
        analyzeCorrelation(history, "dram_utilization", "global_load_throughput");
        analyzeCorrelation(history, "l2_cache_hit_rate", "dram_read_throughput");
        analyzeCorrelation(history, "branch_efficiency", "ipc");
    }
    
private:
    void analyzeCorrelation(const std::vector<PerformanceSnapshot>& history,
                           const std::string& metric1, const std::string& metric2) {
        std::vector<double> values1, values2;
        
        for (const auto& snapshot : history) {
            auto it1 = snapshot.metrics.find(metric1);
            auto it2 = snapshot.metrics.find(metric2);
            
            if (it1 != snapshot.metrics.end() && it2 != snapshot.metrics.end()) {
                values1.push_back(it1->second);
                values2.push_back(it2->second);
            }
        }
        
        if (values1.size() > 10) {
            double correlation = calculateCorrelation(values1, values2);
            
            std::cout << "  " << metric1 << " vs " << metric2 << ": ";
            if (correlation > 0.7) {
                std::cout << "Strong positive correlation (" << correlation << ")" << std::endl;
            } else if (correlation < -0.7) {
                std::cout << "Strong negative correlation (" << correlation << ")" << std::endl;
            } else if (abs(correlation) > 0.3) {
                std::cout << "Moderate correlation (" << correlation << ")" << std::endl;
            } else {
                std::cout << "Weak correlation (" << correlation << ")" << std::endl;
            }
        }
    }
};
```

### Performance Regression Detection

```cpp
class PerformanceRegressionDetector {
private:
    struct BaselineMetrics {
        std::map<std::string, double> values;
        std::map<std::string, double> tolerances;
        std::string version;
        uint64_t timestamp;
    };
    
    BaselineMetrics baseline;
    
public:
    void setBaseline(const std::map<std::string, double>& metrics, const std::string& version) {
        baseline.values = metrics;
        baseline.version = version;
        baseline.timestamp = getCurrentTimestamp();
        
        // Set default tolerances (can be customized)
        for (const auto& [name, value] : metrics) {
            baseline.tolerances[name] = 0.05; // 5% tolerance
        }
    }
    
    void setTolerance(const std::string& metricName, double tolerance) {
        baseline.tolerances[metricName] = tolerance;
    }
    
    bool detectRegression(const std::map<std::string, double>& currentMetrics) {
        std::vector<std::string> regressions;
        
        for (const auto& [name, currentValue] : currentMetrics) {
            auto baselineIt = baseline.values.find(name);
            auto toleranceIt = baseline.tolerances.find(name);
            
            if (baselineIt != baseline.values.end() && toleranceIt != baseline.tolerances.end()) {
                double baselineValue = baselineIt->second;
                double tolerance = toleranceIt->second;
                
                double change = (currentValue - baselineValue) / baselineValue;
                
                // Check if this is a regression (performance decrease)
                if (isPerformanceMetric(name) && change < -tolerance) {
                    regressions.push_back(name + ": " + std::to_string(change * 100) + "% decrease");
                }
            }
        }
        
        if (!regressions.empty()) {
            std::cout << "Performance Regression Detected:" << std::endl;
            for (const auto& regression : regressions) {
                std::cout << "  " << regression << std::endl;
            }
            return true;
        }
        
        return false;
    }
};
```

## Real-World Applications

### GPU Kernel Optimization

```cpp
void optimizeKernelPerformance() {
    PerformanceMetricsCollector collector;
    MetricAnalysisEngine analyzer;
    
    // Configure analysis thresholds
    analyzer.setThresholds("sm_efficiency", 90.0, 80.0, 70.0, 60.0);
    analyzer.setThresholds("achieved_occupancy", 0.8, 0.6, 0.4, 0.2);
    analyzer.setThresholds("dram_utilization", 80.0, 60.0, 40.0, 20.0);
    
    // Test different kernel configurations
    std::vector<dim3> blockSizes = {{16, 16}, {32, 32}, {64, 16}, {128, 8}};
    
    for (const auto& blockSize : blockSizes) {
        std::cout << "Testing block size: " << blockSize.x << "x" << blockSize.y << std::endl;
        
        collector.startCollection();
        
        // Launch kernel with current configuration
        myKernel<<<gridSize, blockSize>>>(data);
        cudaDeviceSynchronize();
        
        collector.stopCollection();
        
        auto metrics = collector.getMetrics();
        
        std::cout << "  Performance metrics:" << std::endl;
        for (const auto& [name, value] : metrics) {
            analyzer.analyzeMetric(name, value);
        }
        
        analyzer.generateOverallAssessment();
        std::cout << std::endl;
    }
}
```

### Memory Access Pattern Analysis

```cpp
void analyzeMemoryPatterns() {
    PerformanceMetricsCollector collector;
    
    // Add memory-specific metrics
    collector.addMetric("global_load_efficiency");
    collector.addMetric("global_store_efficiency");
    collector.addMetric("shared_load_throughput");
    collector.addMetric("shared_store_throughput");
    collector.addMetric("l1_cache_global_hit_rate");
    collector.addMetric("l2_cache_hit_rate");
    
    // Test different memory access patterns
    std::vector<std::string> patterns = {"Sequential", "Strided", "Random", "Coalesced"};
    
    for (const auto& pattern : patterns) {
        std::cout << "Testing " << pattern << " memory access pattern:" << std::endl;
        
        collector.startCollection();
        
        // Execute kernel with specific memory pattern
        executeMemoryPattern(pattern);
        
        collector.stopCollection();
        
        auto metrics = collector.getMetrics();
        
        // Analyze memory efficiency
        analyzeMemoryEfficiency(metrics);
        std::cout << std::endl;
    }
}
```

## Integration with Development Workflow

### Automated Performance Testing

```cpp
class AutomatedPerformanceTesting {
public:
    void runPerformanceTestSuite() {
        PerformanceMetricsCollector collector;
        PerformanceRegressionDetector detector;
        
        // Load baseline from previous "golden" run
        auto baseline = loadBaseline("performance_baseline.json");
        detector.setBaseline(baseline, "v1.0");
        
        // Run current implementation
        collector.addMetricCategory("compute");
        collector.addMetricCategory("memory");
        
        collector.startCollection();
        runApplicationWorkload();
        collector.stopCollection();
        
        auto currentMetrics = collector.getMetrics();
        
        // Check for regressions
        bool hasRegression = detector.detectRegression(currentMetrics);
        
        // Save results
        saveResults(currentMetrics, "current_performance.json");
        
        // Exit with appropriate code for CI/CD
        if (hasRegression) {
            std::exit(1); // Fail CI build
        }
    }
};
```

## Next Steps

- Apply performance monitoring to understand your application's GPU utilization patterns
- Experiment with different metric combinations to identify bottlenecks
- Integrate continuous monitoring into your development and production workflows
- Develop custom analysis algorithms for your specific performance requirements
- Combine PM sampling with other CUPTI features for comprehensive performance analysis 