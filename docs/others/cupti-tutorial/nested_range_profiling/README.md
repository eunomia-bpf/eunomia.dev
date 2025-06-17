# CUPTI Nested Range Profiling Tutorial

> The GitHub repo and complete tutorial is available at <https://github.com/eunomia-bpf/cupti-tutorial>.

## Introduction

The CUPTI Nested Range Profiling sample demonstrates how to implement hierarchical performance analysis using nested profiling ranges. This technique allows you to create detailed performance profiles with multiple levels of granularity, enabling fine-grained analysis of complex algorithms and nested function calls.

## What You'll Learn

- How to create and manage nested profiling ranges
- Implementing hierarchical performance measurement
- Understanding range inheritance and scope management
- Analyzing nested algorithm performance patterns
- Building tree-structured performance reports

## Understanding Nested Range Profiling

Nested range profiling provides several key advantages:

1. **Hierarchical Analysis**: Break down complex algorithms into constituent parts
2. **Scope-based Measurement**: Automatic range management with RAII principles
3. **Contextual Performance Data**: Understand performance within algorithm phases
4. **Multi-level Granularity**: Profile at different levels of detail simultaneously
5. **Call Tree Analysis**: Visualize performance data as an execution tree

## Key Concepts

### Range Hierarchy

Nested ranges create a tree structure:
```
Application
├── Initialization
│   ├── Memory Allocation
│   └── Data Setup
├── Computation
│   ├── Phase 1
│   │   ├── Kernel A
│   │   └── Kernel B
│   └── Phase 2
│       ├── Kernel C
│       └── Memory Transfer
└── Cleanup
```

### Range Attributes

Each range can have:
- **Name**: Descriptive identifier
- **Category**: Grouping mechanism (compute, memory, IO)
- **Color**: Visualization hint
- **Payload**: Custom data attached to the range
- **Metrics**: Performance counters specific to the range

### Inheritance and Scope

Child ranges inherit properties from parents:
- Metric collection configuration
- Output formatting preferences  
- Error handling settings
- Custom attributes

## Building the Sample

### Prerequisites

- CUDA Toolkit with CUPTI
- C++14 compatible compiler (for RAII range management)
- NVTX library for enhanced visualization

### Build Process

```bash
cd nested_range_profiling
make
```

This creates the `nested_range_profiling` executable demonstrating hierarchical profiling techniques.

## Code Architecture

### Range Management Classes

```cpp
class ProfileRange {
private:
    std::string name;
    std::string category;
    uint64_t startTime;
    std::vector<std::unique_ptr<ProfileRange>> children;
    std::map<std::string, double> metrics;

public:
    ProfileRange(const std::string& name, const std::string& category = "default");
    ~ProfileRange(); // Automatically ends the range
    
    ProfileRange* createChild(const std::string& name, const std::string& category = "");
    void addMetric(const std::string& name, double value);
    void generateReport(int depth = 0) const;
};
```

### RAII Range Helper

```cpp
class ScopedRange {
private:
    ProfileRange* range;
    
public:
    ScopedRange(const std::string& name, const std::string& category = "default") {
        range = ProfileManager::getInstance().pushRange(name, category);
    }
    
    ~ScopedRange() {
        ProfileManager::getInstance().popRange();
    }
    
    ProfileRange* operator->() { return range; }
};

// Usage macro for convenience
#define PROFILE_RANGE(name, category) ScopedRange _prof_range(name, category)
```

## Running the Sample

### Basic Execution

```bash
./nested_range_profiling
```

### Sample Output

```
=== Nested Range Profiling Report ===

Application (45.2ms)
├── Initialization (5.1ms)
│   ├── Memory Allocation (2.3ms)
│   │   ├── Device Memory (1.8ms) [Memory: 512MB]
│   │   └── Host Memory (0.5ms) [Memory: 128MB]
│   └── Data Setup (2.8ms)
│       ├── Data Generation (1.2ms) [Items: 1M]
│       └── Data Transfer (1.6ms) [Bandwidth: 8.5GB/s]
├── Computation (35.4ms)
│   ├── Phase 1 (18.7ms)
│   │   ├── Preprocessing Kernel (3.2ms) [Utilization: 89%]
│   │   ├── Main Computation (12.1ms) [FLOPS: 2.1T]
│   │   └── Intermediate Transfer (3.4ms) [Size: 256MB]
│   └── Phase 2 (16.7ms)
│       ├── Reduction Kernel (8.9ms) [Efficiency: 95%]
│       ├── Postprocessing (4.2ms) [Cache Hits: 98%]
│       └── Result Transfer (3.6ms) [Bandwidth: 7.2GB/s]
└── Cleanup (4.7ms)
    ├── Memory Deallocation (2.1ms)
    └── Context Cleanup (2.6ms)

Performance Summary:
- Total Execution: 45.2ms
- Compute Time: 28.4ms (62.8%)
- Memory Time: 12.1ms (26.8%)
- Overhead: 4.7ms (10.4%)
```

## Advanced Features

### Metric Collection Integration

```cpp
class MetricCollector {
private:
    std::map<std::string, CUpti_EventGroup> eventGroups;
    std::vector<CUpti_EventID> activeEvents;

public:
    void beginCollection(const std::string& rangeName) {
        auto& group = eventGroups[rangeName];
        CUPTI_CALL(cuptiEventGroupEnable(group));
    }
    
    void endCollection(const std::string& rangeName, ProfileRange* range) {
        auto& group = eventGroups[rangeName];
        
        uint64_t eventValues[activeEvents.size()];
        size_t valueSize = sizeof(eventValues);
        
        CUPTI_CALL(cuptiEventGroupReadAllEvents(group,
                   CUPTI_EVENT_READ_FLAG_NONE,
                   &valueSize, eventValues,
                   nullptr, nullptr));
        
        // Add metrics to range
        for (size_t i = 0; i < activeEvents.size(); i++) {
            const char* eventName;
            CUPTI_CALL(cuptiEventGetAttribute(activeEvents[i],
                       CUPTI_EVENT_ATTR_NAME, &valueSize, &eventName));
            range->addMetric(eventName, static_cast<double>(eventValues[i]));
        }
        
        CUPTI_CALL(cuptiEventGroupDisable(group));
    }
};
```

### Conditional Profiling

```cpp
class ConditionalProfiler {
private:
    std::map<std::string, bool> enabledCategories;
    int maxDepth;
    
public:
    bool shouldProfile(const std::string& category, int currentDepth) {
        if (currentDepth > maxDepth) return false;
        
        auto it = enabledCategories.find(category);
        return (it != enabledCategories.end()) ? it->second : true;
    }
    
    void setCategory(const std::string& category, bool enabled) {
        enabledCategories[category] = enabled;
    }
    
    void setMaxDepth(int depth) { maxDepth = depth; }
};

// Usage example
void profiledFunction() {
    if (ConditionalProfiler::getInstance().shouldProfile("detailed", getCurrentDepth())) {
        PROFILE_RANGE("Detailed Analysis", "detailed");
        // ... detailed profiling code ...
    } else {
        PROFILE_RANGE("High Level", "summary");
        // ... summary profiling code ...
    }
}
```

### Custom Range Attributes

```cpp
class AttributedRange : public ProfileRange {
private:
    std::map<std::string, std::string> attributes;
    
public:
    template<typename T>
    void setAttribute(const std::string& key, const T& value) {
        std::ostringstream oss;
        oss << value;
        attributes[key] = oss.str();
    }
    
    std::string getAttribute(const std::string& key) const {
        auto it = attributes.find(key);
        return (it != attributes.end()) ? it->second : "";
    }
    
    void printAttributes() const {
        for (const auto& [key, value] : attributes) {
            std::cout << "[" << key << ": " << value << "] ";
        }
    }
};
```

## Real-World Applications

### Algorithm Phase Analysis

```cpp
void analyzeMatrixMultiplication(const Matrix& A, const Matrix& B, Matrix& C) {
    PROFILE_RANGE("Matrix Multiplication", "compute");
    
    {
        PROFILE_RANGE("Memory Preparation", "memory");
        
        {
            PROFILE_RANGE("Input Transfer", "transfer");
            transferToDevice(A, B);
        }
        
        {
            PROFILE_RANGE("Output Allocation", "allocation");  
            allocateDeviceMatrix(C);
        }
    }
    
    {
        PROFILE_RANGE("Computation", "kernel");
        
        {
            PROFILE_RANGE("Tiling Setup", "setup");
            configureTiling(A.rows, B.cols, A.cols);
        }
        
        {
            PROFILE_RANGE("Kernel Execution", "execution");
            launchMatMulKernel(A, B, C);
        }
    }
    
    {
        PROFILE_RANGE("Result Retrieval", "memory");
        transferFromDevice(C);
    }
}
```

### Neural Network Training

```cpp
class NetworkProfiler {
public:
    void profileTrainingEpoch(Network& network, const Dataset& data) {
        PROFILE_RANGE("Training Epoch", "training");
        
        for (const auto& batch : data.getBatches()) {
            PROFILE_RANGE("Batch Processing", "batch");
            
            {
                PROFILE_RANGE("Forward Pass", "forward");
                
                for (size_t i = 0; i < network.getLayers().size(); i++) {
                    std::string layerName = "Layer " + std::to_string(i);
                    PROFILE_RANGE(layerName, "layer");
                    
                    auto& layer = network.getLayer(i);
                    {
                        PROFILE_RANGE("Computation", "compute");
                        layer.forward(batch);
                    }
                    
                    {
                        PROFILE_RANGE("Activation", "activation");
                        layer.applyActivation();
                    }
                }
            }
            
            {
                PROFILE_RANGE("Backward Pass", "backward");
                
                for (int i = network.getLayers().size() - 1; i >= 0; i--) {
                    std::string layerName = "Layer " + std::to_string(i) + " Backward";
                    PROFILE_RANGE(layerName, "layer");
                    
                    auto& layer = network.getLayer(i);
                    layer.backward();
                }
            }
            
            {
                PROFILE_RANGE("Parameter Update", "optimization");
                network.updateParameters();
            }
        }
    }
};
```

### Multi-Stream Application

```cpp
class StreamedComputation {
private:
    std::vector<cudaStream_t> streams;
    
public:
    void processInStreams(const std::vector<DataChunk>& chunks) {
        PROFILE_RANGE("Streamed Processing", "streaming");
        
        std::vector<std::thread> threads;
        
        for (size_t i = 0; i < chunks.size(); i++) {
            threads.emplace_back([this, i, &chunks]() {
                std::string streamName = "Stream " + std::to_string(i);
                PROFILE_RANGE(streamName, "stream");
                
                {
                    PROFILE_RANGE("Data Transfer In", "transfer");
                    transferToDevice(chunks[i], streams[i]);
                }
                
                {
                    PROFILE_RANGE("Kernel Execution", "compute");
                    processChunk(chunks[i], streams[i]);
                }
                
                {
                    PROFILE_RANGE("Data Transfer Out", "transfer");
                    transferFromDevice(chunks[i], streams[i]);
                }
            });
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
    }
};
```

## Visualization and Analysis

### Tree Visualization

```cpp
class RangeTreeVisualizer {
public:
    void exportToGraphviz(const ProfileRange& root, const std::string& filename) {
        std::ofstream file(filename);
        file << "digraph ProfileTree {\n";
        file << "  rankdir=TB;\n";
        file << "  node [shape=box];\n";
        
        exportNode(root, file, 0);
        
        file << "}\n";
    }
    
private:
    void exportNode(const ProfileRange& range, std::ofstream& file, int& nodeId) {
        int currentId = nodeId++;
        
        file << "  node" << currentId << " [label=\"" 
             << range.getName() << "\\n" 
             << range.getDuration() << "ms\"];\n";
        
        for (const auto& child : range.getChildren()) {
            int childId = nodeId;
            exportNode(*child, file, nodeId);
            file << "  node" << currentId << " -> node" << childId << ";\n";
        }
    }
};
```

### Performance Hotspot Detection

```cpp
class HotspotAnalyzer {
public:
    struct Hotspot {
        std::string path;
        double exclusiveTime;
        double inclusiveTime;
        double percentage;
    };
    
    std::vector<Hotspot> findHotspots(const ProfileRange& root, double threshold = 0.05) {
        std::vector<Hotspot> hotspots;
        double totalTime = root.getDuration();
        
        findHotspotsRecursive(root, "", totalTime, threshold, hotspots);
        
        // Sort by exclusive time
        std::sort(hotspots.begin(), hotspots.end(),
                 [](const Hotspot& a, const Hotspot& b) {
                     return a.exclusiveTime > b.exclusiveTime;
                 });
        
        return hotspots;
    }
    
private:
    void findHotspotsRecursive(const ProfileRange& range, const std::string& path,
                              double totalTime, double threshold,
                              std::vector<Hotspot>& hotspots) {
        std::string currentPath = path.empty() ? range.getName() : path + "/" + range.getName();
        
        double inclusiveTime = range.getDuration();
        double exclusiveTime = inclusiveTime;
        
        // Subtract children time for exclusive calculation
        for (const auto& child : range.getChildren()) {
            exclusiveTime -= child->getDuration();
            findHotspotsRecursive(*child, currentPath, totalTime, threshold, hotspots);
        }
        
        double percentage = exclusiveTime / totalTime;
        if (percentage >= threshold) {
            hotspots.push_back({currentPath, exclusiveTime, inclusiveTime, percentage});
        }
    }
};
```

## Integration with Development Tools

### IDE Integration

```cpp
// Generate IDE-friendly format
class IDEReporter {
public:
    void generateVSCodeReport(const ProfileRange& root, const std::string& filename) {
        json report;
        report["version"] = "1.0";
        report["type"] = "profile";
        report["ranges"] = json::array();
        
        exportRangeToJSON(root, report["ranges"]);
        
        std::ofstream file(filename);
        file << report.dump(2);
    }
    
private:
    void exportRangeToJSON(const ProfileRange& range, json& parent) {
        json rangeObj;
        rangeObj["name"] = range.getName();
        rangeObj["duration"] = range.getDuration();
        rangeObj["category"] = range.getCategory();
        rangeObj["children"] = json::array();
        
        for (const auto& child : range.getChildren()) {
            exportRangeToJSON(*child, rangeObj["children"]);
        }
        
        parent.push_back(rangeObj);
    }
};
```

### Continuous Integration

```cpp
class CIReporter {
public:
    bool checkPerformanceRegression(const ProfileRange& current, 
                                   const ProfileRange& baseline,
                                   double threshold = 0.1) {
        std::map<std::string, double> currentTimes = extractTimings(current);
        std::map<std::string, double> baselineTimes = extractTimings(baseline);
        
        for (const auto& [path, currentTime] : currentTimes) {
            auto it = baselineTimes.find(path);
            if (it != baselineTimes.end()) {
                double regression = (currentTime - it->second) / it->second;
                if (regression > threshold) {
                    std::cerr << "Performance regression detected in " << path 
                             << ": " << (regression * 100) << "% slower" << std::endl;
                    return false;
                }
            }
        }
        
        return true;
    }
};
```

## Next Steps

- Apply nested range profiling to understand your algorithm hierarchies
- Experiment with different granularity levels for your specific use cases
- Integrate with visualization tools for better performance understanding
- Develop custom metrics and analysis for your application domains
- Combine with other CUPTI features for comprehensive performance analysis 