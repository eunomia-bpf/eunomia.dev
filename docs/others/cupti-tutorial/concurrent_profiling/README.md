# CUPTI Concurrent Profiling Tutorial

## Introduction

The CUPTI Concurrent Profiling sample demonstrates advanced techniques for profiling complex CUDA applications that use multiple streams, devices, and threads. This tutorial shows how to handle the challenges of profiling concurrent GPU operations while maintaining accuracy and minimizing overhead.

## What You'll Learn

- How to profile applications with multiple CUDA streams
- Techniques for multi-device profiling and analysis
- Understanding concurrency patterns in GPU applications
- Managing profiling overhead in high-throughput scenarios
- Correlating activities across different execution contexts

## Understanding Concurrent Profiling Challenges

Profiling concurrent CUDA applications presents unique challenges:

1. **Overlapping Operations**: Multiple kernels and memory transfers executing simultaneously
2. **Multi-device Coordination**: Synchronizing profiling across multiple GPUs
3. **Thread Safety**: Handling profiling data from multiple CPU threads
4. **Context Management**: Tracking activities across different CUDA contexts
5. **Timeline Correlation**: Maintaining accurate timing relationships

## Key Concepts

### Concurrency Patterns in CUDA

#### Stream-based Concurrency
- Multiple operations on different streams
- Overlapping kernel execution and memory transfers
- Asynchronous API calls

#### Multi-device Concurrency
- Parallel execution across multiple GPUs
- Peer-to-peer memory transfers
- Cross-device synchronization

#### Thread-based Concurrency
- Multiple CPU threads making CUDA calls
- Shared contexts and resources
- Thread-local profiling data

## Building the Sample

### Prerequisites

- CUDA Toolkit with CUPTI
- Multi-GPU system (recommended for full functionality)
- C++11 compatible compiler

### Build Process

```bash
cd concurrent_profiling
make
```

This creates the `concurrent_profiling` executable that demonstrates various concurrency scenarios.

## Sample Architecture

### Test Scenarios

The sample includes several concurrency patterns:

1. **Single Stream Sequential**: Baseline for comparison
2. **Multiple Stream Parallel**: Concurrent kernel execution
3. **Multi-device Execution**: Cross-GPU workload distribution
4. **Mixed Workloads**: Combination of compute and memory operations

### Profiling Components

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

## Running the Sample

### Basic Execution

```bash
./concurrent_profiling
```

### Sample Output

```
=== Concurrent Profiling Analysis ===

Device 0 Analysis:
  Total Streams: 4
  Concurrent Kernels: 8
  Stream Utilization: 85.3%
  
Device 1 Analysis:
  Total Streams: 4
  Concurrent Kernels: 6
  Stream Utilization: 78.1%

Concurrency Metrics:
  Kernel Overlap Ratio: 0.73
  Memory Transfer Overlap: 0.89
  Cross-device Bandwidth: 28.5 GB/s
  
Timeline Analysis:
  Total Execution Time: 45.2ms
  Sequential Equivalent: 124.7ms
  Speedup Factor: 2.76x
```

## Advanced Profiling Techniques

### Stream Timeline Analysis

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
        // Analyze timeline overlaps
        uint64_t totalTime = 0;
        uint64_t overlappedTime = 0;
        
        // Complex overlap calculation algorithm
        return static_cast<double>(overlappedTime) / totalTime;
    }
};
```

### Multi-Device Coordination

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

### Thread-Safe Data Collection

```cpp
class ThreadSafeCollector {
private:
    std::mutex dataMutex;
    std::condition_variable dataReady;
    std::queue<ProfilingEvent> eventQueue;
    
public:
    void recordEvent(const ProfilingEvent& event) {
        std::lock_guard<std::mutex> lock(dataMutex);
        eventQueue.push(event);
        dataReady.notify_one();
    }
    
    void processEvents() {
        std::unique_lock<std::mutex> lock(dataMutex);
        while (profiling) {
            dataReady.wait(lock, [this] { return !eventQueue.empty() || !profiling; });
            
            while (!eventQueue.empty()) {
                ProfilingEvent event = eventQueue.front();
                eventQueue.pop();
                lock.unlock();
                
                // Process event without holding lock
                analyzeEvent(event);
                
                lock.lock();
            }
        }
    }
};
```

## Concurrency Analysis Features

### Overlap Detection

```cpp
struct OverlapAnalysis {
    double kernelOverlap;
    double memoryOverlap;
    double computeMemoryOverlap;
    
    void calculateOverlaps(const TimelineData& timeline) {
        // Analyze different types of overlaps
        auto kernelEvents = timeline.getKernelEvents();
        auto memoryEvents = timeline.getMemoryEvents();
        
        kernelOverlap = calculateKernelOverlap(kernelEvents);
        memoryOverlap = calculateMemoryOverlap(memoryEvents);
        computeMemoryOverlap = calculateComputeMemoryOverlap(kernelEvents, memoryEvents);
    }
};
```

### Resource Utilization

```cpp
class ResourceMonitor {
private:
    std::map<int, GPUUtilization> deviceUtilization;
    std::map<cudaStream_t, StreamUtilization> streamUtilization;

public:
    void updateUtilization() {
        for (auto& [deviceId, util] : deviceUtilization) {
            util.computeUtilization = measureComputeUtilization(deviceId);
            util.memoryBandwidthUtilization = measureMemoryBandwidth(deviceId);
            util.cacheHitRate = measureCachePerformance(deviceId);
        }
    }
    
    void generateUtilizationReport() {
        for (const auto& [deviceId, util] : deviceUtilization) {
            std::cout << "Device " << deviceId << ":" << std::endl;
            std::cout << "  Compute: " << util.computeUtilization * 100 << "%" << std::endl;
            std::cout << "  Memory: " << util.memoryBandwidthUtilization * 100 << "%" << std::endl;
            std::cout << "  Cache Hit Rate: " << util.cacheHitRate * 100 << "%" << std::endl;
        }
    }
};
```

## Performance Optimization Insights

### Identifying Bottlenecks

1. **Stream Underutilization**: Low concurrent kernel execution
2. **Memory Bandwidth Limits**: Saturated memory subsystem
3. **Synchronization Overhead**: Excessive cross-stream dependencies
4. **Load Imbalance**: Uneven work distribution across devices

### Optimization Strategies

```cpp
class OptimizationAdvisor {
public:
    std::vector<std::string> analyzeAndSuggest(const ProfilingData& data) {
        std::vector<std::string> suggestions;
        
        if (data.streamUtilization < 0.7) {
            suggestions.push_back("Increase stream concurrency");
        }
        
        if (data.memoryBandwidthUtilization > 0.9) {
            suggestions.push_back("Consider data compression or caching");
        }
        
        if (data.synchronizationOverhead > 0.1) {
            suggestions.push_back("Reduce synchronization points");
        }
        
        if (data.deviceLoadImbalance > 0.2) {
            suggestions.push_back("Improve load balancing across devices");
        }
        
        return suggestions;
    }
};
```

## Real-World Applications

### High-Throughput Computing

```cpp
// Profile streaming applications
class StreamingProfiler {
private:
    struct BatchMetrics {
        uint64_t processedItems;
        double throughput;
        double latency;
    };
    
public:
    void profileBatch(size_t batchSize) {
        auto startTime = getCurrentTime();
        
        // Process batch with concurrent streams
        processBatchConcurrently(batchSize);
        
        auto endTime = getCurrentTime();
        auto duration = endTime - startTime;
        
        BatchMetrics metrics;
        metrics.processedItems = batchSize;
        metrics.throughput = batchSize / (duration / 1e6); // items per second
        metrics.latency = duration / batchSize; // microseconds per item
        
        recordBatchMetrics(metrics);
    }
};
```

### Multi-GPU Machine Learning

```cpp
// Profile distributed training scenarios
class DistributedTrainingProfiler {
private:
    std::vector<int> gpuIds;
    std::map<int, TrainingMetrics> gpuMetrics;

public:
    void profileTrainingStep() {
        auto stepStart = getCurrentTime();
        
        // Parallel forward pass
        std::vector<std::thread> forwardThreads;
        for (int gpu : gpuIds) {
            forwardThreads.emplace_back([this, gpu]() {
                profileForwardPass(gpu);
            });
        }
        
        for (auto& thread : forwardThreads) {
            thread.join();
        }
        
        // All-reduce synchronization
        profileAllReduce();
        
        // Parallel backward pass
        std::vector<std::thread> backwardThreads;
        for (int gpu : gpuIds) {
            backwardThreads.emplace_back([this, gpu]() {
                profileBackwardPass(gpu);
            });
        }
        
        for (auto& thread : backwardThreads) {
            thread.join();
        }
        
        auto stepEnd = getCurrentTime();
        recordTrainingStep(stepEnd - stepStart);
    }
};
```

## Integration and Visualization

### Timeline Generation

```cpp
// Generate timeline data for visualization tools
class TimelineExporter {
public:
    void exportToNsightSystems(const ProfilingData& data, const std::string& filename) {
        // Export in format compatible with Nsight Systems
        NsightExporter exporter;
        exporter.addStreamData(data.streamActivities);
        exporter.addKernelData(data.kernelActivities);
        exporter.addMemoryData(data.memoryActivities);
        exporter.save(filename);
    }
    
    void exportToChromeTracing(const ProfilingData& data, const std::string& filename) {
        // Export in Chrome tracing format
        json timeline;
        timeline["traceEvents"] = json::array();
        
        for (const auto& event : data.allEvents) {
            timeline["traceEvents"].push_back(convertToTraceEvent(event));
        }
        
        std::ofstream file(filename);
        file << timeline.dump(2);
    }
};
```

## Troubleshooting Concurrent Profiling

### Common Issues

1. **Data Race Conditions**: Multiple threads accessing profiling data
2. **Context Switching Overhead**: Frequent device context changes
3. **Memory Pressure**: High memory usage from profiling buffers
4. **Timeline Synchronization**: Misaligned timestamps across devices

### Debug Strategies

```cpp
class ConcurrencyDebugger {
public:
    void validateTimestamps(const std::vector<ProfilingEvent>& events) {
        for (size_t i = 1; i < events.size(); i++) {
            if (events[i].timestamp < events[i-1].timestamp) {
                std::cerr << "Warning: Out-of-order timestamp detected!" << std::endl;
            }
        }
    }
    
    void checkContextConsistency(const ProfilingData& data) {
        std::set<CUcontext> observedContexts;
        for (const auto& event : data.events) {
            observedContexts.insert(event.context);
        }
        
        std::cout << "Active contexts: " << observedContexts.size() << std::endl;
    }
};
```

## Next Steps

- Apply concurrent profiling to your multi-stream applications
- Experiment with different concurrency patterns and measure their impact
- Integrate profiling into automated performance testing
- Develop custom analysis tools for your specific concurrency patterns
- Combine with other CUPTI features for comprehensive performance analysis 