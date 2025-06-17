# CUPTI Callback-Based Profiling Tutorial

> The GitHub repo and complete tutorial is available at <https://github.com/eunomia-bpf/cupti-tutorial>.

## Introduction

The CUPTI Callback-Based Profiling sample demonstrates how to implement comprehensive profiling using CUPTI's callback API. This approach allows you to intercept CUDA runtime and driver API calls, collect detailed performance metrics, and analyze GPU activity patterns in real-time during application execution.

## What You'll Learn

- How to register and handle CUPTI callbacks for profiling
- Implementing real-time metric collection during CUDA API calls
- Understanding callback timing and synchronization
- Collecting both API timing and GPU performance metrics
- Building a non-intrusive profiling system using callbacks

## Understanding Callback-Based Profiling

Callback-based profiling offers unique advantages:

1. **Real-time interception**: Monitor CUDA operations as they happen
2. **API-level granularity**: Profile individual API calls and their parameters
3. **Minimal overhead**: Efficient data collection without application modification
4. **Flexible filtering**: Choose which operations to profile
5. **Comprehensive coverage**: Access to both runtime and driver API layers

## Key Concepts

### Callback Types

CUPTI provides callbacks for different API domains:
- **Runtime API**: cudaMalloc, cudaMemcpy, cudaLaunchKernel, etc.
- **Driver API**: cuMemAlloc, cuMemcpyHtoD, cuLaunchKernel, etc.
- **Resource API**: Context and stream creation/destruction
- **Synchronization API**: cudaDeviceSynchronize, cudaStreamSynchronize, etc.

### Callback Phases
Each callback can occur at two phases:
- **Entry**: Before the API call executes
- **Exit**: After the API call completes

### Callback Data
Callbacks provide access to:
- Function name and parameters
- Thread and context information
- Timing data
- Return values and error codes

## Building the Sample

### Prerequisites

Ensure you have:
- CUDA Toolkit with CUPTI
- C++ compiler with C++11 support
- CUPTI development headers

### Build Process

```bash
cd callback_profiling
make
```

This creates the `callback_profiling` executable that demonstrates callback-based profiling techniques.

## Code Architecture

### Main Components

1. **Callback Registration**: Setting up callbacks for desired API domains
2. **Data Collection**: Gathering timing and parameter information
3. **Metric Integration**: Collecting GPU performance metrics
4. **Output Generation**: Formatting and presenting profiling results

### Core Implementation

```cpp
// Callback function signature
void CUPTIAPI callbackHandler(void *userdata, CUpti_CallbackDomain domain,
                             CUpti_CallbackId cbid, const CUpti_CallbackData *cbInfo)
{
    const char *funcName = cbInfo->functionName;
    
    if (cbInfo->callbackSite == CUPTI_API_ENTER) {
        // Entry: Record start time, log parameters
        recordAPIEntry(funcName, cbInfo->functionParams);
    }
    else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
        // Exit: Record end time, log results
        recordAPIExit(funcName, cbInfo->functionReturnValue);
    }
}
```

## Running the Sample

### Basic Execution

```bash
./callback_profiling
```

### Sample Output

```
=== CUPTI Callback Profiling Results ===

CUDA Runtime API Calls:
  cudaMalloc: 3 calls, total: 145μs, avg: 48.3μs
  cudaMemcpy: 6 calls, total: 2.1ms, avg: 350μs
  cudaLaunchKernel: 100 calls, total: 5.2ms, avg: 52μs
  cudaDeviceSynchronize: 1 call, total: 15.3ms, avg: 15.3ms

CUDA Driver API Calls:
  cuCtxCreate: 1 call, total: 125μs, avg: 125μs
  cuModuleLoad: 1 call, total: 2.3ms, avg: 2.3ms
  cuLaunchKernel: 100 calls, total: 4.8ms, avg: 48μs

Performance Metrics:
  GPU Utilization: 78.5%
  Memory Bandwidth: 245.2 GB/s
  Cache Hit Rate: 92.3%

Total Profiling Overhead: 0.8ms (0.5% of total execution time)
```

## Detailed Analysis Features

### API Call Tracking

The sample tracks comprehensive information for each API call:

1. **Call Frequency**: How many times each API is called
2. **Timing Statistics**: Min, max, average, and total execution time
3. **Parameter Analysis**: Memory sizes, kernel configurations, etc.
4. **Error Tracking**: Failed calls and error codes

### Memory Usage Analysis

```cpp
// Track memory allocations
void trackMemoryAllocation(size_t size, void* ptr) {
    totalAllocated += size;
    activeAllocations[ptr] = size;
    allocationHistory.push_back({getCurrentTime(), size, true});
}

// Track memory deallocations
void trackMemoryDeallocation(void* ptr) {
    auto it = activeAllocations.find(ptr);
    if (it != activeAllocations.end()) {
        allocationHistory.push_back({getCurrentTime(), it->second, false});
        activeAllocations.erase(it);
    }
}
```

### Kernel Launch Analysis

```cpp
// Analyze kernel launch parameters
void analyzeKernelLaunch(const dim3& gridDim, const dim3& blockDim, 
                        size_t sharedMem, cudaStream_t stream) {
    int totalThreads = gridDim.x * gridDim.y * gridDim.z * 
                      blockDim.x * blockDim.y * blockDim.z;
    
    kernelStats.totalLaunches++;
    kernelStats.totalThreads += totalThreads;
    kernelStats.sharedMemUsage += sharedMem;
    
    if (stream != 0) {
        kernelStats.asyncLaunches++;
    }
}
```

## Advanced Features

### Selective Profiling

Enable profiling for specific API categories:

```cpp
// Runtime API only
CUPTI_CALL(cuptiEnableCallback(1, subscriber, 
           CUPTI_CB_DOMAIN_RUNTIME_API, 
           CUPTI_RUNTIME_TRACE_CBID_INVALID));

// Specific functions only
CUPTI_CALL(cuptiEnableCallback(1, subscriber,
           CUPTI_CB_DOMAIN_RUNTIME_API,
           CUPTI_RUNTIME_TRACE_CBID_cudaMalloc_v3020));
```

### Performance Metric Integration

```cpp
// Collect GPU metrics during callbacks
void collectMetrics(CUcontext context) {
    CUpti_EventGroup eventGroup;
    CUpti_EventID eventIds[NUM_EVENTS];
    
    // Set up event collection
    CUPTI_CALL(cuptiEventGroupCreate(context, &eventGroup, 0));
    
    for (int i = 0; i < NUM_EVENTS; i++) {
        CUPTI_CALL(cuptiEventGroupAddEvent(eventGroup, eventIds[i]));
    }
    
    // Enable and read events
    CUPTI_CALL(cuptiEventGroupEnable(eventGroup));
    // ... kernel execution ...
    
    uint64_t eventValues[NUM_EVENTS];
    CUPTI_CALL(cuptiEventGroupReadAllEvents(eventGroup, 
               CUPTI_EVENT_READ_FLAG_NONE,
               &bytesRead, eventValues, 
               &numEventIds, eventIds));
}
```

### Multi-threaded Analysis

```cpp
// Thread-safe data collection
class ThreadSafeProfiler {
private:
    std::mutex dataMutex;
    std::unordered_map<std::thread::id, ProfileData> threadData;
    
public:
    void recordAPICall(const std::string& apiName, uint64_t duration) {
        std::lock_guard<std::mutex> lock(dataMutex);
        auto threadId = std::this_thread::get_id();
        threadData[threadId].apiCalls[apiName].addSample(duration);
    }
};
```

## Practical Applications

### Performance Bottleneck Detection

1. **API Overhead Analysis**: Identify expensive CUDA API calls
2. **Memory Transfer Optimization**: Analyze data movement patterns
3. **Kernel Launch Efficiency**: Optimize launch configurations
4. **Synchronization Analysis**: Detect unnecessary synchronization points

### Application Characterization

```cpp
// Generate application profile
struct ApplicationProfile {
    double computeToMemoryRatio;
    double asyncUtilization;
    size_t peakMemoryUsage;
    int averageOccupancy;
    
    void generateReport() {
        std::cout << "Compute/Memory Ratio: " << computeToMemoryRatio << std::endl;
        std::cout << "Async Utilization: " << asyncUtilization * 100 << "%" << std::endl;
        std::cout << "Peak Memory Usage: " << peakMemoryUsage / (1024*1024) << " MB" << std::endl;
        std::cout << "Average Occupancy: " << averageOccupancy << "%" << std::endl;
    }
};
```

### Real-time Monitoring

```cpp
// Live performance dashboard
class LiveProfiler {
private:
    std::atomic<uint64_t> totalAPITime{0};
    std::atomic<uint64_t> totalKernelTime{0};
    std::atomic<size_t> memoryAllocated{0};
    
public:
    void updateDashboard() {
        while (profiling) {
            system("clear");
            std::cout << "=== Live CUDA Profiling Dashboard ===" << std::endl;
            std::cout << "API Time: " << totalAPITime.load() / 1000 << "ms" << std::endl;
            std::cout << "Kernel Time: " << totalKernelTime.load() / 1000 << "ms" << std::endl;
            std::cout << "Memory Allocated: " << memoryAllocated.load() / (1024*1024) << "MB" << std::endl;
            
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
};
```

## Integration with Development Workflow

### Automated Performance Testing

```bash
# Performance regression testing
./callback_profiling --baseline > baseline_profile.txt
# ... make code changes ...
./callback_profiling --compare baseline_profile.txt > regression_report.txt
```

### Continuous Integration

```cpp
// CI-friendly output format
void generateCIReport(const ProfileData& data) {
    json report;
    report["total_api_time"] = data.totalAPITime;
    report["memory_efficiency"] = data.memoryEfficiency;
    report["kernel_utilization"] = data.kernelUtilization;
    
    // Fail CI if performance degrades significantly
    if (data.totalAPITime > PERFORMANCE_THRESHOLD) {
        std::exit(1);
    }
}
```

## Troubleshooting

### Common Issues

1. **Callback not triggered**: Verify callback registration and domain selection
2. **High overhead**: Reduce callback frequency or optimize data collection
3. **Thread safety**: Ensure proper synchronization in multi-threaded applications
4. **Memory leaks**: Check proper cleanup of callback data structures

### Debug Tips

1. **Start with simple callbacks**: Begin with basic timing before adding complex analysis
2. **Use selective profiling**: Focus on specific APIs to reduce overhead
3. **Validate with known applications**: Test with CUDA samples first
4. **Monitor overhead**: Measure profiling impact on application performance

## Next Steps

- Extend the sample to profile specific aspects of your applications
- Integrate callback profiling into your development and testing processes
- Combine with other CUPTI features for comprehensive analysis
- Develop custom metrics and analysis algorithms for your use cases
- Create visualization tools for callback profiling data 