# CUPTI API-GPU Activity Correlation Tutorial

## Introduction

This sample demonstrates how to correlate CUDA API calls with their corresponding GPU activities using CUPTI correlation IDs. Understanding this relationship is essential for performance analysis and debugging.

## What You'll Learn

- How to correlate CUDA API calls with GPU activities
- Using correlation IDs to track execution flow
- Building correlation maps for analysis
- Identifying performance bottlenecks through correlation

## Key Concepts

### Correlation IDs
Every CUDA API call gets a unique correlation ID that links it to the GPU activities it generates:

```cpp
// API record with correlation ID
CUpti_ActivityAPI *apiRecord = ...;
uint32_t correlationId = apiRecord->correlationId;

// GPU activity with same correlation ID
CUpti_ActivityKernel9 *kernelRecord = ...;
uint32_t sameId = kernelRecord->correlationId;
```

### Sample Architecture

```cpp
// Maps to store correlated records
static std::map<uint32_t, CUpti_Activity*> s_CorrelationMap;  // GPU activities
static std::map<uint32_t, CUpti_Activity*> s_ConnectionMap;   // API calls

void ProcessActivityRecord(CUpti_Activity* record) {
    switch (record->kind) {
        case CUPTI_ACTIVITY_KIND_API:
            // Store API record
            s_ConnectionMap[apiRecord->correlationId] = record;
            break;
        case CUPTI_ACTIVITY_KIND_KERNEL:
            // Store GPU activity
            s_CorrelationMap[kernelRecord->correlationId] = record;
            break;
    }
}
```

## Sample Walkthrough

The sample performs vector operations and correlates each API call with its GPU activity:

1. **Memory allocation**: `cudaMalloc` → GPU memory allocation
2. **Memory transfer**: `cudaMemcpyAsync` → DMA transfer activity  
3. **Kernel launch**: `VectorAdd<<<>>>` → Kernel execution activity
4. **Synchronization**: `cudaStreamSynchronize` → GPU idle/sync activity

### Correlation Analysis

```cpp
void PrintCorrelationInformation() {
    for (auto& pair : s_CorrelationMap) {
        uint32_t correlationId = pair.first;
        CUpti_Activity* gpuActivity = pair.second;
        
        // Find corresponding API record
        auto apiIter = s_ConnectionMap.find(correlationId);
        if (apiIter != s_ConnectionMap.end()) {
            printf("Correlation ID: %u\n", correlationId);
            PrintActivity(gpuActivity, stdout);
            PrintActivity(apiIter->second, stdout);
        }
    }
}
```

## Building and Running

```bash
cd cupti_correlation
make
./cupti_correlation
```

## Sample Output

```
CUDA_API AND GPU ACTIVITY CORRELATION : correlation 1
CUPTI_ACTIVITY_KIND_MEMCPY : start=1000 end=1500 duration=500
CUPTI_ACTIVITY_KIND_API : start=950 end=1600 name=cudaMemcpyAsync

CUDA_API AND GPU ACTIVITY CORRELATION : correlation 2  
CUPTI_ACTIVITY_KIND_KERNEL : start=2000 end=2100 duration=100
CUPTI_ACTIVITY_KIND_API : start=1900 end=2200 name=cudaLaunchKernel
```

## Use Cases

- **Performance Analysis**: Identify API overhead vs GPU execution time
- **Debugging**: Trace which API calls generate specific GPU activities
- **Optimization**: Find opportunities to overlap operations
- **Profiling**: Build complete execution timelines

## Next Steps

- Extend correlation to include multiple GPU activities per API call
- Add timing analysis to identify bottlenecks
- Implement correlation for multi-stream applications
- Build visualization tools using correlation data

Understanding API-GPU correlation is fundamental to CUDA performance optimization and debugging. This sample provides the foundation for building sophisticated profiling and analysis tools. 