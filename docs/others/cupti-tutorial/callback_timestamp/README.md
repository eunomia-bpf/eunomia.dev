# CUPTI Timestamp Callback Tutorial

> The GitHub repo and complete tutorial is available at <https://github.com/eunomia-bpf/cupti-tutorial>.

## Introduction

Accurately measuring the execution time of CUDA operations is essential for performance optimization. While CPU-based timing can give approximate results, GPU timestamps provide precise measurements of when operations actually execute on the device. This tutorial demonstrates how to use CUPTI callbacks to collect GPU timestamps for CUDA operations, giving you accurate timing information for memory transfers and kernel executions.

## What You'll Learn

- How to set up CUPTI callbacks for CUDA runtime API functions
- Collecting precise GPU timestamps at the beginning and end of operations
- Measuring execution time for memory transfers and kernels
- Analyzing the performance of a complete CUDA workflow

## Understanding GPU Timestamps

GPU timestamps differ from CPU timestamps in several important ways:

1. They measure time from the GPU's perspective
2. They provide nanosecond precision
3. They accurately capture when operations execute on the device, not just when they're submitted
4. They're essential for understanding the true performance characteristics of GPU operations

## Code Walkthrough

### 1. Setting Up the Callback System

First, we need to initialize CUPTI and register our callback function:

```cpp
int main(int argc, char *argv[])
{
    // Initialize CUDA
    RUNTIME_API_CALL(cudaSetDevice(0));
    
    // Subscribe to CUPTI callbacks
    CUpti_SubscriberHandle subscriber;
    CUPTI_CALL(cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)getTimestampCallback, &traceData));
    
    // Enable callbacks for CUDA runtime
    CUPTI_CALL(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API));
    
    // Run our test workload
    runTest();
    
    // Unsubscribe from callbacks
    CUPTI_CALL(cuptiUnsubscribe(subscriber));
    
    return 0;
}
```

This sets up CUPTI to call our `getTimestampCallback` function whenever a CUDA runtime API function is called.

### 2. The Callback Function

The heart of the sample is the callback function that collects timestamps:

```cpp
void CUPTIAPI getTimestampCallback(void *userdata, CUpti_CallbackDomain domain,
                                  CUpti_CallbackId cbid, const CUpti_CallbackData *cbInfo)
{
    RuntimeApiTrace_t *traceData = (RuntimeApiTrace_t*)userdata;
    
    // Only process runtime API callbacks
    if (domain != CUPTI_CB_DOMAIN_RUNTIME_API) return;
    
    // We're interested in memory transfers and kernel launches
    if ((cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020) ||
        (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020) ||
        (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaDeviceSynchronize_v3020)) {
        
        if (cbInfo->callbackSite == CUPTI_API_ENTER) {
            // Record information at the start of the API call
            uint64_t timestamp;
            CUPTI_CALL(cuptiDeviceGetTimestamp(cbInfo->context, &timestamp));
            
            // Store function name
            traceData->functionName = cbInfo->functionName;
            traceData->startTimestamp = timestamp;
            
            // For memory transfers, capture size and direction
            if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020) {
                cudaMemcpy_v3020_params *params = 
                    (cudaMemcpy_v3020_params*)cbInfo->functionParams;
                traceData->memcpyBytes = params->count;
                traceData->memcpyKind = params->kind;
            }
        }
        else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
            // Record information at the end of the API call
            uint64_t timestamp;
            CUPTI_CALL(cuptiDeviceGetTimestamp(cbInfo->context, &timestamp));
            
            // Calculate duration
            traceData->gpuTime = timestamp - traceData->startTimestamp;
            
            // Print the timing information
            printTimestampData(traceData);
        }
    }
}
```

Key aspects of this function:
1. It filters for specific CUDA functions we want to time
2. On API entry, it records the function name, start timestamp, and any relevant parameters
3. On API exit, it records the end timestamp and calculates the duration
4. For memory transfers, it captures the size and direction

### 3. The Test Workload

To demonstrate timing, we run a simple vector addition workflow:

```cpp
void runTest()
{
    int N = 50000;
    size_t size = N * sizeof(float);
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    
    // Allocate host memory
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);
    
    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }
    
    // Allocate device memory
    RUNTIME_API_CALL(cudaMalloc((void**)&d_A, size));
    RUNTIME_API_CALL(cudaMalloc((void**)&d_B, size));
    RUNTIME_API_CALL(cudaMalloc((void**)&d_C, size));
    
    // Transfer data from host to device (these will be timed)
    RUNTIME_API_CALL(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    RUNTIME_API_CALL(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
    
    // Launch kernel (this will be timed)
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    
    // Transfer results back to host (this will be timed)
    RUNTIME_API_CALL(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));
    
    // Synchronize to make sure all operations are complete
    RUNTIME_API_CALL(cudaDeviceSynchronize());
    
    // Clean up
    free(h_A);
    free(h_B);
    free(h_C);
    RUNTIME_API_CALL(cudaFree(d_A));
    RUNTIME_API_CALL(cudaFree(d_B));
    RUNTIME_API_CALL(cudaFree(d_C));
}
```

This workflow includes:
1. Two host-to-device memory transfers
2. One kernel execution
3. One device-to-host memory transfer
4. A device synchronization

### 4. Displaying Timing Results

The results are displayed in a formatted table:

```cpp
void printTimestampData(RuntimeApiTrace_t *traceData)
{
    static bool headerPrinted = false;
    
    if (!headerPrinted) {
        printf("\nstartTimeStamp/gpuTime reported in nano-seconds\n\n");
        printf("%-15s %-24s %-15s %-8s %-10s\n", "Name", "Start Time", "GPU Time", "Bytes", "Kind");
        headerPrinted = true;
    }
    
    // Print function name and timing information
    printf("%-15s %-24llu %-15llu", 
           traceData->functionName,
           (unsigned long long)traceData->startTimestamp,
           (unsigned long long)traceData->gpuTime);
    
    // For memory transfers, print size and direction
    if (strcmp(traceData->functionName, "cudaMemcpy") == 0) {
        printf(" %-8lu %-10s", 
               (unsigned long)traceData->memcpyBytes,
               getMemcpyKindString((cudaMemcpyKind)traceData->memcpyKind));
    } else {
        printf(" %-8s %-10s", "NA", "NA");
    }
    
    printf("\n");
}
```

## Running the Tutorial

1. Build the sample:
   ```bash
   make
   ```

2. Run the timestamp collector:
   ```bash
   ./callback_timestamp
   ```

## Understanding the Output

The sample produces output similar to:

```
startTimeStamp/gpuTime reported in nano-seconds

Name            Start Time              GPU Time        Bytes   Kind
cudaMemcpy      123456789012            5432           200000   HostToDevice
cudaMemcpy      123456794444            5432           200000   HostToDevice
VecAdd          123456799876            10864          NA       NA
cudaMemcpy      123456810740            5432           200000   DeviceToHost
cudaDeviceSync  123456816172            0              NA       NA
```

Let's analyze this output:

1. **Start Time**: The GPU timestamp when the operation began (in nanoseconds)
2. **GPU Time**: The duration of the operation on the GPU (in nanoseconds)
3. **Bytes**: For memory transfers, the amount of data transferred
4. **Kind**: For memory transfers, the direction (HostToDevice or DeviceToHost)

From this output, we can see:
- Memory transfers take about 5.4 microseconds each
- The kernel execution takes about 10.8 microseconds
- The entire workflow completes in about 27 microseconds

## Performance Insights

This timing data reveals several important aspects of CUDA performance:

1. **Memory Transfer Overhead**: Memory transfers can be a significant bottleneck
2. **Kernel Execution Time**: The actual computation time on the GPU
3. **Operation Sequence**: The order and timing of operations in your workflow

By analyzing this data, you can:
- Identify bottlenecks in your application
- Optimize memory transfer patterns
- Balance computation and data movement
- Make informed decisions about kernel optimization priorities

## Advanced Applications

### Timing Custom Operations

To time custom operations:
1. Wrap them in CUDA API calls that will trigger callbacks
2. Use `cudaEventRecord` to create timing points within complex operations

### Correlating with Application Phases

To understand performance in the context of your application:
1. Add markers or identifiers to your timing data
2. Correlate GPU timestamps with application phases
3. Look for patterns or anomalies in specific parts of your workflow

## Next Steps

- Modify the sample to time additional CUDA operations
- Integrate timestamp collection into your own CUDA applications
- Compare GPU timestamps with CPU-based timing to understand the relationship
- Use the timing data to prioritize optimization efforts in your code 