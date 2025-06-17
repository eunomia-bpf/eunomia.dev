# CUPTI Activity Trace Tutorial

> The GitHub repo and complete tutorial is available at <https://github.com/eunomia-bpf/cupti-tutorial>.

## Introduction

Profiling CUDA applications is essential for understanding their performance characteristics. The CUPTI Activity API provides a powerful way to collect detailed traces of both CUDA API calls and GPU activities. This tutorial explains how to use CUPTI to gather and analyze this data.

## What You'll Learn

- How to initialize the CUPTI Activity API
- Setting up and managing activity record buffers
- Processing activity records from multiple sources
- Interpreting activity data for optimization

## Code Walkthrough

### 1. Setting Up Activity Tracing

The core of the activity tracing system revolves around buffer management. CUPTI requests buffers to store activity records and notifies when those buffers are filled.

```cpp
// Buffer request callback - called when CUPTI needs a new buffer
static void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords)
{
  // Allocate buffer for CUPTI records
  *size = BUF_SIZE;
  *buffer = (uint8_t *)malloc(*size + ALIGN_SIZE);
  
  // Ensure buffer is properly aligned
  *buffer = ALIGN_BUFFER(*buffer, ALIGN_SIZE);
  *maxNumRecords = 0;
}
```

This function allocates memory when CUPTI requests a buffer to store activity records. The alignment is important for performance.

### 2. Processing Completed Buffers

```cpp
// Buffer completion callback - called when CUPTI has filled a buffer
static void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize)
{
  CUpti_Activity *record = NULL;
  
  // Process all records in the buffer
  CUptiResult status = CUPTI_SUCCESS;
  while (validSize > 0) {
    status = cuptiActivityGetNextRecord(buffer, validSize, &record);
    if (status == CUPTI_SUCCESS) {
      printActivity(record);
      validSize -= record->common.size;
      buffer += record->common.size;
    }
    else
      break;
  }
  
  free(buffer);
}
```

When CUPTI fills a buffer with activity data, this callback processes each record and then frees the buffer.

### 3. Activity Record Processing

The `printActivity` function is the heart of the analysis, interpreting different types of activities:

```cpp
static void printActivity(CUpti_Activity *record)
{
  switch (record->kind) {
  case CUPTI_ACTIVITY_KIND_DEVICE:
    // Print device information
    ...
  case CUPTI_ACTIVITY_KIND_MEMCPY:
    // Print memory copy details
    ...
  case CUPTI_ACTIVITY_KIND_KERNEL:
    // Print kernel execution details
    ...
    
  // Many more activity types...
  }
}
```

Each activity type provides different insights:
- Device activities show hardware capabilities
- Memory copy activities reveal data transfer patterns and times
- Kernel activities show execution time and parameters

### 4. Initialization and Cleanup

```cpp
void initTrace()
{
  // Register callbacks for buffer management
  CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));
  
  // Enable various activity kinds
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DEVICE));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONTEXT));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));
  // ... more activity kinds ...
  
  // Capture timestamp for normalizing times
  CUPTI_CALL(cuptiGetTimestamp(&startTimestamp));
}

void finiTrace()
{
  // Flush any remaining data
  CUPTI_CALL(cuptiActivityFlushAll(0));
}
```

The initialization function enables specific activity kinds that you want to monitor and registers the callbacks. The cleanup function ensures all data is processed.

### 5. The Test Kernel

The sample uses a simple vector addition kernel (in `vec.cu`) to generate activities to trace:

```cpp
__global__ void vecAdd(const float *A, const float *B, float *C, int numElements)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < numElements)
    C[i] = A[i] + B[i];
}
```

## Running the Sample

1. Build the sample:
   ```bash
   make
   ```

2. Run the activity trace:
   ```bash
   ./activity_trace
   ```

## Understanding the Output

The output shows a chronological trace of activities:

```
DEVICE Device Name (0), capability 7.0, global memory (bandwidth 900 GB/s, size 16000 MB), multiprocessors 80, clock 1530 MHz
CONTEXT 1, device 0, compute API CUDA, NULL stream 1
DRIVER_API cuCtxCreate [ 10223 - 15637 ] 
MEMCPY HtoD [ 22500 - 23012 ] device 0, context 1, stream 7, correlation 1/1
KERNEL "vecAdd" [ 32058 - 35224 ] device 0, context 1, stream 7, correlation 2
MEMCPY DtoH [ 40388 - 41002 ] device 0, context 1, stream 7, correlation 3/3
```

Let's decode this:
1. **Device information**: Shows GPU capabilities
2. **Context creation**: CUDA context initialization
3. **Memory copies**: 
   - `HtoD` (Host to Device) shows data being uploaded to the GPU
   - `DtoH` (Device to Host) shows results being downloaded
4. **Kernel execution**: Shows the execution time of our vector addition

The timestamps (in brackets) are normalized to the start of tracing, making it easy to see the relative timing of operations.

## Performance Insights

With this trace data, you can:
- Identify bottlenecks in memory transfers
- Determine kernel execution efficiency
- Find synchronization points and their impact
- Measure the overhead of CUDA API calls

## Next Steps

- Try modifying the vector size to see how it affects performance
- Enable additional activity kinds to gather more detailed information
- Compare the timings of different GPU operations in your own applications
- Explore CUPTI's other activity-based samples for more advanced tracing 