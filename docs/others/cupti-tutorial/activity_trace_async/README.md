# CUPTI Asynchronous Activity Tracing Tutorial

> The GitHub repo and complete tutorial is available at <https://github.com/eunomia-bpf/cupti-tutorial>.

## Introduction

Profiling GPU applications can significantly impact performance if not done carefully. This tutorial demonstrates how to use CUPTI's asynchronous activity tracing to collect performance data with minimal impact on your application's execution time. You'll learn how to collect detailed GPU and API activity traces while your application continues to run at full speed.

## What You'll Learn

- How to set up asynchronous buffer handling with CUPTI
- Techniques for non-blocking collection of performance data
- Processing profiling information in a separate thread
- Minimizing the performance impact of profiling

## Asynchronous vs. Synchronous Tracing

Before diving into the code, let's understand why asynchronous tracing matters:

| Synchronous Tracing | Asynchronous Tracing |
|---------------------|----------------------|
| Blocks application during buffer processing | Application continues running during buffer processing |
| Simple implementation | Requires thread-safe handling |
| Higher performance impact | Lower performance impact |
| Suitable for short test runs | Suitable for production or long-running applications |

## Code Walkthrough

### 1. Setting Up Asynchronous Buffer Handling

The key to asynchronous tracing is configuring CUPTI to use a separate thread for buffer management:

```cpp
void initTrace()
{
  // Enable asynchronous buffer handling
  CUpti_ActivityAttribute attr = CUPTI_ACTIVITY_ATTR_CALLBACKS;
  CUpti_BuffersCallbackRequestFunc bufferRequested = bufferRequestedCallback;
  CUpti_BuffersCallbackCompleteFunc bufferCompleted = bufferCompletedCallback;
  
  CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));
  
  // Enable the activity kinds you want to track
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DEVICE));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONTEXT));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  
  // Capture timestamp for normalizing times
  CUPTI_CALL(cuptiGetTimestamp(&startTimestamp));
}
```

### 2. Asynchronous Buffer Request Callback

The buffer request callback is similar to the synchronous version, but now it must be thread-safe:

```cpp
static void CUPTIAPI
bufferRequestedCallback(uint8_t **buffer, size_t *size, size_t *maxNumRecords)
{
  *size = BUF_SIZE;
  
  // Allocate a buffer with extra space for alignment
  *buffer = (uint8_t *)malloc(*size + ALIGN_SIZE);
  if (*buffer == NULL) {
    printf("Error: Out of memory\n");
    exit(-1);
  }
  
  // Align the buffer to ALIGN_SIZE
  *buffer = ALIGN_BUFFER(*buffer, ALIGN_SIZE);
  
  // No limit on the number of records
  *maxNumRecords = 0;
}
```

### 3. Asynchronous Buffer Completion Callback

The most important part is the buffer completion callback, which now runs in a separate thread:

```cpp
static void CUPTIAPI
bufferCompletedCallback(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize)
{
  CUptiResult status;
  CUpti_Activity *record = NULL;
  
  // This callback is running in a separate thread, so thread safety is essential
  
  // Process all records in the buffer
  do {
    status = cuptiActivityGetNextRecord(buffer, validSize, &record);
    if (status == CUPTI_SUCCESS) {
      // Process the record - must be quick and non-blocking
      printActivity(record);
      
      // Move to the next record
      validSize -= record->common.size;
      buffer += record->common.size;
    }
    else if (status != CUPTI_ERROR_MAX_LIMIT_REACHED) {
      // Handle any errors
      CUPTI_CALL(status);
    }
  } while (status == CUPTI_SUCCESS);
  
  // Free the buffer - must be the same pointer returned by bufferRequested
  free(buffer);
}
```

**Important notes about this callback:**
- It runs in a separate thread from your application
- It should return quickly to avoid slowing down buffer processing
- Any data structures accessed in this callback must be thread-safe
- The buffer must be freed in this callback to avoid memory leaks

### 4. Record Processing

The record processing is similar to the synchronous version:

```cpp
static void
printActivity(CUpti_Activity *record)
{
  switch (record->kind) {
  case CUPTI_ACTIVITY_KIND_DEVICE:
    {
      // Print device info
      CUpti_ActivityDevice2 *device = (CUpti_ActivityDevice2 *)record;
      printf("DEVICE %s (%u), capability %u.%u, ...\n",
             device->name, device->id,
             device->computeCapabilityMajor, device->computeCapabilityMinor);
      break;
    }
  case CUPTI_ACTIVITY_KIND_KERNEL:
    {
      // Print kernel info
      CUpti_ActivityKernel4 *kernel = (CUpti_ActivityKernel4 *)record;
      printf("KERNEL \"%s\" [ %llu - %llu ] device %u, context %u, stream %u\n",
             kernel->name,
             (unsigned long long)(kernel->start - startTimestamp),
             (unsigned long long)(kernel->end - startTimestamp),
             kernel->deviceId, kernel->contextId, kernel->streamId);
      break;
    }
  // ... other activity types ...
  }
}
```

### 5. Test Kernel and Main Function

The sample includes a simple vector addition kernel to generate profiling data:

```cpp
__global__ void vecAdd(const float *A, const float *B, float *C, int numElements)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < numElements)
    C[i] = A[i] + B[i];
}

int main(int argc, char *argv[])
{
  // Initialize CUPTI tracing
  initTrace();
  
  // Allocate and initialize vectors
  // ...
  
  // Launch the kernel
  dim3 threadsPerBlock(256);
  dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x);
  vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
  
  // Copy results back to host
  cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
  
  // Verify results
  // ...
  
  // Finalize tracing - ensure all data is flushed
  finiTrace();
  
  return 0;
}
```

### 6. Finalizing the Trace

When your application is done, make sure to flush all remaining buffers:

```cpp
void finiTrace()
{
  // Flush any activity records still in buffers
  CUPTI_CALL(cuptiActivityFlushAll(0));
}
```

## Running the Sample

1. Build the sample:
   ```bash
   make
   ```

2. Run the asynchronous tracer:
   ```bash
   ./activity_trace_async
   ```

## Understanding the Output

The output shows all GPU activities with timestamps:

```
DEVICE Device Name (0), capability 7.0, global memory (bandwidth 900 GB/s, size 16000 MB), multiprocessors 80, clock 1530 MHz
CONTEXT 1, device 0, compute API CUDA, NULL stream 1
DRIVER_API cuCtxCreate [ 10223 - 15637 ] 
MEMCPY HtoD [ 22500 - 23012 ] device 0, context 1, stream 7, correlation 1/1
KERNEL "vecAdd" [ 32058 - 35224 ] device 0, context 1, stream 7, correlation 2
MEMCPY DtoH [ 40388 - 41002 ] device 0, context 1, stream 7, correlation 3/3
```

The timestamps (in brackets) show when each operation started and ended, normalized to the beginning of the trace.

## Performance Considerations

Asynchronous tracing offers several performance benefits:

1. **Reduced Overhead**: The main application thread continues running while activity records are processed in a separate thread
2. **Lower Latency**: CUDA operations aren't blocked waiting for buffer processing
3. **Better Scalability**: Works well for long-running applications or production monitoring

## Advanced Usage Tips

1. **Buffer Size Tuning**: Adjust `BUF_SIZE` based on your application:
   - Larger buffers reduce callback frequency but use more memory
   - Smaller buffers use less memory but may increase callback overhead

2. **Selective Tracing**: Only enable the activity kinds you need:
   ```cpp
   // If you only care about kernels, just enable these
   cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL);
   cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
   ```

3. **Thread Safety**: When implementing your own version, ensure any data structures accessed by the buffer completion callback are thread-safe.

## Next Steps

- Try enabling different activity kinds to see different aspects of your application
- Modify the code to store records in a database or file for later analysis
- Implement a custom visualization tool to help understand the timeline data
- Try running the asynchronous tracer on your own CUDA applications 