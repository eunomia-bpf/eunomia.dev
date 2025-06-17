# CUPTI OpenACC Tracing Tutorial

## Introduction

OpenACC is a directive-based programming model that simplifies GPU programming by allowing developers to annotate their code with pragmas that the compiler translates into GPU operations. While this simplifies development, it can make performance analysis challenging because the relationship between your directives and the actual GPU operations isn't always clear. This tutorial demonstrates how to use CUPTI to trace OpenACC API calls and correlate them with GPU activities, giving you insights into how your OpenACC code executes on the GPU.

## What You'll Learn

- How to intercept and trace OpenACC API calls
- Correlating OpenACC directives with the resulting GPU operations
- Measuring performance of different OpenACC operations
- Identifying optimization opportunities in OpenACC applications

## Understanding OpenACC Execution

When you write OpenACC code, the following typically happens:

1. The OpenACC compiler translates your directives into GPU code
2. At runtime, the OpenACC runtime library:
   - Manages memory allocations and transfers
   - Launches GPU kernels
   - Synchronizes execution between host and device

By tracing these operations, we can understand how our high-level directives translate to low-level GPU operations.

## Code Walkthrough

### 1. Setting Up OpenACC API Interception

To trace OpenACC API calls, we need to intercept the functions from the OpenACC runtime library:

```cpp
// Define function pointer types for OpenACC API functions
typedef int (*acc_init_t)(acc_device_t);
typedef int (*acc_get_num_devices_t)(acc_device_t);
typedef void (*acc_set_device_num_t)(int, acc_device_t);
typedef int (*acc_get_device_num_t)(acc_device_t);
typedef void* (*acc_create_t)(void*, size_t);
typedef void* (*acc_copyin_t)(void*, size_t);
typedef void (*acc_delete_t)(void*, size_t);
typedef void* (*acc_copyout_t)(void*, size_t);
// ... more function types ...

// Original function pointers
static acc_init_t real_acc_init = NULL;
static acc_get_num_devices_t real_acc_get_num_devices = NULL;
static acc_set_device_num_t real_acc_set_device_num = NULL;
// ... more function pointers ...

// Initialize function pointers using dlsym
void initOpenACCAPI()
{
    void *handle = dlopen("libopenacc.so", RTLD_NOW);
    if (!handle) {
        fprintf(stderr, "Error: could not load OpenACC library\n");
        exit(1);
    }
    
    real_acc_init = (acc_init_t)dlsym(handle, "acc_init");
    real_acc_get_num_devices = (acc_get_num_devices_t)dlsym(handle, "acc_get_num_devices");
    real_acc_set_device_num = (acc_set_device_num_t)dlsym(handle, "acc_set_device_num");
    // ... load more functions ...
}
```

This code:
1. Defines function pointer types matching the OpenACC API
2. Creates variables to hold the original function pointers
3. Uses `dlsym` to get the addresses of the actual functions in the OpenACC library

### 2. Implementing Wrapper Functions

For each OpenACC function we want to trace, we implement a wrapper function:

```cpp
// Wrapper for acc_init
int acc_init(acc_device_t device_type)
{
    // Record start time
    double startTime = getCurrentTime();
    
    // Call the real function
    int result = real_acc_init(device_type);
    
    // Record end time
    double endTime = getCurrentTime();
    
    // Log the function call
    printf("[%.3f ms] acc_init(", (startTime - programStartTime) * 1000.0);
    printDeviceType(device_type);
    printf(")\n");
    
    return result;
}

// Wrapper for acc_create
void* acc_create(void* host_addr, size_t size)
{
    double startTime = getCurrentTime();
    
    // Call the real function
    void* result = real_acc_create(host_addr, size);
    
    double endTime = getCurrentTime();
    
    // Log the function call with details
    printf("[%.3f ms] acc_create(%p, %zu) [%zu elements]\n", 
           (startTime - programStartTime) * 1000.0,
           host_addr, size, size / sizeof(float));
    
    // Track the CUDA memory allocation
    trackMemoryAllocation(host_addr, result, size);
    
    return result;
}
```

These wrappers:
1. Record the time before and after calling the real function
2. Log information about the function call
3. Track resources like memory allocations
4. Return the result from the real function

### 3. Correlating with CUDA Activities

To correlate OpenACC API calls with CUDA operations, we use CUPTI's Activity API:

```cpp
void initCUPTI()
{
    // Initialize CUPTI activity API
    CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));
    
    // Enable activity kinds we want to track
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMORY));
    
    // Get the start timestamp for normalizing times
    CUPTI_CALL(cuptiGetTimestamp(&startTimestamp));
}

void processActivity(CUpti_Activity *record)
{
    switch (record->kind) {
    case CUPTI_ACTIVITY_KIND_MEMCPY: {
        CUpti_ActivityMemcpy *memcpy = (CUpti_ActivityMemcpy *)record;
        
        // Find the corresponding OpenACC API call
        OpenACCMemoryOp *op = findOpenACCMemoryOp(memcpy->start);
        
        if (op) {
            // This memory copy corresponds to an OpenACC data directive
            printf("[%.3f ms] > CUDA %s Transfer: %zu bytes %s %p\n",
                   (memcpy->start - startTimestamp) / 1000.0,
                   getMemcpyKindString(memcpy->copyKind),
                   memcpy->bytes,
                   (memcpy->copyKind == CUPTI_ACTIVITY_MEMCPY_KIND_HTOD) ? "to" : "from",
                   (void*)memcpy->deviceId);
            
            // Update statistics
            if (memcpy->copyKind == CUPTI_ACTIVITY_MEMCPY_KIND_HTOD) {
                hostToDeviceTime += (memcpy->end - memcpy->start) / 1000.0;
            } else {
                deviceToHostTime += (memcpy->end - memcpy->start) / 1000.0;
            }
        }
        break;
    }
    case CUPTI_ACTIVITY_KIND_KERNEL: {
        CUpti_ActivityKernel5 *kernel = (CUpti_ActivityKernel5 *)record;
        
        // Find the corresponding OpenACC compute region
        OpenACCComputeRegion *region = findOpenACCComputeRegion(kernel->start);
        
        if (region) {
            // This kernel corresponds to an OpenACC compute directive
            printf("[%.3f ms] > CUDA Kernel Launch: %s [grid:%d block:%d]\n",
                   (kernel->start - startTimestamp) / 1000.0,
                   kernel->name,
                   kernel->gridX,
                   kernel->blockX);
            
            // Update statistics
            kernelExecutionTime += (kernel->end - kernel->start) / 1000.0;
        }
        break;
    }
    // ... handle other activity types ...
    }
}
```

This code:
1. Sets up CUPTI to collect activities like memory transfers and kernel launches
2. Processes each activity record and correlates it with OpenACC operations
3. Updates performance statistics for different types of operations

### 4. Sample OpenACC Application

The sample includes a simple OpenACC application to demonstrate the tracing:

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <openacc.h>

#define N 4096

int main()
{
    float *a, *b, *c;
    int i;
    
    // Allocate host memory
    a = (float*)malloc(N * sizeof(float));
    b = (float*)malloc(N * sizeof(float));
    c = (float*)malloc(N * sizeof(float));
    
    // Initialize arrays
    for (i = 0; i < N; i++) {
        a[i] = (float)i;
        b[i] = (float)(N - i);
    }
    
    // OpenACC directives for vector addition
    #pragma acc data create(a[0:N], b[0:N], c[0:N])
    {
        #pragma acc update device(a[0:N], b[0:N])
        
        #pragma acc parallel loop
        for (i = 0; i < N; i++) {
            c[i] = a[i] + b[i];
        }
        
        #pragma acc update host(c[0:N])
    }
    
    // Verify results
    for (i = 0; i < N; i++) {
        if (c[i] != (float)(N)) {
            printf("Error: c[%d] = %f, expected %f\n", i, c[i], (float)N);
            break;
        }
    }
    
    printf("Vector addition completed successfully\n");
    
    // Free memory
    free(a);
    free(b);
    free(c);
    
    return 0;
}
```

This simple application:
1. Allocates and initializes arrays on the host
2. Uses OpenACC directives to:
   - Create device memory
   - Copy data to the device
   - Execute a parallel loop for vector addition
   - Copy results back to the host
3. Verifies the results

## Running the Tutorial

1. Build the sample:
   ```bash
   make
   ```

2. Run the OpenACC tracer:
   ```bash
   ./openacc_trace
   ```

## Understanding the Output

The output shows a timeline of OpenACC API calls and their corresponding GPU activities:

```
OpenACC API Trace:
[00.000 ms] acc_init(acc_device_nvidia)
[00.245 ms] acc_get_num_devices(acc_device_nvidia) returned 1
[00.302 ms] acc_set_device_num(0, acc_device_nvidia)
[00.356 ms] acc_get_device_num(acc_device_nvidia) returned 0

[01.234 ms] acc_create(0x7fff5a1c3000, 16384) [4096 elements]
[01.456 ms] > CUDA Memory Allocation: 16384 bytes at 0xd00000
[01.789 ms] acc_create(0x7fff5a1c7000, 16384) [4096 elements]
[01.856 ms] > CUDA Memory Allocation: 16384 bytes at 0xd04000

[02.123 ms] acc_copyin(0x7fff5a1c3000, 16384) [4096 elements]
[02.234 ms] > CUDA H2D Transfer: 16384 bytes to 0xd00000
[02.345 ms] acc_copyin(0x7fff5a1c7000, 16384) [4096 elements]
[02.456 ms] > CUDA H2D Transfer: 16384 bytes to 0xd04000

[03.012 ms] acc_parallel: entering compute region
[03.123 ms] > CUDA Kernel Launch: Vector Addition [grid:16 block:256]
[03.456 ms] acc_parallel: exiting compute region

[04.123 ms] acc_copyout(0x7fff5a1c3000, 16384) [4096 elements]
[04.234 ms] > CUDA D2H Transfer: 16384 bytes from 0xd00000
[04.345 ms] acc_delete(0x7fff5a1c3000, 16384)
[04.456 ms] > CUDA Memory Free: 0xd00000
[04.567 ms] acc_delete(0x7fff5a1c7000, 16384)
[04.678 ms] > CUDA Memory Free: 0xd04000
```

This output shows:

1. **Initialization Phase**:
   - OpenACC runtime initializes and selects the NVIDIA GPU device

2. **Memory Allocation Phase**:
   - `acc_create` calls allocate memory on the device
   - Each call is matched with a CUDA memory allocation

3. **Data Transfer Phase**:
   - `acc_copyin` calls transfer data from host to device
   - Each call is matched with a CUDA H2D (Host-to-Device) transfer

4. **Computation Phase**:
   - `acc_parallel` marks the beginning and end of a compute region
   - The region is matched with a CUDA kernel launch

5. **Result Retrieval Phase**:
   - `acc_copyout` transfers results from device to host
   - Matched with a CUDA D2H (Device-to-Host) transfer

6. **Cleanup Phase**:
   - `acc_delete` frees device memory
   - Matched with CUDA memory free operations

The trace also includes a performance summary:

```
OpenACC Performance Summary:
  Data Allocation Time: 0.623 ms
  Host-to-Device Transfer Time: 0.344 ms
  Kernel Execution Time: 0.444 ms
  Device-to-Host Transfer Time: 0.111 ms
  Data Deallocation Time: 0.233 ms
  Total OpenACC API Time: 1.755 ms
```

This summary helps identify where time is spent in your OpenACC application.

## Performance Analysis

The trace data reveals several important insights:

1. **Data Movement Overhead**:
   - Compare the time spent in data transfers versus computation
   - Look for unnecessary data transfers

2. **Memory Management**:
   - Check if memory is being allocated and freed efficiently
   - Look for opportunities to reuse device memory

3. **Kernel Efficiency**:
   - Examine the grid and block dimensions of launched kernels
   - Compare the kernel execution time with the overall compute region time

4. **API Overhead**:
   - Measure the overhead of OpenACC runtime API calls
   - Look for cases where manual CUDA code might be more efficient

## Advanced Usage

### Tracing Specific OpenACC Constructs

To focus on specific OpenACC constructs, you can modify the wrapper functions to trace only certain API calls:

```cpp
// Only trace data movement operations
void* acc_copyin(void* host_addr, size_t size)
{
    double startTime = getCurrentTime();
    void* result = real_acc_copyin(host_addr, size);
    double endTime = getCurrentTime();
    
    // Only log if size is above a threshold
    if (size > THRESHOLD) {
        printf("[%.3f ms] acc_copyin(%p, %zu)\n", 
               (startTime - programStartTime) * 1000.0,
               host_addr, size);
    }
    
    return result;
}
```

### Correlating with Source Code

To correlate GPU activities with your source code, you can add source file and line information:

```cpp
// Add source information to OpenACC API traces
#define ACC_TRACE(func, args...) \
    printf("[%.3f ms] %s:%d: " #func "(", \
           (getCurrentTime() - programStartTime) * 1000.0, \
           __FILE__, __LINE__); \
    func(args); \
    printf(")\n")
```

## Next Steps

- Apply OpenACC tracing to your own applications
- Identify performance bottlenecks in data movement and computation
- Use the insights to optimize your OpenACC directives
- Compare different OpenACC implementations for the same algorithm 