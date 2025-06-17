# CUPTI Unified Memory Profiling Tutorial

## Introduction

CUDA Unified Memory creates a single memory space accessible by both CPU and GPU, simplifying memory management in heterogeneous computing. While this abstraction makes programming easier, it introduces behind-the-scenes data migrations that can significantly impact performance. This tutorial demonstrates how to use CUPTI to profile and analyze Unified Memory operations, helping you understand memory migration patterns and optimize your applications for better performance.

## What You'll Learn

- How to track and analyze Unified Memory events
- Monitoring page faults and data migrations between CPU and GPU
- Understanding the performance impact of different memory access patterns
- Applying memory advice to optimize data placement
- Using CUPTI to gain insights into Unified Memory behavior

## Understanding Unified Memory

Unified Memory provides a single pointer that can be accessed from both CPU and GPU code. The CUDA runtime automatically migrates data between host and device as needed. This process involves:

1. **Page Faults**: When a CPU or GPU accesses memory that isn't present locally
2. **Data Migration**: Moving memory pages between host and device
3. **Memory Residency**: Tracking where memory pages currently reside
4. **Access Counters**: Monitoring memory access patterns

On Pascal and newer GPUs, hardware page faults enable fine-grained migration, while older GPUs use a coarser approach.

## Code Walkthrough

### 1. Setting Up CUPTI for Unified Memory Profiling

First, we need to configure CUPTI to track Unified Memory events:

```cpp
void setupUnifiedMemoryProfiling()
{
    // Initialize CUPTI
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMORY));
    
    // Register callbacks for activity buffers
    CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));
    
    // Configure which Unified Memory events to track
    CUpti_ActivityUnifiedMemoryCounterConfig config[2];
    
    // Track CPU page faults
    config[0].scope = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_PROCESS_SINGLE_DEVICE;
    config[0].kind = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT;
    config[0].deviceId = 0;
    config[0].enable = 1;
    
    // Track GPU page faults
    config[1].scope = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_PROCESS_SINGLE_DEVICE;
    config[1].kind = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT_COUNT;
    config[1].deviceId = 0;
    config[1].enable = 1;
    
    // Configure the counters
    CUPTI_CALL(cuptiActivityConfigureUnifiedMemoryCounter(config, 2));
}
```

This code:
1. Enables CUPTI activity tracking for Unified Memory events
2. Sets up callbacks to process activity buffers
3. Configures specific counters for CPU and GPU page faults

### 2. Allocating Unified Memory

Next, we allocate memory using the Unified Memory API:

```cpp
void allocateUnifiedMemory(void **data, size_t size)
{
    // Allocate unified memory accessible from CPU and GPU
    RUNTIME_API_CALL(cudaMallocManaged(data, size));
    
    // Get device properties to check for hardware page fault support
    cudaDeviceProp prop;
    RUNTIME_API_CALL(cudaGetDeviceProperties(&prop, 0));
    
    // Check if the GPU supports hardware page faults
    bool hasPageFaultSupport = (prop.major >= 6);
    printf("GPU %s hardware page fault support\n", 
           hasPageFaultSupport ? "has" : "does not have");
}
```

This function:
1. Allocates memory using `cudaMallocManaged`
2. Checks if the GPU supports hardware page faults (Pascal or newer)

### 3. Testing Different Access Patterns

To demonstrate how access patterns affect Unified Memory performance, we implement several test cases:

```cpp
void testSequentialAccess(float *data, size_t size)
{
    printf("\nTesting Sequential Access (CPU then GPU):\n");
    
    // First, access data on the CPU
    for (size_t i = 0; i < size/sizeof(float); i++) {
        data[i] = i;
    }
    
    // Synchronize to ensure CPU operations complete
    RUNTIME_API_CALL(cudaDeviceSynchronize());
    
    // Then access on the GPU
    vectorAdd<<<(size/sizeof(float) + 255)/256, 256>>>(data, data, data, size/sizeof(float));
    
    // Wait for GPU to finish
    RUNTIME_API_CALL(cudaDeviceSynchronize());
}

void testPrefetchedAccess(float *data, size_t size)
{
    printf("\nTesting Prefetched Access (with cudaMemPrefetchAsync):\n");
    
    // Initialize data on the CPU
    for (size_t i = 0; i < size/sizeof(float); i++) {
        data[i] = i;
    }
    
    // Prefetch data to the GPU before use
    RUNTIME_API_CALL(cudaMemPrefetchAsync(data, size, 0));
    
    // Access on the GPU
    vectorAdd<<<(size/sizeof(float) + 255)/256, 256>>>(data, data, data, size/sizeof(float));
    
    // Wait for GPU to finish
    RUNTIME_API_CALL(cudaDeviceSynchronize());
}

void testConcurrentAccess(float *data, size_t size)
{
    printf("\nTesting Concurrent Access (CPU and GPU):\n");
    
    // Use memory advice to optimize for concurrent access
    RUNTIME_API_CALL(cudaMemAdvise(data, size, cudaMemAdviseSetAccessedBy, 0));
    
    // Launch GPU kernel
    vectorAdd<<<(size/sizeof(float) + 255)/256, 256>>>(data, data, data, size/sizeof(float));
    
    // While GPU is running, access part of the data from CPU
    for (size_t i = 0; i < size/(2*sizeof(float)); i++) {
        data[i] = i;
    }
    
    // Wait for all operations to complete
    RUNTIME_API_CALL(cudaDeviceSynchronize());
}
```

These functions demonstrate:
1. Sequential access from CPU then GPU
2. Prefetching data to GPU before use
3. Concurrent access from both CPU and GPU

### 4. Processing Unified Memory Events

When CUPTI collects activity records, we process them to extract Unified Memory information:

```cpp
void processUnifiedMemoryActivity(CUpti_Activity *record)
{
    switch (record->kind) {
    case CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER:
    {
        CUpti_ActivityUnifiedMemoryCounter *umcRecord = 
            (CUpti_ActivityUnifiedMemoryCounter *)record;
        
        // Process based on the counter kind
        switch (umcRecord->counterKind) {
        case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT:
            cpuPageFaults += umcRecord->value;
            printf("  CPU Page Faults: %llu\n", (unsigned long long)umcRecord->value);
            break;
            
        case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT_COUNT:
            gpuPageFaults += umcRecord->value;
            printf("  GPU Page Faults: %llu\n", (unsigned long long)umcRecord->value);
            break;
            
        case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD:
            bytesTransferredHtoD += umcRecord->value;
            printf("  Host to Device Transfers: %llu bytes\n", 
                   (unsigned long long)umcRecord->value);
            break;
            
        case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH:
            bytesTransferredDtoH += umcRecord->value;
            printf("  Device to Host Transfers: %llu bytes\n", 
                   (unsigned long long)umcRecord->value);
            break;
            
        // Process other counter kinds...
        }
        break;
    }
    
    case CUPTI_ACTIVITY_KIND_MEMORY:
    {
        CUpti_ActivityMemory *memoryRecord = (CUpti_ActivityMemory *)record;
        
        // Process memory allocation/deallocation events
        if (memoryRecord->memoryKind == CUPTI_ACTIVITY_MEMORY_KIND_MANAGED) {
            if (memoryRecord->allocationType == CUPTI_ACTIVITY_MEMORY_ALLOCATION_TYPE_MALLOC_MANAGED) {
                printf("  Unified Memory Allocation: %llu bytes at %p\n",
                       (unsigned long long)memoryRecord->bytes, 
                       (void *)memoryRecord->address);
            }
        }
        break;
    }
    
    // Process other activity kinds...
    }
}
```

This function:
1. Processes different types of Unified Memory activity records
2. Extracts information about page faults and data transfers
3. Tracks memory allocations and deallocations

### 5. Using Memory Advice

Memory advice can significantly improve Unified Memory performance:

```cpp
void applyMemoryAdvice(float *data, size_t size)
{
    // Advise that data will be accessed by the GPU
    RUNTIME_API_CALL(cudaMemAdvise(data, size, cudaMemAdviseSetAccessedBy, 0));
    
    // Advise that the first half should prefer to reside on the GPU
    RUNTIME_API_CALL(cudaMemAdvise(data, size/2, 
                                  cudaMemAdviseSetPreferredLocation, 0));
    
    // Advise that the second half should prefer to reside on the CPU
    RUNTIME_API_CALL(cudaMemAdvise(data + size/(2*sizeof(float)), size/2,
                                  cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
}
```

This function demonstrates:
1. Setting access hints with `cudaMemAdvise`
2. Specifying preferred memory locations for different data regions
3. Optimizing data placement based on access patterns

### 6. Sample Kernel

Here's a simple vector addition kernel that uses Unified Memory:

```cpp
__global__ void vectorAdd(float *a, float *b, float *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
```

This kernel:
1. Uses Unified Memory pointers directly
2. Accesses data without explicit transfers
3. Relies on the runtime to handle data migration

## Running the Tutorial

1. Build the sample:
   ```bash
   make
   ```

2. Run the Unified Memory profiling example:
   ```bash
   ./unified_memory
   ```

## Understanding the Output

When you run the Unified Memory profiling example, you'll see output similar to this:

```
Unified Memory Profiling Results:

Memory Allocation: 256MB
GPU has hardware page fault support

Testing Sequential Access (CPU then GPU):
  CPU Page Faults: 0
  GPU Page Faults: 65536
  Host to Device Transfers: 268435456 bytes
  Device to Host Transfers: 0 bytes
  Migration Time: 25.4ms

Testing Prefetched Access (with cudaMemPrefetchAsync):
  CPU Page Faults: 0
  GPU Page Faults: 0
  Host to Device Transfers: 268435456 bytes
  Device to Host Transfers: 0 bytes
  Migration Time: 15.2ms

Testing Concurrent Access (CPU and GPU):
  CPU Page Faults: 8192
  GPU Page Faults: 32768
  Host to Device Transfers: 268435456 bytes
  Device to Host Transfers: 134217728 bytes
  Migration Time: 42.8ms

Testing Memory Advice:
  CPU Page Faults: 4096
  GPU Page Faults: 16384
  Host to Device Transfers: 134217728 bytes
  Device to Host Transfers: 0 bytes
  Migration Time: 18.7ms
```

Let's analyze this output:

### Sequential Access

```
Testing Sequential Access (CPU then GPU):
  CPU Page Faults: 0
  GPU Page Faults: 65536
  Host to Device Transfers: 268435456 bytes
  Device to Host Transfers: 0 bytes
  Migration Time: 25.4ms
```

- **CPU Page Faults: 0** - No CPU page faults because the CPU initializes the memory first
- **GPU Page Faults: 65536** - Many GPU page faults (256MB / 4KB page size = 65536 pages)
- **Host to Device Transfers: 268435456 bytes** - All data (256MB) is transferred to the GPU
- **Migration Time: 25.4ms** - Time spent migrating data

This pattern shows on-demand migration triggered by GPU access.

### Prefetched Access

```
Testing Prefetched Access (with cudaMemPrefetchAsync):
  CPU Page Faults: 0
  GPU Page Faults: 0
  Host to Device Transfers: 268435456 bytes
  Device to Host Transfers: 0 bytes
  Migration Time: 15.2ms
```

- **GPU Page Faults: 0** - No GPU page faults because data was prefetched
- **Migration Time: 15.2ms** - Faster migration due to bulk prefetching

Prefetching eliminates page faults and reduces migration time by about 40%.

### Concurrent Access

```
Testing Concurrent Access (CPU and GPU):
  CPU Page Faults: 8192
  GPU Page Faults: 32768
  Host to Device Transfers: 268435456 bytes
  Device to Host Transfers: 134217728 bytes
  Migration Time: 42.8ms
```

- **CPU Page Faults: 8192** - CPU faults when accessing data that migrated to GPU
- **GPU Page Faults: 32768** - GPU faults when accessing data not yet migrated
- **Device to Host Transfers: 134217728 bytes** - Data moving back to CPU (128MB)
- **Migration Time: 42.8ms** - Higher migration time due to bidirectional transfers

Concurrent access causes thrashing as pages move back and forth between CPU and GPU.

### Memory Advice

```
Testing Memory Advice:
  CPU Page Faults: 4096
  GPU Page Faults: 16384
  Host to Device Transfers: 134217728 bytes
  Device to Host Transfers: 0 bytes
  Migration Time: 18.7ms
```

- **CPU Page Faults: 4096** - Reduced CPU faults due to memory advice
- **GPU Page Faults: 16384** - Reduced GPU faults due to memory advice
- **Host to Device Transfers: 134217728 bytes** - Only half the data (128MB) is transferred
- **Migration Time: 18.7ms** - Improved migration time due to better data placement

Memory advice significantly reduces page faults and migration overhead.

## Performance Insights

### Page Fault Overhead

Page faults introduce latency as execution must pause while data is migrated. The output shows:

- Sequential access: 65536 GPU page faults
- Prefetched access: 0 page faults
- Concurrent access: 40960 total page faults

Each page fault adds overhead, making prefetching crucial for performance.

### Data Transfer Volume

The amount of data transferred affects performance:

- Sequential access: 256MB (one-way)
- Prefetched access: 256MB (one-way)
- Concurrent access: 384MB (bidirectional)
- Memory advice: 128MB (one-way)

Memory advice reduces transfer volume by keeping data where it's accessed.

### Migration Time

Migration time directly impacts application performance:

- Sequential access: 25.4ms
- Prefetched access: 15.2ms (40% faster than sequential)
- Concurrent access: 42.8ms (68% slower than sequential)
- Memory advice: 18.7ms (26% faster than sequential)

Prefetching and memory advice significantly improve performance.

## Optimization Strategies

### 1. Prefetching

Use `cudaMemPrefetchAsync` to move data before it's needed:

```cpp
// Prefetch data to the GPU before kernel launch
cudaMemPrefetchAsync(data, size, deviceId);

// Launch kernel
myKernel<<<blocks, threads>>>(data);
```

This eliminates page faults and improves performance.

### 2. Memory Advice

Use `cudaMemAdvise` to guide data placement:

```cpp
// Set preferred location
cudaMemAdvise(data, size, cudaMemAdviseSetPreferredLocation, deviceId);

// Mark as accessed-by
cudaMemAdvise(data, size, cudaMemAdviseSetAccessedBy, deviceId);

// Set read-mostly hint
cudaMemAdvise(data, size, cudaMemAdviseSetReadMostly);
```

These hints help the runtime make better migration decisions.

### 3. Access Patterns

Design algorithms with Unified Memory-friendly access patterns:

- Process data in chunks to improve locality
- Avoid frequent switching between CPU and GPU access
- Consider using separate memory regions for CPU-only and GPU-only data

### 4. Hardware Considerations

Different GPUs handle Unified Memory differently:

- Pascal+ GPUs: Hardware page faults for fine-grained migration
- Pre-Pascal GPUs: Coarse-grained migration at kernel boundaries
- Multi-GPU systems: Consider NUMA effects and peer access

## Next Steps

- Profile your own applications with CUPTI Unified Memory tracking
- Experiment with different memory advice settings
- Compare performance with and without prefetching
- Analyze how your application's access patterns affect migration behavior 