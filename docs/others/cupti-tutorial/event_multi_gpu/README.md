# CUPTI Multi-GPU Event Profiling Tutorial

## Introduction

Many high-performance computing systems use multiple GPUs to accelerate computations. When profiling such systems, you need to collect performance data from all GPUs simultaneously without affecting their parallel execution. This tutorial demonstrates how to use CUPTI to collect performance events from multiple GPUs concurrently while maintaining their independent execution.

## What You'll Learn

- How to set up event collection on multiple GPUs
- Techniques for managing multiple CUDA contexts
- Methods to launch and profile kernels across GPUs without serialization
- Properly synchronizing and collecting results from all devices

## Understanding Multi-GPU Profiling Challenges

When profiling multi-GPU applications, several challenges arise:

1. **Context Management**: Each GPU requires its own CUDA context
2. **Avoiding Serialization**: Naive profiling approaches can serialize GPU execution
3. **Resource Coordination**: Event groups and resources must be managed per device
4. **Synchronization Points**: Proper synchronization is needed without blocking parallel execution

This tutorial shows how to address these challenges while maintaining the performance benefits of multi-GPU execution.

## Code Walkthrough

### 1. Detecting Available GPUs

First, we need to identify all available CUDA devices:

```cpp
// Get number of devices
int deviceCount = 0;
RUNTIME_API_CALL(cudaGetDeviceCount(&deviceCount));

// We need at least two devices for this sample
if (deviceCount < 2) {
    printf("This sample requires at least two CUDA capable devices, but only %d devices were found\n", deviceCount);
    return 0;
}

printf("Found %d devices\n", deviceCount);

// Get device names
for (int i = 0; i < deviceCount; i++) {
    cudaDeviceProp prop;
    RUNTIME_API_CALL(cudaGetDeviceProperties(&prop, i));
    printf("CUDA Device Name: %s\n", prop.name);
}
```

### 2. Setting Up Contexts and Event Collection

For each GPU, we create a separate context and set up event collection:

```cpp
// Create a context on each device
CUcontext context[MAX_DEVICES];
CUpti_EventGroup eventGroup[MAX_DEVICES];
CUpti_EventID eventId[MAX_DEVICES];
const char *eventName = "inst_executed"; // Default event

// Use command-line argument if provided
if (argc > 1) {
    eventName = argv[1];
}

// For each device, create a context and set up event collection
for (int i = 0; i < deviceCount; i++) {
    // Set the current device
    RUNTIME_API_CALL(cudaSetDevice(i));
    
    // Create a context for this device
    DRIVER_API_CALL(cuCtxCreate(&context[i], 0, i));
    
    // Get the event ID for the specified event
    CUPTI_CALL(cuptiEventGetIdFromName(i, eventName, &eventId[i]));
    
    // Create an event group for this device
    CUPTI_CALL(cuptiEventGroupCreate(context[i], &eventGroup[i], 0));
    
    // Add the event to the group
    CUPTI_CALL(cuptiEventGroupAddEvent(eventGroup[i], eventId[i]));
    
    // Set collection mode to kernel-level
    CUPTI_CALL(cuptiEventGroupSetAttribute(eventGroup[i], 
                                          CUPTI_EVENT_GROUP_ATTR_COLLECTION_MODE,
                                          sizeof(CUpti_EventCollectionMode), 
                                          &kernel_mode));
    
    // Enable the event group
    CUPTI_CALL(cuptiEventGroupEnable(eventGroup[i]));
}
```

Key aspects of this code:
1. We create a separate context for each GPU
2. We set up event collection for the same event across all devices
3. We use kernel mode collection to focus on kernel execution

### 3. Launching Kernels on All GPUs

Now we launch kernels on all GPUs without waiting for each to complete:

```cpp
// Allocate memory and launch kernels on each device without synchronizing between them
int *d_data[MAX_DEVICES];
size_t dataSize = sizeof(int) * ITERATIONS;

for (int i = 0; i < deviceCount; i++) {
    // Set current device and context
    RUNTIME_API_CALL(cudaSetDevice(i));
    DRIVER_API_CALL(cuCtxSetCurrent(context[i]));
    
    // Allocate memory on this device
    RUNTIME_API_CALL(cudaMalloc((void **)&d_data[i], dataSize));
    
    // Launch the kernel on this device
    dim3 threads(256);
    dim3 blocks((ITERATIONS + threads.x - 1) / threads.x);
    
    dummyKernel<<<blocks, threads>>>(d_data[i], ITERATIONS);
}
```

This is the critical part of the sample - we launch kernels on all GPUs without synchronizing between launches, allowing them to execute concurrently.

### 4. Synchronizing and Reading Event Values

After launching all kernels, we synchronize each device and read the event values:

```cpp
// Synchronize all devices and read event values
uint64_t eventValues[MAX_DEVICES];

for (int i = 0; i < deviceCount; i++) {
    // Set current device and context
    RUNTIME_API_CALL(cudaSetDevice(i));
    DRIVER_API_CALL(cuCtxSetCurrent(context[i]));
    
    // Synchronize the device to ensure kernel completion
    RUNTIME_API_CALL(cudaDeviceSynchronize());
    
    // Read the event value
    size_t valueSize = sizeof(uint64_t);
    CUPTI_CALL(cuptiEventGroupReadEvent(eventGroup[i],
                                      CUPTI_EVENT_READ_FLAG_NONE,
                                      eventId[i],
                                      &valueSize,
                                      &eventValues[i]));
    
    // Print the event value for this device
    printf("[%d] %s: %llu\n", i, eventName, (unsigned long long)eventValues[i]);
    
    // Clean up
    RUNTIME_API_CALL(cudaFree(d_data[i]));
    CUPTI_CALL(cuptiEventGroupDisable(eventGroup[i]));
    CUPTI_CALL(cuptiEventGroupDestroy(eventGroup[i]));
    DRIVER_API_CALL(cuCtxDestroy(context[i]));
}
```

Key aspects of this code:
1. We synchronize each device individually after all kernels are launched
2. We read event values for each device using its specific event group
3. We clean up resources for each device

### 5. The Test Kernel

The kernel used in this sample is a simple dummy kernel that performs a fixed number of iterations:

```cpp
__global__ void dummyKernel(int *data, int iterations)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < iterations) {
        // Do some work to generate events
        int value = 0;
        for (int i = 0; i < 100; i++) {
            value += i;
        }
        data[idx] = value;
    }
}
```

This kernel ensures each GPU has enough work to generate measurable event counts.

## Running the Tutorial

1. Build the sample:
   ```bash
   make
   ```

2. Run with the default event (instruction count):
   ```bash
   ./event_multi_gpu
   ```

3. Try a different event:
   ```bash
   ./event_multi_gpu branch
   ```

## Understanding the Output

When running the sample, you'll see output like:

```
Found 2 devices
CUDA Device Name: NVIDIA GeForce RTX 3080
CUDA Device Name: NVIDIA GeForce RTX 3070
[0] inst_executed: 4194304
[1] inst_executed: 4194304
```

This shows:
1. The number of CUDA devices detected
2. The name of each device
3. The event values collected from each device

In this example, both GPUs executed the same number of instructions because they ran identical kernels. In real applications, you might see different values based on workload distribution and GPU capabilities.

## Performance Considerations

### Non-Serialized Execution

The key benefit of this approach is non-serialized execution:

```
// Traditional approach (serialized):
for (int i = 0; i < deviceCount; i++) {
    cudaSetDevice(i);
    launchKernel();
    collectEvents();  // This forces synchronization before the next GPU starts
}

// Our approach (parallel):
for (int i = 0; i < deviceCount; i++) {
    cudaSetDevice(i);
    launchKernel();   // Launch on all GPUs without waiting
}
for (int i = 0; i < deviceCount; i++) {
    cudaSetDevice(i);
    collectEvents();  // Collect after all GPUs have started working
}
```

The parallel approach allows all GPUs to work simultaneously, giving a more accurate picture of multi-GPU performance.

### Context Switching

Proper context management is crucial for multi-GPU profiling:

1. **Setting the Device**: Use `cudaSetDevice()` to select which GPU to work with
2. **Setting the Context**: Use `cuCtxSetCurrent()` to activate the correct context
3. **Synchronizing**: Use `cudaDeviceSynchronize()` to wait for work completion on a specific device

## Advanced Applications

### Profiling Distributed Workloads

For applications that distribute work across GPUs:
1. Create different kernels or workloads for each GPU
2. Launch them in the same non-serialized manner
3. Compare event values to understand performance differences

### Identifying Load Imbalance

By comparing event values across GPUs:
1. Similar values indicate balanced workloads
2. Significant differences may indicate load imbalance
3. Use this information to optimize work distribution

## Next Steps

- Apply this technique to profile your own multi-GPU applications
- Collect different events to understand various aspects of performance
- Extend the sample to collect multiple events per GPU
- Create visualizations comparing performance across different GPUs 