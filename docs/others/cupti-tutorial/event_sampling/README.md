# CUPTI Event Sampling Tutorial

## Introduction

When profiling CUDA applications, you often need to monitor performance metrics while your application is running. This tutorial demonstrates how to use CUPTI's event sampling capabilities to collect GPU performance data at regular intervals during kernel execution, giving you real-time insights into your application's behavior.

## What You'll Learn

- How to set up continuous event sampling on NVIDIA GPUs
- Techniques for monitoring events while kernels are running
- Creating a multi-threaded profiling system
- Interpreting sampled event data for performance analysis

## Understanding Event Sampling

Unlike one-time event collection that gives you a single value at the end of kernel execution, event sampling allows you to:

1. Monitor how events change over time during kernel execution
2. Detect performance variations and anomalies
3. Correlate GPU activity with specific phases of your algorithm
4. Observe the impact of dynamic workloads

## Code Walkthrough

### 1. Setting Up the Sampling Thread

The core of this sample is a dedicated sampling thread that collects event data while the main thread runs computations:

```cpp
static void *sampling_func(void *arg)
{
    SamplingInfo *info = (SamplingInfo *)arg;
    CUcontext context = info->context;
    CUdevice device = info->device;
    
    // Make this thread use the same CUDA context as the main thread
    cuCtxSetCurrent(context);
    
    // Set up the event we want to monitor
    CUpti_EventGroup eventGroup;
    CUpti_EventID eventId;
    
    // Get the event ID for the specified event (default is "inst_executed")
    CUPTI_CALL(cuptiEventGetIdFromName(device, info->eventName, &eventId));
    
    // Create an event group for the device
    CUPTI_CALL(cuptiEventGroupCreate(context, &eventGroup, 0));
    
    // Add the event to the group
    CUPTI_CALL(cuptiEventGroupAddEvent(eventGroup, eventId));
    
    // Set continuous collection mode (critical for sampling during execution)
    CUPTI_CALL(cuptiEventGroupSetAttribute(eventGroup, 
                                          CUPTI_EVENT_GROUP_ATTR_COLLECTION_MODE,
                                          sizeof(CUpti_EventCollectionMode), 
                                          &continuous));
    
    // Enable the event group
    CUPTI_CALL(cuptiEventGroupEnable(eventGroup));
    
    // Sample until the computation is done
    while (!info->terminate) {
        // Read the current event value
        size_t valueSize = sizeof(uint64_t);
        uint64_t eventValue = 0;
        
        CUPTI_CALL(cuptiEventGroupReadEvent(eventGroup,
                                           CUPTI_EVENT_READ_FLAG_NONE,
                                           eventId,
                                           &valueSize,
                                           &eventValue));
        
        // Print the current value
        printf("%s: %llu\n", info->eventName, (unsigned long long)eventValue);
        
        // Wait before taking the next sample
        millisleep(SAMPLE_PERIOD_MS);
    }
    
    // Cleanup
    CUPTI_CALL(cuptiEventGroupDisable(eventGroup));
    CUPTI_CALL(cuptiEventGroupDestroy(eventGroup));
    
    return NULL;
}
```

Key aspects of this code:

1. **Context Sharing**: The sampling thread uses the same CUDA context as the main thread
2. **Continuous Collection Mode**: Enables reading event values while kernels are running
3. **Regular Sampling**: Reads event values at fixed intervals (50ms by default)
4. **Non-blocking**: Sampling doesn't interrupt kernel execution

### 2. The Computation Thread (Main Thread)

The main thread runs the actual computation we want to profile:

```cpp
int main(int argc, char *argv[])
{
    // Initialize CUDA and get device/context
    CUdevice device;
    CUcontext context;
    
    // Initialize CUDA driver API
    DRIVER_API_CALL(cuInit(0));
    DRIVER_API_CALL(cuDeviceGet(&device, 0));
    DRIVER_API_CALL(cuCtxCreate(&context, 0, device));
    
    // Set up sampling information
    SamplingInfo samplingInfo;
    samplingInfo.device = device;
    samplingInfo.context = context;
    samplingInfo.terminate = 0;
    
    // Default to "inst_executed" or use command line argument
    if (argc > 1) {
        samplingInfo.eventName = argv[1];
    } else {
        samplingInfo.eventName = "inst_executed";
    }
    
    // Create and start the sampling thread
    pthread_t sampling_thread;
    pthread_create(&sampling_thread, NULL, sampling_func, &samplingInfo);
    
    // Allocate memory for vector addition
    float *d_A, *d_B, *d_C;
    float *h_A, *h_B, *h_C;
    size_t size = VECTOR_SIZE * sizeof(float);
    
    // Allocate and initialize host memory
    h_A = (float *)malloc(size);
    h_B = (float *)malloc(size);
    h_C = (float *)malloc(size);
    
    // Initialize vectors
    for (int i = 0; i < VECTOR_SIZE; i++) {
        h_A[i] = i * 1.0f;
        h_B[i] = i * 2.0f;
    }
    
    // Allocate device memory
    RUNTIME_API_CALL(cudaMalloc((void **)&d_A, size));
    RUNTIME_API_CALL(cudaMalloc((void **)&d_B, size));
    RUNTIME_API_CALL(cudaMalloc((void **)&d_C, size));
    
    // Copy host memory to device
    RUNTIME_API_CALL(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    RUNTIME_API_CALL(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
    
    // Launch kernel multiple times to give us time to sample
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((VECTOR_SIZE + threadsPerBlock.x - 1) / threadsPerBlock.x);
    
    for (int i = 0; i < ITERATIONS; i++) {
        vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, VECTOR_SIZE);
    }
    
    // Make sure all kernels are done
    RUNTIME_API_CALL(cudaDeviceSynchronize());
    
    // Signal sampling thread to terminate
    samplingInfo.terminate = 1;
    
    // Wait for sampling thread to finish
    pthread_join(sampling_thread, NULL);
    
    // Cleanup and exit
    free(h_A);
    free(h_B);
    free(h_C);
    RUNTIME_API_CALL(cudaFree(d_A));
    RUNTIME_API_CALL(cudaFree(d_B));
    RUNTIME_API_CALL(cudaFree(d_C));
    
    DRIVER_API_CALL(cuCtxDestroy(context));
    
    return 0;
}
```

Key aspects of this code:

1. **Long-running Workload**: The kernel is run 2000 times to ensure we have enough time to collect samples
2. **Thread Coordination**: The main thread signals the sampling thread when computation is complete
3. **Simple Test Kernel**: Uses vector addition as a test case for sampling

### 3. The Vector Addition Kernel

```cpp
__global__ void vecAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
        C[i] = A[i] + B[i];
}
```

This simple kernel adds two vectors together. While not computationally intensive, running it repeatedly gives us a consistent workload to monitor.

## Running the Tutorial

1. Build the sample:
   ```bash
   make
   ```

2. Run with the default event (instruction count):
   ```bash
   ./event_sampling
   ```

3. Try a different event:
   ```bash
   ./event_sampling branch
   ```

## Understanding the Output

When running with the default "inst_executed" event, you'll see output like:

```
inst_executed: 0
inst_executed: 25600000
inst_executed: 51200000
inst_executed: 76800000
inst_executed: 102400000
...
inst_executed: 4582400000
inst_executed: 4608000000
```

Each line represents:
1. The name of the event being sampled
2. The cumulative count of that event at the time of sampling

In this case, we're seeing the total number of instructions executed by the GPU, which increases steadily as our kernels run. The regular increments show that our workload is executing consistently over time.

## Available Events to Sample

Different GPUs support different events. Some common events you might want to sample include:

- `inst_executed`: Instructions executed
- `branch`: Branch instructions executed
- `divergent_branch`: Divergent branch instructions
- `active_cycles`: Cycles where at least one warp is active
- `active_warps`: Number of active warps per cycle
- `global_load`: Global memory load operations
- `global_store`: Global memory store operations
- `local_load`: Local memory load operations
- `local_store`: Local memory store operations

Use the `cupti_query` sample to discover all available events for your specific GPU.

## Practical Applications

### Performance Monitoring

Event sampling is particularly useful for:
1. **Long-running kernels**: Monitor progress and detect stalls
2. **Iterative algorithms**: Observe convergence behavior
3. **Dynamic workloads**: Detect performance variations over time

### Detecting Performance Anomalies

By watching how event counts change over time, you can detect:
1. **Sudden drops in instruction throughput**: May indicate resource contention
2. **Unexpected spikes in branch divergence**: Could signal inefficient warp execution
3. **Changes in memory access patterns**: Might reveal cache thrashing

## Advanced Techniques

### Sampling Multiple Events

To sample multiple events simultaneously:
1. Add multiple events to the event group
2. Read values for each event during sampling
3. Be aware that some events can't be collected together due to hardware limitations

### Correlating with Application Phases

To correlate samples with algorithm phases:
1. Add timestamps or phase markers to your output
2. Synchronize sampling with algorithm phase changes
3. Analyze how different phases affect GPU performance

## Next Steps

- Modify the sample to collect multiple events simultaneously
- Visualize the sampling data to better understand performance trends
- Apply event sampling to your own CUDA applications
- Experiment with different sampling rates to find the right balance between detail and overhead 