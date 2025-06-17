# CUPTI Performance Metrics Tutorial

## Introduction

Understanding GPU performance requires more than just raw event counts - it requires meaningful metrics that provide insights into how efficiently your code is running. This tutorial demonstrates how to use CUPTI callbacks to collect and calculate performance metrics during CUDA kernel execution, giving you powerful insights into your application's GPU utilization.

## What You'll Learn

- How to collect complex performance metrics during kernel execution
- Techniques for handling metrics that require multiple passes
- Converting raw event counts into meaningful performance indicators
- Interpreting metrics to identify optimization opportunities

## Understanding Performance Metrics

Performance metrics in CUDA provide high-level insights by combining multiple hardware events. For example:

- **IPC (Instructions Per Cycle)**: Measures computational efficiency
- **Memory Throughput**: Measures memory bandwidth utilization
- **Warp Execution Efficiency**: Measures thread utilization within warps

These metrics are more intuitive than raw event counts and directly relate to optimization strategies.

## Code Walkthrough

### 1. Setting Up Metric Collection

First, we need to identify which events are required for our target metric:

```cpp
int main(int argc, char *argv[])
{
    // Default to "ipc" (instructions per cycle) or use command line argument
    const char *metricName = "ipc";
    if (argc > 1) {
        metricName = argv[1];
    }
    
    // Initialize CUDA
    RUNTIME_API_CALL(cudaSetDevice(0));
    
    // Get device properties
    CUdevice device = 0;
    DRIVER_API_CALL(cuDeviceGet(&device, 0));
    
    // Get the metric ID for the requested metric
    CUpti_MetricID metricId;
    CUPTI_CALL(cuptiMetricGetIdFromName(device, metricName, &metricId));
    
    // Determine the events needed for this metric
    uint32_t numEvents = 0;
    CUPTI_CALL(cuptiMetricGetNumEvents(metricId, &numEvents));
    
    // Allocate space for event IDs
    CUpti_EventID *eventIds = (CUpti_EventID *)malloc(numEvents * sizeof(CUpti_EventID));
    CUPTI_CALL(cuptiMetricEnumEvents(metricId, &numEvents, eventIds));
    
    // Determine how many passes are needed to collect all events
    MetricData_t metricData;
    metricData.device = device;
    metricData.eventIdArray = eventIds;
    metricData.numEvents = numEvents;
    metricData.eventValueArray = (uint64_t *)calloc(numEvents, sizeof(uint64_t));
    metricData.eventIdx = 0;
    
    // Create event groups for each pass
    createEventGroups(&metricData);
    
    printf("Collecting events for metric %s\n", metricName);
}
```

This code:
1. Identifies which metric to collect (default is "ipc")
2. Gets the list of events required for that metric
3. Sets up data structures to hold event values
4. Creates event groups for collecting the events

### 2. Creating Event Groups

Some events can't be collected simultaneously due to hardware limitations, so we need to organize them into compatible groups:

```cpp
void createEventGroups(MetricData_t *metricData)
{
    CUcontext context = NULL;
    DRIVER_API_CALL(cuCtxGetCurrent(&context));
    
    // Get number of event domains on the device
    uint32_t numDomains = 0;
    CUPTI_CALL(cuptiDeviceGetNumEventDomains(metricData->device, &numDomains));
    
    // Get the event domains
    CUpti_EventDomainID *domainIds = (CUpti_EventDomainID *)malloc(numDomains * sizeof(CUpti_EventDomainID));
    CUPTI_CALL(cuptiDeviceEnumEventDomains(metricData->device, &numDomains, domainIds));
    
    // For each event, find its domain and available instances
    for (int i = 0; i < metricData->numEvents; i++) {
        CUpti_EventDomainID domainId;
        CUPTI_CALL(cuptiEventGetAttribute(metricData->eventIdArray[i], 
                                        CUPTI_EVENT_ATTR_DOMAIN, 
                                        &domainId));
        
        // Find this domain in our list
        int domainIndex = -1;
        for (int j = 0; j < numDomains; j++) {
            if (domainId == domainIds[j]) {
                domainIndex = j;
                break;
            }
        }
        
        // If this is a new event group or the event can't go in the current group,
        // create a new event group
        if (metricData->numEventGroups == 0 || 
            !canAddEventToGroup(metricData, metricData->eventIdArray[i])) {
            
            // Create a new event group
            CUpti_EventGroup eventGroup;
            CUPTI_CALL(cuptiEventGroupCreate(context, &eventGroup, 0));
            
            // Add the event to the group
            CUPTI_CALL(cuptiEventGroupAddEvent(eventGroup, metricData->eventIdArray[i]));
            
            // Store the event group
            metricData->eventGroups[metricData->numEventGroups++] = eventGroup;
        }
        else {
            // Add the event to the existing group
            CUPTI_CALL(cuptiEventGroupAddEvent(
                metricData->eventGroups[metricData->numEventGroups-1], 
                metricData->eventIdArray[i]));
        }
    }
    
    free(domainIds);
}
```

This function:
1. Gets all event domains on the device
2. For each event, determines its domain
3. Creates event groups that are compatible with hardware limitations
4. May create multiple groups if events can't be collected simultaneously

### 3. Setting Up the Callback Function

Now we register a callback that will be called when kernels are launched:

```cpp
// Subscribe to CUPTI callbacks
CUpti_SubscriberHandle subscriber;
CUPTI_CALL(cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)getMetricValueCallback, &metricData));

// Enable callbacks for CUDA runtime
CUPTI_CALL(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API));
```

### 4. The Callback Function

The heart of the sample is the callback function that collects event data:

```cpp
void CUPTIAPI getMetricValueCallback(void *userdata, CUpti_CallbackDomain domain,
                                    CUpti_CallbackId cbid, const CUpti_CallbackData *cbInfo)
{
    MetricData_t *metricData = (MetricData_t*)userdata;
    
    // Only process runtime API callbacks for kernel launches
    if (domain != CUPTI_CB_DOMAIN_RUNTIME_API ||
        (cbid != CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020 &&
         cbid != CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000)) {
        return;
    }
    
    // Check if we're entering or exiting the function
    if (cbInfo->callbackSite == CUPTI_API_ENTER) {
        // We're about to launch a kernel
        
        // Enable the event groups for this pass
        for (int i = 0; i < metricData->numEventGroups; i++) {
            CUPTI_CALL(cuptiEventGroupEnable(metricData->eventGroups[i]));
        }
        
        // Set the collection mode to collect only during kernel execution
        CUpti_EventCollectionMode mode = CUPTI_EVENT_COLLECTION_MODE_KERNEL;
        for (int i = 0; i < metricData->numEventGroups; i++) {
            CUPTI_CALL(cuptiEventGroupSetAttribute(metricData->eventGroups[i],
                                                 CUPTI_EVENT_GROUP_ATTR_COLLECTION_MODE,
                                                 sizeof(mode), &mode));
        }
    }
    else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
        // The kernel has completed
        
        // Make sure all work is done
        RUNTIME_API_CALL(cudaDeviceSynchronize());
        
        // For each event group, read and normalize the event values
        for (int i = 0; i < metricData->numEventGroups; i++) {
            CUpti_EventGroup group = metricData->eventGroups[i];
            
            // Get the number of events in this group
            uint32_t numEvents = 0;
            CUPTI_CALL(cuptiEventGroupGetAttribute(group, 
                                                 CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS,
                                                 &numEvents));
            
            // Get the event IDs in this group
            CUpti_EventID *eventIds = (CUpti_EventID *)malloc(numEvents * sizeof(CUpti_EventID));
            CUPTI_CALL(cuptiEventGroupGetAttribute(group, 
                                                 CUPTI_EVENT_GROUP_ATTR_EVENTS,
                                                 eventIds));
            
            // Get the number of instances for this group
            uint32_t numInstances = 0;
            CUPTI_CALL(cuptiEventGroupGetAttribute(group, 
                                                 CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT,
                                                 &numInstances));
            
            // Read the event values for each event in this group
            for (int j = 0; j < numEvents; j++) {
                // Find the index of this event in our global list
                int index = -1;
                for (int k = 0; k < metricData->numEvents; k++) {
                    if (eventIds[j] == metricData->eventIdArray[k]) {
                        index = k;
                        break;
                    }
                }
                
                if (index != -1) {
                    // Read the event value
                    uint64_t value = 0;
                    CUPTI_CALL(cuptiEventGroupReadEvent(group,
                                                      CUPTI_EVENT_READ_FLAG_NONE,
                                                      eventIds[j],
                                                      &value));
                    
                    // Store the value
                    metricData->eventValueArray[index] = value;
                    
                    // Get the event name for display
                    char eventName[128];
                    CUPTI_CALL(cuptiEventGetAttribute(eventIds[j],
                                                    CUPTI_EVENT_ATTR_NAME,
                                                    eventName));
                    
                    // Print the raw and normalized values
                    printf("\t%s = %llu (%llu)\n", eventName, 
                           (unsigned long long)value, 
                           (unsigned long long)metricData->eventValueArray[index]);
                    
                    // Normalize the value by the number of instances
                    uint64_t normalized = value * numInstances;
                    printf("\t%s (normalized) (%llu * %u) / %u = %llu\n", 
                           eventName, 
                           (unsigned long long)value,
                           numInstances,
                           numInstances,
                           (unsigned long long)normalized);
                }
            }
            
            free(eventIds);
        }
        
        // Disable the event groups
        for (int i = 0; i < metricData->numEventGroups; i++) {
            CUPTI_CALL(cuptiEventGroupDisable(metricData->eventGroups[i]));
        }
        
        // Calculate the metric value from the collected events
        calculateMetricValue(metricData);
    }
}
```

This function:
1. Enables event collection when a kernel is about to launch
2. Reads event values after the kernel completes
3. Normalizes event values based on the number of instances
4. Displays raw and normalized values for each event

### 5. Calculating the Metric Value

After collecting all events, we calculate the actual metric value:

```cpp
void calculateMetricValue(MetricData_t *metricData)
{
    // Get the metric ID
    CUpti_MetricID metricId;
    CUPTI_CALL(cuptiMetricGetIdFromName(metricData->device, "ipc", &metricId));
    
    // Calculate the metric value from the collected events
    double metricValue = 0.0;
    CUPTI_CALL(cuptiMetricGetValue(metricData->device, 
                                  metricId,
                                  metricData->numEvents * sizeof(CUpti_EventID),
                                  metricData->eventIdArray,
                                  metricData->numEvents * sizeof(uint64_t),
                                  metricData->eventValueArray,
                                  0,
                                  &metricValue));
    
    // Get the metric name
    char metricName[128];
    CUPTI_CALL(cuptiMetricGetAttribute(metricId, 
                                      CUPTI_METRIC_ATTR_NAME,
                                      metricName));
    
    // Print the metric value
    printf("\nMetric %s\n", metricName);
    printf("\tValue: %f\n", metricValue);
}
```

This function:
1. Takes the raw event values we collected
2. Uses CUPTI to calculate the metric value
3. Displays the final metric value

### 6. The Test Kernel

The sample uses a simple vector addition kernel to demonstrate metric collection:

```cpp
__global__ void VecAdd(const int* A, const int* B, int* C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        // Do some extra work to generate more instructions
        int temp = 0;
        for (int j = 0; j < 100; j++) {
            temp += A[i] + B[i];
        }
        C[i] = temp;
    }
}
```

## Running the Tutorial

1. Build the sample:
   ```bash
   make
   ```

2. Run with the default metric (instructions per cycle):
   ```bash
   ./callback_metric
   ```

3. Try a different metric:
   ```bash
   ./callback_metric achieved_occupancy
   ```

## Understanding the Output

When running with the default "ipc" metric, you'll see output like:

```
Collecting events for metric ipc
Pass 1/1 collecting 2 events using 1 event group
        inst_executed = 51200 (51200)
        inst_executed (normalized) (51200 * 1) / 1 = 51200
        active_cycles = 25600 (25600)
        active_cycles (normalized) (25600 * 1) / 1 = 25600

Metric ipc
        Value: 2.000000
```

Let's decode this output:

1. **Events Collected**: The "ipc" metric requires two events - instructions executed and active cycles
2. **Raw Values**: The actual hardware counter values read from the GPU
3. **Normalized Values**: Adjusted values accounting for all instances of the counters
4. **Metric Value**: The calculated IPC value (2.0), meaning on average 2 instructions are executed per clock cycle

## Interpreting Common Metrics

### Instructions Per Cycle (IPC)

- **What it measures**: Computational efficiency
- **Ideal range**: Varies by architecture, but higher is better
- **Optimization hints**:
  - Low IPC (<1): May indicate memory bottlenecks or thread divergence
  - High IPC (>2): Good instruction-level parallelism

### Achieved Occupancy

- **What it measures**: Ratio of active warps to maximum possible warps
- **Ideal range**: Closer to 1.0 is better
- **Optimization hints**:
  - Low occupancy: Consider reducing register usage or shared memory
  - High occupancy doesn't guarantee better performance but provides more latency hiding

### Memory Throughput

- **What it measures**: Memory bandwidth utilization
- **Ideal range**: Closer to peak bandwidth is better
- **Optimization hints**:
  - Low throughput: Consider memory access patterns, coalescing, or caching
  - High throughput with low IPC: Application is memory-bound

## Advanced Usage

### Collecting Multiple Metrics

To collect multiple metrics:
1. Run the application multiple times, each with a different metric
2. Or modify the code to collect all required events and calculate multiple metrics

### Multi-Pass Collection

Some metrics require events that can't be collected simultaneously. The sample handles this by:
1. Determining which events can be collected together
2. Creating multiple event groups if needed
3. Running the kernel multiple times if necessary

## Next Steps

- Try collecting different metrics to understand various aspects of your kernels
- Apply metric collection to your own CUDA applications
- Use metrics to identify optimization opportunities
- Combine multiple metrics to get a complete performance picture 