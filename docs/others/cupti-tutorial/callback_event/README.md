# CUPTI Callback Event Tutorial

> The GitHub repo and complete tutorial is available at <https://github.com/eunomia-bpf/cupti-tutorial>.

## Introduction

Tracking GPU performance events is critical for optimizing CUDA applications. This tutorial demonstrates how to use CUPTI's callback mechanism to collect specific performance events during kernel execution. We'll focus on the "instructions executed" metric to show how many GPU instructions your kernels are running.

## What You'll Learn

- How to set up CUPTI callback functions for CUDA runtime API calls
- Creating and managing event groups for collecting performance data
- Techniques for collecting event data around kernel launches
- Interpreting event values for performance analysis

## Code Walkthrough

### 1. Event Collection Structure

The sample sets up two key data structures:

```cpp
// Stores the event group and event ID
typedef struct cupti_eventData_st {
  CUpti_EventGroup eventGroup;
  CUpti_EventID eventId;
} cupti_eventData;

// Stores event data and values collected by the callback
typedef struct RuntimeApiTrace_st {
  cupti_eventData *eventData;
  uint64_t eventVal;
} RuntimeApiTrace_t;
```

These structures maintain the event state and collected values throughout the application.

### 2. The Callback Function

The heart of the sample is the callback function that gets called when certain CUDA runtime API functions are invoked:

```cpp
void CUPTIAPI getEventValueCallback(void *userdata, CUpti_CallbackDomain domain,
                                    CUpti_CallbackId cbid, const CUpti_CallbackData *cbInfo)
{
  // Only process callbacks for kernel launches
  if ((cbid != CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020) &&
      (cbid != CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000)) {
    return;
  }

  RuntimeApiTrace_t *traceData = (RuntimeApiTrace_t*)userdata;
  
  // When entering the CUDA runtime function (before kernel launches)
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    // Synchronize device to ensure clean event collection
    cudaDeviceSynchronize();
    
    // Set collection mode to kernel-level
    cuptiSetEventCollectionMode(cbInfo->context, CUPTI_EVENT_COLLECTION_MODE_KERNEL);
    
    // Enable the event group to start collecting data
    cuptiEventGroupEnable(traceData->eventData->eventGroup);
  }
  
  // When exiting the CUDA runtime function (after kernel completes)
  if (cbInfo->callbackSite == CUPTI_API_EXIT) {
    // Determine how many instances of the event occurred
    uint32_t numInstances = 0;
    cuptiEventGroupGetAttribute(traceData->eventData->eventGroup, 
                               CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT, 
                               &valueSize, &numInstances);
    
    // Allocate space for event values
    uint64_t *values = (uint64_t *) malloc(sizeof(uint64_t) * numInstances);
    
    // Make sure kernel is done
    cudaDeviceSynchronize();
    
    // Read the event values
    cuptiEventGroupReadEvent(traceData->eventData->eventGroup, 
                            CUPTI_EVENT_READ_FLAG_NONE, 
                            traceData->eventData->eventId, 
                            &bytesRead, values);
    
    // Aggregate values across all instances
    traceData->eventVal = 0;
    for (i=0; i<numInstances; i++) {
      traceData->eventVal += values[i];
    }
    
    // Clean up
    free(values);
    cuptiEventGroupDisable(traceData->eventData->eventGroup);
  }
}
```

This function performs these key operations:
- On API entry (before kernel launch): Enables event collection
- On API exit (after kernel completion): Reads event values and disables collection

### 3. Setting Up the Event and Callback

```cpp
int main(int argc, char *argv[])
{
  // Default event to track (instructions executed)
  const char *eventName = EVENT_NAME;
  
  // Parse command line arguments for device and event
  if (argc > 2)
    eventName = argv[2];
  
  // Set up CUPTI subscriber
  cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)getEventValueCallback, &trace);
  
  // Enable callbacks for kernel launches
  cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API);
  
  // Find the event and create an event group
  cuptiEventGetIdFromName(device, eventName, &cuptiEvent.eventId);
  cuptiEventGroupCreate(context, &cuptiEvent.eventGroup, 0);
  cuptiEventGroupAddEvent(cuptiEvent.eventGroup, cuptiEvent.eventId);
  
  // Store the event data in our trace structure
  trace.eventData = &cuptiEvent;
  trace.eventVal = 0;
  
  // Run the vector addition kernel
  // ...
  
  // Display the event value
  displayEventVal(&trace, eventName);
  
  // Clean up
  cuptiEventGroupDestroy(cuptiEvent.eventGroup);
  cuptiUnsubscribe(subscriber);
  
  return 0;
}
```

The main function sets up the event collection system, runs the kernel, and displays the results.

### 4. The Test Kernel

The sample uses a simple vector addition kernel to demonstrate event collection:

```cpp
__global__ void VecAdd(const int* A, const int* B, int* C, int N)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N)
    C[i] = A[i] + B[i];
}
```

## Running the Sample

1. Build the sample:
   ```bash
   make
   ```

2. Run with default parameters (device 0, tracking "inst_executed"):
   ```bash
   ./callback_event
   ```

3. You can specify a different device or event:
   ```bash
   ./callback_event [device_num] [event_name]
   ```

   For example:
   ```bash
   ./callback_event 0 inst_executed_global_loads
   ```

## Understanding the Output

The sample produces output similar to:

```
CUDA Device Number: 0
CUDA Device Name: NVIDIA GeForce RTX 3080
Compute capability: 8.6
Event Name : inst_executed
Event Value : 2048000
```

This shows:
1. The device used for the test
2. The compute capability (architecture version)
3. The event being tracked ("inst_executed" = instructions executed)
4. The total number of instructions executed by the kernel

## Performance Insights

With event data, you can:
- Compare different implementations of the same algorithm
- Identify inefficient code paths that execute too many instructions
- Determine if your kernel is compute-bound (high instruction count) or memory-bound
- Track specific instruction types (loads, stores, etc.) by using different events

## Available Events

Different GPU architectures support different events. Some common events include:
- `inst_executed` - Total instructions executed
- `global_load` - Global memory loads
- `global_store` - Global memory stores
- `branch` - Branch instructions
- `divergent_branch` - Divergent branches (harmful for performance)

Use `cuptiDeviceEnumEvents` (as shown in the `cupti_query` sample) to discover all available events for your device.

## Next Steps

- Try different events to understand various aspects of kernel performance
- Modify the vector size to see how it affects the instruction count
- Use multiple events to build a more complete performance profile
- Explore CUPTI's other callback-related samples for more advanced profiling 