# CUPTI Query API Tutorial

## Introduction

Before you can effectively profile a CUDA application, you need to know what performance metrics and events are available on your GPU. The CUPTI Query API provides a way to discover and explore the profiling capabilities of your NVIDIA GPU. This tutorial demonstrates how to use this API to list available domains, events, and metrics.

## What You'll Learn

- How to query available event domains on a CUDA device
- Techniques for listing hardware counters (events) in each domain
- Methods to discover available performance metrics
- Understanding the relationships between domains, events, and metrics

## Understanding CUPTI's Profiling Hierarchy

CUPTI organizes GPU profiling capabilities in a hierarchical structure:

1. **Devices**: Your NVIDIA GPUs
2. **Domains**: Groups of related hardware counters on a device
3. **Events**: Raw hardware counters within a domain
4. **Metrics**: Derived measurements calculated from events

This hierarchy allows for organized access to the wide range of performance data available on modern GPUs.

## Code Walkthrough

### 1. Querying Available Devices

First, we need to identify the available CUDA devices:

```cpp
int deviceCount = 0;
CUPTI_CALL(cuptiDeviceGetNumDevices(&deviceCount));
printf("There are %d devices\n", deviceCount);

// Get compute capability for the device
CUdevice device = 0; // Default to first device
CUresult err = cuDeviceGet(&device, dev);
if (err != CUDA_SUCCESS) {
    printf("Error: cuDeviceGet failed with error %d\n", err);
    return;
}

int major = 0, minor = 0;
err = cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
err = cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
printf("Compute capability: %d.%d\n", major, minor);
```

### 2. Enumerating Event Domains

Event domains group related hardware counters. We can list all available domains on a device:

```cpp
void enumEventDomains(CUdevice device)
{
    // Get the number of domains
    uint32_t numDomains = 0;
    CUPTI_CALL(cuptiDeviceGetNumEventDomains(device, &numDomains));
    printf("Device %d has %d domains\n\n", device, numDomains);
    
    if (numDomains == 0) {
        printf("No domains found on device %d\n", device);
        return;
    }
    
    // Allocate space to hold domain IDs
    CUpti_EventDomainID *domainIds = (CUpti_EventDomainID *)malloc(numDomains * sizeof(CUpti_EventDomainID));
    if (domainIds == NULL) {
        printf("Failed to allocate memory for domain IDs\n");
        return;
    }
    
    // Get the domain IDs
    CUPTI_CALL(cuptiDeviceEnumEventDomains(device, &numDomains, domainIds));
    
    // For each domain, print information about it
    for (int i = 0; i < numDomains; i++) {
        char name[CUPTI_MAX_NAME_LENGTH];
        size_t size = CUPTI_MAX_NAME_LENGTH;
        
        // Get domain name
        CUPTI_CALL(cuptiEventDomainGetAttribute(domainIds[i], 
                                              CUPTI_EVENT_DOMAIN_ATTR_NAME,
                                              &size, name));
        
        // Get profiled instance count
        uint32_t profiled = 0;
        size = sizeof(profiled);
        CUPTI_CALL(cuptiDeviceGetEventDomainAttribute(device, domainIds[i],
                                                    CUPTI_EVENT_DOMAIN_ATTR_INSTANCE_COUNT,
                                                    &size, &profiled));
        
        // Get total instance count
        uint32_t total = 0;
        size = sizeof(total);
        CUPTI_CALL(cuptiDeviceGetEventDomainAttribute(device, domainIds[i],
                                                    CUPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT,
                                                    &size, &total));
        
        // Get collection method
        CUpti_EventCollectionMethod method;
        size = sizeof(method);
        CUPTI_CALL(cuptiEventDomainGetAttribute(domainIds[i],
                                              CUPTI_EVENT_DOMAIN_ATTR_COLLECTION_METHOD,
                                              &size, &method));
        
        printf("Domain# %d\n", i+1);
        printf("Id         = %d\n", domainIds[i]);
        printf("Name       = %s\n", name);
        printf("Profiled instance count = %u\n", profiled);
        printf("Total instance count = %u\n", total);
        printf("Event collection method = %s\n\n", 
               getCollectionMethodString(method));
    }
    
    free(domainIds);
}
```

The collection method indicates how events in this domain are collected:
- **PM**: Performance Monitor - Hardware counters
- **SM**: Software Monitor - Software counters
- **Instrumented**: Instrumentation-based collection
- **NVLINK_TC**: NVLink Traffic Counters

### 3. Listing Events in a Domain

Once we have a domain ID, we can list all events available in that domain:

```cpp
void enumEvents(CUdevice device, CUpti_EventDomainID domainId)
{
    // Get number of events in the domain
    uint32_t numEvents = 0;
    CUPTI_CALL(cuptiEventDomainGetNumEvents(domainId, &numEvents));
    printf("Domain %d has %d events\n\n", domainId, numEvents);
    
    if (numEvents == 0) {
        printf("No events found in domain %d\n", domainId);
        return;
    }
    
    // Allocate space to hold event IDs
    CUpti_EventID *eventIds = (CUpti_EventID *)malloc(numEvents * sizeof(CUpti_EventID));
    if (eventIds == NULL) {
        printf("Failed to allocate memory for event IDs\n");
        return;
    }
    
    // Get the event IDs
    CUPTI_CALL(cuptiEventDomainEnumEvents(domainId, &numEvents, eventIds));
    
    // For each event, print information about it
    for (int i = 0; i < numEvents; i++) {
        char name[CUPTI_MAX_NAME_LENGTH];
        size_t size = CUPTI_MAX_NAME_LENGTH;
        
        // Get event name
        CUPTI_CALL(cuptiEventGetAttribute(eventIds[i], 
                                        CUPTI_EVENT_ATTR_NAME,
                                        &size, name));
        
        // Get event description
        char desc[CUPTI_MAX_NAME_LENGTH];
        size = CUPTI_MAX_NAME_LENGTH;
        CUPTI_CALL(cuptiEventGetAttribute(eventIds[i],
                                        CUPTI_EVENT_ATTR_SHORT_DESCRIPTION,
                                        &size, desc));
        
        // Get event category
        CUpti_EventCategory category;
        size = sizeof(category);
        CUPTI_CALL(cuptiEventGetAttribute(eventIds[i],
                                        CUPTI_EVENT_ATTR_CATEGORY,
                                        &size, &category));
        
        printf("Event# %d\n", i+1);
        printf("Id         = %d\n", eventIds[i]);
        printf("Name       = %s\n", name);
        printf("Description= %s\n", desc);
        printf("Category   = %s\n\n", 
               getEventCategoryString(category));
    }
    
    free(eventIds);
}
```

Events are categorized into different types:
- **Instruction**: Related to instruction execution
- **Memory**: Related to memory operations
- **Cache**: Related to cache operations
- **Profile Trigger**: Used for profiling triggers

### 4. Discovering Available Metrics

Metrics are derived measurements calculated from one or more events:

```cpp
void enumMetrics(CUdevice device)
{
    // Get number of metrics for the device
    uint32_t numMetrics = 0;
    CUPTI_CALL(cuptiDeviceGetNumMetrics(device, &numMetrics));
    printf("Device %d has %d metrics\n\n", device, numMetrics);
    
    if (numMetrics == 0) {
        printf("No metrics found for device %d\n", device);
        return;
    }
    
    // Allocate space to hold metric IDs
    CUpti_MetricID *metricIds = (CUpti_MetricID *)malloc(numMetrics * sizeof(CUpti_MetricID));
    if (metricIds == NULL) {
        printf("Failed to allocate memory for metric IDs\n");
        return;
    }
    
    // Get the metric IDs
    CUPTI_CALL(cuptiDeviceEnumMetrics(device, &numMetrics, metricIds));
    
    // For each metric, print information about it
    for (int i = 0; i < numMetrics; i++) {
        char name[CUPTI_MAX_NAME_LENGTH];
        size_t size = CUPTI_MAX_NAME_LENGTH;
        
        // Get metric name
        CUPTI_CALL(cuptiMetricGetAttribute(metricIds[i], 
                                         CUPTI_METRIC_ATTR_NAME,
                                         &size, name));
        
        // Get metric description
        char desc[CUPTI_MAX_NAME_LENGTH];
        size = CUPTI_MAX_NAME_LENGTH;
        CUPTI_CALL(cuptiMetricGetAttribute(metricIds[i],
                                         CUPTI_METRIC_ATTR_SHORT_DESCRIPTION,
                                         &size, desc));
        
        printf("Metric# %d\n", i+1);
        printf("Id         = %d\n", metricIds[i]);
        printf("Name       = %s\n", name);
        printf("Description= %s\n\n", desc);
    }
    
    free(metricIds);
}
```

## Running the Tutorial

### Command Line Options

The sample supports these command line options:

```
-help                                  : displays help message
-device <dev_id> -getdomains           : displays supported domains for specified device
-device <dev_id> -getmetrics           : displays supported metrics for specified device
-device <dev_id> -domain <domain_id> -getevents : displays supported events for specified domain and device
```

### Step-by-Step Usage

1. Build the sample:
   ```bash
   make
   ```

2. List all domains on device 0:
   ```bash
   ./cupti_query -device 0 -getdomains
   ```

3. List all events in domain 0 on device 0:
   ```bash
   ./cupti_query -device 0 -domain 0 -getevents
   ```

4. List all metrics on device 0:
   ```bash
   ./cupti_query -device 0 -getmetrics
   ```

## Understanding the Output

### Domain Information

```
Domain# 1
Id         = 0
Name       = CUPTI_DOMAIN_0
Profiled instance count = 1
Total instance count = 1
Event collection method = CUPTI_EVENT_COLLECTION_METHOD_PM
```

This shows:
- The domain ID (0)
- The domain name
- The number of instances that can be profiled simultaneously
- The total number of instances in the hardware
- The method used to collect events in this domain

### Event Information

```
Event# 1
Id         = 1
Name       = active_warps
Description= Number of active warps per cycle
Category   = CUPTI_EVENT_CATEGORY_INSTRUCTION
```

This shows:
- The event ID (1)
- The event name (active_warps)
- A description of what the event measures
- The category of the event (instruction-related)

### Metric Information

```
Metric# 1
Id         = 1
Name       = achieved_occupancy
Description= Ratio of active warps to maximum supported warps per multiprocessor
```

This shows:
- The metric ID (1)
- The metric name (achieved_occupancy)
- A description of what the metric measures

## Practical Applications

### Finding Relevant Metrics for Performance Analysis

When optimizing a CUDA application, you might be interested in specific aspects of performance:

1. **Memory Bandwidth**: Look for metrics like `dram_read_throughput` or `dram_write_throughput`
2. **Compute Utilization**: Look for metrics like `sm_efficiency` or `achieved_occupancy`
3. **Cache Performance**: Look for metrics like `l2_hit_rate` or `tex_cache_hit_rate`

### Using Events vs. Metrics

- **Events**: Raw hardware counters, useful for low-level analysis
- **Metrics**: Derived measurements, easier to interpret for performance analysis

## Next Steps

- Use the events and metrics discovered with this tool in your profiling applications
- Combine this knowledge with other CUPTI samples like `callback_event` to collect specific metrics
- Create custom metrics by combining events in meaningful ways
- Explore how different GPU architectures offer different sets of events and metrics 