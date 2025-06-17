# CUPTI NVLink Bandwidth Monitoring Tutorial

> The GitHub repo and complete tutorial is available at <https://github.com/eunomia-bpf/cupti-tutorial>.

## Introduction

In multi-GPU systems, the communication bandwidth between GPUs can significantly impact application performance. NVLink is NVIDIA's high-speed GPU interconnect technology that provides much higher bandwidth than traditional PCIe connections. This tutorial demonstrates how to use CUPTI to detect NVLink connections and monitor data transfer rates between GPUs, helping you optimize multi-GPU applications.

## What You'll Learn

- How to detect and identify NVLink connections in a multi-GPU system
- Techniques for measuring NVLink bandwidth utilization
- Using CUPTI metrics to monitor NVLink traffic in real-time
- Optimizing data transfers between GPUs using NVLink

## Understanding NVLink

NVLink is a high-bandwidth interconnect technology that enables direct communication between GPUs at rates significantly higher than PCIe:

- **Bandwidth**: Up to 25-50 GB/s per direction per NVLink connection (depending on GPU generation)
- **Topology**: GPUs can be connected in various configurations (mesh, hybrid cube-mesh, etc.)
- **Scalability**: Multiple NVLink connections can be used between GPU pairs to increase bandwidth
- **Bidirectional**: Simultaneous data transfer in both directions

## Code Walkthrough

### 1. Detecting NVLink Connections

First, we need to identify which GPUs are connected via NVLink:

```cpp
void detectNVLinkConnections()
{
    int deviceCount = 0;
    RUNTIME_API_CALL(cudaGetDeviceCount(&deviceCount));
    
    printf("Detecting NVLink connections among %d GPUs...\n", deviceCount);
    
    // Create a matrix to track connections
    int connections[MAX_DEVICES][MAX_DEVICES] = {0};
    
    // For each device pair, check if they're connected by NVLink
    for (int i = 0; i < deviceCount; i++) {
        for (int j = i + 1; j < deviceCount; j++) {
            // Get the number of NVLinks between these GPUs
            int nvlinkStatus = 0;
            CUdevice device1, device2;
            
            DRIVER_API_CALL(cuDeviceGet(&device1, i));
            DRIVER_API_CALL(cuDeviceGet(&device2, j));
            
            // Check if the devices are connected by NVLink
            DRIVER_API_CALL(cuDeviceGetP2PAttribute(&nvlinkStatus, 
                                                 CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED,
                                                 device1, device2));
            
            // If NVLink is present, determine how many links
            if (nvlinkStatus) {
                int numLinks = 0;
                DRIVER_API_CALL(cuDeviceGetP2PAttribute(&numLinks,
                                                     CU_DEVICE_P2P_ATTRIBUTE_CUDA_ARRAY_ACCESS_SUPPORTED,
                                                     device1, device2));
                
                connections[i][j] = numLinks;
                connections[j][i] = numLinks; // Matrix is symmetric
                
                printf("  GPU %d <-> GPU %d: %d NVLink connections\n", i, j, numLinks);
            }
        }
    }
}
```

This function:
1. Enumerates all CUDA devices in the system
2. For each pair of devices, checks if they're connected by NVLink
3. Determines the number of NVLink connections between each pair
4. Builds a connection matrix representing the NVLink topology

### 2. Setting Up CUPTI Metrics for NVLink Monitoring

To monitor NVLink bandwidth, we need to set up the appropriate CUPTI metrics:

```cpp
void setupNVLinkMetrics(NVLinkMetrics *metrics, int deviceId)
{
    CUdevice device;
    DRIVER_API_CALL(cuDeviceGet(&device, deviceId));
    
    // Get the metric IDs for NVLink transmit and receive bandwidth
    CUPTI_CALL(cuptiMetricGetIdFromName(device, "nvlink_total_data_transmitted", 
                                      &metrics->transmitMetricId));
    CUPTI_CALL(cuptiMetricGetIdFromName(device, "nvlink_total_data_received",
                                      &metrics->receiveMetricId));
    
    // Create event groups for these metrics
    CUcontext context;
    DRIVER_API_CALL(cuCtxCreate(&context, 0, device));
    metrics->context = context;
    
    // Get the events required for the transmit metric
    uint32_t numTransmitEvents = 0;
    CUPTI_CALL(cuptiMetricGetNumEvents(metrics->transmitMetricId, &numTransmitEvents));
    
    CUpti_EventID *transmitEvents = (CUpti_EventID *)malloc(numTransmitEvents * sizeof(CUpti_EventID));
    CUPTI_CALL(cuptiMetricEnumEvents(metrics->transmitMetricId, &numTransmitEvents, transmitEvents));
    
    // Create an event group for transmit events
    CUPTI_CALL(cuptiEventGroupCreate(context, &metrics->transmitEventGroup, 0));
    
    // Add each event to the group
    for (uint32_t i = 0; i < numTransmitEvents; i++) {
        CUPTI_CALL(cuptiEventGroupAddEvent(metrics->transmitEventGroup, transmitEvents[i]));
    }
    
    // Similarly set up the receive metric events
    // ...
    
    // Enable the event groups
    CUPTI_CALL(cuptiEventGroupEnable(metrics->transmitEventGroup));
    CUPTI_CALL(cuptiEventGroupEnable(metrics->receiveEventGroup));
    
    free(transmitEvents);
}
```

This function:
1. Gets the metric IDs for NVLink transmit and receive bandwidth
2. Identifies the events required for these metrics
3. Creates event groups for collecting these events
4. Enables the event groups for monitoring

### 3. Running a Bandwidth Test

To measure NVLink bandwidth, we'll perform memory transfers between connected GPUs:

```cpp
void runBandwidthTest(int srcDevice, int dstDevice, size_t dataSize, NVLinkMetrics *metrics)
{
    printf("Transfer GPU %d -> GPU %d:\n", srcDevice, dstDevice);
    
    // Set source device
    RUNTIME_API_CALL(cudaSetDevice(srcDevice));
    
    // Allocate memory on source device
    void *srcData;
    RUNTIME_API_CALL(cudaMalloc(&srcData, dataSize));
    
    // Initialize source data
    RUNTIME_API_CALL(cudaMemset(srcData, 0xA5, dataSize));
    
    // Set destination device
    RUNTIME_API_CALL(cudaSetDevice(dstDevice));
    
    // Allocate memory on destination device
    void *dstData;
    RUNTIME_API_CALL(cudaMalloc(&dstData, dataSize));
    
    // Enable peer access
    RUNTIME_API_CALL(cudaDeviceEnablePeerAccess(srcDevice, 0));
    
    // Read initial metric values
    uint64_t startTransmit = readNVLinkMetric(metrics, true);
    uint64_t startReceive = readNVLinkMetric(metrics, false);
    
    // Record start time
    cudaEvent_t start, stop;
    RUNTIME_API_CALL(cudaEventCreate(&start));
    RUNTIME_API_CALL(cudaEventCreate(&stop));
    RUNTIME_API_CALL(cudaEventRecord(start));
    
    // Perform the memory transfer
    RUNTIME_API_CALL(cudaMemcpy(dstData, srcData, dataSize, cudaMemcpyDeviceToDevice));
    
    // Record end time
    RUNTIME_API_CALL(cudaEventRecord(stop));
    RUNTIME_API_CALL(cudaEventSynchronize(stop));
    
    // Read final metric values
    uint64_t endTransmit = readNVLinkMetric(metrics, true);
    uint64_t endReceive = readNVLinkMetric(metrics, false);
    
    // Calculate elapsed time
    float milliseconds = 0;
    RUNTIME_API_CALL(cudaEventElapsedTime(&milliseconds, start, stop));
    float seconds = milliseconds / 1000.0f;
    
    // Calculate bandwidth
    float gbps = (dataSize / (1024.0f * 1024.0f * 1024.0f)) / seconds;
    
    // Calculate NVLink utilization
    uint64_t transmittedBytes = endTransmit - startTransmit;
    uint64_t receivedBytes = endReceive - startReceive;
    
    printf("  Data size: %.1f MB\n", dataSize / (1024.0f * 1024.0f));
    printf("  Time: %.3f seconds\n", seconds);
    printf("  Bandwidth: %.1f GB/s\n", gbps);
    printf("  NVLink metrics:\n");
    printf("    Transmitted: %.1f MB\n", transmittedBytes / (1024.0f * 1024.0f));
    printf("    Received: %.1f MB\n", receivedBytes / (1024.0f * 1024.0f));
    
    // Clean up
    RUNTIME_API_CALL(cudaEventDestroy(start));
    RUNTIME_API_CALL(cudaEventDestroy(stop));
    RUNTIME_API_CALL(cudaFree(srcData));
    RUNTIME_API_CALL(cudaFree(dstData));
}
```

This function:
1. Allocates memory on both source and destination GPUs
2. Enables peer access between the GPUs
3. Records starting NVLink metric values
4. Performs a device-to-device memory transfer
5. Records ending NVLink metric values
6. Calculates and displays the achieved bandwidth

### 4. Reading NVLink Metrics

To read the current NVLink metrics:

```cpp
uint64_t readNVLinkMetric(NVLinkMetrics *metrics, bool isTransmit)
{
    CUpti_EventGroup group = isTransmit ? metrics->transmitEventGroup : metrics->receiveEventGroup;
    CUpti_MetricID metricId = isTransmit ? metrics->transmitMetricId : metrics->receiveMetricId;
    
    // Read the event values
    size_t eventValueBufferSize = metrics->numEvents * sizeof(uint64_t);
    uint64_t *eventValues = (uint64_t *)malloc(eventValueBufferSize);
    
    // For each event in the group, read its value
    CUpti_EventID *eventIds = metrics->eventIds;
    for (int i = 0; i < metrics->numEvents; i++) {
        size_t valueSize = sizeof(uint64_t);
        CUPTI_CALL(cuptiEventGroupReadEvent(group, CUPTI_EVENT_READ_FLAG_NONE, 
                                          eventIds[i], &valueSize, &eventValues[i]));
    }
    
    // Calculate the metric value from the event values
    double metricValue = 0.0;
    CUPTI_CALL(cuptiMetricGetValue(metrics->device, metricId,
                                 metrics->numEvents * sizeof(CUpti_EventID), eventIds,
                                 metrics->numEvents * sizeof(uint64_t), eventValues,
                                 0, &metricValue));
    
    free(eventValues);
    
    return (uint64_t)metricValue;
}
```

This function:
1. Reads the raw event values for the specified metric
2. Uses CUPTI to calculate the metric value from these events
3. Returns the metric value (total bytes transmitted or received)

## Running the Tutorial

1. Build the sample:
   ```bash
   make
   ```

2. Run the NVLink bandwidth test:
   ```bash
   ./nvlink_bandwidth
   ```

## Understanding the Output

When running on a system with NVLink-connected GPUs, you'll see output like:

```
Detecting NVLink connections...
Found 4 GPUs with NVLink connections:
  GPU 0 <-> GPU 1: 2 NVLink connections
  GPU 0 <-> GPU 2: 2 NVLink connections
  GPU 1 <-> GPU 3: 2 NVLink connections
  GPU 2 <-> GPU 3: 2 NVLink connections

Running bandwidth test...
Transfer GPU 0 -> GPU 1:
  Data size: 1024.0 MB
  Time: 0.082 seconds
  Bandwidth: 12.5 GB/s
  NVLink metrics:
    Transmitted: 1024.0 MB
    Received: 0.0 MB

Transfer GPU 1 -> GPU 0:
  Data size: 1024.0 MB
  Time: 0.081 seconds
  Bandwidth: 12.6 GB/s
  NVLink metrics:
    Transmitted: 0.0 MB
    Received: 1024.0 MB
```

This output shows:
1. The detected NVLink topology (which GPUs are connected and how many links)
2. For each transfer:
   - The data size being transferred
   - The time taken for the transfer
   - The achieved bandwidth in GB/s
   - The NVLink metrics showing bytes transmitted and received

## NVLink Performance Analysis

### Theoretical vs. Actual Bandwidth

NVLink bandwidth varies by GPU generation:
- Pascal (P100): Up to 20 GB/s per direction per link
- Volta (V100): Up to 25 GB/s per direction per link
- Ampere (A100): Up to 50 GB/s per direction per link

The actual bandwidth you achieve will typically be lower than the theoretical maximum due to:
1. Protocol overhead
2. Memory access patterns
3. Kernel execution overlap
4. System architecture

### Bidirectional Transfers

NVLink supports full-duplex communication, meaning data can be transferred in both directions simultaneously. To test bidirectional bandwidth:

```cpp
// Launch two CUDA streams
cudaStream_t stream1, stream2;
RUNTIME_API_CALL(cudaStreamCreate(&stream1));
RUNTIME_API_CALL(cudaStreamCreate(&stream2));

// Start bidirectional transfers
RUNTIME_API_CALL(cudaMemcpyAsync(dstData, srcData, dataSize, 
                               cudaMemcpyDeviceToDevice, stream1));
RUNTIME_API_CALL(cudaMemcpyAsync(srcData2, dstData2, dataSize, 
                               cudaMemcpyDeviceToDevice, stream2));

// Wait for both transfers to complete
RUNTIME_API_CALL(cudaStreamSynchronize(stream1));
RUNTIME_API_CALL(cudaStreamSynchronize(stream2));
```

Bidirectional transfers can nearly double the effective bandwidth compared to unidirectional transfers.

## Optimizing NVLink Usage

### Best Practices

1. **Peer Access**: Always enable peer access between NVLink-connected GPUs:
   ```cpp
   cudaDeviceEnablePeerAccess(peerDevice, 0);
   ```

2. **Asynchronous Transfers**: Use asynchronous memory copies with streams:
   ```cpp
   cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream);
   ```

3. **Transfer Size**: Use large transfer sizes (>1MB) to maximize bandwidth utilization

4. **Topology Awareness**: Place communicating tasks on directly connected GPUs

5. **Overlap Computation and Communication**: Use CUDA streams to overlap computation with NVLink transfers

## Advanced NVLink Features

### Unified Memory with NVLink

NVLink improves unified memory performance by enabling faster page migrations:

```cpp
// Allocate unified memory
void *unifiedData;
cudaMallocManaged(&unifiedData, size);

// Set preferred location
cudaMemAdvise(unifiedData, size, cudaMemAdviseSetPreferredLocation, srcDevice);

// Set accessing device
cudaMemAdvise(unifiedData, size, cudaMemAdviseSetAccessedBy, dstDevice);
```

### Atomic Operations

NVLink supports hardware-accelerated atomics between GPUs:

```cpp
// Check if native atomics are supported
int nativeAtomicsSupported = 0;
cuDeviceGetP2PAttribute(&nativeAtomicsSupported,
                       CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED,
                       srcDevice, dstDevice);
```

## Next Steps

- Apply NVLink bandwidth monitoring to your own multi-GPU applications
- Experiment with different transfer patterns to maximize bandwidth utilization
- Optimize your application's communication patterns based on the NVLink topology
- Use CUPTI metrics to identify and resolve NVLink bottlenecks 