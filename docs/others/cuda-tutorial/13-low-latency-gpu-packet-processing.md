# Low-Latency GPU Packet Processing

Processing network packets on GPUs can significantly accelerate throughput compared to CPU-only solutions, but achieving low latency requires careful optimization. This document explores techniques for minimizing latency when processing network packets on NVIDIA GPUs.

## Table of Contents
1. [Introduction to GPU Packet Processing](#introduction-to-gpu-packet-processing)
2. [Challenges in Low-Latency GPU Processing](#challenges-in-low-latency-gpu-processing)
3. [Basic Packet Processing Pipeline](#basic-packet-processing-pipeline)
4. [Code Structure and Design](#code-structure-and-design)
5. [Optimization Techniques](#optimization-techniques)
   - [CPU vs GPU Implementation](#cpu-vs-gpu-implementation)
   - [Pinned Memory](#pinned-memory)
   - [Zero-Copy Memory](#zero-copy-memory)
   - [Batching Strategies](#batching-strategies)
   - [Stream Concurrency](#stream-concurrency)
   - [Persistent Kernels](#persistent-kernels)
   - [CUDA Graphs](#cuda-graphs)
6. [Performance Analysis](#performance-analysis)
7. [Conclusion](#conclusion)

## Introduction to GPU Packet Processing

Network packet processing tasks typically involve:
- Packet parsing/header extraction
- Protocol decoding
- Filtering (firewall rules, pattern matching)
- Traffic analysis
- Cryptographic operations
- Deep packet inspection

GPUs excel at these tasks due to their:
- Massive parallelism for processing multiple packets simultaneously
- High memory bandwidth for moving packet data
- Specialized instructions for certain operations (e.g., cryptography)

## Challenges in Low-Latency GPU Processing

Several factors contribute to latency in GPU packet processing:

1. **Data Transfer Overhead**: Moving data between host and device memory is often the primary bottleneck
2. **Kernel Launch Overhead**: Each kernel launch incurs ~5-10μs of overhead
3. **Batching Tension**: Larger batches improve throughput but increase latency
4. **Synchronization Costs**: Coordination between CPU and GPU adds latency
5. **Memory Access Patterns**: Irregular accesses to packet data can cause poor cache utilization

## Basic Packet Processing Pipeline

A typical GPU packet processing pipeline consists of these stages:

1. **Packet Capture**: Receive packets from the network interface
2. **Batching**: Collect multiple packets to amortize transfer and launch costs
3. **Transfer to GPU**: Copy packet data to the GPU memory
4. **Processing**: Execute kernel(s) to process packets
5. **Transfer Results**: Copy processed results back to the host
6. **Response/Forwarding**: Take action based on processing results

### Example Basic Pipeline

```
Network → CPU Buffer → Batch Collection → GPU Transfer → GPU Processing → Results Transfer → Action
```

## Code Structure and Design

Our implementation follows a modular design that separates the core packet processing logic from the optimization strategies. This approach has several benefits:

1. **Separation of Concerns**: The packet processing logic is decoupled from optimization techniques
2. **Easy Comparison**: We can directly compare different optimization approaches using the same processing logic
3. **Maintainability**: Changes to processing logic or optimization strategies can be made independently
4. **Clarity**: The impact of each optimization is clearly visible

### Core Components

1. **Data Structures**:
   - `Packet`: Contains header, payload, size, and status information
   - `PacketResult`: Contains processing results including action to take
   - `PacketBatch`: Groups packets for batch processing

2. **Core Processing Functions**:
   - `processPacketCPU()`: CPU implementation of packet processing
   - `processPacketGPU()`: GPU device function implementation (used by all kernels)

3. **Optimization Stages**:
   - Each optimization strategy is implemented as a separate function
   - All strategies use the same core processing logic
   - Results show the performance impact of each approach

## Optimization Techniques

### CPU vs GPU Implementation

We begin by comparing CPU and GPU implementations to establish a baseline:

```cpp
// CPU implementation
void processPacketCPU(const Packet* packet, PacketResult* result, int packetId) {
    // Core packet processing logic
}

// GPU implementation
__device__ void processPacketGPU(const Packet* packet, PacketResult* result, int packetId) {
    // Same core logic, but as a device function
}
```

The CPU version processes packets sequentially, while the GPU version processes them in parallel across thousands of threads.

### Pinned Memory

**Problem**: Standard pageable memory requires an additional copy when transferring to/from GPU

**Solution**: Use pinned (page-locked) memory to enable direct GPU access

```cuda
// Allocate pinned memory for packet buffers
cudaHostAlloc(&h_packets, packet_buffer_size, cudaHostAllocDefault);
```

**Benefit**: Up to 2x faster transfers between host and device

### Zero-Copy Memory

**Problem**: Even with pinned memory, explicit transfers add latency

**Solution**: Map host memory directly into GPU address space using zero-copy memory

```cuda
// Allocate zero-copy memory
cudaHostAlloc(&h_packets, packet_buffer_size, cudaHostAllocMapped);
cudaHostGetDevicePointer(&d_packets, h_packets, 0);
```

**Benefit**: Eliminates explicit transfers, allows fine-grained access
**Trade-off**: Lower bandwidth via PCIe, but can reduce latency for small transfers

### Batching Strategies

**Problem**: Small batches = high overhead; large batches = high latency

**Solution**: Implement adaptive batching based on traffic conditions

- **Timeout-based batching**: Process after X microseconds or when batch is full
- **Dynamic batch sizing**: Adjust batch size based on load and latency requirements
- **Two-level batching**: Small batches for critical packets, larger for others

### Stream Concurrency

**Problem**: Sequential execution of transfers and kernels wastes time

**Solution**: Use CUDA streams to overlap operations

```cuda
// Create streams for pipelining
cudaStream_t streams[NUM_STREAMS];
for (int i = 0; i < NUM_STREAMS; i++) {
    cudaStreamCreate(&streams[i]);
}

// Pipeline execution
for (int i = 0; i < NUM_BATCHES; i++) {
    int stream_idx = i % NUM_STREAMS;
    // Asynchronously transfer batch i to GPU
    cudaMemcpyAsync(d_packets[i], h_packets[i], batch_size, 
                    cudaMemcpyHostToDevice, streams[stream_idx]);
    // Process batch i
    processPacketsKernel<<<grid, block, 0, streams[stream_idx]>>>(
        d_packets[i], d_results[i], batch_size);
    // Asynchronously transfer results back
    cudaMemcpyAsync(h_results[i], d_results[i], result_size,
                   cudaMemcpyDeviceToHost, streams[stream_idx]);
}
```

**Benefit**: Higher throughput and lower average latency through pipelining

### Persistent Kernels

**Problem**: Kernel launch overhead adds significant latency

**Solution**: Keep a kernel running indefinitely, waiting for new work

```cuda
__global__ void persistentKernel(volatile int* work_queue, volatile int* queue_size,
                                 PacketBatch* batches) {
    while (true) {
        // Check for new work
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            // Wait for new batch (spin-wait or sleep)
            while (*queue_size == 0);
            // Get batch index
            batch_idx = atomicAdd((int*)queue_size, -1);
        }
        // Broadcast batch_idx to all threads using shared memory
        __shared__ int s_batch_idx;
        if (threadIdx.x == 0) s_batch_idx = batch_idx;
        __syncthreads();
        
        // Process packets from the assigned batch using our core function
        processPacketGPU(&batches[s_batch_idx].packets[tid], &results[tid], tid);
        
        // Signal completion
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            batches[s_batch_idx].status = COMPLETED;
        }
    }
}
```

**Benefit**: Eliminates kernel launch overhead, allowing sub-microsecond latency

### CUDA Graphs

**Problem**: Even with streams, each kernel launch has CPU overhead

**Solution**: Use CUDA Graphs to capture and replay entire workflows

```cuda
// Create and capture a CUDA graph
cudaGraph_t graph;
cudaGraphExec_t graphExec;

// Capture operations into a graph
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
for (int i = 0; i < PIPELINE_DEPTH; i++) {
    cudaMemcpyAsync(...); // Copy input
    kernel<<<...>>>(...);  // Process
    cudaMemcpyAsync(...); // Copy output
}
cudaStreamEndCapture(stream, &graph);

// Compile the graph into an executable
cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

// Execute the graph repeatedly with new data
for (int batch = 0; batch < NUM_BATCHES; batch++) {
    updateGraphInputs(batch); // Update memory addresses
    cudaGraphLaunch(graphExec, stream);
}
```

**Benefit**: Reduces CPU overhead by 30-50%, leading to lower latency

## Performance Analysis

When optimizing for low-latency packet processing, measure these metrics:

1. **End-to-end latency**: Time from packet arrival to processing completion
2. **Processing throughput**: Packets processed per second
3. **Batch processing time**: Time to process a single batch
4. **Transfer overhead**: Time spent in host-device transfers
5. **Kernel execution time**: Time spent executing GPU code
6. **Queue waiting time**: Time packets spend waiting in batching queues

Based on our implementation results:

| Method | Processing Time (μs) | Notes |
|--------|-------------------|-------|
| CPU (Baseline) | 6,639 | Sequential processing |
| Basic GPU | 4,124 | ~1.6x faster than CPU |
| Pinned Memory | 2,987 | ~2.2x faster than CPU |
| Batched Streams | 8,488 | Higher total time but low per-packet latency (0.83 μs) |
| Zero-Copy | 61,170 | Much slower due to PCIe bandwidth limitations |
| Persistent Kernel | 200,470 | High total time but includes simulated packet arrival delays |
| CUDA Graphs | 132,917 | Reduces launch overhead but still has synchronization costs |

## Conclusion

Achieving low-latency GPU packet processing requires balancing multiple factors:

1. **Minimize data transfers** wherever possible
2. **Optimize kernel launch overhead** with persistent kernels or CUDA graphs
3. **Use intelligent batching** strategies based on traffic patterns
4. **Pipeline operations** using streams to hide latency
5. **Leverage GPU-specific memory features** like zero-copy when appropriate

By separating core processing logic from optimization strategies, we can clearly see the impact of each approach and select the best technique for our specific use case.

The optimal approach often involves combining multiple techniques based on workload characteristics:
- Use persistent kernels for minimal latency
- Use pinned memory for data that must be transferred
- Use zero-copy for small, latency-sensitive data
- Use adaptive batching based on traffic patterns
- Use CUDA Graphs for complex, repeatable processing pipelines

## References

1. NVIDIA CUDA Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
2. NVIDIA GPUDirect: https://developer.nvidia.com/gpudirect
3. DPDK (Data Plane Development Kit): https://www.dpdk.org/
4. NVIDIA DOCA SDK: https://developer.nvidia.com/networking/doca 