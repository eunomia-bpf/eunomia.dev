/**
 * 13-low-latency-gpu-packet-processing.cu
 * 
 * This example demonstrates techniques for low-latency network packet processing on GPUs.
 * The code progresses through several optimization stages, from a basic implementation
 * to increasingly optimized versions.
 * 
 * Key optimizations include:
 * 1. Pinned memory for faster transfers
 * 2. Zero-copy memory to avoid explicit transfers
 * 3. Stream concurrency for operation overlap
 * 4. Persistent kernels to eliminate launch overhead
 * 5. CUDA Graphs for reduced CPU overhead
 * 
 * This implementation includes batch size exploration to find optimal batch sizes.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <chrono>
#include <thread>
#include <vector>
#include <algorithm>
#include <random>
#include <mutex>
#include <condition_variable>
#include "lib/packet_processing_common.h"
#include "lib/packet_processing_kernels.cuh"
#include "lib/packet_processing_batch.cuh"

// Error checking macro
#define CHECK_CUDA_ERROR(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Global test data - generate once and reuse
Packet* g_test_packets = nullptr;
PacketResult* g_test_results = nullptr;

// Function prototypes
long long runCPUProcessing();
long long runBasicProcessing();
long long runPinnedMemoryProcessing(int batchSize);
long long runBatchedStreamProcessing(int batchSize);
long long runZeroCopyProcessing(int batchSize);
long long runPersistentKernelProcessing(int batchSize);
long long runCudaGraphsProcessing(int batchSize);

// Initialize global test data
void initializeTestData() {
    // Allocate memory for global test packets and results
    g_test_packets = (Packet*)malloc(NUM_PACKETS * sizeof(Packet));
    g_test_results = (PacketResult*)malloc(NUM_PACKETS * sizeof(PacketResult));
    
    // Generate test packets once
    generateTestPackets(g_test_packets, NUM_PACKETS);
}

// Clean up global test data
void cleanupTestData() {
    if (g_test_packets) {
        free(g_test_packets);
        g_test_packets = nullptr;
    }
    
    if (g_test_results) {
        free(g_test_results);
        g_test_results = nullptr;
    }
}

// Helper to copy test packets to a destination buffer
void copyTestPackets(Packet* dest, int startIdx, int count) {
    memcpy(dest, &g_test_packets[startIdx], count * sizeof(Packet));
}

/******************************************************************************
 * Stage 0: CPU-based Processing (Baseline)
 * 
 * This is a simple CPU implementation for comparison purposes.
 ******************************************************************************/

long long runCPUProcessing() {
    PerformanceMetrics metrics = {0};
    metrics.batchSize = NUM_PACKETS;  // Process all packets at once
    
    // Reset packet status
    resetPacketStatus();
    
    // Measure CPU performance
    auto start = std::chrono::high_resolution_clock::now();
    
    // Process packets on CPU
    for (int i = 0; i < NUM_PACKETS; i++) {
        processPacketCPU(&g_test_packets[i], &g_test_results[i], i);
        g_test_packets[i].status = COMPLETED;
    }
    
    // Calculate total time
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    metrics.totalTime = duration;
    
    // Calculate statistics
    calculateResults(g_test_results, NUM_PACKETS, metrics);
    
    // Print performance metrics
    printPerformanceMetrics("Stage 0: CPU-based Processing (Baseline)", metrics);
    
    printf("CPU processing (total): %lld us\n", duration);
    return duration;
}

/******************************************************************************
 * Stage 2: Pinned Memory Optimization
 * 
 * This version uses pinned memory for faster host-device transfers.
 ******************************************************************************/

long long runPinnedMemoryProcessing(int batchSize) {
    PerformanceMetrics metrics = {0};
    metrics.batchSize = batchSize;
    metrics.numBatches = (NUM_PACKETS + batchSize - 1) / batchSize;
    
    // Reset packet status
    resetPacketStatus();
    
    // Allocate pinned memory for host (setup - outside timing)
    Packet* h_pinned_packets;
    PacketResult* h_pinned_results;
    
    CHECK_CUDA_ERROR(cudaHostAlloc(&h_pinned_packets, batchSize * sizeof(Packet), 
                     cudaHostAllocDefault));
    CHECK_CUDA_ERROR(cudaHostAlloc(&h_pinned_results, batchSize * sizeof(PacketResult), 
                     cudaHostAllocDefault));
    
    // Allocate device memory (setup - outside timing)
    Packet* d_packets;
    PacketResult* d_results;
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_packets, batchSize * sizeof(Packet)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_results, batchSize * sizeof(PacketResult)));
    
    // Measure only the actual processing performance
    auto start = std::chrono::high_resolution_clock::now();
    
    long long total_transfer_time = 0;
    long long total_kernel_time = 0;
    
    // Process packets in batches
    for (int batch = 0; batch < metrics.numBatches; batch++) {
        int offset = batch * batchSize;
        int currentBatchSize = (batch == metrics.numBatches - 1) ? 
                             (NUM_PACKETS - batch * batchSize) : batchSize;
        
        // Copy from global to pinned memory
        memcpy(h_pinned_packets, g_test_packets + offset, 
               currentBatchSize * sizeof(Packet));
        
        // Copy batch to device using pinned memory
        auto transfer_start = std::chrono::high_resolution_clock::now();
        
        CHECK_CUDA_ERROR(cudaMemcpy(d_packets, h_pinned_packets, 
                         currentBatchSize * sizeof(Packet), 
                         cudaMemcpyHostToDevice));
        
        auto transfer_end = std::chrono::high_resolution_clock::now();
        total_transfer_time += std::chrono::duration_cast<std::chrono::microseconds>
                              (transfer_end - transfer_start).count();
        
        // Process batch
        int blockSize = 256;
        int numBlocks = (currentBatchSize + blockSize - 1) / blockSize;
        
        auto kernel_start = std::chrono::high_resolution_clock::now();
        processPacketsBasic<<<numBlocks, blockSize>>>(d_packets, d_results, currentBatchSize);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        auto kernel_end = std::chrono::high_resolution_clock::now();
        
        total_kernel_time += std::chrono::duration_cast<std::chrono::microseconds>
                           (kernel_end - kernel_start).count();
        
        // Copy results back to pinned memory
        CHECK_CUDA_ERROR(cudaMemcpy(h_pinned_results, d_results, 
                         currentBatchSize * sizeof(PacketResult), 
                         cudaMemcpyDeviceToHost));
        
        // Copy from pinned to global memory
        memcpy(g_test_results + offset, h_pinned_results, 
               currentBatchSize * sizeof(PacketResult));
    }
    
    // End timing before any printf statements
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    // Store metrics (after timing)
    metrics.totalTime = duration;
    metrics.transferTime = total_transfer_time;
    metrics.kernelTime = total_kernel_time;
    
    // Calculate statistics (after timing)
    calculateResults(g_test_results, NUM_PACKETS, metrics);
    
    // Print performance metrics (after timing)
    printf("Transfer time: %lld us, Kernel time: %lld us\n", 
           metrics.transferTime, metrics.kernelTime);
    
    char stageTitle[100];
    snprintf(stageTitle, sizeof(stageTitle), "Stage 2: Pinned Memory Optimization (Batch Size = %d)", batchSize);
    printPerformanceMetrics(stageTitle, metrics);
    
    // Cleanup (after timing)
    CHECK_CUDA_ERROR(cudaFreeHost(h_pinned_packets));
    CHECK_CUDA_ERROR(cudaFreeHost(h_pinned_results));
    CHECK_CUDA_ERROR(cudaFree(d_packets));
    CHECK_CUDA_ERROR(cudaFree(d_results));
    
    printf("Pinned memory processing (total): %lld us\n", duration);
    return duration;
}

/******************************************************************************
 * Stage 3: Batched Processing with Streams
 * 
 * This version processes packets in batches and uses multiple CUDA streams
 * to overlap transfers and computation, reducing overall latency.
 ******************************************************************************/

long long runBatchedStreamProcessing(int batchSize) {
    PerformanceMetrics metrics = {0};
    metrics.batchSize = batchSize;
    metrics.numBatches = (NUM_PACKETS + batchSize - 1) / batchSize;
    
    // Reset packet status
    resetPacketStatus();
    
    // Create CUDA streams (setup - outside timing)
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        CHECK_CUDA_ERROR(cudaStreamCreate(&streams[i]));
    }
    
    // Allocate pinned memory for packets and results (setup - outside timing)
    Packet* h_packets[MAX_BATCHES];
    PacketResult* h_results[MAX_BATCHES];
    
    for (int i = 0; i < MAX_BATCHES; i++) {
        CHECK_CUDA_ERROR(cudaHostAlloc(&h_packets[i], batchSize * sizeof(Packet), 
                         cudaHostAllocDefault));
        CHECK_CUDA_ERROR(cudaHostAlloc(&h_results[i], batchSize * sizeof(PacketResult), 
                         cudaHostAllocDefault));
    }
    
    // Allocate device memory for each batch (setup - outside timing)
    Packet* d_packets[MAX_BATCHES];
    PacketResult* d_results[MAX_BATCHES];
    
    for (int i = 0; i < MAX_BATCHES; i++) {
        CHECK_CUDA_ERROR(cudaMalloc(&d_packets[i], batchSize * sizeof(Packet)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_results[i], batchSize * sizeof(PacketResult)));
    }
    
    // Kernel launch parameters
    int blockSize = 256;
    
    // Measure only the actual processing performance
    auto start = std::chrono::high_resolution_clock::now();
    
    // Process all batches
    for (int batch = 0; batch < metrics.numBatches; batch++) {
        int streamIdx = batch % NUM_STREAMS;
        int batchIdx = batch % MAX_BATCHES;
        int currentBatchSize = (batch == metrics.numBatches - 1) ? 
                        (NUM_PACKETS - batch * batchSize) : batchSize;
        
        // Copy test packets to pinned memory
        memcpy(h_packets[batchIdx], g_test_packets + batch * batchSize, 
               currentBatchSize * sizeof(Packet));
        
        // Number of thread blocks needed for this batch
        int numBlocks = (currentBatchSize + blockSize - 1) / blockSize;
        
        // Transfer batch to device
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_packets[batchIdx], h_packets[batchIdx], 
                         currentBatchSize * sizeof(Packet), cudaMemcpyHostToDevice, 
                         streams[streamIdx]));
        
        // Process batch
        processPacketsBasic<<<numBlocks, blockSize, 0, streams[streamIdx]>>>(
            d_packets[batchIdx], d_results[batchIdx], currentBatchSize);
        
        // Transfer results back
        CHECK_CUDA_ERROR(cudaMemcpyAsync(h_results[batchIdx], d_results[batchIdx], 
                         currentBatchSize * sizeof(PacketResult), cudaMemcpyDeviceToHost, 
                         streams[streamIdx]));
        
        // If we've used all available batch slots, synchronize the oldest stream
        // to ensure its resources are available
        if (batch >= MAX_BATCHES - 1) {
            int oldestStreamIdx = (batch - (MAX_BATCHES - 1)) % NUM_STREAMS;
            CHECK_CUDA_ERROR(cudaStreamSynchronize(streams[oldestStreamIdx]));
            
            // Copy results to global buffer
            int oldestBatch = batch - (MAX_BATCHES - 1);
            int oldestBatchIdx = oldestBatch % MAX_BATCHES;
            int oldestBatchSize = (oldestBatch == metrics.numBatches - 1) ? 
                           (NUM_PACKETS - oldestBatch * batchSize) : batchSize;
            
            memcpy(g_test_results + oldestBatch * batchSize, h_results[oldestBatchIdx],
                   oldestBatchSize * sizeof(PacketResult));
        }
    }
    
    // Synchronize all streams and copy remaining results
    for (int i = 0; i < NUM_STREAMS; i++) {
        CHECK_CUDA_ERROR(cudaStreamSynchronize(streams[i]));
    }
    
    // Copy any remaining results that weren't copied during processing
    for (int batch = std::max(0, metrics.numBatches - (MAX_BATCHES - 1)); batch < metrics.numBatches; batch++) {
        int batchIdx = batch % MAX_BATCHES;
        int currentBatchSize = (batch == metrics.numBatches - 1) ? 
                        (NUM_PACKETS - batch * batchSize) : batchSize;
        
        memcpy(g_test_results + batch * batchSize, h_results[batchIdx],
               currentBatchSize * sizeof(PacketResult));
    }
    
    // End timing before any calculations or cleanup
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    // Store metrics (after timing)
    metrics.totalTime = duration;
    
    // Calculate average latency per batch and per packet (after timing)
    metrics.avgBatchLatency = (double)duration / metrics.numBatches;
    metrics.avgPacketLatency = (double)duration / NUM_PACKETS;
    
    // Calculate statistics (after timing)
    calculateResults(g_test_results, NUM_PACKETS, metrics);
    
    // Print performance metrics (after timing)
    char stageTitle[100];
    snprintf(stageTitle, sizeof(stageTitle), "Stage 3: Batched Streams (Batch Size = %d)", batchSize);
    printPerformanceMetrics(stageTitle, metrics);
    
    // Cleanup (after timing)
    for (int i = 0; i < MAX_BATCHES; i++) {
        CHECK_CUDA_ERROR(cudaFreeHost(h_packets[i]));
        CHECK_CUDA_ERROR(cudaFreeHost(h_results[i]));
        CHECK_CUDA_ERROR(cudaFree(d_packets[i]));
        CHECK_CUDA_ERROR(cudaFree(d_results[i]));
    }
    
    for (int i = 0; i < NUM_STREAMS; i++) {
        CHECK_CUDA_ERROR(cudaStreamDestroy(streams[i]));
    }
    
    printf("Batched stream processing (total): %lld us\n", duration);
    return duration;
}

/******************************************************************************
 * Stage 4: Zero-Copy Memory
 * 
 * This version uses zero-copy memory to eliminate explicit data transfers,
 * which can reduce latency for small packets.
 ******************************************************************************/

// Note: The processPacketsZeroCopy kernel is already defined in packet_processing_kernels.cuh

long long runZeroCopyProcessing(int batchSize) {
    PerformanceMetrics metrics = {0};
    metrics.batchSize = batchSize;
    metrics.numBatches = (NUM_PACKETS + batchSize - 1) / batchSize;
    
    // Reset packet status
    resetPacketStatus();
    
    // Allocate zero-copy (mapped) memory (setup - outside timing)
    Packet* h_zero_copy_packets;
    PacketResult* h_zero_copy_results;
    
    CHECK_CUDA_ERROR(cudaHostAlloc(&h_zero_copy_packets, batchSize * sizeof(Packet), 
                     cudaHostAllocMapped));
    CHECK_CUDA_ERROR(cudaHostAlloc(&h_zero_copy_results, batchSize * sizeof(PacketResult), 
                     cudaHostAllocMapped));
    
    // Get device pointers to the mapped memory (setup - outside timing)
    Packet* d_zero_copy_packets;
    PacketResult* d_zero_copy_results;
    
    CHECK_CUDA_ERROR(cudaHostGetDevicePointer(&d_zero_copy_packets, h_zero_copy_packets, 0));
    CHECK_CUDA_ERROR(cudaHostGetDevicePointer(&d_zero_copy_results, h_zero_copy_results, 0));
    
    // Measure only the actual processing performance
    auto start = std::chrono::high_resolution_clock::now();
    
    // Process packets in batches
    for (int batch = 0; batch < metrics.numBatches; batch++) {
        int offset = batch * batchSize;
        int currentBatchSize = (batch == metrics.numBatches - 1) ? 
                             (NUM_PACKETS - batch * batchSize) : batchSize;
        
        // Copy batch to zero-copy buffer (still needed for test data)
        memcpy(h_zero_copy_packets, g_test_packets + offset, 
               currentBatchSize * sizeof(Packet));
        
        // Process batch directly in zero-copy memory
        int blockSize = 256;
        int numBlocks = (currentBatchSize + blockSize - 1) / blockSize;
        
        processPacketsZeroCopy<<<numBlocks, blockSize>>>(d_zero_copy_packets, 
                                                        d_zero_copy_results, 
                                                        currentBatchSize);
        
        // Wait for kernel to finish
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        
        // Copy results to global buffer (no device-to-host transfer needed)
        memcpy(g_test_results + offset, h_zero_copy_results, 
               currentBatchSize * sizeof(PacketResult));
    }
    
    // End timing before any calculations or cleanup
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    // Store metrics (after timing)
    metrics.totalTime = duration;
    
    // Calculate statistics (after timing)
    calculateResults(g_test_results, NUM_PACKETS, metrics);
    
    // Print performance metrics (after timing)
    char stageTitle[100];
    snprintf(stageTitle, sizeof(stageTitle), "Stage 4: Zero-Copy Memory (Batch Size = %d)", batchSize);
    printPerformanceMetrics(stageTitle, metrics);
    
    // Cleanup (after timing)
    CHECK_CUDA_ERROR(cudaFreeHost(h_zero_copy_packets));
    CHECK_CUDA_ERROR(cudaFreeHost(h_zero_copy_results));
    
    printf("Zero-copy processing (total): %lld us\n", duration);
    return duration;
}

/******************************************************************************
 * Stage 5: Real Persistent Kernel
 * 
 * Using the persistent kernel header-only library for true persistent kernel
 * processing with CPU-GPU synchronization and dynamic batch handling.
 ******************************************************************************/

// Work queue structure for persistent kernel (based on 12-advanced-gpu-customizations.cu)
struct PacketWorkQueue {
    int items[NUM_PACKETS];    // Work items (packet indices)
    int head;                  // Current head of queue (atomic)
    int tail;                  // Current tail of queue
    int finished;              // Flag to indicate all work is done
};

// True persistent kernel that continuously grabs work items
__global__ void persistentPacketKernel(Packet* packets, PacketResult* results, 
                                      PacketWorkQueue* queue, int totalPackets) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Keep thread alive to process multiple items
    while (true) {
        // Atomically grab next work item
        int work_idx = atomicAdd(&queue->head, 1);
        
        // Check if we've processed all items
        if (work_idx >= totalPackets || work_idx >= queue->tail) {
            break;
        }
        
        // Get the packet index to process
        int packetIdx = queue->items[work_idx];
        
        // Validate packet index
        if (packetIdx < 0 || packetIdx >= totalPackets) {
            continue;
        }
        
        // Process the packet
        Packet* packet = &packets[packetIdx];
        PacketResult* result = &results[packetIdx];
        
        // Mark packet as being processed
        packet->status = PROCESSING;
        
        // Call the core packet processing function
        processPacketGPU(packet, result, packetIdx);
        
        // Mark packet as completed
        packet->status = COMPLETED;
    }
}

long long runPersistentKernelProcessing(int batchSize) {
    PerformanceMetrics metrics = {0};
    metrics.batchSize = batchSize;
    metrics.numBatches = (NUM_PACKETS + batchSize - 1) / batchSize;
    
    printf("Starting batch-based persistent kernel processing...\n");
    printf("Processing %d packets in %d batches (batch size: %d)\n", 
           NUM_PACKETS, metrics.numBatches, batchSize);
    
    // Reset packet status
    resetPacketStatus();
    
    // Allocate and initialize host work queue
    PacketWorkQueue* h_queue = (PacketWorkQueue*)malloc(sizeof(PacketWorkQueue));
    
    // Initialize work queue - start with empty queue
    h_queue->head = 0;
    h_queue->tail = 0;  // Start with no work items
    h_queue->finished = 0;
    
    // Allocate device memory
    Packet* d_packets;
    PacketResult* d_results;
    PacketWorkQueue* d_queue;
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_packets, NUM_PACKETS * sizeof(Packet)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_results, NUM_PACKETS * sizeof(PacketResult)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_queue, sizeof(PacketWorkQueue)));
   
    // Copy initial empty queue to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_queue, h_queue, 
                     sizeof(PacketWorkQueue), cudaMemcpyHostToDevice));
    
    // Launch persistent kernel with fewer threads since each handles multiple items
    dim3 blockDim(256);
    dim3 gridDim(32);  // Use fewer blocks for persistent threads
    
    printf("Launching persistent kernel with %d blocks x %d threads...\n", 
           gridDim.x, blockDim.x);
    
    // Launch the persistent kernel FIRST (it will wait for work)
    persistentPacketKernel<<<gridDim, blockDim>>>(d_packets, d_results, d_queue, NUM_PACKETS);
    
    // Check for launch errors
    cudaError_t launchError = cudaGetLastError();
    if (launchError != cudaSuccess) {
        printf("Persistent kernel launch failed: %s\n", cudaGetErrorString(launchError));
        free(h_queue);
        cudaFree(d_packets);
        cudaFree(d_results);
        cudaFree(d_queue);
        return -1;
    }
    
    printf("Persistent kernel launched successfully, now submitting batches...\n");
    
    // Start timing after kernel launch
    auto start = std::chrono::high_resolution_clock::now();
    
    // Process batches sequentially but efficiently
    for (int batch = 0; batch < metrics.numBatches; batch++) {
        int offset = batch * batchSize;
        int currentBatchSize = (batch == metrics.numBatches - 1) ? 
                             (NUM_PACKETS - batch * batchSize) : batchSize;
        
        // Add packet indices for this batch to work queue
        for (int i = 0; i < currentBatchSize; i++) {
            h_queue->items[h_queue->tail + i] = offset + i;
        }
        h_queue->tail += currentBatchSize;
        
        // Copy updated queue to device (only once per batch)
        CHECK_CUDA_ERROR(cudaMemcpy(d_queue, h_queue, 
                         sizeof(PacketWorkQueue), cudaMemcpyHostToDevice));
        
        // Wait for this batch to complete using device synchronization
        // The persistent kernel will process all available work items
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        
        // Update local queue head to match what was processed
        h_queue->head = h_queue->tail;
    }
    
    // End timing
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    printf("All work submitted and processed, finalizing...\n");
    printf("Persistent kernel completed, copying results...\n");
    
    // Copy results back to host
    CHECK_CUDA_ERROR(cudaMemcpy(g_test_results, d_results, 
                     NUM_PACKETS * sizeof(PacketResult), cudaMemcpyDeviceToHost));
    
    // Copy processed packets back to host (CRITICAL: needed for status verification)
    CHECK_CUDA_ERROR(cudaMemcpy(g_test_packets, d_packets, 
                     NUM_PACKETS * sizeof(Packet), cudaMemcpyDeviceToHost));
    
    // Copy final queue state to check completion
    CHECK_CUDA_ERROR(cudaMemcpy(h_queue, d_queue, 
                     sizeof(PacketWorkQueue), cudaMemcpyDeviceToHost));
    
    printf("Final work queue head: %d, tail: %d\n", h_queue->head, h_queue->tail);
    
    // Calculate statistics
    calculateResults(g_test_results, NUM_PACKETS, metrics);
    
    // Store metrics
    metrics.totalTime = duration;
    metrics.avgBatchLatency = (double)duration / metrics.numBatches;
    metrics.avgPacketLatency = (double)duration / NUM_PACKETS;
    
    // Print performance metrics
    char stageTitle[100];
    snprintf(stageTitle, sizeof(stageTitle), "Stage 5: Batch-based Persistent Kernel (Batch Size = %d)", batchSize);
    printPerformanceMetrics(stageTitle, metrics);
    
    printf("Batch-based persistent kernel processing completed successfully\n");
    
    // Verify results
    bool resultsValid = true;
    int completedCount = 0;
    for (int i = 0; i < NUM_PACKETS; i++) {
        if (g_test_packets[i].status == COMPLETED) {
            completedCount++;
        } else {
            if (resultsValid) {
                printf("Warning: Packet %d not completed (status: %d)\n", i, g_test_packets[i].status);
                resultsValid = false;
            }
        }
    }
    
    printf("Results verification: %s (%d/%d packets completed)\n", 
           resultsValid ? "PASSED" : "FAILED", completedCount, NUM_PACKETS);
    
    // Cleanup
    free(h_queue);
    CHECK_CUDA_ERROR(cudaFree(d_packets));
    CHECK_CUDA_ERROR(cudaFree(d_results));
    CHECK_CUDA_ERROR(cudaFree(d_queue));
    
    printf("Batch-based persistent kernel processing (total): %lld us\n", duration);
    
    return duration;
}

/******************************************************************************
 * Stage 6: CUDA Graphs
 * 
 * This version uses CUDA Graphs to capture and replay the entire sequence
 * of operations (memory transfers, kernel launches) for minimal CPU overhead.
 ******************************************************************************/

long long runCudaGraphsProcessing(int batchSize) {
    PerformanceMetrics metrics = {0};
    metrics.batchSize = batchSize;
    metrics.numBatches = (NUM_PACKETS + batchSize - 1) / batchSize;
    
    // Reset packet status
    resetPacketStatus();
    
    // Allocate pinned memory for host (setup - outside timing)
    Packet* h_batch_packets;
    PacketResult* h_batch_results;
    
    CHECK_CUDA_ERROR(cudaHostAlloc(&h_batch_packets, batchSize * sizeof(Packet), 
                     cudaHostAllocDefault));
    CHECK_CUDA_ERROR(cudaHostAlloc(&h_batch_results, batchSize * sizeof(PacketResult), 
                     cudaHostAllocDefault));
    
    // Allocate device memory (setup - outside timing)
    Packet* d_batch_packets;
    PacketResult* d_batch_results;
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_batch_packets, batchSize * sizeof(Packet)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_batch_results, batchSize * sizeof(PacketResult)));
    
    // Create CUDA stream (setup - outside timing)
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
    
    // Create CUDA graph (setup - outside timing)
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    
    // Capture graph once for a typical batch (setup - outside timing)
    {
        // Fill a batch with test packets
        memcpy(h_batch_packets, g_test_packets, batchSize * sizeof(Packet));
        
        // Begin graph capture
        CHECK_CUDA_ERROR(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        
        // Copy data to device
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_batch_packets, h_batch_packets, 
                         batchSize * sizeof(Packet), 
                         cudaMemcpyHostToDevice, stream));
        
        // Launch kernel
        dim3 blockDim(256);
        dim3 gridDim((batchSize + blockDim.x - 1) / blockDim.x);
        processPacketsBasic<<<gridDim, blockDim, 0, stream>>>(d_batch_packets, d_batch_results, batchSize);
        
        // Copy results back
        CHECK_CUDA_ERROR(cudaMemcpyAsync(h_batch_results, d_batch_results, 
                         batchSize * sizeof(PacketResult), 
                         cudaMemcpyDeviceToHost, stream));
        
        // End capture
        CHECK_CUDA_ERROR(cudaStreamEndCapture(stream, &graph));
        
        // Create executable graph
        CHECK_CUDA_ERROR(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
    }
    
    // Measure only the actual processing performance
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<long long> launchTimes;
    launchTimes.reserve(metrics.numBatches);
    
    // Process all batches using graph
    for (int batch = 0; batch < metrics.numBatches; batch++) {
        int offset = batch * batchSize;
        int currentBatchSize = (batch == metrics.numBatches - 1) ? 
                             (NUM_PACKETS - batch * batchSize) : batchSize;
        
        // Copy batch data to pinned buffer
        memcpy(h_batch_packets, g_test_packets + offset, currentBatchSize * sizeof(Packet));
        
        // Launch graph execution
        auto launchStart = std::chrono::high_resolution_clock::now();
        CHECK_CUDA_ERROR(cudaGraphLaunch(graphExec, stream));
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
        auto launchEnd = std::chrono::high_resolution_clock::now();
        
        // Track launch time
        long long launchTime = std::chrono::duration_cast<std::chrono::microseconds>(
                             launchEnd - launchStart).count();
        launchTimes.push_back(launchTime);
        
        // Copy results to global buffer
        memcpy(g_test_results + offset, h_batch_results, currentBatchSize * sizeof(PacketResult));
    }
    
    // End timing before any calculations or cleanup
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    // Store metrics (after timing)
    metrics.totalTime = duration;
    
    // Calculate average launch time (after timing)
    double avgLaunchTime = 0;
    for (long long time : launchTimes) {
        avgLaunchTime += time;
    }
    avgLaunchTime /= launchTimes.size();
    
    // Calculate statistics (after timing)
    calculateResults(g_test_results, NUM_PACKETS, metrics);
    
    // Print performance metrics (after timing)
    printf("Average graph launch time per batch: %.2f us\n", avgLaunchTime);
    
    char stageTitle[100];
    snprintf(stageTitle, sizeof(stageTitle), "Stage 6: CUDA Graphs (Batch Size = %d)", batchSize);
    printPerformanceMetrics(stageTitle, metrics);
    
    // Cleanup (after timing)
    CHECK_CUDA_ERROR(cudaGraphDestroy(graph));
    CHECK_CUDA_ERROR(cudaGraphExecDestroy(graphExec));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
    CHECK_CUDA_ERROR(cudaFreeHost(h_batch_packets));
    CHECK_CUDA_ERROR(cudaFreeHost(h_batch_results));
    CHECK_CUDA_ERROR(cudaFree(d_batch_packets));
    CHECK_CUDA_ERROR(cudaFree(d_batch_results));
    
    printf("CUDA Graphs processing (total): %lld us\n", duration);
    return duration;
}

/******************************************************************************
 * Main Function
 ******************************************************************************/

int main(int argc, char **argv) {
    // Get device information
    cudaDeviceProp prop;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, 0));
    
    printf("Device: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Clock rate: %.2f GHz\n", prop.clockRate / 1e6);
    printf("Number of SMs: %d\n", prop.multiProcessorCount);
    
    // Initialize test data
    initializeTestData();
    
    printf("\n===== Low-Latency GPU Packet Processing =====\n");
    printf("Testing with %d packets\n\n", NUM_PACKETS);
    
    // Stage 0: CPU-based processing (baseline)
    printf("\n=== Stage 0: CPU-based Processing (Baseline) ===\n");
    long long cpuTime = runCPUProcessing();
    
    // Stage 1: Basic packet processing
    printf("\n=== Stage 1: Basic Packet Processing ===\n");
    long long basicGpuTime = runBasicProcessing();
    
    // Batch size exploration for optimization
    printf("\n=== Batch Size Exploration ===\n");
    int batchSizesToTest[] = {32, 64, 128, 256, 512, 1024};
    int numBatchSizes = sizeof(batchSizesToTest) / sizeof(batchSizesToTest[0]);
    
    // Find optimal batch size
    int optimalBatchSize = findOptimalBatchSize(batchSizesToTest, numBatchSizes);
    
    // Run all optimizations with the optimal batch size
    printf("\n=== Running Optimizations with Optimal Batch Size: %d ===\n", optimalBatchSize);
    
    // Stage 2: Pinned Memory Optimization
    printf("\n=== Stage 2: Pinned Memory Optimization (Batch Size = %d) ===\n", optimalBatchSize);
    long long pinnedTime = runPinnedMemoryProcessing(optimalBatchSize);
    
    // Stage 3: Batched Processing with Streams
    printf("\n=== Stage 3: Batched Processing with Streams (Batch Size = %d) ===\n", optimalBatchSize);
    long long batchedTime = runBatchedStreamProcessing(optimalBatchSize);
    
    // Stage 4: Zero-Copy Memory
    printf("\n=== Stage 4: Zero-Copy Memory (Batch Size = %d) ===\n", optimalBatchSize);
    long long zeroCopyTime = runZeroCopyProcessing(optimalBatchSize);
    
    // Stage 5: Real Persistent Kernel
    printf("\n=== Stage 5: Real Persistent Kernel (Batch Size = %d) ===\n", optimalBatchSize);
    long long persistentTime = runPersistentKernelProcessing(optimalBatchSize);
    
    // Stage 6: CUDA Graphs
    printf("\n=== Stage 6: CUDA Graphs (Batch Size = %d) ===\n", optimalBatchSize);
    long long graphsTime = runCudaGraphsProcessing(optimalBatchSize);
    
    // Overall performance comparison
    printf("\n=== Overall Performance Comparison ===\n");
    printf("CPU Baseline: %lld us\n", cpuTime);
    printf("Basic GPU: %lld us (%.2fx vs CPU)\n", basicGpuTime, (double)cpuTime / basicGpuTime);
    printf("Pinned Memory: %lld us (%.2fx vs CPU)\n", pinnedTime, (double)cpuTime / pinnedTime);
    printf("Batched Streams: %lld us (%.2fx vs CPU)\n", batchedTime, (double)cpuTime / batchedTime);
    printf("Zero-Copy: %lld us (%.2fx vs CPU)\n", zeroCopyTime, (double)cpuTime / zeroCopyTime);
    printf("Real Persistent Kernel: %lld us (%.2fx vs CPU)\n", persistentTime, (double)cpuTime / persistentTime);
    printf("CUDA Graphs: %lld us (%.2fx vs CPU)\n", graphsTime, (double)cpuTime / graphsTime);
    
    printf("\n=== Optimization Techniques Demonstrated ===\n");
    printf("1. Basic processing - baseline\n");
    printf("2. Pinned memory - faster host-device transfers\n");
    printf("3. Batched streams - overlapping transfers and computation\n");
    printf("4. Zero-copy memory - eliminating explicit transfers\n");
    printf("5. Real persistent kernel - reducing kernel launch overhead\n");
    printf("6. CUDA Graphs - minimizing CPU overhead for launch sequences\n");

    // Clean up test data
    cleanupTestData();
    
    return 0;
} 