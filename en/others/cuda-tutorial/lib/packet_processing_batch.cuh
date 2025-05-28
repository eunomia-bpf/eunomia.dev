#ifndef PACKET_PROCESSING_BATCH_CUH
#define PACKET_PROCESSING_BATCH_CUH

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
#include <limits.h>
#include "packet_processing_common.h"
#include "packet_processing_kernels.cuh"

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

// External declarations for global test data
extern Packet* g_test_packets;
extern PacketResult* g_test_results;

// Function to reset packet status
inline void resetPacketStatus() {
    for (int i = 0; i < NUM_PACKETS; i++) {
        g_test_packets[i].status = PENDING;
    }
}

// Kernel for basic packet processing
__global__ void processPacketsBasic(const Packet* packets, PacketResult* results, int numPackets) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < numPackets) {
        // Call the core packet processing function
        processPacketGPU(&packets[tid], &results[tid], tid);
    }
}

/******************************************************************************
 * Stage 1: Basic Packet Processing
 * 
 * This is a simple implementation that:
 * - Transfers packets from host to device
 * - Processes them on the GPU
 * - Transfers results back
 * 
 * This serves as our baseline GPU implementation.
 ******************************************************************************/

inline long long runBasicProcessing() {
    PerformanceMetrics metrics = {0};
    metrics.batchSize = NUM_PACKETS;  // Process all packets at once
    
    // Reset packet status
    resetPacketStatus();
    
    // Allocate device memory
    Packet* d_packets;
    PacketResult* d_results;
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_packets, NUM_PACKETS * sizeof(Packet)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_results, NUM_PACKETS * sizeof(PacketResult)));
    
    // Use manual timing for better control
    auto start = std::chrono::high_resolution_clock::now();
    
    // Copy packets to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_packets, g_test_packets, 
                     NUM_PACKETS * sizeof(Packet), cudaMemcpyHostToDevice));
    
    // Launch kernel to process packets
    int blockSize = 256;
    int numBlocks = (NUM_PACKETS + blockSize - 1) / blockSize;
    
    // Timing for kernel execution
    auto kernel_start = std::chrono::high_resolution_clock::now();
    processPacketsBasic<<<numBlocks, blockSize>>>(d_packets, d_results, NUM_PACKETS);
    
    // Wait for kernel to finish
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    auto kernel_end = std::chrono::high_resolution_clock::now();
    metrics.kernelTime = std::chrono::duration_cast<std::chrono::microseconds>(kernel_end - kernel_start).count();
    
    // Copy results back to host
    CHECK_CUDA_ERROR(cudaMemcpy(g_test_results, d_results, 
                     NUM_PACKETS * sizeof(PacketResult), 
                     cudaMemcpyDeviceToHost));
    
    // Calculate total time
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    metrics.totalTime = duration;
    
    // Calculate statistics
    calculateResults(g_test_results, NUM_PACKETS, metrics);
    
    // Print kernel execution time (after timing)
    printf("Kernel execution time: %lld us\n", metrics.kernelTime);
    
    // Print performance metrics
    printPerformanceMetrics("Stage 1: Basic Packet Processing", metrics);
    
    // Cleanup
    CHECK_CUDA_ERROR(cudaFree(d_packets));
    CHECK_CUDA_ERROR(cudaFree(d_results));
    
    printf("Basic processing (total): %lld us\n", duration);
    return duration;
}

/******************************************************************************
 * Stage 2: Batch Size Exploration
 * 
 * This version explores different batch sizes to find optimal performance.
 * It measures the impact of batch size on:
 * - Total processing time
 * - Transfer time
 * - Kernel execution time
 * - Latency per batch and per packet
 ******************************************************************************/

inline long long runBatchSizeExploration(int batchSize) {
    PerformanceMetrics metrics = {0};
    metrics.batchSize = batchSize;
    metrics.numBatches = (NUM_PACKETS + batchSize - 1) / batchSize;
    
    // Reset packet status
    resetPacketStatus();
    
    // Allocate device memory
    Packet* d_packets;
    PacketResult* d_results;
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_packets, batchSize * sizeof(Packet)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_results, batchSize * sizeof(PacketResult)));
    
    // Measure performance
    auto start = std::chrono::high_resolution_clock::now();
    
    long long total_transfer_time = 0;
    long long total_kernel_time = 0;
    
    // Process packets in batches
    for (int batch = 0; batch < metrics.numBatches; batch++) {
        int offset = batch * batchSize;
        int currentBatchSize = (batch == metrics.numBatches - 1) ? 
                             (NUM_PACKETS - batch * batchSize) : batchSize;
        
        // Copy batch to device
        auto transfer_start = std::chrono::high_resolution_clock::now();
        
        CHECK_CUDA_ERROR(cudaMemcpy(d_packets, g_test_packets + offset, 
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
        
        // Copy results back
        CHECK_CUDA_ERROR(cudaMemcpy(g_test_results + offset, d_results, 
                         currentBatchSize * sizeof(PacketResult), 
                         cudaMemcpyDeviceToHost));
    }
    
    // Get total time and update metrics
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    // Store metrics
    metrics.totalTime = duration;
    metrics.transferTime = total_transfer_time;
    metrics.kernelTime = total_kernel_time;
    
    // Calculate average latency
    metrics.avgBatchLatency = (double)metrics.totalTime / metrics.numBatches;
    metrics.avgPacketLatency = (double)metrics.totalTime / NUM_PACKETS;
    
    // Calculate statistics
    calculateResults(g_test_results, NUM_PACKETS, metrics);
    
    // Print timing results (after timing calculations)
    printf("Batch size %d: total=%lld us, transfer=%lld us, kernel=%lld us\n", 
           batchSize, metrics.totalTime, metrics.transferTime, metrics.kernelTime);
    printf("Average latency per batch: %.2f us, per packet: %.2f us\n", 
           metrics.avgBatchLatency, metrics.avgPacketLatency);
    
    // Print performance metrics
    char stageTitle[100];
    snprintf(stageTitle, sizeof(stageTitle), "Batch Size Exploration (Size = %d)", batchSize);
    printPerformanceMetrics(stageTitle, metrics);
    
    // Cleanup
    CHECK_CUDA_ERROR(cudaFree(d_packets));
    CHECK_CUDA_ERROR(cudaFree(d_results));
    
    return metrics.totalTime;
}

// Function to find the optimal batch size
inline int findOptimalBatchSize(int batchSizesToTest[], int numBatchSizes) {
    int optimalBatchSize = batchSizesToTest[0];
    long long bestTime = LLONG_MAX;
    
    for (int i = 0; i < numBatchSizes; i++) {
        int batchSize = batchSizesToTest[i];
        printf("\n--- Testing Batch Size = %d ---\n", batchSize);
        
        long long time = runBatchSizeExploration(batchSize);
        
        if (time < bestTime) {
            bestTime = time;
            optimalBatchSize = batchSize;
        }
    }
    
    printf("\n=== Optimal Batch Size Found: %d ===\n", optimalBatchSize);
    return optimalBatchSize;
}

#endif // PACKET_PROCESSING_BATCH_CUH 