#ifndef PACKET_PROCESSING_COMMON_H
#define PACKET_PROCESSING_COMMON_H

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

// Timing utilities
#define START_TIMER auto start = std::chrono::high_resolution_clock::now();
#define STOP_TIMER(name) do { \
    auto end = std::chrono::high_resolution_clock::now(); \
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count(); \
    printf("%s: %lld us\n", name, duration); \
    return duration; \
} while(0)

// Packet processing constants
#define MAX_PACKET_SIZE 1500         // Maximum Ethernet packet size
#define PACKET_HEADER_SIZE 42        // Ethernet + IP + TCP headers
#define DEFAULT_BATCH_SIZE 256       // Default number of packets per batch
#define MAX_BATCHES 10               // Maximum number of batches in flight
#define NUM_PACKETS 10000            // Total packets to process
#define NUM_STREAMS 4                // Number of CUDA streams to use

// Status codes for packet processing
enum PacketStatus {
    PENDING = 0,
    PROCESSING = 1,
    COMPLETED = 2,
    ERROR = 3
};

// Simple packet structure
struct Packet {
    unsigned char header[PACKET_HEADER_SIZE];  // Ethernet + IP + TCP headers
    unsigned char payload[MAX_PACKET_SIZE - PACKET_HEADER_SIZE];
    int size;
    int status;
    long long enqueueTime;      // Time when packet was added to processing queue
    long long processingStart;  // Time when processing started
};

// Batch of packets
struct PacketBatch {
    Packet packets[DEFAULT_BATCH_SIZE];
    int count;
    int status;
    volatile int ready;
    long long submitTime;      // Time when batch was submitted for processing
    long long completionTime;  // Time when batch processing completed
};

// Results from packet processing
struct PacketResult {
    int packetId;
    int action;  // 0 = drop, 1 = forward, 2 = modify
    int matches; // Number of pattern matches found
    long long processingEnd;  // Time when processing ended
};

// Global state for persistent kernel
struct GlobalState {
    volatile int batchesReady;
    volatile int batchesCompleted;
    volatile int shutdown;
};

// Performance metrics
struct PerformanceMetrics {
    long long totalTime;         // Total processing time (microseconds)
    long long transferTime;      // Transfer time (microseconds) if applicable
    long long kernelTime;        // Kernel execution time if measured
    double avgBatchLatency;      // Average latency per batch
    double avgPacketLatency;     // Average latency per packet
    double minPacketLatency;     // Minimum packet latency observed
    double maxPacketLatency;     // Maximum packet latency observed
    int batchSize;               // Batch size used
    int numBatches;              // Number of batches processed
    int drops;                   // Number of dropped packets
    int forwards;                // Number of forwarded packets 
    int modifies;                // Number of modified packets
};

// Simple packet generator for testing
void generateTestPackets(Packet* packets, int numPackets) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(64, MAX_PACKET_SIZE);
    
    for (int i = 0; i < numPackets; i++) {
        // Generate random header (first 42 bytes)
        for (int j = 0; j < PACKET_HEADER_SIZE; j++) {
            packets[i].header[j] = rand() % 256;
        }
        
        // Set source and destination IP addresses for easy identification
        // Source IP in header bytes 26-29
        packets[i].header[26] = 10;
        packets[i].header[27] = 0;
        packets[i].header[28] = 0;
        packets[i].header[29] = 1;
        
        // Destination IP in header bytes 30-33
        packets[i].header[30] = 10;
        packets[i].header[31] = 0;
        packets[i].header[32] = 0;
        packets[i].header[33] = 2;
        
        // Random payload
        int payloadSize = dis(gen) - PACKET_HEADER_SIZE;
        for (int j = 0; j < payloadSize; j++) {
            packets[i].payload[j] = rand() % 256;
        }
        
        packets[i].size = PACKET_HEADER_SIZE + payloadSize;
        packets[i].status = PENDING;
        packets[i].enqueueTime = 0;
        packets[i].processingStart = 0;
    }
}

// CPU implementation of packet processing logic
void processPacketCPU(const Packet* packet, PacketResult* result, int packetId) {
    // Extract packet information
    result->packetId = packetId;
    result->matches = 0;
    
    // Simple pattern matching - count occurrences of byte value 0x42
    for (int i = 0; i < packet->size - PACKET_HEADER_SIZE; i++) {
        if (packet->payload[i] == 0x42) {
            result->matches++;
        }
    }
    
    // Decision logic
    if (result->matches > 5) {
        result->action = 0;  // Drop
    } else if (result->matches > 0) {
        result->action = 2;  // Modify
    } else {
        result->action = 1;  // Forward
    }
}

// Print performance metrics in a standardized format
void printPerformanceMetrics(const char* stageName, const PerformanceMetrics& metrics) {
    printf("Batch size: %d packets\n", metrics.batchSize);
    printf("Total time: %lld us\n", metrics.totalTime);
    
    if (metrics.transferTime > 0) {
        printf("Transfer time: %lld us (%.1f%%)\n", 
               metrics.transferTime, 
               (double)metrics.transferTime * 100.0 / metrics.totalTime);
    }
    
    if (metrics.kernelTime > 0) {
        printf("Kernel time: %lld us (%.1f%%)\n", 
               metrics.kernelTime,
               (double)metrics.kernelTime * 100.0 / metrics.totalTime);
    }
    
    if (metrics.avgBatchLatency > 0) {
        printf("Average latency per batch: %.2f us\n", metrics.avgBatchLatency);
    }
    
    printf("Average latency per packet: %.2f us\n", metrics.avgPacketLatency);
    
    if (metrics.minPacketLatency > 0 && metrics.maxPacketLatency > 0) {
        printf("Packet latency min/max: %.2f us / %.2f us\n", 
               metrics.minPacketLatency, metrics.maxPacketLatency);
    }
    
    printf("Processed %d packets: %d drops, %d forwards, %d modifies\n", 
           NUM_PACKETS, metrics.drops, metrics.forwards, metrics.modifies);
}

// GPU implementation of packet processing logic (device function)
__device__ void processPacketGPU(const Packet* packet, PacketResult* result, int packetId) {
    // Extract packet information
    result->packetId = packetId;
    result->matches = 0;
    
    // Simple pattern matching - count occurrences of byte value 0x42
    for (int i = 0; i < packet->size - PACKET_HEADER_SIZE; i++) {
        if (packet->payload[i] == 0x42) {
            result->matches++;
        }
    }
    
    // Decision logic
    if (result->matches > 5) {
        result->action = 0;  // Drop
    } else if (result->matches > 0) {
        result->action = 2;  // Modify
    } else {
        result->action = 1;  // Forward
    }
}

// Calculate results statistics
void calculateResults(const PacketResult* results, int numPackets, PerformanceMetrics& metrics) {
    metrics.drops = 0;
    metrics.forwards = 0;
    metrics.modifies = 0;
    
    for (int i = 0; i < numPackets; i++) {
        switch (results[i].action) {
            case 0: metrics.drops++; break;
            case 1: metrics.forwards++; break;
            case 2: metrics.modifies++; break;
        }
    }
}

#endif // PACKET_PROCESSING_COMMON_H 