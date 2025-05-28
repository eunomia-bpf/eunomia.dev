#ifndef PACKET_PROCESSING_KERNELS_CUH
#define PACKET_PROCESSING_KERNELS_CUH

#include "packet_processing_common.h"

// Kernel for basic packet processing
__global__ void processPacketsBasic(Packet* packets, PacketResult* results, int numPackets) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < numPackets) {
        Packet* packet = &packets[tid];
        PacketResult* result = &results[tid];
        
        // Call the core packet processing function
        processPacketGPU(packet, result, tid);
        
        // Mark packet as processed
        packet->status = COMPLETED;
    }
}

// Kernel for zero-copy processing
__global__ void processPacketsZeroCopy(Packet* packets, PacketResult* results, int numPackets) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < numPackets) {
        Packet* packet = &packets[tid];
        PacketResult* result = &results[tid];
        
        // Call the core packet processing function
        processPacketGPU(packet, result, tid);
        
        // Mark packet as processed directly in host memory
        packet->status = COMPLETED;
    }
}

#endif // PACKET_PROCESSING_KERNELS_CUH 