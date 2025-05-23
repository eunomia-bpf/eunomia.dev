 /**
 * CUDA Basic Example 04 - GPU Organization Hierarchy
 * 
 * This example demonstrates and visualizes the hierarchical organization of:
 * 1. Thread/Block/Grid Structure
 * 2. Memory Hierarchy (Global, Shared, Local, Registers)
 * 3. Hardware Execution Model (SM, Warps)
 * 4. Data Access Patterns and Coalescing
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Print device properties
void printDeviceProperties() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    printf("=== GPU Hardware Organization ===\n\n");
    printf("Found %d CUDA device(s)\n\n", deviceCount);
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        printf("Device %d: %s\n", i, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  SMs (Multiprocessors): %d\n", prop.multiProcessorCount);
        printf("  Warp Size: %d threads\n", prop.warpSize);
        printf("  Max Threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
        printf("  Max Dimensions of a Block: (%d, %d, %d)\n", 
                prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  Max Dimensions of a Grid: (%d, %d, %d)\n", 
                prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        
        printf("\n  === Memory Hierarchy ===\n");
        printf("  Total Global Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  Shared Memory per Block: %lu KB\n", prop.sharedMemPerBlock / 1024);
        printf("  Registers per Block: %d\n", prop.regsPerBlock);
        printf("  L2 Cache Size: %d KB\n", prop.l2CacheSize / 1024);
        printf("  Memory Clock Rate: %.0f MHz\n", prop.memoryClockRate / 1000.0);
        printf("  Memory Bus Width: %d bits\n", prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth: %.2f GB/s\n\n", 
                2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    }
}

/**
 * Kernel to demonstrate Thread/Block/Grid hierarchy
 * Each thread prints its position in the hierarchy
 */
__global__ void threadHierarchyKernel() {
    // Thread ID within a block
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    
    // Block ID within a grid
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;
    
    // Block dimensions
    int bdx = blockDim.x;
    int bdy = blockDim.y;
    int bdz = blockDim.z;
    
    // Grid dimensions
    int gdx = gridDim.x;
    int gdy = gridDim.y;
    int gdz = gridDim.z;
    
    // Calculate unique thread ID
    int threadId = (bz * gdy * gdx + by * gdx + bx) * (bdz * bdy * bdx) + (tz * bdy * bdx + ty * bdx + tx);
    
    // Only some threads print to avoid overwhelming output
    if (threadId % 100 == 0) {
        printf("Thread ID: %d, Position: (%d,%d,%d) in Block (%d,%d,%d) of Grid (%d,%d,%d)\n",
                threadId, tx, ty, tz, bx, by, bz, gdx, gdy, gdz);
    }
}

/**
 * Kernel to demonstrate memory access patterns and coalescing
 * Shows the difference between coalesced and non-coalesced memory access
 */
__global__ void memoryCoalescingKernel(float *data, float *coalesced_output, float *noncoalesced_output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        // Coalesced memory access (threads access adjacent memory locations)
        coalesced_output[tid] = data[tid] * 2.0f;
        
        // Non-coalesced memory access (threads access memory with stride)
        noncoalesced_output[tid] = data[tid * 8 % n] * 2.0f;
    }
}

/**
 * Kernel to demonstrate shared memory usage
 */
__global__ void sharedMemoryKernel(float *input, float *output, int n) {
    // Declare shared memory
    __shared__ float sharedData[256];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int localId = threadIdx.x;
    
    // Load data from global to shared memory
    if (tid < n) {
        sharedData[localId] = input[tid];
    }
    
    // Synchronize to ensure all data is loaded
    __syncthreads();
    
    // Process data in shared memory
    if (tid < n && localId < 255) {
        output[tid] = sharedData[localId] + sharedData[localId + 1];
    } else if (tid < n) {
        output[tid] = sharedData[localId];
    }
}

/**
 * Kernel to demonstrate warp execution
 */
__global__ void warpExecutionKernel(int *output) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int laneId = threadIdx.x % 32; // Lane ID within warp
    int warpId = tid / 32;         // Global warp ID
    
    // Each thread records its warp and lane ID
    output[tid * 2] = warpId;
    output[tid * 2 + 1] = laneId;
    
    // Demonstrate warp divergence
    if (laneId < 16) {
        // First half of the warp does this
        output[tid * 2] += 10000;
    } else {
        // Second half of the warp does this
        output[tid * 2] += 20000;
    }
}

/**
 * Kernel to demonstrate register usage
 */
__global__ void registerUsageKernel(float *input, float *output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Use many registers to demonstrate their effects
    float r0 = 0.0f, r1 = 1.0f, r2 = 2.0f, r3 = 3.0f;
    float r4 = 4.0f, r5 = 5.0f, r6 = 6.0f, r7 = 7.0f;
    float r8 = 8.0f, r9 = 9.0f, r10 = 10.0f, r11 = 11.0f;
    float r12 = 12.0f, r13 = 13.0f, r14 = 14.0f, r15 = 15.0f;
    
    if (tid < n) {
        float val = input[tid];
        
        // Complex computation to force register usage
        val = val + r0 * r1 / r2 + r3;
        val = val + r4 * r5 / r6 + r7;
        val = val + r8 * r9 / r10 + r11;
        val = val + r12 * r13 / r14 + r15;
        
        output[tid] = val;
    }
}

/**
 * Demonstration of various memory types
 */
__constant__ float constData[256]; // Constant memory

__global__ void memoryTypesKernel(float *input, float *output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Local memory (automatically used for large local variables)
    float localArray[32]; // This might be placed in local memory if register pressure is high
    
    // Shared memory
    __shared__ float sharedData[256];
    
    // Register (automatic variables)
    float regVar = 1.5f;
    
    if (tid < n) {
        // Initialize local array
        for (int i = 0; i < 32; i++) {
            localArray[i] = i * 0.1f;
        }
        
        // Load data to shared memory
        sharedData[threadIdx.x] = input[tid];
        
        // Synchronize
        __syncthreads();
        
        // Use all memory types
        float result = sharedData[threadIdx.x] * regVar * constData[tid % 256];
        for (int i = 0; i < 32; i++) {
            result += localArray[i];
        }
        
        output[tid] = result;
    }
}

/**
 * Host function to setup and launch various kernels
 */
void exploreGPUHierarchy() {
    printf("\n=== Thread/Block/Grid Hierarchy ===\n\n");
    
    // Define a 3D grid and block structure
    dim3 block(4, 4, 4);  // 4x4x4 threads per block = 64 threads
    dim3 grid(4, 4, 4);   // 4x4x4 blocks per grid = 64 blocks
    
    printf("Launching kernel with Grid(%d,%d,%d) and Block(%d,%d,%d)...\n", 
           grid.x, grid.y, grid.z, block.x, block.y, block.z);
    printf("Total threads: %d\n\n", grid.x * grid.y * grid.z * block.x * block.y * block.z);
    
    // Launch the kernel
    threadHierarchyKernel<<<grid, block>>>();
    cudaDeviceSynchronize();
    
    // Memory coalescing demonstration
    printf("\n=== Memory Coalescing Demonstration ===\n\n");
    
    int n = 1024 * 1024;  // 1M elements
    size_t size = n * sizeof(float);
    
    float *h_data = (float*)malloc(size);
    float *h_output1 = (float*)malloc(size);
    float *h_output2 = (float*)malloc(size);
    
    // Initialize data
    for (int i = 0; i < n; i++) {
        h_data[i] = (float)i;
    }
    
    float *d_data, *d_output1, *d_output2;
    cudaMalloc(&d_data, size);
    cudaMalloc(&d_output1, size);
    cudaMalloc(&d_output2, size);
    
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
    
    // Setup execution parameters
    block = dim3(256, 1, 1);
    grid = dim3((n + block.x - 1) / block.x, 1, 1);
    
    // Measure performance of coalesced vs non-coalesced
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsedCoalesced, elapsedNonCoalesced;
    
    // Time coalesced access
    cudaEventRecord(start);
    memoryCoalescingKernel<<<grid, block>>>(d_data, d_output1, d_output2, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedCoalesced, start, stop);
    
    // Time non-coalesced access by measuring the whole kernel again
    // In a real scenario, you'd use separate kernels
    cudaEventRecord(start);
    memoryCoalescingKernel<<<grid, block>>>(d_data, d_output1, d_output2, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedNonCoalesced, start, stop);
    
    printf("Memory access timing:\n");
    printf("  Elapsed time: %.3f ms\n", elapsedCoalesced);
    printf("  Note: Coalesced access is typically much faster than non-coalesced access.\n");
    printf("  However, this simplified example may not show a large difference.\n");
    
    // Cleanup
    cudaFree(d_data);
    cudaFree(d_output1);
    cudaFree(d_output2);
    free(h_data);
    free(h_output1);
    free(h_output2);
    
    // Shared memory demonstration
    printf("\n=== Shared Memory Demonstration ===\n\n");
    
    n = 1024;  // Smaller size for simplicity
    size = n * sizeof(float);
    
    h_data = (float*)malloc(size);
    h_output1 = (float*)malloc(size);
    
    // Initialize data
    for (int i = 0; i < n; i++) {
        h_data[i] = (float)i;
    }
    
    cudaMalloc(&d_data, size);
    cudaMalloc(&d_output1, size);
    
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
    
    // Launch kernel
    block = dim3(256, 1, 1);
    grid = dim3((n + block.x - 1) / block.x, 1, 1);
    
    sharedMemoryKernel<<<grid, block>>>(d_data, d_output1, n);
    
    cudaMemcpy(h_output1, d_output1, size, cudaMemcpyDeviceToHost);
    
    printf("Shared memory allows threads in a block to share data.\n");
    printf("First few output values (should be sums of adjacent input values):\n");
    for (int i = 0; i < 5; i++) {
        printf("  output[%d] = %.1f\n", i, h_output1[i]);
    }
    
    // Cleanup
    cudaFree(d_data);
    cudaFree(d_output1);
    free(h_data);
    free(h_output1);
    
    // Warp execution demonstration
    printf("\n=== Warp Execution Demonstration ===\n\n");
    
    n = 256;  // 8 warps (assuming 32 threads per warp)
    int *h_warpData = (int*)malloc(n * 2 * sizeof(int));
    int *d_warpData;
    
    cudaMalloc(&d_warpData, n * 2 * sizeof(int));
    
    block = dim3(n, 1, 1);  // All threads in one block
    grid = dim3(1, 1, 1);
    
    warpExecutionKernel<<<grid, block>>>(d_warpData);
    
    cudaMemcpy(h_warpData, d_warpData, n * 2 * sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("Warp and lane IDs for a subset of threads:\n");
    for (int i = 0; i < n; i += 32) {
        printf("  Thread %3d: Warp %5d, Lane %2d\n", 
               i, h_warpData[i * 2], h_warpData[i * 2 + 1]);
    }
    
    printf("\nNote on warp divergence:\n");
    printf("  When threads within a warp take different paths (diverge),\n");
    printf("  the warp executes both paths, masking threads as appropriate.\n");
    printf("  This reduces efficiency. The kernel demonstrates this with\n");
    printf("  different operations for first and second half of each warp.\n");
    
    // Cleanup
    cudaFree(d_warpData);
    free(h_warpData);
    
    // Register usage demonstration
    printf("\n=== Register Usage Demonstration ===\n\n");
    
    printf("The register usage kernel intentionally uses many registers per thread.\n");
    printf("This can be verified with the compiler option --ptxas-options=-v\n");
    printf("High register usage may limit occupancy (active warps per SM).\n");
    
    // Memory types demonstration
    printf("\n=== Memory Types Demonstration ===\n\n");
    
    printf("CUDA offers several memory types with different characteristics:\n");
    printf("  - Global memory: Large, high latency, accessible by all threads\n");
    printf("  - Shared memory: Small, low latency, accessible by threads in a block\n");
    printf("  - Constant memory: Small, cached, read-only, accessible by all threads\n");
    printf("  - Local memory: Per-thread, spills to global memory\n");
    printf("  - Registers: Fastest, limited number per thread\n");
}

int main() {
    // Print device properties
    printDeviceProperties();
    
    // Explore GPU Hierarchy
    exploreGPUHierarchy();
    
    printf("\n=== Program Complete ===\n");
    return 0;
}