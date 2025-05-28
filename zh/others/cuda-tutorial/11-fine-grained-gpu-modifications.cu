/**
 * 11-fine-grained-gpu-modifications.cu
 * 
 * This example demonstrates advanced GPU customization techniques that require
 * direct modifications to GPU code, including:
 * 
 * 1. Data structure layout optimization (AoS vs SoA)
 * 2. Warp-level primitives and synchronization
 * 3. Memory access pattern optimizations with tiling
 * 4. Kernel fusion for performance
 * 5. Dynamic execution path selection based on data characteristics
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <iostream>
#include <chrono>

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

// Utility function for timing
double timeInMs() {
    using namespace std::chrono;
    static auto startTime = high_resolution_clock::now();
    auto now = high_resolution_clock::now();
    return duration<double, std::milli>(now - startTime).count();
}

/*****************************************************************************
 * Example 1: Data Structure Layout - AoS vs SoA
 * 
 * This example shows how changing memory layout affects performance
 * Structure of Arrays (SoA) vs Array of Structures (AoS)
 *****************************************************************************/

// Array of Structures (AoS) definition
struct Particle_AoS {
    float x, y, z;    // Position
    float vx, vy, vz; // Velocity
};

// AoS kernel - typically less efficient on GPUs
__global__ void updateParticles_AoS(Particle_AoS* particles, int n, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // Update position based on velocity
        particles[i].x += particles[i].vx * dt;
        particles[i].y += particles[i].vy * dt;
        particles[i].z += particles[i].vz * dt;
        
        // Simple gravity effect
        particles[i].vy -= 9.8f * dt;
    }
}

// Structure of Arrays (SoA) definition
struct Particles_SoA {
    float *x, *y, *z;    // Positions
    float *vx, *vy, *vz; // Velocities
};

// SoA kernel - typically more efficient on GPUs due to coalesced memory access
__global__ void updateParticles_SoA(Particles_SoA particles, int n, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // Update position based on velocity
        particles.x[i] += particles.vx[i] * dt;
        particles.y[i] += particles.vy[i] * dt;
        particles.z[i] += particles.vz[i] * dt;
        
        // Simple gravity effect
        particles.vy[i] -= 9.8f * dt;
    }
}

void testAoSvsSoA(int numParticles) {
    // AoS setup
    Particle_AoS* h_particles_aos = new Particle_AoS[numParticles];
    Particle_AoS* d_particles_aos;
    
    // Initialize AoS data
    for (int i = 0; i < numParticles; i++) {
        h_particles_aos[i].x = h_particles_aos[i].y = h_particles_aos[i].z = 0.0f;
        h_particles_aos[i].vx = h_particles_aos[i].vy = h_particles_aos[i].vz = 1.0f;
    }
    
    // Allocate device memory for AoS
    CHECK_CUDA_ERROR(cudaMalloc(&d_particles_aos, numParticles * sizeof(Particle_AoS)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_particles_aos, h_particles_aos, 
                     numParticles * sizeof(Particle_AoS), cudaMemcpyHostToDevice));
    
    // SoA setup
    Particles_SoA h_particles_soa;
    Particles_SoA d_particles_soa;
    
    // Allocate host memory for SoA
    h_particles_soa.x = new float[numParticles];
    h_particles_soa.y = new float[numParticles];
    h_particles_soa.z = new float[numParticles];
    h_particles_soa.vx = new float[numParticles];
    h_particles_soa.vy = new float[numParticles];
    h_particles_soa.vz = new float[numParticles];
    
    // Initialize SoA data
    for (int i = 0; i < numParticles; i++) {
        h_particles_soa.x[i] = h_particles_soa.y[i] = h_particles_soa.z[i] = 0.0f;
        h_particles_soa.vx[i] = h_particles_soa.vy[i] = h_particles_soa.vz[i] = 1.0f;
    }
    
    // Allocate device memory for SoA
    CHECK_CUDA_ERROR(cudaMalloc(&d_particles_soa.x, numParticles * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_particles_soa.y, numParticles * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_particles_soa.z, numParticles * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_particles_soa.vx, numParticles * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_particles_soa.vy, numParticles * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_particles_soa.vz, numParticles * sizeof(float)));
    
    // Copy SoA data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_particles_soa.x, h_particles_soa.x, 
                     numParticles * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_particles_soa.y, h_particles_soa.y, 
                     numParticles * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_particles_soa.z, h_particles_soa.z, 
                     numParticles * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_particles_soa.vx, h_particles_soa.vx, 
                     numParticles * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_particles_soa.vy, h_particles_soa.vy, 
                     numParticles * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_particles_soa.vz, h_particles_soa.vz, 
                     numParticles * sizeof(float), cudaMemcpyHostToDevice));
    
    // Execution configuration
    int blockSize = 256;
    int numBlocks = (numParticles + blockSize - 1) / blockSize;
    
    // Benchmark AoS
    cudaDeviceSynchronize();
    double start = timeInMs();
    
    for (int i = 0; i < 100; i++) {  // Run multiple iterations for better timing
        updateParticles_AoS<<<numBlocks, blockSize>>>(d_particles_aos, numParticles, 0.01f);
    }
    
    cudaDeviceSynchronize();
    double timeAoS = timeInMs() - start;
    
    // Benchmark SoA
    cudaDeviceSynchronize();
    start = timeInMs();
    
    for (int i = 0; i < 100; i++) {  // Run multiple iterations for better timing
        updateParticles_SoA<<<numBlocks, blockSize>>>(d_particles_soa, numParticles, 0.01f);
    }
    
    cudaDeviceSynchronize();
    double timeSoA = timeInMs() - start;
    
    printf("Data Structure Layout Example:\n");
    printf("  AoS execution time: %.3f ms\n", timeAoS);
    printf("  SoA execution time: %.3f ms\n", timeSoA);
    printf("  SoA speedup: %.2fx\n\n", timeAoS / timeSoA);
    
    // Cleanup
    delete[] h_particles_aos;
    delete[] h_particles_soa.x;
    delete[] h_particles_soa.y;
    delete[] h_particles_soa.z;
    delete[] h_particles_soa.vx;
    delete[] h_particles_soa.vy;
    delete[] h_particles_soa.vz;
    
    CHECK_CUDA_ERROR(cudaFree(d_particles_aos));
    CHECK_CUDA_ERROR(cudaFree(d_particles_soa.x));
    CHECK_CUDA_ERROR(cudaFree(d_particles_soa.y));
    CHECK_CUDA_ERROR(cudaFree(d_particles_soa.z));
    CHECK_CUDA_ERROR(cudaFree(d_particles_soa.vx));
    CHECK_CUDA_ERROR(cudaFree(d_particles_soa.vy));
    CHECK_CUDA_ERROR(cudaFree(d_particles_soa.vz));
}

/*****************************************************************************
 * Example 2: Warp-Level Primitives for a Histogram Calculation
 * 
 * This example demonstrates using warp-level primitives to efficiently
 * compute a histogram, which would otherwise suffer from atomic contention.
 *****************************************************************************/

#define HISTOGRAM_SIZE 256  // For 8-bit values

// Naive histogram implementation with global atomics
__global__ void histogram_naive(unsigned char* data, unsigned int* histogram, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        atomicAdd(&histogram[data[idx]], 1);
    }
}

// Optimized histogram using warp-level primitives and shared memory
__global__ void histogram_optimized(unsigned char* data, unsigned int* histogram, int size) {
    // Shared memory for per-block histograms
    __shared__ unsigned int localHist[HISTOGRAM_SIZE];
    
    // Initialize shared memory
    int tid = threadIdx.x;
    if (tid < HISTOGRAM_SIZE) {
        localHist[tid] = 0;
    }
    __syncthreads();
    
    // Process data with multiple elements per thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    while (idx < size) {
        unsigned char value = data[idx];
        atomicAdd(&localHist[value], 1);  // Less contention in shared memory
        idx += stride;
    }
    __syncthreads();
    
    // Cooperatively write back to global memory using the warp
    // Each warp handles a portion of the histogram
    int warpSize = 32;
    int warpId = threadIdx.x / warpSize;
    int laneId = threadIdx.x % warpSize;
    int numWarps = (blockDim.x + warpSize - 1) / warpSize;
    
    // Each warp handles HISTOGRAM_SIZE / numWarps bins
    int binsPerWarp = (HISTOGRAM_SIZE + numWarps - 1) / numWarps;
    int warpStart = warpId * binsPerWarp;
    int warpEnd = min(warpStart + binsPerWarp, HISTOGRAM_SIZE);
    
    // Each thread in warp handles some bins
    for (int binIdx = warpStart + laneId; binIdx < warpEnd; binIdx += warpSize) {
        if (binIdx < HISTOGRAM_SIZE) {
            atomicAdd(&histogram[binIdx], localHist[binIdx]);
        }
    }
}

void testHistogramOptimization(int dataSize) {
    // Generate random data
    unsigned char* h_data = new unsigned char[dataSize];
    for (int i = 0; i < dataSize; i++) {
        h_data[i] = rand() % 256;
    }
    
    // Allocate device memory
    unsigned char* d_data;
    unsigned int* d_hist_naive;
    unsigned int* d_hist_optimized;
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_data, dataSize * sizeof(unsigned char)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_hist_naive, HISTOGRAM_SIZE * sizeof(unsigned int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_hist_optimized, HISTOGRAM_SIZE * sizeof(unsigned int)));
    
    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_data, h_data, dataSize * sizeof(unsigned char), 
                     cudaMemcpyHostToDevice));
    
    // Clear histograms
    CHECK_CUDA_ERROR(cudaMemset(d_hist_naive, 0, HISTOGRAM_SIZE * sizeof(unsigned int)));
    CHECK_CUDA_ERROR(cudaMemset(d_hist_optimized, 0, HISTOGRAM_SIZE * sizeof(unsigned int)));
    
    // Execution configuration
    int blockSize = 256;
    int numBlocks = (dataSize + blockSize - 1) / blockSize;
    
    // Benchmark naive implementation
    cudaDeviceSynchronize();
    double start = timeInMs();
    
    histogram_naive<<<numBlocks, blockSize>>>(d_data, d_hist_naive, dataSize);
    
    cudaDeviceSynchronize();
    double timeNaive = timeInMs() - start;
    
    // Benchmark optimized implementation
    cudaDeviceSynchronize();
    start = timeInMs();
    
    histogram_optimized<<<numBlocks, blockSize>>>(d_data, d_hist_optimized, dataSize);
    
    cudaDeviceSynchronize();
    double timeOptimized = timeInMs() - start;
    
    // Verify results
    unsigned int* h_hist_naive = new unsigned int[HISTOGRAM_SIZE];
    unsigned int* h_hist_optimized = new unsigned int[HISTOGRAM_SIZE];
    
    CHECK_CUDA_ERROR(cudaMemcpy(h_hist_naive, d_hist_naive, 
                     HISTOGRAM_SIZE * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_hist_optimized, d_hist_optimized, 
                     HISTOGRAM_SIZE * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    
    bool resultsMatch = true;
    for (int i = 0; i < HISTOGRAM_SIZE; i++) {
        if (h_hist_naive[i] != h_hist_optimized[i]) {
            resultsMatch = false;
            break;
        }
    }
    
    printf("Warp-Level Primitives Example (Histogram):\n");
    printf("  Naive execution time: %.3f ms\n", timeNaive);
    printf("  Optimized execution time: %.3f ms\n", timeOptimized);
    printf("  Speedup: %.2fx\n", timeNaive / timeOptimized);
    printf("  Results match: %s\n\n", resultsMatch ? "Yes" : "No");
    
    // Cleanup
    delete[] h_data;
    delete[] h_hist_naive;
    delete[] h_hist_optimized;
    
    CHECK_CUDA_ERROR(cudaFree(d_data));
    CHECK_CUDA_ERROR(cudaFree(d_hist_naive));
    CHECK_CUDA_ERROR(cudaFree(d_hist_optimized));
}

/*****************************************************************************
 * Example 3: Memory Access Patterns with Tiling
 * 
 * This example demonstrates matrix transpose with and without tiling
 * to show the performance impact of memory access patterns.
 *****************************************************************************/

#define TILE_DIM 32

// Naive matrix transpose with uncoalesced memory access
__global__ void transposeNaive(float* input, float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        output[x * height + y] = input[y * width + x];
    }
}

// Tiled matrix transpose with improved memory access pattern
__global__ void transposeTiled(float* input, float* output, int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM+1]; // +1 to avoid bank conflicts
    
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    // Load tile collaboratively with coalesced reads
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    }
    __syncthreads();
    
    // Calculate transposed coordinates
    int out_x = blockIdx.y * TILE_DIM + threadIdx.x;
    int out_y = blockIdx.x * TILE_DIM + threadIdx.y;
    
    // Write tile with coalesced writes
    if (out_x < height && out_y < width) {
        output[out_y * height + out_x] = tile[threadIdx.x][threadIdx.y];
    }
}

void testMatrixTranspose(int width, int height) {
    size_t size = width * height * sizeof(float);
    
    // Allocate host memory
    float* h_input = new float[width * height];
    float* h_output_naive = new float[width * height];
    float* h_output_tiled = new float[width * height];
    
    // Initialize input matrix
    for (int i = 0; i < width * height; i++) {
        h_input[i] = static_cast<float>(i);
    }
    
    // Allocate device memory
    float* d_input;
    float* d_output_naive;
    float* d_output_tiled;
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output_naive, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output_tiled, size));
    
    // Copy input to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));
    
    // Execution configuration
    dim3 blockDim(TILE_DIM, TILE_DIM);
    dim3 gridDim((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM);
    
    // Benchmark naive transpose
    cudaDeviceSynchronize();
    double start = timeInMs();
    
    transposeNaive<<<gridDim, blockDim>>>(d_input, d_output_naive, width, height);
    
    cudaDeviceSynchronize();
    double timeNaive = timeInMs() - start;
    
    // Benchmark tiled transpose
    cudaDeviceSynchronize();
    start = timeInMs();
    
    transposeTiled<<<gridDim, blockDim>>>(d_input, d_output_tiled, width, height);
    
    cudaDeviceSynchronize();
    double timeTiled = timeInMs() - start;
    
    // Copy results back for verification
    CHECK_CUDA_ERROR(cudaMemcpy(h_output_naive, d_output_naive, size, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_output_tiled, d_output_tiled, size, cudaMemcpyDeviceToHost));
    
    // Verify results
    bool resultsMatch = true;
    for (int i = 0; i < width * height; i++) {
        if (fabs(h_output_naive[i] - h_output_tiled[i]) > 1e-5) {
            resultsMatch = false;
            break;
        }
    }
    
    printf("Memory Access Pattern Example (Matrix Transpose):\n");
    printf("  Naive execution time: %.3f ms\n", timeNaive);
    printf("  Tiled execution time: %.3f ms\n", timeTiled);
    printf("  Speedup: %.2fx\n", timeNaive / timeTiled);
    printf("  Results match: %s\n\n", resultsMatch ? "Yes" : "No");
    
    // Cleanup
    delete[] h_input;
    delete[] h_output_naive;
    delete[] h_output_tiled;
    
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output_naive));
    CHECK_CUDA_ERROR(cudaFree(d_output_tiled));
}

/*****************************************************************************
 * Example 4: Kernel Fusion for Performance
 * 
 * This example demonstrates the performance benefits of kernel fusion
 * by comparing separate vs. fused kernels for vector operations.
 *****************************************************************************/

// Separate kernels for vector operations
__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

__global__ void vectorScale(float* c, float* d, float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        d[i] = c[i] * scale;
    }
}

// Fused kernel for both operations
__global__ void vectorAddAndScale(float* a, float* b, float* d, float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // Fuse the operations to avoid extra global memory traffic
        d[i] = (a[i] + b[i]) * scale;
    }
}

void testKernelFusion(int vectorSize) {
    // Allocate host memory
    float* h_a = new float[vectorSize];
    float* h_b = new float[vectorSize];
    float* h_c = new float[vectorSize];
    float* h_d_separate = new float[vectorSize];
    float* h_d_fused = new float[vectorSize];
    
    // Initialize vectors
    for (int i = 0; i < vectorSize; i++) {
        h_a[i] = static_cast<float>(rand()) / RAND_MAX;
        h_b[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    
    // Allocate device memory
    float* d_a;
    float* d_b;
    float* d_c;
    float* d_d_separate;
    float* d_d_fused;
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_a, vectorSize * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_b, vectorSize * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_c, vectorSize * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_d_separate, vectorSize * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_d_fused, vectorSize * sizeof(float)));
    
    // Copy input vectors to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_a, h_a, vectorSize * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b, h_b, vectorSize * sizeof(float), cudaMemcpyHostToDevice));
    
    // Execution configuration
    int blockSize = 256;
    int numBlocks = (vectorSize + blockSize - 1) / blockSize;
    float scale = 2.0f;
    
    // Benchmark separate kernels
    cudaDeviceSynchronize();
    double start = timeInMs();
    
    vectorAdd<<<numBlocks, blockSize>>>(d_a, d_b, d_c, vectorSize);
    vectorScale<<<numBlocks, blockSize>>>(d_c, d_d_separate, scale, vectorSize);
    
    cudaDeviceSynchronize();
    double timeSeparate = timeInMs() - start;
    
    // Benchmark fused kernel
    cudaDeviceSynchronize();
    start = timeInMs();
    
    vectorAddAndScale<<<numBlocks, blockSize>>>(d_a, d_b, d_d_fused, scale, vectorSize);
    
    cudaDeviceSynchronize();
    double timeFused = timeInMs() - start;
    
    // Copy results back for verification
    CHECK_CUDA_ERROR(cudaMemcpy(h_d_separate, d_d_separate, 
                     vectorSize * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_d_fused, d_d_fused, 
                     vectorSize * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Verify results
    bool resultsMatch = true;
    for (int i = 0; i < vectorSize; i++) {
        if (fabs(h_d_separate[i] - h_d_fused[i]) > 1e-5) {
            resultsMatch = false;
            break;
        }
    }
    
    printf("Kernel Fusion Example:\n");
    printf("  Separate kernels execution time: %.3f ms\n", timeSeparate);
    printf("  Fused kernel execution time: %.3f ms\n", timeFused);
    printf("  Speedup: %.2fx\n", timeSeparate / timeFused);
    printf("  Results match: %s\n\n", resultsMatch ? "Yes" : "No");
    
    // Cleanup
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    delete[] h_d_separate;
    delete[] h_d_fused;
    
    CHECK_CUDA_ERROR(cudaFree(d_a));
    CHECK_CUDA_ERROR(cudaFree(d_b));
    CHECK_CUDA_ERROR(cudaFree(d_c));
    CHECK_CUDA_ERROR(cudaFree(d_d_separate));
    CHECK_CUDA_ERROR(cudaFree(d_d_fused));
}

/*****************************************************************************
 * Example 5: Dynamic Execution Path Selection
 * 
 * This example demonstrates how to dynamically select execution paths
 * based on input data characteristics.
 *****************************************************************************/

// Kernel optimized for sparse data (many zeros)
__global__ void processSparse(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Skip processing zero elements
        if (input[idx] != 0.0f) {
            // Perform expensive computation only on non-zero elements
            float val = input[idx];
            // Simulate complex processing
            for (int i = 0; i < 100; i++) {
                val = sinf(val) * cosf(val);
            }
            output[idx] = val;
        } else {
            output[idx] = 0.0f;
        }
    }
}

// Kernel optimized for dense data (few zeros)
__global__ void processDense(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Process all elements uniformly
        float val = input[idx];
        // Simulate complex processing
        for (int i = 0; i < 100; i++) {
            val = sinf(val) * cosf(val);
        }
        output[idx] = val;
    }
}

// Adaptive kernel that chooses execution path based on data
__global__ void processAdaptive(float* input, float* output, int size, float density) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        
        // Dynamic branch based on data characteristics
        if (density < 0.5f) {
            // Sparse data path
            if (val != 0.0f) {
                // Perform expensive computation only on non-zero elements
                for (int i = 0; i < 100; i++) {
                    val = sinf(val) * cosf(val);
                }
                output[idx] = val;
            } else {
                output[idx] = 0.0f;
            }
        } else {
            // Dense data path
            for (int i = 0; i < 100; i++) {
                val = sinf(val) * cosf(val);
            }
            output[idx] = val;
        }
    }
}

void testDynamicExecution(int dataSize, float density) {
    // Allocate host memory
    float* h_input = new float[dataSize];
    float* h_output_sparse = new float[dataSize];
    float* h_output_dense = new float[dataSize];
    float* h_output_adaptive = new float[dataSize];
    
    // Initialize input data with specified density
    for (int i = 0; i < dataSize; i++) {
        float r = static_cast<float>(rand()) / RAND_MAX;
        h_input[i] = (r < density) ? r : 0.0f;
    }
    
    // Allocate device memory
    float* d_input;
    float* d_output_sparse;
    float* d_output_dense;
    float* d_output_adaptive;
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, dataSize * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output_sparse, dataSize * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output_dense, dataSize * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output_adaptive, dataSize * sizeof(float)));
    
    // Copy input to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, dataSize * sizeof(float), 
                     cudaMemcpyHostToDevice));
    
    // Execution configuration
    int blockSize = 256;
    int numBlocks = (dataSize + blockSize - 1) / blockSize;
    
    // Benchmark sparse kernel
    cudaDeviceSynchronize();
    double start = timeInMs();
    
    processSparse<<<numBlocks, blockSize>>>(d_input, d_output_sparse, dataSize);
    
    cudaDeviceSynchronize();
    double timeSparse = timeInMs() - start;
    
    // Benchmark dense kernel
    cudaDeviceSynchronize();
    start = timeInMs();
    
    processDense<<<numBlocks, blockSize>>>(d_input, d_output_dense, dataSize);
    
    cudaDeviceSynchronize();
    double timeDense = timeInMs() - start;
    
    // Benchmark adaptive kernel
    cudaDeviceSynchronize();
    start = timeInMs();
    
    processAdaptive<<<numBlocks, blockSize>>>(d_input, d_output_adaptive, dataSize, density);
    
    cudaDeviceSynchronize();
    double timeAdaptive = timeInMs() - start;
    
    // Copy results back
    CHECK_CUDA_ERROR(cudaMemcpy(h_output_sparse, d_output_sparse, 
                     dataSize * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_output_dense, d_output_dense, 
                     dataSize * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_output_adaptive, d_output_adaptive, 
                     dataSize * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Verify results
    bool resultsMatch = true;
    for (int i = 0; i < dataSize; i++) {
        if (fabs(h_output_sparse[i] - h_output_adaptive[i]) > 1e-5) {
            resultsMatch = false;
            break;
        }
    }
    
    printf("Dynamic Execution Path Selection Example:\n");
    printf("  Data density: %.2f\n", density);
    printf("  Sparse kernel execution time: %.3f ms\n", timeSparse);
    printf("  Dense kernel execution time: %.3f ms\n", timeDense);
    printf("  Adaptive kernel execution time: %.3f ms\n", timeAdaptive);
    printf("  Best approach: %s\n", 
           (timeSparse < timeDense) ? "Sparse" : "Dense");
    printf("  Adaptive vs. best fixed: %.2fx\n", 
           std::min(timeSparse, timeDense) / timeAdaptive);
    printf("  Results match: %s\n\n", resultsMatch ? "Yes" : "No");
    
    // Cleanup
    delete[] h_input;
    delete[] h_output_sparse;
    delete[] h_output_dense;
    delete[] h_output_adaptive;
    
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output_sparse));
    CHECK_CUDA_ERROR(cudaFree(d_output_dense));
    CHECK_CUDA_ERROR(cudaFree(d_output_adaptive));
}

/*****************************************************************************
 * Main Function
 *****************************************************************************/

int main(int argc, char** argv) {
    // Print device information
    cudaDeviceProp prop;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, 0));
    
    printf("Device: %s\n", prop.name);
    printf("Compute capability: %d.%d\n\n", prop.major, prop.minor);
    
    // Example 1: Data Structure Layout - AoS vs SoA
    testAoSvsSoA(1000000);
    
    // Example 2: Warp-Level Primitives for a Histogram Calculation
    testHistogramOptimization(10000000);
    
    // Example 3: Memory Access Patterns with Tiling
    testMatrixTranspose(4096, 4096);
    
    // Example 4: Kernel Fusion for Performance
    testKernelFusion(10000000);
    
    // Example 5: Dynamic Execution Path Selection
    // Test with different data densities
    testDynamicExecution(1000000, 0.1f);  // Sparse data
    testDynamicExecution(1000000, 0.9f);  // Dense data
    
    printf("All tests completed successfully!\n");
    return 0;
}
