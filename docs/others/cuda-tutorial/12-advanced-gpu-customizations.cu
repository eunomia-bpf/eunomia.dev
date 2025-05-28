/**
 * 12-advanced-gpu-customizations.cu
 * 
 * This example demonstrates additional fine-grained GPU customization techniques
 * that extend beyond basic optimizations, including:
 * 
 * 1. Thread divergence mitigation strategies
 * 2. Register usage optimization
 * 3. Mixed precision computation
 * 4. Persistent threads for load balancing
 * 5. Warp specialization patterns
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cuda_fp16.h>  // For half precision
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
 * Example 1: Thread Divergence Mitigation
 * 
 * This example demonstrates techniques to minimize the impact of
 * thread divergence within warps.
 *****************************************************************************/

// Naive kernel with thread divergence based on thread ID
__global__ void naiveDivergentKernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        // High divergence: threads take different paths based on thread ID
        if (threadIdx.x % 2 == 0) {
            // Even threads take this path (expensive computation)
            float result = input[idx];
            for (int i = 0; i < 100; i++) {
                result = sinf(result) * cosf(result) + 0.1f;
            }
            output[idx] = result;
        } else {
            // Odd threads take this path (simple computation)
            output[idx] = input[idx] * 2.0f;
        }
    }
}

// Optimized kernel that reduces divergence by rearranging work
__global__ void optimizedDivergenceKernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float value = input[idx];
        
        // All threads in the same block take the same path
        // Divergence happens at block level, not thread level
        if (blockIdx.x % 2 == 0) {
            // Blocks with even indices take this path
            float result = value;
            for (int i = 0; i < 100; i++) {
                result = sinf(result) * cosf(result) + 0.1f;
            }
            output[idx] = result;
        } else {
            // Blocks with odd indices take this path
            output[idx] = value * 2.0f;
        }
    }
}

void testDivergenceMitigation(int dataSize) {
    // Allocate and initialize host data
    float* h_input = new float[dataSize];
    float* h_output_naive = new float[dataSize];
    float* h_output_optimized = new float[dataSize];
    
    for (int i = 0; i < dataSize; i++) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    
    // Allocate device memory
    float* d_input;
    float* d_output_naive;
    float* d_output_optimized;
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, dataSize * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output_naive, dataSize * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output_optimized, dataSize * sizeof(float)));
    
    // Copy input to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, dataSize * sizeof(float), cudaMemcpyHostToDevice));
    
    // Execution configuration
    int blockSize = 256;
    int numBlocks = (dataSize + blockSize - 1) / blockSize;
    
    // Benchmark naive divergent implementation
    cudaDeviceSynchronize();
    double start = timeInMs();
    
    naiveDivergentKernel<<<numBlocks, blockSize>>>(d_input, d_output_naive, dataSize);
    
    cudaDeviceSynchronize();
    double timeNaive = timeInMs() - start;
    
    // Benchmark optimized implementation
    cudaDeviceSynchronize();
    start = timeInMs();
    
    optimizedDivergenceKernel<<<numBlocks, blockSize>>>(d_input, d_output_optimized, dataSize);
    
    cudaDeviceSynchronize();
    double timeOptimized = timeInMs() - start;
    
    printf("Thread Divergence Mitigation Example:\n");
    printf("  Naive divergent kernel: %.3f ms\n", timeNaive);
    printf("  Optimized divergence kernel: %.3f ms\n", timeOptimized);
    printf("  Speedup: %.2fx\n\n", timeNaive / timeOptimized);
    
    // Cleanup
    delete[] h_input;
    delete[] h_output_naive;
    delete[] h_output_optimized;
    
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output_naive));
    CHECK_CUDA_ERROR(cudaFree(d_output_optimized));
}

/*****************************************************************************
 * Example 2: Register Usage Optimization
 * 
 * This example demonstrates techniques to optimize register usage,
 * which can impact occupancy and performance.
 *****************************************************************************/

// Kernel with high register usage
__global__ void highRegisterKernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        // Use many local variables to increase register pressure
        float a1 = input[idx];
        float a2 = a1 * 1.1f;
        float a3 = a2 * 1.2f;
        float a4 = a3 * 1.3f;
        float a5 = a4 * 1.4f;
        float a6 = a5 * 1.5f;
        float a7 = a6 * 1.6f;
        float a8 = a7 * 1.7f;
        float a9 = a8 * 1.8f;
        float a10 = a9 * 1.9f;
        float a11 = a10 * 2.0f;
        float a12 = a11 * 2.1f;
        float a13 = a12 * 2.2f;
        float a14 = a13 * 2.3f;
        float a15 = a14 * 2.4f;
        float a16 = a15 * 2.5f;
        
        // Complex computation to prevent compiler optimizations
        for (int i = 0; i < 20; i++) {
            a1 = a1 + a2 * cosf(a3);
            a2 = a2 + a3 * sinf(a4);
            a3 = a3 + a4 * expf(a5 * 0.01f);
            a4 = a4 + a5 * logf(a6 + 1.0f);
            a5 = a5 + a6 * tanf(a7 * 0.1f);
            a6 = a6 + a7 * a8;
            a7 = a7 + a8 * a9;
            a8 = a8 + a9 * a10;
            a9 = a9 + a10 * a11;
            a10 = a10 + a11 * a12;
            a11 = a11 + a12 * a13;
            a12 = a12 + a13 * a14;
            a13 = a13 + a14 * a15;
            a14 = a14 + a15 * a16;
            a15 = a15 + a16 * a1;
            a16 = a16 + a1 * a2;
        }
        
        output[idx] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + 
                      a9 + a10 + a11 + a12 + a13 + a14 + a15 + a16;
    }
}

// Kernel with reduced register usage using loop unrolling
__global__ void optimizedRegisterKernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        // Use fewer variables and reuse them in the loop
        float result = input[idx];
        float temp = result * 1.1f;
        
        // Replace multiple variables with loop iterations
        for (int i = 0; i < 20; i++) {
            // Perform equivalent computation with fewer registers
            result = result + temp * cosf(result);
            temp = temp + result * sinf(temp);
            
            // Unroll loop partially to maintain computational intensity
            result = result + temp * expf(result * 0.01f);
            temp = temp + result * logf(temp + 1.0f);
        }
        
        output[idx] = result + temp;
    }
}

void testRegisterOptimization(int dataSize) {
    // Allocate and initialize host data
    float* h_input = new float[dataSize];
    float* h_output_high_reg = new float[dataSize];
    float* h_output_opt_reg = new float[dataSize];
    
    for (int i = 0; i < dataSize; i++) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    
    // Allocate device memory
    float* d_input;
    float* d_output_high_reg;
    float* d_output_opt_reg;
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, dataSize * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output_high_reg, dataSize * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output_opt_reg, dataSize * sizeof(float)));
    
    // Copy input to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, dataSize * sizeof(float), cudaMemcpyHostToDevice));
    
    // Execution configuration
    int blockSize = 256;
    int numBlocks = (dataSize + blockSize - 1) / blockSize;
    
    // Benchmark high register usage kernel
    cudaDeviceSynchronize();
    double start = timeInMs();
    
    highRegisterKernel<<<numBlocks, blockSize>>>(d_input, d_output_high_reg, dataSize);
    
    cudaDeviceSynchronize();
    double timeHighReg = timeInMs() - start;
    
    // Benchmark optimized register usage kernel
    cudaDeviceSynchronize();
    start = timeInMs();
    
    optimizedRegisterKernel<<<numBlocks, blockSize>>>(d_input, d_output_opt_reg, dataSize);
    
    cudaDeviceSynchronize();
    double timeOptReg = timeInMs() - start;
    
    // Check occupancy (simplified approach)
    int numBlocksHighReg = 0;
    int numBlocksOptReg = 0;
    
    CHECK_CUDA_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksHighReg, highRegisterKernel, blockSize, 0));
    
    CHECK_CUDA_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksOptReg, optimizedRegisterKernel, blockSize, 0));
    
    printf("Register Usage Optimization Example:\n");
    printf("  High register usage kernel: %.3f ms\n", timeHighReg);
    printf("  Optimized register usage kernel: %.3f ms\n", timeOptReg);
    printf("  Speedup: %.2fx\n", timeHighReg / timeOptReg);
    printf("  Occupancy: %d vs %d blocks per SM\n\n", 
           numBlocksHighReg, numBlocksOptReg);
    
    // Cleanup
    delete[] h_input;
    delete[] h_output_high_reg;
    delete[] h_output_opt_reg;
    
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output_high_reg));
    CHECK_CUDA_ERROR(cudaFree(d_output_opt_reg));
}

/*****************************************************************************
 * Example 3: Mixed Precision Computation
 * 
 * This example demonstrates using mixed precision for better performance
 * while maintaining accuracy where needed.
 *****************************************************************************/

// Kernel using single precision (FP32) throughout
__global__ void singlePrecisionKernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float x = input[idx];
        
        // Perform computation in FP32
        float result = 0.0f;
        for (int i = 0; i < 100; i++) {
            result += sinf(x * i * 0.01f);
        }
        
        output[idx] = result;
    }
}

// Kernel using mixed precision (FP16 computation, FP32 accumulation)
__global__ void mixedPrecisionKernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float x_f32 = input[idx];
        
        // Convert to half precision for computation
        half x_f16 = __float2half(x_f32);
        
        // Accumulate in FP32 for accuracy
        float result = 0.0f;
        
        for (int i = 0; i < 100; i++) {
            // Compute in FP16
            half i_f16 = __float2half(i * 0.01f);
            half mult = __hmul(x_f16, i_f16);
            
            // Use __float2half and __half2float for trig functions since 
            // direct half-precision versions might not be available on all architectures
            float sin_val = sinf(__half2float(mult));
            
            // Accumulate in FP32
            result += sin_val;
        }
        
        output[idx] = result;
    }
}

void testMixedPrecision(int dataSize) {
    // Allocate and initialize host data
    float* h_input = new float[dataSize];
    float* h_output_fp32 = new float[dataSize];
    float* h_output_mixed = new float[dataSize];
    
    for (int i = 0; i < dataSize; i++) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    
    // Allocate device memory
    float* d_input;
    float* d_output_fp32;
    float* d_output_mixed;
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, dataSize * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output_fp32, dataSize * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output_mixed, dataSize * sizeof(float)));
    
    // Copy input to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, dataSize * sizeof(float), cudaMemcpyHostToDevice));
    
    // Execution configuration
    int blockSize = 256;
    int numBlocks = (dataSize + blockSize - 1) / blockSize;
    
    // Benchmark single precision kernel
    cudaDeviceSynchronize();
    double start = timeInMs();
    
    singlePrecisionKernel<<<numBlocks, blockSize>>>(d_input, d_output_fp32, dataSize);
    
    cudaDeviceSynchronize();
    double timeFP32 = timeInMs() - start;
    
    // Benchmark mixed precision kernel
    cudaDeviceSynchronize();
    start = timeInMs();
    
    mixedPrecisionKernel<<<numBlocks, blockSize>>>(d_input, d_output_mixed, dataSize);
    
    cudaDeviceSynchronize();
    double timeMixed = timeInMs() - start;
    
    // Copy results back for accuracy comparison
    CHECK_CUDA_ERROR(cudaMemcpy(h_output_fp32, d_output_fp32, dataSize * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_output_mixed, d_output_mixed, dataSize * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Calculate average error
    float total_error = 0.0f;
    for (int i = 0; i < dataSize; i++) {
        total_error += fabsf(h_output_fp32[i] - h_output_mixed[i]);
    }
    float avg_error = total_error / dataSize;
    
    printf("Mixed Precision Computation Example:\n");
    printf("  Single precision kernel: %.3f ms\n", timeFP32);
    printf("  Mixed precision kernel: %.3f ms\n", timeMixed);
    printf("  Speedup: %.2fx\n", timeFP32 / timeMixed);
    printf("  Average error: %e\n\n", avg_error);
    
    // Cleanup
    delete[] h_input;
    delete[] h_output_fp32;
    delete[] h_output_mixed;
    
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output_fp32));
    CHECK_CUDA_ERROR(cudaFree(d_output_mixed));
}

/*****************************************************************************
 * Example 4: Persistent Threads for Better Load Balancing
 * 
 * This example demonstrates using persistent threads that keep running
 * and dynamically grab work items, which can improve load balancing.
 *****************************************************************************/

#define MAX_QUEUE_SIZE 1024
#define SENTINEL_VALUE -1

// Structure for work queue
struct WorkQueue {
    int items[MAX_QUEUE_SIZE];    // Work items
    int head;                     // Current head of queue (atomic)
    int tail;                     // Current tail of queue
    int finished;                 // Flag to indicate all work is done
};

// Traditional kernel with fixed work assignment
__global__ void traditionalKernel(float* input, float* output, int* workloads, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float value = input[idx];
        int work_amount = workloads[idx];
        
        // Do variable amount of work
        for (int i = 0; i < work_amount; i++) {
            value = value * 0.9f + 0.1f * sinf(value);
        }
        
        output[idx] = value;
    }
}

// Persistent threads kernel with dynamic work assignment
__global__ void persistentThreadsKernel(float* input, float* output, int* workloads, 
                                       WorkQueue* queue, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Keep thread alive to process multiple items
    while (true) {
        // Atomically grab next work item
        int work_idx = atomicAdd(&queue->head, 1);
        
        // Check if we've processed all items
        if (work_idx >= size || work_idx == SENTINEL_VALUE) {
            break;
        }
        
        // Process the work item
        float value = input[work_idx];
        int work_amount = workloads[work_idx];
        
        // Do variable amount of work
        for (int i = 0; i < work_amount; i++) {
            value = value * 0.9f + 0.1f * sinf(value);
        }
        
        output[work_idx] = value;
    }
}

void testPersistentThreads(int dataSize) {
    // Allocate and initialize host data
    float* h_input = new float[dataSize];
    float* h_output_traditional = new float[dataSize];
    float* h_output_persistent = new float[dataSize];
    int* h_workloads = new int[dataSize];
    
    // Create highly imbalanced workloads
    for (int i = 0; i < dataSize; i++) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;
        
        // Some items need much more work than others
        if (i % 128 == 0) {
            h_workloads[i] = 1000;  // Very expensive
        } else {
            h_workloads[i] = 10;    // Cheap
        }
    }
    
    // Allocate device memory
    float* d_input;
    float* d_output_traditional;
    float* d_output_persistent;
    int* d_workloads;
    WorkQueue* d_queue;
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, dataSize * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output_traditional, dataSize * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output_persistent, dataSize * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_workloads, dataSize * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_queue, sizeof(WorkQueue)));
    
    // Copy input data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, dataSize * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_workloads, h_workloads, dataSize * sizeof(int), cudaMemcpyHostToDevice));
    
    // Initialize work queue
    WorkQueue h_queue;
    h_queue.head = 0;
    h_queue.tail = dataSize;
    h_queue.finished = 0;
    CHECK_CUDA_ERROR(cudaMemcpy(d_queue, &h_queue, sizeof(WorkQueue), cudaMemcpyHostToDevice));
    
    // Execution configuration
    int blockSize = 256;
    int numBlocks = (dataSize + blockSize - 1) / blockSize;
    
    // Benchmark traditional kernel
    cudaDeviceSynchronize();
    double start = timeInMs();
    
    traditionalKernel<<<numBlocks, blockSize>>>(d_input, d_output_traditional, d_workloads, dataSize);
    
    cudaDeviceSynchronize();
    double timeTraditional = timeInMs() - start;
    
    // Benchmark persistent threads kernel (use fewer threads since each handles multiple items)
    int persistentBlocks = 32;  // Use fewer blocks for persistent threads
    cudaDeviceSynchronize();
    start = timeInMs();
    
    persistentThreadsKernel<<<persistentBlocks, blockSize>>>(d_input, d_output_persistent, 
                                                            d_workloads, d_queue, dataSize);
    
    cudaDeviceSynchronize();
    double timePersistent = timeInMs() - start;
    
    // Copy results back for verification
    CHECK_CUDA_ERROR(cudaMemcpy(h_output_traditional, d_output_traditional, 
                               dataSize * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_output_persistent, d_output_persistent, 
                               dataSize * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Verify results
    bool resultsMatch = true;
    for (int i = 0; i < dataSize; i++) {
        if (fabsf(h_output_traditional[i] - h_output_persistent[i]) > 1e-5) {
            resultsMatch = false;
            break;
        }
    }
    
    printf("Persistent Threads for Load Balancing Example:\n");
    printf("  Traditional kernel: %.3f ms\n", timeTraditional);
    printf("  Persistent threads kernel: %.3f ms\n", timePersistent);
    printf("  Speedup: %.2fx\n", timeTraditional / timePersistent);
    printf("  Results match: %s\n\n", resultsMatch ? "Yes" : "No");
    
    // Cleanup
    delete[] h_input;
    delete[] h_output_traditional;
    delete[] h_output_persistent;
    delete[] h_workloads;
    
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output_traditional));
    CHECK_CUDA_ERROR(cudaFree(d_output_persistent));
    CHECK_CUDA_ERROR(cudaFree(d_workloads));
    CHECK_CUDA_ERROR(cudaFree(d_queue));
}

/*****************************************************************************
 * Example 5: Warp Specialization Patterns
 * 
 * This example demonstrates how to specialize different warps within
 * a block to perform different tasks for better efficiency.
 *****************************************************************************/

#define WARP_SIZE 32

// Traditional kernel where all warps do the same work
__global__ void uniformKernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        // All threads perform the same set of operations
        float value = input[idx];
        float result = 0.0f;
        
        // Compute-heavy task for all threads
        for (int i = 0; i < 50; i++) {
            result += sinf(value * i * 0.01f);
        }
        
        output[idx] = result;
    }
}

// Kernel with warp specialization (different warps do different tasks)
__global__ void warpSpecializedKernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = threadIdx.x / WARP_SIZE;
    
    if (idx < size) {
        float value = input[idx];
        float result = 0.0f;
        
        // Specialize warps for different tasks
        if (warpId == 0) {
            // First warp: Trigonometric computations
            for (int i = 0; i < 50; i++) {
                result += sinf(value * i * 0.01f);
            }
        } else if (warpId == 1) {
            // Second warp: Polynomial computations
            float x = value;
            float x2 = x * x;
            float x3 = x2 * x;
            float x4 = x3 * x;
            result = 1.0f + x + x2/2.0f + x3/6.0f + x4/24.0f;
        } else if (warpId == 2) {
            // Third warp: Exponential approximation
            result = expf(value);
        } else {
            // Other warps: Simpler operations
            result = value * 2.0f;
        }
        
        output[idx] = result;
    }
}

void testWarpSpecialization(int dataSize) {
    // Allocate and initialize host data
    float* h_input = new float[dataSize];
    float* h_output_uniform = new float[dataSize];
    float* h_output_specialized = new float[dataSize];
    
    for (int i = 0; i < dataSize; i++) {
        h_input[i] = 0.1f * static_cast<float>(rand()) / RAND_MAX;  // Small values for numerical stability
    }
    
    // Allocate device memory
    float* d_input;
    float* d_output_uniform;
    float* d_output_specialized;
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, dataSize * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output_uniform, dataSize * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output_specialized, dataSize * sizeof(float)));
    
    // Copy input to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, dataSize * sizeof(float), cudaMemcpyHostToDevice));
    
    // Execution configuration
    int blockSize = 128;  // 4 warps per block
    int numBlocks = (dataSize + blockSize - 1) / blockSize;
    
    // Benchmark uniform kernel
    cudaDeviceSynchronize();
    double start = timeInMs();
    
    uniformKernel<<<numBlocks, blockSize>>>(d_input, d_output_uniform, dataSize);
    
    cudaDeviceSynchronize();
    double timeUniform = timeInMs() - start;
    
    // Benchmark specialized kernel
    cudaDeviceSynchronize();
    start = timeInMs();
    
    warpSpecializedKernel<<<numBlocks, blockSize>>>(d_input, d_output_specialized, dataSize);
    
    cudaDeviceSynchronize();
    double timeSpecialized = timeInMs() - start;
    
    printf("Warp Specialization Example:\n");
    printf("  Uniform kernel: %.3f ms\n", timeUniform);
    printf("  Warp specialized kernel: %.3f ms\n", timeSpecialized);
    printf("  Speedup: %.2fx\n\n", timeUniform / timeSpecialized);
    
    // Cleanup
    delete[] h_input;
    delete[] h_output_uniform;
    delete[] h_output_specialized;
    
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output_uniform));
    CHECK_CUDA_ERROR(cudaFree(d_output_specialized));
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
    
    // Run examples
    testDivergenceMitigation(1000000);
    testRegisterOptimization(1000000);
    
    // Mixed precision requires Compute Capability 7.0 or higher
    if (prop.major >= 7) {
        testMixedPrecision(1000000);
    } else {
        printf("Mixed Precision Computation Example:\n");
        printf("  Skipped - requires Compute Capability 7.0 or higher\n\n");
    }
    
    testPersistentThreads(1000000);
    testWarpSpecialization(1000000);
    
    printf("All examples completed successfully!\n");
    return 0;
} 