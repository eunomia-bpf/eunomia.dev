/**
 * GPU-Side Profiling and Kernel Instrumentation Example
 * 
 * This example demonstrates techniques for GPU-side profiling and kernel instrumentation,
 * focusing on scenarios where external profilers are insufficient. It implements:
 * 
 * 1. Fine-grained internal kernel timing
 * 2. Divergent path analysis
 * 3. Dynamic workload profiling
 * 4. Adaptive algorithm selection
 * 
 * Compile with: nvcc -o 10-cpu-gpu-profiling-boundaries 10-cpu-gpu-profiling-boundaries.cu
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

// Error checking macro
#define CHECK_CUDA_ERROR(call)                                                 \
do {                                                                           \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
        printf("CUDA error in file '%s' in line %i: %s.\n",                    \
               __FILE__, __LINE__, cudaGetErrorString(err));                   \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while (0)

// Timing structure for kernel sections
struct KernelTimings {
    unsigned long long init_time;
    unsigned long long compute_time;
    unsigned long long finalize_time;
    unsigned long long total_time;
};

// Structure for divergent path analysis
struct PathTimings {
    int path_taken;
    unsigned long long execution_time;
};

// Constants
#define DATA_SIZE (1024 * 1024)  // 1M elements
#define BLOCK_SIZE 256
#define NUM_PATHS 2
#define WORKLOAD_VARIANCE 100    // Max additional work per thread

//==============================================================================
// Example 1: Fine-grained Internal Kernel Timing
//==============================================================================

/**
 * Kernel with internal section timing
 */
__global__ void sectionTimingKernel(float* input, float* output, int n, KernelTimings* timings) {
    extern __shared__ float shared_data[];
    
    // Only first thread in block records timing
    unsigned long long block_start, init_end, compute_end, block_end;
    if (threadIdx.x == 0) {
        block_start = clock64();
    }
    
    // Section 1: Initialization phase - load and preprocess data
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Perform some initialization work
        shared_data[threadIdx.x] = input[idx] * 2.0f;
    }
    
    __syncthreads();
    
    if (threadIdx.x == 0) {
        init_end = clock64();
    }
    
    // Section 2: Computation phase - process data
    if (idx < n) {
        // Simulate computation with shared memory access
        float sum = 0.0f;
        #pragma unroll 8
        for (int i = 0; i < blockDim.x; i++) {
            sum += shared_data[(threadIdx.x + i) % blockDim.x];
        }
        output[idx] = sum / blockDim.x;
    }
    
    __syncthreads();
    
    if (threadIdx.x == 0) {
        compute_end = clock64();
    }
    
    // Section 3: Finalization phase - post-process results
    if (idx < n) {
        // Apply final transformation
        output[idx] = sqrtf(fabsf(output[idx]));
    }
    
    __syncthreads();
    
    // Record final timing and accumulate across blocks
    if (threadIdx.x == 0) {
        block_end = clock64();
        
        // Use atomics to accumulate timings across all blocks
        atomicAdd(&timings->init_time, init_end - block_start);
        atomicAdd(&timings->compute_time, compute_end - init_end);
        atomicAdd(&timings->finalize_time, block_end - compute_end);
        atomicAdd(&timings->total_time, block_end - block_start);
    }
}

//==============================================================================
// Example 2: Conditional Path Analysis
//==============================================================================

/**
 * Kernel with divergent execution paths and timing for each path
 */
__global__ void divergentPathKernel(float* input, float* output, PathTimings* path_timings, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        unsigned long long start_time = clock64();
        
        // Create two divergent paths based on input data
        if (input[idx] > 0.5f) {
            // Path A: Expensive computation (trigonometric functions)
            float val = input[idx];
            for (int i = 0; i < 50; i++) {
                val = sinf(val) * cosf(val) + 0.1f;
            }
            output[idx] = val;
            
            // Record which path was taken (0 for Path A)
            path_timings[idx].path_taken = 0;
        } else {
            // Path B: Different computation (polynomial)
            float val = input[idx];
            for (int i = 0; i < 20; i++) {
                val = val * val + 0.01f;
            }
            output[idx] = val;
            
            // Record which path was taken (1 for Path B)
            path_timings[idx].path_taken = 1;
        }
        
        unsigned long long end_time = clock64();
        
        // Record execution time for this thread's path
        path_timings[idx].execution_time = end_time - start_time;
    }
}

//==============================================================================
// Example 3: Dynamic Workload Profiling
//==============================================================================

/**
 * Kernel with variable workload per thread
 */
__global__ void dynamicWorkloadKernel(float* input, float* output, int* workloads, 
                                     unsigned long long* execution_times, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Each thread gets a different workload based on input value
        // Scale to 1-100 iterations
        int iterations = 1 + (int)(fabsf(input[idx]) * WORKLOAD_VARIANCE);
        workloads[idx] = iterations;
        
        // Record start time
        unsigned long long start_time = clock64();
        
        // Perform variable amount of work
        float result = input[idx];
        for (int i = 0; i < iterations; i++) {
            result = result * 0.9f + 0.1f * sqrtf(fabsf(result));
        }
        
        // Record end time
        unsigned long long end_time = clock64();
        
        // Store result and timing information
        output[idx] = result;
        execution_times[idx] = end_time - start_time;
    }
}

//==============================================================================
// Example 4: Adaptive Algorithm Selection
//==============================================================================

/**
 * Kernel that adapts its algorithm based on performance feedback
 */
__global__ void adaptiveKernel(float* input, float* output, int* algorithm_choices, int n, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float val = input[idx];
        int current_algorithm = 0; // Start with algorithm 0
        
        for (int iter = 0; iter < iterations; iter++) {
            // Record start time
            unsigned long long start_time = clock64();
            
            if (current_algorithm == 0) {
                // Algorithm A: Polynomial approximation
                val = val * val * 0.5f + val * 0.3f + 0.1f;
            } else {
                // Algorithm B: Trigonometric approximation
                val = sinf(val * 0.5f) * 0.8f + 0.2f;
            }
            
            // Record end time
            unsigned long long end_time = clock64();
            long long elapsed = end_time - start_time;
            
            // Every 10 iterations, try the other algorithm and compare
            if (iter % 10 == 9) {
                // Save current value
                float saved_val = val;
                unsigned long long algorithm_a_time, algorithm_b_time;
                
                // Time algorithm A
                start_time = clock64();
                float test_val_a = saved_val * saved_val * 0.5f + saved_val * 0.3f + 0.1f;
                end_time = clock64();
                algorithm_a_time = end_time - start_time;
                
                // Time algorithm B
                start_time = clock64();
                float test_val_b = sinf(saved_val * 0.5f) * 0.8f + 0.2f;
                end_time = clock64();
                algorithm_b_time = end_time - start_time;
                
                // Choose faster algorithm for next iterations
                if (algorithm_a_time <= algorithm_b_time) {
                    current_algorithm = 0;
                    val = test_val_a;
                } else {
                    current_algorithm = 1;
                    val = test_val_b;
                }
                
                // First thread in first block records the choice for this iteration
                if (idx == 0) {
                    algorithm_choices[iter / 10] = current_algorithm;
                }
            }
        }
        
        // Store final result
        output[idx] = val;
    }
}

//==============================================================================
// Host Functions for Running and Analyzing Results
//==============================================================================

/**
 * Run the section timing kernel and analyze results
 */
void runSectionTimingExample() {
    printf("\n===== Example 1: Fine-grained Internal Kernel Timing =====\n");
    
    // Allocate and initialize host data
    float *h_input = (float*)malloc(DATA_SIZE * sizeof(float));
    float *h_output = (float*)malloc(DATA_SIZE * sizeof(float));
    
    for (int i = 0; i < DATA_SIZE; i++) {
        h_input[i] = (float)rand() / RAND_MAX;
    }
    
    // Allocate device memory
    float *d_input, *d_output;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, DATA_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, DATA_SIZE * sizeof(float)));
    
    // Copy input data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, DATA_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    
    // Allocate and initialize timing structure
    KernelTimings *h_timings = (KernelTimings*)calloc(1, sizeof(KernelTimings));
    KernelTimings *d_timings;
    CHECK_CUDA_ERROR(cudaMalloc(&d_timings, sizeof(KernelTimings)));
    CHECK_CUDA_ERROR(cudaMemset(d_timings, 0, sizeof(KernelTimings)));
    
    // Calculate grid dimensions
    int blocks = (DATA_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Launch kernel with section timing
    sectionTimingKernel<<<blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(
        d_input, d_output, DATA_SIZE, d_timings);
    
    // Retrieve timing results
    CHECK_CUDA_ERROR(cudaMemcpy(h_timings, d_timings, sizeof(KernelTimings), cudaMemcpyDeviceToHost));
    
    // Get device properties for clock rate
    cudaDeviceProp prop;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, 0));
    float ms_per_clock = 1000.0f / (prop.clockRate * 1000.0f);
    
    // Calculate average times per block
    float init_ms = h_timings->init_time * ms_per_clock / blocks;
    float compute_ms = h_timings->compute_time * ms_per_clock / blocks;
    float finalize_ms = h_timings->finalize_time * ms_per_clock / blocks;
    float total_ms = h_timings->total_time * ms_per_clock / blocks;
    
    // Report results
    printf("Section timing results (averaged per block):\n");
    printf("  Initialization phase: %.4f ms (%.1f%%)\n", 
           init_ms, 100.0f * init_ms / total_ms);
    printf("  Computation phase:    %.4f ms (%.1f%%)\n", 
           compute_ms, 100.0f * compute_ms / total_ms);
    printf("  Finalization phase:   %.4f ms (%.1f%%)\n", 
           finalize_ms, 100.0f * finalize_ms / total_ms);
    printf("  Total kernel time:    %.4f ms\n", total_ms);
    
    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_timings);
    free(h_input);
    free(h_output);
    free(h_timings);
}

/**
 * Run the divergent path analysis example
 */
void runDivergentPathExample() {
    printf("\n===== Example 2: Conditional Path Analysis =====\n");
    
    // Allocate and initialize host data
    float *h_input = (float*)malloc(DATA_SIZE * sizeof(float));
    float *h_output = (float*)malloc(DATA_SIZE * sizeof(float));
    PathTimings *h_path_timings = (PathTimings*)malloc(DATA_SIZE * sizeof(PathTimings));
    
    for (int i = 0; i < DATA_SIZE; i++) {
        h_input[i] = (float)rand() / RAND_MAX; // 0-1 range
    }
    
    // Allocate device memory
    float *d_input, *d_output;
    PathTimings *d_path_timings;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, DATA_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, DATA_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_path_timings, DATA_SIZE * sizeof(PathTimings)));
    
    // Copy input data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, DATA_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    
    // Launch kernel
    int blocks = (DATA_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    divergentPathKernel<<<blocks, BLOCK_SIZE>>>(d_input, d_output, d_path_timings, DATA_SIZE);
    
    // Retrieve results
    CHECK_CUDA_ERROR(cudaMemcpy(h_path_timings, d_path_timings, DATA_SIZE * sizeof(PathTimings), 
                               cudaMemcpyDeviceToHost));
    
    // Calculate average execution time for each path
    unsigned long long path_counts[NUM_PATHS] = {0};
    unsigned long long path_total_times[NUM_PATHS] = {0};
    
    for (int i = 0; i < DATA_SIZE; i++) {
        int path = h_path_timings[i].path_taken;
        path_counts[path]++;
        path_total_times[path] += h_path_timings[i].execution_time;
    }
    
    // Get device properties for clock rate
    cudaDeviceProp prop;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, 0));
    float ms_per_clock = 1000.0f / (prop.clockRate * 1000.0f);
    
    // Report results
    printf("Divergent path analysis results:\n");
    for (int i = 0; i < NUM_PATHS; i++) {
        if (path_counts[i] > 0) {
            float avg_time = (path_total_times[i] * ms_per_clock) / path_counts[i];
            printf("  Path %d: %lld threads, avg time %.4f ms\n", 
                   i, path_counts[i], avg_time);
        }
    }
    
    if (path_counts[0] > 0 && path_counts[1] > 0) {
        float path0_avg = (path_total_times[0] * ms_per_clock) / path_counts[0];
        float path1_avg = (path_total_times[1] * ms_per_clock) / path_counts[1];
        printf("  Performance ratio: Path 0 is %.2f times slower than Path 1\n", 
               path0_avg / path1_avg);
    }
    
    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_path_timings);
    free(h_input);
    free(h_output);
    free(h_path_timings);
}

/**
 * Run the dynamic workload profiling example
 */
void runDynamicWorkloadExample() {
    printf("\n===== Example 3: Dynamic Workload Profiling =====\n");
    
    // Allocate and initialize host data
    float *h_input = (float*)malloc(DATA_SIZE * sizeof(float));
    float *h_output = (float*)malloc(DATA_SIZE * sizeof(float));
    int *h_workloads = (int*)malloc(DATA_SIZE * sizeof(int));
    unsigned long long *h_execution_times = (unsigned long long*)malloc(DATA_SIZE * sizeof(unsigned long long));
    
    for (int i = 0; i < DATA_SIZE; i++) {
        h_input[i] = (float)rand() / RAND_MAX; // 0-1 range
    }
    
    // Allocate device memory
    float *d_input, *d_output;
    int *d_workloads;
    unsigned long long *d_execution_times;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, DATA_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, DATA_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_workloads, DATA_SIZE * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_execution_times, DATA_SIZE * sizeof(unsigned long long)));
    
    // Copy input data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, DATA_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    
    // Launch kernel
    int blocks = (DATA_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dynamicWorkloadKernel<<<blocks, BLOCK_SIZE>>>(d_input, d_output, d_workloads, d_execution_times, DATA_SIZE);
    
    // Retrieve results
    CHECK_CUDA_ERROR(cudaMemcpy(h_workloads, d_workloads, DATA_SIZE * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_execution_times, d_execution_times, 
                               DATA_SIZE * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    
    // Get device properties for clock rate
    cudaDeviceProp prop;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, 0));
    float ms_per_clock = 1000.0f / (prop.clockRate * 1000.0f);
    
    // Analyze workload distribution and timing correlation
    int workload_buckets[10] = {0}; // Divide into 10 buckets
    unsigned long long bucket_times[10] = {0};
    
    for (int i = 0; i < DATA_SIZE; i++) {
        int bucket = (h_workloads[i] * 10) / (WORKLOAD_VARIANCE + 1);
        if (bucket >= 10) bucket = 9;
        workload_buckets[bucket]++;
        bucket_times[bucket] += h_execution_times[i];
    }
    
    // Report results
    printf("Dynamic workload analysis results:\n");
    printf("  Workload Distribution and Timing:\n");
    printf("  %-15s %-15s %-15s\n", "Workload Range", "Thread Count", "Avg Time (ms)");
    
    for (int i = 0; i < 10; i++) {
        if (workload_buckets[i] > 0) {
            int min_iter = (i * (WORKLOAD_VARIANCE + 1)) / 10;
            int max_iter = ((i + 1) * (WORKLOAD_VARIANCE + 1)) / 10 - 1;
            float avg_time = (bucket_times[i] * ms_per_clock) / workload_buckets[i];
            
            printf("  %-2d-%-12d %-15d %.6f\n", 
                   min_iter, max_iter, workload_buckets[i], avg_time);
        }
    }
    
    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_workloads);
    cudaFree(d_execution_times);
    free(h_input);
    free(h_output);
    free(h_workloads);
    free(h_execution_times);
}

/**
 * Run the adaptive algorithm selection example
 */
void runAdaptiveAlgorithmExample() {
    printf("\n===== Example 4: Adaptive Algorithm Selection =====\n");
    
    // Parameters
    int iterations = 100;
    int num_algorithm_choices = iterations / 10 + 1;
    
    // Allocate and initialize host data
    float *h_input = (float*)malloc(DATA_SIZE * sizeof(float));
    float *h_output = (float*)malloc(DATA_SIZE * sizeof(float));
    int *h_algorithm_choices = (int*)malloc(num_algorithm_choices * sizeof(int));
    
    for (int i = 0; i < DATA_SIZE; i++) {
        h_input[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f; // -1 to 1 range
    }
    
    // Allocate device memory
    float *d_input, *d_output;
    int *d_algorithm_choices;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, DATA_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, DATA_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_algorithm_choices, num_algorithm_choices * sizeof(int)));
    
    // Copy input data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, DATA_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemset(d_algorithm_choices, 0, num_algorithm_choices * sizeof(int)));
    
    // Launch kernel
    int blocks = (DATA_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    adaptiveKernel<<<blocks, BLOCK_SIZE>>>(d_input, d_output, d_algorithm_choices, DATA_SIZE, iterations);
    
    // Retrieve algorithm choices
    CHECK_CUDA_ERROR(cudaMemcpy(h_algorithm_choices, d_algorithm_choices, 
                               num_algorithm_choices * sizeof(int), cudaMemcpyDeviceToHost));
    
    // Report results
    printf("Adaptive algorithm selection results:\n");
    printf("  Algorithm choices over time (0=polynomial, 1=trigonometric):\n  ");
    
    for (int i = 0; i < num_algorithm_choices; i++) {
        printf("%d", h_algorithm_choices[i]);
        if ((i + 1) % 20 == 0) printf("\n  ");
    }
    printf("\n");
    
    // Count distribution of choices
    int alg0_count = 0, alg1_count = 0;
    for (int i = 0; i < num_algorithm_choices; i++) {
        if (h_algorithm_choices[i] == 0) alg0_count++;
        else alg1_count++;
    }
    
    printf("  Algorithm distribution: %d polynomial (%.1f%%), %d trigonometric (%.1f%%)\n",
           alg0_count, 100.0f * alg0_count / num_algorithm_choices,
           alg1_count, 100.0f * alg1_count / num_algorithm_choices);
    
    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_algorithm_choices);
    free(h_input);
    free(h_output);
    free(h_algorithm_choices);
}

/**
 * Main function
 */
int main(int argc, char **argv) {
    // Print device information
    cudaDeviceProp prop;
    int deviceCount;
    
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&deviceCount));
    
    printf("GPU-Side Profiling and Kernel Instrumentation Examples\n");
    printf("=====================================================\n");
    
    if (deviceCount == 0) {
        printf("No CUDA devices found\n");
        return 1;
    }
    
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, 0));
    printf("Using device: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Clock rate: %.2f GHz\n", prop.clockRate / 1e6);
    
    // Run examples
    runSectionTimingExample();
    runDivergentPathExample();
    runDynamicWorkloadExample();
    runAdaptiveAlgorithmExample();
    
    printf("\nAll examples completed successfully\n");
    return 0;
} 