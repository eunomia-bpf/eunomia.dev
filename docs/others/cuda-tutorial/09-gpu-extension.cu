/**
 * CUDA GPU Application Extension Mechanisms
 * 
 * This example demonstrates various techniques for extending and modifying CUDA applications
 * without changing their source code, including:
 * 
 * 1. API Interception - Hooking CUDA Runtime API calls
 * 2. Memory Management - Custom memory allocation and pooling
 * 3. Kernel Launch Optimization - Automatic thread block size adjustment
 * 4. Error Handling Extensions - Automatic retry and resilience
 * 
 * Compile with: nvcc -o basic09 basic09.cu -ldl -lpthread
 */

#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <cuda_runtime.h>
#include <string.h>
#include <unistd.h>
#include <iostream>
#include <vector>
#include <map>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <atomic>

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

// Define a global flag to enable/disable extensions
bool g_extensions_enabled = false;

//==============================================================================
// Part 1: Sample Kernels for Testing Extensions
//==============================================================================

/**
 * Simple vector addition kernel
 */
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

/**
 * Inefficient vector addition with divergence
 */
__global__ void vectorAddInefficient(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // Artificial divergence to demonstrate optimization
        if (i % 2 == 0) {
            // Even threads do extra work
            float temp = 0;
            for (int j = 0; j < 100; j++) {
                temp += a[i] * 0.01f;
            }
            c[i] = a[i] + b[i] + temp * 0.001f;
        } else {
            // Odd threads do normal work
            c[i] = a[i] + b[i];
        }
    }
}

//==============================================================================
// Part 2: Memory Tracking Extension
//==============================================================================

// Track memory allocations
struct MemoryTracker {
    std::map<void*, size_t> allocations;
    size_t total_allocated;
    size_t peak_allocated;
    int num_allocations;
    std::mutex mutex;

    MemoryTracker() : total_allocated(0), peak_allocated(0), num_allocations(0) {}

    void recordAllocation(void* ptr, size_t size) {
        std::lock_guard<std::mutex> lock(mutex);
        allocations[ptr] = size;
        total_allocated += size;
        num_allocations++;
        peak_allocated = std::max(peak_allocated, total_allocated);
        
        printf("CUDA Alloc: %zu bytes at %p (Total: %zu MB, Peak: %zu MB, Count: %d)\n", 
               size, ptr, total_allocated / (1024*1024), 
               peak_allocated / (1024*1024), num_allocations);
    }

    void recordFree(void* ptr) {
        std::lock_guard<std::mutex> lock(mutex);
        if (allocations.count(ptr)) {
            total_allocated -= allocations[ptr];
            allocations.erase(ptr);
            num_allocations--;
            
            printf("CUDA Free: %p (Total: %zu MB, Count: %d)\n", 
                   ptr, total_allocated / (1024*1024), num_allocations);
        } else {
            printf("CUDA Free: Unknown pointer %p\n", ptr);
        }
    }
    
    void printStats() {
        std::lock_guard<std::mutex> lock(mutex);
        printf("\nMemory Tracker Statistics:\n");
        printf("  Current allocations: %d\n", num_allocations);
        printf("  Current memory usage: %zu bytes (%zu MB)\n", 
               total_allocated, total_allocated / (1024*1024));
        printf("  Peak memory usage: %zu bytes (%zu MB)\n", 
               peak_allocated, peak_allocated / (1024*1024));
        
        if (num_allocations > 0) {
            printf("  Potential memory leaks detected:\n");
            for (auto& pair : allocations) {
                printf("    Address: %p, Size: %zu bytes\n", pair.first, pair.second);
            }
        }
    }
};

//==============================================================================
// Part 3: Memory Pool Extension
//==============================================================================

struct MemoryPool {
    struct Block {
        void* ptr;
        size_t size;
        bool in_use;
    };
    
    std::vector<Block> blocks;
    std::mutex mutex;
    size_t min_block_size;
    
    MemoryPool(size_t min_size = 1024) : min_block_size(min_size) {}
    
    ~MemoryPool() {
        // Free all blocks
        std::lock_guard<std::mutex> lock(mutex);
        for (auto& block : blocks) {
            cudaFree(block.ptr);
        }
        blocks.clear();
    }
    
    cudaError_t allocate(void** devPtr, size_t size) {
        std::lock_guard<std::mutex> lock(mutex);
        
        // Round size up to reduce fragmentation
        size_t aligned_size = ((size + min_block_size - 1) / min_block_size) * min_block_size;
        
        // Try to find a free block
        for (auto& block : blocks) {
            if (!block.in_use && block.size >= aligned_size) {
                block.in_use = true;
                *devPtr = block.ptr;
                return cudaSuccess;
            }
        }
        
        // Allocate new block
        void* new_ptr;
        cudaError_t result = cudaMalloc(&new_ptr, aligned_size);
        
        if (result == cudaSuccess) {
            blocks.push_back({new_ptr, aligned_size, true});
            *devPtr = new_ptr;
        }
        
        return result;
    }
    
    cudaError_t free(void* devPtr) {
        std::lock_guard<std::mutex> lock(mutex);
        
        // Find the block and mark it as free
        for (auto& block : blocks) {
            if (block.ptr == devPtr) {
                block.in_use = false;
                return cudaSuccess;
            }
        }
        
        // Not found in our pool
        return cudaErrorInvalidValue;
    }
    
    void printStats() {
        std::lock_guard<std::mutex> lock(mutex);
        size_t total_size = 0;
        size_t used_size = 0;
        int free_blocks = 0;
        int used_blocks = 0;
        
        for (auto& block : blocks) {
            total_size += block.size;
            if (block.in_use) {
                used_size += block.size;
                used_blocks++;
            } else {
                free_blocks++;
            }
        }
        
        printf("\nMemory Pool Statistics:\n");
        printf("  Total blocks: %zu\n", blocks.size());
        printf("  Used blocks: %d\n", used_blocks);
        printf("  Free blocks: %d\n", free_blocks);
        printf("  Total memory: %zu bytes (%zu MB)\n", 
               total_size, total_size / (1024*1024));
        printf("  Used memory: %zu bytes (%zu MB)\n", 
               used_size, used_size / (1024*1024));
        printf("  Free memory: %zu bytes (%zu MB)\n", 
               total_size - used_size, (total_size - used_size) / (1024*1024));
    }
};

//==============================================================================
// Part 4: Kernel Launch Optimizer Extension
//==============================================================================

class KernelOptimizer {
private:
    int device_id;
    cudaDeviceProp props;
    bool initialized;
    
public:
    KernelOptimizer() : initialized(false) {}
    
    void initialize() {
        if (!initialized) {
            CHECK_CUDA_ERROR(cudaGetDevice(&device_id));
            CHECK_CUDA_ERROR(cudaGetDeviceProperties(&props, device_id));
            initialized = true;
            
            printf("Kernel Optimizer initialized for device: %s\n", props.name);
            printf("  Compute capability: %d.%d\n", props.major, props.minor);
            printf("  Multiprocessors: %d\n", props.multiProcessorCount);
            printf("  Max threads per block: %d\n", props.maxThreadsPerBlock);
            printf("  Max threads per SM: %d\n", props.maxThreadsPerMultiProcessor);
        }
    }
    
    dim3 optimizeBlockSize(dim3 gridDim, dim3 blockDim) {
        if (!initialized) initialize();
        
        // Calculate total threads
        int total_threads = gridDim.x * gridDim.y * gridDim.z * 
                           blockDim.x * blockDim.y * blockDim.z;
        
        // Simple optimization: use a multiple of 32 (warp size) for better performance
        int threads_per_block = blockDim.x * blockDim.y * blockDim.z;
        
        // For small block sizes, try to use a larger size for better occupancy
        if (threads_per_block < 128) {
            // Target a larger block size for better SM utilization
            int target_size = 256;
            
            // Ensure we don't exceed device limits
            target_size = std::min(target_size, props.maxThreadsPerBlock);
            
            // Only update x dimension for simplicity
            return dim3(target_size, 1, 1);
        }
        
        // For large block sizes, ensure they're multiples of 32
        if (threads_per_block % 32 != 0) {
            int adjusted_size = ((threads_per_block + 31) / 32) * 32;
            
            // Ensure we don't exceed device limits
            adjusted_size = std::min(adjusted_size, props.maxThreadsPerBlock);
            
            // Only update x dimension for simplicity
            return dim3(adjusted_size, 1, 1);
        }
        
        // Default: return original block size
        return blockDim;
    }
    
    dim3 optimizeGridSize(dim3 gridDim, dim3 blockDim, dim3 newBlockDim) {
        if (!initialized) initialize();
        
        // Calculate total threads
        int total_threads = gridDim.x * gridDim.y * gridDim.z * 
                           blockDim.x * blockDim.y * blockDim.z;
        
        // Calculate new threads per block
        int new_threads_per_block = newBlockDim.x * newBlockDim.y * newBlockDim.z;
        
        // Calculate new grid size to maintain total threads
        int new_blocks = (total_threads + new_threads_per_block - 1) / new_threads_per_block;
        
        // Only update x dimension for simplicity
        return dim3(new_blocks, 1, 1);
    }
};

//==============================================================================
// Part 5: Kernel Resilience Extension
//==============================================================================

class KernelResilience {
private:
    struct KernelInfo {
        const void* func;
        dim3 gridDim;
        dim3 blockDim;
        void** args;
        size_t sharedMem;
        cudaStream_t stream;
        int retries;
    };
    
    static const int MAX_KERNELS = 100;
    KernelInfo history[MAX_KERNELS];
    int count;
    std::mutex mutex;
    
public:
    KernelResilience() : count(0) {}
    
    cudaError_t launchWithRetry(const void* func, dim3 gridDim, dim3 blockDim, 
                              void** args, size_t sharedMem, cudaStream_t stream,
                              int max_retries = 3) {
        std::lock_guard<std::mutex> lock(mutex);
        
        // Store kernel info
        int index = count % MAX_KERNELS;
        history[index].func = func;
        history[index].gridDim = gridDim;
        history[index].blockDim = blockDim;
        history[index].args = args;
        history[index].sharedMem = sharedMem;
        history[index].stream = stream;
        history[index].retries = 0;
        count++;
        
        // Launch kernel
        cudaError_t result = cudaLaunchKernel(func, gridDim, blockDim, 
                                            args, sharedMem, stream);
        
        // Check for errors and retry if needed
        int retry_count = 0;
        while (result != cudaSuccess && retry_count < max_retries) {
            printf("Kernel launch failed with error: %s\n", cudaGetErrorString(result));
            printf("Retrying kernel launch (attempt %d of %d)...\n", 
                   retry_count + 1, max_retries);
            
            // Reset device if necessary
            if (result == cudaErrorLaunchFailure) {
                cudaDeviceReset();
            }
            
            // Wait a moment before retrying
            usleep(100000);  // 100ms
            
            // Retry the launch
            result = cudaLaunchKernel(func, gridDim, blockDim, 
                                     args, sharedMem, stream);
            retry_count++;
        }
        
        if (result != cudaSuccess) {
            printf("All retry attempts failed. Last error: %s\n", 
                   cudaGetErrorString(result));
        } else if (retry_count > 0) {
            printf("Kernel launch successful after %d retries\n", retry_count);
        }
        
        return result;
    }
};

//==============================================================================
// Part 6: Demo Functions
//==============================================================================

// Global instances of our extensions
MemoryTracker g_memory_tracker;
MemoryPool g_memory_pool;
KernelOptimizer g_kernel_optimizer;
KernelResilience g_kernel_resilience;

/**
 * Demonstrates memory tracking extension
 */
void demoMemoryTracking(int vector_size) {
    printf("\n===== Memory Tracking Demo =====\n");
    
    // Allocate host memory
    size_t bytes = vector_size * sizeof(float);
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);
    
    // Initialize host arrays
    for (int i = 0; i < vector_size; i++) {
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }
    
    // Allocate device memory (tracked by our extension)
    float *d_a, *d_b, *d_c;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_a, bytes));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_b, bytes));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_c, bytes));
    
    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (vector_size + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, vector_size);
    
    // Copy results back
    CHECK_CUDA_ERROR(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    
    // Verify results
    for (int i = 0; i < 5; i++) { // Just check first few elements
        printf("h_c[%d] = %.2f\n", i, h_c[i]);
    }
    
    // Free memory (some leaks left for demo)
    cudaFree(d_a);
    cudaFree(d_b);
    // Intentionally don't free d_c to demonstrate leak detection
    
    free(h_a);
    free(h_b);
    free(h_c);
    
    // Print memory tracking stats
    g_memory_tracker.printStats();
}

/**
 * Demonstrates memory pooling extension
 */
void demoMemoryPooling(int iterations, int vector_size) {
    printf("\n===== Memory Pooling Demo =====\n");
    
    // Allocate host memory
    size_t bytes = vector_size * sizeof(float);
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);
    
    // Initialize host arrays
    for (int i = 0; i < vector_size; i++) {
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }
    
    // Start timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    // Perform repeated allocations to demonstrate pooling
    printf("Performing %d allocation/free cycles...\n", iterations);
    
    for (int i = 0; i < iterations; i++) {
        // Allocate device memory with pooling
        float *d_a, *d_b, *d_c;
        g_memory_pool.allocate((void**)&d_a, bytes);
        g_memory_pool.allocate((void**)&d_b, bytes);
        g_memory_pool.allocate((void**)&d_c, bytes);
        
        // Copy data to device
        cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
        
        // Launch kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (vector_size + threadsPerBlock - 1) / threadsPerBlock;
        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, vector_size);
        
        // Copy result back
        cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
        
        // Return memory to pool
        g_memory_pool.free(d_a);
        g_memory_pool.free(d_b);
        g_memory_pool.free(d_c);
    }
    
    // Stop timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("Completed in %.2f ms (%.2f ms per iteration)\n", 
           milliseconds, milliseconds / iterations);
    
    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c);
    
    // Print memory pool stats
    g_memory_pool.printStats();
}

/**
 * Demonstrates kernel optimization extension
 */
void demoKernelOptimization(int vector_size) {
    printf("\n===== Kernel Optimization Demo =====\n");
    
    // Allocate host memory
    size_t bytes = vector_size * sizeof(float);
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c1 = (float*)malloc(bytes);
    float *h_c2 = (float*)malloc(bytes);
    
    // Initialize host arrays
    for (int i = 0; i < vector_size; i++) {
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }
    
    // Allocate device memory
    float *d_a, *d_b, *d_c;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_a, bytes));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_b, bytes));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_c, bytes));
    
    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    
    // Original launch configuration
    int threadsPerBlock = 32; // Intentionally suboptimal
    int blocksPerGrid = (vector_size + threadsPerBlock - 1) / threadsPerBlock;
    dim3 blockDim(threadsPerBlock);
    dim3 gridDim(blocksPerGrid);
    
    printf("Original launch config: grid(%d, %d, %d), block(%d, %d, %d)\n",
           gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z);
    
    // Time original kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    vectorAddInefficient<<<gridDim, blockDim>>>(d_a, d_b, d_c, vector_size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float unoptimized_time = 0;
    cudaEventElapsedTime(&unoptimized_time, start, stop);
    
    // Copy result back
    CHECK_CUDA_ERROR(cudaMemcpy(h_c1, d_c, bytes, cudaMemcpyDeviceToHost));
    
    // Optimized launch configuration
    dim3 optimizedBlockDim = g_kernel_optimizer.optimizeBlockSize(gridDim, blockDim);
    dim3 optimizedGridDim = g_kernel_optimizer.optimizeGridSize(
        gridDim, blockDim, optimizedBlockDim);
    
    printf("Optimized launch config: grid(%d, %d, %d), block(%d, %d, %d)\n",
           optimizedGridDim.x, optimizedGridDim.y, optimizedGridDim.z, 
           optimizedBlockDim.x, optimizedBlockDim.y, optimizedBlockDim.z);
    
    // Time optimized kernel
    cudaEventRecord(start);
    vectorAddInefficient<<<optimizedGridDim, optimizedBlockDim>>>(d_a, d_b, d_c, vector_size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float optimized_time = 0;
    cudaEventElapsedTime(&optimized_time, start, stop);
    
    // Copy result back
    CHECK_CUDA_ERROR(cudaMemcpy(h_c2, d_c, bytes, cudaMemcpyDeviceToHost));
    
    // Report timing
    printf("Unoptimized kernel time: %.4f ms\n", unoptimized_time);
    printf("Optimized kernel time: %.4f ms\n", optimized_time);
    printf("Speedup: %.2fx\n", unoptimized_time / optimized_time);
    
    // Verify results match
    bool match = true;
    for (int i = 0; i < vector_size; i++) {
        if (fabs(h_c1[i] - h_c2[i]) > 1e-5) {
            printf("Results don't match at index %d: %.6f vs %.6f\n", 
                   i, h_c1[i], h_c2[i]);
            match = false;
            break;
        }
    }
    
    if (match) {
        printf("Results match: optimized kernel produces correct output\n");
    }
    
    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c1);
    free(h_c2);
}

/**
 * Demonstrates kernel resilience extension
 */
void demoKernelResilience(int vector_size) {
    printf("\n===== Kernel Resilience Demo =====\n");
    
    // Allocate host memory
    size_t bytes = vector_size * sizeof(float);
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);
    
    // Initialize host arrays
    for (int i = 0; i < vector_size; i++) {
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }
    
    // Allocate device memory
    float *d_a, *d_b, *d_c;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_a, bytes));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_b, bytes));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_c, bytes));
    
    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    
    // Launch configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (vector_size + threadsPerBlock - 1) / threadsPerBlock;
    
    // Prepare kernel arguments for manual launch
    void* args[] = { &d_a, &d_b, &d_c, &vector_size };
    
    printf("Launching kernel with resilience extension...\n");
    printf("(Note: This would normally retry on real errors, but we're simulating normal operation)\n");
    
    // Launch kernel with resilience
    g_kernel_resilience.launchWithRetry(
        (void*)vectorAdd, 
        dim3(blocksPerGrid), 
        dim3(threadsPerBlock), 
        args, 
        0, // shared memory
        0  // stream
    );
    
    // Copy result back
    CHECK_CUDA_ERROR(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    
    // Verify results
    for (int i = 0; i < 5; i++) { // Just check first few elements
        printf("h_c[%d] = %.2f\n", i, h_c[i]);
    }
    
    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
}

//==============================================================================
// Part 7: Main Function
//==============================================================================

/**
 * Simple interceptor demo
 */
cudaError_t intercepted_cudaMalloc(void** devPtr, size_t size) {
    static void* (*real_cudaMalloc)(void**, size_t) = NULL;
    
    // Get the real function pointer on first call
    if (real_cudaMalloc == NULL) {
        *(void**)(&real_cudaMalloc) = dlsym(RTLD_NEXT, "cudaMalloc");
    }
    
    // Call the real function
    cudaError_t result = ((cudaError_t(*)(void**, size_t))real_cudaMalloc)(devPtr, size);
    
    // Add our tracking
    if (result == cudaSuccess && g_extensions_enabled) {
        g_memory_tracker.recordAllocation(*devPtr, size);
    }
    
    return result;
}

cudaError_t intercepted_cudaFree(void* devPtr) {
    static void* (*real_cudaFree)(void*) = NULL;
    
    // Get the real function pointer on first call
    if (real_cudaFree == NULL) {
        *(void**)(&real_cudaFree) = dlsym(RTLD_NEXT, "cudaFree");
    }
    
    // Add our tracking before freeing
    if (g_extensions_enabled) {
        g_memory_tracker.recordFree(devPtr);
    }
    
    // Call the real function
    return ((cudaError_t(*)(void*))real_cudaFree)(devPtr);
}

int main(int argc, char **argv) {
    printf("CUDA GPU Application Extension Mechanisms Demo\n");
    printf("==============================================\n\n");
    
    // Get device info
    int deviceCount = 0;
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&deviceCount));
    
    if (deviceCount == 0) {
        printf("No CUDA-capable devices found\n");
        return 1;
    }
    
    printf("Found %d CUDA-capable device(s)\n", deviceCount);
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, i));
        
        printf("Device %d: %s\n", i, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
    }
    
    printf("\nNote: In a real application, these extensions would be in a shared library\n");
    printf("      loaded with LD_PRELOAD or similar mechanisms. For demo purposes,\n");
    printf("      we're showing the functionality directly.\n\n");
    
    // Enable extensions for the demos
    g_extensions_enabled = true;
    
    // Run demos
    int vector_size = 1024 * 1024; // 1M elements
    
    demoMemoryTracking(vector_size);
    demoMemoryPooling(10, vector_size);
    demoKernelOptimization(vector_size);
    demoKernelResilience(vector_size);
    
    printf("\nAll demos completed successfully!\n");
    
    return 0;
} 