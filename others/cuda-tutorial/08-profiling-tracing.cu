/**
 * CUDA GPU Profiling and Tracing Example
 * 
 * This file demonstrates various profiling and tracing techniques for CUDA applications:
 * 1. CUDA Events for basic timing
 * 2. NVTX Markers and Ranges for custom annotations
 * 3. CUPTI API for advanced profiling
 * 4. Comparing optimized vs. unoptimized kernels
 * 
 * Compile with: nvcc -o basic08 basic08.cu -lcupti -lnvToolsExt
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

// Include CUPTI
#include <cupti.h>

// Include NVTX for markers and ranges
#include <nvToolsExt.h>

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

#define CHECK_CUPTI_ERROR(call, msg)                                           \
do {                                                                           \
    CUptiResult _status = call;                                                \
    if (_status != CUPTI_SUCCESS) {                                            \
        const char *errstr;                                                    \
        cuptiGetResultString(_status, &errstr);                                \
        printf("%s: CUPTI error '%s'\n", msg, errstr);                         \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while (0)

// Constants
#define SIZE (1024 * 1024)  // 1M elements
#define BLOCK_SIZE 256
#define GRID_SIZE ((SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE)

// NVTX color definitions
#define NVTX_COLOR_BLUE     0xFF0000FF
#define NVTX_COLOR_GREEN    0xFF00FF00
#define NVTX_COLOR_RED      0xFFFF0000
#define NVTX_COLOR_YELLOW   0xFFFFFF00
#define NVTX_COLOR_MAGENTA  0xFFFF00FF
#define NVTX_COLOR_CYAN     0xFF00FFFF

// ===== KERNEL IMPLEMENTATIONS =====

/**
 * Unoptimized version of vector addition
 */
__global__ void vectorAddUnoptimized(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // Intentionally add delay with redundant operations to show profiling difference
        float temp = 0.0f;
        for (int j = 0; j < 50; j++) {
            temp = a[i] + b[i];
        }
        c[i] = temp;
    }
}

/**
 * Optimized version of vector addition
 */
__global__ void vectorAddOptimized(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

/**
 * Kernel with non-coalesced memory access
 */
__global__ void nonCoalescedAccess(float *input, float *output, int n) {
    int i = threadIdx.x * n / blockDim.x; // Non-coalesced access pattern
    if (i < n) {
        output[i] = input[i] * 2.0f;
    }
}

/**
 * Kernel with coalesced memory access
 */
__global__ void coalescedAccess(float *input, float *output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Coalesced access pattern
    if (i < n) {
        output[i] = input[i] * 2.0f;
    }
}

/**
 * Kernel with warp divergence
 */
__global__ void divergentKernel(float *input, float *output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (threadIdx.x % 2 == 0) {
            // Even threads do this
            for (int j = 0; j < 100; j++) {
                output[i] = input[i] * input[i];
            }
        } else {
            // Odd threads do this
            for (int j = 0; j < 10; j++) {
                output[i] = input[i] + input[i];
            }
        }
    }
}

/**
 * Kernel without warp divergence
 */
__global__ void nonDivergentKernel(float *input, float *output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // All threads in a warp do the same work
        output[i] = input[i] * input[i];
    }
}

// ===== CUPTI CALLBACK FUNCTIONS =====

// Simplified callback data structure
typedef struct {
    const char *name;
    int value;
} CallbackData;

// Generic callback handler for CUPTI
void CUPTIAPI cuptiCallbackHandler(void *userdata, CUpti_CallbackDomain domain,
                                  CUpti_CallbackId cbid, const void *cbdata) {
    // This is a simplified version that just prints basic information
    switch (domain) {
        case CUPTI_CB_DOMAIN_RESOURCE:
            printf("CUPTI: Resource callback, ID: %d\n", cbid);
            break;
        case CUPTI_CB_DOMAIN_RUNTIME_API:
            printf("CUPTI: Runtime API callback, ID: %d\n", cbid);
            break;
        case CUPTI_CB_DOMAIN_DRIVER_API:
            printf("CUPTI: Driver API callback, ID: %d\n", cbid);
            break;
        default:
            printf("CUPTI: Unknown domain callback, Domain: %d, ID: %d\n", domain, cbid);
            break;
    }
}

// ===== EXAMPLE FUNCTIONS =====

/**
 * Demonstrate basic timing using CUDA events
 */
void demonstrateCudaEvents() {
    printf("\n=== CUDA Events Timing Example ===\n");
    
    // Allocate host memory
    float *h_a = (float *)malloc(SIZE * sizeof(float));
    float *h_b = (float *)malloc(SIZE * sizeof(float));
    float *h_c = (float *)malloc(SIZE * sizeof(float));
    
    // Initialize data
    for (int i = 0; i < SIZE; i++) {
        h_a[i] = i * 0.5f;
        h_b[i] = i * 0.75f;
    }
    
    // Allocate device memory
    float *d_a, *d_b, *d_c;
    CHECK_CUDA_ERROR(cudaMalloc(&d_a, SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_b, SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_c, SIZE * sizeof(float)));
    
    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_a, h_a, SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b, h_b, SIZE * sizeof(float), cudaMemcpyHostToDevice));
    
    // Create CUDA events
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    
    // Time unoptimized kernel
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    vectorAddUnoptimized<<<GRID_SIZE, BLOCK_SIZE>>>(d_a, d_b, d_c, SIZE);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    
    float unoptimized_time = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&unoptimized_time, start, stop));
    printf("Unoptimized kernel time: %.4f ms\n", unoptimized_time);
    
    // Time optimized kernel
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    vectorAddOptimized<<<GRID_SIZE, BLOCK_SIZE>>>(d_a, d_b, d_c, SIZE);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    
    float optimized_time = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&optimized_time, start, stop));
    printf("Optimized kernel time: %.4f ms\n", optimized_time);
    printf("Speedup: %.2fx\n", unoptimized_time / optimized_time);
    
    // Cleanup
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    CHECK_CUDA_ERROR(cudaFree(d_a));
    CHECK_CUDA_ERROR(cudaFree(d_b));
    CHECK_CUDA_ERROR(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);
}

/**
 * Demonstrate NVTX annotations
 */
void demonstrateNVTX() {
    printf("\n=== NVTX Markers and Ranges Example ===\n");
    printf("Note: These annotations are visible in Nsight Systems timeline\n");
    
    // Create NVTX event attributes for different colors
    nvtxEventAttributes_t eventAttrib = {0};
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.colorType = NVTX_COLOR_ARGB;
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    
    // Allocate memory
    float *h_a = (float *)malloc(SIZE * sizeof(float));
    float *h_b = (float *)malloc(SIZE * sizeof(float));
    float *h_c = (float *)malloc(SIZE * sizeof(float));
    
    // Initialize data with NVTX marker
    nvtxMarkA("Data initialization started");
    for (int i = 0; i < SIZE; i++) {
        h_a[i] = i * 0.5f;
        h_b[i] = i * 0.75f;
    }
    
    // Allocate device memory with NVTX range
    nvtxRangePushA("Device memory allocation");
    float *d_a, *d_b, *d_c;
    CHECK_CUDA_ERROR(cudaMalloc(&d_a, SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_b, SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_c, SIZE * sizeof(float)));
    nvtxRangePop();
    
    // Copy data to device with colored NVTX range
    eventAttrib.color = NVTX_COLOR_BLUE;
    eventAttrib.message.ascii = "Host to device transfer";
    nvtxRangePushEx(&eventAttrib);
    CHECK_CUDA_ERROR(cudaMemcpy(d_a, h_a, SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b, h_b, SIZE * sizeof(float), cudaMemcpyHostToDevice));
    nvtxRangePop();
    
    // Run kernel with NVTX range
    eventAttrib.color = NVTX_COLOR_GREEN;
    eventAttrib.message.ascii = "Kernel execution";
    nvtxRangePushEx(&eventAttrib);
    vectorAddOptimized<<<GRID_SIZE, BLOCK_SIZE>>>(d_a, d_b, d_c, SIZE);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    nvtxRangePop();
    
    // Copy results back with NVTX range
    eventAttrib.color = NVTX_COLOR_RED;
    eventAttrib.message.ascii = "Device to host transfer";
    nvtxRangePushEx(&eventAttrib);
    CHECK_CUDA_ERROR(cudaMemcpy(h_c, d_c, SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    nvtxRangePop();
    
    nvtxMarkA("Computation completed");
    
    // Cleanup
    CHECK_CUDA_ERROR(cudaFree(d_a));
    CHECK_CUDA_ERROR(cudaFree(d_b));
    CHECK_CUDA_ERROR(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);
    
    printf("NVTX markers and ranges have been inserted\n");
}

/**
 * Compare coalesced vs non-coalesced memory access
 */
void compareMemoryAccess() {
    printf("\n=== Memory Access Pattern Comparison ===\n");
    
    // Allocate and initialize host memory
    float *h_input = (float *)malloc(SIZE * sizeof(float));
    float *h_output = (float *)malloc(SIZE * sizeof(float));
    
    for (int i = 0; i < SIZE; i++) {
        h_input[i] = i * 0.1f;
    }
    
    // Allocate device memory
    float *d_input, *d_output;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, SIZE * sizeof(float)));
    
    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, SIZE * sizeof(float), cudaMemcpyHostToDevice));
    
    // Create CUDA events
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    
    // Time non-coalesced kernel
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    nonCoalescedAccess<<<GRID_SIZE, BLOCK_SIZE>>>(d_input, d_output, SIZE);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    
    float noncoalesced_time = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&noncoalesced_time, start, stop));
    printf("Non-coalesced access time: %.4f ms\n", noncoalesced_time);
    
    // Time coalesced kernel
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    coalescedAccess<<<GRID_SIZE, BLOCK_SIZE>>>(d_input, d_output, SIZE);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    
    float coalesced_time = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&coalesced_time, start, stop));
    printf("Coalesced access time: %.4f ms\n", coalesced_time);
    printf("Speedup: %.2fx\n", noncoalesced_time / coalesced_time);
    
    // Cleanup
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    free(h_input);
    free(h_output);
}

/**
 * Compare divergent vs non-divergent execution
 */
void compareWarpDivergence() {
    printf("\n=== Warp Divergence Comparison ===\n");
    
    // Allocate and initialize host memory
    float *h_input = (float *)malloc(SIZE * sizeof(float));
    float *h_output = (float *)malloc(SIZE * sizeof(float));
    
    for (int i = 0; i < SIZE; i++) {
        h_input[i] = i * 0.1f;
    }
    
    // Allocate device memory
    float *d_input, *d_output;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, SIZE * sizeof(float)));
    
    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, SIZE * sizeof(float), cudaMemcpyHostToDevice));
    
    // Create CUDA events
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    
    // Time divergent kernel
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    divergentKernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_input, d_output, SIZE);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    
    float divergent_time = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&divergent_time, start, stop));
    printf("Divergent kernel time: %.4f ms\n", divergent_time);
    
    // Time non-divergent kernel
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    nonDivergentKernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_input, d_output, SIZE);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    
    float nondivergent_time = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&nondivergent_time, start, stop));
    printf("Non-divergent kernel time: %.4f ms\n", nondivergent_time);
    printf("Speedup: %.2fx\n", divergent_time / nondivergent_time);
    
    // Cleanup
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    free(h_input);
    free(h_output);
}

/**
 * Demonstrate CUPTI profiling with simplified API
 */
void demonstrateCUPTI() {
    printf("\n=== CUPTI Profiling Example ===\n");
    
    // Initialize CUPTI
    CUpti_SubscriberHandle subscriber;
    CHECK_CUPTI_ERROR(cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)cuptiCallbackHandler, 
                                     NULL), "cuptiSubscribe");
    
    // Enable callbacks for each domain individually instead of using cuptiEnableAllDomains
    // which might not be available in all CUPTI versions
    printf("CUPTI: Enabling callbacks for all domains\n");
    
    // Enable resource domain callbacks (context creation, etc.)
    CHECK_CUPTI_ERROR(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RESOURCE), 
                     "cuptiEnableDomain-RESOURCE");
    
    // Enable runtime API domain callbacks (CUDA runtime calls)
    CHECK_CUPTI_ERROR(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API), 
                     "cuptiEnableDomain-RUNTIME");
    
    // Enable driver API domain callbacks (CUDA driver calls)
    CHECK_CUPTI_ERROR(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API), 
                     "cuptiEnableDomain-DRIVER");
    
    printf("CUPTI: Enabled callbacks for all domains\n");
    
    // Allocate memory and prepare data
    float *h_a = (float *)malloc(SIZE * sizeof(float));
    float *h_b = (float *)malloc(SIZE * sizeof(float));
    float *h_c = (float *)malloc(SIZE * sizeof(float));
    
    for (int i = 0; i < SIZE; i++) {
        h_a[i] = i * 0.5f;
        h_b[i] = i * 0.75f;
    }
    
    float *d_a, *d_b, *d_c;
    CHECK_CUDA_ERROR(cudaMalloc(&d_a, SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_b, SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_c, SIZE * sizeof(float)));
    
    CHECK_CUDA_ERROR(cudaMemcpy(d_a, h_a, SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b, h_b, SIZE * sizeof(float), cudaMemcpyHostToDevice));
    
    // Launch a kernel
    printf("Launching kernel with CUPTI monitoring...\n");
    vectorAddOptimized<<<GRID_SIZE, BLOCK_SIZE>>>(d_a, d_b, d_c, SIZE);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Cleanup
    CHECK_CUPTI_ERROR(cuptiUnsubscribe(subscriber), "cuptiUnsubscribe");
    
    CHECK_CUDA_ERROR(cudaFree(d_a));
    CHECK_CUDA_ERROR(cudaFree(d_b));
    CHECK_CUDA_ERROR(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);
}

/**
 * Demonstrate basic CUPTI metrics collection
 */
void demonstrateCUPTIMetrics() {
    printf("\n=== CUPTI Metrics Collection Example ===\n");
    
    // Check device properties
    cudaDeviceProp prop;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    
    printf("\nCUPTI metrics collection requires Nsight Compute or nvprof.\n");
    printf("Run the program with these tools to collect detailed metrics.\n");
    printf("Example command: nvprof --metrics achieved_occupancy,ipc ./basic08\n");
}

int main(int argc, char **argv) {
    // Print device information
    int deviceCount = 0;
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&deviceCount));
    
    if (deviceCount == 0) {
        printf("Error: No CUDA-capable devices found\n");
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
        printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
        printf("  Max Threads Dimensions: (%d, %d, %d)\n", 
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  Max Grid Dimensions: (%d, %d, %d)\n", 
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("\n");
    }
    
    // Run examples
    demonstrateCudaEvents();
    demonstrateNVTX();
    compareMemoryAccess();
    compareWarpDivergence();
    demonstrateCUPTI();
    demonstrateCUPTIMetrics();
    
    printf("\nProfiling and tracing examples completed.\n");
    printf("For detailed analysis, run with Nsight Systems or Nsight Compute.\n");
    
    return 0;
} 