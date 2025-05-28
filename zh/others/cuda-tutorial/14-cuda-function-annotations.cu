#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#include <assert.h>

// ================================
// UTILITY FUNCTIONS
// ================================

// __host__ __device__ utility functions to avoid linter errors
__host__ __device__ int min_int(int a, int b) {
    return (a < b) ? a : b;
}

__host__ __device__ float min_float(float a, float b) {
    return (a < b) ? a : b;
}

__host__ __device__ int max_int(int a, int b) {
    return (a > b) ? a : b;
}

// ================================
// MEMORY SPACE ANNOTATIONS DEMO
// ================================

// Constant memory for filter coefficients
__constant__ float gaussianKernel[9] = {
    0.0625f, 0.125f, 0.0625f,
    0.125f,  0.25f,  0.125f,
    0.0625f, 0.125f, 0.0625f
};

// ================================
// FUNCTION TYPE ANNOTATIONS DEMO
// ================================

// 1. __device__ function - runs on GPU, called from GPU
__device__ float computeSquaredDistance(float x1, float y1, float x2, float y2) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    return dx * dx + dy * dy;
}

// 2. __device__ function for mathematical operations
__device__ float fastInverseSqrt(float number) {
    long i;
    float x2, y;
    const float threehalfs = 1.5f;
    
    x2 = number * 0.5f;
    y = number;
    i = *(long*)&y;                       // Evil floating point bit level hacking
    i = 0x5f3759df - (i >> 1);           // What the...?
    y = *(float*)&i;
    y = y * (threehalfs - (x2 * y * y));  // 1st iteration
    return y;
}

// 3. __host__ __device__ function - works on both CPU and GPU
__host__ __device__ float linearInterpolation(float a, float b, float t) {
    return a + t * (b - a);
}

// 4. __host__ __device__ utility function
__host__ __device__ int clamp(int value, int min_val, int max_val) {
    return max_int(min_val, min_int(value, max_val));
}

// 5. __host__ __device__ mathematical function
__host__ __device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// ================================
// KERNEL FUNCTIONS (__global__)
// ================================

// Basic kernel demonstrating device function calls
__global__ void calculateDistances(float* x1, float* y1, float* x2, float* y2, 
                                 float* distances, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float squaredDist = computeSquaredDistance(x1[idx], y1[idx], x2[idx], y2[idx]);
        distances[idx] = sqrtf(squaredDist);
    }
}

// Kernel demonstrating shared memory usage
__global__ void gaussianBlur(float* input, float* output, int width, int height) {
    // Shared memory for tile-based processing
    __shared__ float tile[18][18]; // 16x16 + 2-pixel border for 3x3 kernel
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int gx = blockIdx.x * blockDim.x + tx;
    int gy = blockIdx.y * blockDim.y + ty;
    
    // Load data into shared memory with border handling
    int sharedX = tx + 1;
    int sharedY = ty + 1;
    
    // Load center
    if (gx < width && gy < height) {
        tile[sharedY][sharedX] = input[gy * width + gx];
    } else {
        tile[sharedY][sharedX] = 0.0f;
    }
    
    // Load borders (simplified for demonstration)
    if (tx == 0) { // Left border
        int borderX = gx - 1;
        if (borderX >= 0 && gy < height) {
            tile[sharedY][0] = input[gy * width + borderX];
        } else {
            tile[sharedY][0] = 0.0f;
        }
    }
    
    if (tx == blockDim.x - 1) { // Right border
        int borderX = gx + 1;
        if (borderX < width && gy < height) {
            tile[sharedY][sharedX + 1] = input[gy * width + borderX];
        } else {
            tile[sharedY][sharedX + 1] = 0.0f;
        }
    }
    
    if (ty == 0) { // Top border
        int borderY = gy - 1;
        if (borderY >= 0 && gx < width) {
            tile[0][sharedX] = input[borderY * width + gx];
        } else {
            tile[0][sharedX] = 0.0f;
        }
    }
    
    if (ty == blockDim.y - 1) { // Bottom border
        int borderY = gy + 1;
        if (borderY < height && gx < width) {
            tile[sharedY + 1][sharedX] = input[borderY * width + gx];
        } else {
            tile[sharedY + 1][sharedX] = 0.0f;
        }
    }
    
    __syncthreads();
    
    // Apply Gaussian blur using constant memory kernel
    if (gx < width && gy < height) {
        float result = 0.0f;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                result += tile[sharedY + i - 1][sharedX + j - 1] * gaussianKernel[i * 3 + j];
            }
        }
        output[gy * width + gx] = result;
    }
}

// Kernel demonstrating __host__ __device__ function usage
__global__ void activationFunction(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Use the sigmoid function that works on both host and device
        output[idx] = sigmoid(input[idx]);
    }
}

// Kernel demonstrating multiple device function calls
__global__ void normalizeVectors(float* x, float* y, float* norm_x, float* norm_y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float magnitude = sqrtf(x[idx] * x[idx] + y[idx] * y[idx]);
        if (magnitude > 0.0f) {
            float inv_magnitude = fastInverseSqrt(magnitude * magnitude);
            norm_x[idx] = x[idx] * inv_magnitude;
            norm_y[idx] = y[idx] * inv_magnitude;
        } else {
            norm_x[idx] = 0.0f;
            norm_y[idx] = 0.0f;
        }
    }
}

// ================================
// HOST FUNCTIONS
// ================================

// Regular host function (implicit __host__)
void initializeRandomData(float* data, int n, float min_val, float max_val) {
    for (int i = 0; i < n; i++) {
        data[i] = min_val + (max_val - min_val) * ((float)rand() / RAND_MAX);
    }
}

// Explicit __host__ function
__host__ void printResults(const char* title, float* data, int n, int max_print = 10) {
    printf("\n%s (showing first %d elements):\n", title, max_print);
    for (int i = 0; i < min_int(n, max_print); i++) {
        printf("%.4f ", data[i]);
    }
    printf("\n");
}

// Host function using __host__ __device__ function
__host__ void preprocessOnCPU(float* data, int n) {
    printf("\nPreprocessing data on CPU using __host__ __device__ function...\n");
    for (int i = 0; i < n - 1; i++) {
        // Use the linearInterpolation function that works on both host and device
        data[i] = linearInterpolation(data[i], data[i + 1], 0.3f);
        
        // Use the sigmoid function
        data[i] = sigmoid(data[i]);
        
        // Use the clamp function
        int clamped_val = clamp((int)(data[i] * 100), 0, 100);
        data[i] = clamped_val / 100.0f;
    }
}

// Error checking utility
void checkCudaError(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        printf("CUDA Error: %s - %s\n", message, cudaGetErrorString(error));
        exit(1);
    }
}

// ================================
// DEMONSTRATION FUNCTIONS
// ================================

void demonstrateBasicKernelUsage() {
    printf("\n=== Demonstrating Basic Kernel and Device Function Usage ===\n");
    
    const int n = 1000;
    float *h_x1, *h_y1, *h_x2, *h_y2, *h_distances;
    float *d_x1, *d_y1, *d_x2, *d_y2, *d_distances;
    
    // Allocate host memory
    h_x1 = (float*)malloc(n * sizeof(float));
    h_y1 = (float*)malloc(n * sizeof(float));
    h_x2 = (float*)malloc(n * sizeof(float));
    h_y2 = (float*)malloc(n * sizeof(float));
    h_distances = (float*)malloc(n * sizeof(float));
    
    // Initialize data
    initializeRandomData(h_x1, n, -10.0f, 10.0f);
    initializeRandomData(h_y1, n, -10.0f, 10.0f);
    initializeRandomData(h_x2, n, -10.0f, 10.0f);
    initializeRandomData(h_y2, n, -10.0f, 10.0f);
    
    // Allocate device memory
    checkCudaError(cudaMalloc(&d_x1, n * sizeof(float)), "Allocating d_x1");
    checkCudaError(cudaMalloc(&d_y1, n * sizeof(float)), "Allocating d_y1");
    checkCudaError(cudaMalloc(&d_x2, n * sizeof(float)), "Allocating d_x2");
    checkCudaError(cudaMalloc(&d_y2, n * sizeof(float)), "Allocating d_y2");
    checkCudaError(cudaMalloc(&d_distances, n * sizeof(float)), "Allocating d_distances");
    
    // Copy data to device
    checkCudaError(cudaMemcpy(d_x1, h_x1, n * sizeof(float), cudaMemcpyHostToDevice), "Copying x1 to device");
    checkCudaError(cudaMemcpy(d_y1, h_y1, n * sizeof(float), cudaMemcpyHostToDevice), "Copying y1 to device");
    checkCudaError(cudaMemcpy(d_x2, h_x2, n * sizeof(float), cudaMemcpyHostToDevice), "Copying x2 to device");
    checkCudaError(cudaMemcpy(d_y2, h_y2, n * sizeof(float), cudaMemcpyHostToDevice), "Copying y2 to device");
    
    // Launch kernel
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    calculateDistances<<<numBlocks, blockSize>>>(d_x1, d_y1, d_x2, d_y2, d_distances, n);
    checkCudaError(cudaGetLastError(), "Launching calculateDistances kernel");
    checkCudaError(cudaDeviceSynchronize(), "Synchronizing after kernel launch");
    
    // Copy result back
    checkCudaError(cudaMemcpy(h_distances, d_distances, n * sizeof(float), cudaMemcpyDeviceToHost), "Copying distances back");
    
    printResults("Distance calculation results", h_distances, n);
    
    // Cleanup
    free(h_x1); free(h_y1); free(h_x2); free(h_y2); free(h_distances);
    cudaFree(d_x1); cudaFree(d_y1); cudaFree(d_x2); cudaFree(d_y2); cudaFree(d_distances);
}

void demonstrateHostDeviceFunctions() {
    printf("\n=== Demonstrating __host__ __device__ Functions ===\n");
    
    const int n = 100;
    float *h_data = (float*)malloc(n * sizeof(float));
    
    // Initialize data
    initializeRandomData(h_data, n, -5.0f, 5.0f);
    printResults("Original data", h_data, n);
    
    // Process on CPU using __host__ __device__ functions
    preprocessOnCPU(h_data, n);
    printResults("Processed data on CPU", h_data, n);
    
    // Now process the same type of data on GPU
    float *d_input, *d_output;
    float *h_output = (float*)malloc(n * sizeof(float));
    
    checkCudaError(cudaMalloc(&d_input, n * sizeof(float)), "Allocating d_input");
    checkCudaError(cudaMalloc(&d_output, n * sizeof(float)), "Allocating d_output");
    
    // Reset data for GPU processing
    initializeRandomData(h_data, n, -5.0f, 5.0f);
    checkCudaError(cudaMemcpy(d_input, h_data, n * sizeof(float), cudaMemcpyHostToDevice), "Copying input to device");
    
    // Launch activation function kernel
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    activationFunction<<<numBlocks, blockSize>>>(d_input, d_output, n);
    checkCudaError(cudaGetLastError(), "Launching activationFunction kernel");
    checkCudaError(cudaDeviceSynchronize(), "Synchronizing after kernel launch");
    
    checkCudaError(cudaMemcpy(h_output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost), "Copying output back");
    printResults("Sigmoid activation on GPU", h_output, n);
    
    // Cleanup
    free(h_data); free(h_output);
    cudaFree(d_input); cudaFree(d_output);
}

void demonstrateSharedMemoryKernel() {
    printf("\n=== Demonstrating Shared Memory Usage ===\n");
    
    const int width = 64;
    const int height = 64;
    const int size = width * height;
    
    float *h_input = (float*)malloc(size * sizeof(float));
    float *h_output = (float*)malloc(size * sizeof(float));
    float *d_input, *d_output;
    
    // Initialize with a simple pattern
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            h_input[i * width + j] = (i + j) % 10; // Simple pattern
        }
    }
    
    checkCudaError(cudaMalloc(&d_input, size * sizeof(float)), "Allocating d_input");
    checkCudaError(cudaMalloc(&d_output, size * sizeof(float)), "Allocating d_output");
    
    checkCudaError(cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice), "Copying input to device");
    
    // Launch Gaussian blur kernel with 2D thread blocks
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
    
    gaussianBlur<<<gridDim, blockDim>>>(d_input, d_output, width, height);
    checkCudaError(cudaGetLastError(), "Launching gaussianBlur kernel");
    checkCudaError(cudaDeviceSynchronize(), "Synchronizing after kernel launch");
    
    checkCudaError(cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost), "Copying output back");
    
    // Print a small sample of results
    printf("Gaussian blur results (5x5 sample from center):\n");
    int start_y = height / 2 - 2;
    int start_x = width / 2 - 2;
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            int idx = (start_y + i) * width + (start_x + j);
            printf("%.2f ", h_output[idx]);
        }
        printf("\n");
    }
    
    // Cleanup
    free(h_input); free(h_output);
    cudaFree(d_input); cudaFree(d_output);
}

void demonstrateVectorNormalization() {
    printf("\n=== Demonstrating Vector Normalization with Device Functions ===\n");
    
    const int n = 1000;
    float *h_x, *h_y, *h_norm_x, *h_norm_y;
    float *d_x, *d_y, *d_norm_x, *d_norm_y;
    
    // Allocate host memory
    h_x = (float*)malloc(n * sizeof(float));
    h_y = (float*)malloc(n * sizeof(float));
    h_norm_x = (float*)malloc(n * sizeof(float));
    h_norm_y = (float*)malloc(n * sizeof(float));
    
    // Initialize random vectors
    initializeRandomData(h_x, n, -10.0f, 10.0f);
    initializeRandomData(h_y, n, -10.0f, 10.0f);
    
    // Allocate device memory
    checkCudaError(cudaMalloc(&d_x, n * sizeof(float)), "Allocating d_x");
    checkCudaError(cudaMalloc(&d_y, n * sizeof(float)), "Allocating d_y");
    checkCudaError(cudaMalloc(&d_norm_x, n * sizeof(float)), "Allocating d_norm_x");
    checkCudaError(cudaMalloc(&d_norm_y, n * sizeof(float)), "Allocating d_norm_y");
    
    // Copy to device
    checkCudaError(cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice), "Copying x to device");
    checkCudaError(cudaMemcpy(d_y, h_y, n * sizeof(float), cudaMemcpyHostToDevice), "Copying y to device");
    
    // Launch normalization kernel
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    normalizeVectors<<<numBlocks, blockSize>>>(d_x, d_y, d_norm_x, d_norm_y, n);
    checkCudaError(cudaGetLastError(), "Launching normalizeVectors kernel");
    checkCudaError(cudaDeviceSynchronize(), "Synchronizing after kernel launch");
    
    // Copy results back
    checkCudaError(cudaMemcpy(h_norm_x, d_norm_x, n * sizeof(float), cudaMemcpyDeviceToHost), "Copying norm_x back");
    checkCudaError(cudaMemcpy(h_norm_y, d_norm_y, n * sizeof(float), cudaMemcpyDeviceToHost), "Copying norm_y back");
    
    // Verify normalization (check that magnitudes are close to 1.0)
    printf("Verification of vector normalization:\n");
    for (int i = 0; i < 10; i++) {
        float original_mag = sqrtf(h_x[i] * h_x[i] + h_y[i] * h_y[i]);
        float normalized_mag = sqrtf(h_norm_x[i] * h_norm_x[i] + h_norm_y[i] * h_norm_y[i]);
        printf("Vector %d: Original=(%.3f,%.3f) |mag|=%.3f, Normalized=(%.3f,%.3f) |mag|=%.3f\n", 
               i, h_x[i], h_y[i], original_mag, h_norm_x[i], h_norm_y[i], normalized_mag);
    }
    
    // Cleanup
    free(h_x); free(h_y); free(h_norm_x); free(h_norm_y);
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_norm_x); cudaFree(d_norm_y);
}

// ================================
// MAIN FUNCTION
// ================================

int main() {
    printf("CUDA Function Type Annotations Demonstration\n");
    printf("============================================\n");
    
    // Check CUDA capability
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("No CUDA devices found!\n");
        return 1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Using device: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Shared memory per block: %zu bytes\n", prop.sharedMemPerBlock);
    printf("Constant memory: %zu bytes\n", prop.totalConstMem);
    
    // Run demonstrations
    demonstrateBasicKernelUsage();
    demonstrateHostDeviceFunctions();
    demonstrateSharedMemoryKernel();
    demonstrateVectorNormalization();
    
    printf("\n=== Summary of Function Type Annotations ===\n");
    printf("1. __global__: Kernel functions called from host, executed on device\n");
    printf("2. __device__: Functions called from device code, executed on device\n");
    printf("3. __host__: Functions called from host code, executed on host (default)\n");
    printf("4. __host__ __device__: Functions that work on both host and device\n");
    printf("5. __shared__: Variables shared among threads in a block\n");
    printf("6. __constant__: Read-only variables in constant memory cache\n");
    printf("\nAll demonstrations completed successfully!\n");
    
    return 0;
} 