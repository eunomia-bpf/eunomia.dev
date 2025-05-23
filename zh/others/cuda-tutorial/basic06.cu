/**
 * CUDA Basic Example 06 - Convolutional Neural Networks with Shared Memory
 * 
 * This example demonstrates how to implement efficient convolution operations for CNNs
 * using CUDA with shared memory optimization. Convolution is the core operation in
 * Convolutional Neural Networks (CNNs) widely used in computer vision tasks.
 * 
 * The implementation shows:
 * 1. Direct convolution implementation
 * 2. Shared memory optimized convolution
 * 3. Tiled convolution for larger inputs
 * 4. Comparison of different implementations
 * 5. Activation and pooling layers
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

// CNN configuration
#define INPUT_SIZE 28       // Input image width/height (assuming square)
#define INPUT_CHANNELS 1    // Number of input channels (1 for grayscale)
#define KERNEL_SIZE 5       // Convolution kernel width/height
#define KERNEL_COUNT 16     // Number of convolution kernels (output channels)
#define PADDING 2           // Zero padding size
#define STRIDE 1            // Convolution stride
#define BATCH_SIZE 64       // Batch size for processing

// Output dimensions after convolution
#define OUTPUT_SIZE ((INPUT_SIZE + 2*PADDING - KERNEL_SIZE) / STRIDE + 1)

// CUDA error checking macro
#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s, in file '%s', line %d\n", \
                cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

// Initialize convolution kernels (filters)
void initializeKernels(float *kernels) {
    // Xavier/Glorot initialization for CNN kernels
    float scale = sqrtf(6.0f / (KERNEL_SIZE * KERNEL_SIZE * (INPUT_CHANNELS + KERNEL_COUNT)));
    
    for (int k = 0; k < KERNEL_COUNT; k++) {
        for (int c = 0; c < INPUT_CHANNELS; c++) {
            for (int i = 0; i < KERNEL_SIZE; i++) {
                for (int j = 0; j < KERNEL_SIZE; j++) {
                    int idx = k * INPUT_CHANNELS * KERNEL_SIZE * KERNEL_SIZE + 
                              c * KERNEL_SIZE * KERNEL_SIZE + 
                              i * KERNEL_SIZE + j;
                    kernels[idx] = (2.0f * (float)rand() / RAND_MAX - 1.0f) * scale;
                }
            }
        }
    }
}

// Generate random input data (simulating MNIST-like images)
void generateRandomInput(float *input) {
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int c = 0; c < INPUT_CHANNELS; c++) {
            // Generate random "blob" pattern in the center
            int center_x = INPUT_SIZE / 2;
            int center_y = INPUT_SIZE / 2;
            int radius = INPUT_SIZE / 4;
            
            for (int i = 0; i < INPUT_SIZE; i++) {
                for (int j = 0; j < INPUT_SIZE; j++) {
                    float dist = sqrtf((i - center_x) * (i - center_x) + 
                                       (j - center_y) * (j - center_y));
                    
                    // Index for the current pixel
                    int idx = b * INPUT_CHANNELS * INPUT_SIZE * INPUT_SIZE + 
                              c * INPUT_SIZE * INPUT_SIZE + 
                              i * INPUT_SIZE + j;
                    
                    if (dist < radius) {
                        // Higher value inside the blob
                        input[idx] = 0.7f + 0.3f * (float)rand() / RAND_MAX;
                    } else {
                        // Lower values outside (background)
                        input[idx] = 0.1f * (float)rand() / RAND_MAX;
                    }
                }
            }
        }
    }
}

// Direct convolution kernel (baseline implementation without shared memory)
__global__ void convolutionDirectKernel(
    float *input, float *kernels, float *output,
    int batchSize, int inputChannels, int inputSize,
    int kernelSize, int kernelCount, int outputSize,
    int padding, int stride) 
{
    // Calculate output position
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z; // Output channel (kernel number)
    int b = threadIdx.z; // Batch index
    
    // Check bounds
    if (x >= outputSize || y >= outputSize || k >= kernelCount || b >= batchSize)
        return;
    
    // Compute convolution for this output position
    float sum = 0.0f;
    
    // For each input channel
    for (int c = 0; c < inputChannels; c++) {
        // For each kernel position
        for (int ky = 0; ky < kernelSize; ky++) {
            for (int kx = 0; kx < kernelSize; kx++) {
                // Input position
                int in_x = x * stride - padding + kx;
                int in_y = y * stride - padding + ky;
                
                // Skip if input position is outside the input
                if (in_x >= 0 && in_x < inputSize && in_y >= 0 && in_y < inputSize) {
                    // Input value
                    float in_val = input[
                        b * inputChannels * inputSize * inputSize +
                        c * inputSize * inputSize +
                        in_y * inputSize + in_x
                    ];
                    
                    // Kernel value
                    float kernel_val = kernels[
                        k * inputChannels * kernelSize * kernelSize +
                        c * kernelSize * kernelSize +
                        ky * kernelSize + kx
                    ];
                    
                    // Accumulate result
                    sum += in_val * kernel_val;
                }
            }
        }
    }
    
    // Store output
    output[
        b * kernelCount * outputSize * outputSize +
        k * outputSize * outputSize +
        y * outputSize + x
    ] = sum;
}

// Shared memory optimized convolution kernel
__global__ void convolutionSharedKernel(
    float *input, float *kernels, float *output,
    int batchSize, int inputChannels, int inputSize,
    int kernelSize, int kernelCount, int outputSize,
    int padding, int stride) 
{
    // Shared memory for input tile (with padding)
    extern __shared__ float sharedData[];
    
    // Calculate tile dimensions
    int tileSize = blockDim.x; // Assuming blockDim.x == blockDim.y
    int tileSizeWithPadding = tileSize + kernelSize - 1;
    
    // Calculate output position
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int k = blockIdx.z; // Output channel (kernel number)
    int b = threadIdx.z; // Batch index
    
    // Output coordinates
    int out_x = bx * tileSize + tx;
    int out_y = by * tileSize + ty;
    
    // Base input coordinates (top-left of the tile)
    int in_x_base = bx * tileSize * stride - padding;
    int in_y_base = by * tileSize * stride - padding;
    
    // Pointers to shared memory
    float *sharedInput = sharedData;
    
    // Load input data to shared memory
    for (int c = 0; c < inputChannels; c++) {
        // Each thread loads multiple elements to cover the tile with padding
        for (int dy = 0; dy < tileSizeWithPadding; dy += tileSize) {
            for (int dx = 0; dx < tileSizeWithPadding; dx += tileSize) {
                int in_y = in_y_base + ty + dy;
                int in_x = in_x_base + tx + dx;
                
                // Check bounds and apply padding
                float value = 0.0f;
                if (in_y >= 0 && in_y < inputSize && in_x >= 0 && in_x < inputSize) {
                    value = input[
                        b * inputChannels * inputSize * inputSize +
                        c * inputSize * inputSize +
                        in_y * inputSize + in_x
                    ];
                }
                
                // Store in shared memory if within tile bounds
                if (ty + dy < tileSizeWithPadding && tx + dx < tileSizeWithPadding) {
                    sharedInput[
                        c * tileSizeWithPadding * tileSizeWithPadding +
                        (ty + dy) * tileSizeWithPadding + (tx + dx)
                    ] = value;
                }
            }
        }
    }
    
    // Ensure all threads have loaded data to shared memory
    __syncthreads();
    
    // Compute convolution if within output bounds
    if (out_x < outputSize && out_y < outputSize && b < batchSize) {
        float sum = 0.0f;
        
        // For each input channel
        for (int c = 0; c < inputChannels; c++) {
            // For each kernel position
            for (int ky = 0; ky < kernelSize; ky++) {
                for (int kx = 0; kx < kernelSize; kx++) {
                    // Shared memory position
                    int shared_y = ty * stride + ky;
                    int shared_x = tx * stride + kx;
                    
                    // Input value from shared memory
                    float in_val = sharedInput[
                        c * tileSizeWithPadding * tileSizeWithPadding +
                        shared_y * tileSizeWithPadding + shared_x
                    ];
                    
                    // Kernel value
                    float kernel_val = kernels[
                        k * inputChannels * kernelSize * kernelSize +
                        c * kernelSize * kernelSize +
                        ky * kernelSize + kx
                    ];
                    
                    // Accumulate result
                    sum += in_val * kernel_val;
                }
            }
        }
        
        // Store output
        if (out_x < outputSize && out_y < outputSize) {
            output[
                b * kernelCount * outputSize * outputSize +
                k * outputSize * outputSize +
                out_y * outputSize + out_x
            ] = sum;
        }
    }
}

// ReLU activation kernel
__global__ void reluActivationKernel(float *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

// Max pooling kernel
__global__ void maxPoolingKernel(
    float *input, float *output,
    int batchSize, int channels, int inputSize,
    int poolSize, int outputSize, int stride)
{
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z % channels;
    int b = blockIdx.z / channels;
    
    if (out_x >= outputSize || out_y >= outputSize || c >= channels || b >= batchSize)
        return;
    
    // Input position (top-left corner of pooling window)
    int in_x_base = out_x * stride;
    int in_y_base = out_y * stride;
    
    float maxVal = -FLT_MAX;
    
    // Find maximum value in pooling window
    for (int dy = 0; dy < poolSize; dy++) {
        for (int dx = 0; dx < poolSize; dx++) {
            int in_y = in_y_base + dy;
            int in_x = in_x_base + dx;
            
            if (in_y < inputSize && in_x < inputSize) {
                float value = input[
                    b * channels * inputSize * inputSize +
                    c * inputSize * inputSize +
                    in_y * inputSize + in_x
                ];
                maxVal = fmaxf(maxVal, value);
            }
        }
    }
    
    // Store output
    output[
        b * channels * outputSize * outputSize +
        c * outputSize * outputSize +
        out_y * outputSize + out_x
    ] = maxVal;
}

// Forward pass for a simple CNN
void forwardCNN(
    float *d_input,            // Input images
    float *d_kernels,          // Convolution kernels
    float *d_conv_output,      // Convolution output
    float *d_activation,       // Activation output
    float *d_pooling_output,   // Pooling output
    bool useSharedMemory,      // Use shared memory optimization
    float *timing              // Timing output (conv, act, pool)
) {
    // Timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 1. Convolution Layer
    dim3 blockDim(8, 8, BATCH_SIZE); // Each block processes 8x8 output elements for all batches
    dim3 gridDim(
        (OUTPUT_SIZE + blockDim.x - 1) / blockDim.x,
        (OUTPUT_SIZE + blockDim.y - 1) / blockDim.y,
        KERNEL_COUNT
    );
    
    cudaEventRecord(start);
    
    if (useSharedMemory) {
        // Calculate shared memory size
        int tileSize = blockDim.x;
        int tileSizeWithPadding = tileSize + KERNEL_SIZE - 1;
        int sharedMemSize = INPUT_CHANNELS * tileSizeWithPadding * tileSizeWithPadding * sizeof(float);
        
        convolutionSharedKernel<<<gridDim, blockDim, sharedMemSize>>>(
            d_input, d_kernels, d_conv_output,
            BATCH_SIZE, INPUT_CHANNELS, INPUT_SIZE,
            KERNEL_SIZE, KERNEL_COUNT, OUTPUT_SIZE,
            PADDING, STRIDE
        );
    } else {
        convolutionDirectKernel<<<gridDim, blockDim>>>(
            d_input, d_kernels, d_conv_output,
            BATCH_SIZE, INPUT_CHANNELS, INPUT_SIZE,
            KERNEL_SIZE, KERNEL_COUNT, OUTPUT_SIZE,
            PADDING, STRIDE
        );
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timing[0], start, stop);
    
    // 2. ReLU Activation
    int totalElements = BATCH_SIZE * KERNEL_COUNT * OUTPUT_SIZE * OUTPUT_SIZE;
    int blockSize = 256;
    int gridSize = (totalElements + blockSize - 1) / blockSize;
    
    // Copy convolution output to activation buffer
    cudaMemcpy(d_activation, d_conv_output, totalElements * sizeof(float), cudaMemcpyDeviceToDevice);
    
    cudaEventRecord(start);
    
    reluActivationKernel<<<gridSize, blockSize>>>(d_activation, totalElements);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timing[1], start, stop);
    
    // 3. Max Pooling Layer (2x2 with stride 2)
    int poolSize = 2;
    int poolStride = 2;
    int poolOutputSize = OUTPUT_SIZE / poolStride;
    
    dim3 poolBlockDim(8, 8);
    dim3 poolGridDim(
        (poolOutputSize + poolBlockDim.x - 1) / poolBlockDim.x,
        (poolOutputSize + poolBlockDim.y - 1) / poolBlockDim.y,
        BATCH_SIZE * KERNEL_COUNT
    );
    
    cudaEventRecord(start);
    
    maxPoolingKernel<<<poolGridDim, poolBlockDim>>>(
        d_activation, d_pooling_output,
        BATCH_SIZE, KERNEL_COUNT, OUTPUT_SIZE,
        poolSize, poolOutputSize, poolStride
    );
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timing[2], start, stop);
}

int main() {
    // Set random seed for reproducibility
    srand(42);
    
    // Allocate host memory
    float *h_input = (float*)malloc(BATCH_SIZE * INPUT_CHANNELS * INPUT_SIZE * INPUT_SIZE * sizeof(float));
    float *h_kernels = (float*)malloc(KERNEL_COUNT * INPUT_CHANNELS * KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
    float *h_output_direct = (float*)malloc(BATCH_SIZE * KERNEL_COUNT * OUTPUT_SIZE * OUTPUT_SIZE * sizeof(float));
    float *h_output_shared = (float*)malloc(BATCH_SIZE * KERNEL_COUNT * OUTPUT_SIZE * OUTPUT_SIZE * sizeof(float));
    float *h_pooling_output = (float*)malloc(BATCH_SIZE * KERNEL_COUNT * (OUTPUT_SIZE/2) * (OUTPUT_SIZE/2) * sizeof(float));
    
    // Initialize data
    generateRandomInput(h_input);
    initializeKernels(h_kernels);
    
    // Allocate device memory
    float *d_input, *d_kernels;
    float *d_conv_output_direct, *d_activation_direct, *d_pooling_output_direct;
    float *d_conv_output_shared, *d_activation_shared, *d_pooling_output_shared;
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, BATCH_SIZE * INPUT_CHANNELS * INPUT_SIZE * INPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_kernels, KERNEL_COUNT * INPUT_CHANNELS * KERNEL_SIZE * KERNEL_SIZE * sizeof(float)));
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_conv_output_direct, BATCH_SIZE * KERNEL_COUNT * OUTPUT_SIZE * OUTPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_activation_direct, BATCH_SIZE * KERNEL_COUNT * OUTPUT_SIZE * OUTPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_pooling_output_direct, BATCH_SIZE * KERNEL_COUNT * (OUTPUT_SIZE/2) * (OUTPUT_SIZE/2) * sizeof(float)));
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_conv_output_shared, BATCH_SIZE * KERNEL_COUNT * OUTPUT_SIZE * OUTPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_activation_shared, BATCH_SIZE * KERNEL_COUNT * OUTPUT_SIZE * OUTPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_pooling_output_shared, BATCH_SIZE * KERNEL_COUNT * (OUTPUT_SIZE/2) * (OUTPUT_SIZE/2) * sizeof(float)));
    
    // Copy input data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, BATCH_SIZE * INPUT_CHANNELS * INPUT_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_kernels, h_kernels, KERNEL_COUNT * INPUT_CHANNELS * KERNEL_SIZE * KERNEL_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    
    // Run direct convolution
    float direct_timing[3] = {0}; // [conv, act, pool]
    forwardCNN(
        d_input, d_kernels,
        d_conv_output_direct, d_activation_direct, d_pooling_output_direct,
        false, // use direct convolution
        direct_timing
    );
    
    // Run shared memory convolution
    float shared_timing[3] = {0}; // [conv, act, pool]
    forwardCNN(
        d_input, d_kernels,
        d_conv_output_shared, d_activation_shared, d_pooling_output_shared,
        true, // use shared memory
        shared_timing
    );
    
    // Copy results back for verification
    CHECK_CUDA_ERROR(cudaMemcpy(h_output_direct, d_conv_output_direct, 
                              BATCH_SIZE * KERNEL_COUNT * OUTPUT_SIZE * OUTPUT_SIZE * sizeof(float),
                              cudaMemcpyDeviceToHost));
    
    CHECK_CUDA_ERROR(cudaMemcpy(h_output_shared, d_conv_output_shared,
                              BATCH_SIZE * KERNEL_COUNT * OUTPUT_SIZE * OUTPUT_SIZE * sizeof(float),
                              cudaMemcpyDeviceToHost));
    
    CHECK_CUDA_ERROR(cudaMemcpy(h_pooling_output, d_pooling_output_shared,
                              BATCH_SIZE * KERNEL_COUNT * (OUTPUT_SIZE/2) * (OUTPUT_SIZE/2) * sizeof(float),
                              cudaMemcpyDeviceToHost));
    
    // Verify results match between direct and shared memory implementations
    float maxDiff = 0.0f;
    for (int i = 0; i < BATCH_SIZE * KERNEL_COUNT * OUTPUT_SIZE * OUTPUT_SIZE; i++) {
        float diff = fabs(h_output_direct[i] - h_output_shared[i]);
        maxDiff = fmaxf(maxDiff, diff);
    }
    
    // Print results
    printf("=== CNN Convolution with Shared Memory Optimization ===\n\n");
    printf("Configuration:\n");
    printf("  Input size: %dx%d with %d channels\n", INPUT_SIZE, INPUT_SIZE, INPUT_CHANNELS);
    printf("  Kernel size: %dx%d with %d output channels\n", KERNEL_SIZE, KERNEL_SIZE, KERNEL_COUNT);
    printf("  Padding: %d, Stride: %d\n", PADDING, STRIDE);
    printf("  Output size after convolution: %dx%d\n", OUTPUT_SIZE, OUTPUT_SIZE);
    printf("  Output size after pooling: %dx%d\n\n", OUTPUT_SIZE/2, OUTPUT_SIZE/2);
    
    printf("Performance comparison:\n");
    printf("  Direct Convolution: %.3f ms\n", direct_timing[0]);
    printf("  Shared Memory Convolution: %.3f ms\n", shared_timing[0]);
    printf("  Speedup: %.2fx\n\n", direct_timing[0] / shared_timing[0]);
    
    printf("Layer timings (Shared Memory):\n");
    printf("  Convolution: %.3f ms\n", shared_timing[0]);
    printf("  ReLU Activation: %.3f ms\n", shared_timing[1]);
    printf("  Max Pooling: %.3f ms\n\n", shared_timing[2]);
    
    printf("Verification:\n");
    printf("  Max difference between implementations: %e\n", maxDiff);
    if (maxDiff < 1e-5) {
        printf("  Results match: YES\n\n");
    } else {
        printf("  Results match: NO (possible numerical precision issues)\n\n");
    }
    
    // Sample output visualization
    printf("Sample output feature map (first batch, first channel):\n");
    for (int y = 0; y < 5; y++) {
        for (int x = 0; x < 5; x++) {
            printf("%.4f ", h_output_shared[y * OUTPUT_SIZE + x]);
        }
        printf("...\n");
    }
    printf("...\n\n");
    
    // Sample pooling output
    printf("Sample pooling output (first batch, first channel):\n");
    for (int y = 0; y < 5; y++) {
        for (int x = 0; x < 5; x++) {
            printf("%.4f ", h_pooling_output[y * (OUTPUT_SIZE/2) + x]);
        }
        printf("...\n");
    }
    printf("...\n");
    
    // Free memory
    free(h_input);
    free(h_kernels);
    free(h_output_direct);
    free(h_output_shared);
    free(h_pooling_output);
    
    cudaFree(d_input);
    cudaFree(d_kernels);
    cudaFree(d_conv_output_direct);
    cudaFree(d_activation_direct);
    cudaFree(d_pooling_output_direct);
    cudaFree(d_conv_output_shared);
    cudaFree(d_activation_shared);
    cudaFree(d_pooling_output_shared);
    
    return 0;
} 