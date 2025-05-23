/**
 * CUDA Basic Example 05 - Neural Network Forward Pass
 * 
 * This example demonstrates how to implement a basic neural network forward pass on GPU using CUDA.
 * The network consists of:
 * 1. Input layer
 * 2. Fully connected hidden layer with ReLU activation
 * 3. Fully connected output layer with softmax activation
 * 
 * The implementation shows:
 * - Matrix multiplication for fully connected layers
 * - Activation functions (ReLU, softmax)
 * - Memory management for network parameters and activations
 * - Efficient batch processing
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

// Neural network configuration
#define INPUT_SIZE 784     // 28x28 input (e.g., MNIST)
#define HIDDEN_SIZE 128    // Hidden layer neurons
#define OUTPUT_SIZE 10     // 10 output classes
#define BATCH_SIZE 64      // Number of samples in a batch

// CUDA error checking macro
#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s, in file '%s', line %d\n", \
                cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

// Initialize network parameters with Xavier/Glorot initialization
void initializeParameters(float *weights1, float *bias1, float *weights2, float *bias2) {
    float weight1_scale = sqrtf(6.0f / (INPUT_SIZE + HIDDEN_SIZE));
    float weight2_scale = sqrtf(6.0f / (HIDDEN_SIZE + OUTPUT_SIZE));
    
    // Initialize weights with Xavier/Glorot initialization
    for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; i++) {
        weights1[i] = (2.0f * (float)rand() / RAND_MAX - 1.0f) * weight1_scale;
    }
    
    for (int i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; i++) {
        weights2[i] = (2.0f * (float)rand() / RAND_MAX - 1.0f) * weight2_scale;
    }
    
    // Initialize biases to zero
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        bias1[i] = 0.0f;
    }
    
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        bias2[i] = 0.0f;
    }
}

// Generate random input data (simulating MNIST digits)
void generateRandomInput(float *input, int *labels) {
    for (int b = 0; b < BATCH_SIZE; b++) {
        // Generate random class (0-9)
        labels[b] = rand() % OUTPUT_SIZE;
        
        // Generate pixel values centered around the digit pattern
        // This is just a simple simulation, not actual MNIST data
        for (int i = 0; i < INPUT_SIZE; i++) {
            if (i % 28 > 7 && i % 28 < 20 && i / 28 > 7 && i / 28 < 20) {
                // Central region - higher probability of active pixels for the digit
                input[b * INPUT_SIZE + i] = (labels[b] * 0.1f) + 0.5f * (float)rand() / RAND_MAX;
            } else {
                // Background - mostly dark
                input[b * INPUT_SIZE + i] = 0.1f * (float)rand() / RAND_MAX;
            }
        }
    }
}

// CUDA kernel for matrix multiplication (C = A * B)
__global__ void matrixMultiplyKernel(float *A, float *B, float *C, 
                                     int A_rows, int A_cols, int B_cols) {
    // Each thread computes one element of C
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < A_rows && col < B_cols) {
        float sum = 0.0f;
        for (int k = 0; k < A_cols; k++) {
            sum += A[row * A_cols + k] * B[k * B_cols + col];
        }
        C[row * B_cols + col] = sum;
    }
}

// CUDA kernel for adding bias to each row
__global__ void addBiasKernel(float *output, float *bias, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        output[row * cols + col] += bias[col];
    }
}

// CUDA kernel for ReLU activation function
__global__ void reluKernel(float *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

// CUDA kernel for softmax activation function
__global__ void softmaxKernel(float *input, float *output, int batch_size, int num_classes) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size) {
        // Find max value for numerical stability
        float max_val = -FLT_MAX;
        for (int i = 0; i < num_classes; i++) {
            max_val = fmaxf(max_val, input[batch_idx * num_classes + i]);
        }
        
        // Compute exponentials and sum
        float sum = 0.0f;
        for (int i = 0; i < num_classes; i++) {
            output[batch_idx * num_classes + i] = expf(input[batch_idx * num_classes + i] - max_val);
            sum += output[batch_idx * num_classes + i];
        }
        
        // Normalize
        for (int i = 0; i < num_classes; i++) {
            output[batch_idx * num_classes + i] /= sum;
        }
    }
}

// Get class prediction from softmax output
__global__ void getPredictionsKernel(float *softmax_output, int *predictions, int batch_size, int num_classes) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size) {
        float max_prob = -1.0f;
        int max_idx = -1;
        
        for (int i = 0; i < num_classes; i++) {
            float prob = softmax_output[batch_idx * num_classes + i];
            if (prob > max_prob) {
                max_prob = prob;
                max_idx = i;
            }
        }
        
        predictions[batch_idx] = max_idx;
    }
}

// Calculate classification accuracy
float calculateAccuracy(int *predictions, int *labels, int batch_size) {
    int correct = 0;
    for (int i = 0; i < batch_size; i++) {
        if (predictions[i] == labels[i]) {
            correct++;
        }
    }
    return (float)correct / batch_size;
}

// Forward pass of the neural network
void forwardPass(
    // Input
    float *d_input,
    // Network parameters
    float *d_weights1, float *d_bias1,
    float *d_weights2, float *d_bias2,
    // Intermediate activations
    float *d_hidden_preact, float *d_hidden_output,
    float *d_output_preact, float *d_output,
    // Output predictions
    int *d_predictions
) {
    // Define block and grid dimensions
    dim3 block_mm(16, 16);  // 16x16 threads per block for matrix multiplication
    dim3 grid_mm1((OUTPUT_SIZE + block_mm.x - 1) / block_mm.x,
                 (BATCH_SIZE + block_mm.y - 1) / block_mm.y);
    dim3 grid_mm2((HIDDEN_SIZE + block_mm.x - 1) / block_mm.x,
                 (BATCH_SIZE + block_mm.y - 1) / block_mm.y);
    
    dim3 block_bias(16, 16);
    dim3 grid_bias1((HIDDEN_SIZE + block_bias.x - 1) / block_bias.x,
                   (BATCH_SIZE + block_bias.y - 1) / block_bias.y);
    dim3 grid_bias2((OUTPUT_SIZE + block_bias.x - 1) / block_bias.x,
                   (BATCH_SIZE + block_bias.y - 1) / block_bias.y);
    
    int block_act = 256;
    int grid_act1 = (BATCH_SIZE * HIDDEN_SIZE + block_act - 1) / block_act;
    int grid_act2 = (BATCH_SIZE * OUTPUT_SIZE + block_act - 1) / block_act;
    
    int block_pred = 256;
    int grid_pred = (BATCH_SIZE + block_pred - 1) / block_pred;
    
    // Forward pass: input -> hidden layer
    matrixMultiplyKernel<<<grid_mm2, block_mm>>>(d_input, d_weights1, d_hidden_preact, 
                                                BATCH_SIZE, INPUT_SIZE, HIDDEN_SIZE);
    addBiasKernel<<<grid_bias1, block_bias>>>(d_hidden_preact, d_bias1, BATCH_SIZE, HIDDEN_SIZE);
    reluKernel<<<grid_act1, block_act>>>(d_hidden_preact, BATCH_SIZE * HIDDEN_SIZE);
    
    // Copy hidden layer activation to output for next layer
    cudaMemcpy(d_hidden_output, d_hidden_preact, BATCH_SIZE * HIDDEN_SIZE * sizeof(float),
              cudaMemcpyDeviceToDevice);
    
    // Forward pass: hidden -> output layer
    matrixMultiplyKernel<<<grid_mm1, block_mm>>>(d_hidden_output, d_weights2, d_output_preact,
                                                BATCH_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    addBiasKernel<<<grid_bias2, block_bias>>>(d_output_preact, d_bias2, BATCH_SIZE, OUTPUT_SIZE);
    
    // Apply softmax activation
    softmaxKernel<<<grid_pred, block_pred>>>(d_output_preact, d_output, BATCH_SIZE, OUTPUT_SIZE);
    
    // Get predictions
    getPredictionsKernel<<<grid_pred, block_pred>>>(d_output, d_predictions, BATCH_SIZE, OUTPUT_SIZE);
}

int main() {
    // Set random seed for reproducibility
    srand(42);
    
    // Allocate host memory for network parameters
    float *h_weights1 = (float*)malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    float *h_bias1 = (float*)malloc(HIDDEN_SIZE * sizeof(float));
    float *h_weights2 = (float*)malloc(HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
    float *h_bias2 = (float*)malloc(OUTPUT_SIZE * sizeof(float));
    
    // Initialize network parameters
    initializeParameters(h_weights1, h_bias1, h_weights2, h_bias2);
    
    // Allocate device memory for network parameters
    float *d_weights1, *d_bias1, *d_weights2, *d_bias2;
    CHECK_CUDA_ERROR(cudaMalloc(&d_weights1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_bias1, HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_weights2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_bias2, OUTPUT_SIZE * sizeof(float)));
    
    // Copy network parameters to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_weights1, h_weights1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float),
                               cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_bias1, h_bias1, HIDDEN_SIZE * sizeof(float),
                               cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_weights2, h_weights2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float),
                               cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_bias2, h_bias2, OUTPUT_SIZE * sizeof(float),
                               cudaMemcpyHostToDevice));
    
    // Generate input data
    float *h_input = (float*)malloc(BATCH_SIZE * INPUT_SIZE * sizeof(float));
    int *h_labels = (int*)malloc(BATCH_SIZE * sizeof(int));
    generateRandomInput(h_input, h_labels);
    
    // Allocate device memory for input
    float *d_input;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, BATCH_SIZE * INPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, BATCH_SIZE * INPUT_SIZE * sizeof(float),
                               cudaMemcpyHostToDevice));
    
    // Allocate device memory for intermediate results
    float *d_hidden_preact, *d_hidden_output, *d_output_preact, *d_output;
    CHECK_CUDA_ERROR(cudaMalloc(&d_hidden_preact, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_hidden_output, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output_preact, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    
    // Allocate memory for predictions
    int *h_predictions = (int*)malloc(BATCH_SIZE * sizeof(int));
    int *d_predictions;
    CHECK_CUDA_ERROR(cudaMalloc(&d_predictions, BATCH_SIZE * sizeof(int)));
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    printf("=== Neural Network Forward Pass Example ===\n\n");
    printf("Network configuration:\n");
    printf("  Input size: %d\n", INPUT_SIZE);
    printf("  Hidden layer size: %d\n", HIDDEN_SIZE);
    printf("  Output size: %d\n", OUTPUT_SIZE);
    printf("  Batch size: %d\n\n", BATCH_SIZE);
    
    // Time the forward pass
    cudaEventRecord(start);
    
    // Perform forward pass
    forwardPass(d_input, d_weights1, d_bias1, d_weights2, d_bias2,
               d_hidden_preact, d_hidden_output, d_output_preact, d_output,
               d_predictions);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Copy predictions back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_predictions, d_predictions, BATCH_SIZE * sizeof(int),
                               cudaMemcpyDeviceToHost));
    
    // Get some output probabilities for display
    float *h_output = (float*)malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float),
                               cudaMemcpyDeviceToHost));
    
    // Calculate accuracy
    float accuracy = calculateAccuracy(h_predictions, h_labels, BATCH_SIZE);
    
    printf("Forward pass completed in %.3f ms\n\n", milliseconds);
    printf("Example results (first 5 samples):\n");
    
    for (int i = 0; i < 5; i++) {
        printf("Sample %d - True label: %d, Predicted: %d\n", i, h_labels[i], h_predictions[i]);
        printf("  Probabilities: ");
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            printf("%.4f ", h_output[i * OUTPUT_SIZE + j]);
        }
        printf("\n");
    }
    
    printf("\nBatch accuracy: %.2f%%\n", accuracy * 100.0f);
    
    // Free memory
    cudaFree(d_weights1);
    cudaFree(d_bias1);
    cudaFree(d_weights2);
    cudaFree(d_bias2);
    cudaFree(d_input);
    cudaFree(d_hidden_preact);
    cudaFree(d_hidden_output);
    cudaFree(d_output_preact);
    cudaFree(d_output);
    cudaFree(d_predictions);
    
    free(h_weights1);
    free(h_bias1);
    free(h_weights2);
    free(h_bias2);
    free(h_input);
    free(h_labels);
    free(h_predictions);
    free(h_output);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
} 