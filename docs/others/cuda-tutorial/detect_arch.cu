/**
 * CUDA Architecture Detection
 * 
 * This small utility program detects the architecture of the primary CUDA GPU
 * and outputs it in the format required for NVCC's -arch flag (e.g., sm_75).
 * 
 * Used by the Makefile to automatically set the appropriate architecture.
 */

#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int device = 0;
    cudaError_t error = cudaSuccess;
    cudaDeviceProp deviceProp;
    
    error = cudaGetDeviceCount(&device);
    if (error != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(error));
        return 1;
    }
    
    if (device == 0) {
        fprintf(stderr, "No CUDA devices found\n");
        return 1;
    }
    
    // Get properties of the first device
    error = cudaGetDeviceProperties(&deviceProp, 0);
    if (error != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(error));
        return 1;
    }
    
    // Print in the format expected by nvcc -arch flag
    printf("sm_%d%d\n", deviceProp.major, deviceProp.minor);
    
    return 0;
} 