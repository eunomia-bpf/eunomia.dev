#include <stdio.h>
#include <cuda_runtime.h>

// Method 1: Inline PTX assembly within device functions
__device__ int ptx_add(int a, int b) {
    int result;
    asm("add.s32 %0, %1, %2;" : "=r"(result) : "r"(a), "r"(b));
    return result;
}

__device__ int ptx_multiply(int a, int b) {
    int result;
    asm("mul.lo.s32 %0, %1, %2;" : "=r"(result) : "r"(a), "r"(b));
    return result;
}

__device__ int ptx_subtract(int a, int b) {
    int result;
    asm("sub.s32 %0, %1, %2;" : "=r"(result) : "r"(a), "r"(b));
    return result;
}

// Complex PTX operation using multiple instructions
__device__ int ptx_complex_operation(int a, int b, int c) {
    int temp1, temp2, result;
    asm volatile (
        "add.s32 %0, %3, %4;\n\t"      // temp1 = a + b
        "mul.lo.s32 %1, %0, %5;\n\t"   // temp2 = temp1 * c
        "sub.s32 %2, %1, %3;"          // result = temp2 - a
        : "=r"(temp1), "=r"(temp2), "=r"(result)
        : "r"(a), "r"(b), "r"(c)
    );
    return result;
}

// Kernel that calls PTX functions from device
__global__ void kernelCallingPTX(int* a, int* b, int* c, int* results, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Each thread calls different PTX functions
        int operation = idx % 4;
        
        switch(operation) {
            case 0:
                results[idx] = ptx_add(a[idx], b[idx]);
                break;
            case 1:
                results[idx] = ptx_multiply(a[idx], b[idx]);
                break;
            case 2:
                results[idx] = ptx_subtract(a[idx], b[idx]);
                break;
            case 3:
                results[idx] = ptx_complex_operation(a[idx], b[idx], c[idx]);
                break;
        }
    }
}

// Demonstration of calling PTX from nested device functions
__device__ int calculate_with_ptx(int x, int y, int z) {
    // This device function calls multiple PTX functions
    int step1 = ptx_add(x, y);        // step1 = x + y
    int step2 = ptx_multiply(step1, z); // step2 = step1 * z
    int step3 = ptx_subtract(step2, x); // step3 = step2 - x
    return step3;
}

__global__ void nestedPTXKernel(int* input, int* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Call device function that uses PTX internally
        output[idx] = calculate_with_ptx(input[idx], input[idx] + 1, 2);
    }
}

int main() {
    const int n = 1000;
    const size_t size = n * sizeof(int);
    
    // Host memory
    int *h_a, *h_b, *h_c, *h_results, *h_input, *h_output;
    h_a = (int*)malloc(size);
    h_b = (int*)malloc(size);
    h_c = (int*)malloc(size);
    h_results = (int*)malloc(size);
    h_input = (int*)malloc(size);
    h_output = (int*)malloc(size);
    
    // Initialize data
    for (int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = i + 1;
        h_c[i] = 2;
        h_input[i] = i;
    }
    
    // Device memory
    int *d_a, *d_b, *d_c, *d_results, *d_input, *d_output;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    cudaMalloc(&d_results, size);
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    
    // Copy to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    
    printf("=== Simple PTX Calls from GPU Kernels ===\n\n");
    
    // Example 1: Direct PTX calls
    printf("1. Kernel with direct PTX function calls:\n");
    kernelCallingPTX<<<numBlocks, blockSize>>>(d_a, d_b, d_c, d_results, n);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_results, d_results, size, cudaMemcpyDeviceToHost);
    
    printf("First 8 results (operations: add, mul, sub, complex):\n");
    for (int i = 0; i < 8; i++) {
        int op = i % 4;
        const char* op_names[] = {"add", "mul", "sub", "complex"};
        printf("  %s(%d, %d) = %d\n", op_names[op], h_a[i], h_b[i], h_results[i]);
    }
    printf("\n");
    
    // Example 2: Nested PTX calls
    printf("2. Kernel with nested PTX function calls:\n");
    nestedPTXKernel<<<numBlocks, blockSize>>>(d_input, d_output, n);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
    
    printf("First 5 nested PTX results:\n");
    for (int i = 0; i < 5; i++) {
        printf("  calculate_with_ptx(%d) = %d\n", h_input[i], h_output[i]);
    }
    printf("\n");
    
    // Verify results manually for first few elements
    printf("3. Manual verification:\n");
    for (int i = 0; i < 3; i++) {
        int expected = ((h_input[i] + (h_input[i] + 1)) * 2) - h_input[i];
        printf("  Expected: %d, Got: %d %s\n", 
               expected, h_output[i], 
               (expected == h_output[i]) ? "✓" : "✗");
    }
    
    // Cleanup
    free(h_a); free(h_b); free(h_c); free(h_results);
    free(h_input); free(h_output);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); cudaFree(d_results);
    cudaFree(d_input); cudaFree(d_output);
    
    printf("\n=== Summary ===\n");
    printf("✓ PTX functions can be called from within GPU kernels\n");
    printf("✓ Use inline assembly with 'asm' keyword\n");
    printf("✓ PTX functions can be nested (device functions calling other device functions with PTX)\n");
    printf("✓ This is the most practical approach for custom GPU operations\n");
    
    return 0;
} 