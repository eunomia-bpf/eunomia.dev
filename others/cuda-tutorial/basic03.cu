/**
 * CUDA Basic Example 03 - Comparison of GPU Programming Methods
 * 
 * This example demonstrates and compares different GPU programming approaches:
 * 1. Standard CUDA C/C++
 * 2. Inline PTX Assembly
 * 3. Thrust (high-level C++ abstraction)
 * 4. CUDA Unified Memory
 * 5. Shared Memory
 * 6. CUDA Streams
 * 7. Dynamic Parallelism
 * 
 * All implementations perform a simple matrix multiplication and are timed
 * to compare performance.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/generate.h>
#include <thrust/functional.h>
#include <thrust/tuple.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/copy.h>

// Include the placeholders namespace for _1, _2, etc.
using namespace thrust::placeholders;

// Dimensions for our matrices
#define N 1024  // Matrix dimensions (N x N)
#define BLOCK_SIZE 32  // Thread block size

// Utility function for timing
double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

// Initialize matrices with random values
void initialize_matrices(float *A, float *B, float *C, int n) {
    for (int i = 0; i < n * n; i++) {
        A[i] = static_cast<float>(rand()) / RAND_MAX;
        B[i] = static_cast<float>(rand()) / RAND_MAX;
        C[i] = 0.0f;
    }
}

// CPU implementation for verification
void matrix_multiply_cpu(float *A, float *B, float *C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

//=========================================================
// Method 1: Standard CUDA Implementation
//=========================================================
__global__ void matrix_multiply_cuda(float *A, float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

//=========================================================
// Method 2: CUDA with Inline PTX Implementation
//=========================================================
__device__ float multiply_add_ptx(float a, float b, float c) {
    float result;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(result) : "f"(a), "f"(b), "f"(c));
    return result;
}

__global__ void matrix_multiply_ptx(float *A, float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            // Use PTX fused multiply-add instruction for better performance
            sum = multiply_add_ptx(A[row * n + k], B[k * n + col], sum);
        }
        C[row * n + col] = sum;
    }
}

//=========================================================
// Method 3: Unified Memory Implementation
//=========================================================
__global__ void matrix_multiply_unified(float *A, float *B, float *C, int n) {
    // Same kernel as standard CUDA, but memory is managed differently
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

//=========================================================
// Method 4: Shared Memory Implementation
//=========================================================
__global__ void matrix_multiply_shared(float *A, float *B, float *C, int n) {
    __shared__ float shared_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shared_B[BLOCK_SIZE][BLOCK_SIZE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;
    
    float sum = 0.0f;
    
    // Loop over all sub-matrices of A and B
    for (int m = 0; m < n / BLOCK_SIZE; m++) {
        // Load the matrices from global memory to shared memory
        if (row < n && m * BLOCK_SIZE + tx < n)
            shared_A[ty][tx] = A[row * n + m * BLOCK_SIZE + tx];
        else
            shared_A[ty][tx] = 0.0f;
        
        if (m * BLOCK_SIZE + ty < n && col < n)
            shared_B[ty][tx] = B[(m * BLOCK_SIZE + ty) * n + col];
        else
            shared_B[ty][tx] = 0.0f;
        
        __syncthreads();
        
        // Compute the sub-matrix product
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += shared_A[ty][k] * shared_B[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

//=========================================================
// Thrust Implementation Helper Functions
//=========================================================
struct matrix_multiply_functor {
    int n;
    thrust::device_ptr<float> A;
    thrust::device_ptr<float> B;
    
    matrix_multiply_functor(int _n, thrust::device_ptr<float> _A, thrust::device_ptr<float> _B) 
        : n(_n), A(_A), B(_B) {}
    
    __host__ __device__ float operator()(const thrust::tuple<int, int>& index) const {
        int i = thrust::get<0>(index);
        int j = thrust::get<1>(index);
        
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[i * n + k] * B[k * n + j];
        }
        return sum;
    }
};

void run_thrust_implementation(float *h_A, float *h_B, float *h_C, int n) {
    thrust::device_vector<float> d_A(h_A, h_A + n * n);
    thrust::device_vector<float> d_B(h_B, h_B + n * n);
    thrust::device_vector<float> d_C(n * n);
    
    // Create a 2D index space
    thrust::counting_iterator<int> begin(0);
    thrust::counting_iterator<int> end = begin + (n * n);
    
    // Fill the result matrix using our functor
    thrust::transform(
        thrust::make_zip_iterator(thrust::make_tuple(
            thrust::make_transform_iterator(begin, _1 / n),  // row index
            thrust::make_transform_iterator(begin, _1 % n)   // column index
        )),
        thrust::make_zip_iterator(thrust::make_tuple(
            thrust::make_transform_iterator(end, _1 / n),
            thrust::make_transform_iterator(end, _1 % n)
        )),
        d_C.begin(),
        matrix_multiply_functor(n, thrust::device_pointer_cast(d_A.data()), 
                                thrust::device_pointer_cast(d_B.data()))
    );
    
    // Copy result back to host
    thrust::copy(d_C.begin(), d_C.end(), h_C);
}

//=========================================================
// Method 6: CUDA Streams Implementation
//=========================================================
void run_cuda_streams_implementation(float *h_A, float *h_B, float *h_C, int n) {
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, n * n * sizeof(float));
    cudaMalloc(&d_B, n * n * sizeof(float));
    cudaMalloc(&d_C, n * n * sizeof(float));
    
    // Create CUDA streams
    const int numStreams = 4;  // Use multiple streams
    cudaStream_t streams[numStreams];
    for (int i = 0; i < numStreams; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    // Divide the matrices into chunks for each stream
    int streamSize = n / numStreams;
    size_t streamBytes = streamSize * n * sizeof(float);
    
    // Launch kernels in different streams
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    
    for (int i = 0; i < numStreams; i++) {
        int offset = i * streamSize;
        dim3 grid((n + threads.x - 1) / threads.x, (streamSize + threads.y - 1) / threads.y);
        
        // Copy input data for this stream
        cudaMemcpyAsync(d_A + offset * n, h_A + offset * n, streamBytes, 
                       cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(d_B, h_B, n * n * sizeof(float), 
                       cudaMemcpyHostToDevice, streams[i]);
        
        // Launch kernel in this stream
        matrix_multiply_cuda<<<grid, threads, 0, streams[i]>>>(
            d_A + offset * n, d_B, d_C + offset * n, n);
        
        // Copy results back
        cudaMemcpyAsync(h_C + offset * n, d_C + offset * n, streamBytes, 
                       cudaMemcpyDeviceToHost, streams[i]);
    }
    
    // Synchronize all streams
    for (int i = 0; i < numStreams; i++) {
        cudaStreamSynchronize(streams[i]);
    }
    
    // Destroy streams
    for (int i = 0; i < numStreams; i++) {
        cudaStreamDestroy(streams[i]);
    }
    
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

//=========================================================
// Method 7: Dynamic Parallelism Implementation
//=========================================================
__global__ void multiply_submatrix(float *A, float *B, float *C, 
                                 int n, int row_start, int col_start, int subsize) {
    int row = blockIdx.y * blockDim.y + threadIdx.y + row_start;
    int col = blockIdx.x * blockDim.x + threadIdx.x + col_start;
    
    if (row < row_start + subsize && col < col_start + subsize && row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

__global__ void matrix_multiply_dynamic_parent(float *A, float *B, float *C, int n) {
    // Each thread in the parent grid launches a child grid
    int subsize = 256; // Size of sub-matrix
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Only process if we're within the grid dimensions for submatrices
    if (row * subsize < n && col * subsize < n) {
        int row_start = row * subsize;
        int col_start = col * subsize;
        
        // Create the child grid dimensions
        dim3 dimBlock(16, 16);
        dim3 dimGrid((subsize + dimBlock.x - 1) / dimBlock.x, 
                     (subsize + dimBlock.y - 1) / dimBlock.y);
        
        // Launch a child kernel to process this submatrix
        multiply_submatrix<<<dimGrid, dimBlock>>>(A, B, C, n, row_start, col_start, subsize);
    }
}

void run_dynamic_parallelism_implementation(float *h_A, float *h_B, float *h_C, int n) {
    float *d_A, *d_B, *d_C;
    size_t matrix_size = n * n * sizeof(float);
    
    cudaMalloc(&d_A, matrix_size);
    cudaMalloc(&d_B, matrix_size);
    cudaMalloc(&d_C, matrix_size);
    
    cudaMemcpy(d_A, h_A, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, matrix_size, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, matrix_size);
    
    // Launch the parent grid
    dim3 dimBlock(8, 8);
    dim3 dimGrid((n + 255) / 256, (n + 255) / 256); // Each thread handles a 256x256 submatrix
    
    matrix_multiply_dynamic_parent<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, n);
    
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_C, d_C, matrix_size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

//=========================================================
// Main function to compare all methods
//=========================================================
int main() {
    srand(42); // Fixed seed for reproducibility
    
    float *h_A, *h_B, *h_C, *h_R;
    float *d_A, *d_B, *d_C;
    float *u_A, *u_B, *u_C;
    
    size_t matrix_size = N * N * sizeof(float);
    
    printf("Comparing different GPU programming methods for %dx%d matrix multiplication\n\n", N, N);
    
    // Allocate host memory
    h_A = (float*)malloc(matrix_size);
    h_B = (float*)malloc(matrix_size);
    h_C = (float*)malloc(matrix_size);
    h_R = (float*)malloc(matrix_size); // Reference result
    
    // Initialize matrices
    initialize_matrices(h_A, h_B, h_C, N);
    
    // Compute reference solution on CPU
    printf("Computing reference solution on CPU...\n");
    double cpu_start = get_time_ms();
    matrix_multiply_cpu(h_A, h_B, h_R, N);
    double cpu_end = get_time_ms();
    printf("CPU Time: %.2f ms\n\n", cpu_end - cpu_start);
    
    //=========================================================
    // Method 1: Standard CUDA Implementation
    //=========================================================
    cudaMalloc(&d_A, matrix_size);
    cudaMalloc(&d_B, matrix_size);
    cudaMalloc(&d_C, matrix_size);
    
    cudaMemcpy(d_A, h_A, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, matrix_size, cudaMemcpyHostToDevice);
    
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);
    
    printf("Method 1: Standard CUDA Implementation\n");
    cudaMemset(d_C, 0, matrix_size);
    
    double cuda_start = get_time_ms();
    matrix_multiply_cuda<<<grid, threads>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    double cuda_end = get_time_ms();
    
    cudaMemcpy(h_C, d_C, matrix_size, cudaMemcpyDeviceToHost);
    printf("CUDA Time: %.2f ms\n", cuda_end - cuda_start);
    
    // Verify result
    bool correct = true;
    for (int i = 0; i < N * N; i++) {
        if (fabs(h_C[i] - h_R[i]) > 1e-5) {
            correct = false;
            break;
        }
    }
    printf("CUDA Result: %s\n\n", correct ? "CORRECT" : "INCORRECT");
    
    //=========================================================
    // Method 2: CUDA with Inline PTX Implementation
    //=========================================================
    printf("Method 2: CUDA with Inline PTX Implementation\n");
    cudaMemset(d_C, 0, matrix_size);
    
    double ptx_start = get_time_ms();
    matrix_multiply_ptx<<<grid, threads>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    double ptx_end = get_time_ms();
    
    cudaMemcpy(h_C, d_C, matrix_size, cudaMemcpyDeviceToHost);
    printf("PTX Time: %.2f ms\n", ptx_end - ptx_start);
    
    // Verify result
    correct = true;
    for (int i = 0; i < N * N; i++) {
        if (fabs(h_C[i] - h_R[i]) > 1e-5) {
            correct = false;
            break;
        }
    }
    printf("PTX Result: %s\n\n", correct ? "CORRECT" : "INCORRECT");
    
    //=========================================================
    // Method 3: CUDA Unified Memory Implementation
    //=========================================================
    printf("Method 3: CUDA Unified Memory Implementation\n");
    
    cudaMallocManaged(&u_A, matrix_size);
    cudaMallocManaged(&u_B, matrix_size);
    cudaMallocManaged(&u_C, matrix_size);
    
    // Initialize unified memory
    for (int i = 0; i < N * N; i++) {
        u_A[i] = h_A[i];
        u_B[i] = h_B[i];
        u_C[i] = 0.0f;
    }
    
    double unified_start = get_time_ms();
    matrix_multiply_unified<<<grid, threads>>>(u_A, u_B, u_C, N);
    cudaDeviceSynchronize();
    double unified_end = get_time_ms();
    
    printf("Unified Memory Time: %.2f ms\n", unified_end - unified_start);
    
    // Verify result
    correct = true;
    for (int i = 0; i < N * N; i++) {
        if (fabs(u_C[i] - h_R[i]) > 1e-5) {
            correct = false;
            break;
        }
    }
    printf("Unified Memory Result: %s\n\n", correct ? "CORRECT" : "INCORRECT");
    
    //=========================================================
    // Method 4: Shared Memory Implementation
    //=========================================================
    printf("Method 4: Shared Memory Implementation\n");
    cudaMemset(d_C, 0, matrix_size);
    
    double shared_start = get_time_ms();
    matrix_multiply_shared<<<grid, threads>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    double shared_end = get_time_ms();
    
    cudaMemcpy(h_C, d_C, matrix_size, cudaMemcpyDeviceToHost);
    printf("Shared Memory Time: %.2f ms\n", shared_end - shared_start);
    
    // Verify result
    correct = true;
    for (int i = 0; i < N * N; i++) {
        if (fabs(h_C[i] - h_R[i]) > 1e-5) {
            correct = false;
            break;
        }
    }
    printf("Shared Memory Result: %s\n\n", correct ? "CORRECT" : "INCORRECT");
    
    //=========================================================
    // Method 5: Thrust Implementation
    //=========================================================
    printf("Method 5: Thrust High-Level Implementation\n");
    
    double thrust_start = get_time_ms();
    run_thrust_implementation(h_A, h_B, h_C, N);
    double thrust_end = get_time_ms();
    
    printf("Thrust Time: %.2f ms\n", thrust_end - thrust_start);
    
    // Verify result
    correct = true;
    for (int i = 0; i < N * N; i++) {
        if (fabs(h_C[i] - h_R[i]) > 1e-5) {
            correct = false;
            break;
        }
    }
    printf("Thrust Result: %s\n\n", correct ? "CORRECT" : "INCORRECT");
    
    //=========================================================
    // Method 6: CUDA Streams Implementation
    //=========================================================
    printf("Method 6: CUDA Streams Implementation\n");
    
    double streams_start = get_time_ms();
    run_cuda_streams_implementation(h_A, h_B, h_C, N);
    double streams_end = get_time_ms();
    
    printf("CUDA Streams Time: %.2f ms\n", streams_end - streams_start);
    
    // Verify result
    correct = true;
    for (int i = 0; i < N * N; i++) {
        if (fabs(h_C[i] - h_R[i]) > 1e-5) {
            correct = false;
            break;
        }
    }
    printf("CUDA Streams Result: %s\n\n", correct ? "CORRECT" : "INCORRECT");
    
    //=========================================================
    // Method 7: Dynamic Parallelism Implementation
    //=========================================================
    printf("Method 7: Dynamic Parallelism Implementation\n");
    
    double dp_start = get_time_ms();
    run_dynamic_parallelism_implementation(h_A, h_B, h_C, N);
    double dp_end = get_time_ms();
    
    printf("Dynamic Parallelism Time: %.2f ms\n", dp_end - dp_start);
    
    // Verify result
    correct = true;
    for (int i = 0; i < N * N; i++) {
        if (fabs(h_C[i] - h_R[i]) > 1e-5) {
            correct = false;
            break;
        }
    }
    printf("Dynamic Parallelism Result: %s\n\n", correct ? "CORRECT" : "INCORRECT");
    
    //=========================================================
    // Print performance comparison summary
    //=========================================================
    printf("==================================================\n");
    printf("Performance Comparison Summary:\n");
    printf("==================================================\n");
    printf("CPU Time:                 %.2f ms\n", cpu_end - cpu_start);
    printf("CUDA Time:                %.2f ms (%.2fx speedup)\n", 
           cuda_end - cuda_start, (cpu_end - cpu_start) / (cuda_end - cuda_start));
    printf("PTX Time:                 %.2f ms (%.2fx speedup)\n", 
           ptx_end - ptx_start, (cpu_end - cpu_start) / (ptx_end - ptx_start));
    printf("Unified Memory Time:      %.2f ms (%.2fx speedup)\n", 
           unified_end - unified_start, (cpu_end - cpu_start) / (unified_end - unified_start));
    printf("Shared Memory Time:       %.2f ms (%.2fx speedup)\n", 
           shared_end - shared_start, (cpu_end - cpu_start) / (shared_end - shared_start));
    printf("Thrust Time:              %.2f ms (%.2fx speedup)\n", 
           thrust_end - thrust_start, (cpu_end - cpu_start) / (thrust_end - thrust_start));
    printf("CUDA Streams Time:        %.2f ms (%.2fx speedup)\n", 
           streams_end - streams_start, (cpu_end - cpu_start) / (streams_end - streams_start));
    printf("Dynamic Parallelism Time: %.2f ms (%.2fx speedup)\n", 
           dp_end - dp_start, (cpu_end - cpu_start) / (dp_end - dp_start));
    printf("==================================================\n");
    
    // Clean up
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_R);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    cudaFree(u_A);
    cudaFree(u_B);
    cudaFree(u_C);
    
    return 0;
} 