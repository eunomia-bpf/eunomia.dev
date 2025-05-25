# CUDA Basic Example - Vector Addition Explanation

This document provides a detailed explanation of the vector addition CUDA example in `basic01.cu`.

You can find the code in <https://github.com/eunomia-bpf/basic-cuda-tutorial>

## Prerequisites

To run this example, you need:
- NVIDIA GPU with CUDA support
- NVIDIA CUDA Toolkit installed
- A C++ compiler compatible with your CUDA version
- GNU Make (for building with the provided Makefile)

## Building and Running

1. Build the example:
```bash
make
```

2. Run the program:
```bash
./basic01
```

## Code Structure and Explanation

### 1. Header Files and Includes
```cuda
#include <stdio.h>
#include <stdlib.h>
```
These standard C headers provide:
- `stdio.h`: Input/output functions like `printf`
- `stdlib.h`: Memory management functions like `malloc` and `free`

### 2. CUDA Kernel Function
```cuda
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements)
```

- `__global__`: Specifies this is a CUDA kernel function that:
  - Runs on the GPU
  - Can be called from CPU code
  - Must return void
- Parameters:
  - `float *A, *B`: Input vectors in GPU memory
  - `float *C`: Output vector in GPU memory
  - `numElements`: Size of the vectors

Inside the kernel:
```cuda
int i = blockDim.x * blockIdx.x + threadIdx.x;
```
This calculates a unique index for each thread where:
- `threadIdx.x`: Thread index within the block (0 to blockDim.x-1)
- `blockIdx.x`: Block index within the grid
- `blockDim.x`: Number of threads per block

### 3. Main Function Components

#### 3.1 Memory Allocation
```cuda
// Host memory allocation
float *h_A = (float *)malloc(size);  // CPU memory

// Device memory allocation
float *d_A = NULL;
cudaMalloc((void **)&d_A, size);     // GPU memory
```

- Host (CPU) memory uses standard C `malloc`
- Device (GPU) memory uses CUDA's `cudaMalloc`
- The 'h_' prefix denotes host memory
- The 'd_' prefix denotes device memory

#### 3.2 Data Transfer
```cuda
cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
```

`cudaMemcpy` parameters:
1. Destination pointer
2. Source pointer
3. Size in bytes
4. Direction of transfer:
   - `cudaMemcpyHostToDevice`: CPU to GPU
   - `cudaMemcpyDeviceToHost`: GPU to CPU

#### 3.3 Kernel Launch Configuration
```cuda
int threadsPerBlock = 256;
int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
```

- `threadsPerBlock = 256`: Common size for good performance
- `blocksPerGrid`: Calculated to ensure enough threads for all elements
- The formula `(numElements + threadsPerBlock - 1) / threadsPerBlock` rounds up the division
- Launch syntax `<<<blocks, threads>>>` specifies the execution configuration

#### 3.4 Error Checking
```cuda
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    fprintf(stderr, "Failed to launch kernel: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
}
```

Always check for CUDA errors after kernel launches and CUDA API calls.

#### 3.5 Result Verification
```cuda
for (int i = 0; i < numElements; ++i) {
    if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
        fprintf(stderr, "Result verification failed at element %d!\n", i);
        exit(EXIT_FAILURE);
    }
}
```

Verifies the GPU computation by comparing with CPU results.

#### 3.6 Cleanup
```cuda
// Free GPU memory
cudaFree(d_A);
cudaFree(d_B);
cudaFree(d_C);

// Free CPU memory
free(h_A);
free(h_B);
free(h_C);
```

Always free allocated memory to prevent memory leaks.

## Performance Considerations

1. **Thread Block Size**
   - We use 256 threads per block
   - This is a common choice that works well on most GPUs
   - Powers of 2 are typically efficient

2. **Memory Coalescing**
   - Adjacent threads access adjacent memory locations
   - This pattern enables efficient memory access

3. **Error Checking**
   - The code includes robust error checking
   - Important for debugging and reliability

## Common Issues and Debugging

1. **CUDA Installation**
   - Ensure CUDA toolkit is properly installed
   - Check `nvcc --version` works
   - Verify GPU compatibility with `nvidia-smi`

2. **Compilation Errors**
   - Check CUDA path is in system PATH
   - Verify GPU compute capability matches `-arch` flag in Makefile

3. **Runtime Errors**
   - Out of memory: Reduce vector size
   - Kernel launch failure: Check GPU availability
   - Incorrect results: Verify index calculations

## Expected Output

When running successfully, you should see:
```
Vector addition of 50000 elements
CUDA kernel launch with 196 blocks of 256 threads
Test PASSED
Done
```

## Modifying the Example

To experiment with the code:

1. Change vector size (`numElements`)
2. Modify threads per block
3. Add timing measurements
4. Try different data types
5. Implement other vector operations

Remember to handle errors and verify results after modifications.
