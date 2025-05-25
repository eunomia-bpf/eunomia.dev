# CUDA GPU Organization Hierarchy

This document provides a comprehensive overview of the NVIDIA GPU architecture and programming model hierarchy, from both hardware and software perspectives.

You can find the code in <https://github.com/eunomia-bpf/basic-cuda-tutorial>

## Table of Contents

1. [Hardware Organization](#hardware-organization)
2. [Software Programming Model](#software-programming-model)
3. [Memory Hierarchy](#memory-hierarchy)
4. [Execution Model](#execution-model)
5. [Performance Considerations](#performance-considerations)
6. [Example Application](#example-application)

## Hardware Organization

### GPU Architecture Evolution

NVIDIA GPUs have evolved through multiple architecture generations:

| Architecture | Example GPUs | Key Features |
|--------------|--------------|-------------|
| Tesla | GeForce 8/9/200 series | First CUDA-capable GPUs |
| Fermi | GeForce 400/500 series | L1/L2 cache, improved double precision |
| Kepler | GeForce 600/700 series | Dynamic parallelism, Hyper-Q |
| Maxwell | GeForce 900 series | Improved power efficiency |
| Pascal | GeForce 10 series, Tesla P100 | Unified memory improvements, NVLink |
| Volta | Tesla V100 | Tensor Cores, independent thread scheduling |
| Turing | GeForce RTX 20 series | RT Cores, improved Tensor Cores |
| Ampere | GeForce RTX 30 series, A100 | 3rd gen Tensor Cores, sparsity acceleration |
| Hopper | H100 | 4th gen Tensor Cores, Transformer Engine |
| Ada Lovelace | GeForce RTX 40 series | RT improvements, DLSS 3 |

### Hardware Components

A modern NVIDIA GPU consists of:

1. **Streaming Multiprocessors (SMs)**: The basic computational units
2. **Tensor Cores**: Specialized for matrix operations (newer GPUs)
3. **RT Cores**: Specialized for ray tracing (RTX GPUs)
4. **Memory Controllers**: Interface with device memory
5. **L2 Cache**: Shared among all SMs
6. **Scheduler**: Manages execution of thread blocks

### Streaming Multiprocessor (SM) Architecture

Each SM contains:

- **CUDA Cores**: Integer and floating-point arithmetic units
- **Tensor Cores**: Matrix multiply-accumulate units
- **Warp Schedulers**: Manage thread execution
- **Register File**: Ultra-fast storage for thread variables
- **Shared Memory/L1 Cache**: Fast memory shared by threads in a block
- **Load/Store Units**: Handle memory operations
- **Special Function Units (SFUs)**: Calculate transcendentals (sin, cos, etc.)
- **Texture Units**: Specialized for texture operations

![SM Architecture](https://developer.nvidia.com/blog/wp-content/uploads/2018/04/volta-architecture-768x756.png)
*Example SM architecture (diagram not included, reference only)*

## Software Programming Model

CUDA programs are organized in a hierarchical structure:

### Thread Hierarchy

1. **Thread**: The smallest execution unit, runs a program instance
2. **Warp**: Group of 32 threads that execute in lockstep (SIMT)
3. **Block**: Group of threads that can cooperate via shared memory
4. **Grid**: Collection of blocks that execute the same kernel

```
Grid
├── Block (0,0)  Block (1,0)  Block (2,0)
├── Block (0,1)  Block (1,1)  Block (2,1)
└── Block (0,2)  Block (1,2)  Block (2,2)

Block (1,1)
├── Thread (0,0)  Thread (1,0)  Thread (2,0)
├── Thread (0,1)  Thread (1,1)  Thread (2,1)
└── Thread (0,2)  Thread (1,2)  Thread (2,2)
```

### Thread Indexing

Threads can be organized in 1D, 2D, or 3D arrangements. Each thread can be uniquely identified by:

```cuda
// 1D grid of 1D blocks
int tid = blockIdx.x * blockDim.x + threadIdx.x;

// 2D grid of 2D blocks
int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
int tid = tid_y * gridDim.x * blockDim.x + tid_x;

// 3D grid of 3D blocks
int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
int tid_z = blockIdx.z * blockDim.z + threadIdx.z;
```

### Kernel Execution Configuration

A kernel is launched with a specific grid and block configuration:

```cuda
dim3 block(16, 16, 1);   // 16×16 threads per block
dim3 grid(N/16, N/16, 1); // Grid dimensions adjusted to data size
myKernel<<<grid, block>>>(params...);
```

### Synchronization

- **Block-level**: `__syncthreads()` synchronizes all threads in a block
- **System-level**: `cudaDeviceSynchronize()` waits for all kernels to complete
- **Stream-level**: `cudaStreamSynchronize(stream)` waits for operations in a stream
- **Cooperative Groups**: More flexible synchronization patterns (newer CUDA versions)

## Memory Hierarchy

GPUs have a complex memory hierarchy with different performance characteristics:

### Device Memory Types

1. **Global Memory**
   - Largest capacity (several GB)
   - Accessible by all threads
   - High latency (hundreds of cycles)
   - Used for main data storage
   - Bandwidth: ~500-2000 GB/s depending on GPU

2. **Shared Memory**
   - Small capacity (up to 164KB per SM in newer GPUs)
   - Accessible by threads within a block
   - Low latency (similar to L1 cache)
   - Used for inter-thread communication and data reuse
   - Organized in banks for parallel access

3. **Constant Memory**
   - Small (64KB per device)
   - Read-only for kernels
   - Cached and optimized for broadcast
   - Used for unchanging parameters

4. **Texture Memory**
   - Cached read-only memory
   - Optimized for 2D/3D spatial locality
   - Hardware interpolation
   - Used for image processing

5. **Local Memory**
   - Per-thread private storage
   - Used for register spills
   - Actually resides in global memory
   - Automatic variable arrays often stored here

6. **Registers**
   - Fastest memory type
   - Per-thread private storage
   - Limited number per thread
   - Used for thread-local variables

### Memory Management Models

1. **Explicit Memory Management**
   ```cuda
   // Allocate device memory
   float *d_data;
   cudaMalloc(&d_data, size);
   
   // Transfer data to device
   cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
   
   // Launch kernel
   kernel<<<grid, block>>>(d_data);
   
   // Transfer results back
   cudaMemcpy(h_result, d_data, size, cudaMemcpyDeviceToHost);
   
   // Free device memory
   cudaFree(d_data);
   ```

2. **Unified Memory**
   ```cuda
   // Allocate unified memory
   float *data;
   cudaMallocManaged(&data, size);
   
   // Initialize data (on host)
   for (int i = 0; i < N; i++) data[i] = i;
   
   // Launch kernel (data automatically migrated)
   kernel<<<grid, block>>>(data);
   
   // Wait for kernel to finish
   cudaDeviceSynchronize();
   
   // Access results (data automatically migrated back)
   float result = data[0];
   
   // Free unified memory
   cudaFree(data);
   ```

3. **Zero-Copy Memory**
   ```cuda
   float *data;
   cudaHostAlloc(&data, size, cudaHostAllocMapped);
   
   float *d_data;
   cudaHostGetDevicePointer(&d_data, data, 0);
   
   kernel<<<grid, block>>>(d_data);
   ```

### Memory Access Patterns

1. **Coalesced Access**: Threads in a warp access contiguous memory
   ```cuda
   // Coalesced (efficient)
   data[threadIdx.x] = value;
   ```

2. **Strided Access**: Threads in a warp access memory with stride
   ```cuda
   // Strided (inefficient)
   data[threadIdx.x * stride] = value;
   ```

3. **Bank Conflicts**: Multiple threads access the same shared memory bank
   ```cuda
   // Potential bank conflict if threadIdx.x % 32 is the same for multiple threads
   shared[threadIdx.x] = data[threadIdx.x];
   ```

## Execution Model

### SIMT Execution

GPU executes threads in groups of 32 (warps) using Single Instruction, Multiple Thread (SIMT) execution:

- All threads in a warp execute the same instruction
- Divergent paths are serialized (warp divergence)
- Predication is used for short conditional sections

### Scheduling

1. **Block Scheduling**:
   - Blocks are assigned to SMs based on resources
   - Once assigned, a block runs to completion on that SM
   - Blocks cannot communicate with each other

2. **Warp Scheduling**:
   - Warps are the basic scheduling unit
   - Hardware warp schedulers select ready warps for execution
   - Latency hiding through warp interleaving

3. **Instruction-Level Scheduling**:
   - Instructions from different warps can be interleaved
   - Helps hide memory and instruction latency

### Occupancy

Occupancy is the ratio of active warps to maximum possible warps on an SM:

- Limited by resources: registers, shared memory, block size
- Higher occupancy generally improves latency hiding
- Not always linearly correlated with performance

Factors affecting occupancy:
- **Register usage per thread**: More registers = fewer warps
- **Shared memory per block**: More shared memory = fewer blocks
- **Block size**: Very small blocks reduce occupancy

## Performance Considerations

### Memory Optimization

1. **Coalesced Access**: Ensure threads in a warp access contiguous memory
2. **Shared Memory**: Use for data reused within a block
3. **L1/Texture Cache**: Leverage for read-only data with spatial locality
4. **Memory Bandwidth**: Often the limiting factor; minimize transfers

### Execution Optimization

1. **Occupancy**: Balance resource usage to maximize active warps
2. **Warp Divergence**: Minimize divergent paths within warps
3. **Instruction Mix**: Balance arithmetic operations and memory accesses
4. **Kernel Fusion**: Combine multiple operations into one kernel to reduce launch overhead

### Common Optimization Techniques

1. **Tiling**: Divide data into tiles that fit in shared memory
2. **Loop Unrolling**: Reduce loop overhead
3. **Prefetching**: Load data before it's needed
4. **Warp Shuffle**: Exchange data between threads in a warp without shared memory
5. **Persistent Threads**: Keep threads active for multiple work items

## Example Application

The accompanying `basic04.cu` demonstrates:

1. **Hardware inspection**: Querying and displaying device properties
2. **Thread hierarchy**: Visualizing the grid/block/thread structure
3. **Memory types**: Using global, shared, constant, local, and register memory
4. **Memory access patterns**: Demonstrating coalesced vs. non-coalesced access
5. **Warp execution**: Showing warp ID, lane ID, and divergence effects

### Key Code Sections

Thread identification and hierarchy:
```cuda
__global__ void threadHierarchyKernel() {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;
    
    // Print thread position
    printf("Thread (%d,%d,%d) in Block (%d,%d,%d)\n", tx, ty, tz, bx, by, bz);
}
```

Shared memory usage:
```cuda
__global__ void sharedMemoryKernel(float *input, float *output) {
    __shared__ float sharedData[256];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int localId = threadIdx.x;
    
    // Load data to shared memory
    sharedData[localId] = input[tid];
    
    // Synchronize
    __syncthreads();
    
    // Use shared data
    output[tid] = sharedData[localId];
}
```

## Further Reading

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [Professional CUDA C Programming](https://www.amazon.com/Professional-CUDA-Programming-John-Cheng/dp/1118739329)
- [Programming Massively Parallel Processors](https://www.amazon.com/Programming-Massively-Parallel-Processors-Hands/dp/0124159923) 