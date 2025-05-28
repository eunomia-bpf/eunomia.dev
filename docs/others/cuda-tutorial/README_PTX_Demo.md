# PTX File Loading and Execution Demo

This demo shows how to load a PTX (Parallel Thread Execution) file and call it from CUDA C++ code using the CUDA Driver API.

## Files Overview

- `02-ptx-assembly.cu` - Main demo file showing both inline PTX assembly and PTX file loading
- `vector_add_kernel.cu` - Simple CUDA kernel source that gets compiled to PTX
- `vector_add.ptx` - Pre-generated PTX file (can be regenerated from source)
- `Makefile_ptx_demo` - Makefile for building the PTX demo

## What This Demo Demonstrates

1. **Inline PTX Assembly**: Using inline PTX assembly within CUDA kernels
2. **PTX File Loading**: Loading external PTX files using CUDA Driver API
3. **Runtime Execution**: Calling PTX functions at runtime using `cuLaunchKernel`

## Key Concepts

### Inline PTX Assembly
```cuda
__device__ int multiplyByTwo(int x) {
    int result;
    asm("mul.lo.s32 %0, %1, 2;" : "=r"(result) : "r"(x));
    return result;
}
```

### PTX File Loading
```cpp
// Load PTX module from file
CUmodule module;
cuModuleLoadData(&module, ptxSource.c_str());

// Get function from module
CUfunction function;
cuModuleGetFunction(&function, module, "vector_add_ptx");

// Launch kernel using Driver API
cuLaunchKernel(function, numBlocks, 1, 1, blockSize, 1, 1, 0, nullptr, args, nullptr);
```

## Building and Running

### Option 1: Use the dedicated Makefile
```bash
# Generate PTX file and build demo
make -f Makefile_ptx_demo

# Run the demo
make -f Makefile_ptx_demo run

# Clean generated files
make -f Makefile_ptx_demo clean
```

### Option 2: Manual compilation
```bash
# Step 1: Generate PTX file from CUDA kernel
nvcc -ptx vector_add_kernel.cu -o vector_add.ptx

# Step 2: Compile main program (requires CUDA Driver API)
nvcc -std=c++11 -lcuda 02-ptx-assembly.cu -o ptx_demo

# Step 3: Run the demo
./ptx_demo
```

## Expected Output

```
=== Demonstrating PTX Assembly in CUDA ===

1. Inline PTX Assembly - Vector Multiply by 2:
First 5 results (multiply by 2):
0 * 2 = 0
1 * 2 = 2
2 * 2 = 4
3 * 2 = 6
4 * 2 = 8

2. Loading PTX from external file - Vector Addition:
Successfully loaded PTX module from cuda-exp/vector_add.ptx
PTX kernel execution completed successfully
First 5 results (vector addition from PTX file):
0 + 1 = 1
1 + 2 = 3
2 + 3 = 5
3 + 4 = 7
4 + 5 = 9

PTX module unloaded successfully
```

## Technical Details

### CUDA Driver API vs Runtime API

This demo uses both APIs:
- **Runtime API** (`cudaMalloc`, `cudaMemcpy`, `<<<>>>` launches): Higher-level, easier to use
- **Driver API** (`cuModuleLoad`, `cuLaunchKernel`): Lower-level, required for PTX loading

### PTX Assembly Language

PTX is NVIDIA's intermediate assembly language that:
- Is architecture-independent
- Gets compiled to machine code (SASS) at runtime
- Allows fine-grained control over GPU execution
- Can be generated from CUDA C++ or written manually

### Memory Management

The demo shows two memory management approaches:
1. **Runtime API**: `cudaMalloc`/`cudaFree`
2. **Driver API**: `cuMemAlloc`/`cuMemFree`

## Use Cases for PTX Loading

1. **Dynamic Kernel Loading**: Load different kernels based on runtime conditions
2. **Hot-swapping**: Update GPU code without recompiling the host application
3. **JIT Compilation**: Generate and compile PTX code at runtime
4. **Performance Optimization**: Hand-optimized PTX for critical sections
5. **Code Obfuscation**: Distribute only PTX files instead of source code

## Troubleshooting

### Common Issues

1. **"Failed to load PTX module"**: Ensure the PTX file exists and is valid
2. **"Failed to get function"**: Check that the function name matches exactly
3. **Architecture mismatch**: Ensure PTX is compatible with target GPU

### Debugging Tips

- Use `nvdisasm` to examine generated SASS code
- Check CUDA error codes for detailed error information
- Verify GPU compute capability compatibility

## Further Reading

- [CUDA Driver API Documentation](https://docs.nvidia.com/cuda/cuda-driver-api/)
- [PTX ISA Reference](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) 