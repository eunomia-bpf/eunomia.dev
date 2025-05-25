# CUDA PTX Example: Vector Multiplication Using Inline PTX Assembly

This example demonstrates how to use CUDA PTX (Parallel Thread Execution) inline assembly in your CUDA programs. PTX is NVIDIA's low-level parallel thread execution virtual machine and instruction set architecture (ISA).

You can find the code in <https://github.com/eunomia-bpf/basic-cuda-tutorial>

## What is PTX?

PTX is an intermediate assembly language that provides a stable programming model and instruction set for CUDA programs. It serves several purposes:

1. Provides a machine-independent ISA for parallel computing
2. Enables hand-tuning of code for specific GPU architectures
3. Allows direct control over GPU instructions
4. Facilitates optimization of performance-critical code sections

## Example Overview

The example in `basic02.cu` shows how to:
1. Write inline PTX assembly in CUDA C++
2. Use PTX for a simple arithmetic operation (multiplication by 2)
3. Integrate PTX code with regular CUDA kernels

## Code Explanation

### PTX Inline Assembly Function

```cuda
__device__ int multiplyByTwo(int x) {
    int result;
    asm("mul.lo.s32 %0, %1, 2;" : "=r"(result) : "r"(x));
    return result;
}
```

This function uses inline PTX assembly to multiply a number by 2. Let's break down the PTX instruction:

- `mul.lo.s32`: Multiply operation for 32-bit signed integers
- `%0`: First output operand (result)
- `%1`: First input operand (x)
- `2`: Immediate value to multiply by
- `:`: Separates the instruction from the operand mappings
- `"=r"(result)`: Output operand mapping
- `"r"(x)`: Input operand mapping

### CUDA Kernel

```cuda
__global__ void vectorMultiplyByTwoPTX(int* input, int* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = multiplyByTwo(input[idx]);
    }
}
```

The kernel applies our PTX function to each element of the input array in parallel.

## Common PTX Instructions

Here are some commonly used PTX instructions:

1. Arithmetic Operations:
   - `add.s32`: 32-bit integer addition
   - `sub.s32`: 32-bit integer subtraction
   - `mul.lo.s32`: 32-bit integer multiplication (low 32 bits)
   - `div.s32`: 32-bit integer division

2. Memory Operations:
   - `ld.global`: Load from global memory
   - `st.global`: Store to global memory
   - `ld.shared`: Load from shared memory
   - `st.shared`: Store to shared memory

3. Control Flow:
   - `bra`: Branch
   - `setp`: Set predicate
   - `@p`: Predicated execution

## Building and Running

To compile the example:
```bash
nvcc -o basic02 basic02.cu
```

To run:
```bash
./basic02
```

## Performance Considerations

1. PTX inline assembly should be used judiciously, typically only for performance-critical sections
2. Modern CUDA compilers often generate highly optimized code, so PTX may not always be necessary
3. PTX code is architecture-specific and may need adjustment for different GPU generations

## Further Reading

- [NVIDIA PTX ISA Documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [CUDA Inline PTX Assembly](https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html) 