# Tutorial: Understanding GPU Assembly with PTX

**Time Required:** 45-60 minutes
**Difficulty:** Intermediate
**Prerequisites:** Completed Tutorial 01 (Vector Addition), understanding of assembly language concepts helpful but not required

By the end of this tutorial, you will understand PTX (Parallel Thread Execution), NVIDIA's virtual assembly language for GPUs. You'll learn how to read compiler-generated PTX, write inline PTX assembly for performance optimization, and understand the relationship between high-level CUDA code and what actually executes on the GPU.

## What is PTX and Why Should You Care?

When you write CUDA code, it doesn't directly become machine instructions that run on your GPU. Instead, it goes through several compilation stages. Understanding PTX sits at the sweet spot between high-level CUDA C++ and low-level GPU machine code (SASS).

Think of PTX as an intermediate language similar to Java bytecode or LLVM IR. It provides several advantages:

**Architecture Independence:** PTX code can run on different GPU architectures. The driver compiles PTX to actual machine code for your specific GPU at runtime. This means code compiled years ago can still take advantage of newer GPUs.

**Performance Tuning:** Sometimes the CUDA compiler makes conservative choices. By writing inline PTX, you can hand-optimize critical sections when profiling shows they're bottlenecks.

**Understanding Compiler Output:** Reading PTX helps you understand what the compiler does with your code. You can spot inefficiencies and write better high-level code.

**Low-Level Control:** PTX gives you access to special instructions and hardware features not exposed in CUDA C++.

## The Compilation Pipeline

Before diving into PTX, understand where it fits in the compilation process:

```
CUDA C++ (.cu)
    ↓ [nvcc frontend]
PTX Assembly (.ptx)
    ↓ [ptxas assembler]
SASS Machine Code (.cubin)
    ↓ [driver at runtime]
GPU Execution
```

The `nvcc` compiler first translates your CUDA code to PTX. Then `ptxas` (the PTX assembler) converts PTX to SASS (Shader Assembly), which is the actual machine code for your GPU architecture. The CUDA driver can also perform this final step at runtime, enabling forward compatibility.

## Generating and Inspecting PTX

Let's start by looking at the PTX generated from our vector addition example. Compile with the PTX output flag:

```bash
nvcc -ptx 01-vector-addition.cu -o 01-vector-addition.ptx
```

Open `01-vector-addition.ptx` in a text editor. You'll see something like this:

```ptx
.visible .entry vectorAdd(
    .param .u64 vectorAdd_param_0,
    .param .u64 vectorAdd_param_1,
    .param .u64 vectorAdd_param_2,
    .param .u32 vectorAdd_param_3
)
{
    .reg .pred  %p<2>;
    .reg .b32   %r<5>;
    .reg .b64   %rd<11>;

    ld.param.u64    %rd1, [vectorAdd_param_0];
    ld.param.u64    %rd2, [vectorAdd_param_1];
    ld.param.u64    %rd3, [vectorAdd_param_2];
    ld.param.u32    %r1, [vectorAdd_param_3];

    mov.u32         %r2, %ctaid.x;
    mov.u32         %r3, %ntid.x;
    mov.u32         %r4, %tid.x;
    mad.lo.s32      %r1, %r3, %r2, %r4;

    setp.ge.s32     %p1, %r1, %r2;
    @%p1 bra        BB0_2;

    // ... vector addition code ...
}
```

Let's decode what this means.

## PTX Structure and Syntax

### Register Declaration

```ptx
.reg .pred  %p<2>;      // Predicate registers (for conditionals)
.reg .b32   %r<5>;      // 32-bit integer registers
.reg .b64   %rd<11>;    // 64-bit integer registers
.reg .f32   %f<8>;      // 32-bit floating-point registers
```

PTX uses virtual registers with unlimited supply. The actual hardware has a limited number of registers per thread, so `ptxas` maps these virtual registers to physical ones. This is why high register usage can limit occupancy.

### Thread Index Calculation

Remember our CUDA code:
```cuda
int i = blockDim.x * blockIdx.x + threadIdx.x;
```

In PTX, this becomes:
```ptx
mov.u32  %r2, %ctaid.x;      // blockIdx.x → r2
mov.u32  %r3, %ntid.x;       // blockDim.x → r3
mov.u32  %r4, %tid.x;        // threadIdx.x → r4
mad.lo.s32  %r1, %r3, %r2, %r4;  // r1 = r3 * r2 + r4
```

The `mad.lo.s32` instruction performs multiply-add in a single operation. This is a fused multiply-add (FMA), which is both faster and more accurate than separate multiply and add.

The special registers `%ctaid`, `%ntid`, and `%tid` correspond to CUDA built-ins:
- `%ctaid`: Block index (ctaid = compute task ID)
- `%ntid`: Block dimension (ntid = number of threads)
- `%tid`: Thread index within block

### Memory Operations

Loading from global memory:
```ptx
ld.global.f32  %f1, [%rd1];     // Load 32-bit float from address in rd1
```

Storing to global memory:
```ptx
st.global.f32  [%rd2], %f1;     // Store f1 to address in rd2
```

The `.global` qualifier specifies the memory space. Other options include `.shared`, `.local`, `.const`, and `.param`.

### Control Flow

PTX uses predicates and conditional branches:

```ptx
setp.ge.s32  %p1, %r1, %r2;     // p1 = (r1 >= r2)
@%p1 bra  BB0_2;                // Branch to BB0_2 if p1 is true
```

The `setp` instruction sets a predicate register based on a comparison. The `@%p1` prefix makes the branch conditional on that predicate. This is more efficient than traditional if-else branches because the GPU can predicate instructions rather than diverge execution.

## Writing Inline PTX Assembly

Now let's write some inline PTX in CUDA code. Open `02-ptx-assembly.cu` to see practical examples.

### Basic Inline PTX Syntax

The syntax for inline PTX in CUDA is:

```cuda
asm("instruction" : output_operands : input_operands : clobbered_registers);
```

Here's a simple example adding two integers:

```cuda
__device__ int addTwoNumbers(int a, int b) {
    int result;
    asm("add.s32 %0, %1, %2;" : "=r"(result) : "r"(a), "r"(b));
    return result;
}
```

Breaking this down:
- `"add.s32 %0, %1, %2;"`: The PTX instruction with placeholders
- `"=r"(result)`: Output operand (= means write, r means register)
- `"r"(a), "r"(b)`: Input operands (both in registers)
- `%0, %1, %2`: Refer to operands in order they're listed

### More Complex Example: Multiply-Add

Let's implement a fused multiply-add operation:

```cuda
__device__ float fma_custom(float a, float b, float c) {
    float result;
    asm("fma.rn.f32 %0, %1, %2, %3;"
        : "=f"(result)
        : "f"(a), "f"(b), "f"(c));
    return result;
}
```

The `fma.rn.f32` instruction:
- `fma`: Fused multiply-add
- `rn`: Rounding mode (round to nearest even)
- `f32`: 32-bit floating-point
- Computes: result = a * b + c

This single instruction is both faster and more accurate than separate multiply and add because it performs rounding only once.

### Optimizing Memory Access with PTX

Here's an example using vector memory operations:

```cuda
__device__ void load_vector4(float* ptr, float4& vec) {
    asm("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
        : "=f"(vec.x), "=f"(vec.y), "=f"(vec.z), "=f"(vec.w)
        : "l"(ptr));
}
```

The `.v4` qualifier loads four floats in a single transaction, which is much more efficient than four separate loads when accessing contiguous memory.

## Hands-On Exercise: Comparing Compiler Output

Let's write two versions of a simple kernel and compare their PTX:

Version 1 (Regular CUDA):
```cuda
__global__ void saxpy_simple(float a, float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}
```

Version 2 (With inline PTX):
```cuda
__global__ void saxpy_ptx(float a, float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float xi = x[i];
        float yi = y[i];
        float result;
        asm("fma.rn.f32 %0, %1, %2, %3;"
            : "=f"(result) : "f"(a), "f"(xi), "f"(yi));
        y[i] = result;
    }
}
```

Compile both with PTX output and compare:

```bash
nvcc -ptx -o saxpy_simple.ptx saxpy.cu
```

You'll likely find the compiler already uses `fma` instructions for the first version. Modern CUDA compilers are quite good at optimization, so inline PTX is only beneficial for very specific cases.

## Building and Running the PTX Example

Let's compile and run the full example:

```bash
make 02-ptx-assembly
./02-ptx-assembly
```

The program demonstrates different PTX techniques including inline assembly, device function pointers, and cooperative groups. Study the output to understand how each approach works.

## Understanding PTX Instruction Types

### Arithmetic Instructions

```ptx
add.s32  %r1, %r2, %r3;         // Integer addition
sub.f32  %f1, %f2, %f3;         // Float subtraction
mul.lo.s32  %r1, %r2, %r3;      // Integer multiply (low 32 bits)
div.rn.f32  %f1, %f2, %f3;      // Float division (round nearest)
mad.lo.s32  %r1, %r2, %r3, %r4; // Multiply-add
fma.rn.f32  %f1, %f2, %f3, %f4; // Fused multiply-add
```

### Comparison and Selection

```ptx
setp.eq.s32  %p1, %r1, %r2;     // p1 = (r1 == r2)
setp.lt.f32  %p1, %f1, %f2;     // p1 = (f1 < f2)
selp.s32  %r1, %r2, %r3, %p1;   // r1 = p1 ? r2 : r3
```

### Bit Manipulation

```ptx
and.b32  %r1, %r2, %r3;         // Bitwise AND
or.b32  %r1, %r2, %r3;          // Bitwise OR
xor.b32  %r1, %r2, %r3;         // Bitwise XOR
shl.b32  %r1, %r2, 3;           // Shift left by 3
shr.u32  %r1, %r2, 3;           // Unsigned shift right
```

### Special Functions

```ptx
sin.approx.f32  %f1, %f2;       // Fast sine approximation
cos.approx.f32  %f1, %f2;       // Fast cosine approximation
rsqrt.approx.f32  %f1, %f2;     // Fast reciprocal square root
```

## Memory Spaces and Qualifiers

PTX distinguishes different memory spaces:

**Global Memory (.global):**
```ptx
ld.global.f32  %f1, [%rd1];
st.global.f32  [%rd1], %f1;
```
Highest capacity, highest latency. Accessible by all threads.

**Shared Memory (.shared):**
```ptx
ld.shared.f32  %f1, [%rd1];
st.shared.f32  [%rd1], %f1;
```
Low latency, limited capacity. Shared within a thread block.

**Constant Memory (.const):**
```ptx
ld.const.f32  %f1, [%rd1];
```
Read-only, cached. Good for broadcast patterns.

**Local Memory (.local):**
```ptx
ld.local.f32  %f1, [%rd1];
st.local.f32  [%rd1], %f1;
```
Per-thread private memory. Actually stored in global memory but cached.

**Registers:**
```ptx
mov.f32  %f1, %f2;
```
Fastest memory, most limited. Each thread has its own registers.

## Cache Control and Memory Modifiers

PTX allows control over caching behavior:

```ptx
ld.global.ca.f32  %f1, [%rd1];   // Cache at all levels
ld.global.cg.f32  %f1, [%rd1];   // Cache globally
ld.global.cs.f32  %f1, [%rd1];   // Cache streaming
ld.global.cv.f32  %f1, [%rd1];   // Don't cache
```

These modifiers help optimize memory access patterns for different use cases. For example, `.cs` is good for data that won't be reused, while `.ca` is good for data with temporal locality.

## Warp-Level Operations

PTX exposes warp-level primitives for efficient thread cooperation:

### Warp Shuffle

```ptx
shfl.sync.bfly.b32  %r1|%p1, %r2, %r3, %r4, %r5;
```

Shuffle allows threads in a warp to exchange data without using shared memory. This is incredibly fast and useful for reductions and other collective operations.

In CUDA C++, you'd use:
```cuda
__shfl_sync(unsigned mask, int var, int srcLane);
```

But in PTX, you have more fine-grained control over the shuffle patterns (butterfly, up, down, indexed).

### Warp Vote

```ptx
vote.sync.all.pred  %p1, %p2, %r1;
```

Vote instructions test conditions across all threads in a warp. Useful for early exit conditions and divergence detection.

## Debugging PTX Code

When your inline PTX doesn't work as expected:

**Check Register Constraints:**
Make sure you're using the correct register types (r, f, d, p, l).

**Verify Instruction Syntax:**
Consult the [PTX ISA documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/) for exact syntax.

**Use cuobjdump:**
```bash
cuobjdump -ptx executable
```
This extracts the PTX from a compiled executable, useful for seeing what actually got compiled.

**Inspect SASS:**
```bash
cuobjdump -sass executable
```
See the actual machine code generated from your PTX. This helps understand if PTX is being optimized as expected.

**Use Nsight Compute:**
```bash
ncu --set full ./02-ptx-assembly
```
Profile at the instruction level to see execution efficiency.

## Performance Considerations

### When to Use Inline PTX

Only use inline PTX when:

**Profiling Shows a Hotspot:** Don't optimize prematurely. Profile first, identify bottlenecks, then consider PTX.

**Compiler Misses Optimization:** Sometimes the compiler doesn't recognize patterns you can hand-optimize.

**Need Special Instructions:** Some hardware features are only accessible via PTX.

**Require Specific Instruction Ordering:** Control exact execution order for numerical accuracy or memory ordering.

### When Not to Use Inline PTX

Avoid inline PTX when:

**Compiler Does It Better:** Modern CUDA compilers are excellent. Test first before hand-coding.

**Portability Matters:** PTX can vary between architectures. High-level CUDA is more portable.

**Maintainability Concerns:** Assembly is harder to read and maintain than high-level code.

**Rapid Development:** Write correct code first, optimize later if needed.

## Architecture-Specific Optimizations

PTX can target specific GPU architectures using `.target` directives:

```ptx
.target sm_75              // Turing architecture
.target sm_80              // Ampere architecture
.target sm_90              // Hopper architecture
```

Different architectures support different instructions. For example, Tensor Core operations are only available on sm_70+, and FP8 support requires sm_89+.

Check your GPU's compute capability:
```bash
nvidia-smi --query-gpu=compute_cap --format=csv
```

Then target that architecture in your Makefile:
```makefile
NVCC_FLAGS = -arch=sm_89
```

## Real-World Use Cases

### Case Study 1: Custom Atomic Operations

Sometimes you need atomic operations not provided by CUDA:

```cuda
__device__ float atomicMinFloat(float* addr, float value) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed,
                       __float_as_int(fminf(value, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}
```

This requires PTX-level understanding of atomic operations and type punning.

### Case Study 2: Warp-Level Reduction

Using warp shuffle for fast reduction:

```cuda
__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}
```

Understanding the PTX shuffle instructions helps you use these primitives effectively.

### Case Study 3: Memory Fence Control

Precise control over memory ordering:

```cuda
__device__ void writeWithFence(int* ptr, int value) {
    *ptr = value;
    asm volatile("membar.gl;" ::: "memory");
}
```

The `membar.gl` instruction ensures memory operations complete before continuing, critical for concurrent access patterns.

## Advanced Topics Preview

Future exploration areas:

**Dynamic Parallelism:** How kernel launches from device code work in PTX

**Cooperative Groups:** PTX primitives for grid-wide synchronization

**Tensor Cores:** PTX instructions for matrix multiplication acceleration

**Async Copy:** PTX async memory copy for improved pipelining

## Challenge Exercises

1. **Inspect Generated PTX:** Take your vector addition code and generate PTX. Identify the instructions corresponding to memory loads, the addition operation, and the memory store.

2. **Write a Custom Op:** Implement `__device__ int roundUpDiv(int a, int b)` using inline PTX that computes `(a + b - 1) / b`. Compare performance against the C++ version.

3. **Optimize Memory Access:** Write a kernel that loads float4 values using vectorized PTX instructions. Measure bandwidth improvement over scalar loads.

4. **Implement Warp Reduction:** Write a complete block-level sum reduction using warp shuffle instructions. Compare against a shared memory implementation.

5. **Architecture Comparison:** Generate PTX for compute capabilities 7.0, 8.0, and 9.0. Identify differences in available instructions.

## Summary

PTX is NVIDIA's intermediate assembly language that sits between CUDA C++ and machine code. Understanding PTX helps you understand what the compiler does with your code and enables hand-optimization when necessary.

Key PTX concepts include virtual registers, memory space qualifiers, predicated execution, and special GPU instructions. Inline PTX allows embedding assembly directly in CUDA code for fine-grained control.

Modern CUDA compilers generate excellent code, so inline PTX should be used sparingly and only after profiling. When used appropriately, PTX enables optimizations impossible to express in high-level code.

The PTX ISA evolves with each GPU architecture, exposing new hardware capabilities. Understanding PTX provides insight into GPU architecture and helps you write better high-level CUDA code.

## Next Steps

Continue to **Tutorial 03: GPU Programming Methods** to explore different approaches to GPU programming including Thrust, Unified Memory, and Dynamic Parallelism. You'll learn when to use each technique and how to combine them effectively.

## Further Reading

- [PTX ISA Reference](https://docs.nvidia.com/cuda/parallel-thread-execution/) - Complete PTX instruction set
- [Inline PTX Assembly](https://docs.nvidia.com/cuda/inline-ptx-assembly/) - Official guide to inline PTX
- [GPU Architecture Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities) - Hardware capabilities by generation
- [CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/) - Tools for inspecting compiled code
