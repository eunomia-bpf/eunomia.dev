# 教程：使用PTX理解GPU汇编

**所需时间：** 45-60分钟
**难度：** 中级
**前置要求：** 完成教程01（向量加法），了解汇编语言概念会有帮助但不是必需的

完成本教程后，你将理解PTX（并行线程执行），这是NVIDIA的GPU虚拟汇编语言。你将学习如何阅读编译器生成的PTX，编写内联PTX汇编以优化性能，以及理解高级CUDA代码与实际在GPU上执行的代码之间的关系。

## 什么是PTX，为什么要关心它？

当你编写CUDA代码时，它不会直接变成在GPU上运行的机器指令。相反，它会经过几个编译阶段。理解PTX处于高级CUDA C++和低级GPU机器代码（SASS）之间的最佳位置。

可以把PTX看作是一种中间语言，类似于Java字节码或LLVM IR。它提供了几个优势：

**架构独立性：** PTX代码可以在不同的GPU架构上运行。驱动程序在运行时将PTX编译为你特定GPU的实际机器代码。这意味着多年前编译的代码仍然可以利用更新的GPU。

**性能调优：** 有时CUDA编译器会做出保守的选择。通过编写内联PTX，你可以在性能分析显示瓶颈时手动优化关键部分。

**理解编译器输出：** 阅读PTX帮助你理解编译器对你的代码做了什么。你可以发现低效之处并编写更好的高级代码。

**低级控制：** PTX让你访问CUDA C++中未公开的特殊指令和硬件功能。

## 编译管道

在深入PTX之前，理解它在编译过程中的位置：

```
CUDA C++ (.cu)
    ↓ [nvcc前端]
PTX汇编 (.ptx)
    ↓ [ptxas汇编器]
SASS机器代码 (.cubin)
    ↓ [运行时驱动程序]
GPU执行
```

`nvcc`编译器首先将你的CUDA代码转换为PTX。然后`ptxas`（PTX汇编器）将PTX转换为SASS（着色器汇编），这是你的GPU架构的实际机器代码。CUDA驱动程序也可以在运行时执行这最后一步，实现向前兼容性。

## 生成和检查PTX

让我们首先看看从向量加法示例生成的PTX。使用PTX输出标志编译：

```bash
nvcc -ptx 01-vector-addition.cu -o 01-vector-addition.ptx
```

在文本编辑器中打开`01-vector-addition.ptx`。你会看到类似这样的内容：

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

    // ... 向量加法代码 ...
}
```

让我们解码这意味着什么。

## PTX结构和语法

### 寄存器声明

```ptx
.reg .pred  %p<2>;      // 谓词寄存器（用于条件判断）
.reg .b32   %r<5>;      // 32位整数寄存器
.reg .b64   %rd<11>;    // 64位整数寄存器
.reg .f32   %f<8>;      // 32位浮点寄存器
```

PTX使用虚拟寄存器，供应无限。实际硬件每个线程的寄存器数量有限，所以`ptxas`将这些虚拟寄存器映射到物理寄存器。这就是为什么高寄存器使用量会限制占用率。

### 线程索引计算

记住我们的CUDA代码：
```cuda
int i = blockDim.x * blockIdx.x + threadIdx.x;
```

在PTX中，这变成：
```ptx
mov.u32  %r2, %ctaid.x;      // blockIdx.x → r2
mov.u32  %r3, %ntid.x;       // blockDim.x → r3
mov.u32  %r4, %tid.x;        // threadIdx.x → r4
mad.lo.s32  %r1, %r3, %r2, %r4;  // r1 = r3 * r2 + r4
```

`mad.lo.s32`指令在单个操作中执行乘加。这是一个融合乘加（FMA），它既更快又比分离的乘法和加法更精确。

特殊寄存器`%ctaid`、`%ntid`和`%tid`对应于CUDA内置变量：
- `%ctaid`：块索引（ctaid = 计算任务ID）
- `%ntid`：块维度（ntid = 线程数）
- `%tid`：块内的线程索引

### 内存操作

从全局内存加载：
```ptx
ld.global.f32  %f1, [%rd1];     // 从rd1中的地址加载32位浮点数
```

存储到全局内存：
```ptx
st.global.f32  [%rd2], %f1;     // 将f1存储到rd2中的地址
```

`.global`限定符指定内存空间。其他选项包括`.shared`、`.local`、`.const`和`.param`。

### 控制流

PTX使用谓词和条件分支：

```ptx
setp.ge.s32  %p1, %r1, %r2;     // p1 = (r1 >= r2)
@%p1 bra  BB0_2;                // 如果p1为真，分支到BB0_2
```

`setp`指令基于比较设置谓词寄存器。`@%p1`前缀使分支条件依赖于该谓词。这比传统的if-else分支更高效，因为GPU可以谓词指令而不是分歧执行。

## 编写内联PTX汇编

现在让我们在CUDA代码中编写一些内联PTX。打开`02-ptx-assembly.cu`查看实际示例。

### 基本内联PTX语法

CUDA中内联PTX的语法是：

```cuda
asm("指令" : 输出操作数 : 输入操作数 : 破坏的寄存器);
```

这是一个添加两个整数的简单示例：

```cuda
__device__ int addTwoNumbers(int a, int b) {
    int result;
    asm("add.s32 %0, %1, %2;" : "=r"(result) : "r"(a), "r"(b));
    return result;
}
```

分解这个：
- `"add.s32 %0, %1, %2;"`：带有占位符的PTX指令
- `"=r"(result)`：输出操作数（=表示写入，r表示寄存器）
- `"r"(a), "r"(b)`：输入操作数（都在寄存器中）
- `%0, %1, %2`：按列出的顺序引用操作数

### 更复杂的示例：乘加

让我们实现一个融合乘加操作：

```cuda
__device__ float fma_custom(float a, float b, float c) {
    float result;
    asm("fma.rn.f32 %0, %1, %2, %3;"
        : "=f"(result)
        : "f"(a), "f"(b), "f"(c));
    return result;
}
```

`fma.rn.f32`指令：
- `fma`：融合乘加
- `rn`：舍入模式（舍入到最近偶数）
- `f32`：32位浮点
- 计算：result = a * b + c

这个单一指令既更快又比分离的乘法和加法更精确，因为它只执行一次舍入。

### 使用PTX优化内存访问

这是一个使用向量内存操作的示例：

```cuda
__device__ void load_vector4(float* ptr, float4& vec) {
    asm("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
        : "=f"(vec.x), "=f"(vec.y), "=f"(vec.z), "=f"(vec.w)
        : "l"(ptr));
}
```

`.v4`限定符在单个事务中加载四个浮点数，当访问连续内存时，这比四个单独的加载效率高得多。

## 动手练习：比较编译器输出

让我们编写一个简单内核的两个版本并比较它们的PTX：

版本1（常规CUDA）：
```cuda
__global__ void saxpy_simple(float a, float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}
```

版本2（使用内联PTX）：
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

用PTX输出编译两者并比较：

```bash
nvcc -ptx -o saxpy_simple.ptx saxpy.cu
```

你可能会发现编译器已经为第一个版本使用了`fma`指令。现代CUDA编译器在优化方面相当出色，所以内联PTX仅在非常特定的情况下才有益。

## 构建并运行PTX示例

让我们编译并运行完整示例：

```bash
make 02-ptx-assembly
./02-ptx-assembly
```

该程序演示了不同的PTX技术，包括内联汇编、设备函数指针和协作组。研究输出以理解每种方法的工作原理。

## 理解PTX指令类型

### 算术指令

```ptx
add.s32  %r1, %r2, %r3;         // 整数加法
sub.f32  %f1, %f2, %f3;         // 浮点减法
mul.lo.s32  %r1, %r2, %r3;      // 整数乘法（低32位）
div.rn.f32  %f1, %f2, %f3;      // 浮点除法（舍入到最近）
mad.lo.s32  %r1, %r2, %r3, %r4; // 乘加
fma.rn.f32  %f1, %f2, %f3, %f4; // 融合乘加
```

### 比较和选择

```ptx
setp.eq.s32  %p1, %r1, %r2;     // p1 = (r1 == r2)
setp.lt.f32  %p1, %f1, %f2;     // p1 = (f1 < f2)
selp.s32  %r1, %r2, %r3, %p1;   // r1 = p1 ? r2 : r3
```

### 位操作

```ptx
and.b32  %r1, %r2, %r3;         // 按位与
or.b32  %r1, %r2, %r3;          // 按位或
xor.b32  %r1, %r2, %r3;         // 按位异或
shl.b32  %r1, %r2, 3;           // 左移3位
shr.u32  %r1, %r2, 3;           // 无符号右移
```

### 特殊函数

```ptx
sin.approx.f32  %f1, %f2;       // 快速正弦近似
cos.approx.f32  %f1, %f2;       // 快速余弦近似
rsqrt.approx.f32  %f1, %f2;     // 快速平方根倒数
```

## 内存空间和限定符

PTX区分不同的内存空间：

**全局内存（.global）：**
```ptx
ld.global.f32  %f1, [%rd1];
st.global.f32  [%rd1], %f1;
```
最大容量，最高延迟。所有线程都可以访问。

**共享内存（.shared）：**
```ptx
ld.shared.f32  %f1, [%rd1];
st.shared.f32  [%rd1], %f1;
```
低延迟，容量有限。在线程块内共享。

**常量内存（.const）：**
```ptx
ld.const.f32  %f1, [%rd1];
```
只读，缓存。适合广播模式。

**局部内存（.local）：**
```ptx
ld.local.f32  %f1, [%rd1];
st.local.f32  [%rd1], %f1;
```
每线程私有内存。实际存储在全局内存中但被缓存。

**寄存器：**
```ptx
mov.f32  %f1, %f2;
```
最快的内存，最受限。每个线程都有自己的寄存器。

## 缓存控制和内存修饰符

PTX允许控制缓存行为：

```ptx
ld.global.ca.f32  %f1, [%rd1];   // 在所有级别缓存
ld.global.cg.f32  %f1, [%rd1];   // 全局缓存
ld.global.cs.f32  %f1, [%rd1];   // 流式缓存
ld.global.cv.f32  %f1, [%rd1];   // 不缓存
```

这些修饰符帮助优化不同用例的内存访问模式。例如，`.cs`适合不会重用的数据，而`.ca`适合具有时间局部性的数据。

## Warp级操作

PTX公开了warp级原语以实现高效的线程协作：

### Warp洗牌

```ptx
shfl.sync.bfly.b32  %r1|%p1, %r2, %r3, %r4, %r5;
```

洗牌允许warp中的线程交换数据而不使用共享内存。这非常快，对归约和其他集体操作很有用。

在CUDA C++中，你会使用：
```cuda
__shfl_sync(unsigned mask, int var, int srcLane);
```

但在PTX中，你可以更精细地控制洗牌模式（蝶式、向上、向下、索引）。

### Warp投票

```ptx
vote.sync.all.pred  %p1, %p2, %r1;
```

投票指令测试warp中所有线程的条件。对早期退出条件和分歧检测很有用。

## 调试PTX代码

当你的内联PTX不按预期工作时：

**检查寄存器约束：**
确保你使用了正确的寄存器类型（r、f、d、p、l）。

**验证指令语法：**
查阅[PTX ISA文档](https://docs.nvidia.com/cuda/parallel-thread-execution/)以获取确切的语法。

**使用cuobjdump：**
```bash
cuobjdump -ptx executable
```
这从编译的可执行文件中提取PTX，对于查看实际编译的内容很有用。

**检查SASS：**
```bash
cuobjdump -sass executable
```
查看从你的PTX生成的实际机器代码。这有助于理解PTX是否按预期优化。

**使用Nsight Compute：**
```bash
ncu --set full ./02-ptx-assembly
```
在指令级别进行性能分析以查看执行效率。

## 性能考虑

### 何时使用内联PTX

仅在以下情况下使用内联PTX：

**性能分析显示热点：** 不要过早优化。先进行性能分析，识别瓶颈，然后考虑PTX。

**编译器错过优化：** 有时编译器不识别你可以手动优化的模式。

**需要特殊指令：** 某些硬件功能只能通过PTX访问。

**需要特定指令顺序：** 控制确切的执行顺序以获得数值精度或内存排序。

### 何时不使用内联PTX

在以下情况下避免使用内联PTX：

**编译器做得更好：** 现代CUDA编译器非常出色。在手工编码之前先测试。

**可移植性很重要：** PTX在不同架构之间可能有所不同。高级CUDA更具可移植性。

**可维护性问题：** 汇编比高级代码更难阅读和维护。

**快速开发：** 先编写正确的代码，如果需要再优化。

## 架构特定优化

PTX可以使用`.target`指令针对特定的GPU架构：

```ptx
.target sm_75              // Turing架构
.target sm_80              // Ampere架构
.target sm_90              // Hopper架构
```

不同的架构支持不同的指令。例如，Tensor Core操作仅在sm_70+上可用，FP8支持需要sm_89+。

检查你的GPU的计算能力：
```bash
nvidia-smi --query-gpu=compute_cap --format=csv
```

然后在Makefile中针对该架构：
```makefile
NVCC_FLAGS = -arch=sm_89
```

## 实际用例

### 案例研究1：自定义原子操作

有时你需要CUDA未提供的原子操作：

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

这需要对原子操作和类型双关的PTX级理解。

### 案例研究2：Warp级归约

使用warp洗牌进行快速归约：

```cuda
__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}
```

理解PTX洗牌指令帮助你有效地使用这些原语。

### 案例研究3：内存栅栏控制

对内存排序的精确控制：

```cuda
__device__ void writeWithFence(int* ptr, int value) {
    *ptr = value;
    asm volatile("membar.gl;" ::: "memory");
}
```

`membar.gl`指令确保内存操作在继续之前完成，对并发访问模式至关重要。

## 高级主题预览

未来探索领域：

**动态并行：** 设备代码中的内核启动在PTX中如何工作

**协作组：** 用于网格范围同步的PTX原语

**Tensor Core：** 用于矩阵乘法加速的PTX指令

**异步复制：** 用于改进流水线的PTX异步内存复制

## 挑战练习

1. **检查生成的PTX：** 获取你的向量加法代码并生成PTX。识别对应于内存加载、加法操作和内存存储的指令。

2. **编写自定义操作：** 使用内联PTX实现`__device__ int roundUpDiv(int a, int b)`，计算`(a + b - 1) / b`。与C++版本比较性能。

3. **优化内存访问：** 编写一个使用向量化PTX指令加载float4值的内核。测量相对于标量加载的带宽改进。

4. **实现Warp归约：** 使用warp洗牌指令编写完整的块级求和归约。与共享内存实现进行比较。

5. **架构比较：** 为计算能力7.0、8.0和9.0生成PTX。识别可用指令的差异。

## 总结

PTX是NVIDIA的中间汇编语言，位于CUDA C++和机器代码之间。理解PTX帮助你理解编译器对你的代码做了什么，并在必要时实现手动优化。

关键的PTX概念包括虚拟寄存器、内存空间限定符、谓词执行和特殊GPU指令。内联PTX允许在CUDA代码中直接嵌入汇编以实现精细控制。

现代CUDA编译器生成出色的代码，所以应该谨慎使用内联PTX，并且只在性能分析后使用。当适当使用时，PTX能够实现在高级代码中无法表达的优化。

PTX ISA随每个GPU架构演进，公开新的硬件能力。理解PTX提供了对GPU架构的洞察，并帮助你编写更好的高级CUDA代码。

## 下一步

继续学习**教程03：GPU编程方法**，探索不同的GPU编程方法，包括Thrust、统一内存和动态并行。你将学习何时使用每种技术以及如何有效地组合它们。

## 进一步阅读

- [PTX ISA参考](https://docs.nvidia.com/cuda/parallel-thread-execution/) - 完整的PTX指令集
- [内联PTX汇编](https://docs.nvidia.com/cuda/inline-ptx-assembly/) - 内联PTX的官方指南
- [GPU架构文档](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities) - 各代硬件能力
- [CUDA二进制工具](https://docs.nvidia.com/cuda/cuda-binary-utilities/) - 检查编译代码的工具
