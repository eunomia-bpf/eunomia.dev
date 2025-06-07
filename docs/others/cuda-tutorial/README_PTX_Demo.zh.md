# PTX文件加载和执行演示

本演示展示了如何加载PTX（并行线程执行）文件并使用CUDA驱动API从CUDA C++代码中调用它。

## 文件概述

- `02-ptx-assembly.cu` - 主演示文件，展示了内联PTX汇编和PTX文件加载
- `vector_add_kernel.cu` - 简单的CUDA内核源代码，编译为PTX
- `vector_add.ptx` - 预生成的PTX文件（可以从源代码重新生成）
- `Makefile_ptx_demo` - 用于构建PTX演示的Makefile

## 本演示展示的内容

1. **内联PTX汇编**：在CUDA内核中使用内联PTX汇编
2. **PTX文件加载**：使用CUDA驱动API加载外部PTX文件
3. **运行时执行**：使用`cuLaunchKernel`在运行时调用PTX函数

## 关键概念

### 内联PTX汇编
```cuda
__device__ int multiplyByTwo(int x) {
    int result;
    asm("mul.lo.s32 %0, %1, 2;" : "=r"(result) : "r"(x));
    return result;
}
```

### PTX文件加载
```cpp
// 从文件加载PTX模块
CUmodule module;
cuModuleLoadData(&module, ptxSource.c_str());

// 从模块获取函数
CUfunction function;
cuModuleGetFunction(&function, module, "vector_add_ptx");

// 使用驱动API启动内核
cuLaunchKernel(function, numBlocks, 1, 1, blockSize, 1, 1, 0, nullptr, args, nullptr);
```

## 构建和运行

### 选项1：使用专用Makefile
```bash
# 生成PTX文件并构建演示
make -f Makefile_ptx_demo

# 运行演示
make -f Makefile_ptx_demo run

# 清理生成的文件
make -f Makefile_ptx_demo clean
```

### 选项2：手动编译
```bash
# 步骤1：从CUDA内核生成PTX文件
nvcc -ptx vector_add_kernel.cu -o vector_add.ptx

# 步骤2：编译主程序（需要CUDA驱动API）
nvcc -std=c++11 -lcuda 02-ptx-assembly.cu -o ptx_demo

# 步骤3：运行演示
./ptx_demo
```

## 预期输出

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

## 技术细节

### CUDA驱动API与运行时API

本演示使用了两种API：
- **运行时API**（`cudaMalloc`、`cudaMemcpy`、`<<<>>>`启动）：更高级，更易于使用
- **驱动API**（`cuModuleLoad`、`cuLaunchKernel`）：更低级，PTX加载所需

### PTX汇编语言

PTX是NVIDIA的中间汇编语言，它：
- 与架构无关
- 在运行时编译为机器代码（SASS）
- 允许对GPU执行进行细粒度控制
- 可以从CUDA C++生成或手动编写

### 内存管理

演示展示了两种内存管理方法：
1. **运行时API**：`cudaMalloc`/`cudaFree`
2. **驱动API**：`cuMemAlloc`/`cuMemFree`

## PTX加载的使用场景

1. **动态内核加载**：根据运行时条件加载不同的内核
2. **热交换**：在不重新编译主机应用程序的情况下更新GPU代码
3. **JIT编译**：在运行时生成并编译PTX代码
4. **性能优化**：为关键部分手动优化PTX
5. **代码混淆**：仅分发PTX文件而不是源代码

## 故障排除

### 常见问题

1. **"Failed to load PTX module"**：确保PTX文件存在且有效
2. **"Failed to get function"**：检查函数名是否完全匹配
3. **架构不匹配**：确保PTX与目标GPU兼容

### 调试技巧

- 使用`nvdisasm`检查生成的SASS代码
- 检查CUDA错误代码以获取详细的错误信息
- 验证GPU计算能力兼容性

## 进一步阅读

- [CUDA驱动API文档](https://docs.nvidia.com/cuda/cuda-driver-api/)
- [PTX ISA参考](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [CUDA编程指南](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) 