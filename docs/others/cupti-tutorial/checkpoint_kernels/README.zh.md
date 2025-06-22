# CUPTI 检查点 API 教程

> 完整的 GitHub 仓库和教程请访问 <https://github.com/eunomia-bpf/cupti-tutorial>。

## 简介

CUPTI 检查点 API 提供了一个强大的机制来捕获和恢复 GPU 设备状态，即使内核修改了自己的输入数据，也能实现可重现的内核执行。本教程演示如何使用检查点来确保多次内核调用的一致结果。

## 学习内容

- 如何使用 CUPTI 的检查点 API 保存和恢复 GPU 状态
- 确保可重现内核执行的技术
- 了解何时需要检查点来保证正确性
- 跨内核调用管理设备内存状态
- 基于检查点的调试和测试最佳实践

## 理解问题

许多 CUDA 内核在执行过程中修改其输入数据，这可能导致同一内核多次运行时产生不同的结果。这在以下情况中特别常见：

- **归约操作**覆盖输入数组
- **原地变换**在处理过程中修改数据
- **迭代算法**对输入和输出使用相同的缓冲区
- **调试场景**您希望重播完全相同的条件

## 检查点解决方案

CUPTI 的检查点 API 允许您：
1. **保存** GPU 内存在特定点的完整状态
2. **恢复**确切的状态，确保相同的条件
3. **重播**内核执行并保证可重现性

## 代码架构

### 检查点结构

```cpp
// 配置检查点对象
CUpti_Checkpoint cp = { CUpti_Checkpoint_STRUCT_SIZE };
cp.ctx = context;              // 要检查点的 CUDA 上下文
cp.optimizations = 1;          // 启用优化
```

### 基本检查点工作流

```cpp
// 1. 在首次内核执行前保存检查点
CUPTI_API_CALL(cuptiCheckpointSave(&cp));

// 2. 运行内核（可能修改输入数据）
MyKernel<<<blocks, threads>>>(deviceData, size);

// 3. 对于后续运行，首先恢复检查点
CUPTI_API_CALL(cuptiCheckpointRestore(&cp));

// 4. 在相同的初始条件下再次运行内核
MyKernel<<<blocks, threads>>>(deviceData, size);
```

## 示例详解

### 问题内核

我们的示例使用一个演示问题的归约内核：

```cpp
__global__ void Reduce(float *pData, size_t N)
{
    float totalSumData = 0.0;

    // 每个线程在本地求和其元素
    for (int i = threadIdx.x; i < N; i += blockDim.x)
    {
        totalSumData += pData[i];
    }

    // 将每线程的和保存回输入数组（修改输入！）
    pData[threadIdx.x] = totalSumData;
    
    __syncthreads();

    // 线程 0 归约到最终结果
    if (threadIdx.x == 0)
    {
        float totalSum = 0.0;
        size_t setElements = (blockDim.x < N ? blockDim.x : N);
        
        for (int i = 0; i < setElements; i++)
        {
            totalSum += pData[i];
        }
        
        pData[0] = totalSum;  // 最终结果
    }
}
```

**关键问题**：此内核用中间结果覆盖输入数组 `pData`，使后续运行产生不同的结果。

### 不使用检查点

```cpp
// 使用所有 1.0 值初始化数组
for (size_t i = 0; i < elements; i++) {
    pHostA[i] = 1.0;
}
cudaMemcpy(pDeviceA, pHostA, size, cudaMemcpyHostToDevice);

// 多次运行内核
for (int repeat = 0; repeat < 3; repeat++) {
    Reduce<<<1, 64>>>(pDeviceA, elements);
    
    float result;
    cudaMemcpy(&result, pDeviceA, sizeof(float), cudaMemcpyDeviceToHost);
    printf("迭代 %d：结果 = %f\n", repeat + 1, result);
}
```

**输出**：
```
迭代 1：结果 = 1048576.000000  // 100万个1的正确和
迭代 2：结果 = 64.000000       // 错误！输入被修改了
迭代 3：结果 = 1.000000        // 更错误！
```

### 使用检查点

```cpp
// 配置检查点
CUpti_Checkpoint cp = { CUpti_Checkpoint_STRUCT_SIZE };
cp.ctx = context;
cp.optimizations = 1;

float expected;

for (int repeat = 0; repeat < 3; repeat++) {
    // 保存或恢复检查点
    if (repeat == 0) {
        CUPTI_API_CALL(cuptiCheckpointSave(&cp));
    } else {
        CUPTI_API_CALL(cuptiCheckpointRestore(&cp));
    }
    
    // 在相同的初始条件下运行内核
    Reduce<<<1, 64>>>(pDeviceA, elements);
    
    float result;
    cudaMemcpy(&result, pDeviceA, sizeof(float), cudaMemcpyDeviceToHost);
    
    if (repeat == 0) {
        expected = result;  // 保存预期结果
    }
    
    printf("迭代 %d：结果 = %f\n", repeat + 1, result);
    
    // 验证可重现性
    if (result != expected) {
        printf("错误：结果不一致！\n");
        exit(1);
    }
}
```

**输出**：
```
迭代 1：结果 = 1048576.000000  // 正确结果
迭代 2：结果 = 1048576.000000  // 相同结果！
迭代 3：结果 = 1048576.000000  // 一致！
```

## 构建和运行

### 先决条件

- 支持 CUPTI 的 CUDA Toolkit
- 与 CUDA 兼容的 C++ 编译器
- 计算能力 3.5 或更高的 GPU

### 构建过程

```bash
cd checkpoint_kernels
make
```

### 执行

```bash
./checkpoint_kernels
```

## 高级检查点技术

### 检查点优化

```cpp
// 启用优化以获得更好的性能
CUpti_Checkpoint cp = { CUpti_Checkpoint_STRUCT_SIZE };
cp.ctx = context;
cp.optimizations = 1;  // 启用所有优化

// 替代：禁用优化用于调试
cp.optimizations = 0;  // 更慢但更彻底
```

### 选择性内存检查点

```cpp
class SelectiveCheckpoint {
private:
    std::vector<CUpti_Checkpoint> checkpoints;
    std::vector<void*> criticalPointers;

public:
    void addCriticalMemory(void* ptr, size_t size) {
        criticalPointers.push_back(ptr);
        // 为特定内存区域配置检查点
    }
    
    void saveSelectiveState() {
        for (auto& cp : checkpoints) {
            CUPTI_API_CALL(cuptiCheckpointSave(&cp));
        }
    }
    
    void restoreSelectiveState() {
        for (auto& cp : checkpoints) {
            CUPTI_API_CALL(cuptiCheckpointRestore(&cp));
        }
    }
};
```

### 基于检查点的调试

```cpp
class CheckpointDebugger {
private:
    CUpti_Checkpoint debugCheckpoint;
    std::vector<float> expectedResults;

public:
    void setDebugPoint(CUcontext context) {
        debugCheckpoint = { CUpti_Checkpoint_STRUCT_SIZE };
        debugCheckpoint.ctx = context;
        debugCheckpoint.optimizations = 0;  // 完整状态捕获
        
        CUPTI_API_CALL(cuptiCheckpointSave(&debugCheckpoint));
    }
    
    bool verifyReproducibility(int iterations) {
        bool allMatch = true;
        
        for (int i = 0; i < iterations; i++) {
            if (i > 0) {
                CUPTI_API_CALL(cuptiCheckpointRestore(&debugCheckpoint));
            }
            
            // 运行要测试的内核
            runTestKernel();
            
            float currentResult = getResult();
            
            if (i == 0) {
                expectedResults.push_back(currentResult);
            } else {
                if (currentResult != expectedResults[0]) {
                    printf("迭代 %d：不一致结果 %f vs %f\n", 
                           i, currentResult, expectedResults[0]);
                    allMatch = false;
                }
            }
        }
        
        return allMatch;
    }
};
```

## 性能考虑

### 检查点开销

检查点操作会产生以下开销：

1. **内存复制**：保存大型 GPU 内存状态需要时间
2. **存储需求**：检查点需要额外的 GPU 内存存储状态
3. **同步开销**：检查点操作可能需要 GPU 同步

### 优化策略

```cpp
// 最小化检查点大小
void optimizeCheckpointSize() {
    // 只保存关键内存区域
    // 释放不必要的临时缓冲区
    // 使用压缩（如果支持）
}

// 批量检查点操作
void batchCheckpointOperations() {
    // 将多个小的内存区域合并为单个检查点
    // 重用检查点对象以减少分配开销
}
```

## 实际应用

### 单元测试

```cpp
// 内核正确性验证
void testKernelCorrectness() {
    CUpti_Checkpoint cp = setupCheckpoint();
    
    // 运行参考实现
    CUPTI_API_CALL(cuptiCheckpointSave(&cp));
    float reference = runReferenceKernel();
    
    // 测试优化版本
    CUPTI_API_CALL(cuptiCheckpointRestore(&cp));
    float optimized = runOptimizedKernel();
    
    assert(abs(reference - optimized) < 1e-6);
}
```

### 性能基准测试

```cpp
// 可重现的性能测量
double benchmarkKernel(int iterations) {
    CUpti_Checkpoint cp = setupCheckpoint();
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        if (i > 0) {
            CUPTI_API_CALL(cuptiCheckpointRestore(&cp));
        } else {
            CUPTI_API_CALL(cuptiCheckpointSave(&cp));
        }
        
        runBenchmarkKernel();
        cudaDeviceSynchronize();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count() / iterations;
}
```

### 调试工具

```cpp
// 内核状态比较工具
class StateComparator {
    void compareStates(void* state1, void* state2, size_t size) {
        if (memcmp(state1, state2, size) != 0) {
            printf("检测到状态差异\n");
            dumpStateDifferences(state1, state2, size);
        }
    }
    
    void dumpStateDifferences(void* s1, void* s2, size_t size) {
        float* f1 = (float*)s1;
        float* f2 = (float*)s2;
        
        for (size_t i = 0; i < size/sizeof(float); i++) {
            if (f1[i] != f2[i]) {
                printf("位置 %zu：%f vs %f\n", i, f1[i], f2[i]);
            }
        }
    }
};
```

## 故障排除

### 常见问题

1. **检查点保存失败**：检查 GPU 内存是否不足
2. **恢复不完整**：某些内存区域可能被忽略
3. **性能影响过大**：考虑选择性检查点
4. **同步问题**：确保适当的流同步

### 调试技巧

```cpp
// 启用详细的检查点日志
#define DEBUG_CHECKPOINT(fmt, ...) \
    printf("[CHECKPOINT] " fmt "\n", ##__VA_ARGS__)

void debugCheckpointOperation(CUpti_Checkpoint* cp) {
    DEBUG_CHECKPOINT("开始检查点操作");
    
    size_t memoryUsed = getGPUMemoryUsage();
    DEBUG_CHECKPOINT("GPU 内存使用：%zu MB", memoryUsed / (1024*1024));
    
    auto start = std::chrono::high_resolution_clock::now();
    CUptiResult result = cuptiCheckpointSave(cp);
    auto end = std::chrono::high_resolution_clock::now();
    
    if (result == CUPTI_SUCCESS) {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        DEBUG_CHECKPOINT("检查点保存成功，耗时：%ld μs", duration.count());
    } else {
        DEBUG_CHECKPOINT("检查点保存失败，错误代码：%d", result);
    }
}
```

## 最佳实践

### 检查点使用指南

1. **最小范围**：只对真正需要的内存区域使用检查点
2. **早期验证**：在开发过程中验证可重现性
3. **性能测量**：测量检查点开销并相应优化
4. **错误处理**：实现健壮的错误处理和恢复

### 内存管理

1. **资源清理**：始终清理检查点资源
2. **内存池**：为重复操作重用检查点对象
3. **大小限制**：对大型应用设置合理的检查点大小限制

这个检查点 API 教程为在 CUDA 应用程序中实现可重现和可调试的内核执行提供了强大的工具。 