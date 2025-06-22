# CUPTI OpenACC 跟踪教程

> 完整的 GitHub 仓库和教程请访问 <https://github.com/eunomia-bpf/cupti-tutorial>。

## 简介

OpenACC 是一种基于指令的编程模型，通过允许开发者使用编译器翻译为 GPU 操作的编译指示来注释代码，从而简化 GPU 编程。虽然这简化了开发，但可能使性能分析变得困难，因为指令与实际 GPU 操作之间的关系并不总是清晰的。本教程演示如何使用 CUPTI 跟踪 OpenACC API 调用并将其与 GPU 活动关联，让您深入了解 OpenACC 代码如何在 GPU 上执行。

## 学习内容

- 如何拦截和跟踪 OpenACC API 调用
- 将 OpenACC 指令与生成的 GPU 操作关联
- 测量不同 OpenACC 操作的性能
- 识别 OpenACC 应用程序中的优化机会

## 理解 OpenACC 执行

当您编写 OpenACC 代码时，通常会发生以下情况：

1. OpenACC 编译器将您的指令翻译为 GPU 代码
2. 在运行时，OpenACC 运行时库：
   - 管理内存分配和传输
   - 启动 GPU 内核
   - 同步主机和设备之间的执行

通过跟踪这些操作，我们可以了解高级指令如何转换为低级 GPU 操作。

## 代码详解

### 1. 设置 OpenACC API 拦截

要跟踪 OpenACC API 调用，我们需要拦截 OpenACC 运行时库的函数：

```cpp
// 为 OpenACC API 函数定义函数指针类型
typedef int (*acc_init_t)(acc_device_t);
typedef int (*acc_get_num_devices_t)(acc_device_t);
typedef void (*acc_set_device_num_t)(int, acc_device_t);
typedef int (*acc_get_device_num_t)(acc_device_t);
typedef void* (*acc_create_t)(void*, size_t);
typedef void* (*acc_copyin_t)(void*, size_t);
typedef void (*acc_delete_t)(void*, size_t);
typedef void* (*acc_copyout_t)(void*, size_t);
// ... 更多函数类型 ...

// 原始函数指针
static acc_init_t real_acc_init = NULL;
static acc_get_num_devices_t real_acc_get_num_devices = NULL;
static acc_set_device_num_t real_acc_set_device_num = NULL;
// ... 更多函数指针 ...

// 使用 dlsym 初始化函数指针
void initOpenACCAPI()
{
    void *handle = dlopen("libopenacc.so", RTLD_NOW);
    if (!handle) {
        fprintf(stderr, "错误：无法加载 OpenACC 库\n");
        exit(1);
    }
    
    real_acc_init = (acc_init_t)dlsym(handle, "acc_init");
    real_acc_get_num_devices = (acc_get_num_devices_t)dlsym(handle, "acc_get_num_devices");
    real_acc_set_device_num = (acc_set_device_num_t)dlsym(handle, "acc_set_device_num");
    // ... 加载更多函数 ...
}
```

此代码：
1. 定义与 OpenACC API 匹配的函数指针类型
2. 创建变量来保存原始函数指针
3. 使用 `dlsym` 获取 OpenACC 库中实际函数的地址

### 2. 实现包装函数

对于我们想要跟踪的每个 OpenACC 函数，我们实现一个包装函数：

```cpp
// acc_init 的包装器
int acc_init(acc_device_t device_type)
{
    // 记录开始时间
    double startTime = getCurrentTime();
    
    // 调用真实函数
    int result = real_acc_init(device_type);
    
    // 记录结束时间
    double endTime = getCurrentTime();
    
    // 记录函数调用
    printf("[%.3f ms] acc_init(", (startTime - programStartTime) * 1000.0);
    printDeviceType(device_type);
    printf(")\n");
    
    return result;
}

// acc_create 的包装器
void* acc_create(void* host_addr, size_t size)
{
    double startTime = getCurrentTime();
    
    // 调用真实函数
    void* result = real_acc_create(host_addr, size);
    
    double endTime = getCurrentTime();
    
    // 记录带详细信息的函数调用
    printf("[%.3f ms] acc_create(%p, %zu) [%zu 元素]\n", 
           (startTime - programStartTime) * 1000.0,
           host_addr, size, size / sizeof(float));
    
    // 跟踪 CUDA 内存分配
    trackMemoryAllocation(host_addr, result, size);
    
    return result;
}
```

这些包装器：
1. 在调用真实函数前后记录时间
2. 记录函数调用信息
3. 跟踪内存分配等资源
4. 返回真实函数的结果

### 3. 与 CUDA 活动关联

要将 OpenACC API 调用与 CUDA 操作关联，我们使用 CUPTI 的活动 API：

```cpp
void initCUPTI()
{
    // 初始化 CUPTI 活动 API
    CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));
    
    // 启用我们想要跟踪的活动种类
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMORY));
    
    // 获取用于标准化时间的开始时间戳
    CUPTI_CALL(cuptiGetTimestamp(&startTimestamp));
}

void processActivity(CUpti_Activity *record)
{
    switch (record->kind) {
    case CUPTI_ACTIVITY_KIND_MEMCPY: {
        CUpti_ActivityMemcpy *memcpy = (CUpti_ActivityMemcpy *)record;
        
        // 查找对应的 OpenACC API 调用
        OpenACCMemoryOp *op = findOpenACCMemoryOp(memcpy->start);
        
        if (op) {
            // 此内存复制对应 OpenACC 数据指令
            printf("[%.3f ms] > CUDA %s 传输：%zu 字节 %s %p\n",
                   (memcpy->start - startTimestamp) / 1000.0,
                   getMemcpyKindString(memcpy->copyKind),
                   memcpy->bytes,
                   (memcpy->copyKind == CUPTI_ACTIVITY_MEMCPY_KIND_HTOD) ? "到" : "从",
                   (void*)memcpy->deviceId);
            
            // 更新统计信息
            if (memcpy->copyKind == CUPTI_ACTIVITY_MEMCPY_KIND_HTOD) {
                hostToDeviceTime += (memcpy->end - memcpy->start) / 1000.0;
            } else {
                deviceToHostTime += (memcpy->end - memcpy->start) / 1000.0;
            }
        }
        break;
    }
    case CUPTI_ACTIVITY_KIND_KERNEL: {
        CUpti_ActivityKernel5 *kernel = (CUpti_ActivityKernel5 *)record;
        
        // 查找对应的 OpenACC 计算区域
        OpenACCComputeRegion *region = findOpenACCComputeRegion(kernel->start);
        
        if (region) {
            // 此内核对应 OpenACC 计算指令
            printf("[%.3f ms] > CUDA 内核启动：%s [网格:%d 块:%d]\n",
                   (kernel->start - startTimestamp) / 1000.0,
                   kernel->name,
                   kernel->gridX,
                   kernel->blockX);
            
            // 更新统计信息
            kernelExecutionTime += (kernel->end - kernel->start) / 1000.0;
        }
        break;
    }
    // ... 处理其他活动类型 ...
    }
}
```

此代码：
1. 设置 CUPTI 收集内存传输和内核启动等活动
2. 处理每个活动记录并将其与 OpenACC 操作关联
3. 更新不同类型操作的性能统计

### 4. 示例 OpenACC 应用程序

示例包含一个简单的 OpenACC 应用程序来演示跟踪：

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <openacc.h>

#define N 4096

int main()
{
    float *a, *b, *c;
    int i;
    
    // 分配主机内存
    a = (float*)malloc(N * sizeof(float));
    b = (float*)malloc(N * sizeof(float));
    c = (float*)malloc(N * sizeof(float));
    
    // 初始化数组
    for (i = 0; i < N; i++) {
        a[i] = (float)i;
        b[i] = (float)(N - i);
    }
    
    // 向量加法的 OpenACC 指令
    #pragma acc data create(a[0:N], b[0:N], c[0:N])
    {
        #pragma acc update device(a[0:N], b[0:N])
        
        #pragma acc parallel loop
        for (i = 0; i < N; i++) {
            c[i] = a[i] + b[i];
        }
        
        #pragma acc update host(c[0:N])
    }
    
    // 验证结果
    for (i = 0; i < N; i++) {
        if (c[i] != (float)(N)) {
            printf("错误：c[%d] = %f，期望 %f\n", i, c[i], (float)N);
            break;
        }
    }
    
    if (i == N) {
        printf("结果验证通过！\n");
    }
    
    // 清理
    free(a);
    free(b);
    free(c);
    
    return 0;
}
```

## 运行教程

1. 构建示例：
   ```bash
   make
   ```

2. 运行 OpenACC 跟踪器：
   ```bash
   ./openacc_trace
   ```

## 理解输出

示例产生类似的输出：

```
[0.000 ms] acc_init(ACC_DEVICE_NVIDIA)
[2.345 ms] acc_get_num_devices(ACC_DEVICE_NVIDIA) = 1
[2.567 ms] acc_set_device_num(0, ACC_DEVICE_NVIDIA)
[5.123 ms] acc_create(0x7fff12345678, 16384) [4096 元素]
[5.234 ms] > CUDA 内存分配：16384 字节在设备 0
[5.456 ms] acc_create(0x7fff12345abc, 16384) [4096 元素]
[5.567 ms] > CUDA 内存分配：16384 字节在设备 0
[5.789 ms] acc_create(0x7fff12345def, 16384) [4096 元素]
[5.890 ms] > CUDA 内存分配：16384 字节在设备 0
[6.123 ms] acc_copyin(0x7fff12345678, 16384)
[6.234 ms] > CUDA H2D 传输：16384 字节到 0x140000000
[8.456 ms] acc_copyin(0x7fff12345abc, 16384)
[8.567 ms] > CUDA H2D 传输：16384 字节到 0x140004000
[10.789 ms] > CUDA 内核启动：acc_parallel_loop [网格:16 块:256]
[15.234 ms] acc_copyout(0x7fff12345def, 16384)
[15.345 ms] > CUDA D2H 传输：16384 字节从 0x140008000
结果验证通过！

性能摘要：
- 总执行时间：18.5 ms
- 内存分配：0.8 ms (4.3%)
- 主机到设备传输：2.2 ms (11.9%)
- 内核执行：4.8 ms (25.9%)
- 设备到主机传输：2.1 ms (11.4%)
- OpenACC 开销：8.6 ms (46.5%)
```

## 性能分析洞察

### OpenACC 开销分析

```cpp
class OpenACCPerformanceAnalyzer {
private:
    double totalOpenACCTime;
    double totalCUDATime;
    double memoryTransferTime;
    double kernelExecutionTime;

public:
    void generateReport() {
        printf("\n=== OpenACC 性能分析报告 ===\n");
        printf("总 OpenACC 时间：%.2f ms\n", totalOpenACCTime);
        printf("总 CUDA 时间：%.2f ms\n", totalCUDATime);
        printf("OpenACC 开销：%.2f ms (%.1f%%)\n", 
               totalOpenACCTime - totalCUDATime,
               ((totalOpenACCTime - totalCUDATime) / totalOpenACCTime) * 100);
        
        printf("\n详细分解：\n");
        printf("- 内存传输：%.2f ms (%.1f%%)\n", 
               memoryTransferTime, (memoryTransferTime / totalOpenACCTime) * 100);
        printf("- 内核执行：%.2f ms (%.1f%%)\n", 
               kernelExecutionTime, (kernelExecutionTime / totalOpenACCTime) * 100);
    }
    
    void suggestOptimizations() {
        printf("\n=== 优化建议 ===\n");
        
        double transferRatio = memoryTransferTime / kernelExecutionTime;
        if (transferRatio > 0.5) {
            printf("- 内存传输是瓶颈（比率：%.2f）\n", transferRatio);
            printf("  建议：使用数据指令减少传输\n");
        }
        
        double overheadRatio = (totalOpenACCTime - totalCUDATime) / totalOpenACCTime;
        if (overheadRatio > 0.3) {
            printf("- OpenACC 开销较高（%.1f%%）\n", overheadRatio * 100);
            printf("  建议：合并计算区域，减少 API 调用\n");
        }
    }
};
```

### 指令优化分析

```cpp
void analyzeOpenACCDirectives() {
    printf("\n=== OpenACC 指令分析 ===\n");
    
    // 分析数据指令效率
    if (dataReuseCount > 1) {
        printf("数据重用检测：%d 次\n", dataReuseCount);
        printf("建议：使用 data 区域包含多个计算\n");
    }
    
    // 分析并行化效率
    double parallelEfficiency = actualParallelism / theoreticalParallelism;
    printf("并行化效率：%.1f%%\n", parallelEfficiency * 100);
    
    if (parallelEfficiency < 0.8) {
        printf("建议：检查循环依赖和数据访问模式\n");
    }
}
```

## 故障排除

### 常见问题

1. **OpenACC 库加载失败**：确保安装了 OpenACC 运行时
2. **符号未找到**：检查 OpenACC 库版本兼容性
3. **CUPTI 初始化失败**：验证 CUDA 驱动和工具包版本

### 调试技巧

```cpp
// 启用详细的 OpenACC 跟踪
#define DEBUG_OPENACC_TRACE
#ifdef DEBUG_OPENACC_TRACE
#define ACC_TRACE(fmt, ...) printf("[ACC_TRACE] " fmt "\n", ##__VA_ARGS__)
#else
#define ACC_TRACE(fmt, ...)
#endif

void debugOpenACCCall(const char* funcName, double duration) {
    ACC_TRACE("%s 耗时 %.3f ms", funcName, duration);
    
    if (duration > SLOW_CALL_THRESHOLD) {
        printf("警告：%s 调用较慢 (%.3f ms)\n", funcName, duration);
    }
}
```

## 最佳实践

### OpenACC 优化

1. **数据局部性**：最小化主机-设备数据传输
2. **计算粒度**：平衡并行度和开销
3. **内存管理**：重用数据区域，避免重复分配

### 分析策略

1. **基准测试**：建立性能基线
2. **渐进优化**：一次优化一个瓶颈
3. **验证正确性**：确保优化不影响结果

这个 OpenACC 跟踪教程为理解和优化基于指令的 GPU 程序提供了强大的工具。 