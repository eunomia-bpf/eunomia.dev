# CUPTI 自动范围性能分析教程

> GitHub 仓库和完整教程可在 <https://github.com/eunomia-bpf/cupti-tutorial> 获取。

## 简介

分析 CUDA 应用程序通常需要手动在代码中添加插桩来定义要分析的区域。然而，CUPTI 的自动范围分析功能通过自动检测和分析内核启动来简化这一过程。本教程演示了如何使用这一强大功能在不修改内核的情况下收集性能指标。

## 您将学到的内容

- 如何设置 CUDA 内核的自动性能分析
- 在无需手动插桩的情况下收集性能指标
- 使用 NVIDIA 的高级分析 API
- 解释收集的指标以进行性能分析

## 理解自动范围分析

自动范围分析能够自动检测 CUDA 内核启动的时机，并为每个内核收集性能指标。这在以下情况下特别有用：

- 分析第三方代码时无法添加插桩
- 希望在不手动干预的情况下分析应用程序中的所有内核
- 需要在不修改源代码的情况下快速了解性能概况

## 代码演练

### 1. 设置分析环境

首先，我们需要初始化 CUPTI 分析器并将其配置为自动范围分析：

```cpp
// 初始化 CUPTI 和 NVPW 库
NVPW_InitializeHost_Params initializeHostParams = {NVPW_InitializeHost_Params_STRUCT_SIZE};
NVPW_InitializeHost(&initializeHostParams);

// 创建计数器数据镜像
NV_CUPTI_CALL(cuptiProfilerInitialize(&profilerInitializeParams));

// 为我们要收集的指标设置配置
const char* metricName = METRIC_NAME; // 默认为 "smsp__warps_launched.avg+"
```

### 2. 创建计数器数据镜像

计数器数据镜像是 CUPTI 存储原始性能数据的地方：

```cpp
// 创建将存储收集指标的计数器数据镜像
CUpti_Profiler_CounterDataImageOptions counterDataImageOptions;
counterDataImageOptions.size = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
counterDataImageOptions.pCounterDataPrefix = NULL;
counterDataImageOptions.counterDataPrefixSize = 0;
counterDataImageOptions.maxNumRanges = 2;
counterDataImageOptions.maxNumRangeTreeNodes = 2;
counterDataImageOptions.maxRangeNameLength = 64;

CUpti_Profiler_CounterDataImage_CalculateSize_Params calculateSizeParams = {
    CUpti_Profiler_CounterDataImage_CalculateSize_Params_STRUCT_SIZE};
calculateSizeParams.pOptions = &counterDataImageOptions;
NV_CUPTI_CALL(cuptiProfilerCounterDataImageCalculateSize(&calculateSizeParams));

// 为计数器数据镜像分配内存
counterDataImage = (uint8_t*)malloc(calculateSizeParams.counterDataImageSize);
```

### 3. 配置要收集的指标

我们需要指定要收集哪些指标：

```cpp
// 为指标创建配置
CUpti_Profiler_BeginSession_Params beginSessionParams = {
    CUpti_Profiler_BeginSession_Params_STRUCT_SIZE};
beginSessionParams.ctx = NULL;
beginSessionParams.counterDataImageSize = calculateSizeParams.counterDataImageSize;
beginSessionParams.pCounterDataImage = counterDataImage;
beginSessionParams.counterDataScratchBufferSize = calculateSizeParams.counterDataScratchBufferSize;
beginSessionParams.pCounterDataScratchBuffer = counterDataScratchBuffer;
beginSessionParams.range = CUPTI_AutoRange;
beginSessionParams.replayMode = CUPTI_KernelReplay;
beginSessionParams.maxRangesPerPass = 1;
beginSessionParams.maxLaunchesPerPass = 1;

NV_CUPTI_CALL(cuptiProfilerBeginSession(&beginSessionParams));

// 设置要收集的指标（例如 "smsp__warps_launched.avg+"）
CUpti_Profiler_SetConfig_Params setConfigParams = {CUpti_Profiler_SetConfig_Params_STRUCT_SIZE};
setConfigParams.pConfig = metricConfig;
setConfigParams.configSize = configSize;
setConfigParams.passIndex = 0;
NV_CUPTI_CALL(cuptiProfilerSetConfig(&setConfigParams));
```

### 4. 启用分析并运行内核

现在我们启用分析并运行内核：

```cpp
// 启用分析
CUpti_Profiler_EnableProfiling_Params enableProfilingParams = {
    CUpti_Profiler_EnableProfiling_Params_STRUCT_SIZE};
NV_CUPTI_CALL(cuptiProfilerEnableProfiling(&enableProfilingParams));

// 启动第一个内核（VecAdd）
VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
cudaDeviceSynchronize();

// 启动第二个内核（VecSub）
VecSub<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_D, N);
cudaDeviceSynchronize();

// 禁用分析
CUpti_Profiler_DisableProfiling_Params disableProfilingParams = {
    CUpti_Profiler_DisableProfiling_Params_STRUCT_SIZE};
NV_CUPTI_CALL(cuptiProfilerDisableProfiling(&disableProfilingParams));
```

自动范围功能自动检测这些内核启动并为每个内核收集指标。

### 5. 处理结果

分析完成后，我们需要处理收集的数据：

```cpp
// 结束分析会话
CUpti_Profiler_EndSession_Params endSessionParams = {CUpti_Profiler_EndSession_Params_STRUCT_SIZE};
NV_CUPTI_CALL(cuptiProfilerEndSession(&endSessionParams));

// 取消设置分析器配置
CUpti_Profiler_UnsetConfig_Params unsetConfigParams = {CUpti_Profiler_UnsetConfig_Params_STRUCT_SIZE};
NV_CUPTI_CALL(cuptiProfilerUnsetConfig(&unsetConfigParams));

// 获取被分析的范围（内核）数量
CUpti_Profiler_CounterDataImage_GetNumRanges_Params getNumRangesParams = {
    CUpti_Profiler_CounterDataImage_GetNumRanges_Params_STRUCT_SIZE};
getNumRangesParams.pCounterDataImage = counterDataImage;
getNumRangesParams.counterDataImageSize = calculateSizeParams.counterDataImageSize;
NV_CUPTI_CALL(cuptiProfilerCounterDataImageGetNumRanges(&getNumRangesParams));

// 处理每个范围（内核）
for (int rangeIndex = 0; rangeIndex < getNumRangesParams.numRanges; rangeIndex++) {
    // 获取范围名称（内核名称）
    CUpti_Profiler_CounterDataImage_GetRangeName_Params getRangeNameParams = {
        CUpti_Profiler_CounterDataImage_GetRangeName_Params_STRUCT_SIZE};
    getRangeNameParams.pCounterDataImage = counterDataImage;
    getRangeNameParams.counterDataImageSize = calculateSizeParams.counterDataImageSize;
    getRangeNameParams.rangeIndex = rangeIndex;
    NV_CUPTI_CALL(cuptiProfilerCounterDataImageGetRangeName(&getRangeNameParams));
    
    // 为此范围评估指标
    NVPW_MetricsContext_SetCounterData_Params setCounterDataParams = {
        NVPW_MetricsContext_SetCounterData_Params_STRUCT_SIZE};
    setCounterDataParams.pMetricsContext = metricsContext;
    setCounterDataParams.pCounterDataImage = counterDataImage;
    setCounterDataParams.rangeIndex = rangeIndex;
    NVPW_MetricsContext_SetCounterData(&setCounterDataParams);
    
    // 获取指标值
    double metricValue;
    NVPW_MetricsContext_EvaluateToGpuValues_Params evaluateToGpuParams = {
        NVPW_MetricsContext_EvaluateToGpuValues_Params_STRUCT_SIZE};
    evaluateToGpuParams.pMetricsContext = metricsContext;
    evaluateToGpuParams.metricNameBegin = metricName;
    evaluateToGpuParams.metricNameEnd = metricName + strlen(metricName);
    evaluateToGpuParams.pMetricValues = &metricValue;
    NVPW_MetricsContext_EvaluateToGpuValues(&evaluateToGpuParams);
    
    // 打印结果
    printf("Range %d : %s\n  %s: %.1f\n", 
           rangeIndex, getRangeNameParams.pRangeName, metricName, metricValue);
}
```

## 示例内核

示例包含两个简单内核来演示分析多个函数：

```cpp
// 向量加法内核
__global__ void VecAdd(const int* A, const int* B, int* C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

// 向量减法内核
__global__ void VecSub(const int* A, const int* B, int* D, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        D[i] = A[i] - B[i];
}
```

## 运行教程

1. 确保您有所需的依赖项：
   ```bash
   cd ../  # 转到 cupti_samples 根目录
   ./install.sh
   ```

2. 构建示例：
   ```bash
   cd autorange_profiling
   make
   ```

3. 运行示例：
   ```bash
   ./autorange_profiling
   ```

4. 您还可以指定不同的指标：
   ```bash
   ./autorange_profiling smsp__throughput.avg.pct_of_peak_sustained_elapsed
   ```

## 示例输出

运行示例会产生类似于以下的输出：

```
Range 0 : VecAdd(const int *, const int *, int *, int)
  smsp__warps_launched.avg+: 128.0

Range 1 : VecSub(const int *, const int *, int *, int) 
  smsp__warps_launched.avg+: 128.0
```

这显示了：
1. 检测到的内核范围（VecAdd 和 VecSub）
2. 每个内核的指定指标值
3. 自动范围分析如何无缝捕获所有内核启动

## 性能优势

自动范围分析提供了几个关键优势：

### 零代码修改
- 无需在应用程序代码中添加分析调用
- 可以分析第三方库和二进制文件
- 适用于现有应用程序，无需重新编译

### 全面覆盖
- 自动检测所有内核启动
- 不会遗漏任何 GPU 活动
- 为每个独特的内核提供单独的指标

### 最小开销
- 高效的硬件计数器收集
- 自动范围检测的低 CPU 成本
- 可配置的指标选择以控制开销

## 高级配置

### 自定义指标

您可以收集各种性能指标：

```cpp
// 吞吐量指标
const char* throughputMetric = "smsp__throughput.avg.pct_of_peak_sustained_elapsed";

// 内存指标
const char* memoryMetric = "dram__throughput.avg.pct_of_peak_sustained_elapsed";

// 占用率指标
const char* occupancyMetric = "sm__warps_active.avg.pct_of_peak_sustained_active";
```

### 多设备支持

自动范围分析可以扩展到多个 GPU：

```cpp
for (int device = 0; device < deviceCount; device++) {
    cudaSetDevice(device);
    // 为每个设备设置分析
    setupProfilingForDevice(device);
}
```

## 故障排除

### 常见问题

1. **权限错误**：确保以管理员权限运行
2. **CUPTI 初始化失败**：验证 CUDA 驱动程序和工具包版本
3. **指标不可用**：检查 GPU 架构支持
4. **内存不足**：为大型应用程序调整缓冲区大小

### 调试提示

- 使用 `CUPTI_ERROR_PRINT` 宏进行详细错误报告
- 检查 NVIDIA 驱动程序版本兼容性
- 验证 CUDA 上下文在分析期间保持活动状态

## 总结

自动范围分析为 GPU 性能分析提供了一种强大且易于使用的方法。通过自动检测内核边界和收集指标，它简化了性能分析工作流程，同时提供有价值的优化见解。

这种方法特别适用于：
- 快速性能评估
- 第三方代码分析
- 生产系统监控
- 性能回归检测

结合其他 CUPTI 功能，自动范围分析构成了一个全面的 GPU 性能分析工具包的重要组成部分。 