# CUPTI 用户范围分析教程

> GitHub 仓库和完整教程可在 <https://github.com/eunomia-bpf/cupti-tutorial> 获取。

## 简介

在分析 CUDA 应用程序时，您通常需要专注于代码的特定部分，而不是整个内核或完整的应用程序。CUPTI 的用户范围分析功能允许您在代码中定义自定义范围，并仅在这些范围内收集性能指标。这为您提供了对应用程序哪些部分被分析的精确控制，使得在复杂应用程序中更容易识别和优化性能瓶颈。本教程演示如何使用 CUPTI 的分析器 API 定义自定义范围并在其中收集性能指标。

## 您将学到什么

- 如何在 CUDA 代码中定义和仪器化用户指定的范围
- 为用户范围分析设置 CUPTI 分析器 API
- 在您定义的范围内收集 GPU 性能指标
- 处理和分析每个范围收集的指标
- 比较应用程序不同部分的性能

## 理解用户范围分析

与自动分析单个 CUDA API 调用或内核不同，用户范围分析让您：

1. 定义可能包含多个 CUDA 操作的代码逻辑段
2. 为这些段提供有意义的名称以便于分析
3. 将分析资源集中在应用程序最重要的部分
4. 比较不同算法方法的性能指标

这种方法在以下情况下特别有用：
- 您的应用程序具有不同性能特征的不同阶段
- 您想比较同一算法的不同实现
- 您需要将特定的操作序列作为单个单元进行分析
- 您正在优化大型应用程序的特定部分

## 代码演练

### 1. 设置 CUPTI 分析器

首先，我们需要初始化 CUPTI 分析器 API 并为用户范围分析配置它：

```cpp
CUpti_Profiler_Initialize_Params initializeParams = {CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
CUPTI_API_CALL(cuptiProfilerInitialize(&initializeParams));

// 获取当前设备的芯片名称
CUpti_Device_GetChipName_Params getChipNameParams = {CUpti_Device_GetChipName_Params_STRUCT_SIZE};
getChipNameParams.deviceIndex = 0;
CUPTI_API_CALL(cuptiDeviceGetChipName(&getChipNameParams));
const char *chipName = getChipNameParams.pChipName;

// 创建指标配置
NVPW_InitializeHost_Params initializeHostParams = {NVPW_InitializeHost_Params_STRUCT_SIZE};
NVPW_API_CALL(NVPW_InitializeHost(&initializeHostParams));

// 设置要收集的指标
const char *metricNames[] = {"smsp__warps_launched.avg"};
struct MetricNameList metricList;
metricList.numMetrics = 1;
metricList.metricNames = metricNames;

// 创建计数器数据映像和配置
CUpti_Profiler_CounterDataImageOptions counterDataImageOptions = {
    CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE};
counterDataImageOptions.pChipName = chipName;
counterDataImageOptions.counterDataImageSize = 0;
CUPTI_API_CALL(cuptiProfilerCounterDataImageCalculateSize(&counterDataImageOptions));
counterDataImage = (uint8_t *)malloc(counterDataImageOptions.counterDataImageSize);
counterDataImageOptions.pCounterDataImage = counterDataImage;
CUPTI_API_CALL(cuptiProfilerCounterDataImageInitialize(&counterDataImageOptions));

// 为用户范围分析配置分析器
CUpti_Profiler_UserRange_Config_Params configParams = {
    CUpti_Profiler_UserRange_Config_Params_STRUCT_SIZE};
configParams.pCounterDataPrefixImage = counterDataImage;
configParams.counterDataPrefixImageSize = counterDataImageOptions.counterDataImageSize;
configParams.maxRangesPerPass = 1;
configParams.maxLaunchesPerPass = 1;
CUPTI_API_CALL(cuptiProfilerUserRangeConfigureScratchBuffer(&configParams));
```

这段代码：
1. 初始化 CUPTI 分析器 API
2. 获取当前设备的芯片名称
3. 设置我们想要收集的指标（在这种情况下是 "smsp__warps_launched.avg"）
4. 创建并配置计数器数据映像
5. 专门为用户范围分析配置分析器

### 2. 在代码中定义用户范围

接下来，我们围绕想要分析的代码段定义范围：

```cpp
void profileVectorOperations(int *d_A, int *d_B, int *d_C, int numElements)
{
    // 定义范围名称
    const char *rangeName = "Vector Add-Subtract";
    
    // 开始用户范围
    CUpti_Profiler_BeginRange_Params beginRangeParams = {CUpti_Profiler_BeginRange_Params_STRUCT_SIZE};
    beginRangeParams.pRangeName = rangeName;
    CUPTI_API_CALL(cuptiProfilerBeginRange(&beginRangeParams));
    
    // 在范围内执行操作
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("启动内核：块数 %d，每块线程数 %d\n", blocksPerGrid, threadsPerBlock);
    
    // 第一个向量操作：加法
    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    
    // 第二个向量操作：减法
    VecSub<<<blocksPerGrid, threadsPerBlock>>>(d_C, d_B, d_C, numElements);
    
    // 结束用户范围
    CUpti_Profiler_EndRange_Params endRangeParams = {CUpti_Profiler_EndRange_Params_STRUCT_SIZE};
    CUPTI_API_CALL(cuptiProfilerEndRange(&endRangeParams));
}
```

这个函数：
1. 为范围定义有意义的名称（"Vector Add-Subtract"）
2. 使用 `cuptiProfilerBeginRange` 开始用户范围
3. 在范围内执行多个 GPU 操作（两个内核启动）
4. 使用 `cuptiProfilerEndRange` 结束范围

### 3. 启动和停止分析器会话

我们需要在范围之前启动分析器，之后停止它：

```cpp
int main(int argc, char *argv[])
{
    // 初始化 CUDA 和 CUPTI
    initializeCuda();
    initializeProfiler();
    
    // 分配和初始化数据
    int *h_A, *h_B, *h_C;
    int *d_A, *d_B, *d_C;
    // ... 分配和初始化代码 ...
    
    // 开始分析器会话
    CUpti_Profiler_BeginSession_Params beginSessionParams = {CUpti_Profiler_BeginSession_Params_STRUCT_SIZE};
    beginSessionParams.counterDataImageSize = counterDataImageOptions.counterDataImageSize;
    beginSessionParams.pCounterDataImage = counterDataImage;
    beginSessionParams.maxRangesPerPass = 1;
    beginSessionParams.maxLaunchesPerPass = 1;
    CUPTI_API_CALL(cuptiProfilerBeginSession(&beginSessionParams));
    
    // 启用分析
    CUpti_Profiler_EnableProfiling_Params enableProfilingParams = {
        CUpti_Profiler_EnableProfiling_Params_STRUCT_SIZE};
    CUPTI_API_CALL(cuptiProfilerEnableProfiling(&enableProfilingParams));
    
    // 执行包含用户范围的分析代码
    profileVectorOperations(d_A, d_B, d_C, numElements);
    
    // 禁用分析
    CUpti_Profiler_DisableProfiling_Params disableProfilingParams = {
        CUpti_Profiler_DisableProfiling_Params_STRUCT_SIZE};
    CUPTI_API_CALL(cuptiProfilerDisableProfiling(&disableProfilingParams));
    
    // 结束分析器会话
    CUpti_Profiler_EndSession_Params endSessionParams = {CUpti_Profiler_EndSession_Params_STRUCT_SIZE};
    CUPTI_API_CALL(cuptiProfilerEndSession(&endSessionParams));
    
    // 处理结果
    processCounterData();
    
    // 清理
    // ... 清理代码 ...
    
    return 0;
}
```

这段代码：
1. 初始化 CUDA 和 CUPTI 分析器
2. 使用 `cuptiProfilerBeginSession` 开始分析器会话
3. 使用 `cuptiProfilerEnableProfiling` 启用分析
4. 执行包含我们用户范围的代码
5. 禁用分析并结束会话
6. 处理收集的数据

### 4. 处理收集的指标

分析后，我们需要处理计数器数据以计算指标：

```cpp
void processCounterData()
{
    // 获取当前设备的芯片名称
    CUpti_Device_GetChipName_Params getChipNameParams = {CUpti_Device_GetChipName_Params_STRUCT_SIZE};
    getChipNameParams.deviceIndex = 0;
    CUPTI_API_CALL(cuptiDeviceGetChipName(&getChipNameParams));
    const char *chipName = getChipNameParams.pChipName;
    
    // 初始化评估上下文
    NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params scratchBufferSizeParams = {
        NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params_STRUCT_SIZE};
    scratchBufferSizeParams.pChipName = chipName;
    scratchBufferSizeParams.pCounterDataImage = counterDataImage;
    NVPW_API_CALL(NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize(&scratchBufferSizeParams));
    
    // 分配和初始化评估器
    void *scratchBuffer = malloc(scratchBufferSizeParams.scratchBufferSize);
    NVPW_CUDA_MetricsEvaluator_Initialize_Params initializeParams = {
        NVPW_CUDA_MetricsEvaluator_Initialize_Params_STRUCT_SIZE};
    initializeParams.scratchBufferSize = scratchBufferSizeParams.scratchBufferSize;
    initializeParams.pScratchBuffer = scratchBuffer;
    initializeParams.pChipName = chipName;
    initializeParams.pCounterDataImage = counterDataImage;
    NVPW_API_CALL(NVPW_CUDA_MetricsEvaluator_Initialize(&initializeParams));
    
    // 评估指标
    NVPW_MetricsEvaluator_EvaluateToGpuValues_Params evaluateParams = {
        NVPW_MetricsEvaluator_EvaluateToGpuValues_Params_STRUCT_SIZE};
    evaluateParams.pMetricNames = metricNames;
    evaluateParams.numMetrics = 1;
    evaluateParams.pCounterDataImage = counterDataImage;
    NVPW_API_CALL(NVPW_MetricsEvaluator_EvaluateToGpuValues(&evaluateParams));
    
    // 打印结果
    printf("用户范围分析结果：\n");
    for (size_t i = 0; i < evaluateParams.numMetrics; i++) {
        printf("  %s: %.2f\n", metricNames[i], evaluateParams.pGpuValues[i]);
    }
    
    // 清理
    free(scratchBuffer);
}
```

这段代码：
1. 获取设备芯片名称以进行指标评估
2. 计算并分配评估器的暂存缓冲区
3. 初始化指标评估器
4. 评估我们指定的指标
5. 打印结果并清理资源

## 高级用例

### 多范围分析

```cpp
class MultiRangeProfiler {
private:
    std::vector<std::string> rangeNames;
    std::map<std::string, std::map<std::string, double>> rangeMetrics;

public:
    void profileAlgorithmComparison() {
        // 定义要分析的算法变体
        std::vector<std::string> algorithms = {
            "Naive Implementation",
            "Shared Memory Optimized", 
            "Coalesced Access",
            "Tiled Algorithm"
        };
        
        for (const auto& algorithm : algorithms) {
            // 开始范围
            CUpti_Profiler_BeginRange_Params beginParams = {
                CUpti_Profiler_BeginRange_Params_STRUCT_SIZE};
            beginParams.pRangeName = algorithm.c_str();
            CUPTI_API_CALL(cuptiProfilerBeginRange(&beginParams));
            
            // 执行特定的算法实现
            if (algorithm == "Naive Implementation") {
                runNaiveKernel();
            } else if (algorithm == "Shared Memory Optimized") {
                runSharedMemoryKernel();
            } else if (algorithm == "Coalesced Access") {
                runCoalescedKernel();
            } else if (algorithm == "Tiled Algorithm") {
                runTiledKernel();
            }
            
            // 结束范围
            CUpti_Profiler_EndRange_Params endParams = {
                CUpti_Profiler_EndRange_Params_STRUCT_SIZE};
            CUPTI_API_CALL(cuptiProfilerEndRange(&endParams));
            
            // 收集此范围的指标
            auto metrics = collectRangeMetrics();
            rangeMetrics[algorithm] = metrics;
        }
        
        // 比较结果
        compareAlgorithmPerformance();
    }
    
private:
    void compareAlgorithmPerformance() {
        printf("=== 算法性能比较 ===\n");
        
        // 找到最佳性能作为基线
        double bestTime = std::numeric_limits<double>::max();
        std::string bestAlgorithm;
        
        for (const auto& [name, metrics] : rangeMetrics) {
            double execTime = metrics.at("sm__cycles_elapsed.avg");
            if (execTime < bestTime) {
                bestTime = execTime;
                bestAlgorithm = name;
            }
        }
        
        // 打印相对性能
        for (const auto& [name, metrics] : rangeMetrics) {
            double execTime = metrics.at("sm__cycles_elapsed.avg");
            double speedup = bestTime / execTime;
            
            printf("%s:\n", name.c_str());
            printf("  执行时间: %.2f 周期\n", execTime);
            printf("  相对于最佳的速度提升: %.2fx", speedup);
            if (name == bestAlgorithm) {
                printf(" (最佳)");
            }
            printf("\n\n");
        }
    }
};
```

### 性能回归检测

```cpp
class PerformanceRegressionDetector {
private:
    std::map<std::string, double> baselineMetrics;
    double regressionThreshold = 0.95; // 5% 性能下降阈值

public:
    void setBaseline(const std::string& rangeName) {
        // 运行基线测试
        profileRange(rangeName);
        auto metrics = collectRangeMetrics();
        baselineMetrics[rangeName] = metrics["sm__cycles_elapsed.avg"];
        
        printf("为 %s 设置基线: %.2f 周期\n", 
               rangeName.c_str(), baselineMetrics[rangeName]);
    }
    
    bool checkForRegression(const std::string& rangeName) {
        // 运行当前版本
        profileRange(rangeName);
        auto metrics = collectRangeMetrics();
        double currentPerformance = metrics["sm__cycles_elapsed.avg"];
        
        // 比较与基线（较低的周期 = 更好的性能）
        double baseline = baselineMetrics[rangeName];
        double performanceRatio = baseline / currentPerformance;
        
        printf("性能检查 %s:\n", rangeName.c_str());
        printf("  基线: %.2f 周期\n", baseline);
        printf("  当前: %.2f 周期\n", currentPerformance);
        printf("  比率: %.3f", performanceRatio);
        
        if (performanceRatio < regressionThreshold) {
            printf(" - 检测到回归！\n");
            return true;
        } else {
            printf(" - 通过\n");
            return false;
        }
    }
    
private:
    void profileRange(const std::string& rangeName) {
        CUpti_Profiler_BeginRange_Params beginParams = {
            CUpti_Profiler_BeginRange_Params_STRUCT_SIZE};
        beginParams.pRangeName = rangeName.c_str();
        CUPTI_API_CALL(cuptiProfilerBeginRange(&beginParams));
        
        // 执行要测试的代码
        executeTestCode(rangeName);
        
        CUpti_Profiler_EndRange_Params endParams = {
            CUpti_Profiler_EndRange_Params_STRUCT_SIZE};
        CUPTI_API_CALL(cuptiProfilerEndRange(&endParams));
    }
};
```

### 自适应优化

```cpp
class AdaptiveOptimizer {
public:
    void optimizeKernelLaunchParameters() {
        std::vector<int> blockSizes = {128, 256, 512, 1024};
        std::map<int, double> performanceResults;
        
        for (int blockSize : blockSizes) {
            std::string rangeName = "BlockSize_" + std::to_string(blockSize);
            
            // 分析此配置
            CUpti_Profiler_BeginRange_Params beginParams = {
                CUpti_Profiler_BeginRange_Params_STRUCT_SIZE};
            beginParams.pRangeName = rangeName.c_str();
            CUPTI_API_CALL(cuptiProfilerBeginRange(&beginParams));
            
            // 使用当前块大小运行内核
            runKernelWithBlockSize(blockSize);
            
            CUpti_Profiler_EndRange_Params endParams = {
                CUpti_Profiler_EndRange_Params_STRUCT_SIZE};
            CUPTI_API_CALL(cuptiProfilerEndRange(&endParams));
            
            // 收集性能数据
            auto metrics = collectRangeMetrics();
            performanceResults[blockSize] = metrics["sm__cycles_elapsed.avg"];
        }
        
        // 找到最佳配置
        int bestBlockSize = findOptimalBlockSize(performanceResults);
        printf("最佳块大小: %d\n", bestBlockSize);
        
        // 应用最佳配置
        applyOptimalConfiguration(bestBlockSize);
    }

private:
    int findOptimalBlockSize(const std::map<int, double>& results) {
        int bestSize = 0;
        double bestTime = std::numeric_limits<double>::max();
        
        for (const auto& [size, time] : results) {
            printf("块大小 %d: %.2f 周期\n", size, time);
            if (time < bestTime) {
                bestTime = time;
                bestSize = size;
            }
        }
        return bestSize;
    }
};
```

## 最佳实践

### 范围命名约定

```cpp
// 使用分层命名以便于分析
class RangeNamingConventions {
public:
    static std::string createRangeName(const std::string& module,
                                      const std::string& function,
                                      const std::string& variant = "") {
        std::string name = module + "::" + function;
        if (!variant.empty()) {
            name += "_" + variant;
        }
        return name;
    }
    
    // 示例用法
    void demonstrateNaming() {
        // 模块级范围
        profileRange("LinearAlgebra::MatrixMultiply");
        profileRange("ImageProcessing::Convolution");
        
        // 变体比较
        profileRange("GEMM::Naive");
        profileRange("GEMM::Tiled");
        profileRange("GEMM::SharedMemory");
        
        // 参数化测试
        profileRange("Sort::QuickSort_Size1M");
        profileRange("Sort::QuickSort_Size10M");
    }
};
```

### 错误处理和验证

```cpp
class ProfilerValidator {
public:
    static bool validateProfilerState() {
        // 检查分析器是否正确初始化
        CUpti_Profiler_GetInitializationStatus_Params statusParams = {
            CUpti_Profiler_GetInitializationStatus_Params_STRUCT_SIZE};
        
        CUptiResult result = cuptiProfilerGetInitializationStatus(&statusParams);
        if (result != CUPTI_SUCCESS) {
            printf("分析器未正确初始化\n");
            return false;
        }
        
        return true;
    }
    
    static void handleProfilerError(CUptiResult result, const char* operation) {
        if (result != CUPTI_SUCCESS) {
            const char* errorString;
            cuptiGetResultString(result, &errorString);
            printf("分析器错误在 %s: %s\n", operation, errorString);
            exit(1);
        }
    }
};
```

## 故障排除

### 常见问题

1. **范围嵌套**：确保范围正确嵌套，避免重叠
2. **指标兼容性**：验证指标在目标 GPU 架构上可用
3. **内存限制**：监控分析器内存使用情况
4. **性能开销**：平衡分析粒度与开销

### 调试技巧

```cpp
void debugProfilingSession() {
    // 启用详细的 CUPTI 日志记录
    setenv("CUPTI_DEBUG", "1", 1);
    
    // 检查可用指标
    listAvailableMetrics();
    
    // 验证范围边界
    validateRangeBoundaries();
    
    // 监控内存使用
    checkMemoryUsage();
}
```

用户范围分析为精确的 CUDA 性能分析提供了强大的工具。通过仔细定义有意义的范围并收集相关指标，您可以获得对应用程序性能特征的深入洞察，并做出数据驱动的优化决策。 