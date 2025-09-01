# CUPTI 跟踪注入教程

> 完整的 GitHub 仓库和教程请访问 <https://github.com/eunomia-bpf/cupti-tutorial>。

## 简介

CUPTI 跟踪注入示例演示如何创建一个轻量级跟踪库，可以自动注入到任何 CUDA 应用程序中。这种方法无需修改源代码即可实现全面的活动跟踪，非常适合分析现有应用程序、第三方库或生产工作负载。

## 学习内容

- 如何构建用于自动 CUDA 活动跟踪的注入库
- 理解 CUDA 注入机制以实现无缝集成
- 实现 NVTX 活动记录以增强时间线可视化
- 跨平台注入技术（Linux 和 Windows）
- 在无需修改应用程序的情况下收集全面的跟踪数据

## 理解跟踪注入

跟踪注入为 CUDA 性能分析提供了几个关键优势：

1. **零应用程序修改**：无需重新编译即可分析任何 CUDA 应用程序
2. **自动激活**：CUDA 运行时自动加载和初始化跟踪
3. **全面覆盖**：捕获所有 CUDA 操作和活动
4. **NVTX 集成**：记录用户定义的范围和标记
5. **时间线可视化**：生成适用于时间线分析工具的数据

## 架构概述

跟踪注入系统包含：

1. **注入库**：`libcupti_trace_injection.so`（Linux）或 `libcupti_trace_injection.dll`（Windows）
2. **CUDA 注入钩子**：通过 `CUDA_INJECTION64_PATH` 自动加载
3. **NVTX 集成**：通过 `NVTX_INJECTION64_PATH` 可选的 NVTX 活动记录
4. **活动收集**：全面的 CUDA API 和 GPU 活动跟踪
5. **输出生成**：用于分析工具的结构化跟踪数据

## 主要功能

### 自动初始化
- 无需修改源代码
- 适用于任何 CUDA 应用程序
- 支持运行时和驱动 API
- 处理复杂的多线程应用程序

### 全面的活动跟踪
- CUDA 运行时 API 调用
- CUDA 驱动 API 调用
- 内核执行活动
- 内存传输操作
- 上下文和流管理

### NVTX 支持
- 用户定义的范围记录
- 自定义标记和注释
- 增强的时间线可视化
- 应用程序阶段关联

## 构建示例

### 先决条件

确保您具备：
- 带有 CUPTI 的 CUDA Toolkit
- 开发工具（gcc/Visual Studio）
- 对于 Windows：Microsoft Detours 库

### Linux 构建过程

1. 导航到示例目录：
   ```bash
   cd cupti_trace_injection
   ```

2. 使用提供的 Makefile 构建：
   ```bash
   make
   ```
   
   这会创建 `libcupti_trace_injection.so`。

### Windows 构建过程

对于 Windows 构建，您需要 Microsoft Detours 库：

1. **下载并构建 Detours**：
   ```cmd
   # 从 https://github.com/microsoft/Detours 下载
   # 解压到一个文件夹
   cd Detours
   set DETOURS_TARGET_PROCESSOR=X64
   "Program Files (x86)\Microsoft Visual Studio\<version>\Professional\VC\Auxiliary\Build\vcvarsall.bat" x64
   NMAKE
   ```

2. **复制所需文件**：
   ```cmd
   copy detours.h <cupti_trace_injection_folder>
   copy detours.lib <cupti_trace_injection_folder>
   ```

3. **构建示例**：
   ```cmd
   nmake
   ```
   
   这会创建 `libcupti_trace_injection.dll`。

## 运行示例

### Linux 使用方法

1. **设置注入环境**：
   ```bash
   export CUDA_INJECTION64_PATH=/root/yunwei37/cupti-tutorial/cupti_trace_injection/libcupti_trace_injection.so
   export NVTX_INJECTION64_PATH=/usr/local/cuda-13.0/extras/CUPTI/lib64/libcupti.so
   export LD_LIBRARY_PATH=/usr/local/cuda-13.0/extras/CUPTI/lib64:$LD_LIBRARY_PATH
   ```

2. **运行您的 CUDA 应用程序**：
   ```bash
   ./your_cuda_application
   ```

### Windows 使用方法

1. **设置注入环境**：
   ```cmd
   set CUDA_INJECTION64_PATH=C:\full\path\to\libcupti_trace_injection.dll
   set NVTX_INJECTION64_PATH=C:\full\path\to\cupti.dll
   ```

2. **运行您的 CUDA 应用程序**：
   ```cmd
   your_cuda_application.exe
   ```

### 环境变量

#### CUDA_INJECTION64_PATH
指定注入库的路径。设置后，CUDA 自动：
- 在初始化时加载共享库
- 调用 `InitializeInjection()` 函数
- 为所有后续 CUDA 操作启用跟踪

#### NVTX_INJECTION64_PATH
CUPTI 库的可选路径，用于 NVTX 活动记录：
- 启用用户定义的范围收集
- 记录自定义标记和注释
- 提供增强的时间线上下文

## 理解输出

### 跟踪数据格式

注入库生成包含以下内容的全面跟踪数据：

```
CUDA 运行时 API 调用：
  cudaMalloc: Start=1234567890, End=1234567925, Duration=35μs
  cudaMemcpy: Start=1234567950, End=1234568100, Duration=150μs
  cudaLaunchKernel: Start=1234568150, End=1234568175, Duration=25μs

GPU 活动：
  Kernel: vectorAdd, Start=1234568200, End=1234568500, Duration=300μs
  MemcpyHtoD: Size=4096KB, Start=1234567950, End=1234568100, Duration=150μs
  MemcpyDtoH: Size=4096KB, Start=1234568600, End=1234568750, Duration=150μs

NVTX 范围：
  Range: "数据准备", Start=1234567800, End=1234568150, Duration=350μs
  Range: "计算", Start=1234568150, End=1234568550, Duration=400μs
  Range: "结果验证", Start=1234568600, End=1234568900, Duration=300μs
```

### 关键指标

1. **API 调用时序**：CUDA 运行时和驱动 API 调用的持续时间
2. **GPU 活动时间线**：实际内核执行和内存传输时间
3. **内存使用**：分配大小和传输模式
4. **并发分析**：重叠操作和流利用率
5. **用户定义上下文**：提供应用程序语义的 NVTX 范围

## 实际应用

### 性能分析

使用跟踪注入进行：
- **瓶颈识别**：找到应用程序中最慢的操作
- **并发分析**：了解操作重叠的程度
- **内存带宽利用率**：分析数据传输效率
- **API 开销测量**：量化 CUDA API 调用成本

### 时间线可视化

跟踪数据可以导入到：
- **NVIDIA Nsight Systems**：全面的时间线分析
- **Chrome Tracing**：基于 Web 的可视化
- **自定义分析工具**：程序化跟踪处理
- **性能比较工具**：优化前后的分析

### 生产监控

在生产环境中部署以：
- 随时间监控应用程序性能
- 检测性能回归
- 分析真实世界的工作负载模式
- 生成自动化性能报告

## 高级用法

### 自定义活动过滤

修改注入库以专注于特定活动：

```cpp
// 过滤特定 API 调用
bool shouldTraceAPI(const char* apiName) {
    return (strstr(apiName, "Launch") != nullptr ||
            strstr(apiName, "Memcpy") != nullptr);
}

// 过滤内核活动
bool shouldTraceKernel(const char* kernelName) {
    return !strstr(kernelName, "internal_");
}
```

### 增强的 NVTX 集成

利用 NVTX 获得更好的应用程序上下文：

```cpp
// 在您的应用程序中（可选，但增强跟踪）
nvtxRangePush("关键段");
// ... CUDA 操作 ...
nvtxRangePop();

nvtxMark("检查点 A");
```

### 多 GPU 分析

注入库自动处理：
- 多个 GPU 上下文
- 跨设备内存传输
- 点对点通信
- 设备特定的活动时间线

## 输出格式和分析

### 原始数据处理

跟踪数据可以进行后处理以：

```cpp
// 解析跟踪文件的示例
class TraceParser {
public:
    struct TraceEvent {
        std::string name;
        uint64_t start;
        uint64_t end;
        std::string category;
        std::map<std::string, std::string> args;
    };
    
    std::vector<TraceEvent> parseTraceFile(const std::string& filename) {
        // 实现跟踪文件解析
        std::vector<TraceEvent> events;
        // ... 解析逻辑 ...
        return events;
    }
    
    void generateReport(const std::vector<TraceEvent>& events) {
        // 生成性能报告
        double totalKernelTime = 0;
        double totalMemcpyTime = 0;
        
        for (const auto& event : events) {
            double duration = (event.end - event.start) / 1000.0; // 转换为微秒
            
            if (event.category == "kernel") {
                totalKernelTime += duration;
            } else if (event.category == "memcpy") {
                totalMemcpyTime += duration;
            }
        }
        
        printf("总内核时间：%.2f μs\n", totalKernelTime);
        printf("总内存传输时间：%.2f μs\n", totalMemcpyTime);
        printf("内核/内存比率：%.2f\n", totalKernelTime / totalMemcpyTime);
    }
};
```

### 自动化分析流水线

```cpp
// 自动化性能分析流水线
class PerformanceAnalyzer {
private:
    std::string traceDirectory;
    std::vector<std::string> baselineFiles;
    
public:
    void analyzePerformanceRegression() {
        auto currentTrace = loadTraceFile("current_run.json");
        auto baselineTrace = loadTraceFile("baseline.json");
        
        auto currentMetrics = extractMetrics(currentTrace);
        auto baselineMetrics = extractMetrics(baselineTrace);
        
        compareMetrics(currentMetrics, baselineMetrics);
    }
    
    struct PerformanceMetrics {
        double avgKernelDuration;
        double memoryBandwidth;
        double apiOverhead;
        int concurrentKernels;
    };
    
    PerformanceMetrics extractMetrics(const std::vector<TraceEvent>& events) {
        PerformanceMetrics metrics = {};
        // ... 指标提取逻辑 ...
        return metrics;
    }
    
    void compareMetrics(const PerformanceMetrics& current, 
                       const PerformanceMetrics& baseline) {
        printf("性能比较报告：\n");
        
        double kernelRegression = (current.avgKernelDuration - baseline.avgKernelDuration) 
                                 / baseline.avgKernelDuration * 100;
        
        if (kernelRegression > 5.0) {
            printf("警告：内核性能回归 %.1f%%\n", kernelRegression);
        }
        
        double bandwidthChange = (current.memoryBandwidth - baseline.memoryBandwidth) 
                                / baseline.memoryBandwidth * 100;
        
        printf("内存带宽变化：%.1f%%\n", bandwidthChange);
    }
};
```

## 故障排除

### 常见问题

1. **注入库加载失败**：
   - 检查环境变量路径是否正确
   - 验证库文件权限
   - 确保所有依赖项都可用

2. **符号解析错误**：
   - 确保 CUDA 版本兼容性
   - 检查库导出符号
   - 验证链接器配置

3. **性能影响过大**：
   - 调整跟踪粒度
   - 使用选择性事件过滤
   - 考虑采样而非完整跟踪

### 调试技巧

```cpp
// 启用详细日志记录
#ifdef DEBUG_INJECTION
#define DEBUG_LOG(fmt, ...) \
    printf("[INJECTION DEBUG] " fmt "\n", ##__VA_ARGS__)
#else
#define DEBUG_LOG(fmt, ...)
#endif

// 在注入库中使用
void InitializeInjection() {
    DEBUG_LOG("开始初始化注入库");
    
    // 验证 CUPTI 版本
    uint32_t cuptiVersion;
    cuptiGetVersion(&cuptiVersion);
    DEBUG_LOG("CUPTI 版本：%u", cuptiVersion);
    
    // 设置活动缓冲区
    setupActivityBuffers();
    DEBUG_LOG("活动缓冲区设置完成");
    
    // 注册回调
    registerCallbacks();
    DEBUG_LOG("回调注册完成");
}
```

## 最佳实践

### 性能影响最小化

1. **选择性跟踪**：只跟踪感兴趣的活动类型
2. **缓冲区优化**：调整缓冲区大小以平衡内存使用和性能
3. **异步处理**：使用后台线程处理跟踪数据

### 生产部署

1. **配置管理**：使用配置文件控制跟踪行为
2. **数据轮转**：实现跟踪文件轮转以管理存储
3. **监控集成**：与现有监控系统集成

### 数据安全

1. **敏感数据过滤**：避免记录敏感的内核参数
2. **访问控制**：限制对跟踪文件的访问
3. **数据清理**：定期清理旧的跟踪文件

这个跟踪注入教程为在任何 CUDA 应用程序中实现透明、全面的性能监控提供了强大的基础，无需任何源代码修改。 