# CUPTI 分析 API 注入教程

> GitHub 仓库和完整教程可在 <https://github.com/eunomia-bpf/cupti-tutorial> 获取。

## 简介

CUPTI 分析 API 注入示例演示如何创建一个分析库，可以注入到任何 CUDA 应用程序中，无需修改源代码。这种强大的技术允许您使用 CUDA 的注入机制从现有应用程序收集详细的性能指标。

## 您将学到什么

- 如何为 CUPTI 分析注入构建共享库
- 理解 CUDA 注入机制（`CUDA_INJECTION64_PATH`）
- 实现基于回调的内核启动和上下文创建分析
- 使用带内核重放和自动范围模式的分析器 API
- 实时收集和分析 GPU 性能指标

## 理解分析注入

分析注入相对于传统分析方法提供几个优势：

1. **无需源代码修改**：无需重新编译即可分析任何 CUDA 应用程序
2. **自动初始化**：CUDA 自动加载和初始化您的分析库
3. **全面覆盖**：拦截目标应用程序中的所有 CUDA 操作
4. **灵活配置**：通过环境变量控制分析行为
5. **生产就绪**：可与发布版本和第三方应用程序一起使用

## 架构概览

注入系统由几个关键组件组成：

1. **注入库**：`libinjection.so` - 主要分析库
2. **CUDA 注入机制**：通过 `CUDA_INJECTION64_PATH` 自动加载
3. **回调系统**：拦截 CUDA API 调用以进行分析设置
4. **分析器 API 集成**：为每个上下文配置指标收集
5. **目标应用程序**：用于测试的 `simple_target` 和 `complex_target`

## 示例应用程序

### simple_target
一个基本的可执行文件，多次调用内核，每次调用的工作量递增。适用于：
- 测试注入机制
- 理解基本分析工作流
- 验证指标收集

### complex_target
一个复杂的示例，具有多种内核启动模式：
- 默认流执行
- 多流并发
- 多设备执行（如果可用）
- 基于线程的并行性

这反映了 `concurrent_profiling` 示例的复杂性，并证明注入处理各种执行模式。

## 构建示例

### 先决条件

确保您有：
- 带 CUPTI 的 CUDA 工具包
- profilerHostUtils 库（从 `cuda/extras/CUPTI/samples/extensions/src/profilerhost_util/` 构建）
- 适当的开发工具（gcc、make）

### 构建过程

```bash
# 设置 CUDA 安装路径
export CUDA_INSTALL_PATH=/path/to/cuda

# 构建所有组件
make CUDA_INSTALL_PATH=/path/to/cuda
```

这会创建三个构建目标：
1. `libinjection.so` - 注入库
2. `simple_target` - 基本测试应用程序
3. `complex_target` - 高级测试应用程序

### 构建组件

#### libinjection.so
核心分析库：
- 为 `cuLaunchKernel` 和上下文创建注册回调
- 为每个上下文创建分析器 API 配置
- 配置内核重放和自动范围模式
- 跟踪内核启动并管理分析通道
- 在通道完成或退出时打印指标

#### 目标应用程序
两个目标应用程序为测试提供不同的复杂度级别：
- `simple_target`：具有不同工作量的顺序内核启动
- `complex_target`：跨多个流和设备的并发执行模式

## 配置选项

### 环境变量

#### INJECTION_KERNEL_COUNT
控制单个分析会话中包含多少个内核：

```bash
export INJECTION_KERNEL_COUNT=20  # 默认值为 10
```

当启动了这么多内核时，会话结束并打印指标，然后开始新会话。

#### INJECTION_METRICS
指定要收集的指标：

```bash
# 默认指标
export INJECTION_METRICS="sm__cycles_elapsed.avg smsp__sass_thread_inst_executed_op_dadd_pred_on.avg smsp__sass_thread_inst_executed_op_dfma_pred_on.avg"

# 自定义指标（空格、逗号或分号分隔）
export INJECTION_METRICS="sm__cycles_elapsed.avg,dram__bytes_read.sum,dram__bytes_write.sum"
```

默认指标关注：
- `sm__cycles_elapsed.avg`：总体执行时间
- `smsp__sass_thread_inst_executed_op_dadd_pred_on.avg`：双精度加法操作
- `smsp__sass_thread_inst_executed_op_dfma_pred_on.avg`：双精度融合乘加操作

## 运行示例

### 基本用法

设置注入路径并运行目标应用程序：

```bash
env CUDA_INJECTION64_PATH=./libinjection.so ./simple_target
```

### 高级配置

```bash
# 配置内核计数和自定义指标
env CUDA_INJECTION64_PATH=./libinjection.so \
    INJECTION_KERNEL_COUNT=15 \
    INJECTION_METRICS="sm__cycles_elapsed.avg,dram__bytes_read.sum" \
    ./complex_target
```

### 测试不同模式

```bash
# 使用简单目标测试
env CUDA_INJECTION64_PATH=./libinjection.so ./simple_target

# 使用复杂多流目标测试
env CUDA_INJECTION64_PATH=./libinjection.so ./complex_target

# 使用您自己的应用程序测试
env CUDA_INJECTION64_PATH=./libinjection.so /path/to/your/cuda/app
```

## 理解输出

### 示例输出格式

```
=== 分析会话 1 ===
上下文：0x7f8b2c000000
会话中的内核：10

指标结果：
sm__cycles_elapsed.avg: 125434.2
smsp__sass_thread_inst_executed_op_dadd_pred_on.avg: 8192.0
smsp__sass_thread_inst_executed_op_dfma_pred_on.avg: 16384.0

=== 分析会话 2 ===
上下文：0x7f8b2c000000
会话中的内核：10
...
```

### 关键信息

1. **会话边界**：每个会话包含可配置数量的内核
2. **上下文信息**：显示指标适用于哪个 CUDA 上下文
3. **指标值**：指定指标的收集性能数据
4. **自动轮换**：当达到内核计数时自动开始新会话

## 代码架构

### 注入入口点

```cpp
extern "C" void InitializeInjection(void)
{
    // 当 CUDA 加载注入库时自动调用
    // 注册回调并设置分析基础设施
}
```

### 回调注册

库为以下内容注册回调：

1. **上下文创建**：为新上下文设置分析器 API 配置
2. **内核启动**：跟踪启动计数并管理会话边界
3. **cuLaunchKernel**：处理大多数内核启动场景

### 分析会话管理

```cpp
class ProfilingSession {
private:
    std::map<CUcontext, ProfilerState> contextStates;
    int kernelCount;
    int maxKernelsPerSession;

public:
    void handleKernelLaunch(CUcontext ctx) {
        kernelCount++;
        if (kernelCount >= maxKernelsPerSession) {
            endCurrentSession();
            startNewSession();
        }
    }
    
    void endCurrentSession() {
        // 收集并打印指标
        for (auto& [ctx, state] : contextStates) {
            collectAndPrintMetrics(ctx, state);
        }
        kernelCount = 0;
    }
};
```

## 实际应用

### 生产分析

```bash
# 分析生产应用程序
env CUDA_INJECTION64_PATH=./libinjection.so \
    INJECTION_KERNEL_COUNT=50 \
    INJECTION_METRICS="sm__cycles_elapsed.avg,dram__throughput.avg" \
    production_app
```

### 持续集成

```bash
#!/bin/bash
# CI 脚本中的性能回归检测

export CUDA_INJECTION64_PATH=./libinjection.so
export INJECTION_METRICS="sm__cycles_elapsed.avg"

# 运行基准测试
./benchmark_app > current_results.txt

# 与基线比较
if python compare_performance.py baseline.txt current_results.txt; then
    echo "性能测试通过"
else
    echo "检测到性能回归"
    exit 1
fi
```

### 第三方库分析

```bash
# 分析使用第三方 CUDA 库的应用程序
env CUDA_INJECTION64_PATH=./libinjection.so \
    LD_LIBRARY_PATH=/path/to/vendor/libs:$LD_LIBRARY_PATH \
    vendor_application
```

## 高级特性

### 自定义指标过滤

```cpp
bool shouldIncludeMetric(const std::string& metricName) {
    // 只包含特定模式的指标
    return metricName.find("cycles") != std::string::npos ||
           metricName.find("throughput") != std::string::npos;
}
```

### 多 GPU 支持

```cpp
void handleMultiGPU() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    for (int i = 0; i < deviceCount; i++) {
        cudaSetDevice(i);
        // 为每个设备设置分析
        setupProfilingForDevice(i);
    }
}
```

### 实时指标导出

```cpp
class MetricsExporter {
public:
    void exportToInfluxDB(const std::map<std::string, double>& metrics) {
        // 将指标发送到时间序列数据库
        for (const auto& [name, value] : metrics) {
            influxClient.writePoint(name, value, timestamp);
        }
    }
    
    void exportToPrometheus(const std::map<std::string, double>& metrics) {
        // 导出到 Prometheus 监控
        prometheusRegistry.update(metrics);
    }
};
```

## 故障排除

### 常见问题

1. **库加载失败**：
   ```bash
   # 检查库依赖
   ldd libinjection.so
   
   # 验证路径
   echo $CUDA_INJECTION64_PATH
   ```

2. **权限问题**：
   ```bash
   # 某些指标可能需要管理员权限
   sudo env CUDA_INJECTION64_PATH=./libinjection.so ./app
   ```

3. **内存不足**：
   ```bash
   # 减少每会话的内核数
   export INJECTION_KERNEL_COUNT=5
   ```

### 调试模式

```bash
# 启用详细日志
export CUPTI_DEBUG=1
export INJECTION_DEBUG=1

# 运行并检查日志
env CUDA_INJECTION64_PATH=./libinjection.so ./app 2>&1 | tee debug.log
```

## 最佳实践

1. **从小开始**：首先使用简单的目标应用程序验证设置
2. **选择性指标**：只收集分析目标所需的指标
3. **会话大小**：平衡数据粒度与内存使用
4. **环境隔离**：在受控环境中测试注入
5. **性能影响**：监控分析开销并相应调整

分析注入为分析 CUDA 应用程序提供了强大而灵活的方法。通过利用 CUDA 的内置注入机制，您可以获得任何 CUDA 应用程序的深入性能洞察，而无需访问其源代码。 