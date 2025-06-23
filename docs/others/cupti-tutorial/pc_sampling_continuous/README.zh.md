# CUPTI 程序计数器 (PC) 连续采样教程

> GitHub 仓库和完整教程可在 <https://github.com/eunomia-bpf/cupti-tutorial> 获取。

## 简介

程序计数器 (PC) 采样是一种强大的分析技术，允许您在汇编指令级别了解 CUDA 内核的执行时间分布。本教程演示如何实现连续 PC 采样，可以监控任何 CUDA 应用程序，无需修改源代码。

## 您将学到什么

- 如何构建用于 PC 采样注入的动态库
- 连续分析 CUDA 应用程序的技术
- 理解 PC 采样数据和停顿原因
- 跨平台实现（Linux 和 Windows）
- 将分析库与现有应用程序配合使用

## 理解 PC 连续采样

PC 连续采样与其他分析方法的不同之处在于：

1. **在汇编级别操作**：提供对实际 GPU 指令执行的洞察
2. **无需源代码修改**：可以分析任何 CUDA 应用程序
3. **通过库注入工作**：使用动态加载拦截 CUDA 调用
4. **提供停顿原因分析**：显示线程束为什么没有取得进展
5. **支持实时监控**：可以在执行期间观察性能

## 架构概览

连续 PC 采样系统包含：

1. **动态库**：`libpc_sampling_continuous.so`（Linux）或 `pc_sampling_continuous.lib`（Windows）
2. **注入机制**：使用 `LD_PRELOAD`（Linux）或 DLL 注入（Windows）
3. **CUPTI 集成**：利用 CUPTI 的 PC 采样 API
4. **辅助脚本**：`libpc_sampling_continuous.pl` 便于执行

## 构建示例

### Linux 构建过程

1. 导航到 pc_sampling_continuous 目录：
   ```bash
   cd pc_sampling_continuous
   ```

2. 使用提供的 Makefile 构建：
   ```bash
   make
   ```
   
   这会在当前目录创建 `libpc_sampling_continuous.so`。

### Windows 构建过程

对于 Windows，您需要首先构建 Microsoft Detours 库：

1. **下载 Detours 源代码**：
   - 从 GitHub：https://github.com/microsoft/Detours
   - 或 Microsoft：https://www.microsoft.com/en-us/download/details.aspx?id=52586

2. **构建 Detours**：
   ```cmd
   # 解压并导航到 Detours 文件夹
   set DETOURS_TARGET_PROCESSOR=X64
   "Program Files (x86)\Microsoft Visual Studio\<version>\Professional\VC\Auxiliary\Build\vcvarsall.bat" x64
   NMAKE
   ```

3. **复制所需文件**：
   ```cmd
   copy detours.h <pc_sampling_continuous_folder>
   copy detours.lib <pc_sampling_continuous_folder>
   ```

4. **构建示例**：
   ```cmd
   nmake
   ```
   
   这会创建 `pc_sampling_continuous.lib`。

## 运行示例

### Linux 执行

1. **设置库路径**：
   ```bash
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/CUPTI/lib64:/path/to/pc_sampling_continuous:/path/to/pcsamplingutil
   ```

2. **使用辅助脚本**：
   ```bash
   ./libpc_sampling_continuous.pl --help
   ```
   
   这会显示所有可用选项。

3. **与您的应用程序一起运行**：
   ```bash
   ./libpc_sampling_continuous.pl --app /path/to/your/cuda/application
   ```

### Windows 执行

1. **设置库路径**：
   ```cmd
   set PATH=%PATH%;C:\path\to\CUPTI\bin;C:\path\to\pc_sampling_continuous;C:\path\to\pcsamplingutil
   ```

2. **与您的应用程序一起运行**：
   ```cmd
   pc_sampling_continuous.exe your_cuda_application.exe
   ```

## 辅助脚本选项

`libpc_sampling_continuous.pl` 脚本提供各种配置选项：

```bash
# 显示帮助
./libpc_sampling_continuous.pl --help

# 基本用法
./libpc_sampling_continuous.pl --app ./my_cuda_app

# 指定采样频率
./libpc_sampling_continuous.pl --app ./my_cuda_app --frequency 1000

# 设置输出文件
./libpc_sampling_continuous.pl --app ./my_cuda_app --output samples.data

# 启用详细输出
./libpc_sampling_continuous.pl --app ./my_cuda_app --verbose
```

## 理解输出

### 采样数据格式

PC 采样生成包含以下内容的数据文件：

1. **函数信息**：内核名称和地址
2. **PC 采样**：带时间戳的程序计数器值
3. **停顿原因**：每个采样点线程束停顿的原因
4. **源代码关联**：汇编到源代码的映射（当调试信息可用时）

### 示例输出结构

```
Kernel: vectorAdd(float*, float*, float*, int)
PC: 0x7f8b2c001000, Stall: MEMORY_DEPENDENCY, Count: 15
PC: 0x7f8b2c001008, Stall: EXECUTION_DEPENDENCY, Count: 8
PC: 0x7f8b2c001010, Stall: NOT_SELECTED, Count: 12
...
```

## 关键特性

### 自动注入

库自动：
- 拦截 CUDA 运行时和驱动 API 调用
- 为每个内核启动设置 PC 采样
- 在内核执行期间收集采样
- 生成详细报告

### 无需源代码修改

优势包括：
- 无需重新编译即可分析现有应用程序
- 分析第三方 CUDA 库
- 监控生产工作负载
- 比较不同的优化策略

### 跨平台支持

实现处理：
- 不同的动态加载机制（Linux vs Windows）
- 平台特定的库格式
- 不同的 CUDA 安装路径
- 不同的调试符号格式

## 实际应用

### 性能热点识别

使用 PC 采样来：
1. **找到瓶颈指令**：识别执行时间消耗的汇编指令
2. **分析停顿模式**：了解线程束为什么没有取得进展
3. **优化内存访问**：检测内存绑定操作
4. **改进指令调度**：识别依赖停顿

### 算法分析

应用于：
1. **比较实现**：分析不同的算法方法
2. **验证优化**：测量代码更改的影响
3. **了解 GPU 利用率**：查看代码如何使用可用资源
4. **调试性能回归**：识别性能何时何地降级

## 高级用法

### 自定义采样配置

您可以通过修改库配置来调整采样行为：

```c
// 示例配置参数
typedef struct {
    int samplingPeriod;          // 采样周期（微秒）
    int maxSamples;              // 每个内核的最大采样数
    bool enableStallReasons;     // 是否收集停顿原因
    bool enableSourceMapping;    // 是否启用源代码映射
} SamplingConfig;
```

### 批量分析

对于大规模分析：

```bash
# 批量处理多个应用程序
for app in *.out; do
    ./libpc_sampling_continuous.pl --app $app --output ${app}.samples
done

# 合并和分析结果
./analyze_batch_results.py *.samples
```

### 实时监控

实现实时性能监控：

```c
void setupRealtimeMonitoring() {
    // 设置回调函数处理采样数据
    registerSamplingCallback(processSamplesInRealtime);
    
    // 配置低延迟模式
    enableLowLatencyMode();
    
    // 启动监控线程
    startMonitoringThread();
}
```

## 故障排除

### 常见问题

1. **库加载失败**：
   - 检查 `LD_LIBRARY_PATH` 是否包含所有必需的库
   - 验证 CUPTI 版本兼容性
   - 确保有适当的权限

2. **没有采样数据**：
   - 验证 GPU 支持 PC 采样
   - 检查内核是否实际执行
   - 确认采样配置正确

3. **性能影响**：
   - 调整采样频率
   - 限制采样到关键内核
   - 使用异步数据收集

### 调试技巧

```bash
# 启用详细日志记录
export CUPTI_DEBUG=1
export CUDA_INJECTION_DEBUG=1

# 检查库依赖项
ldd libpc_sampling_continuous.so

# 验证 CUDA 安装
nvidia-smi
nvcc --version
```

## 性能考虑

### 采样开销

PC 采样会引入一些开销：
- 典型开销：5-15% 的执行时间
- 开销取决于采样频率和内核复杂性
- 可以通过调整采样参数来平衡精度和性能

### 数据存储

```c
// 优化数据存储
typedef struct {
    uint64_t pc;           // 程序计数器值
    uint32_t stallReason;  // 停顿原因（压缩）
    uint16_t count;        // 采样计数
    uint16_t contextId;    // 上下文 ID（用于多 GPU）
} CompactSample;
```

### 内存使用

管理内存消耗：
- 使用循环缓冲区处理长时间运行的应用程序
- 压缩重复的采样数据
- 实现数据流式处理到磁盘

## 扩展功能

### 多 GPU 支持

```c
void setupMultiGPUSampling() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    for (int i = 0; i < deviceCount; i++) {
        cudaSetDevice(i);
        initializeSamplingForDevice(i);
    }
}
```

### 源代码关联

启用源代码映射：

```bash
# 使用调试信息编译
nvcc -g -lineinfo your_kernel.cu

# 运行时启用源映射
export INJECTION_ENABLE_SOURCE_MAPPING=1
```

### 自定义过滤器

实现选择性采样：

```c
bool shouldSampleKernel(const char* kernelName) {
    // 只采样特定的内核
    return strstr(kernelName, "optimize_me") != NULL;
}
```

## 最佳实践

1. **从小规模开始**：首先在小型测试用例上验证设置
2. **选择性采样**：专注于关键的性能瓶颈
3. **迭代分析**：使用结果指导后续的优化工作
4. **版本控制结果**：跟踪不同优化的性能变化
5. **文档化发现**：记录性能洞察供未来参考

连续 PC 采样为理解 GPU 性能提供了无与伦比的洞察力。通过自动注入和详细的指令级分析，它是优化 CUDA 应用程序的强大工具。 