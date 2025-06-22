# CUPTI PC 采样分析工具教程

> GitHub 仓库和完整教程可在 <https://github.com/eunomia-bpf/cupti-tutorial> 获取。

## 简介

PC 采样工具是一个强大的后处理工具，用于分析由 `pc_sampling_continuous` 样本收集的数据。它将原始 PC 采样数据转换为可操作的洞察，通过将汇编指令与停顿原因关联，并在调试信息可用时提供源级映射。

## 您将学到什么

- 如何分析连续采样生成的 PC 采样数据文件
- 理解汇编指令级别的停顿原因计数器
- 将汇编代码与 CUDA C 源代码关联的技术
- 使用 CUDA cubin 文件进行详细分析
- 从 PC 采样结果解释性能瓶颈

## 理解 PC 采样数据分析

PC 采样分析与实时监控的不同之处在于：

1. **离线处理收集的数据**：允许详细分析而无运行时开销
2. **提供汇编级洞察**：显示哪些指令导致性能问题
3. **与源代码关联**：将性能热点映射回原始 C/C++ 代码
4. **量化停顿原因**：解释 GPU 执行单元空闲的原因
5. **支持批处理**：可以一起分析多个采样会话

## 关键概念

### 停顿原因

GPU 线程束可能因各种原因停顿：

- **MEMORY_DEPENDENCY**：等待内存操作完成
- **EXECUTION_DEPENDENCY**：等待流水线中的前一条指令
- **NOT_SELECTED**：线程束就绪但调度器选择了其他线程束
- **MEMORY_THROTTLE**：内存子系统饱和
- **PIPE_BUSY**：执行流水线完全利用
- **CONSTANT_MEMORY_DEPENDENCY**：等待常量内存访问
- **TEXTURE_MEMORY_DEPENDENCY**：等待纹理内存访问

### 汇编到源代码的关联

当以下条件满足时，工具可以将汇编指令映射回源代码：
- 调试信息编译到应用程序中（`-g` 标志）
- CUDA cubin 文件被提取并正确命名
- 源文件在分析时可访问

## 构建工具

### 先决条件

确保您有：
- 已安装的 CUDA 工具包
- 可用的 CUPTI 库
- 访问目标应用程序的 cubin 文件

### 构建过程

1. 导航到 pc_sampling_utility 目录：
   ```bash
   cd pc_sampling_utility
   ```

2. 使用提供的 Makefile 构建：
   ```bash
   make
   ```
   
   这会创建 `pc_sampling_utility` 可执行文件。

## 准备输入数据

### 生成 PC 采样数据

首先，使用连续采样库收集 PC 采样数据：

```bash
# 使用连续采样库
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/cupti/lib64:/path/to/pc_sampling_continuous
./libpc_sampling_continuous.pl --app ./your_cuda_application --output samples.data
```

### 提取 CUDA Cubin 文件

为了进行源代码关联，从应用程序中提取 cubin 文件：

```bash
# 从可执行文件提取所有 cubin 文件
cuobjdump -xelf all your_cuda_application

# 从库文件提取
cuobjdump -xelf all libmy_cuda_library.so
```

**重要**：`cuobjdump` 版本必须与用于构建应用程序的 CUDA 工具包版本匹配。

### 命名 Cubin 文件

按顺序重命名提取的 cubin 文件：

```bash
# 按顺序重命名 cubin 文件
mv first_extracted_file.cubin 1.cubin
mv second_extracted_file.cubin 2.cubin
mv third_extracted_file.cubin 3.cubin
# ... 依此类推
```

工具期望 cubin 文件命名为 `1.cubin`、`2.cubin`、`3.cubin` 等。

## 运行分析

### 基本用法

```bash
./pc_sampling_utility --input samples.data
```

### 命令行选项

查看所有可用选项：

```bash
./pc_sampling_utility --help
```

常用选项包括：

```bash
# 指定输入文件
./pc_sampling_utility --input samples.data

# 设置 cubin 目录
./pc_sampling_utility --input samples.data --cubin-path ./cubins/

# 启用详细输出
./pc_sampling_utility --input samples.data --verbose

# 过滤特定内核
./pc_sampling_utility --input samples.data --kernel vectorAdd

# 设置输出格式
./pc_sampling_utility --input samples.data --format csv
```

## 理解输出

### 示例输出格式

```
Kernel: vectorAdd(float*, float*, float*, int)
================================================================================

汇编分析：
PC: 0x008 | INST: LDG.E.SYS R2, [R8] | Stall: MEMORY_DEPENDENCY | Count: 245 (15.3%)
PC: 0x010 | INST: LDG.E.SYS R4, [R10] | Stall: MEMORY_DEPENDENCY | Count: 198 (12.4%)
PC: 0x018 | INST: FADD R6, R2, R4 | Stall: EXECUTION_DEPENDENCY | Count: 89 (5.6%)
PC: 0x020 | INST: STG.E.SYS [R12], R6 | Stall: MEMORY_DEPENDENCY | Count: 156 (9.7%)

源代码关联：
PC: 0x008 | File: vector_add.cu | Line: 42 | Code: float a = A[i];
PC: 0x010 | File: vector_add.cu | Line: 43 | Code: float b = B[i];
PC: 0x018 | File: vector_add.cu | Line: 44 | Code: float result = a + b;
PC: 0x020 | File: vector_add.cu | Line: 45 | Code: C[i] = result;

性能摘要：
总采样数：1599
内存绑定：599 采样 (37.5%)
执行绑定：234 采样 (14.6%)
调度器限制：445 采样 (27.8%)
其他：321 采样 (20.1%)
```

### 要分析的关键指标

1. **停顿分布**：哪些停顿原因主导内核执行
2. **热点指令**：具有最高采样计数的汇编指令
3. **内存访问模式**：内存操作如何导致停顿
4. **源行关联**：哪些源行对应性能问题

## 实际分析工作流

### 识别内存瓶颈

1. **查找 MEMORY_DEPENDENCY 停顿**：高计数表明内存绑定内核
2. **分析访问模式**：检查访问是否合并
3. **考虑缓存策略**：评估共享内存或纹理内存使用

示例工作流：
```bash
# 专注于内存相关停顿
./pc_sampling_utility --input samples.data --filter-stall MEMORY_DEPENDENCY

# 分析特定内存指令
./pc_sampling_utility --input samples.data --filter-instruction "LDG\|STG"
```

### 优化指令依赖

1. **识别 EXECUTION_DEPENDENCY 热点**：显示指令流水线停顿
2. **分析指令排序**：查找重新排序机会
3. **检查资源冲突**：识别寄存器或功能单元竞争

### 分析分支效率

```bash
# 分析控制流指令
./pc_sampling_utility --input samples.data --filter-instruction "BRA\|EXIT\|RET"

# 检查分歧模式
./pc_sampling_utility --input samples.data --analyze-divergence
```

## 高级分析技术

### 热点识别

```bash
# 找到前 10 个热点指令
./pc_sampling_utility --input samples.data --top-hotspots 10

# 按停顿类型分组
./pc_sampling_utility --input samples.data --group-by-stall
```

### 性能比较

```bash
# 比较不同版本
./pc_sampling_utility --input before.data --compare after.data

# 生成差异报告
./pc_sampling_utility --input samples.data --baseline baseline.data --diff-report
```

## 最佳实践

### 有效分析

1. **从总体视图开始**：首先查看停顿分布
2. **深入热点**：专注于高计数的指令
3. **关联源代码**：将汇编发现映射回源代码
4. **迭代优化**：使用分析结果指导代码更改

### 常见陷阱

1. **忽略低频事件**：有时罕见但昂贵的操作很重要
2. **过度优化**：平衡优化努力与潜在收益
3. **忽略上下文**：考虑整体应用程序性能而不仅仅是内核

PC 采样工具提供了无与伦比的 GPU 性能洞察。通过将汇编级数据与源代码关联，它为 CUDA 内核优化提供了强大的基础。 