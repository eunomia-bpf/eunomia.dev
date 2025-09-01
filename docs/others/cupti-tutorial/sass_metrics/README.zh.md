# SASS 指标收集教程

> GitHub 仓库和完整教程可在 <https://github.com/eunomia-bpf/cupti-tutorial> 获取。

## 简介

SASS 指标收集示例演示如何使用 CUPTI 在 SASS（着色器汇编）级别收集详细的性能指标。本教程向您展示如何收集关于 GPU 指令执行的低级性能数据，提供对 CUDA 内核实际汇编代码性能的洞察。

## 您将学到什么

- 如何收集 SASS 级别的性能指标
- 理解 GPU 指令级性能分析
- 分析汇编代码效率和瓶颈
- 将高级 CUDA 代码与 SASS 指令关联
- 基于汇编级洞察优化内核

## 理解 SASS 指标

SASS（着色器汇编）指标通过 CUPTI 提供最低级别的性能洞察：

1. **指令级分析**：单个 GPU 汇编指令的性能
2. **执行效率**：硬件如何高效执行指令
3. **资源利用率**：内核如何使用可用的 GPU 执行单元
4. **流水线分析**：理解指令流水线行为
5. **瓶颈识别**：精确定位特定指令级性能问题

## 关键 SASS 指标

### 指令执行指标
- **inst_executed**：执行的总指令数
- **inst_issued**：发送到执行单元的指令数
- **inst_fp_32/64**：浮点指令计数
- **inst_integer**：整数指令计数
- **inst_bit_convert**：位操作指令
- **inst_control**：控制流指令

### 内存指令指标
- **inst_compute_ld_st**：加载/存储计算指令
- **global_load/store**：全局内存访问指令
- **shared_load/store**：共享内存访问指令
- **local_load/store**：本地内存访问指令

### 执行效率指标
- **thread_inst_executed**：每线程执行的指令数
- **warp_execution_efficiency**：线程束中活跃线程的百分比
- **branch_efficiency**：非分歧分支的百分比

## 构建示例

### 先决条件

- 带 CUPTI 的 CUDA 工具包
- 计算能力 5.0 或更高的 GPU
- 访问详细 GPU 性能计数器

### 构建过程

```bash
cd sass_metrics
make
```

这会创建用于收集 SASS 级性能数据的 `sass_metrics` 可执行文件。

## 运行示例

### 基本执行

```bash
./sass_metrics
```

### 示例输出

```
evice Num: 0
Lazy Patching Enabled
Device Name: NVIDIA H100 MIG 1c.4g.48gb
Device compute capability: 9.0
Metric Name: smsp__sass_inst_executed, MetricID: 8913560818632243504, Metric Description: # of warp instructions executed
Enable SASS Patching
Launching VectorAdd

Module cubinCrc: 2917979105
Kernel Name: _Z9VectorAddPKiS0_Pii
metric Name: smsp__sass_inst_executed
                [Inst] pcOffset: 0x0    metricValue:    [0]: 1000
                [Inst] pcOffset: 0x10   metricValue:    [0]: 1000
                [Inst] pcOffset: 0x20   metricValue:    [0]: 1000
                [Inst] pcOffset: 0x30   metricValue:    [0]: 1000
                [Inst] pcOffset: 0x40   metricValue:    [0]: 1000
                [Inst] pcOffset: 0x50   metricValue:    [0]: 1000
                [Inst] pcOffset: 0x60   metricValue:    [0]: 1000
                [Inst] pcOffset: 0x70   metricValue:    [0]: 1000
                [Inst] pcOffset: 0x80   metricValue:    [0]: 1000
                [Inst] pcOffset: 0x90   metricValue:    [0]: 1000
                [Inst] pcOffset: 0xa0   metricValue:    [0]: 1000
                [Inst] pcOffset: 0xb0   metricValue:    [0]: 1000
                [Inst] pcOffset: 0xc0   metricValue:    [0]: 1000
                [Inst] pcOffset: 0xd0   metricValue:    [0]: 1000
                [Inst] pcOffset: 0xe0   metricValue:    [0]: 1000
                [Inst] pcOffset: 0xf0   metricValue:    [0]: 1000
                [Inst] pcOffset: 0x100  metricValue:    [0]: 1000
                [Inst] pcOffset: 0x110  metricValue:    [0]: 1000
                [Inst] pcOffset: 0x120  metricValue:    [0]: 1000
                [Inst] pcOffset: 0x130  metricValue:    [0]: 1000

Launching VectorSubtract
Disable SASS Patching
Launching VectorMultiply
```

## 代码架构

### SASS 指标收集器

```cpp
class SASSMetricsCollector {
private:
    struct SASSMetrics {
        uint64_t totalInstructions;
        uint64_t fp32Instructions;
        uint64_t integerInstructions;
        uint64_t memoryInstructions;
        uint64_t controlInstructions;
        double warpEfficiency;
        double branchEfficiency;
        double threadEfficiency;
    };
    
    std::map<std::string, SASSMetrics> kernelMetrics;
    std::vector<CUpti_EventID> sassEventIds;
    CUpti_EventGroup eventGroup;

public:
    void setupSASSMetrics(CUcontext context, CUdevice device);
    void startCollection();
    void stopCollection();
    void analyzeKernelSASS(const std::string& kernelName);
    void generateSASSReport();
};
```

### 汇编代码分析

```cpp
class AssemblyAnalyzer {
private:
    struct InstructionProfile {
        std::string opcode;
        uint64_t executionCount;
        double averageLatency;
        double throughput;
        std::string category;
    };
    
    std::map<std::string, InstructionProfile> instructionProfiles;

public:
    void analyzeInstructionMix(const SASSMetrics& metrics) {
        std::cout << "指令混合分析：" << std::endl;
        
        uint64_t totalInsts = metrics.totalInstructions;
        
        double fpPercentage = (double)metrics.fp32Instructions / totalInsts * 100;
        double intPercentage = (double)metrics.integerInstructions / totalInsts * 100;
        double memPercentage = (double)metrics.memoryInstructions / totalInsts * 100;
        double ctrlPercentage = (double)metrics.controlInstructions / totalInsts * 100;
        
        std::cout << "  浮点：" << fpPercentage << "%" << std::endl;
        std::cout << "  整数：" << intPercentage << "%" << std::endl;
        std::cout << "  内存：" << memPercentage << "%" << std::endl;
        std::cout << "  控制：" << ctrlPercentage << "%" << std::endl;
        
        // 分析指令平衡
        if (memPercentage > 40) {
            std::cout << "  警告：检测到内存绑定内核" << std::endl;
        }
        if (ctrlPercentage > 15) {
            std::cout << "  警告：控制开销高" << std::endl;
        }
    }
};
```

## 指令级优化

### 内存指令分析

```cpp
void analyzeMemoryInstructions() {
    // 收集内存相关的 SASS 指标
    std::vector<std::string> memoryMetrics = {
        "global_load_inst_executed",
        "global_store_inst_executed", 
        "shared_load_inst_executed",
        "shared_store_inst_executed",
        "local_load_inst_executed",
        "local_store_inst_executed"
    };
    
    for (const auto& metric : memoryMetrics) {
        uint64_t count = getMetricValue(metric);
        std::cout << metric << ": " << count << std::endl;
    }
    
    // 分析内存访问效率
    double globalLoadRatio = getMetricValue("global_load_inst_executed") / 
                           getMetricValue("total_inst_executed");
    
    if (globalLoadRatio > 0.3) {
        std::cout << "建议：考虑使用共享内存缓存频繁访问的数据" << std::endl;
    }
}
```

### 计算指令分析

```cpp
void analyzeComputeInstructions() {
    std::map<std::string, uint64_t> computeInsts = {
        {"fp32_add", getMetricValue("inst_fp_32_add")},
        {"fp32_mul", getMetricValue("inst_fp_32_mul")},
        {"fp32_fma", getMetricValue("inst_fp_32_fma")},
        {"integer_add", getMetricValue("inst_integer_add")},
        {"integer_mul", getMetricValue("inst_integer_mul")}
    };
    
    uint64_t totalCompute = 0;
    for (const auto& [name, count] : computeInsts) {
        totalCompute += count;
        std::cout << name << ": " << count << std::endl;
    }
    
    // 计算密度分析
    double computeDensity = (double)totalCompute / getMetricValue("total_inst_executed");
    std::cout << "计算密度: " << computeDensity * 100 << "%" << std::endl;
    
    if (computeDensity < 0.2) {
        std::cout << "建议：增加计算密度以更好地隐藏内存延迟" << std::endl;
    }
}
```

## 实际优化示例

### 向量化指令分析

```cpp
class VectorizationAnalyzer {
public:
    void analyzeVectorization() {
        // 检查向量化加载/存储使用
        uint64_t scalarLoads = getMetricValue("ld_32_inst");
        uint64_t vectorLoads = getMetricValue("ld_64_inst") + 
                              getMetricValue("ld_128_inst");
        
        double vectorizationRatio = (double)vectorLoads / (scalarLoads + vectorLoads);
        
        std::cout << "向量化比率: " << vectorizationRatio * 100 << "%" << std::endl;
        
        if (vectorizationRatio < 0.7) {
            std::cout << "建议：使用向量化数据类型（float2、float4）" << std::endl;
            std::cout << "这可以提高内存带宽利用率" << std::endl;
        }
    }
};
```

### 分支效率分析

```cpp
class BranchAnalyzer {
public:
    void analyzeBranchEfficiency() {
        uint64_t branchInsts = getMetricValue("branch_inst_executed");
        uint64_t divergentBranches = getMetricValue("divergent_branch");
        
        double branchEfficiency = 1.0 - ((double)divergentBranches / branchInsts);
        
        std::cout << "分支效率: " << branchEfficiency * 100 << "%" << std::endl;
        
        if (branchEfficiency < 0.8) {
            std::cout << "警告：高分支分歧检测" << std::endl;
            std::cout << "建议：" << std::endl;
            std::cout << "- 重构条件逻辑" << std::endl;
            std::cout << "- 使用按位操作替代分支" << std::endl;
            std::cout << "- 考虑数据重组以减少分歧" << std::endl;
        }
    }
};
```

## 高级分析技术

### 热点指令识别

```cpp
void identifyHotspotInstructions() {
    std::vector<std::pair<std::string, uint64_t>> instructionCounts;
    
    // 收集所有指令类型的计数
    std::vector<std::string> allInstructions = {
        "inst_fp_32", "inst_integer", "inst_control",
        "inst_misc", "inst_compute_ld_st"
    };
    
    for (const auto& inst : allInstructions) {
        uint64_t count = getMetricValue(inst);
        instructionCounts.push_back({inst, count});
    }
    
    // 按计数排序
    std::sort(instructionCounts.begin(), instructionCounts.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    std::cout << "热点指令类型：" << std::endl;
    for (size_t i = 0; i < std::min(size_t(5), instructionCounts.size()); i++) {
        const auto& [name, count] = instructionCounts[i];
        std::cout << i+1 << ". " << name << ": " << count << std::endl;
    }
}
```

### 性能瓶颈诊断

```cpp
class BottleneckDiagnostics {
public:
    void diagnoseBottlenecks() {
        double memoryRatio = getMemoryInstructionRatio();
        double computeRatio = getComputeInstructionRatio();
        double controlRatio = getControlInstructionRatio();
        
        std::cout << "=== 性能瓶颈诊断 ===" << std::endl;
        
        if (memoryRatio > 0.4) {
            std::cout << "内存绑定内核检测 (" << memoryRatio * 100 << "% 内存指令)" << std::endl;
            suggestMemoryOptimizations();
        }
        
        if (computeRatio < 0.3) {
            std::cout << "计算利用率不足 (" << computeRatio * 100 << "% 计算指令)" << std::endl;
            suggestComputeOptimizations();
        }
        
        if (controlRatio > 0.1) {
            std::cout << "控制开销高 (" << controlRatio * 100 << "% 控制指令)" << std::endl;
            suggestControlOptimizations();
        }
    }

private:
    void suggestMemoryOptimizations() {
        std::cout << "内存优化建议：" << std::endl;
        std::cout << "- 使用共享内存进行数据重用" << std::endl;
        std::cout << "- 优化内存访问模式以实现合并" << std::endl;
        std::cout << "- 考虑纹理内存进行只读数据" << std::endl;
    }
    
    void suggestComputeOptimizations() {
        std::cout << "计算优化建议：" << std::endl;
        std::cout << "- 增加每线程的工作量" << std::endl;
        std::cout << "- 使用融合乘加指令 (FMA)" << std::endl;
        std::cout << "- 重新平衡计算与内存访问" << std::endl;
    }
    
    void suggestControlOptimizations() {
        std::cout << "控制流优化建议：" << std::endl;
        std::cout << "- 减少分支分歧" << std::endl;
        std::cout << "- 使用按位操作替代条件语句" << std::endl;
        std::cout << "- 重构循环结构" << std::endl;
    }
};
```

## 最佳实践

### 指标选择策略

1. **从高级指标开始**：首先查看指令混合和效率
2. **深入特定领域**：基于初始发现专注于内存或计算
3. **关联源代码**：将 SASS 发现映射回高级代码
4. **迭代优化**：使用指标验证优化效果

### 常见优化模式

```cpp
// 示例：基于 SASS 指标的优化决策
void makeOptimizationDecisions() {
    double memRatio = getMemoryInstructionRatio();
    double branchEff = getBranchEfficiency();
    double warpEff = getWarpExecutionEfficiency();
    
    if (memRatio > 0.5 && warpEff > 0.8) {
        std::cout << "优化焦点：内存系统" << std::endl;
        std::cout << "建议：实现内存合并优化" << std::endl;
    }
    
    if (branchEff < 0.7) {
        std::cout << "优化焦点：控制流" << std::endl;
        std::cout << "建议：减少分支分歧" << std::endl;
    }
    
    if (warpEff < 0.6) {
        std::cout << "优化焦点：线程束利用率" << std::endl;
        std::cout << "建议：调整线程块大小和占用率" << std::endl;
    }
}
```

SASS 指标收集为理解和优化 CUDA 内核提供了最深入的洞察。通过分析实际执行的汇编指令，您可以识别性能瓶颈并进行精确的优化，这些优化在源代码级别可能不明显。 