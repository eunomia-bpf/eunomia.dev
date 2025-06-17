# SASS Metrics Collection Tutorial

## Introduction

The SASS Metrics Collection sample demonstrates how to collect detailed performance metrics at the SASS (Shader ASSembly) level using CUPTI. This tutorial shows you how to gather low-level performance data about GPU instruction execution, providing insights into the actual assembly code performance of your CUDA kernels.

## What You'll Learn

- How to collect SASS-level performance metrics
- Understanding GPU instruction-level performance analysis
- Analyzing assembly code efficiency and bottlenecks
- Correlating high-level CUDA code with SASS instructions
- Optimizing kernels based on assembly-level insights

## Understanding SASS Metrics

SASS (Shader ASSembly) metrics provide the lowest level of performance insight available through CUPTI:

1. **Instruction-Level Analysis**: Performance of individual GPU assembly instructions
2. **Execution Efficiency**: How efficiently instructions are executed by the hardware
3. **Resource Utilization**: How well kernels use available GPU execution units
4. **Pipeline Analysis**: Understanding instruction pipeline behavior
5. **Bottleneck Identification**: Pinpointing specific instruction-level performance issues

## Key SASS Metrics

### Instruction Execution Metrics
- **inst_executed**: Total instructions executed
- **inst_issued**: Instructions issued to execution units
- **inst_fp_32/64**: Floating-point instruction counts
- **inst_integer**: Integer instruction counts
- **inst_bit_convert**: Bit manipulation instructions
- **inst_control**: Control flow instructions

### Memory Instruction Metrics
- **inst_compute_ld_st**: Load/store compute instructions
- **global_load/store**: Global memory access instructions
- **shared_load/store**: Shared memory access instructions
- **local_load/store**: Local memory access instructions

### Execution Efficiency Metrics
- **thread_inst_executed**: Instructions executed per thread
- **warp_execution_efficiency**: Percentage of active threads in warps
- **branch_efficiency**: Percentage of non-divergent branches

## Building the Sample

### Prerequisites

- CUDA Toolkit with CUPTI
- GPU with compute capability 5.0 or higher
- Access to detailed GPU performance counters

### Build Process

```bash
cd sass_metrics
make
```

This creates the `sass_metrics` executable for collecting SASS-level performance data.

## Running the Sample

### Basic Execution

```bash
./sass_metrics
```

### Sample Output

```
=== SASS Metrics Analysis ===

Kernel: vectorAdd
SASS Instruction Analysis:
  Total Instructions: 2,048,576
  FP32 Instructions: 1,024,288 (50.0%)
  Integer Instructions: 512,144 (25.0%)
  Memory Instructions: 409,716 (20.0%)
  Control Instructions: 102,428 (5.0%)

Execution Efficiency:
  Warp Execution Efficiency: 87.5%
  Branch Efficiency: 94.2%
  Thread Instruction Efficiency: 91.8%

Memory Access Patterns:
  Global Load Instructions: 204,858
  Global Store Instructions: 204,858
  Shared Memory Access: 0
  Local Memory Access: 0

Pipeline Utilization:
  ALU Pipeline: 85.3%
  FPU Pipeline: 78.9%
  Memory Pipeline: 92.1%
  Control Pipeline: 15.2%

Performance Bottlenecks:
  - Memory bandwidth limited: 45.2% of execution time
  - ALU utilization could be improved
  - No significant control divergence detected
```

## Code Architecture

### SASS Metrics Collector

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

void SASSMetricsCollector::setupSASSMetrics(CUcontext context, CUdevice device) {
    // Set up SASS-level event collection
    CUPTI_CALL(cuptiEventGroupCreate(context, &eventGroup, 0));
    
    // Add SASS instruction counting events
    std::vector<std::string> sassEvents = {
        "inst_executed",
        "inst_fp_32", 
        "inst_integer",
        "inst_compute_ld_st",
        "inst_control",
        "thread_inst_executed",
        "warp_execution_efficiency",
        "branch_efficiency"
    };
    
    for (const auto& eventName : sassEvents) {
        CUpti_EventID eventId;
        CUptiResult result = cuptiEventGetIdFromName(device, eventName.c_str(), &eventId);
        if (result == CUPTI_SUCCESS) {
            CUPTI_CALL(cuptiEventGroupAddEvent(eventGroup, eventId));
            sassEventIds.push_back(eventId);
        }
    }
}
```

### Assembly Code Analysis

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
        std::cout << "Instruction Mix Analysis:" << std::endl;
        
        uint64_t totalInsts = metrics.totalInstructions;
        
        double fpPercentage = (double)metrics.fp32Instructions / totalInsts * 100;
        double intPercentage = (double)metrics.integerInstructions / totalInsts * 100;
        double memPercentage = (double)metrics.memoryInstructions / totalInsts * 100;
        double ctrlPercentage = (double)metrics.controlInstructions / totalInsts * 100;
        
        std::cout << "  Floating Point: " << fpPercentage << "%" << std::endl;
        std::cout << "  Integer: " << intPercentage << "%" << std::endl;
        std::cout << "  Memory: " << memPercentage << "%" << std::endl;
        std::cout << "  Control: " << ctrlPercentage << "%" << std::endl;
        
        // Analyze instruction balance
        if (memPercentage > 40) {
            std::cout << "  WARNING: Memory-bound kernel detected" << std::endl;
        }
        if (ctrlPercentage > 20) {
            std::cout << "  WARNING: High control overhead detected" << std::endl;
        }
    }
    
    void suggestOptimizations(const SASSMetrics& metrics) {
        std::cout << "Optimization Suggestions:" << std::endl;
        
        if (metrics.warpEfficiency < 0.8) {
            std::cout << "  - Consider optimizing warp utilization" << std::endl;
            std::cout << "  - Check for thread divergence patterns" << std::endl;
        }
        
        if (metrics.branchEfficiency < 0.9) {
            std::cout << "  - Optimize branch patterns to reduce divergence" << std::endl;
            std::cout << "  - Consider predicated execution for simple branches" << std::endl;
        }
        
        double memRatio = (double)metrics.memoryInstructions / metrics.totalInstructions;
        if (memRatio > 0.3) {
            std::cout << "  - Consider memory access optimization" << std::endl;
            std::cout << "  - Evaluate shared memory usage opportunities" << std::endl;
        }
    }
};
```

## Advanced SASS Analysis

### Instruction Pipeline Analysis

```cpp
class PipelineAnalyzer {
private:
    struct PipelineMetrics {
        double aluUtilization;
        double fpuUtilization;
        double memoryUtilization;
        double controlUtilization;
        std::vector<double> stallReasons;
    };

public:
    PipelineMetrics analyzePipelineUtilization(const SASSMetrics& metrics) {
        PipelineMetrics pipelineMetrics;
        
        // Calculate pipeline utilization based on instruction mix
        uint64_t totalInsts = metrics.totalInstructions;
        
        pipelineMetrics.aluUtilization = 
            (double)(metrics.integerInstructions) / totalInsts;
        pipelineMetrics.fpuUtilization = 
            (double)(metrics.fp32Instructions) / totalInsts;
        pipelineMetrics.memoryUtilization = 
            (double)(metrics.memoryInstructions) / totalInsts;
        pipelineMetrics.controlUtilization = 
            (double)(metrics.controlInstructions) / totalInsts;
        
        return pipelineMetrics;
    }
    
    void identifyBottlenecks(const PipelineMetrics& pipeline) {
        std::cout << "Pipeline Bottleneck Analysis:" << std::endl;
        
        double maxUtilization = std::max({
            pipeline.aluUtilization,
            pipeline.fpuUtilization, 
            pipeline.memoryUtilization,
            pipeline.controlUtilization
        });
        
        if (maxUtilization == pipeline.memoryUtilization) {
            std::cout << "  Primary bottleneck: Memory pipeline" << std::endl;
            std::cout << "  Suggestions: Optimize memory access patterns, use shared memory" << std::endl;
        } else if (maxUtilization == pipeline.fpuUtilization) {
            std::cout << "  Primary bottleneck: FP pipeline" << std::endl;
            std::cout << "  Suggestions: Balance compute with memory operations" << std::endl;
        } else if (maxUtilization == pipeline.controlUtilization) {
            std::cout << "  Primary bottleneck: Control pipeline" << std::endl;
            std::cout << "  Suggestions: Reduce branch divergence, simplify control flow" << std::endl;
        }
    }
};
```

### Performance Model Validation

```cpp
class PerformanceModel {
private:
    struct ModelParameters {
        double peakALUThroughput;
        double peakFPUThroughput;
        double memoryBandwidth;
        double instructionIssueRate;
    };
    
    ModelParameters getGPUParameters(int deviceId) {
        ModelParameters params;
        
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, deviceId);
        
        // Estimate peak throughput based on GPU architecture
        params.peakALUThroughput = prop.multiProcessorCount * prop.clockRate * 2; // Simplified
        params.peakFPUThroughput = prop.multiProcessorCount * prop.clockRate;
        params.memoryBandwidth = prop.memoryBusWidth * prop.memoryClockRate * 2 / 8; // GB/s
        params.instructionIssueRate = prop.multiProcessorCount * prop.clockRate;
        
        return params;
    }

public:
    void validatePerformanceModel(const SASSMetrics& metrics, int deviceId) {
        ModelParameters params = getGPUParameters(deviceId);
        
        // Calculate achieved vs theoretical performance
        double achievedALUUtilization = 
            (double)metrics.integerInstructions / metrics.totalInstructions;
        double achievedFPUUtilization = 
            (double)metrics.fp32Instructions / metrics.totalInstructions;
        
        std::cout << "Performance Model Validation:" << std::endl;
        std::cout << "  ALU Utilization: " << achievedALUUtilization * 100 << "%" << std::endl;
        std::cout << "  FPU Utilization: " << achievedFPUUtilization * 100 << "%" << std::endl;
        
        // Compare with theoretical limits
        double efficiencyGap = 1.0 - std::max(achievedALUUtilization, achievedFPUUtilization);
        if (efficiencyGap > 0.3) {
            std::cout << "  Significant efficiency gap detected: " << efficiencyGap * 100 << "%" << std::endl;
        }
    }
};
```

## Real-World Applications

### Matrix Multiplication Analysis

```cpp
void analyzeMatrixMultiplication() {
    SASSMetricsCollector collector;
    AssemblyAnalyzer analyzer;
    
    // Set up metrics collection
    collector.setupSASSMetrics(context, device);
    
    // Launch matrix multiplication kernel
    collector.startCollection();
    
    dim3 grid(32, 32);
    dim3 block(16, 16);
    matrixMulKernel<<<grid, block>>>(d_A, d_B, d_C, N);
    
    cudaDeviceSynchronize();
    collector.stopCollection();
    
    // Analyze results
    SASSMetrics metrics = collector.getMetrics("matrixMulKernel");
    analyzer.analyzeInstructionMix(metrics);
    analyzer.suggestOptimizations(metrics);
    
    std::cout << "Matrix Multiplication SASS Analysis:" << std::endl;
    std::cout << "  Compute Intensity: " << calculateComputeIntensity(metrics) << std::endl;
    std::cout << "  Memory Efficiency: " << calculateMemoryEfficiency(metrics) << std::endl;
}
```

### Reduction Operation Profiling

```cpp
void analyzeReductionKernel() {
    SASSMetricsCollector collector;
    
    collector.setupSASSMetrics(context, device);
    collector.startCollection();
    
    // Launch reduction kernel
    reductionKernel<<<grid, block, sharedMemSize>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    collector.stopCollection();
    
    SASSMetrics metrics = collector.getMetrics("reductionKernel");
    
    std::cout << "Reduction Kernel Analysis:" << std::endl;
    std::cout << "  Shared Memory Usage Efficiency: " << 
        analyzeSharedMemoryUsage(metrics) << std::endl;
    std::cout << "  Synchronization Overhead: " << 
        analyzeSynchronizationOverhead(metrics) << std::endl;
    std::cout << "  Warp Divergence: " << 
        (100.0 - metrics.branchEfficiency) << "%" << std::endl;
}
```

## Integration with Development Tools

### Compiler Optimization Analysis

```cpp
class CompilerAnalyzer {
public:
    void compareOptimizationLevels() {
        std::vector<std::string> optimizationFlags = {"-O0", "-O1", "-O2", "-O3"};
        
        for (const auto& flag : optimizationFlags) {
            std::cout << "Testing with " << flag << ":" << std::endl;
            
            // Recompile with different optimization level
            recompileKernel(flag);
            
            // Collect SASS metrics
            SASSMetrics metrics = collectMetricsForOptimization(flag);
            
            std::cout << "  Instructions: " << metrics.totalInstructions << std::endl;
            std::cout << "  Efficiency: " << metrics.warpEfficiency << std::endl;
            
            analyzeOptimizationImpact(metrics, flag);
        }
    }
    
private:
    void analyzeOptimizationImpact(const SASSMetrics& metrics, const std::string& flag) {
        // Analyze how compiler optimizations affect SASS metrics
        if (flag == "-O3" && metrics.controlInstructions > baseline.controlInstructions) {
            std::cout << "    Warning: High optimization may have increased control overhead" << std::endl;
        }
        
        if (metrics.warpEfficiency < 0.8) {
            std::cout << "    Consider different optimization strategies" << std::endl;
        }
    }
};
```

### Assembly Code Correlation

```cpp
class CodeCorrelator {
public:
    void correlateCUDAtoSASS(const std::string& cudaFile, const std::string& sassFile) {
        // Parse CUDA source code
        auto cudaLines = parseCUDASource(cudaFile);
        
        // Parse SASS assembly
        auto sassInstructions = parseSASSAssembly(sassFile);
        
        // Correlate performance metrics with source lines
        for (const auto& [lineNum, sassInsts] : sassInstructions) {
            if (lineNum < cudaLines.size()) {
                std::cout << "CUDA Line " << lineNum << ": " << cudaLines[lineNum] << std::endl;
                std::cout << "  SASS Instructions: " << sassInsts.size() << std::endl;
                
                for (const auto& inst : sassInsts) {
                    std::cout << "    " << inst.opcode << " (executed " 
                             << inst.executionCount << " times)" << std::endl;
                }
                
                // Identify performance hotspots
                if (getTotalExecutionCount(sassInsts) > hotspotThreshold) {
                    std::cout << "  >>> PERFORMANCE HOTSPOT DETECTED <<<" << std::endl;
                }
            }
        }
    }
};
```

## Troubleshooting SASS Metrics

### Common Issues

1. **Metric Availability**: Not all GPUs support all SASS metrics
2. **Event Multiplexing**: Some metrics cannot be collected simultaneously
3. **Precision Limitations**: SASS counters may have limited precision
4. **Context Sensitivity**: Metrics may vary with different execution contexts

### Debug Strategies

```cpp
class SASSDebugger {
public:
    void validateMetricSupport(CUdevice device) {
        uint32_t numEvents;
        CUPTI_CALL(cuptiDeviceGetNumEventDomains(device, &numEvents));
        
        std::cout << "Available event domains: " << numEvents << std::endl;
        
        // Check specific SASS events
        std::vector<std::string> requiredEvents = {
            "inst_executed", "inst_fp_32", "warp_execution_efficiency"
        };
        
        for (const auto& eventName : requiredEvents) {
            CUpti_EventID eventId;
            CUptiResult result = cuptiEventGetIdFromName(device, eventName.c_str(), &eventId);
            
            if (result == CUPTI_SUCCESS) {
                std::cout << "✓ " << eventName << " supported" << std::endl;
            } else {
                std::cout << "✗ " << eventName << " not supported" << std::endl;
            }
        }
    }
    
    void checkEventGroupCompatibility(const std::vector<CUpti_EventID>& events, CUcontext context) {
        // Test if events can be collected together
        CUpti_EventGroup testGroup;
        CUptiResult result = cuptiEventGroupCreate(context, &testGroup, 0);
        
        for (auto eventId : events) {
            result = cuptiEventGroupAddEvent(testGroup, eventId);
            if (result != CUPTI_SUCCESS) {
                std::cout << "Event " << eventId << " cannot be added to group" << std::endl;
            }
        }
        
        cuptiEventGroupDestroy(testGroup);
    }
};
```

## Next Steps

- Apply SASS metrics analysis to understand your kernel performance at the lowest level
- Experiment with different compiler optimizations and measure their SASS-level impact
- Correlate SASS metrics with high-level algorithm performance
- Develop custom analysis tools for your specific instruction patterns
- Integrate SASS analysis into your optimization workflow for detailed performance tuning 