# CUPTI PC 采样启动/停止控制教程

> GitHub 仓库和完整教程可在 <https://github.com/eunomia-bpf/cupti-tutorial> 获取。

## 简介

CUPTI PC 采样启动/停止控制示例演示如何使用启动和停止命令精确控制程序计数器 (PC) 采样会话。本教程向您展示如何实现对 PC 采样何时发生的细粒度控制，允许您分析应用程序的特定阶段或响应运行时条件。

## 您将学到什么

- 如何为 PC 采样实现启动/停止控制
- 理解精确的分析会话管理
- 基于应用程序状态控制采样
- 实现条件和触发分析
- 使用 PC 采样分析特定执行阶段

## 理解 PC 采样控制

带启动/停止控制的 PC 采样提供几个优势：

1. **目标分析**：仅采样执行的特定阶段
2. **减少开销**：通过选择性采样最小化分析影响
3. **条件分析**：基于运行时条件启动/停止
4. **阶段关联**：将性能数据与应用程序阶段关联
5. **资源效率**：为关键时期保存采样资源

## 关键概念

### 采样控制模式

#### 手动控制
由应用程序逻辑触发的显式启动和停止命令

#### 条件控制
基于性能阈值或条件的自动启动/停止

#### 事件驱动控制
由特定 CUDA 事件或 API 调用触发的启动/停止

#### 基于时间的控制
周期性采样窗口或持续时间限制的会话

### 控制粒度

- **应用程序级别**：控制整个应用程序阶段的采样
- **函数级别**：围绕特定函数调用启动/停止
- **内核级别**：采样单个内核执行
- **线程级别**：控制每线程或每上下文的采样

## 构建示例

### 先决条件

- 带 CUPTI 的 CUDA 工具包
- 支持 PC 采样的 GPU
- 具有可识别执行阶段的应用程序

### 构建过程

```bash
cd pc_sampling_start_stop
make
```

这会创建演示受控 PC 采样的 `pc_sampling_start_stop` 可执行文件。

## 运行示例

### 基本执行

```bash
./pc_sampling_start_stop
```

### 示例输出

```
=== PC 采样启动/停止控制 ===

应用程序阶段分析：
阶段 1：初始化
  采样：禁用
  持续时间：2.3ms
  原因：设置阶段 - 无计算

阶段 2：数据加载
  采样：在 t=2.3ms 启用
  持续时间：15.7ms
  收集的 PC 采样：1,247
  热点排名：
    - memcpy 操作：45.2%
    - 数据验证：23.1%
    - 缓冲区设置：18.9%

阶段 3：核心计算
  采样：在 t=18.0ms 启用
  持续时间：124.5ms
  收集的 PC 采样：8,956
  热点排名：
    - 矩阵乘法：78.3%
    - 归约操作：12.4%
    - 同步：5.1%
  采样：在 t=142.5ms 禁用

阶段 4：结果处理
  采样：禁用
  持续时间：8.2ms
  原因：I/O 绑定阶段

阶段 5：关键算法
  采样：在 t=150.7ms 启用（由条件触发）
  持续时间：67.3ms
  收集的 PC 采样：4,832
  热点排名：
    - 优化内核：89.7%
    - 收敛检查：6.2%
  采样：在 t=218.0ms 禁用

总采样时间：207.5ms（计算阶段的 89.1% 覆盖率）
收集的总采样：15,035
平均采样率：72.5 采样/ms
```

## 代码架构

### 采样控制器

```cpp
class PCSamplingController {
private:
    struct SamplingSession {
        std::string phaseName;
        uint64_t startTime;
        uint64_t endTime;
        uint32_t samplesCollected;
        bool isActive;
    };
    
    std::vector<SamplingSession> sessions;
    std::atomic<bool> samplingActive;
    std::mutex controlMutex;
    CUpti_PCSamplingConfig config;

public:
    void initializeSampling();
    void startSampling(const std::string& phaseName);
    void stopSampling();
    void conditionalStart(std::function<bool()> condition);
    void conditionalStop(std::function<bool()> condition);
    void generatePhaseReport();
};

void PCSamplingController::startSampling(const std::string& phaseName) {
    std::lock_guard<std::mutex> lock(controlMutex);
    
    if (samplingActive.load()) {
        std::cout << "警告：采样已激活，停止之前的会话" << std::endl;
        stopSamplingInternal();
    }
    
    SamplingSession session;
    session.phaseName = phaseName;
    session.startTime = getCurrentTimestamp();
    session.isActive = true;
    
    // 启动 PC 采样
    CUPTI_CALL(cuptiPCSamplingStart(context, &config));
    
    samplingActive = true;
    sessions.push_back(session);
    
    std::cout << "为阶段启动 PC 采样：" << phaseName 
             << " 在 t=" << session.startTime << "ms" << std::endl;
}

void PCSamplingController::stopSampling() {
    std::lock_guard<std::mutex> lock(controlMutex);
    stopSamplingInternal();
}

void PCSamplingController::stopSamplingInternal() {
    if (!samplingActive.load()) {
        return;
    }
    
    // 停止 PC 采样
    CUPTI_CALL(cuptiPCSamplingStop(context));
    
    // 更新最后一个会话
    if (!sessions.empty()) {
        auto& lastSession = sessions.back();
        lastSession.endTime = getCurrentTimestamp();
        lastSession.isActive = false;
        
        // 从此会话收集采样
        lastSession.samplesCollected = collectPCSamples();
        
        std::cout << "为阶段停止 PC 采样：" << lastSession.phaseName
                 << " 在 t=" << lastSession.endTime << "ms" << std::endl;
        std::cout << "  收集的采样：" << lastSession.samplesCollected << std::endl;
    }
    
    samplingActive = false;
}
```

### 条件控制系统

```cpp
class ConditionalController {
private:
    struct Condition {
        std::function<bool()> predicate;
        std::string description;
        bool isStartCondition;
        bool isTriggered;
    };
    
    std::vector<Condition> conditions;
    std::thread monitoringThread;
    std::atomic<bool> monitoring;

public:
    void addStartCondition(std::function<bool()> condition, const std::string& desc) {
        Condition cond;
        cond.predicate = condition;
        cond.description = desc;
        cond.isStartCondition = true;
        cond.isTriggered = false;
        conditions.push_back(cond);
    }
    
    void addStopCondition(std::function<bool()> condition, const std::string& desc) {
        Condition cond;
        cond.predicate = condition;
        cond.description = desc;
        cond.isStartCondition = false;
        cond.isTriggered = false;
        conditions.push_back(cond);
    }
    
    void startMonitoring(PCSamplingController* controller) {
        monitoring = true;
        monitoringThread = std::thread([this, controller]() {
            while (monitoring) {
                for (auto& condition : conditions) {
                    if (!condition.isTriggered && condition.predicate()) {
                        condition.isTriggered = true;
                        
                        if (condition.isStartCondition) {
                            std::cout << "触发启动条件：" << condition.description << std::endl;
                            controller->startSampling(condition.description);
                        } else {
                            std::cout << "触发停止条件：" << condition.description << std::endl;
                            controller->stopSampling();
                        }
                    }
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        });
    }
};
```

### 实际使用示例

```cpp
int main() {
    PCSamplingController controller;
    ConditionalController condController;
    
    // 初始化 PC 采样
    controller.initializeSampling();
    
    // 设置条件控制
    condController.addStartCondition([]() {
        return getCurrentMemoryUsage() > 1024 * 1024 * 1024; // 1GB
    }, "高内存使用");
    
    condController.addStopCondition([]() {
        return getCurrentGPUUtilization() < 0.3; // 30%
    }, "低 GPU 利用率");
    
    // 启动条件监控
    condController.startMonitoring(&controller);
    
    // 应用程序阶段
    std::cout << "开始应用程序执行..." << std::endl;
    
    // 阶段 1：手动控制的初始化
    controller.startSampling("应用程序初始化");
    initializeApplication();
    controller.stopSampling();
    
    // 阶段 2：自动条件控制的主要计算
    runMainComputations(); // 条件将自动控制采样
    
    // 阶段 3：手动控制的清理
    controller.startSampling("清理阶段");
    cleanupApplication();
    controller.stopSampling();
    
    // 生成报告
    controller.generatePhaseReport();
    
    return 0;
}
```

## 高级控制策略

### 性能阈值采样

```cpp
class PerformanceThresholdController {
private:
    double gpuUtilizationThreshold;
    double memoryBandwidthThreshold;
    std::chrono::milliseconds samplingDuration;

public:
    void setupThresholds(double gpuUtil, double memBW, int durationMs) {
        gpuUtilizationThreshold = gpuUtil;
        memoryBandwidthThreshold = memBW;
        samplingDuration = std::chrono::milliseconds(durationMs);
    }
    
    bool shouldStartSampling() {
        double currentGPUUtil = getCurrentGPUUtilization();
        double currentMemBW = getCurrentMemoryBandwidth();
        
        return (currentGPUUtil > gpuUtilizationThreshold) || 
               (currentMemBW > memoryBandwidthThreshold);
    }
    
    bool shouldStopSampling(const std::chrono::time_point<std::chrono::steady_clock>& startTime) {
        auto now = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - startTime);
        
        return duration >= samplingDuration;
    }
};
```

### 多阶段分析

```cpp
class MultiPhaseAnalyzer {
private:
    struct Phase {
        std::string name;
        std::vector<std::string> kernelPatterns;
        double expectedDuration;
        bool isOptimizationTarget;
    };
    
    std::vector<Phase> phases;
    PCSamplingController* controller;

public:
    void definePhase(const std::string& name, 
                    const std::vector<std::string>& patterns,
                    double duration, bool isTarget) {
        Phase phase;
        phase.name = name;
        phase.kernelPatterns = patterns;
        phase.expectedDuration = duration;
        phase.isOptimizationTarget = isTarget;
        phases.push_back(phase);
    }
    
    void monitorPhases() {
        for (const auto& phase : phases) {
            if (phase.isOptimizationTarget) {
                std::cout << "开始监控关键阶段：" << phase.name << std::endl;
                controller->startSampling(phase.name);
                
                // 等待阶段完成
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(static_cast<int>(phase.expectedDuration * 1000))
                );
                
                controller->stopSampling();
            }
        }
    }
};
```

## 性能分析和报告

### 详细的阶段分析

```cpp
void PCSamplingController::generatePhaseReport() {
    std::cout << "\n=== 详细阶段分析报告 ===" << std::endl;
    
    uint64_t totalSamples = 0;
    uint64_t totalDuration = 0;
    
    for (const auto& session : sessions) {
        uint64_t duration = session.endTime - session.startTime;
        totalSamples += session.samplesCollected;
        totalDuration += duration;
        
        std::cout << "\n阶段：" << session.phaseName << std::endl;
        std::cout << "  持续时间：" << duration << "ms" << std::endl;
        std::cout << "  采样数：" << session.samplesCollected << std::endl;
        
        if (duration > 0) {
            double samplingRate = static_cast<double>(session.samplesCollected) / duration;
            std::cout << "  采样率：" << samplingRate << " 采样/ms" << std::endl;
        }
        
        // 计算采样密度
        if (session.samplesCollected > 0) {
            double samplingDensity = static_cast<double>(session.samplesCollected) / totalSamples * 100;
            std::cout << "  采样密度：" << samplingDensity << "%" << std::endl;
        }
    }
    
    // 总体统计
    std::cout << "\n=== 总体统计 ===" << std::endl;
    std::cout << "总阶段数：" << sessions.size() << std::endl;
    std::cout << "总采样数：" << totalSamples << std::endl;
    std::cout << "总持续时间：" << totalDuration << "ms" << std::endl;
    
    if (totalDuration > 0) {
        double avgSamplingRate = static_cast<double>(totalSamples) / totalDuration;
        std::cout << "平均采样率：" << avgSamplingRate << " 采样/ms" << std::endl;
    }
}
```

## 最佳实践

### 有效的采样策略

1. **识别关键阶段**：专注于性能关键的代码段
2. **平衡精度和开销**：调整采样频率以平衡详细程度和性能影响
3. **使用条件控制**：让系统自动响应性能变化
4. **记录采样决策**：跟踪为什么启动或停止采样的原因

### 故障排除技巧

```cpp
class SamplingDiagnostics {
public:
    static void checkSamplingCapability() {
        CUdevice device;
        DRIVER_API_CALL(cuCtxGetDevice(&device));
        
        int major, minor;
        DRIVER_API_CALL(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
        DRIVER_API_CALL(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
        
        if (major < 5) {
            std::cout << "警告：GPU 计算能力 " << major << "." << minor 
                     << " 可能不支持 PC 采样" << std::endl;
        }
    }
    
    static void validateSamplingConfiguration(const CUpti_PCSamplingConfig& config) {
        if (config.samplingPeriod < 1000) {
            std::cout << "警告：采样周期过短可能导致高开销" << std::endl;
        }
        
        if (config.maxSamples > 1000000) {
            std::cout << "警告：最大采样数过高可能导致内存问题" << std::endl;
        }
    }
};
```

PC 采样启动/停止控制为 CUDA 应用程序提供了精确的性能分析能力。通过智能地控制何时收集性能数据，您可以专注于最重要的代码段，同时最小化分析开销。这种方法对于复杂的应用程序特别有用，在这些应用程序中，不同的阶段具有不同的性能特征和优化需求。 