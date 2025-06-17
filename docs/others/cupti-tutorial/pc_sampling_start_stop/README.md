# CUPTI PC Sampling Start/Stop Control Tutorial

> The GitHub repo and complete tutorial is available at <https://github.com/eunomia-bpf/cupti-tutorial>.

## Introduction

The CUPTI PC Sampling Start/Stop Control sample demonstrates how to precisely control Program Counter (PC) sampling sessions with start and stop commands. This tutorial shows you how to implement fine-grained control over when PC sampling occurs, allowing you to profile specific phases of your application or respond to runtime conditions.

## What You'll Learn

- How to implement start/stop control for PC sampling
- Understanding precise profiling session management
- Controlling sampling based on application state
- Implementing conditional and triggered profiling
- Analyzing specific execution phases with PC sampling

## Understanding PC Sampling Control

PC sampling with start/stop control provides several advantages:

1. **Targeted Profiling**: Sample only specific phases of execution
2. **Reduced Overhead**: Minimize profiling impact by sampling selectively
3. **Conditional Analysis**: Start/stop based on runtime conditions
4. **Phase Correlation**: Correlate performance data with application phases
5. **Resource Efficiency**: Conserve sampling resources for critical periods

## Key Concepts

### Sampling Control Modes

#### Manual Control
Explicit start and stop commands triggered by application logic

#### Conditional Control
Automated start/stop based on performance thresholds or conditions

#### Event-Driven Control
Start/stop triggered by specific CUDA events or API calls

#### Time-Based Control
Periodic sampling windows or duration-limited sessions

### Control Granularity

- **Application-Level**: Control sampling for entire application phases
- **Function-Level**: Start/stop around specific function calls
- **Kernel-Level**: Sample individual kernel executions
- **Thread-Level**: Control sampling per-thread or per-context

## Building the Sample

### Prerequisites

- CUDA Toolkit with CUPTI
- GPU with PC sampling support
- Applications with identifiable execution phases

### Build Process

```bash
cd pc_sampling_start_stop
make
```

This creates the `pc_sampling_start_stop` executable demonstrating controlled PC sampling.

## Running the Sample

### Basic Execution

```bash
./pc_sampling_start_stop
```

### Sample Output

```
=== PC Sampling Start/Stop Control ===

Application Phase Analysis:
Phase 1: Initialization
  Sampling: DISABLED
  Duration: 2.3ms
  Reason: Setup phase - no computation

Phase 2: Data Loading
  Sampling: ENABLED at t=2.3ms
  Duration: 15.7ms
  PC Samples Collected: 1,247
  Top Hotspots:
    - memcpy operations: 45.2%
    - data validation: 23.1%
    - buffer setup: 18.9%

Phase 3: Core Computation
  Sampling: ENABLED at t=18.0ms
  Duration: 124.5ms
  PC Samples Collected: 8,956
  Top Hotspots:
    - matrix multiplication: 78.3%
    - reduction operations: 12.4%
    - synchronization: 5.1%
  Sampling: DISABLED at t=142.5ms

Phase 4: Result Processing
  Sampling: DISABLED
  Duration: 8.2ms
  Reason: I/O bound phase

Phase 5: Critical Algorithm
  Sampling: ENABLED at t=150.7ms (triggered by condition)
  Duration: 67.3ms
  PC Samples Collected: 4,832
  Top Hotspots:
    - optimization kernel: 89.7%
    - convergence check: 6.2%
  Sampling: DISABLED at t=218.0ms

Total Sampling Time: 207.5ms (89.1% coverage of compute phases)
Total Samples Collected: 15,035
Average Sample Rate: 72.5 samples/ms
```

## Code Architecture

### Sampling Controller

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
        std::cout << "Warning: Sampling already active, stopping previous session" << std::endl;
        stopSamplingInternal();
    }
    
    SamplingSession session;
    session.phaseName = phaseName;
    session.startTime = getCurrentTimestamp();
    session.isActive = true;
    
    // Start PC sampling
    CUPTI_CALL(cuptiPCSamplingStart(context, &config));
    
    samplingActive = true;
    sessions.push_back(session);
    
    std::cout << "PC Sampling started for phase: " << phaseName 
             << " at t=" << session.startTime << "ms" << std::endl;
}

void PCSamplingController::stopSampling() {
    std::lock_guard<std::mutex> lock(controlMutex);
    stopSamplingInternal();
}

void PCSamplingController::stopSamplingInternal() {
    if (!samplingActive.load()) {
        return;
    }
    
    // Stop PC sampling
    CUPTI_CALL(cuptiPCSamplingStop(context));
    
    // Update the last session
    if (!sessions.empty()) {
        auto& lastSession = sessions.back();
        lastSession.endTime = getCurrentTimestamp();
        lastSession.isActive = false;
        
        // Collect samples from this session
        lastSession.samplesCollected = collectPCSamples();
        
        std::cout << "PC Sampling stopped for phase: " << lastSession.phaseName
                 << " at t=" << lastSession.endTime << "ms" << std::endl;
        std::cout << "  Samples collected: " << lastSession.samplesCollected << std::endl;
    }
    
    samplingActive = false;
}
```

### Conditional Sampling

```cpp
class ConditionalSamplingManager {
private:
    struct SamplingCondition {
        std::string name;
        std::function<bool()> startCondition;
        std::function<bool()> stopCondition;
        bool isMonitoring;
        std::thread monitorThread;
    };
    
    std::vector<SamplingCondition> conditions;
    PCSamplingController& controller;
    std::atomic<bool> monitoringActive;

public:
    ConditionalSamplingManager(PCSamplingController& ctrl) : controller(ctrl), monitoringActive(false) {}
    
    void addCondition(const std::string& name,
                     std::function<bool()> startCond,
                     std::function<bool()> stopCond) {
        SamplingCondition condition;
        condition.name = name;
        condition.startCondition = startCond;
        condition.stopCondition = stopCond;
        condition.isMonitoring = false;
        
        conditions.push_back(condition);
    }
    
    void startMonitoring() {
        monitoringActive = true;
        
        for (auto& condition : conditions) {
            condition.isMonitoring = true;
            condition.monitorThread = std::thread([this, &condition]() {
                monitorCondition(condition);
            });
        }
    }
    
    void stopMonitoring() {
        monitoringActive = false;
        
        for (auto& condition : conditions) {
            condition.isMonitoring = false;
            if (condition.monitorThread.joinable()) {
                condition.monitorThread.join();
            }
        }
    }
    
private:
    void monitorCondition(SamplingCondition& condition) {
        bool samplingForThisCondition = false;
        
        while (condition.isMonitoring && monitoringActive) {
            if (!samplingForThisCondition && condition.startCondition()) {
                controller.startSampling(condition.name);
                samplingForThisCondition = true;
                std::cout << "Conditional sampling started: " << condition.name << std::endl;
            } else if (samplingForThisCondition && condition.stopCondition()) {
                controller.stopSampling();
                samplingForThisCondition = false;
                std::cout << "Conditional sampling stopped: " << condition.name << std::endl;
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
};
```

### Performance-Triggered Sampling

```cpp
class PerformanceTriggeredSampling {
private:
    struct PerformanceMetrics {
        double gpuUtilization;
        double memoryBandwidth;
        double computeEfficiency;
        uint64_t timestamp;
    };
    
    std::queue<PerformanceMetrics> metricsHistory;
    PCSamplingController& controller;
    double utilizationThreshold;
    double bandwidthThreshold;
    bool autoSamplingEnabled;

public:
    PerformanceTriggeredSampling(PCSamplingController& ctrl) 
        : controller(ctrl), utilizationThreshold(0.7), bandwidthThreshold(0.5), autoSamplingEnabled(false) {}
    
    void enableAutoSampling(double utilThreshold, double bwThreshold) {
        utilizationThreshold = utilThreshold;
        bandwidthThreshold = bwThreshold;
        autoSamplingEnabled = true;
    }
    
    void updateMetrics(double utilization, double bandwidth, double efficiency) {
        PerformanceMetrics metrics;
        metrics.gpuUtilization = utilization;
        metrics.memoryBandwidth = bandwidth;
        metrics.computeEfficiency = efficiency;
        metrics.timestamp = getCurrentTimestamp();
        
        metricsHistory.push(metrics);
        
        // Keep only recent history
        while (metricsHistory.size() > 100) {
            metricsHistory.pop();
        }
        
        if (autoSamplingEnabled) {
            evaluateSamplingTriggers(metrics);
        }
    }
    
private:
    void evaluateSamplingTriggers(const PerformanceMetrics& current) {
        // Trigger sampling during high-utilization periods
        if (current.gpuUtilization > utilizationThreshold && 
            current.memoryBandwidth > bandwidthThreshold) {
            
            if (!controller.isSamplingActive()) {
                controller.startSampling("High Performance Period");
            }
        }
        // Stop sampling during low-utilization periods
        else if (current.gpuUtilization < utilizationThreshold * 0.7) {
            if (controller.isSamplingActive()) {
                controller.stopSampling();
            }
        }
        
        // Trigger sampling when efficiency drops (potential issue)
        if (current.computeEfficiency < 0.6 && !controller.isSamplingActive()) {
            controller.startSampling("Low Efficiency Investigation");
        }
    }
};
```

## Advanced Control Strategies

### Kernel-Level Sampling Control

```cpp
class KernelSamplingController {
private:
    std::set<std::string> targetKernels;
    std::map<std::string, uint32_t> kernelSampleCounts;
    PCSamplingController& controller;

public:
    KernelSamplingController(PCSamplingController& ctrl) : controller(ctrl) {}
    
    void addTargetKernel(const std::string& kernelName) {
        targetKernels.insert(kernelName);
    }
    
    void onKernelLaunch(const std::string& kernelName) {
        if (targetKernels.find(kernelName) != targetKernels.end()) {
            controller.startSampling("Kernel: " + kernelName);
        }
    }
    
    void onKernelComplete(const std::string& kernelName) {
        if (targetKernels.find(kernelName) != targetKernels.end()) {
            controller.stopSampling();
            kernelSampleCounts[kernelName]++;
        }
    }
    
    void generateKernelReport() {
        std::cout << "Kernel Sampling Summary:" << std::endl;
        for (const auto& [kernelName, count] : kernelSampleCounts) {
            std::cout << "  " << kernelName << ": sampled " << count << " times" << std::endl;
        }
    }
};
```

### Time-Window Sampling

```cpp
class TimeWindowSampling {
private:
    struct SamplingWindow {
        uint64_t startTime;
        uint64_t duration;
        uint64_t interval;
        bool isActive;
        std::thread windowThread;
    };
    
    std::vector<SamplingWindow> windows;
    PCSamplingController& controller;
    std::atomic<bool> windowingActive;

public:
    TimeWindowSampling(PCSamplingController& ctrl) : controller(ctrl), windowingActive(false) {}
    
    void addPeriodicWindow(uint64_t durationMs, uint64_t intervalMs) {
        SamplingWindow window;
        window.duration = durationMs;
        window.interval = intervalMs;
        window.isActive = false;
        
        windows.push_back(window);
    }
    
    void startWindowedSampling() {
        windowingActive = true;
        
        for (auto& window : windows) {
            window.isActive = true;
            window.windowThread = std::thread([this, &window]() {
                executeWindowedSampling(window);
            });
        }
    }
    
    void stopWindowedSampling() {
        windowingActive = false;
        
        for (auto& window : windows) {
            window.isActive = false;
            if (window.windowThread.joinable()) {
                window.windowThread.join();
            }
        }
    }
    
private:
    void executeWindowedSampling(SamplingWindow& window) {
        while (window.isActive && windowingActive) {
            // Start sampling for this window
            controller.startSampling("Time Window");
            
            // Sample for the specified duration
            std::this_thread::sleep_for(std::chrono::milliseconds(window.duration));
            
            // Stop sampling
            controller.stopSampling();
            
            // Wait for the interval period
            std::this_thread::sleep_for(std::chrono::milliseconds(window.interval - window.duration));
        }
    }
};
```

## Real-World Applications

### Iterative Algorithm Profiling

```cpp
void profileIterativeAlgorithm() {
    PCSamplingController controller;
    ConditionalSamplingManager condManager(controller);
    
    // Set up conditions for sampling
    condManager.addCondition("Convergence Issues",
        []() { return getCurrentError() > ERROR_THRESHOLD; },  // Start condition
        []() { return getCurrentError() < ERROR_THRESHOLD; }   // Stop condition
    );
    
    condManager.addCondition("High Iteration Count",
        []() { return getCurrentIteration() > 1000; },         // Start condition
        []() { return getCurrentIteration() > 1500; }          // Stop condition (timeout)
    );
    
    condManager.startMonitoring();
    
    // Run iterative algorithm
    for (int i = 0; i < maxIterations; i++) {
        performIteration();
        
        // Manual control for specific phases
        if (i % 100 == 0) {
            controller.startSampling("Checkpoint Iteration " + std::to_string(i));
            performDetailedAnalysis();
            controller.stopSampling();
        }
    }
    
    condManager.stopMonitoring();
    controller.generatePhaseReport();
}
```

### Multi-Phase Application Analysis

```cpp
void analyzeMultiPhaseApplication() {
    PCSamplingController controller;
    PerformanceTriggeredSampling perfSampling(controller);
    
    // Enable performance-triggered sampling
    perfSampling.enableAutoSampling(0.8, 0.6); // 80% GPU, 60% bandwidth thresholds
    
    // Phase 1: Data preprocessing
    // (No manual sampling - let performance triggers handle it)
    preprocessData();
    
    // Phase 2: Core computation (always sample)
    controller.startSampling("Core Computation");
    performCoreComputation();
    controller.stopSampling();
    
    // Phase 3: Optimization loop (conditional sampling)
    for (int iter = 0; iter < optimizationIterations; iter++) {
        if (iter % 10 == 0 || isConvergenceSlowing()) {
            controller.startSampling("Optimization Iter " + std::to_string(iter));
        }
        
        performOptimizationStep();
        
        // Update performance metrics for auto-sampling
        double util = measureGPUUtilization();
        double bw = measureMemoryBandwidth();
        double eff = measureComputeEfficiency();
        perfSampling.updateMetrics(util, bw, eff);
        
        if ((iter % 10 == 9) || hasConverged()) {
            controller.stopSampling();
        }
    }
    
    controller.generatePhaseReport();
}
```

## Performance Analysis and Reporting

### Phase-Based Analysis

```cpp
class PhaseAnalyzer {
public:
    void analyzePhasePerformance(const std::vector<SamplingSession>& sessions) {
        std::cout << "Phase Performance Analysis:" << std::endl;
        
        for (const auto& session : sessions) {
            double duration = session.endTime - session.startTime;
            double sampleRate = session.samplesCollected / duration;
            
            std::cout << "Phase: " << session.phaseName << std::endl;
            std::cout << "  Duration: " << duration << "ms" << std::endl;
            std::cout << "  Samples: " << session.samplesCollected << std::endl;
            std::cout << "  Sample Rate: " << sampleRate << " samples/ms" << std::endl;
            
            // Analyze sample distribution
            analyzeSampleDistribution(session);
            
            // Identify hotspots
            identifyPhaseHotspots(session);
        }
    }
    
private:
    void analyzeSampleDistribution(const SamplingSession& session) {
        // Analyze how samples are distributed across functions/modules
        auto samples = getPCSamplesForSession(session);
        std::map<std::string, uint32_t> functionCounts;
        
        for (const auto& sample : samples) {
            std::string functionName = getFunctionName(sample.pc);
            functionCounts[functionName]++;
        }
        
        std::cout << "  Top Functions:" << std::endl;
        auto sortedFunctions = sortByCount(functionCounts);
        for (int i = 0; i < std::min(5, (int)sortedFunctions.size()); i++) {
            const auto& [funcName, count] = sortedFunctions[i];
            double percentage = (double)count / session.samplesCollected * 100;
            std::cout << "    " << funcName << ": " << percentage << "%" << std::endl;
        }
    }
};
```

## Next Steps

- Implement start/stop control in your own applications to focus on critical execution phases
- Experiment with different triggering conditions for automated sampling control
- Develop custom control strategies for your specific application patterns
- Integrate with real-time performance monitoring for adaptive sampling
- Combine with other CUPTI features for comprehensive application analysis 