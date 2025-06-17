# CUPTI External Correlation Tutorial

> The GitHub repo and complete tutorial is available at <https://github.com/eunomia-bpf/cupti-tutorial>.

## Introduction

External correlation allows you to correlate CUDA activities with high-level application phases or external events. This sample demonstrates how to use CUPTI's external correlation API to track different phases of execution (initialization, computation, cleanup) and correlate them with GPU activities.

## What You'll Learn

- How to use external correlation IDs to track application phases
- Correlating CUDA activities with high-level application events
- Using push/pop external correlation for hierarchical tracking
- Analyzing performance across different application phases
- Building comprehensive phase-based performance analysis

## Key Concepts

### External Correlation
External correlation allows you to:
- **Tag Phases**: Mark different phases of your application
- **Correlate Activities**: Link GPU activities to application phases  
- **Hierarchical Tracking**: Nest correlations for complex applications
- **Performance Analysis**: Analyze performance by application phase

### Push/Pop Model
```cpp
// Start tracking a phase
cuptiActivityPushExternalCorrelationId(
    CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, 
    INITIALIZATION_PHASE_ID);

// Run CUDA operations - they get tagged with this ID
cudaMalloc(...);
cudaMemcpy(...);

// Stop tracking this phase
cuptiActivityPopExternalCorrelationId(
    CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, 
    &id);
```

## Sample Architecture

### Phase Definitions
```cpp
typedef enum ExternalId_st {
    INITIALIZATION_EXTERNAL_ID = 0,  // Memory allocation, setup
    EXECUTION_EXTERNAL_ID = 1,       // Kernel execution, computation
    CLEANUP_EXTERNAL_ID = 2,         // Memory deallocation, cleanup
    MAX_EXTERNAL_ID = 3
} ExternalId;
```

### Correlation Tracking
```cpp
// Map external ID to correlation IDs
static std::map<uint64_t, std::vector<uint32_t>> s_externalCorrelationMap;

void ProcessExternalCorrelation(CUpti_Activity* record) {
    if (record->kind == CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION) {
        CUpti_ActivityExternalCorrelation* extCorr = 
            (CUpti_ActivityExternalCorrelation*)record;
        
        // Store which correlation IDs belong to which external phase
        s_externalCorrelationMap[extCorr->externalId].push_back(
            extCorr->correlationId);
    }
}
```

## Sample Walkthrough

### Phase 1: Initialization
```cpp
// Push initialization phase ID
cuptiActivityPushExternalCorrelationId(
    CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, 
    INITIALIZATION_EXTERNAL_ID);

// These operations get tagged with INITIALIZATION_EXTERNAL_ID
cuCtxCreate(&context, 0, device);
cudaMalloc((void**)&pDeviceA, size);
cudaMalloc((void**)&pDeviceB, size);
cudaMalloc((void**)&pDeviceC, size);
cudaMemcpy(pDeviceA, pHostA, size, cudaMemcpyHostToDevice);
cudaMemcpy(pDeviceB, pHostB, size, cudaMemcpyHostToDevice);

// Pop the phase ID
cuptiActivityPopExternalCorrelationId(
    CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, &id);
```

### Phase 2: Execution
```cpp
// Push execution phase ID
cuptiActivityPushExternalCorrelationId(
    CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, 
    EXECUTION_EXTERNAL_ID);

// These operations get tagged with EXECUTION_EXTERNAL_ID
VectorAdd<<<blocksPerGrid, threadsPerBlock>>>(pDeviceA, pDeviceB, pDeviceC, N);
cuCtxSynchronize();
cudaMemcpy(pHostC, pDeviceC, size, cudaMemcpyDeviceToHost);

// Pop the phase ID
cuptiActivityPopExternalCorrelationId(
    CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, &id);
```

### Phase 3: Cleanup
```cpp
// Push cleanup phase ID
cuptiActivityPushExternalCorrelationId(
    CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, 
    CLEANUP_EXTERNAL_ID);

// These operations get tagged with CLEANUP_EXTERNAL_ID
cudaFree(pDeviceA);
cudaFree(pDeviceB);
cudaFree(pDeviceC);

// Pop the phase ID
cuptiActivityPopExternalCorrelationId(
    CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, &id);
```

## Analysis and Reporting

### Phase Summary
```cpp
void ShowExternalCorrelation() {
    printf("\n=== PHASE ANALYSIS ===\n");
    
    for (auto& [externalId, correlationIds] : s_externalCorrelationMap) {
        const char* phaseName = GetPhaseName(externalId);
        
        printf("Phase: %s (ID: %llu)\n", phaseName, externalId);
        printf("  Operations: %zu\n", correlationIds.size());
        printf("  Correlation IDs: ");
        
        for (auto correlationId : correlationIds) {
            printf("%u ", correlationId);
        }
        printf("\n");
    }
}

const char* GetPhaseName(uint64_t externalId) {
    switch (externalId) {
        case INITIALIZATION_EXTERNAL_ID: return "INITIALIZATION";
        case EXECUTION_EXTERNAL_ID: return "EXECUTION";
        case CLEANUP_EXTERNAL_ID: return "CLEANUP";
        default: return "UNKNOWN";
    }
}
```

## Building and Running

```bash
cd cupti_external_correlation
make
./cupti_external_correlation
```

## Sample Output

```
=== PHASE ANALYSIS ===
Phase: INITIALIZATION (ID: 0)
  Operations: 5
  Correlation IDs: 1 2 3 4 5

Phase: EXECUTION (ID: 1)  
  Operations: 3
  Correlation IDs: 6 7 8

Phase: CLEANUP (ID: 2)
  Operations: 3
  Correlation IDs: 9 10 11

Activity records processed: 28
```

## Advanced Use Cases

### Hierarchical Correlation
```cpp
class HierarchicalTracker {
public:
    void TrackNestedPhases() {
        // Main application phase
        PushPhase(APPLICATION_START_ID);
        
        // Nested initialization phase
        PushPhase(MEMORY_INITIALIZATION_ID);
        AllocateMemory();
        PopPhase();
        
        // Nested computation phases
        for (int i = 0; i < iterations; i++) {
            PushPhase(ITERATION_START_ID + i);
            RunComputation(i);
            PopPhase();
        }
        
        // Nested cleanup phase
        PushPhase(MEMORY_CLEANUP_ID);
        FreeMemory();
        PopPhase();
        
        PopPhase(); // End main phase
    }
    
private:
    std::stack<uint64_t> phaseStack;
    
    void PushPhase(uint64_t phaseId) {
        cuptiActivityPushExternalCorrelationId(
            CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, phaseId);
        phaseStack.push(phaseId);
    }
    
    void PopPhase() {
        if (!phaseStack.empty()) {
            uint64_t id;
            cuptiActivityPopExternalCorrelationId(
                CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, &id);
            phaseStack.pop();
        }
    }
};
```

### Performance Analysis by Phase
```cpp
class PhasePerformanceAnalyzer {
private:
    struct PhaseMetrics {
        std::string name;
        double totalTime;
        size_t operationCount;
        double averageTime;
        std::vector<uint32_t> correlationIds;
    };
    
    std::map<uint64_t, PhaseMetrics> phaseMetrics;

public:
    void AnalyzePhasePerformance() {
        for (auto& [externalId, correlationIds] : s_externalCorrelationMap) {
            PhaseMetrics metrics;
            metrics.name = GetPhaseName(externalId);
            metrics.operationCount = correlationIds.size();
            metrics.correlationIds = correlationIds;
            
            // Calculate timing metrics
            metrics.totalTime = CalculatePhaseTotalTime(correlationIds);
            metrics.averageTime = metrics.totalTime / metrics.operationCount;
            
            phaseMetrics[externalId] = metrics;
        }
        
        PrintPhaseAnalysis();
    }
    
private:
    void PrintPhaseAnalysis() {
        printf("\n=== PHASE PERFORMANCE ANALYSIS ===\n");
        printf("Phase\t\tOperations\tTotal Time\tAvg Time\tPercentage\n");
        printf("-----\t\t----------\t----------\t--------\t----------\n");
        
        double totalApplicationTime = CalculateTotalApplicationTime();
        
        for (auto& [externalId, metrics] : phaseMetrics) {
            double percentage = (metrics.totalTime / totalApplicationTime) * 100.0;
            
            printf("%-15s\t%zu\t\t%.3f ms\t\t%.3f ms\t%.1f%%\n",
                   metrics.name.c_str(),
                   metrics.operationCount,
                   metrics.totalTime,
                   metrics.averageTime,
                   percentage);
        }
    }
};
```

## Real-World Applications

### ML Training Pipeline Tracking
```cpp
class MLPipelineTracker {
public:
    void TrackTrainingPipeline() {
        for (int epoch = 0; epoch < numEpochs; epoch++) {
            // Track entire epoch
            PushPhase(EPOCH_START + epoch);
            
            // Data loading phase
            PushPhase(DATA_LOADING_PHASE);
            LoadBatchData();
            PopPhase();
            
            // Forward pass phase
            PushPhase(FORWARD_PASS_PHASE);
            RunForwardPass();
            PopPhase();
            
            // Backward pass phase
            PushPhase(BACKWARD_PASS_PHASE);
            RunBackwardPass();
            PopPhase();
            
            // Parameter update phase
            PushPhase(PARAMETER_UPDATE_PHASE);
            UpdateParameters();
            PopPhase();
            
            PopPhase(); // End epoch
        }
    }
};
```

### Scientific Simulation Tracking
```cpp
class SimulationTracker {
public:
    void TrackSimulation() {
        // Initialization phase
        PushPhase(SIM_INITIALIZATION);
        InitializeSimulation();
        PopPhase();
        
        // Time step iterations
        for (int step = 0; step < timeSteps; step++) {
            PushPhase(TIME_STEP_BASE + step);
            
            // Physics computation
            PushPhase(PHYSICS_COMPUTATION);
            ComputePhysics();
            PopPhase();
            
            // Boundary conditions
            PushPhase(BOUNDARY_CONDITIONS);
            ApplyBoundaryConditions();
            PopPhase();
            
            // Data output (if needed)
            if (step % outputFrequency == 0) {
                PushPhase(DATA_OUTPUT);
                OutputData(step);
                PopPhase();
            }
            
            PopPhase(); // End time step
        }
        
        // Finalization phase
        PushPhase(SIM_FINALIZATION);
        FinalizeSimulation();
        PopPhase();
    }
};
```

## Best Practices

### Proper Phase Management
```cpp
class PhaseManager {
private:
    std::stack<uint64_t> activePhases;
    std::map<uint64_t, std::string> phaseNames;

public:
    void RegisterPhase(uint64_t phaseId, const std::string& name) {
        phaseNames[phaseId] = name;
    }
    
    void StartPhase(uint64_t phaseId) {
        if (phaseNames.find(phaseId) == phaseNames.end()) {
            printf("WARNING: Unregistered phase ID %llu\n", phaseId);
        }
        
        cuptiActivityPushExternalCorrelationId(
            CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, phaseId);
        activePhases.push(phaseId);
    }
    
    void EndPhase() {
        if (activePhases.empty()) {
            printf("ERROR: No active phase to end\n");
            return;
        }
        
        uint64_t id;
        cuptiActivityPopExternalCorrelationId(
            CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, &id);
        
        uint64_t expectedId = activePhases.top();
        activePhases.pop();
        
        if (id != expectedId) {
            printf("WARNING: Phase ID mismatch. Expected %llu, got %llu\n", 
                   expectedId, id);
        }
    }
    
    bool HasActivePhases() const {
        return !activePhases.empty();
    }
};
```

## Integration with Profiling Tools

### JSON Export for Visualization
```cpp
class PhaseDataExporter {
public:
    void ExportPhaseData(const std::string& filename) {
        json exportData;
        exportData["metadata"]["tool"] = "cupti_external_correlation";
        exportData["metadata"]["timestamp"] = GetCurrentTimestamp();
        
        for (auto& [externalId, correlationIds] : s_externalCorrelationMap) {
            json phaseData;
            phaseData["externalId"] = externalId;
            phaseData["phaseName"] = GetPhaseName(externalId);
            phaseData["operationCount"] = correlationIds.size();
            phaseData["correlationIds"] = correlationIds;
            
            exportData["phases"].push_back(phaseData);
        }
        
        std::ofstream file(filename);
        file << exportData.dump(2);
    }
};
```

## Use Cases

- **Application Profiling**: Track performance across different application phases
- **Pipeline Analysis**: Analyze ML training or data processing pipelines
- **Scientific Computing**: Track simulation phases and identify bottlenecks
- **Multi-stage Applications**: Correlate GPU activities with high-level application logic
- **Performance Debugging**: Identify which application phases consume the most GPU time

## Next Steps

- Implement hierarchical phase tracking for complex applications
- Add automatic phase detection based on API patterns
- Integrate phase analysis with existing profiling workflows
- Build visualization tools for phase-based performance analysis
- Develop phase-specific optimization recommendations

External correlation provides a powerful way to bridge the gap between high-level application logic and low-level GPU activities, enabling more meaningful performance analysis. 