# CUPTI 外部关联教程

> GitHub 仓库和完整教程可在 <https://github.com/eunomia-bpf/cupti-tutorial> 获取。

## 简介

外部关联允许您将 CUDA 活动与高级应用程序阶段或外部事件关联起来。本示例演示了如何使用 CUPTI 的外部关联 API 来跟踪不同的执行阶段（初始化、计算、清理）并将它们与 GPU 活动关联。

## 您将学到的内容

- 如何使用外部关联 ID 跟踪应用程序阶段
- 将 CUDA 活动与高级应用程序事件关联
- 使用推入/弹出外部关联进行分层跟踪
- 分析不同应用程序阶段的性能
- 构建基于阶段的综合性能分析

## 关键概念

### 外部关联

外部关联允许您：
- **标记阶段**：标记应用程序的不同阶段
- **关联活动**：将 GPU 活动链接到应用程序阶段
- **分层跟踪**：为复杂应用程序嵌套关联
- **性能分析**：按应用程序阶段分析性能

### 推入/弹出模型

```cpp
// 开始跟踪一个阶段
cuptiActivityPushExternalCorrelationId(
    CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, 
    INITIALIZATION_PHASE_ID);

// 运行 CUDA 操作 - 它们被标记为此 ID
cudaMalloc(...);
cudaMemcpy(...);

// 停止跟踪此阶段
cuptiActivityPopExternalCorrelationId(
    CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, 
    &id);
```

## 示例架构

### 阶段定义

```cpp
typedef enum ExternalId_st {
    INITIALIZATION_EXTERNAL_ID = 0,  // 内存分配、设置
    EXECUTION_EXTERNAL_ID = 1,       // 内核执行、计算
    CLEANUP_EXTERNAL_ID = 2,         // 内存释放、清理
    MAX_EXTERNAL_ID = 3
} ExternalId;
```

### 关联跟踪

```cpp
// 将外部 ID 映射到关联 ID
static std::map<uint64_t, std::vector<uint32_t>> s_externalCorrelationMap;

void ProcessExternalCorrelation(CUpti_Activity* record) {
    if (record->kind == CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION) {
        CUpti_ActivityExternalCorrelation* extCorr = 
            (CUpti_ActivityExternalCorrelation*)record;
        
        // 存储哪些关联 ID 属于哪个外部阶段
        s_externalCorrelationMap[extCorr->externalId].push_back(
            extCorr->correlationId);
    }
}
```

## 示例演练

### 阶段 1：初始化

```cpp
// 推入初始化阶段 ID
cuptiActivityPushExternalCorrelationId(
    CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, 
    INITIALIZATION_EXTERNAL_ID);

// 这些操作被标记为 INITIALIZATION_EXTERNAL_ID
cuCtxCreate(&context, 0, device);
cudaMalloc((void**)&pDeviceA, size);
cudaMalloc((void**)&pDeviceB, size);
cudaMalloc((void**)&pDeviceC, size);
cudaMemcpy(pDeviceA, pHostA, size, cudaMemcpyHostToDevice);
cudaMemcpy(pDeviceB, pHostB, size, cudaMemcpyHostToDevice);

// 弹出阶段 ID
cuptiActivityPopExternalCorrelationId(
    CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, &id);
```

### 阶段 2：执行

```cpp
// 推入执行阶段 ID
cuptiActivityPushExternalCorrelationId(
    CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, 
    EXECUTION_EXTERNAL_ID);

// 这些操作被标记为 EXECUTION_EXTERNAL_ID
VectorAdd<<<blocksPerGrid, threadsPerBlock>>>(pDeviceA, pDeviceB, pDeviceC, N);
cuCtxSynchronize();
cudaMemcpy(pHostC, pDeviceC, size, cudaMemcpyDeviceToHost);

// 弹出阶段 ID
cuptiActivityPopExternalCorrelationId(
    CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, &id);
```

### 阶段 3：清理

```cpp
// 推入清理阶段 ID
cuptiActivityPushExternalCorrelationId(
    CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, 
    CLEANUP_EXTERNAL_ID);

// 这些操作被标记为 CLEANUP_EXTERNAL_ID
cudaFree(pDeviceA);
cudaFree(pDeviceB);
cudaFree(pDeviceC);

// 弹出阶段 ID
cuptiActivityPopExternalCorrelationId(
    CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, &id);
```

## 分析和报告

### 阶段摘要

```cpp
void ShowExternalCorrelation() {
    printf("\n=== 阶段分析 ===\n");
    
    for (auto& [externalId, correlationIds] : s_externalCorrelationMap) {
        const char* phaseName = GetPhaseName(externalId);
        
        printf("阶段: %s (ID: %llu)\n", phaseName, externalId);
        printf("  操作数: %zu\n", correlationIds.size());
        printf("  关联 ID: ");
        
        for (auto correlationId : correlationIds) {
            printf("%u ", correlationId);
        }
        printf("\n");
    }
}

const char* GetPhaseName(uint64_t externalId) {
    switch (externalId) {
        case INITIALIZATION_EXTERNAL_ID: return "初始化";
        case EXECUTION_EXTERNAL_ID: return "执行";
        case CLEANUP_EXTERNAL_ID: return "清理";
        default: return "未知";
    }
}
```

## 构建和运行

```bash
cd cupti_external_correlation
make
./cupti_external_correlation
```

## 示例输出

```
=== 阶段分析 ===
阶段: 初始化 (ID: 0)
  操作数: 5
  关联 ID: 1 2 3 4 5

阶段: 执行 (ID: 1)  
  操作数: 3
  关联 ID: 6 7 8

阶段: 清理 (ID: 2)
  操作数: 3
  关联 ID: 9 10 11

处理的活动记录: 28
```

## 高级用例

### 分层关联

```cpp
class HierarchicalTracker {
public:
    void TrackNestedPhases() {
        // 主应用程序阶段
        PushPhase(APPLICATION_START_ID);
        
        // 嵌套初始化阶段
        PushPhase(MEMORY_INITIALIZATION_ID);
        AllocateMemory();
        PopPhase();
        
        // 嵌套计算阶段
        PushPhase(COMPUTATION_PHASE_ID);
        
        // 进一步嵌套的内核启动
        PushPhase(KERNEL_LAUNCH_ID);
        LaunchKernels();
        PopPhase();
        
        // 嵌套的内存传输
        PushPhase(MEMORY_TRANSFER_ID);
        TransferResults();
        PopPhase();
        
        PopPhase(); // 结束计算阶段
        PopPhase(); // 结束应用程序阶段
    }
    
private:
    void PushPhase(uint64_t phaseId) {
        cuptiActivityPushExternalCorrelationId(
            CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, phaseId);
    }
    
    void PopPhase() {
        uint64_t id;
        cuptiActivityPopExternalCorrelationId(
            CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, &id);
    }
};
```

### 时间线分析

```cpp
class PhaseTimelineAnalyzer {
private:
    struct PhaseInfo {
        uint64_t externalId;
        std::string phaseName;
        uint64_t startTime;
        uint64_t endTime;
        std::vector<uint32_t> correlationIds;
        uint64_t totalDuration;
    };
    
    std::vector<PhaseInfo> phaseTimeline;

public:
    void BuildPhaseTimeline() {
        // 从外部关联映射构建时间线
        for (const auto& [externalId, correlationIds] : s_externalCorrelationMap) {
            PhaseInfo phase;
            phase.externalId = externalId;
            phase.phaseName = GetPhaseName(externalId);
            phase.correlationIds = correlationIds;
            
            // 查找此阶段的时间范围
            CalculatePhaseTimeRange(phase);
            
            phaseTimeline.push_back(phase);
        }
        
        // 按开始时间排序阶段
        std::sort(phaseTimeline.begin(), phaseTimeline.end(),
                 [](const PhaseInfo& a, const PhaseInfo& b) {
                     return a.startTime < b.startTime;
                 });
    }
    
    void AnalyzePhasePerformance() {
        printf("\n=== 阶段性能分析 ===\n");
        
        uint64_t totalApplicationTime = 0;
        for (const auto& phase : phaseTimeline) {
            totalApplicationTime += phase.totalDuration;
        }
        
        for (const auto& phase : phaseTimeline) {
            double percentage = (double)phase.totalDuration / totalApplicationTime * 100.0;
            
            printf("阶段 %s:\n", phase.phaseName.c_str());
            printf("  持续时间: %llu ns\n", phase.totalDuration);
            printf("  占总时间: %.2f%%\n", percentage);
            printf("  操作数: %zu\n", phase.correlationIds.size());
            printf("  平均操作时间: %llu ns\n", 
                   phase.totalDuration / phase.correlationIds.size());
        }
    }
    
private:
    void CalculatePhaseTimeRange(PhaseInfo& phase) {
        uint64_t minStart = UINT64_MAX;
        uint64_t maxEnd = 0;
        
        // 查找与此阶段关联的所有活动的时间范围
        for (uint32_t correlationId : phase.correlationIds) {
            auto activities = GetActivitiesForCorrelation(correlationId);
            for (const auto& activity : activities) {
                minStart = std::min(minStart, activity.startTime);
                maxEnd = std::max(maxEnd, activity.endTime);
            }
        }
        
        phase.startTime = minStart;
        phase.endTime = maxEnd;
        phase.totalDuration = maxEnd - minStart;
    }
};
```

### 性能瓶颈识别

```cpp
class BottleneckIdentifier {
public:
    void IdentifyBottlenecks() {
        PhaseTimelineAnalyzer analyzer;
        analyzer.BuildPhaseTimeline();
        
        // 识别最慢的阶段
        auto slowestPhase = FindSlowestPhase();
        
        printf("\n=== 瓶颈分析 ===\n");
        printf("最慢阶段: %s\n", slowestPhase.phaseName.c_str());
        printf("持续时间: %llu ns\n", slowestPhase.totalDuration);
        
        // 分析阶段内的操作
        AnalyzePhaseOperations(slowestPhase);
        
        // 提供优化建议
        SuggestOptimizations(slowestPhase);
    }
    
private:
    PhaseInfo FindSlowestPhase() {
        PhaseInfo slowest;
        slowest.totalDuration = 0;
        
        for (const auto& phase : phaseTimeline) {
            if (phase.totalDuration > slowest.totalDuration) {
                slowest = phase;
            }
        }
        
        return slowest;
    }
    
    void AnalyzePhaseOperations(const PhaseInfo& phase) {
        printf("\n阶段 %s 的操作分析:\n", phase.phaseName.c_str());
        
        std::map<std::string, uint64_t> operationTimes;
        
        for (uint32_t correlationId : phase.correlationIds) {
            auto activities = GetActivitiesForCorrelation(correlationId);
            for (const auto& activity : activities) {
                std::string opType = GetOperationType(activity);
                operationTimes[opType] += (activity.endTime - activity.startTime);
            }
        }
        
        // 按时间排序并显示
        std::vector<std::pair<std::string, uint64_t>> sortedOps(
            operationTimes.begin(), operationTimes.end());
        
        std::sort(sortedOps.begin(), sortedOps.end(),
                 [](const auto& a, const auto& b) {
                     return a.second > b.second;
                 });
        
        for (const auto& [opType, duration] : sortedOps) {
            printf("  %s: %llu ns\n", opType.c_str(), duration);
        }
    }
    
    void SuggestOptimizations(const PhaseInfo& phase) {
        printf("\n优化建议:\n");
        
        if (phase.phaseName == "初始化") {
            printf("- 考虑批量内存分配\n");
            printf("- 使用内存池减少分配开销\n");
            printf("- 异步内存传输\n");
        } else if (phase.phaseName == "执行") {
            printf("- 优化内核启动配置\n");
            printf("- 考虑使用 CUDA 流重叠\n");
            printf("- 分析内核占用率\n");
        } else if (phase.phaseName == "清理") {
            printf("- 批量释放内存\n");
            printf("- 异步清理操作\n");
        }
    }
};
```

## 实际应用场景

### 训练循环分析

```cpp
class TrainingLoopAnalyzer {
public:
    void AnalyzeTrainingLoop() {
        for (int epoch = 0; epoch < numEpochs; epoch++) {
            // 推入 epoch 阶段
            PushPhase(EPOCH_START_ID + epoch);
            
            // 数据加载阶段
            PushPhase(DATA_LOADING_ID);
            LoadBatch();
            PopPhase();
            
            // 前向传播阶段
            PushPhase(FORWARD_PASS_ID);
            ForwardPass();
            PopPhase();
            
            // 反向传播阶段
            PushPhase(BACKWARD_PASS_ID);
            BackwardPass();
            PopPhase();
            
            // 参数更新阶段
            PushPhase(PARAMETER_UPDATE_ID);
            UpdateParameters();
            PopPhase();
            
            PopPhase(); // 结束 epoch
        }
        
        // 分析每个 epoch 的性能
        AnalyzeEpochPerformance();
    }
    
private:
    void AnalyzeEpochPerformance() {
        printf("\n=== Epoch 性能分析 ===\n");
        
        for (int epoch = 0; epoch < numEpochs; epoch++) {
            uint64_t epochId = EPOCH_START_ID + epoch;
            auto correlationIds = s_externalCorrelationMap[epochId];
            
            uint64_t epochDuration = CalculateEpochDuration(correlationIds);
            
            printf("Epoch %d: %llu ns\n", epoch, epochDuration);
        }
    }
};
```

## 故障排除

### 常见问题

1. **未匹配的推入/弹出**：
   ```cpp
   void ValidatePushPopBalance() {
       static int pushPopBalance = 0;
       
       // 在推入时增加
       void OnPush() { pushPopBalance++; }
       
       // 在弹出时减少
       void OnPop() { pushPopBalance--; }
       
       // 验证平衡
       void ValidateBalance() {
           if (pushPopBalance != 0) {
               printf("警告: 未匹配的推入/弹出操作\n");
           }
       }
   }
   ```

2. **丢失的外部关联**：
   ```cpp
   void ValidateExternalCorrelations() {
       for (const auto& [externalId, correlationIds] : s_externalCorrelationMap) {
           if (correlationIds.empty()) {
               printf("警告: 外部 ID %llu 没有关联的活动\n", externalId);
           }
       }
   }
   ```

## 总结

CUPTI 外部关联为理解应用程序性能提供了强大的高级视角。通过将 GPU 活动与应用程序阶段关联：

### 关键优势

- **阶段感知分析**：理解每个应用程序阶段的性能
- **瓶颈识别**：快速识别最慢的阶段
- **优化指导**：针对特定阶段的优化建议
- **分层洞察**：支持复杂应用程序的嵌套分析

### 最佳实践

1. **定义清晰的阶段**：使用有意义的阶段标识符
2. **平衡推入/弹出**：确保每个推入都有对应的弹出
3. **分层组织**：对复杂应用程序使用嵌套关联
4. **性能监控**：定期分析阶段性能趋势

外部关联是构建智能、上下文感知性能分析工具的关键技术。 