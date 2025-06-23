# CUPTI 终结和清理教程

> GitHub 仓库和完整教程可在 <https://github.com/eunomia-bpf/cupti-tutorial> 获取。

## 简介

CUPTI 终结示例演示了基于 CUPTI 的性能分析应用程序的正确清理程序和终结技术。本教程涵盖了资源管理的最佳实践、优雅关闭程序，以及确保在应用程序终止前完成数据收集。

## 您将学到的内容

- 如何正确终结 CUPTI 性能分析会话
- 理解资源清理要求
- 实现优雅关闭程序
- 确保完整的数据收集和报告
- 处理终结期间的边界情况和错误条件

## 理解 CUPTI 终结

正确的 CUPTI 终结对以下方面至关重要：

1. **数据完整性**：确保所有收集的数据都正确刷新和保存
2. **资源清理**：释放 CUPTI 资源并避免内存泄漏
3. **优雅关闭**：处理应用程序终止而不丢失数据
4. **错误处理**：在错误条件下管理清理
5. **性能影响**：最小化终结开销

## 关键终结步骤

### 活动缓冲区终结
- 刷新剩余的活动记录
- 处理挂起的活动
- 关闭活动流

### 事件组清理
- 禁用活动事件组
- 销毁事件组对象
- 释放关联资源

### 回调注销
- 取消订阅回调
- 清理回调数据结构
- 确保没有挂起的回调

### 上下文和设备清理
- 销毁性能分析上下文
- 释放设备资源
- 清理每设备数据结构

## 构建示例

### 先决条件

- 带有 CUPTI 的 CUDA 工具包
- 具有 CUPTI 性能分析集成的应用程序
- 理解 CUPTI 资源管理

### 构建过程

```bash
cd cupti_finalize
make
```

这将创建演示正确 CUPTI 清理程序的 `cupti_finalize` 可执行文件。

## 运行示例

### 基本执行

```bash
./cupti_finalize
```

### 示例输出

```
=== CUPTI 终结过程 ===

开始性能分析会话...
性能分析活动 1500ms

开始终结过程...

活动缓冲区终结:
  刷新活动缓冲区...
  处理了 2,847 个活动记录
  活动缓冲区终结完成

事件组清理:
  禁用 3 个活动事件组
  销毁事件组对象
  事件清理完成

回调注销:
  取消订阅 5 个回调域
  清理回调数据结构
  回调清理完成

上下文和设备清理:
  清理 2 个设备上下文
  释放性能分析资源
  上下文清理完成

资源验证:
  内存泄漏: 0
  活动句柄: 0
  挂起操作: 0

终结在 45ms 内成功完成
所有数据保存到: profiling_results.json

=== 应用程序终止 ===
```

## 代码架构

### 终结管理器

```cpp
class CUPTIFinalizationManager {
private:
    struct ResourceTracker {
        std::vector<CUpti_EventGroup> activeEventGroups;
        std::vector<CUpti_SubscriberHandle> activeCallbacks;
        std::vector<CUcontext> managedContexts;
        std::vector<CUpti_ActivityBufferState> activityBuffers;
        bool isFinalized;
    };
    
    ResourceTracker resources;
    std::mutex finalizationMutex;
    std::atomic<bool> finalizationInProgress;

public:
    void registerEventGroup(CUpti_EventGroup group);
    void registerCallback(CUpti_SubscriberHandle subscriber);
    void registerContext(CUcontext context);
    void beginFinalization();
    void finalizeActivities();
    void finalizeEventGroups();
    void finalizeCallbacks();
    void finalizeContexts();
    bool validateCleanup();
};
```

### 活动缓冲区终结

```cpp
void CUPTIFinalizationManager::finalizeActivities() {
    std::cout << "活动缓冲区终结:" << std::endl;
    
    // 强制刷新所有挂起的活动记录
    CUPTI_CALL(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));
    
    // 处理任何剩余的缓冲活动
    CUpti_Activity* record = nullptr;
    size_t processedRecords = 0;
    
    do {
        CUptiResult status = cuptiActivityGetNextRecord(buffer, validSize, &record);
        if (status == CUPTI_SUCCESS) {
            processActivityRecord(record);
            processedRecords++;
        } else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
            break;
        } else {
            CUPTI_ERROR_CHECK(status);
        }
    } while (record != nullptr);
    
    std::cout << "  处理了 " << processedRecords << " 个活动记录" << std::endl;
    
    // 禁用活动记录
    CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL));
    CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY));
    CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_OVERHEAD));
    
    std::cout << "  活动缓冲区终结完成" << std::endl;
}
```

### 事件组清理

```cpp
void CUPTIFinalizationManager::finalizeEventGroups() {
    std::cout << "事件组清理:" << std::endl;
    std::cout << "  禁用 " << resources.activeEventGroups.size() << " 个活动事件组" << std::endl;
    
    for (auto& eventGroup : resources.activeEventGroups) {
        // 首先禁用事件组
        CUptiResult status = cuptiEventGroupDisable(eventGroup);
        if (status != CUPTI_SUCCESS) {
            std::cerr << "警告: 禁用事件组失败" << std::endl;
        }
        
        // 读取任何最终事件值
        readFinalEventValues(eventGroup);
        
        // 销毁事件组
        status = cuptiEventGroupDestroy(eventGroup);
        if (status != CUPTI_SUCCESS) {
            std::cerr << "警告: 销毁事件组失败" << std::endl;
        }
    }
    
    resources.activeEventGroups.clear();
    std::cout << "  销毁事件组对象" << std::endl;
    std::cout << "  事件清理完成" << std::endl;
}
```

### 回调注销

```cpp
void CUPTIFinalizationManager::finalizeCallbacks() {
    std::cout << "回调注销:" << std::endl;
    std::cout << "  取消订阅 " << resources.activeCallbacks.size() << " 个回调域" << std::endl;
    
    for (auto& subscriber : resources.activeCallbacks) {
        // 禁用所有回调域
        CUPTI_CALL(cuptiEnableDomain(0, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API));
        CUPTI_CALL(cuptiEnableDomain(0, subscriber, CUPTI_CB_DOMAIN_DRIVER_API));
        CUPTI_CALL(cuptiEnableDomain(0, subscriber, CUPTI_CB_DOMAIN_RESOURCE));
        CUPTI_CALL(cuptiEnableDomain(0, subscriber, CUPTI_CB_DOMAIN_SYNCHRONIZE));
        CUPTI_CALL(cuptiEnableDomain(0, subscriber, CUPTI_CB_DOMAIN_NVTX));
        
        // 取消订阅
        CUPTI_CALL(cuptiUnsubscribe(subscriber));
    }
    
    resources.activeCallbacks.clear();
    std::cout << "  清理回调数据结构" << std::endl;
    std::cout << "  回调清理完成" << std::endl;
}
```

### 上下文和设备清理

```cpp
void CUPTIFinalizationManager::finalizeContexts() {
    std::cout << "上下文和设备清理:" << std::endl;
    std::cout << "  清理 " << resources.managedContexts.size() << " 个设备上下文" << std::endl;
    
    for (auto& context : resources.managedContexts) {
        // 确保上下文是当前的
        DRIVER_API_CALL(cuCtxSetCurrent(context));
        
        // 清理上下文特定的 CUPTI 资源
        cleanupContextResources(context);
        
        // 销毁上下文（如果我们拥有它）
        if (isOwnedContext(context)) {
            DRIVER_API_CALL(cuCtxDestroy(context));
        }
    }
    
    resources.managedContexts.clear();
    std::cout << "  释放性能分析资源" << std::endl;
    std::cout << "  上下文清理完成" << std::endl;
}
```

## 高级终结技术

### 优雅关闭处理

```cpp
class GracefulShutdownHandler {
private:
    static std::atomic<bool> shutdownRequested;
    static CUPTIFinalizationManager* finalizationManager;

public:
    static void setupSignalHandlers() {
        signal(SIGINT, signalHandler);
        signal(SIGTERM, signalHandler);
        signal(SIGABRT, signalHandler);
        
        // Windows 特定
        #ifdef _WIN32
        SetConsoleCtrlHandler(consoleHandler, TRUE);
        #endif
    }
    
private:
    static void signalHandler(int signal) {
        std::cout << "\n收到信号 " << signal << "，开始优雅关闭..." << std::endl;
        shutdownRequested = true;
        
        if (finalizationManager) {
            finalizationManager->beginFinalization();
        }
        
        exit(0);
    }
    
    #ifdef _WIN32
    static BOOL WINAPI consoleHandler(DWORD ctrlType) {
        switch (ctrlType) {
            case CTRL_C_EVENT:
            case CTRL_BREAK_EVENT:
            case CTRL_CLOSE_EVENT:
                signalHandler(SIGINT);
                return TRUE;
            default:
                return FALSE;
        }
    }
    #endif
};
```

### 错误恢复

```cpp
class ErrorRecoveryManager {
public:
    static void handleFinalizationError(CUptiResult error, const std::string& operation) {
        std::cerr << "终结错误在 " << operation << ": " << 
                     cuptiGetResultString(error) << std::endl;
        
        // 尝试恢复策略
        if (error == CUPTI_ERROR_NOT_INITIALIZED) {
            std::cerr << "CUPTI 未初始化，跳过 " << operation << std::endl;
            return;
        }
        
        if (error == CUPTI_ERROR_INVALID_PARAMETER) {
            std::cerr << "无效参数在 " << operation << "，继续其他清理" << std::endl;
            return;
        }
        
        // 对于关键错误，记录并继续
        std::cerr << "继续清理尽管有错误..." << std::endl;
        logErrorToFile(error, operation);
    }
    
private:
    static void logErrorToFile(CUptiResult error, const std::string& operation) {
        std::ofstream errorLog("cupti_finalization_errors.log", std::ios::app);
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        
        errorLog << "[" << std::ctime(&time_t) << "] "
                 << "错误在 " << operation << ": " << cuptiGetResultString(error) 
                 << std::endl;
    }
};
```

### 资源验证

```cpp
class ResourceValidator {
public:
    bool validateCleanup(const CUPTIFinalizationManager& manager) {
        bool isValid = true;
        
        // 检查内存泄漏
        if (!checkMemoryLeaks()) {
            std::cerr << "检测到内存泄漏" << std::endl;
            isValid = false;
        }
        
        // 验证所有句柄都已关闭
        if (!validateHandles()) {
            std::cerr << "检测到未关闭的句柄" << std::endl;
            isValid = false;
        }
        
        // 检查挂起操作
        if (!checkPendingOperations()) {
            std::cerr << "检测到挂起操作" << std::endl;
            isValid = false;
        }
        
        return isValid;
    }
    
private:
    bool checkMemoryLeaks() {
        // 实现内存泄漏检测
        // 这可能涉及跟踪分配/释放
        return true; // 简化实现
    }
    
    bool validateHandles() {
        // 验证所有 CUPTI 句柄都已正确关闭
        return true; // 简化实现
    }
    
    bool checkPendingOperations() {
        // 检查是否有挂起的 CUPTI 操作
        return true; // 简化实现
    }
};
```

## 数据保存和报告

### 最终数据导出

```cpp
class FinalDataExporter {
private:
    struct ProfilingResults {
        std::vector<ActivityRecord> activities;
        std::vector<EventRecord> events;
        std::vector<MetricRecord> metrics;
        std::map<std::string, std::string> metadata;
        uint64_t totalProfilingTime;
        size_t totalDataPoints;
    };
    
    ProfilingResults results;

public:
    void collectFinalData() {
        // 收集所有剩余数据
        collectActivityData();
        collectEventData();
        collectMetricData();
        collectMetadata();
        
        // 计算摘要统计
        calculateSummaryStatistics();
    }
    
    void exportToJson(const std::string& filename) {
        nlohmann::json jsonResults;
        
        // 导出活动数据
        jsonResults["activities"] = nlohmann::json::array();
        for (const auto& activity : results.activities) {
            nlohmann::json activityJson;
            activityJson["name"] = activity.name;
            activityJson["start"] = activity.startTime;
            activityJson["end"] = activity.endTime;
            activityJson["duration"] = activity.duration;
            jsonResults["activities"].push_back(activityJson);
        }
        
        // 导出事件数据
        jsonResults["events"] = nlohmann::json::array();
        for (const auto& event : results.events) {
            nlohmann::json eventJson;
            eventJson["name"] = event.name;
            eventJson["value"] = event.value;
            eventJson["timestamp"] = event.timestamp;
            jsonResults["events"].push_back(eventJson);
        }
        
        // 导出元数据
        jsonResults["metadata"] = results.metadata;
        jsonResults["summary"] = {
            {"total_profiling_time", results.totalProfilingTime},
            {"total_data_points", results.totalDataPoints}
        };
        
        // 写入文件
        std::ofstream file(filename);
        file << jsonResults.dump(2);
        
        std::cout << "所有数据保存到: " << filename << std::endl;
    }
    
private:
    void collectActivityData() {
        // 实现活动数据收集
    }
    
    void collectEventData() {
        // 实现事件数据收集
    }
    
    void collectMetricData() {
        // 实现指标数据收集
    }
    
    void collectMetadata() {
        // 收集环境和配置信息
        results.metadata["cuda_version"] = getCudaVersionString();
        results.metadata["driver_version"] = getDriverVersionString();
        results.metadata["device_name"] = getDeviceName();
        results.metadata["finalization_time"] = getCurrentTimestampString();
    }
    
    void calculateSummaryStatistics() {
        results.totalDataPoints = results.activities.size() + 
                                 results.events.size() + 
                                 results.metrics.size();
        
        // 计算总性能分析时间
        if (!results.activities.empty()) {
            uint64_t minStart = std::numeric_limits<uint64_t>::max();
            uint64_t maxEnd = 0;
            
            for (const auto& activity : results.activities) {
                minStart = std::min(minStart, activity.startTime);
                maxEnd = std::max(maxEnd, activity.endTime);
            }
            
            results.totalProfilingTime = maxEnd - minStart;
        }
    }
};
```

## 性能考虑

### 快速终结

```cpp
class FastFinalization {
public:
    void performQuickFinalization() {
        // 仅执行必要的清理以进行快速关闭
        
        // 1. 禁用新的活动记录
        cuptiActivityDisableAll();
        
        // 2. 快速刷新而不处理
        cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
        
        // 3. 禁用回调
        disableAllCallbacks();
        
        // 4. 跳过详细验证
        std::cout << "快速终结完成" << std::endl;
    }
    
    void performCompleteFinalization() {
        // 执行完整的清理和验证
        finalizationManager.beginFinalization();
        finalizationManager.finalizeActivities();
        finalizationManager.finalizeEventGroups();
        finalizationManager.finalizeCallbacks();
        finalizationManager.finalizeContexts();
        
        if (finalizationManager.validateCleanup()) {
            std::cout << "完整终结和验证成功" << std::endl;
        }
    }
};
```

## 最佳实践

### 终结检查清单

```cpp
class FinalizationChecklist {
public:
    void runFinalizationChecklist() {
        std::cout << "\n=== 终结检查清单 ===\n";
        
        checkItem("活动缓冲区已刷新", verifyActivityBuffersFlushed());
        checkItem("所有事件组已禁用", verifyEventGroupsDisabled());
        checkItem("回调已注销", verifyCallbacksUnregistered());
        checkItem("上下文已清理", verifyContextsCleaned());
        checkItem("内存已释放", verifyMemoryReleased());
        checkItem("句柄已关闭", verifyHandlesClosed());
        checkItem("数据已保存", verifyDataSaved());
        
        std::cout << "=== 检查清单完成 ===\n";
    }
    
private:
    void checkItem(const std::string& item, bool status) {
        std::cout << "[" << (status ? "✓" : "✗") << "] " << item << std::endl;
    }
    
    bool verifyActivityBuffersFlushed() { return true; } // 简化实现
    bool verifyEventGroupsDisabled() { return true; }
    bool verifyCallbacksUnregistered() { return true; }
    bool verifyContextsCleaned() { return true; }
    bool verifyMemoryReleased() { return true; }
    bool verifyHandlesClosed() { return true; }
    bool verifyDataSaved() { return true; }
};
```

## 故障排除

### 常见终结问题

1. **不完整的数据刷新**：
   ```cpp
   void ensureCompleteFlush() {
       // 多次尝试刷新以确保完整性
       for (int i = 0; i < 3; i++) {
           cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
           std::this_thread::sleep_for(std::chrono::milliseconds(10));
       }
   }
   ```

2. **资源泄漏**：
   ```cpp
   void detectResourceLeaks() {
       // 实现资源泄漏检测
       if (getActiveHandleCount() > 0) {
           std::cerr << "警告: 检测到活动句柄泄漏" << std::endl;
       }
   }
   ```

3. **挂起操作**：
   ```cpp
   void waitForPendingOperations() {
       // 等待所有挂起操作完成
       while (hasPendingOperations()) {
           std::this_thread::sleep_for(std::chrono::milliseconds(1));
       }
   }
   ```

## 总结

正确的 CUPTI 终结对于健壮的性能分析应用程序至关重要。通过遵循本教程中演示的最佳实践：

### 关键要点

- **彻底性**：确保清理所有资源
- **数据完整性**：在终结前保存所有收集的数据
- **错误处理**：优雅地处理终结期间的错误
- **性能**：平衡完整性和关闭速度

### 最佳实践

1. **始终刷新活动缓冲区**：防止数据丢失
2. **实现信号处理**：支持优雅关闭
3. **验证清理**：确保没有资源泄漏
4. **记录错误**：帮助调试终结问题
5. **测试终结路径**：验证在各种条件下的行为

正确的 CUPTI 终结确保了可靠和专业的性能分析工具，为用户提供了完整和一致的体验。 