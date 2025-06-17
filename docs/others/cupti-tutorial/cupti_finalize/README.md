# CUPTI Finalization and Cleanup Tutorial

> The GitHub repo and complete tutorial is available at <https://github.com/eunomia-bpf/cupti-tutorial>.

## Introduction

The CUPTI Finalization sample demonstrates proper cleanup procedures and finalization techniques for CUPTI-based profiling applications. This tutorial covers best practices for resource management, graceful shutdown procedures, and ensuring complete data collection before application termination.

## What You'll Learn

- How to properly finalize CUPTI profiling sessions
- Understanding resource cleanup requirements
- Implementing graceful shutdown procedures  
- Ensuring complete data collection and reporting
- Handling edge cases and error conditions during finalization

## Understanding CUPTI Finalization

Proper CUPTI finalization is crucial for:

1. **Data Integrity**: Ensuring all collected data is properly flushed and saved
2. **Resource Cleanup**: Releasing CUPTI resources and avoiding memory leaks
3. **Graceful Shutdown**: Handling application termination without data loss
4. **Error Handling**: Managing cleanup in error conditions
5. **Performance Impact**: Minimizing finalization overhead

## Key Finalization Steps

### Activity Buffer Finalization
- Flush remaining activity records
- Process pending activities
- Close activity streams

### Event Group Cleanup
- Disable active event groups
- Destroy event group objects
- Release associated resources

### Callback Deregistration
- Unsubscribe from callbacks
- Clean up callback data structures
- Ensure no pending callbacks

### Context and Device Cleanup
- Destroy profiling contexts
- Release device resources
- Clean up per-device data structures

## Building the Sample

### Prerequisites

- CUDA Toolkit with CUPTI
- Application with CUPTI profiling integration
- Understanding of CUPTI resource management

### Build Process

```bash
cd cupti_finalize
make
```

This creates the `cupti_finalize` executable demonstrating proper CUPTI cleanup procedures.

## Running the Sample

### Basic Execution

```bash
./cupti_finalize
```

### Sample Output

```
=== CUPTI Finalization Process ===

Starting profiling session...
Profiling active for 1500ms

Beginning finalization process...

Activity Buffer Finalization:
  Flushing activity buffers...
  Processed 2,847 activity records
  Activity buffer finalization complete

Event Group Cleanup:
  Disabling 3 active event groups
  Destroying event group objects
  Event cleanup complete

Callback Deregistration:
  Unsubscribing from 5 callback domains
  Cleaning up callback data structures
  Callback cleanup complete

Context and Device Cleanup:
  Cleaning up 2 device contexts
  Releasing profiling resources
  Context cleanup complete

Resource Validation:
  Memory leaks: 0
  Active handles: 0
  Pending operations: 0

Finalization completed successfully in 45ms
All data saved to: profiling_results.json

=== Application Termination ===
```

## Code Architecture

### Finalization Manager

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

### Activity Buffer Finalization

```cpp
void CUPTIFinalizationManager::finalizeActivities() {
    std::cout << "Activity Buffer Finalization:" << std::endl;
    
    // Force flush of all pending activity records
    CUPTI_CALL(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));
    
    // Process any remaining buffered activities
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
    
    std::cout << "  Processed " << processedRecords << " activity records" << std::endl;
    
    // Disable activity recording
    CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL));
    CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY));
    CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_OVERHEAD));
    
    std::cout << "  Activity buffer finalization complete" << std::endl;
}
```

### Event Group Cleanup

```cpp
void CUPTIFinalizationManager::finalizeEventGroups() {
    std::cout << "Event Group Cleanup:" << std::endl;
    std::cout << "  Disabling " << resources.activeEventGroups.size() << " active event groups" << std::endl;
    
    for (auto& eventGroup : resources.activeEventGroups) {
        // Disable the event group first
        CUptiResult status = cuptiEventGroupDisable(eventGroup);
        if (status != CUPTI_SUCCESS) {
            std::cerr << "Warning: Failed to disable event group" << std::endl;
        }
        
        // Read any final event values
        readFinalEventValues(eventGroup);
        
        // Destroy the event group
        CUPTI_CALL(cuptiEventGroupDestroy(eventGroup));
    }
    
    resources.activeEventGroups.clear();
    std::cout << "  Event cleanup complete" << std::endl;
}

void CUPTIFinalizationManager::readFinalEventValues(CUpti_EventGroup eventGroup) {
    // Get the number of events in the group
    uint32_t numEvents;
    size_t size = sizeof(numEvents);
    CUPTI_CALL(cuptiEventGroupGetAttribute(eventGroup, 
               CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS, &size, &numEvents));
    
    if (numEvents > 0) {
        // Read final event values before destruction
        std::vector<CUpti_EventID> eventIds(numEvents);
        std::vector<uint64_t> eventValues(numEvents);
        
        size = numEvents * sizeof(CUpti_EventID);
        CUPTI_CALL(cuptiEventGroupGetAttribute(eventGroup,
                   CUPTI_EVENT_GROUP_ATTR_EVENTS, &size, eventIds.data()));
        
        size = numEvents * sizeof(uint64_t);
        CUPTI_CALL(cuptiEventGroupReadAllEvents(eventGroup, 
                   CUPTI_EVENT_READ_FLAG_NONE, &size, eventValues.data(),
                   &numEvents, eventIds.data()));
        
        // Store final values for reporting
        storeFinalEventValues(eventIds, eventValues);
    }
}
```

### Callback Deregistration

```cpp
void CUPTIFinalizationManager::finalizeCallbacks() {
    std::cout << "Callback Deregistration:" << std::endl;
    std::cout << "  Unsubscribing from " << resources.activeCallbacks.size() << " callback domains" << std::endl;
    
    for (auto& subscriber : resources.activeCallbacks) {
        // First disable all callbacks for this subscriber
        CUPTI_CALL(cuptiEnableCallback(0, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, 
                   CUPTI_RUNTIME_TRACE_CBID_INVALID));
        CUPTI_CALL(cuptiEnableCallback(0, subscriber, CUPTI_CB_DOMAIN_DRIVER_API,
                   CUPTI_DRIVER_TRACE_CBID_INVALID));
        
        // Wait for any pending callbacks to complete
        waitForPendingCallbacks(subscriber);
        
        // Unsubscribe from callbacks
        CUPTI_CALL(cuptiUnsubscribe(subscriber));
    }
    
    resources.activeCallbacks.clear();
    std::cout << "  Callback cleanup complete" << std::endl;
}

void CUPTIFinalizationManager::waitForPendingCallbacks(CUpti_SubscriberHandle subscriber) {
    // Ensure all pending callbacks have completed
    // This is important to avoid race conditions during cleanup
    
    const int maxWaitMs = 1000;
    const int pollIntervalMs = 10;
    int waitedMs = 0;
    
    while (hasPendingCallbacks(subscriber) && waitedMs < maxWaitMs) {
        std::this_thread::sleep_for(std::chrono::milliseconds(pollIntervalMs));
        waitedMs += pollIntervalMs;
    }
    
    if (waitedMs >= maxWaitMs) {
        std::cerr << "Warning: Timeout waiting for callbacks to complete" << std::endl;
    }
}
```

## Advanced Finalization Techniques

### Graceful Shutdown with Signal Handling

```cpp
class GracefulShutdownManager {
private:
    static std::atomic<bool> shutdownRequested;
    static CUPTIFinalizationManager* finalizationManager;
    
public:
    static void signalHandler(int signal) {
        std::cout << "Shutdown signal received (" << signal << "), beginning graceful finalization..." << std::endl;
        shutdownRequested = true;
        
        if (finalizationManager) {
            finalizationManager->beginFinalization();
        }
    }
    
    static void setupSignalHandlers(CUPTIFinalizationManager* manager) {
        finalizationManager = manager;
        signal(SIGINT, signalHandler);
        signal(SIGTERM, signalHandler);
        signal(SIGABRT, signalHandler);
    }
    
    static bool isShutdownRequested() {
        return shutdownRequested.load();
    }
};

// Usage in main application
int main() {
    CUPTIFinalizationManager finalizationManager;
    GracefulShutdownManager::setupSignalHandlers(&finalizationManager);
    
    // Main application loop
    while (!GracefulShutdownManager::isShutdownRequested()) {
        // Perform profiling work
        doProfilingWork();
        
        // Check for completion
        if (isWorkComplete()) {
            break;
        }
    }
    
    // Always finalize, whether through normal completion or signal
    finalizationManager.beginFinalization();
    return 0;
}
```

### Error-Resilient Finalization

```cpp
class ResilientFinalizer {
private:
    struct FinalizationStep {
        std::string name;
        std::function<void()> action;
        bool isOptional;
        bool completed;
    };
    
    std::vector<FinalizationStep> finalizationSteps;
    
public:
    void addFinalizationStep(const std::string& name, 
                           std::function<void()> action, 
                           bool optional = false) {
        finalizationSteps.push_back({name, action, optional, false});
    }
    
    void executeFinalization() {
        std::vector<std::string> errors;
        
        for (auto& step : finalizationSteps) {
            try {
                std::cout << "Executing: " << step.name << std::endl;
                step.action();
                step.completed = true;
                std::cout << "  ✓ " << step.name << " completed" << std::endl;
            } catch (const std::exception& e) {
                std::string error = step.name + ": " + e.what();
                errors.push_back(error);
                
                if (step.isOptional) {
                    std::cout << "  ⚠ " << step.name << " failed (optional): " << e.what() << std::endl;
                } else {
                    std::cout << "  ✗ " << step.name << " failed (critical): " << e.what() << std::endl;
                }
            }
        }
        
        // Report finalization status
        reportFinalizationStatus(errors);
    }
    
private:
    void reportFinalizationStatus(const std::vector<std::string>& errors) {
        int completed = 0;
        int critical_failed = 0;
        
        for (const auto& step : finalizationSteps) {
            if (step.completed) {
                completed++;
            } else if (!step.isOptional) {
                critical_failed++;
            }
        }
        
        std::cout << "Finalization Summary:" << std::endl;
        std::cout << "  Steps completed: " << completed << "/" << finalizationSteps.size() << std::endl;
        std::cout << "  Critical failures: " << critical_failed << std::endl;
        
        if (!errors.empty()) {
            std::cout << "Errors encountered:" << std::endl;
            for (const auto& error : errors) {
                std::cout << "  - " << error << std::endl;
            }
        }
    }
};
```

### Resource Validation and Leak Detection

```cpp
class ResourceValidator {
private:
    struct ResourceSnapshot {
        size_t allocatedMemory;
        int activeHandles;
        int pendingOperations;
        std::chrono::steady_clock::time_point timestamp;
    };
    
    ResourceSnapshot initialSnapshot;
    
public:
    void takeInitialSnapshot() {
        initialSnapshot = getCurrentResourceSnapshot();
    }
    
    bool validateFinalState() {
        ResourceSnapshot finalSnapshot = getCurrentResourceSnapshot();
        
        bool isValid = true;
        
        // Check for memory leaks
        if (finalSnapshot.allocatedMemory > initialSnapshot.allocatedMemory) {
            size_t leakedMemory = finalSnapshot.allocatedMemory - initialSnapshot.allocatedMemory;
            std::cout << "Memory leak detected: " << leakedMemory << " bytes" << std::endl;
            isValid = false;
        }
        
        // Check for unclosed handles
        if (finalSnapshot.activeHandles > 0) {
            std::cout << "Active handles remaining: " << finalSnapshot.activeHandles << std::endl;
            isValid = false;
        }
        
        // Check for pending operations
        if (finalSnapshot.pendingOperations > 0) {
            std::cout << "Pending operations: " << finalSnapshot.pendingOperations << std::endl;
            isValid = false;
        }
        
        if (isValid) {
            std::cout << "Resource validation: PASSED" << std::endl;
        } else {
            std::cout << "Resource validation: FAILED" << std::endl;
        }
        
        return isValid;
    }
    
private:
    ResourceSnapshot getCurrentResourceSnapshot() {
        ResourceSnapshot snapshot;
        snapshot.allocatedMemory = getCurrentMemoryUsage();
        snapshot.activeHandles = getActiveHandleCount();
        snapshot.pendingOperations = getPendingOperationCount();
        snapshot.timestamp = std::chrono::steady_clock::now();
        return snapshot;
    }
};
```

## Integration with Application Lifecycle

### RAII-Based Resource Management

```cpp
class CUPTISession {
private:
    CUPTIFinalizationManager finalizationManager;
    ResourceValidator validator;
    bool isActive;
    
public:
    CUPTISession() : isActive(false) {
        validator.takeInitialSnapshot();
    }
    
    ~CUPTISession() {
        if (isActive) {
            finalize();
        }
    }
    
    void initialize() {
        if (isActive) {
            throw std::runtime_error("Session already active");
        }
        
        // Initialize CUPTI components
        initializeActivityTracing();
        initializeEventCollection();
        initializeCallbacks();
        
        isActive = true;
    }
    
    void finalize() {
        if (!isActive) {
            return;
        }
        
        try {
            finalizationManager.beginFinalization();
            isActive = false;
            
            // Validate cleanup
            validator.validateFinalState();
        } catch (const std::exception& e) {
            std::cerr << "Error during finalization: " << e.what() << std::endl;
        }
    }
    
    // Move semantics to prevent copying
    CUPTISession(CUPTISession&& other) noexcept 
        : finalizationManager(std::move(other.finalizationManager)),
          validator(std::move(other.validator)),
          isActive(other.isActive) {
        other.isActive = false;
    }
    
    CUPTISession& operator=(CUPTISession&& other) noexcept {
        if (this != &other) {
            finalize(); // Clean up current state
            finalizationManager = std::move(other.finalizationManager);
            validator = std::move(other.validator);
            isActive = other.isActive;
            other.isActive = false;
        }
        return *this;
    }
    
    // Disable copy semantics
    CUPTISession(const CUPTISession&) = delete;
    CUPTISession& operator=(const CUPTISession&) = delete;
};
```

### Finalization in Multi-threaded Applications

```cpp
class ThreadSafeFinalizationManager {
private:
    std::mutex finalizationMutex;
    std::atomic<bool> finalizationComplete;
    std::vector<std::thread::id> activeThreads;
    std::condition_variable finalizationCV;
    
public:
    void registerThread() {
        std::lock_guard<std::mutex> lock(finalizationMutex);
        activeThreads.push_back(std::this_thread::get_id());
    }
    
    void unregisterThread() {
        std::lock_guard<std::mutex> lock(finalizationMutex);
        auto it = std::find(activeThreads.begin(), activeThreads.end(), 
                           std::this_thread::get_id());
        if (it != activeThreads.end()) {
            activeThreads.erase(it);
        }
        
        // Notify if this was the last thread
        if (activeThreads.empty()) {
            finalizationCV.notify_all();
        }
    }
    
    void waitForAllThreads() {
        std::unique_lock<std::mutex> lock(finalizationMutex);
        finalizationCV.wait(lock, [this] { return activeThreads.empty(); });
    }
    
    void finalizeThreadSafe() {
        // Signal all threads to stop
        signalShutdown();
        
        // Wait for all threads to complete
        waitForAllThreads();
        
        // Perform final cleanup
        performFinalization();
        
        finalizationComplete = true;
    }
};
```

## Best Practices

### Finalization Checklist

1. **Activity Buffers**: Flush and process all pending activities
2. **Event Groups**: Disable and destroy all event groups
3. **Callbacks**: Unsubscribe from all callback domains
4. **Contexts**: Clean up all CUPTI contexts
5. **Memory**: Verify no memory leaks
6. **Handles**: Ensure all handles are closed
7. **Files**: Close all output files and streams

### Common Pitfalls

1. **Premature Finalization**: Finalizing while operations are still pending
2. **Resource Leaks**: Forgetting to clean up event groups or callbacks
3. **Race Conditions**: Finalizing while callbacks are still executing
4. **Data Loss**: Not flushing activity buffers before shutdown
5. **Error Handling**: Not handling cleanup errors gracefully

## Next Steps

- Integrate proper finalization into your CUPTI applications
- Implement graceful shutdown procedures for production systems
- Add resource validation to detect leaks and improper cleanup
- Test finalization under various failure scenarios
- Combine with other CUPTI samples for comprehensive profiling solutions 