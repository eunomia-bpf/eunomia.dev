# CUDA Memory Tracing Tutorial

## Introduction

The CUDA Memory Tracing sample demonstrates how to track and analyze memory operations in CUDA applications using CUPTI's activity tracing capabilities. This tutorial focuses specifically on memory management, transfer patterns, and memory usage optimization through detailed tracing and analysis.

## What You'll Learn

- How to trace all types of CUDA memory operations
- Understanding memory transfer patterns and bottlenecks  
- Analyzing memory allocation and deallocation patterns
- Detecting memory leaks and usage inefficiencies
- Optimizing memory bandwidth utilization

## Understanding CUDA Memory Operations

CUDA applications involve various types of memory operations:

1. **Memory Allocation**: cudaMalloc, cudaMallocPitch, cudaMallocManaged
2. **Memory Transfer**: cudaMemcpy, cudaMemcpyAsync, peer-to-peer transfers
3. **Memory Mapping**: cudaHostAlloc, cudaHostRegister
4. **Unified Memory**: cudaMallocManaged, automatic migration
5. **Memory Deallocation**: cudaFree, cudaFreeHost

## Key Concepts

### Memory Domains

#### Device Memory
- Global memory on GPU
- Texture and surface memory
- Constant memory
- Local memory (registers/shared)

#### Host Memory
- Pageable system memory
- Pinned (page-locked) memory
- Unified memory regions

#### Transfer Types
- Host-to-Device (H2D)
- Device-to-Host (D2H)  
- Device-to-Device (D2D)
- Peer-to-Peer (P2P)

## Building the Sample

### Prerequisites

- CUDA Toolkit with CUPTI
- Application with diverse memory operations
- Sufficient memory for tracing buffers

### Build Process

```bash
cd cuda_memory_trace
make
```

This creates the `cuda_memory_trace` executable that demonstrates memory operation tracing.

## Code Architecture

### Memory Activity Tracking

```cpp
class MemoryTracer {
private:
    struct MemoryActivity {
        CUpti_ActivityKind kind;
        uint64_t start;
        uint64_t end;
        size_t bytes;
        CUdeviceptr srcPtr;
        CUdeviceptr dstPtr;
        int srcDevice;
        int dstDevice;
        cudaMemcpyKind copyKind;
    };
    
    std::vector<MemoryActivity> activities;
    std::map<CUdeviceptr, AllocationInfo> allocations;

public:
    void processActivity(CUpti_Activity* record);
    void analyzeMemoryPatterns();
    void generateMemoryReport();
};
```

### Memory Allocation Tracking

```cpp
class AllocationTracker {
private:
    struct AllocationInfo {
        size_t size;
        uint64_t allocTime;
        uint64_t freeTime;
        bool isActive;
        std::string allocationType;
    };
    
    std::map<void*, AllocationInfo> hostAllocations;
    std::map<CUdeviceptr, AllocationInfo> deviceAllocations;
    size_t peakMemoryUsage;
    size_t currentMemoryUsage;

public:
    void recordAllocation(void* ptr, size_t size, const std::string& type, uint64_t timestamp);
    void recordDeallocation(void* ptr, uint64_t timestamp);
    void detectMemoryLeaks();
    void calculateMemoryStatistics();
};
```

## Running the Sample

### Basic Execution

```bash
./cuda_memory_trace
```

### Sample Output

```
=== CUDA Memory Tracing Analysis ===

Memory Allocation Summary:
  Total Device Allocations: 1,024 MB
  Total Host Allocations: 512 MB
  Peak Memory Usage: 1,536 MB
  Active Allocations: 768 MB
  Memory Leaks Detected: 0

Memory Transfer Analysis:
  Host-to-Device: 2,048 MB (avg: 8.5 GB/s)
  Device-to-Host: 1,024 MB (avg: 7.2 GB/s)
  Device-to-Device: 512 MB (avg: 450 GB/s)
  Peer-to-Peer: 256 MB (avg: 28.5 GB/s)

Transfer Patterns:
  Sequential Transfers: 75.5%
  Concurrent Transfers: 24.5%
  Optimal Coalescing: 89.3%
  Bandwidth Efficiency: 85.7%

Memory Hotspots:
  Large Transfers (>100MB): 12 transfers, 2.1 GB total
  Frequent Small Transfers (<1MB): 847 transfers, 45 MB total
  Redundant Transfers: 23 transfers, 128 MB total

Performance Issues:
  - Uncoalesced transfers detected: 18 instances
  - Memory fragmentation: 12.3% overhead
  - Synchronous transfers on default stream: 156 instances
```

## Advanced Memory Analysis

### Bandwidth Analysis

```cpp
class BandwidthAnalyzer {
private:
    struct TransferMetrics {
        double achievedBandwidth;
        double theoreticalBandwidth;
        double efficiency;
        size_t transferSize;
        cudaMemcpyKind direction;
    };
    
    std::vector<TransferMetrics> transferHistory;

public:
    void analyzeTransfer(const MemoryActivity& activity) {
        double duration = (activity.end - activity.start) * 1e-9; // Convert to seconds
        double achievedBandwidth = activity.bytes / duration / 1e9; // GB/s
        
        TransferMetrics metrics;
        metrics.achievedBandwidth = achievedBandwidth;
        metrics.theoreticalBandwidth = getTheoreticalBandwidth(activity.copyKind);
        metrics.efficiency = achievedBandwidth / metrics.theoreticalBandwidth;
        metrics.transferSize = activity.bytes;
        metrics.direction = activity.copyKind;
        
        transferHistory.push_back(metrics);
    }
    
    void generateBandwidthReport() {
        std::map<cudaMemcpyKind, std::vector<double>> bandwidthByType;
        
        for (const auto& metrics : transferHistory) {
            bandwidthByType[metrics.direction].push_back(metrics.achievedBandwidth);
        }
        
        for (const auto& [direction, bandwidths] : bandwidthByType) {
            double avgBandwidth = std::accumulate(bandwidths.begin(), bandwidths.end(), 0.0) / bandwidths.size();
            std::cout << "Average bandwidth for " << getDirectionName(direction) 
                     << ": " << avgBandwidth << " GB/s" << std::endl;
        }
    }
};
```

### Memory Leak Detection

```cpp
class MemoryLeakDetector {
private:
    struct LeakInfo {
        void* address;
        size_t size;
        uint64_t allocTime;
        std::string allocationType;
        std::string stackTrace;
    };
    
    std::vector<LeakInfo> detectedLeaks;

public:
    void checkForLeaks(const AllocationTracker& tracker) {
        for (const auto& [ptr, info] : tracker.getActiveAllocations()) {
            if (info.isActive && !isValidPointer(ptr)) {
                LeakInfo leak;
                leak.address = ptr;
                leak.size = info.size;
                leak.allocTime = info.allocTime;
                leak.allocationType = info.allocationType;
                leak.stackTrace = getStackTrace(info.allocTime);
                
                detectedLeaks.push_back(leak);
            }
        }
    }
    
    void reportLeaks() {
        if (detectedLeaks.empty()) {
            std::cout << "No memory leaks detected!" << std::endl;
            return;
        }
        
        std::cout << "Memory Leaks Detected:" << std::endl;
        size_t totalLeaked = 0;
        
        for (const auto& leak : detectedLeaks) {
            std::cout << "  Address: " << leak.address 
                     << ", Size: " << leak.size << " bytes"
                     << ", Type: " << leak.allocationType << std::endl;
            totalLeaked += leak.size;
        }
        
        std::cout << "Total leaked memory: " << totalLeaked << " bytes" << std::endl;
    }
};
```

### Memory Access Pattern Analysis

```cpp
class AccessPatternAnalyzer {
private:
    struct AccessPattern {
        CUdeviceptr baseAddress;
        size_t stride;
        size_t accessCount;
        bool isCoalesced;
        double coalescingEfficiency;
    };

public:
    void analyzeMemoryAccess(const std::vector<MemoryActivity>& activities) {
        std::map<CUdeviceptr, std::vector<size_t>> accessOffsets;
        
        // Group accesses by base address
        for (const auto& activity : activities) {
            if (activity.kind == CUPTI_ACTIVITY_KIND_MEMCPY) {
                size_t offset = activity.dstPtr - getBaseAddress(activity.dstPtr);
                accessOffsets[getBaseAddress(activity.dstPtr)].push_back(offset);
            }
        }
        
        // Analyze patterns for each memory region
        for (const auto& [baseAddr, offsets] : accessOffsets) {
            AccessPattern pattern = analyzePattern(baseAddr, offsets);
            
            if (!pattern.isCoalesced) {
                std::cout << "Warning: Uncoalesced memory access detected at " 
                         << std::hex << baseAddr << std::dec
                         << " (efficiency: " << pattern.coalescingEfficiency * 100 << "%)" << std::endl;
            }
        }
    }
};
```

## Memory Optimization Insights

### Transfer Optimization

```cpp
class TransferOptimizer {
public:
    struct OptimizationSuggestion {
        std::string issue;
        std::string suggestion;
        double potentialSpeedup;
    };
    
    std::vector<OptimizationSuggestion> analyzeTransfers(const std::vector<MemoryActivity>& activities) {
        std::vector<OptimizationSuggestion> suggestions;
        
        // Check for small, frequent transfers
        int smallTransferCount = 0;
        size_t totalSmallBytes = 0;
        
        for (const auto& activity : activities) {
            if (activity.bytes < 1024) { // Less than 1KB
                smallTransferCount++;
                totalSmallBytes += activity.bytes;
            }
        }
        
        if (smallTransferCount > 100) {
            OptimizationSuggestion suggestion;
            suggestion.issue = "Many small memory transfers detected";
            suggestion.suggestion = "Consider batching small transfers or using unified memory";
            suggestion.potentialSpeedup = estimateSpeedup(smallTransferCount, totalSmallBytes);
            suggestions.push_back(suggestion);
        }
        
        // Check for synchronous transfers
        int syncTransferCount = 0;
        for (const auto& activity : activities) {
            if (isSynchronousTransfer(activity)) {
                syncTransferCount++;
            }
        }
        
        if (syncTransferCount > 50) {
            OptimizationSuggestion suggestion;
            suggestion.issue = "Many synchronous memory transfers";
            suggestion.suggestion = "Use asynchronous transfers with streams for better overlap";
            suggestion.potentialSpeedup = 1.2 + (syncTransferCount * 0.01);
            suggestions.push_back(suggestion);
        }
        
        return suggestions;
    }
};
```

### Memory Pool Analysis

```cpp
class MemoryPoolAnalyzer {
private:
    struct PoolStatistics {
        size_t totalAllocated;
        size_t peakUsage;
        size_t fragmentationWaste;
        double utilizationEfficiency;
        int allocationCount;
        int deallocationCount;
    };

public:
    PoolStatistics analyzeMemoryPool(const std::vector<AllocationInfo>& allocations) {
        PoolStatistics stats = {};
        
        // Calculate fragmentation
        std::map<size_t, int> sizeBuckets;
        for (const auto& alloc : allocations) {
            size_t bucket = roundToPowerOfTwo(alloc.size);
            sizeBuckets[bucket]++;
            stats.totalAllocated += alloc.size;
        }
        
        // Estimate fragmentation waste
        for (const auto& [bucketSize, count] : sizeBuckets) {
            size_t avgWaste = bucketSize / 4; // Estimate internal fragmentation
            stats.fragmentationWaste += avgWaste * count;
        }
        
        stats.utilizationEfficiency = 1.0 - (double(stats.fragmentationWaste) / stats.totalAllocated);
        
        return stats;
    }
};
```

## Real-World Applications

### Deep Learning Memory Profiling

```cpp
class DLMemoryProfiler {
public:
    void profileTrainingStep(const std::vector<MemoryActivity>& activities) {
        std::map<std::string, size_t> phaseMemory;
        
        // Categorize memory operations by training phase
        for (const auto& activity : activities) {
            std::string phase = classifyTrainingPhase(activity);
            phaseMemory[phase] += activity.bytes;
        }
        
        std::cout << "Memory usage by training phase:" << std::endl;
        for (const auto& [phase, bytes] : phaseMemory) {
            std::cout << "  " << phase << ": " << bytes / (1024*1024) << " MB" << std::endl;
        }
        
        // Detect gradient accumulation patterns
        detectGradientAccumulation(activities);
        
        // Analyze batch size impact
        analyzeBatchSizeEfficiency(activities);
    }
    
private:
    std::string classifyTrainingPhase(const MemoryActivity& activity) {
        // Use heuristics to classify memory operations
        if (activity.copyKind == cudaMemcpyHostToDevice) {
            return "Data Loading";
        } else if (isGradientOperation(activity)) {
            return "Gradient Computation";
        } else if (isWeightUpdate(activity)) {
            return "Parameter Update";
        } else {
            return "Forward Pass";
        }
    }
};
```

### Scientific Computing Memory Analysis

```cpp
class ScientificMemoryAnalyzer {
public:
    void analyzeComputationPattern(const std::vector<MemoryActivity>& activities) {
        // Detect stencil computation patterns
        detectStencilPatterns(activities);
        
        // Analyze temporal locality
        analyzeTemporalLocality(activities);
        
        // Check for memory streaming patterns
        analyzeStreamingPatterns(activities);
        
        // Evaluate cache efficiency
        evaluateCacheEfficiency(activities);
    }
    
private:
    void detectStencilPatterns(const std::vector<MemoryActivity>& activities) {
        // Look for regular access patterns characteristic of stencil computations
        std::map<CUdeviceptr, std::vector<size_t>> accessSequences;
        
        for (const auto& activity : activities) {
            CUdeviceptr baseAddr = getBaseAddress(activity.srcPtr);
            size_t offset = activity.srcPtr - baseAddr;
            accessSequences[baseAddr].push_back(offset);
        }
        
        for (const auto& [baseAddr, sequence] : accessSequences) {
            if (isStencilPattern(sequence)) {
                std::cout << "Stencil pattern detected at " << std::hex << baseAddr << std::dec << std::endl;
                suggestStencilOptimizations(sequence);
            }
        }
    }
};
```

## Integration with Performance Tools

### NVIDIA Nsight Integration

```cpp
class NsightIntegration {
public:
    void exportMemoryTrace(const std::vector<MemoryActivity>& activities, const std::string& filename) {
        // Export in format compatible with Nsight Systems/Compute
        std::ofstream file(filename);
        
        file << "timestamp,operation,size,bandwidth,efficiency\n";
        
        for (const auto& activity : activities) {
            double bandwidth = calculateBandwidth(activity);
            double efficiency = calculateEfficiency(activity);
            
            file << activity.start << ","
                 << getOperationName(activity.kind) << ","
                 << activity.bytes << ","
                 << bandwidth << ","
                 << efficiency << "\n";
        }
    }
};
```

### Custom Visualization

```cpp
class MemoryVisualizer {
public:
    void generateTimelineChart(const std::vector<MemoryActivity>& activities) {
        // Generate data for memory timeline visualization
        json timeline;
        timeline["events"] = json::array();
        
        for (const auto& activity : activities) {
            json event;
            event["name"] = getOperationName(activity.kind);
            event["cat"] = "memory";
            event["ph"] = "X"; // Complete event
            event["ts"] = activity.start / 1000; // Convert to microseconds
            event["dur"] = (activity.end - activity.start) / 1000;
            event["args"]["size"] = activity.bytes;
            event["args"]["bandwidth"] = calculateBandwidth(activity);
            
            timeline["events"].push_back(event);
        }
        
        std::ofstream file("memory_timeline.json");
        file << timeline.dump(2);
    }
    
    void generateMemoryMap(const AllocationTracker& tracker) {
        // Create visual representation of memory layout
        auto allocations = tracker.getActiveAllocations();
        
        std::cout << "Memory Map:" << std::endl;
        std::cout << "Address Range          | Size     | Type" << std::endl;
        std::cout << "----------------------|----------|----------" << std::endl;
        
        for (const auto& [ptr, info] : allocations) {
            std::cout << std::hex << ptr << "-" << (ptr + info.size) << std::dec
                     << " | " << formatSize(info.size)
                     << " | " << info.allocationType << std::endl;
        }
    }
};
```

## Troubleshooting Memory Issues

### Common Memory Problems

1. **Memory Leaks**: Unfreed allocations
2. **Fragmentation**: Inefficient memory usage
3. **Bandwidth Underutilization**: Poor transfer patterns
4. **Excessive Synchronization**: Blocking memory operations

### Debug Strategies

```cpp
class MemoryDebugger {
public:
    void validateMemoryOperations(const std::vector<MemoryActivity>& activities) {
        // Check for invalid memory accesses
        std::set<CUdeviceptr> validPointers;
        
        for (const auto& activity : activities) {
            if (activity.kind == CUPTI_ACTIVITY_KIND_MEMCPY) {
                if (validPointers.find(activity.srcPtr) == validPointers.end() &&
                    validPointers.find(activity.dstPtr) == validPointers.end()) {
                    std::cerr << "Warning: Memory operation on potentially invalid pointer" << std::endl;
                }
            }
        }
    }
    
    void checkMemoryAlignment(const std::vector<MemoryActivity>& activities) {
        for (const auto& activity : activities) {
            if (activity.srcPtr % 256 != 0 || activity.dstPtr % 256 != 0) {
                std::cout << "Warning: Unaligned memory access detected" << std::endl;
            }
        }
    }
};
```

## Next Steps

- Apply memory tracing to identify bottlenecks in your applications
- Experiment with different memory optimization strategies
- Integrate memory analysis into your development workflow
- Develop custom memory management patterns based on trace analysis
- Combine with other CUPTI features for comprehensive performance profiling 