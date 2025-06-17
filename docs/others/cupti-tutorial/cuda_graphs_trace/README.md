# CUDA Graphs Tracing Tutorial

> The GitHub repo and complete tutorial is available at <https://github.com/eunomia-bpf/cupti-tutorial>.

## Introduction

CUDA Graphs provide a powerful way to optimize GPU workloads by capturing and replaying sequences of operations with minimal CPU overhead. This tutorial demonstrates how to use CUPTI to trace CUDA Graph execution, correlate graph node launches with their creation APIs, and analyze graph performance characteristics.

## What You'll Learn

- How to trace CUDA Graph construction and execution using CUPTI
- Techniques for correlating graph node launches with creation APIs
- Understanding CUDA Graph performance benefits and overhead
- Analyzing graph node dependencies and execution patterns
- Best practices for debugging and optimizing CUDA Graphs

## Understanding CUDA Graphs

CUDA Graphs represent a collection of operations and their dependencies as a single executable unit. They provide several advantages:

- **Reduced Launch Overhead**: Eliminate repeated kernel launch costs
- **Optimized Scheduling**: GPU can optimize execution across the entire graph
- **Better Resource Utilization**: Improved parallelism and memory bandwidth usage
- **Reproducible Execution**: Consistent performance across multiple runs

### Graph Components

1. **Nodes**: Individual operations (kernels, memory copies, etc.)
2. **Dependencies**: Edges defining execution order
3. **Graph**: Collection of nodes and their relationships
4. **Executable Graph**: Optimized version ready for repeated execution

## Code Architecture

### Graph Construction Flow

```cpp
// 1. Create empty graph
cudaGraph_t graph;
cudaGraphCreate(&graph, 0);

// 2. Add nodes with dependencies
cudaGraphNode_t memcpyNode, kernelNode;
cudaGraphAddMemcpyNode(&memcpyNode, graph, dependencies, numDeps, &memcpyParams);
cudaGraphAddKernelNode(&kernelNode, graph, &memcpyNode, 1, &kernelParams);

// 3. Instantiate for execution
cudaGraphExec_t graphExec;
cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

// 4. Execute multiple times efficiently
cudaGraphLaunch(graphExec, stream);
```

### CUPTI Tracing Setup

```cpp
class GraphTracer {
private:
    std::map<uint64_t, ApiData> nodeCorrelationMap;

public:
    void setupGraphTracing() {
        // Enable activity tracing for graph-related events
        cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME);
        cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);
        cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
        
        // Enable callbacks for graph node creation
        cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RESOURCE, 
                           CUPTI_CBID_RESOURCE_GRAPHNODE_CREATED);
        cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RESOURCE, 
                           CUPTI_CBID_RESOURCE_GRAPHNODE_CLONED);
        cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API);
    }
};
```

## Sample Walkthrough

### The Graph Workload

Our sample creates a computation graph with multiple operations:

```cpp
void DoPass(cudaStream_t stream)
{
    // Create vectors and allocate memory
    int *pHostA, *pHostB, *pHostC;
    int *pDeviceA, *pDeviceB, *pDeviceC;
    
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    cudaGraphNode_t nodes[5];
    
    // Create the graph
    cudaGraphCreate(&graph, 0);
    
    // Node 0 & 1: Host-to-Device memory copies
    cudaMemcpy3DParms memcpyParams = {0};
    memcpyParams.kind = cudaMemcpyHostToDevice;
    memcpyParams.srcPtr.ptr = pHostA;
    memcpyParams.dstPtr.ptr = pDeviceA;
    memcpyParams.extent.width = size;
    cudaGraphAddMemcpyNode(&nodes[0], graph, NULL, 0, &memcpyParams);
    
    memcpyParams.srcPtr.ptr = pHostB;
    memcpyParams.dstPtr.ptr = pDeviceB;
    cudaGraphAddMemcpyNode(&nodes[1], graph, NULL, 0, &memcpyParams);
    
    // Node 2: Vector addition kernel (depends on nodes 0 & 1)
    cudaKernelNodeParams kernelParams;
    void* kernelArgs[] = {&pDeviceA, &pDeviceB, &pDeviceC, &num};
    kernelParams.func = (void*)VectorAdd;
    kernelParams.gridDim = dim3(blocksPerGrid, 1, 1);
    kernelParams.blockDim = dim3(threadsPerBlock, 1, 1);
    kernelParams.kernelParams = kernelArgs;
    cudaGraphAddKernelNode(&nodes[2], graph, &nodes[0], 2, &kernelParams);
    
    // Node 3: Vector subtraction kernel (depends on node 2)
    kernelParams.func = (void*)VectorSubtract;
    cudaGraphAddKernelNode(&nodes[3], graph, &nodes[2], 1, &kernelParams);
    
    // Node 4: Device-to-Host memory copy (depends on node 3)
    memcpyParams.kind = cudaMemcpyDeviceToHost;
    memcpyParams.srcPtr.ptr = pDeviceC;
    memcpyParams.dstPtr.ptr = pHostC;
    cudaGraphAddMemcpyNode(&nodes[4], graph, &nodes[3], 1, &memcpyParams);
    
    // Instantiate and execute
    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
    cudaGraphLaunch(graphExec, stream);
}
```

This creates a dependency graph:
```
[H2D Copy A] ──┐
               ├─► [VectorAdd] ──► [VectorSubtract] ──► [D2H Copy]
[H2D Copy B] ──┘
```

### Correlation Tracking

The sample tracks the relationship between graph node creation and execution:

```cpp
typedef struct ApiData_st {
    const char *pFunctionName;  // API that created the node
    uint32_t correlationId;     // Unique ID for correlation
} ApiData;

// Map graph node ID to its creation API
std::map<uint64_t, ApiData> nodeIdCorrelationMap;

void GraphsCallbackHandler(void *pUserData, CUpti_CallbackDomain domain,
                          CUpti_CallbackId callbackId, 
                          const CUpti_CallbackData *pCallbackInfo)
{
    static const char *s_pFunctionName;
    static uint32_t s_correlationId;
    
    switch (domain) {
        case CUPTI_CB_DOMAIN_RUNTIME_API:
            if (pCallbackInfo->callbackSite == CUPTI_API_ENTER) {
                // Record API call information
                s_correlationId = pCallbackInfo->correlationId;
                s_pFunctionName = pCallbackInfo->functionName;
            }
            break;
            
        case CUPTI_CB_DOMAIN_RESOURCE:
            if (callbackId == CUPTI_CBID_RESOURCE_GRAPHNODE_CREATED) {
                // Skip nodes created during instantiation
                if (strncmp(s_pFunctionName, "cudaGraphInstantiate", 20) == 0)
                    break;
                    
                CUpti_ResourceData *pResourceData = 
                    (CUpti_ResourceData*)pCallbackInfo;
                CUpti_GraphData *callbackData = 
                    (CUpti_GraphData*)pResourceData->resourceDescriptor;
                
                // Get node ID and store correlation info
                uint64_t nodeId;
                cuptiGetGraphNodeId(callbackData->node, &nodeId);
                
                ApiData apiData;
                apiData.correlationId = s_correlationId;
                apiData.pFunctionName = s_pFunctionName;
                nodeIdCorrelationMap[nodeId] = apiData;
            }
            break;
    }
}
```

### Activity Record Processing

```cpp
void GraphTraceRecords(CUpti_Activity *pRecord)
{
    switch (pRecord->kind) {
        case CUPTI_ACTIVITY_KIND_MEMCPY:
        {
            CUpti_ActivityMemcpy6 *pMemcpyRecord = 
                (CUpti_ActivityMemcpy6*)pRecord;
            
            // Find creation API for this memory copy node
            auto it = nodeIdCorrelationMap.find(pMemcpyRecord->graphNodeId);
            if (it != nodeIdCorrelationMap.end()) {
                printf("Memcpy node created by %s (correlation: %u)\n",
                       it->second.pFunctionName, it->second.correlationId);
            }
            break;
        }
        
        case CUPTI_ACTIVITY_KIND_KERNEL:
        case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
        {
            CUpti_ActivityKernel9 *pKernelRecord = 
                (CUpti_ActivityKernel9*)pRecord;
            
            // Find creation API for this kernel node
            auto it = nodeIdCorrelationMap.find(pKernelRecord->graphNodeId);
            if (it != nodeIdCorrelationMap.end()) {
                printf("Kernel node created by %s (correlation: %u)\n",
                       it->second.pFunctionName, it->second.correlationId);
            }
            break;
        }
    }
}
```

## Building and Running

### Prerequisites

- CUDA Toolkit 10.0 or later (for CUDA Graphs support)
- CUPTI library
- GPU with compute capability 3.5 or higher

### Build Process

```bash
cd cuda_graphs_trace
make
```

### Execution

```bash
./cuda_graphs_trace
```

### Sample Output

```
Device Name: NVIDIA GeForce RTX 3080

[CUPTI] CUPTI: Found context 0x7f8b4c002020
Graph node was created using API cudaGraphAddMemcpyNode with correlationId 1
Graph node was created using API cudaGraphAddMemcpyNode with correlationId 2
Graph node was created using API cudaGraphAddKernelNode with correlationId 3
Graph node was created using API cudaGraphAddKernelNode with correlationId 4
Graph node was created using API cudaGraphAddMemcpyNode with correlationId 5

Activity records processed: 23
```

## Advanced Graph Analysis

### Graph Topology Analysis

```cpp
class GraphAnalyzer {
private:
    struct NodeInfo {
        uint64_t nodeId;
        std::string nodeType;
        std::vector<uint64_t> dependencies;
        std::vector<uint64_t> dependents;
        double executionTime;
    };
    
    std::map<uint64_t, NodeInfo> graphNodes;

public:
    void analyzeGraphStructure(const std::vector<CUpti_Activity*>& activities) {
        // Build graph topology from activity records
        for (auto* activity : activities) {
            NodeInfo info = extractNodeInfo(activity);
            graphNodes[info.nodeId] = info;
        }
        
        // Analyze critical path
        std::vector<uint64_t> criticalPath = findCriticalPath();
        
        // Report results
        printf("Graph contains %zu nodes\n", graphNodes.size());
        printf("Critical path length: %zu nodes\n", criticalPath.size());
        printf("Estimated critical path time: %.3f ms\n", 
               calculateCriticalPathTime(criticalPath));
    }
    
private:
    std::vector<uint64_t> findCriticalPath() {
        // Implement topological sort and longest path algorithm
        std::vector<uint64_t> path;
        std::map<uint64_t, double> distances;
        
        // Initialize distances
        for (auto& [nodeId, info] : graphNodes) {
            distances[nodeId] = 0.0;
        }
        
        // Find longest path (critical path)
        std::queue<uint64_t> queue;
        std::map<uint64_t, int> inDegree;
        
        // Calculate in-degrees
        for (auto& [nodeId, info] : graphNodes) {
            inDegree[nodeId] = info.dependencies.size();
            if (inDegree[nodeId] == 0) {
                queue.push(nodeId);
            }
        }
        
        // Process nodes in topological order
        while (!queue.empty()) {
            uint64_t current = queue.front();
            queue.pop();
            
            for (uint64_t dependent : graphNodes[current].dependents) {
                double newDistance = distances[current] + 
                                   graphNodes[current].executionTime;
                if (newDistance > distances[dependent]) {
                    distances[dependent] = newDistance;
                }
                
                inDegree[dependent]--;
                if (inDegree[dependent] == 0) {
                    queue.push(dependent);
                }
            }
        }
        
        // Backtrack to find critical path
        return backtrackCriticalPath(distances);
    }
};
```

### Performance Comparison

```cpp
class GraphPerformanceAnalyzer {
public:
    struct PerformanceMetrics {
        double graphExecutionTime;
        double traditionalExecutionTime;
        double speedup;
        size_t launchOverheadReduction;
    };
    
    PerformanceMetrics compareGraphVsTraditional() {
        PerformanceMetrics metrics;
        
        // Measure graph execution
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; i++) {
            cudaGraphLaunch(graphExec, stream);
            cudaStreamSynchronize(stream);
        }
        auto end = std::chrono::high_resolution_clock::now();
        metrics.graphExecutionTime = 
            std::chrono::duration<double>(end - start).count();
        
        // Measure traditional execution
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; i++) {
            executeTraditionalWorkload(stream);
            cudaStreamSynchronize(stream);
        }
        end = std::chrono::high_resolution_clock::now();
        metrics.traditionalExecutionTime = 
            std::chrono::duration<double>(end - start).count();
        
        // Calculate metrics
        metrics.speedup = metrics.traditionalExecutionTime / 
                         metrics.graphExecutionTime;
        metrics.launchOverheadReduction = 
            estimateLaunchOverheadReduction();
        
        return metrics;
    }
    
private:
    void executeTraditionalWorkload(cudaStream_t stream) {
        // Execute same operations without graphs
        cudaMemcpyAsync(pDeviceA, pHostA, size, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(pDeviceB, pHostB, size, cudaMemcpyHostToDevice, stream);
        VectorAdd<<<blocks, threads, 0, stream>>>(pDeviceA, pDeviceB, pDeviceC, N);
        VectorSubtract<<<blocks, threads, 0, stream>>>(pDeviceA, pDeviceB, pDeviceC, N);
        cudaMemcpyAsync(pHostC, pDeviceC, size, cudaMemcpyDeviceToHost, stream);
    }
};
```

### Graph Optimization Analysis

```cpp
class GraphOptimizer {
public:
    struct OptimizationReport {
        std::vector<std::string> suggestions;
        double estimatedImprovement;
        size_t unnecessaryDependencies;
        size_t parallelizableNodes;
    };
    
    OptimizationReport analyzeOptimizationOpportunities(
        const std::map<uint64_t, NodeInfo>& graphNodes) {
        
        OptimizationReport report;
        
        // Check for unnecessary serialization
        report.unnecessaryDependencies = findUnnecessaryDependencies(graphNodes);
        if (report.unnecessaryDependencies > 0) {
            report.suggestions.push_back(
                "Remove " + std::to_string(report.unnecessaryDependencies) + 
                " unnecessary dependencies to improve parallelism");
        }
        
        // Identify parallelizable operations
        report.parallelizableNodes = findParallelizableNodes(graphNodes);
        if (report.parallelizableNodes > 0) {
            report.suggestions.push_back(
                "Consider fusing " + std::to_string(report.parallelizableNodes) + 
                " nodes for better performance");
        }
        
        // Check memory access patterns
        if (hasInefficiientMemoryAccess(graphNodes)) {
            report.suggestions.push_back(
                "Optimize memory access patterns to reduce bandwidth usage");
        }
        
        // Estimate overall improvement potential
        report.estimatedImprovement = calculateImprovementPotential(report);
        
        return report;
    }
    
private:
    size_t findUnnecessaryDependencies(
        const std::map<uint64_t, NodeInfo>& nodes) {
        // Analyze dependency graph for unnecessary edges
        size_t unnecessary = 0;
        
        for (auto& [nodeId, info] : nodes) {
            for (uint64_t dep : info.dependencies) {
                if (!isNecessaryDependency(nodeId, dep, nodes)) {
                    unnecessary++;
                }
            }
        }
        
        return unnecessary;
    }
};
```

## Real-World Applications

### Streaming Workloads

```cpp
class StreamingGraphManager {
private:
    std::vector<cudaGraphExec_t> pipelineStages;
    std::vector<cudaStream_t> streams;

public:
    void setupStreamingPipeline() {
        // Create graphs for different pipeline stages
        for (int stage = 0; stage < numStages; stage++) {
            cudaGraph_t graph;
            cudaGraphCreate(&graph, 0);
            
            // Add stage-specific operations
            addStageOperations(graph, stage);
            
            cudaGraphExec_t graphExec;
            cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
            pipelineStages.push_back(graphExec);
            
            // Create stream for this stage
            cudaStream_t stream;
            cudaStreamCreate(&stream);
            streams.push_back(stream);
        }
    }
    
    void executePipeline(void* inputData, size_t dataSize) {
        // Launch all pipeline stages concurrently
        for (int stage = 0; stage < numStages; stage++) {
            cudaGraphLaunch(pipelineStages[stage], streams[stage]);
        }
        
        // Synchronize all stages
        for (auto stream : streams) {
            cudaStreamSynchronize(stream);
        }
    }
};
```

### Machine Learning Inference

```cpp
class MLInferenceGraph {
private:
    cudaGraphExec_t inferenceGraph;
    std::vector<float*> layerInputs;
    std::vector<float*> layerOutputs;

public:
    void buildInferenceGraph(const ModelConfig& config) {
        cudaGraph_t graph;
        cudaGraphCreate(&graph, 0);
        
        std::vector<cudaGraphNode_t> layerNodes;
        
        // Add nodes for each layer
        for (int layer = 0; layer < config.numLayers; layer++) {
            cudaGraphNode_t node;
            cudaKernelNodeParams params = createLayerParams(layer, config);
            
            // Add dependencies to previous layer
            cudaGraphNode_t* deps = (layer > 0) ? &layerNodes[layer-1] : nullptr;
            int numDeps = (layer > 0) ? 1 : 0;
            
            cudaGraphAddKernelNode(&node, graph, deps, numDeps, &params);
            layerNodes.push_back(node);
        }
        
        // Instantiate the complete inference graph
        cudaGraphInstantiate(&inferenceGraph, graph, NULL, NULL, 0);
    }
    
    void runInference(float* inputData, float* outputData) {
        // Copy input data
        cudaMemcpy(layerInputs[0], inputData, inputSize, cudaMemcpyHostToDevice);
        
        // Execute entire inference pipeline as single graph
        cudaGraphLaunch(inferenceGraph, defaultStream);
        cudaStreamSynchronize(defaultStream);
        
        // Copy output data
        int lastLayer = layerOutputs.size() - 1;
        cudaMemcpy(outputData, layerOutputs[lastLayer], outputSize, 
                   cudaMemcpyDeviceToHost);
    }
};
```

## Debugging Graph Issues

### Graph Validation

```cpp
class GraphValidator {
public:
    struct ValidationResult {
        bool isValid;
        std::vector<std::string> errors;
        std::vector<std::string> warnings;
    };
    
    ValidationResult validateGraph(cudaGraph_t graph) {
        ValidationResult result;
        result.isValid = true;
        
        // Check for cycles
        if (hasCycles(graph)) {
            result.isValid = false;
            result.errors.push_back("Graph contains cycles");
        }
        
        // Check for disconnected nodes
        std::vector<cudaGraphNode_t> disconnectedNodes = 
            findDisconnectedNodes(graph);
        if (!disconnectedNodes.empty()) {
            result.warnings.push_back(
                "Found " + std::to_string(disconnectedNodes.size()) + 
                " disconnected nodes");
        }
        
        // Validate memory dependencies
        if (!validateMemoryDependencies(graph)) {
            result.isValid = false;
            result.errors.push_back("Invalid memory dependencies detected");
        }
        
        return result;
    }
    
private:
    bool hasCycles(cudaGraph_t graph) {
        // Implement cycle detection algorithm
        std::map<cudaGraphNode_t, int> colors; // 0=white, 1=gray, 2=black
        std::vector<cudaGraphNode_t> nodes = getGraphNodes(graph);
        
        for (auto node : nodes) {
            colors[node] = 0;
        }
        
        for (auto node : nodes) {
            if (colors[node] == 0) {
                if (dfsHasCycle(node, colors, graph)) {
                    return true;
                }
            }
        }
        
        return false;
    }
};
```

### Performance Debugging

```cpp
class GraphPerformanceDebugger {
public:
    void analyzeGraphBottlenecks(const std::vector<CUpti_Activity*>& activities) {
        std::map<uint64_t, double> nodeExecutionTimes;
        std::map<uint64_t, std::string> nodeTypes;
        
        // Extract execution times for each node
        for (auto* activity : activities) {
            uint64_t nodeId = extractNodeId(activity);
            double execTime = extractExecutionTime(activity);
            std::string type = extractNodeType(activity);
            
            nodeExecutionTimes[nodeId] = execTime;
            nodeTypes[nodeId] = type;
        }
        
        // Find bottlenecks
        std::vector<std::pair<uint64_t, double>> sortedNodes;
        for (auto& [nodeId, time] : nodeExecutionTimes) {
            sortedNodes.push_back({nodeId, time});
        }
        
        std::sort(sortedNodes.begin(), sortedNodes.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        
        // Report top bottlenecks
        printf("Top 5 bottleneck nodes:\n");
        for (int i = 0; i < std::min(5, (int)sortedNodes.size()); i++) {
            uint64_t nodeId = sortedNodes[i].first;
            double time = sortedNodes[i].second;
            std::string type = nodeTypes[nodeId];
            
            printf("  Node %lu (%s): %.3f ms (%.1f%% of total)\n",
                   nodeId, type.c_str(), time * 1000,
                   (time / getTotalExecutionTime()) * 100);
        }
    }
};
```

## Integration with Profiling Tools

### NVIDIA Nsight Integration

```cpp
class NsightGraphExporter {
public:
    void exportGraphProfile(const std::string& filename,
                           const std::vector<CUpti_Activity*>& activities) {
        json profile;
        profile["traceEvents"] = json::array();
        
        for (auto* activity : activities) {
            json event;
            event["name"] = getNodeName(activity);
            event["cat"] = "graph";
            event["ph"] = "X";
            event["ts"] = getTimestamp(activity);
            event["dur"] = getDuration(activity);
            event["pid"] = 0;
            event["tid"] = getNodeId(activity);
            
            // Add graph-specific metadata
            event["args"]["nodeId"] = getNodeId(activity);
            event["args"]["nodeType"] = getNodeType(activity);
            event["args"]["graphId"] = getGraphId(activity);
            
            profile["traceEvents"].push_back(event);
        }
        
        std::ofstream file(filename);
        file << profile.dump(2);
    }
};
```

## Best Practices and Optimization Tips

### Graph Construction Best Practices

1. **Minimize Graph Construction Overhead**: Build graphs once, execute many times
2. **Optimize Dependencies**: Only add necessary dependencies
3. **Consider Memory Locality**: Group related operations
4. **Use Appropriate Node Types**: Choose the most efficient node type for each operation

### Performance Optimization

```cpp
class GraphOptimizationTips {
public:
    void applyOptimizations(cudaGraph_t& graph) {
        // 1. Fuse compatible kernels
        fuseCompatibleKernels(graph);
        
        // 2. Optimize memory transfers
        optimizeMemoryTransfers(graph);
        
        // 3. Remove redundant synchronization
        removeRedundantSynchronization(graph);
        
        // 4. Apply memory coalescing optimizations
        optimizeMemoryCoalescing(graph);
    }
    
private:
    void fuseCompatibleKernels(cudaGraph_t& graph) {
        // Identify kernels that can be fused
        auto fusableKernels = findFusableKernels(graph);
        
        for (auto& kernelGroup : fusableKernels) {
            if (kernelGroup.size() > 1) {
                printf("Fusing %zu kernels for better performance\n", 
                       kernelGroup.size());
                fuseKernelGroup(graph, kernelGroup);
            }
        }
    }
};
```

## Next Steps

- Experiment with different graph topologies to understand performance implications
- Implement graph-based optimizations in your own CUDA applications
- Explore advanced graph features like conditional nodes and nested graphs
- Combine graph tracing with other CUPTI profiling capabilities
- Develop automated graph optimization tools based on trace analysis

CUDA Graphs represent a significant advancement in GPU programming, and proper profiling with CUPTI can help you maximize their benefits in your applications. 