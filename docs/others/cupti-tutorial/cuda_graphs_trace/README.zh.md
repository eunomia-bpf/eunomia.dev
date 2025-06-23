# CUDA 图追踪教程

> GitHub 仓库和完整教程可在 <https://github.com/eunomia-bpf/cupti-tutorial> 获取。

## 简介

CUDA 图为优化 GPU 工作负载提供了一种强大的方法，通过捕获和重放操作序列来最小化 CPU 开销。本教程演示了如何使用 CUPTI 追踪 CUDA 图的执行，将图节点启动与其创建 API 关联，并分析图的性能特征。

## 您将学到的内容

- 如何使用 CUPTI 追踪 CUDA 图的构建和执行
- 将图节点启动与创建 API 关联的技术
- 理解 CUDA 图的性能优势和开销
- 分析图节点依赖关系和执行模式
- 调试和优化 CUDA 图的最佳实践

## 理解 CUDA 图

CUDA 图将一系列操作及其依赖关系表示为单个可执行单元。它们提供了几个优势：

- **减少启动开销**：消除重复的内核启动成本
- **优化调度**：GPU 可以优化整个图的执行
- **更好的资源利用**：改善并行性和内存带宽使用
- **可重现的执行**：多次运行的一致性能

### 图组件

1. **节点**：单个操作（内核、内存复制等）
2. **依赖关系**：定义执行顺序的边
3. **图**：节点及其关系的集合
4. **可执行图**：准备好重复执行的优化版本

## 代码架构

### 图构建流程

```cpp
// 1. 创建空图
cudaGraph_t graph;
cudaGraphCreate(&graph, 0);

// 2. 添加带依赖关系的节点
cudaGraphNode_t memcpyNode, kernelNode;
cudaGraphAddMemcpyNode(&memcpyNode, graph, dependencies, numDeps, &memcpyParams);
cudaGraphAddKernelNode(&kernelNode, graph, &memcpyNode, 1, &kernelParams);

// 3. 实例化以执行
cudaGraphExec_t graphExec;
cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

// 4. 高效地多次执行
cudaGraphLaunch(graphExec, stream);
```

### CUPTI 追踪设置

```cpp
class GraphTracer {
private:
    std::map<uint64_t, ApiData> nodeCorrelationMap;

public:
    void setupGraphTracing() {
        // 为图相关事件启用活动追踪
        cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME);
        cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);
        cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
        
        // 为图节点创建启用回调
        cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RESOURCE, 
                           CUPTI_CBID_RESOURCE_GRAPHNODE_CREATED);
        cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API);
    }
};
```

## 示例演练

### 图工作负载

我们的示例创建了一个包含多个操作的计算图：

```cpp
void DoPass(cudaStream_t stream)
{
    // 创建向量并分配内存
    int *pHostA, *pHostB, *pHostC;
    int *pDeviceA, *pDeviceB, *pDeviceC;
    
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    cudaGraphNode_t nodes[5];
    
    // 创建图
    cudaGraphCreate(&graph, 0);
    
    // 节点 0 & 1：主机到设备内存复制
    cudaMemcpy3DParms memcpyParams = {0};
    memcpyParams.kind = cudaMemcpyHostToDevice;
    memcpyParams.srcPtr.ptr = pHostA;
    memcpyParams.dstPtr.ptr = pDeviceA;
    memcpyParams.extent.width = size;
    cudaGraphAddMemcpyNode(&nodes[0], graph, NULL, 0, &memcpyParams);
    
    // 节点 2：向量加法内核（依赖于节点 0 & 1）
    cudaKernelNodeParams kernelParams;
    void* kernelArgs[] = {&pDeviceA, &pDeviceB, &pDeviceC, &num};
    kernelParams.func = (void*)VectorAdd;
    kernelParams.gridDim = dim3(blocksPerGrid, 1, 1);
    kernelParams.blockDim = dim3(threadsPerBlock, 1, 1);
    kernelParams.kernelParams = kernelArgs;
    cudaGraphAddKernelNode(&nodes[2], graph, &nodes[0], 2, &kernelParams);
    
    // 实例化并执行
    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
    cudaGraphLaunch(graphExec, stream);
}
```

这创建了一个依赖图：
```
[H2D Copy A] ──┐
               ├─► [VectorAdd] ──► [VectorSubtract] ──► [D2H Copy]
[H2D Copy B] ──┘
```

### 关联追踪

示例追踪图节点创建和执行之间的关系：

```cpp
typedef struct ApiData_st {
    const char *pFunctionName;  // 创建节点的 API
    uint32_t correlationId;     // 关联的唯一 ID
} ApiData;

// 将图节点 ID 映射到其创建 API
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
                // 记录 API 调用信息
                s_correlationId = pCallbackInfo->correlationId;
                s_pFunctionName = pCallbackInfo->functionName;
            }
            break;
            
        case CUPTI_CB_DOMAIN_RESOURCE:
            if (callbackId == CUPTI_CBID_RESOURCE_GRAPHNODE_CREATED) {
                // 跳过实例化期间创建的节点
                if (strncmp(s_pFunctionName, "cudaGraphInstantiate", 20) == 0)
                    break;
                    
                CUpti_ResourceData *pResourceData = 
                    (CUpti_ResourceData*)pCallbackInfo;
                CUpti_GraphData *callbackData = 
                    (CUpti_GraphData*)pResourceData->resourceDescriptor;
                
                // 获取节点 ID 并存储关联信息
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

## 运行示例

### 构建和执行

```bash
cd cuda_graphs_trace
make
./cuda_graphs_trace
```

### 示例输出

```
=== CUDA 图追踪分析 ===

图构建阶段:
  创建图节点: cudaGraphAddMemcpyNode (关联 ID: 1001)
  创建图节点: cudaGraphAddKernelNode (关联 ID: 1003)

图执行阶段:
  图节点执行:
    节点 ID: 0x7f8b40001000
    创建 API: cudaGraphAddMemcpyNode
    内核名称: VectorAdd
    持续时间: 255 ns

性能比较:
  图执行时间: 1250000 ns
  顺序执行时间: 2100000 ns
  加速比: 1.68x
```

## 性能分析

### 启动开销比较

图执行通过消除重复的启动开销显著提高性能。示例展示了如何测量和比较图执行与顺序执行的性能差异。

### 图拓扑分析

通过分析图的依赖结构，可以识别：
- 关键路径长度
- 并行度机会
- 潜在的优化点

## 优化策略

### 图节点合并

- 合并小的连续操作
- 减少图节点数量
- 优化内存传输模式

### 动态图更新

- 重用图结构
- 更新参数而非重建
- 缓存常用图模式

## 调试技巧

### 图验证

- 检查循环依赖
- 验证节点参数
- 识别资源冲突

## 总结

CUDA 图追踪为理解和优化图形工作负载提供了强大的见解。通过使用 CUPTI 追踪图的构建和执行：

### 关键优势

- **可见性**：完整的图生命周期视图
- **关联**：将执行链接到创建 API
- **优化机会**：识别瓶颈和改进区域
- **性能验证**：量化图的好处

### 最佳实践

1. **使用图进行重复工作负载**：最大化启动开销减少
2. **优化图拓扑**：最小化同步点和依赖链
3. **合并小操作**：减少图节点开销
4. **验证图正确性**：在部署前进行彻底测试

CUDA 图代表了 GPU 计算的重要进步，通过 CUPTI 追踪提供了掌握这项强大技术所需的见解。 