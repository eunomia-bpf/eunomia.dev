# CUPTI NVTX 集成教程

> GitHub 仓库和完整教程可在 <https://github.com/eunomia-bpf/cupti-tutorial> 获取。

## 简介

NVTX（NVIDIA 工具扩展）提供了一种强大的方法来为您的 CUDA 应用程序添加自定义范围、标记和元数据注释。本示例演示了如何将 NVTX 与 CUPTI 集成以捕获和分析自定义注释，从而更好地理解应用程序结构和性能瓶颈。

## 您将学到的内容

- 如何使用 NVTX 注释对应用程序进行插桩
- 创建用于有组织分析的自定义域
- 使用 push/pop 和 start/end 范围模式
- 注册字符串和命名资源
- 将 NVTX 与 CUPTI 集成以进行全面分析
- NVTX 插桩的最佳实践

## 关键概念

### NVTX 注释

NVTX 提供几种注释类型：
- **范围**：标记代码段的持续时间（push/pop 或 start/end）
- **标记**：时间点事件
- **域**：注释的逻辑分组
- **类别**：事件的分类
- **消息**：注释的描述性文本

### NVTX 域

```cpp
// 为向量操作创建自定义域
nvtxDomainHandle_t domain = nvtxDomainCreateA("Vector Addition");

// 使用域特定的注释
nvtxDomainRangePushA(domain, "Memory Allocation");
// ... 代码 ...
nvtxDomainRangePop(domain);
```

## 示例架构

### NVTX 设置和集成

```cpp
// 必需的环境变量设置
// Linux: export NVTX_INJECTION64_PATH=<path>/libcupti.so
// Windows: set NVTX_INJECTION64_PATH=<path>/cupti.dll

#include "nvtx3/nvToolsExt.h"
#include "nvtx3/nvToolsExtCuda.h"
#include "nvtx3/nvToolsExtCudaRt.h"
#include "generated_nvtx_meta.h"

// CUPTI 回调集成
void CUPTIAPI NvtxCallbackHandler(void* userdata, 
                                 CUpti_CallbackDomain domain,
                                 CUpti_CallbackId callbackId, 
                                 const CUpti_CallbackData* callbackInfo);
```

### 事件属性结构

```cpp
nvtxEventAttributes_t eventAttrib = {0};
eventAttrib.version = NVTX_VERSION;
eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
eventAttrib.colorType = NVTX_COLOR_ARGB;
eventAttrib.color = 0x0000ff;  // 蓝色
eventAttrib.message.ascii = "Custom Operation";
```

## 示例演练

### 带有 NVTX 的应用程序结构

```cpp
void DoVectorAddition() {
    // 创建自定义域
    nvtxDomainHandle_t domain = nvtxDomainCreateA("Vector Addition");
    
    // 为更好的识别命名 CUDA 资源
    CUdevice device;
    cuDeviceGet(&device, 0);
    nvtxNameCuDeviceA(device, "CUDA Device 0");
    
    // 配置事件属性
    nvtxEventAttributes_t eventAttrib = {0};
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.colorType = NVTX_COLOR_ARGB;
    eventAttrib.color = 0x0000ff;
    
    // 标记主要操作
    eventAttrib.message.ascii = "vectorAdd";
    nvtxDomainRangePushEx(domain, &eventAttrib);
    
    // 阶段 1：在默认域上进行内存分配
    nvtxRangePushA("Allocate host memory");
    // 分配主机内存
    pHostA = (int*)malloc(size);
    pHostB = (int*)malloc(size);
    pHostC = (int*)malloc(size);
    nvtxRangePop();
    
    // 阶段 2：在自定义域上进行设备内存分配
    eventAttrib.message.ascii = "Allocate device memory";
    nvtxDomainRangePushEx(domain, &eventAttrib);
    cudaMalloc((void**)&pDeviceA, size);
    cudaMalloc((void**)&pDeviceB, size);
    cudaMalloc((void**)&pDeviceC, size);
    nvtxDomainRangePop(domain);
    
    // 阶段 3：使用注册字符串进行内存传输
    nvtxStringHandle_t string = nvtxDomainRegisterStringA(domain, "Memcpy operation");
    eventAttrib.message.registered = string;
    nvtxDomainRangePushEx(domain, &eventAttrib);
    cudaMemcpy(pDeviceA, pHostA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(pDeviceB, pHostB, size, cudaMemcpyHostToDevice);
    nvtxDomainRangePop(domain);
    
    // 阶段 4：使用 start/end 模式执行内核
    eventAttrib.message.ascii = "Launch kernel";
    nvtxRangeId_t id = nvtxDomainRangeStartEx(domain, &eventAttrib);
    VectorAdd<<<blocksPerGrid, threadsPerBlock>>>(pDeviceA, pDeviceB, pDeviceC, N);
    cudaDeviceSynchronize();
    nvtxDomainRangeEnd(domain, id);
    
    // 阶段 5：结果传输
    eventAttrib.message.registered = string; // 重用注册字符串
    nvtxDomainRangePushEx(domain, &eventAttrib);
    cudaMemcpy(pHostC, pDeviceC, size, cudaMemcpyDeviceToHost);
    nvtxDomainRangePop(domain);
    
    // 清理
    nvtxDomainRangePop(domain); // 结束主要 vectorAdd 范围
}
```

### 资源命名

```cpp
void NameCudaResources() {
    // 命名 GPU 设备
    CUdevice device;
    cuDeviceGet(&device, 0);
    nvtxNameCuDeviceA(device, "Primary GPU");
    
    // 命名 CUDA 上下文
    CUcontext context;
    cuCtxCreate(&context, 0, device);
    nvtxNameCuContextA(context, "Vector Addition Context");
    
    // 命名 CUDA 流
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    nvtxNameCudaStreamA(stream1, "Computation Stream");
    nvtxNameCudaStreamA(stream2, "Memory Transfer Stream");
}
```

## 高级 NVTX 技术

### 多域组织

```cpp
class NvtxDomainManager {
private:
    std::map<std::string, nvtxDomainHandle_t> domains;
    
public:
    nvtxDomainHandle_t GetDomain(const std::string& name) {
        auto it = domains.find(name);
        if (it == domains.end()) {
            nvtxDomainHandle_t domain = nvtxDomainCreateA(name.c_str());
            domains[name] = domain;
            return domain;
        }
        return it->second;
    }
    
    void AnnotateMemoryOperations() {
        auto memDomain = GetDomain("Memory Operations");
        
        nvtxDomainRangePushA(memDomain, "Host to Device Transfer");
        // ... 内存传输代码 ...
        nvtxDomainRangePop(memDomain);
    }
    
    void AnnotateComputation() {
        auto computeDomain = GetDomain("Computation");
        
        nvtxDomainRangePushA(computeDomain, "Matrix Multiplication");
        // ... 计算代码 ...
        nvtxDomainRangePop(computeDomain);
    }
};
```

### 分层注释

```cpp
class HierarchicalAnnotator {
public:
    void AnnotateTrainingLoop() {
        nvtxRangePushA("Training Loop");
        
        for (int epoch = 0; epoch < numEpochs; epoch++) {
            std::string epochName = "Epoch " + std::to_string(epoch);
            nvtxRangePushA(epochName.c_str());
            
            AnnotateForwardPass();
            AnnotateBackwardPass();
            AnnotateParameterUpdate();
            
            nvtxRangePop(); // 结束 epoch
        }
        
        nvtxRangePop(); // 结束训练循环
    }
    
private:
    void AnnotateForwardPass() {
        nvtxRangePushA("Forward Pass");
        
        nvtxRangePushA("Data Loading");
        // 数据加载代码
        nvtxRangePop();
        
        nvtxRangePushA("Model Inference");
        // 模型推理代码
        nvtxRangePop();
        
        nvtxRangePushA("Loss Calculation");
        // 损失计算代码
        nvtxRangePop();
        
        nvtxRangePop(); // 结束前向传播
    }
};
```

### 颜色编码和分类

```cpp
class ColorCodedAnnotator {
private:
    enum class OperationType {
        MEMORY = 0xFF0000,    // 红色
        COMPUTE = 0x00FF00,   // 绿色
        COMMUNICATION = 0x0000FF, // 蓝色
        SYNCHRONIZATION = 0xFFFF00 // 黄色
    };

public:
    void AnnotateWithColor(const std::string& message, OperationType type) {
        nvtxEventAttributes_t eventAttrib = {0};
        eventAttrib.version = NVTX_VERSION;
        eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
        eventAttrib.message.ascii = message.c_str();
        eventAttrib.colorType = NVTX_COLOR_ARGB;
        eventAttrib.color = static_cast<uint32_t>(type);
        eventAttrib.category = static_cast<uint32_t>(type);
        
        nvtxRangePushEx(&eventAttrib);
    }
    
    void EndAnnotation() {
        nvtxRangePop();
    }
};
```

## 性能分析集成

### CUPTI 回调处理

```cpp
void CUPTIAPI NvtxCallbackHandler(void* userdata, 
                                 CUpti_CallbackDomain domain,
                                 CUpti_CallbackId callbackId, 
                                 const CUpti_CallbackData* callbackInfo) {
    
    if (domain == CUPTI_CB_DOMAIN_NVTX) {
        switch (callbackId) {
            case CUPTI_CBID_NVTX_nvtxRangePushA:
            case CUPTI_CBID_NVTX_nvtxRangePushEx:
                printf("NVTX 范围开始: %s\n", 
                       getNvtxMessage(callbackInfo));
                break;
                
            case CUPTI_CBID_NVTX_nvtxRangePop:
                printf("NVTX 范围结束\n");
                break;
                
            case CUPTI_CBID_NVTX_nvtxMarkA:
            case CUPTI_CBID_NVTX_nvtxMarkEx:
                printf("NVTX 标记: %s\n", 
                       getNvtxMessage(callbackInfo));
                break;
        }
    }
}
```

### 活动记录分析

```cpp
class NvtxActivityAnalyzer {
private:
    struct NvtxRange {
        std::string message;
        uint64_t startTime;
        uint64_t endTime;
        uint32_t threadId;
        std::string domain;
    };
    
    std::vector<NvtxRange> ranges;

public:
    void processNvtxActivity(CUpti_ActivityNvtx* nvtxActivity) {
        NvtxRange range;
        range.message = std::string(nvtxActivity->text);
        range.startTime = nvtxActivity->start;
        range.endTime = nvtxActivity->end;
        range.threadId = nvtxActivity->threadId;
        
        ranges.push_back(range);
    }
    
    void generateAnalysisReport() {
        printf("\n=== NVTX 分析报告 ===\n");
        
        // 按持续时间排序范围
        std::sort(ranges.begin(), ranges.end(),
                 [](const NvtxRange& a, const NvtxRange& b) {
                     return (a.endTime - a.startTime) > (b.endTime - b.startTime);
                 });
        
        printf("耗时最长的操作:\n");
        for (int i = 0; i < std::min(10, (int)ranges.size()); i++) {
            const auto& range = ranges[i];
            uint64_t duration = range.endTime - range.startTime;
            printf("  %s: %llu ns\n", range.message.c_str(), duration);
        }
        
        // 分析嵌套层次
        analyzeNesting();
    }
    
private:
    void analyzeNesting() {
        printf("\n嵌套分析:\n");
        
        std::stack<const NvtxRange*> rangeStack;
        int maxDepth = 0;
        int currentDepth = 0;
        
        // 按开始时间排序以分析嵌套
        auto sortedRanges = ranges;
        std::sort(sortedRanges.begin(), sortedRanges.end(),
                 [](const NvtxRange& a, const NvtxRange& b) {
                     return a.startTime < b.startTime;
                 });
        
        for (const auto& range : sortedRanges) {
            // 弹出已结束的范围
            while (!rangeStack.empty() && 
                   rangeStack.top()->endTime <= range.startTime) {
                rangeStack.pop();
                currentDepth--;
            }
            
            rangeStack.push(&range);
            currentDepth++;
            maxDepth = std::max(maxDepth, currentDepth);
        }
        
        printf("最大嵌套深度: %d\n", maxDepth);
    }
};
```

## 构建和运行

### 环境设置

```bash
# Linux
export NVTX_INJECTION64_PATH=/path/to/libcupti.so

# Windows  
set NVTX_INJECTION64_PATH=C:\path\to\cupti.dll
```

### 编译

```bash
cd cupti_nvtx
make
```

### 执行

```bash
./cupti_nvtx
```

## 示例输出

```
=== NVTX 集成分析 ===

NVTX 范围开始: vectorAdd
NVTX 范围开始: Allocate host memory
NVTX 范围结束
NVTX 范围开始: Allocate device memory  
NVTX 范围结束
NVTX 范围开始: Memcpy operation
NVTX 范围结束
NVTX 范围开始: Launch kernel
NVTX 范围结束
NVTX 范围开始: Memcpy operation
NVTX 范围结束
NVTX 范围结束

=== NVTX 分析报告 ===
耗时最长的操作:
  Launch kernel: 125000 ns
  Memcpy operation: 45000 ns
  Allocate device memory: 12000 ns
  Allocate host memory: 8000 ns

嵌套分析:
最大嵌套深度: 2
```

## 最佳实践

### 命名约定

```cpp
class NvtxNamingConventions {
public:
    // 使用描述性和一致的命名
    static void annotateKernelLaunch(const std::string& kernelName) {
        std::string annotation = "Kernel: " + kernelName;
        nvtxRangePushA(annotation.c_str());
    }
    
    static void annotateMemoryOperation(const std::string& operation, 
                                       size_t bytes) {
        std::string annotation = "Memory " + operation + " (" + 
                               std::to_string(bytes) + " bytes)";
        nvtxRangePushA(annotation.c_str());
    }
    
    static void annotatePhase(const std::string& phaseName, int iteration) {
        std::string annotation = phaseName + " [Iter " + 
                               std::to_string(iteration) + "]";
        nvtxRangePushA(annotation.c_str());
    }
};
```

### 性能考虑

```cpp
class PerformanceOptimizedNvtx {
private:
    static bool profilingEnabled;
    
public:
    // 条件注释以减少开销
    static void conditionalAnnotate(const std::string& message) {
        if (profilingEnabled) {
            nvtxRangePushA(message.c_str());
        }
    }
    
    static void conditionalPop() {
        if (profilingEnabled) {
            nvtxRangePop();
        }
    }
    
    // RAII 封装器用于自动清理
    class ScopedRange {
    private:
        bool active;
        
    public:
        ScopedRange(const std::string& message) 
            : active(profilingEnabled) {
            if (active) {
                nvtxRangePushA(message.c_str());
            }
        }
        
        ~ScopedRange() {
            if (active) {
                nvtxRangePop();
            }
        }
    };
};
```

## 总结

NVTX 与 CUPTI 的集成为理解 CUDA 应用程序的结构和性能提供了强大的能力。通过战略性地放置注释：

### 关键优势

- **可见性**：清晰的应用程序结构视图
- **上下文**：将性能数据与代码段关联
- **分层分析**：理解嵌套操作的影响
- **自定义分析**：根据应用程序需求定制分析

### 最佳实践

1. **使用描述性名称**：清晰地标识代码段
2. **组织域**：逻辑地分组相关操作
3. **平衡粒度**：避免过度注释导致开销
4. **利用颜色编码**：可视化区分操作类型
5. **实现条件注释**：在生产中控制开销

NVTX 注释是现代 CUDA 性能分析工作流程的重要组成部分，为理解复杂 GPU 应用程序提供了必要的上下文。 