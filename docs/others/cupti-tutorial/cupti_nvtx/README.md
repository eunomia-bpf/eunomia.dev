# CUPTI NVTX Integration Tutorial

> The GitHub repo and complete tutorial is available at <https://github.com/eunomia-bpf/cupti-tutorial>.

## Introduction

NVTX (NVIDIA Tools Extension) provides a powerful way to annotate your CUDA applications with custom ranges, markers, and metadata. This sample demonstrates how to integrate NVTX with CUPTI to capture and analyze custom annotations, enabling better understanding of application structure and performance bottlenecks.

## What You'll Learn

- How to instrument applications with NVTX annotations
- Creating custom domains for organized profiling
- Using push/pop and start/end range patterns
- Registering strings and naming resources
- Integrating NVTX with CUPTI for comprehensive analysis
- Best practices for NVTX instrumentation

## Key Concepts

### NVTX Annotations
NVTX provides several annotation types:
- **Ranges**: Mark duration of code sections (push/pop or start/end)
- **Markers**: Point-in-time events
- **Domains**: Logical groupings for annotations
- **Categories**: Classification of events
- **Messages**: Descriptive text for annotations

### NVTX Domains
```cpp
// Create custom domain for vector operations
nvtxDomainHandle_t domain = nvtxDomainCreateA("Vector Addition");

// Use domain-specific annotations
nvtxDomainRangePushA(domain, "Memory Allocation");
// ... code ...
nvtxDomainRangePop(domain);
```

## Sample Architecture

### NVTX Setup and Integration
```cpp
// Required environment variable setup
// Linux: export NVTX_INJECTION64_PATH=<path>/libcupti.so
// Windows: set NVTX_INJECTION64_PATH=<path>/cupti.dll

#include "nvtx3/nvToolsExt.h"
#include "nvtx3/nvToolsExtCuda.h"
#include "nvtx3/nvToolsExtCudaRt.h"
#include "generated_nvtx_meta.h"

// CUPTI callback integration
void CUPTIAPI NvtxCallbackHandler(void* userdata, 
                                 CUpti_CallbackDomain domain,
                                 CUpti_CallbackId callbackId, 
                                 const CUpti_CallbackData* callbackInfo);
```

### Event Attributes Structure
```cpp
nvtxEventAttributes_t eventAttrib = {0};
eventAttrib.version = NVTX_VERSION;
eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
eventAttrib.colorType = NVTX_COLOR_ARGB;
eventAttrib.color = 0x0000ff;  // Blue color
eventAttrib.message.ascii = "Custom Operation";
```

## Sample Walkthrough

### Application Structure with NVTX
```cpp
void DoVectorAddition() {
    // Create custom domain
    nvtxDomainHandle_t domain = nvtxDomainCreateA("Vector Addition");
    
    // Name CUDA resources for better identification
    CUdevice device;
    cuDeviceGet(&device, 0);
    nvtxNameCuDeviceA(device, "CUDA Device 0");
    
    // Configure event attributes
    nvtxEventAttributes_t eventAttrib = {0};
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.colorType = NVTX_COLOR_ARGB;
    eventAttrib.color = 0x0000ff;
    
    // Mark main operation
    eventAttrib.message.ascii = "vectorAdd";
    nvtxDomainRangePushEx(domain, &eventAttrib);
    
    // Phase 1: Memory allocation on default domain
    nvtxRangePushA("Allocate host memory");
    // Allocate host memory
    pHostA = (int*)malloc(size);
    pHostB = (int*)malloc(size);
    pHostC = (int*)malloc(size);
    nvtxRangePop();
    
    // Phase 2: Device memory allocation on custom domain
    eventAttrib.message.ascii = "Allocate device memory";
    nvtxDomainRangePushEx(domain, &eventAttrib);
    cudaMalloc((void**)&pDeviceA, size);
    cudaMalloc((void**)&pDeviceB, size);
    cudaMalloc((void**)&pDeviceC, size);
    nvtxDomainRangePop(domain);
    
    // Phase 3: Memory transfer with registered string
    nvtxStringHandle_t string = nvtxDomainRegisterStringA(domain, "Memcpy operation");
    eventAttrib.message.registered = string;
    nvtxDomainRangePushEx(domain, &eventAttrib);
    cudaMemcpy(pDeviceA, pHostA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(pDeviceB, pHostB, size, cudaMemcpyHostToDevice);
    nvtxDomainRangePop(domain);
    
    // Phase 4: Kernel execution with start/end pattern
    eventAttrib.message.ascii = "Launch kernel";
    nvtxRangeId_t id = nvtxDomainRangeStartEx(domain, &eventAttrib);
    VectorAdd<<<blocksPerGrid, threadsPerBlock>>>(pDeviceA, pDeviceB, pDeviceC, N);
    cudaDeviceSynchronize();
    nvtxDomainRangeEnd(domain, id);
    
    // Phase 5: Result transfer
    eventAttrib.message.registered = string; // Reuse registered string
    nvtxDomainRangePushEx(domain, &eventAttrib);
    cudaMemcpy(pHostC, pDeviceC, size, cudaMemcpyDeviceToHost);
    nvtxDomainRangePop(domain);
    
    // Cleanup
    nvtxDomainRangePop(domain); // End main vectorAdd range
}
```

### Resource Naming
```cpp
void NameCudaResources() {
    // Name GPU device
    CUdevice device;
    cuDeviceGet(&device, 0);
    nvtxNameCuDeviceA(device, "Primary GPU");
    
    // Name CUDA context  
    CUcontext context;
    cuCtxCreate(&context, 0, device);
    nvtxNameCuContextA(context, "Vector Addition Context");
    
    // Name CUDA streams
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    nvtxNameCudaStreamA(stream1, "Computation Stream");
    nvtxNameCudaStreamA(stream2, "Memory Transfer Stream");
}
```

## Advanced NVTX Techniques

### Multi-Domain Organization
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
        // ... memory transfer code ...
        nvtxDomainRangePop(memDomain);
    }
    
    void AnnotateComputation() {
        auto computeDomain = GetDomain("Computation");
        
        nvtxDomainRangePushA(computeDomain, "Matrix Multiplication");
        // ... computation code ...
        nvtxDomainRangePop(computeDomain);
    }
};
```

### Hierarchical Annotations
```cpp
class HierarchicalAnnotator {
public:
    void AnnotateTrainingLoop() {
        nvtxDomainHandle_t mlDomain = nvtxDomainCreateA("ML Training");
        
        for (int epoch = 0; epoch < numEpochs; epoch++) {
            std::string epochName = "Epoch " + std::to_string(epoch);
            nvtxDomainRangePushA(mlDomain, epochName.c_str());
            
            // Data loading phase
            nvtxDomainRangePushA(mlDomain, "Data Loading");
            LoadBatchData();
            nvtxDomainRangePop(mlDomain);
            
            // Forward pass
            nvtxDomainRangePushA(mlDomain, "Forward Pass");
            RunForwardPass();
            nvtxDomainRangePop(mlDomain);
            
            // Backward pass
            nvtxDomainRangePushA(mlDomain, "Backward Pass");
            RunBackwardPass();
            nvtxDomainRangePop(mlDomain);
            
            // Parameter update
            nvtxDomainRangePushA(mlDomain, "Parameter Update");
            UpdateParameters();
            nvtxDomainRangePop(mlDomain);
            
            nvtxDomainRangePop(mlDomain); // End epoch
        }
    }
};
```

### Automatic Scope-Based Annotations
```cpp
class ScopedNvtxRange {
private:
    nvtxDomainHandle_t domain;
    bool useStartEnd;
    nvtxRangeId_t rangeId;
    
public:
    // Constructor for push/pop pattern
    ScopedNvtxRange(nvtxDomainHandle_t d, const char* message) 
        : domain(d), useStartEnd(false) {
        nvtxDomainRangePushA(domain, message);
    }
    
    // Constructor for start/end pattern
    ScopedNvtxRange(nvtxDomainHandle_t d, const nvtxEventAttributes_t* attrib)
        : domain(d), useStartEnd(true) {
        rangeId = nvtxDomainRangeStartEx(domain, attrib);
    }
    
    ~ScopedNvtxRange() {
        if (useStartEnd) {
            nvtxDomainRangeEnd(domain, rangeId);
        } else {
            nvtxDomainRangePop(domain);
        }
    }
};

// Usage with RAII
void ProcessBatch() {
    nvtxDomainHandle_t domain = GetDomain("Data Processing");
    ScopedNvtxRange range(domain, "Batch Processing");
    
    // All code in this scope is automatically annotated
    PreprocessData();
    RunInference();
    PostprocessResults();
    
    // Range automatically ends when scope exits
}
```

## Building and Running

### Prerequisites
```bash
# Set NVTX injection library path
export NVTX_INJECTION64_PATH=/usr/local/cuda/lib64/libcupti.so

# Or for custom CUPTI installation
export NVTX_INJECTION64_PATH=/path/to/your/libcupti.so
```

### Build and Execute
```bash
cd cupti_nvtx
make
./cupti_nvtx
```

## Sample Output

```
Device Name: NVIDIA GeForce RTX 3080

[CUPTI] NVTX Domain: Vector Addition
[CUPTI] NVTX Range: vectorAdd (Start)
[CUPTI] NVTX Range: Allocate host memory (Default Domain)
[CUPTI] NVTX Range: Allocate device memory (Vector Addition)
[CUPTI] NVTX Range: Memcpy operation (Vector Addition)
[CUPTI] NVTX Range: Launch kernel (Vector Addition)
[CUPTI] NVTX Range: Memcpy operation (Vector Addition)
[CUPTI] NVTX Range: vectorAdd (End)

Activity records processed: 15
NVTX ranges captured: 6
```

## Integration with Profiling Tools

### NVIDIA Nsight Integration
NVTX annotations automatically appear in:
- **Nsight Systems**: Timeline view with custom ranges
- **Nsight Compute**: Kernel-level annotations
- **Visual Profiler**: Legacy tool support

### Custom Analysis Tools
```cpp
class NvtxAnalyzer {
private:
    struct NvtxRange {
        std::string domain;
        std::string message;
        uint64_t startTime;
        uint64_t endTime;
        uint32_t threadId;
    };
    
    std::vector<NvtxRange> ranges;

public:
    void ProcessNvtxRecord(CUpti_ActivityNvtxRange* nvtxRange) {
        NvtxRange range;
        range.domain = GetDomainName(nvtxRange->domainId);
        range.message = GetRangeMessage(nvtxRange);
        range.startTime = nvtxRange->start;
        range.endTime = nvtxRange->end;
        range.threadId = nvtxRange->threadId;
        
        ranges.push_back(range);
    }
    
    void GenerateReport() {
        printf("\n=== NVTX Range Analysis ===\n");
        
        // Group by domain
        std::map<std::string, std::vector<NvtxRange*>> domainRanges;
        for (auto& range : ranges) {
            domainRanges[range.domain].push_back(&range);
        }
        
        for (const auto& [domain, rangeList] : domainRanges) {
            printf("Domain: %s\n", domain.c_str());
            
            double totalTime = 0.0;
            for (const auto* range : rangeList) {
                double duration = (range->endTime - range->startTime) / 1e6; // ms
                printf("  %s: %.3f ms\n", range->message.c_str(), duration);
                totalTime += duration;
            }
            
            printf("  Total: %.3f ms\n\n", totalTime);
        }
    }
};
```

## Best Practices

### Annotation Strategy
```cpp
class OptimalAnnotationStrategy {
public:
    void AnnotateApplication() {
        // 1. Use domains to organize annotations logically
        auto memoryDomain = nvtxDomainCreateA("Memory Management");
        auto computeDomain = nvtxDomainCreateA("Computation");
        auto ioDomain = nvtxDomainCreateA("Input/Output");
        
        // 2. Annotate at the right granularity
        AnnotateCoarseGrainedOperations();  // High-level phases
        AnnotateFineGrainedOperations();    // Detailed operations
        
        // 3. Use meaningful names and colors
        UseDescriptiveNames();
        UseConsistentColors();
        
        // 4. Avoid over-annotation
        AvoidTrivialAnnotations();
    }
    
private:
    void AnnotateCoarseGrainedOperations() {
        // Good: High-level application phases
        nvtxRangePushA("Model Training");
        nvtxRangePushA("Data Preprocessing");
        nvtxRangePushA("Inference Pass");
    }
    
    void AnnotateFineGrainedOperations() {
        // Moderate: Important inner operations
        nvtxRangePushA("Matrix Multiplication");
        nvtxRangePushA("Activation Function");
        
        // Avoid: Too fine-grained
        // nvtxRangePushA("Single Add Operation"); // Too detailed
    }
    
    void UseDescriptiveNames() {
        // Good
        nvtxRangePushA("ResNet50 Forward Pass - Layer 15");
        nvtxRangePushA("Adam Optimizer - Parameter Update");
        
        // Avoid
        // nvtxRangePushA("Function1"); // Not descriptive
        // nvtxRangePushA("Loop");      // Too generic
    }
    
    void UseConsistentColors() {
        nvtxEventAttributes_t memoryAttrib = CreateEventAttributes("Memory Operation", 0x00FF0000); // Red
        nvtxEventAttributes_t computeAttrib = CreateEventAttributes("Compute Operation", 0x0000FF00); // Green
        nvtxEventAttributes_t ioAttrib = CreateEventAttributes("I/O Operation", 0x000000FF);      // Blue
    }
};
```

### Performance Considerations
```cpp
class PerformanceOptimizedNvtx {
public:
    void OptimizeAnnotations() {
        // 1. Use registered strings for frequently used messages
        static nvtxStringHandle_t memcpyString = 
            nvtxDomainRegisterStringA(domain, "Memory Transfer");
        
        // Reuse registered string (more efficient)
        nvtxEventAttributes_t attrib = {};
        attrib.message.registered = memcpyString;
        nvtxDomainRangePushEx(domain, &attrib);
        
        // 2. Minimize annotation overhead in hot paths
        #ifdef ENABLE_DETAILED_PROFILING
            nvtxRangePushA("Hot Path Operation");
        #endif
        
        // 3. Use conditional compilation for production builds
        #ifndef NDEBUG
            nvtxRangePushA("Debug-only annotation");
        #endif
    }
    
    void BatchAnnotations() {
        // Avoid individual annotations in tight loops
        nvtxRangePushA("Process 1000 Elements");
        for (int i = 0; i < 1000; i++) {
            ProcessElement(i);
            // No annotation here - would be too expensive
        }
        nvtxRangePop();
    }
};
```

## Real-World Applications

### Scientific Computing
```cpp
class SimulationAnnotator {
public:
    void AnnotateSimulation() {
        nvtxDomainHandle_t simDomain = nvtxDomainCreateA("Physics Simulation");
        
        for (int timeStep = 0; timeStep < totalSteps; timeStep++) {
            std::string stepName = "Time Step " + std::to_string(timeStep);
            nvtxDomainRangePushA(simDomain, stepName.c_str());
            
            // Physics computation phases
            nvtxDomainRangePushA(simDomain, "Force Calculation");
            CalculateForces();
            nvtxDomainRangePop(simDomain);
            
            nvtxDomainRangePushA(simDomain, "Integration");
            IntegrateEquations();
            nvtxDomainRangePop(simDomain);
            
            nvtxDomainRangePushA(simDomain, "Boundary Conditions");
            ApplyBoundaryConditions();
            nvtxDomainRangePop(simDomain);
            
            nvtxDomainRangePop(simDomain); // End time step
        }
    }
};
```

### Deep Learning Framework
```cpp
class DeepLearningAnnotator {
public:
    void AnnotateTraining() {
        nvtxDomainHandle_t dlDomain = nvtxDomainCreateA("Deep Learning");
        
        // Register commonly used strings
        nvtxStringHandle_t forwardString = nvtxDomainRegisterStringA(dlDomain, "Forward Pass");
        nvtxStringHandle_t backwardString = nvtxDomainRegisterStringA(dlDomain, "Backward Pass");
        
        for (int batch = 0; batch < numBatches; batch++) {
            std::string batchName = "Batch " + std::to_string(batch);
            nvtxDomainRangePushA(dlDomain, batchName.c_str());
            
            // Forward pass
            nvtxEventAttributes_t attrib = {};
            attrib.message.registered = forwardString;
            nvtxDomainRangePushEx(dlDomain, &attrib);
            ForwardPass(batch);
            nvtxDomainRangePop(dlDomain);
            
            // Backward pass
            attrib.message.registered = backwardString;
            nvtxDomainRangePushEx(dlDomain, &attrib);
            BackwardPass(batch);
            nvtxDomainRangePop(dlDomain);
            
            nvtxDomainRangePop(dlDomain); // End batch
        }
    }
};
```

## Use Cases

- **Application Profiling**: Add semantic meaning to timeline profiling
- **Performance Debugging**: Identify bottlenecks in application logic
- **Code Organization**: Visualize application structure and flow
- **Team Collaboration**: Share profiling data with meaningful annotations
- **Optimization Validation**: Verify that optimizations affect the right code sections

## Next Steps

- Integrate NVTX annotations into your existing applications
- Develop automated annotation tools for common patterns
- Create domain-specific annotation libraries
- Build custom analysis tools that leverage NVTX data
- Establish team conventions for consistent annotation practices

NVTX provides a powerful way to bridge the gap between application semantics and low-level profiling data, enabling more effective performance analysis and optimization. 