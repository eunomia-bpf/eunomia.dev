# CUPTI PC 采样教程

> GitHub 仓库和完整教程可在 <https://github.com/eunomia-bpf/cupti-tutorial> 获取。

## 简介

在优化CUDA内核时，了解代码的哪些部分消耗最多执行时间至关重要。程序计数器（PC）采样是一种强大的技术，允许您以最小的性能开销识别GPU代码中的热点。本教程演示如何使用CUPTI的PC采样API在内核执行期间收集程序计数器样本并将其映射回源代码，帮助您将优化工作集中在最有影响的地方。

## 您将学到什么

- 如何为CUDA内核配置和启用PC采样
- 在内核执行期间收集和处理PC样本
- 将PC地址映射到源代码位置
- 识别CUDA代码中的性能热点
- 分析采样数据以指导优化工作

## 理解PC采样

程序计数器（PC）采样通过定期记录每个活跃线程当前执行的指令来工作。此过程如下进行：

1. GPU硬件定期采样活跃warp的程序计数器
2. 这些样本在内核执行期间被收集和缓冲
3. 内核完成后，分析样本以确定哪些代码区域被执行得最频繁
4. 使用调试信息将PC地址映射回源代码

与基于仪器的性能分析不同，PC采样对内核性能的影响最小，并提供GPU执行时间消耗的统计视图。

## 代码演练

### 1. 设置PC采样

首先，我们需要配置和启用PC采样：

```cpp
CUpti_PCSamplingConfig config;
memset(&config, 0, sizeof(config));
config.size = sizeof(config);
config.samplingPeriod = 5;  // 每5个周期采样一次
config.samplingPeriod2 = 0; // 此示例中未使用
config.samplingPeriodRatio = 0; // 此示例中未使用
config.collectingMode = CUPTI_PC_SAMPLING_COLLECTION_MODE_KERNEL;
config.samplingBufferSize = 0x2000; // 8KB缓冲区
config.stoppingCount = 0; // 无停止条件

// 获取当前CUDA上下文
CUcontext context;
DRIVER_API_CALL(cuCtxGetCurrent(&context));

// 启用PC采样
CUPTI_CALL(cuptiPCSamplingEnable(&config));
```

此代码：
1. 创建PC采样的配置结构
2. 设置采样周期（多久采样一次PC）
3. 配置收集模式以在内核执行期间采样
4. 设置存储样本的缓冲区大小
5. 在当前CUDA上下文上启用PC采样

### 2. 注册缓冲区回调

要处理PC采样数据，我们需要注册在缓冲区满时将被调用的回调：

```cpp
// 注册缓冲区处理的回调
CUPTI_CALL(cuptiPCSamplingRegisterBufferHandler(handleBuffer, userData));

// 处理PC采样缓冲区的回调函数
void handleBuffer(uint8_t *buffer, size_t size, size_t validSize, void *userData)
{
    PCData *pcData = (PCData *)userData;
    
    // 处理缓冲区中的所有记录
    CUpti_PCSamplingData *record = (CUpti_PCSamplingData *)buffer;
    
    while (validSize >= sizeof(CUpti_PCSamplingData)) {
        // 处理PC样本
        processPCSample(record, pcData);
        
        // 移动到下一个记录
        validSize -= sizeof(CUpti_PCSamplingData);
        record++;
    }
}
```

此代码：
1. 注册一个在PC采样缓冲区准备就绪时将被调用的回调函数
2. 在回调中，处理每个PC采样记录
3. 根据收集的样本更新统计信息

### 3. 处理PC样本

对于每个PC样本，我们需要处理和存储信息：

```cpp
void processPCSample(CUpti_PCSamplingData *sample, PCData *pcData)
{
    // 从样本中提取信息
    uint64_t pc = sample->pc;
    uint32_t functionId = sample->functionId;
    uint32_t stall = sample->stallReason;
    
    // 更新PC直方图
    pcData->totalSamples++;
    
    // 为此PC查找或创建条目
    PCEntry *entry = findPCEntry(pcData, pc);
    if (!entry) {
        entry = createPCEntry(pcData, pc);
    }
    
    // 更新此PC的样本计数
    entry->sampleCount++;
    
    // 如果需要，更新阻塞原因计数
    if (stall != CUPTI_PC_SAMPLING_STALL_NONE) {
        entry->stallCount[stall]++;
    }
}
```

此函数：
1. 从样本中提取程序计数器（PC）值
2. 更新总样本计数
3. 在我们的数据结构中为此PC查找或创建条目
4. 更新此PC的统计信息，包括阻塞原因

### 4. 将PC映射到源代码

要使PC值有意义，我们需要将它们映射回源代码：

```cpp
void mapPCsToSource(PCData *pcData)
{
    // 获取内核的CUDA模块
    CUmodule module;
    DRIVER_API_CALL(cuModuleGetFunction(&module, pcData->function));
    
    // 对于每个PC条目，获取源信息
    for (int i = 0; i < pcData->numEntries; i++) {
        PCEntry *entry = &pcData->entries[i];
        
        // 获取源文件和行信息
        CUpti_LineInfo lineInfo;
        lineInfo.size = sizeof(lineInfo);
        
        CUPTI_CALL(cuptiGetLineInfo(module, entry->pc, &lineInfo));
        
        // 存储源信息
        if (lineInfo.lineInfoValid) {
            entry->fileName = strdup(lineInfo.fileName);
            entry->lineNumber = lineInfo.lineNumber;
            entry->functionName = strdup(lineInfo.functionName);
        } else {
            entry->fileName = strdup("unknown");
            entry->lineNumber = 0;
            entry->functionName = strdup("unknown");
        }
    }
}
```

此函数：
1. 获取我们内核的CUDA模块
2. 对于每个PC条目，调用CUPTI获取源行信息
3. 为每个PC存储文件名、行号和函数名

### 5. 带热点的示例内核

为了演示PC采样，我们将使用一个带有一些故意热点的内核：

```cpp
__global__ void sampleKernel(float *data, int size, int iterations)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float value = data[idx];
        
        // 热点1：计算密集型循环
        for (int i = 0; i < iterations; i++) {
            value = value * value + 0.5f;
        }
        
        // 热点2：带发散的条件分支
        if (idx % 2 == 0) {
            for (int i = 0; i < iterations / 2; i++) {
                value = sinf(value);
            }
        } else {
            for (int i = 0; i < iterations / 4; i++) {
                value = cosf(value);
            }
        }
        
        // 热点3：内存访问模式
        int offset = (idx * 17) % size;
        value += data[offset];
        
        data[idx] = value;
    }
}
```

此内核具有几个将在PC采样中显示的特征：
1. 一个将成为主要热点的计算密集型循环
2. 导致线程发散的条件分支
3. 非合并内存访问模式

### 6. 分析和显示结果

收集样本后，我们分析和显示结果：

```cpp
void analyzeResults(PCData *pcData)
{
    // 按样本计数排序PC条目
    qsort(pcData->entries, pcData->numEntries, sizeof(PCEntry), comparePCEntries);
    
    // 打印摘要
    printf("\nPC采样结果：\n");
    printf("  收集的总样本数：%llu\n", pcData->totalSamples);
    printf("  唯一PC地址：%d\n\n", pcData->numEntries);
    
    // 打印前5个热点
    printf("前5个热点：\n");
    int numToShow = (pcData->numEntries < 5) ? pcData->numEntries : 5;
    
    for (int i = 0; i < numToShow; i++) {
        PCEntry *entry = &pcData->entries[i];
        float percentage = 100.0f * entry->sampleCount / pcData->totalSamples;
        
        printf("  %d. PC: 0x%llx（%.1f%%的样本）- %s:%d - %s\n",
               i + 1, entry->pc, percentage,
               entry->fileName, entry->lineNumber, entry->functionName);
        
        // 如果可用，打印阻塞原因
        if (entry->stallCount[CUPTI_PC_SAMPLING_STALL_MEMORY_THROTTLE] > 0) {
            printf("     内存限制阻塞：%d\n", 
                  entry->stallCount[CUPTI_PC_SAMPLING_STALL_MEMORY_THROTTLE]);
        }
        if (entry->stallCount[CUPTI_PC_SAMPLING_STALL_SYNC] > 0) {
            printf("     同步阻塞：%d\n", 
                  entry->stallCount[CUPTI_PC_SAMPLING_STALL_SYNC]);
        }
        // ... 打印其他阻塞原因 ...
    }
}
``` 