# CUPTI 查询 API 教程

> GitHub 仓库和完整教程可在 <https://github.com/eunomia-bpf/cupti-tutorial> 获取。

## 简介

在您能够有效地对CUDA应用程序进行性能分析之前，您需要知道您的GPU上有哪些性能指标和事件可用。CUPTI查询API提供了一种发现和探索您的NVIDIA GPU性能分析能力的方法。本教程演示如何使用此API列出可用域、事件和指标。

## 您将学到什么

- 如何查询CUDA设备上的可用事件域
- 列出每个域中硬件计数器（事件）的技术
- 发现可用性能指标的方法
- 理解域、事件和指标之间的关系

## 理解CUPTI的性能分析层次结构

CUPTI将GPU性能分析能力组织在层次结构中：

1. **设备**：您的NVIDIA GPU
2. **域**：设备上相关硬件计数器的组
3. **事件**：域内的原始硬件计数器
4. **指标**：从事件计算出的派生测量值

这种层次结构允许有组织地访问现代GPU上可用的广泛性能数据。

## 代码演练

### 1. 查询可用设备

首先，我们需要识别可用的CUDA设备：

```cpp
int deviceCount = 0;
CUPTI_CALL(cuptiDeviceGetNumDevices(&deviceCount));
printf("有 %d 个设备\n", deviceCount);

// 获取设备的计算能力
CUdevice device = 0; // 默认为第一个设备
CUresult err = cuDeviceGet(&device, dev);
if (err != CUDA_SUCCESS) {
    printf("错误：cuDeviceGet失败，错误代码%d\n", err);
    return;
}

int major = 0, minor = 0;
err = cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
err = cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
printf("计算能力：%d.%d\n", major, minor);
```

### 2. 枚举事件域

事件域将相关的硬件计数器分组。我们可以列出设备上的所有可用域：

```cpp
void enumEventDomains(CUdevice device)
{
    // 获取域的数量
    uint32_t numDomains = 0;
    CUPTI_CALL(cuptiDeviceGetNumEventDomains(device, &numDomains));
    printf("设备 %d 有 %d 个域\n\n", device, numDomains);
    
    if (numDomains == 0) {
        printf("在设备 %d 上未找到域\n", device);
        return;
    }
    
    // 分配空间来保存域ID
    CUpti_EventDomainID *domainIds = (CUpti_EventDomainID *)malloc(numDomains * sizeof(CUpti_EventDomainID));
    if (domainIds == NULL) {
        printf("为域ID分配内存失败\n");
        return;
    }
    
    // 获取域ID
    CUPTI_CALL(cuptiDeviceEnumEventDomains(device, &numDomains, domainIds));
    
    // 对于每个域，打印相关信息
    for (int i = 0; i < numDomains; i++) {
        char name[CUPTI_MAX_NAME_LENGTH];
        size_t size = CUPTI_MAX_NAME_LENGTH;
        
        // 获取域名称
        CUPTI_CALL(cuptiEventDomainGetAttribute(domainIds[i], 
                                              CUPTI_EVENT_DOMAIN_ATTR_NAME,
                                              &size, name));
        
        // 获取已分析实例计数
        uint32_t profiled = 0;
        size = sizeof(profiled);
        CUPTI_CALL(cuptiDeviceGetEventDomainAttribute(device, domainIds[i],
                                                    CUPTI_EVENT_DOMAIN_ATTR_INSTANCE_COUNT,
                                                    &size, &profiled));
        
        // 获取总实例计数
        uint32_t total = 0;
        size = sizeof(total);
        CUPTI_CALL(cuptiDeviceGetEventDomainAttribute(device, domainIds[i],
                                                    CUPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT,
                                                    &size, &total));
        
        // 获取收集方法
        CUpti_EventCollectionMethod method;
        size = sizeof(method);
        CUPTI_CALL(cuptiEventDomainGetAttribute(domainIds[i],
                                              CUPTI_EVENT_DOMAIN_ATTR_COLLECTION_METHOD,
                                              &size, &method));
        
        printf("域# %d\n", i+1);
        printf("ID         = %d\n", domainIds[i]);
        printf("名称       = %s\n", name);
        printf("已分析实例计数 = %u\n", profiled);
        printf("总实例计数 = %u\n", total);
        printf("事件收集方法 = %s\n\n", 
               getCollectionMethodString(method));
    }
    
    free(domainIds);
}
```

收集方法指示此域中的事件如何收集：
- **PM**：性能监视器 - 硬件计数器
- **SM**：软件监视器 - 软件计数器
- **Instrumented**：基于仪器的收集
- **NVLINK_TC**：NVLink流量计数器

### 3. 列出域中的事件

一旦我们有了域ID，我们就可以列出该域中可用的所有事件：

```cpp
void enumEvents(CUdevice device, CUpti_EventDomainID domainId)
{
    // 获取域中的事件数量
    uint32_t numEvents = 0;
    CUPTI_CALL(cuptiEventDomainGetNumEvents(domainId, &numEvents));
    printf("域 %d 有 %d 个事件\n\n", domainId, numEvents);
    
    if (numEvents == 0) {
        printf("在域 %d 中未找到事件\n", domainId);
        return;
    }
    
    // 分配空间来保存事件ID
    CUpti_EventID *eventIds = (CUpti_EventID *)malloc(numEvents * sizeof(CUpti_EventID));
    if (eventIds == NULL) {
        printf("为事件ID分配内存失败\n");
        return;
    }
    
    // 获取事件ID
    CUPTI_CALL(cuptiEventDomainEnumEvents(domainId, &numEvents, eventIds));
    
    // 对于每个事件，打印相关信息
    for (int i = 0; i < numEvents; i++) {
        char name[CUPTI_MAX_NAME_LENGTH];
        size_t size = CUPTI_MAX_NAME_LENGTH;
        
        // 获取事件名称
        CUPTI_CALL(cuptiEventGetAttribute(eventIds[i], 
                                        CUPTI_EVENT_ATTR_NAME,
                                        &size, name));
        
        // 获取事件描述
        char desc[CUPTI_MAX_NAME_LENGTH];
        size = CUPTI_MAX_NAME_LENGTH;
        CUPTI_CALL(cuptiEventGetAttribute(eventIds[i],
                                        CUPTI_EVENT_ATTR_SHORT_DESCRIPTION,
                                        &size, desc));
        
        // 获取事件类别
        CUpti_EventCategory category;
        size = sizeof(category);
        CUPTI_CALL(cuptiEventGetAttribute(eventIds[i],
                                        CUPTI_EVENT_ATTR_CATEGORY,
                                        &size, &category));
        
        printf("事件# %d\n", i+1);
        printf("ID         = %d\n", eventIds[i]);
        printf("名称       = %s\n", name);
        printf("描述       = %s\n", desc);
        printf("类别       = %s\n\n", 
               getEventCategoryString(category));
    }
    
    free(eventIds);
}
```

事件被分类为不同类型：
- **Instruction**：与指令执行相关
- **Memory**：与内存操作相关
- **Cache**：与缓存操作相关
- **Profile Trigger**：用于性能分析触发器

### 4. 发现可用指标

指标是从一个或多个事件计算出的派生测量值： 