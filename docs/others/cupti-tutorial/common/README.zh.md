# CUPTI 通用辅助文件

> GitHub 仓库和完整教程可在 <https://github.com/eunomia-bpf/cupti-tutorial> 获取。

## 概述

此目录包含在多个CUPTI示例中使用的共享辅助文件和工具。这些文件为CUPTI初始化、错误处理、活动记录处理和其他常用操作提供通用功能。

## 文件

### helper_cupti.h

一个全面的头文件，提供：

- **错误处理宏**：简化CUPTI和CUDA API调用的错误检查
- **通用初始化函数**：CUPTI性能分析的标准设置例程
- **内存管理工具**：安全分配和释放辅助函数
- **设备管理函数**：GPU设备枚举和选择工具

主要宏和函数包括：

```cpp
// 错误检查宏
#define CUPTI_API_CALL(apiFuncCall)
#define RUNTIME_API_CALL(apiFuncCall)  
#define DRIVER_API_CALL(apiFuncCall)
#define MEMORY_ALLOCATION_CALL(var)

// 通用初始化
void initCuda();
void cleanupCuda();
CUdevice pickDevice();
```

### helper_cupti_activity.h

用于CUPTI活动记录处理的扩展头文件：

- **活动记录管理**：处理不同活动类型的结构和函数
- **缓冲区管理**：活动记录的高效缓冲区分配和处理
- **回调注册**：设置CUPTI回调的工具
- **数据提取**：从活动记录中提取有意义数据的辅助函数
- **输出格式化**：用于美化打印性能分析结果的函数

主要功能包括：

```cpp
// 活动记录处理
void processActivityRecord(CUpti_Activity* record);
void handleActivityBuffer(uint8_t* buffer, size_t validSize);

// 缓冲区管理
CUpti_BufferAlloc bufferRequested;
CUpti_BufferCompleted bufferCompleted;

// 回调工具
void enableActivityRecords();
void registerCallbacks();
```

## 在示例中的使用

这些辅助文件包含在大多数CUPTI示例中，用于：

1. **减少代码重复**：通用操作被集中化
2. **简化错误处理**：在示例中实现一致的错误检查
3. **标准化初始化**：统一的CUPTI设置程序
4. **简化活动处理**：可重用的活动记录处理

## 典型包含模式

```cpp
#include "helper_cupti.h"
#include "helper_cupti_activity.h"

int main() {
    // 初始化CUDA和CUPTI
    initCuda();
    
    // 设置性能分析
    enableActivityRecords();
    registerCallbacks();
    
    // 运行您的应用程序代码
    runKernels();
    
    // 处理结果并清理
    processResults();
    cleanupCuda();
    
    return 0;
}
```

## 与示例的集成

此仓库中的大多数示例通过以下方式使用这些辅助器：

1. 包含适当的头文件
2. 在main()中调用初始化函数
3. 在整个代码中使用错误检查宏
4. 利用活动处理工具
5. 遵循标准清理程序

## 自定义

虽然这些辅助器提供通用功能，但个别示例可能：

- 扩展提供的结构
- 添加特定于示例的处理函数
- 自定义回调处理程序
- 实现额外的错误处理

## 好处

使用这些通用辅助器提供：

- **一致性**：所有示例遵循相似模式
- **可靠性**：经过良好测试的错误处理和初始化
- **可维护性**：对通用功能的更改影响所有示例
- **学习性**：清晰的CUPTI最佳实践示例
- **效率**：优化的缓冲区管理和处理

## 贡献

在添加新示例或修改现有示例时：

1. 尽可能使用现有的辅助函数
2. 将新的通用功能添加到适当的辅助文件中
3. 保持一致的错误处理模式
4. 添加新辅助器时更新文档
5. 跨多个示例测试更改以确保兼容性

这些辅助文件对于在CUPTI示例套件中保持一致性和可靠性至关重要。 