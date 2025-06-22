# CUPTI 嵌套范围分析教程

> 完整的 GitHub 仓库和教程请访问 <https://github.com/eunomia-bpf/cupti-tutorial>。

## 简介

CUPTI 嵌套范围分析示例演示如何使用嵌套分析范围实现分层性能分析。这种技术允许您创建具有多个粒度级别的详细性能配置文件，支持复杂算法和嵌套函数调用的细粒度分析。

## 学习内容

- 如何创建和管理嵌套分析范围
- 实现分层性能测量
- 理解范围继承和作用域管理
- 分析嵌套算法性能模式
- 构建树结构的性能报告

## 理解嵌套范围分析

嵌套范围分析提供几个关键优势：

1. **分层分析**：将复杂算法分解为组成部分
2. **基于作用域的测量**：使用 RAII 原则的自动范围管理
3. **上下文性能数据**：了解算法阶段内的性能
4. **多级粒度**：同时在不同详细级别进行分析
5. **调用树分析**：将性能数据可视化为执行树

## 关键概念

### 范围层次结构

嵌套范围创建树结构：
```
应用程序
├── 初始化
│   ├── 内存分配
│   └── 数据设置
├── 计算
│   ├── 阶段 1
│   │   ├── 内核 A
│   │   └── 内核 B
│   └── 阶段 2
│       ├── 内核 C
│       └── 内存传输
└── 清理
```

### 范围属性

每个范围可以有：
- **名称**：描述性标识符
- **类别**：分组机制（计算、内存、IO）
- **颜色**：可视化提示
- **载荷**：附加到范围的自定义数据
- **指标**：特定于范围的性能计数器

### 继承和作用域

子范围从父范围继承属性：
- 指标收集配置
- 输出格式首选项
- 错误处理设置
- 自定义属性

## 构建示例

### 先决条件

- 带有 CUPTI 的 CUDA Toolkit
- C++14 兼容编译器（用于 RAII 范围管理）
- 用于增强可视化的 NVTX 库

### 构建过程

```bash
cd nested_range_profiling
make
```

这创建了演示分层分析技术的 `nested_range_profiling` 可执行文件。

## 代码架构

### 范围管理类

```cpp
class ProfileRange {
private:
    std::string name;
    std::string category;
    uint64_t startTime;
    std::vector<std::unique_ptr<ProfileRange>> children;
    std::map<std::string, double> metrics;

public:
    ProfileRange(const std::string& name, const std::string& category = "default");
    ~ProfileRange(); // 自动结束范围
    
    ProfileRange* createChild(const std::string& name, const std::string& category = "");
    void addMetric(const std::string& name, double value);
    void generateReport(int depth = 0) const;
};
```

### RAII 范围辅助类

```cpp
class ScopedRange {
private:
    ProfileRange* range;
    
public:
    ScopedRange(const std::string& name, const std::string& category = "default") {
        range = ProfileManager::getInstance().pushRange(name, category);
    }
    
    ~ScopedRange() {
        ProfileManager::getInstance().popRange();
    }
    
    ProfileRange* operator->() { return range; }
};

// 便利宏
#define PROFILE_RANGE(name, category) ScopedRange _prof_range(name, category)
```

## 运行示例

### 基本执行

```bash
./nested_range_profiling
```

### 示例输出

```
=== 嵌套范围分析报告 ===

应用程序 (45.2ms)
├── 初始化 (5.1ms)
│   ├── 内存分配 (2.3ms)
│   │   ├── 设备内存 (1.8ms) [内存: 512MB]
│   │   └── 主机内存 (0.5ms) [内存: 128MB]
│   └── 数据设置 (2.8ms)
│       ├── 数据生成 (1.2ms) [项目: 1M]
│       └── 数据传输 (1.6ms) [带宽: 8.5GB/s]
├── 计算 (35.4ms)
│   ├── 阶段 1 (18.7ms)
│   │   ├── 预处理内核 (3.2ms) [利用率: 89%]
│   │   ├── 主计算 (12.1ms) [FLOPS: 2.1T]
│   │   └── 中间传输 (3.4ms) [大小: 256MB]
│   └── 阶段 2 (16.7ms)
│       ├── 归约内核 (8.9ms) [效率: 95%]
│       ├── 后处理 (4.2ms) [缓存命中: 98%]
│       └── 结果传输 (3.6ms) [带宽: 7.2GB/s]
└── 清理 (4.7ms)
    ├── 内存释放 (2.1ms)
    └── 上下文清理 (2.6ms)

性能摘要：
- 总执行时间：45.2ms
- 计算时间：28.4ms (62.8%)
- 内存时间：12.1ms (26.8%)
- 开销：4.7ms (10.4%)
```

## 高级功能

### 指标收集集成

```cpp
class MetricCollector {
private:
    std::map<std::string, CUpti_EventGroup> eventGroups;
    std::vector<CUpti_EventID> activeEvents;

public:
    void beginCollection(const std::string& rangeName) {
        auto& group = eventGroups[rangeName];
        CUPTI_CALL(cuptiEventGroupEnable(group));
    }
    
    void endCollection(const std::string& rangeName, ProfileRange* range) {
        auto& group = eventGroups[rangeName];
        
        uint64_t eventValues[activeEvents.size()];
        size_t valueSize = sizeof(eventValues);
        
        CUPTI_CALL(cuptiEventGroupReadAllEvents(group,
                   CUPTI_EVENT_READ_FLAG_NONE,
                   &valueSize, eventValues,
                   nullptr, nullptr));
        
        // 向范围添加指标
        for (size_t i = 0; i < activeEvents.size(); i++) {
            const char* eventName;
            CUPTI_CALL(cuptiEventGetAttribute(activeEvents[i],
                       CUPTI_EVENT_ATTR_NAME, &valueSize, &eventName));
            range->addMetric(eventName, static_cast<double>(eventValues[i]));
        }
        
        CUPTI_CALL(cuptiEventGroupDisable(group));
    }
};
```

### 条件分析

```cpp
class ConditionalProfiler {
private:
    std::map<std::string, bool> enabledCategories;
    int maxDepth;
    
public:
    bool shouldProfile(const std::string& category, int currentDepth) {
        if (currentDepth > maxDepth) return false;
        
        auto it = enabledCategories.find(category);
        return (it != enabledCategories.end()) ? it->second : true;
    }
    
    void setCategory(const std::string& category, bool enabled) {
        enabledCategories[category] = enabled;
    }
    
    void setMaxDepth(int depth) { maxDepth = depth; }
};

// 使用示例
void profiledFunction() {
    if (ConditionalProfiler::getInstance().shouldProfile("detailed", getCurrentDepth())) {
        PROFILE_RANGE("详细分析", "detailed");
        // ... 详细分析代码 ...
    } else {
        PROFILE_RANGE("高级概述", "summary");
        // ... 摘要分析代码 ...
    }
}
```

### 自定义范围属性

```cpp
class AttributedRange : public ProfileRange {
private:
    std::map<std::string, std::string> attributes;
    
public:
    void setAttribute(const std::string& key, const std::string& value) {
        attributes[key] = value;
    }
    
    void generateDetailedReport() {
        printf("范围：%s\n", name.c_str());
        for (const auto& attr : attributes) {
            printf("  %s: %s\n", attr.first.c_str(), attr.second.c_str());
        }
    }
};
```

## 实际应用

### 算法优化

```cpp
void matrixMultiplication() {
    PROFILE_RANGE("矩阵乘法", "compute");
    
    {
        PROFILE_RANGE("数据预处理", "memory");
        // 数据加载和预处理
    }
    
    {
        PROFILE_RANGE("计算阶段", "compute");
        for (int block = 0; block < numBlocks; block++) {
            PROFILE_RANGE("块计算", "kernel");
            // 执行矩阵块乘法
        }
    }
    
    {
        PROFILE_RANGE("结果收集", "memory");
        // 收集和后处理结果
    }
}
```

### 内存使用分析

```cpp
class MemoryProfiler {
public:
    void trackAllocation(const std::string& rangeName, size_t size) {
        auto range = getCurrentRange();
        range->setAttribute("allocated_memory", std::to_string(size));
        range->addMetric("memory_usage", static_cast<double>(size));
    }
    
    void analyzeMemoryPattern() {
        // 分析内存分配模式
        for (const auto& range : allRanges) {
            if (range->hasAttribute("allocated_memory")) {
                printf("范围 %s 分配了 %s 字节\n", 
                       range->getName().c_str(),
                       range->getAttribute("allocated_memory").c_str());
            }
        }
    }
};
```

## 最佳实践

### 范围设计

1. **逻辑分组**：根据功能逻辑创建范围
2. **平衡粒度**：避免过度细化或过于粗糙
3. **一致命名**：使用清晰、一致的命名约定
4. **范围最小化**：只在必要时创建嵌套

### 性能考虑

1. **最小开销**：保持分析开销低于 5%
2. **选择性启用**：在生产中禁用详细分析
3. **缓冲输出**：批量写入分析数据

### 报告生成

```cpp
class ReportGenerator {
public:
    void generateHTMLReport(const ProfileRange& root) {
        std::ofstream html("profile_report.html");
        html << "<html><head><title>性能报告</title></head><body>";
        generateHTMLNode(html, root, 0);
        html << "</body></html>";
    }
    
    void generateJSONReport(const ProfileRange& root) {
        Json::Value jsonRoot;
        convertToJSON(root, jsonRoot);
        
        std::ofstream json("profile_report.json");
        json << jsonRoot;
    }
};
```

这个嵌套范围分析教程为构建详细、分层的性能分析系统提供了强大的框架。 