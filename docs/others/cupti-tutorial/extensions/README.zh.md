# CUPTI 扩展工具

> 完整的 GitHub 仓库和教程请访问 <https://github.com/eunomia-bpf/cupti-tutorial>。

此目录包含 CUPTI 示例的额外工具和扩展功能。

## 概述

这些扩展提供了简化 CUPTI API 使用的辅助函数和工具，特别适用于更复杂的任务，如性能指标收集、评估和结果处理。

## 目录结构

扩展分为两个主要区域：

- **include**：定义接口和工具的头文件
  - **c_util**：基本的 C 工具函数（文件操作、作用域管理等）
  - **profilerhost_util**：用于 CUPTI Profiler API 的工具

- **src**：工具的实现代码
  - **profilerhost_util**：分析器主机工具的源代码

## 核心组件

### 分析器主机工具

profilerhost_util 库提供以下功能：

1. **性能指标管理**：
   - 列出可用的性能指标
   - 获取指标描述和属性
   - 在不同指标格式间转换

2. **评估功能**：
   - 处理计数器数据
   - 从原始计数器数据计算指标值
   - 解释分析结果

3. **文件操作**：
   - 读取/写入指标数据
   - 管理分析配置

## 兼容性说明

这些扩展设计为与特定的 CUDA 和 CUPTI 版本配合工作。使用不同版本的 CUDA 时可能出现兼容性问题。

如果在构建 profilerhost_util 库时遇到构建错误：

1. 检查您的 CUDA 版本兼容性
2. 使用 install.sh 脚本创建的虚拟库来获得基本功能
3. 要获得完整功能，您可能需要更新代码以匹配您的 CUDA/CUPTI 版本

## 使用方法

需要这些扩展的示例包括：
- autorange_profiling
- userrange_profiling

这些示例演示了依赖于此目录中提供的辅助工具的更高级 CUPTI 功能。

## 构建

扩展由主目录中的 install.sh 脚本自动构建。但是，如果您需要手动构建：

```bash
cd src/profilerhost_util
make
cp libprofilerHostUtil.* ../../../lib64/
```

## 相关资源

- [CUPTI Profiler API 文档](https://docs.nvidia.com/cuda/cupti/modules.html#group__CUPTI__PROFILER__API)
- [NVPERF API 文档](https://docs.nvidia.com/cupti/Cupti/modules.html#group__NVPERF__API)

## 故障排除

### 常见构建问题

1. **找不到 CUDA 路径**：确保设置了 `CUDA_PATH` 环境变量
2. **库链接错误**：检查 `LD_LIBRARY_PATH` 是否包含 CUDA 库路径
3. **版本不匹配**：验证 CUDA Toolkit 版本与 CUPTI 版本兼容

### 运行时问题

1. **库加载失败**：确保所有依赖库都在系统路径中
2. **权限错误**：某些分析功能可能需要管理员权限
3. **设备兼容性**：验证目标 GPU 支持所需的分析功能

## 性能考虑

### 优化建议

1. **内存使用**：大型应用程序的分析可能消耗大量内存
2. **分析开销**：启用详细分析可能影响应用程序性能
3. **数据收集频率**：平衡数据粒度与性能影响

### 最佳实践

- 首先使用轻量级分析确定热点
- 对关键代码段使用详细分析
- 定期清理分析数据以避免内存泄漏
- 在生产环境中使用时要小心性能影响 