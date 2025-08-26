# CUPTI 程序计数器 (PC) 连续采样教程

> GitHub 仓库和完整教程可在 <https://github.com/eunomia-bpf/cupti-tutorial> 获取。

## 简介

程序计数器 (PC) 采样是一种强大的分析技术，允许您在汇编指令级别了解 CUDA 内核的执行时间分布。本教程演示如何实现连续 PC 采样，可以监控任何 CUDA 应用程序，无需修改源代码。

## 您将学到什么

- 如何构建用于 PC 采样注入的动态库
- 连续分析 CUDA 应用程序的技术
- 理解 PC 采样数据和停顿原因
- 跨平台实现（Linux 和 Windows）
- 将分析库与现有应用程序配合使用

## 理解 PC 连续采样

PC 连续采样与其他分析方法的不同之处在于：

1. **在汇编级别操作**：提供对实际 GPU 指令执行的洞察
2. **无需源代码修改**：可以分析任何 CUDA 应用程序
3. **通过库注入工作**：使用动态加载拦截 CUDA 调用
4. **提供停顿原因分析**：显示线程束为什么没有取得进展
5. **支持实时监控**：可以在执行期间观察性能

## 架构概览

连续 PC 采样系统包含：

1. **动态库**：`libpc_sampling_continuous.so`（Linux）或 `pc_sampling_continuous.lib`（Windows）
2. **注入机制**：使用 `LD_PRELOAD`（Linux）或 DLL 注入（Windows）
3. **CUPTI 集成**：利用 CUPTI 的 PC 采样 API
4. **辅助脚本**：`libpc_sampling_continuous.pl` 便于执行

## 详细组件分析

### 1. 辅助脚本 (`libpc_sampling_continuous.pl`)

Perl 脚本作为包装器，简化了 PC 采样过程。以下是其工作原理：

#### 脚本工作流程

1. **命令行解析（第 29-41 行）**
   ```perl
   GetOptions( 'help'                => \$help
             , 'app=s'               => \$applicationName
             , 'collection-mode=i'   => \$collectionMode
             , 'sampling-period=i'   => \$samplingPeriod
             # ... 更多选项
   ```
   - 使用 Perl 的 `Getopt::Long` 模块解析命令行参数
   - 支持各种采样配置参数

2. **参数验证（第 44-104 行）**
   - **采集模式**：验证值（1 为连续模式，2 为内核序列化模式）
   - **采样周期**：确保值在 5-31 之间（代表 2^n 个周期）
   - **缓冲区大小**：验证临时缓冲区和硬件缓冲区大小
   - 构建命令行选项字符串传递给注入库

3. **库路径验证（`init` 函数，第 150-233 行）**
   ```perl
   sub init {
       my $ldLibraryPath = $ENV{'LD_LIBRARY_PATH'};
       my @libPaths = split /:/, $ldLibraryPath;
   ```
   - 在系统路径中检查所需库：
     - `libpc_sampling_continuous.so`：主分析库
     - `libcupti.so`：用于 GPU 分析的 CUPTI 库
     - `libpcsamplingutil.so`：PC 采样的实用工具库
   - 设置 `CUDA_INJECTION64_PATH` 环境变量用于 CUDA 注入

4. **应用程序执行（`RunApplication` 函数，第 235-244 行）**
   ```perl
   sub RunApplication {
       $ENV{INJECTION_PARAM} = $injectionParameters;
       my $returnCode = system($applicationName);
   }
   ```
   - 将注入参数设置为环境变量
   - 启动加载了注入库的目标应用程序

#### 关键配置参数

| 参数 | 描述 | 默认值 | 有效范围 |
|-----|------|--------|----------|
| `--collection-mode` | 采样模式（1=连续，2=序列化） | 1 | 1-2 |
| `--sampling-period` | 设置采样为 2^n 个周期 | - | 5-31 |
| `--scratch-buf-size` | 临时 PC 记录缓冲区 | 1 MB | 任意大小 |
| `--hw-buf-size` | 硬件缓冲区大小 | 512 MB | 任意大小 |
| `--pc-config-buf-record-count` | 配置的 PC 记录数 | 5000 | 任意数量 |
| `--pc-circular-buf-record-count` | 每个循环缓冲区的记录数 | 500 | 任意数量 |
| `--circular-buf-count` | 循环缓冲区数量 | 10 | 任意数量 |
| `--file-name` | 输出文件名 | pcsampling.dat | 任意名称 |

### 2. 核心实现 (`pc_sampling_continuous.cpp`)

C++ 实现通过 CUPTI 回调和数据收集处理实际的 PC 采样。

#### 初始化流程

1. **入口点（`InitializeInjection`，第 987-1025 行）**
   ```cpp
   extern "C" int InitializeInjection(void) {
       // 读取环境参数
       ReadInputParams();
       
       // 订阅 CUPTI 回调
       CUPTI_API_CALL(cuptiSubscribe(&subscriber, 
                      (CUpti_CallbackFunc)&CallbackHandler, NULL));
       
       // 为所有内核启动变体启用回调
       CUPTI_API_CALL(cuptiEnableCallback(1, subscriber, 
                      CUPTI_CB_DOMAIN_DRIVER_API, 
                      CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel));
       // ... 更多启动回调
       
       // 启用资源回调
       CUPTI_API_CALL(cuptiEnableCallback(1, subscriber, 
                      CUPTI_CB_DOMAIN_RESOURCE, 
                      CUPTI_CBID_RESOURCE_CONTEXT_CREATED));
   }
   ```
   - 当 CUDA 加载注入库时自动调用
   - 订阅所有内核启动回调和资源事件
   - 设置退出处理程序进行清理

#### 数据结构

2. **上下文信息管理（第 95-106 行）**
   ```cpp
   typedef struct ContextInfo_st {
       uint32_t contextUid;
       CUpti_PCSamplingData pcSamplingData;
       std::vector<CUpti_PCSamplingConfigurationInfo> pcSamplingConfigurationInfo;
       PcSamplingStallReasons pcSamplingStallReasons;
       bool ctxDestroyed;
       uint8_t *pPcSamplingBuffer;
   } ContextInfo;
   ```
   - 存储每个上下文的 PC 采样配置
   - 维护采样数据和停顿原因信息
   - 跟踪上下文生命周期

#### 回调系统

3. **主回调处理程序（`CallbackHandler`，第 758-983 行）**
   ```cpp
   void CallbackHandler(void *pUserdata, 
                       CUpti_CallbackDomain domain,
                       CUpti_CallbackId callbackId, 
                       void *pCallbackData) {
       switch (domain) {
           case CUPTI_CB_DOMAIN_DRIVER_API:
               // 处理内核启动
               HandleDriverApiCallback(callbackId, pCallbackData);
               break;
           case CUPTI_CB_DOMAIN_RESOURCE:
               // 处理上下文和模块事件
               HandleResourceCallback(callbackId, pCallbackData);
               break;
       }
   }
   ```

4. **上下文创建处理**
   - 为新上下文启用 PC 采样
   - 配置采样参数（停顿原因、缓冲区大小）
   - 创建工作线程进行数据处理（仅第一个上下文）
   - 分配循环缓冲区用于数据收集

5. **内核启动处理**
   - **序列化模式**：每个内核后刷新所有 PC 记录
   - **连续模式**：缓冲区达到阈值时刷新
   - 使用 `cuptiPCSamplingGetData()` 检索样本
   - 将数据推送到队列以进行文件写入

6. **模块加载事件**
   - 处理动态模块加载/卸载
   - 模块更改时刷新任何待处理的 PC 记录
   - 确保跨模块边界的数据一致性

#### 数据收集工作流

7. **PC 采样数据检索**
   ```cpp
   bool GetPcSamplingDataFromCupti(
       CUpti_PCSamplingGetDataParams &params,
       ContextInfo *pContextInfo) {
       // 分配循环缓冲区
       params.pPcData = g_circularBuffer[g_bufferIndexForCupti];
       params.pcDataBufferSize = g_circularbufSize;
       
       // 从 CUPTI 获取数据
       CUptiResult result = cuptiPCSamplingGetData(&params);
       
       // 处理缓冲区并排队数据进行写入
       if (pContextInfo->pcSamplingData.totalNumPcs > 0) {
           g_pcSampDataQueue.push(
               std::make_pair(&pContextInfo->pcSamplingData, 
                            pContextInfo));
       }
   }
   ```

8. **工作线程（`StoreDataInFile`，第 541-583 行）**
   ```cpp
   void StoreDataInFile() {
       while (g_running || !g_pcSampDataQueue.empty()) {
           if (!g_pcSampDataQueue.empty()) {
               // 从队列获取数据
               auto pcSampData = g_pcSampDataQueue.front();
               g_pcSampDataQueue.pop();
               
               // 使用 CUPTI 实用工具写入文件
               CuptiUtilPutPcSampData(fileName, 
                                    &pContextInfo->pcSamplingStallReasons,
                                    &pcSamplingConfigurationInfo,
                                    &pcSamplingData);
           }
       }
   }
   ```
   - 在后台持续运行
   - 处理排队的 PC 采样数据
   - 使用 CUPTI 实用工具将数据写入二进制文件
   - 创建每个上下文的输出文件

#### 清理和退出

9. **退出处理程序（`AtExitHandler`，第 595-673 行）**
   ```cpp
   void AtExitHandler() {
       // 为所有活动上下文禁用 PC 采样
       for (auto& itr: g_contextInfoMap) {
           // 刷新剩余数据
           while (itr.second->pcSamplingData.remainingNumPcs > 0) {
               GetPcSamplingData(pcSamplingGetDataParams, itr.second);
           }
           
           // 禁用采样
           cuptiPCSamplingDisable(&pcSamplingDisableParams);
           
           // 排队最终缓冲区进行写入
           g_pcSampDataQueue.push(
               std::make_pair(&itr.second->pcSamplingData, 
                            itr.second));
       }
       
       // 加入工作线程并清理
       g_thread.join();
       FreeAllocatedMemory();
   }
   ```

### 3. 数据流架构

```
应用程序启动
        |
        v
[libpc_sampling_continuous.pl]
        |
        | 设置 CUDA_INJECTION64_PATH
        | 设置 INJECTION_PARAM
        v
[CUDA 运行时加载库]
        |
        v
[InitializeInjection()]
        |
        | 订阅回调
        v
[上下文创建] -----> [配置 PC 采样]
        |                    |
        v                    v
[内核启动] -----> [收集 PC 样本]
        |                    |
        v                    v
[模块事件] -----> [刷新到循环缓冲区]
        |                    |
        v                    v
[工作线程] <------ [排队 PC 数据]
        |
        v
[写入文件]
        |
        v
[N_pcsampling.dat]
```

## 构建示例

### Linux 构建过程

1. 导航到 pc_sampling_continuous 目录：
   ```bash
   cd pc_sampling_continuous
   ```

2. 使用提供的 Makefile 构建：
   ```bash
   make
   ```
   
   这会在当前目录创建 `libpc_sampling_continuous.so`。

### Windows 构建过程

对于 Windows，您需要先构建 Microsoft Detours 库：

1. **下载 Detours 源代码**：
   - 从 GitHub：https://github.com/microsoft/Detours
   - 或 Microsoft：https://www.microsoft.com/en-us/download/details.aspx?id=52586

2. **构建 Detours**：
   ```cmd
   # 解压并导航到 Detours 文件夹
   set DETOURS_TARGET_PROCESSOR=X64
   "Program Files (x86)\Microsoft Visual Studio\<version>\Professional\VC\Auxiliary\Build\vcvarsall.bat" x64
   NMAKE
   ```

3. **复制所需文件**：
   ```cmd
   copy detours.h <pc_sampling_continuous_folder>
   copy detours.lib <pc_sampling_continuous_folder>
   ```

4. **构建示例**：
   ```cmd
   nmake
   ```
   
   这会创建 `pc_sampling_continuous.lib`。

## 运行示例

### Linux 执行

1. **设置库路径**：
   ```bash
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/CUPTI/lib64:/path/to/pc_sampling_continuous:/path/to/pcsamplingutil
   ```

2. **使用辅助脚本**：
   ```bash
   ./libpc_sampling_continuous.pl --help
   ```
   
   这会显示所有可用选项。

3. **运行您的应用程序**：
   ```bash
   ./libpc_sampling_continuous.pl --app /path/to/your/cuda/application
   ```

### Windows 执行

1. **设置库路径**：
   ```cmd
   set PATH=%PATH%;C:\path\to\CUPTI\bin;C:\path\to\pc_sampling_continuous;C:\path\to\pcsamplingutil
   ```

2. **运行您的应用程序**：
   ```cmd
   pc_sampling_continuous.exe your_cuda_application.exe
   ```

## 理解输出

### 二进制数据格式

输出文件（`N_pcsampling.dat`）包含 CUPTI PC 采样格式的二进制数据：

1. **头部部分**
   - 魔术字符串："CUPS"（CUPTI 统一分析样本）
   - 版本信息
   - 缓冲区数量
   - 配置参数

2. **配置数据**
   - 停顿原因名称和索引
   - 采样周期和收集模式
   - 缓冲区大小和计数

3. **PC 样本记录**
   - 程序计数器地址
   - 停顿原因代码
   - 样本计数
   - 时间戳信息

### 解析数据

使用 `pc_sampling_utility` 工具解析和分析二进制数据：

```bash
# 基本解析
./pc_sampling_utility --file-name pcsampling.dat

# 带源代码关联
./pc_sampling_utility --file-name pcsampling.dat --enable-source-correlation

# 详细输出
./pc_sampling_utility --file-name pcsampling.dat --verbose
```

### 解析输出示例

```
函数：vectorAdd
模块：module_1
总样本数：10000

PC 地址      | 停顿原因              | 计数  | 百分比
0x7f8b2c1000 | MEMORY_DEPENDENCY    | 3500  | 35.0%
0x7f8b2c1008 | EXECUTION_DEPENDENCY | 2000  | 20.0%
0x7f8b2c1010 | NOT_SELECTED        | 1500  | 15.0%
0x7f8b2c1018 | SYNCHRONIZATION     | 1000  | 10.0%
...
```

## 高级配置

### 性能调优

1. **采样周期优化**
   - 较低值（5-10）：高细节，更多开销
   - 中等值（11-20）：平衡方法
   - 较高值（21-31）：低开销，较少细节

2. **缓冲区大小考虑**
   ```
   临时缓冲区大小 = (预期 PC 数) × (16 字节 + 16 字节 × 停顿原因数)
   
   示例：1000 个 PC，4 个停顿原因
   = 1000 × (16 + 16 × 4)
   = 1000 × 80 字节
   = 最少 80 KB
   ```

3. **收集模式选择**
   - **连续模式**：最适合长时间运行的内核
   - **序列化模式**：更适合短暂、频繁的内核启动

### 内存管理

系统使用多缓冲区方法：

1. **配置缓冲区**：保存 PC 采样设置（默认：5000 条记录）
2. **循环缓冲区**：收集数据的临时存储（默认：10 个缓冲区 × 500 条记录）
3. **硬件缓冲区**：GPU 端存储（默认：512 MB）
4. **临时缓冲区**：数据处理的工作内存（默认：1 MB）

### 停顿原因分析

常见停顿原因及其含义：

| 停顿原因 | 描述 | 优化策略 |
|---------|------|----------|
| MEMORY_DEPENDENCY | 等待内存操作 | 改善内存合并，使用共享内存 |
| EXECUTION_DEPENDENCY | 等待先前指令 | 重新排序指令，增加 ILP |
| NOT_SELECTED | 线程束未被调度 | 平衡工作负载，减少分歧 |
| SYNCHRONIZATION | 在同步点等待 | 最小化 __syncthreads()，优化屏障 |
| TEXTURE | 等待纹理获取 | 优化纹理缓存使用 |
| CONSTANT_MEMORY | 等待常量内存 | 对频繁访问的常量使用共享内存 |

## 故障排除

### 常见问题和解决方案

1. **库加载失败**
   ```
   错误：库 libpc_sampling_continuous.so 不存在
   ```
   解决方案：确保 LD_LIBRARY_PATH 包含库目录

2. **CUPTI 初始化错误**
   ```
   CUPTI_ERROR_NOT_INITIALIZED
   ```
   解决方案：验证 CUDA 和 CUPTI 版本匹配

3. **缓冲区溢出**
   ```
   警告：cuptiPCSamplingDisable() 期间丢弃了 N 条记录
   ```
   解决方案：使用命令行参数增加缓冲区大小

4. **权限被拒绝**
   ```
   无法打开文件进行写入
   ```
   解决方案：确保输出目录中的写入权限

### 调试技巧

1. **启用详细日志记录**
   ```bash
   ./libpc_sampling_continuous.pl --app ./myapp --verbose
   ```

2. **检查库依赖项**
   ```bash
   ldd libpc_sampling_continuous.so
   ```

3. **监控系统资源**
   ```bash
   # 检查可用内存
   free -h
   
   # 执行期间监控
   watch -n 1 'free -h'
   ```

4. **首先使用简单内核测试**
   ```cuda
   __global__ void simpleKernel() {
       int idx = blockIdx.x * blockDim.x + threadIdx.x;
       // 用于测试的简单操作
   }
   ```

## 性能影响

PC 采样开销取决于：

1. **采样频率**：频率越高 = 开销越大
2. **内核持续时间**：较长的内核分摊设置成本
3. **缓冲区大小**：较大的缓冲区减少刷新频率
4. **收集模式**：连续模式的开销低于序列化模式

典型开销范围：
- 低采样（周期=20-31）：1-5% 开销
- 中等采样（周期=10-19）：5-15% 开销
- 高采样（周期=5-9）：15-30% 开销

## 最佳实践

1. **从保守设置开始**
   - 初始使用较大的采样周期
   - 根据需要逐渐增加细节

2. **分析代表性工作负载**
   - 确保分析涵盖典型用例
   - 运行多次迭代以获得统计意义

3. **与其他指标关联**
   - 与 nvprof/ncu 指标结合
   - 与应用程序级别的计时交叉引用

4. **自动化分析**
   - 编写数据解析和分析脚本
   - 创建性能回归测试

## 下一步

- 尝试不同的采样频率以找到最佳设置
- 将连续 PC 采样应用于您自己的 CUDA 应用程序
- 结合 `pc_sampling_utility` 分析收集的数据
- 使用调试符号探索与源代码的关联
- 将 PC 采样数据集成到您的性能分析工作流程中