# CUPTI SASS 源代码映射教程

> GitHub 仓库和完整教程可在 <https://github.com/eunomia-bpf/cupti-tutorial> 获取。

## 简介

在优化 CUDA 内核时，理解源代码如何转换为实际的 GPU 汇编指令可以提供强大的洞察。SASS（流式汇编）是 NVIDIA GPU 执行的本机汇编语言，映射 CUDA C/C++ 代码与生成的 SASS 指令之间的关系可以揭示在源代码级别不明显的优化机会。本教程演示如何使用 CUPTI 提取内核的 SASS 代码并将其映射回原始源代码，帮助您理解高级代码与 GPU 上实际执行的指令之间的关系。

## 您将学到什么

- 如何提取 CUDA 内核的 SASS 汇编代码
- 将 SASS 指令映射回源代码行
- 解释 SASS 以理解指令级行为
- 通过分析 SASS 识别优化机会
- 使用 SASS 信息做出明智的优化决策

## 理解 CUDA 编译

当您编译 CUDA 代码时，它经过几个阶段：

1. **CUDA C/C++**：您的高级源代码
2. **PTX**：中间表示（并行线程执行）
3. **SASS**：GPU 执行的最终机器代码

每个阶段代表不同的抽象级别，理解最终的 SASS 代码可以提供在源代码级别不可见的洞察。

## 代码演练

### 1. 加载 CUDA 模块

首先，我们需要加载 CUDA 模块以访问其代码：

```cpp
CUmodule module;
CUfunction function;

// 加载模块
DRIVER_API_CALL(cuModuleLoad(&module, "kernel.cubin"));

// 获取内核函数
DRIVER_API_CALL(cuModuleGetFunction(&function, module, "vectorAdd"));
```

这段代码：
1. 加载编译的 CUDA 二进制（cubin）文件
2. 获取该模块内特定内核函数的句柄

### 2. 提取 SASS 代码

接下来，我们提取内核的 SASS 代码：

```cpp
// 获取函数的代码
CUdeviceptr code;
size_t codeSize;
DRIVER_API_CALL(cuFuncGetAttribute(&code, CU_FUNC_ATTRIBUTE_CODE, function));
DRIVER_API_CALL(cuFuncGetAttribute(&codeSize, CU_FUNC_ATTRIBUTE_BINARY_SIZE, function));

// 为代码分配内存
unsigned char *sassCode = (unsigned char *)malloc(codeSize);
if (!sassCode) {
    fprintf(stderr, "为 SASS 代码分配内存失败\n");
    return -1;
}

// 从设备内存复制代码
DRIVER_API_CALL(cuMemcpyDtoH(sassCode, code, codeSize));
```

这段代码：
1. 获取函数代码的设备指针及其大小
2. 分配内存来保存 SASS 代码
3. 将代码从设备内存复制到主机内存

### 3. 反汇编 SASS

现在我们将二进制 SASS 代码反汇编为人类可读的格式：

```cpp
// 创建反汇编器
CUpti_Activity_DisassembleData disassembleData;
memset(&disassembleData, 0, sizeof(disassembleData));
disassembleData.size = sizeof(disassembleData);
disassembleData.cubin = sassCode;
disassembleData.cubinSize = codeSize;
disassembleData.function = (const char *)function;

// 反汇编代码
CUPTI_CALL(cuptiActivityDisassembleKernel(&disassembleData));

// 获取反汇编的 SASS
const char *sassText = disassembleData.sass;
```

这段代码：
1. 设置反汇编的结构
2. 调用 CUPTI 反汇编内核
3. 获取生成的 SASS 文本

### 4. 将 SASS 映射到源代码

要将 SASS 指令映射到源代码，我们使用 CUPTI 的行信息 API：

```cpp
// 获取模块中的函数数量
uint32_t numFunctions = 0;
CUPTI_CALL(cuptiModuleGetNumFunctions(module, &numFunctions));

// 获取函数 ID
CUpti_ModuleResourceData *functionIds = 
    (CUpti_ModuleResourceData *)malloc(numFunctions * sizeof(CUpti_ModuleResourceData));
CUPTI_CALL(cuptiModuleGetFunctions(module, numFunctions, functionIds));

// 对于每个函数
for (uint32_t i = 0; i < numFunctions; i++) {
    // 检查这是否是我们的目标函数
    if (strcmp(functionIds[i].resourceName, "vectorAdd") == 0) {
        // 获取行信息
        uint32_t numLines = 0;
        CUPTI_CALL(cuptiGetNumLines(functionIds[i].function, &numLines));
        
        // 为行信息分配内存
        CUpti_LineInfo *lineInfo = 
            (CUpti_LineInfo *)malloc(numLines * sizeof(CUpti_LineInfo));
        
        // 获取行信息
        CUPTI_CALL(cuptiGetLineInfo(functionIds[i].function, numLines, lineInfo));
        
        // 处理行信息
        for (uint32_t j = 0; j < numLines; j++) {
            printf("偏移 0x%x 处的 SASS 指令映射到 %s:%d\n",
                   lineInfo[j].pcOffset, lineInfo[j].fileName, lineInfo[j].lineNumber);
        }
        
        free(lineInfo);
    }
}

free(functionIds);
```

这段代码：
1. 获取模块中的函数列表
2. 找到我们的目标函数
3. 获取该函数的行信息
4. 将每个 SASS 指令偏移映射到源文件和行号

### 5. 创建源代码注释的 SASS 列表

现在我们将 SASS 代码与源行信息结合：

```cpp
void printSourceAnnotatedSass(const char *sassText, CUpti_LineInfo *lineInfo, uint32_t numLines)
{
    // 解析 SASS 文本
    char *sassCopy = strdup(sassText);
    char *line = strtok(sassCopy, "\n");
    
    int currentSourceLine = -1;
    const char *currentFileName = NULL;
    
    // 处理每行 SASS
    while (line != NULL) {
        // 提取指令偏移
        unsigned int offset;
        if (sscanf(line, "/*%x*/", &offset) == 1) {
            // 为此偏移找到源行
            for (uint32_t i = 0; i < numLines; i++) {
                if (lineInfo[i].pcOffset == offset) {
                    // 如果我们移动到新的源行，打印它
                    if (currentSourceLine != lineInfo[i].lineNumber || 
                        currentFileName != lineInfo[i].fileName) {
                        currentSourceLine = lineInfo[i].lineNumber;
                        currentFileName = lineInfo[i].fileName;
                        
                        // 读取源文件并获取行
                        char sourceLine[1024];
                        FILE *sourceFile = fopen(currentFileName, "r");
                        if (sourceFile) {
                            for (int j = 0; j < currentSourceLine; j++) {
                                if (!fgets(sourceLine, sizeof(sourceLine), sourceFile)) {
                                    break;
                                }
                            }
                            fclose(sourceFile);
                            // 移除换行符
                            sourceLine[strcspn(sourceLine, "\n")] = 0;
                            
                            printf("\n源代码第 %d 行：%s\n", currentSourceLine, sourceLine);
                        }
                    }
                    break;
                }
            }
        }
        
        // 打印 SASS 指令
        printf("    %s\n", line);
        
        // 获取下一行
        line = strtok(NULL, "\n");
    }
    
    free(sassCopy);
}
```

这个函数：
1. 解析 SASS 文本行
2. 提取每条指令的偏移
3. 找到对应的源行
4. 当移动到新源行时打印源代码
5. 缩进打印 SASS 指令

## 实际应用

### 性能热点分析

通过源代码映射的 SASS，您可以：

1. **识别昂贵的源行**：查看哪些源代码行生成了最多的指令
2. **分析内存访问模式**：理解内存操作如何在汇编级别实现
3. **优化算法**：看到编译器如何转换您的算法

### 示例输出

```
源代码第 42 行：float a = A[i];
    /*0008*/         LDG.E.SYS R2, [R8];

源代码第 43 行：float b = B[i];
    /*0010*/         LDG.E.SYS R4, [R10];

源代码第 44 行：float result = a + b;
    /*0018*/         FADD R6, R2, R4;

源代码第 45 行：C[i] = result;
    /*0020*/         STG.E.SYS [R12], R6;
```

这个输出显示：
- 每个源行如何转换为特定的 SASS 指令
- 内存加载操作（LDG）用于读取数组元素
- 浮点加法（FADD）执行计算
- 内存存储操作（STG）写入结果

## 优化洞察

### 指令效率

通过检查 SASS，您可以识别：

1. **冗余指令**：编译器生成的不必要操作
2. **内存合并机会**：访问模式的优化潜力
3. **寄存器使用**：寄存器压力和溢出问题

### 编译器优化

```cpp
// 源代码
for (int i = 0; i < N; i++) {
    C[i] = A[i] + B[i];
}

// 可能生成展开的 SASS（如果编译器优化）
/*0008*/ LDG.E.64 R2, [R8];      // 加载 A[i] 和 A[i+1]
/*0010*/ LDG.E.64 R4, [R10];     // 加载 B[i] 和 B[i+1]
/*0018*/ FADD R6, R2, R4;        // 添加第一对
/*0020*/ FADD R7, R3, R5;        // 添加第二对
/*0028*/ STG.E.64 [R12], R6;     // 存储两个结果
```

这显示编译器如何展开循环并使用64位加载进行向量化。

## 调试技术

### 性能回归

当性能回归时，比较 SASS 输出可以揭示：

1. **指令计数变化**：新版本是否生成更多指令
2. **内存访问模式**：访问是否变得不那么高效
3. **分支行为**：控制流是否变得更复杂

### 代码验证

```cpp
void validateOptimization() {
    // 比较优化前后的 SASS
    char *beforeSass = extractSassForKernel("kernel_v1");
    char *afterSass = extractSassForKernel("kernel_v2");
    
    int beforeInstructions = countInstructions(beforeSass);
    int afterInstructions = countInstructions(afterSass);
    
    printf("优化前指令数：%d\n", beforeInstructions);
    printf("优化后指令数：%d\n", afterInstructions);
    
    if (afterInstructions < beforeInstructions) {
        printf("✓ 优化成功减少了指令数\n");
    } else {
        printf("⚠ 优化增加了指令数\n");
    }
}
```

## 最佳实践

### 设置编译标志

为了获得最佳的源映射：

```bash
# 包含调试信息
nvcc -g -lineinfo -o kernel kernel.cu

# 生成 cubin 文件
nvcc --cubin -o kernel.cubin kernel.cu

# 同时生成 PTX 以进行比较
nvcc -ptx -o kernel.ptx kernel.cu
```

### 分析工作流

1. **建立基线**：为当前实现创建 SASS 映射
2. **实施更改**：修改源代码
3. **比较 SASS**：查看汇编级别的差异
4. **测量性能**：验证 SASS 变化是否带来预期的性能提升

### 常见模式

```cpp
// 识别内存合并问题
void checkMemoryCoalescing(const char *sass) {
    if (strstr(sass, "LDG.E.32") != NULL) {
        printf("发现32位标量加载 - 考虑向量化\n");
    }
    if (strstr(sass, "LDG.E.128") != NULL) {
        printf("发现128位向量加载 - 良好的合并\n");
    }
}

// 分析分支效率
void analyzeBranches(const char *sass) {
    int branchCount = 0;
    char *pos = (char*)sass;
    while ((pos = strstr(pos, "BRA")) != NULL) {
        branchCount++;
        pos += 3;
    }
    printf("发现 %d 个分支指令\n", branchCount);
}
```

SASS 源代码映射为理解 CUDA 编译器行为和优化 GPU 代码提供了无价的洞察。通过桥接高级源代码和低级汇编之间的差距，您可以做出更明智的优化决策并验证您的更改产生了预期的效果。 