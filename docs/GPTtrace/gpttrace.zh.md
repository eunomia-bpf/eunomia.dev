---
title: GPTtrace 文档
description: 使用自然语言和AI生成并运行eBPF程序。通过GPT驱动的自动化和智能分析追踪您的Linux系统。
keywords: GPTtrace, AI eBPF, 自然语言追踪, GPT追踪, eBPF自动化, LLM内核代理, 智能系统监控
author: eunomia-bpf 社区
---

## GPTtrace 🤖

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Actions Status](https://github.com/eunomia-bpf/GPTtrace/workflows/Pylint/badge.svg)](https://github.com/eunomia-bpf/GPTtrace/actions)
[![DeepSource](https://deepsource.io/gh/eunomia-bpf/eunomia-bpf.svg/?label=active+issues&show_trend=true&token=rcSI3J1-gpwLIgZWtKZC-N6C)](https://deepsource.io/gh/eunomia-bpf/eunomia-bpf/?ref=repository-badge)
[![CodeFactor](https://www.codefactor.io/repository/github/eunomia-bpf/eunomia-bpf/badge)](https://www.codefactor.io/repository/github/eunomia-bpf/eunomia-bpf)
[![DOI](https://zenodo.org/badge/603351016.svg)](https://zenodo.org/badge/latestdoi/603351016)

使用GPT和自然语言生成eBPF程序并进行追踪的实验项目

想要在线版本？请查看 [GPTtrace-web](https://github.com/eunomia-bpf/GPTtrace-web) 获取**在线演示**！

### **请查看我们在 eBPF'24 的论文 [Kgent: Kernel Extensions Large Language Model Agent](https://dl.acm.org/doi/10.1145/3672197.3673434)！**

## 主要特性 💡

### 使用自然语言与您的Linux交互和追踪

示例：按进程统计页面错误

<img src="https://github.com/eunomia-bpf/GPTtrace/raw/main/doc/trace.png" alt="Image" width="600">

- 使用自然语言开始追踪
- 让AI为您解释结果

### 使用自然语言生成eBPF程序

示例：编写一个eBPF程序，打印所有运行中的shell输入的bash命令，将bpf程序保存到文件并退出，不实际运行它。

<img src="https://github.com/eunomia-bpf/GPTtrace/raw/main/doc/generate.png" alt="Image" width="600">

我们使用来自 [bpftrace tools](https://github.com/eunomia-bpf/GPTtrace/tree/main/tools) 的示例创建向量存储并进行搜索。

有关如何编写eBPF程序的更多详细文档和教程，请参考：[`bpf-developer-tutorial`](https://github.com/eunomia-bpf/bpf-developer-tutorial)（一个libbpf工具教程，教ChatGPT编写eBPF程序）

### 选择合适的bcc命令行工具来完成追踪任务

使用合适的bcc工具追踪内核

```console
$ python3 gpttrace "追踪内存分配并显示每个单独的分配器函数调用"
 Run:  sudo memleak-bpfcc --trace 
Attaching to kernel allocators, Ctrl+C to quit.
(b'Relay(35)', 402, 6, b'd...1', 20299.252425, b'alloc exited, size = 4096, result = ffff8881009cc000')
(b'Relay(35)', 402, 6, b'd...1', 20299.252425, b'free entered, address = ffff8881009cc000, size = 4096')
(b'Relay(35)', 402, 6, b'd...1', 20299.252426, b'free entered, address = 588a6f, size = 4096')
(b'Relay(35)', 402, 6, b'd...1', 20299.252427, b'alloc entered, size = 4096')
(b'Relay(35)', 402, 6, b'd...1', 20299.252427, b'alloc exited, size = 4096, result = ffff8881009cc000')
(b'Relay(35)', 402, 6, b'd...1', 20299.252428, b'free entered, address = ffff8881009cc000, size = 4096')
(b'sudo', 6938, 10, b'd...1', 20299.252437, b'alloc entered, size = 2048')
(b'sudo', 6938, 10, b'd...1', 20299.252439, b'alloc exited, size = 2048, result = ffff88822e845800')
(b'node', 410, 18, b'd...1', 20299.252455, b'alloc entered, size = 256')
(b'node', 410, 18, b'd...1', 20299.252457, b'alloc exited, size = 256, result = ffff8882e9b66400')
(b'node', 410, 18, b'd...1', 20299.252458, b'alloc entered, size = 2048')
```

## 工作原理

![GPTtrace/doc/how-it-works.png](https://github.com/eunomia-bpf/GPTtrace/raw/main/doc/how-it-works.png)

1. **用户输入**：用户提供其操作系统信息和内核版本。这些信息至关重要，因为它有助于根据用户的特定环境定制eBPF程序。
2. **提示构建**：用户的输入以及操作系统信息和内核版本用于构建提示。此提示旨在指导eBPF程序的生成。
3. **向量数据库查询**：构建的提示用于查询向量数据库中的eBPF程序示例。这些示例作为生成将插入内核的eBPF程序的基础。
4. **挂钩点识别**：使用GPT API识别eBPF程序中的潜在挂钩点。这些挂钩点是代码中可以插入eBPF程序以监控或修改内核行为的位置。
5. **eBPF程序生成**：识别的挂钩点以及向量数据库中的示例用于生成eBPF程序。该程序旨在插入内核以执行所需的追踪任务。
6. **内核插入**：生成的eBPF程序被插入内核。如果此过程中出现任何错误，工具将重试从查询向量数据库到内核插入的步骤几次。
7. **结果解释**：一旦eBPF程序成功插入内核，AI将向用户解释结果。这包括解释eBPF程序正在做什么以及它如何与内核交互。

此过程确保eBPF程序针对用户的特定环境和需求进行定制，并且用户了解程序的工作原理和功能。

## 安装 🔧

```sh
pip install gpttrace
```

## 使用和设置 🛠

```console
$ python3 -m gpttrace -h
usage: GPTtrace [-h] [-c CMD_NAME QUERY] [-v] [-k OPENAI_API_KEY]
                input_string

使用ChatGPT编写eBPF程序（bpftrace等）

positional arguments:
  input_string          您对bpf程序的问题或请求

options:
  -h, --help            显示此帮助消息并退出
  -c CMD_NAME QUERY, --cmd CMD_NAME QUERY
                        使用bcc工具完成追踪任务
  -v, --verbose         显示更多详细信息
  -k OPENAI_API_KEY, --key OPENAI_API_KEY
                        Openai api密钥，参见
                        `https://platform.openai.com/docs/quickstart/add-
                        your-api-key` 或通过 `OPENAI_API_KEY` 传递
```

### 首先：登录ChatGPT

- 访问 <https://platform.openai.com/docs/quickstart/add-your-api-key>，然后按照以下步骤创建您的openai api密钥：

  ![image-20230402163041886](https://github.com/eunomia-bpf/GPTtrace/raw/main/doc/api-key.png)

- 记住您的密钥，然后将其设置为环境变量 `OPENAI_API_KEY` 或使用 `-k` 选项。

### 开始您的追踪！ 🚀

例如：

```sh
python3 gpttrace "按进程统计页面错误"
```

如果eBPF程序无法加载到内核中，错误消息将用于纠正ChatGPT，结果将打印到控制台。

## 示例

- 按进程打开的文件
- 按程序统计系统调用
- 按进程读取的字节数：
- 按进程读取大小分布：
- 显示每秒系统调用率：
- 按进程追踪磁盘大小
- 按进程统计页面错误
- 按进程名称和PID统计LLC缓存未命中（使用PMC）：
- 以99赫兹的频率对PID 189进行用户级堆栈分析：
- 为根cgroup-v2中的进程打开的文件

## 引用

```bibtex
@inproceedings{10.1145/3672197.3673434,
author = {Zheng, Yusheng and Yang, Yiwei and Chen, Maolin and Quinn, Andrew},
title = {Kgent: Kernel Extensions Large Language Model Agent},
year = {2024},
isbn = {9798400707124},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3672197.3673434},
doi = {10.1145/3672197.3673434},
abstract = {扩展的伯克利包过滤器（eBPF）生态系统允许扩展Linux和Windows内核，但由于需要了解操作系统内部知识和eBPF验证器强制执行的编程限制，编写eBPF程序具有挑战性。这些限制确保只有专家级内核开发人员才能扩展他们的内核，使得初级系统管理员、补丁制作者和DevOps人员难以维护扩展。本文提出了Kgent，这是一个替代框架，通过允许使用自然语言编写内核扩展来减轻编写eBPF程序的难度。Kgent使用大型语言模型（LLM）的最新进展，根据用户的英语语言提示合成eBPF程序。为确保LLM的输出在语义上等同于用户的提示，Kgent采用了LLM驱动的程序理解、符号执行和一系列反馈循环的组合。Kgent的关键创新是这些技术的组合。特别是，该系统使用符号执行的新结构，使其能够结合程序合成和程序理解的结果，并建立在LLM在这些任务中分别显示的最近成功基础上。为了评估Kgent，我们开发了一个新的eBPF程序自然语言提示语料库。我们表明，Kgent在80%的情况下生成正确的eBPF程序——与GPT-4程序合成基线相比，提高了2.67倍。此外，我们发现Kgent很少合成"假阳性"eBPF程序——即Kgent验证为正确但手动检查显示对输入提示语义不正确的eBPF程序。Kgent的代码可在<https://github.com/eunomia-bpf/KEN>公开访问。},
booktitle = {Proceedings of the ACM SIGCOMM 2024 Workshop on EBPF and Kernel Extensions},
pages = {30–36},
numpages = {7},
keywords = {大型语言模型, 符号执行, eBPF},
location = {Sydney, NSW, Australia},
series = {eBPF '24}
}
```

## 许可证

MIT

## 🔗 链接

- 关于我们如何训练ChatGPT编写eBPF程序的详细文档和教程：<https://github.com/eunomia-bpf/bpf-developer-tutorial>（基于CO-RE（一次编写，到处运行）libbpf的eBPF开发者教程：通过20个小工具一步步学习eBPF（尝试教会ChatGPT编写eBPF程序））
- bpftrace：<https://github.com/iovisor/bpftrace>
- ChatGPT：<https://chat.openai.com/>