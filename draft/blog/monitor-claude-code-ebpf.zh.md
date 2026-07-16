---
date: 2026-07-21
slug: monitor-claude-code-ebpf
description: "用 AgentSight 和 eBPF 零侵入监控 Claude Code，无需 SDK 或代码修改，实时捕获每个 prompt、工具调用、文件操作和子进程。"
---

# 用 eBPF 监控 Claude Code：完整运行时可见性

你给 Claude Code 一个任务：重构认证模块。十五分钟后任务完成，测试通过，diff 看起来很干净。但在 prompt 和 commit 之间，到底发生了什么？它发了多少次 API 请求？除了改动的文件之外还读取了哪些文件？是否启动了你没有预料到的子进程？

Claude Code 的内置 hooks 机制可以拦截特定的工具调用，但仅限于 agent 框架暴露出来的那些。如果某个子进程调用了 `curl` 或通过子进程写入文件，hooks 不会触发。传统的可观测工具（Langfuse、LangSmith 以及基于 OpenTelemetry 的方案）全都需要在应用代码中集成 SDK，而 Claude Code 是一个闭源编译二进制文件，你无法对它做任何插桩。

[AgentSight](https://github.com/eunomia-bpf/agentsight) 采用了完全不同的方式。它利用 eBPF 从进程外部拦截 TLS 加密的 LLM 流量和内核级系统事件，不需要修改任何代码。本教程将带你完成 AgentSight 的安装、Claude Code 会话录制，以及捕获数据的探索。

<!-- more -->

## 为什么 Hooks 和 SDK 插桩不够用

Claude Code 提供了一个 hooks 系统，在工具调用等事件前后执行 shell 命令。Hooks 适合做策略执行（比如"阻止 `git push`"），但它们只能看到 agent 框架选择暴露的事件。工具调用启动的子进程、子进程打开的文件、意外的网络连接，这些都发生在 hook 层之下。

基于 SDK 的可观测工具走的是另一条路：在应用代码中埋点来产生 traces 和 spans。Langfuse、SigNoz、Arize Phoenix 这类工具在你拥有源代码并且可以添加 `@trace` 装饰器或回调处理器时效果很好。Claude Code 是一个编译好的 Bun 二进制文件，内部静态链接了 BoringSSL 并且符号已被剥离。你没法往里面加 SDK。

| 方式 | 能看到什么 | 盲区 |
|------|-----------|------|
| **Claude Code hooks** | 框架暴露的工具调用 | 子进程、文件 I/O、hook 层以下的网络调用 |
| **SDK/OTel 插桩** | SDK 包裹的内容 | 需要源代码访问权限，无法插桩闭源 CLI |
| **AgentSight (eBPF)** | 所有 TLS 流量、进程树、文件操作、网络活动 | 需要支持 eBPF 的 Linux 和 sudo 权限 |

AgentSight 在内核层面工作。它把 uprobes 挂载到 Claude Code 二进制文件内部的 SSL 库函数上，在 `SSL_read`/`SSL_write` 边界捕获明文的请求和响应数据，然后将这些流量与内核 tracepoints 产生的进程执行事件进行关联。agent 本身完全不需要修改。

## 安装 AgentSight

AgentSight 是一个单一二进制文件。通过 Cargo 安装或直接下载预编译版本：

```bash
cargo install agentsight
# 或者直接下载 release 二进制：
wget https://github.com/eunomia-bpf/agentsight/releases/latest/download/agentsight && chmod +x agentsight
```

**前置条件：** Linux 内核 4.1+（推荐 5.0+）并支持 eBPF，以及 eBPF 探针所需的 sudo 权限。

## 录制 Claude Code 会话

一条命令就能在 AgentSight 下启动 Claude Code：

```bash
sudo agentsight record -- claude
```

这条命令自动完成所有工作：通过 `$PATH` 解析 `claude` 二进制文件，跟踪符号链接找到真正的可执行文件（例如 `claude` -> `~/.local/share/claude/versions/2.1.150`），通过字节模式匹配发现静态链接的 BoringSSL 库，然后挂载 uprobes 进行 SSL 捕获，同时启动进程和系统监控。Claude Code 的 TUI 在终端中照常工作，AgentSight 在后台安静地录制。

退出 Claude Code 后，AgentSight 将整个会话保存到当前目录下的 `agentsight-*.db` SQLite 文件中。

## 探索捕获的数据

### 实时 Web UI

录制期间，在浏览器中打开 [http://127.0.0.1:7395](http://127.0.0.1:7395) 就能看到实时仪表盘，包含四个视图：

- **Timeline**（`/timeline`）：所有 LLM 调用、进程启动、文件操作和网络事件的统一时间线
- **Process Tree**（`/tree`）：以 Claude Code 为根的完整进程树，展示子进程、参数和退出状态
- **Event Log**（`/logs`）：支持过滤的原始事件流
- **Metrics**（`/metrics`）：CPU 和内存使用的实时变化

### 命令行报告

会话结束后，查询保存的数据库：

```bash
agentsight report                    # 会话概览
agentsight report token              # token 用量统计
agentsight report prompts --json     # 完整的 prompt/response 载荷
agentsight report audit --json       # 进程启动、文件打开、API 调用
agentsight report serve              # 对已保存的会话重新打开 Web UI
```

### 实时 Top 视图

想要跨多个 agent 会话持续监控，`agentsight top` 提供类似 Unix `top` 的实时排名视图：

```bash
sudo agentsight top
```

会话按模型、token 用量、健康度、进程族、工具调用、文件活动和网络活动排名显示。

## eBPF 如何实现这一切

Claude Code 是一个 Bun 应用，内部编译了 BoringSSL 且符号已被剥离。系统中没有共享的 `libssl.so` 可供传统拦截工具挂载。AgentSight 的 `record` 命令透明地处理了这个问题：通过字节模式匹配检测嵌入的 BoringSSL，将 uprobes 挂载到正确的函数偏移位置，在加密之前捕获明文 HTTP 载荷。

进程监控使用内核 tracepoints 追踪 Claude Code 启动的每一个子进程，包括深层嵌套的子进程。文件监控捕获打开、写入、截断、重命名和删除操作。所有这些内核级事件都与 LLM 流量时间线关联在一起，你可以追踪每个 API 调用触发了哪些系统操作。

开销很低：我们的评估显示，典型的 agent 追踪工作负载下 CPU 开销低于 3%。被监控的 agent 以你的普通用户身份运行，只有 eBPF 探针需要 sudo 权限。

想深入了解 BoringSSL 拦截的原理，参见 [用 eBPF 逆向 Claude Code 的 SSL 流量](https://eunomia.dev/blog/2026/02/13/reverse-engineering-claude-codes-ssl-traffic-with-ebpf/)。

## 常见问题

**能实时看到 Claude Code 在做什么吗？**
可以。运行 `sudo agentsight record -- claude`，然后打开 `http://127.0.0.1:7395/timeline`。你会看到 prompt、模型响应、工具调用、子进程执行和文件操作实时出现。

**AgentSight 会拖慢 Claude Code 吗？**
我们的评估显示，典型追踪工作负载下 CPU 开销低于 3%。eBPF 探针在内核层面运行，不会向 agent 进程注入代码。

**AgentSight 支持其他 AI 编程 agent 吗？**
AgentSight 支持任何发起 TLS 加密 API 调用的进程。`record` 可以自动发现 Gemini CLI、Codex、Python agent（aider、open-interpreter）、Node.js 工具以及 Docker 容器中的 agent。对任意 agent 运行 `sudo agentsight record -- <command>` 即可。

**能把数据导出到现有的可观测平台吗？**
AgentSight 可以将捕获的 LLM 调用导出为 OpenTelemetry GenAI（`gen_ai.*`）spans，通过 OTLP/HTTP 协议发送，让你无需进程内插桩就能将 agent 遥测数据发送到任何兼容 OTel 的后端：

```bash
sudo agentsight debug trace --otel --otel-endpoint http://localhost:4318
```

## 下一步

AgentSight 将 Claude Code 从一个黑盒变成了可观测的系统，且不需要对 agent 做任何改动。想进一步了解：

- **[AgentSight GitHub](https://github.com/eunomia-bpf/agentsight)**：安装、文档和源代码
- **[用语义火焰图分析 AI agent](https://eunomia.dev/blog/2026/06/24/profiling-ai-agents-with-semantic-flamegraphs/)**：将多会话 agent 行为聚合为 pprof 兼容的火焰图
- **[AgentSight：用 eBPF 掌控你的 AI agent](https://eunomia.dev/blog/2025/08/26/agentsight-keeping-your-ai-agents-under-control-with-ebpf-powered-system-observability/)**：最初的发布公告和架构概览
