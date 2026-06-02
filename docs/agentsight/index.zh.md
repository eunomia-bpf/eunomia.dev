---
title: AgentSight
description: 基于 eBPF 的 LLM 与 AI agent 零插桩可观测工具。
---

AgentSight 从系统边界观测 LLM 和 AI agent 行为。它把进程活动、TLS/网络流量、文件访问、工具执行和资源行为关联起来，不需要在 agent 应用里接入 SDK。

当应用日志不够可信或不够完整时，可以使用 AgentSight：例如会执行子进程的 coding agent、调用远程 LLM API 的 CLI agent、多 agent workflow，或者需要审计证据的生产试点。

项目链接：

- GitHub: <https://github.com/eunomia-bpf/agentsight>
- Paper: <https://arxiv.org/abs/2508.02736>
- 旧页面：[/GPTtrace/agentsight/](/GPTtrace/agentsight/)

## 能捕获什么

- 在支持的 TLS 库路径上，捕获加密前的 LLM API 请求与响应。
- 进程树、子进程执行、命令参数和生命周期事件。
- 与 agent 行为相关的文件读写和元数据事件。
- 可与进程活动关联的网络和 HTTP 信号。
- 用于回放 agent session 的运行时指标和事件时间线。

## 为什么在系统层观测

应用层 tracing 只能看到应用主动上报的内容。AgentSight 位于应用之下，更适合观察 closed-source 工具、agent 子进程、shell 命令，以及绕过框架插桩的提示词、工具调用或文件访问。

AgentSight 不是要替代应用层 tracing，而是补足 LangSmith、OpenTelemetry、proxy 日志或自定义 SDK 之外的 ground-truth 层。

## 文档

- [快速开始](quickstart.md)：安装二进制，记录 Claude Code、Gemini CLI 或 Python agent，并打开 UI。
- [架构](architecture.md)：eBPF probe、Rust collector、analyzer、存储和 UI 如何协作。
- [可视化](visualization.md)：如何阅读进程树、事件时间线、日志视图和指标。
- [运行说明](operational-notes.md)：权限、隐私、支持环境和部署建议。
- [故障排查](troubleshooting.md)：事件缺失或 TLS hook 挂载失败时的常见原因。

## 最小流程

```bash
wget https://github.com/eunomia-bpf/agentsight/releases/download/v0.1.1/agentsight
chmod +x agentsight

# 记录 Claude Code。
sudo ./agentsight record -c claude

# 记录 Gemini CLI；它通常以 node 进程运行。
sudo ./agentsight record -c node

# 记录 Python AI 工具。
sudo ./agentsight record -c python
```

然后打开 <http://127.0.0.1:8080> 查看捕获到的 session。

## 当前范围

AgentSight 主要面向 Linux agent 进程，以及常见 userspace TLS 和进程可观测路径。它提供证据和可见性；策略执行、sandbox 和 runtime guardrail 应交给 ActPlane 或其他专门的安全控制层。
