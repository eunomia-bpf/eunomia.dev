---
title: AgentSight 架构
description: 从 eBPF probe 到分析、存储和可视化的数据路径。
---

AgentSight 是一条 capture pipeline。eBPF 程序收集底层事件，Rust collector 对事件做规范化和分析，frontend 把 session 渲染成进程、时间线、日志和指标视图。

## Pipeline

```text
Agent process
  -> eBPF probes
  -> JSON event stream
  -> Rust runners
  -> analyzer chain
  -> storage and web UI
```

## eBPF 捕获层

eBPF 层在 agent framework 之下观察行为：

- 支持的 userspace TLS 库的 read/write 路径。
- 进程生命周期事件。
- 文件活动和命令执行。
- 系统级时间和资源信号。

这一层的价值在于可以看到子进程和 closed-source 工具，即使它们没有发出应用层 trace。

## Collector

Rust collector 负责运行 capture program，并把原始事件转换成一致的事件流。它负责：

- 启动和停止 runner。
- 规范化 timestamp 和 process metadata。
- 在可行时解析 HTTP 或 LLM payload。
- 在输出前过滤或脱敏事件。
- 提供本地 Web UI。

## Analyzer chain

Analyzer 是事件流上的小型处理阶段。常见工作包括：

- HTTP 解析。
- Server-sent event chunk 合并。
- Authentication header 移除。
- Timestamp 规范化。
- File logging 或 OpenTelemetry export。

这种结构把 capture code 和 interpretation logic 分开，新增 parser 或输出路径时不需要改 eBPF 程序。

## Frontend

Frontend 读取规范化后的事件流，并提供几个运维视图：

- Process tree：查看父子进程关系。
- Timeline view：按时间顺序查看 request、tool call、文件访问和命令执行。
- Log view：检查原始 event。
- Resource metrics：查看 CPU、memory 和 runtime 行为。

## 限制

AgentSight 是可观测工具。它可以提供发生了什么的证据，但 enforcement 应由 policy layer 实现。它也依赖目标进程、TLS 库、kernel 能力以及必要 eBPF hook 的可用性。
