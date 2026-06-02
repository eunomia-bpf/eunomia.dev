---
title: AgentSight 可视化
description: 如何阅读 AgentSight UI，并把事件关联成一个 agent session。
---

AgentSight UI 面向 session investigation。通常先看进程树，再用 timeline 和 raw logs 检查 prompt、tool call 或子进程执行附近的具体事件。

## Process tree

进程树展示 agent 进程及其子进程。它适合先回答这些问题：

- 哪个命令启动了 agent？
- agent 创建了哪些子进程？
- 某个 tool call 是通过 shell、Python、Node.js 还是其他 binary 执行的？
- 哪个进程执行了文件或网络操作？

![AgentSight process tree](https://github.com/eunomia-bpf/agentsight/raw/master/docs/demo-tree.png)

## Timeline

Timeline 按时间排列事件。可以用来关联：

- LLM request/response event。
- Tool call 和子进程执行。
- 模型响应前后的文件访问。
- 特定 agent step 周围的网络活动。

![AgentSight timeline](https://github.com/eunomia-bpf/agentsight/raw/master/docs/demo-timeline.png)

## Logs

Log view 展示规范化和原始 event payload。适合在这些情况下使用：

- 解析后的 event 看起来不完整。
- 需要精确的 HTTP payload、命令参数或文件路径。
- 调试新的 analyzer 或 filter。

## Metrics

Metrics view 用来解释 agent session 附近的 runtime 行为，例如 agent reasoning、调用工具或创建进程时的 CPU 和 memory 变化。

![AgentSight metrics](https://github.com/eunomia-bpf/agentsight/raw/master/docs/demo-metrics.png)

## 调查流程

1. 在 process tree 里找到 root agent process。
2. 展开子进程，确认 tool execution。
3. 跳到 timeline 中对应的时间范围。
4. 对需要证据的事件检查 raw log payload。
5. 用 metrics 解释性能或资源异常。
