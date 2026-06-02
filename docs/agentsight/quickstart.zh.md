---
title: AgentSight 快速开始
description: 安装 AgentSight，记录一个 agent 进程，并在本地 UI 中查看 session。
---

这个指南展示如何记录一个本地 agent 进程，并打开内置 Web UI。

## 前置条件

- 支持 eBPF 的 Linux。推荐 kernel 5.x 或更新版本。
- 加载 eBPF 程序需要 root 权限。
- 一个目标进程，例如 `claude`、`node` 或 `python`。
- 如果从源码构建：需要 Rust、Node.js、clang/LLVM 和 libelf 开发头文件。

## 安装 release 二进制

```bash
wget https://github.com/eunomia-bpf/agentsight/releases/download/v0.1.1/agentsight
chmod +x agentsight
```

## 记录 agent

选择 agent 实际使用的进程 command name。

```bash
# Claude Code。
sudo ./agentsight record -c claude

# Gemini CLI 通常表现为 node 进程。
sudo ./agentsight record -c node

# Python-based agent 应用。
sudo ./agentsight record -c python
```

如果 agent 使用的 Node.js 二进制静态打包了 OpenSSL，可以显式传入 binary path：

```bash
sudo ./agentsight record --binary-path /usr/bin/node -c node
```

## 打开 UI

record 启动后，打开：

```text
http://127.0.0.1:8080
```

UI 会展示当前 session 的进程树、事件时间线、日志和资源指标。

## 先检查什么

1. 确认目标进程出现在进程树里。
2. 查看 agent 下面的子进程和命令执行。
3. 在 LLM 请求或工具调用附近检查 timeline event。
4. 调试 parser 行为时，用 log view 查看原始 event payload。

## 停止记录

在运行 AgentSight 的终端按 `Ctrl-C`。如果异常中断后仍有 eBPF 程序残留，可以重新启动 AgentSight 做清理，或使用仓库里的 cleanup script 卸载 stale probe。
