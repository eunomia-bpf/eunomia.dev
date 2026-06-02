---
title: AgentSight 故障排查
description: 事件缺失、TLS hook 失败和 UI 问题的常见处理方法。
---

这一页覆盖本地运行 AgentSight 时最常见的问题。

## 目标进程没有出现

先确认 command name 是否匹配真实进程名：

```bash
ps -eo pid,comm,args | grep -E 'claude|node|python'
```

然后把 `comm` 列中的值传给 AgentSight：

```bash
sudo ./agentsight record -c node
```

## 缺少 TLS event

可能原因：

- 目标进程使用的 TLS 库与当前 probe 预期不同。
- Runtime 静态打包了 OpenSSL。
- 真正发起网络请求的不是被 trace 的进程。
- 流量经过了另一个未被 trace 的 proxy 进程。

如果 Node.js binary path 明确，可以尝试：

```bash
sudo ./agentsight record --binary-path /usr/bin/node -c node
```

## 权限错误

使用 root 权限运行 AgentSight，并确认 host 允许加载 eBPF 程序。一些限制严格的 container、kernel 或 security profile 会阻止所需操作。

## UI 无法打开

确认本地 server 正在运行，且 `8080` 端口没有被其他进程占用：

```bash
ss -ltnp | grep 8080
```

如果端口已被占用，停止对应进程；如果当前 build 支持，也可以配置 AgentSight 使用其他端口。

## event 太多

尽量缩小目标进程范围，并在可行时加入 analyzer filter。对于敏感 session，应在 recording 前减少 capture scope，而不是只依赖事后脱敏。

## 数据顺序看起来不对

比较事件时优先使用 collector 规范化后的 timestamp。原始 kernel、userspace 和 frontend timestamp 可能因 buffering 或 clock source 不同而有差异。
