---
title: AgentSight 运行说明
description: AgentSight 的权限、隐私、支持环境和部署建议。
---

AgentSight 会捕获敏感运行时数据。应把它的输出当作审计证据处理，并以包含 prompt、API payload、文件路径和命令参数的日志同等级别保护。

## 权限

AgentSight 需要加载 eBPF 程序的权限。本地开发时通常使用 `sudo`。在生产或类生产环境中，只应在 operator 有权限捕获进程和网络 metadata 的主机上运行。

## 隐私和脱敏

捕获数据可能包含：

- Prompt 和 response 文本。
- HTTP header 和 request metadata。
- 文件路径和命令参数。
- Tool output 和子进程行为。

使用 analyzer filter 移除 authentication header，并避免存储调查不需要的数据。共享 trace 时，应脱敏 prompt、secret、token 和私有文件路径。

## 适用环境

AgentSight 最适合 agent 进程直接运行、或运行在具备足够 eBPF 捕获权限的容器中的 Linux host。不适合的情况包括：

- Host 不允许加载 eBPF 程序。
- 目标进程使用不支持的 TLS 路径。
- 所有流量都隐藏在另一个未被 trace 的 proxy 进程之后。
- 从法律或组织规则上不能捕获 prompt payload。

## 部署方式

开发和事故复现时，可以围绕单个 agent 命令本地运行 AgentSight。做 pilot 部署时，建议限定少量 host 或 process，定义 retention policy，并决定哪些字段在入库前需要脱敏。

## 证据处理

用于审计或 incident 时：

1. 记录启动 AgentSight 的命令。
2. 记录目标 process name 和 binary path。
3. 保留 timestamp 和 host metadata。
4. 当 parser output 有争议时，保留 raw event log。
5. 把 trace 存储在有访问控制的位置。
