---
title: 每日报告
description: "面向系统研究与生产实践的技术分析，比较一手证据，识别尚未解决的问题，并提出具有学术价值和生产价值的可验证方向。"
---

# 每日报告

Eunomia 每日报告围绕具体系统问题展开，比较一手证据，分析现有研究或生产实践仍然解释不了、测不出来或落不了地的部分，并提出可以继续做成研究系统或工程机制的方向。

## 当前报告

### [用户态 eBPF 要成为真正的运行时，还缺什么？](https://eunomia.dev/zh/research/userspace-ebpf-runtime-contract/)

一个 BPF VM 可以执行指令，却未必定义程序如何挂载、拥有哪些能力、状态由谁持有，以及扩展如何撤销和做资源归属。本文比较 Linux eBPF、uBPF、bpftime 与 eBPF for Windows，并提出机器可读的运行时契约、绑定能力的 attach handle 和 per-extension 资源账本。

### [多个 AI Agent 同时工作时，谁来保证最终结果是对的？](https://eunomia.dev/zh/research/parallel-agent-effect-serializability/)

`worktree`、沙箱和并行工具调用可以隔离 worker，却仍可能产生错误的组合结果。本文用代码修改、共享预算、审批和不可逆操作说明：并行 Agent 的结果在真正生效前，需要经过一次统一的验证和提交。文章还分析现有 benchmark 与工具效果契约的不足，并提出 Agent 事务层、语义冲突 benchmark 和自适应并发控制等方向。

### [AI Agent 轨迹到底该保留什么：固定证据预算下的可观测性设计](https://eunomia.dev/zh/research/agent-trace-evidence-budget/)

一次模型调用周围可能产生数百个系统事件，完整轨迹却仍可能缺少决定性的状态、权限和 provenance。本文提出 evidence portfolio，并进一步分析证据效用、可移植 schema、无偏自适应采集与同预算评测仍然存在的研究空白。
