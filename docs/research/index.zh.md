---
title: 研究
description: "基于一手来源的系统研究简报，提出可验证论点，比较互相竞争的证据，并明确架构影响、适用范围和可证伪条件。"
---

# 研究

Eunomia Research 发布基于一手来源的系统研究简报。目标不是总结一轮新闻，而是从一个精确技术问题出发，比较独立证据，识别冲突与替代解释，并说明结论会改变哪项架构或工程决策。

这些页面属于经过审查的研究简报，除非页面另有说明，并不等同于同行评审论文。每项判断都受证据范围、来源截止时间、假设和可证伪条件约束。后续证据改变结论时，应更新稳定页面，而不是再创建一篇高度重复的文章。

## 当前研究

### [并行 Agent 需要 Commit Protocol：从 Effect Serializability 到契约有效执行](https://eunomia.dev/zh/research/parallel-agent-effect-serializability/)

并行调用、worktree、reducer 和可串行化的资源更新，仍然可能组合成违背用户任务或使用过期权限的结果。本文审视当前 Agent runtime，并针对代码、API、预算、审批与不可逆动作提出 contract-valid effect serializability。

### [AI Agent 轨迹到底该保留什么：固定证据预算下的可观测性设计](https://eunomia.dev/zh/research/agent-trace-evidence-budget/)

一次模型调用周围可能产生数百个系统事件，完整轨迹却仍可能缺少决定性的状态、权限和 provenance。本文提出一种 evidence portfolio，把代表性采样、异常捕获、因果 flight recorder 与 outcome evidence 组合起来。

## 发布标准

一篇研究简报必须提供比流畅综述更多的价值：

- 问题会影响真实的系统或工程决策；
- 使用相互独立的一手来源，并在可能时加入实现或测量证据；
- 存在真实张力、矛盾或替代解释；
- 综合结论不能从任意单篇来源直接复制；
- 明确列出假设、适用范围、不确定性和可证伪条件；
- 给出能够推进判断的实验、artifact 或后续观察。

[博客](https://eunomia.dev/zh/blog/)继续保存项目文章、教程、release、工程解释和相对成熟的观点。研究简报单独放在这里，使不断变化的证据状态保持可见，同时不降低博客本身的编辑约定。
