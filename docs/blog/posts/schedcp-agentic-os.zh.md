---
date: 2026-07-10
slug: schedcp-agentic-linux-scheduler
description: SchedCP 为 AI Agent 建立一条从 workload 意图到经过验证的 sched_ext 策略的受控路径，在论文评测中实现最高 1.79x 性能提升，并把优化成本降低 13x。
---

# 让 AI Agent 调优 Linux 调度器：SchedCP 如何接入 sched_ext

并行编译 Linux kernel 追求吞吐量，交互式服务关心稳定的唤醒延迟，混合工作负载则会随着关键路径变化而改变目标。Linux scheduler 能看到 runnable tasks、优先级和运行时统计，却不知道 operator 真正想优化什么。两个在内核看来十分相似的 workload，可能需要完全不同的调度策略。

AI Agent 能阅读 benchmark 描述、检查性能计数器，并用自然语言理解优化目标。直接给 Agent 一个 shell，让它改写 scheduler，又会产生新的风险：生成的 kernel policy code 仍然需要成功编译、通过安全检查、正确部署，并用实验确认它改善了整个 workload，而非只优化了某个容易测量的指标。

[SchedCP](https://github.com/eunomia-bpf/schedcp) 为 Agent 提供了一条范围更窄、约束更清晰的路径。它通过 Model Context Protocol（MCP）server 暴露 scheduler 的观测和控制能力，再使用 Linux `sched_ext` 选择或生成 eBPF 调度策略。对应论文 [**Towards Agentic OS: An LLM Agent Framework for Linux Schedulers**](https://arxiv.org/abs/2509.01245) 研究了这个 control plane 如何把语义推理与特权执行分开。

<!-- more -->

## Scheduler 看不到的语义

Linux 已经提供了丰富的 scheduler telemetry，`sched_ext` 也允许管理员加载用 eBPF 实现的自定义调度类。真正缺少的是两者之间的连接。运行时计数器描述已经发生的事情，优化请求表达 operator 真正在意的目标。像“降低前台服务的 tail latency，同时避免 batch jobs 饥饿”这样的请求，需要被转换成可测量目标、候选策略和安全部署计划。

通用 Agent 可以尝试在一次对话中完成所有步骤，但它会反复消耗 token 重新理解 scheduler interface、解析噪声输出并修复生成的代码，同时把特权操作紧贴在开放式推理旁边。一条幻觉命令或无效配置，就可能让实验在产生有效反馈之前结束。

SchedCP 把这条 semantic gap 当作 control plane 问题处理。AI 负责理解 workload 意图并选择优化方向，稳定的系统层负责 profiling、policy construction、verification、deployment 和 measurement。两侧分别使用自己最擅长处理的表示。

## 两个阶段组成一个受控闭环

论文把优化分为 goal inference 和 policy synthesis。Goal inference 把 workload 描述与测量数据转换成明确目标；policy synthesis 根据目标选择已有 scheduler，或者生成新的 `sched_ext` policy。

两个阶段组成下面的闭环：

```text
workload + goal
      |
      v
profile and infer objective
      |
      v
select or synthesize sched_ext policy
      |
      v
compile, verify, deploy, and measure
      |
      +---------- feedback ----------+
```

这个闭环很重要，因为 scheduler optimization 最终要靠实验判断。一个听起来合理的策略，可能因为 cache behavior、wake-up pattern 或 contention 与 Agent 的初始判断不同，在真实 workload 上反而变慢。SchedCP 把测量结果送回下一轮决策，并保留有效策略，供后续相似 workload 复用。

## 从观测走到经过验证的策略

三个服务让闭环始终由证据驱动。Workload analysis engine 运行目标命令，收集 CPU、内存、scheduler 和 application-level measurements，再把结构化证据交给 Agent。Policy repository 保存已有 schedulers、配置和历史结果，让 Agent 在生成新代码之前先利用已经验证过的起点。

已有策略达不到目标时，execution verifier 接管 synthesis path。Agent 通过专用接口提交 scheduler source，SchedCP 完成编译、静态检查和动态检查，策略通过后才能进入实验。测量也由同一个 control plane 完成，因此进入 repository 的策略会带上 workload context 和真实结果，不会只留下一个缺少证据的性能声明。

MCP boundary 把这些操作组织为明确工具，包括 scheduler listing、monitored execution、system measurement，以及带 verification 的 scheduler creation。Agent 可以围绕这些工具规划，无需获得所有底层命令的无限制访问。随着 scheduler implementation 和 benchmark harness 演进，MCP 还能保持上层接口稳定。

![SchedCP control-plane 设计](https://raw.githubusercontent.com/eunomia-bpf/schedcp/master/document/design.png)

## 为什么 `sched_ext` 适合 Agentic Optimization

传统 scheduler 实验通常需要 kernel patches、reboot cycle 或 out-of-tree modules，这些成本会让迭代搜索变慢，也会放大错误候选策略的影响。`sched_ext` 把 scheduling policy 放进经过验证的 eBPF 程序中，同时保留由内核管理的安全边界。策略退出或失败后，系统可以恢复普通 scheduler，机器无需长期运行修改过的 kernel。

这个模型提供了 Agent 所需的迭代速度。SchedCP 可以编译策略，为一次 benchmark run 加载它，收集结果后卸载，再继续尝试下一个候选。Kernel verifier 与 SchedCP execution verifier 处理不同风险：前者检查 eBPF 安全属性，后者检查生成的 scheduler 及其部署方式是否符合即将执行的实验。

## 评测结果说明了什么

论文使用 Linux kernel build、`schbench` 和 batch processing 等 workload 评测 SchedCP。在这组评测中，生成或选择的策略带来最高 1.79x 的性能提升。复用 control plane、policy repository 和结构化工具，也让优化成本相对论文中的 naive agentic baseline 降低 13x。

这些数字对应论文评测范围，并不意味着每个 workload 都会自动变快。更重要的结果是，Agent 能以更小的搜索成本完成端到端 scheduler optimization loop，同时保留 verification 和测量反馈。仓库公开了 prompts、workloads、生成的 schedulers 和 benchmark paths，可以沿着 artifact 检查这些结果如何产生。

## 运行 Artifact

构建仓库及其 `sched_ext` submodules 后，`autotune` 接受一条 workload command 作为优化目标。论文 artifact 包含 Linux build workload 和 `schbench`，可以这样启动：

```bash
./autotune/target/release/autotune cc \
  "make -C workloads/linux-build-bench/linux -j"

./autotune/target/release/autotune cc \
  workloads/basic/schbench/schbench
```

[SchedCP 仓库](https://github.com/eunomia-bpf/schedcp)记录了 kernel requirements、build steps、MCP tools 和论文 workloads，[论文](https://arxiv.org/abs/2509.01245)则给出完整架构与评测方法。两者合在一起，可以从三个层面检查结果：Agent 的 reasoning interface、经过验证的 `sched_ext` policy，以及决定候选策略是否保留的 workload measurement。

SchedCP 展示了一种面向操作系统的 control plane：Agent 表达并持续修正目标，经过验证的机制掌握特权变更的最终执行权。Scheduler tuning 很适合作为验证场景，因为成功标准足够具体，生成策略需要安全运行，目标 workload 也必须真的得到改善。
