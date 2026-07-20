---
date: 2026-07-10
slug: schedcp-agentic-linux-scheduler
description: Linux 调度器调优要先明确工作负载目标，SchedCP 论文报告基于 sched_ext 的验证闭环最高带来 1.79x 加速，并把搜索成本降低 13x。
---

# 让 AI Agent 调优 Linux 调度器：SchedCP 如何接入 sched_ext

一个让 Linux 内核编译更快的调度策略，换到 `schbench` 上可能反而拉高唤醒延迟。两次运行都能让 CPU 忙起来，但运维者在前者关心吞吐量，在后者关心唤醒时间，所以 AI Agent 要安全调优调度器，必须先把工作负载目标放进控制闭环。

AI Agent 可以理解“降低尾延迟，同时避免批处理任务饥饿”这样的请求，再把它转换成实验。一个不受限制的 root shell 也会放大错误，一条幻觉命令、损坏的配置或无效调度器都可能让实验在产生有效反馈前结束。生成的内核策略代码仍需成功编译、通过安全检查、正确部署，并在测量中改善目标工作负载。

[SchedCP](https://github.com/eunomia-bpf/schedcp) 把开放式 shell 会话收束成一个受控优化循环。Model Context Protocol（MCP）服务器只暴露明确的观测和调度器管理操作，Linux `sched_ext` 则提供可加载的 eBPF 策略。论文 [**Towards Agentic OS: An LLM Agent Framework for Linux Schedulers**](https://arxiv.org/abs/2509.01245) 研究了这种拆分如何保留 Agent 的推理空间，同时让系统控制面掌握特权执行权。

<!-- more -->

## 到底要让哪个指标变快

以优化 `schbench` 为例，CPU 利用率无法单独判断候选策略是否有效。Agent 需要提取延迟目标，固定工作负载参数，再比较多轮运行结果。混合前台和批处理工作负载还会引入额外约束，因为平均吞吐量改善时，关键路径上的进程仍可能遭遇饥饿。运行时计数器描述发生了什么，用户请求决定哪些测量结果才算成功。

通用 Agent 可以尝试在一次对话中完成所有步骤，但它会反复消耗 token 重新理解 scheduler interface、解析噪声输出并修复生成的代码，同时把特权操作紧贴在开放式推理旁边。一条幻觉命令或无效配置，就可能让实验在产生有效反馈之前结束。

SchedCP 把从请求走到实验的转换作为控制面问题处理。Agent 理解工作负载意图并提出优化方向，稳定的系统层负责分析、策略构造、验证、部署和测量，让每个候选都沿同一条路径从源代码走到证据。

## 把请求转换成实验

SchedCP 先执行目标推断，把工作负载描述和基线测量转换成明确目标。策略生成随后从仓库中寻找合适的调度器，或者生成新的 `sched_ext` 策略。两个阶段分开后，一个听起来合理的调度器选择就无法悄悄改写自己原本应该优化的目标。

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

## 把生成的调度策略放在 Verifier 后面

三个服务让闭环始终由证据驱动。Workload analysis engine 运行完全相同的目标命令，返回 CPU、内存、scheduler 和 application-level measurements。Policy repository 把 schedulers 连同配置、workload context 和历史结果一起保存，让 Agent 在生成代码前先检查已有证据。

已有策略达不到目标时，执行验证器（execution verifier）接管生成路径。论文描述了三道门槛。内核 eBPF verifier 检查内存安全与终止性，调度器专用的静态分析检查饥饿、不公平等内核 verifier 不覆盖的逻辑问题，动态验证再把候选放进安全 micro-VM。验证通过后系统签发部署令牌，通过带监控的灰度部署运行。如果实测性能下降，熔断器可以回滚策略。

MCP boundary 把这些操作暴露为职责清晰的工具：`list_schedulers` 查找已有 policies，`system_monitor` 收集 measurements，`create_and_verify_scheduler` 只在 verification 通过后接纳生成的 source。Agent 围绕这些工具规划，deployment details 则留在 SchedCP 内部。即使 benchmark harness 和 scheduler implementation 继续演进，上层 reasoning interface 也能保持稳定。

![SchedCP control-plane 设计](https://raw.githubusercontent.com/eunomia-bpf/schedcp/master/document/design.png)

## 为什么 `sched_ext` 适合 Agent 迭代优化

传统 scheduler 实验通常需要 kernel patches、reboot cycle 或 out-of-tree modules，这些成本会让迭代搜索变慢，也会放大错误候选策略的影响。`sched_ext` 把 scheduling policy 放进经过验证的 eBPF 程序中，同时保留由内核管理的安全边界。策略退出或失败后，系统可以恢复普通 scheduler，机器无需长期运行修改过的 kernel。

这个模型提供了 Agent 所需的迭代速度。SchedCP 可以编译策略，为一次 benchmark run 加载它，收集结果后卸载，再继续尝试下一个候选。Kernel verifier 与 SchedCP execution verifier 处理不同风险：前者检查 eBPF 安全属性，后者检查生成的 scheduler 及其部署方式是否符合即将执行的实验。

## 评测结果说明了什么

论文把这组结果明确称为初步评测。实验使用两台机器，分别运行 Linux 6.13 和 6.14，由 Claude Code 使用 Opus 4 完成优化，每个案例测量三次后取平均值。结果证明闭环可以纠正一次糟糕的初始选择，但还不能证明它能在任意机器和工作负载上自动改善调度。

在 Linux 内核编译上，第一次选择的 `scx_rusty` 相对 EEVDF 获得 1.63x 加速，后续迭代改用 `scx_layered`，最终达到 1.79x。更能说明反馈价值的是 `schbench`。Agent 第一次配置的 `scx_bpfland` 反而弱于 EEVDF，经过三轮反馈后改用 `scx_rusty`，P99 延迟相对 EEVDF 改善 2.11x，吞吐量提升 1.60x。

对包含 39 个短任务和 1 个长任务的 8 组批处理工作负载，sched-agent 推导出 Longest Job First 策略，让平均延迟降低 20%。论文还报告调度器生成效率提升 13x，每个工作负载的生成时间降到 2.5 分钟，生成成本为 $0.45。

这些数据支持的结论比“Agent 总能让调度器更快”窄得多。结构化控制面可以降低搜索成本、拒绝不安全候选，并用工作负载反馈推翻 Agent 的初始选择。仓库公开了 prompts、工作负载、生成的调度器和基准测试路径，便于检查这个过程。更完整的基准测试仍属于后续工作。

## 运行实验 Artifact

构建仓库及其 `sched_ext` submodules 后，`autotune` 接受一条 workload command 作为优化目标。论文 artifact 包含 Linux build workload 和 `schbench`，可以这样启动：

```bash
./autotune/target/release/autotune cc \
  "make -C workloads/linux-build-bench/linux -j"

./autotune/target/release/autotune cc \
  workloads/basic/schbench/schbench
```

[SchedCP 仓库](https://github.com/eunomia-bpf/schedcp)记录了 kernel requirements、build steps、MCP tools 和论文 workloads，[论文](https://arxiv.org/abs/2509.01245)则给出完整架构与评测方法。两者合在一起，可以从三个层面检查结果：Agent 的 reasoning interface、经过验证的 `sched_ext` policy，以及决定候选策略是否保留的 workload measurement。

调度器调优为 Agent 控制操作系统提供了一个要求很高的验证场景。Agent 必须把意图转换成指标，生成内核可以运行的策略，并在测量结果推翻初始判断时接受证据。SchedCP 把这些转换显式化，由 Agent 做语义选择，再由能够验证、测量和回滚每次尝试的机制负责特权变更。

## 参考资料

- [Towards Agentic OS：面向 Linux 调度器的 LLM Agent 框架](https://arxiv.org/abs/2509.01245)
- [SchedCP 仓库](https://github.com/eunomia-bpf/schedcp)
- [Linux 内核文档：可扩展调度器类](https://docs.kernel.org/scheduler/sched-ext.html)
- [Model Context Protocol 规范](https://modelcontextprotocol.io/specification/)
- [schbench 调度器基准测试](https://kernel.googlesource.com/pub/scm/linux/kernel/git/mason/schbench/)
