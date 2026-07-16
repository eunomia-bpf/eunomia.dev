---
date: 2026-07-10
slug: schedcp-agentic-linux-scheduler
description: SchedCP 为 AI Agent 建立一条从 workload 意图到经过验证的 sched_ext 策略的受控路径，在论文评测中实现最高 1.79x 性能提升，并把优化成本降低 13x。
---

# 让 AI Agent 调优 Linux 调度器：SchedCP 如何接入 sched_ext

加载一个面向吞吐量的 scheduler，并行编译 Linux kernel 可能更快完成。把同一策略直接用于 `schbench`，wake-up latency 却可能朝相反方向变化。两次运行都让 CPU 保持繁忙，scheduler counters 无法说明 operator 更在意哪一个结果，缺失的输入是 workload goal。

AI Agent 可以理解“降低 tail latency，同时避免 batch jobs 饥饿”这样的请求，再把它转换成实验。一个不受限制的 root shell 也会放大错误，一条幻觉命令、损坏的配置或无效 scheduler 都可能让实验在产生有效反馈前结束。生成的 kernel policy code 仍需成功编译、通过安全检查、正确部署，并在测量中改善目标 workload。

[SchedCP](https://github.com/eunomia-bpf/schedcp) 把这次开放式 shell session 收束成一个受控优化循环。Model Context Protocol（MCP）server 只暴露明确的观测和 scheduler management 操作，Linux `sched_ext` 则提供可加载的 eBPF policies。论文 [**Towards Agentic OS: An LLM Agent Framework for Linux Schedulers**](https://arxiv.org/abs/2509.01245) 研究了这种拆分如何保留 Agent 的推理空间，同时让 systems control plane 掌握特权执行权。

<!-- more -->

## 到底要让哪个指标变快

以优化 `schbench` 为例，CPU utilization 无法单独判断候选策略是否有效。Agent 需要提取 latency objective，固定 workload parameters，再比较多轮运行结果。混合前台和 batch workload 还会引入额外约束，因为平均吞吐量改善时，关键路径上的进程仍可能遭遇饥饿。Runtime counters 描述发生了什么，用户请求决定哪些 measurements 才算成功。

通用 Agent 可以尝试在一次对话中完成所有步骤，但它会反复消耗 token 重新理解 scheduler interface、解析噪声输出并修复生成的代码，同时把特权操作紧贴在开放式推理旁边。一条幻觉命令或无效配置，就可能让实验在产生有效反馈之前结束。

SchedCP 把从请求走到实验的转换作为 control-plane 问题处理。Agent 理解 workload 意图并提出优化方向，稳定的系统层负责 profiling、policy construction、verification、deployment 和 measurement，让每个候选都沿同一条路径从 source code 走到 evidence。

## 把请求转换成实验

SchedCP 先执行 goal inference，把 workload 描述和 baseline measurements 转换成明确目标。Policy synthesis 随后从 repository 中寻找合适的 scheduler，或者生成新的 `sched_ext` policy。两个阶段分开后，一个听起来合理的 scheduler choice 就无法悄悄改写自己原本应该优化的目标。

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

## 把生成的 Scheduler 放在 Verifier 后面

三个服务让闭环始终由证据驱动。Workload analysis engine 运行完全相同的目标命令，返回 CPU、内存、scheduler 和 application-level measurements。Policy repository 把 schedulers 连同配置、workload context 和历史结果一起保存，让 Agent 在生成代码前先检查已有证据。

已有策略达不到目标时，execution verifier 接管 synthesis path。Agent 通过专用接口提交 scheduler source，SchedCP 完成编译、静态检查和动态检查，策略通过后才能进入实验。测量也由同一个 control plane 完成，因此进入 repository 的策略会带上 workload context 和真实结果，不会只留下一个缺少证据的性能声明。

MCP boundary 把这些操作暴露为职责清晰的工具：`list_schedulers` 查找已有 policies，`system_monitor` 收集 measurements，`create_and_verify_scheduler` 只在 verification 通过后接纳生成的 source。Agent 围绕这些工具规划，deployment details 则留在 SchedCP 内部。即使 benchmark harness 和 scheduler implementation 继续演进，上层 reasoning interface 也能保持稳定。

![SchedCP control-plane 设计](https://raw.githubusercontent.com/eunomia-bpf/schedcp/master/document/design.png)

## 为什么 `sched_ext` 适合 Agentic Optimization

传统 scheduler 实验通常需要 kernel patches、reboot cycle 或 out-of-tree modules，这些成本会让迭代搜索变慢，也会放大错误候选策略的影响。`sched_ext` 把 scheduling policy 放进经过验证的 eBPF 程序中，同时保留由内核管理的安全边界。策略退出或失败后，系统可以恢复普通 scheduler，机器无需长期运行修改过的 kernel。

这个模型提供了 Agent 所需的迭代速度。SchedCP 可以编译策略，为一次 benchmark run 加载它，收集结果后卸载，再继续尝试下一个候选。Kernel verifier 与 SchedCP execution verifier 处理不同风险：前者检查 eBPF 安全属性，后者检查生成的 scheduler 及其部署方式是否符合即将执行的实验。

## 评测结果说明了什么

论文使用 Linux kernel build、`schbench` 和 batch processing 等 workload 评测 SchedCP。在这组评测中，生成或选择的策略带来最高 1.79x 的性能提升。复用 control plane、policy repository 和结构化工具，也让优化成本相对论文中的 naive agentic baseline 降低 13x。

1.79x 是被评测 workloads 中的最高改善，13x 则来自与论文中 naive agentic baseline 的比较。这些数据支持一个范围更明确的结论：结构化 control plane 让 Agent 用更低成本完成端到端搜索，同时每个被接受的候选仍需经过 verification 和 workload measurement。仓库公开了 prompts、workloads、生成的 schedulers 和 benchmark paths，可以沿着 artifact 检查整个过程。

## 运行 Artifact

构建仓库及其 `sched_ext` submodules 后，`autotune` 接受一条 workload command 作为优化目标。论文 artifact 包含 Linux build workload 和 `schbench`，可以这样启动：

```bash
./autotune/target/release/autotune cc \
  "make -C workloads/linux-build-bench/linux -j"

./autotune/target/release/autotune cc \
  workloads/basic/schbench/schbench
```

[SchedCP 仓库](https://github.com/eunomia-bpf/schedcp)记录了 kernel requirements、build steps、MCP tools 和论文 workloads，[论文](https://arxiv.org/abs/2509.01245)则给出完整架构与评测方法。两者合在一起，可以从三个层面检查结果：Agent 的 reasoning interface、经过验证的 `sched_ext` policy，以及决定候选策略是否保留的 workload measurement。

Scheduler tuning 给 agentic OS control 提供了一个要求很高的验证场景。Agent 必须把意图转换成 metric，生成 kernel 能运行的 policy，并在测量结果推翻初始判断时接受证据。SchedCP 把这些转换显式化，由 Agent 做语义选择，由能够验证、测量和回滚每次尝试的机制负责特权变更。
