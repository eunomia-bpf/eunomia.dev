---
date: 2026-08-19
title: "性能分析器的采样什么时候会产生偏差？"
description: "采样结果看起来很精确，但相位锁定、skid 和短函数漏采都可能造成系统性偏差。本文讨论怎样让 profiler 给出可验证的置信度和退化路径。"
tags:
  - 每日报告
  - 性能分析
  - Sampling
  - Linux perf
  - 可观测性
  - 统计
research_question: "采样型性能分析器怎样识别并量化相位锁定、覆盖不足、skid 和其他系统性误差，而不是把 sample 百分比当成精确计数？"
source_cutoff: 2026-08-19
status: daily-report
---

# 性能分析器的采样什么时候会产生偏差？

一个 profiler 每秒对某个服务采样 1000 次。服务内部正好有一个重复的 1 ms 控制循环。报告显示函数 A 占 8.1% CPU，函数 B 占 5.7%。第二次运行时，两者的顺序反过来了。把采样频率再调高，两边的百分比又一起变了。

这时候应该相信哪一次？

最常见的回答是多采一点。这个办法只在主要问题是随机方差时有效。如果 sampler 总是在周期性 workload 的同一个相位打断程序，或者持续漏掉短函数，或者 PMU overflow 发生后实际记录的 IP 已经滑到了后面的指令，多跑十分钟只会收集更多同一种错误。

<!-- more -->

问题的本质是：sampling profile 是一个**估计器**，但很多 profiler 的 UI 把它显示得像精确计数器。一个 `8.1%` 隐藏了大量条件：sample 是怎样安排的、哪些代码有机会被看到、丢了多少 sample、采样周期有没有和程序周期同步、不同独立运行是不是得到相同排名。

这并不是新发现。Steven McCanne 和 Chris Torek 在 1993 年发表的 [A Randomized Sampling Clock for CPU Utilization Estimation and Code Profiling](https://www.usenix.org/conference/usenix-winter-1993-conference/randomized-sampling-clock-cpu-utilization-estimation-and) 就专门处理周期采样和程序行为同步的问题。后来一篇 USENIX profiler 工作也讨论了 DCPI 怎样随机化 sample interval，以及不可预测的一次性 timer 为什么能减少意外同步。到了今天，Linux kernel 的 Propeller 文档在收集硬件 sample 时仍然建议使用类似 `500009` 这样的合适大质数作为 event period，而不是随手选一个整齐的数。

但相位锁定只是其中一类误差。OSDI 2026 的 operational systems paper [When Sampling Lies: Trustworthy Performance Profiling for Flat Workloads with Blink](https://www.usenix.org/conference/osdi26/presentation/devsot) 研究了一类很平的 mobile workload：成千上万个生命周期很短的 routine 分摊整体执行时间，没有一个明显热点。作者发现 `perf` 一类 sampling profiler 会因为 skid、shadow effect 和 function coverage 不完整而出现系统性错误，不只是方差变大。Blink 改用轻量 instrumentation，在论文评测 workload 上报告了 99.999% accuracy 和约 1% overhead。

这些结果说明，比“提高采样频率”更有价值的目标是：**profiler 应该能告诉用户，当前 workload 下自己的估计到底可不可信；如果不可信，还应该有一个有预算上限的精确化路径。**

本文刻意把这个问题作为 adjacent-systems 主题，而不是 eBPF 主题。它同样适用于 `perf`、PMU profiler、runtime profiler、mobile profiler、GPU profiler 和其他周期或 event-driven sampler。eBPF 可以参与实现，但并不是核心机制成立的必要条件。

## 一个 sample 百分比背后有很多假设

Linux `perf_event_open()` 同时提供 `sample_period` 和 `sample_freq`。使用 period 时，每累计到指定数量的 event 就触发一次 overflow。使用 frequency mode 时，kernel 会调整 period，尽量达到目标采样频率。sample 本身还可以带上当前 period、timestamp、CPU、instruction pointer、callchain，以及丢 sample 等信息。

这些接口足够构建很强的 profiler，但下面这个比例：

```text
归因到 X 的 samples / 总 samples
```

只有在采样过程足够有代表性时，才能成为 CPU 时间或事件占比的好估计。

现实里有几类假设很容易失效。

### 采样时钟会和 workload 同步

假设程序一直重复下面的周期：

```text
A 运行 300 us -> B 运行 300 us -> C 运行 400 us -> 重复
```

如果 profiler 每 1 ms 在相同相位取一次 sample，它可能几乎只看到其中一段。继续收集十倍时间没有帮助，因为新增观察一直重复原来的偏差。

这也是为什么 randomized sampling 这个三十多年前的想法今天仍然值得认真对待。1993 年的 randomized clock 工作把 synchronization 当成 sampling 设计问题。Linux Propeller 文档建议使用较大质数 period，也是在工程实践里尽量避免简单周期关系。

大质数不是独立性的证明，frequency feedback 也不等于随机采样。真正重要的是：**sample timing 本身属于测量设计，不只是 overhead 参数。**

### Hardware sample 会 skid

PMU overflow 并不总能精确落在逻辑上造成 event 的那条指令。现代硬件和 `perf` 在部分 event 上提供更精确的采样能力，但不同事件和架构之间差异很大。

如果只是找一个占 CPU 40% 的大热点，一点 skid 可能无关紧要。如果是大量短函数、很平的 profile，或者需要指令级 event attribution，skid 就可能直接改变排名。Blink 在 OSDI 2026 的结果里把它当成系统性误差来源之一，而不是简单的随机噪声。

### 很短的函数可能几乎没有被观察到的机会

一个函数每次只执行 20 微秒，但被调用很多次，总 CPU 成本仍然可能很高。可是在任意一个 sampler interrupt 到来的瞬间，它恰好在运行的概率可能很低。周围更长的函数反而更容易被 sample 命中，于是占比被高估。

这和“样本数不够”不是一回事。如果 observation process 系统性 shadow 掉某一类函数，延长采样时间甚至可能越来越稳定地收敛到错误分布。

### Profiler 会动态改变自己的采样过程

Sampling 有成本。Linux 的 `perf_cpu_time_max_percent` 可以在 profiler 消耗过多 CPU 时触发 throttling。frequency mode 也会动态调整 period 来追踪目标频率。

这些机制本身很合理，但它们意味着有效采样过程会随时间变化。如果最终产物只保存一张 symbol histogram，就会丢掉判断这张 histogram 是怎样采出来的证据。

## 现有工作还薄弱在哪里

### Randomization 常常只是一个调参技巧，而不是测量契约

随机化 sample period 本身并不新。真正缺的是 profiler 把**实际发生的采样 schedule**保存成一等证据，让用户能在事后判断是否发生 aliasing。

一个更可靠的 profile 至少应该能回答：

- 下一次 sample interval 来自什么分布？
- 最终实际 interval 是多少？
- kernel 或 runtime 有没有 throttle 或重新调整？
- sample 时间相对于某个重复 workload 周期是否集中在固定相位？
- 换一个独立 schedule 后，hot-code 排名是否仍然一致？

如果没有这些信息，用户只能不断改 `-F` 和 `-c`，然后肉眼比较两张报告，却不知道结果为什么变了。

### Sample 数量很多，不等于置信度很高

一千万个 sample 听起来很有说服力。但如果 workload 有强周期性，或者大量 sample 都落在同一个相位，或者一类短函数一直被漏掉，那么这些 sample 并不是一千万个独立观察。

因此，只基于独立 Bernoulli sample 假设得到的经典 confidence interval 可能过于乐观。Profiler 的 uncertainty 应该反映时间相关性、独立 collection epoch 和 coverage，而不是只看 `sqrt(n)`。

### Profiler UI 很少区分随机方差和结构性偏差

这两个问题需要完全不同的处理：

1. **随机方差**：采样过程大体正确，只是 sample 不够多；
2. **结构性偏差**：sampler 系统性地看到了错误的执行区域。

多采样对第一类有帮助。对第二类，它只会让错误答案看起来更“稳定”。

一个好的 profiler 应该把两种状态分开。多个独立 randomized epoch 的排名反复变化，说明证据还不够。sample 结果长期和 selective instrumentation 或已知 ground truth 不一致，则更像结构性偏差。

### Instrumentation 不应该总被当成 sampling 的对立面

Sampling 的优势是 overhead 低、部署范围广。Instrumentation 的覆盖更完整，但传统上成本更高，也可能改变程序行为。

Blink 表明这个 trade-off 取决于 workload 和实现。论文里的约 1% overhead 当然不能直接推广到所有程序，但它说明 instrumentation 可以作为一个**局部 oracle**，而不是只能在“全量 sampling”和“全量 instrumentation”之间二选一。

这带来一个更实用的问题：profiler 能不能只在 sampling 证据明显不足的地方花 instrumentation budget？

## 值得继续做的研究和工程方向

### 1. 带 aliasing 诊断的 randomized sampling contract

**Gap.** 现有接口能设置 period 或目标频率，但最终 profile 通常没有完整描述实际采样随机过程，因此出了 synchronization 以后很难解释。

**Mechanism.** 把 sampling schedule 当成 profile 的一等元数据。每个 collection epoch 给出目标 overhead budget 和 interval distribution，例如 bounded exponential 或带 jitter 的 renewal process，而不是固定周期。每个 sample 同时记录 intended interval、realized interval、trigger source、当前 hardware period、CPU、timestamp，以及 throttle 和 loss 信息。

分析层再做 aliasing 检查。它可以从 scheduler、request、runtime 或 application marker 中寻找主要 workload period，然后检查 sample timestamp 对这些周期取模后的 phase distribution。如果样本集中在很窄的 phase 范围，就说明 sampling clock 没有均匀探索整个执行周期。

Sampler 也不应该无脑随机化。某些有明确语义的硬件 event 更适合按 event count 采样，有些 profiler 还需要可复现的确定性 schedule。因此 contract 应该明确记录 fixed、frequency-controlled、randomized 或 mixed mode，而不是强制只有一种模式。

**Delta.** Randomized clock 已经是已有工作。新的系统贡献不是“再随机一次 timer”，而是把 schedule 及其真实执行结果写进 profile artifact，并提供明确的 aliasing diagnostic。

**Artifact.** 一个兼容 `perf` 数据路径的 prototype，或者独立 collector。它输出 sampling manifest，并在报告里显示 interval distribution、phase concentration、throttling、sample loss 和不同 epoch 之间的 profile 差异。

**Evaluation.** 构造周期长度已知的 microbenchmark，在相同 overhead budget 下比较 fixed period、prime period、frequency mode 和几种 randomized schedule。再加入 phase drift、CPU migration 和负载变化。测量 per-function estimation error、top-k rank error、aliasing detection precision/recall 和 overhead。

**学术价值。** 可以直接测量 sampler 的随机过程与 profile bias 之间的关系。

**生产价值。** Hot-function 排名变化时，工程师能判断到底是 workload 变了，还是 sampler 和 workload 同步了。

**Failure condition.** 如果普通 `perf` frequency mode 或经过良好选择的固定 period 已经能在现实周期 workload 上消除有意义的 phase bias，那么更复杂的 randomized contract 不值得维护。

### 2. 用独立 profile epoch 表达 uncertainty 和 rank stability

**Gap.** 传统 profile 百分比没有 uncertainty；sample 数量又不能区分独立证据和强相关观察。

**Mechanism.** 把总采集预算分成几个彼此独立的 epoch，每个 epoch 使用独立 seed 或 schedule。对每个 symbol 和 stack 保留 per-epoch estimate，不要一开始就把全部 samples 合并掉。最后输出 central estimate、跨 epoch uncertainty 和 rank-stability score。

对于有强时间相关性的长 workload，应把 epoch 当成 resampling unit，而不是假设每一个 sample 都 IID。可以使用 block bootstrap 或其他 dependence-aware estimator 来估计变化范围。

UI 还应该明确显示“尚未解决”的排序。如果 A 是 8%，B 是 7%，但两个 epoch-level interval 大量重叠，就不应该用两位小数把 A 固定排在 B 前面。相反，如果独立 schedule 下 top five 始终一致，这比一张巨大的合并 histogram 更有说服力。

**Delta.** 统计学里的 uncertainty 当然不是新东西。这里真正的系统问题，是 profile collection pipeline 是否保留了足够结构，让 uncertainty 和 rank stability 成为合法、可验证的测量属性。

**Artifact.** 扩展 profile format 和 analysis tool，保存 epoch identity、effective sample period、lost sample、per-epoch histogram、uncertainty interval 和 stable/unstable rank group。

**Evaluation.** 用 CPU share 已知的 workload、重复 phase change 和 flat short-function workload 比较 naive sample-count interval、epoch-based interval 和 ground truth。测 interval coverage、top-k stability、false-confidence rate，以及达到稳定优化决策需要的采集时间。

**学术价值。** 问题会从“收了多少 sample”变成“这个 profile decision 在统计上什么时候真正 resolved”。

**生产价值。** 工程师可以在决策已经稳定时停止采集，也可以避免把随机排名当成热点去优化。

**Failure condition.** 如果正常生产时间窗口下 epoch-level uncertainty 总是太宽，几乎无法给出有用排序，那么应该改进 stratification 或 estimator，而不是只在 UI 上加一个 confidence badge。

### 3. 由 uncertainty 触发 selective instrumentation

**Gap.** Sampling 的最大价值仍然是便宜，但最容易被 sampling 骗到的 workload 恰好需要更强覆盖。

**Mechanism.** 先低成本 sampling。用前面的诊断找出证据不足的区域，例如 rank 不稳定的 symbol、疑似覆盖率过低的短函数、phase-sensitive function group，或者 skid 风险很大的 event。然后只在这些区域开启一个有严格预算的短期 instrumentation window。

这个 instrumented window 作为局部 oracle，可以记录 entry count、受控 timing 或选定的 exact event。Profiler 把 sampling estimate 和 oracle 对比，判断当前区域能否安全用 sampling；如果不能，就明确标记 unresolved 或临时保留 instrumentation。

Correction 必须保守。不能用一次 ratio 永久“修正”后续 profile，因为 workload 结构可能已经变化。输出里要保留哪些值来自 sampled、instrumented、corrected 或 unresolved。

**Delta.** Hybrid profiler 并不新，Blink 也证明了轻量 instrumentation 在一些 workload 上很实用。这里新的问题是：**能不能让 uncertainty 自己成为 instrumentation placement 的控制信号**，并且始终遵守固定 overhead budget？

**Artifact.** 一个 hybrid profiler，包含 budget manager、sampling diagnostics、临时 function instrumentation，以及带 provenance 的 report format。

**Evaluation.** 复现 Blink 一类 flat workload，同时加入少数函数占主导的普通 server workload。在同样 overhead budget 下比较 sampling-only、full instrumentation 和 uncertainty-triggered instrumentation。测 coverage、attribution error、top-k correctness、instrumentation footprint 和 time-to-diagnosis。

**学术价值。** Measurement confidence 从一个离线统计值变成 profiler 的在线控制信号。

**生产价值。** Fleet profiler 大多数时间保持低成本，只有 sampling 证据确实不足时才为精度付费。

**Failure condition.** 如果识别 uncertain region 本身就需要接近全量 instrumentation 的成本，或者动态 instrumentation 对 flat workload 的扰动大于收益，那么 hybrid design 就不成立。

## Benchmark 应该专门包含会“骗过” sampler 的 workload

一个 profiler 在“一个函数占 50% CPU”的 workload 上表现很好，不代表它面对周期和 flat workload 也可靠。因此 benchmark 不能只选常见 application，还要专门攻击 estimator。

可以包含下面这些 case：

| Workload | 主要检查的问题 |
| --- | --- |
| 固定周期 phase | phase locking 和 aliasing |
| 缓慢漂移的 phase | randomization 是否仍有代表性 |
| 上千个等权短函数 | coverage 和 rank error |
| 一个明显热点函数 | sampling 应该表现很好的简单场景 |
| bursty request phase | 时间相关性和 epoch 划分 |
| 支持不同 precision 的 PMU event | skid sensitivity |
| CPU 压力下的 sampler | throttling 和实际 sample rate 变化 |
| 有 selective instrumentation ground truth 的 case | estimator error 和 interval coverage |

评价指标也不能只有 overhead 和图看起来像不像。至少需要：

- per-symbol relative error；
- top-k precision / recall；
- pairwise rank reversal；
- 实际执行过的函数中，有多少从未被 sample 看见；
- uncertainty interval coverage；
- false-confidence rate，也就是 profiler 声称排名已经稳定，但 ground truth 证明它错了的比例；
- effective sample rate 和 sample loss；
- CPU / memory overhead。

最有意义的指标可能是 **decision error**。如果 profiler 导致工程师或 optimizer 去优化错误的函数，那么整体 histogram 只有很小的数值误差也没有太大安慰作用。

## 第一版 prototype 应该尽量普通

这个问题一开始不需要发明新的 kernel subsystem。

一个有价值的第一版完全可以围绕现有 `perf_event_open()` 做：

1. 收几个短而独立的 epoch；
2. 在总 overhead budget 不变的条件下，改变或 jitter 支持的 sampling period；
3. 保存 perf 已经能给出的 timestamp、period、loss 和 throttle 证据；
4. 计算 phase concentration 和 rank stability；
5. 在 benchmark 里只 instrumentation 一小组 ambiguous function；
6. 把两条路径都和已知 ground truth 对比。

只有这个实验明确证明缺少底层 primitive 之后，才应该讨论 kernel 侧支持，例如更直接的 randomized overflow mode 或更完整的 sample-schedule provenance。

这个顺序很重要。McCanne 和 Torek 三十多年前就已经证明 sampling clock 会影响结果。2026 年真正值得做的系统工作不应该只是“再随机一下 timer”。它应该证明：**现代 profiler 能识别自己的 measurement 什么时候有偏差，能量化还没解决的 uncertainty，并且在不突破 overhead budget 的前提下升级精度。**

## 哪些结果会改变这个判断？

有三类结果会明显削弱上面的方向。

第一，如果当前 `perf` frequency control 加上常见 event-period 选择，在周期、flat、短函数和 phase-changing workload 上已经能以低 overhead 给出无明显偏差且稳定的排名，那么没有必要再增加 sampling contract。

第二，如果 Blink 一类 instrumentation 或其他低成本 instrumentation 方法在同样 workload 上持续更准确，而且成本已经足够低，那么更直接的工程答案可能是 instrumentation，而不是让 sampling 变得越来越复杂。

第三，如果 epoch-level uncertainty 和 aliasing diagnostic 与真实 ground-truth error 没有预测关系，那么这些统计值只是看起来合理。Benchmark 必须证明 warning 真正对应更高的错误优化决策概率。

在这些实验完成之前，sample 百分比更应该被看成有明确 measurement design 的估计，而不是精确 CPU accounting。真正有意思的问题已经不是“怎样收更多 sample”，而是 profiler 怎样解释：**自己的这些 sample 到底什么时候已经足够可信。**
