---
date: 2026-09-11
title: "一个 eBPF 优化结果，到什么程度才值得上线？"
description: "eBPF 优化可能在单个 microbenchmark 上很快，却让真实应用退化。本文讨论 evidence envelope、holdout 测试和上线 promotion gate。"
tags:
  - Daily Report
  - eBPF
  - Optimization
  - JIT
  - Benchmarking
  - Verification
research_question: "怎样证明一个 eBPF 优化的收益能跨 verifier、JIT、kernel、architecture、application 和 workload 成立，而不是只对某个被调过的 benchmark 有效？"
source_cutoff: 2026-09-11
status: daily-report
---

# 一个 eBPF 优化结果，到什么程度才值得上线？

假设一个 eBPF 优化把某个 `BPF_PROG_TEST_RUN` tight loop 加速了 18%。Verifier 还是通过，JIT 出来的指令也更少，同一台机器重复跑结果很稳定。现在能不能把它默认打开，然后用于 Cilium、Katran、Tracee，甚至一整个同时有 x86 和 ARM 的 fleet？

还不够。

这个优化可能在语义上完全正确，却仍然不适合生产。它也许把一个 arithmetic loop 做快了，却因为 code size 变大，在大型 program 里造成 I-cache 压力；也许只对某一个 JIT backend 有收益；也许被优化的 program 只占 application 总运行时间的一小部分；profile-guided pass 还可能只对训练时的 branch bias 有利，一换 workload phase 就开始掉速。如果优化过程本身是自动搜索甚至 agentic search，它还可能学会“把 benchmark 做快”，而不是把真正的 application 做快。

所以这里的问题已经不是 semantic equivalence。**Correctness 只能告诉我们 transformed program 能不能合法替代原程序，它不能告诉我们 performance claim 到底覆盖多大范围、优化在哪些环境里真的 profitable，也不能直接告诉我们证据是否足够支持生产上线。**

<!-- more -->

本文是当前 optimization series 的收尾篇。前面已经讨论过 [runtime-profile specialization](https://eunomia.dev/zh/research/ebpf-runtime-profile-specialization/)、[跨架构 specialization](https://eunomia.dev/zh/research/ebpf-portable-architecture-specialization/)、[execution provenance](https://eunomia.dev/zh/research/ebpf-specialization-debug-provenance/)、[native-operation trust boundary](https://eunomia.dev/zh/research/ebpf-native-operation-trust-boundary/) 和 [cross-backend operation semantics](https://eunomia.dev/zh/research/ebpf-cross-backend-operation-semantics/)。

前五篇分别解决 transformation 是否等价、runtime assumption 是否还成立、当前 architecture 能不能执行、事故时到底是哪一代 specialization 在跑、以及 delegated stateful operation 是否保持同一份 semantic contract。这里把这些 gate 都假设为已经通过，只剩最后一个问题：**发现一个 speedup 以后，我们凭什么认为它不是某一次实验的偶然结果？**

## 同一个优化，其实可以同时存在几种不同的“性能真相”

Kops 是一个很好的例子，因为它同时报告了 microbenchmark 和 application 结果。EInsn 用 native machine idiom 替换 verifier-visible BPF instruction sequence。论文里 microbenchmark 最高提升 24%，production application 最高提升 12%，并且在 x86-64 与 ARM64 上做了评估。

这两个数字并不矛盾，它们回答的是不同问题。

Microbenchmark 可以隔离出 rotate、conditional select、extract 之类的 sequence 到底有没有变便宜。但真实 application 还要问：这个 sequence 实际执行多少次？周围 instruction layout 怎么变？helper 和 map cost 是否占主导？code size 有没有破坏 cache？最终 workload 对省下来的这些 cycle 到底敏不敏感？

Linux BPF 的现有基础设施其实已经体现了这种分层。BPF CI 会用 `veristat` 跑复杂 BPF program，比较 verifier behavior，并发现 verifier-performance regression。`veristat` 很适合比较 processed instruction 等 verifier statistic，但 verifier 变快或变慢，并不等价于 runtime throughput 或 p99 latency 的变化。一个 optimization 因此不应该只有一个万能分数，而是需要几个不同的 oracle。

当前 [`bpf-bench`](https://github.com/eunomia-bpf/bpf-benchmark) 也把这个问题显式化了。现在的 corpus 包含 6 个 production eBPF application、146 个可比较的 BPF program measurement 和 42 个 microbenchmark task。它同时记录 verifier outcome、JIT code size、application lifecycle、workload correctness、raw workload metric 和 per-program kernel run counter。更重要的是，这套 framework 直接承认两个现实：static rewrite count 并不能可靠预测 speedup，而且一个 pass 可以让某些 program 更快，同时让另一些 program 退化。

这就是 evidence gap 的核心。Optimization result 不是一个 scalar，而是对 program、runtime、architecture、kernel、workload 和 measurement procedure 的条件化陈述。

## 现在的 evaluation 还缺什么

第一个缺口是 **performance claim 的单位不清楚**。“最高加速 20%”到底是某个 instruction sequence、某个 BPF program、某个 application phase，还是 end-to-end workload？如果 claim boundary 不显式，一个完全正确的局部结果很容易被读成更大的结论。

第二个缺口是 **profitability coverage**。Correctness 往往可以做成 binary：transformed program 等价或者不等价。Profitability 通常不是。一个 pass 可能让 30 个 program 变快、100 个不变、16 个变慢。一个 geomean 很容易把真正重要的问题藏起来：这 16 个 regression 是冷门 utility，还是恰好吃掉 fleet 大部分 BPF CPU time 的 hot program？

第三个缺口是 **environment generalization**。eBPF 位于一条很长的 compilation pipeline 里，architecture-specific JIT、kernel version、distribution backport、helper implementation、map behavior、CPU microarchitecture 和 application loader 都会影响结果。一个 optimization 可以 semantic portable，却在另一个 JIT 或 CPU 上没有收益。9 月 6 日那篇解决的是 architecture eligibility 和 safe fallback，而不是“需要多少 performance evidence 才能说收益也 portable”。

第四个缺口是 **adaptive-search bias**。普通 compiler research 已经会担心 benchmark selection；自动优化会再多一层风险。Optimizer 可以尝试很多 transformation，反复观察同一个 benchmark，最后过拟合 measurement noise、warm-cache state、某个固定 workload phase，甚至 harness 的漏洞。闭环搜索里，benchmark 本身可能被变成真正的 optimization target。

第五个缺口是 **production promotion**。Paper 做完 evaluation 通常就结束了。但生产环境必须继续回答：明天到底在哪些 machine 和 program 上 enable？这个 decision 需要携带什么证据？出现什么 observation 时应该自动撤销？

## 兼具学术价值和生产价值的方向

### 1. 让每一个 optimization claim 都带一份 evidence envelope

不要只保存一个 speedup。把结果变成结构化 artifact，把 optimization identity 和“这个收益在什么条件下测出来”绑定在一起。

最小记录可以是：

```text
optimization = rotate_v3
semantic_witness = proof-sequence-hash
kernel = 6.x + commit
jit = x86_64 / commit
cpu = family-model-stepping
program = object + program tag
application = katran + revision
workload = replay-manifest + digest
baseline = exact artifact
samples = raw run counters + workload metrics
correctness = pass/fail + oracle identity
performance = distribution + confidence interval
regression_budget = <= 1% on guarded metrics
claim_scope = {this program, this workload family, this target class}
```

和普通 benchmark metadata 最大的区别是最后一项：artifact 明确写出这份结果**允许支持什么 claim**。

Microbenchmark 可以支持“这个 instruction idiom 在这个 target 上更便宜”，但不能自动推出“application 更快”。真实 application replay 可以支持更高层 claim，但也只能覆盖被 replay 到的 workload family 与 target family。

Research artifact 可以是一份 schema 加 comparison engine。只有 claim scope 相容的 envelope 才能被 aggregate。Evaluation 可以拿常见的 optimization summary，观察弱 metadata 下会被接受的结论，有多少在离开隐藏 assumption 后无法 reproduce。可以把这个指标叫做 *claim escape*。

如果普通 benchmark manifest 已经能完整记录所有 materially relevant condition，而且 reviewer 和 deployment system 真的会严格限制 claim scope，那么这层 artifact 就是多余的。实验应该主动尝试证明它不需要存在。

### 2. 给 optimizer 准备 holdout 和 counterexample suite

传统 optimization evaluation 通常问 selected program 有没有变快。Adaptive optimizer 需要更强的测试：它能不能在看得到的 search set 上优化，同时没有学会怎么 exploit 这个测试？

可以把 evidence 分成至少三部分。Development set 对 optimizer 可见。Holdout set 放它没见过的 application、workload phase、kernel 或 architecture。Counterexample set 则故意包含会惩罚常见 shortcut 的 case：code-size growth 应该带来 I-cache 代价的 program、目标 sequence 实际很冷的 workload、和 training profile 相反的 branch distribution、verifier-sensitive program，以及必须通过真实 loader/lifecycle 才算成功的 application。

[`bpf-bench`](https://github.com/eunomia-bpf/bpf-benchmark) 的 integrity model 已经采用类似思路：把 optimizer 当成不可信组件，禁止它缩短 workload、过滤失败 program、绕过真实 loader 或伪造 result file。Research system 可以进一步把这件事量化。

Primary score 不应该只剩“找到的最好 speedup”。至少要报告 acceptance coverage、worst guarded regression、holdout speedup distribution、invalid-result rate，以及 development win 中有多少能在 holdout target 上 reproduce。对于 agentic search，holdout oracle 应该一直冻结到最终 evaluation，避免搜索过程继续对它适应。

Ablation 可以故意把 holdout result 泄露回 optimizer。如果 benchmark score 上升，但独立 reproduction 变差，就直接展示了我们想捕获的 overfitting。

这个方向的 artifact 不是另一个 optimization pass，而是一套可以复用的 optimization benchmark methodology。生产里的对应物也很自然：canary workload 和 canary machine 就是新 optimization policy 的 holdout。

### 3. 用 profitability contract 上线，而不是只有一个 global enable bit

很多 optimization 不需要 everywhere profitable，只需要一个足够可靠的 applicability rule。

可以从 evidence envelope 推导或学习一个小的 profitability predicate：target architecture、JIT capability、program feature、code-size delta、runtime profile stability、application/workload class。只有 predicate match 时才 enable。Runtime 再观察实际效果，一旦 guarded regression budget 被突破，就 disable 或 rollback specialization。

这和 9 月 5 日讨论的 deoptimization 不一样。那篇关心 stale runtime assumption 会不会改变 program semantics；这里 semantics 一直是合法的，predicate 只控制“这个优化值不值得”。Fallback 可以因为 p99 latency 退化 3%、JIT size 超过 I-cache budget，或者 program 的 runtime share 太低，根本不值得增加复杂度而触发。

Evaluation 至少比较三种 policy：全局打开、手写 target allowlist、evidence-driven promotion。跨 kernel version、x86-64 与 ARM64、多个 CPU generation、microbenchmark 和真实 application workload 测试。指标应该包括 fleet-weighted realized speedup、worst regression、eligible execution fraction、decision overhead、rollback frequency，以及 workload/kernel 变化之后 policy 需要多久才能重新收敛。

如果一个简单 static allowlist 已经能达到同样的 realized benefit 和 regression bound，那么 online controller 就没有必要。这个 failure condition 很重要，不是每个 compiler pass 都值得造一个 control plane。

## 一份更实用的 eBPF optimization evaluation contract

把上面的想法合起来，可以得到一个比较清晰的层次。

第一层先验证 **semantic admissibility**：transformation 能不能合法替换原程序。这部分属于 verifier、equivalence、trust 与 stateful operation contract。

第二层测 **local mechanism gain**：用 microbenchmark 确认目标 machine-level effect 的确存在。

第三层测 **program gain**：结合 kernel run counter、JIT size 和真实 BPF program 的 repeated execution，确认 surrounding bytecode 没把收益吃掉。

第四层测 **application gain**：通过真实 loader 和真实 workload，看这个 BPF program 对系统到底重要不重要。

第五层测 **generalization**：在 holdout kernel、architecture 和 workload phase 上测试，决定 claim 到底能覆盖多大范围。

最后一层是 **promotion safety**：applicability rule 会漏过多少 regression、多快能发现、rollback 能不能恢复 baseline。

不是每篇 paper 都需要在每个 experiment 上做到所有层，但 claim 应该停在它真正测到的最高层。Production rollout 更应该这样，因为最终决定收益的是 workload distribution，而不是 benchmark 里最漂亮的那个数字。

## 什么证据会改变这个结论？

如果 local eBPF benchmark win 已经可以非常可靠地预测 production benefit，那么上述复杂 evidence machinery 就没有必要。可以直接用一个大 corpus 检验：在多个 architecture 和 workload 上，isolated instruction/program speedup 与 application throughput、latency、BPF CPU time 的 correlation 是否强且稳定。如果答案是肯定的，evidence envelope 和 holdout gate 可能只是增加流程成本。

如果现有 BPF CI 已经提供一套完整 end-to-end optimization contract，同时覆盖 verifier behavior、exact JIT output、真实 application lifecycle、workload correctness、runtime performance、architecture diversity 和 regression promotion，这个研究空间也会大幅缩小。现在这些组件已经分别存在，但通常还没有被组合成同一个 promotion evidence chain：`veristat` 很擅长 verifier evidence，application benchmark suite 提供 workload evidence，各种 optimization paper 再各自定义 performance matrix。

另外，如果某个 optimization 在所有 supported target 上都拥有压倒性收益，而且几乎没有 downside，就不应该给它再造 profitability control plane。最简单的答案仍然是把 compiler 本身做好。

但对于 architecture-sensitive、profile-sensitive，或者由 adaptive search 找出来的优化，“benchmark 变快了”只能算 evidence 的起点。真正足够上线的结果应该能回答四件事：**这个 claim 在哪里成立、我们主动测过哪些反例、哪些 regression 被约束住，以及现实离开 measured envelope 后，什么 observation 会撤销这个优化。**

## References

- Yusheng Zheng et al., [Kops: Safely Extending the eBPF Compilation Pipeline with Native Operations](https://arxiv.org/abs/2606.24213), 2026.
- Linux BPF CI, [kernel-patches/bpf](https://github.com/kernel-patches/bpf), accessed 2026-09-11.
- libbpf, [`veristat`](https://github.com/libbpf/veristat), accessed 2026-09-11.
- Eunomia, [`bpf-bench`](https://github.com/eunomia-bpf/bpf-benchmark), accessed 2026-09-11.
- sched_ext, [Developer Guide](https://github.com/sched-ext/scx/blob/main/DEVELOPER_GUIDE.md), accessed 2026-09-11.
