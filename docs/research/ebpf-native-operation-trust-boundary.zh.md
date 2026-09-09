---
date: 2026-09-09
title: "eBPF 把工作交给原生代码后，哪些东西必须被信任？"
description: "eBPF verifier 可以证明调用侧安全，但原生操作本身仍属于受信任代码。本文分析这条边界，并提出可绑定版本、效果与证明证据的信任契约。"
tags:
  - Daily Report
  - eBPF
  - JIT
  - Verification
  - Security
  - Compilers
research_question: "eBPF 运行时把已经验证的工作委托给原生操作时，怎样限制并检查新增的受信任代码，而不是让每个优化、helper 或 backend 都变成无法解释的 TCB？"
source_cutoff: 2026-09-09
status: daily-report
---

# eBPF 把工作交给原生代码后，哪些东西必须被信任？

假设一个 XDP 程序里有一段很热的八条指令，用来完成 rotate。专门化 backend 识别出这段模式，最后只生成一条原生 `ROL` 指令。普通 BPF 版本已经通过 verifier，新的 fast path 更短，看起来没有太多争议。

把例子再往前推一步，问题就会变得不同。替换内容不再是一条指令，而是一段会访问内存、依赖 CPU feature、带 compiler-generated prologue，并且可能独立升级的原生函数。verifier 仍然能证明原来的 BPF sequence 安全，但它不会再对这段任意 machine code 做同样的符号执行。此时系统已经把一部分正确性判断移出了 verifier 的边界。

这条边界并不是未来才会出现。BPF helper 与 kfunc 的实现本来就在受信任的内核代码里；每个 JIT backend 也必须把已经验证的 BPF 忠实翻译成机器指令。Kops 进一步把这种取舍显式化：一个 operation 同时带普通 BPF proof sequence 和 native emit。真正值得研究的问题因此不是“能不能用原生代码”，而是：**到底新增了多少必须信任的代码，每一部分需要保持什么性质，以及运行时怎样确认当前真正执行的实现仍然对应当初验证或审查过的那份契约。**

<!-- more -->

本文接着前面的 [运行时特化](https://eunomia.dev/zh/research/ebpf-runtime-profile-specialization/)、[跨架构 fast path 可移植性](https://eunomia.dev/zh/research/ebpf-portable-architecture-specialization/) 与 [动态优化执行 provenance](https://eunomia.dev/zh/research/ebpf-specialization-debug-provenance/) 继续往下问。前几篇分别讨论一个 specialization 是否仍然保持语义、某个 native implementation 是否适合当前机器，以及事故时究竟运行的是哪一代。这里假设 implementation 已经被选中、身份也知道，剩下的问题是：这份 implementation 为什么值得被信任，以及这种信任能不能被限制和检查。

## verifier 证明的是 BPF 程序，不是任意原生实现

Linux verifier 围绕 BPF 指令建立抽象状态，追踪 register、stack、pointer type、control flow，以及 helper 和 kfunc 调用需要满足的约束。它的状态搜索、pruning 和安全判断都建立在 BPF program representation 上，而不是建立在 JIT 后面产生的任意 machine code 上。

这种分工本身有合理性。如果 verifier 还要同时建模每一种 CPU backend、compiler optimization、kernel routine、NIC firmware 与未来 accelerator，维护和信任 verifier 本身都会更困难。因此 Linux 采用流水线式的边界：verifier 先证明 BPF 层面的性质，后面的受信任 runtime component 再负责保持这些性质。

代价是 verifier acceptance 与 native correctness 从来不是同一个命题。JIT 里的一处 bug 完全可能让 verifier-safe 的程序产生错误行为。Jitterbug 把这个问题做得很具体：它建立 BPF JIT correctness specification，并在多个已部署 backend 中找到并修复了 16 个此前未知的问题。这个结果也说明，即使语义关系很清楚，从 formal model 到真正 production implementation 之间仍然需要一条可以检查的桥。

## kfunc 已经体现了“调用者可验证、被调用者受信任”的分工

Linux kfunc 在另一条边界上展示了类似结构。BTF 和 verifier annotation 会描述 BPF 程序允许怎样调用一个 kernel function。`KF_ACQUIRE`、`KF_RELEASE`、`KF_RET_NULL`、`KF_SLEEPABLE`、`KF_RCU` 等规则让 verifier 可以追踪 reference ownership、nullability、sleepability 和 pointer validity。

这些 annotation 的价值在于，它们把一部分 kernel function contract 变成 verifier 可以直接检查的状态。但是 Linux 文档同时明确要求，暴露一个现有 kernel function 给 BPF 时，仍然需要人工判断它在相应调用上下文中是否安全。verifier 可以证明传给 kfunc 的 pointer 符合声明的要求，却不会自动证明每一个 kfunc body 都正确。

所以 kfunc 很适合拿来理解 native optimization：一个好的接口不一定需要从零形式化证明所有 implementation，但它应该尽量缩小“只能靠人相信”的那部分。一个调用背后隐藏的 memory effect、lifetime、同步与 control effect 越多，reviewer 和 operator 真正需要信任的语义表面就越大。

## Kops 把“信任多少代码”变成了优化设计的一部分

Kops 为一个 native operation 同时保留两种形式。proof sequence 由普通 BPF 指令组成，继续通过现有 verifier；native emit 则产生真正用于执行的架构相关机器指令。在 EInsn 中，论文用 Lean 4 证明七种 operation 的 native emit 与 proof sequence 计算相同结果，并报告 microbenchmark 最高 24%、实际应用最高 12% 的加速。

同一套机制也可以替换更大范围的代码。Kops 的 whole-program native replacement 可以达到 2.358x，但论文也明确指出，这会扩大 trusted computing base。这里最有意思的并不是某一个性能数字，而是优化实际上至少有两条轴：一条是速度，另一条是有多少 implementation correctness 不再直接由 verifier-visible program 推导，而需要额外信任。

只替换一条 rotate 指令，与把上千条 machine instruction 当作一个 native replacement，不应该只用同一个 `verified=true` 来描述。它们需要的证据强度和 operation risk budget 明显不同。

## 还缺一个面向 delegated execution 的信任契约

现有机制已经有不少碎片。verifier 有 instruction semantics 与 abstract state；kfunc 有 BTF signature 和 verifier flag；JIT verification 工作定义 BPF 与 generated instruction 的语义等价；Kops 有 proof sequence、native emit，以及针对部分 operation 的形式化证明；前一篇 Daily Report 又可以记录哪一个 generation 真正执行过。

但这些机制还没有形成一个统一答案。对于一次 delegated operation，系统至少应该能回答四件事：

1. **verifier 已经知道什么？** 例如普通 BPF proof sequence、argument type、允许访问的 memory region、reference ownership 或必须为 constant 的输入。
2. **还额外信任什么？** 应该明确到具体 native implementation、architecture/backend、compiler 或 emitter version，以及任何被假设正确的 kernel 或 firmware component。
3. **允许产生哪些 effect？** Register clobber、memory read/write、helper call、sleep、allocation、synchronization、fault 和 control transfer 应该有显式边界，而不是散落在 implementation convention 里。
4. **契约怎样绑定到真正执行的代码？** 对 version A 的 proof 或 review，不能自动成为 version B 的证据。runtime 必须把当前加载的 implementation identity 与当时检查过的 artifact 绑定起来。

缺少这些信息时，我们可以统计 source line、引用一份 proof，甚至知道事故时运行的是哪一个 generation，却仍然回答不了生产环境里最具体的问题：现在正在执行的 implementation，是否真的还是那份曾经被验证或审查过的实现？

## 现有研究还缺什么

第一个缺口是 **trust granularity**。JIT correctness 常常以整个 backend 为对象，kfunc safety 通常逐函数 review，而 native-operation 机制可以从一条机器指令一直扩展到 whole-program replacement。当前缺少一个共同尺度，用来比较有多少语义表面已经移出 BPF verifier。

第二个缺口是 **effect completeness**。只证明 return value 相同，对会访问 memory、改变 synchronization、调用其他 kernel code、产生 fault 或暴露 timing-sensitive effect 的 native operation 并不够。formal model 本身可能完全正确，但如果 contract 没有描述生产环境真正关心的 effect，proof 仍然无法覆盖实际风险。

第三个缺口是 **artifact binding**。source-level proof、code review 或 CI result 都可能在 compiler、kernel、module、firmware 或 backend 更新之后失效。execution provenance 能告诉我们运行了什么，而信任判断还需要知道“被检查的实现”和“真正运行的实现”是不是同一份 artifact 及依赖上下文。

第四个缺口是 **固定 trust budget 下的评估**。优化论文经常比较 latency、throughput 和 code size，却很少问：两个同样能带来 10% 加速的方案，是否一个只多信任二十条机器指令，另一个却多引入了一整个 compiler/runtime component？如果更慢一点的方案能显著缩小、独立检查 TCB，它可能才是生产环境里更合理的选择。

## 兼具学术价值与生产价值的方向

### 1. 与 verifier 相连的 native-operation contract

每个 native operation 可以在 BPF proof sequence 旁边携带一份很小的 contract：输入输出 register type、可访问 memory range、允许的 side effect、clobber、sleepability、failure behavior、architecture requirement，以及 native implementation 的 content digest。verifier 不需要理解全部 machine code，但要检查 BPF-side assumption 与 operation contract 一致；runtime 则只允许 identity 与已批准 entry 一致的 implementation 被激活。

可以做出的 artifact 是一个小型 kernel/runtime interface，加上生成和检查 contract 的 tooling。最强 baseline 是今天“verifier-visible BPF + 人工 review native code”的组合。评估时故意注入 undeclared memory write、额外 helper call、错误 clobber set、过期 implementation digest 等问题，观察系统能否在错误 path 真正执行之前拒绝 activation。Ablation 则逐个移除 contract field，看看哪一种错误因此重新变成不可见。

学术问题是：为了让 delegation 可以组合，而又不要求 verifier 学会所有 backend，需要把多少语义信息跨过 verifier/native boundary？生产价值则很直接，maintainer 可以 review 一份几十行的 effect contract，而不是把整个 backend 当成一个不透明的 trust decision。

### 2. 用 proof-carrying trust tier 代替一个 `verified` bit

不同 native operation 适合的 assurance mechanism 不一样。简单 arithmetic idiom 可能可以做 exhaustive equivalence check 或 Lean proof；较大的 routine 更适合对每一个 emitted artifact 做 translation validation；hardware 或 firmware implementation 也许只能提供 conformance test 加签名 version identity。

runtime 因此可以给每一个 operation generation 带上 trust tier，例如 verifier-only、verifier + differential test、per-artifact translation validation、machine-checked proof。tier 表示真实 evidence，而不是一个高低等级标签。部署策略随后可以要求：effect 更大、trusted code 更多的 operation 必须提供更强 evidence，否则自动 fallback 到普通 BPF。

可以实现一个 validation pipeline 与 policy engine，同时在 latency target 和 assurance budget 下选择 implementation。评估覆盖小型 instruction idiom、中等 native routine 与 whole-program replacement，测 proof/validation latency、trusted code size、performance、escaped semantic fault 与 fallback frequency。如果一种简单而便宜的 validation 已经能抓住同一组问题，那么多层 trust tier 就没有必要。

### 3. 用对抗式 native fault 建立 trust-budget benchmark

需要一个把 correctness failure 当作一等指标的 benchmark。起点是 verifier-safe BPF workload，再提供若干语义上应该等价的 native replacement，并在 native side 注入已知错误：rare input 才出现的 arithmetic bug、未声明 memory write、reference leak、architecture-dependent flag 行为、升级后的 stale code，以及 proof 与 implementation 配错。

比较 stock JIT、helper/kfunc 风格 delegation、proof-linked native operation、verified JIT 与 whole-program native replacement。尽量固定 performance target，再测 escaped fault、bad artifact detection time、trusted source/binary surface、validation cost，以及单位 trusted code 所得到的 speedup。

学术价值是把“TCB 大小”从定性描述变成可以复现实验的 trust/performance trade-off。生产价值则是 release gate：maintainer 能看到一个 fast path 带来 8% throughput 时，到底只多信任了二十条机器指令，还是实际上悄悄接受了一个新的 compiler/runtime component。

## 哪些结果会改变这个判断？

如果现有 verifier、kfunc、JIT 或 native-operation 机制已经提供一套 machine-readable、与版本绑定的 contract，能够完整描述 native effect，并让 operator 跨 implementation 比较 trust surface，那么本文提出的缺口会明显缩小。

另一个反例是，真实故障实验发现普通 regression test 已经可以和额外 contract/validation 一样可靠地抓住 native-operation bug，而新增机制只增加维护成本。更强的反例则是一套足够便宜的 translation validation，可以在生产中对每一个 native artifact 检查其与 verifier-visible BPF semantics 的等价关系，而且覆盖相关 memory 与 side effect。那时受信任边界可以缩到 validator 与 runtime binding mechanism，本身就不再需要复杂的 trust hierarchy。

在出现这些证据之前，性能特化最好把“信任扩大了多少”直接暴露出来。真正有用的判断并不是一个 native fast path 是否“看起来足够安全”，而是系统能否准确说清它信任什么、把这份声明绑定到真正执行的代码，并证明新增的 trusted surface 值得换取对应的性能收益。

## 参考资料

- Yusheng Zheng 等，[Kops: Safely Extending the eBPF Compilation Pipeline with Native Operations](https://arxiv.org/abs/2606.24213)，arXiv:2606.24213v1，2026。
- Linux kernel documentation，[BPF Kernel Functions (kfuncs)](https://docs.kernel.org/bpf/kfuncs.html)。
- Linux kernel documentation，[eBPF verifier](https://docs.kernel.org/bpf/verifier.html)。
- Luke Nelson 等，[Specification and verification in the field: Applying formal methods to BPF just-in-time compilers in the Linux kernel](https://www.usenix.org/conference/osdi20/presentation/nelson)，OSDI 2020。
- Jitterbug artifact，[Verification of BPF JIT compilers](https://github.com/uw-unsat/jitterbug)。
