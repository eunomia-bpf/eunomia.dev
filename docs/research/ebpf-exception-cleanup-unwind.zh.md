---
date: 2026-09-20
slug: ebpf-exception-cleanup-unwind
title: "eBPF 异常展开时，能安全释放资源吗？"
description: "新的 BPF cleanup table 让 bpf_throw 有机会在展开栈帧时运行清理逻辑。本文分析 verifier、JIT 与编译器必须共同满足的资源释放契约。"
tags:
  - Daily Report
  - eBPF
  - Linux
  - Verifier
  - Compilers
  - Rust
research_question: "bpf_throw() 要安全展开持有内核资源的栈帧，需要怎样的证明与运行时契约，才能避免泄漏 verifier 跟踪的状态，并保持不同 JIT 后端的一致性？"
source_cutoff: 2026-09-20
status: daily-report
---

# eBPF 异常展开时，能安全释放资源吗？

设想一个 BPF 函数进入 RCU read-side critical section，并创建一个生命周期上“拥有”这把锁的对象，随后调用另一个 BPF 函数。被调用函数遇到无法继续的错误，于是执行 `bpf_throw()`。

按照普通 Rust 的语义，结果很直接：stack unwind 会运行 `Drop`，而 `Drop` 会释放 guard。今天的 BPF 却不能这样理解。`bpf_throw()` 会丢弃中间的 BPF 栈帧；如果其中某个栈帧仍然持有 lock、reference 或其他由 verifier 跟踪的资源，verifier 会直接拒绝程序，因为运行时没有机会执行原本应该发生的 cleanup。

这个限制目前就写在 `bpf_throw` 的文档里：带着未释放的 lock、reference 等资源抛出异常，会导致 verification error。这不是单纯的语言易用性问题，而是 verifier 在运行时缺少 cleanup path 时，继续维护资源生命周期不变量的结果。

9 月 16 日提交到 `bpf-next` 的一组 20 个 patch 正在改变这个边界。该方案引入由编译器生成的 `.bpf_cleanup` table，让 verifier 从被覆盖的 call site 继续探索 cleanup landing pad，并让 `bpf_throw()` 在遍历调用栈时真正执行这些 landing pad；新的 `bpf_unwind_resume()` kfunc 再把控制权交回 unwinder。LLVM 23 已经合入了生成 cleanup table 的编译器侧支持。

这使 BPF 距离语言级资源管理近了一大步，但也带来一个新的系统问题：**编译器生成的 cleanup、verifier 的资源推理、libbpf 对元数据的搬运，以及不同架构 JIT 的运行时 unwind，怎样才能被证明描述的是同一个资源生命周期 transition？**

<!-- more -->

## 今天的限制本质上是资源生命周期问题，不是异常语法问题

BPF 已经有异常机制。`bpf_throw()` 从 Linux 6.7 开始存在，可以终止当前执行并沿 BPF call stack 展开到 exception boundary。在默认路径下，传入的 cookie 会成为程序返回值；在支持 exception callback 的模式里，也可以由 callback 处理该 cookie。

真正困难的是，被丢弃的栈帧拥有的义务怎么办。

Verifier 并不只是逐条检查指令是否会越界，它还会追踪必须成对结束的资源生命周期。例如，一个操作取得 reference，后续路径就必须把它释放。Linux verifier 文档用 socket reference 展示了同样的原则：程序退出时仍有未释放的 reference，会被拒绝。BPF kfunc 中的 task reference、cgroup reference、RCU read-side section 等也依赖类似的 acquire/release 关系。

因此，异常不能只是“跳到函数结尾”的控制流捷径。假设某一帧的 verifier state 是：

```text
RCU read lock: held
referenced kptr: owned
preemption: disabled
```

如果 `bpf_throw()` 直接把这一帧扔掉，这些义务虽然从程序控制流里消失了，却没有从内核真实状态里消失。没有可执行 cleanup path 时，拒绝这种 throw 才是安全的选择。

Rust 只是把矛盾表现得特别明显，因为 RAII 和 `Drop` 会把资源释放结构化地放进 unwind path。C++ destructor 或其他由编译器生成的 cleanup 机制也会遇到同样的问题。即使是手写 BPF，只要系统希望支持 non-local exit，也必须让 verifier 能看到资源究竟在哪里被释放。

## 9 月的新方案把 cleanup 变成一等控制流路径

这组 patch 把此前没有共享同一 cleanup 模型的四个层次连接起来。

第一，LLVM BPF backend 可以把带 cleanup landing pad 的 `invoke` 降低，并生成 `.bpf_cleanup` section。每条记录是一个 12-byte triple：

```text
(begin, end, landing_pad)
```

`[begin, end)` 表示一个 call-site region。Unwind 穿过某个栈帧时，如果它的 return address 落在这个区间，就应先执行对应 landing pad，再丢弃该帧。LLVM PR #192164 在 2026 年 4 月合入，加入了保留这些 cleanup edge 并生成 table 的 BPF backend 支持。

第二，proposal 中的 libbpf 会收集这些记录，在 `BPF_PROG_LOAD` 时传给内核，把编译器产生的 `_Unwind_Resume` 解析为内核侧 `bpf_unwind_resume()` kfunc，并让 light skeleton 和 static linker 继续携带这份信息。

第三，verifier 会把被 cleanup table 覆盖的 call 看成多了一条可能后继边：cleanup landing pad。它会探索 pad，并按照其中真正执行的 cleanup operation 更新 resource state。以 RCU 为例，verifier 可以确认 `bpf_rcu_read_unlock()` 在继续 unwind 之前确实把 lock 释放掉。

第四，运行时路径与 verifier 模型保持对应。`bpf_throw()` 原本就需要 architecture support 来遍历 BPF stack；新的设计会根据每一帧的 return PC 查询 cleanup table，运行对应 pad，再继续 unwind。首版 patch 提供 x86-64 和 arm64 的 JIT 支持。

这套设计最重要的优点就是这种对称性。Verifier 不是相信 side table 写着“已经 cleanup”，而是实际分析 cleanup code；运行时也没有创造第二套 unwind language，而是执行 verifier 看过的同一个 landing pad。

## 这仍然不是通用的 C++ 异常机制

如果只看到 `.bpf_cleanup`，很容易把它理解成“BPF 要支持完整 C++ exception 了”。这个说法太大。

当前 patch series 支持的是 cleanup pad，不是 catch pad。Cleanup pad 的目标是释放资源，然后继续 unwind；它不能把已经被丢弃的帧重新交还给普通执行。Proposal 还要求目标 JIT 本身能够 dispatch cleanup pad。第一版支持 x86-64 和 arm64；x86-64 依赖现有 `bpf_throw()` 路径所需的 ORC unwinding。其他还没实现该 ABI 的 JIT 会在加载时返回 `-EOPNOTSUPP`，而不是默默跳过 cleanup。

方案还会拒绝运行时无法安全处理的组合和 shape，例如 offloaded program、private stack、cleanup table 与 exception callback 同时使用，以及 landing pad 内的 tail call、indirect jump、on-stack call arguments 等操作。

更重要的是，Rust toolchain 目前还不是一条完整可生产使用的 BPF exception path。Patch series 的 selftest 会手写与 frontend 输出等价的 cleanup record 和 landing pad，以便先验证 kernel 与 libbpf 的端到端行为，而不是宣称 Rust 集成已经完成。

这些限制并不是坏事。它们把初始 claim 控制在一个可以验证的范围里：**受 cleanup table 覆盖的 BPF call，在支持的 JIT 上可以通过 verifier 检查过的 cleanup pad 继续 unwind。** 这并不等于支持任意语言异常、portable catch semantics，也不等于所有 cleanup 自动安全。

## 尚未解决的缺口是跨层 cleanup correctness

新方案解决了最明显的资源泄漏问题，但也建立了一条横跨 compiler、ELF metadata、libbpf、verifier、JIT、architecture stack walker 与 runtime exception path 的 correctness boundary。

普通 BPF branch 的行为主要由指令和 verifier state 决定。Cleanup unwind 则还依赖 compiler 生成的 side metadata；这份 metadata 会经过 userspace tooling，再由 kernel 消费。到了运行时，architecture-specific JIT 与 stack walker 还必须把 machine return PC 对回 verifier 当初认为属于某个 landing pad 的同一个逻辑 region。

因此，一个正确的 unwind 至少要求下面两组关系同时成立：

```text
compiler region
    == libbpf-carried region
    == verifier-covered call region
    == JIT/runtime PC region

以及

verifier 看到的 cleanup resource transition
    == runtime 实际执行的 cleanup resource transition
```

当前 proposal 正是围绕这种一致性设计的，但现在还缺少一种通用的 deployment/test abstraction，可以把这组一致性明确记录下来，并在 toolchain version 或 architecture 变化时主动找反例。

这和之前讨论的 [内核 capability admission](https://eunomia.dev/zh/research/ebpf-kernel-capability-evidence/) 不一样。Capability admission 问的是一个 artifact 在一台目标机上能不能使用某个能力。它也不同于 [跨内核 semantic compatibility](https://eunomia.dev/zh/research/ebpf-kernel-upgrade-semantic-compatibility/)，后者问的是已经成功加载的应用在 kernel upgrade 前后是否还保持应用层行为。这里的边界更窄、更底层：一个 non-local control-flow operation 跨过 compiler metadata 与 JIT runtime machinery 时，能不能保持 verifier 对资源生命周期做出的证明。

它也与 [native eBPF operation 的 trust boundary](https://eunomia.dev/zh/research/ebpf-native-operation-trust-boundary/) 有联系。Landing pad 自己仍是 verifier 检查过的 BPF code，但正确 dispatch 依赖受信任的 native unwind machinery。因此值得研究的不是再设计一套 exception syntax，而是怎样把这个跨层 transition 做成可审计、可证伪的 contract。

## 研究方向一：给 cleanup pad 明确的 resource-effect discipline

第一个方向，是把 landing pad 应该做什么表达成 verifier property，而不是只靠一组不断增加的 syntactic restriction。

**缺口。** Runtime 需要 cleanup pad 消除当前 frame 已经拥有的义务。如果 pad 在 unwind 中又获得新的 long-lived resource、制造新的 ownership cycle，或者把义务转移到 unwinder 无法表达的位置，整体语义会迅速变得难以推理。首版 patch 已经限制了一些危险控制流 shape，但更深层的不变量其实是 resource effect。

**机制。** 在 verifier state 上定义 cleanup-effect discipline。对每个 landing pad 比较进入与调用 `bpf_unwind_resume()` 前的资源集合。一个保守的初版规则可以要求 pad 对 unwind obligation 保持单调：它可以释放该 frame 原来已经拥有的资源，也可以执行在 pad 内完全配对的临时 acquire/release，但不能在 resume 时留下新的 live reference、lock、preemption-disable state、iterator 或其他被跟踪的义务。

概念上可以写成：

```text
owned_at_resume ⊆ owned_at_pad_entry
```

临时 acquire/release 则要求在 resume 前闭合。

第一版甚至不需要新的 source-language annotation，因为 verifier 在探索 landing pad 时已经有 resource state，可以直接推导这条性质。更进一步，verifier 可以用 effect 语言给出错误，例如“landing pad 新增了 reference id N”，而不是只说某种 unwind shape 不支持。

**相对现状的增量。** 当前 patch 通过已有 verifier machinery 验证具体 path。这里建议的是把所有合法 cleanup pad 应满足的共同不变量显式命名出来，并在以后有机会用一个 resource-state rule 替代部分零散限制。

**原型。** 在 cleanup selftest 上扩展一组不同资源：RCU lock、preemption-disable section、referenced kptr/task object，以及适用时的 iterator-like lifetime state。生成正确释放、double release、新 acquire、通过 map/kptr 转移 ownership、条件分支后残留 obligation 等不同 pad。

**评测。** 把 verifier acceptance 与人工标注的预期集合比较，并 fuzz 嵌套资源组合和不同 cleanup 顺序。再比较 effect-based rule 与首版 syntactic restriction：目标是减少无必要拒绝，同时绝不接受在 `bpf_unwind_resume()` 时仍残留 unmatched obligation 的程序。

**学术价值。** Non-local BPF control flow 可以被表述成明确的 resource logic，而不是越来越多的 exception special case。

**生产价值。** Compiler 与 language-runtime 作者会得到一个稳定目标：生成的 `Drop`/RAII cleanup 到底允许产生哪些 effect，并得到更接近资源义务本身的诊断。

**失败条件。** 如果现有 verifier path exploration 已经以同样精确的方式自动保证这条不变量，且 diagnostics 足够清楚、限制也没有继续扩张，那么再增加 cleanup-effect abstraction 只会制造复杂度。

## 研究方向二：把 cleanup metadata 绑定到可审计的 artifact contract

第二个方向不是让内核更相信 compiler，而是让跨层 provenance 与 diagnosis 更清楚。

**缺口。** `.bpf_cleanup` 来自 compiler，可能经过 static linker 或 light-skeleton generation，随后被转换成 kernel load metadata。Kernel 最终仍然必须验证实际代码，但失败时很难快速回答：compiler 描述的是哪个 region？linker 有没有改变 offset？libbpf 携带的是不是 stale table？目标 kernel 又是因为哪一个 architecture-specific 限制拒绝它？

**机制。** 随加载 artifact 产生一个 normalized cleanup descriptor。这份 descriptor 是 evidence，不是 authority。它可以记录 object digest、compiler/backend identity、subprogram 与 call-site region ID、link 前后的 landing-pad offset、verifier 实际观察到的 cleanup outcome，以及目标 JIT 用于 admission 的能力。

例如 loader 可以导出：

```text
artifact: sha256:...
compiler: llvm-bpf ...
cleanup_region: subprog=foo callsite=3
object_range: [0x..., 0x...)
landing_pad: 0x...
verifier_result: accepted
resource_delta: rcu_lock 1 -> 0
jit_dispatch: x86_64/orc
```

这绝不能允许 compiler metadata 绕过 verifier。目的只是把 compiler intention、userspace transformation、kernel interpretation 与 target runtime capability 绑定到一个可复现 artifact 上。

**相对现状的增量。** Proposal 中的 libbpf 负责正确搬运 table；这个方向要求额外记录 table 在各阶段怎样变化，以及 kernel 最终如何理解它。它把 provenance 从“feature 存在”推进到“这个具体 cleanup region 以这个 resource transition 被接受”。

**原型。** 给实验性 libbpf loader 增加 debug/export mode，加载前输出 normalized cleanup record，再结合加载后的 verifier log 或结构化 verifier identifier。覆盖普通 ELF、static linking 和 light skeleton 三条路径。

**评测。** 用多个 LLVM revision、optimization level 构建同一批 cleanup 程序，再经过不同 libbpf/linker generation。每次故意让其中一个阶段出现 stale/corrupt metadata，比较 descriptor 与“只看 verifier log + `llvm-objdump`”两种方法定位 mismatch 的时间和准确率。

**学术价值。** 可以检验 proof-relevant compiler metadata 能否跨多阶段 systems toolchain 保持可审计，同时又不退化成“compiler 自己声明证明成立”。

**生产价值。** Deployment 失败时，可以区分 compiler generation、object transformation、kernel admission 和 JIT support，而不是全部归类为“exception cleanup unsupported”。

**失败条件。** 如果现有 verifier/libbpf diagnostics 已经能把几乎所有失败直接定位，或者 descriptor 在正常 toolchain upgrade 中变化过快、无法稳定解释，那么这个 provenance layer 没有维护价值。

## 研究方向三：把 verifier 与 JIT unwind 当成一个机制做 differential test

第三个方向针对最危险的错误：verifier 与 runtime 对同一 cleanup path 理解不同。

**缺口。** Verifier 因为探索 landing pad 并更新 resource state 而接受程序。Runtime 安全则依赖 architecture JIT 与 stack walker 在同一 frame、同一 call-site region 上 dispatch 对应的 pad。PC range 的 off-by-one、特殊 prologue、嵌套 unwind 或 toolchain transformation，都可能把“verifier 认为一定 cleanup”变成“runtime 实际跳过或跑错 cleanup”。

**机制。** 构建 unwind conformance harness，在一组自动生成的程序里，对每个可 throw call boundary 注入 `bpf_throw()`。每个测试同时记录 verifier 预期的 resource transition，以及 runtime 真正执行过哪些 landing pad、顺序是什么。最终比较 resource state，而不只是 cookie 返回值。

测试矩阵至少变化：

- nested call depth 与一个 subprogram 中多个 cleanup region；
- 单个和多个同时持有的资源；
- conditional cleanup path；
- 每个 `[begin, end)` 区间两侧的 boundary PC；
- static-link 与 light-skeleton transformation；
- x86-64、arm64，以及以后实现同一 ABI 的其他 JIT；
- 必须稳定 load-fail 的 unsupported combination。

在 kernel test VM 里还可以加入 resource-specific invariant，例如 throw 返回后直接确认 RCU lock 或 reference 已真正退休。

**相对普通 selftest 的增量。** 当前 patch series 已经带了较完整的 end-to-end 与 rejection selftest。这里把 cross-architecture differential behavior 与自动生成的 boundary coverage 作为主要评测目标，并主动用 verifier 接受的 resource transition 去攻击 runtime implementation。

**原型。** 从 series 现有的手写 cleanup 程序开始，自动生成 region permutation 与 nested ownership pattern，在 x86-64 和 arm64 VM 上跑同一 corpus，记录 verifier result、cleanup execution trace、最终 resource state 和 program result。

**评测。** 人为注入 cleanup range lookup、JIT return-PC normalization、pad dispatch 或 resume handling 的错误，测检测率与 false positive。之后在未修改的实现上跨 kernel/JIT revision 运行，观察它是否能发现普通 positive test 没覆盖的 architecture-specific regression。

**学术价值。** 这是一个很具体的 differential-testing 问题：静态安全模型与负责实现该模型的 native runtime machinery 是否真正一致。

**生产价值。** Distribution kernel 或新 architecture 可以用可复现 conformance result 决定是否开启 exception cleanup，而不是把“能编译”当成“unwind 行为等价”。

**失败条件。** 如果现有 BPF selftest 已经提供等价的 cross-JIT boundary coverage，而且注入 verifier/runtime mismatch 后总能被现有测试捕获，那么单独的 differential harness 就是重复建设。

## 今天怎样使用这项能力

目前应该把它当作正在开发中的 kernel/toolchain feature，而不是 portable production contract。

对生产 BPF 程序，frame 持有 verifier-tracked resource 时，显式 cleanup + error-return path 仍然是更稳妥的默认方案。现有 kernel 会有意拒绝仍有 live obligation 的 `bpf_throw()`。

如果要实验 cleanup unwind，应固定 exact kernel series、LLVM revision、libbpf revision、architecture 和 JIT config，并测试最终真正加载的 artifact，包括 static-link/light-skeleton 路径。至少要有一条针对真实资源状态的 runtime witness，而不能只验证“program load 成功”或 cookie 值正确。

Language runtime 也应该 fail closed。如果目标机不支持 cleanup-table ABI 或必要的 JIT dispatch，就应该拒绝 unwind-capable artifact，或者选择另一条已经验证过的 error/abort implementation，而不能默认 RAII 的 `Drop` 一定会运行。

最终适合生产使用的 boundary 应该很容易说清楚：**只有当 kernel 验证了 cleanup path，并且 runtime/JIT 被证明会为同一个 call-site region 执行同一条 path 时，目标环境才能启用 unwind cleanup。**

## 什么证据会改变这个结论？

有三类结果会降低额外 cleanup contract 与 conformance machinery 的必要性。

第一，如果最终 kernel implementation 能证明普通 verifier path exploration 已经自然给出完整、清晰的 landing-pad resource-effect invariant，而且不会出现不断增长的 special-case restriction 或模糊 diagnostics，那么没有必要再单独引入 cleanup-effect discipline。

第二，如果 compiler、linker、libbpf 与 kernel 已经通过一种 mechanically checked representation 保证 cleanup metadata 在各阶段严格一致，并且 mismatch 能被现有工具精确解释，那么额外的 artifact-level provenance descriptor 收益会很小。

第三，如果大规模 x86-64 与 arm64 fault-injection 表明 cleanup-region dispatch 与 JIT code generation 的绑定足够机械，verifier/runtime divergence 在实践上几乎无法发生，那么单独做 differential promotion gate 可能过重。

目前的证据还没有达到这些更强的结论。Patch series 刚提交并仍在 review，Rust BPF exception path 也还不是完整生产 toolchain，而且这项能力跨越了多个长期独立演化的层次。

## 结论

BPF exception cleanup 最有意思的地方，并不是 `bpf_throw()` 对 Rust 变得更友好，而是 non-local control flow 开始真正接入 verifier 原本用于保证安全的资源记账体系。

9 月的新方案基本形状是合理的：compiler 标出 cleanup region，libbpf 携带它，verifier 探索 landing pad，JIT/runtime 在 unwind 时执行同一个 pad。这样一来，“持有 RCU guard 时 throw”这种今天无法通过 verifier 的程序，才有机会变成 cleanup 既被静态验证、又在运行时真的发生的程序。

下一步值得解决的是如何让这种跨层一致性可证伪。Cleanup pad 应该有明确 resource effect，metadata 应该能在 toolchain 中被审计，而 verifier 认可的 cleanup 还应该与真正执行它的 architecture runtime 做 differential test。只有这些关系能持续成立，BPF 才可能获得 RAII 风格的 unwind cleanup，同时不放弃最基本的安全规则：每一个由程序拥有的内核资源，都必须具有 verifier 可见并且运行时真实执行的生命周期。

## Sources

- Yonghong Song, `[PATCH bpf-next 00/20] bpf: Run exception cleanup landing pads when bpf_throw() unwinds`, September 16, 2026: https://lwn.net/Articles/1095028/
- LLVM PR #192164, `[BPF] Add exception handling support with .bpf_cleanup section`, merged April 16, 2026: https://github.com/llvm/llvm-project/pull/192164
- eBPF Docs, `bpf_throw`: https://docs.ebpf.io/linux/kfuncs/bpf_throw/
- Linux kernel documentation, eBPF verifier: https://docs.kernel.org/bpf/verifier.html
- Linux kernel documentation, BPF kfuncs: https://docs.kernel.org/bpf/kfuncs.html
