---
date: 2026-09-07
title: "eBPF 动态优化后，怎么知道事故时真正跑的是哪份代码？"
description: "eBPF 动态特化会在运行时更换 bytecode 与 JIT image。本文分析现有 bpftool/BTF 能看到什么，并提出把优化决策、执行 generation 和机器码样本连起来的 provenance contract。"
tags:
  - Daily Report
  - eBPF
  - JIT
  - Debugging
  - Observability
  - Compilers
research_question: "动态特化 eBPF 的运行时需要保留哪些 provenance，才能在事故后还原当时真正执行的 BPF generation、优化假设与 native JIT image？"
source_cutoff: 2026-09-07
status: daily-report
---

# eBPF 动态优化后，怎么知道事故时真正跑的是哪份代码？

假设生产里的一个 XDP 程序在一下午被重新特化了好几次。用户态 optimizer 先看到某个 branch 长时间高度偏向，于是生成一份更快的 BPF variant，照常通过 verifier 和 JIT；之后 workload 改变，又触发 deoptimization，退回另一份代码。14:07 出现了 30 秒延迟尖峰。等 operator 打开 `bpftool` 时，出问题的那个 generation 已经被替换掉了。

源文件还在。当前加载的 BPF 程序也能 dump。incident trace 也没有丢。但这三件事放在一起，仍然不能自动证明 14:07 那批 packet 到底经过哪一份 rewritten BPF、哪一份 native JIT image，以及 runtime 当时为什么选中了它。

Linux 现在的 BPF introspection 已经很强：program ID、tag、translated instructions、JIT machine code、BTF line info、runtime statistics 和 load metadata 都可以拿到。问题出现在 optimization 变成动态过程之后：**debugging 需要一条 execution-time provenance chain，把一次 observation 绑定到准确的 specialization generation，再一路连回生成它的 assumption、transform，以及当时真正 active 的 native image。**

<!-- more -->

这个问题和上一篇 [runtime profile-guided eBPF specialization](https://eunomia.dev/zh/research/ebpf-runtime-profile-specialization/) 不一样。上一篇讨论的是 rewrite 在什么条件下还能被认为和原程序语义一致，并提出 generation-scoped 的 equivalence 与 assumption evidence。这里先假设 optimizer 已经生成了一个合法 generation。现在问的是更偏运维和取证的问题：后面又经历了 re-JIT 和 deoptimization 之后，postmortem 还能不能把某个 sample、packet 或 latency interval 准确接回当时那个 generation？

它也不同于 [architecture-specific specialization portability](https://eunomia.dev/zh/research/ebpf-portable-architecture-specialization/)。Portability 问的是某个 native fast path 在这台机器上能不能用、不可用时有没有 portable fallback；provenance 问的是在所有可选实现里，某一个时刻到底真的跑了哪个，以及我们拿什么证明。

## Linux 已经能看到 loaded BPF 与 JIT image

当前 BPF userspace API 其实已经暴露了很多 debugger 需要的基础信息。

内核 [`struct bpf_prog_info`](https://github.com/torvalds/linux/blob/master/include/uapi/linux/bpf.h) 包含 program ID 和 tag、translated program 的长度与内容、JITed program 的长度与内容、BTF identity、function info、source line info、JITed line info、map IDs、load time，以及可选的 runtime statistics。[BTF 文档](https://kernel.org/doc/html/next/bpf/btf.html) 也明确说明，introspection tool 可以拿到 `bpf_prog_info`、BTF、translated bytecode 和 JIT line information，再把 source line 与 BPF/JIT code 对起来。

`bpftool` 把这些接口变成了直接可用的操作。当前 [`prog dump`](https://manpages.debian.org/trixie-backports/bpftool/bpftool-prog.8) 可以 dump translated BPF、dump JITed host image、输出 raw opcode，并在有 line info 时显示源码位置。`prog show` 则能看到 ID、tag、load time、xlated/JIT size、maps、持有 FD 的 process，以及启用统计后的 run count/runtime。

对于静态部署，这已经是很好的 baseline。很多时候我们能回答：现在加载了什么程序、verifier/JIT 最终产生了什么、这条 instruction 大致对应哪一行源码。

Linux 的 BPF tag 也提供了一个很实用的 content-derived identity。当前 [`kernel/bpf/core.c`](https://github.com/torvalds/linux/blob/master/kernel/bpf/core.c) 里的 `bpf_prog_calc_tag()` 会对 BPF instruction stream 做 hash，并先把不稳定的 map FD 字段归一化；内核给 JITed BPF symbol 命名时也会使用这个 tag。相比只看一个 process-local FD，它更适合成为工具之间的 join key。

但 tag 回答的是“这份 BPF bytecode 是谁”，不是“它为什么出现、什么时候真正跑过”。

## Dynamic re-JIT 把 debugging 变成 version-history 问题

公开的 [BpfReJIT 设计](https://lpc.events/event/20/contributions/2445/) 让这个问题变得很具体。它的 userspace shim 可以拦截 load/attach，根据 configuration、workload behavior 与 kernel version 重写 bytecode，然后把每个 candidate 再送回原来的 verifier 和 JIT。设计里明确包含 runtime speculative optimization。

一旦 runtime 能这么做，同一个逻辑应用就可能形成这样的序列：

```text
source object
   |
   +-- generation 41: portable BPF -> JIT image A
   |
   +-- generation 42: branch-specialized BPF -> JIT image B
   |
   +-- generation 43: deoptimized BPF -> JIT image C
```

三个 generation 都可以 verifier-safe，也都可以有合法的 BPF tag，甚至都带着不错的 BTF line info。真正的 debugging gap 是：这些信息通常绑定在一个个 loaded object 上，而 incident 绑定的是**时间与执行**。

所以 operator 实际上要回答四个不同问题：

1. observation 发生时，这个 hook/link active 的到底是哪一个 generation？
2. 哪个 source artifact 和哪串 transformation 生成了它？
3. 哪个 profile、configuration、capability 或其他 assumption 让 optimizer 选择了它？
4. 在这台机器上，哪个 JIT backend 和哪份 native image 最终执行了？

现有 kernel introspection 已经提供了这条链上的很多零件，但还没有定义一个 durable record，把四件事一次连起来。

## BTF line info 是 source mapping，不是 optimization provenance

BTF 让 BPF debugging 好用很多，因为 translated 与 JITed instruction 都可以保留 source line 信息。对于普通 compiler pipeline，看到 instruction 后能够跳回源码，很多时候已经足够。

动态 optimizer 会多出一层 mapping：

```text
source line
   -> original BPF instruction
      -> rewritten BPF instruction
         -> JIT instruction range
            -> observed sample/event
```

如果一次 rewrite clone 了 block、fold 了 condition、插入 guard、把一串 instruction 换成 native operation，或者之后发生 deoptimization，同一行源码会在不同时间对应多个 generated instruction range。保留原始 line number 当然有用，但它并不能说明是哪一个 transform 造出了这条 instruction，也不能说明那个 transform 当时依赖了什么 assumption。

这和 compiler debugging 里 debug location 与 optimization history 的区别很接近。location 告诉你代码从哪里来；provenance 告诉你中间发生过什么。

对 BPF 来说这个区别更明显，因为 verifier 与 JIT 本来就是两个独立阶段。一次完整 postmortem 可能需要区分：

- 原始 application BPF artifact；
- userspace optimizer 生成的 rewritten BPF；
- verifier 对这份 rewritten artifact 的 acceptance；
- 当时实际使用的 kernel/JIT version；
- 可选的 architecture-specific fast path；
- 真正 attach 并接收 execution 的 generation；
- 这个 generation 在什么时候被 retire 或 replace。

## 现有研究还缺什么

### Loaded-object identity 不是 durable incident identity

BTF 文档说明，一个 loaded BPF program 的 ID 在它的 lifetime 内保持唯一。这个 object-lifetime 语义本身没有问题，但 incident archive 往往比 loaded object 活得更久。程序可能在 postmortem 开始之前就已经被 replace、release。

BPF tag 因为是 content-derived，比 ID 更耐用；可它仍然只标识 BPF instruction，而不是完整 optimization decision。两次 deployment 完全可以使用相同 rewritten BPF bytes，却来自不同 optimizer version、profile evidence、kernel build 或 native lowering。反过来，两份语义等价的 generation 也可能因为无害 rewrite 而得到不同 tag。

缺少的是 durable execution-generation identity：它的含义要覆盖足够的 transform 与 deployment context，才能把当时的 code path 复现出来。

### Current dump 能告诉你代码长什么样，却不能解释为什么选中了它

`bpftool prog dump xlated` 和 `bpftool prog dump jited` 是很强的 forensic primitive。只要对象还在，它们可以直接揭示 translated instruction 和 native instruction。但它们不会记录 optimizer 当时用于 selection 的 predicate。

对于 profile-guided optimizer，真正有用的解释可能是：`branch 17 被特化，是因为 profile epoch P 观察到 99.8% bias，并且 guard G 当时仍有效`。对于 architecture-specific operation，则可能是：`native implementation N 被选中，因为 capability set C 匹配，同时 proof witness W 被接受`。

没有这层 decision record，看到 machine code 可以证明 **what ran**，却解释不了 **why this variant existed**。

### Generation 快速切换后，sample attribution 会变得含糊

Profiler 观察的是随时间发生的 execution；re-JIT runtime 改变的是随时间发生的 code。两条 timeline 必须能对齐。

如果 trace 只存 source symbol 或 logical program name，不同 generation 的 sample 会被折叠到一起；如果只存 transient program ID 或 address，对象释放后又可能无法 symbolization；如果只存 BPF tag，却没有 activation interval 和 transform metadata，operator 虽然知道 bytecode 是哪一份，还是不知道它为什么被生成。

所以动态 specialization 需要一种低开销方式，把 execution evidence 绑定到 generation，而不是给每个 sample 都复制一整份 manifest。

## 值得继续做、同时有论文和生产价值的方向

### 1. 每个 specialization generation 都生成一份 execution receipt

第一个 artifact 可以是一份 append-only、content-addressed 的 generation receipt。它主要放在 userspace，不需要为了 provenance 把 hot-path kernel ABI 变得很重。

一个最小版本可以包含：

```text
generation_id
parent_generation_id
source_build_id
original_bpf_tag
specialized_bpf_tag
optimizer_build_id
transform_ids[]
assumption_ids[]
verifier_result_digest
kernel_build_id
jit_backend
native_image_digest
line_mapping_digest
activated_at
retired_at
retirement_reason
```

大的 verifier log、profile、proof object、JIT image 可以继续存在外部 artifact store，只在 receipt 里保留 digest。真正重要的是让整条链 immutable 且可 join。

这里最值得研究的是 identity。`generation_id` 不应该只是某个 process 里会复用的 counter。它至少要稳定到足以让另一台机器收集的 trace 在事故后还能 join 到 optimizer 与 deployment evidence。一个原型可以把 parent receipt、specialized BPF bytes、optimizer version、assumption set 与 target context 一起做 content hash。

这并不是重复 9 月 5 日报告里的 equivalence certificate。Certificate 主要说明这个 generation 为什么在语义上可接受；execution receipt 要进一步保留 deployment evidence，证明**真正被 activate 和 observed 的就是这一代**。

### 2. 用 generation interval map 给 sample 做归属

第二个 artifact 应该把 runtime observation 接到 receipt 上，同时避免每个 event 都携带大块 metadata。

一种设计是在 activation 时维护 interval map：

```text
[time_start, time_end, hook/link identity, JIT address range]
    -> generation_id
```

新 generation active 时，runtime 记录旧 generation 的 retirement boundary，再记录新 generation 的 JIT address range 与 stable generation ID。Perf sample、BPF-side counter、packet trace 或 application event 只需要留下足够的 identity 和时间信息，就能在离线阶段 join 回这张表。

更 kernel-integrated 的 prototype 可以在现有 BPF program metadata 或 JIT symbol event 周围暴露一个短 generation cookie；userspace-first 的版本则可以先订阅 load/attach/replace，并在外部保留已经 retire 的 mapping。真正的问题是：要做到 race-free 的 activation boundary，到底需要多少 kernel participation？

最值得拿来攻击系统的是这些边界：link replacement 时仍在 CPU 上结束的旧 invocation、tail call 跨 program、freplace/fentry 关系、per-CPU execution、快速 deoptimization，以及 JIT image free 之后的 address reuse。如果 collector 丢失 handoff，表示里必须保留明确的 `unknown`，而不是默认把 sample 归给最新 generation。

### 3. 用 adversarial re-JIT 做真正的 forensic benchmark

第三个 artifact 可以直接把 ground truth 写进 workload，再让 debugger 事后找答案。

从同一份 BPF source 出发，连续生成几十到几百个 generation。在某一个特定 assumption epoch 下，让某个 transformation 引入一个已知的 performance anomaly 或 observable behavior。随后继续 churn 多轮 specialization，并在分析开始前把出问题的 generation unload 掉。

至少比较四种 baseline：

- 当前 `bpftool` program metadata、translated/JIT dump 与 BTF line info；
- Linux 正常 profiler/JIT symbol record；
- 完整 optimizer log 加每代 snapshot；
- execution receipt 加 generation-aware sample attribution。

主要指标不应该只有 storage overhead，还要测：

- exact-generation attribution precision / recall；
- evidence 丢失时，正确保留 `unknown` 的比例；
- source -> rewrite -> verifier -> JIT -> execution 链能否完整重建；
- 能否指出真正引入 anomaly 的 optimization decision；
- 在原 kernel/JIT 与另一版本上的 replay success；
- 每 generation 的保留字节数，以及 activation/sample 的 runtime overhead；
- 一个没有亲眼看到 optimization 发生的 operator，需要多久才能找到 root cause。

好的系统应该让 stale 或 missing provenance 显式暴露。一个自信地把 sample 归错 generation 的 debugger，比直接告诉你“证据不完整”更危险。

## 哪些结果会改变这个判断？

有三种结果会明显削弱单独做 specialization-provenance layer 的必要性。

第一，现有 Linux metadata、BPF tag、BTF/JIT line info 与常规 profiler load record 也许已经足以在 unload、replace 之后准确还原历史 generation，而且连 optimizer decision 都能无歧义重建。如果现实的 re-JIT benchmark 证明不加任何 metadata 也能做到，就没有必要再造 receipt format。

第二，dynamic BPF specialization 也许长期都很少见，以至于每次 rewrite 直接保存完整 bytecode/JIT snapshot 和 optimizer log 就足够便宜。如果 full snapshot 在 storage、同步和 retention 上都没有问题，精细的 provenance graph 只会增加复杂度。

第三，execution boundary 可能根本不值得 always-on 精确观测。如果 race-free generation attribution 需要给每个 event 增加 metadata 或同步，并明显吃掉 optimization 带来的收益，那么更合理的答案可能是 coarse epoch logging 加 targeted debug mode，而不是永久开启细粒度 attribution。

现在的证据说明，这个问题还没有大规模出现，但机制已经值得提前定义。Linux 已经能给出 bytecode、JIT image、tag、BTF line mapping 和 runtime metadata；BpfReJIT 也把“运行时反复重写 BPF”变成了具体设计，而不是假想 compiler pass。**真正缺少的不是另一个 disassembler，而是一个 durable join key，把 incident observation 接到当时真正产生并执行那份代码的 optimization generation。**

## 参考资料

- Linux kernel UAPI. [`struct bpf_prog_info` in `include/uapi/linux/bpf.h`](https://github.com/torvalds/linux/blob/master/include/uapi/linux/bpf.h)，访问于 2026-09-07。
- Linux kernel documentation. [BPF Type Format (BTF)](https://kernel.org/doc/html/next/bpf/btf.html)，访问于 2026-09-07。
- Linux bpftool documentation. [`bpftool-prog(8)`](https://manpages.debian.org/trixie-backports/bpftool/bpftool-prog.8)，访问于 2026-09-07。
- Linux kernel source. [`kernel/bpf/core.c`](https://github.com/torvalds/linux/blob/master/kernel/bpf/core.c)，访问于 2026-09-07。
- Linux Plumbers Conference 2026. [kops and rejit: Safely Optimizing eBPF for Hardware and Workloads](https://lpc.events/event/20/contributions/2445/)，访问于 2026-09-07。
