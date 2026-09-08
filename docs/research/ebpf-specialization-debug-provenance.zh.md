---
date: 2026-09-07
title: "eBPF 动态优化后，怎么知道事故时真正跑的是哪份代码？"
description: "eBPF 动态特化会在运行时更换字节码与 JIT 机器码。本文分析 bpftool/BTF 已能提供的证据，并提出把优化决策、执行代次和机器码样本连起来的来源记录。"
tags:
  - Daily Report
  - eBPF
  - JIT
  - Debugging
  - Observability
  - Compilers
research_question: "动态特化 eBPF 的运行时需要保留哪些来源与执行记录，才能在事故后还原真正执行的 BPF 代次、优化假设与 JIT 机器码？"
source_cutoff: 2026-09-07
status: daily-report
---

# eBPF 动态优化后，怎么知道事故时真正跑的是哪份代码？

假设生产里的一个 XDP 程序在一下午被重新特化了好几次。用户态优化器先观察到某个分支长期高度偏向，于是生成一份更快的 BPF 变体，照常通过 verifier 和 JIT；之后负载改变，又触发去优化，退回另一份代码。14:07 出现了 30 秒延迟尖峰。等运维人员打开 `bpftool` 时，出问题的那一代程序已经被替换掉了。

源文件还在，当前加载的 BPF 程序也能导出，事故期间的 trace 也没有丢。但这三类证据放在一起，仍然不能自动证明 14:07 那批数据包到底经过哪一份重写后的 BPF、哪一份 JIT 机器码，以及运行时当时为什么选中了它。

Linux 现在的 BPF 自省能力已经很强：程序 ID、tag、翻译后的指令、JIT 机器码、BTF 行号信息、运行统计和加载元数据都可以取得。问题出现在优化变成动态过程之后：**调试需要一条面向执行时刻的来源链，把一次观测绑定到准确的特化代次，再一路连回生成它的假设、变换，以及当时真正生效的机器码。**

<!-- more -->

这个问题和上一篇 [运行时画像驱动的 eBPF 特化](https://eunomia.dev/zh/research/ebpf-runtime-profile-specialization/) 不一样。上一篇讨论重写在什么条件下仍能被视为和原程序语义一致，并提出按代次记录的等价性与假设证据。这里先假设优化器已经生成了一个合法代次。现在问的是更偏运维和取证的问题：后面又经历多次 re-JIT 和去优化之后，事故复盘还能不能把某个 sample、数据包或延迟区间准确接回当时那一代程序？

它也不同于 [面向特定架构的 eBPF 特化可移植性](https://eunomia.dev/zh/research/ebpf-portable-architecture-specialization/)。可移植性关注某个原生快速路径在这台机器上能不能用，以及不可用时是否有可移植的回退实现；这里关注的是在所有可选实现里，某一个时刻到底真正运行了哪个，以及我们拿什么证明。

## Linux 已经能看到加载后的 BPF 和 JIT 机器码

当前 BPF 用户态 API 已经暴露了很多调试器需要的基础信息。

内核 [`struct bpf_prog_info`](https://github.com/torvalds/linux/blob/master/include/uapi/linux/bpf.h) 包含程序 ID 和 tag、翻译后程序的长度与内容、JIT 后程序的长度与内容、BTF 标识、函数信息、源码行信息、JIT 行信息、map ID、加载时间，以及可选的运行统计。[BTF 文档](https://kernel.org/doc/html/next/bpf/btf.html) 也明确说明，自省工具可以取得 `bpf_prog_info`、BTF、翻译后的字节码和 JIT 行信息，再把源码位置与 BPF/JIT 指令对应起来。

`bpftool` 把这些接口变成了直接可用的操作。当前 [`prog dump`](https://manpages.debian.org/trixie-backports/bpftool/bpftool-prog.8) 可以导出翻译后的 BPF 和 JIT 后的主机机器码、输出原始 opcode，并在有行号信息时显示源码位置。`prog show` 则能看到 ID、tag、加载时间、xlated/JIT 大小、maps、持有 FD 的进程，以及启用统计后的运行次数和运行时间。

对于静态部署，这已经是很好的基础。很多时候我们能回答：现在加载了什么程序、verifier/JIT 最终产生了什么、这条指令大致对应哪一行源码。

Linux 的 BPF tag 也提供了一个实用的内容派生标识。当前 [`kernel/bpf/core.c`](https://github.com/torvalds/linux/blob/master/kernel/bpf/core.c) 里的 `bpf_prog_calc_tag()` 会对 BPF 指令流做哈希，并先把不稳定的 map FD 字段归一化；内核给 JIT 后的 BPF symbol 命名时也会使用这个 tag。相比只看某个进程内的 FD，它更适合成为工具之间的关联键。

但 tag 回答的是“这份 BPF 字节码是谁”，不是“它为什么出现、什么时候真正跑过”。

## 动态 re-JIT 把调试变成了版本历史问题

公开的 [BpfReJIT 设计](https://lpc.events/event/20/contributions/2445/) 让这个问题变得很具体。它的用户态 shim 可以拦截 load/attach，根据配置、负载行为与内核版本重写字节码，然后把每个候选程序再送回原来的 verifier 和 JIT。设计里明确包含运行时的推测式优化。

一旦运行时能这么做，同一个逻辑应用就可能形成这样的序列：

```text
source object
   |
   +-- generation 41: portable BPF -> JIT image A
   |
   +-- generation 42: branch-specialized BPF -> JIT image B
   |
   +-- generation 43: deoptimized BPF -> JIT image C
```

这三代程序都可以通过 verifier，也都可以有合法的 BPF tag，甚至都带着完整的 BTF 行号信息。真正的调试缺口在于：这些信息通常绑定在一个个已加载对象上，而事故绑定的是**时间与实际执行**。

所以运维人员实际要回答四个不同问题：

1. 观测发生时，这个 hook/link 上真正生效的是哪一代程序？
2. 哪个源构建产物和哪一串变换生成了它？
3. 哪个 profile、配置、硬件能力或其他假设让优化器选择了它？
4. 在这台机器上，哪个 JIT 后端和哪份机器码最终执行了？

现有内核自省接口已经提供了这条链上的很多零件，但还没有定义一份可长期保存的记录，把四件事一次连起来。

## BTF 行号信息能映射源码，但不能解释优化历史

BTF 让 BPF 调试好用很多，因为翻译后的指令与 JIT 指令都可以保留源码行信息。对于普通编译流水线，看到指令后能够跳回源码，很多时候已经足够。

动态优化器会多出一层映射：

```text
source line
   -> original BPF instruction
      -> rewritten BPF instruction
         -> JIT instruction range
            -> observed sample/event
```

如果一次重写复制了基本块、折叠了条件、插入 guard、把一串指令换成原生操作，或者之后发生去优化，同一行源码会在不同时间对应多个生成后的指令区间。保留原始行号当然有用，但它不能说明是哪一个变换造出了这条指令，也不能说明那个变换当时依赖了什么假设。

这和编译器调试里“源码位置”和“优化历史”的区别很接近。前者告诉你代码从哪里来，后者告诉你中间发生过什么。

对 BPF 来说这个区别更明显，因为 verifier 与 JIT 本来就是两个独立阶段。一次完整事故复盘可能需要区分：

- 原始应用 BPF 构建产物；
- 用户态优化器生成的重写后 BPF；
- verifier 对这份程序的接受结果；
- 当时实际使用的内核与 JIT 版本；
- 可选的架构专用快速路径；
- 真正 attach 并接收执行的程序代次；
- 这一代程序何时被退役或替换。

## 现有研究还缺什么

### 已加载对象的标识不能直接充当长期事故标识

BTF 文档说明，一个已加载 BPF 程序的 ID 在它的生命周期内保持唯一。这个语义本身没有问题，但事故档案往往比已加载对象活得更久。程序可能在复盘开始之前就已经被替换并释放。

BPF tag 因为由内容派生，比 ID 更耐用；可它仍然只标识 BPF 指令，而不是完整的优化决策。两次部署完全可以使用相同的重写后 BPF 字节，却来自不同优化器版本、不同 profile 证据、不同内核构建或不同原生 lowering。反过来，两份语义等价的程序也可能因为无害重写而得到不同 tag。

缺少的是一个可长期保存的执行代次标识：它要覆盖足够的变换与部署上下文，才能在事后复现当时的代码路径。

### 当前 dump 能告诉你代码长什么样，却不能解释为什么选中了它

`bpftool prog dump xlated` 和 `bpftool prog dump jited` 是很强的取证基础。只要对象还在，它们可以直接揭示翻译后的指令和机器码，但不会记录优化器当时用于选择变体的条件。

对于 profile-guided optimizer，真正有用的解释可能是：`branch 17 被特化，是因为 profile epoch P 观察到 99.8% bias，并且 guard G 当时仍有效`。对于架构专用操作，则可能是：`native implementation N 被选中，因为 capability set C 匹配，同时 proof witness W 被接受`。

没有这层决策记录，看到机器码可以证明**跑了什么**，却解释不了**为什么会有这个变体**。

### 程序代次快速切换后，sample 归属会变得含糊

Profiler 观察的是随时间发生的执行；re-JIT 运行时改变的是随时间发生的代码。两条时间线必须能对齐。

如果 trace 只存源码 symbol 或逻辑程序名，不同代次的 sample 会被折叠到一起；如果只存短生命周期的程序 ID 或地址，对象释放后又可能无法做符号化；如果只存 BPF tag，却没有生效区间和变换元数据，运维人员虽然知道字节码是哪一份，还是不知道它为什么被生成。

所以动态特化需要一种低开销方式，把执行证据绑定到程序代次，而不是给每个 sample 都复制一整份 manifest。

## 值得继续做、同时有论文和生产价值的方向

### 1. 每个特化代次都生成一份执行收据

第一个可实现的产物可以是一份只追加、按内容寻址的执行收据（execution receipt）。它主要放在用户态，不需要为了来源记录把热路径上的内核 ABI 变得很重。

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

大的 verifier log、profile、proof object、JIT image 可以继续存在外部 artifact store，只在收据里保留 digest。真正重要的是让整条链不可变，而且能稳定关联。

这里最值得研究的是标识设计。`generation_id` 不应该只是某个进程里会复用的计数器。它至少要稳定到足以让另一台机器收集的 trace 在事故后还能关联到优化器与部署证据。一个原型可以把父收据、特化后的 BPF 字节、优化器版本、假设集合与目标环境一起做内容哈希。

这并不是重复 9 月 5 日报告里的 equivalence certificate。Certificate 主要说明这一代程序为什么在语义上可接受；执行收据要进一步保留部署证据，证明**真正被激活和观测到的就是这一代**。

### 2. 用代次区间表给 sample 做归属

第二个产物应该把运行时观测接到执行收据上，同时避免每个事件都携带大块元数据。

一种设计是在激活时维护区间表：

```text
[time_start, time_end, hook/link identity, JIT address range]
    -> generation_id
```

新一代程序生效时，运行时记录旧一代的退役边界，再记录新一代的 JIT 地址范围与稳定代次 ID。Perf sample、BPF 侧计数器、数据包 trace 或应用事件只需要留下足够的标识与时间信息，就能在离线阶段关联回这张表。

更贴近内核的原型可以在现有 BPF 程序元数据或 JIT symbol 事件周围暴露一个短代次 cookie；先从用户态实现的版本则可以订阅 load/attach/replace，并在外部保留已经退役的映射。真正的问题是：要做到没有竞态的激活边界，到底需要多少内核参与？

最值得拿来攻击系统的是这些边界：link replacement 时仍在 CPU 上结束的旧 invocation、跨程序 tail call、freplace/fentry 关系、per-CPU 执行、快速去优化，以及 JIT 机器码释放之后的地址复用。如果 collector 丢失交接事件，表示里必须保留明确的 `unknown`，而不是默认把 sample 归给最新一代。

### 3. 用对抗式 re-JIT 做真正的取证 benchmark

第三个产物可以直接把 ground truth 写进负载，再让调试器事后找答案。

从同一份 BPF 源码出发，连续生成几十到几百个代次。在某一个特定假设 epoch 下，让某个变换引入一个已知的性能异常或可观测行为。随后继续多轮特化，并在分析开始前把出问题的那一代程序卸载掉。

至少比较四种 baseline：

- 当前 `bpftool` 程序元数据、translated/JIT dump 与 BTF 行号信息；
- Linux 常规 profiler/JIT symbol 记录；
- 完整优化器日志加每代 snapshot；
- 执行收据加按代次归属的 sample。

主要指标不应该只有存储开销，还要测：

- 精确代次归属的 precision / recall；
- 证据丢失时，正确保留 `unknown` 的比例；
- source -> rewrite -> verifier -> JIT -> execution 链能否完整重建；
- 能否指出真正引入异常的优化决策；
- 在原内核/JIT 与另一版本上的 replay success；
- 每一代需要保留的字节数，以及激活和采样时的运行时开销；
- 一个没有亲眼看到优化发生的运维人员，需要多久才能找到根因。

好的系统应该让陈旧或缺失的来源证据显式暴露。一个自信地把 sample 归错代次的调试器，比直接告诉你“证据不完整”更危险。

## 哪些结果会改变这个判断？

有三种结果会明显削弱单独做动态特化来源层的必要性。

第一，现有 Linux 元数据、BPF tag、BTF/JIT 行号信息与常规 profiler 加载记录，也许已经足以在 unload、replace 之后准确还原历史代次，而且连优化决策都能无歧义重建。如果现实的 re-JIT benchmark 证明不加任何额外元数据也能做到，就没有必要再造执行收据格式。

第二，动态 BPF 特化也许长期都很少见，以至于每次重写直接保存完整字节码/JIT snapshot 和优化器日志就足够便宜。如果完整 snapshot 在存储、同步和保留周期上都没有问题，精细的来源图只会增加复杂度。

第三，执行边界可能根本不值得长期精确观测。如果无竞态的代次归属需要给每个事件增加元数据或同步，并明显吃掉优化带来的收益，那么更合理的答案可能是粗粒度 epoch 日志加按需调试模式，而不是永久开启细粒度归属。

现在的证据说明，这个问题还没有大规模出现，但机制已经值得提前定义。Linux 已经能给出字节码、JIT 机器码、tag、BTF 行号映射和运行元数据；BpfReJIT 也把“运行时反复重写 BPF”变成了具体设计，而不是假想的编译器 pass。**真正缺少的不是另一个反汇编器，而是一个可长期保存的关联键，把事故观测接到当时真正产生并执行那份代码的优化代次。**

## 参考资料

- Linux kernel UAPI. [`struct bpf_prog_info` in `include/uapi/linux/bpf.h`](https://github.com/torvalds/linux/blob/master/include/uapi/linux/bpf.h)，访问于 2026-09-07。
- Linux kernel documentation. [BPF Type Format (BTF)](https://kernel.org/doc/html/next/bpf/btf.html)，访问于 2026-09-07。
- Linux bpftool documentation. [`bpftool-prog(8)`](https://manpages.debian.org/trixie-backports/bpftool/bpftool-prog.8)，访问于 2026-09-07。
- Linux kernel source. [`kernel/bpf/core.c`](https://github.com/torvalds/linux/blob/master/kernel/bpf/core.c)，访问于 2026-09-07。
- Linux Plumbers Conference 2026. [kops and rejit: Safely Optimizing eBPF for Hardware and Workloads](https://lpc.events/event/20/contributions/2445/)，访问于 2026-09-07。
