---
date: 2026-08-09
title: "多个 eBPF 程序如何安全共享同一个 Hook？"
description: "多个独立 eBPF 程序共享同一 hook 时，执行顺序只是第一步。本文比较 Linux 多程序链、libxdp、vBPF 与 KRAKENGUARD，分析返回值、状态共享、权限、更新和故障语义，并提出可验证、可版本化的组合契约，让独立程序在加载前声明影响范围并安全更新整条链。"
tags:
  - Daily Report
  - eBPF
  - Runtime Systems
  - Composition
  - Linux
research_question: "多个独立开发的 eBPF 程序共享同一 hook、修改共同状态、产生竞争结果并各自升级时，运行时应该暴露哪些可验证的组合语义？"
source_cutoff: 2026-08-09
status: daily-report
---

# 多个 eBPF 程序如何安全共享同一个 Hook？

设想同一个 ingress 路径上同时挂着三个由不同团队维护的 eBPF 程序。安全程序可以丢包，遥测程序希望看到每一个包，流量控制程序会修改元数据再继续转发。三个程序单独看都通过 verifier，也都能正常 attach。真正困难的问题出现在它们同时工作之后：这三个程序组成的系统到底应该是什么语义？

这就是 **eBPF hook 组合** 问题。执行顺序当然重要，但顺序本身无法回答这些问题：一次 `drop` 是否必须终止后续执行，后面的程序看到的是原始数据还是已经修改的数据，两个程序能不能安全写同一个 map，以及替换其中一个程序时是否可能短暂出现一个不安全的中间状态。

<!-- more -->

本文的判断是：**多程序 eBPF 要做到可独立部署而且安全共存，需要的是组合契约，而不只是 dispatcher。** 这个契约至少要把四件事写清楚：每个程序能产生什么影响，中间结果如何传递，多个返回结果如何合并，以及当前生效的是整套组合的哪个版本。dispatcher、隔离、静态分析、虚拟化和 BPF link 都已经提供了重要机制，但还缺少一个把这些机制绑定到 attach 与更新语义上的机器可检查接口。

这也是上一篇[用户态 eBPF 运行时报告](https://eunomia.dev/zh/research/userspace-ebpf-runtime-contract/)的直接延续。上一篇讨论的是单个扩展进入不同运行时之后，attach、能力、状态和生命周期如何显式化。多个扩展共享同一个 attach point 后，这些问题会进一步变成组合问题。

## Linux 已经支持多程序组合，但每个 Hook 的规则并不一样

Linux 并不存在一条通用规则来定义“多个 BPF 程序一起运行”。不同 hook 的底层操作不同，所以它们有意采用不同的组合语义。

[`BPF_SK_LOOKUP` 文档](https://docs.kernel.org/bpf/prog_sk_lookup.html)是一个很直接的例子。多个程序可以挂到同一个 network namespace，并按 attach 顺序执行。但最后结果并不是简单取最后一个程序的 return value。程序可以通过 `bpf_sk_assign()` 选择 socket。如果多个程序都选择了 socket 并返回 `SK_PASS`，最后一次有效选择生效；如果某个程序返回 `SK_DROP`，只有在没有任何程序通过 `SK_PASS` 提供有效 socket 时才会导致 lookup 失败。这里实际存在的是“verdict 加累积选中对象”的结果合并规则。

[`BPF_PROG_TYPE_CGROUP_SOCKOPT`](https://docs.kernel.org/bpf/prog_cgroup_sockopt.html)又是另一套模型。在 cgroup 层级中，程序会从子 cgroup 向父 cgroup 执行。前一个程序对 `bpf_sockopt` context 的修改会被后面的程序看到。返回 `0` 表示拒绝操作，返回 `1` 表示继续执行下一段 BPF。这里的组合既包含有序的数据变换，也包含是否继续的控制结果。

[HID-BPF](https://docs.kernel.org/hid/hid-bpf.html)把共享修改的问题写得更明确。除少数 attachment type 外，同一个设备可以挂多个程序。它们依次操作同一块数据 buffer，后一个程序看到的是已经修改的数据，而不是原始输入。返回负数会丢弃整个事件。接口允许用 `BPF_F_BEFORE` 把程序插到最前面，但这个 flag 并不能告诉一个独立开发的程序：它是否真的能正确处理前面程序重写过的数据。

网络生态还在 kernel primitive 之上增加了自己的组合机制。[`libxdp`](https://github.com/xdp-project/xdp-tools/blob/master/lib/libxdp/README.org)通过 dispatcher 让多个 XDP 程序共享一个 interface。每个 component 有运行优先级和 chain-call action。一个程序的 return code 可以决定继续执行下一个程序，或者直接终止链。这比互相覆盖 attach 好很多，但它仍然默认各个程序已经对 packet 修改、map 使用和结果语义达成一致。

新的 TCX 在生命周期和顺序上更进一步。[Eunomia 的 TCX 教程](https://eunomia.dev/zh/tutorials/50-tcx/)展示了 BPF link 所有权、`BPF_F_BEFORE` / `BPF_F_AFTER` 相对排序、`BPF_F_REPLACE`、chain revision，以及 `TCX_NEXT`、`TCX_PASS`、`TCX_DROP` 这类明确的返回语义。TCX 说明相对顺序和 revision-aware attach 完全可以成为正式接口，而不是 loader 之间的约定。

这些设计没有谁是“错”的。socket 选择、packet 分类、HID 数据变换和 socket option 过滤，本来就不应该强行使用同一套结果合并规则。真正值得注意的是，它们的差异大多隐藏在各自 hook 的 API 和文档里，没有被表示成可以让部署工具统一检查的组合契约。

| 机制 | 执行顺序 | 后续程序看到什么 | 结果如何合并 | 状态与更新 |
| --- | --- | --- | --- | --- |
| `BPF_SK_LOOKUP` | attach 顺序 | 可能继承已选择的 socket | 最后的有效 socket 选择可能覆盖之前选择，drop 还受是否存在有效选择影响 | hook 专用规则 |
| cgroup sockopt | 子 cgroup 到父 cgroup | 前面程序修改后的 context | return 控制拒绝或继续 | 可使用 cgroup/socket local storage |
| HID-BPF | list 顺序，可插到最前 | 同一个可修改 buffer | 负数终止并丢弃事件 | link 生命周期，共享修改可见 |
| libxdp | run priority | 前面 component 处理后的 packet | chain-call action 决定继续还是终止 | dispatcher 管理 component set |
| TCX | before/after 相对顺序 | packet 与 `__sk_buff` 状态 | `TCX_NEXT` 与终止类结果明确区分 | BPF link、replace、revision-aware 变更 |

## 执行顺序只是 eBPF Hook 组合的一条轴

面对多程序冲突，一个常见做法是增加 priority 或显式 chain。这确实解决了 deterministic ordering，但至少还有四类交互没有被解决。

### Hook 自己需要明确的结果合并规则

假设安全过滤器后面是遥测程序。安全过滤器决定拒绝一次操作后，遥测还要不要运行，以便记录被拒绝的事件？如果继续运行，它的 return value 是否可能把拒绝结果覆盖掉？再把遥测换成 socket selector，“第一个结果赢”和“最后一个结果赢”都讲得通，但安全含义完全不同。

Linux 已经针对不同 hook 回答了这些问题。`BPF_SK_LOOKUP` 有 socket selection 的优先规则；TCX 用 `TCX_NEXT` 把继续执行和终止结果分开；libxdp 允许 component metadata 声明哪些 XDP action 应继续执行。它们本质上都是小型的“结果代数”，而不只是整数返回值。

因此，一个通用组合层不应该把 return value 当作 opaque integer。它至少需要知道 `continue`、`deny`、`select`、`redirect`、`transform`、`terminal` 等结果类别，再由 hook adapter 映射到原生语义。

### 数据修改对后续程序是否可见，必须成为显式选择

HID-BPF 明确规定多个程序操作同一 buffer，后面的程序只能自然看到被前面改过的数据。cgroup sockopt 同样把 context 修改传给后面的程序。

这很有用，因为它可以构成 pipeline。但它同时创建了隐藏依赖。一个 parser 也许只对原始 packet 正确，前面插入一个 rewriter 后就会悄悄产生错误。反过来，如果强制每个程序都看 immutable snapshot，又会破坏很多有价值的 staged transformation，并引入复制成本。

所以组合契约需要表达程序究竟要求哪一种视图：原始输入、上一个程序的输出，或者某个有名字的派生视图。如果运行时无法提供要求的视图，应该在 attach 之前失败，而不是让线上流量帮我们发现问题。

### 共享 Map 是并发协议，不只是共享数据结构

Linux 当然允许多个程序有意共享 map。[`BPF_MAP_TYPE_CGROUP_STORAGE` 文档](https://docs.kernel.org/bpf/map_cgroup_storage.html)指出，从 Linux 5.9 开始 storage 可以被多个程序共享，而且多个 CPU 上的访问没有隐含同步，使用者需要自己处理同步。

这在 primitive 层面是合理的。但对独立部署的扩展来说，“两个程序都能打开这个 map”远远不够。一个程序可能假设自己是唯一 writer，另一个程序可能会 reset entry，第三个程序又把同一 entry 当成 lease。verifier 能证明 memory safety，却不会证明这些状态机假设能组合。

至少需要区分 private state、shared read-only、single-writer shared state、以及带明确同步协议的 multi-writer state。更复杂的 invariant 可以留给上层，但 ownership 不能继续隐含。

### 替换一个程序，实际上可能是整条链的版本切换

BPF link 让单个程序的 lifetime 更容易管理，相对顺序让插入位置更确定，TCX 的 revision checking 还能发现并发修改。剩下的问题是 semantic atomicity。

例如安全程序从 A 版本升级到 B 版本，而 B 依赖一个新版 metadata producer 先运行。如果只原子地 replace 安全程序，B 仍然可能短暂运行在旧 composition 上。即使每一次 `BPF_LINK_CREATE` 或 replace 都是原子的，真正需要上线的配置也可能要求多项变化一起生效。

因此，组合本身需要 generation。运行时应该先验证完整的新链，在 hot path 之外尽可能准备好成员和状态，然后在当前 generation 仍然符合 expected revision 时一次提交。

## 最近的工作已经解决不少问题，但还没有统一成组合协议

多程序 eBPF 并不是一个没人研究的空白方向。

[OSDI 2026 的 vBPF](https://www.usenix.org/conference/osdi26/presentation/zhang-jing)解决 multi-tenant eBPF contention。它把 logical program 到 physical hook 的 binding 延迟到运行时，用 Sniffer 做 tenant attribution，用 O(1) Dispatcher 选择程序，并通过 compiler-assisted 方法隔离状态。这说明多个 tenant 共享基础设施时，简单 linear attach 不够。它主要解决 virtualization 和 tenant isolation，而不是那些本来就要共同处理一个事件的程序应该怎样组合语义。

[NSDI 2026 的 KRAKENGUARD](https://www.usenix.org/conference/nsdi26/presentation/patel)使用 trusted userspace manager 和 symbolic execution，在 load time 检查 helper、memory access、return value 和 cross-program interference。它的 XDP-as-a-Service case 表明，组合在真正 attach 前可以接受相当深入的安全分析。这对组合层很有价值，但 attachment system 仍然要决定“允许的组合到底应该是什么语义”。

[Yaksha-Prashna](https://arxiv.org/abs/2602.11232)直接分析 eBPF bytecode network function 的规范一致性和它对其他 bytecode 的依赖。这一点很重要，因为未来的组合系统不能假设总能拿到源码 annotation。同时它也意味着，单纯提出“静态分析两个 eBPF 程序是否冲突”已经不足以构成新的系统方向。还需要回答分析结果怎样变成可 enforce 的运行时状态。

[ACM SIGCOMM 2026 已公布的 BPFChain tutorial](https://conferences.sigcomm.org/sigcomm/2026/ttbpfchain/)也是一个很直接的信号。它的 program 已经把 execution-order conflict、return-value override、shared-map race、trampoline chainer 和 chain monitoring 列为重点。本文 source cutoff 是 8 月 9 日，而 SIGCOMM 2026 将在 8 月 17 日之后举行，所以这里引用的是已公布的教程计划，而不是把未来活动写成已经完成。它至少说明一件事：再做一个 dispatcher 或 chainer 本身已经不是足够的新意。

剩余的问题在这些工作之间。我们已经有 chain、有 tenant isolation、有 effect analysis、有 individual link lifecycle，但还缺一个被广泛使用的对象，能够机器可读地说明：**哪些影响可以组合，结果使用什么 resolver，共享哪些状态，以及整套组合的哪个 generation 正在运行。**

## 组合契约需要把影响范围和依赖关系显式化

这个契约不需要取代现有 hook API。它可以位于它们上方，再编译成 TCX、libxdp、cgroup attach、kernel link，或者 [bpftime](https://github.com/eunomia-bpf/bpftime) 这样的用户态运行时。

最小的 per-program manifest 可以描述：程序期望的 hook 和 target；它可能读取或写入哪些 context field 或 packet region；它能调用哪些 helper class 和外部 side effect；它读写哪些 map，以及每个 map 的 ownership mode；它要求原始事件、前一程序输出还是某个派生视图；它与其他组件的 `before` / `after` 约束；它可能产生哪些结果类别，以及哪些结果是 terminal；还有 failure behavior 与可选 execution budget。

hook adapter 再补上原生结果语义。XDP adapter 可以把 XDP action 映射成继续与终止类别；TCX 直接利用 `TCX_NEXT` 等已有语义；`BPF_SK_LOOKUP` 则必须同时表示 verdict 和 selected socket；HID-BPF 需要表示共同 buffer 与 event-discard 规则。

manifest 中有些事实可以由开发者声明，有些应该从 bytecode、BTF、map reference、helper call 或 program analysis 中推断。关键区别在于，静态分析只是组合契约的证据来源，不是契约本身。loader 应该比较声明与推断效果，在两者矛盾时报告问题，并在无法建立所要求保证时拒绝 attach。

## 现有研究还缺什么

### 不同 Hook 的组合语义仍然难以比较和复用

kernel documentation 在单个 hook 内往往已经定义得很精确，但维护大量 BPF component 的 operator 仍然需要手工理解每一种 family。现在缺少一个共同词汇去表达“terminal result”“selected object”“shared mutation”“original-view requirement”等概念，让部署工具能跨 hook 检查。

真正需要的下一步证据，是一组真实 multi-program deployment 的 corpus，观察 XDP、TCX、cgroup、tracing、HID 和 userspace backend 到底重复出现哪些语义维度。如果每一个 hook 都需要完全特制的规则，那么通用 contract 只会退化成 documentation metadata。如果少量 effect 和 outcome pattern 能覆盖大多数部署，抽象才站得住。

### 隔离分析还没有成为 Attach 协议

KRAKENGUARD 能分析 helper、memory、return 和 interference，Yaksha-Prashna 能分析 conformance 与 bytecode dependency。这些能力已经比简单的 declaration file 强得多。

问题在于分析结果怎样进入 live composition。分析器可以知道 B 读取 A 写入的字段，但 attach system 还要决定这是否意味着 `A before B`，A 给出 terminal result 后 B 是否仍然应该运行，以及更新过程中能不能短暂违反这个 dependency。

因此可以把边界划得很清楚：analysis 产生 evidence，composition protocol 把 evidence 变成可 enforce 的 attach 和 generation rule。

### Revision-aware Link 还不等于整个组合的事务

TCX 是一个很重要的反例。它说明 BPF 并不缺排序、link ownership 或 revision control。真正较窄的 gap 是：一组程序、状态 contract 和 result resolver 怎样作为一个 semantic unit 一起变化。

好的实验必须在多个 coordinated update 之间注入 failure。如果普通 TCX 或其他现有机制加一层很薄的 userspace wrapper 就能守住所有 invariant，那么所谓 versioned composition generation 就不值得成为新的系统抽象。

### 共享状态正确性仍然大多在 Verifier 之外

verifier 可以证明单程序的很多安全属性，map type 与 lock 提供并发访问机制，fine-grained isolation 还能限制程序接触哪些状态。

但这些机制都不会自动证明两个独立状态机对 ownership、reset、epoch 或 update order 有相同假设。因此组合问题会从 memory safety 延伸到 protocol compatibility。真正的研究难点是：究竟能捕获多少这类 protocol，而不把接口膨胀成一套通用 formal specification language。

## 兼具学术价值与生产价值的方向

### 用 Effect Inference 支撑的类型化组合 Manifest

**Gap。** 部署系统能排序程序，分析系统能发现部分 effect，但两者之间缺少 portable attachment contract。

**Mechanism。** 定义一个紧凑 schema，描述 read/write effect、map ownership、outcome category、visibility requirement 与 partial-order constraint。再利用 BTF、ELF metadata、helper-call inspection，以及可选的 symbolic 或 abstract interpretation，从编译后的程序推断 effect summary。loader 在 attach 前对比声明与推断结果，并解出合法的 partial order。

**Delta。** 这不是另一个 dispatcher，也不是另一个独立 analyzer。libxdp 或 TCX 继续负责实际执行，KRAKENGUARD 或 Yaksha-Prashna 一类分析继续提供证据。新 artifact 是 analysis 和 attachment 之间的机器可检查 contract。

**Artifact。** 一份公开 schema，一个基于 libbpf 的 loader，优先实现 TCX 和 libxdp adapter，再扩展到 cgroup hook 与 bpftime。可选 compiler plugin 可以生成 source-derived effect metadata。

**Evaluation。** 收集独立开发的 observability、security、networking 程序，构造有效和无效的 pair / triple。分别测 write/write conflict、缺失 ordering dependency、map ownership mismatch、undeclared side effect 的检测率，同时报告 false rejection 与 false acceptance。attach-time analysis cost 与 steady-state overhead 要分开测，后者在验证结束后应该尽量接近 native backend。

**Academic value。** 可以研究一个 eBPF composition effect system 是否既能表达真实程序，又保持可判定并且不绑定单一 backend。

**Production value。** operator 能在部署前回答“两个由不同团队发布的程序能不能安全共享这个 hook”，而不是等到丢流量或 policy bypass 后再排查。

**Failure condition。** 如果 annotation 成本过高、推断制造大量 false conflict，或者多数真实程序都必须依赖 hook-specific escape hatch，这个 common type system 就没有价值。

### 为不同 Hook 建立显式的 Outcome Algebra Adapter

**Gap。** 多程序 hook 都有 return value，但这些返回值没有统一的组合含义。把它们压成 generic priority 很容易改变安全或路由语义。

**Mechanism。** 每个 hook adapter 都提供类型化结果模型。共同 vocabulary 可以包括 `continue`、`deny`、`select`、`redirect`、`transform`、`terminal`，而 adapter 保留 selected socket 之类 hook-specific payload。composition plan 指定 resolver。如果两个 terminal 或 stateful result 的组合存在歧义，planner 在没有显式 resolver 时直接拒绝。

**Delta。** 现有 chain mechanism 主要决定下一个程序要不要运行。这里关注的是“多个程序共同产生的结果究竟是什么意思”，并让这个含义可检查、可测试。只要 kernel 已经有足够 primitive，就应该编译到原生 return convention，而不是再发明一个 packet-processing runtime。

**Artifact。** 为 `BPF_SK_LOOKUP`、TCX、XDP/libxdp、cgroup sockopt 和 HID-BPF 做 adapter，并提供一个 test harness，把同一组 abstract composition case 映射到真实 kernel behavior。

**Evaluation。** 构造 policy、telemetry、transformation、selection、redirect 程序的不同排列，比较 declared outcome 与 kernel observed outcome。加入恶意或偶然的 return-value override。除了 adapter overhead，更重要的是 attach 前能发现多少 semantic ambiguity。

**Academic value。** 研究问题是能否找到一个足够小的 algebra，在不假装所有 BPF hook 相同的前提下复用主要组合性质。

**Production value。** security 和 networking 团队可以清楚看到为什么某个结果最终生效，并在 observability 或 routing component 可能削弱 policy 时提前拒绝部署。

**Failure condition。** 如果每个 hook 都必须有完全独立的 algebra，几乎没有可复用性质，那么这个抽象应该留在 hook-local 层，而不是强行做 cross-hook interface。

### 给整个 Hook 组合建立 Versioned Generation

**Gap。** 单个 BPF link 可以安全 replace，TCX 也已经支持 revision-aware chain change，但一个真正的 semantic composition 可能包含多个程序、ordering constraint、state contract 和一个 result resolver，它们需要一起演化。

**Mechanism。** 把 active composition 表示成 generation object。先 off-path 构造下一代，验证所有成员与 dependency，解出顺序，准备兼容 map 或 state view，然后对 expected current generation 做 compare-and-swap commit。任何准备或验证失败都保持旧 generation 继续运行。如果 backend 根本无法原子切换完整 chain，adapter 必须暴露这个限制，而不能伪装成 transactional behavior。

**Delta。** 这个方向并不试图解决任意 stateful eBPF application 的完整升级问题，那是本系列下一篇更大的问题。这里范围更窄，只让“同一个共享 hook 上有哪些成员，以及它们采用什么组合语义”成为一个 coherent versioned object。

**Artifact。** 一个 userspace composition manager。先以 TCX 为 prototype，因为 TCX 已经暴露 ordering 与 revision，再做 libxdp dispatcher adapter。control-plane object 记录 generation ID、member program/link ID、effect contract 与 resolver choice，并允许 operator inspect。

**Evaluation。** 在持续流量或事件下反复 add、remove、reorder、replace program，并在每一个 preparation step 注入 process crash 和故障。定义“deny policy 从未缺席”“parser 与 consumer version 永远匹配”“old writer 不会写 new map layout”之类 safety invariant。测 interruption window、rollback time、throughput impact，以及是否出现任何 intermediate invalid composition。

**Academic value。** 可以回答 transactional configuration 是否能在不改 kernel 的情况下覆盖 heterogeneous BPF attachment mechanism，以及哪个边界最终必须得到 kernel support。

**Production value。** fleet operator 可以独立 rollout 多个 eBPF component，而不用把每次 chain update 都变成人工协调的 maintenance operation。

**Failure condition。** 如果已有 link 与 revision primitive 加很薄的 userspace wrapper 就能提供所需 multi-object atomicity，那么它应该只是工程 library，而不是新的系统抽象。

## 更实际的架构是复用原生机制的 Planner，而不是万能 Dispatcher

最有可能落地的设计并不是一个 universal mega-dispatcher，而是一层 planner 和 contract，针对每个 hook 复用最合适的 native primitive。

对 TCX，可以直接使用 native link、relative ordering 和 revision；对 XDP，可以把验证后的 plan 编译成 libxdp dispatcher；对 cgroup hook，保留 hierarchy 与各自 return semantics；对 HID-BPF，如果前面的 transformer 会破坏原始数据，而后面程序声明自己必须看到 original event，planner 就拒绝这个组合。对 bpftime 这样的 userspace runtime，同一 contract 也可以驱动本地 dispatcher 与 capability model，即使物理 attach mechanism 完全不同。

这样做还会给 observability 一个稳定单位。工具不再只能列出 program ID，而可以报告当前 composition generation、dependency、哪个程序终止了事件、当前共享状态采用什么 contract，以及 live chain 是否仍然等于验证过的 plan。

这很重要，因为组合故障在表面上经常只是普通 application failure。packet 消失、syscall 被拒绝、input event 形状变化，都不一定直接指向 BPF。如果没有 first-class composition identity，operator 只能从多个 loader 的状态里重新拼出整条 chain，再猜是哪一段交互造成了结果。

## 哪些结果会改变这个判断？

有三类证据会明显削弱“需要新的 eBPF hook 组合契约”这个判断。

第一，如果已有生产机制实际上已经提供了跨 hook 的 machine-readable effect、outcome resolution、shared-state ownership、ordering dependency 和 versioned whole-chain update，那么更值得做的是 adoption 和 conformance test，而不是再发明一层抽象。

第二，如果大规模 empirical study 证明，只要有 deterministic ordering 和 memory isolation，独立 eBPF 程序几乎不会再出现 semantic conflict，那么额外 contract 可能只是在为少数故障增加复杂度。

第三，如果 prototype 证明 effect inference 太不精确，或者 transactional composition 为了获得一致性需要大量 copying、pause 或 backend-specific code，最终 operator 还不如继续使用每个 hook 专用的 manager，那么这个 cross-hook 方案也应该被放弃。

目前的证据更支持相反方向。Linux 已经暴露多种不同的组合语义，libxdp 和 TCX 让 multi-program chain 成为现实，vBPF 与 KRAKENGUARD 说明多 tenant 共存不能只依赖传统 verifier，Yaksha-Prashna 说明 bytecode dependency 可以被分析，已公布的 BPFChain 内容则把 multi-program conflict 明确当作 production discipline。下一步真正缺的不是再加一条 chain，而是把这些机制连接成一个可测试的声明：共享同一个 hook 的多个程序，究竟被允许组成什么样的系统。