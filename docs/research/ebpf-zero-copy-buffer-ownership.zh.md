---
date: 2026-08-24
title: "零拷贝 eBPF 数据路径里，谁拥有数据包缓冲区？"
description: "AF_XDP、io_uring ZC Rx 与 DPDK 都依赖缓冲区复用，但所有权和回收协议不同。本文提出跨内核、用户态与网卡的缓冲区能力、策略 provenance 与故障注入评测。"
tags:
  - Daily Report
  - eBPF
  - Networking
  - Security
  - XDP
  - Zero Copy
research_question: "零拷贝 eBPF 网络路径应该怎样表示缓冲区所有权、DMA 可达性、回收状态和策略 provenance，使这些语义能跨内核、用户态与网卡 handoff 保持一致？"
source_cutoff: 2026-08-24
status: daily-report
---

# 零拷贝 eBPF 数据路径里，谁拥有数据包缓冲区？

一个包先进入 XDP 程序，被重定向到 AF_XDP，在用户态 dataplane 里检查，然后再送回网卡。应用可能反复看到同一个 UMEM 地址，数据本身一次都没有复制。这正是 zero copy 想得到的性能收益。

但它也带来一个更容易被忽略的问题：**每个时刻到底是谁有权访问这个 buffer，哪个设备可以向它 DMA，什么时候可以回收，以及这个地址里当前这批字节还应该关联哪一代策略决定？**

<!-- more -->

Linux 对这个问题的局部答案其实已经很明确。[AF_XDP](https://docs.kernel.org/networking/af_xdp.html) 用 FILL 和 COMPLETION ring 在用户态与内核之间移交 UMEM frame 的所有权。[io_uring zero-copy Rx](https://docs.kernel.org/networking/iou-zcrx.html) 让 TCP payload 直接进入注册的用户态内存，再通过 refill ring 把已经消费的 buffer 交还内核。内核 [page_pool](https://docs.kernel.org/networking/page_pool.html) 负责 packet page 的 in-flight 计数、DMA mapping、同步和回收。DPDK 则用 `rte_mbuf`、mempool、reference count 和 external-buffer callback 管理用户态 packet buffer 生命周期。

这些机制在各自路径内部都很合理。真正困难的是，一个高性能 datapath 同时跨过几种路径时，能不能让同一套安全语义和 provenance 跟着 buffer 一起走。

先看 AF_XDP。XDP 程序可以通过 `XSKMAP` 把 ingress frame 重定向到用户态。AF_XDP 文档对 ownership 的描述非常直接：FILL ring 把 UMEM frame 从用户态交给内核，COMPLETION ring 把已经完成 TX 处理的 frame 从内核交回用户态。UMEM 可以跨 socket、queue 甚至 device 共享，但单生产者/单消费者 ring 需要应用自己保证同步。文档还明确提醒，如果同一个 frame 同时被放进互相冲突的 ownership path，NIC 可能并发收发同一块内存，从而造成 packet corruption。

再看 io_uring ZC Rx。它没有绕过整个 TCP stack。packet header 仍然在内核里正常处理，payload 才直接放到用户态内存。NIC 需要 header/data split、RSS 和 flow steering。当前 kernel API 并不负责配置这些 NIC 状态，用户需要在 io_uring 注册之前从带外完成。应用消费完数据后，再通过 refill ring 把对应的内存区域交还给内核。

DPDK 把边界继续推到用户态。它的 packet buffer 由 `rte_mbuf` 和 mempool 管理。external buffer 有独立的 reference count 和 free callback，只有最后一个关联 mbuf 脱离后，buffer 才能真正释放。这同样是一套有效的 lifetime protocol，但它不是 AF_XDP 的 ring protocol，也不是 io_uring ZC Rx 的 refill protocol。

常见的理解是，zero copy 主要是在优化 data movement。既然每个子系统已经有 ring、reference count 或 page lifecycle，那么 correctness 应该只是各自实现内部的问题。

对 programmable networking 来说，这个理解还不够。一个 BPF 程序可能在 AF_XDP handoff 之前做完安全决定；之后 NIC steering 又可能把 flow 切到另一个 queue，而这个 queue 配置并不属于 BPF control plane；用户态 runtime 还可能再执行一层策略。buffer 被回收之后，同一个物理或虚拟地址很快就会承载另一个完全无关的 packet。局部 lifetime rule 能避免很多 use-after-free，但它不会自动保留 **policy generation、provenance 和 authority**。

本文的观点是：zero-copy buffer handoff 应该被看成一种 typed capability transition，而不只是 ring operation。这个 capability 必须有 generation，fast path 开销要足够小，同时要能在事故发生后回答三个问题：当时谁拥有 buffer，哪条策略授权了这次 handoff，以及这个地址是不是已经被回收到另一个逻辑 packet 上。

这个问题和之前的 [io_uring eBPF 可编程性报告](https://eunomia.dev/zh/research/io-uring-bpf-programmability/) 不一样。那篇讨论的是 BPF 在 io_uring execution loop 里应该拥有什么 control surface。它也不同于 [异构 eBPF 执行位置报告](https://eunomia.dev/zh/research/heterogeneous-ebpf-execution-placement/)，后者讨论程序应该运行在内核、用户态还是设备侧。这里假设执行位置已经选好，缺的是跨这些位置移动的内存怎样保持统一的 lifetime 与 provenance contract。

## Zero copy 并没有消除所有权，它把所有权变得更重要

zero copy 的核心是高频复用同一块内存。因此 ownership 不会消失，反而更需要被准确表达。

### AF_XDP 把所有权移交显式放进 ring

AF_XDP socket 把 XDP 程序连接到用户态提供的 UMEM。RX 和 TX descriptor 通过 UMEM offset 指向数据。内核文档明确把 FILL 和 COMPLETION ring 描述为在用户态与内核之间转移 frame ownership 的机制。

这个措辞很重要。FILL entry 不是普通 free-list hint。用户态一旦发布一个地址，内核就可能把它用于 ingress。COMPLETION entry 表示内核已经处理完相应 TX descriptor，frame 可以重新由用户态使用。文档还特别说明，completion 只证明 ownership 已经回来，并不等价于 packet 一定成功发送。

shared UMEM 让 ownership topology 更复杂。多个 socket 可以引用同一块内存，包括不同 queue 和不同 device 上的 socket。ring 是 single-producer/single-consumer，因此应用必须自己同步。实际上，AF_XDP 已经形成了一个由 XDP program、kernel socket path、userspace worker 和 NIC 共同参与的小型 ownership protocol。

问题在于，这个 protocol 只描述 AF_XDP 自己的路径。它不会告诉我们某个 frame 是在 policy generation 84 下由 rule 19 放行，之后又在 userspace policy generation 52 下处理，最后才被回收。

### io_uring ZC Rx 把控制路径和数据路径拆开

io_uring ZC Rx 暴露的是另一种边界。header 继续走内核 TCP stack，payload 直接进入注册的 userspace memory。应用处理后通过 refill ring 把内存交回内核。

NIC setup 目前需要带外完成。用户要配置 header/data split、RSS 和 flow steering，之后才能注册对应的 io_uring queue。这是一个很好的反例：packet 实际走哪条路径，可能取决于 BPF policy object 完全看不到的 hardware queue state。

因此，一个只看到 XDP 或 socket verdict 的 debugger 可能会漏掉关键变化：flow steering 已经把流量切进或切出 zero-copy queue。buffer 生命周期完全正确，数据也没有损坏，但系统对“为什么这个 flow 走了这条路径”的解释仍然是错的。

### page_pool 保护的是内核局部的 recycling contract

kernel page_pool 专门优化 skb 和 XDP frame 使用的 page 或 page fragment。文档描述了 in-flight accounting、reference handling、DMA mapping、device synchronization，以及在 NAPI 等安全 context 里直接 recycle 的规则。

这说明高性能 packet memory 本来就需要比 `malloc/free` 更强的生命周期语义。allocator 必须知道 page 能不能回收，什么时候需要 DMA sync。不过 page_pool 的目标就是解决 kernel networking allocator 的问题，它并不试图定义跨 AF_XDP、io_uring、用户态 BPF runtime 或 DPDK 的逻辑 packet identity。

### DPDK 使用另一套用户态 lifetime model

DPDK 的 `rte_mbuf` 把 packet metadata 放在用户态，并用 mempool 做复用。external buffer 有自己的 shared info、reference count 与 free callback；当最后一个 mbuf 不再引用它时，callback 才会执行。

这套模型很适合表达共享引用，也同时暴露了 abstraction mismatch：AF_XDP 用 ring 表示 ownership，page_pool 用 kernel reference 和 recycle API，io_uring ZC Rx 用 completion/refill state，DPDK 用 mbuf ownership 和 reference count。

目标不应该是强迫它们全部换成一个 allocator。更有价值的是在 allocator 之上定义一个 **共同的 handoff contract**，每条 native path 用 adapter 保留自己的 fast path。

## 真正危险的是 stale meaning，而不只是 stale memory

假设 UMEM frame `0x4000` 第一次装的是 tenant A 的 packet。运行 policy generation 84 的 XDP 程序把它送到 AF_XDP。用户态消费后，frame 经过 completion 或 refill path 回收。很快，同一个地址又装上 tenant B 的 packet，而此时 policy 已经是 generation 85。

如果 trace 只记录 `addr=0x4000`，后续 join 就不可靠。地址只是 storage location，不是稳定的 packet identity。

这和 PID reuse 后仍把 PID 当作 process identity、或者在没有 configuration generation 的情况下解释 map entry 是同一类错误。zero-copy networking 让这个问题更明显，因为地址复用本来就是设计目标。

最小的逻辑 identity 至少要包含一次 lease generation：

```text
buffer_ref = {
  region_id,
  offset,
  lease_generation
}
```

每次 handoff 再带上当时有效的 policy 与 execution context：

```text
handoff = {
  buffer_ref,
  from_owner,
  to_owner,
  queue_or_device,
  dma_domain,
  policy_generation,
  decision_id,
  transition_seq
}
```

这些 metadata 不应该塞进每个 packet header。它们可以存在 bounded side table、userspace manifest、采样 transition event，或者某条 native path 本来就能携带的 compact token 里。真正需要研究的是：为了证明重要 invariant，最少需要保存多少状态。

## 现有工作还弱在哪里

### 1. 各路径自己的 ownership protocol 还不能组合成一个跨边界 contract

AF_XDP、io_uring ZC Rx、page_pool 和 DPDK 都定义了自己什么时候可以复用 buffer，但它们并没有共享一套 owner、lease generation、DMA reachability 与 handoff authority 语义。

缺的不是 universal memory allocator，而是一套机器可读的 transition model。它应该能表达“用户态已经把这个 AF_XDP frame 交给 kernel RX”“NIC 当前有权向这个 lease DMA”“这个 lease 已经回收，不能再继承上一批 packet 的 policy witness”。

决定性的实验是把两种以上 native zero-copy path 串起来，在它们的边界注入 ownership error，再比较 common contract 和原始 path-local API。如果所有 fault 都已经能被 native boundary 拒绝，而且诊断信息同样完整，那就没有必要引入额外抽象。

### 2. Buffer 地址的复用速度比 policy provenance 消失得更快

fast path 希望地址尽快复用，incident analysis 希望 identity 稳定。如果 observability 只按地址 join，这两个目标会直接冲突。

缺的是 generation-scoped buffer identity。它只需要活到能够把 BPF decision 和准确的 logical packet lease 关联起来，然后在 recycle 时失效。同一个物理地址应该可以随时间产生很多不同逻辑 identity。

测试方式很直接：用很小的 UMEM 强制高频 recycle，同时切换 policy generation 并交错多个 tenant。analyzer 绝不能因为地址相同，就把 generation 84 的 verdict 归到 generation 85 的数据上。

### 3. Hardware steering 可以在 BPF control plane 之外改变 zero-copy 边界

io_uring ZC Rx 当前依赖带外 NIC 配置完成 header/data split、RSS 和 flow steering。AF_XDP 可以跨 queue 和 device 共享 UMEM，DPDK 甚至可能绕过普通 kernel networking path。

缺的是一个可以与 BPF policy state join 的 **realized path description**：NIC queue、steering generation、memory region，以及 packet 进入 userspace 的具体 handoff。否则某个 hook 上的 security policy 可能完全正确，但 operator 对“哪些流量实际上会经过这个 hook”的理解已经过期。

区分能力最强的评测是只修改 steering rule，不修改 BPF policy。如果系统解释不出 flow 已经被切到不同 queue 或 userspace path，那说明 policy observability boundary 还不完整。

### 4. 如果不能复制 payload，zero-copy failure 很难诊断

最直接的 debug 方法是抓包，但这可能破坏正在调试的性能性质、暴露敏感 payload，而且依然无法解释 buffer ownership。

缺的是 metadata-first evidence：ownership transition、lease generation、queue identity、policy witness 和 drop/recycle reason，默认不保留 packet content。

评测应该在同一组 fault 上比较 full packet capture、普通 counter 和 metadata-only witness。好的 witness 设计应该能用远小于 payload capture 的数据量诊断 double reuse、stale policy attribution 和 wrong-path steering。

## 具有学术价值和生产价值的方向

### 1. 给 zero-copy handoff 一个 generation-scoped buffer capability

**Gap。** Native API 有各自的 lifetime rule，但缺少跨 API 的 ownership 与 policy provenance 表示。

**Mechanism。** 为每个活跃 zero-copy buffer lease 分配一个 compact capability：

```text
cap = {
  region_id,
  offset,
  generation,
  owner,
  access_mode,
  dma_target,
  policy_generation
}
```

transition 显式表示为 `USER_TO_RX`、`RX_TO_USER`、`USER_TO_TX`、`TX_COMPLETE`、`USER_TO_IOURING_REFILL`，以及 DPDK adapter 的 attach/detach transition。buffer recycle 时旧 capability 失效，新 lease 使用新 generation。native ring 或 mbuf 仍然是 fast path 的事实来源，capability layer 只 mirror 跨路径检查和 provenance 需要的最小状态。

在 eBPF 一侧，可以把 immutable region descriptor 放到 map，用 bounded per-CPU 或 per-queue state 记录活跃 generation。像 [bpftime](https://github.com/eunomia-bpf/bpftime) 这样的用户态 eBPF runtime 也可以在 attach boundary 使用同一 schema，而不是再定义一套 packet lifetime 词汇。

**Delta。** 现有 API 保护的是自己的 object lifetime。新的性质是一个 typed capability，它的 generation 与 authority 可以跨 API handoff 保留，同时不要求所有路径使用相同 allocator。

**Artifact。** 一个小型、尽量 UAPI-neutral 的 schema，AF_XDP 和 io_uring adapter，可选 DPDK adapter，针对关键 transition 的 BPF checker，以及能够重建 lease state machine 的 userspace validator。

**Evaluation。** 分别运行 AF_XDP zero-copy、io_uring ZC Rx 和 DPDK line-rate forwarding。测 packets/s、cycles/packet、cache miss、内存开销与 tail latency。注入 double-fill、premature recycle、completion 后错误复用、stale policy generation、worker crash 与 NIC reset，比较 prevention rate 和 diagnostic precision。

**Academic value。** 这个问题可以检验 linear type 或 capability 式 ownership 能否跨 kernel、userspace 和 device packet-memory protocol 落地，而不需要把通用 type system 放进 fast path。

**Production value。** runtime 可以对非法 handoff fail closed，并报告具体 lease transition，而不是只给出模糊的 packet corruption。

**Failure condition。** 如果 shadow capability state 引入的 cache traffic 足以抵消 zero-copy 收益，或者 native API 已经能捕获同样的跨路径 fault 并给出等价 provenance，这个机制就不值得采用。

### 2. 把 compact handoff witness 和 policy generation 绑定

**Gap。** Security decision 与 buffer lifetime 往往存在不同 evidence stream。地址反复回收后，单纯用 timestamp join 并不安全。

**Mechanism。** 给每个相关 BPF policy decision 分配一个属于某个 policy generation 的 compact `decision_id`。zero-copy handoff 时保存或输出一个 witness，包含 buffer lease generation、decision ID、源 owner 与目标 owner 类型、device/queue 和 transition sequence。userspace 保存 `decision_id` 到 policy object 或 rule 的 reverse mapping。

witness 默认不包含 packet payload。它可以只在 deny、异常 transition、policy generation 切换，以及少量普通 allow sample 时输出。目标不是 trace 每一个 packet，而是在 buffer 地址已经被复用几百次之后，关键 evidence 仍然不会产生歧义。

**Delta。** 现有 packet trace 可以记录地址、queue event 或 policy verdict。新的性质是 logical buffer lease 与授权 handoff 的 policy generation 之间存在显式 join key。

**Artifact。** witness format、BPF map/event helper、AF_XDP userspace library wrapper，以及一个无需读取 payload 就能重建单次 packet lease 的命令行工具。

**Evaluation。** 用很小的 UMEM 制造极高 reuse rate，同时轮换 policy 和 queue steering。比较 address-only correlation、timestamp correlation 和 generation-scoped witness。统计 false join、missed join、evidence volume 与 root-cause accuracy，并加入同一个 scheduler tick 内复用相同 frame 的 adversarial timing。

**Academic value。** 这把 provenance 变成一个可以测量的 high-performance I/O 与 systems security 边界问题，而不是默认无限制日志就能解决。

**Production value。** operator 可以回答“这些 bytes 在进入 userspace 之前到底被哪一代 policy 放行”，而不需要 full packet capture，也不必永久保存旧 control-plane state。

**Failure condition。** 如果 generation-scoped witness 相比更便宜的 queue-local counter 与 timestamp 并没有明显降低 false attribution，它就只是多余 metadata。

### 3. 在标准化 contract 之前，先做一个跨路径 zero-copy fault benchmark

**Gap。** 设计一套漂亮的 ownership schema 很容易，但必须先证明真实 mixed datapath 确实需要它。

**Mechanism。** benchmark 不单独测每个 API，而是组合 native path。场景包括 XDP 到 AF_XDP、shared UMEM 上的 AF_XDP forwarding、io_uring ZC Rx 的 TCP payload、DPDK external buffer，以及可选的 userspace eBPF policy stage。每个场景先声明正确的 ownership 与 policy-generation sequence，再由 fault injector 一次破坏一个边界。

至少加入这些 fault：

- 同一个 AF_XDP frame 被发布到互相冲突的 ownership path；
- recycle 地址在新的 tenant 或 policy generation 下重新使用；
- queue steering 更新使 flow 离开预期 BPF path；
- userspace 在仍持有 outstanding buffer 时退出；
- completion 延迟时提前复用 frame；
- policy update 过程中发生 NIC reset 或 queue reconfiguration；
- DPDK external buffer 在仍有逻辑 packet reference 时被 detach。

benchmark 比较三个设计：完全未修改的 native API、metadata-only witness，以及 active capability checking。

**Delta。** 现有 API selftest 主要验证单个机制。这里把 **composition failure** 作为评测单位，让 proposed contract 有明确失败条件。

**Artifact。** 可复现 traffic generator、fault injector、reference ownership trace，以及针对 safety、provenance correctness、diagnosis time 和 performance overhead 的 grader。

**Evaluation。** 除了 throughput 和 latency，还要报告 fault-detection recall、false positive、buffer reuse 后的 false policy join、recovery correctness，以及每百万 packet 需要多少 evidence bytes。至少跨两组 NIC/driver 测试，因为 zero-copy 与 DMA 行为依赖硬件。

**Academic value。** benchmark 能把真正的 cross-boundary abstraction gap 和一组普通 implementation bug 区分开。

**Production value。** networking runtime 可以获得覆盖高负载、queue reconfiguration 与高频 buffer reuse 故障的 regression test，而这些问题平时很难稳定复现。

**Failure condition。** 如果真实组合场景没有出现 native API 自己无法检测的新 fault，正确结论应该是改进局部 diagnostics，而不是推动新的通用 contract。

## 什么会改变这个结论？

最强的反对意见是：zero-copy ownership 本来就应该保持 path-specific。AF_XDP ring、io_uring refill state、page_pool reference 和 DPDK mbuf reference count 面向的 execution model 不同。common capability layer 可能增加 cache pressure、重复状态和新的 bug，却只换来一套更复杂的 tracing convention。

如果实验同时证明三件事，这个反对意见就成立：native API 已经能拒绝重要的 cross-path misuse；在真实 reuse rate 下，address 加 timestamp 已经足够可靠；hardware steering 的变化也可以低成本从现有 control-plane state 重建。

反过来，如果 mixed datapath 反复出现这种 failure：每个 subsystem 单独看都合法，但组合起来已经违反 buffer ownership 或 policy provenance，那么缺的就不是另一种 zero-copy transport。真正需要的是一套足够小的 contract，用来说明一份 buffer lease 在 BPF、kernel networking、userspace 与 NIC 之间移动时到底代表什么。
