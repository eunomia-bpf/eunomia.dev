---
date: 2026-08-14
title: "eBPF 可编程能力能在 io_uring 里面走多远？"
description: "当前 Linux 的 io_uring 已经同时出现请求过滤和 eBPF struct_ops 控制路径。本文分析这些机制怎样与静态限制、LSM 安全策略和新型 I/O 对象组合，以及 io_uring 要成为安全的可编程运行时边界还缺什么。"
tags:
  - Daily Report
  - eBPF
  - io_uring
  - Linux
  - I/O
  - Security
  - Runtime
research_question: "当当前 Linux io_uring 同时拥有 ring 内 BPF 请求过滤和 eBPF struct_ops 执行路径后，哪些机制第一次变得可行，以及在让 eBPF 安全控制异步 I/O 之前，权限、上下文、生命周期和评测还缺哪些明确契约？"
source_cutoff: 2026-08-14
status: daily-report
---

# eBPF 可编程能力能在 io_uring 里面走多远？

设想一个存储或网络服务只维护一个 `io_uring`，却替多个逻辑租户提交工作。这个 ring 可能同时负责打开文件、连接 socket、发送设备专用的 `URING_CMD`、把网络 payload 直接收进预注册的用户态内存，甚至承载 FUSE 的文件系统请求。到了这个阶段，`io_uring` 已经不只是把 syscall 批量化得更快。它开始像一个有自己队列、注册资源、调度规则和安全边界的执行环境。

过去如果要约束这些操作，通常会想到两个位置。应用创建 ring 时可以配置静态 restriction，系统级安全策略则可以通过 Linux Security Module 在真正执行文件、网络或凭据相关操作时作最终判断。当前 Linux 源码里已经出现了第三类机制：BPF 程序可以直接运行在 `io_uring` 自己的执行路径里。

<!-- more -->

这里有一个很容易写错、但恰好最值得研究的细节：当前 `io_uring` 里面其实有两套不同的 BPF 机制。

`IORING_REGISTER_BPF_FILTER` 安装的是按 opcode 绑定的 **classic BPF（cBPF）** 过滤器，用来决定一个 submission 能不能进入后续执行。另一边，`io_uring_bpf_ops` 是真正的 **eBPF `struct_ops`** 接口，可以接管 ring 的一次 loop step，并通过 io_uring 专用 kfunc 提交 SQE 或访问受限的 ring memory region。

前者刻意做得很小，接近“对当前请求做一次纯粹的 allow/deny 判断”；后者已经进入执行控制。这个分裂比“Linux 又多了一个 eBPF hook”更重要，因为它说明同一个 I/O 子系统里正在形成几层不同的可编程控制面，而它们的权限、上下文和生命周期并不相同。

因此，真正的问题已经从“eBPF 能不能观察 I/O”变成了：**eBPF 应该怎样参与 I/O 执行，同时不在 Linux 里面再造一套互相打架的安全、调度和资源控制系统？**

本文延续 [eBPF runtime 与 extensibility 系列](https://eunomia.dev/zh/research/async-ebpf-causal-profiler/)。前面的文章分别讨论过用户态 eBPF runtime contract、多个程序共享 hook、状态化应用升级，以及异步执行中的因果关系。`io_uring` 是一个很好的具体例子，因为这几个问题现在开始在同一个真实内核子系统里汇合。

## io_uring 里面已经存在四层不同的控制

先看当前 Linux 已经提供了什么。

| 机制 | 作用范围 | 可编程程度 | 主要用途 |
| --- | --- | --- | --- |
| `IORING_RESTRICTION_*` | 单个 ring | 静态声明式规则 | 限制允许的注册操作、SQE opcode 和 SQE flags |
| `IORING_REGISTER_BPF_FILTER` | 单个 ring + opcode | cBPF 程序 | 查看 submission 上下文并允许或拒绝请求 |
| Linux Security Modules | 系统安全边界 | LSM policy，包括适用场景下的 BPF LSM | 独立于单个 ring 做主机级安全判定 |
| `io_uring_bpf_ops` | 单个 ring | eBPF `struct_ops` + io_uring kfunc | 参与 ring execution loop、提交 SQE、读取允许访问的 ring region |

当前 [io_uring UAPI](https://github.com/torvalds/linux/blob/master/include/uapi/linux/io_uring.h) 仍然保留早期的 restriction 接口。它可以允许某个 registration opcode、允许某类 SQE，或者限制 SQE flags。对于 sandbox 风格的 ring，这很有用，因为能力可以在创建阶段就被收窄。

同一份 UAPI 现在还包含 `IORING_REGISTER_BPF_FILTER`。与静态 allowlist 不同，filter 可以根据当前请求的状态决定是否允许。这自然形成了两层关系：restriction 定义粗粒度的 capability envelope，filter 再判断某一个具体请求是不是满足策略。

而 eBPF `struct_ops` 更进一步。它不是更复杂的 allowlist，而是允许 eBPF 参与一个受支持 ring 的执行推进。

问题在于，这四层机制原本解决的是不同事情。如果生产系统开始同时使用它们，就必须回答几个很实际的问题：谁拥有最终权限，哪一层可以读写什么状态，更新时怎样保证版本一致，以及事后如何解释一条请求到底被哪层允许、拒绝或生成。

## 请求过滤器叫 BPF，但它现在刻意不是 eBPF

当前 [BPF filter UAPI](https://github.com/torvalds/linux/blob/master/include/uapi/linux/io_uring/bpf_filter.h) 定义了 `io_uring_bpf_ctx`。基础字段包括 `user_data`、SQE opcode、SQE flags，以及可选的 opcode-specific auxiliary data。当前的辅助结构已经覆盖了几类有语义的信息：

- `SOCKET` 的 family、type、protocol；
- `OPENAT` / `OPENAT2` 的 flags、mode、resolve；
- `CONNECT` 的地址族、port、IPv4/IPv6 地址。

内核的 [filter implementation](https://github.com/torvalds/linux/blob/master/io_uring/bpf_filter.c) 也把语义写得很直接。过滤器按 opcode 注册，返回 1 表示允许，返回 0 表示拒绝。同一个 opcode 如果挂了多个 filter，所有 filter 都必须通过。拒绝会在请求真正进入队列之前变成 `-EACCES`。

[`io_uring.c`](https://github.com/torvalds/linux/blob/master/io_uring/io_uring.c) 的 submission path 会在 `io_init_req()` 之后执行这些 filter，然后才进入常规的 submit trace 和 queue 路径。这个位置适合 admission control，因为 kernel 已经把 SQE 解释成 request，能准备出一定语义上下文，但 work 还没有逃逸到 worker、poll、device 或 completion 路径里。

这套接口还有一个值得注意的兼容机制。用户态 filter 会声明自己期望的 auxiliary PDU 大小；kernel 也会告诉用户态该 opcode 的实际 size。`IO_URING_BPF_FILTER_SZ_STRICT` 可以在 size 不匹配时直接拒绝注册，`IO_URING_BPF_FILTER_DENY_REST` 则可以把所有没有明确配置的 opcode 设成默认拒绝。

这两个小功能其实已经反映出两个更大的 runtime 需求：上下文版本要明确，默认权限边界也要明确。

最关键的是程序怎么被加载。当前实现构造 `sock_fprog`，然后调用 `bpf_prog_create_from_user()`，因此这个路径是受限的 cBPF filter，而不是通过普通 `BPF_PROG_LOAD` 加载的 eBPF 程序。

对于一个热路径上的小型 admission gate，这个选择是合理的。固定 context 的 cBPF predicate 比带 map、kfunc、共享可变状态和不断扩展 helper surface 的通用 eBPF 更容易分析。

所以，`IORING_REGISTER_BPF_FILTER` 不应该被包装成“任意 eBPF 已经进入 io_uring”。真正更深的 eBPF 边界在另一条路径上。

## eBPF struct_ops 已经开始参与 ring 的 execution loop

当前 [`io_uring/bpf-ops.h`](https://github.com/torvalds/linux/blob/master/io_uring/bpf-ops.h) 定义了 `struct io_uring_bpf_ops`，其中包含 `loop_step` callback 和 `ring_fd`。[`bpf-ops.c`](https://github.com/torvalds/linux/blob/master/io_uring/bpf-ops.c) 把它注册成名为 `io_uring_bpf_ops` 的 BPF `struct_ops` 类型。

这才是真正的 eBPF programmability。实现还为 `BPF_PROG_TYPE_STRUCT_OPS` 注册了 io_uring 专用的 kfunc，其中两个特别重要：

- `bpf_io_uring_submit_sqes()` 可以让程序为这个 ring 提交指定数量的 SQE；
- `bpf_io_uring_get_region()` 可以返回经过 size 检查的指针，让程序访问选定的 io_uring memory region，例如 parameter memory、completion queue region 或 submission queue region。

verifier 还会限制 callback context 的访问范围，region kfunc 在返回指针前也会检查请求长度。因此，这个接口并不是把任意 `io_ring_ctx *` 整个交给 eBPF，而是在做一个小型、subsystem-specific 的 capability surface。

安装条件也说明它目前并不打算覆盖所有 io_uring 模式。当前代码会拒绝带 `IORING_SETUP_SQPOLL` 或 `IORING_SETUP_IOPOLL` 的 ring，要求 `IORING_SETUP_DEFER_TASKRUN`，而且一个 ring 只能装一个 `bpf_ops` instance。BPF object 通过 `ring_fd` 绑定到具体 ring，并随着 `struct_ops` link 生命周期被卸载。

这些限制不应该简单理解成“功能还不完整”。如果目标是建立可维护的内核接口，限制本身就是设计的一部分。我们终于可以讨论一个明确的问题：一个 eBPF 程序被允许参与某个 defer-task-run ring 的执行循环，但它只能调用指定 kfunc，也只能看到明确授权的 region。

从架构上，可以把当前几层关系粗略理解成：

```text
application 提交 SQE
        |
        v
静态 ring restrictions
        |
        v
按 opcode 的 cBPF admission filter
        |
        v
普通 request 初始化 / security / issue 路径
        |
        +----------> worker、poll、device 等异步执行
        |
        v
eBPF struct_ops 控制的 ring loop
        |
        +----------> submit SQE / 读取允许的 ring region
        v
completion handling
```

这不是每一个 opcode 的精确 call graph，但它抓住了设计上最重要的区别。cBPF filter 决定请求能不能进来，eBPF `struct_ops` 可以影响一个受支持 ring 怎样推进执行，而 LSM 和各子系统自己的安全检查仍然是独立的权限边界。

## io_uring 开始承载更多 I/O 对象后，这个问题会放大

如果 `io_uring` 永远只是在批量提交 `read()` 和 `write()`，ring 内执行控制仍然有价值，但边界会相对简单。当前 Linux 正把更多 subsystem-specific 工作放进 ring。

内核的 [FUSE-over-io-uring 文档](https://docs.kernel.org/next/filesystems/fuse-io-uring.html) 描述了用户态 FUSE daemon 使用 `IORING_OP_URING_CMD` 在 FUSE connection 上注册请求条目。注册完成后，kernel 可以把 FUSE request 放到 per-CPU io_uring queue，daemon 返回结果时同时获取下一条请求。文档也明确说这套接口仍在开发中，并没有覆盖所有 request 类型。

这意味着 ring 已经可以成为文件系统控制路径的 transport，而不只是“应用主动发起 syscall 的队列”。

[io_uring zero-copy receive 文档](https://docs.kernel.org/networking/iou-zcrx.html) 又移动了一层边界。packet header 仍由普通 kernel TCP stack 处理，但 payload 可以直接进入预注册的用户态内存。flow steering 和 RSS 负责把 NIC receive queue 对接到 zero-copy path，应用还要通过 io_uring refill ring 回收 buffer。

于是一个 ring 绑定的对象不再只有 syscall 参数，还包括 NIC queue、registered memory、refill metadata、multishot receive state 和应用自己的 buffer lifetime。

block 层也在发生类似变化。[ublk 文档](https://docs.kernel.org/block/ublk.html) 描述了基于 io_uring command 的 userspace block server，较新的 zero-copy 模式还会使用 registered kernel buffer。trusted userspace server 需要在服务 client I/O 时维护 buffer correctness。

这些例子让 eBPF `struct_ops` 的意义变得更大。一个能控制 ring loop 的 eBPF 程序，未来可能非常靠近文件系统、网络、block 或设备专用的执行路径。某个在 microbenchmark 里看起来无害的 kfunc，一旦 ring 持有大量 registered resource，就可能变成真正的权限或资源成本边界。

因此，正确抽象不应该是“把更多 io_uring internal 暴露给 eBPF”，而应该说明程序到底控制哪些 I/O capability，哪些系统安全决定永远不属于它。

## eBPF 应该和 LSM 组合，而不是再造一套平行安全栈

当前 [`io_uring.c`](https://github.com/torvalds/linux/blob/master/io_uring/io_uring.c) 会在 ring setup 期间调用 `security_uring_allowed()`，io_uring 在 credential 和具体 operation path 上也有独立安全检查。更大的原则是，LSM 用来表达跨子系统的 host-level security policy，而不是只管理某一个 file descriptor。

ring-local BPF 解决的事情不一样。它可以从“这个应用、这个 ring”的视角做更窄的约束。例如，一个服务可能希望某个 tenant-facing ring 只能连接特定地址范围、禁止带某些 flag 的 file open，或者使用一个自定义 deferred-task execution policy。即使 host LSM 允许整个进程执行这些底层操作，这种局部收窄仍然有价值。

真正危险的设计，是让 ring-local eBPF 重新解释系统权限。如果 struct_ops program 能通过一条不会经过同等 LSM 检查的新路径生成工作，programmability 就可能变成 bypass surface。反过来，如果每一个 ring-local decision 都要重新跑一遍大型 host policy engine，hot path 的价值又可能消失。

更清楚的关系应该是非对称的：

1. **LSM 仍然是更高层的安全权限。** ring-local program 可以进一步限制或调度，但不能授予 host policy 已经拒绝的操作。
2. **cBPF filter 做低成本 request admission。** 小 context + allow/deny 语义适合做窄 predicate。
3. **eBPF struct_ops 只在明确授权的 ring mode 里参与 execution policy。** kfunc 和 memory access surface 要保持 capability-limited、可审计。
4. **静态 restriction 在需要时定义不可随意扩大的 capability envelope。** 动态程序不应该悄悄把一个本来故意收窄的 ring 权限重新放大。

Linux 已经有这些关系的零件，但还缺一个足够明确的机器可读契约，让 application author、verifier 和 operator 都能依赖它。

## 现有工作还薄弱在哪里

### 不同 opcode 能拿到的语义上下文差异很大

cBPF base context 总有 opcode、SQE flags 和 `user_data`。当前 [`opdef.c`](https://github.com/torvalds/linux/blob/master/io_uring/opdef.c) 还给 `CONNECT`、`OPENAT` / `OPENAT2`、`SOCKET` 添加了 semantic filter payload，但很多其他 operation 并没有对应上下文。

于是，同一个高层策略在不同 opcode 上可能有完全不同的可实现性。filter 能看到 `CONNECT` 的目标地址，但如果以后要约束 `URING_CMD`、zero-copy receive resource、registered file 或设备专用字段，现有 context 可能根本没有对应语义。

这里缺的并不是一句“再加几个字段”，而是一条稳定原则：**哪些语义状态适合在 admission boundary 被暴露，而且能跨 kernel version 保持可解释性。** 当前 PDU size negotiation 是一个不错的兼容性起点，但还没有定义新字段怎样获得长期语义，也没有解决 policy portability。

一个很直接的实验，是把“这个 ring 只能访问 tenant T 的资源”这类策略同时应用到 file、socket、FUSE、ublk 和 zero-copy receive。如果每个 opcode 最后都要写完全不同的 ad hoc parser 或依赖额外 privileged side channel，那么 context model 还不够统一。

### 多层控制面缺少共同的优先级与 provenance

静态 restriction、cBPF filter、LSM 和 eBPF `struct_ops` 各自回答不同问题，但它们的关系主要隐含在 kernel code path 里，没有形成一个可查询的 policy graph。

生产环境很快会问这些简单问题：

- 这条 SQE 到底被哪条规则拒绝？
- 当时生效的是哪个 ring-local policy 版本？
- 这个请求是 userspace 提交的，还是 eBPF loop 生成或推进的？
- BPF 生成的请求仍然经过哪一层 LSM 判断？
- 一个 filter set 从别的 restriction object copy 过来后，当前 lineage 是什么？

cBPF filter implementation 本身已经有 filter set 的 copy-on-write 行为，而 eBPF struct_ops 有另一套 link lifecycle。每一部分单独看都能理解，但没有共同的 provenance record。

如果无法稳定回答“谁允许、谁拒绝、谁生成、当时是什么版本”，programmability 可能让控制更灵活，却让 incident 更难解释。

### 一个 ring 的可编程状态更新还不是事务

前面的 [有状态 eBPF 应用升级](https://eunomia.dev/zh/research/stateful-ebpf-transactional-upgrade/) 已经讨论过通用的 transactional upgrade 问题。`io_uring` 把这个问题变得更窄，也更容易进入生产 hot path。

一个 ring 可能同时依赖静态 restriction、一组 cBPF opcode filter、一个 eBPF struct_ops link、registered file、memory region、personality、buffer ring、FUSE entry 或 zero-copy network resource。单独更新其中一个对象，即使每个 API 自己都正确，也可能产生一个任何正式 generation 都没有定义过的临时状态。

例如，operator 可能先安装了新 execution policy，但匹配的新 filter context 还没准备好；或者 resource registration 已经切到新布局，旧 struct_ops program 却仍按老布局理解 region。

这里缺的是跨**相关 programmable I/O state** 的 generation boundary。它最终应该放在 io_uring kernel API 里，还是由 userspace controller 管理，可以继续讨论，但评测一定要覆盖并发 submission 和更新中途失败。

### 新的 execution boundary 已经超过传统 eBPF accounting

当 eBPF 能驱动 ring loop，而 ring 又持有大量 registered resource 时，只统计 BPF instruction count 不够。程序可能触发 submission、长期占用注册内存、推进 FUSE queue，或者改变用户态和 kernel 之间的交互频率。

至少需要归属这些成本：

- eBPF 执行时间和 invocation count；
- userspace 原始提交与 BPF-controlled loop 生成或推进的 SQE；
- ring 可访问 registered memory 的字节数和 lifetime；
- completion queue pressure，以及 deferred / dropped work；
- FUSE queue entry、zero-copy receive buffer 等 subsystem-specific resource。

否则，一个程序可以 verifier-safe，却仍然在生产系统里非常昂贵。这和 [用户态 eBPF runtime contract](https://eunomia.dev/zh/research/userspace-ebpf-runtime-contract/) 讨论的区别一样：memory safety 不等于 runtime resource authority。

## 哪些方向值得继续做

### 1. 给 programmable io_uring 一个类型化 capability contract

**缺口。** 现有接口混合了静态 restriction、cBPF context field、eBPF `struct_ops` callback 和 kfunc-accessible region，却没有一个统一描述告诉系统“这个 ring-local program 到底能观察什么、能造成什么 effect”。

**机制。** 为 programmable ring 定义机器可读的 capability descriptor。它声明允许的 SQE class、semantic context field、registered-resource class、eBPF kfunc、可读写 region，以及是否允许 BPF-originated submission。descriptor 与 ring 绑定，userspace 可以查询。eBPF 侧可以继续用 BTF 描述类型，而 cBPF gate 保留现有 PDU-size negotiation 作为兼容机制。

关键不是强行把 cBPF 和 eBPF 合并成一种程序模型，而是让两者共享一套 capability vocabulary，同时保留不同 trust model。例如 descriptor 可以说：`CONNECT` admission 允许 cBPF 看到 destination family/address/port，而 eBPF loop 只能提交 opcode A/B，并且只能读 CQ region。

**与现有机制的差别。** 这不是单纯增加 filter field 或 kfunc，真正的 artifact 是约束并解释这些接口的 contract。

**实现与评测。** 可以先围绕现有 io_uring query/registration API 做 prototype，再配一个 liburing inspection tool。用 file、network、FUSE、ublk workload 检查同一个高层 capability policy 能否跨 opcode 映射，统计需要多少 kernel-version conditional，并测试 verifier/controller 能否在 activation 前拒绝不兼容的 program/ring 组合。

**学术价值。** 这是一个更一般的问题：subsystem-specific eBPF runtime 怎样暴露异构 capability，而不退化成不稳定的 kernel-internal API。

**生产价值。** operator 在 attach 前就能查询 ring-local program 的真实权限，不必把 BPF source 和内核实现一起人工审计。

**失败条件。** 如果 descriptor 最终只是照抄几十个不稳定 implementation field，而且没有减少兼容代码，它只增加了管理负担。

### 2. 用 versioned ring policy generation 明确权限顺序

**缺口。** 一个 ring 的有效策略分散在独立更新的对象里，而且 precedence 隐含在执行路径中。

**机制。** 把相关 restriction、cBPF filter、eBPF struct_ops 和 registered-resource assumption 视为一个 versioned policy generation。新 generation 先离线构造，验证和目标 ring capability 匹配后，再通过一次 generation switch 激活。请求进入 ring 时记录自己的 generation；即使 policy 在请求完成前已经更新，completion 和 audit record 仍保留原 generation。

同一个 generation 还显式定义 authority ordering：静态 restriction envelope 先约束能力，host LSM 始终保持有效，ring-local cBPF admission 在其下进一步收窄，eBPF execution policy 只能在授予的 capability 里工作。BPF 生成的 submission 继承同一个或更窄的 generation，不能凭空产生新的 authority chain。

**与现有工作的差别。** 前面的 transactional-upgrade report 讨论通用 stateful BPF application。这里的 testcase 更严格，因为它必须处理 live async I/O queue，新旧 request 会同时在飞。

**实现与评测。** 可以先在 userspace controller 里实现 generation tagging，再判断是否真的需要一个小型 kernel atomic-activation primitive。用 fio、network proxy、FUSE 和 ublk 持续提交请求，同时注入 process crash 和 update failure。指标包括 unauthorized window、request loss、generation ambiguity、rollback latency 和 hot-path overhead。

**学术价值。** 它把 policy versioning 与 asynchronous execution 的 concurrency 问题变成可验证对象。

**生产价值。** 服务可以更新 ring-local policy，而不必先完全 drain queue，也不会接受一个很难审计的混合配置窗口。

**失败条件。** 如果普通 link replacement 配合 application-level quiescence 已经能做到接近零停顿，也不会产生可测的 inconsistent window，就没有必要新增 generation primitive。

### 3. 建一个 in-ring eBPF control 与外部 hook 的对照 benchmark

**缺口。** 很容易宣称 ring 内 eBPF 比 LSM、tracing、userspace validation 或静态 constraint 更快、更有语义。但现在缺少一个标准 workload 告诉我们，额外的 control surface 到底什么时候值得。

**机制。** 同一组 policy 分别放到几个 boundary 上实现：

1. SQE submit 前的 userspace validation；
2. 能表达时使用静态 `IORING_RESTRICTION_*`；
3. cBPF `IORING_REGISTER_BPF_FILTER`；
4. LSM-relevant host security boundary；
5. 对需要 ring loop control 的 workload 使用 eBPF `io_uring_bpf_ops`；
6. 只观察、不 enforcement 的 tracing baseline。

policy 要故意选择那些能拉开机制差异的场景，例如 destination-aware socket admission、file-open constraint、tenant-specific registered buffer、completion pressure 下的 request scheduling，以及根据负载自适应 batch submission 的 BPF-controlled loop。

**实现与评测。** 除 microbenchmark 外，再加入真实 network、storage、FUSE 和 ublk service。测 per-I/O latency、throughput、CPU cycles、cache effect、policy precision、bypass attempt、overload tail latency 和 update behavior。还要做几个 ablation：去掉 semantic PDU、去掉 region kfunc access、去掉 BPF-originated submission。

**学术价值。** 这个 benchmark 可以回答什么控制决策真正值得放进 asynchronous I/O runtime，而不是在它前后挂 hook。

**生产价值。** kernel/runtime maintainer 能据此判断一个新的 io_uring BPF context field 或 kfunc 是否解决真实部署问题。

**失败条件。** 如果外部 hook 和 userspace validation 在真实 workload 上同时达到同样的开销和 policy precision，正确方向就是保持 io_uring BPF surface 足够小。

## 目标应该是窄而明确的可编程边界，而不是无限扩张的内核插件

当前 Linux 源码已经足以排除两个极端判断。

第一个极端认为 io_uring 只需要 observability。这个判断已经过时，因为当前 kernel 同时存在 submission-time BPF decision path 和 eBPF execution-control interface。

第二个极端则认为，既然已经有 eBPF struct_ops，就应该继续扩成能访问任意 io_uring internal 的通用 in-kernel runtime。当前实现反而说明边界控制很重要。cBPF filter 刻意只拿小型 predicate context，eBPF path 只暴露经过 BTF/verifier 检查的 callback state 和 selected kfunc，struct_ops 也限制可 attach 的 ring mode。

如果目标是可维护的 subsystem interface，这些限制应该被当作 feature。

更合理的设计目标是一个**窄而权限明确的 programmable boundary**：

- admission 需要语义时，暴露稳定而有限的 semantic context；
- 只有当 ring 内控制具有可测价值时，才增加小型 eBPF execution interface；
- host security policy 始终保持最高权限；
- lifecycle 和 provenance 能跨多层对象做 versioned tracking；
- resource accounting 能衡量 verifier safety 之外的真实成本。

`io_uring` 因此也成为一个更一般的 eBPF case study。当 BPF 从“观察某个 subsystem”走向“参与 subsystem control loop”，最重要的接口问题会从“哪里有 hook 可以 attach”转成“这个 attachment 到底授予了什么 capability，它又怎样和系统其他控制面组合”。

## 什么证据会改变这个结论？

本文的判断依赖一个前提：ring 内 decision 至少在 overhead、semantic context 或 execution control 中有一项明显优于放在 io_uring 外部的 policy。

上面的 benchmark 完全可能证明这个前提不成立。如果静态 restriction、现有 LSM 和 userspace validation 已经覆盖绝大多数生产 policy，并且性能没有明显差距，那么 io_uring 应该把新的 BPF surface 保持得很窄。如果 eBPF struct_ops 只有暴露大量不稳定 ring internal 才能提供有用 control，那它更适合作为实验性 optimization hook，而不是稳定 runtime boundary。如果 cBPF filter 在极高 IOPS 下也出现明显 hot-path cost，正确方向可能是把更多 decision 前移到静态 capability，而不是把 filter 做得更复杂。

反过来，如果 FUSE、zero-copy networking、ublk 和其他 ring-centric workload 反复需要外部 hook 无法低成本表达的 request semantic 与 execution control，那么 `io_uring` 就提供了一条很具体的 subsystem-native eBPF programmability 路径。那时下一步不该继续堆更多 hook，而应该把 capability、authority、update、provenance 和 resource cost 做成一套能长期依赖的契约。