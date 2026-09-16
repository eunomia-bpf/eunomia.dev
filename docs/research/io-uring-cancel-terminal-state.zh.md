---
date: 2026-09-16
slug: io-uring-cancel-terminal-state
title: "io_uring 取消 I/O 后，这个操作真的结束了吗？"
description: "io_uring 取消与 I/O 完成存在竞争。本文分析 CQE、关闭文件描述符与硬件在途请求，并提出终态回执、资源生命周期栅栏和可复现的取消竞争安全测试。"
tags:
  - Daily Report
  - Linux
  - io_uring
  - 异步 I/O
  - 取消语义
research_question: "io_uring 运行时应如何区分取消请求和目标操作的真正终态，避免过早复用缓冲区、文件相关状态或逻辑请求身份？"
source_cutoff: 2026-09-16
status: daily-report
---

# io_uring 取消 I/O 后，这个操作真的结束了吗？

一个服务器提交了 `io_uring` 读请求，连接开始关闭，于是运行时又提交 `IORING_OP_ASYNC_CANCEL`。取消请求的 CQE 返回成功。此时能不能立刻复用原来的 buffer、释放连接对象，并把这次 I/O 当成“没有发生过”？

不能把这件事当成默认成立。

`io_uring` 的取消本身就是一场竞争：目标 I/O 可能已经正常完成，可能正处于无法立即取消的完成阶段，也可能已经下发到硬件。应用关闭自己的文件描述符，也不等于所有使用这个 fd 提交的 ring 请求都会自动消失，因为 `io_uring` 请求可以继续持有自己的 file reference。

所以更稳妥的边界是：**取消请求只是尝试改变目标操作接下来会不会继续执行；目标操作自己的最终 completion，才是判断它是否真正结束、资源是否可以回收的依据。**

<!-- more -->

## cancel CQE 和目标 CQE 回答的是两个不同问题

liburing 当前的取消文档把这两个事件明确分开。取消操作本身会产生一个 CQE：返回 `0` 表示找到了目标并完成了可执行的取消路径，`-ENOENT` 表示没有找到匹配请求，`-EALREADY` 则表示请求找到了，但它已经进入完成过程。

与此同时，原来的 I/O 请求还会有自己的 CQE。取消真正获胜时，目标通常以 `-ECANCELED` 结束；如果目标先赢了竞争，它完全可能返回正常结果或另一个错误。两个 CQE 的先后顺序也没有保证。

这不是 API 的偶然细节，而是两个不同状态机。cancel CQE 说明“取消动作发生了什么”，target CQE 说明“原操作最终怎么结束”。

当前 Linux 的 `io_uring/cancel.c` 也保留了这种区别。对 io-wq 中的请求，已经运行的工作会映射为 `-EALREADY`，找不到则是 `-ENOENT`；随后内核还会继续检查 poll、waitid、futex、timeout 等不同取消路径。不同 opcode、不同执行阶段并不存在一个统一的“已经停下”瞬间。

另一个容易踩坑的地方来自同步 I/O 的习惯。liburing 文档明确说明，`close(fd)` 不会自动取消已经提交的 `io_uring` 请求。请求已经拿到了自己的 file reference，因此应用侧那个整数 fd 消失之后，内核里的 I/O 仍然可能存活。

`io_uring_register_sync_cancel()` 提供了同步取消接口，适合某个线程需要等待取消处理完成的场景。但“同步”等待取消处理，并不能让已经跨过不可撤销边界的操作倒退。上游文档也明确指出，已经提交到硬件的磁盘 I/O 通常无法再取消。

对运行时来说，至少应该保留下面这些状态，而不是压成一个 `cancelled=true`：

| 运行时状态 | 已知事实 | 可以做什么 |
| --- | --- | --- |
| 已请求取消 | cancel SQE 已提交 | 继续保留目标请求及其资源 |
| 取消已接受 | 找到了目标，并进入支持的取消路径 | 仍然等待目标终态 |
| 目标正在完成 | cancel 返回 `-EALREADY`，或已经观察到同类竞争 | 等待 target CQE |
| 目标已终结 | 原请求已经给出最终 CQE，并满足该 opcode 的结束条件 | 才能按契约回收或复用资源 |

对于 multishot 请求，终态还要包括最后一个 CQE 不再带 `IORING_CQE_F_MORE`。如果运行时看到取消相关事件就释放对象，后面合法到来的 completion 可能落到已经复用的新对象上。

这和之前 [AI Agent 重试与副作用幂等性的报告](https://eunomia.dev/zh/research/agent-tool-retry-effect-idempotency/) 有一点相似，但层次不同。Agent 场景的问题是外部副作用可能已经提交，而响应丢了；这里内核其实提供了明确的 completion 通道，真正容易出错的是用户态把“取消控制流”和“目标操作结束”混为一谈。

## 常见的几个运行时捷径为什么会出错

第一种做法是：用 `user_data` 把 pending operation 放进表里，发出 cancel，cancel CQE 一成功就把表项删除。表看起来清爽了，但资源生命周期并没有结束。原请求仍然欠着一个 completion；此时复用 buffer、request object 或 generation slot，后到的 CQE 就可能被错误归到新的请求上。

第二种做法是把 `close(fd)` 当成 teardown barrier。对 `io_uring` 来说，这个边界不成立。应用的 descriptor lifetime 和 ring request lifetime 有关系，但不是同一个生命周期。

第三种做法是看到 `-ENOENT` 就理解成“这个 I/O 从来没有发生”。实际上它也可能表示目标在 cancel lookup 之前就完成了。如果 target CQE 还没被消费或对账，`-ENOENT` 不能证明什么都没发生。

第四种做法是把 timeout 当成 effect rollback。linked timeout 可以帮应用触发取消，但 timeout 和目标 I/O 仍然会竞争。对于网络发送、存储写入或设备命令，应用必须另外定义“成功完成”到底产生了什么外部效果，以及每一种终态之后哪些资源才能释放。

## 现有研究和工程实践还缺什么

第一个缺口是**跨 opcode 的用户态终态契约**。`io_uring` 本身保留了不同操作的真实行为，但语言运行时、网络框架和存储引擎往往会统一包装成 future、promise 或 callback。这个统一抽象需要回答一个很具体的问题：什么时候 buffer、registered resource、file-related state 和逻辑 request identity 才真的不会再被内核触及？一个有价值的测试应该让通用运行时在高频 cancel/reuse 下配合 ASan、generation-tagged buffer 运行，检查是否出现 stale completion 归错对象。

第二个缺口是**和外部效果关联的取消语义**。target CQE 能说明内核请求怎么结束，但应用真正关心的是更高层的效果：数据是否已经发送、write 是否已经进入更低层、设备命令是否跨过不可逆边界、multishot source 是否还能产出事件。不同 opcode 和 backing object 的答案并不一样，因此评估不能只数 `-ECANCELED` 有多少，而要故意制造 cancel race，并观察外部效果。

第三个缺口是**可解释的 teardown 证据**。线上运行时出了问题之后，维护者需要知道某个资源为什么被释放：正常完成、取消获胜、ring shutdown drain，还是请求进入不可取消阶段后又晚一些完成。只留下 `user_data` 和结果码，在 identifier 被复用、或者请求是 multishot 时，往往不够恢复真实生命周期。

第四个缺口是**跨内核、跨 opcode 的取消竞争 benchmark**。请求可能在 poll、io-wq、协议栈或硬件里，不同阶段的可取消性完全不同。只测一个阻塞 socket read，无法说明文件 I/O、multishot network operation、`uring_cmd` 或已经下发到设备的请求。benchmark 应该按操作类型和执行阶段统计 race outcome，而不是给出一个总的 cancellation success rate。

## 兼具学术价值与生产价值的方向

### 1. 给每个异步操作生成“终态回执”

**缺口。** cancel CQE 和 target CQE 各自只描述一半事实，通用运行时却需要一个可以用于资源回收和事后分析的完整结论。

**机制。** 为每个逻辑 operation 分配带 generation 的稳定身份，在目标真正终结前保留一份很小的状态记录。记录里包括 opcode、持有的资源、cancel attempt、cancel CQE、target CQE、`IORING_CQE_F_MORE` 以及 operation-specific effect classification。状态只能单向前进：cancel 成功可以把请求推进到 `cancel-pending`，但只有 target terminal evidence 才能推进到 `retirable`。

例如：

```text
operation = {slot, generation}
opcode = RECV_MULTISHOT
resources = {fixed_file_slot, buffer_group}
submit_seq = 1842
cancel_attempt = {seq=1901, result=0}
target_terminal = {result=-ECANCELED, more=false}
effect = no_additional_payload_after_terminal
retire_after = target_terminal
```

**和现有方案的区别。** 这不是再造一个 kernel cancel primitive。内核已经提供了需要的 completion。新增的东西是运行时层面的不变量：把取消控制流和目标 completion 合并成一份 retirement proof，而不是让每个 callback 自己猜资源何时安全。

**可实现产物。** 做一个兼容 liburing 的轻量 runtime shim，加上 trace/replay 格式，覆盖普通 one-shot、multishot、按 fd 取消、linked timeout 和 sync cancel。

**评估。** 和普通 future/callback wrapper 对比，在 socket read/write、poll、file I/O 和 multishot operation 上调整 cancel 与 completion 的相对时序。主要测 stale completion 误归因、过早资源复用、memory-safety failure、每个 live request 的状态开销、completion latency 和吞吐。关键 ablation 是去掉 generation identity；如果 slot 或 `user_data` 重用后开始出现错误 join，就能证明 generation 不是装饰。

**学术价值。** 可以研究一个通用问题：取消和完成分离的异步操作，是否能定义跨 operation class 的可组合终态性质。

**生产价值。** 网络运行时、存储引擎、语言 runtime 和 proxy 都可以在自己的 `io_uring` abstraction boundary 使用这份回执，让 teardown 可审计，并避免内核尚未结束时复用对象。

**失败条件。** 如果所有支持的 one-shot 和 multishot operation 都只需要“等待一个 target CQE 再 free”就能达到同样安全性，而且 generation 和记录没有发现任何额外错误，那么这套回执就是多余开销。

### 2. 把资源回收做成 fence，而不是 callback 里的约定

**缺口。** 即便 cancellation state machine 写对了，只要 buffer pool、fixed-file table、connection object 或 request arena 提前复用，同样可能出错。

**机制。** 给资源 generation 增加一个轻量 retirement fence，只要还有引用它的 operation 没有 target terminal evidence，这一代资源就不能被重用。取消只能表示“应用已经不想要这项工作继续”，真正解除 kernel ownership 的仍然是终态 completion。

普通请求收到一次最终 CQE 即可释放对应 fence reference；multishot 则等到最后一个没有 `IORING_CQE_F_MORE` 的 CQE。shutdown 时也不需要把 `close(fd)` 当隐式 barrier，而是直接统计这个 scope 里还有哪些 operation identity 未终结。

**和现有方案的区别。** 普通 reference count 只有在每个 subsystem 都准确拿住并释放引用时才有效。generation-aware fence 把“异步内核仍然可能回来访问/完成这项工作”变成显式边界，因此 stale CQE 不会因为 slot 恰好复用了就被默默接到新对象上。

**可实现产物。** 做一个带 generation slot 的 connection/request arena，并给 registered buffer、fixed file 等资源加适配器，再接入一个小型 echo proxy 和 file-I/O worker。

**评估。** 压测 connect/cancel/close/reuse、buffer ring recycling、fixed-file slot reuse 和 ring teardown。baseline 是传统 callback lifetime 管理，以及没有 generation check 的普通 reference counting。测 stale CQE 接受数、额外保留内存、teardown latency、slot reuse delay 和 steady-state throughput。再故意让某个 completion 延迟到 slot 已经尝试复用之后，检查 fence 是否能拒绝旧 generation。

**学术价值。** 可以验证 asynchronous kernel ownership 是否适合表达成一种可组合的 lifetime capability，而不是库内部零散 cleanup 规则。

**生产价值。** 它直接针对高性能运行时最容易出问题的阶段：大量连接同时断开、timeout 爆发、滚动 shutdown，同时系统又在 aggressively reuse buffer 和 request object。

**失败条件。** 如果普通 reference counting 加“等 target CQE”已经在同样的 reuse 压力下达到同等安全，而且内存和 latency 更低，就不应该引入 generation fence。

### 3. benchmark 不测“取消成功率”，而测 race outcome 和外部效果

**缺口。** “90% cancel 返回成功”不是安全指标。cancel 输掉竞争并不代表错误，只要正常 target completion 被正确处理；反过来，大量 `-ECANCELED` 也不能证明运行时没有提前复用资源。

**机制。** 建一个 adversarial harness，主动控制 submission、执行、cancel、`close(fd)`、资源复用和 target-CQE consumption 的相对时序。每次运行同时记录 kernel-visible terminal result 和 operation-specific external-effect oracle。

网络 I/O 可以由 peer 记录实际收到的 bytes；文件写入可以在终态后和强制 shutdown 后回读；poll/multishot 可以统计 final CQE 前后产生的事件；io-wq 路径可以设置可控 blocking point，把 `-EALREADY` 的竞争窗口放大。

**和现有方案的区别。** liburing 自己已经有大量 regression/unit tests。这里要测试的不是内核 API 是否通过回归，而是**用户态 lifetime/effect contract 在恶意时序下是否仍然正确**。`0`、`-ENOENT`、`-EALREADY`、target success、target error、`-ECANCELED` 都只是 oracle 的输入，不是最终分数。

**可实现产物。** 发布 workload matrix、race scheduler、ground-truth effect collector 和紧凑 outcome schema，能够在 CI 里跨内核版本运行。

**评估。** 对多个 kernel version 和多类 operation 使用固定随机种子加针对性 race window。分别报告 forbidden resource reuse、completion 误归因、effect disagreement、cancel latency 和复现率。还要加入完全不做 cancellation 的 baseline，确认 harness 自己没有制造 lifetime bug。

**学术价值。** 这是一个取消语义的测量方法：不再假设取消是单一原子事件，而是直接研究不同执行阶段的竞争结果。

**生产价值。** runtime maintainer 可以把 kernel/liburing 升级以及 teardown-path 改动放到同一套 race corpus 下做发布门禁，让测试更接近真实的 connection storm 和 timeout-heavy workload。

**失败条件。** 如果现有 liburing regression tests 配合 sanitizer 已经能稳定发现所有这套 harness 能找到的 contract violation，就没有必要再维护独立 benchmark。

## 现在写运行时，应该遵守什么规则？

把 cancellation 当成 control flow，不要把它直接当成 resource reclamation。

给 cancel SQE 使用独立的 `user_data`，避免 cancel CQE 和 target CQE 混淆。target identity 和它占用的资源一直保留到目标真正进入 terminal completion。`-ENOENT` 只能解释为“取消时没有找到”，不能推导成“没有 completion”；`-EALREADY` 则应直接进入等待 target CQE 的路径。multishot 请求要等最后一个不带 `IORING_CQE_F_MORE` 的 CQE，再释放状态。也不要用 `close(fd)` 代替 ring request 的取消屏障。

如果 operation 会产生外部效果，还要把 effect lifetime 和 kernel request lifetime 分开定义。cancel race 之后正常完成的 target 仍然是真实完成，应用协议必须处理它。之前 [Linux 原子写与崩溃一致性的报告](https://eunomia.dev/zh/research/linux-atomic-write-crash-semantics/) 讨论过类似的系统性错误：一个局部机制不能被自动提升成更大的应用契约。这里局部机制是 cancellation，而更大的契约是安全回收资源并正确对账外部效果。

## 哪些结果会改变这个判断？

如果未来 `io_uring` 提供一个对所有 opcode 和 backend 都成立的统一取消 primitive，并且它成功返回时能保证目标之后绝不会再 completion、不会再持有任何用户可见资源，也不会留下未对账的外部效果，那么运行时就不需要自己组合这么多状态。

对于操作集合很窄的应用，这个判断也会弱化。例如 runtime 只支持 one-shot pollable read，从不在消费 target CQE 前复用 identity，也没有 multishot 和 device I/O，那么一个很小的 state machine 可能已经够用。终态回执和 generation fence 必须在真实的 reuse/cancel race 下证明价值，不能只因为抽象看起来整齐就加入系统。

最后，这些研究方向本身也必须允许被推翻。如果在 socket、file、poll、multishot 和 device-backed I/O 上做受控竞争测试后，普通的“cancel，然后等 target CQE”包装始终没有 lifetime 或 attribution failure，而且更复杂状态也没有提供额外诊断价值，就应该放弃额外机制。

因此真正值得记住的是：**cancel request 只是尝试阻止未来工作；target operation 的最终 completion，才证明原异步操作已经结束。运行时需要把这条边界保留下来。**

## 参考资料

- liburing，[`io_uring_cancelation(7)`](https://github.com/axboe/liburing/blob/master/man/io_uring_cancelation.7)，上游手册，访问于 2026-09-16。
- liburing，[`io_uring_register_sync_cancel(3)`](https://github.com/axboe/liburing/blob/master/man/io_uring_register_sync_cancel.3)，上游手册，访问于 2026-09-16。
- Linux kernel，[`io_uring/cancel.c`](https://github.com/torvalds/linux/blob/master/io_uring/cancel.c)，当前取消实现，访问于 2026-09-16。
- Linux kernel UAPI，[`include/uapi/linux/io_uring.h`](https://github.com/torvalds/linux/blob/master/include/uapi/linux/io_uring.h)，取消 flags 与 CQE 定义，访问于 2026-09-16。
- Jens Axboe，[Reliable cancelation or wait for completion of specified SQE](https://github.com/axboe/liburing/discussions/608)，liburing 维护者讨论，2022-06-17。