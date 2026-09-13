---
date: 2026-09-13
slug: io-uring-checkpoint-recovery
title: "使用 io_uring 的进程，检查点究竟必须保存什么？"
description: "io_uring 检查点不能只保存进程内存，还必须处理飞行中的请求、ring 资源、完成事件与外部副作用，才能避免恢复后丢结果或重复执行。"
tags:
  - Daily Report
  - Linux
  - io_uring
  - Checkpoint Restore
  - Async I/O
research_question: "怎样保存和恢复 io_uring 应用，才能避免丢失完成事件、重复外部副作用，以及把 ring 内资源索引重新绑定到错误对象？"
source_cutoff: 2026-09-13
status: daily-report
---

# 使用 io_uring 的进程，检查点究竟必须保存什么？

给普通进程做 checkpoint 时，我们通常会先停住线程，再保存内存、寄存器、文件描述符以及其他内核对象，恢复时把这些状态重新建立起来。到了 `io_uring`，这幅图就不完整了，因为应用的一部分活跃状态正处在用户态和内核之间。

一个已经提交的请求可能仍然持有内核里的文件引用，即使应用已经关闭了普通文件描述符；一个注册缓冲区可能因为还有 I/O 在使用而继续被 pin；一个 multishot accept 可以产生多次完成事件后继续保持活跃；一次写入或网络操作的外部副作用也可能已经发生，只是用户态还没有消费对应的 CQE。更麻烦的是，有些请求进入硬件后已经无法取消，取消动作本身也存在竞态。

因此，`io_uring` checkpoint 不能定义成“把 SQ 和 CQ 所在的内存映射一起拷走”。真正需要回答的是：哪些异步请求属于这个检查点，哪些资源仍由 ring 持有，哪些副作用已经发生，哪些完成事件已经被应用看见，以及恢复后哪些操作允许重放。

这里的工程结论很明确：**正确的 io_uring 检查点需要按操作定义恢复语义，而不只是逐字节复制进程状态。** 否则恢复后可能丢掉一个已经完成的结果，重复一次写入或网络动作，把 fixed-file 索引绑定到错误对象，或者过早复用仍被旧请求占用的缓冲区。

<!-- more -->

## 为什么冻结内存还不够完成 io_uring checkpoint

`io_uring` 的价值之一，就是把工作从同步 syscall 的边界里移出去。用户态准备 SQE，内核随后消费请求，再通过 CQE 报告结果。性能因此更好，但 checkpoint 系统也必须区分更多中间状态。

假设一个服务器已经提交了 multishot accept，注册了一组 fixed files，并使用 provided-buffer ring 接收网络数据。冻结进程的那一刻，一条连接可能已经被 accept 并放进 direct descriptor slot，只是应用还没消费 CQE；另一条连接仍在同一个 multishot 请求里等待；一次 receive 也可能已经从共享 buffer pool 里拿走了一个 buffer ID。单纯复制用户态内存，并不能说明这些状态转移里哪些已经成为真实的内核状态或外部状态。

liburing 当前文档把这些边界写得很清楚。Registered files 不是进程 fd table 的简单副本，内核会长期持有自己的文件引用，应用甚至可以关闭原来的 fd；direct descriptor 还可以只存在于 ring 的 registered file table 中。Registered buffers 会被 pin，并在飞行中的请求结束之前继续有效，即使应用已经开始替换或注销它们。Provided-buffer ring 又增加了一层 ownership 转移：内核会消费 buffer ID，并在 CQE 中告诉应用最终选择了哪个缓冲区。

“先把所有请求 cancel 掉”也不能把问题简化掉。`IORING_OP_ASYNC_CANCEL` 与正常完成之间存在竞态，取消请求和原请求对应的 CQE 也没有固定先后顺序。当前 liburing 文档还明确指出，并非所有操作都能取消，已经提交到硬件的磁盘 I/O 通常就无法撤回。关闭普通 fd 也不会自动停止对应的 pending `io_uring` 请求，因为请求可能仍然持有自己的文件引用。

Multishot 操作让状态空间更大。一条 SQE 可以不断产生 CQE，同时继续保持活跃。Multishot accept 可以连续产生多个 socket，multishot receive 可以连续消耗多个 provided buffer。于是到了 checkpoint 时，“这条 SQE 是否已经完成”本身就不再是一个二值问题。

CRIU 的现状也说明，这仍然是真实的兼容性边界。它的 **#2131 `io_uring support in CRIU`** 到 2026 年 9 月 13 日仍处于 open 状态。该 issue 的复现程序在 dump 已建立 `io_uring` 的进程时，会因为 `anon_inode:[io_uring]` 映射而失败。将这个映射本身支持起来只是第一层；真正的生产级 restore 还必须决定如何处理活跃请求，以及那些普通内存和 fd image 无法完整描述的 ring-owned 资源。

## 需要分开的三个边界：请求、完成事件与外部副作用

对 checkpoint/restore 来说，一条请求至少会跨过三类不同的边界：

| 边界 | 需要回答的问题 | 为什么恢复时重要 |
| --- | --- | --- |
| 请求归属 | SQE 仍只在用户态、已被内核接收、已排队，还是已经执行？ | 决定旧 ring 是否还可能产生结果，以及是否能安全重放。 |
| 完成可见性 | CQE 是否已经生成，用户态是否已经消费？ | 防止丢结果或让同一个逻辑结果被交付两次。 |
| 外部副作用 | 操作是否已经改变文件、socket、namespace、peer 或设备状态？ | 防止 restore 重放已经发生过的动作。 |

这三个边界有关联，却不是同一回事。一次 write 可能已经进入存储路径，而用户态还没看到 CQE；一次 accept 可能已经创建 socket，而应用还没处理 direct-descriptor index；一次 receive 也可能已经从远端取走字节并消费一个 provided buffer，只是上层逻辑还没记录这次进展。

因此，盲目 replay 并不安全。重新建一个 ring，再把所有“应用尚未消费 CQE”的请求全部提交一次，可能重复外部副作用；完全不 replay，又可能丢掉旧 ring 已经接收、但还没真正执行的操作。Checkpoint 层需要比“pending/done”更细的分类。

可以先采用这样一组恢复类别：

- **recreate**：根据持久身份重新建立 ring 配置和资源表；
- **replay-safe**：允许再次提交，因为重复执行不会产生非法外部状态；
- **reconcile**：恢复前先查询或重建外部状态，再决定是否重放；
- **deliver-only**：副作用已经发生，但应用还没消费结果，只需要补交完成事件；
- **quiesce-required**：必须等待请求进入明确状态后才能 checkpoint；
- **non-restorable**：当前环境不存在安全重建路径，迁移应直接失败。

具体 runtime 可以使用不同名称，但必须存在类似区分。否则系统实际上是在把一个异步执行协议误当成内存复制问题。

## 现有研究还缺什么

第一个缺口是**可检查、可导出的 ring 状态边界**。Linux 和 liburing 已经提供 setup、registration、cancel 和 completion 等接口，但没有一个通用 checkpoint image，可以把 `io_uring` 实例表达成一组可恢复的操作与资源身份。只恢复用户态映射不够，因为请求引用、fixed-file entry、被 pin 的 buffer、provided-buffer 消费状态等都包含内核持有的部分。

第二个缺口是**理解副作用的 replay 语义**。`user_data` 可以关联请求和 CQE，却不能告诉 checkpoint 系统一次操作是否幂等、外部副作用是否已经发生、迁移后该如何核对。Write、send、accept、open 以及会分配 direct descriptor 的操作尤其需要这类语义。

第三个缺口是**恢复时的资源重新绑定**。Fixed-file index 只在某一个 ring 的 registered file table 内有意义，buffer ID 只在某一个 buffer group 内有意义，direct descriptor 甚至可能根本不存在于普通 fd table。只恢复数值索引，而不恢复它背后的对象身份和 ownership graph，完全可能得到一个格式正确、语义却错误的 ring。

第四个缺口是**面向故障类别的 benchmark**。一个测试如果只验证“进程恢复后还能继续处理请求”，就可能漏掉重复写入、CQE 静默丢失、accept 重复、fixed-file 绑定错误、buffer 尚未归还就被复用，或者 multishot 历史从错误位置重新开始。这些错误需要理解逻辑操作和外部副作用的 ground-truth oracle。

## 兼具学术价值与生产价值的方向

### 1. 为操作和资源定义 io_uring recovery manifest

第一个方向是建立一个机器可读的 checkpoint manifest，同时描述 ring 配置以及它引用的逻辑资源。它不需要序列化任意内核内部结构，而是保存足够稳定的信息，用来决定一项状态可以重建还是必须拒绝：

```text
ring = setup flags + supported features + registration generation
files = fixed index -> stable object identity + reopen/reconnect method
buffers = group/index -> memory region generation + ownership state
requests = logical request id + opcode + dependencies + recovery class
multishot = logical request id + delivered completion frontier
completion = result present? userspace consumed? associated resource/effect?
```

核心机制是**带 generation 的资源身份**。恢复后的 fixed index 7 只有重新绑定到 checkpoint 时同一个逻辑文件或 socket generation，才可以继续被当作原来的 index 7。Buffer group、direct descriptor 也应遵循同样规则。数字相同本身不构成身份连续性。

一个原型可以接到 CRIU plugin，或者放在封装 liburing 的 userspace runtime 中。评测应覆盖文件 I/O、TCP server、multishot accept/recv、registered files、registered buffers、provided-buffer rings 和 direct descriptors，并比较三种方案：只保存内存和 fd、完全 drain 后重建、以及 recovery manifest。首要指标应该是 invalid recovery event，其次才是 checkpoint downtime、restore latency、额外 bookkeeping，以及有多少 workload 最终仍然必须进入 quiescent 状态。

学术价值在于：为异步、内核持有的资源建立一种可移植的重建模型，而不要求把整个内核实现状态原样序列化。生产价值则很直接，越来越多 storage engine、proxy、runtime 和 server 都依赖 `io_uring`，它们未来做 live migration 或快速 restart 时会碰到同一问题。

如果现实 workload 里简单的 drain、destroy、recreate 已经能在可接受停顿内达到相同正确性，这个方向就应该放弃。更复杂的 manifest 只有在它真正降低停顿或支持无法轻易 quiesce 的请求时才值得存在。

### 2. 保存 completion/effect frontier，而不只是 pending SQE

第二个方向是维护一个 operation ledger，把每个逻辑请求至少拆成四个事实：已经提交、外部副作用可能已经提交、CQE 已生成、应用已经消费结果。Checkpoint 保存这些状态之间的 frontier，而不是只根据 SQ/CQ index 猜测请求处于什么阶段。

对只读操作，如果数据源本身允许，replay 可能安全；对有副作用的操作，runtime 可以要求应用提供 reconciliation key 或幂等规则。一次存储写入可以绑定到应用 transaction/generation ID；一次网络发送可以依赖协议已有的 sequence/request ID；一次 accept 如果连接已经真实存在，就应该恢复并补交 completion，而不是再次 accept 出一条新的连接。

这里有一个重要边界：不应要求内核理解应用事务。Runtime 只需要保存足够的 operation identity，把内核 completion 与应用已有的恢复协议连接起来。某个应用完全没有这种协议时，诚实的答案可能就是“checkpoint 前必须 quiesce”，而不是猜测该不该 replay。

评测时应在多个阶段强制冻结：内核消费请求之前、请求飞行中、外部副作用已经发生但 CQE 尚未被应用看到、CQE 已生成但尚未消费、以及应用已经确认结果之后。分别统计 duplicate effect、lost result、重复协议消息和不必要的 quiescence。还必须加入 cancel race 与不可取消的磁盘 I/O，避免方案把“cancel 成功”误当成可靠 oracle。

学术问题是：当内核并不控制远端或持久化副作用时，怎样为异步内核接口定义 exactly-once 或 at-least-once 的恢复边界。生产集成点则是 I/O runtime，因为 `user_data`、请求 metadata 与应用 transaction ID 通常已经在那里汇合。

如果应用本身已经有足够强的持久化幂等信息，使普通 checkpoint 加 replay 就能得到相同结果，那么独立 ledger 就没有必要。

### 3. 建立对抗式 io_uring checkpoint benchmark

一个真正有用的 benchmark 应该让常见的错误实现主动暴露出来。可以组合一组拥有明确 ground truth 的 workload：

- 每个逻辑写入都带唯一 generation 的文件 writer，可以检测重复或丢失；
- 使用 multishot accept 与 direct descriptor 的 TCP server，由 client 记录实际建立的连接；
- 使用 provided buffer 的 multishot receive，记录每段数据与 buffer ID 的 ownership 历史；
- fixed-file table 更新与飞行中的请求竞争；
- registered buffer replacement，在旧内存仍被请求引用时冻结；
- cancel race，包括已经进入不可取消阶段的操作。

Benchmark 在可控的请求生命周期节点冻结进程，然后在同一 kernel 上恢复，或者迁移到兼容 host。结果按 failure class 分开统计：lost CQE、duplicate effect、stale request replay、wrong resource binding、buffer ownership violation、multishot truncation/restart，以及无法恢复的外部状态。正确性通过之后，再报告 downtime 和 throughput。

最强的 baseline 不应该是一个故意做坏的 snapshotter，而应是保守的 **drain-everything** 方案：等待 ring 完全 quiescent，重建 ring，再继续运行。新系统只有在保持同等正确性的前提下减少停顿，或者支持难以全局 idle 的请求，才真正有价值。

这个方向的学术贡献是给异步 I/O migration 建立可重复的语义 benchmark；生产产物则是一套 qualification suite，让 runtime 和 checkpoint 维护者可以在新 kernel、liburing 版本和新应用 I/O pattern 上跑回归。

如果现有 CRIU 与 io_uring 测试已经能用同等级别的 ground-truth oracle 捕获上述逻辑故障，那么就没必要再造一个 benchmark。第一轮实验应该先尝试证明这一点。

## 现在可以采用的 checkpoint 规则

在出现更强的 restore 接口之前，应用和 checkpoint 系统应该优先选择**显式 quiescence，而不是猜测式 replay**。

先停止新的 submission，对能够安全取消的请求发起取消，同时继续消费 CQE，直到 runtime 能明确分类剩余状态。不要假设关闭普通 fd 会终止对应的 pending `io_uring` 操作。记录 fixed files、direct descriptors、registered buffers 与 provided-buffer groups 到应用对象的映射。把 multishot 请求当作带 completion frontier 的长期流，而不是一次性 SQE。对于任何 completion 与外部副作用关系不确定的请求，要么利用应用自己的协议进行 reconcile，要么直接拒绝在这个点创建 checkpoint。

这不意味着每个服务都需要新的 kernel API。很多应用完全可以主动建立一个很短的 quiescent point，再从普通持久状态重建 ring。任何更复杂的 checkpoint 机制都应该拿这个方案作为 baseline，而不是默认它落后。

此前的 [io_uring 可编程能力报告](https://eunomia.dev/zh/research/io-uring-bpf-programmability/) 讨论的是当越来越多资源进入 `io_uring` 后，policy 与 execution control 应该怎样组合。本文讨论的是另一条边界：这些异步资源怎样跨过进程 restart 或 migration。[GPU checkpoint 报告](https://eunomia.dev/zh/research/gpu-checkpoint-recovery-consistency/) 也提供了一个有用对照，它同样区分“某个组件能 restore”和“整个应用回到一个合法 recovery cut”，只是本文面对的具体状态是 Linux async I/O 与 ring-owned resource。

## 哪些结果会改变这个判断？

如果未来 CRIU 或其他通用 checkpoint 系统提供 `io_uring` restore 接口，并能证明它正确重建 live request、completion、registered resource、multishot progress 与副作用边界，那么保存更丰富 userspace recovery metadata 的必要性会明显下降。Kernel-supported export/import 甚至可以把许多难处理的状态直接纳入接口，不过外部副作用仍然需要应用语义。

如果实际测量发现，真实 `io_uring` 服务总能在可接受的迁移停顿内停止 submission，并快速 drain 或 cancel 整个 ring，那么更复杂的 live replay 也没有价值。便宜可靠的 quiescence 本身就是更好的设计。

还有一类应用本来就拥有很强的幂等协议。例如带持久 request ID 的 replicated storage engine，或能够精确检测 replay 的网络协议，都可能直接从高层状态恢复 pending work。对这些 workload，正确的 checkpoint contract 可以比通用 operation ledger 小得多。

最值得记住的边界是：**io_uring ring 不只是共享内存，它还是一个包含内核持有资源和异步副作用的执行协议。安全的 checkpoint 要么先让这个协议进入明确的 quiescent 状态，要么保存足够的身份与进度信息，有意识地完成恢复。**

## 参考资料

- CRIU，[`io_uring support in CRIU` issue #2131](https://github.com/checkpoint-restore/criu/issues/2131)，截至 2026-09-13 仍为 open。
- liburing manual，[`io_uring_cancelation(7)`](https://man7.org/linux/man-pages/man7/io_uring_cancelation.7.html)，访问于 2026-09-13。
- liburing manual，[`io_uring_multishot(7)`](https://man7.org/linux/man-pages/man7/io_uring_multishot.7.html)，访问于 2026-09-13。
- liburing manual，[`io_uring_registered_files(7)`](https://man7.org/linux/man-pages/man7/io_uring_registered_files.7.html)，访问于 2026-09-13。
- liburing manual，[`io_uring_registered_buffers(7)`](https://man7.org/linux/man-pages/man7/io_uring_registered_buffers.7.html)，访问于 2026-09-13。
- liburing manual，[`io_uring_provided_buffers(7)`](https://man7.org/linux/man-pages/man7/io_uring_provided_buffers.7.html)，访问于 2026-09-13。
