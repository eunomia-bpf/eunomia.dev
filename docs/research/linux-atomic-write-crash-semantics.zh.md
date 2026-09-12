---
date: 2026-09-12
slug: linux-atomic-write-crash-semantics
title: "Linux 原子写成功后，数据就一定能抗崩溃吗？"
description: "Linux 原子写可以防止单个数据范围被撕裂，但崩溃一致性仍需组合持久化、写入顺序、文件系统元数据与应用恢复，应用只有验证这些边界后才能安全简化 WAL 或同步屏障。"
tags:
  - Daily Report
  - Linux
  - Storage
  - Atomic I/O
  - Crash Consistency
research_question: "Linux 应用应该怎样组合 atomic write、持久化、写入顺序、文件系统元数据和 recovery guarantee，避免把 RWF_ATOMIC 误解成完整的 crash-consistency contract？"
source_cutoff: 2026-09-12
status: daily-report
---

# Linux 原子写成功后，数据就一定能抗崩溃吗？

Linux 现在已经有了真正面向普通文件的 atomic write 接口。应用可以通过 `pwritev2(..., RWF_ATOMIC)` 请求 torn-write protection：只要文件系统和底层存储支持，在掉电或硬件故障后，这一段数据必须全部保持旧值，或者全部变成新值，不能一半新、一半旧。

听起来，这已经很像数据库更新一个 page 时想要的性质了。但这里很容易多走一步，把“这一段不会被撕裂”理解成“这次事务已经安全落盘”。

其实两者差得很远。

Untorn write 只回答一个很窄的问题：**一个满足条件的数据范围，在故障后会不会出现新旧字节混在一起？** 它本身并没有回答 syscall 返回时新数据是否已经持久化、两个 atomic write 是否按应用需要的顺序进入稳定存储、目录项或文件长度是否和数据属于同一代，也没有保证跨多个文件和对象的更新构成一个可恢复的应用事务。

Linux 也正是把这些性质拆成不同机制暴露出来的。`RWF_ATOMIC` 负责 torn-write protection；`O_SYNC`、`O_DSYNC`、`RWF_SYNC`、`fsync()` 以及文件系统 journal 分别处理不同层次的持久化和元数据问题。如果应用最后只留下一个 `atomic_write_supported=true`，就很可能把一条窄的底层 guarantee 放大成并不存在的应用级 guarantee。

<!-- more -->

## Linux 原子写解决的是 torn write，不是全部 crash consistency

当前 Linux manual 对 `RWF_ATOMIC` 的定义非常明确：在 block-based filesystem 的普通文件上请求 torn-write protection。存储路径支持时，即使发生掉电或其他 hardware failure，也应该是整个 write 全部生效或者全部不生效，而不能同时看到 old/new data。

同一份 manual 紧接着又规定了几个重要条件：它只支持 `O_DIRECT`，write size 和 offset 必须满足 `statx()` 返回的 atomic unit 与 alignment 约束；如果应用还需要保证文件的 in-core state 和 storage device 之间一致，就要另外使用 `O_SYNC`、`O_DSYNC`、`RWF_SYNC` 等 synchronized I/O 语义。

这句话实际上已经把最容易混淆的两件事拆开了：**atomicity 和 synchronization 不是同一个维度。**

`statx()` 会暴露 `stx_atomic_write_unit_min`、`stx_atomic_write_unit_max`、`stx_atomic_write_segments_max` 和 `stx_atomic_write_unit_max_opt`。它们告诉应用，当前这个文件在当前 filesystem/storage path 上，什么形状的数据范围可以做 atomic write。它们并没有定义应用事务，更不会把两次独立提交的 atomic write 自动合并成一个 atomic unit。

Ext4 的实现更能说明为什么这个边界不能省略。最新文档要求 atomic write 使用 Direct I/O、regular extent file，并依赖底层 block device 的硬件 atomic write。Linux 6.13 开始支持 single-fsblock atomic write；multi-fsblock 则依赖 bigalloc，而且最大/最小 atomic unit 同时受 filesystem 和 device 限制。如果目标范围同时包含 mapped 和 unwritten extent，ext4 还要先把它整理成一个连续 extent，并可能在真正 data I/O 前强制提交当前 journal transaction。否则 crash 后可能同时出现已经更新的 mapped 区和还没有完成 extent conversion 的区域，直接破坏原子性。

也就是说，一个看起来只是“这次 write 不要撕裂”的接口，实际上已经横跨 allocation metadata、unwritten extent conversion、journal、iomap 和 storage device。

Ext4 journal 又是另一层 guarantee。JBD2 的核心目标是避免 filesystem metadata 在 crash 后停在一个半完成的 metadata transaction 里。默认 `data=ordered` 并不会把普通 file data 全部写进 journal；`data=journal` 更强，但代价也更高。所以“文件系统有 journal”同样不能直接推出“我的应用事务已经 durable”。

更实用的思考方式，是把 crash behavior 至少拆成下面五个性质：

| 性质 | 典型机制 | 它真正回答的问题 |
| --- | --- | --- |
| Range atomicity | `RWF_ATOMIC` + supported storage | 这一段 write 会不会 torn？ |
| Completion / persistence | `O_SYNC`、`O_DSYNC`、`RWF_SYNC`、`fsync()` | syscall/同步点完成时，哪些数据必须已经稳定？ |
| Ordering | flush/barrier + application protocol | A 必须先于 B 时，recovery 会不会看到 B 却看不到 A？ |
| Filesystem metadata consistency | ext4/JBD2 mode 与 metadata rule | allocation、size、rename、directory state 能否恢复到合法 filesystem 状态？ |
| Application transaction consistency | WAL、commit record、COW、recovery logic | 多个对象组合后，哪些 post-crash state 才是合法事务状态？ |

第一项完全成立，第五项仍然可能失败。

比如数据库先 atomic overwrite 数据页 P，再 atomic overwrite commit record C，但没有建立需要的 persistence ordering。两次 write 都永远不会 torn，并不代表 crash 后不会出现 C 已经 durable、P 却没有 durable 的组合。反过来，一个 WAL protocol 可能完全允许数据页还没写下去，因为 recovery 会从 log 重放。**应用允许哪些 crash cut，是由 recovery protocol 定义的，不是由某一个 syscall 里出现了 `ATOMIC` 这个名字定义的。**

这和此前的 [GPU checkpoint recovery 报告](https://eunomia.dev/zh/research/gpu-checkpoint-recovery-consistency/) 有一个相似的系统边界：某一个 component 可以恢复，并不等于整个应用已经回到一个合法 global cut。这里的底层机制完全不同，但局部 guarantee 不能自动放大成全局 guarantee 这一点是一样的。

## 现有工作还缺什么

第一个缺口是 **capability description 不够完整**。Linux 已经能通过 `statx()` 给出很有用的 per-file atomic-write 参数，filesystem 也会记录自己的 requirement。但 storage engine 真正决定 protocol 时，仍然要把 open flags、filesystem mode、device capability、metadata operation 和自己的 recovery assumption 拼起来。“支持 atomic write”这个 Boolean 太小了。

第二个缺口是 **composition**。现在没有一个通用 artifact 能表达：一次 application commit 包含这两段 atomic data write、一个 metadata update、一次 rename 和一个 commit record，故障后只允许出现哪些组合。Filesystem journal 和 database WAL 各自在自己的边界里做了 composition，但跨两层的 contract 很大程度上还是藏在 bespoke implementation 和经验里。

第三个缺口是 **evaluation 对 failure class 的区分不够**。能检测 torn sector 的测试可以很好地验证 atomic-write mechanism，却可能完全看不到 whole write 丢失、write reordering、metadata/data generation 不一致，或者 recovery 接受了一个本来不可能存在的 transaction state。Linux 已经有成熟的 filesystem/block test framework，包括 blktests，但 application-level atomic I/O 仍需要一个高于“这次 range 没 torn”的 oracle。

第四个缺口是 **deployment evidence 的可移植性**。Atomic write 本来就应该按 capability 检测，而不是按 kernel version 猜。同一个 application binary 放到不同 host 上，可能面对不同的 `statx()` limit、filesystem 配置和 storage hardware。Storage engine 不但要选择 fast path，还应该能在事故后解释：为什么这台机器走 atomic path，那台机器走 WAL/fsync fallback。

## 兼具学术价值和生产价值的方向

### 1. 做一份可组合的 crash-semantics descriptor

不要把 atomic write support 暴露成一个 bool，而是为真实 storage path 生成一份机器可读的 descriptor，把应用真正依赖的条件绑定在一起，例如：

```text
file = inode/mount identity
atomic_range = statx min/max/segments/alignment
io_mode = O_DIRECT + sync semantics
filesystem = ext4 + relevant feature/mount mode
metadata_scope = rename/size/allocation 需要的持久化规则
storage_path = device/controller capability identity
application_protocol = expected recovery contract version
fallback = WAL/fsync 或 copy-on-write path
```

它不是用来替代 filesystem spec 或 device spec，而是在 I/O engine 选择 protocol 的 integration boundary，把“应用认为现在拥有什么 crash semantics”显式记录下来，并且可以版本化。

研究 prototype 可以在多种 kernel、ext4 config 和 device capability profile 上，比较 descriptor 预测的 crash state 与 fault injection 真正观测到的结果。Primary metric 不应该是 coverage，而是 **false confidence**：descriptor 认为 protocol 安全，却实际到达 forbidden state 的次数。还可以测 false rejection、生成成本，以及它相对 conservative baseline 有多少次真正改变 fast-path selection。

学术价值在于把分散在不同 API、filesystem 和 device contract 里的 crash guarantee 变成可组合模型。生产使用者是 database、object store 或 storage runtime，接入点就是 I/O backend 初始化和 protocol selection。

如果只用现有 `statx()`、open flags、filesystem discovery 和 storage-engine config 就已经能够无歧义推导所有重要 recovery state，而且 descriptor 从不改变 safe/unsafe decision，也发现不了错误 assumption，那这个方向应该直接放弃。否则只是多了一份 manifest。

### 2. 给每次 application commit 生成 crash-cut witness

第二个方向是把一次 commit 写成一张很小的 persistence obligation graph，而不是让真正的语义只存在于 syscall sequence 和代码注释里。

例如一个 page-oriented database 可以描述：

```text
WAL record W 必须 durable 后，page P 才能成为 authoritative
P 满足 range constraint 时可以使用一次 RWF_ATOMIC
commit marker C 只能在 W 和 P 都持久化之后 durable
只有 allocation/layout 改变时才需要 metadata operation M
recovery 可以接受 {old, W-only, W+P, W+P+C}
其他所有 visible combination 都必须拒绝
```

Runtime 可以只在 debug/test build 输出这份 witness，checker 则在每个 persistence edge 枚举 crash cut。这里并不是要把 transaction manager 塞进 kernel，而是让 application test 第一次拥有一个明确 oracle，知道底层 untorn-write、sync、ordering 和 metadata guarantee 最终应该组合成什么。

Evaluation 至少拿 atomic-write-aware protocol 和传统 WAL + `fsync()` baseline 比。故障点覆盖 submission 前、device completion 后、sync operation 周围和 metadata change 周围。第一指标必须是 forbidden recovered state，然后再比较 barrier 数、write amplification、commit latency 和 recovery work。最有价值的结果不是“atomic write 更快”，而是证明它确实能删掉一部分 logging/copy 成本，同时没有扩大 legal recovery-state set。

学术贡献是给 untorn range、persistence ordering 和 application recovery 一个具体的 composition semantics。生产接入点是 database/storage engine 的 commit path，witness 更适合 CI、hardware qualification 和 incident reproduction，不一定需要让每次 production I/O 都付出额外成本。

如果固定的 WAL/`fsync()` protocol 更简单，而且在 latency、write amplification 上一样好或更好，同时始终保持零 forbidden recovery state，那么 witness-guided design 没有存在价值。Atomic I/O 不应该为了“新”而增加 protocol complexity。

### 3. 按 failure class 测 crash guarantee，而不是一个 pass/fail

一个有用的 benchmark 应该故意把下面几类失败拆开：

- 同一个 requested range 里面出现 torn data；
- syscall 已完成但没有同步的 whole write 在 power loss 后完全消失；
- 两次各自合法的 write 以 application-invalid ordering 持久化；
- data 与 filesystem metadata 属于不同 generation；
- create/rename 等 namespace operation 与 file data 落在不同 crash cut；
- process crash、kernel crash、controller reset 和 full power-loss model。

现有 kernel infrastructure 可以继续复用。`blktests` 已经是 Linux block layer 和 storage stack 的测试框架，filesystem tooling 也可以直接驱动 atomic-write path。真正缺的是 application-level oracle：在一个声明好的 protocol 下，把每种 recovered state 标成 allowed 或 forbidden。

同一 protocol 要跨多种 atomic-unit size、extent layout、sync mode、filesystem mode，以及模拟/真实 failure model 测试。结果不要合成一个“crash test passed”。分别报告 torn-write escape、forbidden recovered state、capability selector 的 false accept/reject、recovery time 和 overhead，这样才不会把不同 guarantee 再次混到一起。

学术价值是形成一套能分辨 atomicity、durability、ordering、metadata consistency 和 transaction recovery 的 measurement methodology。生产使用者是 storage engine 或 platform qualification team，接入点是 fleet 上线 fast path 之前的 kernel/filesystem/device matrix。

如果已有 block/filesystem test suite 在相同成本下已经能预测 application recovery，而且覆盖完全相同的 failure class 和 target matrix，那么不需要再做一套新 benchmark。实验应该主动证明 application oracle 是否真的能发现额外错误。

## 一个更稳妥的上线规则

现在更合理的做法，是把 `RWF_ATOMIC` 当作简化 crash protocol 某一个局部步骤的 capability，而不是删除 crash protocol 的许可证。

先对这个具体文件调用 `statx()` 并请求 `STATX_WRITE_ATOMIC`，不要根据 kernel version 猜。读取 atomic-write 字段前，必须确认返回的 `stx_mask` 包含 `STATX_WRITE_ATOMIC`，随后严格满足 Direct I/O、alignment、range 和 segment constraint。然后单独决定 completion 是否需要 synchronized I/O。多次 write 之间的 ordering 要继续显式表达。Filesystem metadata 和 namespace operation 按 filesystem 自己的 persistence rule 处理。最后在真的删除 WAL、COW step 或 barrier 之前，用 crash injection 验证 application recovery state machine。

此前的 [有状态 eBPF 原子升级报告](https://eunomia.dev/zh/research/stateful-ebpf-transactional-upgrade/) 用 generation-gated commit 和 recovery 避免把 multi-object update 的中间状态误认成新 generation。Storage software 虽然使用完全不同的 primitive，但会犯同一类 composition mistake：**一个 locally atomic operation 并不会自动让 multi-object transition 也 atomic。**

## 什么证据会改变这个结论？

如果未来 Linux 提供一套 end-to-end interface，明确把 untorn write、persistence completion、cross-write ordering、相关 filesystem metadata，以及 multi-object transaction boundary 组合在一个有正式 recovery semantics 的接口里，那么应用当然不需要自己组合这么多独立 guarantee。

另一个会削弱本文结论的情况，是实际 storage engine 证明一个更小的 contract 已经足够。例如某个 application 的全部 authoritative state 真正只落在一个 eligible atomic range 里，没有额外 metadata/object 参与 commit，那么一次 atomic + synchronized write 本身就可以成为 transaction boundary，不需要额外 composition layer。

最后，fault injection 也可能直接否定上面的研究方向。如果现有 capability discovery 加传统 WAL/fsync practice 已经能在不同 kernel、filesystem、device 上准确预测所有 post-crash state，那么新 descriptor 和 witness language 只是在增加流程，而不是增加安全性。

因此最有用的 mental model 其实很窄：**Linux atomic write 能保证一个受支持的 range 不被 torn；真正的 crash consistency 仍然是 protocol property，持久化、顺序、metadata 和 recovery boundary 都要分别说清楚。**

## 参考资料

- Linux kernel documentation, [Atomic Block Writes](https://www.kernel.org/doc/html/latest/filesystems/ext4/atomic_writes.html)，访问于 2026-09-12。
- Linux man-pages, [`pwritev2(2)` / `RWF_ATOMIC`](https://man7.org/linux/man-pages/man2/pwritev2.2.html)，man-pages 6.19，访问于 2026-09-12。
- Linux man-pages, [`statx(2)` / `STATX_WRITE_ATOMIC`](https://man7.org/linux/man-pages/man2/statx.2.html)，访问于 2026-09-12。
- Linux kernel documentation, [ext4 Journal (jbd2)](https://www.kernel.org/doc/html/latest/filesystems/ext4/journal.html)，访问于 2026-09-12。
- linux-blktests, [blktests](https://github.com/linux-blktests/blktests)，访问于 2026-09-12。
