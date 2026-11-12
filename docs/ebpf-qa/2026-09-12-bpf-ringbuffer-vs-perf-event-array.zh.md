# 什么时候应该用 BPF ring buffer 而不是 perf event 数组来采集 eBPF 样本？

**简短回答：** perf event 数组（`BPF_MAP_TYPE_PERF_EVENT_ARRAY`）和 BPF ring
buffer（`BPF_MAP_TYPE_RINGBUF`）用不同的所有权模型把 eBPF 样本从内核搬到用户
空间。perf event 是 per-CPU 的：每个 CPU 一个 perf fd、一块固定大小的数据区，
按 CPU 消费，某个 CPU 自己的区域填满时只丢那个 CPU 的样本。ring buffer 是一个
共享的多生产者单消费者（MPSC）环，没有 perf fd，由一个消费者统一消费；
`bpf_ringbuf_reserve()` 把样本零拷贝直接写进跨 CPU 共享的那一个环。决定性
边界是生产者在哪里。如果生产者和消费者在同一个 CPU 上，或你希望 per-CPU 隔离
和固定大小的数据区，perf event 数组是合适的工具。如果生产者跨很多 CPU 或线程，
想要一条共享的消费流而不必为每个 CPU 建 perf fd，就用 ring buffer。ring
buffer 可能丢掉 perf 数组保留下的记录：共享环满时 `bpf_ringbuf_reserve()` 会
原子地失败；即使还有空闲空间，在 NMI 上下文里争抢环锁时也会失败；而 perf
event 数组的溢出是 per-CPU 的，每个 CPU 的区域各自停止接受新样本。

## 两种所有权模型

perf event 数组每个 CPU 一个槽位。每个槽位是一个 perf buffer，有一块固定大小
的数据区和一个 perf fd。消费者独立地读每个 CPU 的 buffer。某个 CPU 的区域溢出
时，只有那个 CPU 停止接受新样本，其他 CPU 继续工作。CPU 之间没有共享，除了
map 本身，所以某个 CPU 上的突发不会影响另一个 CPU 的样本。

ring buffer 正好相反。一个 `BPF_MAP_TYPE_RINGBUF` map 向所有 CPU 呈现一个
2 的幂大小的共享环（通过 `BPF_MAP_TYPE_HASH_OF_MAPS` 可以得到一组环，用于分
片设计）。任意 CPU 上的生产者都在同一个环里预留空间，由一个消费者统一抽走。
没有 perf fd，也没有 per-CPU 区域。内核文档直接给出了动机：跨 CPU 共享一个
环，内存利用率更高，并且省掉 perf event 数组所需的 per-CPU perf buffer。

差异在 helper API 上看得最清楚。`bpf_perf_event_output()` 把记录拷贝进
per-CPU 的 perf buffer。`bpf_ringbuf_output()` 做同样的拷贝，但目标是 ring
buffer。`bpf_ringbuf_reserve()` / `bpf_ringbuf_commit()` /
`bpf_ringbuf_discard()` 则直接把一个指向环内存储的指针交给程序，程序原地写入，
零拷贝。`bpf_ringbuf_query()` 报告 `BPF_RB_AVAIL_DATA`（未消费字节数）和
`BPF_RB_RING_SIZE`，这是消费者用来决定吞吐和节奏的依据。

## 失效模式并不相同

丢弃行为在形态上不同，这正是最容易被人忽略的地方：

- **perf event 数组溢出**是 per-CPU、彼此独立的。每个 CPU 的区域各自填满，
  那个 CPU 的后续样本被丢，其他 CPU 继续。存在稳定的 per-CPU 背压模型。
- **ring buffer 溢出**对共享环来说是全局的。共享环满时，
  `bpf_ringbuf_reserve()` 原子地失败，对所有生产者一次性丢样本。
- **即使没满，ring buffer 预留也会失败。** 在 NMI 上下文里，
  `bpf_ringbuf_reserve()` 可能拿不到环锁，因此即便有空闲空间预留也会失败。
  在 NMI 或紧密原子路径里采样的程序必须把“没满”当作“不保证成功”。

## 如何验证你需要哪一种

1. **先定位生产者。** 如果采样发生在与消费者相同的 CPU 上，或你希望 per-CPU
   隔离，per-CPU 的 perf event 数组能避免共享环的全局竞争和跨 CPU 饥饿。
2. **盯着丢弃的来源，而不只是数量。** 对 ring buffer，在一次负载前后轮询
   `bpf_ringbuf_query(BPF_RB_AVAIL_DATA)`，确认消费者能抽到接近零。对 perf
   event 数组，则逐个 CPU 检查溢出。ring buffer 在突发后仍保持高 available
   data，说明在全局丢；perf 数组只在那些热点 CPU 上显示溢出。
3. **检查生产者上下文。** 如果程序可能跑在 NMI 或紧密原子路径里，专门记录
   `bpf_ringbuf_reserve()` 的失败。那里空闲空间不保证预留成功，缓解办法是
   用基于拷贝的 `bpf_ringbuf_output()` 或降低锁竞争，而不是单纯把环调大。
4. **匹配拓扑。** 生产者跨多核时，共享环去掉了 perf event 数组每个 CPU 一个
   perf fd 的开销；如果消费者线程已经绑核，per-CPU 的 perf event 数组正好
   对得上。

## 决定性的局限

共享 ring buffer 用 per-CPU 隔离换来了单一消费流。某个 CPU 上的突发可以把
共享环填满，对所有生产者一次性丢记录；共享设计里没有 per-CPU 背压。perf event
数组保留了这种隔离：某个 CPU 上的突发只丢那个 CPU 的样本。当共享消费流和
零拷贝 reserve 值得接受全局溢出时，用 ring buffer；当 per-CPU 隔离和固定的
per-CPU 数据区是硬性要求时，继续用 perf event 数组。

## 参考资料

- [The Linux kernel documentation: BPF ring buffer](https://docs.kernel.org/bpf/ringbuf.html)
- [perf_event_open(2)](https://man7.org/linux/man-pages/man2/perf_event_open.2.html)
- [The Linux kernel documentation: BPF maps](https://docs.kernel.org/bpf/)

## 当日社区讨论

受监测的窗口没有技术内容。过去 24 小时两个 opt-in Slack 归档没有返回任何
消息；7 天回退窗口里只有会议安排——日程帖子、一个致谢、一个社区场次链接——
没有 eBPF 问题、症状或设计争论。两个白名单 chat 工作区本次没有可见的浏览器
会话；公开邮件列表与论坛归档本次未复核。这些来源被记录为不可用覆盖，而不是
“安静”。由于可读归档没有产出任何技术问题，本条 Q&A 依据公开一手文档而非社区
消息撰写：问题是一个反复出现的实践者抉择——在把样本搬到用户空间时，在 BPF
ring buffer 与 perf event 数组之间做选择——并对照内核的 ring buffer 设计文档
与 `perf_event_open(2)` 接口做了核实。本页不复现任何私有文本、身份、频道或
链接。
