# 为什么通过 Cilium eBPF Go 库 attach 的 tracepoint eBPF 程序在新内核上只在 CPU 0 上触发，而同一程序在旧发行版上却能系统级地记录系统调用？

这不是库的 bug，也不是靠"每个 CPU 各开一个 perf event"就能绕开的问题。`link.Tracepoint` 只打开一个 tracepoint 类型的 perf event，并给 `perf_event_open` 传 `pid == -1`、`cpu == 0`，所以这个事件是"限定在 CPU 0"的。在健康的内核上，这个单事件仍然会在所有 CPU 上触发，因为 tracepoint BPF 程序不像硬件 PMU 计数器那样被 CPU 过滤。一次内核回归在系统调用追踪路径上破坏了这一点，于是限定在 CPU 0 的事件不再跨 CPU 投递，程序就只在事件恰好落在 CPU 0 时才触发。库只开一个事件是对的；回归在内核上游已经修复。

## 机制

边界在"库的 attach"与"内核的事件投递"之间：

1. **attach 侧（库）。** `link.Tracepoint` 在 tracefs 的 `events` 树下解析出事件 id，再调用 `openTracepointPerfEvent`，并传入"全线程"哨兵 `perfAllThreads`（即 `-1`）。在该函数里，当 `pid` 是这个全线程哨兵时，局部 `cpu` 保持为 `0`，于是系统调用就是 `PerfEventOpen`，参数为 `pid = -1`、`cpu = 0`，并带一个 `PERF_TYPE_TRACEPOINT` 属性。一个事件，限定在 CPU 0。
2. **投递侧（内核）。** `perf_event_open(2)` 的约定是：`pid == -1` 且 `cpu >= 0` 时，事件只度量那一个 CPU 上的所有线程。所以按 man 页，它确实是个 CPU-0 事件。但历史上这从没有真正限制过 tracepoint BPF 程序：tracepoint BPF 事件是系统级的，并不受 CPU 过滤约束（见 iovisor/bcc issue 1490）。attach 在 CPU 0 上的单个事件会在每个 CPU 上触发，所以同一程序在 Debian 13/14、Ubuntu 24.04 上能记到所有 `sys_enter_open` 调用。
3. **回归。** 在新内核上（例如内核 7.0 / Ubuntu 26.04），读取系统调用用户态参数的 trace event 路径可能触发抢占与迁移，此时每 CPU 的 perf 事件链表头可能指向了错误的 CPU，于是限定在 CPU 0 的事件就不再投递跨 CPU 的事件。可观察现象：用 `taskset -c 1,15` 把忙循环钉离 CPU 0，`trace_pipe` 就为空；用 `taskset -c 0` 钉到 CPU 0，日志又恢复。

## 验证与调试路径

1. **确认 attach 只开一个事件。** 用 `strace` 跟踪 attach：只有单个 `perf_event_open`，`pid = -1, cpu = 0`。对照 `perf stat -a` 会每个 CPU 各开一个 `perf_event_open`——这个差异解释了库与 `perf` 在内核侧看起来不同。
2. **复现作用域。** 跑一个紧凑的 `open()` 循环，钉离 CPU 0（`taskset -c 1,15 ./prog`）：在受影响的内核上 `trace_pipe` 为空；钉到 CPU 0（`taskset -c 0`）日志就出现。这就是整个诊断。
3. **排除库。** 同一个 `cilium/ebpf` 构建在 Debian 13/14、Ubuntu 24.04 上表现一致；只有新内核不同，且库是刻意只开一个事件。回归不在 attach 这一侧。
4. **确认内核回归与修复。** 维护者把它追溯到系统调用 trace 事件路径上每 CPU 头指针的失效，上游已用内核提交 `2c2b322acdcc`（"tracing: perf: Fix stale head for perf syscall tracing"，它修复了更早的 `edca33a56297d`）更正。在内核达到该修复或某发行版已回合之后，程序就会重新在所有 CPU 上触发。请用上游内核做诊断，避免发行版内核的噪声干扰判断。

## 局限

- 行为取决于内核版本。健康内核是系统级触发；某一段内核上表现为只在 CPU 0 上触发。"在我的机器上正常"在这里是真实的，不是代码缺陷。
- 对 tracepoint 而言，只开一个 CPU 0 事件是库的正确设计；只有在这个受影响内核上它才显得"不对"，因为内核"tracepoint 事件是系统级"的保证发生了回归。自己循环每个 `perf_event_open` 只是临时绕过，并非通用要求，真正的解法是内核的回修。
- 发行版内核很关键：不同发行版是否已回修该修复并不一致。要在实际内核上验证，并在上游内核上复现。
- `perf_event_open(2)` 关于 CPU 作用的措辞描述的是事件的作用域契约；它并不意味着"正确 attach 的 tracepoint 程序在修复后的内核上应该只看得到一个 CPU"。

## 参考

- [cilium/ebpf link/tracepoint.go (v0.22.0)](https://github.com/cilium/ebpf/blob/v0.22.0/link/tracepoint.go) —— `Tracepoint` 助手及其单事件 attach。
- [cilium/ebpf link/perf_event.go](https://github.com/cilium/ebpf/blob/v0.22.0/link/perf_event.go) —— `openTracepointPerfEvent`、`perfAllThreads` 与带 `cpu = 0` 的 `perf_event_open` 调用。
- [iovisor/bcc issue 1490](https://github.com/iovisor/bcc/issues/1490) —— tracepoint BPF 事件是系统级的，且不遵循 CPU 过滤。
- [kernel commit 2c2b322acdcc](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/commit/?id=2c2b322acdcc78575b8d6afa64a085cf92e03c12) —— "tracing: perf: Fix stale head for perf syscall tracing"。
- [cilium/ebpf discussion 2126](https://github.com/cilium/ebpf/discussions/2126) —— 本页回答的那个实践者问题。

## 当日社区讨论

本 run 审查的两个 opt-in 存档渠道（两个 CNCF OpenTelemetry 插桩渠道）共产出四条消息。本页回答的正是其中一条：一位实践者把 Cilium eBPF Go 库里的一个开放问题交叉转发到这边，问为什么一个 `link.Tracepoint` 程序在新发行版上只记录了归属于 CPU 0 的调用，以及如何让一个 tracepoint 做系统级监测。存档里只带了那个公开讨论的指针，所以问题实质与维护者的诊断在此以匿名化形式复现。窗口里另外两个话题早已被往期回答覆盖——可用技能专属属性的建模问题、以及 model-serving-signals / GenAI 指标目录边界问题——因此本 run 选取了那个尚未在本站回答过、且非重复的 Cilium tracepoint 问题。

本 run 的渠道覆盖：两个 opt-in 存档渠道共提供四条消息。可见浏览器专属的渠道——eunomia-bpf 与 sched-ext 的 Discord 服务器、bpf 邮件列表、以及 r/eBPF——本 run 没有可用的可见浏览器会话，因此标记为"未覆盖"，而非"安静"。
