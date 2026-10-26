# 为什么不可睡眠的 BPF fentry 与 fexit 程序会悄悄跳过同一 CPU 上其他任务的调用，被跳过的调用又为什么会记成递归丢失？

因为 trampoline 的递归保护是 per-CPU 的，而且是在 preemption 仍然开启的状态下获取的。不可睡眠的 fentry/fexit 程序经由 BPF trampoline 执行，其入口路径会递增 `prog->active`，一个 per-CPU 计数器，而它是在 `rcu_read_lock_dont_migrate()` 下获取的。这个锁只禁用了 CPU 迁移，并没有禁用 preemption。如果进入程序的任务在程序执行中途被抢占，它那个 CPU 上的计数器会一直停在 1。随后当一个不相关的任务被调度到同一个 CPU 并命中同一个 hook 时，trampoline 看到的计数器是 2 而不是 1，便认定程序正在递归，递增 `recursion_misses`，并悄悄跳过该程序。其实并没有发生递归；这条 miss 是一个误报。

在内核 7.0 之前，大多数发行版内核默认运行 `PREEMPT_VOLUNTARY`，而不可睡眠的 BPF 程序内部不含自愿 preemption 点，所以这个窗口很少被触发。提交 `7dadeaa6e851`（"sched: Further restrict the preemption modes"）在 x86 与 arm64 上移除了 `PREEMPT_NONE` 和 `PREEMPT_VOLUNTARY`，只留下 lazy 与 full preemption。自此以后，每个发行版内核在正常多任务运行中都会在不可睡眠的 fentry/fexit 程序内部发生 preemption，于是这些丢失现在是常规现象，而且它们在 trace 输出中是不可见的：事件只是干脆没有运行。

## 机制

1. 保护本身。`kernel/bpf/trampoline.c` 里的 `__bpf_prog_enter_recur()` 在 `rcu_read_lock_dont_migrate()` 下获取 `prog->active`——一个在 `include/linux/bpf.h` 中声明为 per-CPU 计数器的字段。`bpf_prog_get_recursion_context()` 要求这个 per-CPU 值恰好等于 1；任何别的值都意味着"已经在里面"，于是 trampoline 调用 `bpf_prog_inc_misses_counter()`，可选地触发 `recursion_detected` 回调，并直接返回而不运行 BPF 程序。

2. Preemption 的漏洞。`rcu_read_lock_dont_migrate()` 把任务钉在它当前的 CPU 上，但仍允许它被抢占。任务 A 进入 `fentry/foo`，CPU N 上的 `prog->active` 变成 1；任务 A 在程序中途被抢占，计数器仍为 1。任务 B 运行在 CPU N 上并调用 `foo()`，计数器递增到 2，`val != 1` 检查失败，任务 B 的这次调用被记成递归丢失并跳过。等任务 A 恢复并退出时，计数器降回 0。这条 miss 纯粹是 preemption 干扰，不是重新进入。

3. 为什么是 per-CPU 而不是 per-task。自从提交 `a76ab5731e32`（"bpf: Find eligible subprogs for private stack support"）起，做递归检查的 trampoline 程序还用 `prog->active` 来保证对唯一的 per-CPU 私有栈（`priv_stack_ptr`）以及 per-CPU scratch map 的独占所有权。把计数器改成 per-task 存储可以消除误报，但会破坏那个所有权保证，所以修复必须是一个 preemption 屏障，而不是换一个计数器。

4. 与 kprobe 和 tracepoint 的对照。`kernel/trace/bpf_trace.c` 里的 `trace_call_bpf()` 断言 `cant_sleep()`，而提交 `16718274ee75`（"tracing: perf: Have perf tracepoint callbacks always disable preemption"）特意保持 perf tracepoint 回调始终禁用 preemption。因此，同一段 BPF 程序逻辑，仅凭它是通过 fentry/fexit 还是 kprobe/经典 tracepoint 挂载，就会得到不同的完整性保证。

## 验证与排障路径

1. 统计丢失。用 `bpf_enable_stats()` 打开 `BPF_STATS_RUN_TIME`，跑一段负载，再用 `bpf_prog_get_info_by_fd()` 读取 `prog_info.run_cnt` 与 `prog_info.recursion_misses`。一个从不自重入的程序在可抢占内核上却报了非零的 `recursion_misses`，就是这个 bug 的特征；该计数器自提交 `ca06f55b9002`（"bpf: Add per-program recursion prevention mechanism"）起就存在，也就是 RFC 引用的 `Fixes:` 目标。

2. 确定性复现。把产生负载的任务钉在一个 CPU 上，在一个热点函数上挂一对 fentry/fexit 探针，并让探针体足够长以可被抢占（RFC 的 47 行复现器在 `fentry/__set_task_comm` 与 `fexit/__set_task_comm` 里都循环一百万次 `bpf_loop`，同时 fork 短命任务）。RFC 实测：stock 6.8 内核 voluntary 模式每 1000 次 0 丢失，同一内核 full 模式 25-26 百分丢失，7.3 mainline lazy 模式 28 百分以上。

3. 检查 preemption 模式。`uname -v` 与内核配置能告诉你是否设置了 `CONFIG_PREEMPT_LAZY` 或 `CONFIG_PREEMPT`。用旧 voluntary 模式编译的内核不会复现这些丢失；这就是为什么代码路径很旧，但这个 bug 却是 7.0 时代才暴露。

4. 交叉对照挂载类型。把同样逻辑作为 kprobe 和经典 tracepoint 程序挂到同一个 hook。由于这些路径以禁用 preemption 运行，kprobe/tracepoint 变体会对每个任务都继续触发，而 fentry/fexit 变体会继续产生误报递归丢失。这个差异是最快的确认方式，用来区分是原因还是程序自身的逻辑 bug。

## 局限

- RFC 的修复只作用于 `!CONFIG_PREEMPT_RT` 内核：在 PREEMPT_RT 上 trampoline 保留 `rcu_read_lock_dont_migrate()`，因为可睡眠、可抢占执行正是那里的设计目标。非 RT 内核改为在 enter/exit 这一对路径周围使用 `preempt_disable_notrace()`。
- 这条 miss 本身就是按设计静默的：`recursion_misses` 是一个轮询得到的统计量，而 `recursion_detected` 回调是可选的，所以只看成功事件的监控系统不会自己察觉完整性上的损失。
- 私有栈与 per-CPU scratch map 的所有权，正是计数器不能简单改成 per-task 的原因；任何替代修复仍需在 CPU 上串行化，所以该系列选了一个 preemption 屏障而不是重新设计计数器。
- 用 kprobe/tracepoint 兜底并非没有代价：在热点 socket 路径上，RFC 报告 fentry 大约比 kprobe 快 2 倍，而 kretprobe 依赖一个有界的 `maxactive` 池，在阻塞调用如 `inet_recvmsg()` 上可能耗尽。

## References

- [Public bpf mailing-list RFC: do not let preemption cause false recursion drops in trampoline programs](https://lore.kernel.org/bpf/20261025-b4-bpf-trampoline-preempt-v1-0-198dcdc0c6ac@sentinelone.com/) — 一封公开的 bpf 列表 RFC，提议在非 RT 内核上围绕 `__bpf_prog_enter_recur()` / `__bpf_prog_exit_recur()` 使用 `preempt_disable_notrace()`，并附一个可复现的 selftest。
- [Kernel `kernel/bpf/trampoline.c`](https://github.com/torvalds/linux/blob/master/kernel/bpf/trampoline.c) — `__bpf_prog_enter_recur` 与 miss 计数路径。
- [Kernel `include/linux/bpf.h`](https://github.com/torvalds/linux/blob/master/include/linux/bpf.h) — per-CPU `active` 计数器与 `bpf_prog_get_recursion_context`。
- [Kernel `include/uapi/linux/bpf.h`](https://github.com/torvalds/linux/blob/master/include/uapi/linux/bpf.h) — `bpf_prog_info.recursion_misses` 与 `run_cnt`。
- [Kernel `kernel/trace/bpf_trace.c`](https://github.com/torvalds/linux/blob/master/kernel/trace/bpf_trace.c) — `trace_call_bpf`，禁用 preemption 的 kprobe/tracepoint 对照。
- [Commit `7dadeaa6e851` ("sched: Further restrict the preemption modes")](https://github.com/torvalds/linux/commit/7dadeaa6e851) — 在 x86 与 arm64 上移除了 `PREEMPT_NONE` / `PREEMPT_VOLUNTARY`，暴露了这个窗口。

## 当日社区讨论

一封公开的 bpf 邮件列表 RFC 提议在非 RT 内核上禁用 trampoline 递归 enter/exit 周围的 preemption，从而 per-CPU 递归跟踪与 per-CPU 私有栈不会被跨被抢占的任务共享。该系列附一个 47 行复现器，并实测了误报递归丢失率：voluntary-preemption 内核 0 百分，对照 full- 与 lazy-preemption 内核 25-28 百分。一个自动邮件列表评审在补丁里只挑了一个注释风格的细节。到目前为止的讨论很窄：焦点是在所有非 RT 内核上是否可接受在热点 trampoline 路径里加 preemption 屏障，以及私有栈所有权论证是否足以证明保留 per-CPU 计数器是合理的。

本次运行的存档来源在 7 天窗口内产出 0 条消息（两个 opt-in 存档来源在快照时为空），因此上面的问题取自被允许的公开 bpf 邮件列表，经其公开的 atom feed 审阅，RFC 线程作为一手参考。eunomia-bpf 与 sched-ext 的 Discord 频道以及 r/eBPF 在本次运行未覆盖（没有可用的可见浏览器会话），故标记为未覆盖而非安静。
