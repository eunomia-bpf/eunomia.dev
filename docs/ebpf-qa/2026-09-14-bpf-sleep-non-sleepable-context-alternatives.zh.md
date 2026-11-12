# 为什么 BPF 程序不能睡眠或阻塞，在不可睡眠上下文里应该用什么替代？

**简短回答：** 一个 BPF 程序只有在两个条件同时成立时才可以睡眠：它是以*可睡眠*方式加载的（`BPF_F_SLEEPABLE`，附着到内核上下文允许睡眠的钩子上），**并且**当前执行不在任何临界区内。验证器会静态地同时检查这两点。如果一个可能睡眠的 helper 被从不可睡眠的程序里调用——或者从 `bpf_spin_lock`、`bpf_rcu_read_lock`、禁止抢占的区域、禁止中断的区域内部调用——加载就会被拒绝，报错为 `sleepable helper <name>#<id> in <context>`，其中 `<context>` 精确指出原因（`non-sleepable prog`、`lock region`、`rcu_read_lock region`、`non-preemptible region` 或 `IRQ-disabled region`）。这不是一个可以通过调优绕过的运行时风险，而是加载期的错误。正确的替代不是“少睡一点”，而是换一种机制——per-CPU map 槽位、在严格约束下使用 `bpf_spin_lock`、BPF 原子指令，或者把工作推迟出去（ring buffer、`bpf_loop`、尾调用，或可睡眠的异步回调）。

## 规则是上下文的属性，不是 BPF 的属性

内核在程序加载标志里直接写明了这个约定：如果设置了 `BPF_F_SLEEPABLE`，“验证器会限制这类程序的 map 与 helper 用法。可睡眠 BPF 程序只能附着到内核执行上下文允许睡眠的钩子上。这类程序可以使用可能睡眠的 helper，例如 `bpf_copy_from_user()`”。也就是说，可睡眠性是按程序授予的，而且只在附着点的上下文允许时才成立。

只有少数几种程序类型是可睡眠的。在 libbpf 的程序类型表里，`Sleepable` 一列为 `Yes` 的是 `fentry.s`、`fexit.s`、`fmod_ret.s`、`fsession.s`、`iter.s`、`lsm.s`、`struct_ops.s`，以及 uprobe 家族的 `uprobe.s`、`uretprobe.s`、`usdt.s`、`uprobe.multi.s`、`uretprobe.multi.s`、`uprobe.session.s`，再加 `BPF_PROG_TYPE_SYSCALL`（`syscall`）。同一张表里其余各行的该列均为空：XDP、tc、socket filter、cgroup 钩子、普通的（无 `.s` 后缀的）tracing 与 kprobe 变体、tracepoint 以及 perf-event 程序都**不是**可睡眠的。附着到这些钩子上的程序，即使你想让它睡眠也做不到。

随后验证器检查程序*内部*的动态上下文。它对“可睡眠上下文”的定义是五个否定条件的合取（`kernel/bpf/verifier.c`）：

```c
static inline bool in_sleepable_context(struct bpf_verifier_env *env)
{
	return !env->cur_state->active_rcu_locks &&
	       !env->cur_state->active_preempt_locks &&
	       !env->cur_state->active_locks &&
	       !env->cur_state->active_irq_id &&
	       in_sleepable(env);
}
```

两半都重要：程序必须是可睡眠的（`in_sleepable`），并且四个“处于临界区内”的计数器都必须为零。当检查在某个 helper 调用处失败时，验证器通过 `non_sleepable_context_description()` 打印失败原因，该函数把每个计数器映射为一个名字：`rcu_read_lock region`、`non-preemptible region`、`IRQ-disabled region`、`lock region`，或者——当程序本身就不是可睡眠的——`non-sleepable prog`。

这正是这个错误信息有用的原因：它告诉你违反了两个条件中的哪一个。`sleepable helper bpf_copy_from_user#190 in non-sleepable prog` 意味着程序类型不对。同样的信息以 `lock region` 或 `rcu_read_lock region` 结尾，则意味着程序类型没问题，但调用位于某个临界区内，必须移出去。

这条规则同样适用于 BPF 到 BPF 的调用以及 kfunc。当一个被标记为可能睡眠的全局函数在不可睡眠的上下文中被调用时，验证器会输出 `sleepable global function <name>() called in <context>` 并拒绝加载。对可睡眠 kfunc，验证器会区分两种条件：从不可睡眠程序调用时输出 `program must be sleepable to call sleepable kfunc <name>`，从临界区内部调用时输出 `kernel func <name> is sleepable within <context>`。持锁时还有一条相关限制：`global function calls are not allowed while holding a lock, use static function instead`。

## 为什么内核不能简单地放行

睡眠意味着当前任务主动让出 CPU，并要求调度器之后重新运行这个任务。只有在以下前提成立时才合理：内核拥有一个可恢复的任务上下文，并且没有关掉调度器所需的各种机制。在 RCU 读侧临界区内、自旋锁内、禁止抢占的区域里，或者中断被关闭时，这些前提都不成立——代码可能运行在软中断或硬中断上下文，或者位于一个绝不能被抢占的区域。此时一个阻塞调用要么死锁（等待一个根本无法送达的唤醒），要么破坏该临界区所保证的原子性。因此验证器的静态拒绝是一项证明义务，而非启发式判断：它拒绝生成任何“沿着某条路径*可能*在原子区域中抵达睡眠调用”的程序。

这也是“把程序改成可睡眠”并非万能解药的原因。即便程序是可睡眠的，它也可以显式进入这些区域：`bpf_rcu_read_lock()`——一个 kfunc——会打开 RCU 读侧临界区，而验证器自己的 `in_rcu_cs()` 就把这样的区域在对可睡眠性的判定上视为禁止抢占。kfunc 文档中写道，`KF_RCU_PROTECTED`“在不可睡眠程序中默认成立，而在可睡眠程序中必须通过调用 `bpf_rcu_read_lock` 显式保证”——所以需要访问 RCU 保护数据的可睡眠程序必须自行管理该区域，而位于其中的睡眠调用会像自旋锁内的调用一样被拒。

## 应该用什么替代

正确的替代取决于当初想用睡眠做什么。下面三类几乎覆盖了所有情况。

**1. 消除共享状态（per-CPU map）。** 对于计数器和 per-CPU 暂存区，per-CPU map 给每个 CPU 独立的一份槽位，两个 CPU 永不争用，也无需加锁。内核对这一内存模型有明确表述：`BPF_MAP_TYPE_PERCPU_ARRAY`“为每个 CPU 使用不同的内存区域，而 `BPF_MAP_TYPE_ARRAY` 使用相同的内存区域”。per-CPU 值上限为 `PCPU_MIN_UNIT_SIZE`（32 kB），必须按 CPU 读取；用户态负责聚合各 CPU 的副本。`bpf_this_cpu_ptr()` 返回某个 per-CPU ksym 在当前 CPU 上的副本，且“永远不会返回 NULL”；`bpf_per_cpu_ptr()` 则接收一个显式的 CPU。per-CPU map **不**提供跨 CPU 一致性——如果你需要跨 CPU 的单一一致值，那它就是用错了工具。

**2. 就地保护共享值（`bpf_spin_lock`、原子操作）。** 当两个 CPU 确实必须更新同一个 map 值时，`bpf_spin_lock` 是受支持的原语，但它在 `bpf-helpers(7)` 中带有硬性约束：

- 手册页的历史规则是“只能用在 `BPF_MAP_TYPE_HASH` 或 `BPF_MAP_TYPE_ARRAY` 类型的 map 里”；当前验证器更宽，也接受 `bpf_obj_new()` 分配的 BTF（`MEM_ALLOC`）对象内的锁；无论哪种情况都必须有 map 的 BTF 描述；
- “BPF 程序一次只能取一把锁，因为取两把或更多可能造成死锁”；
- “每个 map 元素只允许一个 `struct bpf_spin_lock`”，位于值的顶层、按 4 字节对齐、不可嵌套，且**不能**在栈上或数据包里；
- “锁被持有时，不允许调用（无论 BPF 到 BPF 还是 helper 调用）”，且锁区内禁止 `BPF_LD_ABS`/`BPF_LD_IND`；
- “BPF 程序必须在返回前、在所有执行路径上调用 `bpf_spin_unlock()` 释放锁”；
- “`bpf_spin_lock` 仅对 root 可用”，并且 tracing 与 socket filter 程序“由于抢占检查不足”而不能使用它；
- map-in-map 的内层 map 中不允许使用。

自旋锁同样不是睡眠的替代品：因为持锁期间不能调用 helper，而且它会禁止抢占，所以持有它做任何较长的工作本身就是 bug。对于单字更新，普通的 BPF 原子指令和原子 map 更新可以完全绕开锁——`bpf_map_update_elem()`“以原子方式替换已有元素”，而 BPF 指令集也将原子操作定义为其 ISA 的一部分。

**3. 把工作推迟到可以睡眠的上下文。** 如果这项工作确实需要阻塞——usercopy、分配内存、I/O——就不要在你不能阻塞的地方做它，把它移走。按隔离程度递增，可选方案有：

- **Ring buffer**：`bpf_ringbuf_reserve()` / `bpf_ringbuf_submit()` / `bpf_ringbuf_output()` 把负载交给用户态，由普通进程上下文去做阻塞工作。
- **有界迭代**：`bpf_loop()` 最多执行 `1 << 23` 次回调，回调上下文在栈上；它让长时间工作*能够终止*，但**不会**让它变得可睡眠。
- **尾调用**把程序拆开，同样是有界化工作，而不改变上下文。
- **异步回调**会替你改变上下文。这是微妙之处：`bpf_timer_start()` 会在“某个 CPU 的软中断上下文中”调用所配置的回调，而验证器按可睡眠性对回调分类——在 `is_async_cb_sleepable()` 中，一条注释写道“bpf_timer 回调永远不可睡眠”，而“bpf_wq 和 bpf_task_work 回调永远可睡眠”。所以 timer 回调是一个*不同的*不可睡眠上下文，而不是对规则的逃脱；`bpf_wq`/`bpf_task_work` 回调*才*是逃脱之道。
- **可睡眠程序**：如果这项工作属于一个可以睡眠的钩子，就在那里附着 `*.s`（可睡眠）程序——例如 `fentry.s`/`fexit.s`/`lsm.s` 或 `BPF_PROG_TYPE_SYSCALL`——在这些程序里，`bpf_copy_from_user()` 这类睡眠 helper 是被允许的。

## 如何判断自己撞上了哪种情况

验证器错误已经点明了上下文。照字面读它：

```text
sleepable helper bpf_copy_from_user#190 in non-sleepable prog   -> 程序类型不对
sleepable helper ... in lock region                             -> 调用位于 bpf_spin_lock 内
sleepable helper ... in rcu_read_lock region                    -> 调用位于 bpf_rcu_read_lock 内
sleepable helper ... in non-preemptible region                  -> 抢占被禁止
sleepable helper ... in IRQ-disabled region                     -> 中断被关闭
sleepable global function foo() called in <context>             -> 某个可能睡眠的 BPF 到 BPF 被调者
program must be sleepable to call sleepable kfunc bar           -> kfunc 需要一个可睡眠程序
kernel func bar is sleepable within <context>                   -> kfunc 被从临界区内调用
```

一条最小的复现与排查路径：

```sh
# 1. 找到确切的拒绝信息以及被点名的上下文。
sudo bpftool prog load ./obj.o /sys/fs/bpf/p 2>&1 | grep -i 'sleepable\|context\|lock'

# 2. 确认程序不是可睡眠的，且其附着点是原子的：
#    检查 ELF section（没有 '.s' 后缀 => 不可睡眠）以及钩子类型。

# 3. 出问题的 helper 是否必须睡眠？检查待拷贝的值能否不用睡眠就取到。
grep -n 'BPF_FUNC_copy_from_user\|bpf_probe_read_kernel' ./prog.c

# 4. 如果调用在锁内，把它移出去，并重新确认锁在 helper 调用前的每条路径上都已释放。
```

如果信息以 `non-sleepable prog` 结尾，而这个 helper 又确实必须睡眠，那就是程序类型不对——把这段逻辑移到可睡眠程序或 `bpf_wq`/`bpf_task_work` 回调中。如果它以某个临界区名字结尾，就把调用移出该临界区（或者换用一个不睡眠的 helper，例如 `bpf_probe_read_kernel()`）。

## 决定性的局限

规则不是“BPF 不能睡眠”，而是一个必须在每个程序点都成立的合取：程序可睡眠 **并且** 没有临界区处于活动状态。由此有两条边界，且都会在实践中咬人。第一，可睡眠性是*附着点*的属性：你无法通过更换 helper 把它加到 XDP、tc 或普通 kprobe 程序上——钩子的上下文不允许睡眠。第二，异步回调并非自动可睡眠：`bpf_timer` 回调本身就是不可睡眠的，所以把工作搬进 timer 可能重现同样的错误；只有 `bpf_wq`/`bpf_task_work` 回调（以及可睡眠程序类型）才给出一个睡眠 helper 合法的上下文。当你无法满足这个合取时，答案永远是同一个形状：在程序里做原子性的那一部分（per-CPU 槽位、自旋锁或原子指令），把阻塞性的那一部分推迟到允许睡眠的上下文。

## 参考资料

- [Linux 内核文档：Program Types and ELF Sections](https://docs.kernel.org/bpf/libbpf/program_types.html)
- [Linux 内核文档：BPF_MAP_TYPE_ARRAY 与 BPF_MAP_TYPE_PERCPU_ARRAY](https://docs.kernel.org/bpf/map_array.html)
- [Linux 内核文档：BPF_MAP_TYPE_HASH](https://docs.kernel.org/bpf/map_hash.html)
- [Linux 内核文档：BPF Instruction Set Architecture (ISA)](https://docs.kernel.org/bpf/standardization/instruction-set.html)
- [bpf-helpers(7) — Linux 手册页](https://man7.org/linux/man-pages/man7/bpf-helpers.7.html)
- [Linux 内核源码：`kernel/bpf/verifier.c`](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/verifier.c)
- [Linux 内核源码：`include/uapi/linux/bpf.h`](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/include/uapi/linux/bpf.h)

## 当日社区讨论

本监控窗口不具备技术性。过去 24 小时内，两个选择加入的 Slack 归档只返回了一小段回退内容，全部是会议事务——一条周期性排期通知、一句共同演讲致谢、一个会议链接——没有任何 eBPF 问题、症状或设计争议。两个白名单中的聊天工作区本次运行没有可见的浏览器会话，公开邮件列表与论坛归档也未审阅。这些来源按“不可用覆盖”如实记录，而不是“安静”。由于可读归档没有产出问题，本篇问答以公开一手资料为依据，而非某条社区消息：这是一个反复出现的实践者问题——某个可能睡眠的 BPF helper 被从绝不能阻塞的上下文里调用，加载随即以带上下文信息的验证器错误失败——并针对内核 BPF 文档、上游验证器与 UAPI 源码、BPF 指令集规范以及 `bpf-helpers(7)` 做了核对。此处不复制任何私有文本、身份、频道或链接。
