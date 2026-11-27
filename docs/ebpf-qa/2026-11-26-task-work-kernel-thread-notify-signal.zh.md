# 为什么给内核线程安排一个 BPF task work 回调，会让它的可中断睡眠立刻返回？

**简短回答：** 因为带信号语义的那个调度 kfunc 用一个只有「返回用户态」才会清掉的线程标志去通知目标。kfunc 以信号通知模式排队回调，而这个模式会在目标线程上置位 `TIF_NOTIFY_SIGNAL`。清掉这个标志的地方只有三处：`get_signal()` 开头、`ptrace_stop()` 结尾，以及 io_uring 等待 inflight 请求时手写的一处；内核线程一处都走不到，所以这个标志会伴随它余生。`signal_pending()` 把这个标志当作信号，于是线程里每一个 `TASK_INTERRUPTIBLE` 睡眠都不阻塞、直接返回 `-ERESTARTSYS`，`schedule()` 也把任务留在运行队列上。回调也不是「排队了只是晚点跑」：task work 就跑在同样那些退出路径上，对内核线程而言只在退出时执行。已合入的修复是对内核线程目标返回 `-EPERM`，而不是假装通知成功了。

## 机制

### 1. kfunc 对目标做了什么

两个调度 kfunc 都汇入同一个函数，真正的入队工作被推迟到 `irq_work`，这样从 NMI 上下文和任何程序类型里调用都合法：

```c
static int bpf_task_work_schedule(struct task_struct *task, struct bpf_task_work *tw,
				  struct bpf_map *map, void *callback_fn,
				  struct bpf_prog_aux *aux, enum task_work_notify_mode mode)
{
	...
	irq_work_queue(&ctx->irq_work);
	return 0;
}

__bpf_kfunc int bpf_task_work_schedule_signal(...)
{
	return bpf_task_work_schedule(task, tw, map__const_map, callback, aux, TWA_SIGNAL);
}
```

那个 `irq_work` 最终调用 `task_work_add(task, &ctx->work, TWA_SIGNAL)`，目标就是在这一步被动的：

```c
	case TWA_SIGNAL:
		set_notify_signal(task);
		break;
```

`set_notify_signal()` 做两件事：置位线程标志；如果任务此刻正处于可中断睡眠，把它唤醒，好让通知被及时处理而不是等到下一次无关的唤醒：

```c
static inline bool __set_notify_signal(struct task_struct *task)
{
	return !test_and_set_tsk_thread_flag(task, TIF_NOTIFY_SIGNAL) &&
	       !wake_up_state(task, TASK_INTERRUPTIBLE);
}
```

注意形状：标志只置一次、不会重设；真正起作用的是那次唤醒。这条路径没有任何一处是用户任务专属的，也没有任何调用者检查目标会不会回到用户态。

### 2. 标志在哪里被清掉

`clear_notify_signal()` 在树里只有三个调用点：`get_signal()` 开头、`ptrace_stop()` 结尾，以及 io_uring 等待 inflight 请求处手写的一处。`get_signal()` 由 `arch_do_signal_or_restart()` 调用，而后者由返回用户态之前的循环在看到这个标志时执行：

```c
		if (ti_work & (_TIF_SIGPENDING | _TIF_NOTIFY_SIGNAL)) {
			futex_fixup_robust_unlock(regs);
			arch_do_signal_or_restart(regs);
		}
```

这个循环跑在「离开内核、回到用户态」的路上。内核线程永远不走这条路，所以这条路径对它不可达。剩下的一条路径是线程死亡：`do_exit()` 调用 `exit_task_work()`，后者就是 `task_work_run()`，排队的回调此刻才执行。io_uring 那一处很有参考价值，因为内核在自己代码里遇到过同一个问题，并用手动清标志绕开：

```c
		if (test_thread_flag(TIF_NOTIFY_SIGNAL)) {
			clear_notify_signal();
			if (task_work_pending(current))
				set_notify_resume(current);
		}
```

### 3. 为什么可中断睡眠不再睡眠

`signal_pending()` 把通知标志当成信号，只留一个例外口子：

```c
	if (unlikely(test_tsk_thread_flag(p, TIF_NOTIFY_SIGNAL)) &&
	    likely(!(READ_ONCE(p->flags) & PF_NO_NOTIFY_SIGNAL)))
		return 1;
	return task_sigpending(p);
```

`signal_pending_state()` 由此把它变成「不要阻塞」，而真正要紧的两处都会查它：completion 等待在睡前先查一次；`schedule()` 在阻塞任务时再查一次，命中就把状态写回 `TASK_RUNNING`、把任务留在运行队列上，而不是真正让出去。

于是一条写成「睡到 completion 被 signal，反正 kworker 不收信号」的等待，在标志被置位的那一刻就变成了忙等。内核自己的线程创建循环是同一个形状，这也是 io_uring 那份报告除了崩溃之外还多出 100% CPU 空转的原因：

```c
	for (;;) {
		set_current_state(TASK_INTERRUPTIBLE);
		if (list_empty(&kthread_create_list))
			schedule();
```

### 4. 报告里的损坏链

报告中的受害者是 ring 的拆除流程：

```c
		/*
		 * This is really an uninterruptible wait ... But it's also run from
		 * a kworker, which doesn't take signals, so it's fine to make it
		 * interruptible.
		 */
	} while (!wait_for_completion_interruptible_timeout(&ctx->ref_comp, interval));
```

注释写明了通知所破坏的那个前提：一个正等引用计数降到零、切到原子模式的 kworker 被当作收不到信号，所以在此处用可中断等待是安全的，还能避免卡住任务的报告。标志一旦被置位，这个等待在第一轮就返回 `-ERESTARTSYS`，循环在 `ctx->refs` 归零之前退出，`io_ring_ctx_free()` 于是在原子切换仍在飞行时就调用 `percpu_ref_exit()`——正是引用计数代码里那条告警描述的状态：

```c
	if (percpu_count) {
		/* non-NULL confirm_switch indicates switching in progress */
		WARN_ON_ONCE(ref->data && ref->data->confirm_switch);
		free_percpu(percpu_count);
```

在一个仍会去累加并 put 它、已排队的 RCU 回调之下释放 per-CPU 计数器，就是同一份报告里的 slab-use-after-free。

### 5. 修复，以及为什么是「拒绝」

已合入的改动是共享调度函数里的三行，位于所有分配和引用计数之前：

```c
	if (task->flags & PF_KTHREAD)
		return -EPERM;
```

`PF_KTHREAD` 在 `copy_process()` 里按 `args->kthread` 置位、其余情况清掉，`/proc/<pid>/status` 用 `Kthread:` 字段暴露同一位。`-EPERM` 也是更老的辅助函数对内核线程目标的答案：`bpf_send_signal()` 与 `bpf_probe_write_user()` 早已用 `-EPERM` 拒绝 `PF_KTHREAD`（以及 `PF_EXITING`）。

选择拒绝而不是修复，来自第 2 节：对内核线程来说，即使通知完全正常，回调也不会在退出之前跑起来，所以「接受但迟到」和「接受但永不执行」对程序而言无法区分。让调用失败，是程序唯一能据以动作的答案。

## 在活动内核上验证

1. 先确认目标是内核线程。`/proc/<pid>/status` 里 `Kthread: 1` 就是它；`ps` 输出里方括号中的调度器名是同一批任务。工作队列 worker 也是内核线程，`get_task_comm()` 对此有显式处理。
2. 别去怪排队的信号。`kthreadd()` 在派生任何线程之前就把自己的所有 handler 改成忽略，子线程继承这张表，所以内核线程没有 handler 可跑；`/proc/<pid>/status` 里的 `SigPnd`、`ShdPnd` 若有值，那才是真信号。一个信号队列为空的内核线程却返回 `-ERESTARTSYS`，指向的就是通知标志，因为 `ignore_signals()` 只清 `TIF_SIGPENDING`。
3. 看任务状态，而不是症状。`__schedule()` 在把任务出队之前先做信号检查；命中时它把状态写回 `TASK_RUNNING`、把任务留在运行队列上，于是 `/proc/<pid>/status` 里的 `voluntary_ctxt_switches` 几乎不动，而线程在一个它自认为是睡眠的等待里烧 CPU。
4. 在 vmcore 里直接读标志。通知是一位线程标志，调试器可以在 dump 里列出带 `TIF_NOTIFY_SIGNAL` 的任务，确认其中之一是没有任何用户态状态的任务——这个组合是任何信号都造不出来的。
5. 读 kfunc 的返回值。返回 0 说明内核接受了内核线程目标；返回 `-EPERM` 说明检查在位。

## 边界

- `TWA_RESUME` 不是同样坏，也不是同样对。它置的是 `TIF_NOTIFY_RESUME`，`signal_pending()` 不看这个位，所以线程里的睡眠照常工作；但回调仍要等 `resume_user_mode_work()` 或线程退出，对内核线程而言一样迟到。
- 这个检查是拒绝，不是修复。任何其他 `task_work_add()` 调用者排到内核线程上的工作，仍然只在退出时执行；`task_work.c` 里也没有任何地方要求目标必须是用户任务。
- `PF_NO_NOTIFY_SIGNAL` 可作为逃生口，但它是 `current` 上的一个标志、通过保存/恢复的 guard 设置，所以一个线程只能替自己压制这次唤醒，替不了别人；它无法让「别人排到我身上的通知」变得无害。
- 不是所有内核侧 worker 都能被这个测试识别。io_uring 自己的 worker 以用户 worker 创建（内核线程位对它们被清掉），并且在循环里显式跑 task work，所以新的拒绝既抓不到、也不该抓它们。

## 参考资料

- [BPF 邮件列表：在 task work kfunc 中拒绝内核线程](https://lore.kernel.org/bpf/20261126053016.1145482-1-rhkrqnwk98@gmail.com/) —— 报告、空转，以及引用计数告警。
- [BPF 树中的提交](https://git.kernel.org/bpf/bpf/c/3c625090b46b) —— 那三行检查，以及指出 kfunc 系列的 `Fixes` 标签。
- [kfunc 文档](https://docs.kernel.org/bpf/kfuncs.html) —— 隐式参数 kfunc，以及 task work 的声明与调用示例。
- [task work](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/task_work.c) —— `task_work_add()` 及其通知模式、`task_work_run()`。
- [调度器信号辅助函数](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/include/linux/sched/signal.h) —— `clear_notify_signal()`、`set_notify_signal()`、`PF_NO_NOTIFY_SIGNAL` guard，以及 `signal_pending()`。
- [信号投递](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/signal.c) —— 两个 `clear_notify_signal()` 调用点、`ignore_signals()`、`flush_signals()`。
- [返回用户态](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/entry/common.c) —— `_TIF_NOTIFY_SIGNAL` 分派到 `arch_do_signal_or_restart()`。
- [线程创建](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/kthread.c) —— 带可中断 `schedule()` 的创建循环，以及启动时的 `ignore_signals()`。
- [进程退出](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/exit.c) —— `do_exit()` 里的 `exit_task_work()`，内核线程唯一的路径。
- [completion](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/sched/completion.c) —— 可中断等待与它的 `-ERESTARTSYS` 返回。
- [核心调度器](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/sched/core.c) —— `try_to_block_task()` 和那个把任务留在运行队列上的信号检查。
- [io_uring](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/io_uring/io_uring.c) —— 拆除循环及其可中断 completion 等待，以及 inflight 等待里的手动清标志。
- [per-CPU 引用计数](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/lib/percpu-refcount.c) —— `confirm_switch` 告警，以及那个释放计数器的 RCU 回调。
- [基础辅助函数](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/trace/bpf_trace.c) —— 信号与用户写辅助函数早已在用的 `PF_KTHREAD` 拒绝。
- [proc 文档](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/Documentation/filesystems/proc.rst) —— `status` 里的 `Kthread` 字段与上下文切换计数。

## 今日社区讨论

本次覆盖情况：通过授权的只读归档读了两个选择加入归档的 Slack 工作区（共三个白名单频道），24 小时与 7 天窗口里都没有消息，所以 Slack 一侧是「安静」而不是「读不到」。两个白名单 Discord 工作区（共十个频道）读不到——网页客户端停在登录页——因此它们的缺席是覆盖缺口，不是没有讨论。r/eBPF 的公开页面按普通网页读到了，确实带了一个真实的技术问题，记在下面；本篇选的问题取自公开的 bpf 邮件列表归档，本节其余主题也来自那里。

**一个只有返回用户态才会清掉的标志。** 今天的主线就是上面那次内核线程拒绝，对读者有价值的是机制而不是那三行。要凑齐崩溃需要三个条件同时成立：目标是内核线程、通知模式置了一个线程标志、而线程里有一个可中断等待，其作者假定这个线程收不到信号。少任何一个都不会触发。这也是修复落在共享函数而不是 io_uring 等待处的原因：「内核线程不收信号」对**信号**成立，对 task work 的通知不成立，于是内核上下文里的每一个可中断等待都是候选。仍然开放的部分是工具箱里的另一件工具：`PF_NO_NOTIFY_SIGNAL` guard 只替 `current` 压制唤醒，所以一个想让自己在一段关键等待里不可被打断的线程可以做到，但没有任何任务能替它正在安排工作的目标做到。

**存在目的地址槽位里的一个 socket。** 同一天的第二个报告是一个与 eBPF 相邻的 sockmap 别名 bug。stream parser 程序可以调用 socket redirect 辅助函数，后者把目标 socket 存进 skb 的 redirect 字段；这个字段与目的地址指针、引用计数位共用同一个字，而 parser 路径从不清它。verdict 路径会在跑程序之前清、在跑完之后重新取判定，于是两条路径的差别恰好就是修复加的那两行：parser 之前 drop 掉 dst，parser 之后清掉 redirect。症状是 KASAN 在 per-CPU 计数器代码深处报空指针解引用，来自 skb 拆除时的 `dst_release()`，因为一个 socket 指针被当成 `dst_entry`。这条教训不止适用于这个 bug：任何把 skb 交给程序、随后又可能在「没跑过那个程序」的路径上释放它的代码，都得先决定这个字归谁。

**对齐，第三版。** 那个选择去补齐 map 布局而不是改校验器规则的系列这次带来了实测表：每种 map 类型的 value 地址对 8 取模，两种位宽各一份，64 位上是 lpm trie IPv4 与 devmap 错位，32 位上是分配器头部的几种情形错位。它还说明了评审发现的既有 32 位隐患——per-CPU 指针修补时对 4 字节 per-CPU 指针做 8 字节 load——并把它留在范围之外，这是部分修复的诚实形态。对读者可迁移的一点是：布局改动与校验器规则改动失败方式不同——布局改动可以用一条对返回地址的断言来测，规则改动在没有会触发故障的架构时根本无从测起。

**modify-return 程序可以挂到哪些函数上。** 一个 v2 系列收窄了构建期标记为「返回错误」的函数集合，排除 BPF 自己依赖的那些：带 BPF 类型参数的函数、kfunc 与 helper、vmlinux 里的 `bpf_` 与 `__bpf_` 符号，以及 BPF 目录和两个 JIT 下的代码。它给出了剔除的数量（arm64 构建上约 14 400 个里剔掉 795 个），这才是有用的部分：标记不是函数原型自身的属性，而是「调用者会检查返回值」的一个断言，而一个消费该标记的校验器或 JIT，本身会被一个程序改动，改动之后校验器又要重查这个程序自己的返回值。仍然开放的边界是调用点覆盖：调用者忽略返回值的已标记函数仍然带着标记。

**一处递归，notrace 属性和 deny 列表都能修。** 挂在 RCU unlock-special 路径上的 trampoline `fentry` 程序，可以经由 trampoline 自己的 read-unlock 一路递归到栈保护页。报告的分析是 ftrace、fprobe、kprobes 都拿递归 trylock 而 BPF trampoline 不拿，所以 BPF 侧的修法是 deny 列表；RCU 维护者更偏好通用修法——把该函数标 `notrace`——线程最后收敛到先取第一版。值得记住的模式是：「哪个机制在递归」和「该排除哪个机制」是两个不同的问题，而能扩展的答案在能承载它的最低那一层。

**local storage 的内存统计，以及它下面的分配策略。** 两条 local storage 线程并行：一条把每个元素的分配计入 map 上报的内存用量，做法是遍历 bucket 链表，并指出并发插入删除时这个计数不精确；另一条在调用点选择分配变体——可睡眠程序用可睡眠分配，调用方可能拿 zone 锁时用 nowait，其余走无锁路径——而不是永远付无锁路径的代价。两条线程从两侧描述同一个张力：storage 元素占据真实内存用量的大头，而统计的精度不会超过它测量的那条分配路径。

**copy 模式下的 AF_XDP 发送。** 一条公开社区帖问：一个发送密集、64 字节小包的工作负载，是该按队列分片 AF_XDP socket 并共享 UMEM，还是干脆放弃 copy 模式下的 AF_XDP 发送；它测到在 veth 上单个 TX socket 约 24k 帧/秒，并把上限归因于每次 `sendto` 32 个描述符的预算。原始代码对这个归因给出了有用的更正。copy 模式下 `sendto` 走到通用发送路径，持有 socket mutex，对每个描述符先预留一个 completion 槽位、分配 skb、把帧拷进去、再以一次设备发送提交。限制这个循环的预算是 socket 选项（`XDP_MAX_TX_SKB_BUDGET`），下限 32、默认 32、允许取到 TX ring 大小——所以 32 是用户能要求的最小值，不是内核强加的上限。调高它只能摊薄每次调用的开销；每描述符一个 skb 加一次拷贝仍在，而 mutex 会串行化共享该 socket 的每个发送者。这才是「每队列一个 socket、每核自己发」的论据：跨队列共享 UMEM 需要每个 netdev/queue 组合各一对 FILL 与 COMPLETION 环，而这对环是单生产者单消费者，所以「每核一个 TX 环」是 API 支持的形状。问题的另一半——某个网卡能否 zero-copy——是设备声明的特性，不是 netdev 类型的属性：veth 只声明 basic、redirect 与 RX scatter-gather，所以 veth 上的测量必然是 copy 模式，而 virtio-net、gve、mlx5、ice、i40e 都声明了 zero-copy 标志，ena 一个都没声明。