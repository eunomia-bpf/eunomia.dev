# BPF task 迭代器为什么分为可睡眠与不可睡眠两种程序，哪些进程视图分别落在哪一侧？

简短回答：人们常说的那条分界 —— 不可睡眠程序读 exe 视图、可睡眠程序读 cmdline 视图 —— 恰好把实际的分界说反了。RESCHED 是一种权限，而不是睡眠性闸门：task 家族的三个 seq 迭代器全部设置了它，而它只是授权迭代器在对象之间调用 `cond_resched()`。唯一的挂载期拒绝是单向的：可睡眠程序挂到没有该位的 target 上会被拒，而不可睡眠程序挂到 `task`、`task_file`、`task_vma` 上都不会有任何问题。各进程视图的分界取决于数据存在哪里，而不是程序能否睡眠：`comm` 是结构体内部的 16 字节无锁字段；`exe` 是 `mm_struct` 里带 RCU 保护的文件指针，在两种执行上下文中都可读；`cmdline` 是用户态内存块，只能拷贝、不能解引用。真正与可睡眠执行路径耦合的视图是 `task_vma` 遍历：它的 seq 循环在 vma 之间持有 mm 的 mmap 读锁，并在返回用户态之前放下它；新的 kfunc task 迭代器遍历任务链表时完全不碰 mmap 锁，可以在不可睡眠程序中使用。

## 机制

**RESCHED 是一个 feature 位，不是睡眠闸门。** 每个 task 家族的 seq 迭代器都注册了它：

```c
static struct bpf_iter_reg task_reg_info = {
	...
	.feature		= BPF_ITER_RESCHED,
	...
};
```

`task_file` 与 `task_vma` 的注册块、以及 cgroup 迭代器上都有同一个位。这个位真正起作用的地方在定制的 seq 读循环 `bpf_seq_read` 里，它运行在持有 seq file mutex 的过程中：

```c
can_resched = bpf_iter_support_resched(seq);
while (1) {
	...
	if (num_objs >= MAX_ITER_OBJECTS) { ... }
	...
	if (can_resched)
		cond_resched();
}
```

也就是说，RESCHED 只是让读循环在对象之间可以自愿重新调度，每次读取封顶 `MAX_ITER_OBJECTS` 即一百万个对象；当一次 read 在触到上限前就填满了缓冲，用户态读取方就会按 `-EAGAIN` 循环到 EOF 为止。这条路径里没有任何东西要求、或授予程序睡眠能力。

**sleepable 标记属于程序，挂载闸门是单向的。** 程序通过 `BPF_F_SLEEPABLE` 加载标志变成可睡眠的，verifier 会把这一类记录在程序上，并相应限制它的 map 与 helper 使用。迭代器的挂载路径只检查一个条件，在 `bpf_iter_attach_iter` 中：

```c
/* Only allow sleepable program for resched-able iterator */
if (prog->sleepable && !bpf_iter_target_support_resched(tinfo))
	return -EINVAL;
```

后果是不对称的：不可睡眠程序可以挂到任何 RESCHED target，包括全部三个 task 家族迭代器，所以 exe、cmdline、comm 视图没有任何睡眠性要求。反过来，可睡眠程序只能挂到声明了该位的 target；挂到没有 RESCHED 的 target 会在创建 link 时失败。

**各视图按数据位置分界。** `comm` 不是一个独立查找：它就是 `task_struct` 里的 16 字节 `char` 数组，没有任何锁保护，因此在 task 迭代器上运行的程序可以无锁读取，完全不需要访问 mm。`exe` 是 `mm_struct` 的字段 `exe_file`，带 RCU 标记声明：

```c
/* store ref to file /proc/<pid>/exe symlink points to */
struct file __rcu *exe_file;
```

`bpf_iter_run_prog` 的两种执行上下文都在运行程序前取 RCU 读锁，所以这个指针在两侧都合法可读；把指针变成可打印的路径字符串是另一个独立步骤，走内核的文件路径 helper，而不是指针读取本身。`cmdline` 根本不在 `task_struct` 或 `mm_struct` 里。内核暴露的是用户态 argv 内存块，即 `mm_struct` 的 `arg_start` 与 `arg_end` 这一对，在 exec 时写入、写方由 `arg_lock` 保护；BPF 侧用用户内存拷贝原语 `bpf_probe_read_user` 与 `bpf_probe_read_user_str` 把它拷贝出来。这些是拷贝原语，在不可睡眠上下文同样可用，所以 cmdline 读取本身并不强制可睡眠性。它真正强制的是：数据住在另一个进程的地址空间里，因此必须拷贝而不能解引用。

**task_vma 遍历才是与睡眠性耦合的视图。** 它的 seq 循环在遍历某个 task 的 vma 时持有 mm 的 mmap 读锁，并在返回用户态之前刻意放下这把锁，让长时间读循环不跨越用户态边界持有它；下一次 read 时重新获取并用 `find_vma` 重新定位。当锁被争用时，遍历会先放下再重取，用的是 killable 变体，使中断信号能杀掉卡住的读：

```c
if (mmap_lock_is_contended(curr_mm)) {
	info->prev_vm_start = curr_vma->vm_start;
	info->prev_vm_end = curr_vma->vm_end;
	op = task_vma_iter_find_vma;
	mmap_read_unlock(curr_mm);
	if (mmap_read_lock_killable(curr_mm)) {
		mmput(curr_mm);
		goto finish;
	}
}
```

`task_vma_seq_stop` 在交还控制权给读取方之前释放这把锁。可睡眠执行路径才是这把锁的睡眠侧真正所在：

```c
if (prog->sleepable) {
	rcu_read_lock_trace();
	migrate_disable();
	might_fault();
	...
}
```

内核侧的对应物是 `bpf_find_vma`：它用 task 的 `alloc_lock` 钉住 task 的 mm，用 trylock 取 mmap 读锁，争用时返回 `-EBUSY` 而不是阻塞，并把解锁通过 per-CPU 的 helper 延迟到 IRQ work，使其绝不在不能睡眠的上下文中睡眠。

**kfunc task 迭代器不带任何 mmap 锁。** 另一个较新的入口用不透明句柄包装同样的任务链表遍历：

```c
struct bpf_iter_task {
	__u64 __opaque[3];
} __attribute__((aligned(8)));
```

`bpf_iter_task_new` 返回句柄，`bpf_iter_task_next` 在线程场景走 `__next_thread`、在进程场景走 `next_task`，`bpf_iter_task_destroy` 收尾。由于遍历本身不碰 mm 也不碰 mmap 锁，这对函数可以在不可睡眠程序中使用。

由此得出的实用规则：一个每个对象做超出琐碎工作的 task 迭代器程序，应当放在可睡眠一侧；task 家族里真正迫使这一选择的视图是 vma 遍历，因为它的内核循环持有并重新获取一把对睡眠敏感的锁。`comm`、`exe`、`cmdline` 不带来这种约束；它们的约束来自数据住在哪里，而不是程序能否睡眠。

## 验证与调试路径

1. 导出目标内核的 BTF：`bpftool btf dump file /sys/kernel/btf/vmlinux | grep -E "comm|arg_start|arg_end|exe_file"`，确认字段布局：`task_struct` 里 16 字节的 `comm` 数组、`mm_struct` 的 arg 范围对、带 RCU 标记的 `exe_file` 指针。
2. 搭挂载矩阵。把同一个程序体分别以可睡眠与不可睡眠两种变体加载，两种都挂到 `task`、`task_file`、`task_vma`；两者都应成功。再把可睡眠变体挂到没有 RESCHED 位的 target 上，确认 link 创建以 `-EINVAL` 失败，即那条单向闸门在起作用。
3. 验证读循环契约。对大表跑迭代器，确认用户态按 `-EAGAIN` 循环到 EOF，并且 `task_vma` 在缓冲填满后重入时不会重复或漏掉任何 vma。
4. 与各视图的 ground truth 对账。从 BPF 侧 dump `(pid, comm, exe)` 三元组，与用 `bpf_probe_read_user_str` 从 arg 范围读出的 argv 字节对照；BPF 侧的 argv 必须与内核自己的 /proc 数据对同一进程报告的一致。

一个最小的 BPF-C 三视图读取示例，跑在 task 迭代器上：

```c
SEC("iter/task")
int read_views(struct bpf_iter__task *ctx)
{
	struct task_struct *task = ctx->task;
	char comm[TASK_COMM_LEN];
	bpf_probe_read_kernel(comm, sizeof(comm), &task->comm);

	struct mm_struct *mm = task->mm;
	if (!mm)
		return 0;

	struct file *exe;
	bpf_probe_read_kernel(&exe, sizeof(exe), &mm->exe_file);

	char cmdline[256] = {};
	if (bpf_probe_read_user_str(cmdline, sizeof(cmdline),
	    (void *)mm->arg_start))
		cmdline[0] = 0;

	/* key = pid, value = { comm, exe pointer, argv } */
	return 0;
}
```

## 局限

- mmap 读锁的耦合是 `task_vma` seq 遍历独有的。本主线快照里没有任何 task 家族的 kfunc 以程序可调用的加锁/解锁函数对形式持有 mm 锁；遍历在内核内部以 killable 锁获取它，而 `bpf_find_vma` 用 trylock 加 IRQ-work 延迟解锁包装同一把锁。
- `exe` 是指针读取。把 mm 的 `exe_file` 解析成路径字符串，是独立于指针读取的另一个步骤；RCU 标记只保证指针读取在两种执行上下文都安全。
- 内核线程没有 `mm_struct`，因此 argv 范围与 exe 指针都不存在。task 迭代器仍然交出 `task_struct` 指针，程序在推导任一用户内存视图之前必须对 `task->mm` 做空检查。
- 每次读取一百万个对象是每缓冲上限，不是总量。用户态读取方必须预期 `-EAGAIN` 并循环到 EOF；`task_vma` 在缓冲填满后重入，正是 vma 遍历跨越该边界仍保持一致的机制。

## 参考

- task 家族 seq 迭代器、mmap 锁遍历、kfunc task 迭代器与 `bpf_find_vma`：https://elixir.bootlin.com/linux/latest/source/kernel/bpf/task_iter.c
- seq 读循环、RESCHED 权限、单向可睡眠挂载闸门与执行路径：https://elixir.bootlin.com/linux/latest/source/kernel/bpf/bpf_iter.c
- `BPF_ITER_RESCHED` feature 位与迭代器注册结构：https://elixir.bootlin.com/linux/latest/source/include/linux/bpf.h
- `mm_struct` 的 arg 范围字段与带 RCU 标记的 `exe_file` 指针：https://elixir.bootlin.com/linux/latest/source/include/linux/mm_types.h
- `task_struct` 的 `comm` 数组与 `TASK_COMM_LEN`：https://elixir.bootlin.com/linux/latest/source/include/linux/sched.h
- `bpf_probe_read_user` 与 `bpf_probe_read_user_str` 原型：https://elixir.bootlin.com/linux/latest/source/kernel/bpf/helpers.c
- `bpf_link` 与迭代器 link 类型：https://man7.org/linux/man-pages/man7/bpf_link.7.html

## 当日社区讨论

覆盖情况：通过只读快照可触达 **2** 个已加入 watchlist 的 Slack 归档，滚动窗口内共 **4** 条消息；其中 **1** 条是仓库内部 bug 报告，按匿名化规则弃用，**1** 条与技术无关。允许名单里的浏览器专属社区在本次运行中无可见浏览器会话可用，标记为**不可触达，而非无讨论**。下文不复现任何 Slack/Discord 工作区、频道、参与者或消息 URL。

可达窗口里的两条线索：

- **到底谁测过 tracepoint 与 LSM hook 的成本差？** 一个关于仅用 tracepoint 构建端点 agent 的线程，反复假设 hook 成本相对 LSM 方案可忽略。悬而未决的问题是：在同一个固定负载上做差分成本研究 —— 相同事件、两类 hook，热路径上每事件的墙钟时间。
- **在大型主机上，周期性 /proc 轮询正走向生命尾声。** 在大型多路机器上，完整遍历 /proc 要数秒，而 eBPF task 迭代器远小于一秒即可完成，线程因此推荐用迭代器替代轮询。该线程里的一条附带说法是：exe 视图需要不可睡眠程序，cmdline 视图需要可睡眠程序。这个说法说反了：exe 视图正是无锁、可 RCU 读取的那一侧，cmdline 视图是一次用户内存读取，task 家族里真正有 mmap 锁耦合的视图是 vma 遍历。修正后的分界，就是本页回答的内容。
- **经 `sock:inet_sock_set_state` tracepoint 做 socket 归属**，承自上日的回答：connect/accept 探针会漏掉经 socket 激活或 `SCM_RIGHTS` 交接而换了主人的 socket，`TIME_WAIT` socket 会在新属主下复活，而 UDP 严格来说根本不是"连接"。

4 条消息的可达窗口即今日完整可达集；浏览器专属社区仍未审阅，未擅自当作无讨论。
