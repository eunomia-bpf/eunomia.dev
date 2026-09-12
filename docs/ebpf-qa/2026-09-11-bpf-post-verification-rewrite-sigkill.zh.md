# 为什么 SIGKILL 无法终止 BPF 程序加载，即使验证已经成功？

**简短回答：** 加载过程中可以被终止的只有主验证循环；验证完成之后的指令重写尾段完全没有取消点：

1. **验证可以被终止，重写尾段不行。** 验证器的主循环会检查挂起的信号并返回 `-EAGAIN`，内核要求时也会重新调度。但这个循环只是 `bpf_check()` 的一个阶段，不是整个加载过程。
2. **验证返回之后，内核会执行既不检查信号、也不重新调度的指令重写通道。** 特权路径逐条硬化死代码分支、删除死代码、删除 NOP；非特权路径净化死代码；JIT 常量致盲则逐条重写每个含常量的指令。每次重写都走同一个补丁辅助函数，它会移动指令数组和辅助数据数组、重新调整所有分支偏移——整体工作量与程序长度成平方关系。因此，挂起的 SIGKILL 要等重写尾段结束才能终止任务；而特权加载者可以用一个足够大的程序把尾段拖得很长：上限是 100 万条指令，报告的复现用例用了约 13.1 万条"无条件跳转 + 返回"指令对，验证很快完成，但 `bpf_opt_remove_nops()` 逐条删除它们会花很长时间。

目前有一个补丁系列正在评审中，把重写辅助函数变成取消点和重新调度点；今天上午，一位 bpf 列表维护者在回复中认可了这条路线。

## 可终止性到底停在哪里

加载路径上仅有的取消点位于验证器主循环（`kernel/bpf/verifier.c` 的 `do_check()`，主线内核第 18422-18426 行）：

```c
		if (signal_pending(current))
			return -EAGAIN;

		if (need_resched())
			cond_resched();
```

这就是为什么杀掉一个卡住的*验证*是有效的。但 `do_check()` 返回后，`kernel/bpf/core.c` 里的 `bpf_check()` 会继续执行指令级重写通道：

```c
	if (is_priv) {
		if (ret == 0)
			bpf_opt_hard_wire_dead_code_branches(env);
		if (ret == 0)
			ret = bpf_opt_remove_dead_code(env);
		if (ret == 0)
			ret = bpf_opt_remove_nops(env);
	} else {
		if (ret == 0)
			sanitize_dead_code(env);
	}
```

每个通道都通过共享辅助函数 `bpf_patch_insn_data()` 和 `verifier_remove_insns()` 逐条修补或移除指令。每次这样的调用都要重新分配或移动 `insns` 和 `aux` 数组并调整所有分支偏移，总工作量随程序长度呈平方增长。在主线内核中，这些辅助函数没有一个检查 `fatal_signal_pending()`，它们的逐条循环内部也没有重新调度——所以在重写窗口内送达的 SIGKILL 会被记为挂起，但无法让任务在重写尾段结束前退出。

JIT 常量致盲的形状与它相同。`bpf_jit_blind_constants()`（`kernel/bpf/core.c`，主线第 1562 行）遍历程序，通过 `bpf_jit_blind_insn()` 加 `bpf_patch_insn_data()` 逐条重写常量指令，循环内部没有信号检查。它在 `bpf_prog_need_blind()` 为真时从 `bpf_prog_jit_compile()` 运行，也用于 `kernel/bpf/fixups.c` 中 `jit_subprogs()` 的卸载子程序。还有一个致盲特有的细节：JIT 把补丁失败当作"退回解释器"处理，因此如果在补丁过程中收到致命信号并被当作补丁失败消费掉，它会被这条回退路径悄悄吞掉。正在评审的补丁因此在 `bpf_check()` 里加了两处 `fatal_signal_pending(current)` 复查，置 `-EINTR`——一处在 `bpf_fixup_call_args()` 之后，另一处在 `__bpf_prog_select_runtime()` 之后——确保解释器回退不会把取消吞掉。

### 放大器：复杂度上限数的是指令数，不是重写工作量

`include/linux/bpf.h` 把 `BPF_COMPLEXITY_LIMIT_INSNS` 设为 1000000——源码树里的注释是 "yes. 1M insns"。特权加载者可以提交到该上限的程序。报告的利用形状是 131072 条指向零的无条件跳转，每条后面跟一个合法返回：这样的程序验证会很快完成（状态空间很平凡），但随后 `bpf_opt_remove_nops()` 必须逐条删除这些跳转，每次删除都是一次完整的数组搬移加分支偏移重定向。整个窗口内任务都处于 `D`（不可中断）状态。

### 正在推进的修复

一个 7 补丁系列（v2 评审中，`Fixes: 52875a04f4b2`，"bpf: verifier: remove dead code"），由一位外部安全研究者报告，把 `bpf_patch_insn_data()` 和 `verifier_remove_insns()` 变成公共的取消与重新调度点。理由是：它们本就在 `BPF_PROG_LOAD` 进程上下文中运行，补丁辅助函数在重新分配时本来就可以睡眠，而大多数调用方都会直接传播补丁失败。今天一位 bpf 列表维护者的回复认可了"正面解决"的方向——改掉 `bpf_patch_insn_data()` 的实现，而不是继续绕着走。讨论了两个设计——重写前把指令数组换成链表，以及在循环中累积补丁、一次性应用——其中单遍应用被认为最自包含，是计划先尝试的方案。线程中悬而未决的问题是：那两处致命信号复查是否应该放进常量致盲路径里。

## 如何验证

1. **复现卡住的加载。** 构造一个约 13.1 万条"无条件跳转 + 返回"指令对的程序，从专用进程以特权加载。验证完成后（`bpf()` 系统调用仍在飞行中）发送 SIGKILL。未打补丁的内核上，任务停留在 `D` 状态——看 `/proc/<pid>/stat` 的 stat 字段——`bpf()` 调用在重写尾段结束前不会返回。打上补丁后，`BPF_PROG_LOAD` 会迅速返回 `-EINTR`，任务随即消失。
2. **在源码中确认缺口。** 在加载路径上 grep：主线内核中 `fatal_signal_pending` 检查存在于 `do_check()` 循环，而 `bpf_opt_remove_nops()`、`bpf_opt_remove_dead_code()`、`sanitize_dead_code()`、`bpf_jit_blind_constants()` 的逐条循环内部既没有信号检查，也没有 `cond_resched()`。
3. **确认修复的落点。** 该系列在 `bpf_check()` 中加了两处 `fatal_signal_pending()` 复查（`bpf_fixup_call_args()` 之后、`__bpf_prog_select_runtime()` 之后），并把重写辅助函数变成重新调度点。

## 回答的适用边界

- 不可终止的重写是*当前*主线内核的行为；修复是评审中的 v2 系列，尚未合入。文中的行号指向今天核对的主线快照。
- 修复让尾段变为*可抢占*，而不是变快：重写的平方成本仍然存在。超大型程序的验证本身仍可能耗时很长，但它可以经由 `-EAGAIN` 被终止；补丁关掉的是重写窗口。
- 致盲回退是真实行为：致盲失败时程序跑在解释器上；这两处复查正是为了它——此前致命信号会被这条回退吞掉。
- 卸载子程序（`fixups.c` 路径）有自己独立的致盲调用点；补丁后的辅助函数覆盖了它们，但子程序 JIT 有自己的终结路径，系列合入后值得再核对一遍。
- 这是 [2026-09-08 回答](/zh/ebpf-qa/2026-09-08-malicious-ebpf-payload-boundaries/) 所讨论的武器库中又多出的一件：一个杀不掉的加载本身就是一种拒绝服务向量，即便加载行为本身是被允许的。

## 参考

- bpf 列表，线程 "bpf: Make post-verification instruction rewrites killable"（7 补丁系列的 v2，含公开的维护者回复）：[回复消息](https://lore.kernel.org/bpf/917a177ed71802d933c03fa154662b50f5fffd14.camel@gmail.com/) — 补丁描述、报告的 131072 条跳转复现、维护者回复及两个备选设计
- [kernel/bpf/verifier.c](https://github.com/torvalds/linux/blob/master/kernel/bpf/verifier.c) — `do_check()` 主循环中 `signal_pending(current)` → `-EAGAIN` 与 `need_resched()` → `cond_resched()`，加载路径上仅有的取消点
- [kernel/bpf/core.c](https://github.com/torvalds/linux/blob/master/kernel/bpf/core.c) — `bpf_check()` 中验证之后的重写段落；`bpf_jit_blind_constants()` 及致盲失败时退回解释器的路径
- [kernel/bpf/fixups.c](https://github.com/torvalds/linux/blob/master/kernel/bpf/fixups.c) — `jit_subprogs()`，卸载子程序的致盲调用点
- [include/linux/bpf.h](https://github.com/torvalds/linux/blob/master/include/linux/bpf.h) — `BPF_COMPLEXITY_LIMIT_INSNS` 设为 1000000
- 正在评审的系列本身（7 补丁，v2）：`bpf_patch_insn_data()` 与 `verifier_remove_insns()` 作为公共取消/重新调度点，外加 `bpf_check()` 中的两处 `fatal_signal_pending()` 复查

## 当日社区讨论

本次覆盖情况：bpf@vger.kernel.org 公开存档通过其常规公开 feed 完成了 24 小时窗口的审阅，上面这个问题和当日大部分讨论都来自这里。CNCF 工作区已 opted-in 的存档频道可读；其 24 小时窗口安静，7 天回退窗口里只有会议事务——稳定化例会安排、一个会议场次链接、致谢——没有技术问题，如实记为安静而非缺口。Cilium & eBPF 工作区的存档不携带其 allowlisted 频道，记为不可访问而非安静。两个 allowlisted 的 Discord 工作区是纯浏览器可见面，本次运行没有可用的可见浏览器会话，按覆盖缺口上报，而不是当作沉默。公开从业者论坛（r/eBPF）本次无法抓取——HTTP 客户端被阻断——因此上报为未审阅，而不是宣称已审阅。

### 成为今日回答的那个问题

bpf 列表当日最强的线程："Make post-verification instruction rewrites killable"，7 补丁系列的 v2，由一位外部安全研究者报告。报告的形状：特权加载者可以提交一个验证迅速完成、但验证后重写尾段呈平方成本的程序，于是验证成功后发来的 SIGKILL 无法终止任务。一位 bpf 列表维护者当天回复，认可了这个方向——别再绕着走，直接改掉 `bpf_patch_insn_data()` 的实现——桌上摆着两个设计（重写前把指令换成链表，与循环内累积补丁、单遍应用），计划先试其中最自包含的单遍方案。上面的回答在此发布；线程里还留着一个未决问题：那两处致命信号复查是否该放进常量致盲路径。

### 列表上的其余讨论

- **helper/kfunc 参数校验统一。** 一个 23 补丁的 bpf-next 系列（v2）把 helper 与 kfunc 之间重复的参数检查合并成单一参数类型枚举和单一运行时类型解析路径，并附 kfunc 包内存直写与可空 per-CPU kptr 的 selftests。验证器的参数机制是近来 kfunc 工作的主要落点，该系列的目标是让它可维护。
- **ring buffer 的内存核算。** 一个 RFC 提议把 ring buffer 后备页从丢失内存中单独核算——这正是 [2026-09-10 回答](/zh/ebpf-qa/2026-09-10-process-behavior-reconstruction-event-loss/) 在完整性一侧讨论的"事件被静默丢弃"故事的内存侧。
- **JIT 工具链。** MIPS 用户态汇编器支持：为 BPF JIT 增加带符号 div/mod 与符号扩展发射器。
- **selftests。** 新的 kfunc 自测继续为上面的参数校验系列补齐覆盖。
