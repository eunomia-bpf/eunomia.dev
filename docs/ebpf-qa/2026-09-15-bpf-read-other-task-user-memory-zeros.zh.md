# BPF 程序能否安全读取属于另一个进程的用户态内存？

**简短回答：** BPF 程序无法用 `bpf_probe_read_user*` 这类 helper 直接读取*另一个*任务的用户态内存——这些 helper 通过 `copy_from_user_nofault()` 读取的是*当前任务*的用户态地址空间，一旦访问出错就静默返回零填充的数据。要在 BPF 里检查*别的*任务的用户态状态，有两条受支持的路径：要么在目标任务自己的上下文里跑（uprobe 打在该任务的用户库上、fentry/fexit 打在该任务的内核入口上），让探针本身落在那个任务上，然后 `bpf_probe_read_user()` 自然读它的内存；要么用 `bpf_task_storage_*`（按任意 `task_struct` 作为 key）去读该任务自己存进 `BPF_MAP_TYPE_TASK_STORAGE` 的每任务数据。当前 BPF helper 集合里并没有“从一个与目标任务不匹配的上下文读取任意其他任务用户内存”的单一 helper。

## 为什么 `bpf_probe_read_user` 返回全零而不是错误

`bpf_probe_read_user()` 是故意为容错的。手册页把它描述为“安全地尝试从用户态地址 `unsafe_ptr` 读取 `size` 字节”，成功返回 `0`，失败返回负错误。但这里的“出错”指*当前任务的内存上下文*里该地址不可读。看 `kernel/trace/bpf_trace.c` 里的实现：

```c
static __always_inline int
bpf_probe_read_user_common(void *dst, u32 size, const void __user *unsafe_ptr)
{
	int ret;

	ret = copy_from_user_nofault(dst, unsafe_ptr, size);
	if (unlikely(ret < 0))
		memset(dst, 0, size);
	return ret;
}
```

任何一次访问故障，`copy_from_user_nofault` 都返回 `-EFAULT`，helper 随即把目标缓冲区清零，而不是把错误返回给调用方去当作“内存不存在”处理。两个直接后果：

1. “地址找不到”这一症状表现为 `0` 字节，而不是 helper 返回的 `-EFAULT`。
2. 读取发生在*当前任务的*页表上下文里，因为 `copy_from_user_nofault()` 走的是当前任务的 `mm_struct`，而不是任意某个任务的。

于是，当一个附着在任务 A 上下文里的 BPF 程序对“属于任务 B 地址空间”的用户态地址调用 `bpf_probe_read_user()` 时，要么发生访问故障（B 的页没有映射进 A 的地址空间——跨进程地址的常见情形）结果被清零；要么——在更罕见的情形里，同一个虚拟地址在两边都恰好被映射时——你读到的是 B 的值，而这并不是调用方想要的那个任务的值。helper 本身无法分辨这两种情况。

对大多数内核入口的 kprobe，该 helper 读的是*正在执行被探针函数的任务的*地址空间里的用户态地址——也就是说，是发起系统调用的那个进程。这正是实践者容易踩坑的地方：helper 读的是*当前任务*的内存，而不是你 map 里持有的某个 `task_struct*` 所对应任务的用户态地址空间。如果拥有该地址的任务并不是此刻运行探针的任务，读到的结果就不是你以为的。

目前并没有“从任务 A 上下文直接读任务 B 用户内存”的 helper，因为内核不想把一个不受信任的 BPF 程序变成任意跨任务内存访问的入口。两条受支持的做法：

**路径一：程序运行在 B 上时读 B 的用户数据。**
用 `bpf_get_current_task_btf()` 拿当前 `task_struct`（在任务上下文的 kprobe / uprobe / fentry 里有用），或者用 `bpf_task_storage_get()` 按 `task_struct` 作为 key 去读*B 自己*存进 `BPF_MAP_TYPE_TASK_STORAGE` 的每任务状态。手册页对该 helper 的说明：

> `bpf_task_storage_get(struct bpf_map *map, struct task_struct *task, void *value, u64 flags)` —— “从 `task` 获取一个 `bpf_local_storage`。从逻辑上可以理解为以 `task` 为 key 从一个 map 里取值……该 map 必须是 `BPF_MAP_TYPE_TASK_STORAGE`。”

这正是读*任务局部*数据的正解：把程序附着到在 B 上触发的钩子（B 的用户库上的 uprobe、B 的入口上的 fentry），B 的 BPF task storage 里存着 B 自己的每任务状态。如果 B 需要看到自己的用户态内存，那应该是 B 自己的上下文在读，而不是外来上下文伸进来。

**路径二：以任意 `task_struct` 为 key 的 BPF local storage。**
`BPF_MAP_TYPE_TASK_STORAGE` 加 `bpf_task_storage_get`/`bpf_task_storage_put` 是把任意用户态数据挂到任务上、再让任意任务（包括稍后读它的那个）取回的规范化手段。这个 map 以 `task_struct` 为 key，所以打在 B 的某个内核入口上的 kprobe 可以 `bpf_task_storage_get(map, target_task, …)` 去读 B 的 BPF 局部数据，而稍后在 B 内部触发的 uprobe 也可以读回。这就是内核把“任务维度的 BPF 状态”变成一等公民的方式，无需新增 helper：任务*自己*在自己的上下文里写入，其他上下文通过 task key 读取。

对于**任意的跨任务用户态指针解引用**（即“我想在任务 A 的探针里跟随一个由任务 B 拥有的指针”），当前内核没有专门的 BPF helper。用户态指针不会被自动替你解引用：`bpf_probe_read_user` 是错的工具，因为它解引用的是*当前任务*的页表。实际解法是改变程序*运行的时机*：附着到在任务 B 内部触发的钩子（B 的 `sys_enter_*` 上的 kprobe、B 的 libc / 用户库上的 uprobe、或 B 调用的内核入口上的 fentry），再在那里读。

## 如何判断自己撞上了哪种情况

最小化的排查代码：

```c
struct task_struct *t = (struct task_struct *)bpf_get_current_task_btf();
long r = bpf_probe_read_user(&out, sizeof(out), ptr_from_B);
if (r < 0)          /* -EFAULT：当前任务里该地址不可读 */
    /* 回退：该地址大概率属于一个不是当前任务的任务 */
```

两条检查可以区分两种失败模式：

1. 如果目标地址已知属于任务 B、而程序上下文不是 B，那么对于任何跨进程地址，该读取*都会*返回 `-EFAULT`（被映射为清零），改目标地址也救不回来——问题出在*上下文*。
2. 如果目标地址是 B 的、而 B 其实正是当前任务（例如打在 B 自己的 `sys_openat` 上的 kprobe），那普通的 `bpf_probe_read_user()` 就是好的——所以在怀疑 helper 之前先核对 `task->pid == target_pid`。

对任务维度的数据，优先用 `bpf_task_storage_get` 而不是裸指针读取；它是不依赖页表巧合的内核准机制。

## 决定性的局限

一旦目标是*另一个任务*的用户态内存，`bpf_probe_read_user()` 就是错的工具，因为这个 helper 的语义是“在当前任务的地址空间里读，出错就静默返回全零”。两条受支持的逃生路线：把程序附着到在目标任务上触发的钩子；对任务维度的状态用 `BPF_MAP_TYPE_TASK_STORAGE` 加 `bpf_task_storage_get`/`bpf_task_storage_put`。当前 BPF helper 集合里没有“从一个与目标任务不匹配的 BPF 上下文读取任意其他任务用户内存”的 helper。

## 参考资料

- [bpf-helpers(7) — Linux 手册页，`bpf_probe_read_user`、`bpf_probe_read_kernel`、`bpf_task_storage_get`、`bpf_get_current_task_btf`](https://man7.org/linux/man-pages/man7/bpf-helpers.7.html)
- [Linux 内核源码：`kernel/trace/bpf_trace.c`](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/trace/bpf_trace.c)
- [Linux 内核源码：`include/uapi/linux/bpf.h`](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/include/uapi/linux/bpf.h)
- [Linux 内核文档：Program Types and ELF Sections](https://docs.kernel.org/bpf/libbpf/program_types.html)

## 当日社区讨论

本监控窗口不具备技术性。过去 24 小时内，两个选择加入的 CNCF / Cilium Slack 归档只返回会议事务——一条“stabilization effort”例行会议通知（周一 / 周三 / 周五，美西时间）、一个带访问链接的 Zoom 会议、来自一位共同演讲者的两句致谢、一个 KubeCon 场次链接。没有任何 eBPF 问题、症状或设计争议。两个白名单中的 Discord 工作区（eunomia-bpf、sched-ext）本次运行没有可见的浏览器会话，公开邮件列表与论坛归档也未审阅。这些来源按“不可用覆盖”如实记录，而不是“安静”。由于可读归档没有产出问题，本篇问答以公开一手资料为依据，而非某条社区消息：这是一个反复出现的实践者问题——一个在任务 A 上触发的 BPF 探针无法安全地用 `bpf_probe_read_user()` 读取任务 B 的用户态指针，加载时要么失败、要么静默返回全零，而不是给出一个可读的错误——已对照 `bpf-helpers(7)` 手册页、上游 `bpf_trace.c` 实现以及 `include/uapi/linux/bpf.h` 做了核对。此处不复制任何私有文本、身份、频道或链接。
