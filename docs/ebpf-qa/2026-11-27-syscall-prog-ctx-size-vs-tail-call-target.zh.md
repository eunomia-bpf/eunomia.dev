# 为什么一个 syscall 程序能读到运行时上下文缓冲区之外，而一次进入另一个 syscall 程序的尾调用又决定了这个缓冲区要多大？

**简短回答：** 因为对 `BPF_PROG_TYPE_SYSCALL` 来说，上下文就是调用者自己给定大小的一块不透明缓冲区，而运行路径里唯一的尺寸检查看的是调用者点名的那个程序——不是尾调用可能跳过去的那个程序。`bpf_prog_test_run_syscall()` 只要求上下文不小于入口程序的 `max_ctx_offset`，也就是校验器记录的这个程序自己碰过的最远字节。而保护程序数组插入的兼容性检查比较程序类型、JIT 与否、可睡眠性以及几个 cookie，唯独不比这个偏移。于是两个 syscall 程序在「上下文有多大」上意见不一，却互相合法尾调用；运行时被调程序原样继承调用者的指针：入口程序一个字节都不读，8 字节的上下文就能跑，被调程序在这块分配之外偏移 8192 处取一个字，而这个字作为 `retval` 回到用户态。

修复的做法是：只要入口程序可能尾调用，就把复制出来的上下文补零到该程序类型允许的最大上下文——65535 字节；同时禁止扩展程序给一个本来没有尾调用的目标加上尾调用。

## 机制

### 1. 尺寸检查在哪里，它比较的是什么

syscall 程序类型是唯一一个上下文不由类型专属结构描述的类型。它的访问回调接受 65535 以内的任意偏移：

```c
static bool syscall_prog_is_valid_access(int off, int size,
					 enum bpf_access_type type,
					 const struct bpf_prog *prog,
					 struct bpf_insn_access_aux *info)
{
	if (off < 0 || off >= U16_MAX)
		return false;
	/* No alignment requirements for syscall ctx accesses. */
	return true;
}
```

由于偏移可以是变量，校验器把这类访问交给通用的内存范围检查，上界给 65535 字节，同时记下到达的最远字节：

```c
	if (is_var_ctx_off_allowed(env->prog)) {
		int err = check_mem_region_access(env, reg, argno, 0, access_size, U16_MAX,
						  zero_size_allowed);
		if (err)
			return err;
		if (env->prog->aux->max_ctx_offset < reg_umax(reg) + access_size)
			env->prog->aux->max_ctx_offset = reg_umax(reg) + access_size;
		return 0;
	}
```

固定偏移访问走 `__check_ctx_access()`，用另一条路径维护同一个字段。无论如何，`max_ctx_offset` 都是**单个程序**的属性，由它自己的代码推出来。

运行路径用的恰恰就是这个属性，而且只针对它收到的那个程序：

```c
	if (ctx_size_in < prog->aux->max_ctx_offset ||
	    ctx_size_in > U16_MAX)
		return -EINVAL;
```

所以可接受的区间是「至少是这个程序要读的，最多 65535 字节」。对别的程序类型来说，左半边几乎是空的，因为上下文布局由类型或挂载目标固定；而对 syscall 类型，它就是全部约定——上下文是调用者传进来的那块缓冲区。

### 2. 尾调用对这个指针做了什么

尾调用不建新栈帧就转移控制权：解释器里直接采用被调程序的指令数组，寄存器一律保持原状，`R1` 仍持有上下文指针；x86 JIT 用同一个 `rdi` 跳进被调程序入口，生成代码的注释自己就写着（"rdi == ctx (1st arg)"）。没有任何第二次尺寸检查，也不存在第二块缓冲区——被调程序读的就是入口程序拿到的那个指针。

这在设计上是合理的，通常也无害：tracepoint、raw tracepoint 或 extension 程序的上下文布局是固定的，同类型的调用者与被调者对同一批字节的理解一致。syscall 类型打破了这个假设，因为「这里有多少字节」是**按程序**给出的判断。

### 3. 为什么兼容性检查抓不到它

程序数组在插入程序时做兼容性检查：

```c
		ret = map->owner->type  == prog_type &&
		      map->owner->jited == fp->jited &&
		      map->owner->xdp_has_frags == aux->xdp_has_frags &&
		      map->owner->sleepable == fp->sleepable;
		if (ret &&
		    map->map_type == BPF_MAP_TYPE_PROG_ARRAY &&
		    map->owner->expected_attach_type != fp->expected_attach_type)
			ret = false;
```

类型、JIT 与解释器的一致性、分片能力、可睡眠性、期望挂载类型、cgroup storage 的 cookie 以及挂载函数原型。`max_ctx_offset` 不在其中，而对其他类型它本来就是多余的。一个从不碰上下文的 syscall 程序和一个在偏移 8192 处取字的 syscall 程序，被报告为互相兼容。

把三件事串起来，一次运行就是一次直白的越界读：

1. 入口程序的 `max_ctx_offset` 是 0，于是用 8 字节上下文调用 `BPF_PROG_RUN` 成功；
2. 被调程序在同一个程序数组里，类型相同所以能进来；
3. 被调程序通过继承来的指针加载，读在 8 字节分配之外 8184 字节处，并把这个值当返回值，最终被拷回用户态成为 `retval`。

如果上下文长度为 0，这条路径甚至没有分配：`ctx` 保持 `NULL`，被调程序的位移就变成一次低地址加载，而不是读邻近的 slab 对象。

### 4. 修复改了什么

三处改动，各关掉上面三个事实之一。

`bpf_prog_test_run_syscall()` 现在按「这个程序能否走到尾调用」而不是按 `max_ctx_offset` 决定复制多大：

```c
	/* the target of a tail call can use that much */
	if (prog->aux->tail_call_reachable)
		size = max_t(u32, size, U16_MAX);

	if (size) {
		ctx = kvzalloc(size, GFP_KERNEL_ACCOUNT | __GFP_NOWARN);
		if (!ctx)
			return -ENOMEM;
		if (copy_from_user(ctx, ctx_in, ctx_size_in)) {
			err = -EFAULT;
			goto out;
		}
	}
```

`tail_call_reachable` 是校验器对程序及其子程序的汇总，只要任一可达路径里有尾调用，就会置在入口程序上。它正是合适的判据：它的含义就是「除这个程序之外，可能还有别的程序去解引用这个指针」。`kvzalloc` 提供补零的填充，所以被调程序读到填充区里时看到的是 0，而不是这块分配之前的使用者留下的内容。

内核侧的调用者没法补，因为它把调用者的指针原样传进去、不做复制。于是 `kern_sys_bpf()` 在程序可能尾调用时直接要求完整大小：

```c
		if (attr->test.ctx_size_in < prog->aux->max_ctx_offset ||
		    (prog->aux->tail_call_reachable && attr->test.ctx_size_in < U16_MAX) ||
		    attr->test.ctx_size_in > U16_MAX) {
			bpf_prog_put(prog);
			return -EINVAL;
		}
```

另外，扩展程序不能再给一个本来没有尾调用的目标加上尾调用，因为补零的决定取自入口程序的标志：

```c
	if (prog->type == BPF_PROG_TYPE_EXT &&
	    tgt_prog->type == BPF_PROG_TYPE_SYSCALL &&
	    prog->aux->tail_call_reachable && !tgt_prog->aux->tail_call_reachable) {
		err = -EINVAL;
		goto out_unlock;
	}
```

形状与紧邻其上的一条检查一致：那条已经拒绝「上下文可达范围超过 tracepoint 或 syscall 目标」的 freplace 程序。

## 如何验证

两个探针，都很便宜，也不需要打补丁的内核。

1. **不借任何新接口就能读出单个程序的有效上下文大小。** 一次测试运行能接受的最小 `ctx_size_in` 就是该程序的 `max_ctx_offset`，因为这项检查就是与它比较。用 0、8、64 去跑入口程序，用 0、8、8192、8196 去跑被调程序：入口程序从 0 就开始成功，被调程序只有在 8196 时才成功（若它的访问是偏移 8192 处的 4 字节读）。两个数字之间的差就是整个 bug，而且在真正走尾调用之前就能看到。
2. **然后再观察这次转移。** 把两个程序都插进同一个 `BPF_MAP_TYPE_PROG_ARRAY`，用一个入口程序自己读不了的较小上下文运行它，打印 `retval`。如果入口程序一个字节都不读而运行返回非零，被调程序就读到了缓冲区之外。每次运行换一个尺寸档位，让分配落在不同地址；尺寸固定时容易复现同一批陈旧字节，让效果看起来像是确定性的。

第三个是否定检查，值得保留：修复之后，同一个入口程序会直接拒绝 4096 字节的上下文，只接受 `ctx_size_in == 65535`。如果它还接受 4096，那么跑的是旧路径，而不是补零后的路径。

## 边界

- **这是一次补零，不是一条兼容性规则。** 两个 syscall 程序仍然可以在「上下文有多大」上意见不一；它们现在总能拿到 65535 字节的补零内存，于是这个分歧对读没有后果。报告里提出的另一条路——把 16 位偏移塞进 map owner 结构——会让插入失败，代价是拒绝掉本来能正常工作的组合。
- **写到调用者缓冲区之外的部分会被丢弃。** 只有 `ctx_size_in` 字节会被拷回，所以被调程序写到调用者尺寸之外的内容落进临时空间。对安全有利，但在结果里看不见。
- **65535 字节是按运行、不是按程序付出的。** 一次测试运行里，能尾调用的 syscall 程序每次都要清零并复制这么多，并计入 memory cgroup；修复前 `memdup_user` 只要精确大小。没有可达尾调用的程序仍走精确大小那条路径。
- **`kern_sys_bpf()` 的调用者在自己的缓冲区里付账。** 那里不复制，所以调用者必须给足 65535 字节；在这个改动之前构建的 light skeleton loader 传的结构更小，现在会失败。
- **上界来自程序类型，而不是内核关于「够用」的某种判断。** `U16_MAX`（65535）同时被访问回调与运行路径使用，所以它就是真的上限；将来若有上下文更大的程序类型，需要它自己的补零规则。

## 参考资料

- [BPF 邮件列表：syscall 程序配合尾调用时的越界上下文访问](https://lore.kernel.org/bpf/CAAYXoE8ORWnqLVofqZf2AxUvnV_Bd1Gsqb3qdJv+dCM2bVkvAg@mail.gmail.com/) — 报告、双程序复现器，以及往 map owner 里放偏移的建议。
- [补丁：让做尾调用的 syscall 程序拿到最大的上下文](https://lore.kernel.org/bpf/20261116152626.4032587-4-alexei.starovoitov@gmail.com/) — 带提交正文与 diff 的那封信。
- [BPF next 树中的提交](https://git.kernel.org/pub/scm/linux/kernel/git/bpf/bpf-next.git/commit/?id=08c9879049705193e77fd98a8103148b620ef0f1) — 三处改动被合入后的形态。
- [syscall 程序的运行路径](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/net/bpf/test_run.c) — `bpf_prog_test_run_syscall()`、尺寸检查、补零与拷回。
- [syscall 与测试运行入口](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/syscall.c) — `kern_sys_bpf()`、扩展程序对目标上下文的检查，以及以 `U16_MAX` 为界的测试运行门槛。
- [校验器的上下文记账](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/verifier.c) — `max_ctx_offset` 的更新、变量偏移的范围检查，以及 `is_var_ctx_off_allowed()`。
- [尾调用的兼容性判断与转移](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/core.c) — map owner 的比较，以及解释器中的尾调用分支。
- [程序 map 的插入](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/arraymap.c) — 程序数组在何处执行兼容性检查。
- [x86 JIT 的尾调用](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/arch/x86/net/bpf_jit_comp.c) — 以第一个参数寄存器携带上下文跳进被调入口的间接跳转。
- [程序类型与属性声明](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/include/uapi/linux/bpf.h) — syscall 程序类型、测试运行属性，以及文档化的上下文尺寸约定。

## 今日社区讨论

覆盖情况：两个选择加入只读归档的 Slack 工作区（三个白名单频道）通过经批准的只读归档读过，24 小时与 7 天窗口内均无消息，所以那一边是安静，而不是未读。两个白名单 Discord 工作区（十个频道）读不到——网页客户端停在登录界面——因此它们的活跃度是覆盖缺口，而不是沉寂。公开的 r/eBPF 页面以普通网页方式读过，其中有一个实用的 AF_XDP 问题，摘要见下。上面的问题与本节其余部分来自公开的 bpf 邮件列表归档。

**只有尾调用一侧才看得到的上下文描述。** 主线那封就是这样一种通用形状最新的清晰例子：只要某个程序类型让调用者决定输入的大小，就一定会存在一条「这个程序读了多少」的记录，而每一条把同一指针递给另一个程序的路径都必须查那条记录。这里记录存在（`max_ctx_offset`），运行路径查了，尾调用路径没查——因为对其他所有程序类型来说答案是个常数。另一点值得带走的是把 `tail_call_reachable` 当判据：代价按运行付出，而且恰好只由那些能把上下文交给别人的程序付出，交不出去的程序不受影响。

**在 generic XDP 里活过了数据帧的 mac 头。** 一个 v2 修复在 generic 模式下、`bpf_xdp_adjust_tail()` 之后重算 `skb->tail`，改为减去 `mac_len`，因为 XDP 帧起点在 `skb->data` 之前 `mac_len` 字节处。评审指向了边界：辅助函数只保证 `xdp->data` 之后至少 `ETH_HLEN` 字节，而 `mac_len` 可以更大——一个 header_ops 里没有 remote 的 GRE 设备会把 mac 头留在外层 IP 头上，`mac_len` 至少是 24。在这种设备上把帧缩到 14 字节，减法就把 tail 放到 data 之前并使 `skb->len` 回绕；之后再调整 tail 的代码会信任这个回绕过的长度。`adjust_head` 一侧存在同一个缺口。可推广的诊断是：信任任何一个常数之前先比较这两个常数——辅助函数的下限是辅助函数的属性，`mac_len` 是设备 header 操作的属性，而 generic XDP 恰好是两者同时在场的地方。

**seccomp 路径里的 classic BPF，以及锁时间花在哪。** 一个 seccomp 过滤器缓存提案被重新对准了一个实测症状：采样显示大量短命实例在程序 pack 分配器的一把全局锁上排队并塌陷。后续分析把分配路径与释放路径分开：分配在复用一个曾经装过 classic BPF 的 pack 时可能触发一次间接跳转预测器刷新；而释放路径执行架构的代码失效，在 arm64 上是一次对代码区的 `memset` 式改写，外加向所有 CPU 发核间中断，等 icache 刷新完成。因为 pack 锁横跨了这一步，每个 loader 都在一次释放后面排队。有用的读法是：一把锁的持有时长可以由它并不保护的东西支配——被失效的区域在 bitmap 清掉之前不会被再次分配出去，所以那个核间中断可以挪到临界区之外，而分配器遍历 pack 才是真正的串行点。

**解释器不支持的指令，被回移。** 那个让程序在用到只有 JIT 才实现的指令形态时拒绝解释器回退的系列，本周同时被排入 6.18 与 6.12 稳定树。对已经运行这两个内核之一的读者，这就是要留意的版本：它防的失败是「在开了 JIT 的机器上加载成功、在 JIT 被关掉或耗尽的机器上之后才崩」，而加载日志区分不出这两者。做这种铺开前该跑的检查，正是本栏此前描述过的那一条：去读程序是否被 JIT（`bpf_prog_info` 的 jited 长度，或测试运行的行为），而不是假定解释器总是可用的回退。

**CO-RE 重定位与立即数的符号位。** libbpf 的一行修复处理这样一次重定位：本地值是零扩展的，而指令立即数是符号扩展的。自从允许 64 位重定位值之后，比较就把一个符号扩展的 32 位立即数与一个 64 位值相比，于是零扩展、第 31 位置位的无符号值——比如编译成 32 位 move 的 `0x80000000` 枚举成员——被判为意外指令。修复比较低 32 位，那正是立即数持有的部分。实际后果是：用较新 clang 构建的 BPF 目标文件可能在较旧 libbpf 上加载失败，而错误指向一条指令，而不是产生它的那个枚举值。

**虚拟链路上 copy 模式的 AF_XDP 发送。** 一个公开提问问：一个只发 64 字节小包的扫描器，是该按队列分片 AF_XDP socket 并共享 UMEM，还是把发送留在 raw socket 上；实测通过单个发送 socket、在 veth 对上约 24k 帧/秒，并把原因归给每次调用 32 个描述符的预算。这个预算是 socket 选项，下限 32、默认 32、上限是发送环大小，所以 32 是用户能要求的最小值，而不是内核施加的上限。copy 模式同时意味着走通用发送路径：每个描述符都要在 socket mutex 下建一个 skb 并把帧拷进去，所以提高预算只是摊薄调用开销，拷贝与串行化都还在。设备能否零拷贝是一个可查询的能力标志，这也是 veth 测量必然是 copy 模式的原因；这正是「每队列一个发送 socket、按核发送」的论据。