# 为什么 BPF uprobe 程序能写到决定内核返回用户态的那个寄存器，校验器又该拒绝哪些偏移？

**简短回答：** kprobe 或 uprobe 程序的上下文就是当前任务真实的 `struct pt_regs`，而校验器对写这块上下文的唯一约束是结构体的大小。它的每一个字都可写，包括异常返回路径会重新装回控制状态的那几个字：riscv 的 `status`、arm64 的 `pstate`，以及 x86-64 的 `cs`、`ss` 和 `flags`。ptrace 写的是同一个结构体，但每个架构都有一份代码把这些字段限制在合法范围里；BPF 的上下文检查没有这样一道门。

在 riscv 上，实际后果只有一条指令。`struct pt_regs` 里偏移 256 就是 `status`，返回路径会在 `sret` 之前把它装回 `sstatus`。在那里存一个 `0x100` 就把 `SR_SPP` 置上了，于是这次陷入会以特权态返回到一个用户地址。这是确有其事的缺陷，而且不止 riscv 一家；截至今天，修复只以邮件列表回复里的一个 diff 形式存在，还没有合入。

## 机制

1. **上下文检查限定的是大小，不是字段集合。** `kprobe_prog_is_valid_access()` 只要 `off >= 0` 且 `off + size <= sizeof(struct pt_regs)` 就接受任意偏移，拒绝不对齐的访问，并且一旦是写就把 `prog->aux->kprobe_write_ctx` 置为真。除此之外什么都不看：

   ```c
   if (type == BPF_WRITE)
           prog->aux->kprobe_write_ctx = true;
   ```

   校验器通过一张按程序类型索引的表走到这个函数：`__check_ctx_access()` 调用 `env->ops->is_valid_access()`，而 `BPF_PROG_TYPE_KPROBE` 对应的就是 `kprobe_verifier_ops`。kprobe 与 uprobe 程序共用这一个类型，所以这个窗口对两者同时成立。因为这个类型没有 `convert_ctx_access`，校验器还允许对上下文做变量偏移；边界仍然是结构体大小。

2. **在 riscv 上，导出寄存器集合之后多出的那 32 字节就是控制状态。** riscv 的 `struct pt_regs` 是 288 字节，`status` 在 256，紧接 `t6` 之后；而导出的 `user_regs_struct`——也就是 uapi 交给 BTF 和 BPF 的那个——是 256 字节，止于 `t6`。所以可写窗口恰好比寄存器集合宽 32 字节，`status` 正是这段尾巴的第一个字，后面跟着 `badaddr`、`cause` 和 `orig_a0`。

3. **返回路径信任它。** `ret_from_exception` 把 `PT_STATUS(sp)` 载入 `a0`，用它恢复用户栈指针并写 `CSR_STATUS`。`SR_SPP` 是 `0x00000100`，而 `user_mode()` 就是 `(regs->status & SR_PP) == 0`：同一个位既描述陷入自何处来，也决定 `sret` 回何处去。把它置上，内核就以特权级返回到一个用户虚拟地址。报告者的程序只有一条指令：操作码 `0x7a`（存立即数，双字），目标 `r1`（上下文），偏移 `0x100`，立即数 `0x100`。

4. **arm64 在 `pstate` 上是同样的形状。** `struct pt_regs` 依次是 `regs[31]`、`sp`、`pc`，然后 `pstate` 在 264。内核出口序列从 `S_PC` 取出 ELR 与 SPSR 这一对，然后在 `eret` 之前执行 `msr spsr_el1, x22`。`pstate` 带着运行态和中断屏蔽位，所以写了它的程序等于选定了返回时的异常级和 DAIF 状态。这里的 BPF 上下文是 `user_pt_regs`，它的最后一个字段就是 `pstate`。

5. **x86-64 有三个彼此独立的洞。** `cs` 在 136，`flags` 在 144，`ss` 在 160。返回用户态的路径弹完寄存器后，用 `testb $3, 8(%rsp)` 检查 `CS` 的低字节，RPL 不是 3 就执行 `ud2`；因此写了 `cs` 的程序把一次返回变成一次致命断言，而不是一次受控的转换。`ss` 会被拷进同一个帧。`flags` 由 `iretq` 装入，`IF` 和 `IOPL` 都取自它：清掉 `IF` 就让用户态带着关中断返回，把 `IOPL` 置成 3 则让用户程序可以发端口 I/O。

6. **ptrace 已经把答案按架构写好了。** 调试器写的是同一个结构体，每个架构都会拒绝自己返回路径会盲目重装的那些字段。x86 的 `set_segment_reg()` 拒绝 `cs` 与 `ss` 的空选择子，也拒绝任何 RPL 不是用户的合法选择子，`set_flags()` 用 `FLAG_MASK` 掩出认识的位，而 `FLAG_MASK` 是算术与条件标志加 `NT`，所以 `IF` 与 `IOPL` 设不上去。arm64 把每次写都过一遍 `valid_user_regs()`，它清掉 `pstate` 的保留位，值不成形时就强行捏成一个合法的 EL0 状态。riscv 干脆把 `status` 留在寄存器集合之外：regset 读写的长度是 `sizeof(struct user_regs_struct)`，所以 `PTRACE_GETREGSET` 和 `PTRACE_SETREGSET` 根本到不了偏移 256。三种机制，一条规则：程序可以设置那些架构会按原值重装回去的寄存器。

7. **`kprobe_write_ctx` 是个标记，不是白名单。** 这一位只记录“这个程序写过上下文”。真正消费它的只有一条挂载路径：`kprobe_multi` 挂载返回 `-EINVAL`，注释说 kprobe 不允许写上下文。`uprobe_multi` 挂载完全不检查这一位，所以一个写 `pstate`、`cs` 或 `flags` 的 multi-uprobe 程序今天照样能挂上。单个 kprobe 或 uprobe 程序在挂载侧同样没有检查；kprobes 文档也一直告诉探针处理函数可以去改 `regs->ip`。

8. **提议的修复把规则搬进校验器。** 诊断出这个缺陷的那封回复草图里给了 `arch_uprobe_regs_writable(off, size)`：一个默认返回 false 的弱函数，各架构覆写——riscv 允许到 `offsetof(struct pt_regs, status)`，arm64 到 `offsetof(struct pt_regs, pstate)`，x86-64 允许通用寄存器加 `ip` 与 `sp`，并声明 32 位不支持。`kprobe_prog_is_valid_access()` 会在 `BPF_WRITE` 分支里调用它，先拒绝再置标记。没有覆写的架构就一个上下文写都不许，这是保守的默认值。uprobe 代码的维护者回复说，这个改动由他来发。

## 验证方法

在一台 riscv 机器上，整份报告可以化简为一次加载、一次反汇编和一次存数：

```bash
bpftool prog load uprobe_write.bpf.o /sys/fs/bpf/uw autoattach
bpftool prog dump xlated id "$(bpftool prog show name uprobe_write | head -1 | awk -F: '{print $1}')"
```

反汇编里应当能看到一条存到 `r1 + 256` 的双字写。然后在你手上那个内核构建的源码里核对路径的两端：上下文边界在 `kernel/trace/bpf_trace.c`，`PT_STATUS` 的载入在 `arch/riscv/kernel/entry.S`。偏移会随版本移动，所以要从构建里读，而不是从这一页抄；arm64 由 `asm-offsets` 生成 `S_PSTATE`、`S_PC` 和 `S_SP`，x86 的偏移则跟随结构体字段顺序。

不做 riscv 的验证时，可以在 arm64 上写 `pstate`，或在 x86-64 上写 `cs`，然后看内核停在哪里：x86-64 上错的选择子应当撞上返回用户态路径上的那条断言，注意内核态那条检查才需要 `CONFIG_DEBUG_ENTRY`，用户态那条不需要。把程序、翻译后的指令、内核构建和确切偏移放在一起记录，因为没有偏移的报告在别的版本上无法复现。

这一页没有跑本地复现：这台主机既没有 riscv 硬件也没有 `bpftool`，所以上面的结论全部来自源码和报告，偏移则是对公开结构体布局的算术结果。

## 边界

修复还只是回复里的草图。写作时 `mainline` 和 `bpf-next` 都没有合入对应提交，所以今天编出来的内核仍然接受这个写，说它已经被拒的页面过几天就会错：依赖这个拒绝之前先看一眼树。

`arch_uprobe_regs_writable` 会让规则变成架构相关的，形状是对的，但也是一份维护负担：每个把 pt_regs 导出给 BPF 的架构都要列出自己可写的寄存器，而布局一变边界就悄悄移动。没有覆写的架构会彻底失去上下文写，这安全，但对那些确实在改自己寄存器的程序是一次可见的行为变化。

边界在一个方向上还偏粗：检查针对访问的字节区间，所以一次从 `pstate` 之前开始、落进它内部的写会被拒绝，而恰好压在该字段上的单字写，只有在架构的边界正好停在它前面时才会被抓到。子字长的写和不对齐的组合值得各自有测试，而回复里的 diff 依赖的是 `off + size <=` 比较，不是一张允许区间表。

最后，这是一个已经受信任的程序类别内部的权限问题。它不扩大非特权用户能做的事，也和 `bpf_probe_write_user` 无关：程序写的是内核保存的帧，不是进程的内存。

## 参考

- [报告：riscv uprobe 写 status](https://lore.kernel.org/bpf/30318.1794850523@30-11-166.wireless.csail.mit.edu/) — 那条单指令程序、偏移，以及 `sstatus.SPP` 的后果。
- [回复：不止 riscv，以及与 ptrace 的对照](https://lore.kernel.org/bpf/DMWUY4M0QXMH.33XAPR76ZXKRT@gmail.com/) — 按架构列出的出口路径字段与 ptrace 检查，以及 `arch_uprobe_regs_writable()` 的补丁草图。
- [回复：uprobe 维护者接过这个改动](https://lore.kernel.org/bpf/avt5WnmsPkYkeZrz@krava/) — 确认修复即将发出。
- [内核 `kernel/trace/bpf_trace.c`](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/trace/bpf_trace.c) — `kprobe_prog_is_valid_access()`、写标记，以及 `kprobe_multi` 挂载检查。
- [内核 `kernel/bpf/verifier.c`](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/verifier.c) — `__check_ctx_access()` 与按类型的 ops 表。
- [内核 `arch/riscv/include/asm/ptrace.h`](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/arch/riscv/include/asm/ptrace.h) — 结构体布局与 `user_mode()`。
- [内核 `arch/riscv/kernel/entry.S`](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/arch/riscv/kernel/entry.S) — `PT_STATUS` 的载入与 `CSR_STATUS` 的写入。
- [内核 `arch/riscv/include/asm/csr.h`](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/arch/riscv/include/asm/csr.h) — `SR_SPP` 与 `SR_PP`。
- [内核 `arch/riscv/kernel/ptrace.c`](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/arch/riscv/kernel/ptrace.c) — 止于 `t6` 的寄存器集合。
- [内核 `arch/arm64/kernel/entry.S`](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/arch/arm64/kernel/entry.S) — ELR/SPSR 这一对与 `msr spsr_el1`。
- [内核 `arch/arm64/kernel/ptrace.c`](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/arch/arm64/kernel/ptrace.c) — `valid_user_regs()` 与强制的 EL0 状态。
- [内核 `arch/x86/entry/entry_64.S`](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/arch/x86/entry/entry_64.S) — `CS` 断言与 `iretq`。
- [内核 `arch/x86/kernel/ptrace.c`](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/arch/x86/kernel/ptrace.c) — `set_segment_reg()`、`set_flags()` 与 `FLAG_MASK`。
- [内核 `kernel/events/uprobes.c`](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/events/uprobes.c) — 在真实的 `pt_regs` 上运行程序的恢复时处理函数。
- [内核 kprobes 文档](https://docs.kernel.org/trace/kprobes.html) — 长期以来“处理函数可以修改保存的寄存器”那句说明。

## 今天的社区讨论

这一轮在可见浏览器里读了公开的 bpf 邮件列表，并把选择性加入的聊天归档当作核对过的次要来源。归档快照覆盖两个工作区、三个频道：24 小时窗口里没有消息，七天窗口里两条，都不是技术问题，一条是会议纪要通告，一条是贡献者自我介绍。两条都不是提问，所以上面这个问题来自公开列表。

两个聊天面这一轮读不到，按缺口报告而不是按安静报告：两个 Discord 工作区只能在可见浏览器里看，这一轮没有抓取；归档里两个 Slack 来源指向的邀请页都止于加入页面，其中一个还在同意墙后面，没有任何频道列表可浏览。公开网络上的那个实践者论坛今天只返回一个靠脚本渲染的空壳，线程列表同样读不到。

列表本身很忙：第一页上有 42 个日期为 2026-11-16 的线程根。除了上面这个问题，还有四个线程值得一读。

**抢占下的裸 tracepoint 寄存器槽位。** 裸 tracepoint 与 tracing 程序没有 `pt_regs`，于是有三个辅助函数按每 CPU 的嵌套计数器在三个槽位里挑一个来填，前提是调用者按相反顺序离开。抢占在任务层面打破了这个前提，报告者的复现装置在 140 万个样本里测到每个程序 7 到 18 个样本带着另一个程序的返回地址，另外还有 `bpf_get_stack()` 与 `bpf_get_stackid()` 返回的空栈。修复在 get 到 put 之间关掉抢占。评审者的回复接受这个缺陷，但问了更大的问题：为什么还要每 CPU 存一份 `pt_regs`，每次调用花 168 到 384 字节栈就能消掉这份共享资源；他还给了一张表，列出十四种架构变体的 `pt_regs` 与 BPF 上下文大小。

**一个还没填好的受信任指针。** 两个 backing file 的 LSM 钩子把 `f_path` 尚未装好的 `struct file` 交给程序：分配期间 `f_path.dentry` 是 NULL，而打开失败的文件在释放路径上又可能已经被清空。两个钩子都不在不可信清单里，所以校验器允许程序把这条路径交给按受信任参数取用的 kfunc 去格式化，而 `d_path()` 在 overlayfs 挂载上第一次打开文件时就空指针解引用。把两个钩子加进不可信清单，程序就变成加载期拒绝；这个补丁已经以 `ab86b62012b9` 进入 bpf 树。

**成批拆除校验器的上限。** 一个 21 个补丁的系列把校验器在地图数量、子程序、kfunc、调用帧和指令数上的固定上限换成由可用内存推导的边界，用 cgroup 的内存上限作为新的约束，并据此调整日志与跳转修正结构的大小。它还放宽了远跳、把支配关系计算降到 `O(n log n)`，并加了一个规模很大的自测，去加载超过旧上限的程序。对使用者来说有意思的不是新的最大值，而是失败模式的变化：过去在某个具名上限上失败的程序，现在只会在内存不足时失败，而这同时改变了诊断信息和拒绝服务预算的形状。

**经由 page-pool 内存提供者的 AF_XDP 零拷贝。** 一个十一个补丁的系列让驱动在 bind 时安装 page-pool 内存提供者，由核心用新的 `page_pool_refill_done()` 而不是 NAPI 驱动补充，并把提供者的块不拷贝地送到 socket，同时用一个按内存模型的能力位说明协议栈是否可以保留缓冲区。单驱动单队列上的数字是接收 21.2 Mpps、发送 28.9 Mpps，`XDP_DROP` 在每包 30 条指令下开销 1.9%。仍未解决的是：用户态如何在 bind 之前得知块大小的约束，以及通用 XDP 与 cpumap 无法给提供者的 socket 供料。

窗口里其余的线程小而精确：把 200005 次系统调用换成五次的批量地图转储、BTF 加载器对超过结构体长度的头做的跨端序构建修复、把类型标签从编译器根本无法记录的每 CPU **变量**声明上撤掉，以及终于写明哪些程序类型可以改动 socket map 的文档。