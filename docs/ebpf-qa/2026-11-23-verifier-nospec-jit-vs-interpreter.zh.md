# 为什么在 JIT 编译的 arm64 BPF 程序里一条推测屏障就够了，而在 BPF 解释器里不行？

**简短回答：** 关键在于屏障**落在哪里**，而不只是有没有屏障。校验器允许某些只在推测路径上出错的程序通过，做法是在它证明不了安全的那条指令**前面**插一条 `nospec`（内部指令：`BPF_ST` 类、模式 `0xc0`），然后停止校验该推测路径。JIT 会把这它编译成直线指令流里的一条屏障：arm64 上是 `sb`（或 `dsb nsh; isb`），x86-64 上是 `lfence`。解释器却是通过跳表分发的——`goto *jumptable[insn->code]`——于是屏障落在了一条**间接分支之后**。arm64 没有 retpoline，x86-64 在选中 eIBRS 或 IBRS 时会在运行时关掉 retpoline；一个能训练该间接分支的程序可以让 CPU 跳过屏障、推测执行下一条指令。此外，arm64 解释器里干脆**没有**屏障：它调用 `barrier_nospec()`，而 arm64 只定义了 `spec_bar()`、没定义 `barrier_nospec()`，于是用的是 `linux/nospec.h` 里那个空定义。正在评审的修法不是换更强的屏障，而是把解释器从「启用了 bpf() 系统调用的 arm64 与 x86-64 内核」里去掉。

## 机制

### 1. 校验器发什么，什么时候发

校验器对 Spectre v1 的处理是「插入屏障、放行程序」，而不是拒绝。当校验一个**推测**状态时遇到非致命错误，校验器在它即将检查的那条指令上打标记并停止探索该路径：

```c
if (error_recoverable_with_nospec(err) && state->speculative) {
        insn_aux->nospec = true;
        insn_aux->alu_state = 0;
        goto process_bpf_exit;
}
```

而 `error_recoverable_with_nospec()` 只认可 `-EPERM`、`-EACCES` 和 `-EINVAL`。校验结束后，`bpf_convert_ctx_accesses()` 会在每条被标记的指令前补一条 `nospec`。这条指令不属于 UAPI，反汇编器把它打印成 `(c0) nospec`；因为语义被拓宽过，源码注释写的是它同时覆盖 Spectre v1 和 v4。

关键性质是：屏障只有在「CPU 不能在屏障退休前执行它之后的指令」时，才真正保护**架构可见**的程序。下游的一切推论——包括解释器能否满足它——都建立在这个前提上，而这也是那组补丁争论的核心。

### 2. JIT：屏障就在指令流里

两个 JIT 都在 `nospec` 指令所在的位置直接发一条屏障，所以 CPU 看到的序列是 `... load; barrier; next-insn ...`，中间没有分支：

- arm64：CPU 报告 `ARM64_HAS_SB`（来自 `ID_AA64ISAR1_EL1` 的 SB 字段）时发 `sb`，否则发 `dsb nsh` 加 `isb`。
- x86-64：`lfence`。

这就是 JIT 路径被认为是安全的原因：程序前面某条条件分支被错误预测，也不会让下一条指令越过 `sb` 或 `lfence`，因为屏障在程序顺序上紧邻其后。

### 3. 解释器：屏障在间接分支之后

解释器是一张 256 项的表上的 computed goto：

```c
select_insn:
        goto *jumptable[insn->code];
```

`ST_NOSPEC` 只是表里的一个标签，标签体就是 `barrier_nospec()`。于是两件独立的事情同时出问题：

**分支本身。** arm64 没有 retpoline；x86-64 在拥有 enhanced/legacy IBRS 的 CPU 上会在运行时关掉 retpoline。一个能塑造分支预测器的程序，可以让 CPU 从 `nospec` 标签的**下一条**指令继续推测执行，从而整条跳过屏障。这是分发方式的性质，与屏障本身无关。

**arm64 上的屏障是空的。** `include/linux/nospec.h` 提供

```c
#ifndef barrier_nospec
# define barrier_nospec() do { } while (0)
#endif
```

x86 在自己的 `asm/barrier.h` 里覆盖了这个名字，用 `alternative()` 在 CPU 具备 `X86_FEATURE_LFENCE_RDTSC` 时打入 `lfence`。arm64 根本没定义它：它在 `arch/arm64/include/asm/barrier.h` 里有一个叫 `spec_bar()` 的东西，供 Spectre v4 缓解代码使用，名字不同。所以在一台仍然带解释器的 arm64 内核上，校验器插入的 `nospec` 执行起来什么也不做；而 `kernel/bpf/core.c` 里 `ST_NOSPEC` 旁边那句「arm64 可以依赖固件的 SSBD 缓解」说的是 Spectre v4，不是 v1。SSBD 是 store-bypass 缓解，挡不住一次被错误预测的加载。

### 4. 内核是怎么带上解释器的

`CONFIG_BPF_JIT` 是个独立布尔项，而只要 `CONFIG_BPF_JIT_ALWAYS_ON` 没开，解释器就存在：

```c
#ifndef CONFIG_BPF_JIT_ALWAYS_ON
static u64 ___bpf_prog_run(u64 *regs, const struct bpf_insn *insn)
...
#endif
```

`BPF_JIT_ALWAYS_ON` 原本只是用户的选择。第一个修法（已进入 bpf-next，等下个合并窗口）新增了一个架构符号 `ARCH_WANT_BPF_JIT_ALWAYS_ON`，由 x86-64 和 arm64 选中，隐藏提示符、令其默认为开；同时把 sysctl 变成固定值，并把 JIT 失败从「静默回退到解释器」改成加载错误（`-ENOTSUPP`）。但它仍留下一个缺口：用 `BPF_JIT=n` 构建的内核根本没有编译进 JIT，`BPF_JIT_ALWAYS_ON` 也就无从生效。后续补丁让 `BPF_SYSCALL` 在这两个架构上选中 `BPF_JIT`，只留一个刻意的例外，因为 sanitizer 内核需要解释器：

```c
select BPF_JIT if ARCH_WANT_BPF_JIT_ALWAYS_ON && !KMSAN
```

KMSAN 看不见 JIT 程序所做的写，会把落在程序栈上的一把 map key 报成未初始化；解释器正因如此显式地对该栈做了 unpoison，所以 sanitizer 构建继续用它。32 位 x86 保留自由选择，因为那套 JIT 并不覆盖全部指令。

## 在一台真实内核上验证

1. 先问内核它构建了什么、解释器到底有没有编进去。`/proc/sys/net/core` 下的 `bpf_jit_enable` 只在 `CONFIG_BPF_JIT` 打开时存在；开了 `BPF_JIT_ALWAYS_ON` 时它的最小值和最大值都是 1，任何写入都返回 `EPERM`，内核只打印一次「已启用、值被固定」的提示。

2. 把同一个程序的两种形态都 dump 出来。`bpftool prog dump xlated` 会显示 `(c0) nospec` 位于校验器证明不了的那条指令之前；`bpftool prog dump jited` 则在同一点显示 arm64 的 `sb`（或 `dsb nsh` 加 `isb`）、x86-64 的 `lfence`。如果第二个 dump 报 "not jited"，这个程序走的是解释器，屏障就是上面描述的 computed goto 版本。

3. 看清楚缓解是给谁用的。当 JIT 声明自己不需要缓解、CPU 缓解被关闭、或加载进程持有 `CAP_PERFMON` 时，`env->bypass_spec_v1` 为真；对这类加载者，校验器根本不插 `nospec`。非特权 BPF 默认关闭（`kernel.unprivileged_bpf_disabled` 为 2），所以 arm64 解释器的暴露面要求管理员显式打开非特权 BPF，或者某个特权程序被刻意用不含 `CAP_PERFMON` 的 token 加载。

4. 正面确认 arm64 的 v1 空洞，而不是靠「没找到」。解释器里的屏障是 `barrier_nospec()`；在 `arch/arm64/` 下 grep 这个名字什么也找不到，而 `spec_bar()` 存在。那就是空定义，不是更慢的屏障。

## 限制

解释器问题是**纵深防御**层面的论证，这些修法并不让解释器变安全；它们只是把解释器从「另有真正屏障可用」的两个架构上删掉。解释器还留下的地方——32 位 x86、没有 `ARCH_WANT_BPF_JIT_ALWAYS_ON` 的架构、KMSAN 构建、以及后续补丁落地前以 `BPF_JIT=n` 构建的任何内核——同样的推理成立，「我的 `nospec` 生效了吗」的答案是「只有程序被 JIT 了才生效」。邮件列表上的争论也没有声称相反；分歧在于：用改 Kconfig 默认值的方式关掉它，是否值得在同一个窗口里做这种改动，而这同时仍把针对跳表的 Spectre v2 留着。即使有了 JIT，arm64 的屏障也只和 SB 能力一样强：没有 `ARM64_HAS_SB` 的 CPU 拿到的是 `dsb nsh; isb`——正确但更重，而且 JIT 是在编译时依据系统级能力做出这个选择的。

## References

- [BPF 邮件列表：arm64 解释器的 Spectre v1，v1](https://lore.kernel.org/bpf/20261117172341.3152407-1-ryan.roberts@arm.com/) —— 重命名 arm64 屏障并新增 uaccess 判定宏的那个补丁。
- [BPF 邮件列表：首次评审，基于 bpf-next 的 nack](https://lore.kernel.org/bpf/DMYSI3BOQWOC.2P2FZB0H872DR@gmail.com/) —— "in bpf-next arm64 has no interpreter when the JIT is built"。
- [BPF 邮件列表：Kconfig 与既有内核](https://lore.kernel.org/bpf/DN07ZMOAD4GP.37PG3XUPBG4CM@gmail.com/) —— 解释器内核在 stable 和 `BPF_JIT=n` 情形下意味着什么。
- [BPF 邮件列表：间接分支的反对意见](https://lore.kernel.org/bpf/DN2ALO12MI1Z.3HI553HDMO2FL@gmail.com/) —— `goto *jumptable[insn->code]`、arm64 无 retpoline，以及提议的 Kconfig 片段。
- [BPF 邮件列表：维护者对 v2 与 CSV2 的回复](https://lore.kernel.org/bpf/awRd1mHcQDVNi9U9@arm.com/) —— 若强制打开 JIT，这个补丁就不再必要。
- [BPF 邮件列表：作者收尾该线程](https://lore.kernel.org/bpf/0fad3d48-54cc-4919-bff9-254347c933c4@arm.com/) —— 补丁被放弃，改用 Kconfig 变更。
- [BPF 邮件列表：在 x86-64 与 arm64 上强制 JIT](https://lore.kernel.org/bpf/20261123205713.4094553-1-alexei.starovoitov@gmail.com/) —— 由 `BPF_SYSCALL` 选中 `BPF_JIT`，并对 KMSAN 留例外。
- [BPF 解释器与跳表](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/core.c) —— `___bpf_prog_run()`、`jumptable`、`ST_NOSPEC` 与 `BPF_JIT_ALWAYS_ON` 的编译开关。
- [arm64 JIT 屏障](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/arch/arm64/net/bpf_jit_comp.c) —— `sb` 与 `dsb nsh` 加 `isb` 两种形式，以及 Spectre v4 的 bypass 钩子。
- [x86-64 JIT 屏障](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/arch/x86/net/bpf_jit_comp.c) —— 同一指令位置上的 `EMIT_LFENCE()`。
- [arm64 屏障名称](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/arch/arm64/include/asm/barrier.h) —— 有 `spec_bar()`，没有 `barrier_nospec()`。
- [x86 屏障定义](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/arch/x86/include/asm/barrier.h) —— `barrier_nospec()` 是一个 `lfence` alternative。
- [通用空屏障](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/include/linux/nospec.h) —— 架构未定义时的空实现。
- [校验器插入 nospec](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/verifier.c) —— `error_recoverable_with_nospec()` 与推测状态分支。
- [校验器补丁阶段](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/fixups.c) —— 在标记指令前插入 `nospec`。
- [推测屏障判定](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/include/linux/bpf.h) —— `bpf_bypass_spec_v1()` 与 `CAP_PERFMON` 例外。
- [把 nospec 变成 v1 屏障的提交](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/patch/?id=dff883d9e93a) —— 语义拓宽，以及各架构的易感/缓解表。
- [校验器回退到 nospec 的提交](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/patch/?id=d6f1c85f2253) —— 插入屏障即放行的方案，以及哪些架构会真的发屏障。
- [强制 BPF_JIT_ALWAYS_ON 的提交](https://git.kernel.org/pub/scm/linux/kernel/git/bpf/bpf-next.git/patch/?id=8db2aca155a4) —— 在 bpf-next 中：sysctl 固定、`-ENOTSUPP` 回退变更、32 位 x86 例外。
- [变更后的 bpf-next Kconfig](https://git.kernel.org/pub/scm/linux/kernel/git/bpf/bpf-next.git/plain/kernel/bpf/Kconfig) —— `ARCH_WANT_BPF_JIT_ALWAYS_ON`、隐藏的提示符、KMSAN 例外。
- [sysctl 处理函数](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/net/core/sysctl_net_core.c) —— `bpf_jit_enable` 只在 `CONFIG_BPF_JIT` 下存在，`BPF_JIT_ALWAYS_ON` 时固定为 1。
- [内核 Spectre v2 文档](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/Documentation/admin-guide/hw-vuln/spectre.rst) —— 按缓解模式选择 retpoline，并在 IBRS 下运行时关闭。

## 今日社区讨论

本次覆盖情况：通过授权的只读归档读了两个选择加入归档的 Slack 工作区（共三个白名单频道），24 小时与 7 天窗口里都没有消息，所以 Slack 一侧是「安静」而非「读不到」。两个白名单 Discord 工作区（共十个频道）读不到：网页客户端停在登录页，因此它们的缺席是覆盖缺口，不是没有讨论。r/eBPF 作为普通公开页面读过，内容以项目公告为主，另有两篇真实的实践者帖子。当天真正的技术论证发生在公开的 bpf 邮件列表归档上，本篇的问题就取自那里；同一批列表流量也提供了下面这些主题。

**一条屏障，两种安全故事。** 当天开场的那个 arm64 补丁想让解释器也能执行屏障：把 arm64 的辅助函数改名，让通用的 `barrier_nospec()` 变成真屏障；同时避免让 `copy_from_user()` 为此付代价——告诉 uaccess 代码「arm64 本来就在软件层掩码用户指针」（`uaccess_mask_ptr()` 清掉第 55 位，所以一次被错误预测的 `access_ok()` 也到不了内核地址）。评审者否掉的不是补丁而是前提：在 bpf-next 上，只要 JIT 编进去了，arm64 就没有解释器，所以仍在跑解释型 BPF 的内核都是刻意用 `BPF_JIT=n` 构建的；而且就算屏障本身正确，它也落在 computed goto 之后，能训练该分支的程序会直接越过它。线程最终以补丁被放弃、改用让 Kconfig 选择消失收尾。留在这条线程里的未决边界是 Spectre v2：CSV2 只是在**上下文之间**隔离分支预测器，而 BPF 程序可以从内核内部训练它，整组补丁都没有处理这一点。读者今天能用的诊断就是验证一节那套——dump xlated、dump jited，把「not jited」当成屏障论证会变的那种情况。

**一个 Kconfig 变更，三个可见副作用。** 「强制 JIT」的补丁很小，但会改变读者会遇到的行为：`bpf_jit_enable` 变成固定的 1；JIT 编译不了的程序从「静默走解释器」变成以 `-ENOTSUPP` 拒绝加载，其中包含 classic socket filter 与 seccomp filter；而且只对 x86-64 和 arm64 取消选择，因为 32 位 x86 的 JIT 没实现全部指令。最容易被忽略的是 KMSAN 例外：sanitizer 内核保留解释器，因为 sanitizer 观察不到 JIT 程序的写，表现为对程序栈上一把 map key 的误报。如果你用 KMSAN 构建调试内核并期待「没有解释器」这一性质，它并不成立。

**一个没有 8 字节对齐的 map value，以及一次 arm64 panic。** 另一条报告显示，校验器对 map value 的对齐规则是关于**基指针**的断言，而不是对它的证明。`check_ptr_alignment()` 只把 `off` 与 value 起点比对（`check_generic_ptr_alignment()` 只看寄存器偏移加 `off`），代码假定基址至少 8 字节对齐——哈希 map 满足这一点，因为 value 放在向上取整到 8 的 key 之后。LPM trie 不取整：`data_size` 是 `key_size - 4`，value 紧跟 key 字节且没有填充，于是 IPv4 trie（key 8、data 4）返回的 value 指针只有 4 字节对齐。在偏移 0 做一次 64 位原子加能通过校验，然后在 arm64 上触发缺页处理不修复的对齐故障，从 `bpf_prog_test_run` 一路 panic。实用的检查很便宜：拿一个对 LPM trie value 做 64 位原子的程序，在对齐严格的那个架构上跑，而不是 x86-64——后者容忍同样的访问。修复必须接受的点是：value 布局是 map 内存记账的一部分，加填充会改变 `leaf_size` 和 `memlock` 计费，而用户看到的 key 格式不能变。

**一次以错误类型读取 union 成员导致的校验器崩溃。** 一个 struct_ops 程序调用返回 `PTR_TO_MEM | PTR_MAYBE_NULL` 的 helper，可以让校验器解引用一个空的 `struct btf *`。`check_return_code()` 的 struct_ops 分支读 `reg->btf` 和 `reg->btf_id` 时没有先检查寄存器是不是 BTF 指针；在寄存器状态 union 里这两个字段与 `mem_size` 重叠，于是校验器把一个大小（报告里是 64）当作 BTF id 传给了 `btf_type_by_id()`。报告给出了复现的指令列表和跳转偏移，它暴露的边界是通用的：struct_ops 返回路径是唯一按类型去读 union 成员的地方，任何新的寄存器类型只要与这些成员重叠，就需要这个检查。对读者有用的一个边界：这是校验器崩溃而不是程序被拒，所以一次「本该安静失败」的加载可以把内核干掉，而且发生在加载路径上。

**hook 原型就是 API，而它移动了。** 7.4 的一个 inode hook 新增了**第一个**参数，这让按旧原型编译的 BPF LSM 程序所有既有参数都平移了一位：它们照样加载成功，然后读到错误的参数，对安全程序来说是最糟的失败模式。线程的答案收敛在机制而非政策上：程序已经可以靠声明一个匹配的函数指针 union，让 BTF 检查出的签名不一致从而强制加载失败；一个更新的系列提议对 tracing 程序参数做 CO-RE 重定位，在加载时把「名为 dentry 的参数」解析成下标，这样只要新参数不追加在末尾就不会静默平移。未决的部分是抽象层归谁：维护者对稳定 API 层应放在 LSM 框架、BPF 一侧，还是哪儿都不放，意见并不一致。

**会在读者眼皮底下变化的 ring buffer 记录。** 一个 libbpf 系列在 ring buffer 之上加了逐条记录 API，针对 overlay 模式——内核可能在用户态读取时改写最旧的记录。对任何消费这类 map 的人值得记住的形态是：`ring__next()` 返回记录但不消费，`ring__release()` 消费这一批，`ring__invalidated()` 告诉读者自上次释放以来返回的记录是否可能被改写过——所以调用方先拷贝字节，只有在事后失效检查通过时才使用拷贝。配套的内核侧系列把记录数据排在生产者位置之后，这样改写不会发布一条写了一半的记录。仍然开放的是这个检查在繁忙 ring 上的开销：每批多读一次共享状态，系列并没有量化它。