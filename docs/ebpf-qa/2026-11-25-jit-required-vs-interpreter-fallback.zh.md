# 为什么一个已经通过校验的 BPF 程序会在解释器里崩溃，又是什么让内核拒绝这次回退？

**简短回答：** 校验和代码生成是两个不同的阶段，而校验器自己**插入**的若干指令只在 JIT 里有意义。程序先按它自己的指令流被校验；随后 `bpf_do_misc_fixups()` 会改写它——一次 helper 调用变成一条加上 per-CPU 基址偏移的内部 `mov`，一次 arena 访问变成一种只有 JIT 才解码的探测模式，一个会被 JIT 内联的 helper 保留原始 helper 编号作为 `imm`。解释器对这些一无所知。于是内核为每个程序保留一个 `jit_required` 位，由每一处这样的改写置起，并在 JIT 不可用时拒绝加载，而不是用解释器去跑。一旦回退被拒绝，失败形态就是一次加载错误；在此之前，它是内核上下文里的缺页或 `BUG_ON()`。

## 机制

### 1. 改写发生在选择运行时之前

`bpf_check()` 按固定顺序跑各趟：先是主校验循环，然后 `bpf_convert_ctx_accesses()`，然后 `bpf_do_misc_fixups()`，然后 32 位优化，然后 `bpf_fixup_call_args()`，最后才是 `__bpf_prog_select_runtime()` —— 由它决定用解释器还是编译程序。因此 fixup 趟插入的每一条指令，都构成「如果 JIT 没跑，解释器将要执行」的那条指令流：

```c
if (ret == 0)
        ret = bpf_convert_ctx_accesses(env);
if (ret == 0)
        ret = bpf_do_misc_fixups(env);
...
if (ret == 0)
        ret = bpf_fixup_call_args(env);
```

这些指令的生产者是校验器，不是 JIT。这正是解释器「必须能执行它们」的原因，而它做不到的原因，是这些属于内部编码，不在 UAPI 里。

### 2. per-CPU 的那条 `mov` 与解释器的 switch

在原生代码里解析一个 per-CPU 地址就是一条指令，所以 fixup 趟就生成一条。编码复用了 `BPF_ALU64 | BPF_MOV | BPF_X`，把区别放在带外偏移里：

```c
#define BPF_ADDR_PERCPU (-1)
#define BPF_MOV64_PERCPU_REG(DST, SRC)                         \
        ((struct bpf_insn) { .code = BPF_ALU64 | BPF_MOV | BPF_X, \
                             .dst_reg = DST, .src_reg = SRC,       \
                             .off = BPF_ADDR_PERCPU, .imm = 0 })
```

指令集文档只允许三条算术指令带非零 `off`（`SDIV`、`SMOD`、`MOVSX`），所以 `off = -1` 从构造上就在 ABI 之外。支持它的 JIT 在普通 `mov` 分支之前先认它，并发出基址加法——x86-64 上是对 `this_cpu_off` 的 `gs:` 加法，arm64 上是读 per-CPU 基址寄存器再 `add`。

解释器没有这个分支。它的双字搬移是对 `off` 的 switch，只处理 0、8、16、32，**没有 default 分支**：

```c
ALU64_MOV_X:
        switch (OFF) {
        case 0:  DST = SRC;       break;
        case 8:  DST = (s8) SRC;  break;
        case 16: DST = (s16) SRC; break;
        case 32: DST = (s32) SRC; break;
        }
        CONT;
```

生成的序列都是把一个寄存器搬给它自己（`BPF_MOV64_PERCPU_REG(r0, r0)`），所以在 `off = -1` 时该寄存器内容原封不动：它仍是前面 `ld_imm64` 取来的 per-CPU **偏移**，而不是地址。紧随其后的加载就通过它去读。这就是被报告出来的失败形态——一个刚通过校验的程序，在 per-CPU 数据区的一段地址上触发 supervisor read 缺页。

### 3. JIT 内联、而解释器不内联的 helper

某些 JIT 直接发出 `bpf_get_smp_processor_id()` 与两个 current-task helper 的函数体。当架构报告这一点时，fixup 趟就跳过对该调用 `imm` 的改写——`imm` 本应是相对 `__bpf_call_base` 的偏移——把原始 helper 枚举值留在那里：

```c
/* Skip inlining the helper call if the JIT does it. */
if (bpf_jit_inlines_helper_call(insn->imm)) {
        prog->jit_required = 1;
        goto next_insn;
}
```

解释器的调用分支是 `(__bpf_call_base + insn->imm)(...)`。用一个个位数大小的原始 id 顶替真正的偏移，就是往内核正文中间去调用——要么是指令对齐故障，要么跳进恰好落在那里的一段字节。这是通向同一场崩溃的**第二条独立路径**：完全不涉及 per-CPU 指令，只涉及一个 JIT 平时会内联的 helper。

### 4. 解释器根本不解码的 arena 编码

arena 访问被改写为 `BPF_PROBE_MEM32` 的加载与存储，以及 `BPF_PROBE_ATOMIC`；地址空间转换则保留自己的 `off` 标记。解释器的分发表是一张 256 项的 computed goto，带一个 default 分支，而探测模式不在其中：

```c
default_label:
        pr_warn("BPF interpreter: unknown opcode %02x (imm: 0x%x)\n",
                insn->code, insn->imm);
        BUG_ON(1);
```

于是一条被解释执行的 arena 存储抵达 `default_label`，并在一条校验器自己生成的指令上把内核带走。这一类与前两类在性质上不同：fixup 趟早已把程序变成不可解释的，唯一缺的只是那句拒绝。

### 5. 这个位在哪里被置起

`struct bpf_prog` 有 `jit_required:1` 标志，分配时按配置初始化，因此在 `CONFIG_BPF_JIT_ALWAYS_ON` 内核上每个程序都带着它，任何 JIT 失败都是加载失败：

```c
fp->jit_requested = ebpf_jit_enabled();
fp->jit_required = IS_ENABLED(CONFIG_BPF_JIT_ALWAYS_ON);
```

凡是只有 JIT 才实现的构造，校验器都在引入它的那一趟里置位：

- kfunc 调用 —— 上一层 `check_kfunc_call()` 会调用 `bpf_add_kfunc_call()`；
- 返回值跨多个寄存器的子程序 —— `bpf_compute_subprog_ret_regs()` 里的返回寄存器对检查；
- 间接调用（callx）—— `require_callx_jit()`；
- 栈展开 —— `check_unwind_insns()` 里的检查；
- arena map —— `check_map_prog_compatibility()`；
- instruction-array map —— `claim_used_maps()`；
- 被 `map_gen_lookup` 内联的 per-CPU map 查找 —— `bpf_do_misc_fixups()`；
- 被改写为特殊 `mov` 的 per-CPU `ld_imm64`，以及内联的 `get_smp_processor_id`、`get_current_task` 序列 —— `bpf_do_misc_fixups()`；
- 被 JIT 内联的 helper —— `bpf_do_misc_fixups()`；
- 带超时的 `may_goto` —— `bpf_do_misc_fixups()`；
- 返回 64 位的 modify-return 目标 —— `check_attach_btf_id()`。

### 6. 这个位在哪里被执行——以及 errno 为何不同

执行点有两个，哪个先命中决定了你看到的错误。

第一个在校验器内部，位于 `bpf_fixup_call_args()` 末尾，由 `#ifndef CONFIG_BPF_JIT_ALWAYS_ON` 包住，且只在 JIT 拆分没有成功时才会走到：

```c
if (prog->jit_required) {
        verbose(env, "program requires BPF JIT compiler but it is not available\n");
        return -EINVAL;
}
```

第二个在 `__bpf_prog_select_runtime()`，走的是「请求了 JIT，但拿回来的程序没有 jited」这条路径：

```c
bool jit_needed = fp->jit_required;
...
if (!bpf_prog_select_interpreter(fp))
        jit_needed = true;
...
if (!fp->jited && jit_needed) {
        *err = -ENOTSUPP;
        return fp;
}
```

因此 `net.core.bpf_jit_enable = 0` 的内核会在校验阶段以 `EINVAL` 加上述消息拒绝；而尝试过 JIT 却没成功的（例如常量盲化失败）内核，会在选择运行时处以 `ENOTSUPP` 拒绝。这两个数字不同，后者还很容易看错：`ENOTSUPP` 是 524，一个内核内部码，与 95 的 `EOPNOTSUPP` 不是一回事，所以 `strerror()` 会把它报成未知错误而不是「operation not supported」。两者都是加载错误，都不会留下一个稍后才出错的程序。

注意这套机制的形状：解释器是在尝试 JIT **之前**就被装进 `fp->bpf_func` 的，所以一次失败的编译正常来说只是把它留在原处，程序照跑——静默地跑在一条解释器执行不了的指令流上。`jit_required` 存在的意义，正是把这个静默回退变成一次拒绝。

### 7. 一个正常程序如何仍然落到解释器

被报告的路径并不是「JIT 被关掉」，而是常量盲化。`net.core.bpf_jit_harden = 2` 时，每条带常量的指令都被改写成随机立即数序列，指令数成倍增长，而超过 `S16_MAX` 的条件跳转撑不住：

```c
if (off < S16_MIN)
        return -ERANGE;
```

这个错误会中止盲化，而盲化失败是有明文记录的回落路径：恢复原始程序——也就是 fixup 趟已经改写过的那个——交给解释器。于是在没有 `BPF_JIT_ALWAYS_ON` 的内核上，加载成功，崩溃发生在执行时，落在挂载该程序的上下文里；在有该选项的内核上，同一失败就是加载错误。

## 在一个真实内核上验证

1. 把同一个程序的两种形态都 dump 出来。`bpftool prog dump xlated` 会把特殊指令按原样打印——基址加法显示为 `r0 = &(void __percpu *)(r0)`，其它内部操作码显示为类似 `(c0) nospec` 的标记——而 `bpftool prog dump jited` 给出原生序列。若 jited dump 报 `not jited`，说明程序正在被解释执行，xlated dump 里每一条内部指令都正由一个可能没有对应分支的解码器执行。

2. 读决定路径的两个 sysctl。`bpf_jit_enable` 只在 `CONFIG_BPF_JIT` 编进内核时存在；在 `BPF_JIT_ALWAYS_ON` 下它的最小值与最大值都是 1，写入会失败，内核只打一行日志说明该值被钉死。`bpf_jit_harden = 2` 对所有用户开启盲化，包括你正在用来测试的特权加载器。

3. 用 errno 与消息区分两条错误路径。`EINVAL` + `program requires BPF JIT compiler but it is not available` 来自校验器，意味着本次加载从未使用 JIT。`ENOTSUPP`——524，不是 95——来自 `__bpf_prog_select_runtime()`，意味着试过 JIT 但什么都没产出。

4. 主动复现这次回退：设 `bpf_jit_harden = 2`，构造一个含 per-CPU map 查找、并带一条跨度约一万一千条指令以上的条件跳转的程序，加载到没有 `BPF_JIT_ALWAYS_ON` 的内核上。盲化改写会溢出 16 位跳转，程序于是回退。在修复之前的内核上，执行会缺页；在当前内核上，加载会以第 3 步的消息失败。

## 边界

`jit_required` 是一道加载期护栏，不是对解释器的修正。它不能让解释器理解某种内部编码，只保证含这种编码的程序不会被交给解释器。在未置位的地方，旧行为照旧。

这留下两个仍然现实的情形。

**长期维护分支落后。** 这四个提交都在 7.2.y 里，但 6.12.y 与 6.18.y 都没有它们——而这两条正是当前发行版实际发布的长期分支，被报告的崩溃也正来自这些内核。回移之所以别扭，是因为这项检查如今落在 `kernel/bpf/fixups.c` 里，而该文件自 7.1 才存在，老树需要把代码搬过去而不是直接 cherry-pick。所以一个稳定内核仍可能在当前内核返回 `EINVAL` 的地方崩溃。

**modify-return 程序的 64 位返回值。** 解释器最外层帧的返回类型是 `unsigned int`，`R0` 的高 32 位在回给调用者时被丢掉。对于返回 64 位值的目标，一个返回 `ERR_PTR(-EINVAL)` 的程序变成 `0xffffffea`，`IS_ERR()` 认不出来。目前只有被构建期标记为「会返回错误」的函数要求 JIT；一份在评审中的系列把它扩展到所有 8 字节返回值的 modify-return 目标，并限制目标返回指针时程序可返回的取值。在它落地之前，一个未被打标的 64 位目标仍可被解释执行的程序修改，而那个错误值已被静默改坏。

第三个较轻的边界：per-CPU 那条指令只存在于其 JIT 报告支持它的架构上。别处 fixup 趟不动 helper 调用，因此 per-CPU 查找在被解释执行时依然可用——代价是失去内联，而不是正确性。

## 参考资料

- [BPF 邮件列表：被解释器回退后崩溃的报告](https://lore.kernel.org/bpf/MR1P264MB331494502C9D13D8A9EAFB99B7E82@MR1P264MB3314.FRAP264.PROD.OUTLOOK.COM/) — 一个通过校验的程序、一次 map 查找，以及 JIT 加固下的长跳转。
- [BPF 邮件列表：回移请求与回复](https://lore.kernel.org/bpf/809f91e2-638d-4e54-9b60-e66098d173e0@linux.dev/) — 哪些提交修好了它、哪些长期分支仍然缺、以及回移为何不能干净应用。
- [引入该标志的提交](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/patch/?id=9a6df65d5c6a) — `jit_required`，以及对「只查 kfunc」的替换。
- [拒绝内联 helper 却无 JIT 的提交](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/patch/?id=f1c27922576e) — 被跳过的 `imm` 改写与未修正的调用 id。
- [无 JIT 时拒绝 per-CPU 指令的提交](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/patch/?id=7a0855e73757) — 含故障报告，以及置位的两处代码。
- [无 JIT 时拒绝 arena 指令的提交](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/patch/?id=34746b5a84ec) — `unknown opcode` 那次崩溃与 map 兼容性检查钩子。
- [引入 per-CPU 指令的提交](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/patch/?id=7bdbf7446305) — 内部编码与它在反汇编器里的形态。
- [BPF 解释器与运行时选择](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/core.c) — `ALU64_MOV_X` 那个 switch、分发表、`bpf_prog_select_interpreter()`、`__bpf_prog_select_runtime()` 与常量盲化。
- [校验器的改写趟](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/fixups.c) — 每一处置位的内联点，以及校验期的拒绝。
- [校验器](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/verifier.c) — `bpf_check()` 末尾的趟次顺序，以及要求 JIT 的各项检查。
- [程序标志的定义](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/include/linux/bpf.h) — `struct bpf_prog` 的位域。
- [per-CPU 指令与 JIT 能力钩子](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/include/linux/filter.h) — 编码、`bpf_jit_supports_percpu_insn()` 与 `ebpf_jit_enabled()`。
- [反汇编器](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/disasm.c) — xlated dump 如何呈现这些内部指令。
- [per-CPU array 查找的内联](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/arraymap.c) — 生成的指令序列与它的能力门。
- [x86-64 JIT](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/arch/x86/net/bpf_jit_comp.c) 与 [arm64 JIT](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/arch/arm64/net/bpf_jit_comp.c) — per-CPU 指令与被内联 helper 的原生生成。
- [eBPF 指令集文档](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/Documentation/bpf/standardization/instruction-set.rst) — `off` 字段，以及被允许使用它的那三条指令。
- [BPF sysctl 文档](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/Documentation/admin-guide/sysctl/net.rst) — `bpf_jit_enable` 与 `bpf_jit_harden` 的取值。
- [sysctl 处理函数](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/net/core/sysctl_net_core.c) — `BPF_JIT_ALWAYS_ON` 下被钉死的值与那行一次性日志。
- [Kconfig](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/Kconfig) — `BPF_JIT_ALWAYS_ON` 及其移除的东西。
- [BPF 邮件列表：modify-return 程序与 64 位返回值](https://lore.kernel.org/bpf/20261125102810.699464-1-a.s.protopopov@gmail.com/) — 系列封面信、范围，以及解释器被截断的返回。
- [BPF 邮件列表：64 位 modify-return 目标的 JIT 要求](https://lore.kernel.org/bpf/20261125102810.699464-2-a.s.protopopov@gmail.com/) — 为什么只查被标记的函数还不够。

## 今日社区讨论

覆盖情况：两个选择加入归档的 Slack 工作区（三个在允许清单里的频道）通过只读归档读过，24 小时与 7 天窗口都没有消息，所以那一侧是安静而非未读。两个允许清单里的 Discord 工作区（十个频道）读不到——网页客户端停在登录界面——因此它们的活动是覆盖缺口，不是沉默。r/eBPF 的公开页面按普通网页读过，只有项目公告，没有决定性的技术问题。上面的问题取自公开的 bpf 邮件列表归档，本节其余主题也来自那里。

**一个从未被设计为正确的回退。** 当天的技术锚点，是上面答案里那四个提交的稳定分支回移请求。对读者有用的是线程里关于「拒绝应该放在哪」的分歧：报告者的复现需要三个条件同时成立——没有 `BPF_JIT_ALWAYS_ON`、`bpf_jit_harden = 2`、以及一条在常量被盲化后涨出 16 位范围之外的跳转。去掉任意一个，问题就消失，这也是为什么它需要一个被削减过的复现才能定位，以及为什么报告者最初对 trace 的解读（「解释器错误地执行了 map 查找」）比「JIT 坏了」更接近真相。维护者的回复补上了 trace 藏起来的那一环：崩溃的前提是 JIT 曾经**可用**，因为那条特殊指令只在 fixup 趟之后才存在，而 fixup 趟只在架构报告支持时才发出它。仍然开放的边界：任何未来只存在于 JIT 里的改写都能造出同样形状的问题，因此这个标志是每个新改写都必须置位的地方。

**依赖「哪条路径跑过」的校验器记账。** 一份五补丁系列修正了四处「每指令的辅助状态活得比产生它的校验路径更久」的问题。最有意思的一处与推测执行回退的交互有关：一次在推测路径上失败的检查会标记该指令并结束该路径，但同一条指令稍后会在非推测路径上、带着另一种类型的指针被再次校验——而失败那次写下的 `nospec_result` 与 `ctx_field_size` 仍然生效。修法是围绕每指令检查把两者存起来再恢复。两个配套补丁补齐了对应的发出缺口：对栈的 `BPF_ATOMIC` 存储之后现在也会发出屏障，而不只在普通存储之后；推测路径不再在调用处结束，因为调用之后并不发出屏障。第四个补丁让只读 map value 的读取使用非对齐访问器，因为那次读取跑在对齐检查之前。这一类 bug 对读者的实用诊断是：让同一个程序的同一条指令带着不同来源的指针各校验一次，然后对比——辅助状态是按指令索引的，不是按状态索引的。

**一个不能算错误的错误指针。** 当天 modify-return 讨论的另一半是值得记住的机制。`IS_ERR()` 看的是指针的全部 64 位，而解释器只返回 32 位。因此一个挂到返回 64 位函数上、并返回错误指针的程序只在 jited 时正确；被解释执行时，那个值以 `0xffffffea` 之类的垃圾到达调用者，且通过所有 `IS_ERR()` 判断。线程里还给出两条边界：校验器本来就允许这类程序挂到返回指针的函数上，于是它可以返回一个调用者会去解引用的任意指针；而构建期对「会返回错误」的函数打标只覆盖整数返回，所以返回 ERR_PTR 的目标未被标记，目前也不要求 JIT。仍未解决的是调用点覆盖：即便在被标记的函数上，也有调用者从不检查返回值，还有少数函数在返回前把错误指针改成了 NULL，因此打标是对原型的陈述，不是对调用点的陈述。

**只在初始命名空间里有效的权限。** 一份七补丁系列针对一个具体的可用性缺口：在用户命名空间里、带着委派出去的 BPF token 运行的追踪应用，能挂到 vmlinux 函数上，却挂不到模块里的函数上。两个独立原因，在排查同样症状时都值得知道。内核的 `BPF_BTF_GET_NEXT_ID` 要求全局 `CAP_SYS_ADMIN`，而 token 提供不了；同时 libbpf 在程序加载时才取 token，这已经晚于它枚举内核 BTF 的时刻，所以即使内核愿意接受 token，请求时也根本没带。修法是把权限检查放宽到 `CAP_BPF` 加 token，并把 token 获取提前到对象创建时。诊断路径：如果失败的是模块 BTF id 查找，而加载器在容器里，先怀疑枚举那一步，而不是挂载权限——后者本来就已经认识 token。

**为最坏上下文选定的分配变体。** 本地存储（task、cgroup、inode）此前一律用 `kmalloc_nolock()` 分配新对象，它唯一的优点是既不会睡眠，也不会在 zone 锁上自旋。代价在内存压力下显现：无锁路径失败的地方，普通分配本可以成功，比如某个 zone 被抽到水位线以下而空闲页在另一个 zone 里。一份四补丁系列改为在调用时挑变体——可睡眠程序用睡眠分配，上下文允许拿 zone 锁时用 nowait 分配，其余走无锁路径——并把程序的辅助元数据一路传下去以做出这个判断。它留下的边界是「上下文允许拿 zone 锁」在运行时的确切含义：系列读取抢占与 NMI 状态以及程序的可睡眠性，而一个已经持有该锁的调用者仍只能自己判断。

**空 mm 检查该放在哪里。** 一份十补丁 RFC 把用户访问检查改成：在没有当前地址空间时拒绝并告警，而不是在某段过期的地址上限上静默成功。线程里的设计问题是放置位置：把检查放进 `access_ok()` 里让人无法遗漏，但需要各架构自己的实现都加上，并让 `__access_ok()` 退回纯范围检查。评审结论是一种拆分——带检查的包装函数负责检查，`__` 前缀的访问器在调用者改造完成前保留显式检查，而使用掩码指针的调用者自己检查地址空间，因为它们本来就已经清掉了标记位。仍未覆盖的部分在线程里被点名而非隐藏：x86 的用户访问在汇编里自己做范围检查，另有一些架构专属路径直接调用未检查的形式。

**一个安全程序读错东西的两种方式。** 同一天的两份独立报告都展示了「加载成功、运行成功、行为错误」的 LSM 程序。第一份里，某个在信息类命令下传 NULL 对象的钩子把参数标注为可空，而该参数实际是**可信**的；可信指针恰恰是校验器**不**改写成带故障保护的加载的那一类，于是一次无条件解引用在内核上下文里缺页，而不是读到 0。评审的更正对稳定分支很关键：行为是在「可信参数不再做探测转换」那次改动里变的，所以它是一次回归而非长期约定；而对不可空的那一种，加载错误是内存访问检查给出的 `EACCES`，不是挂载路径的 `EPERM`。第二份是一个在子 socket 尚未初始化完成时就运行的 socket 钩子，写在那里的选项会触到共享的或尚未分配的状态；线程里未决的问题是：应当把钩子移到 clone 路径更晚的位置，还是把它从「socket 可能已被修改」的钩子列表里移除——答案是在「选项可以早早设置」和「选项触及的一切都已存在」之间的取舍。

**会在读者脚下被改写的 ring buffer 记录。** 一份 libbpf 系列为可覆盖 ring buffer 新增了逐条记录 API，在那里内核可能在用户态读取时改写最旧的记录。值得记住的形状是：`ring__next()` 返回记录但不消费，`ring__release()` 消费这一批，`ring__rewind()` 回到第一条未释放的记录，`ring__invalidated()` 告诉读者已返回的记录里是否可能有被覆盖的——因此读者先拷贝，并且只有在那次检查事后通过时才相信这份拷贝。配套的内核侧工作把记录数据排在生产者位置之后，使一次覆盖不会发布出半写好的记录。仍未量化的是这个检查在繁忙 ring 上的开销：每批多读一次共享状态，而系列没有测量它。