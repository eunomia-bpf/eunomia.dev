# 为什么 x86 上的 BPF 程序在每一次不可信指针加载前都要付一次地址检查，而去掉它必须先改内核的缺页处理？

**简短回答：** 这次加载需要一条异常表项，因为校验器证明不了这个指针有效；而在 x86 上，异常表只有在故障落在内核地址空间时才会被查。开了 SMAP 之后，内核态访问**用户**地址且 EFLAGS.AC 为 0，会被当成缺少一次 `STAC`：`do_user_addr_fault()` 直接调用 `page_fault_oops()`，根本不去看异常表，那条本该把目的寄存器清零并跳过加载的修复因此永远跑不到。所以 JIT 只能自己在加载前把地址过一遍——用 `lea` 算出偏移后的地址、移位、分支——凡是不是内核地址的，就直接把目的寄存器清零。要把这段检查去掉，就得让缺页处理在「今天必定 oops」的那条分支上解析 BPF 程序的异常表项：`arch/x86/mm/fault.c` 里十一行，加一次 `is_bpf_text_address()` 查询。这个查询对每一个可能触发此类故障的程序是否都答得对，就是全部的安全性论证，也正是当前评审分歧所在。

## 机制

### 1. 校验器标记加载，JIT 给每一条配上异常表项

当校验器证明不了指针时，这次加载会被改成 `PROBE_MEM`。改写只在校验结束之后做一次，位置是 `bpf_convert_ctx_accesses()`：

```c
if (bpf_is_ptr_to_mem_or_btf_id(ptr_type) &&
    bpf_may_fault_on_deref(ptr_type) &&
    type == BPF_READ) {
        if (BPF_MODE(insn->code) == BPF_MEM)
                insn->code = BPF_LDX | BPF_PROBE_MEM | BPF_SIZE(insn->code);
        else
                insn->code = BPF_LDX | BPF_PROBE_MEMSX | BPF_SIZE(insn->code);
        env->prog->aux->num_exentries++;
}
```

而 `bpf_may_fault_on_deref()` 只有两种情形：普通的 `PTR_TO_BTF_ID`（沿着指针字段走一层、而该字段又没有标记为继承信任时得到的类型），以及任何带 `PTR_UNTRUSTED` 的类型。经这类指针的**写**更早就被拒掉了，所以被改写的只有加载。

x86 JIT 随后给每个这样的加载记一条异常表项：`data` 是 `EX_TYPE_BPF`，`fixup` 字里打包两个数——它实际发出的那条 x86 加载指令有多少字节，以及目的寄存器落在 `pt_regs` 的哪个槽位（对于读进寄存器、但 `dst_reg` 不指向它的 load-acquire 形式，用 `DONT_CLEAR`）。发生故障时，`fixup_exception()` 依次查内核表、模块表，最后是 `search_bpf_extables()`：按指令指针找到程序，再在程序自己的表里查；`ex_handler_bpf()` 把对应的槽位清零，并把记录下来的指令长度加到 `regs->ip` 上。这就是全部恢复动作：不重执行、不发信号，程序带着一个零继续跑。

### 2. 在用户地址上，开了 SMAP 就永远走不到这次恢复

有两个不同的故障入口会走到 `fixup_exception()`。内核地址上的故障走 `do_kern_addr_fault()`，最终进 `kernelmode_fixup_or_oops()`，那里的第一个动作就是 `fixup_exception()`。用户地址上的故障走 `do_user_addr_fault()`，而在开了 SMAP 的机器上它最先做的是这个判断：

```c
if (unlikely(cpu_feature_enabled(X86_FEATURE_SMAP) &&
             !(error_code & X86_PF_USER) &&
             !(regs->flags & X86_EFLAGS_AC))) {
        /*
         * No extable entry here.  This was a kernel access to an
         * invalid pointer.  get_kernel_nofault() will not get here.
         */
        page_fault_oops(regs, error_code, address);
        return;
}
```

注释把设计说得很清楚：普通内核代码想扛住坏地址，就用 nofault 系列访问器，而那些访问器在访问之前就在软件里把用户地址滤掉了——`copy_from_kernel_nofault_allowed()` 拒绝低于 `TASK_SIZE_MAX + PAGE_SIZE` 的地址、也拒绝 vsyscall 页里的地址，理由正是缺页处理会对它们 oops。JIT 出来的 BPF 程序没有这样的软件过滤，除非 JIT 自己发一段，而那段就是本文讨论的检查。

### 3. 为什么这个检查不能只跟 TASK_SIZE_MAX 比

vsyscall 页在 `0xffffffffff600000`，位于 `TASK_SIZE_MAX` 之上，但 `fault_in_kernel_space()` 把它特判回了用户空间：

```c
if (IS_ENABLED(CONFIG_X86_64) && is_vsyscall_vaddr(address))
        return false;
return address >= TASK_SIZE_MAX;
```

所以从那一页加载会走用户地址分支，在 SMAP 下就会 oops。这不是假设：提交 `b599d7d26d6a`（"bpf, x86: Fix PROBE_MEM runtime load check"）就是因为一个从 `sock` 上读 `sk_tsq_flags` 的程序能够到那一页、把内核打崩。于是这个检查两头都夹住：接受区间是 `[TASK_SIZE_MAX + PAGE_SIZE, VSYSCALL_ADDR)`，实现成对 `addr - VSYSCALL_ADDR` 与 `TASK_SIZE_MAX + PAGE_SIZE - VSYSCALL_ADDR` 的一次无符号比较。

现在的编码已经不需要把常量放进寄存器。十月合并的两个 JIT 提交把它重写过：`74309a9` 把偏移后的地址折进一条 `lea`（7 字节、一个 uop），`181576e` 改成跟「高于上限的下一个 2 的幂」比较，把比较变成一次移位加一个 `jne`：

```
lea  r11, [src + off - VSYSCALL_ADDR]   ; 偏移整个放进 disp32
shr  r11, fls64(limit)                  ; 4 级页表 48，5 级 57
jne  load                                ; 非零 => 让加载执行
xor  dst, dst                            ; 否则把目的寄存器清零
jmp  done
```

处在「上限」与那个 2 的幂之间的地址本来就是非规范地址，以前也由缺页处理清零，所以提前排除它们不改变任何可观测行为。

### 4. 这段检查花多少，提案又量了什么

在上面两个提交之前，这段守卫是一条 `movabs` 取 `VSYSCALL_ADDR`、一次搬移加一次 `add` 算偏移地址、一次 `sub`、一条 `movabs` 取上限、一次 `cmp`、一个分支、一次 `xor` 和一次 `jmp`——九条指令、39 字节（偏移为 0 时 32 字节），挡在一条四到七字节的加载前面。正在评审的这个系列贴在 `e1d84a37cba9` 的 `bpf-next` 上，2026-10-09 发出，用 veristat 遍历全部 BPF selftest 目标文件和 466 个生产目标文件，量了去掉它值多少：

- BPF selftests：3324 个程序，其中 91 个含 `PROBE_MEM` 加载。平均小 35.0%，
  中位 37.5%，最多 74.4%。
- 生产目标文件：1710 个程序，其中 228 个含 `PROBE_MEM` 加载。平均小 7.0%，
  中位 1.8%，最多 69.2%。

没有程序变大。相对节省最多的是 1039 字节降到 266 字节的那个；selftest 里的 socket 与 task 迭代器掉了大约一半代码。两个数都是在仍有九指令形式的树上量的代码尺寸，所以它们只是「现在这段更短的守卫花多少」的上界。

### 5. 提案本身

补丁 1 在 SMAP 分支里、oops 之前加一个判断：

```c
if (is_bpf_text_address(regs->ip) &&
    fixup_exception(regs, X86_TRAP_PF, error_code, address))
        return;
```

它落在「代码本来就要死」的位置，所以今天能处理好的故障一条新指令都不多执行；来自 BPF 以外的内核态用户访问照样 oops。补丁 2 让 JIT 只在没有 SMAP 时保留检查：

```c
probe_mem = BPF_MODE(insn->code) == BPF_PROBE_MEM ||
            BPF_MODE(insn->code) == BPF_PROBE_MEMSX;
bounds_check = probe_mem && !cpu_feature_enabled(X86_FEATURE_SMAP);
```

这个特性位在任何程序被 JIT 之前就定了：`setup_smap()` 在早期 CPU 初始化时按启动 CPU 的能力设 CR4，`clearcpuid=smap` 更早地从能力掩码里去掉这一位，所以 JIT 时刻的判断在整个内核生命周期内稳定。没有 SMAP 的内核保留检查，这正好保住了契约的另一半：SMAP 机器上从已映射用户页加载会变成故障、因而变成零；而没有 SMAP 的内核上同一次加载本来会成功读到用户内存，一旦把守卫去掉就真的读到了。

### 6. 为什么这个查询就是安全性论证

`is_bpf_text_address()` 是 `bpf_tree` 里的一次 latch-tree 查询，不是对地址布局的判断。这棵树由 `bpf_prog_kallsyms_add()` 填，而它除非「程序已 JIT、不是 classic socket filter、加载者持有 `CAP_BPF`（直接持有或通过授予它的 token）」否则直接返回：

```c
if (!bpf_prog_kallsyms_candidate(fp) ||
    !bpf_token_capable(fp->aux->token, CAP_BPF))
        return;
```

所以修复恰好只对在这棵树里的程序可达，而 `search_bpf_extables()` 还额外要求 `num_exentries` 非零。其余在用户地址上故障的代码仍然走老路。解释器两边都不受影响：`LDX_PROBE_MEM` 在解释器里走 `bpf_probe_read_kernel_common()` 和 nofault 访问器，软件过滤在那边自己做了。

## 在一台真实内核上验证

1. 看发出来的守卫，而不是 BPF 汇编。`bpftool prog dump jited`（或程序在 `/proc/kallsyms` 里的那一项）给的是原生代码；守卫就是每条 `PROBE_MEM` 加载前那条位移带 `0xA00000` 的 `lea`，加上移位与分支。应用这个系列之后，加载单独站着，只剩异常表项。

2. 跑那张地址表，而不只是 NULL。selftest 用一个由用户态决定取值的指针驱动同一个程序，检查六个地址：`0`、`4096`、最后一个用户页（`(1UL << 47) - 4096`）、非规范的 `1UL << 63`、vsyscall 页，以及该页内的一个偏移。每次加载都必须读出零、内核必须活着；同一个程序经当前 task 指针读时必须读出真实值——这才是「清零是有条件的」的证据。

3. 用这个测试区分两套机制。selftest 在「有守卫」和「有缺页修复」两种内核上都通过，而它自己的说明写着：只打补丁 2 时第一个子测试会 oops——那个 oops 就是 SMAP 分支，补丁 1 把它变成一次修复。

4. 确认清的是对的寄存器。`ex_handler_bpf()` 清的是 `fixup` 字里点名的 `pt_regs` 槽位，并按记录下来的加载长度推进 `regs->ip`。槽位索引写错不会触发故障：程序继续跑，读到的是旧值，所以要把程序上报的某个字段与已知的用户态值对一遍。

5. 确认检查真的以 SMAP 为条件。用 `clearcpuid=smap` 启动（这是个仅供测试、会给内核打上 taint 的选项），再 dump 一次 JIT，守卫就在一个同样跑了本系列的内核上回来了。

## 限制

修复的好坏完全取决于它查的那棵树。`is_bpf_text_address()` 只对「已 JIT、非 classic、以 `CAP_BPF` 加载」的程序给肯定答案；不在这个集合里、又在用户地址上故障的程序照样 oops，而缺页处理无法区分「一个不在 kallsyms 里的 BPF 程序」和「忘了 `STAC` 的普通内核代码」。这也正是 x86 维护者的反对意见指向查询而不是指向 SMAP 本身的原因：在这条路径上，SMAP 不再负责阻止读成功，它只负责让这次读发生故障，之后能否恢复就取决于一张符号表了。

量到的是代码尺寸，不是时间。带守卫的加载每次执行都要多几条指令；不带守卫的加载一旦真的故障，要付一次完整缺页、一次 `is_bpf_text_address()` 查询、一次异常表搜索，然后才恢复。系列给出的数字只数字节，所以一个不可信加载几乎总是落空的程序会变小、变慢，而评审正是要求把运行时这一侧讲清楚。

changelog 里「arm64、riscv、s390、loongarch 本来就这么发」的说法，对这四个里的三个并不成立。arm64、riscv、loongarch 都选了 Kconfig 符号 `ARCH_HAS_NON_OVERLAPPING_ADDRESS_SPACE`，使弱定义的 `bpf_arch_uaddress_limit()` 返回 `TASK_SIZE`；对它们来说，是**校验器**（不是 JIT）把每一条 `PROBE_MEM` 加载改写成七条 BPF 守卫指令——地址右移 32 位、与 `limit >> 32` 比较。这些架构根本不让用户地址走到加载那里。x86 把钩子覆盖成返回 0，因为它的 JIT 自己发检查；s390 是分离地址空间，靠故障路径。所以 cover letter 里那个对比是一个架构，不是四个。

这个改动也没有让用户内存变得可读，而这正是设计目的而不是副作用：SMAP 仍然拦住访问，加载仍然故障，故障仍然产出一个零。移动的是**谁**来判断「这不是内核地址」——加载时由 JIT 判断，或者运行时由缺页处理判断。x86 那边要接受的事是：处理里从此多一条 BPF 专用分支；系列提出可以把它关在 `CONFIG_BPF_SYSCALL` 后面，那只改变代码住在哪里，不改变依赖关系。截至目前，这个系列有一位 BPF 评审的 Ack，也有 x86 维护者的明确否决——他要求去「调理」程序本身，所以主线里的检查还在。

最后，这里的任何东西都不改变别的故障类别：vmalloc 上的故障、机密 guest 里未 accept 页上的故障、内核地址上的故障都保持原有处理。新增的分支在 SMAP 条件内部，没有 SMAP 的内核根本不会求值它。

## References

- [BPF 邮件列表：在 SMAP 下去掉 PROBE_MEM 地址范围检查（cover letter）](https://lore.kernel.org/bpf/20261009024925.3169077-1-memxor@gmail.com/) —— 机制、veristat 数据，以及那个「四个架构」的说法。
- [BPF 邮件列表：缺页处理的改动](https://lore.kernel.org/bpf/20261009024925.3169077-2-memxor@gmail.com/) —— `arch/x86/mm/fault.c` 里那十一行，以及 `is_bpf_text_address()` 这道门。
- [BPF 邮件列表：JIT 的改动](https://lore.kernel.org/bpf/20261009024925.3169077-3-memxor@gmail.com/) —— `bounds_check = probe_mem && !cpu_feature_enabled(X86_FEATURE_SMAP)`，以及改动前后的生成汇编。
- [BPF 邮件列表：selftest](https://lore.kernel.org/bpf/20261009024925.3169077-4-memxor@gmail.com/) —— 六个地址、有效地址的对照，以及「只打补丁 2 会 oops」的说明。
- [BPF 邮件列表：x86 维护者的反对](https://lore.kernel.org/bpf/20261009041346.GHashp-hYpLl-o-l3M@fat_crate.local/) —— 「在 #PF 处理里加 BPF 专用逻辑，一点也不好笑」。
- [BPF 邮件列表：结束这一轮的回复](https://lore.kernel.org/bpf/20261123003036.GCawOJLKK1iX47XTDm@fat_crate.local/) —— 「唯一的防线是 `is_bpf_text_address()` 真的做对」。
- [BPF 邮件列表：2026-11-22 的反对](https://lore.kernel.org/bpf/20261122200848.GBawNL0FcQoK918YDI@fat_crate.local/) —— 「我们基本上是在 BPF 程序周围关掉 SMAP」，以及「这点节省不值这个风险」的判断。
- [BPF 邮件列表：作者 2026-11-22 的回答](https://lore.kernel.org/bpf/CAP01T746W=8FFiJJUM3H-Qv09uRO6-XEi9B-8pjSgtpAPBx4EA@mail.gmail.com/) —— 「严格限定在 BPF 程序内」，以及把分支关进 `CONFIG_BPF_SYSCALL` 的提议。
- [x86 缺页处理](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/arch/x86/mm/fault.c) —— 以 `page_fault_oops()` 收尾的 SMAP 分支、`kernelmode_fixup_or_oops()`、以及 `fault_in_kernel_space()`。
- [x86 异常修复](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/arch/x86/mm/extable.c) —— `fixup_exception()` 与 `EX_TYPE_BPF` 分支。
- [x86 BPF JIT](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/arch/x86/net/bpf_jit_comp.c) —— 地址范围检查、`ex_handler_bpf()`、fixup 字的布局，以及返回 0 的 `bpf_arch_uaddress_limit()`。
- [BPF 核心](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/core.c) —— `search_bpf_extables()`、`bpf_prog_ksym_find()`，以及 `bpf_prog_kallsyms_add()` 里的 `CAP_BPF` 条件。
- [校验器改写](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/fixups.c) —— `BPF_LDX` 到 `BPF_PROBE_MEM` 的改写，以及 arm64、riscv、loongarch 上用的 `bpf_arch_uaddress_limit()` 守卫。
- [校验器指针信任判定](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/include/linux/bpf_verifier.h) —— `bpf_may_fault_on_deref()` 与 `bpf_is_ptr_to_mem_or_btf_id()`。
- [nofault 访问器的门](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/arch/x86/mm/maccess.c) —— `copy_from_kernel_nofault_allowed()` 在软件里拒绝用户地址与 vsyscall 页。
- [加上 vsyscall 边界的提交](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/commit/?id=b599d7d26d6ad1fc9975218574bc2ca6d0293cfd) —— 2024 年那次崩溃，以及为什么只比 `TASK_SIZE_MAX` 不够。
- [把地址计算折进 lea](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/commit/?id=74309a9294a74975fee75e4d6259eef026495325) —— 20 字节、四个 uop 变成 7 字节、一个。
- [用移位替代寄存器比较](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/commit/?id=181576ebd87d90aafeedbecf7a2dcc109dcff391) —— 15 字节变 6 字节，以及它吸收掉的那段非规范地址区间。

## 今日社区讨论

本次通过获批的只读归档查看了两个归档选择加入的 Slack 工作区，共三个白名单频道：24 小时窗口内没有消息，7 天窗口内只有一条。那条消息提议了一个内核内的二进制模式查找 kfunc；它是真实的开放问题，但这里已经有专门的一篇回答，所以今天这页建在公开的 bpf 邮件列表归档上，而当天真正的设计争论也发生在那里。r/eBPF 作为可见浏览器页面读了，看到的是项目发布而不是提问。两个白名单 Discord 工作区共十个频道本次没有查看：web 客户端停在登录页、没有可用会话，所以它们在这里的缺席是覆盖缺口，不是「没人说话」。

**九条指令对一次缺页。** 今天的线程是一次复活：沉寂六周之后，十月初的三补丁系列又被翻出来。作者的论证有两条：守卫挡在**每一条**那种形态的不可信指针加载前面，而它防的那次故障是处理本来就会解析的——处理只是拒绝在 SMAP 打开时去看异常表。x86 一侧读同一个补丁，读成对 SMAP 的颠覆：改完之后，「我们基本上是在 BPF 程序要访问可能过期的内存时把 SMAP 关掉，唯一的防线是 `is_bpf_text_address()` 真的做对」。维护者的结论是这点节省「完全不值这个风险」，并在随后一封回复里要求去「调理」程序，让故障看起来像任何一次普通用户态故障，而不是教 #PF 处理认识 BPF。作者的回应是：改动严格受限——读仍然不成功、目的寄存器仍然清零、分支还能关进 `CONFIG_BPF_SYSCALL`；被搬走的开销本来付在一条最终会 panic 的路径上。

**两位评审发现「其他架构」的说法不对。** changelog 说这个补丁让 x86 的行为像 arm64、riscv、s390、loongarch。对这个补丁的自动评审指出，四个里有三个从不为用户地址发出裸的 `PROBE_MEM` 加载：它们选了非重叠地址空间的 Kconfig，于是校验器在 JIT 之前就把每次加载改写成七条 BPF 守卫指令，只有 s390 靠故障路径。同一份评审还指出 veristat 只量了代码尺寸，而一次 NULL 加偏移的加载现在要付完整缺页，以前只要一次比较加一个分支。同一系列的另一个自动评审提出的，正是本页机制部分的核心问题：这个查询是 kallsyms 树查询，所以任何「已 JIT 但不在树里」的程序仍然会 oops。

**一段已经被改写过两次的守卫。** 对一个想弄清这段守卫今天花多少的读者，有意思的背景是：系列引用的那段序列在主线里已经不是这样了。系列贴出后一周合并的两个提交，把地址计算换成一条 `lea`、把范围比较换成一次移位加分支，把守卫从 39 字节压到 17 字节左右。没有人针对更短的形式重跑 veristat，所以被广泛引用的「平均降 35%」只是上界。同一周的 JIT 工作还动了 arena 原子加载的异常表元数据，那是相邻的情形：那些加载带同样的 `EX_TYPE_BPF` 项，但还需要把**源**寄存器也报出来，因为一次故障的 arena 访问必须指明它在算的地址。