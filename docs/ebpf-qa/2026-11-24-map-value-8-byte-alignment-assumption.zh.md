# 为什么校验器会接受在它从未证明过 8 字节对齐的 map value 上做 64 位原子操作或 load-acquire？

**简短回答：** 因为这条对齐规则说的是 value **内部的偏移**，而不是 value 起始的那个地址。`check_ptr_alignment()` 把 `PTR_TO_MAP_VALUE` 交给 `check_generic_ptr_alignment()`，后者把寄存器已知的内部偏移加上指令偏移，再和访问宽度比。map value 指针本身对这个和没有任何贡献：校验器手里没有关于它低位的任何事实，于是假定那几位是零。对预分配的哈希 map，这个假定成立。有三种生产者让它不成立——64 位上的 `lpm_trie`、32 位上的 `bpf_mem_alloc` 头部、以及两种位宽上的 `devmap`——在 arm64 上，由此产生的 64 位 `LDAR`/`STLR` 或原子操作会触发一个没有人修复的对齐故障，于是一个加载成功的程序在 BPF 程序内部直接 oops。2026 年 11 月 24 日发出的四补丁系列选择去补齐这三种布局，而不是改这条规则，因为规则本身正是 libbpf、BTF 和已有程序都依赖的东西。

## 机制

### 1. 校验器只查偏移，从不查基址

`check_ptr_alignment()` 按寄存器类型分支。`PTR_TO_MAP_VALUE` 除 `INSN_ARRAY` 外没有特例，因此落到 `check_generic_ptr_alignment()`：

```c
reg_off = tnum_add(reg->var_off, tnum_const(off));
if (!tnum_is_aligned(reg_off, size))
        return -EACCES;
```

`reg->var_off` 是校验器对「距 value 起点的偏移」的全部认知。对一次 lookup 直接返回的指针，它是零。而基址的低位根本不在寄存器状态里，所以任何检查都看不到它们。源码直接给出两个后果：

- 偏移检查只在 `strict` 为真时执行。`strict` 等于 `env->strict_alignment || strict_alignment_once`，而 `env->strict_alignment` 来自 `BPF_F_STRICT_ALIGNMENT` 程序标志；内核未配置 `CONFIG_HAVE_EFFICIENT_UNALIGNED_ACCESS` 时它被强制为真，`BPF_F_ANY_ALIGNMENT` 会把它强制为假。arm64 与 x86-64 都选中了该项配置，所以在这些架构上，对 map value 的普通 8 字节读写根本不查。
- 原子操作是仍然会被查的例外：`check_atomic_rmw()` 与 `check_atomic_load()` 传 `strict_alignment_once=true`，因此偏移为奇数处的 8 字节原子操作或 load-acquire 会被拒绝。但检查的仍然只是偏移，偏移为 0 时永远通过。

校验器真正强制的宽度规则只有一条，且是架构性的：当 `BITS_PER_LONG != 64` 时，`check_atomic()` 用 `-EOPNOTSUPP` 拒绝 64 位 load-acquire 与 store-release。64 位 `BPF_ATOMIC` 算术在 32 位上是被允许的。

### 2. 三种把 value 放在 4 mod 8 处的生产者

下面的偏移是结构体布局，不是实测值；每一个都按当前定义重新算过。

- **LPM trie，64 位。** `struct lpm_trie_node` 依次是两个 child 指针、`u32 prefixlen`、`u32 flags`、`u8 data[]`，所以 `data` 在偏移 24。value 是 `found->data + trie->data_size`，而 `data_size` 等于 `key_size - 4`。IPv4 trie 的 key 是 8 字节，于是 `data_size` 为 4，value 落在节点的偏移 28，即 4 mod 8。IPv6（key 20、data 16）落在 40，是对齐的。没有任何地方把 `data_size` 向上取整。
- **`bpf_mem_alloc` 对象，32 位。** `bpf_mem_alloc()` 返回 `unit_alloc(...) + LLIST_NODE_SZ`，而 `LLIST_NODE_SZ` 就是 `sizeof(struct llist_node)`。这个头部在 64 位上是 8 字节，在 32 位上是 4 字节，于是 32 位内核上分配器交出的每个对象都比 `kmalloc` 返回地址偏 4 字节。非预分配的 hash 与 rhash 元素就从这个分配器取 value，这正是评审表里哈希 value 只在 32 位错位的原因。`bpf_obj_new()` 造出的对象同理。32 位上的 IPv4 LPM trie 反而恰好对齐（基址 4、`data` 在 20、value 在 24）——这是今天唯一靠「碰巧」成立的情形。
- **devmap，两种位宽。** `dev_map_lookup_elem()` 与 `dev_map_hash_lookup_elem()` 返回 `&obj->val`，而 `struct bpf_dtab_netdev` 里 `val` 紧跟在 `unsigned int idx` 之后：64 位偏移 52，32 位偏移 28，都是 4 mod 8。devmap 对 BPF 程序只读，所以没有原子操作能落到它上面，但 64 位上的 64 位 load-acquire 位于偏移 0 时是能通过的。

### 3. arm64 会缺页，而故障不被修复

arm64 JIT 把这几条指令编成恰好要求自然对齐的形式：load-acquire 与 store-release 用 `LDAR`/`STLR`，`BPF_ATOMIC` 用 LSE 原子指令或 `LDXR`/`STXR` 循环。`FEAT_LSE2` 放宽的是「不越出 16 字节块」的 LSE 原子；对 `LDAR`/`STLR`，只有当 `SCTLR_EL1.nAA` 置位时才放宽——Linux 并不置位它，所以在 `INIT_SCTLR_EL1_MMU_ON` 启动的内核上，即使硬件有 LSE2，这两种装载指令仍保留对齐要求。

接着是故障路径不肯帮忙。`do_alignment_fault()` 只在「内核带 `CONFIG_COMPAT_ALIGNMENT_FIXUPS` 且故障来自 32 位 compat 用户态任务」时才调 `do_compat_alignment_fixup()`；其余全部走 `do_bad_area()`，而在内核态它最终落到 `die_kernel_fault()`，报 "alignment fault"。BPF 程序运行在内核态，所以结果是一次 oops，而不是一个信号。系列里带了一份：

```
ESR = 0x0000000096000021
FSC = 0x21: alignment fault
Internal error: Oops: 0x96000021 [#1] SMP
pc : bpf_prog_cc9155b98965fc5e+0x64/0x90
```

这条 trace 是 IPv4 LPM trie value 偏移 8 处的 64 位原子加：地址是 4 mod 8，且访问跨过了 16 字节块。它是在有硬件 LSE2 的主机上、用 QEMU 跑 arm64 复现的——这正是要点：硬件支持救不了 `LDAR`/`STLR` 那一路。

32 位 arm 的故障来源不同，结局一样：JIT 完全没有实现 `BPF_ATOMIC`，于是解释器用 `atomic64_*()` 执行，而 ARMv6K 及以后用 `LDREXD`/`STREXD` 实现它们，这两条要求地址 8 字节对齐。

### 4. x86-64 把它藏了起来，除非访问被拆分

x86-64 对普通访问没有对齐陷阱；4 mod 8 的加锁 8 字节访问只有在跨缓存行时才特殊：那是一次 split lock，在支持 split lock detection 的 CPU 上触发 `#AC`。`exc_alignment_check()` 在内核态并不参考缓解状态——只要故障不来自用户态，它直接 `die("Split lock detected")`。`split_lock_detect=` 的取值（当 CPU 被确认支持检测时默认 `warn`）只影响用户态故障的处理：`warn` 会记一条日志并暂时关掉该 CPU 的检测，`fatal` 发 `SIGBUS`。

所以在一台开着检测的机器上，恰好跨缓存行的这些访问会致命，不跨的只是更慢。一个在 x86-64 上跑了多年的测试，对 arm64 什么也没说明。

### 5. 修复改了什么，以及为什么两个补丁必须同行

系列调整的是三种布局，而不是校验器规则。

- 分配器头部在 `memalloc.c` 里变成 `ALIGN(sizeof(struct llist_node), 8)`，哈希 map 的内存用量统计里用同一个常量，percpu 指针改存到 `obj + LLIST_NODE_SZ` 而不是 `obj[1]`。这是 Felix Fietkau 的补丁，被 rebase 过来。
- `lpm_trie` 在 lookup、节点填充、leaf 大小和内存用量四处把 `data_size` 向上取整到 8 的倍数，并加上 `BUILD_BUG_ON(offsetof(struct lpm_trie_node, data) % 8)`，因为 value 的对齐现在依赖那个偏移。
- `devmap` 把 `val` 标为 `__aligned(8)`。64 位上它移到偏移 56、结构体仍是 64 字节；32 位上结构体从 36 涨到 40 字节，而那里 64 位 load-acquire 本来就被校验器拒绝。

顺序约束是真的，cover letter 里明说了。只打分配器补丁，会让 32 位的 IPv4 LPM value 从 0 变成 4 mod 8，破坏今天唯一能用的情形。只打 trie 补丁、而 32 位节点仍在 4 mod 8，则把 `data_size` 取整到 8，value 落在基址为 4 mod 8 的节点的偏移 28。任一个单独打，在 32 位上都是回归；一起打才是 0 mod 8。

评审里有一处更正值得记住：给 `u8 data[]` 加 `__aligned(8)` 什么也改变不了，因为 `data` 本来就在 24 或 16。错位的是节点，所以 trie 补丁必须改偏移算术，而不是改声明。

## 在自己的内核上核对

1. 从 map 的实现里算偏移，不要从程序里看。LPM trie 是 `offsetof(data) + data_size`；32 位内核上任何来自 `bpf_mem_alloc` 的对象，都要再加 4 字节头部。
2. 能测就别只推。一个把 `bpf_map_lookup_elem()` 返回的指针存进全局变量、再在用户态断言 `addr & 7` 的程序，正是系列里 selftest 对 IPv4 trie、`BPF_F_NO_PREALLOC` 哈希和 devmap 做的事。因为程序本身不执行任何对对齐敏感的指令，受影响的内核是断言失败而不是 oops。不打补丁时，trie 与 devmap 的检查在 64 位失败，哈希与 devmap 的检查在 32 位失败。
3. 看生成的代码。`bpftool prog dump jited` 能看出 8 字节 load-acquire 是否变成 `ldar`、原子操作是否变成 `ldadd` 或 `ldxr`/`stxr` 对（arm64），以及是否变成带 `lock` 前缀的指令（x86-64）。
4. 在架构较真的地方跑。在有 LSE2 的主机上用 QEMU 就足以复现 oops；x86-64 不会告诉你任何东西。
5. 在 x86-64 上，先确认检测是否开着再下结论：查 `/proc/cmdline` 里的 `split_lock_detect`，并在程序跑完后到日志里找 `#AC` 或 "Split lock detected"。

## 限制

- 这是 v2，2026 年 11 月 24 日时尚未合入。你今天能启动的每个内核都还是上面那些布局，而 trie 那一路上有 7.3.5 的 oops 报告。
- 它修的是三个生产者，不是规则。`check_ptr_alignment()` 仍然假定每个 map value 从 8 字节对齐处开始，所以任何新的 map 类型只要返回 4 mod 8 的 value 指针，就把这个 bug 重新打开；selftest 覆盖三种 map 类型，而不是这条不变式。
- x86-64 的失败模式是从 split-lock 路径推出来的，系列里没有实测，而且取决于 CPU 是否支持 split lock detection。不支持的 CPU 只会付一次慢的 bus lock，然后活下来。
- 32 位审计的对象是 arm32：JIT 里没有 `BPF_ATOMIC`，解释器里是 `LDREXD`/`STREXD`。线程里没有考察其他 32 位 JIT 与解释器。
- trie 的修复改变内存计费：IPv4 节点涨 4 字节，map 的 `memlock` 计费随之增加；用户态看到的 key 格式不变。

## References

- [BPF 邮件列表：v2 系列的 cover letter](https://lore.kernel.org/bpf/20261124225055.87344-1-tphilipp@gmail.com/) —— 三种错位布局、两补丁的顺序约束，以及系列没有改什么。
- [BPF 邮件列表：32 位上的分配器头部](https://lore.kernel.org/bpf/20261124225055.87344-2-tphilipp@gmail.com/) —— `LLIST_NODE_SZ`、percpu 指针的搬迁，以及哈希 map 计费的改动。
- [BPF 邮件列表：lpm_trie 的 value 偏移](https://lore.kernel.org/bpf/20261124225055.87344-3-tphilipp@gmail.com/) —— 节点布局、`round_up` 补丁，以及 arm64 的 oops trace。
- [BPF 邮件列表：devmap 的 value 偏移](https://lore.kernel.org/bpf/20261124225055.87344-4-tphilipp@gmail.com/) —— `val` 在 52，以及 `__aligned(8)` 改动。
- [BPF 邮件列表：对齐 selftest](https://lore.kernel.org/bpf/20261124225055.87344-5-tphilipp@gmail.com/) —— XDP 程序、三次 lookup，以及 `addr & 7` 断言。
- [BPF 邮件列表：最初的 32 位分配器补丁](https://lore.kernel.org/all/20260717183632.99195-1-nbd@nbd.name/) —— 打开 32 位那一半的报告。
- [校验器的对齐检查](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/verifier.c) —— `check_ptr_alignment()`、`check_generic_ptr_alignment()`、`strict` 的来处，以及 `check_atomic()`。
- [LPM trie](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/lpm_trie.c) —— `struct lpm_trie_node` 与未取整的 `data_size` 偏移。
- [哈希 map 布局](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/hashtab.c) —— `htab_elem_value()`、`rhtab_elem_value()`，以及分配器的使用。
- [devmap](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/devmap.c) —— `struct bpf_dtab_netdev`，以及两个返回 `&obj->val` 的 lookup。
- [BPF 内存分配器](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/memalloc.c) —— `LLIST_NODE_SZ` 与 `ret + LLIST_NODE_SZ`。
- [arm64 BPF JIT](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/arch/arm64/net/bpf_jit_comp.c) —— `emit_atomic_ld_st()` 与独占指令序列那条路径。
- [arm64 缺页处理](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/arch/arm64/mm/fault.c) —— `do_alignment_fault()` 与 `do_bad_area()`。
- [arm64 启动控制寄存器](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/arch/arm64/include/asm/sysreg.h) —— `INIT_SCTLR_EL1_MMU_ON`，它没有置 `nAA`。
- [x86 陷阱处理](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/arch/x86/kernel/traps.c) —— `exc_alignment_check()` 在内核态直接 die。
- [x86 split lock 状态机](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/arch/x86/kernel/cpu/bus_lock.c) —— `sld_state`、启动默认值，以及只处理用户态故障的那一段。
- [Split lock 文档](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/Documentation/arch/x86/buslock.rst) —— warn/fatal 的表格，两种模式下内核态都写 "Kernel OOPs"。
- [BPF 解释器](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/core.c) —— `STX_ATOMIC` 那一族，以及 32 位走 `atomic64_*` 的位置。
- [ARM 原子操作](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/arch/arm/include/asm/atomic.h) —— `atomic64_*` 在 ARMv6K 及以后是 `ldrexd`/`strexd`。
- [arm32 BPF JIT](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/arch/arm/net/bpf_jit_32.c) —— `BPF_ATOMIC` 回退到解释器。

## 今日社区讨论

本次覆盖情况：通过授权的只读归档读了两个选择加入归档的 Slack 工作区（共三个白名单频道），24 小时与 7 天窗口里都没有消息，所以 Slack 一侧是「安静」而非「读不到」。两个白名单 Discord 工作区（共十个频道）读不到：网页客户端停在登录页，因此它们的缺席是覆盖缺口，不是没有讨论。r/eBPF 这次也读不到——普通的公开页面每次都返回一个网络安全拦截页——这是本次的第二个覆盖缺口。本篇的问题取自公开的 bpf 邮件列表归档，当日其余主题同样来自那里。

**对齐假定，以及一次找出回归的评审。** 「8 字节」是校验器代表所有 map 类型向 JIT 作出的承诺，而上面这个系列是第一次有人拿实现（而不是文档）去核对它。线程真正的技术价值在评审给出的那张表：修 32 位分配器头部的同一个补丁，会把 32 位的 IPv4 trie value 从对齐变成错位，因为节点的 `data` 偏移和分配器头部原本在互相补偿。任何改动「被 map 类型当作 value 返回」的结构的人，都该为自己的布局建同样一张表——基址对齐、成员偏移、value 偏移——并在两种位宽上各查一遍。仍然开放的边界是规则本身：校验器没有手段表达「这个 map 类型的 value 是 8 字节对齐的」，于是下一个 map 类型必须重新发现这个约束。

**一次 kfunc 原型变更，和一个读 BTF 的解析器。** 一个后续系列让内核把旧的 kfunc 原型留在新原型旁边，并按 BTF 原型（而不是按名字）去匹配 BPF 程序的声明。作者拿 1302 个对象、对三个内核 BTF 镜像重放了一遍 kfunc 解析，报告了哪些 `__weak` 声明会改绑、哪些保持未解析；树内的 `compat.bpf.h` 现在优先尝试 `___v2` 名字。读者该带走的是机制：`__weak` 声明不会大声失败，所以决定权在加载器，而只按名字回退会静默绑到内核留下来的那个变体上。仍然开放的问题是「被指向的结构体改了名」的声明会怎样——解析按原型匹配，于是改名后的结构体会被读成一个不同的函数，而不是一个不兼容的函数。

**可睡眠的 struct_ops，以及被复制四次的同一个检查。** 一个系列给 `Qdisc_ops`、`tcp_congestion_ops` 和 `smc_hs_ctrl` 各加一个 `check_member`，以此拒绝可睡眠程序。评审对两半都提出了异议：`Qdisc_ops` 的 `init` 与 `destroy` 跑在 `rtnl` 之下、可以睡眠，所以那里放可睡眠程序今天就是合法的；而且同一个函数抄四份说明默认值才是 bug——正确的修法是在 `check_struct_ops_btf_id()` 里做一次，已有 `check_member` 的 struct_ops 则保留自己的可睡眠程序。仍然开放的边界是「这个成员可以睡眠」与「这个程序可以睡眠」的区分：带 `check_member` 的 struct_ops 仍然得自己去测 `prog->sleepable`。

**现在可以睡眠的 local storage 分配。** task 与 inode storage 原先只走 `kmalloc_nolock()`，在本来可以阻塞分配成功的地方失败——报告描述的是 `ZONE_NORMAL` 争用叠加 hugepage 把 `ZONE_DMA32` 压到水位以下，且在生产环境观察到过。该系列在调用点选择分配方式：程序可睡眠时用 `GFP_KERNEL`，调用方可以自旋时用 `GFP_NOWAIT`，其余用 `kmalloc_nolock()`，依据是程序的 aux 元数据和当前抢占状态。值得记住的性质是「用哪个分配器」变成了一个有三种答案的运行时问题，于是过去在某种上下文里不会失败的 storage 分配，现在每种上下文都是不同的代码路径。

**对象加载完成后再 clone 程序。** 一个被报出来的 libbpf 缺陷：针对模块的 fentry 程序在 `bpf_object__prepare()` 之后可以 clone，`bpf_object__load()` 之后就失败，因为加载过程把模块 BTF fd 和 type id 缓存在程序里，而加载收尾时的清理关掉了那个 fd。复用释放掉的 fd 编号，能让 clone 绑到另一个模块的 type id 上加载成功——一次「成功但错误」的加载，比 `-EINVAL` 更糟。维护者的答复是：clone 是为「从不加载的对象里单独加载一个程序」设计的，另一种情形用 load strategy；仍然开放的边界是缓存下来的 attach 状态该在清理时失效，还是活到对象关闭为止。