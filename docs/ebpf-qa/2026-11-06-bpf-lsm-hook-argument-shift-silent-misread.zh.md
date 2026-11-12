# 为什么在 inode 系列 LSM 钩子的最前面加一个 mount idmap 参数，会悄悄让 BPF LSM 的每一个参数错位，又有什么办法能发现它？

因为 BPF LSM 程序并不按名字接收参数。`BPF_PROG` 宏把声明的参数列表展开成对一个无类型 64 位上下文槽数组的按位置读取，而内核按*内核侧*钩子原型在该下标处的类型来标记每一个槽，与程序自己声明的类型无关。于是，当六个 inode 钩子在位置 0 插入一个新参数时，用旧头文件编译出来的程序里，每一个原有参数都向右挪了一个槽；加载路径上没有任何检查会发现这件事：程序照常加载，字段仍按程序自己声明的类型算出偏移去读，读出来的却是另一个对象里的值。做出这一改动的内核提交进入了 7.4-rc1；同样的形状此前在 6.9 已经发生在 `inode_setattr` 钩子上。

## 机制

1. **内核改了什么。** 这个提交的说明是：security 层在 overlay 类操作中无法区分 identity-mapped 的上层与 idmapped 的上层，因此把 mount idmap 一路传进基于 inode 的钩子，让安全模块看到 VFS 已经看到的东西。签名改动是机械的：`inode_create`、`inode_link`、`inode_symlink`、`inode_mkdir`、`inode_mknod`、`inode_permission` 各增加 `const struct mnt_idmap *idmap` 作为第一个参数。mode 或 mask 参数、dentry 参数、inode 参数的相对顺序不变，但全部后移一位。6.9 的先例是一样的：当钩子定义与 EVM 注册的实现对齐时，`inode_setattr` 从 `(dentry, attr)` 变成了 `(idmap, dentry, attr)`。

2. **程序是怎么收到它们的。** 对 LSM 程序，libbpf 的 `BPF_PROG` 生成一个 `name(unsigned long long *ctx)` 包装函数，由它调用一个内联函数，而被内联函数的参数列表由 `___bpf_ctx_castN` 展开为 `ctx[0]`、`ctx[1]`、`ctx[2]`……，顺序就是声明顺序，并抑制 `-Wint-conversion` 把整数转成声明的指针类型。所以这份声明实际是一张*槽下标*清单，而不是按名字或按类型的绑定。把编译产物里的 BTF 打印出来可以看到，程序入口只有一个参数 `unsigned long long *ctx`：内联之后，声明的参数只存在于 C 源码里。

3. **内核怎么给它们定类型。** 加载时只有内核侧有原型。`btf_ctx_access` 把上下文偏移映射为参数下标，从挂载目标函数原型里取出该下标的参数类型，如果是指向结构体的指针，就返回带*所指向类型* BTF id 的 `PTR_TO_BTF_ID`。所以在一个 7.4 内核上，槽 0 是 `mnt_idmap` 指针、槽 1 是 `inode` 指针，不管程序以为它们是什么。最后一个参数之后的那个槽留给钩子的返回值，这正是声明参数多于钩子参数的程序仍然能加载的原因。

4. **为什么类型不匹配没被拦住。** 直接解引用上下文参数的程序，至少会按照错误的类型被检查一次，通常直接拒绝。用 `BPF_CORE_READ` 走字段的程序完全不被检查：宏展开为 `bpf_probe_read_kernel(dst, size, (const void *)__builtin_preserve_access_index(src))`，第三个参数类型是 `ARG_ANYTHING`，任何指针都接受。字段偏移在加载时也不取自内核对象；CO-RE 只把字段在*程序*声明类型里的偏移重定位成常量，再加上槽里到达的那个指针。于是类型不匹配退化成一次看起来合法的内核内存读取，辅助函数还报告成功。

5. **为什么读出来恰好是这些值。** 偏移可以从运行内核自己的 BTF 得到。在报告机器上的布局里，`i_ino` 位于 `struct inode` 的 64 字节处，`d_name` 位于 `struct dentry` 的 32 字节处、`qstr.name` 再往后 8 字节，所以读字符串用的是偏移 40。错位之后，程序用来读 `inode->i_ino` 的指针其实是一个 `mnt_idmap`，而在 136 字节的 `mnt_idmap` 里偏移 64 处是第二个 id map：当没有任何 id 映射时 VFS 传入的静态对象 `nop_mnt_idmap`，它只初始化了引用计数，所以那块区域是零，inode 号打印为 0。程序用来读 dentry 名字的指针其实是一个 `inode`，而 inode 偏移 40 处是超级块指针，于是字符串辅助函数拿到的是一个指向超级块的指针，报告的名字是空的。`bpf_probe_read_kernel_str` 在拷贝失败时会把目标清零，所以症状是干净的空串，而不是明显的垃圾。

6. **这不只是 CO-RE 的问题。** CO-RE 重定位的是*字段偏移*和类型是否存在；它没有"某个参数来自哪个槽"这种重定位。因此同一个二进制必然在改动的一侧出错：用 7.3 时代的头文件编译，在 7.4 上读错；用 7.4 的头文件编译，在 7.3 上读错，因为新的第一个参数被绑到了旧的第一个槽。只有按内核分别构建，或者在加载时显式检查，才能补上这个缺口。

## 验证与排障路径

1. **打印运行内核真正暴露的钩子原型。** `bpftool btf dump file /sys/kernel/btf/vmlinux format raw | grep -A2 "FUNC 'bpf_lsm_inode_create'"` 会显示校验器将要使用的 `FUNC_PROTO`，顺序与真实一致。在 7.3 或更早内核上第一个参数是 `struct inode *`；在 7.4 上是 `const struct mnt_idmap *`。任何希望熬过一次内核升级的 LSM 程序，都应在发布前做这一步，并在 CI 里和测试内核的 BTF 做对比。

2. **不要把源码里的声明当作 ABI 文档。** 导出编译产物，例如 `bpftool btf dump file prog.bpf.o format raw`，确认内核实际看到的东西：一个 `unsigned long long *` 参数。你写的参数名和类型到那时已经不存在了，这正解释了为什么没有任何加载器侧的参数个数检查会被触发。

3. **区分"读错"和"被拒绝"。** 落在错误类型结构体大小之外的字段偏移会在校验阶段以越过结构体的消息被拒绝；落在其内部的偏移，就像本例，会通过。因此"程序能加载"并不能证明参数正确，只有"在错误对象里读出了看起来合理的值"才能暴露它。

4. **用标记复现。** 让钩子返回一个固定值，同时把程序读到的两个值，与从稳定来源（例如同一个 inode 上的 kprobe）对同一操作观测到的值打印在一起对比。对一个确实存在的文件，目录 inode 号为 0、名字为空，就是参数错位的特征。在程序正确的内核上，同一次打印会显示真实 inode 号和被创建的名字。

5. **把原型变化当成一次新的 ABI。** 在已知布局之后，按运行内核的 `vmlinux.h` 重新构建可以为该内核消除歧义；再加一个针对目标内核 BTF 的钩子参数个数构建期断言，就能把静默情形变成加载期失败。

## 局限

- 这不是一个能被修复掉、从此消失的运行时 bug；它是"上下文按位置绑定"加上"在已发布分支中部改动签名"这两件事的性质。在出现显式规则或检查之前，任何给 inode 钩子参数命名的 LSM 程序，只要这些钩子在任何非末尾位置增加参数，就会暴露。
- 内核文档一方面声明 BPF 程序的参数属于 ABI，另一方面又把用 `bpf_probe_read` 之类辅助函数遍历内核内部结构的 tracing 程序排除在外。LSM 程序是另一种程序类型，但做的正是同一类事，所以文档给出的契约可以有两种读法；实际的问题就出在这种含糊上，而不在机制本身。
- 如果某个字段在两种类型里恰好位于同一偏移，上面所有检查都无法区分。于是错误的读取可以看起来完全合理：一个恰好是小整数的 inode 号，或者一个恰好指向合法字符串的名字。
- 如果把同一个参数加在签名末尾，对已有程序本是无害的（参数偏少是被接受的）；但那只是惯例而非强制规则，没有任何东西阻止下一次改动重复同样的模式。
- 程序作者可用的两条补救都是"按内核"的：按目标内核头文件重新构建，或者自己核对原型。两者都不能产生一个跨这次改动都正确的二进制。

## References

- [内核提交：lsm: expose mount idmaps to inode hooks](https://github.com/torvalds/linux/commit/16959c469f232f5c76cfb628297cafbf6dbdba79) — 动机说明、获得首位 idmap 参数的六个钩子，以及随之更新的树内调用方。
- [内核提交：security: Align inode_setattr hook definition with EVM](https://github.com/torvalds/linux/commit/314a8dc728d0) — 6.9 中给 `inode_setattr` 加上同一参数的改动，是这一模式的更早一次出现。
- [内核 `include/linux/lsm_hook_defs.h`](https://github.com/torvalds/linux/blob/master/include/linux/lsm_hook_defs.h) — 当前钩子签名，idmap 是 create、link、symlink、mkdir、mknod、permission、setattr 这些钩子的第一个参数。
- [v7.3 的内核 `include/linux/lsm_hook_defs.h`](https://github.com/torvalds/linux/blob/v7.3/include/linux/lsm_hook_defs.h) — 改动之前的同一批钩子，可直接做位置对比。
- [内核 `kernel/bpf/btf.c`](https://github.com/torvalds/linux/blob/master/kernel/bpf/btf.c) — `btf_ctx_access` 按挂载目标在该下标的参数来给上下文槽定类型；`btf_struct_access` 在给定类型内部解析字段。
- [内核 `kernel/bpf/verifier.c`](https://github.com/torvalds/linux/blob/master/kernel/bpf/verifier.c) — `check_ptr_to_btf_access`，以及 LSM 程序上下文参数的访问规则。
- [内核 `kernel/trace/bpf_trace.c`](https://github.com/torvalds/linux/blob/master/kernel/trace/bpf_trace.c) — `bpf_probe_read_kernel` 的原型（源参数为 `ARG_ANYTHING`），以及失败时清零目标的字符串版本。
- [libbpf `src/bpf_tracing.h`](https://github.com/libbpf/libbpf/blob/master/src/bpf_tracing.h) — `BPF_PROG` 包装函数、按位置展开的 `___bpf_ctx_cast`，以及被抑制的转换告警。
- [libbpf `src/bpf_core_read.h`](https://github.com/libbpf/libbpf/blob/master/src/bpf_core_read.h) — `bpf_core_read` 展开为带 preserve-access-index 参数的内核 probe read 辅助函数。
- [内核 `Documentation/bpf/bpf_design_QA.rst`](https://github.com/torvalds/linux/blob/master/Documentation/bpf/bpf_design_QA.rst) — BPF 程序参数属于 ABI 的表述、probe read 类 tracing 程序的例外，以及针对 kprobe 与 tracepoint 的单独裁定。
- [内核 `Documentation/bpf/prog_lsm.rst`](https://github.com/torvalds/linux/blob/master/Documentation/bpf/prog_lsm.rst) — LSM 程序如何声明上下文，以及 CO-RE 被期望如何适配字段访问。
- [内核 `fs/mnt_idmapping.c`](https://github.com/torvalds/linux/blob/master/fs/mnt_idmapping.c) — `nop_mnt_idmap` 的定义与 id map 结构的布局。
- [公开 bpf 邮件列表报告：7.4 的 inode 钩子新增首位参数会静默破坏 BPF LSM](https://lore.kernel.org/bpf/356db94e-f5d6-4fb2-839a-b97aef72256b@app.fastmail.com/) — 复现程序、7.1 与 7.4-rc1 的输出，以及两条建议：把参数加在末尾、在加载时检查声明参数。
- [公开 bpf 邮件列表缺陷报告：开启 clang KCSAN 时 tcp kfunc 从 BTF 中消失](https://lore.kernel.org/bpf/a69dd097-7e71-41f0-9a6b-eda001773236@linux.dev/) — 参数被重新赋值导致 pahole 丢弃 kfunc、resolve_btfids 的致命告警，以及没有该告警时的注册失败。

## 当日社区讨论

本次运行检查了六个白名单社区。其中两个是可通过已批准的私有只读归档访问的 Slack 工作区，合计覆盖三个白名单频道：一个频道在 24 小时窗口内没有消息，另一个 24 小时内没有消息、七天内有三条。两个没有归档的聊天工作区未被覆盖，因为这个环境没有它们的登录会话，可见页面停在登录页；它们记为不可访问，而不是安静。内核邮件列表和实践者论坛通过公开页面阅读，两边都有活动。合计四个来源有内容或确认的空窗口，两个未覆盖。

邮件列表带来了上面回答的问题，其窗口的其余部分形成两个主题：不带错误就消失的接口，以及错误路径上缓冲区的归属。第一个主题今天还有第二个实例。某个 kfunc 在开启了 sanitizer 的构建里从 vmlinux BTF 中消失了：编译器在一条路径上对函数参数重新赋值，其位置列表不再描述入口值，BTF 生成器于是丢掉这个参数，进而丢掉整个 kfunc。把告警当作错误时这会构建失败；不当作错误时构建成功，但子系统根本不注册，其 initcall 返回一个参数错误，运行期它的 kfunc 全部消失且没有任何消息。报告者的绕法是不再对参数重新赋值，而一位评审的反对意见对每个写 kfunc 的人都成立：这会让"参数不许重新赋值"看起来像规则，而真正的缺陷在工具链。同一主题也贯穿上面回答的问题，以及窗口里一个 struct ops 补丁：本应在某条状态转换路径上重置的字段被漏掉了，自动评审的回复则同时指出，重置辅助函数在补丁自称基于的树里并不存在，而且在该路径上调用它会把调度器已经设置好的默认时间片清零。

第二个主题是操作失败时缓冲区与状态归谁所有。一个系列修复 XDP 重定向路径：核心在告知驱动重定向失败之后仍然释放了零拷贝缓冲区，而驱动本应继续持有它；那些稍后才释放、或稍后才使用缓冲区的驱动，会在帧可能已被重新分配之后释放它，把同一个用户态内存帧交给用户空间两次。补丁把释放移到能安全释放的调用方，并调整了依赖旧行为的驱动。第二个系列延续另一个页计数问题：某条接收路径把每个页的引用计数偏置到最大值，于是来自 splice 已收数据的一次额外引用就会把它溢出；修复下调偏置，但前提是先恢复一次后续提交丢掉的队列格式检查，因为它使某条无法保持引用的重定向路径变得可达。同一列表上的第三个系列处理比回调活得更久的 map 值：删除值的时候取消 task work、工作队列任务或定时器，并不能阻止已经开始的回调，于是该系列在发布回调上下文之后重新查找该值以核对成员关系，并给出了早期修订版仍会在已释放元素上挂上工作的反例。

可通过归档访问的可观测性频道当天很安静，其七天窗口里有一条三条消息的讨论，主题是谁该拥有那个"压下厂商 SDK 自己的 trace 导出、让插桩路径成为唯一来源"的开关。技术内容是关于配置归属的争论而非 eBPF 本身：反对意见认为这个决定属于 SDK，回答则是插桩本身已经修好了手工插桩路径，所以这个开关只是去掉重复。实践者论坛在 24 小时窗口内没有新帖，最新的帖子是四天前的；其中讨论最多的一条是此前某篇回答中提到的"先学习再处置"代理，其未解问题没有变化：当它最先学到的就是已被入侵的状态时，行为基线该如何处理。

本页回答的问题来自公开邮件列表，页面里每一条机制性论断都对照内核源码、运行内核自身的 BTF 以及一份实际编译的对象文件核对过，而不只是采信报告本身。