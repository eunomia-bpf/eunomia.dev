# 内核 kfunc 的 C 原型没变，为什么它会从 vmlinux BTF 里消失，又为什么会因此构建失败？

因为把 DWARF 转成 BTF 的编码器是依据编译器记录的参数位置，而不是依据 C ABI，来决定是否输出一个函数。一个 kfunc 的参数如果被编译器描述成"溢出后位于被调用者保存寄存器"，或者只在函数入口之后才有描述，在编码器看来就是"参数用了意外的寄存器"；于是它跳过整个函数，vmlinux BTF 里就没有了这个 C 原型从未变过的 kfunc 对应的 FUNC 条目。后面有两个步骤依赖这个条目：给内核 kfunc ID 集合填值的那一步，以及内核自己注册 kfunc 的那一步。前者在 CONFIG_WERROR 下把缺条目变成构建失败；后者把缺条目变成一次返回 -EINVAL 的 initcall，于是一个在内核源码、配置、C 原型三处都没变的情况下，kfunc 在启动后依然没有注册成功。

## 机制

1. **kfunc 宏到底保证了什么。** 内核 BTF 头文件里，`__bpf_kfunc` 展开为 `__used __retain __noclone noinline`。由此得到的、写进文档的结论是：kfunc 保持其 C 原型的调用约定，它不会被内联、不会被克隆、也不会因为"没人用"而被删掉。kfunc 文档还规定 kfunc 不能声明为 static，因为它的对外可见名字正是 BTF 解析用来找它的依据。所以 C 里写下的原型就是 ABI，而按源码签名生成的 BTF 函数原型在构造上就是正确的。

2. **BTF 从哪里来。** 编译器在构建内核时产生 DWARF。`scripts/gen-btf.sh` 先对这份 DWARF 跑编码器，得到一个独立的 BTF 数据块，再对数据块和 ELF 跑 `resolve_btfids`，施加内核特有的变换，并把 kfunc 集合所用的 `.BTF_ids` 条目填上值。所以 BTF 是"编译器 + 编码器"的构建产物，不是源码的属性。

3. **编码器在什么条件下跳过。** 编码器为每个函数定义保存一份状态，稍后统一输出。每个参数都带着 DWARF 描述的位置：寄存器、入口值表达式、栈槽、或常量。对签名未变的 clang 产物，编码器会把每个参数与该位置期望的 ABI 实参寄存器比对，记录在别的寄存器里的参数就被标成"意外的寄存器使用"。真正输出已保存的函数时，这个标记、不确定的参数位置、参数顺序被打乱、参数被优化掉、原型不一致、符号地址有歧义，任何一条都会给出一个跳过原因，函数干脆不进 BTF。在 verbose 模式下编码器会打印原因，例如因参数用了意外寄存器而被跳过。

4. **kfunc 为什么容易中招。** kfunc 带 `noinline`，于是 clang 会在自己发起的调用前后溢出并重读参数。产生的 location list 对"值住在哪"是准确的，但在那个位置上并不是 ABI 实参寄存器：报告的案例里，第二个参数的位置列表从函数入口之后才开始，且点名一个被调用者保存寄存器；同一周里另一个 kfunc 通过另一种 DWARF 形状命中了同一个跳过。这仍然是对值所在位置的准确描述，而且对 ABI 是中性的，因为 `__bpf_kfunc` 早就钉住了调用约定。编码器这项检查本意是保护调用方免受真正变了的签名之害，却分辨不出这两种情况。这个形状可以复现：编译一个小的 `noinline` 函数，在用到第二个参数之前先向外调用，就会得到第一段为入口寄存器、后续段为被调用者保存寄存器的 location list。在一台 arm64 的 KCSAN 构建上，那个参数在被使用前被重新赋值过一次，`bpf_tcp_ops_set_rcvlowat` 从 vmlinux BTF 中消失。

5. **函数缺失在构建期打断了什么。** `resolve_btfids` 遍历带有 kfunc 标记的 BTF 集合定义，把每个列出的名字解析成 `BTF_KIND_FUNC` 条目，解析失败就告警。该集合条目于是留着占位值，第二个告警就把这个符号报告为未解析。两者都只是告警，但内核的 BTF makefile 在 `CONFIG_WERROR` 打开时会给 `resolve_btfids` 追加"告警即失败"开关。告警于是变成非零退出，`gen-btf.sh` 失败，链接步骤打印 `Failed to generate BTF for vmlinux`，并建议关掉 `CONFIG_DEBUG_INFO_BTF`。

6. **没有 CONFIG_WERROR 时它打断了什么。** 构建成功，那个条目被填成零。子系统在 `late_initcall` 注册其 kfunc 集合时，内核查询函数 ID 并校验它确实指向一个函数，而 ID 零是 void 类型，校验返回 `-EINVAL`。initcall 返回 -22，这个 kfunc 集合以及它背后的 `struct_ops` 注册都没发生，唯一的痕迹是日志里的那一行 initcall。此后 BPF 程序在加载时失败，而内核的源码、配置、C 原型全都没变。

7. **绕行与真正的修复。** 立刻能用的绕行是源码层、且与编译器形状挂钩的：不要让参数被重新赋值，从而让它留在入口寄存器里，例如把钳位折进调用的实参表达式。同一形状的另一次报告用了编译器屏障，让参数的位置描述成内存而不是寄存器，选它是因为那个在某个架构上奏效的"藏寄存器"宏在另一个架构上会出问题。真正的修复属于编码器：对已被识别为 kfunc 的函数，保存状态时清掉由位置推导出的标记，输出已保存函数时跳过基于位置的检查，并且只要编码 BTF 就收集 kfunc，而不再只在 decl tag 特性打开时才收集。原型不一致与地址有歧义两项检查保留，因为它们描述的是真实的 ABI 疑点。decl tag 那个条件必须去掉，因为 `resolve_btfids` 现在自己发 kfunc decl tag，内核已经不再给编码器传那个特性了。

## 验证与排障路径

1. **直接问运行中的内核有什么。** 缺 FUNC 条目就是全部缺陷，所以要查产物而不是查源码：

   ```console
   $ bpftool btf dump file /sys/kernel/btf/vmlinux format raw | grep "FUNC 'bpf_tcp_ops_set_rcvlowat'"
   $ bpftool btf dump file /sys/kernel/btf/vmlinux format raw | grep -c "DECL_TAG 'bpf_kfunc'"
   ```

   计数可以作为健全性检查：在当前构建链下 decl tag 来自 ID 集合，所以一个仍有集合条目却连 tag 都没有的函数，指向的是另一种失败，而不是被丢弃。

2. **读编码器自己给出的原因。** 对失败的那个目标文件以 verbose 重跑编码器，grep 那条跳过信息。跳过原因会点明是哪一类参数，这正是把它与"原型真不一致"区分开的依据。

3. **看编码器看到的 DWARF。** 反汇编调试位置列表就能直接看到形状：一段在函数入口之后才开始、且点名被调用者保存寄存器的区间，或者一个入口值表达式。再对照反汇编确认该区间里值确实在那个寄存器；位置列表本身只是一份描述，并不是 bug。

4. **区分"构建失败"和"静默未注册"。** 如果构建停了，先找缺函数告警，再找"告警即失败"的退出。如果没停，就在启动日志里 grep 那行 initcall 和它的 -22，再确认 kfunc 集合到底注册了没有。两种结果同一个根因，证据不同。

5. **用 BTF 差异确认修复，而不是重建两次凭感觉。** 用修好的编码器重建，导出两份 BTF，比较函数列表。已发出的修复报告说只有那两个被丢掉的 kfunc 回来了，vmlinux BTF 里别的都没变；差异更大就说明编码器改动超出了这个豁免。

## 局限

- 这个豁免与编码器版本绑定。已发布的编码器版本仍会丢掉这些函数，所以在修复落地前，需要源码层的绕行、钉住编码器版本、或者回移补丁。在一台编码器早于 clang 位置匹配的主机上，同一个目标文件会编码得很干净——所以不记录编码器版本，"我们这边构建没问题"这个结论并不可靠。
- 这不是 FUNC 条目丢失的唯一途径。内联、原型不一致、签名变化、符号地址有歧义都会让函数被丢掉，而那些情况下的丢弃是正确的。kfunc 之所以特殊，只是因为宏已经把调用约定钉死了。
- 完整的检查是一个"构建再导出"的循环，所以它属于内核构建验证，不属于某个 BPF 程序的单元测试。程序在加载时无法察觉它的 kfunc 是否在 BTF 里；它看到的是普通的"unknown func"或缺少 `struct_ops` 成员，指向的层是错的。
- 这些都不能让一份运行中内核的 BTF 在多次重建之间变得可信。同一份源码用不同的编译器、编码器或优化级别构建，就可能把某个参数放进不同的寄存器，从而改变哪些函数会被输出。

## References

- [内核 `include/linux/btf.h`](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/include/linux/btf.h) — `__bpf_kfunc` 的展开，正是它让 C 原型成为 ABI。
- [内核文档：BPF kernel functions](https://docs.kernel.org/bpf/kfuncs.html) — kfunc 标注规则、ID 集合宏，以及"resolve_btfids 在构建期发出 decl tag"这句陈述。
- [内核文档：BPF Type Format](https://docs.kernel.org/bpf/btf.html) — BTF 描述什么，以及它在加载期如何被使用。
- [内核 `scripts/gen-btf.sh`](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/scripts/gen-btf.sh) — 先编码器、后 `resolve_btfids` 的两遍流程。
- [内核 `scripts/Makefile.btf`](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/scripts/Makefile.btf) — 与 `CONFIG_WERROR` 绑定的"告警即失败"开关，以及编码器特性列表。
- [内核 `scripts/link-vmlinux.sh`](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/scripts/link-vmlinux.sh) — `Failed to generate BTF for vmlinux` 那条路径。
- [内核 `include/linux/btf_ids.h`](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/include/linux/btf_ids.h) — 集合的内存布局，以及标记"这个集合是 kfunc"的那个标志位。
- [内核 `kernel/bpf/btf.c`](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/btf.c) — kfunc 原型校验，以及未解析 ID 在注册时产生的 `-EINVAL`。
- [内核 `net/ipv4/bpf_tcp_ops.c`](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/net/ipv4/bpf_tcp_ops.c) — 因参数重新赋值而触发丢弃的那个 kfunc、它的 ID 集合，以及返回 -22 的 initcall。
- [`resolve_btfids` 源码](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/tools/bpf/resolve_btfids/main.c) — kfunc 收集、缺函数与未解析符号两个告警，以及"告警即失败"的退出。
- [`pahole` 编码器源码](https://git.kernel.org/pub/scm/devel/pahole/pahole.git/plain/btf_encoder.c) — 已保存函数的各种跳过原因，以及已发补丁里的 kfunc 豁免。
- [`pahole` DWARF 加载器](https://git.kernel.org/pub/scm/devel/pahole/pahole.git/plain/dwarf_loader.c) — 参数位置的记录，以及 ABI 实参寄存器窗口的检查。
- [公开报告：clang KCSAN 构建下 kfunc 从 BTF 消失](https://lore.kernel.org/bpf/9ccfc1f1-1214-4374-bac4-f4d91103d149@linux.dev/) — 症状、去掉重新赋值的绕行，以及 initcall 的返回值。
- [公开补丁：保留 kfunc 的源码原型](https://lore.kernel.org/bpf/20261111222447.322978-1-yonghong.song@linux.dev/) — 为 kfunc 清掉由位置推导的标记，并改为无条件收集。
- [公开补丁：sched_ext 的编码器绕行](https://lore.kernel.org/bpf/7c139810be5ffc9764ed4236efc28794@kernel.org/) — 更早的同形状丢弃、它的编译器屏障，以及"藏寄存器"方案为何与架构相关。
- [把 decl tag 移交给 `resolve_btfids` 的内核提交](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/commit/?id=692393104909ee14686488995ad5009a618fb0c7) — 为什么编码器不再需要内核停止传入的那个 decl tag 特性。

## 当日社区讨论

本次运行通过已批准的私有只读归档检查了两个白名单社区，覆盖三个白名单频道：一个频道在 24 小时窗口内没有消息，七天窗口内也没有；另外两个 24 小时内没有消息、七天内共四条，而这四条都属于同一个讨论会议订桌子的主题，没有技术内容。两个 Discord 工作区（共十个白名单频道）未被覆盖：本环境没有已认证会话，公开邀请页虽然现在能打开，但只提供了创建账号的入口。它们按不可访问记录，而不是按安静记录。从业者论坛本次可以读取，窗口内没有帖子；最新的帖子是四天前的，其中两篇有实质内容。内核邮件列表通过其公开归档读取，上面这个问题和下面这些主题都来自那里。

### BTF 是从调试信息里"生成"的，所以元数据缺陷就是构建缺陷

上面回答的案例是本次窗口里最清楚的一个，而同一周里还有第二个实例。另一个调度器补丁早就为另一个 kfunc 绕过了同一编码器行为：把参数强行留在内存里，让它的位置不点名任何寄存器；那个补丁明确写道，"藏寄存器"的辅助宏会修好一个架构却弄坏另一个。两件事合起来描述的是同一条边界：一个 kfunc 是否存在于 BTF，由一个 DWARF 启发式决定，所以同一份 C 原型可以在这次构建里存在、下次构建里消失。实用的诊断是在记录内核构建时同时记录编码器版本，并跨构建比较 BTF 导出，因为 BPF 程序在加载时看到的错误点的是 kfunc 的名字，不是编码器。尚未确定的是部署面：在编码器修复发布之前，使用受影响编码器加上 `CONFIG_WERROR` 的发行版内核会构建失败，而不带那个开关的内核会带着未注册的 kfunc 集合启动。

### 循环迭代计数换了形状，而它可能改变判定结果

一个活跃的六补丁系列把校验器标量演化代码里那八处手写边界换成一个小的标量抽象，并把循环的 backedge-taken 计数收进一个函数，取代原先分给常量和区间的两条路径。文中陈述（而非暗示）了两个后果。更多循环现在能得到计数，包括一些今天会退回 may_goto 机制的循环；而计数可以带来指令成本：cover 里给出，当循环体里某个值没有线性形式时，32 次迭代的循环从 90 条指令变成 247 条。另外，有计数的循环会被概括成"初值 + 步长 × 次数"，于是一个出口给出的次数可能让另一个出口并未约束的索引越界；cover 展示了一个今天能加载的循环，在计数补丁之后被拒、在锁存条件补丁之后又能加载。它给出的验证异常完备：约 110 万个公开程序判定不变，连两个指令数变多的 Rust 程序都点了名。对实践者而言，实用规则是在升级内核时保留失败程序和它的校验器日志，并把指令数的变化读成"校验器需要证明的东西真的变了"。

### 网络元数据正在被统一，而挂载点决定机制

本次窗口里有两个相关系列。一个提议为每个 `skb` 增加扩展区，让接收侧元数据能活过 `XDP_PASS` 和重定向，理由是哈希、VLAN 标签和校验和正被分散地一次次修补；一位评审的顾虑是 XDP 正在长出第二个元数据区、慢慢变成 `skb`。另一个为每个设备增加发送侧预留头部空间，让在出口压入头部的程序不再重新分配包的头部，它特意把这个新值和驱动已有的 needed headroom 区分开，因为后者在设备创建时固定、抬高它会拖累现有路径。交换里有两个细节值得记下：预留量可以由 tc 程序上的 BTF 声明标签给出，多个挂载程序的标签会相加，总量再被截断，所以标签是一个请求而不是保证；预留只在分配包的地方生效，这正是 TCP 被明确排除在外的原因。尚未解决的边界是 veth 对端的原生 XDP 仍会复制，因为头部不仅要够大，还必须是页片段。

### AF_XDP 零拷贝正在被改写成 page-pool provider

一个更大的系列把 UMEM 变成 page-pool 的内存提供者，于是本来就有 page pool 的驱动不再为零拷贝维护第二个接收分配器。结构性改动包括：用重新填充契约取代 NAPI 唤醒、用显式能力位说明栈是否可以保留 provider 的缓冲区、以及保持 `xdp_buff` 不变而把头部指针放在驱动内存里。cover 如实写了自己的代价：被线性化后重定向到设备映射或 cpumap 的多缓冲区帧会被丢弃，通用 XDP 与 cpumap 无法为 provider socket 供包，且只支持对齐的 chunk。它还留给用户态一个开放问题：socket 在 bind 之前如何得知 chunk 约束。对今天就在跑 AF_XDP 的人，有用的读法是：零拷贝快路被保留，改变的是拷贝模式的管道，所以吞吐结论应该在系列落地后按驱动重新测量。

更小的信号，每一个都是"子系统承诺什么"与"工具推导出什么"之间的边界。一个 kallsyms 符号查找优化系列已到第三版，提醒人们大型生成的符号表有自己的构建期成本曲线。而预留头部空间的讨论里只在 tc 挂载的设计，显示了那个反复出现的取舍：按设备挂载让内核知道该更新哪个设备，而被多个设备共享的过滤器不行。