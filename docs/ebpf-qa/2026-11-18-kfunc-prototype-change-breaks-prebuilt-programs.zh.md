# 当 kfunc 的原型发生不兼容变更时，为什么已经编译好的 BPF 程序就不再能加载，又需要怎样才能让它们继续工作？

**简短回答：** libbpf 在加载时先按名字、再按原型解析 kfunc 调用。它向内核查询的名字是 BPF 声明里的名字去掉最后一个 `___<flavor>` 后缀，而内核侧的这个查询是精确匹配：不会退回去找一个带后缀的 kfunc。所以当一个命名空间把不带后缀的名字让给新原型、把旧原型挪到后缀名下之后，用旧原型编译出来的目标文件仍然去查不带后缀的名字，原型检查失败，于是要么加载直接失败，要么变成一条被投毒的调用，一旦被校验器走到就被拒绝。

按目前的内核与 libbpf，只有两条路能让旧目标继续加载：把**旧**原型留在内核的不带后缀名字上、把后缀给新的；或者让 BPF 源码显式声明内核那一侧带后缀的名字，并用弱符号做守卫。一个在内核侧搜索 `<name>___<flavor>` 的 libbpf 回退已经在邮件列表上评审，但尚未合入，而且它只会对 libbpf 1.9 及以后生效。

## 机制

### 1. BPF 一侧对后缀宽容，内核一侧不宽容

kfunc 的名字匹配用的是*本质名*（essential name），也就是最后一个 `___` 分隔符之前的部分。`bpf_core_essential_name_len()` 从字符串末尾往前找 `X___Y` 形态，返回前缀长度。libbpf 为它找到的每个 extern 符号记下这个前缀：

```c
ext_essent_len = bpf_core_essential_name_len(ext->name);
ext->essent_name = NULL;
if (ext_essent_len != strlen(ext->name))
        ext->essent_name = strndup(ext->name, ext_essent_len);
```

解析时查的就是本质名：

```c
kfunc_id = find_ksym_btf_id(obj, ext->essent_name ?: ext->name, BTF_KIND_FUNC, ...);
```

而 `find_ksym_btf_id()` 就是一次普通的「名字到 BTF ID」查找：先在 vmlinux 上做 `btf__find_by_name_kind()`，再依次在每个模块自己的 BTF 上做。它从不去枚举同名的这一族函数。这个不对称就是全部问题的来源。BPF 程序完全可以写 `bpf_foo___local`，libbpf 照样去找内核的 `bpf_foo`；但内核若把旧函数改名成 `bpf_foo___v1`，libbpf 就没有任何办法从一个 `bpf_foo` 的声明找到它。

### 2. 「兼容」到底是什么意思

解析到的函数原型会与声明用 `bpf_core_types_are_compat()` 比较——这正是 libbpf 对 kfunc 一直在用的那套 CO-RE 类型匹配。它的规则一边宽松一边严格：

- 函数原型兼容的条件是参数个数相同，并且每个参数和返回值都兼容；
- 整数之间互相兼容，大小与有无符号都被忽略；
- struct、union、enum 和前置声明的名字被完全忽略；
- typedef、`const`、`volatile`、`restrict` 会被穿透；
- 指针必须指向兼容的类别，所以 `struct file *` 与 `struct task_struct *` 不同，而 `struct file *` 与 `struct file___x *` 相同。

后果是：一大类变更对这项检查完全不可见，因此它们都是*原地*做的——把参数从 `u32` 加宽到 `u64`、加 `const`、改类型名、改参数名。旧目标会静默地绑定到新定义，没有任何版本信号。只有改变形状的变更——增删参数、改变返回类别、改变回调原型、改变指针所指对象的类别——才会让检查失败。

### 3. 失败的样子：强符号与弱符号

对强的 `__ksym` extern，原型不兼容是硬错误：

```
extern (func ksym) 'scx_bpf_dsq_insert': func_proto [..] incompatible with vmlinux [..]
```

`bpf_object__load()` 返回 `-EINVAL`。

对弱 extern，失败被推后，而且安静得多。libbpf 让这个 extern 保持未解析状态，在重定位阶段把调用改写成一条刻意非法的 helper 调用：

```c
insn->code = BPF_JMP | BPF_CALL;
insn->imm = POISON_CALL_KFUNC_BASE + ext_idx;
```

如果校验器走到了这条指令，会报一条「调用了未知函数」的非法调用，libbpf 的校验器日志修补再把那一行改写成缺失的 kfunc 名字：

```
invalid func unknown#2002000000
kfunc 'scx_bpf_dsq_insert___v2' is referenced but wasn't resolved
```

如果校验器根本没走到，那就什么都不会报：不可达的路径在校验之后被替换成一条 trap，加载成功，而那次调用从未执行。

compat 宏正是依赖「没走到」。`bpf_ksym_exists()` 在编译器确认符号是 `__weak` 之后展开成 `!!sym`，它守卫的分支是静态决定的：已解析的类型化 kfunc 是一个指针，而对任何已被证明非空的指针，校验器都会把 `指针 != 0` 直接判定为真；未解析的弱 kfunc 则是一个普通常量 0。没走到的那个分支成为死代码，它那条被投毒的调用从不会被校验。

### 4. 引出这个问题的真实案例

一个面向调度器的 kfunc 家族把返回值从 `void` 改成了 `bool`。第一次尝试把新原型放在不带后缀的名字上、把旧原型留作 `<name>___compat`，理由是认为 libbpf 会把旧二进制匹配到带后缀的那个函数。它并不会，于是所有先前编译好的调度器都加载失败；修法是把命名反过来：`void` 留在不带后缀的名字上，`bool` 版本加 `___v2`。

第二个缺陷源自同一个不对称，也是更有教益的那个。vmlinux.h 把新函数拼成 `<name>___v2`，于是 compat 头文件就那样声明它。libbpf 去掉后缀、查内核的不带后缀名字、找到的是*旧*原型、判定不兼容，于是让这个弱 extern 保持未解析。`bpf_ksym_exists()` 就在每一个拥有新 kfunc 的内核上都返回 false，宏落到更老的那个变体，一个参数被静默丢掉——没有任何警告，因为缺失的弱符号不算错误。修法是在 BPF 源码里再追加一个后缀，这样去掉一层之后剩下的正好是内核的精确名字。这就是当前模型「内核命名泄漏进 BPF 程序」的代价：只有当去后缀之后的名字恰好与内核一致时，这个守卫才有效。

### 5. 在评审的系列改了什么，以及它的构建期守卫

在评审的系列让 libbpf 从失败改为恢复：如果去后缀名字对应的 kfunc 缺失，或它的原型不兼容，libbpf 就在名为 `<name>___<flavor>` 的 kfunc 中搜索——先 vmlinux，再第一个有这类函数的模块——范围限定在带 `bpf_kfunc` decl tag 的函数里，并绑定原型兼容的那一个。原有的精确查找先跑，所以今天能工作的绑定全都保持不变。多于一个兼容的 flavor 就是错误，即使声明是弱符号也一样，理由是「歧义不等于缺失」。

如果一份声明可以同时匹配两个家族成员，这个搜索就不安全，因此 `resolve_btfids` 增加了一项内核构建期检查：在去重之后，用同一个 libbpf 函数比较每一对共享本质名的 kfunc，只要它们兼容就让构建失败。例外是别名——一个带后缀、且原型与不带后缀者逐字相同的 kfunc——因为不带后缀的查找先命中，别名本来也轮不到。这项检查也是为什么该命名空间里最新的 compat 宏还能继续工作：被别名化的 `___v2` 与普通函数在构造上就是兼容的。

规则文档随同一系列一起提交，作为生命周期章节的新一节，其中明确建议：当旧版 libbpf 或 light skeleton 必须继续工作时，给新原型起一个不以 `<name>___` 开头的名字；以及不要原地重命名一个 `<name>___vN` 的 kfunc，因为程序可能正按那个名字调用它。

## 如何验证

1. 在运行中的内核上枚举这一族函数，直接读真实原型，而不是相信 vmlinux.h：

   ```
   bpftool btf dump file /sys/kernel/btf/vmlinux format raw | grep 'FUNC .*<name>'
   ```

   每个带后缀的兄弟函数都会带着自己的 `FUNC_PROTO` 出现，所以「去后缀之后为什么失败」的那个参数个数差异是直接可见的。

2. 问 libbpf 它到底绑定了谁。打开 debug 日志（环境变量 `LIBBPF_LOG_LEVEL=debug`）后，成功的解析会说明符号取自哪个对象、选中了哪个 BTF ID：

   ```
   extern (func ksym) 'scx_bpf_dsq_insert': resolved to vmlinux [61234]
   ```

   不兼容的原型则会打印 `func_proto ... incompatible` 警告，并写明是模块还是 vmlinux 以及两侧的类型 ID。一个加载成功、行为却像少了一个参数的程序，应该先看这里：debug 行会告诉你实际绑定到了这一族的哪一个成员。

3. 对弱 extern，强制走到被守卫的路径。如果日志声称该 kfunc 被引用但未被解析，说明声明去掉后缀后的名字没有匹配到内核里的某个原型，程序走的是回退分支：

   ```
   kfunc 'scx_bpf_dsq_move_to_local___v2' is referenced but wasn't resolved
   ```

## 局限

兼容性检查是结构性的，因此抓不住「形状没变、含义变了」的改动：两个整数参数互换、调用方在 32 位里算过的偏移被加宽、`bool` 返回值的 `false` 有了新含义。这些都能干净地加载到新内核上，然后出错。

flavor 只有在内核开发者把旧 wrapper 编译进来、并以相同 flag 注册、在整个弃用窗口内保持时才有效，而且当前树里能找到它的回退还没合入；一份在本月 libbpf 上能工作的声明，在发行版的 libbpf 上仍然失败，而 light skeleton 与内核的精确名字查找根本看不到 flavor。别名例外还给内核侧带来一项没有任何检查能强制的义务：两个兼容的家族成员必须保持语义相同。

最后，kfunc 被移除的正式信号是弃用 flag 加上 kernel-doc 说明，而不是 ABI 承诺；原型随时可以变，树外目标真正的保护只有两件事：写清楚自己是按哪个原型编译的，并让这项守卫足够便宜。

## References

- [libbpf 的 kfunc 解析，上游源码](https://github.com/torvalds/linux/blob/master/tools/lib/bpf/libbpf.c) — 本质名剥离、内核侧精确查找、`bpf_core_types_are_compat()`、弱 extern 投毒，以及校验器日志修补。
- [libbpf 的 CO-RE 类型兼容规则](https://github.com/torvalds/linux/blob/master/tools/lib/bpf/relo_core.c) — 逐类别的精确比较，包括整数与 struct 名字的放宽。
- [内核 kfunc 生命周期约定](https://github.com/torvalds/linux/blob/master/Documentation/bpf/kfuncs.rst) — kfunc 没有硬性稳定性保证，以及弃用如何表示。
- [公开 bpf 邮件列表系列封面信：内核侧 kfunc flavors](https://lore.kernel.org/bpf/20261117192153.3384031-1-ihor.solodrai@linux.dev/) — 动机、sched_ext 的历史，以及各版本之间同名原型变更数量的实测统计。
- [公开 bpf 邮件列表补丁：回退到内核 flavor](https://lore.kernel.org/bpf/20261117192153.3384031-2-ihor.solodrai@linux.dev/) — 搜索顺序、歧义报错，以及未改动的精确查找。
- [公开 bpf 邮件列表补丁：拒绝有歧义的 kfunc 家族](https://lore.kernel.org/bpf/20261117192153.3384031-4-ihor.solodrai@linux.dev/) — `resolve_btfids` 里的逐对检查、别名例外，以及按 decl tag 识别家族。
- [公开 bpf 邮件列表补丁：记录如何修改 kfunc 原型](https://lore.kernel.org/bpf/20261117192153.3384031-5-ihor.solodrai@linux.dev/) — 规则本来的文档形态，包括无害变更清单与「换个名字」的建议。
- [内核 sched_ext 兼容性修复提交](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/patch/?id=2dbbdeda77a6) — 把旧原型放回不带后缀名字的对调命名。
- [内核 sched_ext 声明修复提交](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/patch/?id=56bbc91219c0) — 被静默丢掉的参数，以及双层后缀的绕行办法。
- [sched_ext 的 kfunc 定义](https://github.com/torvalds/linux/blob/master/kernel/sched/ext/ext.c) — 两个家族成员都已注册、`___v2` 名字，以及标记了待删除的 wrapper。
- [bpf_ksym_exists 的定义](https://github.com/torvalds/linux/blob/master/tools/lib/bpf/bpf_helpers.h) — 对弱符号的要求，以及校验器能静态判定的 `!!sym` 形态。
- [内核校验器的调用指令检查](https://github.com/torvalds/linux/blob/master/kernel/bpf/verifier.c) — 伪 kfunc 调用的编码、指针与零比较的分支判定，以及死代码清理。
- [libbpf 版本头文件](https://github.com/torvalds/linux/blob/master/tools/lib/bpf/libbpf_version.h) — 回退会随 1.9 发布的基线。

## 今天的社区讨论

本次通过获批的只读归档查看了两个选择了归档的 Slack 工作区，共三个白名单频道：24 小时窗口里没有消息，七天窗口里有三条，一条是会议纪要公告，一条是贡献者自我介绍，还有一条昨天已经答过。因此本页的问题来自公开的 bpf 邮件列表，而不是今天的聊天窗口。邮件列表是通过它普通的公开归档页读取的。两个白名单 Discord 工作区（合计十个频道）和 r/eBPF 本次没有查看：它们都是可见浏览器界面，而这次没有打开浏览器会话，所以它们的缺席是覆盖缺口，不是安静。

**上面这个问题，从另一侧看。** 列表上的这个线程谈的不只是加载失败。评审者追问的是回退该如何规定：为什么首先需要一个构建期拒绝有歧义家族的措施、原型完全相同的别名该不该允许、带后缀的名字到底是加载期回退还是语义 wrapper。作者的回答是：去重之后对每个家族逐一做两两检查，就能让歧义绑定不可能出现；而别名之所以必要，恰恰是因为某个命名空间已经发布了一个同时声明普通名和带后缀名的 compat 宏。仍未解决的问题是别名带来的义务：两个兼容的函数必须保持语义相同，而除了评审之外没有任何东西强制它。

**硬件接收卸载与已挂载的 XDP。** 一个较新的系列把「XDP 排除硬件 GRO 与 LRO」这条规则从各驱动搬到网络核心，因为驱动今天的做法不一致：有的拒绝挂载，有的在 `ndo_fix_features()` 里清掉特性，至少有一个在 XDP 之下照旧开着硬件合并。它报告出来的失败模式值得每个调试接收路径的人记住：驱动在自己的 fixup 里把硬件 GRO 又加回去，于是核心的警告被触发；挂载仍然成功，XDP 看到的是已合并的帧；而且这个特性位在卸载之后会永久丢掉——在某个驱动上这还会带走 header-data split，并连带带走零拷贝接收路径。诊断方法是：在挂载和卸载前后读设备的 wanted 与当前特性位，而不是相信接口自报的状态。

**调用栈里重复的调用者帧。** 在 arm64 上，由调用者寄存器构造的内核调用链顶部会把调用者报两次，因为寄存器集把返回地址放在 PC 槽里，而帧记录本身已经指向同一个返回地址。它影响 tracepoint 程序、调度器软件事件，以及 tracepoint、raw tracepoint、启用 BTF 的 tracepoint 和 fentry 程序里的栈 helper，而且在一段很短的 tracepoint 栈里就能直接看到。x86 用一个标志位区分这两种寄存器集；arm64 的修法是给它们标记一个内核永远不会返回到的线程模式 PSTATE 值，并跳过对应的叶子 PC，这同时改变了 perf-event 程序在保存寄存器里看到的内容。在修复落地之前，用这类栈去数帧数、或把两个样本折叠成一个，都是错的。

**接收元数据的读接口与写接口。** 一个大型系列让接收元数据从描述符一直存活到 skb 出现，评审争论的是形态而不是必要性：读的一侧必须走各驱动自己的回调，因为每个驱动都按自己的描述符格式保存元数据；而写的一侧永远作用于通用的 frame 或 skb。评审者的反对意见是「写 API 必须持有信息的一份独立副本」这一点并未被证明，而不是认为数据没必要。在它定下来之前，把这条元数据通路当作仍在变动的 API。