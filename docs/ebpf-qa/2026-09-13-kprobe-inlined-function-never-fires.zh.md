# 为什么函数被编译器内联后，给它的 kprobe 永远不会触发？

**简短回答：** 基于符号的内核探针只能附着到在附着那一刻仍然作为独立符号存在的函数上。当编译器把一个 static 或小函数内联掉——或者代码里写了 `__always_inline`——这个符号就会从 kallsyms 中消失，也不再带有 ftrace 的 `fentry`/`mcount` 调用点，于是附着在注册阶段就失败了，而不是静默地挂在那里不触发。对 kprobe event，写入 `kprobe_events` 会被拒绝，诊断信息是 `Invalid probed address or symbol`；对 BPF 的 `fentry`/`fexit` 或 `kprobe.multi`，则根本没有 ftrace 调用点可绑定。解决办法不是重试同一个探针，而是强制生成一份非内联副本（`noinline`、GCC 的 `-fno-inline`、`-fkeep-inline-functions`），改成探测编译器保留下来的调用者或包装函数，或者按一个具体地址来探测。这和 kprobe 黑名单是两种不同的失败：黑名单里符号存在，只是 kprobes 拒绝在其上断点。

## 两张表决定可否附着

一个内核函数可探测，当且仅当它出现在附着路径会查询的某张表里。内联会同时把它踢出两张表。

第一张表是 **kallsyms**。kprobe event 在内核 tracing 代码里通过 `kallsyms_lookup_name()` 解析符号；如果查不到地址，符号校验会在真正尝试注册之前直接返回 `-ENOENT`。随后对 `kprobe_events` 文件的写入失败，trace-probe 日志把原因记作 `Invalid probed address or symbol`。这套解析本身要依赖内核开启 `CONFIG_KALLSYMS`（通常还包括 `CONFIG_KALLSYMS_ALL`）。

第二张表是 **ftrace 的可插桩函数集合**。ftrace 的每个点就是每个函数的调用点——“被跟踪函数的指令指针（也就是函数内 fentry 或 mcount 所在的位置）”。tracefs 中的 `available_filter_functions` 列出了“ftrace 已经处理并且可以跟踪的函数”。`fprobe`（BPF `kprobe.multi` 的底层）是“一种基于 ftrace 的函数图跟踪特性实现的功能入口/出口探针”，而 BPF `fentry`/`fexit` 的 trampoline 会用 `ftrace_location()` 解析目标；如果它返回空，trampoline 就没有东西可绑定。被编译器内联掉的函数，在调用者里没有这样的调用点。

BTF 不会创造附着点。`BTF_KIND_FUNC`“表示一个已定义的子程序”——它描述并标注已定义的子程序类型，并不会把内联后的代码重新变回一个独立、可插桩的函数实例。于是探针本可瞄准的三样东西——kallsyms 符号、ftrace 调用点、BTF 函数条目——会随着那份非内联副本一起消失。

有一个分支不是失败：**模块限定**的符号如果模块尚未加载，会被延迟处理而不是拒绝。内核会告警 “This probe might be able to register after target module is loaded”，并保留该 event，等模块加载后再重试。这和内核核心符号根本不存在是两回事。

## 如何判断自己处在哪种情况

```sh
# 1. 符号到底存不存在？（能否通过 kallsyms 被 kprobe 探测）
grep -w my_func /proc/kallsyms

# 2. 它是否在 ftrace 可插桩集合里？（能否被 fentry/fprobe 探测）
grep -w my_func /sys/kernel/tracing/available_filter_functions
grep -w my_func /sys/kernel/tracing/available_filter_functions_addrs

# 3. kprobe event 附着测试：符号缺失会拒绝写入
echo 'p:my_probe my_func' > /sys/kernel/tracing/kprobe_events
dmesg | grep -E 'Invalid probed address|module is loaded'

# 4. 有没有 BTF 条目？只被内联的函数没有 FUNC 条目
bpftool btf dump file /sys/kernel/btf/vmlinux | grep -w my_func

# 5. 证明发生了内联：目标文件里没有独立函数体
objdump -d vmlinux | grep -A4 my_func

# 6. perf probe 默认会枚举 DWARF 中的内联实例（需要 debuginfo）
perf probe -v --add='my_func' -k /path/to/vmlinux
```

如果第 1 步和第 2 步都查不到，而第 5 步显示函数体只出现在另一个函数的反汇编里，那说明它被内联了，根本没有独立的探测目标。

## 强制生成可探测的副本

解法在编译器一侧，而不是换一种探针：

- 给函数加 `noinline`。clang 属性参考指出该属性“在函数的调用点抑制对该函数的内联”；GCC 文档也记录了同样的按函数豁免。
- 用 GCC 的 `-fno-inline` 构建，它“除了标记为 `always_inline` 的函数外，不对任何函数做内联展开”。
- 用 GCC 的 `-fkeep-inline-functions` 保留非内联副本，它会把声明为 `inline` 的 static 函数“即便已经内联进所有调用者”也输出到目标文件（clang 没有 `-fno-inline`；对应形式是 `-fno-inline-functions` / `-fkeep-inline-functions`）。
- 或者改变探测目标：探测一个没有被内联的调用者或包装函数，探测内联代码真正落在调用者中的那个地址，或让 `perf probe` 从 DWARF 里解析内联实例，其中 `-N`/`--no-inlines` 会把搜索限定在非内联函数上。

## 决定性的局限

你无法把基于符号的探针附着到一个编译器没有作为独立符号生成的函数上，也不能靠“探针应该会一直不触发”在创建时判断——实际行为取决于路径。kprobe event 的写入会被 `Invalid probed address or symbol` 拒绝；BPF `fentry`/`fexit`/`kprobe.multi` 的附着会因为找不到 ftrace 调用点而失败；模块限定的符号则相反，会先成功并等待。不要把它和 kprobe 黑名单混淆：黑名单里符号存在，只是 kprobes 拒绝在其上断点（带 `__kprobes`/`nokprobe_inline` 注解或被 `NOKPROBE_SYMBOL` 标记的函数）；内联意味着符号不存在，黑名单意味着符号存在但被禁止。先确认符号在 kallsyms 和 `available_filter_functions` 中是否存在，再决定是重新用 `noinline` 编译，还是把探针移到调用者。

## 参考资料

- [Linux 内核文档：Kernel Probes (Kprobes)](https://docs.kernel.org/trace/kprobes.html)
- [Linux 内核文档：Kprobe-based Event Tracing](https://docs.kernel.org/trace/kprobetrace.html)
- [Linux 内核文档：ftrace — Function Tracer](https://docs.kernel.org/trace/ftrace.html)
- [Linux 内核文档：Using ftrace to hook to functions](https://docs.kernel.org/trace/ftrace-uses.html)
- [Linux 内核文档：Fprobe-based Event Tracing](https://docs.kernel.org/trace/fprobetrace.html)
- [Linux 内核文档：BPF Type Format (BTF)](https://docs.kernel.org/bpf/btf.html)
- [Linux 内核源码：`kernel/trace/trace_kprobe.c`](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/trace/trace_kprobe.c)
- [Linux 内核源码：`kernel/trace/trace_probe.h`](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/trace/trace_probe.h)
- [Linux 内核源码：`kernel/bpf/trampoline.c`](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/trampoline.c)
- [perf-probe(1)](https://man7.org/linux/man-pages/man1/perf-probe.1.html)
- [GCC：Options That Control Optimization](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html)
- [Clang：Attributes in Clang](https://clang.llvm.org/docs/AttributeReference.html)

## 当日社区讨论

本监控窗口不具备技术性。过去 24 小时内，两个选择加入的 Slack 归档只返回了一小段回退内容，全部是会议事务——一条周期性排期通知、一句共同演讲致谢、一个会议链接——没有任何 eBPF 问题、症状或设计争议。两个白名单中的聊天工作区本次运行没有可见的浏览器会话，公开邮件列表与论坛归档也未审阅。这些来源按“不可用覆盖”如实记录，而不是“安静”。由于可读归档没有产出问题，本篇问答以公开一手资料为依据，而非某条社区消息：这是一个反复出现的实践者问题——动态探针附着了错误的目标，或者看似附着却永不触发，原因是编译器把函数内联掉了——并针对内核 tracing 文档、上游 kprobe 与 trampoline 源码、以及 `perf probe` 与编译器选项参考做了核对。此处不复制任何私有文本、身份、频道或链接。
