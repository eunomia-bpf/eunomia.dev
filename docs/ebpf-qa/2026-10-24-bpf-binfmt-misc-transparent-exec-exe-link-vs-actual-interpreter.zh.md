# BPF binfmt-misc 处理器以透明模式执行 exec 时，为什么进程的 exe 链接、comm 和命令行仍然描述的是被执行的二进制，而不是实际运行的解释器？

因为内核刻意把"身份"固定到二进制上。BPF binfmt-misc 处理器通过 `struct linux_binprm` 上的 BPF 选择字段来挑选解释器；当这个选择带有透明标志时，exec 路径把*实际运行的代码*换成了解释器，却把进程身份——`exe` 链接、线程 `comm`、参数向量——继续指向二进制。解释器的选择被暂存在 `bprm->bpf_interp`、`bprm->bpf_interp_file` 和 `bprm->bpf_flags` 里，转换成 `MISC_FMT_*` 位之后就在 exec 路径内部被清零。exec 完成后，没有任何用户可见的接口，也没有任何 tracepoint 记录是哪个 BPF 处理器做的选择、用了哪些标志。

这正是观察盲区之所以是结构性的、而不是某处疏漏的全部原因：这个选择是一个加载期、内核内部的事实，而它被刻意不留在进程上，所以标准的 exec 观测工具天生看不见 BPF 这一侧。

## 机制

1. BPF 处理器是一个 `struct binfmt_misc_ops` struct_ops map，由一个 `'B'` binfmt-misc 条目激活。它的 load 程序是唯一能调用选择 kfunc 的地方——验证器把 `bpf_binprm_set_interp`、`bpf_binprm_select_interp`、`bpf_binprm_set_interp_arg` 和 `bpf_binprm_set_flags` 限制在这个 map 的 load 程序里。选中的解释器暂存在 `bprm->bpf_interp` / `bprm->bpf_interp_file`，标志放在 `bprm->bpf_flags`。

2. `load_misc_binary` 调用 `entry_invocation_flags`，把 BPF 标志翻译成 `MISC_FMT_*` 调用位——透明标志对应 `MISC_FMT_TRANSPARENT` 加 `MISC_FMT_OPEN_BINARY`——然后把 `bprm->bpf_flags` 清零，避免标志累加进嵌套的解释器层级。在透明路径上它跳过 `build_interp_argv`，调用 `bprm_change_interp` 把 `bprm->interp` 设成解释器字符串，并把 `bprm->interpreter` 设成解释器文件、置上 `BINPRM_FLAGS_TRANSPARENT_INTERP`。

3. `exec_binprm` 循环接下来以 `bprm->file = bprm->interpreter` 跑下一轮——解释器成为下一轮的文件。当 `bprm->have_execfd` 被置位时，原始二进制被暂存为 `bprm->executable`，用于交给解释器的 `AT_EXECFD`，循环把解释器深度上限设成 5，超过就返回 `-ELOOP`。

4. 只要置了 `BINPRM_FLAGS_TRANSPARENT_INTERP`，`bprm_identity_file` 就返回 `bprm->executable`——也就是二进制——而 `set_mm_exe_file` 和 comm 的推导都取自它。于是 `/proc/PID/exe` 和 `comm` 呈现的是二进制，`argv` 原样保留，实际运行的代码是通过 `AT_EXECFD` 到达的解释器。按内核自己的注释，透明模式正是那条"为下一轮更新 interp"的路径，让这次 exec"看起来像一次直接执行"。

## 验证与排障路径

1. 直接读 exec 身份：对比 `/proc/PID/exe` 加 `comm`，与 `/proc/PID/maps` 和 `AT_EXECFD` auxv 项。透明模式下，exe 链接、`cmdline`、`comm` 全都指向二进制，而实际运行的代码是解释器。

2. 对比同一次 exec 的两个 sched 事件：`sched_prepare_exec` 带有 `interp` 路径，但依旧没有 handler 名或标志；而 `sched_process_exec`——多数追踪器以它为准——只有 `filename` 加 `pid` 加 `old_pid`。两者对同一次 exec 报告的内容不同，而 BPF handler 名和选择标志在两者里都缺失。

3. 用 BPF 处理器复现：注册 `'B'` handler，建一个 `D` 状态条目，绑定一个解释器，再走一次 exec。二进制的 `exe` 链接保留，argv 保留，实际运行的代码是解释器。

4. 解读观察到的标志值：一次记录为 `0x48000000` 的选择就是 `MISC_FMT_TRANSPARENT` 或 `MISC_FMT_OPEN_BINARY`——`entry_invocation_flags` 里的翻译会附加透明标志自带的那个 `OPEN_BINARY` 含义。

## 局限

- 这个盲区不是彻底的：`sched_prepare_exec` 确实会报告解释器路径。没有任何现成事件或接口暴露的，是 BPF handler 名和选择标志。

- 这个身份行为是刻意的，不是 bug。透明模式的存在就是为了让这次 exec 以二进制的身份呈现，所以 exe 链接和 comm 跟着二进制走，`argv` 被保留。

- RFC 提议的那个默认关闭的 `binfmt_misc_select` tracepoint 是观察，不是强制。它事后报告选择，而 tracefs 访问仅限 root，root 可以把它关掉。

- 一个 BPF binfmt-misc 处理器注册在它自己的用户命名空间里，并不从祖先用户命名空间继承，所以哪个处理器生效取决于这次 exec 所在的那个命名空间。

- Loader 模式是另一套机制：它用 loader 替代二进制的 `PT_INTERP`，原生运行二进制，并排除其它所有标志——它不是透明路径。

## References

- [公开的 bpf 邮件列表 RFC：一个默认关闭的 `binfmt_misc_select` tracepoint](https://lore.kernel.org/bpf/21bBRWi4dDGR7l5TdmShL09exFRFwQA8c-1U_QlNoY2L7LGzDCNfypM9VECtuwD7mkrG1Cw8uXSLLdHmXsYsMARBbGn-mIl3Mutmeku-2zg=@proton.me/) - 一封公开的 bpf 列表 RFC，提议在 BPF 处理器选好解释器之后触发一个默认关闭的 tracepoint，报告 filename、解释器、handler、屏蔽后的标志与 pid，明确定位为观察而非强制。
- [内核 `fs/binfmt_misc_bpf.c`](https://github.com/torvalds/linux/blob/master/fs/binfmt_misc_bpf.c) - 暂存解释器与标志的 BPF struct_ops handler kfuncs。
- [内核 `fs/binfmt_misc.c`](https://github.com/torvalds/linux/blob/master/fs/binfmt_misc.c) - `load_misc_binary`、`entry_invocation_flags` 与 `MISC_FMT_*` 标志位。
- [内核 `fs/exec.c`](https://github.com/torvalds/linux/blob/master/fs/exec.c) - `bprm_identity_file`、`bprm_change_interp` 与 `exec_binprm` 循环。
- [内核 `include/trace/events/sched.h`](https://github.com/torvalds/linux/blob/master/include/trace/events/sched.h) - `sched_process_exec` 与 `sched_prepare_exec` 的字段列表。

## 当日社区讨论

一封公开的 bpf 邮件列表 RFC 提议一个默认关闭的 `binfmt_misc_select` tracepoint，在 BPF 处理器选好解释器之后触发，报告 filename、解释器、handler、屏蔽后的标志与 pid。它被明确定位为对 BPF 这一侧选择的观察——而这一事实，按现状，没有任何现有 tracepoint 或用户可见接口会暴露。该 RFC 在列表上公开、处于讨论中。

本次运行的存档来源在 7 天窗口内产出 0 条消息（两个 opt-in 存档来源在快照时为空），因此上面的问题取自被允许的公开 bpf 邮件列表，经其公开的 atom feed 审阅，RFC 线程作为一手参考。eunomia-bpf 与 sched-ext 的 Discord 频道以及 r/eBPF 在本次运行未覆盖（没有可用的可见浏览器会话），故标记为未覆盖而非安静。
