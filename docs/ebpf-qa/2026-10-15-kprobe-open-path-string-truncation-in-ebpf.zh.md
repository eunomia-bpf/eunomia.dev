# 为什么 eBPF kprobe 的字符串读取会截断较长的 open 与 stat 路径，又如何读取完整路径？

因为字符串读取的目标缓冲必须放在 BPF 程序 512 字节的栈上，一次 `bpf_probe_read_user_str` 调用最多只能取到用户态字符串的 511 个字符，而内核允许一条完整路径达到 `PATH_MAX`，即 4096。open 与 stat 交给程序的是一根指向用户态路径的指针，合法情况下它可以长到几千个字符，所以任何读进 `sizeof(buf)` 字节栈缓冲的读取都会丢弃 511 个字符之后的部分，并用 NUL 补齐尾部。要读取完整路径，得把字符串搬出栈：在较新的内核上，这些读取助手的动态指针（dynamic-pointer）变体，以及可睡眠程序里的可睡眠 copy-from-user 字符串助手，都能写到 512 字节上限之外。或者，当路径较短时，把栈缓冲设到 511，并从返回值里检测是否被截断。

## 机制

三个限制对齐在一起：

1. **系统调用侧。** `open`、`openat`、`stat` 等接收指向 NUL 结尾路径的用户态指针。`openat` 的文件名参数落在 `kprobe/sys_openat` 的 `ctx->di`，而 UAPI 助手文档展示的正是这种形状：`bpf_probe_read_user_str` 助手把文件名指针从 `ctx->di` 读进栈上的 `char buf[PATHLEN]`（`PATHLEN` 定义为 256）。它指向的字符串可以是一条完整的绝对路径，直到 `PATH_MAX`，即 4096 个字符。
2. **助手契约。** `bpf_probe_read_user_str` 把一条 NUL 结尾字符串从不安全的用户地址拷入 `dst`，其中 `size` 包括结尾的 NUL 字节。若字符串长于 `size`，只拷贝 `size - 1` 个字节，并把最后一个字节置为 NUL；成功时返回写入的字节数（含结尾 NUL）。`bpf_probe_read_kernel_str` 对内核地址有相同语义。
3. **BPF 栈上限。** 目标 `buf` 是 BPF 程序的局部变量，住在 BPF 栈上，而验证器把栈深限制在 `MAX_BPF_STACK`，即 512 字节。你无法声明一个 4096 字节的栈缓冲。因此栈缓冲字符串读取能传的最大 `size` 是 512，一次调用最多取到 511 个字符加 NUL。

所以边界是：栈缓冲的字符串读取被硬性封顶在 511 个可用字符，而 `PATH_MAX` 是 4096。任何超过 511 字符的真实路径（深层嵌套的挂载路径、经 `/proc/self/cwd` 解析的符号链接、很长的 CWD 相对路径）都会在每次栈缓冲读取中被截断到前 511 个字符。

## 验证与调试路径

1. **在助手层面复现。** 在 `sys_openat` 上 attach 一个 kprobe，用 `bpf_probe_read_user_str` 把 `ctx->di` 读进 `char buf[512]`，再比较返回的长度 `res` 与 `sizeof(buf)`。
2. **从返回值检测截断。** 成功时助手返回写入的字节数（含结尾 NUL）。若字符串放得下，`res` 是 `len + 1`，严格小于 `sizeof(buf)`；若放不下，`res` 等于 `sizeof(buf)`（整 512 字节：511 个字符加 NUL）。因此 `res == sizeof(buf)` 就是截断信号。
3. **理解上限处的歧义。** 该信号在 `len >= 511` 时触发。恰好 511 字符的路径会填满整个缓冲并同样返回 `sizeof(buf)`，所以单看返回值无法区分"在 511 处完整捕获"与"超过 511 被截断"。若需要在上限处取得精确长度，应改用非栈目标而不是栈缓冲。
4. **与外部追踪器交叉核对。** 并行运行 `strace -e trace=openat` 或一条 `perf trace` 的 openat sysenter；两者都会打印内核经 `getname` 记录的完整路径。当你的 eBPF 路径更短、且在半截处结束，那就是 512 字节上限，而非探针本身的 bug。

## 局限

- **511 是栈缓冲读取的硬墙。** `MAX_BPF_STACK` 是 512 字节，验证器会拒绝超限的程序，所以你不能简单地把缓冲做大越过上限。`char buf[512]` 已是极限。
- **`res == sizeof(buf)` 混淆了两种情形。** 它对恰好 511 字符的字符串（完整捕获）与更长的字符串（被截断）都成立，助手并不区分二者。
- **逃逸通道依赖内核版本。** 动态指针的 str 变体与可睡眠的 `bpf_copy_from_user_str` 助手是较新的 kfunc，需要带 BTF 的内核，且动态指针 str 变体还挂在 events 特性开关之后；只暴露经典栈缓冲助手的旧内核上没有它们。
- **`bpf_probe_read_user_str` 不适用于别的任务的路径。** 它读的是当前任务的用户内存。要读另一个任务的用户路径，需要跨任务的 str 变体 `bpf_copy_from_user_task_str` 及其动态指针形式。

## 参考

- [kernel include/uapi/linux/bpf.h](https://github.com/torvalds/linux/blob/master/include/uapi/linux/bpf.h) —— `bpf_probe_read_user_str` / `bpf_probe_read_kernel_str` 契约：`size - 1` 拷贝上限、置 NUL、按写入字节数返回，以及 `kprobe/sys_open` 示例。
- [kernel include/linux/filter.h](https://github.com/torvalds/linux/blob/master/include/linux/filter.h) —— 验证器强加在 BPF 程序栈上的 `MAX_BPF_STACK` 512 字节上限。
- [kernel kernel/bpf/verifier.c](https://github.com/torvalds/linux/blob/master/kernel/bpf/verifier.c) —— 验证器强制 512 字节 BPF 栈深限制之处。
- [kernel include/uapi/linux/limits.h](https://github.com/torvalds/linux/blob/master/include/uapi/linux/limits.h) —— `PATH_MAX` 4096 与 `NAME_MAX` 255，内核允许的完整路径长度。
- [kernel kernel/bpf/helpers.c](https://github.com/torvalds/linux/blob/master/kernel/bpf/helpers.c) —— 可睡眠的 `bpf_copy_from_user_str` / `bpf_copy_from_user_task_str` kfunc，以及动态指针 str 助手声明。

## 当日社区讨论

本 run 审查的两个 opt-in 存档渠道（两个 CNCF OpenTelemetry 插桩渠道）共产出十一条消息。窗口内的三个话题都已被回答或并非问题：Cilium eBPF 讨论的交叉转发是 10-13 已发布条目，FUSE 与 OBI trace 传播线程是 10-14 已发布条目，GenAI semantic-conventions 的 PR 备注是评审状态更新而非问题。因此本页没有依托窗口内的某个线程，而是立足于公开的一手资料：它回答的是一个反复出现、可用公开资料核对的 eBPF 机制问题——open 与 stat 探针处被栈上限封顶的字符串读取，这一主题此前没有任何已发布条目覆盖。

本 run 的渠道覆盖：两个 opt-in 存档渠道共提供十一条消息。可见浏览器专属的渠道——eunomia-bpf 与 sched-ext 的 Discord 服务器、bpf 邮件列表、以及 r/eBPF——本 run 没有可用的可见浏览器会话，因此标记为"未覆盖"，而非"安静"。
