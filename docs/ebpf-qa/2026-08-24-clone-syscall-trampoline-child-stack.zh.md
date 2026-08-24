# 为什么 syscall rewriting trampoline 会让 `clone` 或 `clone3` 创建的线程崩溃？

**简短回答：** x86-64 tracer 可能把普通 `syscall` 改写成一条类似函数调用、最终以 `syscall; ret` 结束的路径。对于一般的单次返回系统调用，这通常能够工作，因为进入 trampoline 的 `call` 已经在当前栈上压入 continuation address。`clone` 和 `clone3` 则不同：系统调用会分别在 parent 和 child 返回，而线程 child 会使用 caller 指定的新栈继续执行。Child 到达 trampoline 的 `ret` 时，parent 旧栈上由 `call` 压入的返回地址并不存在于新栈。于是 `ret` 会把 child 新栈的第一个 word 当成 instruction pointer，常见结果是在 thread start routine 运行前出现一个看似无关的 `SIGSEGV`。

问题不是 Linux 返回到了错误的指令，而是 rewriter 引入了原始 `syscall` 不具备的、依赖栈的 return contract。

## 原始指令不使用 userspace stack

在 x86-64 上，硬件 `SYSCALL` entry 把 userspace return instruction pointer 保存到 `rcx`，把 flags 保存到 `r11`。Linux entry code 明确说明：该指令不会在 userspace stack 上保存任何内容，也不会改变 `rsp`。

原始 `syscall` 只有两个字节，instruction rewriter 可能需要更多空间，因此一种设计是把它替换成 indirect `call`，跳到附近或特殊映射的 stub。Stub 运行 hook，最后执行真正的 system call：

```text
application
    call trampoline        # 在当前栈压入 application continuation

trampoline
    ... run hook ...
    syscall
    ret                    # 从当前栈弹出 continuation
```

对于 `read`、`write` 等普通系统调用，控制流只回到发起调用的线程。它的 `rsp` 仍指向 `call` 压入 continuation 的那一个栈，因此末尾 `ret` 可以正常工作。

`clone` 和 `clone3` 打破了这个假设。Raw interface 会在两个 execution context 中返回：parent 得到 child ID，child 得到零。Kernel 复制 parent 保存的 register image，把 child 的返回寄存器设成零；如果 caller 提供了新栈，则把 child stack pointer 替换成这个新栈。Instruction pointer 仍然从 trampoline 内 `syscall` 之后继续。

因此，两条控制路径并不对称：

- **Parent path：** 下一条指令是 trampoline 的 `ret`，而 `rsp` 仍指向含有已压入 continuation 的原始栈，因此可以正常返回。
- **使用新栈的 child：** 下一条指令仍是同一个 `ret`，但 `rsp` 已指向 fresh child stack。那里没有 trampoline continuation，所以 `ret` 会弹出无关数据并跳转。
- **使用 null、fork-like stack 的 child：** child 保留 parent stack layout 的副本，因此可能保留预期 continuation。

Intel 对 near `RET` 的定义是：从栈顶弹出下一个 instruction pointer。处理器并不知道程序员希望它与哪一条 `call` 配对。如果栈顶 word 是零、data pointer、canary 或其他非代码值，fault address 就可能与 tracer 完全无关。因此，表面症状常被误认为 libc 或应用自身崩溃，而不是 syscall rewriting bug。

## 为什么 `pthread_create` 是高价值 reproducer

glibc 当前的 `pthread_create` 路径会准备带有 thread-sharing flags、显式 stack address 和 size、TLS 及 parent/child TID locations 的 `clone_args`，再调用内部 clone wrapper。因此，一个最小 `pthread_create`/`pthread_join` 程序，比只循环调用 `getpid`、`read` 或 `write` 的单线程程序更适合作为 regression test。

有效 reproducer 应分别证明：

1. 同一个 binary 在没有 instruction rewriting 时能启动并 join worker。
2. Rewriting mode 确实处于 active 状态；stale preload path 或 transformer 未加载必须让测试失败，而不能静默产生绿色结果。
3. Parent 看到了 thread creation success。
4. Worker 到达自己的第一条指令，并改变一个可观察 flag。
5. Parent 成功 join worker，process 正常退出。

第四项非常关键。Parent 中的 `pthread_create` 成功返回，并不能证明 child 活过了 return path 或进入 user start routine。

## 诊断 continuation，而不只是 fault address

先用最小 threaded program 对比 rewriting off 和 on。若只有 rewritten case 失败，应检查 instruction boundary 的路径，而不是直接把 faulting symbol 当成原因。

在 x86-64 上，应在真正 system call 返回后，分别采集 parent 和 child 的这些状态：

- system call number 与 return value；
- `rip`，确认两个 context 都从 trampoline 内同一条 `syscall` 之后恢复；
- `rsp`，确认 child 是否切换到显式指定的新栈；
- `ret` 前两个栈顶的 word；
- 原始 rewriting `call` 压入的预期 application continuation。

同时反汇编 patched application site 和 trampoline，确认 site 确实经过预期 stub，而且正在调试的就是实际 fallback path。Instruction fetch fault 落在 data page 与 bad `ret` 一致，但仅凭这一点还不够；stack 与 continuation 的对照才真正建立因果机制。

还要区分 glibc wrapper 与 raw interface。glibc `clone()` wrapper 会安排 child 调用给定函数；raw `clone` system call 和 `clone3` 则在 parent 与 child 中都从 system call 返回。Rewriter 位于 library abstraction 之下，必须保持 library 所依赖的 raw architectural behavior。

## 修复选项

最安全的设计规则是：除非 trampoline 为每个返回 context 显式构造有效 continuation，否则不要把一个 two-return system call 建模成普通 one-return function。

实践中有两类修复。

### 1. 让特殊 system call 绕过 hook

把 `clone` 和 `clone3` dispatch 到一条小型 untraced path，执行真正的 system call，但不进入 C hook。这样可以避免在尚未完整启动的 child context 中运行复杂 hook code；不过，它本身不能自动修复末尾的 `ret`。如果 child 使用新栈，untraced path 仍必须提供有效 continuation，或者改用不依赖栈的 transfer。

这一 trade-off 应在产品行为和测试中明确说明。如果 creation calls 绕过 hook，tracing output 就不能声称完整 syscall coverage。

### 2. 在 child stack 中预置 continuation

对于使用非 null child stack 的 x86-64 raw `clone`，可以在请求的 stack pointer 下方保留一个 word，写入 application continuation，再把调整后的 pointer 交给 kernel。Child 的 `ret` 弹出该地址，同时 `rsp` 恢复到原本请求的 stack top。

对于 `clone3`，`stack` 指向 child stack 的最低字节，`stack_size` 表示范围。对应修复可以把 continuation 写到 `stack + stack_size - 8`，并在 system call 前把 `stack_size` 减八。当前 bpftime x86-64 transformer 采用这一形状，同时让两种创建调用绕过 C hook。对于 null child stack，它保留 fork-like path，因为此时 parent stack layout 被复制而不是替换。

这个修复不能作为跨 architecture 直接粘贴的通用配方。它依赖：

- architecture 的 syscall ABI 和 syscall number；
- stack growth direction 与 alignment；
- raw `clone` 的精确 argument order；
- `clone3` structure layout 与 size validation；
- 请求的 memory 是否可写；
- userspace shadow stack 等 control-flow protection；
- rewriter 自身关于 register、red zone、vector state、signal 和 unwind 的契约。

特别是，在没有专门测试前，不应声称 regular-stack repair 已覆盖 Intel CET shadow-stack execution。启用 shadow stack 后，普通栈与 shadow return-address stack 必须同时同意，`ret` 才能成功。

另一种 trampoline 可以用 `jmp` 而不是 `ret` 跳到保存的 continuation，但 child 仍需要一个可信位置来取得这个 continuation。只存在于 parent stack 的 pointer，不会因为最后一条指令改成 `jmp` 就自动变得有效。

## Regression tests 必须覆盖两个返回 context

完整 test matrix 不能只检查“process 不再崩溃”：

- `pthread_create` 后接 `pthread_join`，并证明 worker 确实运行；
- 使用显式 child stack 的 raw `clone`；
- flags 允许时，使用 null、fork-like stack 的 raw `clone`；
- 使用显式 stack 和合法最小 size 的 `clone3`；
- invalid stack 与 size input，确认 error behavior 没有改变；
- 大量并发 creation，确认 stack preparation 和 continuation storage 没有被错误共享；
- 每个 child 启动后继续发出真实 syscalls；
- single-threaded victim，确认 common path 没有 regression；
- tracing-enabled 与 tracing-disabled mode，并断言 transformer 确实加载；
- 每个受支持的 architecture 和 control-flow-protection mode，而不是把 x86-64 结果直接外推。

除了 process exit status，还应检查 semantic coverage。如果修复故意 bypass creation calls，就要断言 downstream consumer 能看到并理解这个已记录的 gap。如果 hook 被保留，则要验证 parent/child return value、preserved registers、stack alignment、signal behavior 和 unwindability。

错误 `ret` 会从无关栈内容派生 instruction pointer，因此属于 control-flow integrity defect。当前证据证明的是 crash；exploitability 取决于 memory layout 和 mitigations，不能在没有独立证据时推断。

## 参考资料

- [Linux `clone(2)` 手册：raw return behavior 与 child-stack semantics](https://man7.org/linux/man-pages/man2/clone.2.html)
- [Linux x86 `copy_thread`：child return value 与可选 stack-pointer replacement](https://github.com/torvalds/linux/blob/master/arch/x86/kernel/process.c)
- [Linux x86-64 syscall entry：`SYSCALL` 不在 userspace stack 保存数据，也不改变 `rsp`](https://github.com/torvalds/linux/blob/master/arch/x86/entry/entry_64.S)
- [Intel 64 与 IA-32 的 `RET` 指令参考](https://www.intel.com/content/dam/www/public/us/en/documents/manuals/64-ia-32-architectures-software-developer-vol-2b-manual.pdf)
- [glibc `pthread_create` 实现及其 `clone_args` setup](https://github.com/bminor/glibc/blob/master/nptl/pthread_create.c)
- [当前 bpftime x86-64 syscall transformer 与 `clone`/`clone3` handling](https://github.com/eunomia-bpf/bpftime/blob/master/attach/text_segment_transformer/text_segment_transformer.cpp)
- [Intel Control-flow Enforcement Technology 与 shadow-stack return 概览](https://www.intel.com/content/www/us/en/developer/articles/technical/technical-look-control-flow-enforcement-technology.html)
- [Linux BPF selftests](https://github.com/torvalds/linux/tree/master/tools/testing/selftests/bpf)
- [Linux BPF verifier 实现](https://github.com/torvalds/linux/blob/master/kernel/bpf/verifier.c)
- [Linux AF_XDP socket 实现](https://github.com/torvalds/linux/blob/master/net/xdp/xsk.c)
- [OpenTelemetry GenAI operation-cost proposal](https://github.com/open-telemetry/semantic-conventions-genai/issues/287)
- [OpenTelemetry GenAI event 约定](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-events.md)

## 当日社区讨论

今天通过普通可见浏览器检查了全部 6 个批准社区和 15 个 allowlist 频道或公开页面，所有目标均可访问。选题来自过去 24 小时，因此没有使用七天 fallback。姓名、账号、雇主、workspace 和频道身份、message link、精确时间、私有拓扑、原始日志及可搜索回原讨论的措辞均已删除，也没有保留原始 transcript。

### Runtime instrumentation 必须保持 system call 的 control-flow shape

最强工程问题是：syscall rewriting mode 可以运行单线程程序，却在程序创建线程时失败。诊断关键从“哪个 library address 崩溃”转为“每个返回 context 中究竟存在哪个 continuation”。Linux 的 [x86 child setup](https://github.com/torvalds/linux/blob/master/arch/x86/kernel/process.c)与[syscall entry contract](https://github.com/torvalds/linux/blob/master/arch/x86/entry/entry_64.S)解释了为什么 `call`/`ret` wrapper 会增加 raw `clone` 原本没有的义务。当前 [runtime transformer](https://github.com/eunomia-bpf/bpftime/blob/master/attach/text_segment_transformer/text_segment_transformer.cpp)已为 child 提供 continuation，并让 creation calls 绕过 C hook。

更一般的教训是 coverage evidence。绿色 single-threaded benchmark 可以证明 ordinary return 正常，却无法说明创建第二个 userspace continuation 的 system call 是否正确。Loader success 也不能证明预期 transformer 确实存在。测试必须包含可观察的 child-side action，并显式断言 rewriting 已启用。

### Kernel review 聚焦 failure path、ownership 与 diagnostic precision

公开 kernel surface 当天非常活跃。反复出现的主题包括 private-stack JIT state、link update 的 stack bounds、resizable hash table lifetime、memory limit 下的 arena reclaim、generic XDP 与 AF_XDP refill path 的 page ownership，以及 composite return value 的 verifier diagnostics。它们属于不同 subsystem，却共享一个不变量：error、retry 或 alternate execution path 不能留下 stale ownership 或不可能的 machine state。

实际应对方式是把每项 review concern 变成 [BPF selftests](https://github.com/torvalds/linux/tree/master/tools/testing/selftests/bpf) 中的 failure-injection 或 boundary test。Verifier work 应指出究竟是哪一个 member 或 register state 让 return type 不受支持，而不只是拒绝 program；[verifier implementation](https://github.com/torvalds/linux/blob/master/kernel/bpf/verifier.c)是语义边界。Packet-buffer work 则应通过 retry 与 release test 证明 [AF_XDP socket state](https://github.com/torvalds/linux/blob/master/net/xdp/xsk.c)的 ownership 只转移一次。

### Observability convention 仍在区分 observed data 与 standardized meaning

另一条活跃讨论追问：当 exporter 已经发出 proprietary attributes 时，如何表示 model-operation cost。公开的 [operation-cost proposal](https://github.com/open-telemetry/semantic-conventions-genai/issues/287)仍处于 open 状态，因此 exporter-specific field 不应被呈现为已经发布的 OpenTelemetry convention。Producer 还需要说明数值是 billed、estimated 还是 derived，以及采用何种 currency 或 unit。

关于让 evaluation result 引用外部 evidence 的讨论也仍在继续。合适边界仍然是：标准化 optional reference 与 integrity binding，但不把 telemetry 变成 artifact store，也不暗示证据已经获得信任。在新名称完成 specification process 以前，应继续以[当前 GenAI event convention](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-events.md)为 baseline。

### 安静目标仍然完成了检查

Scheduler help surfaces、公开 practitioner forum 和多数 project-specific support channels 在当日窗口内没有新的实质技术交流。一个 networking surface 主要讨论 meeting 和 contribution onboarding。若干 project channels 只有 introductions、自动维护通知或没有消息。公开论坛的最新帖子讨论 crash-safe BPF loader，但已超出 24 小时窗口，因此没有作为 fallback evidence 使用。
