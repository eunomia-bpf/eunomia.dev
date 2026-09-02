# 为什么 eBPF 程序不能读取 `bpf_tail_call()` 的返回值？

**简短回答：** BPF tail call 是控制流转移，不是带有可用返回值的普通 helper 调用。成功时，执行会跳到选中的程序，不再回到调用者；失败时，执行会落到下一条指令，但内核 verifier 明确把这个 helper 建模为 `RET_VOID`。因此调用后的 `R0` 不可读，尝试保存或打印所谓返回值的程序会被 `R0 !read_ok` 之类的错误拒绝。

实际规则很简单：把 `bpf_tail_call()` 当作一条独立语句调用。紧接其后的代码就是失败路径，不要接收它表面上的 C 返回值。

## 为什么 C 声明会误导人

很多 BPF helper 通过类似函数指针的 C 接口暴露。根据所用 header 和生成文档，`bpf_tail_call()` 可能看起来返回 `long`，较旧的 helper 说明甚至会写成成功返回零、失败返回负数。但这些表面语法不能决定 verifier 允许 BPF 程序读取什么。

当前 Linux 源码中的权威 verifier 原型把 `bpf_tail_call_proto.ret_type` 设为 [`RET_VOID`](https://github.com/torvalds/linux/blob/master/kernel/bpf/core.c#L3037-L3047)。`RET_VOID` 表示该调用不会在 `R0` 中建立可读结果。libbpf 的常量槽位封装 [`bpf_tail_call_static()`](https://github.com/torvalds/linux/blob/master/tools/lib/bpf/bpf_helpers.h#L133-L162) 也反映了这一点：它虽然会发出 helper call 12，但 C 返回类型是 `void`。

这正好解释了下面的 verifier 序列：

```text
call bpf_tail_call#12
...
w3 = w0
R0 !read_ok
```

调用本身合法，后续复制 `R0` 的动作不合法。普通 BPF 调用约定确实使用 `R0` 承载函数结果，但前提是被调用操作真的定义了结果；[BPF ABI 指南](https://docs.kernel.org/bpf/standardization/abi.html)不会覆盖具体 helper 的 verifier 原型。

## 成功和失败时分别发生什么

解释器清楚地展示了两条路径。它先检查下标越界、tail-call 预算耗尽和程序数组槽位为空。任何一项失败都会跳到 `out`，从调用者的下一条指令继续执行。如果槽位有效，解释器就把当前指令指针替换为目标程序并从那里继续分派。整个过程不会把错误码写入 `R0`，见 [`JMP_TAIL_CALL` 的实现](https://github.com/torvalds/linux/blob/master/kernel/bpf/core.c#L2057-L2082)。

x86-64 JIT 实现的是同一套契约：检查下标、链预算和目标指针；检查失败时跳到 fall-through 标签，成功时直接跳入目标程序。源码在 [`emit_bpf_tail_call_indirect()`](https://github.com/torvalds/linux/blob/master/arch/x86/net/bpf_jit_comp.c#L701-L715) 前直接写出了这段伪代码。

因此不可能存在有意义的“成功返回值”：成功路径根本不会回来。失败码同样不能由 BPF 字节码可移植地读取，因为失败路径没有定义 `R0`。如果读取底层本机寄存器中碰巧残留的内容，结果会依赖架构且不安全，所以 verifier 会阻止这种用法。

目标程序最终执行 `EXIT` 时产生的结果，会成为当前程序链的最终结果；它不会返回给发起 tail call 的程序。

## verifier 可接受的写法

应把 tail call 和 fallback 写成控制流：

```c
SEC("xdp")
int dispatch(struct xdp_md *ctx)
{
    __u32 slot = choose_slot(ctx);

    bpf_tail_call(ctx, &programs, slot);

    /* 只有 tail call 未转移控制流时才会执行。 */
    count_tail_call_fallthrough(slot);
    return XDP_ABORTED;
}
```

不要这样写：

```c
long err = bpf_tail_call(ctx, &programs, slot);
bpf_printk("tail call returned %ld", err);
```

把结果转换成其他整数类型，或先保存再打印，都没有帮助，因为这些写法仍然要求 verifier 读取未定义的 `R0`。把源表达式转为 `void`，或直接忽略表达式结果即可。如果槽位是编译期常量且工具链支持，使用 `bpf_tail_call_static()` 可以更清楚地表达 `void` 契约，还可能帮助 JIT 优化。

## 没有返回码时怎样诊断失败

把 fall-through 视为一种可观测结果，再到拥有相应状态的层次验证原因：

1. 在 per-CPU map 中统计 fall-through，并只按小而有界的分派类别或槽位标记。这样能测量失败，而不依赖未定义寄存器。
2. 调用前证明下标落在程序数组声明范围内。如果下标来自数据包或任务数据，应显式限制或拒绝越界值。
3. 在用户态确认每个预期程序数组槽位都装有兼容的已加载程序，并在更新或重载后复查。不能从 BPF 调用者的“返回值”推断槽位是否存在。
4. 从设计上限制调用链深度。循环或意外过长的分派链最终会在耗尽内核 tail-call 预算时落回调用者。
5. 分别测试一个已知有效槽位、一个空槽位、一个越界下标和一条故意过长的链，同时断言最终程序返回值与调用者的 fall-through 计数。

调用者无法通过 `bpf_tail_call()` 本身区分槽位为空还是调用链预算耗尽。如果运维上必须区分，应由用户态记录配置状态，并暴露独立且不含敏感信息的健康指标。不要随手增加第二张可变 map 作为“镜像”，除非两张 map 的更新已有明确的一致性协议。

## 文档和工具应该怎样描述它

更准确的用户接口契约是：“成功时，控制流转移到选中的程序且不会返回；失败时，从下一条指令继续执行；BPF 程序拿不到返回值。”一个看起来返回整数的 C 声明，不应被解释成可以读取 `R0`。

这不只是措辞偏好。verifier 接受规则、解释器行为和不同架构的 JIT 行为必须一致。`RET_VOID` 可以防止源代码依赖执行引擎刻意没有定义的寄存器内容。文档生成器和库封装应表达这种控制流契约；测试也应编译一个仅用于失败路径的 continuation，而不是断言某个数值 helper 结果。

## 参考资料

- [Linux 内核：`bpf_tail_call` 的 verifier 原型](https://github.com/torvalds/linux/blob/master/kernel/bpf/core.c#L3037-L3047)
- [Linux 内核：解释器中的 `JMP_TAIL_CALL` 实现](https://github.com/torvalds/linux/blob/master/kernel/bpf/core.c#L2057-L2082)
- [Linux 内核：x86-64 JIT 的 tail-call 检查与控制转移](https://github.com/torvalds/linux/blob/master/arch/x86/net/bpf_jit_comp.c#L701-L715)
- [libbpf：返回 `void` 的常量槽位 `bpf_tail_call_static()` 封装](https://github.com/torvalds/linux/blob/master/tools/lib/bpf/bpf_helpers.h#L133-L162)
- [Linux 内核文档：BPF 寄存器与调用约定](https://docs.kernel.org/bpf/standardization/abi.html)
- [Linux 内核 selftest：程序数组与 tail call 测试构造](https://github.com/torvalds/linux/blob/master/tools/testing/selftests/bpf/test_verifier.c)

## 当日社区讨论

本次通过普通可见浏览器检查了全部 6 个批准社区和 15 个白名单频道或公开页面，所有目标均可访问。选题出现在严格的过去 24 小时窗口内，因此没有使用七天回退。以下综合分析已删除参与者和频道身份、消息链接、精确时间、私有拓扑、原始日志以及可回搜的原文措辞；没有保存原始 transcript，也没有执行回复、表情互动、私信、关注、邀请或管理操作。

### Tail call 控制流暴露了文档陷阱

当天最明确的排障讨论起于 helper call 12 之后的 verifier 拒绝。关键线索是程序尝试把 `R0` 移到另一个参数寄存器。讨论随后对照了用户可见的 helper 声明、verifier 原型和执行路径，并确认了前文解释的不一致：成功路径不会返回，而失败后的 fall-through 不会定义可读结果。正确修复是设置仅在失败路径执行的 continuation，并使用独立健康计数；强制类型转换或更换整数类型都不能解决问题。

### 内核审查聚焦失败状态的正确性

公开内核开发归档当天活跃主题包括 verifier 回溯、socket 引用生命周期、BPF LSM attach、终止型控制流指令、batch map 的溢出处理，以及涉及 socket context 字段的编译器回归。虽然这些改动分属不同子系统，共同问题都是异常路径能否保持所有权与寄存器状态不变量。这里也遵循同一原则：fall-through 边确实存在，但不能凭空制造执行引擎从未定义的数值。

### 实践者同时关心开销与语义价值

一个公开实践者论坛在窗口内出现了性能测量文章和进程行为重建项目。两类讨论体现了可观测性工具的互补要求：一方面要量化指令与数据路径开销，另一方面要证明低层事件能被关联成有用且可复现的解释。对 tail-call 分派器而言，更合适的输出是有界 fall-through 计数和配置状态，而不是误导性的逐次错误码。

### 插桩工作强调迁移与可审查性

当天活跃的可观测性讨论主要是实现协调，包括迁移库插桩、拆分相互依赖的改动，以及判断哪些任务已经可供审查。那里没有出现更强的新用户问题。运维上的共同经验是显式表达依赖与就绪状态；对应到 BPF 程序数组，就是在把分派器判定为健康之前，验证槽位已填充且程序类型兼容。

### 安静目标也完成了检查

项目帮助与功能区在窗口内为空、只有自动通知，或没有新讨论。调度器支持区没有新的 24 小时问题，eBPF 插桩区在该窗口内也没有新增交流。这些是已访问但安静的结果，不是把覆盖缺口伪装成零活动。
