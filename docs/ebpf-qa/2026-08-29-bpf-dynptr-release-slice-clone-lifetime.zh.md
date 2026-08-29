# 为什么释放 BPF dynptr 时必须让所有派生 slice 和 clone 失效？

**简短回答：**因为 dynptr 是 verifier 跟踪的 backing memory view，不是可以自由复制的 owning buffer。`bpf_dynptr_slice()` 或 `bpf_dynptr_slice_rdwr()` 返回的 slice 仍然 alias 同一块 memory，clone 也只是通过另一个 stack object 表示相同的 underlying lifetime。`bpf_ringbuf_submit_dynptr()`、`bpf_ringbuf_discard_dynptr()` 等 release operation 一旦结束这个 lifetime，所有能够到达的 alias 都必须在同一次 verifier state transition 中变为不可用。

因此，只让传给 release helper 的 dynptr 失效并不够。遗留的 slice 仍可能表现为 verifier 已批准的 `PTR_TO_MEM`，另一个 BPF call frame 里的 clone 也可能仍被视为 live。任一缺口都可能把看似合法的 load、store 或第二次 release 变成 use-after-free。安全规则必须是 transitive 的：撤销 released dynptr、它的全部 clone，以及从这个 lifetime group 任一成员派生的所有 slice。

## Lifetime 属于 backing object，而不是 C variable

以 ring-buffer reservation 为例：

```c
struct bpf_dynptr ptr;
struct event *event;

if (bpf_ringbuf_reserve_dynptr(&events, sizeof(*event), 0, &ptr))
    return 0;

event = bpf_dynptr_slice_rdwr(&ptr, 0, NULL, sizeof(*event));
if (!event) {
    bpf_ringbuf_discard_dynptr(&ptr, 0);
    return 0;
}

event->kind = 1;
bpf_ringbuf_submit_dynptr(&ptr, 0);
/* event 与 ptr 从这里起都已失效。 */
```

在 submit 之前，`event` 不是独立 allocation，而是 reserved record 内部的 alias。Submit 会把 record 发布给 consumer，并结束 BPF program 对 reservation 的 ownership。Discard 虽然会让 consumer 跳过该 record，也同样结束 ownership。如果之后仍允许读写 `event->kind`，program 就能访问 lifetime 已经改变的 memory。

Dynptr clone 也遵循同一逻辑。Clone 提供另一个 view，并有自己的 verifier identity，但不会创建新的 backing object。释放任意一个 referenced dynptr 会结束 shared lifetime，因此 original、所有 clone 与各自的 slice 必须一起失效。

## 一个 logical lifetime 可以有多个 verifier identity

Stable kernel 讨论中的 bug，来自把两类 identity 误当成可以互换。

在受影响的 stable verifier representation 中：

- dynptr stack object 有自己的 ID；
- slice 是携带 source `dynptr_id` 的 `PTR_TO_MEM` register；
- referenced object 还用 `ref_obj_id` 把共享 acquired lifetime 的 value 归为一组；
- clone 可以位于任意 active BPF call frame 的 dynptr stack slot 中。

`release_reference(ref_obj_id)` 可以使关联 referenced lifetime 的 register 失效，但 `bpf_dynptr_slice()` 和 `bpf_dynptr_slice_rdwr()` 返回的 slice 携带 `dynptr_id`，`ref_obj_id` 则保持 unset。只遍历 `ref_obj_id` 的 release path 因而会漏掉这些 slice。相比之下，一些 `bpf_dynptr_data()` path 已携带 reference ID，所以只测试该 API 可能掩盖缺失的 `dynptr_id` invalidation。

Stable fix 会显式遍历 register，找出 `dynptr_id` 与 released dynptr 相同的 `PTR_TO_MEM` value。读取 `dynptr_id` 前先检查 register base type 很重要，因为该 metadata field 与其他 register type 的 metadata 共用 storage；同时又不能用过于严格的 exact type match，否则会漏掉带 dynptr-specific type flag 的 slice。

## Clone invalidation 必须跨越 BPF call frame

当 subprogram 处理 clone，而 original 或另一个 clone 位于不同 frame 时，会出现第二个缺口。只扫描当前 `bpf_func_state` 不够：

```text
caller frame:  original dynptr + derived slice
                    |
                    +---- shared lifetime ----+
                                              |
callee frame:                              clone
                                              |
                                           release
```

Callee 释放 referenced dynptr 后，caller 不能带着 verifier 仍视为有效的 original 或 slice 继续运行。因此 invalidation pass 必须检查所有 active frame 中的 dynptr stack slot，而不只是执行 helper 的 frame。

这个 scan 还需要 structural guard。普通 spilled register 可能被部分覆盖，stack byte 中残留的内容看起来像旧 dynptr metadata。如果把这些 byte 当成 live dynptr，可能错误拒绝有效 program，甚至破坏 verifier bookkeeping。Stable repair 会先确认 slot 确实标记为 `STACK_DYNPTR`，再读取 dynptr field。

## Verifier rejection 必须保持为普通 error

还有一个相关的 control-flow edge case：callback 可能尝试释放属于 caller 的 referenced dynptr。这个 operation 应被拒绝，因为 callback 不能消费不属于自身 lifetime context 的 reference。

Release function 已经可以为这种情况返回 error。把该 error 转成 `WARN_ON_ONCE()` 有两个问题：它会隐藏有用的 verifier rejection；而在 `panic_on_warn=1` 的 system 上，一个 invalid BPF program 甚至可能在 verification 阶段触发 kernel panic。Stable series 因此把 error 传播到正常 verifier failure path。

这个区别在运行时非常重要。Program 被拒绝，说明 untrusted input 得到了预期处理；kernel warning 则表示 internal invariant 失败。Verifier code 不应把可预见的 program error 升级成 invariant failure。

## Mainline 与 stable kernel 可以用不同机制执行同一规则

Mainline commit `308c7a0ae885` 用显式 parent-child tracking 替换了旧 relationship bookkeeping。每个 object 有自己的 identity，`parent_id` 则把 derived object 连接到控制其 lifetime 的 object。释放 reference 时会遍历 object tree 并使所有 descendant 失效。Referenced dynptr 使用 intermediate lifetime anchor，让 clone 共享 release boundary，同时保留各自的 identity 来追踪自己的 slice。

这是一个跨多个文件的大型 verifier refactor。仍使用 `dynptr_id` 与 `ref_obj_id` 的 stable branch 不能只挑其中一部分机械 cherry-pick。当前 stable proposal 改用旧 representation 实现等价 safety property：

1. 通过匹配 `dynptr_id` 使 slice register 失效；
2. 让 referenced-dynptr release error 沿 verifier 正常 error path 返回；
3. 扫描全部 active call frame 中经过确认的 dynptr stack slot，使 clone 失效。

实现机制不同，但 contract 相同：任何 verifier-visible descendant 都不能比授权其访问 backing memory 的 object 活得更久。

## 怎样编写遵守这条边界的 dynptr code

对 BPF program author，应采用 lexical ownership pattern：

1. reserve 或 construct dynptr；
2. 只在 owning block 内派生 slice；
3. 检查每个 slice-returning call 是否返回 `NULL`；
4. 保证每条 path 都只 submit、discard 或 release 一次；
5. release 后不再 read、write、return 或传递任何 dynptr-derived pointer。

除非 API 明确证明 lifetime，否则不要把 slice 存入会跨 callback 或 subprogram boundary 存活的 state。Clone 可以帮助 structured code 传递 dynptr，但它不代表 ownership split：referenced lifetime 被释放时，所有 alias 都会失效。

对 kernel/verifier backport，可以使用以下 regression matrix：

| Case | 预期 verifier 结果 |
| --- | --- |
| Submit 后使用 `bpf_dynptr_slice()` 结果 | 拒绝 |
| Discard 后使用 `bpf_dynptr_slice_rdwr()` 结果 | 拒绝 |
| Callee 释放 clone，caller 随后使用 original | 拒绝 |
| Callback 释放 caller-owned referenced dynptr | 拒绝且不产生 warning |
| 扫描带 stale-looking metadata 的 partially overwritten ordinary spill | 接受有效 control program |
| 在唯一一次 release 之前使用 live dynptr 与 slice | 接受 |

Negative 与 positive control 都不可缺少。Negative test 证明 stale alias 已被关闭；positive test 证明 invalidation walk 没有把 unrelated register 或 stack slot 错当成 descendant。

## 参考资料

- [Linux mainline commit：重构 verifier object relationship 并修复 dynptr use-after-free](https://github.com/torvalds/linux/commit/308c7a0ae8859b34d9d90a3dff953b2d14242145)
- [Linux kernel 文档：BPF ring-buffer reservation、commit、discard 与 verifier reference tracking](https://docs.kernel.org/bpf/ringbuf.html)
- [Linux BPF selftest：dynptr invalidation 与 failure path](https://github.com/torvalds/linux/blob/308c7a0ae8859b34d9d90a3dff953b2d14242145/tools/testing/selftests/bpf/progs/dynptr_fail.c)
- [Mainline lifetime-tracking fix 对应的 Linux verifier source](https://github.com/torvalds/linux/blob/308c7a0ae8859b34d9d90a3dff953b2d14242145/kernel/bpf/verifier.c)
- [Linux 6.12.y verifier source：stable branch 的 dynptr 与 reference representation](https://github.com/gregkh/linux/blob/linux-6.12.y/kernel/bpf/verifier.c)
- [Linux 6.12.y dynptr failure selftest](https://github.com/gregkh/linux/blob/linux-6.12.y/tools/testing/selftests/bpf/progs/dynptr_fail.c)
- [Linux BPF UAPI：dynptr ring-buffer helper definition](https://github.com/torvalds/linux/blob/master/include/uapi/linux/bpf.h)

## 当日社区讨论

今天通过普通可见浏览器检查了全部 6 个批准社区和 15 个 allowlist 频道或公开页面，所有目标均可访问。选题来自严格的过去 24 小时窗口，因此没有使用七天 fallback。姓名、账号、雇主、workspace 与频道身份、message link、精确时间、私有拓扑、原始日志和可搜索回原讨论的措辞均已删除。没有保留原始 transcript，也没有进行任何社交互动。

### Lifetime tracking 是最明确的 correctness concern

最强的技术讨论是 stable-kernel dynptr release fix。Review 不只覆盖明显的 stale slice，还测试了完整 alias closure：read-only 与 writable slice、位于另一个 BPF call frame 的 clone、尝试 invalid release 的 callback，以及带 stale-looking metadata 的 ordinary spill slot。重要的工程结论是：backport 应复现 mainline safety property，而不是机械移植 target branch 中不存在的 object model。

### Instrumentation chat 主要关心 review dependency

当天 instrumentation 讨论主要围绕 prerequisite patch 与 follow-up work 的协调，而不是新的 user question。Daily window 内 thread 仍有回复，但 visible request 是 review 与 merge ordering。这表明大家关心如何让相互依赖的 schema 与 runtime change 保持足够小、能够按顺序 review；它不足以支持虚构另一个 troubleshooting question。

### Public kernel 工作集中在 invalidation 与 verifier boundary

除 dynptr 外，public archive 还在积极讨论 BPF test execution 中的 use-after-free prevention、concurrent lifetime handling、by-value 返回的 arena pointer、JIT memory checking、bounded map iteration 与更安全的 BTF dumping。这些主题具有同一模式：object 跨越 subsystem boundary 时，implementation 必须保持 ownership、type 与 failure information。讨论反复使用 focused selftest 区分真实 stale reference 与过度宽泛的 verifier rejection。

### 若干 user-facing surface 当天较安静

Project-specific help 与 feature area、scheduler support surface、general eBPF chat 和 public forum 在严格 daily window 内都没有新的 technical question。一个 general chat 出现新成员介绍，project feed 则主要是 automated repository activity，而非 human support request。这些目标仍被检查并计为 accessible；安静或自动化的活动没有被包装成 community demand。
