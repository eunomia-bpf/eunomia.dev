# BPF 可扩展调度生效时，cgroup v2 的 `cpu.max` 仍会限制 CPU 时间吗？

**简短回答：** 不一定。`cpu.max` 是 fair scheduler 为其管理的 task 提供的硬带宽控制。当 `sched_ext` scheduler 接管一个 task 后，kernel 可以把该 cgroup 的 period、quota 与 burst 值交给 BPF scheduler，但 BPF scheduler 必须自行实现 accounting、throttling 与 requeue，才能让这些值真正生效。因此，成功写入 `cpu.max` 只能证明配置被接受，不能证明 `sched_ext` workload 已被限流。

这个差异对把 `cpu.max` 当作隔离边界的 container 和 service 很重要。一个 scheduler 可以感知 cgroup、遵守相对的 `cpu.weight`，却仍忽略 hard ceiling；它也可以暴露 `cgroup_set_bandwidth` callback，却不实际 throttle task。安全结论只能来自精确的 scheduler implementation 与端到端 runtime test。

## 为什么 `sched_ext` 下的 contract 会改变

cgroup v2 CPU controller 把 `cpu.max` 定义为 `$MAX $PERIOD`：该 group 在每个 `$PERIOD` 内最多消耗 `$MAX` 微秒。当前 kernel 文档还明确限定了决定性的适用范围：`cpu.max` 影响的是 fair-class scheduler 管理的进程。

通常这已经足够，因为 fair scheduler 同时负责 quota accounting 与 throttling。Scheduler 把 runtime 计入 per-cgroup budget；budget 用尽后停止 runnable task；下一个 period 补充 budget 后再让它们重新可运行。

`sched_ext` 有意把 scheduling policy 移入 BPF。在普通的 full-switch mode 中，`SCHED_NORMAL`、`SCHED_BATCH`、`SCHED_IDLE` 与 `SCHED_EXT` task 都交给 BPF scheduler。Fair scheduler 不再拥有这些 task 的 runnable lifecycle，也就不能在正确时点直接替它们停下和释放。

Kernel-side interface 是通知与数据传递路径，不是 fallback limiter：

- cgroup 为 `sched_ext` 初始化时，`scx_cgroup_init_args` 会携带当前 weight、period、quota 与 burst，因此能覆盖 BPF scheduler 加载前已经存在的 limit。
- bandwidth 配置变化时，`ops.cgroup_set_bandwidth(cgrp, period_us, quota_us, burst_us)` 可以通知当前 BPF scheduler。
- callback 文档明确说明，control mechanism 以及 period、burst 的具体解释由 BPF scheduler 决定。
- core implementation 只在 scheduler 注册该 operation 时调用它，随后保存新值；core 不会替 scheduler 扣减 runtime 或 throttle dispatch queue。

因此，callback 存在是响应配置变化的必要条件，却不足以证明 enforcement。示例 scheduler 可能只把这些值打印出来。Production scheduler 需要完整 state machine：用已有配置初始化 budget、计费 execution time、确保每条 dispatch path 都不能绕过耗尽的 budget、安全保存被 throttle 的 task、在正确边界补充 budget、处理 hierarchy 与 task migration，并保证自身 control thread 可以继续推进。

## Relative weight 不是 hard quota

不要用 `cpu.weight` 替代 `cpu.max`。Weight 告诉 scheduler 在多个 runnable group 竞争时如何分配 CPU time，却不会限制一台空闲机器。没有竞争 group 时，weight 很小的 group 仍可能占满一个 CPU。

反过来也一样：实现 hierarchical weight sharing 的 scheduler 并不因此具备 bandwidth control。当前 cgroup 文档说明，`cpu.weight` 对 BPF scheduler 的影响依赖 `cgroup_set_weight` 以及 callback 实际做了什么；而 `cpu.max` 仍明确属于 fair class。Weight、bandwidth、utilization clamp 与 scheduling priority 应被当作彼此独立的 capability。

## 验证运行系统，不要相信配置表象

先确认 workload 由哪个 scheduler 管理。下面的检查都是只读的：

```bash
cat /sys/kernel/sched_ext/state
cat /sys/kernel/sched_ext/root/ops
SCX_PID=1234
grep -E '^ext\.enabled' "/proc/$SCX_PID/sched"
```

把 `1234` 替换为目标 service 或 container 中某个进程的 PID。如果系统使用 partial-switch mode，普通 `SCHED_NORMAL` task 可能仍由 fair scheduler 管理，并保留正常的 `cpu.max` 行为。不能仅凭系统里存在一个 `sched_ext` process 判断 ownership。

接着检查实际 cgroup：

```bash
SCX_CGROUP=/sys/fs/cgroup/path/to/workload
cat "$SCX_CGROUP/cpu.max"
cat "$SCX_CGROUP/cpu.max.burst"
cat "$SCX_CGROUP/cpu.stat"
```

记录 `usage_usec`，在这个已有 cgroup 内运行一个 wall-clock interval 已知的有限 CPU workload，再读取一次 `usage_usec`。把 CPU-time delta 与配置 quota 比较；然后回到 fair scheduler，或在匹配的 control host 上重复同一个 workload。持续运行时，half-CPU limit 不应在每秒内累积接近一个完整 CPU-second。

解释 `cpu.stat` 时要谨慎。总体 usage fields 包含 cgroup 中所有进程，但文档中的 `nr_periods`、`nr_throttled` 与 `throttled_usec` 描述的是 fair-scheduler bandwidth accounting。因此，`sched_ext` 下 throttle count 为零不能证明 workload 没有超出 limit；也可能只是 fair bandwidth controller 从未拥有这个 task。

最后检查 scheduler source 与版本。不要只搜索 callback name，还要确认以下属性：

1. cgroup 初始化时消费已有 quota state。
2. 每一条 execution path 都计费 runtime，包括 direct dispatch 与 re-enqueue。
3. budget 耗尽的 group 中的 task 无法进入 runnable dispatch queue。
4. replenishment 使用 monotonic scheduler clock，并且不会丢失 task。
5. task migration 与 cgroup hierarchy 不会重置或重复 budget。
6. scheduler 暴露足够 counter，以区分正常 enforcement、overrun 与实现故障。

当前 `scx_lavd` tree 是一个有用的公开例子：它记录实际 runtime，在 direct dispatch 与普通 enqueue 前都检查 group 是否已被 throttle，并把 throttled task 暂存到后续可运行的位置。这并不表示每个版本对每个 workload 都正确；它展示的是 quota enforcement 远不只是从 kernel 收到三个整数。

## 明确选择 deployment boundary

如果 hard CPU quota 属于 service-level 或 multi-tenant isolation contract，可以选择三种明确设计之一：

- 让 quota-bound workload 继续使用 fair scheduler。采用 partial-switch mode 的 `sched_ext` scheduler 可以让普通 task 留在 fair scheduling，只把选中的 `SCHED_EXT` task 交给 BPF policy。
- 使用为精确 kernel 与 scheduler 版本明确记录了 bandwidth enforcement 的 BPF scheduler，并在上线前用持续 quota test 验证。
- 增加 admission 或 startup check，拒绝“有限 `cpu.max` + 未验证 BPF scheduler”的组合。Orchestrator 承诺 hard limit 时，静默 best-effort 行为并不安全。

不要依赖成功的 cgroup write、operation callback 存在，或 scheduler 笼统的“支持 cgroup”声明。也不要假设 kernel warning 能捕获每种未实现 policy；`sched_ext` 本来就允许 scheduler 忽略其他输入，包括 nice level。Capability documentation 与 runtime validation 比 kernel 猜测所有遗漏更可靠。

ABI 仍与版本紧密相关。`sched_ext` operation callback 没有稳定性保证，旧 kernel 也可能完全没有 bandwidth callback。发行版 backport 同样会让 release number 失真。应把 kernel build、scheduler commit 或 package version、switching mode 与测量结果一起记录。

## 参考资料

- [Linux kernel 文档：cgroup v2 CPU interface files](https://docs.kernel.org/next/admin-guide/cgroup-v2.html#cpu-interface-files)
- [Linux kernel 文档：Extensible Scheduler Class](https://docs.kernel.org/scheduler/sched-ext.html)
- [Linux source：`sched_ext_ops.cgroup_set_bandwidth` contract](https://kernel.googlesource.com/pub/scm/linux/kernel/git/torvalds/linux/+/248951ddc14de84de3910f9b13f51491a8cd91df/kernel/sched/ext/internal.h)
- [Linux source：`sched_ext` 中的 cgroup 初始化与 bandwidth 更新传递](https://linux.googlesource.com/linux/kernel/git/torvalds/linux/+/2b414a95b8f7307d42173ba9e580d6d3e2bcbfce/kernel/sched/ext.c)
- [`scx_lavd` source：runtime charging 与 cgroup throttling path](https://github.com/sched-ext/scx/blob/main/scheds/rust/scx_lavd/src/bpf/main.bpf.c)
- [Linux kernel 文档：CPU frequency policy attributes](https://docs.kernel.org/admin-guide/pm/cpufreq.html)
- [Linux kernel 文档：通过 sysfs 导出的 CPU topology](https://docs.kernel.org/admin-guide/cputopology.html)
- [OpenTelemetry GenAI span semantic conventions](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-spans.md)
- [OpenTelemetry MCP span model](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/model/mcp/spans.yaml)
- [OpenTelemetry Metrics SDK cardinality limits](https://opentelemetry.io/docs/specs/otel/metrics/sdk/#cardinality-limits)
- [Linux kernel 文档：AF_XDP](https://docs.kernel.org/networking/af_xdp.html)
- [Linux kernel 文档：AF_XDP TX metadata](https://docs.kernel.org/networking/xsk-tx-metadata.html)
- [Linux commit：BPF 私有栈 eligibility](https://github.com/torvalds/linux/commit/a76ab5731e32d50ff5b1ae97e9dc4b23f41c23f5)
- [跟踪 classic-uprobe 私有 BPF 栈崩溃的公开报告](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/issues/3056)

## 今日社区讨论

今天通过普通可见浏览器检查了全部 6 个获准社区、共 15 个 allowlist 频道或公开页面，所有目标均可访问。入选问题来自过去 24 小时，因此没有使用七天回退。公开论坛只给最新条目显示 day-level age，精度不足以证明它落在巡检边界内，所以没有把它计入当日证据。以下内容已删除姓名、账号、雇主、社区和频道身份、消息链接、精确时间、私有拓扑、原始日志与可回搜措辞；没有保留原始 transcript。

### CPU policy input 需要明确的 fallback semantics

核心调度讨论暴露了 policy input 消失的两种方式。Cgroup 层面，有限 bandwidth value 可以被接受，但当前 BPF scheduler 不执行 throttling。Hardware 层面，按 advertised maximum frequency 给 CPU 排名的 scheduler 可能遇到没有可用 CPUFreq policy 的 virtual CPU，并计算出 zero capacity；另一个每 core 单 thread 的系统也让 unconditional “SMT-adjusted” aggregate 受到质疑。

两者都是 missing-data 问题，不能让它们以 numeric zero 身份静默参与算术。CPUFreq 文档说明 policy attributes 位于 `/sys/devices/system/cpu/cpufreq/policyX/`；virtual hardware 可能根本不暴露 policy。Topology 文档则说明，每个 CPU 的 sysfs topology 目录中的 `thread_siblings` 才是对应 sibling relation。健壮实现应验证这些来源，把“不可用”与数字零分开，根据 sibling mask 而不是通用 log label 判断 SMT，并选择有文档的保守 fallback，或者拒绝 unsupported mode。尚未解决的边界，是隐藏频率信息的 VM 应使用哪种 fallback capacity；它取决于 scheduler objective，无法只靠 `/proc/cpuinfo` 可靠重建。

### GenAI telemetry 需要分离 denominator，并限制 dimension

Agent observability 讨论集中在如何把详细 span 转成有用 metric，同时不改变其含义。安全的第一步是按 `gen_ai.operation.name` 分类。Chat、content generation、text completion 与 embeddings 等 model inference operation 可以构成 model-request denominator；agent invocation、workflow、retrieval 与 tool execution 是独立 operation，不能仅因为处于同一 trace 就算成额外 model call。

Token field 是 optional observation。Input 或 output token usage 缺失时，正确 aggregate state 是 coverage missing，而不是 zero usage。字段存在时，GenAI conventions 说明 cache-read 与 cache-creation input token 已应包含在 total input tokens 中；再次相加会 double-count。Dashboard 应同时报告 measured total 与 coverage ratio，从而区分低成本请求和未测量请求。

MCP session ID、resource URI 与 JSON-RPC request ID 可以是单条 trace 上有用的 span attribute，却不适合做 metric label。MCP model 已经因为 high cardinality，默认不把 resource URI 放进 span name。Metric view 应在 aggregation 前 allowlist low-cardinality dimension；SDK cardinality limit 与 `otel.metric.overflow` 是最后安全网，而不是主要 schema。关闭 content capture 也不能自动解决 metric cardinality 或 identifier leakage。

### 只有先证明 probe 确实执行，crash reproducer 才有效

私有栈 uprobe 调查仍在继续，但没有产生区别于前一日回答的新问题。新出现的实用教训有关 negative evidence：重建测试 kernel 后，一组 probe 可以成功 load，却不再看到预期的 uprobe activity。在这种状态下无法触发 panic，不能说明候选修复有效。

比较 kernel 或 application guard 前，应确认 target symbol 被命中、BPF program run count 增长、JIT/private-stack path 存在，并且 workload 产生并发调用。还应保留一个必须触发的 known-good positive control。只有这样，“baseline 崩溃、只改变一项后不崩溃”才能支持 mitigation claim。底层 kernel 与 application patch 仍在 review，因此选择性禁用 probe 或使用经过验证的 kernel build，仍是保守边界。

### Network patch 强调 UAPI layout 与 ownership，而不只是速度

公开开发列表还出现多项围绕 frame layout、TX metadata 与 generic XDP assumption 的 AF_XDP 修复。它们共享一条规则：descriptor address、UMEM frame、metadata headroom 与 packet boundary 共同组成 ABI contract。Copy 与 zero-copy path、multi-buffer packet 以及 driver 必须对 metadata 的位置和 ownership 返回 userspace 的时点达成一致。

实际诊断应分别强制 copy 与 zero-copy mode，验证 descriptor address 与 completion ownership，测试 single-buffer 和 multi-buffer packet，并按照固定 `xsk_tx_metadata` layout 对照 offload 行为。某个 driver 成功不能证明 generic path 正确。这些仍是活跃 patch review，而不是稳定行为；用户应测试精确的 kernel 与 NIC 组合，不能假设最新 proposed layout 已经部署。

其余 project-focused chat surface 较安静、只包含 onboarding 或 automated build notice，或在当日窗口内没有实质技术讨论。没有把任何不可访问页面误报为 quiet。
