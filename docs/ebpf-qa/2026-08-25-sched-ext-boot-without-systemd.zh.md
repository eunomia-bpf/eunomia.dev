# 没有 systemd 时，怎样在开机后启动 `sched_ext` 调度器？

**简短回答：** 应把调度器当作需要监督的 foreground process，而不是会被 kernel 永久保存的配置。`sched_ext` policy 只在对应 BPF scheduler 已加载并持续运行时生效。在 OpenRC、runit、dinit 或其他 init system 上，需要创建 native service：等 `/sys/kernel/sched_ext` 出现后，以确定的 binary 和 arguments 启动 scheduler，让它保持 foreground，以有界 delay 进行失败重启，并把 service 加入正常 boot target。启动后必须检查 `/sys/kernel/sched_ext/state` 和 `/sys/kernel/sched_ext/root/ops`；service 显示绿色，不能单独证明预期 scheduling policy 已生效。

如果 scheduler 退出、stall 或被拒绝，kernel 会禁用它，并让任务回到 fair scheduler。这是重要的安全 fallback，但也意味着机器可能正常启动，却静默运行了另一种 scheduling policy。

## Reboot 后真正需要恢复的是什么

Kernel 不会把上一次选择的 `sched_ext` scheduler 保存成下次启动偏好。Linux 文档把 `sched_ext` 描述为动态机制：加载 BPF scheduler 时启用，scheduler 卸载或失败时禁用。因此，真正持久化的是一份 init-system declaration，其中包括：

- scheduler executable 的 absolute path；
- 与当前版本匹配、经过验证的 command-line arguments；
- boot 过程中允许启动它的时点；
- process supervision 与 restart policy；
- standard output 和 error 的记录位置；
- 激活它的 boot target 或 service bundle。

即使目标机器不使用 systemd，`scx` upstream 的 systemd unit 仍可用来理解这份 contract：它检查 `/sys/kernel/sched_ext` 是否存在，从 `/etc/default/scx` 读取配置，把选择的 scheduler 作为 foreground service process 启动，失败后重启，并加入 multi-user boot target。需要把这些语义翻译为本机 init format；在 OpenRC、runit 或 dinit 上复制一份 systemd unit 本身没有作用。

自动启动前，应先在受控 session 中手动运行选定 scheduler，并检查当前版本的 `--help`。`sched_ext` ABI 明确不稳定，scheduler options 也会变化。Kernel update、package update、binary rename 或过时 flag 是否导致 command invalid，不应等到 boot supervision 阶段才发现。

## 简单部署优先直接监督 scheduler process

如果只需要一个固定 policy，直接监督 `scx_lavd` 这类 scheduler 是最小设计。让它保持 foreground，由 init system 管理 PID 与生命周期。不要额外加入 `&`、`nohup`、shell daemonizer 或 background mode，除非 init system 明确要求：supervisor 无法可靠重启一个刚启动就与它脱离的 child。

以下只是**示意性的 service shape**，不是可以直接安装的 distribution package。Path、scheduler choice、arguments、dependency、logging 和 service directory 都必须匹配实际系统。

### OpenRC

`/etc/init.d/scx-scheduler` 可以使用 OpenRC 自带的 supervisor：

```sh
#!/sbin/openrc-run

description="Local sched_ext scheduler"
command="/usr/bin/scx_lavd"
command_args="--some-verified-option"
supervisor="supervise-daemon"
respawn_delay=5
respawn_max=5
respawn_period=60

depend() {
    after modules
}
```

OpenRC 指南要求被 `supervise-daemon` 监督的 daemon 留在 foreground。`respawn_delay`、`respawn_max` 和 `respawn_period` 可以避免 invalid scheduler 进入无限的高频 restart loop。Dependency name 与 distribution 有关，`after modules` 只是一种示例。本地 pre-start check 还应在 `/sys/kernel/sched_ext` 不存在时拒绝启动。只有 manual start 已成功后，才应通过 distribution 的正常 OpenRC runlevel 机制启用 service。

### runit

在 runit service directory 中，`run` 可以直接把自己替换为 scheduler：

```sh
#!/bin/sh
exec 2>&1
exec /usr/bin/scx_lavd --some-verified-option
```

`runsv` 会启动 `./run`，并通常在它退出后重新启动。应通过 distribution 的正常 link 或 package 机制，把 service directory 加入受监督的 boot set。Runit 会有意 restart service，因此第一次 boot 后要检查日志，避免永久不兼容的 binary 持续循环。如果 distribution 本身没有提供足够 backoff，可以在本地 `finish` policy 中加入一个小而明确的 delay。

### dinit

一个 process service 可以表达相同生命周期：

```text
type = process
command = /usr/bin/scx_lavd --some-verified-option
restart = true
waits-for: local-preconditions
```

Dinit 文档中的 process service 期望 foreground command，并支持 automatic restart。应把 service description 放入配置的 service directory，把 `local-preconditions` 替换为该系统真实存在的 service，再把 scheduler 加入恰当的 boot dependency chain。不能因为另一个 distribution 的 dependency name 看起来合理，就直接复制过来。

无论使用哪一种 init system，都应把可变 arguments 放入 root-owned configuration file，使用 absolute path，并避免让 shell evaluate 不可信文本。除非选择的 scheduler 和本地 policy 已专门验证过更窄的 privilege model，否则应以 root 启动。

## `scx_loader` 是另一种 deployment model

`scx_loader` 是通过 system D-Bus 暴露 scheduler control 的 management daemon；`scxctl` 是它的 command-line client。Loader configuration 可以选择 `default_sched` 和 mode，因此 loader 启动后可以继续启动配置的 scheduler。需要通过统一 control plane 切换 policy 时，这种模式很有用。

它也会引入更多 boot prerequisites。Upstream loader 当前提供的是 systemd D-Bus service 和带 systemd hardening 的 unit。移植到 non-systemd 时，不能只调用一次 `scxctl`，而要同时具备：

1. 正在运行的 system D-Bus；
2. 安装到本机约定路径的 loader D-Bus service、policy 和 interface files；
3. 监督 foreground `scx_loader` 的 native init service；
4. 对 system bus 与 kernel prerequisite 的显式 dependency；
5. 已验证的 `default_sched` configuration，或另一项有明确 ordering 的 control action。

如果不需要这些能力，direct supervision 可以减少一个 daemon 与 authorization surface。如果确实需要，应把 loader 当作 durable service，把它拥有的 scheduler process 当作 subordinate state。不要同时独立监督两者，否则两个 controller 可能争相替换 active `sched_ext` policy。

## 每次 boot 后都要验证 kernel state

Kernel 提供了直接 status surface：

```sh
test -d /sys/kernel/sched_ext
cat /sys/kernel/sched_ext/state
cat /sys/kernel/sched_ext/root/ops
cat /sys/kernel/sched_ext/enable_seq
```

Service 启动后应满足：

- `state` 报告 `enabled`；
- `root/ops` 标识预期 scheduler operations；
- `enable_seq` 显示 scheduler 已经被启用过；
- supervised process 仍存活，且没有陷入 restart loop。

这些观察可以区分 generic “service started” check 无法发现的失败：

- **没有 `/sys/kernel/sched_ext`：** 当前 kernel 缺少该能力或必要 kernel configuration。
- **Service inactive：** boot activation、dependency ordering、executable path 或 permission 有误。
- **Service 反复退出且 state 为 disabled：** 捕获第一次 verifier 或 loader error；停止循环，并手动测试一次 invocation。
- **State enabled，但 `root/ops` 不符合预期：** 另一个 controller 或 package 选择了不同 scheduler。
- **Kernel upgrade 前可用，之后 scheduler 不再加载：** 检查 exact kernel、BTF、scheduler build 与 command line 组合；不要用无限 restart 掩盖 ABI mismatch。

正式依赖前还要演练 failure path。主动停止 scheduler，确认任务继续由 kernel fair class 执行，supervisor 记录到 failure，restart behavior 符合 policy，且 state files 能返回预期值。还应保留一条已记录的路径，用于禁用 boot service 并保持 fair scheduler。这个 rollback 比试图让 boot path 永不失败更有价值。

## 参考资料

- [Linux kernel `sched_ext` 文档：动态启用、failure fallback、status files 与 ABI warning](https://docs.kernel.org/scheduler/sched-ext.html)
- [`scx` upstream systemd service](https://github.com/sched-ext/scx/blob/main/services/scx.service)
- [`scx` upstream default service configuration](https://github.com/sched-ext/scx/blob/main/services/scx)
- [`scx` service 文档](https://github.com/sched-ext/scx/blob/main/services/README.md)
- [`scx_loader` architecture 与 system D-Bus interface](https://github.com/sched-ext/scx-loader/blob/main/README.md)
- [`scx_loader` configuration，包括 `default_sched`](https://github.com/sched-ext/scx-loader/blob/main/crates/scx_loader/configuration.md)
- [`scx_loader` upstream systemd unit](https://github.com/sched-ext/scx-loader/blob/main/services/scx_loader.service)
- [`scxctl` client 文档](https://github.com/sched-ext/scx-loader/blob/main/crates/scxctl/README.md)
- [OpenRC service-script guide](https://github.com/OpenRC/openrc/blob/master/service-script-guide.md)
- [OpenRC `supervise-daemon` guide](https://github.com/OpenRC/openrc/blob/master/supervise-daemon-guide.md)
- [runit `runsv(8)` manual](https://smarden.org/runit/runsv.8)
- [Dinit service-description overview](https://github.com/davmac314/dinit/blob/master/README.md)
- [bpftime fixed-hash iteration 修复与 regression test](https://github.com/eunomia-bpf/bpftime/pull/658)
- [OpenTelemetry eBPF metric-label change](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/pull/3168)
- [OpenTelemetry GenAI evidence-reference proposal](https://github.com/open-telemetry/semantic-conventions-genai/issues/470)
- [Linux BPF patch discussion：跨 page 的 USDT instruction patching](https://lore.kernel.org/bpf/20260825150444.31603-1-jiayuan.chen@linux.dev/T/#t)
- [Linux BPF patch discussion：XDP path 中的 page ownership](https://lore.kernel.org/bpf/20260824030257.263179-1-jiayuan.chen@linux.dev/T/#t)

## 当日社区讨论

今天通过普通可见浏览器检查了全部 6 个批准社区和 15 个 allowlist 频道或公开页面，所有目标均可访问。选题来自过去 24 小时，因此没有使用七天 fallback。姓名、账号、雇主、workspace 与频道身份、message link、精确时间、私有拓扑、原始日志和可搜索回原讨论的措辞均已删除。没有保留原始 transcript，也没有进行任何社交互动。

### Boot automation 需要 kernel-state acceptance test

最强的未解决问题是：一台不使用 systemd 的机器如何在 reboot 后恢复 `sched_ext` 选择。关键重构是 policy 本身并不持久化；boot automation 必须监督 userspace scheduler，重新创建 live attachment。Upstream 的 [kernel status files](https://docs.kernel.org/scheduler/sched-ext.html)让 acceptance test 不依赖本地使用 OpenRC、runit、dinit 还是其他 init system。

讨论由此落到两个实际问题。第一，service manager 应监督 foreground process 并限制 retry，因为 invalid command 即使立即退出，kernel fallback 仍会让 host 看起来可用。第二，service liveness 与 scheduler activation 是两个不同事实。Wrapper 或 management daemon 可能保持 alive，但预期 scheduling operations 并未 attach，因此 `state` 和 `root/ops` 必须进入 post-boot checks。

### Runtime 与 kernel 工作主要在处理边界条件

一个 userspace BPF runtime fix 处理了 fixed hash map 的 iteration：实际分配的 bucket count 被调整为 prime number，lookup 仍能找到 entry，但 key iteration 使用了调整前的 size，因而可能跳过 buckets。公开[修复](https://github.com/eunomia-bpf/bpftime/pull/658)让 lookup 与 iteration 共享 effective capacity，并加入 regression test。这个教训不只适用于该 map：遍历结构的 API 必须使用与写入 API 相同的 realized geometry。

Kernel 讨论也反复回到 ownership 与 boundary failure。一个 patch 处理跨 page boundary 的 userspace static-tracing instruction sequence；另一个检查 XDP path 中的 page ownership 与 release。Memory limit 下的 reclaim、program signing、stream-buffer sizing、private BPF stack、verifier bounds 与 HID BPF changes 也很活跃。这些主题共同说明：高价值测试应主动触发少见 boundary 或 failure path，而不只证明 common case。

### Telemetry change 需要 migration semantics，而不只是更干净的 schema

一个 observability implementation 提议从 default metric labels 移除 service identity，同时继续把它保留为 resource data，并允许显式 opt-in。这样可以避免在每个 data point 重复 identity，但仍然会打破已有 query surface。用户需要 migration path——例如 resource-to-target metadata join 或临时 compatibility option——而不能等 dashboard 失败后才发现新的 label set。公开[变更](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/pull/3168)仍是当前状态的 source of truth。

前一日关于让 GenAI evaluation result 携带 verifiable evidence 的讨论，也推进为公开的 [specification issue](https://github.com/open-telemetry/semantic-conventions-genai/issues/470)。这项 follow-up 没有被提升为第二个每日问题。有效边界仍是：standard 可以定义 reference、digest 与 media type，但不能声称 referenced content 已被信任，也不应把 telemetry storage 变成 artifact archive。

### 安静目标仍然完成了检查

若干 project help surfaces 在 daily window 内没有实质消息，一个 public practitioner forum 的最新帖子已经超过一周。某 networking community 的活动主要是 contributor onboarding、documentation 与 meeting logistics。其他 project-specific surfaces 只有 automated build notices、introductions、older threads 或没有消息。这些目标被记录为 accessible and quiet，没有被误写成零活动，也没有用作 fallback evidence。
