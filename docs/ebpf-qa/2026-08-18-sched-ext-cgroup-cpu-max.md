# Does cgroup v2 `cpu.max` still limit CPU time under BPF extensible scheduling?

**Short answer:** not automatically. `cpu.max` is a hard-bandwidth control for tasks handled by the fair scheduler. When a `sched_ext` scheduler owns a task, the kernel can pass the cgroup's period, quota, and burst values to the BPF scheduler, but the BPF scheduler must implement the accounting, throttling, and requeue behavior that gives those values effect. A successful write to `cpu.max` therefore proves that configuration was accepted; it does not prove that a `sched_ext` workload is being capped.

This distinction matters for containers and services that treat `cpu.max` as an isolation boundary. A scheduler can be cgroup-aware, honor relative `cpu.weight`, and still ignore the hard ceiling. It can also expose the `cgroup_set_bandwidth` callback without actually throttling tasks. The only safe conclusion comes from the exact scheduler implementation plus an end-to-end runtime test.

## Why the contract changes under `sched_ext`

The cgroup v2 CPU controller describes `cpu.max` as `$MAX $PERIOD`: the group may consume at most `$MAX` microseconds in each `$PERIOD`. Current kernel documentation also states the decisive scope explicitly: `cpu.max` affects processes under the fair-class scheduler.

That is normally enough because the fair scheduler owns both quota accounting and throttling. The scheduler accounts runtime against a per-cgroup budget, stops runnable tasks after the budget is exhausted, and makes them eligible again when the next period replenishes the budget.

`sched_ext` deliberately moves scheduling policy into BPF. In its normal full-switch mode, `SCHED_NORMAL`, `SCHED_BATCH`, `SCHED_IDLE`, and `SCHED_EXT` tasks are all handed to the BPF scheduler. The fair scheduler no longer owns the runnable lifecycle of those tasks, so its bandwidth controller cannot simply stop and release them at the right points.

The kernel-side interface is a notification and data-delivery path, not a fallback limiter:

- When a cgroup is initialized for `sched_ext`, `scx_cgroup_init_args` carries its current weight, period, quota, and burst. This covers limits that existed before the BPF scheduler was loaded.
- When the bandwidth configuration changes, `ops.cgroup_set_bandwidth(cgrp, period_us, quota_us, burst_us)` can notify the active BPF scheduler.
- The callback documentation says that the control mechanism and the interpretation of period and burst behavior are up to the BPF scheduler.
- The core implementation calls the operation only when the scheduler registered it, then stores the new values. It does not debit runtime or throttle a dispatch queue on the scheduler's behalf.

This is why callback presence is necessary for reacting to configuration changes but insufficient as an enforcement claim. A demonstration scheduler may only print the values. A production scheduler needs a complete state machine: initialize pre-existing budgets, charge execution time, prevent every dispatch path from bypassing a depleted budget, hold throttled tasks safely, replenish them at the correct boundary, handle hierarchy and task migration, and preserve forward progress for its own control threads.

## Relative weight is not a hard quota

Do not substitute `cpu.weight` for `cpu.max`. Weight tells a scheduler how to divide contested CPU time between runnable groups. It does not cap an otherwise idle machine. A group with a small weight may still consume a whole CPU when no competing group is runnable.

The same caution applies in the other direction: a scheduler that implements hierarchical weight sharing has not thereby implemented bandwidth control. Current cgroup documentation describes `cpu.weight` as affecting a BPF scheduler only through `cgroup_set_weight` and depending on what that callback actually does. For `cpu.max`, the documentation remains fair-class-only. Treat weight, bandwidth, utilization clamps, and scheduling priority as separate capabilities.

## Verify the running system instead of trusting configuration

Start by identifying which scheduler owns the workload. These checks are read-only:

```bash
cat /sys/kernel/sched_ext/state
cat /sys/kernel/sched_ext/root/ops
SCX_PID=1234
grep -E '^ext\.enabled' "/proc/$SCX_PID/sched"
```

Replace `1234` with the PID of a process in the target service or container. If the system is using partial-switch mode, ordinary `SCHED_NORMAL` tasks may still be on the fair scheduler and retain the normal `cpu.max` behavior. Do not infer ownership from the presence of a `sched_ext` process alone.

Next inspect the actual cgroup:

```bash
SCX_CGROUP=/sys/fs/cgroup/path/to/workload
cat "$SCX_CGROUP/cpu.max"
cat "$SCX_CGROUP/cpu.max.burst"
cat "$SCX_CGROUP/cpu.stat"
```

Record `usage_usec`, run a bounded CPU workload inside that existing cgroup for a known wall-clock interval, and read `usage_usec` again. Compare the CPU-time delta with the configured quota. Repeat the same workload after reverting to the fair scheduler or on a matched control host. A half-CPU limit should not accumulate close to one full CPU-second per second over a sustained interval.

Interpret `cpu.stat` carefully. Its overall usage fields include all processes in the cgroup, but the documented `nr_periods`, `nr_throttled`, and `throttled_usec` fields describe fair-scheduler bandwidth accounting. A zero throttling count under `sched_ext` is therefore not proof that the workload stayed below the limit; it may mean that the fair bandwidth controller never owned the task.

Finally, inspect the scheduler source and version. Look for all of these properties, not just the callback name:

1. Existing quota state is consumed during cgroup initialization.
2. Runtime is charged on every execution path, including direct dispatch and re-enqueue paths.
3. A task from an exhausted group cannot reach a runnable dispatch queue.
4. Replenishment uses a monotonic scheduler clock and cannot lose tasks.
5. Task migration and cgroup hierarchy cannot reset or duplicate budget.
6. The scheduler exports enough counters to distinguish enforcement, overruns, and implementation failure.

The current `scx_lavd` tree is a useful public example of the required shape: it records consumed runtime, checks whether a group is throttled before direct dispatch as well as normal enqueue, and puts throttled tasks aside for later execution. That does not make every version correct for every workload; it shows why quota enforcement is more than receiving three integers from the kernel.

## Choose a deployment boundary deliberately

If hard CPU quotas are part of a service-level or multi-tenant isolation contract, use one of three explicit designs:

- Keep quota-bound workloads on the fair scheduler. A `sched_ext` scheduler using partial-switch mode can leave ordinary tasks under fair scheduling while selected `SCHED_EXT` tasks use BPF policy.
- Use a BPF scheduler that documents bandwidth enforcement for the exact kernel and scheduler version, then validate it with a sustained quota test before production rollout.
- Add an admission or startup check that rejects the combination of a finite `cpu.max` and an unverified BPF scheduler. Silent best-effort behavior is unsafe when an orchestrator promises a hard limit.

Do not rely on a successful cgroup write, the existence of an operation callback, or a scheduler's general claim of cgroup support. Also do not assume that a kernel warning will catch every unsupported policy. `sched_ext` intentionally permits schedulers to ignore other inputs, including nice levels; capability documentation and runtime validation scale better than trying to infer every omission in the kernel.

The ABI remains version-sensitive. `sched_ext` operation callbacks have no stability guarantee, and older kernels may not expose the bandwidth callback at all. Vendor backports can also make a release number misleading. Record the kernel build, scheduler commit or package version, switching mode, and the measured result together.

## References

- [Linux kernel documentation: cgroup v2 CPU interface files](https://docs.kernel.org/next/admin-guide/cgroup-v2.html#cpu-interface-files)
- [Linux kernel documentation: Extensible Scheduler Class](https://docs.kernel.org/scheduler/sched-ext.html)
- [Linux source: `sched_ext_ops.cgroup_set_bandwidth` contract](https://kernel.googlesource.com/pub/scm/linux/kernel/git/torvalds/linux/+/248951ddc14de84de3910f9b13f51491a8cd91df/kernel/sched/ext/internal.h)
- [Linux source: cgroup initialization and bandwidth update delivery in `sched_ext`](https://linux.googlesource.com/linux/kernel/git/torvalds/linux/+/2b414a95b8f7307d42173ba9e580d6d3e2bcbfce/kernel/sched/ext.c)
- [`scx_lavd` source: runtime charging and cgroup throttling paths](https://github.com/sched-ext/scx/blob/main/scheds/rust/scx_lavd/src/bpf/main.bpf.c)
- [Linux kernel documentation: CPU frequency policy attributes](https://docs.kernel.org/admin-guide/pm/cpufreq.html)
- [Linux kernel documentation: CPU topology exported through sysfs](https://docs.kernel.org/admin-guide/cputopology.html)
- [OpenTelemetry GenAI span semantic conventions](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-spans.md)
- [OpenTelemetry MCP span model](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/model/mcp/spans.yaml)
- [OpenTelemetry Metrics SDK cardinality limits](https://opentelemetry.io/docs/specs/otel/metrics/sdk/#cardinality-limits)
- [Linux kernel documentation: AF_XDP](https://docs.kernel.org/networking/af_xdp.html)
- [Linux kernel documentation: AF_XDP TX metadata](https://docs.kernel.org/networking/xsk-tx-metadata.html)
- [Linux commit: private BPF stack eligibility](https://github.com/torvalds/linux/commit/a76ab5731e32d50ff5b1ae97e9dc4b23f41c23f5)
- [Public report tracking classic-uprobe crashes on private BPF stacks](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/issues/3056)

## Community discussion today

Today's ordinary visible-browser review covered all 6 approved communities and all 15 allowlisted channels or public pages. Every target was accessible, and the selected question came from the 24-hour window, so the seven-day fallback was not used. The public forum exposed only day-level age for its newest item, which was not precise enough to prove that it fell inside the review boundary; it was not counted as same-day evidence. Names, accounts, employers, community and channel identities, message links, exact times, private topology, raw logs, and searchable phrasing have been removed. No raw transcript was retained.

### CPU policy inputs need explicit fallback semantics

The main scheduler discussion exposed two separate ways a policy input can disappear. At the cgroup layer, a finite bandwidth value can be accepted while the active BPF scheduler performs no throttling. At the hardware layer, a scheduler that ranks CPUs by advertised maximum frequency can encounter a virtual CPU with no usable CPUFreq policy and derive a zero capacity. A separate single-thread-per-core system also raised doubts about an unconditional “SMT-adjusted” aggregate.

These are both missing-data problems, not values that should silently participate in arithmetic. The CPUFreq documentation says policy attributes exist under `/sys/devices/system/cpu/cpufreq/policyX/`; virtual hardware may expose no policy at all. The topology documentation identifies `thread_siblings` under each CPU's sysfs topology directory as the relevant sibling relation. A robust scheduler should validate those sources, distinguish “unavailable” from numeric zero, derive SMT from sibling masks rather than a generic log label, and either select a documented conservative fallback or refuse the unsupported mode. What remains unresolved is the correct fallback capacity for frequency-obscured virtual machines; it depends on the scheduler's objective and cannot be reconstructed reliably from `/proc/cpuinfo` alone.

### GenAI telemetry needs separate denominators and bounded dimensions

Agent observability discussions centered on how to turn detailed spans into useful metrics without changing their meaning. The safe first step is to group by `gen_ai.operation.name`. Model inference operations such as chat, content generation, text completion, and embeddings can form a model-request denominator; agent invocation, workflow, retrieval, and tool execution are separate operations and should not be counted as extra model calls merely because they occur in the same trace.

Token fields are optional observations. If input or output token usage is absent, the correct aggregate state is missing coverage, not zero usage. When reported, the GenAI conventions say cache-read and cache-creation input tokens should already be included in total input tokens; adding them again double-counts. Report both the measured total and a coverage ratio so dashboards can distinguish a cheap request from an unmeasured one.

MCP session IDs, resource URIs, and JSON-RPC request IDs can be valuable span attributes for one trace but are poor metric labels. The MCP model already avoids resource URIs in span names by default because of high cardinality. Metric views should allowlist low-cardinality dimensions before aggregation, while the SDK cardinality limit and `otel.metric.overflow` remain a last-resort safety net rather than the primary schema. Content capture being disabled does not by itself solve metric cardinality or identifier leakage.

### A crash reproducer is valid only after probe execution is proven

The private-stack uprobe investigation continued, but it did not produce a new question distinct from the previous day's answer. The useful new operational lesson was about negative evidence: after rebuilding a test kernel, a probe suite could load while expected uprobe activity was no longer visible. Failure to trigger a panic in that state says nothing about the candidate fix.

Before comparing kernels or application guards, verify that the target symbol is reached, the BPF program's run count increases, the JIT/private-stack path is present, and the workload creates concurrent invocations. Keep a known-good positive control that must fire. Only then does “crash on baseline, no crash with one change” support a mitigation claim. The underlying kernel and application patches remain under review, so selective probe disablement or a validated kernel build remains the conservative boundary.

### Network patches emphasized UAPI layout and ownership, not only speed

The public development list also carried several AF_XDP fixes around frame layout, TX metadata, and generic XDP assumptions. These changes share one rule: a descriptor address, UMEM frame, metadata headroom, and packet boundary form an ABI contract. Copy and zero-copy paths, multi-buffer packets, and drivers must agree on where metadata lives and when ownership returns to userspace.

The practical diagnostic is to force copy and zero-copy modes separately, validate descriptor addresses and completion ownership, test single- and multi-buffer packets, and compare offload behavior against the fixed `xsk_tx_metadata` layout. A driver-specific success does not prove that the generic path is correct. These were active patch reviews, not settled behavior, so users should test the exact kernel and NIC combination rather than assuming the newest proposed layout is already deployed.

The remaining project-focused chat surfaces were quiet, contained onboarding or automated build notices, or had no substantive technical discussion in the daily window. No inaccessible surface was misclassified as quiet.
