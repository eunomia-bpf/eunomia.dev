# Why do non-sleepable BPF fentry and fexit programs silently skip calls from another task on the same CPU, and why do the skipped calls show up as recursion misses?

Because the trampoline recursion guard is per-CPU and is taken with preemption still enabled. A non-sleepable fentry/fexit program runs through the BPF trampoline, whose entry path increments `prog->active`, a per-CPU counter, under `rcu_read_lock_dont_migrate()`. That lock only disables CPU migration; it does not disable preemption. If the task that entered the program is preempted mid-program, its counter stays at 1 on that CPU. When an unrelated task is then scheduled onto that CPU and hits the same hook, the trampoline sees the counter go to 2 instead of 1, concludes that the program is recursing, increments `recursion_misses`, and silently skips the program. No recursion happened; the miss is a false positive.

Before kernel 7.0 most distribution kernels ran `PREEMPT_VOLUNTARY` by default, and non-sleepable BPF programs contain no voluntary preemption points, so the window was rarely hit. Commit `7dadeaa6e851` ("sched: Further restrict the preemption modes") removed `PREEMPT_NONE` and `PREEMPT_VOLUNTARY` on x86 and arm64, leaving lazy and full preemption. Every distribution kernel since then preempts inside non-sleepable fentry/fexit programs during normal multi-tasking, so the drops are now routine, and they are invisible in the trace output: the event simply never runs.

## The mechanism

1. The guard. `__bpf_prog_enter_recur()` in `kernel/bpf/trampoline.c` acquires `prog->active` — a per-CPU counter declared `u8 __percpu *active` in `include/linux/bpf.h` — under `rcu_read_lock_dont_migrate()`. `bpf_prog_get_recursion_context()` then requires the per-CPU value to be exactly 1; any other value means "already inside", the trampoline calls `bpf_prog_inc_misses_counter()`, optionally fires the `recursion_detected` callback, and returns without running the BPF program.

2. The preemption hole. `rcu_read_lock_dont_migrate()` keeps the task pinned to its CPU but leaves it preemptible. Task A enters `fentry/foo`, `prog->active` on CPU N becomes 1, Task A is preempted mid-program with the counter still 1. Task B runs on CPU N and calls `foo()`; the counter increments to 2, the check `val != 1` fails, and Task B's invocation is counted as a recursion miss and skipped. When Task A resumes and exits, the counter drops back to 0. The miss is pure preemption interference, not re-entry.

3. Why per-CPU, not per-task. Since commit `a76ab5731e32` ("bpf: Find eligible subprogs for private stack support"), recursion-checked trampoline programs also use `prog->active` to guarantee exclusive ownership of the single per-CPU private stack (`priv_stack_ptr`) and per-CPU scratch maps. Switching the counter to per-task storage would remove the false misses but break that ownership guarantee, which is why the fix has to be a preemption barrier, not a different counter.

4. The contrast with kprobes and tracepoints. `trace_call_bpf()` in `kernel/trace/bpf_trace.c` asserts `cant_sleep()` and commit `16718274ee75` ("tracing: perf: Have perf tracepoint callbacks always disable preemption") kept preemption disabled for perf tracepoint callbacks. So the same BPF program logic gets different completeness guarantees depending only on whether it attaches via fentry/fexit or via kprobe/classic tracepoint.

## How to verify and debug

1. Count the misses. Enable `BPF_STATS_RUN_TIME` with `bpf_enable_stats()`, run the workload, then read `prog_info.run_cnt` against `prog_info.recursion_misses` via `bpf_prog_get_info_by_fd()`. A program that never re-enters itself but reports non-zero `recursion_misses` under a preemptible kernel is the signature of this bug; the counter has existed since commit `ca06f55b9002` ("bpf: Add per-program recursion prevention mechanism"), which the RFC cites as the `Fixes:` target.

2. Reproduce deterministically. Pin a producer task to one CPU, attach a paired fentry/fexit probe to a hot function, and make the probe body long enough to be preemptible (the RFC's 47-line reproducer loops a `bpf_loop` of a million iterations inside both `fentry/__set_task_comm` and `fexit/__set_task_comm` while forking short-lived tasks). The RFC measured 0 skips per 1000 on a stock 6.8 kernel in voluntary-preemption mode, 25-26 percent on the same kernel in full-preemption mode, and 28 percent or more on 7.3 mainline in lazy mode.

3. Check the preemption mode. `uname -v` and the kernel config tell you whether `CONFIG_PREEMPT_LAZY` or `CONFIG_PREEMPT` is set. A kernel built with the old voluntary mode does not reproduce the drops; that is why the bug is new to 7.0-era kernels even though the code path is much older.

4. Cross-check attach types. Re-attach the same logic as a kprobe and as a classic tracepoint program on the same hook. Because those paths run with preemption disabled, the kprobe/tracepoint variants keep firing for every task; the fentry/fexit variant keeps the false recursion misses. That differential is the fastest way to confirm the cause rather than suspect a logic bug in the program.

## The limitation

- The RFC's fix applies to `!CONFIG_PREEMPT_RT` kernels only: on PREEMPT_RT the trampoline keeps `rcu_read_lock_dont_migrate()`, since sleepable and preemptible execution is the design goal there. Non-RT kernels get `preempt_disable_notrace()` around the enter/exit pair instead.
- The miss itself is silent by design: `recursion_misses` is a polled stat and the `recursion_detected` callback is optional, so a monitoring system that only watches successful events will not notice the loss of completeness on its own.
- Private-stack and per-CPU scratch-map ownership is the reason the counter cannot simply become per-task; any alternative fix still has to serialize the CPU, which is why the series chose a preemption barrier over a counter redesign.
- The kprobe/tracepoint workaround is not free: on hot socket paths the RFC reports fentry roughly 2x faster than kprobe, and kretprobe relies on a bounded `maxactive` pool that can exhaust on blocking calls such as `inet_recvmsg()`.

## References

- [Public bpf mailing-list RFC: do not let preemption cause false recursion drops in trampoline programs](https://lore.kernel.org/bpf/20261025-b4-bpf-trampoline-preempt-v1-0-198dcdc0c6ac@sentinelone.com/) — a public bpf-list RFC proposing `preempt_disable_notrace()` around `__bpf_prog_enter_recur()` / `__bpf_prog_exit_recur()` on non-RT kernels, plus a reproducing selftest.
- [Kernel `kernel/bpf/trampoline.c`](https://github.com/torvalds/linux/blob/master/kernel/bpf/trampoline.c) — `__bpf_prog_enter_recur` and the miss-counting path.
- [Kernel `include/linux/bpf.h`](https://github.com/torvalds/linux/blob/master/include/linux/bpf.h) — the per-CPU `active` counter and `bpf_prog_get_recursion_context`.
- [Kernel `include/uapi/linux/bpf.h`](https://github.com/torvalds/linux/blob/master/include/uapi/linux/bpf.h) — `bpf_prog_info.recursion_misses` and `run_cnt`.
- [Kernel `kernel/trace/bpf_trace.c`](https://github.com/torvalds/linux/blob/master/kernel/trace/bpf_trace.c) — `trace_call_bpf`, the preemption-disabled kprobe/tracepoint contrast.
- [Commit `7dadeaa6e851` ("sched: Further restrict the preemption modes")](https://github.com/torvalds/linux/commit/7dadeaa6e851) — removed `PREEMPT_NONE` / `PREEMPT_VOLUNTARY` on x86 and arm64, exposing the window.

## Community discussion today

A public bpf mailing-list RFC proposes disabling preemption around the trampoline recursion enter/exit on non-RT kernels so that per-CPU recursion tracking and per-CPU private stacks are not shared across preempted tasks. The series includes a 47-line reproducer and measured false recursion drop rates of 0 percent on voluntary-preemption kernels against 25-28 percent on full- and lazy-preemption kernels. An automated mailing-list review flagged only a comment-style nit in the patch. The discussion so far is narrow: the boundary is whether a preemption barrier in the hot trampoline path is acceptable on all non-RT kernels, and whether the private-stack ownership argument fully justifies keeping the per-CPU counter.

This run's archive-backed sources produced 0 messages in the 7-day window (the two opt-in archive sources were empty at snapshot time), so the question above was sourced from the permitted public bpf mailing list, reviewed through its public atom feed, and the RFC thread was used as the primary reference. The Discord channels for eunomia-bpf and sched-ext and r/eBPF were not covered this run (no visible-browser session was available), so they are marked not-covered rather than quiet.
