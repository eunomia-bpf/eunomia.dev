# Why does a tracepoint eBPF program attached through the Cilium eBPF Go library fire only on CPU 0 on a new kernel, when the identical program logged system calls system-wide on older distributions?

It is not a bug in the library and not something your code needs to work around by opening one perf event per CPU. `link.Tracepoint` opens exactly one tracepoint-type perf event and passes `pid == -1` with `cpu == 0` to `perf_event_open`, so the event is CPU-0 scoped. On a healthy kernel that single event still fires on every CPU, because tracepoint BPF programs are not CPU-filtered the way hardware PMU counters are. A recent kernel regression broke that guarantee on the syscall-tracing path, so a CPU-0-scoped event no longer delivered the cross-CPU events, and the program now fires only when its events land on CPU 0. The library is correct to open one event; the regression is fixed upstream in the kernel.

## The mechanism

The boundary is between the library's attach and the kernel's event delivery:

1. **Attach side (library).** `link.Tracepoint` resolves the event id under the tracefs `events` tree and calls `openTracepointPerfEvent` with the all-threads sentinel `perfAllThreads` (which is `-1`). In that function, when `pid` is the all-threads sentinel the local `cpu` is left at `0`, so the syscall is `PerfEventOpen` with `pid = -1`, `cpu = 0`, and a `PERF_TYPE_TRACEPOINT` attribute. One event, scoped to CPU 0.
2. **Delivery side (kernel).** The `perf_event_open(2)` contract says that with `pid == -1` and `cpu >= 0` the event measures all threads on that one CPU. So by the man page this really is a CPU-0 event. Historically that never limited a tracepoint BPF program, though: tracepoint BPF events are system-wide and do not honor a CPU filter (see iovisor/bcc issue 1490). One event attached on CPU 0 fired on every CPU, which is why the identical program logged all `sys_enter_open` calls on Debian 13/14 and Ubuntu 24.04.
3. **The regression.** On the new kernel (for example kernel 7.0 / Ubuntu 26.04) the syscall trace-event path that reads user-space syscall arguments can preempt and migrate the task, and the per-CPU perf event list head could point at the wrong CPU. A CPU-0-scoped event then stopped delivering the cross-CPU events, and the program now fires only when its events land on CPU 0. The library is correct to open one event; the regression is fixed upstream in the kernel.
## Verification and debugging path

1. **Confirm the attach is one event.** `strace` the attach: a single `perf_event_open` with `pid = -1, cpu = 0`. Contrast with `perf stat -a`, which opens one `perf_event_open` per CPU — that difference is why the library and `perf` look different on the wire.
2. **Reproduce the scope.** Run a tight `open()` loop pinned off CPU 0 (`taskset -c 1,15 ./prog`): no lines in `trace_pipe` on the affected kernel; pin onto CPU 0 (`taskset -c 0`) and the lines appear. This is the whole diagnosis.
3. **Rule the library out.** The same `cilium/ebpf` build behaves identically on Debian 13/14 and Ubuntu 24.04; only the new kernel differs, and the library deliberately opens one event. The regression is not in the attach.
4. **Confirm the kernel regression and its fix.** The maintainers trace it to a stale per-CPU head pointer on the syscall trace event path, corrected upstream by kernel commit `2c2b322acdcc` ("tracing: perf: Fix stale head for perf syscall tracing", which fixes the earlier change `edca33a56297d`). On a kernel at or after that fix, or a distribution that backported it, the program fires on all CPUs again. Diagnose on an upstream kernel so vendor-kernel noise does not muddy the result.

## The limitation

- The behavior is kernel-version-dependent. Healthy kernels fire system-wide; a specific window of kernels fires CPU-0-only. "It works on my machine" is real here, not a code defect.
- Opening one CPU-0 event is the correct library design for tracepoints; it only becomes wrong on the affected kernel, because the kernel's "tracepoint events are system-wide" guarantee regressed. Looping your own `perf_event_open` over every CPU is a workaround, not a general requirement, and the kernel backport is the real remedy.
- Vendor kernels matter: a distribution may or may not have backported the fix. Test the actual kernel and reproduce on an upstream kernel.
- The `perf_event_open(2)` wording about CPU scoping describes the event's scoping contract; it does not mean a correctly attached tracepoint program should observe only one CPU on a fixed kernel.

## References

- [cilium/ebpf link/tracepoint.go (v0.22.0)](https://github.com/cilium/ebpf/blob/v0.22.0/link/tracepoint.go) - the `Tracepoint` helper and its single-event attach.
- [cilium/ebpf link/perf_event.go](https://github.com/cilium/ebpf/blob/v0.22.0/link/perf_event.go) - `openTracepointPerfEvent`, `perfAllThreads`, and the `perf_event_open` call with `cpu = 0`.
- [iovisor/bcc issue 1490](https://github.com/iovisor/bcc/issues/1490) - tracepoint BPF events are system-wide and do not honor a CPU filter.
- [kernel commit 2c2b322acdcc](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/commit/?id=2c2b322acdcc78575b8d6afa64a085cf92e03c12) - "tracing: perf: Fix stale head for perf syscall tracing".
- [cilium/ebpf discussion 2126](https://github.com/cilium/ebpf/discussions/2126) - the practitioner question this page answers.

## Community discussion today

Two opt-in archive channels (the two CNCF OpenTelemetry instrumentation channels) produced four messages this run. The thread this page answers: a practitioner cross-posted an open question from the Cilium eBPF Go library asking why a `link.Tracepoint` program only logged calls attributed to CPU 0 on a new distribution, and how to monitor a tracepoint system-wide. The archive carried only the pointer to that public discussion, so the substance of the question and the maintainers' diagnosis are reproduced here in anonymized form. The other two threads in the window had already been answered by earlier posts - the available-skills dedicated-attribute modeling question and the model-serving-signals / GenAI metric-catalog boundary question - so this run selected the non-duplicative Cilium tracepoint question.

Channel coverage for this run: the two opt-in archives supplied four messages. The visible-browser-only sources - the eunomia-bpf and sched-ext Discord servers, the bpf mailing list, and r/eBPF - could not be reviewed this run because no visible-browser session was available, so they are marked not-covered, not quiet.
