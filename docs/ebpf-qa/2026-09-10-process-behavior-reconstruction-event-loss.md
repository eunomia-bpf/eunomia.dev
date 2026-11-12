# What eBPF/kernel limitations should a process behavior reconstruction tool account for in its architecture?

**Short answer:** two of them, and both are architectural rather than tuning issues:

1. **Event delivery is ordered but not complete.** The kernel's BPF output paths — the per-CPU perf event array and the shared BPF ring buffer — *silently drop* events when a buffer is full. The perf buffer at least attempts a `PERF_RECORD_LOST` record, and only "when possible"; the ring buffer writes no loss record at all. A reconstructed timeline is therefore a *lower bound* on what the process actually did, not a complete record — unless the tool measures loss itself.
2. **`pid` is not a stable identity.** PIDs are assigned per namespace (a new PID namespace starts at 1), are unique only within that namespace, and are allocated cyclically — they are reused as tasks exit. A correlation engine keyed on pid alone silently merges two unrelated processes, or attributes an event to the wrong one.

The other limitations (attach availability, tracepoint coverage, CO-RE drift between kernel versions) determine *which* events you can capture; these two determine the completeness and correctness of everything you do capture.

## Why the kernel cannot promise a complete event stream

A process behavior reconstruction tool is, at its core, an event consumer: it attaches BPF programs to the hooks it cares about (the clone/fork/exit family, `execve`, `openat`, `socket`/`connect`, …) and each hook emits a record into one of two kernel output channels:

- **Per-CPU perf event array** (`bpf_perf_event_output()`): each CPU gets its own mmap'd ring, set up through `perf_event_open(2)`, consumed separately in userspace.
- **Shared BPF ring buffer** (`bpf_ringbuf_output()` / `bpf_ringbuf_reserve()`, map type `BPF_MAP_TYPE_RINGBUF`): a single ring shared by all CPUs. The kernel's ring-buffer design document exists precisely because the perf buffer "fails" at two things: efficient memory use across CPUs, and "preserving ordering of events that happen sequentially in time, even across multiple CPUs (e.g., fork/exec/exit events for a task)" — that parenthetical is exactly the workload of a process behavior reconstruction tool.

Neither channel gives you completeness. On the perf-buffer side, `perf_event_open(2)` is explicit: when the consumer cannot keep up, the kernel simply discards samples, which "are considered lost, and cause a PERF_RECORD_LOST sample to be generated when possible." Read that qualifier: even the loss notification is best-effort. The lost count can be read when supported (`PERF_FORMAT_LOST`, since Linux 6.0), and libbpf's `perf_buffer__new()` accepts a `lost_cb` that fires "when record loss has occurred" — that callback is your only signal, and it only fires if the kernel managed to write the record.

The ring buffer is cleaner but gives you less: its design document states the overflow rule in one line — "if there is no more space left in ring buffer, reservation fails, no blocking." A full ring means `bpf_ringbuf_reserve()` returns `NULL` and `bpf_ringbuf_output()` returns 0, and the event is gone with no loss record, no notification, and nothing in the stream that marks a gap. The `bpf_ringbuf_query()` helpers (`BPF_RB_AVAIL_DATA` and friends) are documented as "momentarily snapshots" for "debugging/reporting reasons" — a heuristic, not a loss counter. (Recent kernel work adds an opt-in overwrite mode in which a new event replaces the oldest one; that trades silent drops for silent *overwrite* — still undetectable from the stream alone.)

The architectural consequence: **completeness is a property your tool must manufacture.** Size the buffer for peak, not average, event rate; increment a counter in the BPF program on every event you attempt to emit; compare the producer-side counter against the consumer-side count at shutdown or on an interval; when the two disagree, mark that window "possibly incomplete" instead of presenting it as fact. Without that, the tool will report a gap as if it were an absence.

## Key tasks by start time, not by pid

The correlation engine is the second silent failure mode, and it is documented in the same primary sources:

- `pid_namespaces(7)`: "PIDs in a new PID namespace start at 1 … and calls to fork(2), vfork(2), or clone(2) will produce processes with PIDs that are unique within the namespace." Uniqueness is a per-namespace property.
- The kernel's PID allocator (`kernel/pid.c`) assigns ids cyclically from the namespace's range: after the top of the range, allocation wraps back, so a PID that was live ten minutes ago can be handed to a new task.

Consequences for a reconstruction tool:

- A PID you observed at time `t` can belong to a *different* task at `t + 1s` — the task that held it exited and the allocator recycled the number.
- In containerized deployments the same task has different PIDs in each namespace level; a tracer attached in the host namespace and one inside the container see different ids for the same task. Neither is wrong, but they cannot be joined on pid alone.

Key your correlation objects on **(pid, task start time)**, or assign a tool-generated id at first sight of the task (capture the start time from the first event you see, or read it from `/proc`). PID reuse and namespace differences then become a display detail instead of a correctness bug. (This is a different boundary from the connection-attribution problem for multiplexed network traffic — see the [2026-08-31 entry](/ebpf-qa/2026-08-31-tls-http2-sse-connection-correlation/) — where even a stable pid cannot attribute a flow across shared sockets; here the identity itself is unstable.)

## How to verify

1. **Ring-buffer loss demo.** Attach a small program to a high-frequency tracepoint; on each hit, increment a counter map and attempt to emit a record into a deliberately small ring buffer (4 KiB will do). Drain the consumer slowly. Expect the producer counter to exceed the consumed record count, and `bpf_ringbuf_reserve()` to return `NULL` / `bpf_ringbuf_output()` to return 0 when the buffer is full — the documented baseline. If your target kernel has the overwrite mode, records are *replaced* instead: the stream still looks complete, which is the failure the opt-in flag changes, not one it cures.
2. **Perf-buffer loss demo.** Same shape, with `bpf_perf_event_output()` and libbpf's `perf_buffer` with a `lost_cb`. Under burst the lost callback fires with a count; when it does not fire while the consumer is observably slow, the "when possible" clause has already applied — treat the window as lost.
3. **Identity demo.** Run your tracer inside a PID namespace (`unshare --pid --fork`, with a fresh `/proc` if you need to inspect it) and watch PIDs restart at 1 for the same set of tasks; then spawn a burst of short-lived processes on a quiet host and observe a recycled PID carrying a different comm and start time inside the same capture window.
4. **For "which events are essential":** start from the fork/exec family (`clone`, `fork`, `exit`, `execve`), `openat`, and `socket`/`connect` for network actions — but treat that list as a design input, not a kernel guarantee. The tool can only correlate what the target kernel's hooks expose, so probe the target before committing to an event schema: available tracepoints, CO-RE relocation success, and kprobe attachability all decide what your event table can contain.

## Where the answer stops applying

- This covers the *output* path — how captured events reach userspace. It does not cover attach availability (kernel version, configuration, architecture, whether the hook you need exists and relocates), which determines which events exist at all.
- The loss semantics cited are for the standard `BPF_MAP_TYPE_RINGBUF` and perf event array behavior. User-ring-buffer variants and the proposed overwrite mode change overflow behavior, so check the target kernel before quoting guarantees.
- A reconstructed timeline is a lower bound on *observed* behavior. For forensic-grade claims, cross-check suspected loss windows against independent evidence (audit logs, captured traffic, process accounting) — and the tool should say so explicitly instead of smoothing the gap.
- The pid-recycling boundary assumes the standard allocator; a small `pid_max` makes recycling much faster, which makes start-time keying more important, not less.

## References

- Linux kernel documentation: [BPF ring buffer](https://docs.kernel.org/bpf/ringbuf.html) — shared MPSC design; "reservation fails, no blocking"; `BPF_RB_*` as debugging-only snapshots
- [perf_event_open(2)](https://man7.org/linux/man-pages/man2/perf_event_open.2.html) — mmap ring layout; `PERF_RECORD_LOST`; discarded samples "considered lost … when possible"; `PERF_FORMAT_LOST`
- libbpf API, `tools/lib/bpf/libbpf.h` in the kernel tree — `perf_buffer__new()` with `lost_cb` ("called when record loss has occurred") and `ring_buffer__new()` without any loss callback, because there is no loss record to deliver
- [pid_namespaces(7)](https://man7.org/linux/man-pages/man7/pid_namespaces.7.html) — PIDs start at 1 in a new namespace; uniqueness is per-namespace
- Kernel PID allocator, `kernel/pid.c` — cyclic allocation that wraps back after the top of the namespace's range
- LWN: [Make BPF ring buffer over writable](https://lwn.net/Articles/904407/) and [Add overwrite mode for bpf ring buffer](https://lwn.net/Articles/1032293/) — baseline behavior ("when BPF ring buffer are full … calling `bpf_ringbuf_reserve()` from eBPF code returns NULL") and the proposed overwrite semantics

## Community discussion today

Coverage this run: one of the two archive-opted-in Slack workspaces was readable — the strict 24-hour window was quiet and the seven-day fallback window held exactly one message, the research-scoping question already answered in the [2026-09-08 entry](/ebpf-qa/2026-09-08-malicious-ebpf-payload-boundaries/). The other opted-in workspace's allowlisted channels were absent from its archive, recorded as inaccessible rather than quiet. The two allowlisted Discord workspaces are visible-browser-only surfaces and no visible browser session was available for them this run, reported here as a coverage gap rather than silence. The public bpf@vger.kernel.org archive and the public practitioner forum were reviewed through their ordinary public pages.

### The question that became today's answer

The strongest unresolved practitioner question on the public forum this week came from an engineer building a process-behavior reconstruction tool on eBPF/CO-RE — correlating raw syscall events (`execve`, `openat`, `socket`/`connect`, `clone`) into a behavior timeline, with a versioned capture/replay format — who asked, among other things, which eBPF/kernel limitations to build into the architecture, which events are essential, and which correlations are actually useful during investigation. The thread carries no substantive technical answer yet (a single reply, in jest), so the answer above is published here: delivery is ordered but not complete, pid is not a stable key, and the tool must manufacture both properties itself.

### The kernel mailing list: policy objects, verifier scaling, reclaim

The public bpf archive today was patch-level work with three substantive threads:

- **Security policy as a BPF object.** A 15-patch bpf-next series (v3, ~30 messages) makes Landlock rulesets applicable from BPF through an LSM "policy object" and a family of `bpf_lsm_policy_*` kfuncs. The kernel is turning confinement policy into something a loaded program can manage, which raises the stakes for the visibility boundary discussed in the [2026-09-08 entry](/ebpf-qa/2026-09-08-malicious-ebpf-payload-boundaries/).
- **Verifier scaling.** A bpf series (6 patches, 11+ messages) fixes a quadratic successor rescan in `bpf_compute_scc`, bounds the number of indirect jump edges per program, and caches subprogram jump tables. Verifier runtime is a practical ceiling on what an operator will load, and worth tracking for anyone running large policy programs.
- **Memory management from BPF.** A bpf-next series (v9) adds a `bpf_proactive_reclaim` kfunc for BPF-driven proactive memcg reclaim, alongside a 2-patch series preserving escaped dynptr slice ancestry across subprogram returns. The BTF inline-function location series (v2) extending libbpf's `LOC` kinds continues the same kernel-ABI movement.

Also in the queue: a soft-lockup fix in the hash-map batch delete path, a sockmap fix for `copied_seq` double-counting on self-redirect, and a `bpf_fib_lookup()` neighbor read fix. Net: the capability surface keeps expanding into security policy and memory management while the verifier and map machinery get their background scaling work.

### The practitioner forum

The two-part public replatforming series continued, its second part detailing eleven Linux-ecosystem gaps that a large operator closed with eBPF; no discussion had started as of this writing. The thesis-scoping thread from earlier this week — the one that produced the 2026-09-08 entry — remained the most-discussed security thread, and the behavior-reconstruction question above was the one open architecture question in the window.
