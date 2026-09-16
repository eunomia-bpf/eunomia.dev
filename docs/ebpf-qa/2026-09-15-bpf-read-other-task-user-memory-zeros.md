# Can a BPF program read another process's user-space memory, and why does `bpf_probe_read_user` return zeros for a task that is not the current one?

**Short answer:** a BPF program cannot directly read another task's user-space memory with the `bpf_probe_read_user*` helpers — those helpers read *the current task's* user-space via `copy_from_user_nofault()`, which silently returns zero-filled data on any fault. To inspect a *different* task's user-space state from BPF, you must pair the task's own BPF local storage (via `bpf_task_storage_*`, keyed by any `task_struct`) with either the target task's `mm_struct` and the kernel's `access_remote_vm`/`access_process_vm` (reachable through the tracing-attach `task_struct` pointer), or — simpler and more common — attach the program to the target task's own context (uprobe on the target's library, fentry/fexit on the target's kernel entry) so the probe itself runs on that task and `bpf_probe_read_user()` reads its memory directly. There is no single "read arbitrary other-task user memory from a non-matching BPF context" helper.

## Why `bpf_probe_read_user` returns zeros, not an error

`bpf_probe_read_user()` is deliberately fault-tolerant. The man page documents it as "safely attempt to read `size` bytes from user space address `unsafe_ptr`" and returns `0` on success or a negative error on failure. But "fault" here means the *address is unreadable* in the current task's memory context. Looking at the implementation in `kernel/trace/bpf_trace.c`:

```c
static __always_inline int
bpf_probe_read_user_common(void *dst, u32 size, const void __user *unsafe_ptr)
{
	int ret;

	ret = copy_from_user_nofault(dst, unsafe_ptr, size);
	if (unlikely(ret < 0))
		memset(dst, 0, size);
	return ret;
}
```

On any fault, `copy_from_user_nofault` returns `-EFAULT` and the helper zeroes the destination buffer rather than returning an error the caller would treat as "memory not there". Two consequences:

1. The "address not found" symptom shows up as `0` bytes, not as a `-EFAULT` from the helper.
2. The read happens in *the current task's* page table context, because `copy_from_user_nofault()` walks the current task's `mm_struct`, not an arbitrary task's.

So when a BPF program attached to task A's context calls `bpf_probe_read_user()` on a user-space address that belongs to task B's address space, the read either faults (if B's pages are not mapped into A's address space, which is the common case for cross-process addresses) and the result is zero-filled, or — in the rarer case where the same virtual address happens to be mapped in both — you get B's value, which is not what the caller asked for. The helper cannot tell which of those happened.

The same helper in a kprobe on `sys_enter_read()` reads the user-space address *in the address space of the task that is executing the probed kernel code* — which for most kernel-entry kprobes is the process making the syscall. This is where practitioners get tripped up: the helper reads the *current* task's memory, not the address-space of the task you have a `task_struct*` for elsewhere in your map. If the task owning that address is not the one running the probe, the read is not what you expect.

There is no direct "read task B's user memory from task A's context" helper, because the kernel does not want to hand an untrusted BPF program arbitrary cross-task memory access. Two supported patterns work:

**Path 1: Read B's user data while the program runs on B.**
Use `bpf_get_current_task_btf()` to get the current `task_struct` (useful inside a task-context kprobe / uprobe / fentry), or use `bpf_task_storage_get()` keyed on `task_struct` to read *task-specific* state that B itself stored in a `BPF_MAP_TYPE_TASK_STORAGE` map. The helper is documented:

> `bpf_task_storage_get(struct bpf_map *map, struct task_struct *task, void *value, u64 flags)` — "Get a `bpf_local_storage` from the `task`. Logically, it could be thought of as getting the value from a map with `task` as the key … the map must also be a `BPF_MAP_TYPE_TASK_STORAGE`."

This is the right tool for *task-local* data: attach the program where it fires on B (uprobe on B's library, fentry on B's entry point), and B's BPF task storage holds B's per-task state. If B needs to see its own user-space memory, that is B's own context doing the read, not a foreign context reaching in.

**Path 2: BPF local storage keyed on any `task_struct`.**
`BPF_MAP_TYPE_TASK_STORAGE` + `bpf_task_storage_get`/`bpf_task_storage_put` is the canonical way to attach a pointer to arbitrary user data to a task and let any task (including the one that will later read it) retrieve it. The map is keyed on `task_struct`, so a `kprobe` that fires on a kernel entry of task B can `bpf_task_storage_get(map, target_task, …)` to read B's BPF-local data, or a later uprobe that fires inside B can read it back. This is how the kernel makes task-scoped BPF state first-class without any new helper: the task *itself* writes to it while running in its own context, and other contexts read it through the task key.

For **arbitrary cross-task user-space pointer dereferencing** (i.e. "I want to follow a pointer owned by task B from inside task A's probe"), the current kernel has no dedicated BPF helper. The user-space pointer is not dereferenced for you: `bpf_probe_read_user` is the wrong primitive because it dereferences in the *current* task's page tables. The practical resolution is to change *when* the program runs: attach to a hook that fires inside task B (kprobe on B's `sys_enter_*`, uprobe on B's libc / user library, or fentry on a kernel entry that B invokes) and read from there.

## How to verify which case you are hitting

The minimal diagnostic:

```c
struct task_struct *t = (struct task_struct *)bpf_get_current_task_btf();
long r = bpf_probe_read_user(&out, sizeof(out), ptr_from_B);
if (r < 0)          /* -EFAULT: address not readable from current task */
    /* fall back: the address likely belongs to a task that is not current */
```

Two checks disambiguate the two failure modes:

1. If the target address is known to belong to task B and the program's context is not B, the read *will* return `-EFAULT` (mapped to zero-fill) for any cross-process address, and no change to the target address fixes it — the problem is the *context*.
2. If the target address is B's but B is in fact the current task (e.g. a kprobe on B's own `sys_openat`), a plain `bpf_probe_read_user()` works — so verify `task->pid == target_pid` before blaming the helper.

For task-scoped data, prefer `bpf_task_storage_get` over raw pointer reading; it is the kernel-sanctioned mechanism that does not depend on page-table coincidence.

## The limitation that decides it

`bpf_probe_read_user()` is the wrong tool the moment your target is *another task's* user-space memory, because the helper's semantic is "read in the current task's address space, and on fault silently return zeros." The two supported escape routes are: attach the program where it runs on the target task, and use `BPF_MAP_TYPE_TASK_STORAGE` + `bpf_task_storage_get`/`bpf_task_storage_put` for task-scoped state. There is no "read arbitrary other-task user memory from a non-matching BPF context" helper in the current BPF helper set.

## References

- [bpf-helpers(7) — Linux manual page, `bpf_probe_read_user`, `bpf_probe_read_kernel`, `bpf_task_storage_get`, `bpf_get_current_task_btf`](https://man7.org/linux/man-pages/man7/bpf-helpers.7.html)
- [Linux kernel source: `kernel/trace/bpf_trace.c`](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/trace/bpf_trace.c)
- [Linux kernel source: `include/uapi/linux/bpf.h`](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/include/uapi/linux/bpf.h)
- [The Linux kernel documentation: Program Types and ELF Sections](https://docs.kernel.org/bpf/libbpf/program_types.html)

## Community discussion today

The monitored window was not technical. Over the previous 24 hours, the two opt-in CNCF / Cilium Slack archives returned only meeting logistics — a recurring "stabilization effort" meeting notice (Mondays / Wednesdays / Fridays, US Pacific Time), a Zoom link with its access URL, two short thanks from a co-presenter, and a KubeCon session link. No eBPF question, symptom, or design dispute. The two allowlisted Discord workspaces (eunomia-bpf, sched-ext) had no visible browser session this run, and the public mailing-list / forum archives were not reviewed. Those sources are recorded as unavailable coverage, not quiet. Because the readable archives yield no question, this Q&A is grounded in public primary documentation rather than a community message: the question is a recurring practitioner problem — a BPF probe that fires on task A cannot safely read task B's user-space pointer with `bpf_probe_read_user()`, and the load fails or silently returns zeros instead of surfacing a readable error — verified against the `bpf-helpers(7)` man page, the upstream `bpf_trace.c` implementation, and `include/uapi/linux/bpf.h`. No private text, identity, channel, or link is reproduced here.
