# Why do BPF task iterators split between sleepable and non-sleepable programs, and which process views fall on each side?

Short answer: the boundary people usually state — non-sleepable programs for the exe view, sleepable programs for the cmdline view — inverts the actual split. RESCHED is a permission, not a sleepability gate: all three task-family seq iterators set it, and it only authorises the iterator to call cond_resched() between objects. The one attach-time rejection is one-directional: a sleepable program on a target without the bit is refused, while a non-sleepable program attaches to task, task_file, and task_vma without complaint. The process views split by where their data lives, not by whether the program sleeps: comm is a lockless 16-byte in-struct field, exe is an RCU-protected file pointer in mm_struct readable in either run context, and cmdline is a user-memory block that is copied, never dereferenced. The view that genuinely couples to the sleepable execution path is the task_vma walk, whose seq loop holds the mm's mmap read lock across vmas and drops it before returning to user space; the new kfunc task iterator walks the task list with no mmap lock at all and is usable from non-sleepable programs.

## The mechanism

**RESCHED is a feature bit, not a sleep gate.** Every task-family seq iterator registers it:

```c
static struct bpf_iter_reg task_reg_info = {
	...
	.feature		= BPF_ITER_RESCHED,
	...
};
```

and the same bit appears on the task_file and task_vma registrations, plus on the cgroup iterator. What the bit actually does lives in the custom seq read loop, bpf_seq_read, which runs while holding the seq file's mutex:

```c
can_resched = bpf_iter_support_resched(seq);
while (1) {
	...
	if (num_objs >= MAX_ITER_OBJECTS) { ... }
	...
	if (can_resched)
		cond_resched();
}
```

So RESCHED makes the read loop voluntarily reschedulable between objects, with a per-read cap of MAX_ITER_OBJECTS, one million objects; when a single read fills the buffer before hitting the cap, the userspace reader loops on -EAGAIN until EOF. Nothing in that path requires, or grants, program sleepability.

**The sleepable flag belongs to the program, and the attach gate is one-directional.** A program becomes sleepable through the BPF_F_SLEEPABLE load flag, which the verifier records on the program class and restricts its map and helper usage to match. The iterator attach path then checks exactly one condition, in bpf_iter_attach_iter:

```c
/* Only allow sleepable program for resched-able iterator */
if (prog->sleepable && !bpf_iter_target_support_resched(tinfo))
	return -EINVAL;
```

The consequence is asymmetric: a non-sleepable program attaches to any RESCHED target, including all three task-family iterators, so there is no sleepability requirement on the exe, cmdline, or comm views. A sleepable program, in turn, may only attach to targets that advertise the bit; attaching it to a target without RESCHED fails at link creation.

**The views split by data location.** comm is not a separate lookup: it is a 16-byte char array inside task_struct, guarded by nothing, so a program running on the task iterator reads it locklessly with no mm access at all. exe is the mm_struct field exe_file, declared with the RCU marker:

```c
/* store ref to file /proc/<pid>/exe symlink points to */
struct file __rcu *exe_file;
```

Both execution contexts of bpf_iter_run_prog take an RCU read lock before running the program, so the pointer is legitimately readable from either side; turning the pointer into a printable path string is a separate step performed through the kernel's file path helpers, not by the pointer read itself. cmdline does not live in task_struct or mm_struct at all. What the kernel exposes is the user-space argv block, the mm_struct pair arg_start and arg_end, written at exec time and guarded by arg_lock for writers; the BPF side copies it out with the user-memory read primitives, bpf_probe_read_user and bpf_probe_read_user_str. Those are copy primitives that also work in non-sleepable contexts, so the cmdline read does not by itself force sleepable-ness. What it does force is that the data lives in another process's address space, which is why it must be copied rather than dereferenced.

**The task_vma walk is the view that couples to sleepability.** Its seq loop takes the mm's mmap read lock while it walks a task's vmas, and it deliberately drops that lock before returning to user space so a long read loop never holds it across the userspace boundary; on the next read it re-acquires and re-positions via find_vma. Where the lock is contended the walk drops and retakes it, using the killable variant so an interrupting signal can kill a stuck read:

```c
if (mmap_lock_is_contended(curr_mm)) {
	info->prev_vm_start = curr_vma->vm_start;
	info->prev_vm_end = curr_vma->vm_end;
	op = task_vma_iter_find_vma;
	mmap_read_unlock(curr_mm);
	if (mmap_read_lock_killable(curr_mm)) {
		mmput(curr_mm);
		goto finish;
	}
}
```

and task_vma_seq_stop releases the lock before handing control back to the reader. The sleepable execution path is where the sleeping side of that lock actually lives:

```c
if (prog->sleepable) {
	rcu_read_lock_trace();
	migrate_disable();
	might_fault();
	...
}
```

The kernel-side counterpart is bpf_find_vma, which pins the task's mm under the task's alloc_lock, takes the mmap read lock with a trylock, and on contention returns -EBUSY rather than blocking, deferring the unlock to IRQ work through a per-CPU helper so it never sleeps in a context that cannot sleep.

**The kfunc task iterator carries no mmap lock.** A separate, newer entry point wraps the same task list walk in an opaque handle:

```c
struct bpf_iter_task {
	__u64 __opaque[3];
} __attribute__((aligned(8)));
```

bpf_iter_task_new returns the handle, bpf_iter_task_next walks __next_thread for the thread case and next_task for the process case, and bpf_iter_task_destroy closes it. Because the walk itself touches no mm and no mmap lock, the whole pair is usable from non-sleepable programs.

The practical rule that falls out of all of this: a task-iterator program that does more than trivial per-object work belongs on the sleepable side, and the one task-family view that forces that choice is the vma walk, because its kernel loop holds and retakes a sleep-sensitive lock. comm, exe, and cmdline impose none of that; their constraints come from where the data lives, not from whether the program may sleep.

## Verification and debugging path

1. Dump the BTF of the target kernel, `bpftool btf dump file /sys/kernel/btf/vmlinux | grep -E "comm|arg_start|arg_end|exe_file"`, and confirm the field layout matches: a 16-byte comm array in task_struct, the arg range pair in mm_struct, and the exe_file pointer with its RCU marker.
2. Build the attach matrix. Load one program body as both a sleepable and a non-sleepable variant, attach both to task, task_file, and task_vma; both should succeed. Then attach the sleepable variant to a target registered without the RESCHED bit and confirm the link creation fails with -EINVAL, which is the one-directional gate in action.
3. Exercise the read-loop contract. Run the iterator against a large table and confirm that userspace loops on -EAGAIN until EOF, and that a task_vma resume after the buffer fills re-enters the walk without duplicating or skipping a vma.
4. Cross-check the views against ground truth. Dump the (pid, comm, exe) tuple from BPF and compare it against the argv bytes read from the arg range with bpf_probe_read_user_str; the BPF-side argv must match what the kernel's own /proc data reports for the same process.

A minimal BPF-C read of all three views on the task iterator:

```c
SEC("iter/task")
int read_views(struct bpf_iter__task *ctx)
{
	struct task_struct *task = ctx->task;
	char comm[TASK_COMM_LEN];
	bpf_probe_read_kernel(comm, sizeof(comm), &task->comm);

	struct mm_struct *mm = task->mm;
	if (!mm)
		return 0;

	struct file *exe;
	bpf_probe_read_kernel(&exe, sizeof(exe), &mm->exe_file);

	char cmdline[256] = {};
	if (bpf_probe_read_user_str(cmdline, sizeof(cmdline),
	    (void *)mm->arg_start))
		cmdline[0] = 0;

	/* key = pid, value = { comm, exe pointer, argv } */
	return 0;
}
```

## The limitation

- The mmap read lock coupling is specific to the task_vma seq walk. No task-family kfunc in this mainline snapshot takes the mm lock as a program-callable lock/unlock pair; the walk acquires it internally with the killable lock, and bpf_find_vma wraps the same lock in a trylock with an IRQ-work deferred unlock.
- exe is a pointer read. Resolving mm's exe_file into a path string is a separate step from the pointer read itself; the RCU marker only guarantees the pointer read is safe in both run contexts.
- Kernel threads have no mm_struct, so the argv range and the exe pointer are absent. The task iterator still passes the task_struct pointer, so a program must null-check task->mm before deriving either user-memory view.
- One million objects per read is a per-buffer cap, not a total. A userspace reader must expect -EAGAIN and loop until EOF, and the task_vma re-entry after a buffer fill is what keeps the vma walk consistent across that boundary.

## References

- Task-family seq iterators, the mmap lock walk, the kfunc task iterator, and bpf_find_vma: https://elixir.bootlin.com/linux/latest/source/kernel/bpf/task_iter.c
- The seq read loop, RESCHED permission, one-directional sleepable attach gate, and run paths: https://elixir.bootlin.com/linux/latest/source/kernel/bpf/bpf_iter.c
- BPF_ITER_RESCHED feature bit and the iterator registration structure: https://elixir.bootlin.com/linux/latest/source/include/linux/bpf.h
- mm_struct arg range fields and the RCU-marked exe_file pointer: https://elixir.bootlin.com/linux/latest/source/include/linux/mm_types.h
- task_struct comm array and TASK_COMM_LEN: https://elixir.bootlin.com/linux/latest/source/include/linux/sched.h
- bpf_probe_read_user and bpf_probe_read_user_str prototypes: https://elixir.bootlin.com/linux/latest/source/kernel/bpf/helpers.c
- bpf_link and the iterator link type: https://man7.org/linux/man-pages/man7/bpf_link.7.html

## Community discussion today

Coverage: **2** watchlist-opted-in Slack archives were reachable via the read-only snapshot, **4 messages** in the rolling window; **1** was a repository-internal bug report and was discarded under anonymization, and **1** was non-technical. The browser-only communities in the allowlist had no visible-browser session available this run and are marked **unavailable, not quiet**. No Slack/Discord workspace, channel, participant, or message URL is reproduced below.

Two threads from the reachable window:

- **Who actually measured the tracepoint-versus-LSM hook cost gap?** A thread about a tracepoint-only endpoint agent kept assuming the hook cost was negligible compared with an LSM-based design. The open question is a differential cost study on one fixed workload: same events, both hook classes, wall time per event on the hot path.
- **Interval /proc polling is nearing the end of its useful life on big hosts.** On a large multi-socket machine a full /proc trawl takes multiple seconds, while an eBPF task iterator closes it in well under a second, and the thread recommended the iterator as the polling replacement. An adjacent claim in that thread said the exe view needs a non-sleepable program while the cmdline view needs a sleepable one. That framing is inverted: the exe view is the lockless, RCU-readable one, the cmdline view is a user-memory read, and the only task-family view with genuine mmap-lock coupling is the vma walk. The corrected boundary is what this page answers.
- **Socket attribution through the sock:inet_sock_set_state tracepoint**, carried over from yesterday: connect/accept probes miss sockets that changed owner through socket activation or an SCM_RIGHTS handoff, TIME_WAIT sockets revive under a new owner, and UDP is not strictly a connected protocol at all.

The 4-message reachable window is the full set for today; the browser-only communities remain unreviewed rather than assumed silent.
