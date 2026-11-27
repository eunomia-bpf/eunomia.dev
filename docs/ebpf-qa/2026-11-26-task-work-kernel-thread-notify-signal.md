# Why does scheduling a BPF task-work callback for a kernel thread make its interruptible sleeps return at once?

**Short answer:** because the signal-flavoured scheduling kfunc notifies the
target through a thread flag that only a return to user mode clears. The kfunc
queues the callback with the signal notification mode, and that mode sets
`TIF_NOTIFY_SIGNAL` on the target thread. The only places that clear the flag
are the signal path that runs on the way out to user mode — at the top of
`get_signal()` and at the tail of `ptrace_stop()` — plus one hand-written clear
in the io_uring inflight wait; a kernel thread does none of them, so the flag
stays set for the rest of its life. `signal_pending()` counts that flag as a
signal, so every `TASK_INTERRUPTIBLE` sleep in the thread returns
`-ERESTARTSYS` without blocking, and `schedule()` leaves the task runnable.
The callback is not "pending but delayed" either: task work runs from those
same exit paths, so for a kernel thread it runs only at thread exit. The merged
fix rejects a kernel-thread target with `-EPERM` instead of pretending the
notification worked.

## The mechanism

### 1. What the kfunc actually does to the target

Both scheduling kfuncs funnel into one helper, which defers the real work to
`irq_work`, so that the call is legal from NMI context and from any program
type:

```c
static int bpf_task_work_schedule(struct task_struct *task, struct bpf_task_work *tw,
				  struct bpf_map *map, void *callback_fn,
				  struct bpf_prog_aux *aux, enum task_work_notify_mode mode)
{
	...
	irq_work_queue(&ctx->irq_work);
	return 0;
}

__bpf_kfunc int bpf_task_work_schedule_signal(...)
{
	return bpf_task_work_schedule(task, tw, map__const_map, callback, aux, TWA_SIGNAL);
}
```

The `irq_work` eventually calls `task_work_add(task, &ctx->work, TWA_SIGNAL)`,
and that is where the target is touched:

```c
	case TWA_SIGNAL:
		set_notify_signal(task);
		break;
```

`set_notify_signal()` does two things: it sets the thread flag, and it wakes
the task if it is currently in an interruptible sleep, so that the notification
is acted on promptly instead of at the next unrelated wakeup:

```c
static inline bool __set_notify_signal(struct task_struct *task)
{
	return !test_and_set_tsk_thread_flag(task, TIF_NOTIFY_SIGNAL) &&
	       !wake_up_state(task, TASK_INTERRUPTIBLE);
}
```

Note the shape: the flag is set once and never re-armed; the wakeup is what
makes it matter. Nothing in this path is specific to user tasks, and no caller
checks whether the target ever returns to user mode.

### 2. Where the flag is cleared

`clear_notify_signal()` has three call sites in the tree: at the top of
`get_signal()`, at the tail of `ptrace_stop()`, and by hand in the io_uring
inflight wait. `get_signal()` is called from
`arch_do_signal_or_restart()`, which the exit-to-user-mode loop runs when it
sees the flag:

```c
		if (ti_work & (_TIF_SIGPENDING | _TIF_NOTIFY_SIGNAL)) {
			futex_fixup_robust_unlock(regs);
			arch_do_signal_or_restart(regs);
		}
```

That loop runs on the way out of the kernel to user mode. A kernel thread never
takes that exit, so on that path the flag is unreachable. The remaining path is
the thread's death: `do_exit()` calls `exit_task_work()`, which is
`task_work_run()`, and that is when a queued callback finally executes. The
io_uring occurrence is instructive because the kernel had the same problem in
its own code and worked around it by clearing the flag explicitly, for a task
that may sit in kernel-side sleeps without returning to user mode:

```c
		if (test_thread_flag(TIF_NOTIFY_SIGNAL)) {
			clear_notify_signal();
			if (task_work_pending(current))
				set_notify_resume(current);
		}
```

### 3. Why an interruptible sleep stops sleeping

`signal_pending()` treats the notification flag as a signal, with one escape:

```c
	if (unlikely(test_tsk_thread_flag(p, TIF_NOTIFY_SIGNAL)) &&
	    likely(!(READ_ONCE(p->flags) & PF_NO_NOTIFY_SIGNAL)))
		return 1;
	return task_sigpending(p);
```

`signal_pending_state()` then turns that into "do not block" for any
interruptible wait, and both of the two places that matter consult it: the
completion wait checks it before it sleeps, and `schedule()` re-checks it while
blocking the task and puts the task back to `TASK_RUNNING` instead of
deactivating it.

So a wait written as "sleep until the completion is signalled, the kworker
does not take signals" becomes a busy loop the moment the flag is set. The
kernel's own thread creation loop has the same shape, which is what turns the
io_uring report below into a 100% CPU spin on top of the crash:

```c
	for (;;) {
		set_current_state(TASK_INTERRUPTIBLE);
		if (list_empty(&kthread_create_list))
			schedule();
```

### 4. The damage chain in the reported case

The consumer in the report is ring teardown:

```c
		/*
		 * This is really an uninterruptible wait ... But it's also run from
		 * a kworker, which doesn't take signals, so it's fine to make it
		 * interruptible.
		 */
	} while (!wait_for_completion_interruptible_timeout(&ctx->ref_comp, interval));
```

The comment states the invariant the notification breaks: a kworker that is
waiting for a reference count to drop to zero and switch to atomic mode is
taken to be unsignallable, so an interruptible wait is safe and avoids stuck-
task reports. With the flag set, the wait returns `-ERESTARTSYS` on its first
iteration, the loop exits before `ctx->refs` reaches zero, and
`io_ring_ctx_free()` runs `percpu_ref_exit()` while the atomic switch is still
in flight — which is exactly what the warning condition in the refcount code
describes:

```c
	if (percpu_count) {
		/* non-NULL confirm_switch indicates switching in progress */
		WARN_ON_ONCE(ref->data && ref->data->confirm_switch);
		free_percpu(percpu_count);
```

Freeing the per-CPU counter under a queued RCU callback that will still sum and
put it is the slab-use-after-free in the same report.

### 5. The fix, and why it is a rejection

The merged change is three lines in the shared scheduling helper, ahead of every
allocation and every refcount:

```c
	if (task->flags & PF_KTHREAD)
		return -EPERM;
```

`PF_KTHREAD` is set in `copy_process()` for `args->kthread` and cleared for
everything else, and `/proc/<pid>/status` exposes the same bit as `Kthread:`.
`-EPERM` is the same answer the older helpers give for a kernel-thread target:
`bpf_send_signal()` and `bpf_probe_write_user()` already refuse `PF_KTHREAD`
(and `PF_EXITING`) with it.

The choice to reject rather than repair follows from section 2: for a kernel
thread, a correctly working notification still would not run the callback
before exit, so "accepted but late" is indistinguishable from "accepted and
never run". Failing the call is the only answer a program can act on.

## Verifying it on a live kernel

1. Establish that the target is a kernel thread. `/proc/<pid>/status` carries
   `Kthread: 1` for such a task, and the scheduler abbreviations in brackets
   in `ps` are the same set. Workqueue workers are kernel threads too, a point
   `get_task_comm()` handles explicitly.
2. Do not blame a queued signal. `kthreadd()` marks every handler ignored on
   itself before it spawns anything, and children are created with a copied
   handler table, so a kernel thread has no handler to run; `SigPnd` and
   `ShdPnd` in `/proc/<pid>/status` would show a real pending signal. An
   `-ERESTARTSYS` from a kernel thread whose signal queues are empty points at
   the notification flag instead, because `ignore_signals()` only clears
   `TIF_SIGPENDING`.
3. Watch the task state, not the symptom. `__schedule()` runs the signal check
   before it dequeues the task; on a hit it writes `TASK_RUNNING` back and
   leaves the task on the runqueue, so `voluntary_ctxt_switches` in
   `/proc/<pid>/status` almost stops moving while the thread burns CPU inside a
   wait it believes is a sleep.
4. In a vmcore, read the flag directly. The notification is a thread flag, so
   a debugger session on the dump can list the tasks with `TIF_NOTIFY_SIGNAL`
   set and confirm one of them is a task with no user-mode state — the
   combination no signal can produce.
5. Read the kfunc's return value. Zero means the kernel accepted a
   kernel-thread target; `-EPERM` means the check is present.

## The limits

- `TWA_RESUME` is not equally broken, and not equally right. It sets
  `TIF_NOTIFY_RESUME`, which `signal_pending()` does not look at, so sleeps in
  the thread keep working; but the callback still waits for
  `resume_user_mode_work()` or for thread exit, so for a kernel thread it is
  just as late.
- The check is a rejection, not a repair. A work item queued on a kernel
  thread by any other `task_work_add()` caller still runs only at thread exit,
  and nothing in `task_work.c` requires the target to be a user task.
- `PF_NO_NOTIFY_SIGNAL` is available as an escape hatch, but it is a flag on
  `current` set through a save/restore guard, so a thread can suppress the
  wakeup for itself and for nobody else; it cannot make a remote target's
  notification harmless.
- Not every kernel-side worker is a kernel thread by this test. io_uring's own
  workers are created as user workers (the kernel-thread bit is cleared for
  them), and they run task work explicitly in their loop, so the new rejection
  does not and should not catch them.

## References

- [BPF mailing list: reject kernel threads in the task-work kfunc](https://lore.kernel.org/bpf/20261126053016.1145482-1-rhkrqnwk98@gmail.com/) — the report, the spin, and the refcount warning.
- [Commit in the BPF tree](https://git.kernel.org/bpf/bpf/c/3c625090b46b) — the three-line check and the `Fixes` tag naming the kfunc series.
- [Kfunc documentation](https://docs.kernel.org/bpf/kfuncs.html) — implicit-argument kfuncs, and the task-work declaration and usage example.
- [Task work](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/task_work.c) — `task_work_add()` and its notification modes, `set_notify_signal()` is in the header below, and `task_work_run()`.
- [Scheduler signal helpers](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/include/linux/sched/signal.h) — `clear_notify_signal()`, `set_notify_signal()`, the `PF_NO_NOTIFY_SIGNAL` guard, and `signal_pending()`.
- [Signal delivery](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/signal.c) — the two `clear_notify_signal()` sites, `ignore_signals()`, and `flush_signals()`.
- [Exit to user mode](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/entry/common.c) — the `_TIF_NOTIFY_SIGNAL` dispatch into `arch_do_signal_or_restart()`.
- [Thread creation](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/kthread.c) — the creation loop with its interruptible `schedule()`, and `ignore_signals()` at startup.
- [Process exit](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/exit.c) — `exit_task_work()` in `do_exit()`, the only path a kernel thread has.
- [Completions](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/sched/completion.c) — the interruptible wait and its `-ERESTARTSYS` return.
- [Core scheduler](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/sched/core.c) — `try_to_block_task()` and the signal check that keeps a task runnable.
- [io_uring](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/io_uring/io_uring.c) — the teardown loop and its interruptible completion wait, and the manual flag clear in the inflight wait.
- [Per-CPU refcount](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/lib/percpu-refcount.c) — the `confirm_switch` warning and the RCU callback that frees the counter.
- [Base helpers](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/trace/bpf_trace.c) — the `PF_KTHREAD` rejection already used by the signal and user-write helpers.
- [proc documentation](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/Documentation/filesystems/proc.rst) — the `Kthread` field and the context-switch counters in `status`.

## Community discussion today

Coverage for this run: two archive-opted Slack workspaces (three allowlisted
channels) were read through the approved read-only archive and held no message
in either the 24-hour or the 7-day window, so that side is quiet rather than
unread. The two allowlisted Discord workspaces (ten channels) could not be read
— the web client sat at its login screen — so their activity is a coverage gap,
not silence. The r/eBPF public page was read as an ordinary web page and did
carry a real technical question, noted below; the question chosen for this page
came from the public bpf mailing-list archive, which also supplies the rest of
this section's themes.

**A notification flag that only a user-mode return can clear.** The day's
anchor was the kernel-thread rejection above, and its value for a reader is the
mechanism rather than the three lines. Three conditions have to line up for the
crash: a target that is a kernel thread, a scheduling mode that sets a
thread flag, and an interruptible wait in that thread whose author relied on
the thread being unsignallable. Remove any one and nothing happens. That is why
the fix went to the shared helper instead of the io_uring wait: the invariant
"kernel threads do not take signals" is true of *signals* and false of the
task-work notification, and every interruptible wait in kernel context is a
candidate. The unresolved part is the other tool in the toolbox: the
`PF_NO_NOTIFY_SIGNAL` guard suppresses the wakeup for `current` only, so a
thread that decides to make itself unsignallable for a critical wait can do so,
but no task can do it on behalf of a target it is scheduling work onto.

**A socket stored in the destination-entry slot.** A second report on the same
day shows an eBPF-adjacent aliasing bug in sockmap. A stream parser program may
call the socket-redirect helpers, which store the target socket in the skb's
redirect field; that field is a union member over the same word as the
destination-entry pointer and refcount bits, and the parser path never cleared
it. The verdict path does clear it before running the program and re-derives the
verdict after, so the difference between the two paths is exactly the two lines
the fix adds: drop the dst before the parser runs, clear the redirect after it.
The symptom is a KASAN null-pointer dereference deep inside the per-CPU counter
code, reached from `dst_release()` during skb teardown, because the socket
pointer is treated as a `dst_entry`. The lesson generalises past this bug: any
code that hands an skb to a program and then lets the skb be freed on a path
that was not the one that ran the program has to decide who owns that word.

**Alignment, third revision.** The series that pads map layouts instead of
changing the verifier rule came back with a measured table this time: value
address modulo 8, per map type, on both word sizes, with lpm trie IPv4 and
devmap misaligned on 64-bit and the allocator-header cases misaligned on 32-bit.
It also states the pre-existing 32-bit hazard the reviewers found — an 8-byte
load of the 4-byte per-CPU pointer in the per-CPU pointer fixup — and leaves it
out of scope, which is the honest form of a partial fix. For a reader, the
transferable bit is that a layout change and a verifier rule change fail
differently: the layout change is testable with one assertion on the returned
address, the rule change is not testable at all without an architecture that
faults.

**Which functions a modify-return program may attach to.** A v2 series narrows
the set of functions the build tags as error-returning, excluding the ones BPF
itself depends on: functions taking BPF types, kfuncs and helpers, `bpf_` and
`__bpf_` symbols of vmlinux, and the code under the BPF directory and the two
JITs. It reports the count it removes (795 of about 14 400 on an arm64 build),
which is the useful part: tagging is not a property of a function's prototype
alone, it is a claim that the caller checks the return, and a verifier or a JIT
that consumes the tag can be modified by a program whose own return value the
verifier then re-checks. The boundary that stays open is call-site coverage:
tagged functions whose callers ignore the return value are still tagged.

**One recursion that a notrace attribute and a deny list both fix.** A
trampoline `fentry` program on an RCU unlock-special path can recurse through
the trampoline's own read-unlock until the stack guard page is hit. The
report's analysis is that ftrace, fprobe and kprobes take the recursion
trylock while the BPF trampoline does not, so the BPF-specific fix is a deny
list; the RCU maintainers preferred the general fix, marking the function
`notrace`, and the thread converged on taking the first revision. Worth
remembering as a pattern: "which mechanism recurses" and "which mechanism
should be excluded" are different questions, and the answer that scales is the
one at the lowest layer that can carry it.

**Local-storage memory accounting, and the allocation policy under it.** Two
threads on local storage ran in parallel: one adds the per-element allocation
to a map's reported memory usage by walking the bucket list, noting that the
count is imprecise under concurrent insert and delete; the other picks the
allocation variant at call time — sleeping allocation for sleepable programs,
nowait when the caller may take the zone lock, the lock-free path otherwise —
instead of always paying for the lock-free path. Together they describe the
same tension from two sides: storage elements dominate real memory usage, and
the accounting is only as good as the allocation path it measures.

**AF_XDP transmit in copy mode.** A public community post asked whether a
TX-heavy workload of 64-byte packets should shard AF_XDP sockets per queue with
a shared UMEM, or give up on AF_XDP transmit in copy mode; it measured about
24k frames/s through one TX socket on veth, and attributed the ceiling to the
per-`sendto` budget of 32 descriptors. The sources correct that attribution in
a useful way. In copy mode `sendto` reaches the generic transmit path, which
holds the socket mutex, and for each descriptor reserves a completion slot,
allocates an skb, copies the frame into it, and submits it with one device
transmit call. The budget that bounds the loop is a socket option
(`XDP_MAX_TX_SKB_BUDGET`) with a floor of 32, defaulting to 32 and accepting
any value up to the TX ring size — so 32 is the minimum a user may ask for, not
a cap the kernel imposes. Raising it amortises per-call overhead only; the
per-descriptor skb and copy remain, and the mutex serialises every sender that
shares the socket. That is the argument for one socket per queue with per-core
transmit: a shared UMEM across queue ids needs one FILL and COMPLETION pair per
netdev/queue tuple, and those pairs are single-producer/single-consumer, so
"one TX ring per core" is the shape the API supports. On the second half of the
question, whether a NIC can do zero-copy is an advertised device feature, not a
property of the netdev type: veth advertises only basic, redirect and RX
scatter-gather, which is why a veth measurement is necessarily copy mode, while
virtio-net, gve, mlx5, ice and i40e all advertise the zero-copy flag and ena
advertises none.