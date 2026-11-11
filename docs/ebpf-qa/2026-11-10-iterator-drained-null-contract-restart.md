# Why must an open-coded BPF iterator keep returning NULL after it drains, and what changes when a next method starts over instead?

Because the verifier verifies an iterator loop by assuming that assumption: at every
`next` call it forks one state that treats the result as NULL and marks the iterator
drained, and one that treats it as a live pointer. On the drained path the return
value is a constant zero and no further fork happens, so the rest of the program is
verified as if `next` can only ever return NULL from then on. A `next` method that
treats "drained" and "not started" as the same state restarts and hands back a real
pointer the verifier has already stopped accounting for. Three separate open-coded
iterators had exactly that shape, and three patches posted on 10 November fix them
one by one.

## The mechanism

1. **The contract, in the documentation.** The iterator documentation states it
   plainly: a `next` method "guarantees that it will eventually return NULL when
   elements are exhausted. Once NULL is returned, subsequent next calls should keep
   returning NULL." The same section requires a constructor to initialise the state so
   that `next` returns NULL on failure, and requires the state struct to be
   stack-sized and ABI-stable, because its size and layout are part of the
   user-visible API.

2. **Why the verifier needs it.** Verification of a loop whose exit depends on kernel
   data cannot be exhaustive. So at each `next` call the verifier runs
   `process_iter_next_call()`, which splits the current state: one copy is queued
   with the iterator still active, its depth incremented, and the return register
   marked non-NULL; the current state is switched to drained and the return register
   set to the constant zero. The queued copy is verified and, if it reaches the same
   `next` call with an equivalent state, the loop is accepted as convergent. The
   drained copy is what proves the exit is reachable. Both halves of that argument
   depend on the iterator never coming back to life.

3. **What a drained iterator means to the rest of verification.** A call to `next` on
   an iterator whose state is already drained does not fork. It just marks the
   return register constant zero again. Every conditional that tests that register is
   therefore decided statically by `is_branch_taken()`, which follows only one side
   and pushes the other as a speculative path.

4. **What happens to the side that was not followed.** Instructions reached only
   through the pruned branch are verified, if at all, under speculation, and
   speculative simulation deliberately does not mark them as seen. Two
   post-verification passes then act on that marking, and which one runs depends on
   the load's privilege. For a privileged load, the pass that hard-wires dead
   branches rewrites the conditional into an unconditional jump to the side the
   verifier kept, and `bpf_opt_remove_dead_code()` deletes the other side. For an
   unprivileged load, `sanitize_dead_code()` replaces each unseen instruction with
   a jump to itself, a trap. Either way the loaded program encodes the verifier's
   belief rather than a test of the register.

5. **So a restart diverges at run time.** The kernel's `next` returns a valid pointer,
   but the loaded code no longer tests it. In the privileged shape the branch was
   removed, so execution unconditionally takes the drained side and the element is
   silently skipped — and if the program goes on to loop, it loops on a fixed path.
   In the unprivileged shape the conditional survives, the restart sends execution
   into the branch the verifier never explored, and every instruction there is a
   self-jump: the program hangs in the verifier's trap until a watchdog reports it.

6. **The two shapes are distinguishable at load time.** If the program calls `next`
   again after the drained call and loops back to it, there is no new state to fork,
   so `is_state_visited()` sees equivalent states and the load fails with "infinite
   loop detected". The in-tree selftest for that shape is named
   `iter_remake_drained`, and it expects exactly that message. So the same
   implementation defect shows up either as a rejected program or as an accepted
   program that misbehaves.

7. **Three iterators shared the defect.** A dma-buf iterator used a NULL element
   pointer for "iteration has not started", but the end of that list is also NULL, so
   the call after exhaustion started over at the head; the fix uses an invalid
   pointer value as the drained sentinel and skips it in the destructor, which keeps
   the state one pointer wide. A cgroup iterator cleared its position on exhaustion
   but kept its start pointer, and every `css_next_*()` helper restarts from the
   start when the position is NULL; the fix clears the start pointer on exhaustion,
   and the destructor is a no-op so nothing else depended on it. A scheduler DSQ
   iterator unlinks its cursor at the end of the queue, and the next call then saw an
   empty cursor and began again at the queue head; the fix records the drained state
   in a spare cursor flag bit, leaving the BPF-visible iterator layout unchanged.
   Three unrelated iterators, one contract nobody had written down as a requirement
   on the implementation.

8. **The iterators that are already correct show the intended shape.** The number
   iterator, once past its range, sets both its cursor and its end to zero and returns
   NULL forever after. The task iterator returns NULL as soon as its position is
   NULL. A dma-buf-style "NULL means not started" encoding is the only thing that
   loses the distinction, and a sentinel, a cleared start, or a flag bit restores it.

## Verification and debugging path

1. **Read the iterator state in the log.** A level 2 verifier log prints each
   iterator stack slot as `iter_<type>(id=N,state=…,depth=N)` and names the three
   states explicitly: active, drained, invalid. Load the program with `log_level` 2
   and find the first line where the state becomes drained; every later `next` call
   in that log is one where the verifier is no longer forking.

2. **Look at what the branch became.** `bpftool prog dump xlated` shows the
   post-verification instruction stream. A conditional that you wrote on the `next`
   result appears as an unconditional jump when the load was privileged, and the
   other side's instructions are gone; when the load was unprivileged, the untaken
   side appears as a jump to itself.

3. **Reproduce deliberately.** Write a program that calls `next` once more after it
   has returned NULL, and uses the result. If the extra call is inside a loop you
   get "infinite loop detected" at load. If it is straight-line code you get an
   accepted program; run it and observe that the guarded branch is skipped, or that
   the unprivileged build hangs in the trap.

4. **Audit your own `next` against three questions.** Does the constructor leave a
   state whose first `next` returns NULL? Is exhaustion distinguishable from "not
   started"? Does anything after exhaustion reset the position? A one-line selftest
   that calls `next` a second time after it returned NULL and asserts NULL catches
   all three, and it is the check the in-tree registry does not perform.

## Limitations

- The three patches are from 10 November and describe a bug class, not a merged
  invariant. Only the scheduler one had been applied in the window, to a fixes
  branch; the cgroup one carries an ack and an open question about which tree takes
  it; the dma-buf one had no reply yet. Kernels in the field can still carry all
  three defects.
- Nothing enforces the contract at kfunc registration. The verifier trusts the
  implementation, and the documentation states the requirement as prose, so a new
  iterator can reintroduce the same restart without any load-time diagnostic.
- The fixes are constrained by ABI. The iterator state struct's size is visible to
  programs through generated type information, so the dma-buf fix spends a sentinel
  value instead of adding a field, and the scheduler fix uses a spare flag bit.
- From inside a program the restart is invisible: the register holds a real pointer
  while the loaded code no longer tests it, and nothing in the program can tell which
  of the two post-verification rewrites it was compiled with.

## References

- [BPF iterators documentation](https://github.com/torvalds/linux/blob/master/Documentation/bpf/bpf_iterators.rst) — the `next` contract ("once NULL is returned, subsequent next calls should keep returning NULL"), the constructor and destructor requirements, and why the state layout is ABI.
- [Kernel `kernel/bpf/verifier.c`](https://github.com/torvalds/linux/blob/master/kernel/bpf/verifier.c) — `process_iter_next_call()` with the drained and active fork, the constant-zero result on the drained path, `sanitize_dead_code()`, and the privileged and unprivileged post-verification branches.
- [Kernel `kernel/bpf/fixups.c`](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/fixups.c) — `bpf_opt_hard_wire_dead_code_branches()` turning a conditional into an unconditional jump once one side is unseen, and `bpf_opt_remove_dead_code()`.
- [Kernel `kernel/bpf/log.c`](https://github.com/torvalds/linux/blob/master/kernel/bpf/log.c) — the iterator stack-slot print with its type, id, state and depth, and the active, drained and invalid state names.
- [Kernel `kernel/bpf/dmabuf_iter.c`](https://github.com/torvalds/linux/blob/master/kernel/bpf/dmabuf_iter.c) — the iterator that treats a NULL element as "not started", and its destructor.
- [Kernel `kernel/bpf/cgroup_iter.c`](https://github.com/torvalds/linux/blob/master/kernel/bpf/cgroup_iter.c) — the constructor that stores the start css, the `next` that clears only the position, and the no-op destructor.
- [Kernel `kernel/sched/ext.c`](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/sched/ext.c) — the DSQ iterator's cursor flags and the cursor unlink at the end of the queue.
- [Kernel `kernel/bpf/bpf_iter.c`](https://github.com/torvalds/linux/blob/master/kernel/bpf/bpf_iter.c) — the number iterator, which zeroes both cursor and end when the range is exhausted and so returns NULL forever after.
- [Selftest program `iters.c`](https://github.com/torvalds/linux/blob/master/tools/testing/selftests/bpf/progs/iters.c) — the `iter_remake_drained` case and the manual call sequences that expect NULL after exhaustion.
- [Patch: make the dma-buf iterator stay drained after NULL](https://lore.kernel.org/bpf/20261110125106.2935677-1-k.chen@smail.nju.edu.cn/) — the NULL-means-not-started collision, the invalid-pointer sentinel for the one-pointer state, and the destructor change.
- [Patch: make the css iterator stay drained after NULL](https://lore.kernel.org/bpf/20261110125110.2935707-1-k.chen@smail.nju.edu.cn/) — the restart inside `css_next_*()` when the position is NULL, and why clearing the start pointer is enough.
- [Ack and routing question for the css fix](https://lore.kernel.org/bpf/d81460af8053a51f9be8a73186025ce7@kernel.org/) — the ack and the open question of which tree takes it.
- [Patch: make the DSQ iterator stay drained after NULL](https://lore.kernel.org/bpf/20261110125113.2935742-1-k.chen@smail.nju.edu.cn/) — the cursor unlink that restarts the walk, and the spare flag bit used to remember the drain.
- [Reply applying the DSQ fix](https://lore.kernel.org/bpf/9a45ee48b0984bd959becb01d1d11f1c@kernel.org/) — the branch the DSQ fix went to.
- [libbpf `bpf_helpers.h`](https://github.com/torvalds/linux/blob/master/tools/lib/bpf/bpf_helpers.h) — the `bpf_for_each` macro, whose loop condition is exactly `next(...)` being non-NULL.

## Community discussion today

This run reviewed six allowlisted communities. Two are Slack workspaces reached
through the approved private read-only archive, covering three allowlisted channels.
One channel carried no message in the 24-hour window and none in the seven-day
window; the other two carried none in 24 hours and four in seven days, all four
belonging to a single thread about booking a table at a conference event — logistics,
with no technical content. The two Discord workspaces, ten allowlisted channels in
total, were not covered: this environment has no authenticated session and both
public invite links answer with an invalid-invite page, so they are recorded as
inaccessible rather than quiet. The practitioner forum was reachable this run and had
no post inside the window; its three visible posts are two days old or older. The
kernel mailing list was read through its public archive, and the question above plus
the themes below come from it.

The first theme is the one answered above: three iterators whose `next` method
restarts after returning NULL, found in the same week and fixed separately. Three
details make it more than three small patches. The defect is invisible at load for
straight-line code and fatal only at run time, so a program's behaviour depends on
whether it was loaded with `CAP_BPF` or without. The fixes are constrained by the
iterator state being ABI, which is why one spends a sentinel pointer value and
another a flag bit rather than adding a field. And the requirement itself lives in
documentation rather than in a registration-time check, which is what allowed three
independent implementations to miss it.

The second theme is what the verifier does to the rest of the system while it runs. A
verifier loop reported stalling RCU-tasks on debug kernels for over twenty minutes
per load: the main loop calls `cond_resched()`, which does nothing on a kernel with
preemption enabled, and involuntary preemption is not a tasks-RCU quiescent state, so
the loading task stays a holdout until the load returns while every
`synchronize_rcu_tasks()` caller waits — the reported example being a dynamic ftrace
shutdown. The fix swaps in `cond_resched_tasks_rcu_qs()`, which reports the
quiescent state even when the reschedule itself is unnecessary; it also drops the
redundant `need_resched()` test. The diagnostic worth remembering is that a stall
report naming a task inside the verifier is not a hung program, and the stack in the
report shows which verifier allocation was in progress, not the instruction being
verified.

The third theme is registration cost. A thirteen-patch series indexes registered
kprobes by code address in an rbtree and adds address-sorted indexes for the
static-call and jump-label reservation checks, so instruction recovery and
optimisation stop looking up addresses one at a time; the measured result on 20,000
probes is roughly a tenfold reduction in cycles and elapsed time. Two decisions in it
are worth copying: if the sorted index allocation fails the code keeps its linear
scans rather than failing registration, and module reservation checks are left linear
because the module case was not where the cost was. The stated motivation is
inline-function tracing, which will register far more probes than today.

The fourth theme is a per-CPU counter that conflates recursion with preemption. A
non-sleepable fentry or fexit program takes a per-CPU recursion counter before its
first instruction. If the task running it is preempted, the counter stays set, and
the next task on that CPU to enter the same function sees a count that is not one,
increments the recursion-miss statistic, and skips the program. On distribution
kernels this was hidden while voluntary preemption was the default, because a
non-sleepable program has no voluntary preemption point; with only lazy and full
preemption left on x86 and arm64, every distribution kernel now shows it. The
measured skip rates are tens of percent, and the practical consequence for paired
entry and exit probes is that one half of the pair is dropped, desynchronising any
state machine built from them. The proposal disables preemption around the call,
matching kprobes and perf tracepoints; a reviewer's alternative of relaxing the check
to "same task" was rejected because a second task on the same CPU would then share a
single per-CPU private stack and the existing per-CPU scratch-buffer idiom. The
thread's open point is that a program cannot work around this itself: the counter is
tested before the first instruction, and disabling preemption from BPF cannot span a
tail call.

The fifth theme is where error injection may attach. An eighteen-patch series
replaces the static list of functions an fmodret program may modify with BTF
declaration tags plus a compiler plugin that scans IR for candidates whose path to a
plain `return -errno` has no side effects, so the list can be generated rather than
maintained. Two boundaries are stated explicitly rather than assumed: the option is
off by default and needs a recent compiler and only x86-64 and arm64, and the
decision that a given function is safe to fail is delegated to the operator, on the
argument that such an operator can already live-patch the kernel.

Smaller signals, each with the same shape — a property that only holds for a
particular class of caller. `bpftool` map dumps gain batch lookups with a fallback
from the last printed key, because a batch walk can lose its cursor when a key is
deleted under it and otherwise report a short walk as the end of the map. The
BPF CI was down for the day after the patchwork instance blocked the bot that copies
series to CI, so new series carried no CI result; the advice was not to resend but to
run the local VM test script. A `cpu.max.runtime` series that adds a per-cgroup
bandwidth runtime file, lets it feed BPF schedulers, and lets those schedulers charge
their tasks back to it was called too maximalist for an unproven use case, with the
suggestion to keep the scheduler-side credit in the scheduler's own maps until the
need is real. And a capability check reached through the wrong function: setting
process securebits tested `CAP_SETPCAP` with the plain capability check instead of
the namespace-aware one, so an LSM restricting capabilities through the capable hook
— including a BPF LSM program — could not deny a privileged securebit change that
`PR_CAPBSET_DROP` does allow it to deny; the fix aligns the two and asks the BPF LSM
side to check what else becomes deniable.