# Why can't a BPF program sleep or block, and what should you use instead in a non-sleepable context?

**Short answer:** a BPF program may sleep only when two conditions hold at once:
the program was loaded as *sleepable* (`BPF_F_SLEEPABLE`, attached to a hook
whose kernel context allows sleeping), **and** execution is not inside a
critical section. The verifier enforces both statically. If a helper that may
sleep is called from a non-sleepable program — or from inside `bpf_spin_lock`,
`bpf_rcu_read_lock`, a preemption-disabled region, or an IRQ-disabled region —
the load is rejected with `sleepable helper <name>#<id> in <context>`, where
`<context>` names the exact reason (`non-sleepable prog`, `lock region`,
`rcu_read_lock region`, `non-preemptible region`, or `IRQ-disabled region`).
This is not a runtime hazard you can tune away: it is a load-time error. The
replacement is not "sleep less" but a different mechanism — per-CPU map slots,
`bpf_spin_lock` under tight constraints, BPF atomic instructions, or deferring
the work (ring buffer, `bpf_loop`, tail calls, or a sleepable async callback).

## The rule is a property of the context, not of BPF

The kernel states the contract directly in the program-load flag. If
`BPF_F_SLEEPABLE` is set, "the verifier will restrict map and helper usage for
such programs. Sleepable BPF programs can only be attached to hooks where
kernel execution context allows sleeping. Such programs are allowed to use
helpers that may sleep like `bpf_copy_from_user()`". So sleepability is granted
per program, and only where the attach point's context permits it.

Only a small set of program types are sleepable. In the libbpf program-type
table the `Sleepable` column is `Yes` for `fentry.s`, `fexit.s`, `fmod_ret.s`,
`fsession.s`, `iter.s`, `lsm.s`, `struct_ops.s`, and the uprobe-family sections
`uprobe.s`, `uretprobe.s`, `usdt.s`, `uprobe.multi.s`, `uretprobe.multi.s`,
`uprobe.session.s`, plus `BPF_PROG_TYPE_SYSCALL` (`syscall`). Every other row's
cell is blank: XDP, tc, socket filters, cgroup hooks, the plain (non-`.s`)
tracing and kprobe variants, tracepoints, and perf-event programs are **not**
sleepable. A program attached to one of those hooks cannot sleep even if you
wanted it to.

The verifier then checks the *dynamic* context inside the program. Its
definition of a sleepable context is the conjunction of five negatives
(`kernel/bpf/verifier.c`):

```c
static inline bool in_sleepable_context(struct bpf_verifier_env *env)
{
	return !env->cur_state->active_rcu_locks &&
	       !env->cur_state->active_preempt_locks &&
	       !env->cur_state->active_locks &&
	       !env->cur_state->active_irq_id &&
	       in_sleepable(env);
}
```

Both halves matter: the program must be sleepable (`in_sleepable`), and none of
the four "inside a critical section" counters may be nonzero. When the check
fails at a helper call, the verifier prints the failing reason through
`non_sleepable_context_description()`, which maps each counter to a name:
`rcu_read_lock region`, `non-preemptible region`, `IRQ-disabled region`,
`lock region`, or — when the program itself is simply not sleepable —
`non-sleepable prog`.

That is why the error is useful: it tells you *which* of the two conditions you
violated. `sleepable helper bpf_copy_from_user#190 in non-sleepable prog` means
the program type is wrong. The same message ending in `lock region` or
`rcu_read_lock region` means the program type is fine but the call sits inside
a critical section and must move out.

The rule applies to BPF-to-BPF calls and to kfuncs as well. When a global
function marked as potentially sleeping is called in a non-sleepable context,
the verifier emits `sleepable global function <name>() called in <context>` and
rejects the load. For a sleepable kfunc it distinguishes the two conditions:
from a non-sleepable program it prints `program must be sleepable to call
sleepable kfunc <name>`, and from inside a critical section it prints `kernel
func <name> is sleepable within <context>`. There is a related restriction while
holding a lock: `global function calls are not allowed while holding a lock, use
static function instead`.

## Why the kernel cannot simply allow it

Sleeping means the current task voluntarily yields the CPU and requires the
scheduler to run again on that task. That is only sound where the kernel owns a
resumable task context and has not disabled the mechanisms the scheduler needs.
Inside an RCU read-side critical section, a spinlock, a preemption-disabled
region, or with IRQs disabled, none of that holds — the code may run in softirq
or hardirq context, or in a region that must not be preempted. A blocking call
there would either deadlock (waiting for a wakeup that cannot be delivered) or
violate the atomicity the section guarantees. The verifier's static rejection is
therefore a proof obligation, not a heuristic: it refuses to emit a program that
*could* reach a sleeping call in an atomic region along any path.

This is also why "make the program sleepable" is not a blanket fix. Even a
sleepable program can enter these regions explicitly: `bpf_rcu_read_lock()` — a
kfunc — opens an RCU read-side critical section, and the verifier's own
`in_rcu_cs()` treats such a region as non-preemptible for the purposes of the
sleepability check. In the kfunc documentation, `KF_RCU_PROTECTED` "is assumed
by default in non-sleepable programs, and must be explicitly ensured by calling
`bpf_rcu_read_lock` for sleepable ones" — so a sleepable program that needs
RCU-protected data must manage that region itself, and a sleeping call inside it
is rejected exactly like one inside a spin lock.

## What to use instead

The correct replacement depends on what the sleep was for. Three families cover
almost every case.

**1. Remove the shared state (per-CPU maps).** For counters and per-CPU
scratch, a per-CPU map gives each CPU its own slot, so two CPUs never contend
and no lock is needed. The kernel documents the memory model explicitly:
`BPF_MAP_TYPE_PERCPU_ARRAY` "uses a different memory region for each CPU whereas
`BPF_MAP_TYPE_ARRAY` uses the same memory region". Per-CPU values are capped at
`PCPU_MIN_UNIT_SIZE` (32 kB) and must be read per-CPU; user space aggregates the
per-CPU copies. `bpf_this_cpu_ptr()` returns the current CPU's copy of a per-CPU
ksym and "would never return NULL", while `bpf_per_cpu_ptr()` takes an explicit
CPU. Per-CPU maps do **not** give you cross-CPU consistency — if you need one
consistent value across CPUs, they are the wrong tool.

**2. Protect the shared value in place (`bpf_spin_lock`, atomics).** When two
CPUs really must update the same map value, `bpf_spin_lock` is the supported
primitive, but it carries hard constraints documented in `bpf-helpers(7)`:

- the man page's historical rule is "only allowed inside maps of types
  `BPF_MAP_TYPE_HASH` and `BPF_MAP_TYPE_ARRAY`"; the current verifier is broader,
  also accepting a lock inside BTF-allocated (`bpf_obj_new()`) objects, with BTF
  description of the map mandatory either way;
- "The BPF program can take ONE lock at a time, since taking two or more could
  cause dead locks";
- "Only one `struct bpf_spin_lock` is allowed per map element", at the top level
  of the value, 4-byte aligned, not nested and **not** on the stack or in a
  packet;
- "When the lock is taken, calls (either BPF to BPF or helpers) are not
  allowed", and `BPF_LD_ABS`/`BPF_LD_IND` are forbidden in the locked region;
- "The BPF program MUST call `bpf_spin_unlock()` to release the lock, on all
  execution paths, before it returns";
- "`bpf_spin_lock` is available to root only", and tracing and socket-filter
  programs cannot use it "due to insufficient preemption checks";
- it is not allowed in inner maps of map-in-map.

A spin lock is not a substitute for sleeping either: because helpers may not be
called while it is held, and it disables preemption, holding it for anything
long is itself the bug. For single-word updates, plain BPF atomic instructions
and atomic map updates avoid the lock entirely — `bpf_map_update_elem()`
"replaces existing elements atomically", and the BPF instruction set defines
atomic operations as part of the ISA.

**3. Defer the work to a context that may sleep.** If the work genuinely needs
to block — a usercopy, an allocation, I/O — do not do it where you cannot; move
it. Options, in increasing order of separation:

- **Ring buffer**: `bpf_ringbuf_reserve()` / `bpf_ringbuf_submit()` /
  `bpf_ringbuf_output()` hand the payload to user space, which does the blocking
  work in a normal process context.
- **Bounded iteration**: `bpf_loop()` runs a callback for up to `1 << 23`
  iterations with the callback context on the stack; it makes long work
  *terminate* but does **not** make it sleepable.
- **Tail calls** to split the program, which likewise bounds work without
  changing the context.
- **Async callbacks**, which change the context for you. This is the subtle
  case: `bpf_timer_start()` invokes "the configured callback … in soft irq
  context on some cpu", and the verifier classifies callbacks by sleepability —
  inside `is_async_cb_sleepable()`, a comment states "bpf_timer callbacks are
  never sleepable", while "bpf_wq and bpf_task_work callbacks are always
  sleepable". So a timer callback is a *different* non-sleepable context, not an
  escape from the rule; `bpf_wq`/`bpf_task_work` callbacks *are* the escape.
- **A sleepable program**: if the work belongs to a hook that can sleep, attach
  a `*.s` (sleepable) program there — for example `fentry.s`/`fexit.s`/`lsm.s` or
  `BPF_PROG_TYPE_SYSCALL` — where a sleeping helper such as
  `bpf_copy_from_user()` is permitted.

## How to diagnose which case you hit

The verifier error already names the context. Read it literally:

```text
sleepable helper bpf_copy_from_user#190 in non-sleepable prog   -> wrong program type
sleepable helper ... in lock region                             -> call inside bpf_spin_lock
sleepable helper ... in rcu_read_lock region                    -> call inside bpf_rcu_read_lock
sleepable helper ... in non-preemptible region                  -> preemption disabled
sleepable helper ... in IRQ-disabled region                     -> IRQs disabled
sleepable global function foo() called in <context>             -> a may-sleep BPF-to-BPF callee
program must be sleepable to call sleepable kfunc bar           -> kfunc needs a sleepable prog
kernel func bar is sleepable within <context>                   -> kfunc called in a critical section
```

A minimal path to reproduce and resolve it:

```sh
# 1. Find the exact rejection and the named context.
sudo bpftool prog load ./obj.o /sys/fs/bpf/p 2>&1 | grep -i 'sleepable\|context\|lock'

# 2. Confirm the program is not sleepable and its attach point is atomic:
#    check the ELF section (no '.s' suffix => not sleepable) and the hook type.

# 3. Is the offending helper sleeping on purpose? Check whether the value
#    being copied can be fetched without sleeping instead.
grep -n 'BPF_FUNC_copy_from_user\|bpf_probe_read_kernel' ./prog.c

# 4. If the call is inside a lock, move it out and re-check that the lock is
#    released on every path before the helper call.
```

If the message ends in `non-sleepable prog` and the helper genuinely must
sleep, the program type is wrong — move that logic to a sleepable program or a
`bpf_wq`/`bpf_task_work` callback. If it ends in a critical-section name, move
the call outside that section (or replace the helper with a non-sleeping one
such as `bpf_probe_read_kernel()`).

## The limitation that decides it

The rule is not "BPF cannot sleep". It is a conjunction that must hold at every
program point: the program is sleepable **and** no critical section is active.
Two boundaries follow and both bite in practice. First, sleepability is a
property of the *attach point*: you cannot add it to an XDP, tc, or plain kprobe
program by changing the helper — the hook's context does not permit sleeping.
Second, an async callback is not automatically sleepable: a `bpf_timer` callback
is itself non-sleepable, so moving work into a timer can reproduce the same
error; only `bpf_wq`/`bpf_task_work` callbacks (and sleepable program types)
give you a context where a sleeping helper is legal. When you cannot satisfy the
conjunction, the answer is always the same shape: do the atomic part in the
program (per-CPU slot, spin lock, or atomic instruction) and defer the blocking
part to a context that is allowed to sleep.

## References

- [The Linux kernel documentation: Program Types and ELF Sections](https://docs.kernel.org/bpf/libbpf/program_types.html)
- [The Linux kernel documentation: BPF_MAP_TYPE_ARRAY and BPF_MAP_TYPE_PERCPU_ARRAY](https://docs.kernel.org/bpf/map_array.html)
- [The Linux kernel documentation: BPF_MAP_TYPE_HASH](https://docs.kernel.org/bpf/map_hash.html)
- [The Linux kernel documentation: BPF Instruction Set Architecture (ISA)](https://docs.kernel.org/bpf/standardization/instruction-set.html)
- [bpf-helpers(7) — Linux manual page](https://man7.org/linux/man-pages/man7/bpf-helpers.7.html)
- [Linux kernel source: `kernel/bpf/verifier.c`](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/verifier.c)
- [Linux kernel source: `include/uapi/linux/bpf.h`](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/include/uapi/linux/bpf.h)

## Community discussion today

The monitored window was not technical. Over the previous 24 hours the two
opt-in Slack archives returned a small fallback set consisting only of meeting
logistics — a recurring scheduling note, a co-presenting thank-you, and a
session link — with no eBPF question, symptom, or design dispute. The two
allowlisted chat workspaces had no visible browser session this run, and the
public mailing-list and forum archives were not reviewed. Those sources are
recorded as unavailable coverage, not quiet. Because the readable archive
yielded no question, this Q&A is grounded in public primary documentation rather
than a community message: the question is a recurring practitioner problem —
a BPF helper that may sleep is called from a context that must not block, and
the load fails with a context-specific verifier error — verified against the
kernel BPF documentation, the upstream verifier and UAPI source, the BPF
instruction-set specification, and `bpf-helpers(7)`. No private text, identity,
channel, or link is reproduced here.
