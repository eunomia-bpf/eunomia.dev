# Why can't SIGKILL stop a BPF program load after verification has succeeded?

**Short answer:** the killable part of a load is only the main verification loop; the instruction-rewrite tail that runs after it has no cancellation points at all:

1. **Verification is killable; the rewrite tail is not.** The verifier's main loop checks pending signals and returns `-EAGAIN`, and reschedules when the kernel asks. But that loop is one stage of `bpf_check()`, not the whole load.
2. **After verification returns, the kernel runs instruction-rewrite passes that neither check signals nor reschedule.** The privileged path hard-wires dead-code branches, removes dead code, and removes NOPs one instruction at a time; the unprivileged path sanitizes dead code; and JIT constant blinding rewrites every constant-bearing instruction individually. Each rewrite goes through the same patch helper, which moves the instruction and auxiliary-data arrays and retargets every branch offset — quadratic in program length. A pending SIGKILL therefore cannot terminate the task until that tail finishes, and a privileged loader can stretch the tail by loading a large program: the limit is 1,000,000 instructions, and the reported reproduction used ~131K unconditional jump/return pairs, which verify quickly but take `bpf_opt_remove_nops()` a long time to remove one at a time.

A patch series now in review makes the rewrite helpers cancellation and rescheduling points, and a bpf list maintainer endorsed that approach in a reply this morning.

## Where killability actually stops

The only cancellation points on the load path live in the verifier's main loop (`do_check()` in `kernel/bpf/verifier.c`, mainline lines 18422-18426):

```c
		if (signal_pending(current))
			return -EAGAIN;

		if (need_resched())
			cond_resched();
```

That is why killing a stuck *verification* works. But when `do_check()` returns, `bpf_check()` in `kernel/bpf/core.c` continues with instruction-level rewrite passes:

```c
	if (is_priv) {
		if (ret == 0)
			bpf_opt_hard_wire_dead_code_branches(env);
		if (ret == 0)
			ret = bpf_opt_remove_dead_code(env);
		if (ret == 0)
			ret = bpf_opt_remove_nops(env);
	} else {
		if (ret == 0)
			sanitize_dead_code(env);
	}
```

Each pass patches or removes one instruction at a time through the shared helpers `bpf_patch_insn_data()` and `verifier_remove_insns()`. Every such call reallocates or moves the `insns` and `aux` arrays and adjusts all branch offsets, which is quadratic in program length overall. In mainline, none of these helpers checks `fatal_signal_pending()`, and none reschedules inside its per-instruction loop — so a SIGKILL delivered during that window is recorded as pending but cannot make the task exit until the rewrite tail completes.

JIT constant blinding has the same shape. `bpf_jit_blind_constants()` (`kernel/bpf/core.c`, mainline line 1562) walks the program and rewrites each constant instruction through `bpf_jit_blind_insn()` plus `bpf_patch_insn_data()`, with no per-instruction signal check. It runs from `bpf_prog_jit_compile()` when `bpf_prog_need_blind()` is true, and from `jit_subprogs()` in `kernel/bpf/fixups.c` for offloaded subprograms. There is a second subtlety specific to blinding: the JIT treats a patching failure as "fall back to the interpreter", so a fatal signal that arrives and gets consumed as a patching failure would otherwise be swallowed by that fallback. The in-flight patch therefore adds two `fatal_signal_pending(current)` rechecks that set `-EINTR` — one after `bpf_fixup_call_args()` and one after `__bpf_prog_select_runtime()` — so the interpreter fallback cannot consume the cancellation.

### The amplifier: the complexity limit counts instructions, not rewrite work

`include/linux/bpf.h` sets `BPF_COMPLEXITY_LIMIT_INSNS` to 1,000,000 — the comment in the tree is "yes. 1M insns". A privileged loader can submit a program up to that size. The reported attack shape is 131072 unconditional jumps to zero, each followed by a valid return: verification of such a program finishes quickly because the states are trivial, but `bpf_opt_remove_nops()` must then remove each of those jumps individually, and each removal is a full array move plus branch-offset retargeting. The task sits in `D` (uninterruptible) state for the whole window.

### The fix in flight

A 7-patch series (v2 in review, `Fixes: 52875a04f4b2`, "bpf: verifier: remove dead code"), reported by an external security researcher, makes `bpf_patch_insn_data()` and `verifier_remove_insns()` common cancellation and rescheduling points. The argument is that they already run from `BPF_PROG_LOAD` process context, the patching helper can already sleep while reallocating, and most callers propagate patching failures directly. A bpf list maintainer's reply today endorses addressing the problem properly by changing the `bpf_patch_insn_data()` implementation rather than working around it. Two designs were discussed — converting the instruction array to a linked list before rewrites, and accumulating patches in a loop and applying them in a single pass — with the single-pass approach the one planned as the most self-contained. The open question in the thread is whether the two fatal-signal rechecks belong in the constants-blinding path instead.

## How to verify

1. **Reproduce the stuck load.** Build a program of roughly 131K unconditional jump/return pairs and load it from a dedicated process with privilege. After verification completes (the `bpf()` syscall is still in flight), send SIGKILL. On an unpatched kernel the task stays in `D` state — check `stat` in `/proc/<pid>/stat` — and the `bpf()` call does not return until the rewrite tail finishes. With the patch, `BPF_PROG_LOAD` returns `-EINTR` promptly and the task is gone.
2. **Confirm the gap in source.** Grep the load path: in mainline, `fatal_signal_pending` checks exist in the `do_check()` loop, but the per-instruction loops of `bpf_opt_remove_nops()`, `bpf_opt_remove_dead_code()`, `sanitize_dead_code()`, and `bpf_jit_blind_constants()` contain no signal check and no `cond_resched()`.
3. **Confirm the fix's placement.** The series adds the two `fatal_signal_pending()` rechecks in `bpf_check()` (after `bpf_fixup_call_args()` and after `__bpf_prog_select_runtime()`) and turns the rewrite helpers into rescheduling points.

## Where the answer stops applying

- The unkillable-rewrite behavior is *current* mainline; the fix is a v2 series in review, not landed. The line numbers above refer to the mainline snapshot checked today.
- The fix makes the tail *preemptible*, not fast: the quadratic cost of the rewrites remains. A very large verification can still take a long time, but it is killable through `-EAGAIN`; the rewrite window is what the patch closes.
- The blinding fallback is a real behavior: when blinding fails, the program runs on the interpreter; the rechecks exist precisely because that fallback previously swallowed fatal signals.
- Offloaded subprograms (the `fixups.c` path) have their own blinding call sites; the patched helpers cover them, but the subprog JIT has its own finalize path worth re-checking when the series lands.
- This is one more primitive in the arsenal discussed in the [2026-09-08 entry](/ebpf-qa/2026-09-08-malicious-ebpf-payload-boundaries/): a load that cannot be killed is a denial-of-service vector even where loading itself is permitted.

## References

- bpf list, thread "bpf: Make post-verification instruction rewrites killable" (v2 of a 7-patch series, with the public maintainer reply): [reply message](https://lore.kernel.org/bpf/917a177ed71802d933c03fa154662b50f5fffd14.camel@gmail.com/) — the patch description, the reported 131072-jump reproduction, and the maintainer's reply with the two design alternatives
- [kernel/bpf/verifier.c](https://github.com/torvalds/linux/blob/master/kernel/bpf/verifier.c) — the `do_check()` main loop with `signal_pending(current)` → `-EAGAIN` and `need_resched()` → `cond_resched()`, the only cancellation points on the load path
- [kernel/bpf/core.c](https://github.com/torvalds/linux/blob/master/kernel/bpf/core.c) — the post-verification rewrite section of `bpf_check()`; `bpf_jit_blind_constants()` and the interpreter fallback on blinding failure
- [kernel/bpf/fixups.c](https://github.com/torvalds/linux/blob/master/kernel/bpf/fixups.c) — `jit_subprogs()`, the blinding call site for offloaded subprograms
- [include/linux/bpf.h](https://github.com/torvalds/linux/blob/master/include/linux/bpf.h) — `BPF_COMPLEXITY_LIMIT_INSNS` set to 1,000,000
- The in-flight series itself (7 patches, v2): `bpf_patch_insn_data()` and `verifier_remove_insns()` as common cancellation/rescheduling points, plus the two `fatal_signal_pending()` rechecks in `bpf_check()`

## Community discussion today

Coverage this run: the public bpf@vger.kernel.org archive was reviewed through its ordinary public feed for the 24-hour window, and it supplied the question above plus most of the day's discussion. The CNCF workspace's opted-in archive channel was readable; its 24-hour window was quiet and the seven-day fallback held only meeting logistics — a stabilization-meeting schedule, a conference session link, and thanks — no technical question, recorded as quiet rather than a gap. The Cilium & eBPF workspace's archive does not carry its allowlisted channel, recorded as inaccessible rather than quiet. The two allowlisted Discord workspaces are visible-browser-only surfaces and no visible browser session was available this run, reported as a coverage gap rather than silence. The public practitioner forum (r/eBPF) could not be fetched this run — the HTTP client was blocked — so it is reported as not reviewed, not claimed as reviewed.

### The question that became today's answer

The day's strongest thread on the bpf list: "Make post-verification instruction rewrites killable", v2 of a 7-patch series, reported by an external security researcher. The report's shape: a privileged loader can submit a program whose verification finishes fast but whose post-verification rewrite tail is quadratic, so a SIGKILL sent after verification succeeds cannot terminate the task. A bpf list maintainer replied the same day, endorsing the direction — stop working around the problem and change the `bpf_patch_insn_data()` implementation — with two designs on the table (linked list of instructions before rewrites, versus accumulating patches and applying them in a single pass) and the single-pass approach planned as the most self-contained. The answer above is published here; the thread also carries the open question of whether the two fatal-signal rechecks belong in the constants-blinding path.

### The rest of the day on the list

- **Helper/kfunc argument validation unification.** A 23-patch bpf-next series (v2) collapses the duplicated argument checking between helpers and kfuncs into a single argument-kind enum and a single runtime type-resolution path, with selftests for kfunc packet memory direct writes and nullable per-CPU kptr ids. The verifier's argument machinery is where most kfunc work has landed recently, and the series aims to make it maintainable.
- **Memory accounting for ring buffers.** An RFC proposes accounting ring-buffer backing pages separately from lost RAM — the memory side of the "events are silently dropped" story that the [2026-09-10 entry](/ebpf-qa/2026-09-10-process-behavior-reconstruction-event-loss/) covered on the completeness side.
- **JIT tooling.** MIPS user-mode assembler support: signed div/mod and sign-extension emitters for the BPF JIT.
- **Selftests.** New kfunc selftests continue the coverage build-out for the argument-validation series above.
