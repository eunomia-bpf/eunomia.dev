# Why is a speculation barrier enough in JITed arm64 BPF but not in the BPF interpreter?

**Short answer:** because of *where* the barrier lands, not just whether it exists.
The verifier accepts a program that misbehaves only on a speculative path by
inserting a `nospec` instruction (an internal `BPF_ST` with mode `0xc0`) in front
of the instruction it could not prove safe, and then stops verifying that path.
A JIT compiles that instruction into a barrier in the straight-line instruction
stream: `sb` (or `dsb nsh; isb`) on arm64, `lfence` on x86-64. The interpreter
instead dispatches through a jump table — `goto *jumptable[insn->code]` — so the
barrier sits *behind an indirect branch*. On arm64 there is no retpoline, and on
x86-64 the kernel turns retpoline off at run time when it selects eIBRS or IBRS;
a program that trains that indirect branch can make the CPU skip the barrier and
execute the next instruction speculatively. On top of that, the arm64 interpreter
does not even contain a barrier: it calls `barrier_nospec()`, and arm64 defines
`spec_bar()` but not `barrier_nospec()`, so the generic empty definition in
`linux/nospec.h` applies. The fix path being reviewed is not a stronger barrier
but removing the interpreter from arm64 and x86-64 kernels that have the `bpf()`
system call.

## The mechanism

### 1. What the verifier emits, and when

Spectre v1 mitigation in the verifier is an accept-with-barrier scheme. When
verification of a *speculative* state hits an error that is not fatal, the
verifier sets `insn_aux->nospec` on the instruction it was about to check and
stops exploring that path:

```c
if (error_recoverable_with_nospec(err) && state->speculative) {
        insn_aux->nospec = true;
        insn_aux->alu_state = 0;
        goto process_bpf_exit;
}
```

and `error_recoverable_with_nospec()` is exactly `-EPERM`, `-EACCES` and
`-EINVAL`. After verification, `bpf_convert_ctx_accesses()` patches a `nospec`
in front of every marked instruction. The instruction is not part of the UAPI;
the disassembler prints it as `(c0) nospec`, and it is documented in the source
as covering both Spectre v1 and v4 since the semantics were widened for this use.

The key property is that the barrier protects the *architectural* program only if
the CPU cannot run instructions after it before the barrier retires. Everything
downstream of that assumption — including whether an interpreter can honour it —
is what the patch series argues about.

### 2. JIT: the barrier is in the instruction stream

Both JITs emit a plain barrier where the `nospec` instruction sits, so the
sequence the CPU sees is `... load; barrier; next-insn ...` with no branch in
between:

- arm64: `sb` when the CPU reports `ARM64_HAS_SB` (from the SB field of
  `ID_AA64ISAR1_EL1`), otherwise `dsb nsh` followed by `isb`.
- x86-64: `lfence`.

That is why the JIT path is considered covered: a mistrained conditional branch
earlier in the program cannot move the following instruction ahead of an `sb` or
an `lfence`, because the barrier is the next instruction in program order.

### 3. Interpreter: a barrier behind an indirect branch

The interpreter is a computed goto over one 256-entry table:

```c
select_insn:
        goto *jumptable[insn->code];
```

`ST_NOSPEC` is one label in that table and its body is `barrier_nospec()`. Two
independent things then break:

**The branch itself.** With arm64 there is no retpoline, and on x86-64 the
retpoline mitigation is disabled at run time on CPUs that have enhanced or
legacy IBRS. A program that shapes the branch predictor can therefore have the
CPU resume at the instruction *after* the `nospec` label, skipping the barrier
entirely. This is a property of the dispatch, not of the barrier.

**The barrier on arm64 is empty.** `include/linux/nospec.h` supplies

```c
#ifndef barrier_nospec
# define barrier_nospec() do { } while (0)
#endif
```

x86 overrides that name in its own `asm/barrier.h` with an `alternative()` that
patches an `lfence` in when the CPU has `X86_FEATURE_LFENCE_RDTSC`.
arm64 does not define it at all: it has `spec_bar()` in
`arch/arm64/include/asm/barrier.h`, used by the Spectre-v4 mitigation code, with
a different name. So on an arm64 kernel that still has the interpreter, the
`nospec` the verifier inserted executes as nothing, and the comment in
`kernel/bpf/core.c` next to `ST_NOSPEC` — that arm64 can rely on the firmware
SSBD mitigation instead — describes Spectre v4, not v1. SSBD is a store-bypass
mitigation; it does not stop a mispredicted load.

### 4. How a kernel ends up with the interpreter

`CONFIG_BPF_JIT` is an independent boolean, and the interpreter exists whenever
`CONFIG_BPF_JIT_ALWAYS_ON` is off:

```c
#ifndef CONFIG_BPF_JIT_ALWAYS_ON
static u64 ___bpf_prog_run(u64 *regs, const struct bpf_insn *insn)
...
#endif
```

`BPF_JIT_ALWAYS_ON` was previously just a user choice. The first fix, merged
into bpf-next for the next merge window, adds an arch symbol
`ARCH_WANT_BPF_JIT_ALWAYS_ON` that x86-64 and arm64 select, hides the prompt,
and makes the option default on; it also turns the sysctl into a pinned value
and makes a failed JIT a load error (`-ENOTSUPP`) instead of a silent fallback.
That still leaves the case the follow-up patch addresses: a kernel built with
`BPF_JIT=n` never compiles the JIT in, so `BPF_JIT_ALWAYS_ON` cannot apply. The
follow-up makes `BPF_SYSCALL` select `BPF_JIT` on those two architectures, with
one deliberate exception, because a sanitizer kernel needs the interpreter:
```c
select BPF_JIT if ARCH_WANT_BPF_JIT_ALWAYS_ON && !KMSAN
```

KMSAN cannot see stores performed by JITed programs, so it reports a map key
living on the program stack as uninitialized; the interpreter unpoisons its
stack explicitly for exactly that reason, and the sanitizer builds keep using it.
32-bit x86 keeps the free choice because that JIT does not cover every
instruction.

## Verifying it on a real kernel

1. Ask the kernel what it was built with, and whether the interpreter is
   compiled in at all. The `bpf_jit_enable` file under `/proc/sys/net/core` only
   exists when `CONFIG_BPF_JIT` is enabled; when `BPF_JIT_ALWAYS_ON` is on, its
   minimum and maximum are both 1, so any write fails with `EPERM` and the
   kernel logs once that the module is enabled and the value is pinned.

2. Dump both forms of the same program. `bpftool prog dump xlated` shows
   `(c0) nospec` before the instruction the verifier could not prove;
   `bpftool prog dump jited` shows `sb` (or `dsb nsh` + `isb`) at that point on
   arm64 and `lfence` on x86-64. If the second dump fails with "not jited", the
   program is running through the interpreter and the barrier is the computed-goto
   version described above.

3. Check who the mitigation is for. `env->bypass_spec_v1` is set when the JIT
   says it does not need the mitigation, when CPU mitigations are off, or when
   the loading process holds `CAP_PERFMON`; for that loader the verifier inserts
   no `nospec` at all. Unprivileged BPF is disabled by default
   (`kernel.unprivileged_bpf_disabled = 2`), so the arm64 interpreter exposure
   requires an admin who enabled unprivileged BPF, or a privileged program that
   was deliberately loaded through a token without `CAP_PERFMON`.

4. Confirm the arm64 V1 hole positively, not by absence of evidence: the barrier
   in the interpreter is `barrier_nospec()`, and `grep` for that name under
   `arch/arm64/` returns nothing while `spec_bar()` exists. That is the empty
   definition, not a slower barrier.

## The limits

The interpreter problem is a *defence depth* argument, and the fixes do not make
the interpreter safe; they delete it from the two architectures where the
alternative is a real barrier. Where the interpreter survives — 32-bit x86,
architectures without `ARCH_WANT_BPF_JIT_ALWAYS_ON`, KMSAN builds, any kernel
built with `BPF_JIT=n` before the follow-up lands — the same reasoning applies,
and the answer to "is my `nospec` enforced?" is "only if the program is JITed".
The mailing-list thread does not claim otherwise; the disagreement is about
whether closing it by changing Kconfig defaults is acceptable churn in the same
window that also leaves Spectre v2 against the jump table open. Even with the
JIT, the arm64 barrier is only as good as the SB capability: a CPU without
`ARM64_HAS_SB` gets `dsb nsh; isb`, which is a correct but heavier barrier, and
the JIT picks it at compile time from a system-wide capability.

## References

- [BPF mailing list: arm64 Spectre v1 in the BPF interpreter, v1](https://lore.kernel.org/bpf/20261117172341.3152407-1-ryan.roberts@arm.com/) — the patch that renames the arm64 barrier and adds a uaccess predicate.
- [BPF mailing list: first review, nack over bpf-next](https://lore.kernel.org/bpf/DMYSI3BOQWOC.2P2FZB0H872DR@gmail.com/) — "in bpf-next arm64 has no interpreter when the JIT is built".
- [BPF mailing list: Kconfig and existing kernels](https://lore.kernel.org/bpf/DN07ZMOAD4GP.37PG3XUPBG4CM@gmail.com/) — what an interpreted kernel means for stable and for `BPF_JIT=n`.
- [BPF mailing list: the indirect-branch objection](https://lore.kernel.org/bpf/DN2ALO12MI1Z.3HI553HDMO2FL@gmail.com/) — `goto *jumptable[insn->code]`, no retpoline on arm64, and the proposed Kconfig hunk.
- [BPF mailing list: maintainer reply on v2 and CSV2](https://lore.kernel.org/bpf/awRd1mHcQDVNi9U9@arm.com/) — what forcing the JIT on would make this patch unnecessary.
- [BPF mailing list: author closes the thread](https://lore.kernel.org/bpf/0fad3d48-54cc-4919-bff9-254347c933c4@arm.com/) — the patch is dropped in favour of the Kconfig change.
- [BPF mailing list: force the JIT on x86-64 and arm64](https://lore.kernel.org/bpf/20261123205713.4094553-1-alexei.starovoitov@gmail.com/) — select `BPF_JIT` from `BPF_SYSCALL`, with a KMSAN exception.
- [BPF interpreter and jump table](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/core.c) — `___bpf_prog_run()`, the `jumptable`, `ST_NOSPEC`, and the `BPF_JIT_ALWAYS_ON` guard.
- [arm64 JIT barrier](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/arch/arm64/net/bpf_jit_comp.c) — `sb` versus `dsb nsh` plus `isb`, and the bypass hook for Spectre v4.
- [x86-64 JIT barrier](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/arch/x86/net/bpf_jit_comp.c) — `EMIT_LFENCE()` for the same instruction.
- [arm64 barrier names](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/arch/arm64/include/asm/barrier.h) — `spec_bar()` exists, `barrier_nospec()` does not.
- [x86 barrier definition](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/arch/x86/include/asm/barrier.h) — `barrier_nospec()` as an `lfence` alternative.
- [Generic empty barrier](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/include/linux/nospec.h) — the empty fallback when the architecture does not define one.
- [Verifier nospec insertion](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/verifier.c) — `error_recoverable_with_nospec()` and the speculative-state branch.
- [Verifier patching](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/fixups.c) — the `nospec` instruction emitted in front of a marked instruction.
- [Speculation-barrier predicate](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/include/linux/bpf.h) — `bpf_bypass_spec_v1()` and the `CAP_PERFMON` bypass.
- [Commit that makes nospec a v1 barrier](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/patch/?id=dff883d9e93a) — widened semantics and the per-arch vulnerability table.
- [Commit that adds verifier fallback to nospec](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/patch/?id=d6f1c85f2253) — the accept-with-barrier scheme, and the architectures that emit a barrier.
- [Commit that forces BPF_JIT_ALWAYS_ON](https://git.kernel.org/pub/scm/linux/kernel/git/bpf/bpf-next.git/patch/?id=8db2aca155a4) — in bpf-next: the sysctl pinning, the `-ENOTSUPP` fallback change, and the 32-bit x86 exception.
- [bpf-next Kconfig after the change](https://git.kernel.org/pub/scm/linux/kernel/git/bpf/bpf-next.git/plain/kernel/bpf/Kconfig) — `ARCH_WANT_BPF_JIT_ALWAYS_ON`, the hidden prompt, and the KMSAN comment.
- [Sysctl handler](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/net/core/sysctl_net_core.c) — `bpf_jit_enable` existing only under `CONFIG_BPF_JIT`, pinned to 1 under `BPF_JIT_ALWAYS_ON`.
- [Kernel Spectre v2 documentation](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/Documentation/admin-guide/hw-vuln/spectre.rst) — retpoline chosen by mitigation mode and disabled at run time under IBRS.

## Community discussion today

Coverage this run: two archive-opted Slack workspaces (three allowlisted
channels) were read through the approved read-only archive and held no message in
either the 24-hour or the 7-day window, so the Slack side is quiet rather than
unread. The two allowlisted Discord workspaces (ten channels) could not be read:
the web client was at its login screen, so their activity is a coverage gap, not
silence. r/eBPF was read as a normal public page and carried project
announcements plus two real practitioner threads. The day's technical argument
itself ran on the public bpf mailing-list archive, which is where the question
above comes from; the rest of that day's list traffic supplies the themes below.

**One barrier, two safety stories.** The arm64 patch that opened the day wanted a
version of the barrier the interpreter could execute: rename the arm64 helper so
the generic `barrier_nospec()` becomes a real barrier, and keep `copy_from_user()`
from paying for it by teaching the uaccess code that arm64 masks user pointers in
software anyway (`uaccess_mask_ptr()` clears bit 55, so a mispredicted
`access_ok()` cannot reach a kernel address). Reviewers rejected the premise
rather than the patch: on bpf-next, arm64 compiles without an interpreter
whenever the JIT is built, so the only kernels that run interpreted BPF are the
ones deliberately built with `BPF_JIT=n`; and even a correct barrier would sit
behind the computed goto, so a program that trains that branch resumes past it.
The thread ended with the patch dropped in favour of making the Kconfig choice
disappear. The unresolved boundary in that thread is Spectre v2: with CSV2 the
branch predictor is isolated *between* contexts, but a BPF program can train the
predictor from inside the kernel, and nothing in the series addresses that. The
diagnostic a reader can apply today is the one in the verification section — dump
the xlated form, dump the jited form, and treat "not jited" as the case where
the barrier argument changes.

**A Kconfig change with three visible side effects.** The force-the-JIT patch is
small but changes behaviour a reader will meet: `bpf_jit_enable` becomes a pinned
1; a program the JIT cannot compile is refused with `-ENOTSUPP` instead of
silently running interpreted, which includes classic socket filters and seccomp
filters; and the choice is removed only for x86-64 and arm64, because the 32-bit
x86 JIT does not implement every instruction. The KMSAN exemption is the part
most likely to be forgotten: sanitizer kernels keep the interpreter because the
sanitizer does not observe JITed stores, which shows up as a false report about a
map key on the program stack. If you build a debug kernel with KMSAN and expect
the "no interpreter" property, it is not there.

**A map value that is not 8-byte aligned, and an arm64 panic.** A separate report
showed the verifier's alignment rule for map values is a claim about the base
pointer, not a proof of it. `check_ptr_alignment()` checks `off` against the
value start (`check_generic_ptr_alignment()` tests only the register offset plus
`off`), and the code assumes the base is at least 8-byte aligned — true for hash
maps, whose values are placed after a key size rounded up to 8. The LPM trie does
not round: `data_size` is `key_size - 4`, and the value follows the key bytes
with no padding, so an IPv4 trie (key 8, data 4) returns a value pointer that is
4-byte aligned. A 64-bit atomic add at offset 0 passes verification and then, on
arm64, takes an alignment fault that the fault handler does not fix up, ending in
a panic from `bpf_prog_test_run`. The practical check is cheap: take a program
that does a 64-bit atomic on an LPM trie value and run it on the architecture
that is strict about alignment, not on x86-64, which tolerates the same access.
What the fix must accept is that the value layout is part of the map's memory
accounting, so padding it changes `leaf_size` and the `memlock` charge, and the
key format seen by users must stay unchanged.

**A verifier crash from a union member read at the wrong type.** A struct_ops
program calling a helper that returns `PTR_TO_MEM | PTR_MAYBE_NULL` can make the
verifier dereference a null `struct btf *`. `check_return_code()`'s struct_ops
branch reads `reg->btf` and `reg->btf_id` without first checking that the
register is a BTF pointer; in the register state union those fields overlap
`mem_size`, so the verifier passes a size (64 in the report) to `btf_type_by_id()`
as a BTF id. The report includes the instruction list and the jump offsets that
reproduce it, and the boundary it exposes is general: the struct_ops return path
is the only place that inspects union members by type, so a new register type
that aliases them needs that check. A boundary reader-facing point: this is a
verifier crash, not a program rejection, so a load that "should" fail quietly can
instead take the kernel down in the load path.

**The hook prototype is the API, and it moved.** A 7.4 inode hook gained a new
*first* argument, which shifted every existing argument by one for BPF LSM
programs compiled against the old prototype: they keep loading and then read the
wrong argument, which is the worst failure mode for a security program. Thread
answers converged on mechanics rather than policy: a program can already force a
load failure on a prototype mismatch by declaring a matching union of function
pointers so that the BTF-checked signature differs, and a newer series proposes
CO-RE relocation of tracing-program arguments, resolving "the argument named
dentry" at load time so an added argument anywhere but the end cannot silently
shift. The unresolved part is ownership of the abstraction: maintainers disagree
whether the stable-API layer belongs in the LSM framework, the BPF side, or
nowhere.

**Ring buffer records that may change under the reader.** A libbpf series added a
record-at-a-time API on top of the ring buffer, aimed at overwrite mode where the
kernel may rewrite the oldest record while user space is reading it. The shape
worth remembering for anyone consuming such a map: `ring__next()` returns a
record without consuming it, `ring__release()` consumes the batch, and
`ring__invalidated()` tells the reader whether any returned record may have been
overwritten — so the caller copies the bytes first and uses the copy only if the
invalidation check passes afterwards. The related kernel-side series orders the
record data after the producer position so an overwrite cannot publish a
half-written record. What is still open is the cost of that check on a busy ring:
it is a second read of shared state per batch, and the series does not quantify
it.