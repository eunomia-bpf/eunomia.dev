# Why can a verified BPF program panic in the interpreter, and what makes the kernel refuse that fallback?

**Short answer:** verification and code generation are separate passes, and several
instructions the verifier *inserts* have meaning only inside a JIT. A program is
verified against its own instruction stream; then `bpf_do_misc_fixups()` rewrites
it — a helper call becomes an internal `mov` that adds a per-CPU base offset, an
arena access becomes a probe load with a mode only the JIT decodes, a helper the
JIT inlines keeps its raw helper id in `imm`. The interpreter knows none of
these. So the kernel carries one per-program bit, `jit_required`, raised by every
such rewrite, and refuses to load the program when the JIT is unavailable instead
of running it interpreted. Once the fallback is refused, the failure mode is a
load error; before that it was a page fault or a `BUG_ON()` in kernel context.

## The mechanism

### 1. Rewrites happen before the JIT decision

`bpf_check()` runs the passes in a fixed order: the main verification loop, then
`bpf_convert_ctx_accesses()`, then `bpf_do_misc_fixups()`, then the 32-bit
optimizations, then `bpf_fixup_call_args()`, and only at the very end
`__bpf_prog_select_runtime()`, which picks the interpreter or compiles the
program. Everything the fixup pass inserts is therefore part of the instruction
stream that the interpreter will execute if the JIT never runs:

```c
if (ret == 0)
        ret = bpf_convert_ctx_accesses(env);
if (ret == 0)
        ret = bpf_do_misc_fixups(env);
...
if (ret == 0)
        ret = bpf_fixup_call_args(env);
```

The JIT is not the producer of these instructions; the verifier is. That is the
whole reason the interpreter has to be able to execute them, and the reason it
cannot is that they are internal encodings, not UAPI.

### 2. The per-CPU `mov` and the interpreter switch

Resolving a per-CPU address in native code is a single instruction, so the fixup
pass emits one. The encoding reuses `BPF_ALU64 | BPF_MOV | BPF_X` with an
out-of-band offset:

```c
#define BPF_ADDR_PERCPU (-1)
#define BPF_MOV64_PERCPU_REG(DST, SRC)                         \
        ((struct bpf_insn) { .code = BPF_ALU64 | BPF_MOV | BPF_X, \
                             .dst_reg = DST, .src_reg = SRC,       \
                             .off = BPF_ADDR_PERCPU, .imm = 0 })
```

The instruction-set document says only three arithmetic instructions carry a
non-zero `off` (`SDIV`, `SMOD`, `MOVSX`), so `off = -1` is outside the ABI by
construction. The JITs that support it decode it before the ordinary `mov` path
and emit the base addition — on x86-64 a `gs:` add of `this_cpu_off`, on arm64 a
read of the per-CPU base register and an `add`.

The interpreter has no such case. Its double-word move is a switch over `off`
with arms for 0, 8, 16 and 32 and **no default arm**:

```c
ALU64_MOV_X:
        switch (OFF) {
        case 0:  DST = SRC;       break;
        case 8:  DST = (s8) SRC;  break;
        case 16: DST = (s16) SRC; break;
        case 32: DST = (s32) SRC; break;
        }
        CONT;
```

Three of the four generated sequences move a register onto itself
(`BPF_MOV64_PERCPU_REG(r0, r0)`), so with `off = -1` the register is left
holding exactly what it held before: a per-CPU *offset* taken from the preceding
`ld_imm64`, not an address. The load that follows then reads through it. That is
the reported failure shape — a supervisor read fault at an address in the
per-CPU data range, from a program that had just passed verification.

### 3. Helper calls the JIT inlines but the interpreter does not

Some JITs emit the body of `bpf_get_smp_processor_id()` and the two
current-task helpers directly. When the architecture reports that, the fixup pass
skips rewriting the call's `imm` — which is supposed to be an offset from
`__bpf_call_base` — and leaves the raw helper enum id there:

```c
/* Skip inlining the helper call if the JIT does it. */
if (bpf_jit_inlines_helper_call(insn->imm)) {
        prog->jit_required = 1;
        goto next_insn;
}
```

The interpreter's call arm is `(__bpf_call_base + insn->imm)(...)`. With a raw
id of a few units in place of a real offset, that is a call into the middle of
the kernel text — an instruction-alignment fault or a jump into whatever bytes
happen to sit there. This is a *second*, independent way to reach the same
panic: no per-CPU instruction is involved at all, only a helper the JIT normally
inlines.

### 4. Arena encodings that the interpreter does not decode at all

Arena accesses are rewritten to `BPF_PROBE_MEM32` loads and stores and to
`BPF_PROBE_ATOMIC`, and an address-space cast keeps its own `off` marker. The
interpreter's dispatch table is a 256-entry computed goto with a default arm, and
the probe modes are not in it:

```c
default_label:
        pr_warn("BPF interpreter: unknown opcode %02x (imm: 0x%x)\n",
                insn->code, insn->imm);
        BUG_ON(1);
```

So an interpreted arena store reaches `default_label` and takes the kernel down
on an instruction the verifier itself produced. This case differs from the two
above in kind: the fixup pass had already made the program uninterpretable, and
the only thing missing was the refusal.

### 5. Where the bit is raised

`struct bpf_prog` has a `jit_required:1` flag. It is initialised at allocation
from the config, so on a `CONFIG_BPF_JIT_ALWAYS_ON` kernel every program carries
it and any JIT failure is a load failure:

```c
fp->jit_requested = ebpf_jit_enabled();
fp->jit_required = IS_ENABLED(CONFIG_BPF_JIT_ALWAYS_ON);
```

The verifier raises it for every construct that only a JIT implements, in the
pass that introduces the construct:

- a kfunc call — `check_kfunc_call()` above it calls `bpf_add_kfunc_call()`;
- a subprogram whose return value spans more than one register — the
  return-register-pair pass in `bpf_compute_subprog_ret_regs()`;
- an indirect call (`callx`) — `require_callx_jit()`;
- stack unwinding — the check in `check_unwind_insns()`;
- an arena map — `check_map_prog_compatibility()`;
- an instruction-array map — `claim_used_maps()`;
- a per-CPU map lookup inlined by `map_gen_lookup` — `bpf_do_misc_fixups()`;
- a per-CPU `ld_imm64` rewritten into the special `mov`, and the inlined
  `get_smp_processor_id` / `get_current_task` sequences — `bpf_do_misc_fixups()`;
- a helper the JIT inlines — `bpf_do_misc_fixups()`;
- a timed `may_goto` — `bpf_do_misc_fixups()`;
- a modify-return target returning 64 bits — `check_attach_btf_id()`.

### 6. Where the bit is enforced — and why the errno differs

There are two enforcement points, and which one fires decides the error you see.

The first is inside the verifier, at the end of `bpf_fixup_call_args()`, guarded
by `#ifndef CONFIG_BPF_JIT_ALWAYS_ON` and reached only when the JIT split did not
succeed:

```c
if (prog->jit_required) {
        verbose(env, "program requires BPF JIT compiler but it is not available\n");
        return -EINVAL;
}
```

The second is in `__bpf_prog_select_runtime()`, on the path where the JIT was
requested but returned a program that is not jited:

```c
bool jit_needed = fp->jit_required;
...
if (!bpf_prog_select_interpreter(fp))
        jit_needed = true;
...
if (!fp->jited && jit_needed) {
        *err = -ENOTSUPP;
        return fp;
}
```

So a kernel with `net.core.bpf_jit_enable = 0` rejects such a program during
verification with `EINVAL` and the message above, while a kernel that tried to
JIT and could not (constant blinding failing, for instance) rejects it from
runtime selection with `EOPNOTSUPP`. Both are load errors; neither leaves a
program that will fault later.

Note the shape of the mechanism: the interpreter is installed in `fp->bpf_func`
*before* the JIT is attempted, so a failed compile normally just leaves it in
place and the program runs — silently, on a stream of instructions the
interpreter cannot execute. `jit_required` exists to turn exactly that silent
fallback into a refusal.

### 7. How a working program reaches the interpreter anyway

The reported path was not "JIT disabled". It was constant blinding. With
`net.core.bpf_jit_harden = 2`, every constant-bearing instruction is rewritten
into a random-immediate sequence, which multiplies the instruction count, and a
branch offset that grows past the 16-bit field cannot survive it:

```c
} else {
        off = insn->off;
        off_min = S16_MIN;
        off_max = S16_MAX;
}
...
if (off < off_min || off > off_max)
        return -ERANGE;
```

That error aborts blinding, and blinding failure is a documented fallback: the
original program — the one the fixup pass already rewrote — is restored and the
interpreter runs it. On a kernel without `BPF_JIT_ALWAYS_ON` the load therefore
succeeds, and the crash happens at execution time, in whatever context attached
the program. On a kernel with the flag, the same failure is a load error.

## Verifying it on a real kernel

1. Dump both forms of the same program. `bpftool prog dump xlated` prints the
   special instructions as what they are — `r0 = &(void __percpu *)(r0)` for the
   base addition, a `(c0) nospec` style marker for other internal opcodes — while
   `bpftool prog dump jited` shows the native sequence. If the jited dump fails
   with `not jited`, the program is running interpreted and every internal
   instruction in the xlated dump is being executed by a decoder that may not
   have a case for it.

2. Read the two sysctls that decide the path. `bpf_jit_enable` exists only when
   `CONFIG_BPF_JIT` is built in; under `BPF_JIT_ALWAYS_ON` its minimum and
   maximum are both 1, so a write fails and the kernel logs once that the value
   is pinned. `bpf_jit_harden = 2` enables blinding for every user, including
   the privileged loader you are testing with.

3. Distinguish the two error paths by errno and message. `EINVAL` with
   `program requires BPF JIT compiler but it is not available` comes from the
   verifier, and means the JIT was never used for this load. `EOPNOTSUPP` comes
   from `__bpf_prog_select_runtime()`, and means the JIT was attempted and
   produced nothing.

4. Reproduce the fallback deliberately: set `bpf_jit_harden = 2`, build a
   program with a per-CPU map lookup and a conditional jump of more than about
   11,000 instructions, and load it on a kernel built without
   `BPF_JIT_ALWAYS_ON`. The blinding rewrite overflows the 16-bit jump and the
   program falls back. On a kernel that predates the fix, execution faults; on a
   current one the load fails with the message from step 3.

## The limits

`jit_required` is a load-time guard, not a correction of the interpreter. It
cannot make the interpreter understand an internal encoding; it only guarantees
that a program containing one is never handed to it. Where the bit is not raised,
the old behaviour still applies.

**Longterm trees lag.** The four commits are in the 7.2.y tree, but neither
6.12.y nor 6.18.y has them — and those are the LTS trees a current distribution
ships, which is where the reported panics came from. The backport is awkward
because the check now lives in `kernel/bpf/fixups.c`, a file that only exists
since 7.1, so the older trees need the code moved rather than cherry-picked. A
stable kernel can therefore still panic where a current one returns `EINVAL`.

**The 64-bit return of a modify-return program.** The interpreter's outermost
frame returns `unsigned int`, so the upper 32 bits of `R0` are dropped on the
way back to the caller. For a target whose return is a 64-bit value, a program
that returns `ERR_PTR(-EINVAL)` becomes `0xffffffea`, which `IS_ERR()` does not
recognise. Only functions the build tags as error-returning require the JIT
today; a series under review extends that to every modify-return target with an
8-byte return, and also restricts what such a program may return when the target
returns a pointer. Until it lands, an untagged 64-bit target can still be
modified by an interpreted program whose error value is silently corrupted.

A third, milder boundary: the per-CPU instruction only exists on architectures
whose JIT reports support for it. Elsewhere the fixup pass leaves the helper
call alone, so per-CPU lookups keep working when interpreted — at the cost of the
inlining, not of correctness.

## References

- [BPF mailing list: report of an interpreter fallback that panics](https://lore.kernel.org/bpf/MR1P264MB331494502C9D13D8A9EAFB99B7E82@MR1P264MB3314.FRAP264.PROD.OUTLOOK.COM/) — a verified program, a map lookup, and a long jump under JIT hardening.
- [BPF mailing list: backport request and its answer](https://lore.kernel.org/bpf/809f91e2-638d-4e54-9b60-e66098d173e0@linux.dev/) — which commits fix it, which longterm trees still lack them, and why the backport does not apply cleanly.
- [Commit that introduces the flag](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/patch/?id=9a6df65d5c6a) — `jit_required`, and the replacement of the kfunc-only check.
- [Commit that rejects inlined helpers without a JIT](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/patch/?id=f1c27922576e) — the skipped `imm` rewrite and the unpatched call id.
- [Commit that refuses the per-CPU instruction without a JIT](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/patch/?id=7a0855e73757) — the fault report included, and the two places that raise the bit.
- [Commit that refuses arena instructions without a JIT](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/patch/?id=34746b5a84ec) — the `unknown opcode` panic and the map-compatibility hook.
- [Commit that introduces the per-CPU instruction](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/patch/?id=7bdbf7446305) — the internal encoding and its disassembler form.
- [BPF interpreter and runtime selection](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/core.c) — the `ALU64_MOV_X` switch, the dispatch table, `bpf_prog_select_interpreter()`, `__bpf_prog_select_runtime()`, and constant blinding.
- [Verifier fixups](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/fixups.c) — every inlining site that raises the bit, and the verifier-time refusal.
- [Verifier](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/verifier.c) — the pass order at the end of `bpf_check()`, and the checks that require a JIT.
- [Program flag definition](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/include/linux/bpf.h) — `struct bpf_prog` bitfields.
- [Per-CPU instruction and JIT capability hooks](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/include/linux/filter.h) — the encoding, `bpf_jit_supports_percpu_insn()`, and `ebpf_jit_enabled()`.
- [Disassembler](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/disasm.c) — how the xlated dump renders the internal instructions.
- [Per-CPU array lookup inlining](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/arraymap.c) — the generated instruction sequence and its capability gate.
- [x86-64 JIT](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/arch/x86/net/bpf_jit_comp.c) and [arm64 JIT](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/arch/arm64/net/bpf_jit_comp.c) — native emission for the per-CPU instruction and the inlined helpers.
- [eBPF instruction-set document](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/Documentation/bpf/standardization/instruction-set.rst) — the `off` field and the three instructions allowed to use it.
- [BPF sysctl documentation](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/Documentation/admin-guide/sysctl/net.rst) — `bpf_jit_enable` and `bpf_jit_harden` values.
- [Sysctl handler](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/net/core/sysctl_net_core.c) — the pinned value and the one-time log line under `BPF_JIT_ALWAYS_ON`.
- [Kconfig](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/Kconfig) — `BPF_JIT_ALWAYS_ON` and what it removes.
- [BPF mailing list: modify-return programs and 64-bit returns](https://lore.kernel.org/bpf/20261125102810.699464-1-a.s.protopopov@gmail.com/) — the series cover letter, its scope, and the interpreter's truncated return.
- [BPF mailing list: the JIT requirement for a 64-bit modify-return target](https://lore.kernel.org/bpf/20261125102810.699464-2-a.s.protopopov@gmail.com/) — why the tagged-function check is not enough.

## Community discussion today

Coverage: two archive-opted Slack workspaces (three allowlisted channels) were
read through the approved read-only archive and held no message in either the
24-hour or the 7-day window, so that side is quiet rather than unread. The two
allowlisted Discord workspaces (ten channels) could not be read — the web client
was at its login screen — so their activity is a coverage gap, not silence. The
r/eBPF public page was read as an ordinary web page and carried project
announcements rather than a decisive technical question. The question above came
from the public bpf mailing-list archive, which also supplies the themes of the
rest of this section.

**A fallback that was never designed to be correct.** The day's technical anchor
was a stable-tree backport request for the four commits in the answer above. What
makes the thread useful for a reader is the disagreement about *where* the
refusal belongs: the reporter's reproducer needs three conditions at once — a
kernel without `BPF_JIT_ALWAYS_ON`, `bpf_jit_harden = 2`, and a jump that grows
past the 16-bit range when constants are blinded. Reduce any one of them and the
bug disappears, which is why it took a reduced reproducer to find and why the
reporter's first reading of the trace ("the interpreter mis-executes the map
lookup") was closer to the truth than "the JIT is broken". The maintainer
answer added the piece the trace hides: the failure is that the crash needs a JIT
*available* first, because the special instruction only exists after the fixup
pass and the fixup pass only emits it when the architecture reports support.
Boundary that remains open: the same shape can be created by any future
JIT-only rewrite, so the flag is now the place each new one must be raised.

**Verifier bookkeeping that depends on which path ran.** A five-patch series
fixes four places where per-instruction auxiliary state outlives the verification
path that produced it. The most interesting is the interaction with the
speculative-execution fallback: a check that fails on a speculative path marks
the instruction and ends the path, but the same instruction is later verified
again on a non-speculative path with a pointer of a different type — and the
`nospec_result` and `ctx_field_size` written by the failed attempt still apply.
The fix saves and restores both around the per-instruction check. Two companion
patches close the matching emission gaps: a barrier is now emitted after a
`BPF_ATOMIC` store to the stack, not only after a plain store, and a speculative
path no longer ends at a call, because no barrier is emitted after a call. A
fourth makes the read of a read-only map value use an unaligned accessor, since
that read runs before the alignment check. The practical diagnostic for readers
of this class of bug: verify the same program twice with different pointer
provenance at one instruction, and compare — aux state is indexed by
instruction, not by state.

**An error pointer that is not an error pointer.** The second half of the day's
modify-return discussion is the mechanism worth remembering.
`IS_ERR()` works on the full 64 bits of a pointer, and the interpreter returns
32. A program that attaches to a function returning 64 bits and returns an error
pointer is therefore correct only when jited; interpreted, the value reaches the
caller as `0xffffffea`-style garbage that passes every `IS_ERR()` test. Two
further boundaries come out of the thread: the verifier already let such a
program attach to a function returning a pointer, so a program could return an
arbitrary pointer that the caller dereferences; and the build-time tagging of
error-returning functions only covers integer returns, so an ERR_PTR-returning
target is untagged and needs no JIT today. The unresolved part is caller
coverage: some callers of even the tagged functions never check the return
value, and a few functions turn an error pointer into NULL before returning, so
tagging is a statement about the prototype, not about the call sites.

**Permissions that only work in the initial namespace.** A seven-patch series
targets a concrete usability hole: a tracing application run inside a user
namespace with a delegated BPF token can attach to a vmlinux function but not to
a function in a module. Two independent causes, both worth knowing when
debugging the same symptom. The kernel's `BPF_BTF_GET_NEXT_ID` demands global
`CAP_SYS_ADMIN`, which a token does not supply; and libbpf loads the token at
program load, which is after the point where it enumerates kernel BTF, so even a
kernel willing to accept the token would be asked without one. The fix relaxes
the permission check to `CAP_BPF` plus a token and moves token acquisition to
object creation. Diagnostic: if the failing lookup is a module BTF id and the
loader is a container, suspect the enumeration step rather than the attach
permission, which is already token-aware.

**Allocation variants that were chosen for the worst context.** Local storage
(task, cgroup, inode) allocated every new object with `kmalloc_nolock()`, whose
only virtue is that it cannot sleep or spin on the zone lock. The cost shows up
under memory pressure: the lock-free path fails where an ordinary allocation
would have succeeded, for instance when one zone is drained below its watermark
and free pages sit in another. A four-patch series picks the variant at call
time — sleeping allocation for a sleepable program, a nowait allocation when the
context permits the zone lock, and the lock-free path otherwise — and threads the
program's auxiliary metadata down to make that decision. The boundary it leaves
open is what "context permits the zone lock" means at runtime: the series reads
preemption and NMI state and the program's sleepability, and a caller that
already holds the lock is still on its own.

**Where the null-mm check belongs.** A ten-patch RFC converts user-access
checking so that a user access with no current address space is refused and
warned about, instead of silently succeeding on a stale address limit. The design
question in the thread is placement: putting the check inside `access_ok()` makes
it impossible to forget, but requires the architecture-specific `access_ok()`
implementations to gain it, and leaves `__access_ok()` as a plain range check.
The review converged on a split — checked wrappers carry the check, the
`__`-prefixed accessors keep an explicit one until their callers are converted,
and the masked-pointer users check the address space themselves because they
already clear the tag bits. What is still uncovered is named in the thread rather
than hidden: x86 user accesses do their own range check in assembly, and a few
architecture-specific paths call the unchecked form directly.

**Two ways for a security program to read the wrong thing.** Two independent
reports on the same day both show an LSM program that loads, runs, and
misbehaves. In the first, a hook that passes a NULL object for informational
commands was annotated as nullable while the argument is in fact *trusted*, and
trusted pointers are the ones the verifier does **not** rewrite into a
fault-protected load — so an unconditional dereference faults in kernel context
rather than reading zero. The review's correction mattered for stable: the
behaviour changed when trusted arguments stopped being probe-converted, which
makes it a regression rather than a long-standing invariant, and the load error
for the non-nullable variant is `EACCES` from the memory-access check, not
`EPERM` from the attach path. The second report is a socket hook that runs while
a child socket is still half-initialised, so an option written there reaches
shared or not-yet-allocated state; the open question in the thread is whether
the hook should be moved later in the clone path or removed from the list of
hooks where the socket may already be modified, and the answer is a trade between
"options can be set early" and "everything the option touches exists".

**Ring buffer records that change under the reader.** A libbpf series adds a
record-at-a-time API for overwritable ring buffers, where the kernel may rewrite
the oldest record while user space reads it. The shape worth remembering:
`ring__next()` returns a record without consuming it, `ring__release()` consumes
the batch, `ring__rewind()` returns to the first unreleased record, and
`ring__invalidated()` tells the reader whether any returned record may have been
overwritten — so the reader copies first and trusts the copy only if that check
passes afterwards. The related kernel-side work orders record data after the
producer position so an overwrite cannot publish a half-written record. What is
still unquantified is the cost of the check on a busy ring: it is a second read
of shared state per batch, and the series does not measure it.