# Why do signed division and sign-extending moves yield unsigned results in the i386 BPF JIT, and how does that escape a verifier range?

Because the BPF instruction set selects three instructions by the `off` field, and the 32-bit x86
JIT decides which instruction it is emitting from the opcode alone. Signed division and signed
modulo are `DIV` and `MOD` with `off = 1`; a sign-extending move is `MOV` with `off` of 8, 16 or
32. The interpreter switches on that offset, the verifier proves ranges from it, and the x86-64,
arm64, RISC-V and s390 JITs branch on it, but the i386 emitter takes only the opcode, so it emits a
plain `div` and a plain move. A verifier-accepted program therefore gets a value its range proof
excluded: `w0 = -1; w0 s/= 2` returns `0x7fffffff` instead of 0, and `r0 = (s8)0x80` returns 128
instead of -128. The gap is not a verifier bypass by itself; it becomes one when the program uses
the value as a bound, which is what makes it worth detecting.

## The mechanism

1. **The offset is the opcode extension.** The instruction-set document says it directly: only
   `SDIV`, `SMOD` and `MOVSX` have a non-zero offset, and the offset is what distinguishes
   `dst s/ src` from `dst / src`, and a sign-extending move from a plain move. The interpreter
   implements that as a `switch (OFF)` in the `ALU_MOV_X`, `ALU_DIV_X`, `ALU_MOD_X` and their
   `ALU64` counterparts, so the offset is load-bearing for execution, not a hint.

2. **The verifier derives its proof from the same field.** `check_alu_fields` accepts `off` of 0
   or 1 for `DIV` and `MOD`, and 0, 8, 16 for a 32-bit `MOV` register form, 0, 8, 16, 32 for the
   64-bit one. Range tracking then branches on it: with `off == 1` the division case calls the
   signed range helper, and the move case calls the sign-extending coercion. For `-1 s/ 2` the
   signed helper computes `res1 = -1 / 2` and `res2 = -1 / 2`, so the destination is proved to be
   exactly 0. That is the range the JIT is then trusted to honour.

3. **The JIT family is split on where the offset is read.** The x86-64 emitter branches on
   `insn->off == 1` both in its constant-divisor strength reduction and in the register path, where
   `off == 0` emits `xor edx, edx` followed by `div` and the other branch emits `cdq` or `cqo`
   followed by `idiv`; a register move with a non-zero offset goes to the sign-extending move
   emitter. arm64 emits `SDIV`, RISC-V emits `divw`/`remw`, s390 emits `dsgfr`/`dsgr`, and parisc
   routes to a signed libgcc helper, all keyed on `off == 1`. So the mechanism is known to every
   architecture that implements it.

4. **The i386 helper cannot see the instruction.** `emit_ia32_div_mod_r` takes an operation byte,
   a destination, a source and two stack flags; the caller passes `BPF_OP(code)`. The helper loads
   the operands, then emits `xor edx, edx` and `div ecx` unconditionally, because that is the only
   form a 32-bit unsigned division needs. The `MOV` case is the same shape: the code path for
   `BPF_ALU | BPF_MOV | BPF_X` and `BPF_ALU64 | BPF_MOV | BPF_X` calls the plain move helper,
   which takes `is64` but not `off`. Nothing in either path reads `insn->off`.

5. **Only the 32-bit signed division is miscompiled, because the 64-bit one never reaches the
   JIT.** The i386 emitter sends `BPF_ALU64 | BPF_DIV` and `BPF_ALU64 | BPF_MOD` to its
   "not yet" label, which returns `-EFAULT`; the compile loop then returns the program unjited. A
   failed compilation falls back to the interpreter unless the program is one that requires the
   JIT — a kernel built with JIT always-on, or a program that needs it, such as one using arena or
   a subprogram returning a pair of registers — so a 64-bit division on i386 is slow but correct.
   A 32-bit division with `off = 1` matches the `BPF_ALU` case and is emitted as unsigned, and
   both the 32-bit and the 64-bit signed moves are emitted as plain moves. The dangerous shape is
   exactly this one: an instruction the JIT accepts and compiles wrong, rather than rejects.

6. **What "outside the proved range" buys an attacker.** The verifier only knows what it proved, so
   every later check built from that range is now unsound. A helper argument length is the textbook
   case: if the program computes a length with a signed 32-bit division or a sign-extending cast,
   the verifier may prove the length is non-negative and small, while the emitted code produces
   `0x7fffffff`, and the length check that follows is a check the program already passed in the
   verifier's model. The series reports exactly that: on an unpatched i386 kernel three
   verifier-accepted test programs passed helper lengths beyond the proved bound and the helper
   wrote that many bytes.

7. **The fix is smaller than the bug's class.** The posted series reads `insn->off` inside the two
   emitters: for `off == 1` it emits `cdq` and `idiv` instead of `xor`/`div`, and for a move it
   emits `movsx` for the 8- and 16-bit widths and uses `cdq` to supply the high word for the
   32-bit width, filling the high half in the 64-bit case and zeroing it in the 32-bit case when
   the verifier has not already inserted the zero extension. It deliberately adds no new
   instruction support, which is why it is small.

8. **`idiv` cannot trap here, and that is why the fix can be that simple.** Two exceptions make a
   raw `idiv` unsafe: a zero divisor, and `INT_MIN / -1` (or `LLONG_MIN`), where the quotient does
   not fit. Both are removed before the JIT runs. A constant `-1` divisor on a signed operation is
   rewritten into a negation or a zero move, and the register form is wrapped in a check that
   turns a zero divisor into 0 and the minimum divided by -1 into the minimum itself. That is the
   precondition the interpreter relies on too.

## Verifying it

1. **Confirm the offset semantics from the specification first.** Read the arithmetic table in the
   instruction-set document: `SDIV` and `SMOD` are listed with offset 1, `MOVSX` with 8, 16 or 32,
   and every other arithmetic entry with 0. Without that, "the JIT ignores the offset" is a claim
   about a field that looks optional.

2. **Read the two emitters side by side.** In the 32-bit x86 JIT, find the division helper and
   check its parameter list: if it has no instruction pointer, it cannot branch on the offset. Then
   find the 64-bit x86 emitter's division case and note the `if (insn->off == 0)` that separates
   `xor`/`div` from `cdq`/`idiv`. The difference between the two files is the whole bug.

3. **Run the selftests, not a synthetic program.** The instruction-sweep test has a `movsx` subtest
   that compares JIT output against expected values for the sign-extension widths, and the
   verifier's own signed-division and sign-extension tests exercise the range checks. On a 32-bit
   x86 kernel six of the instruction-sweep subtests, four of the blinded sweep and fifty of the
   signed-division cases fail; the sign-extension patch clears five sign-extension verifier cases.
   The six remaining sweep failures fail without the series as well.

   ```console
   # ./test_progs -t jit_insns,verifier_sdiv,verifier_movsx
   ```

4. **For a running fleet, compare interpreter against JIT.** This needs no source reading and no
   selftest build: on a 32-bit x86 host, run one program that uses `s/=` and `(s8)` casts with the
   JIT disabled and enabled, and diff the results. A divergence is the bug, and the size of the
   divergence tells you whether the value is being used as a bound.

   ```console
   # echo 0 > /proc/sys/net/core/bpf_jit_enable
   ```

5. **Check whether a given kernel carries it.** Both emitters are small and the fix is local, so the
   reliable check is the tree rather than the release number: search the 32-bit emitter for the
   `cdq` opcode (`0x99`) in the division helper and for a `movsx` form in the move helper. A kernel
   whose helper still hard-codes `xor edx, edx` predates the series.

## Limits

- **The series is unmerged.** It was posted as a fix for the bpf tree with two `Fixes` tags, and the
  patchwork state is new. Read it as the intended shape, and re-read the patch before citing the
  offset values it reads.

- **The miscompile is i386-only, and i386 is a shrinking target.** The same series' discussion
  records the question of whether the 32-bit JIT should keep taking work at all. The useful way to
  read the result is the bug class, not the architecture: an instruction whose meaning lives in one
  field of its encoding, dispatched on everything except that field.

- **A plain `MOV` with offset 0 is not the same instruction as a plain `MOV` after the verifier.**
  The verifier marks some 32-bit definitions as needing a zero extension, and the 32-bit x86 JIT
  answers that marker with a `MOV` whose immediate is 1. A reader grepping the emitter for the move
  case sees that special form first and can mistake it for the missing sign extension.

- **`ALU64` division on i386 is correct but not JITed.** Anything that needs it pays the interpreter
  cost, which means a performance result measured on i386 mixes the JIT and the interpreter and is
  not comparable with the 64-bit kernel.

- **This page's verdicts come from source and selftest assertions, not from a local load.** The
  environment used to write it could compile, but its kernel had unprivileged BPF disabled and the
  process had no `CAP_BPF`, so no program could be loaded to observe the divergence directly. The
  reproduction path in section 4 is the one to run on a host that can load programs.

## References

- [BPF instruction-set standard](https://docs.kernel.org/bpf/standardization/instruction-set.html) — the arithmetic table that lists `SDIV`, `SMOD` and `MOVSX` as the only instructions with a non-zero offset, and the `MOVSX` definitions.
- [Kernel `arch/x86/net/bpf_jit_comp32.c`](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/arch/x86/net/bpf_jit_comp32.c) — the 32-bit division helper that receives only an operation byte, the move case that ignores the offset, and the 64-bit division path that bails out to the interpreter.
- [Kernel `arch/x86/net/bpf_jit_comp.c`](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/arch/x86/net/bpf_jit_comp.c) — the 64-bit division case branching on the offset between `xor`/`div` and `cdq`/`idiv`, and the sign-extending move emitter.
- [Kernel `kernel/bpf/core.c`](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/core.c) — the interpreter switches on the offset, and the runtime selection that falls back to the interpreter when JIT compilation is not required.
- [Kernel `kernel/bpf/verifier.c`](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/verifier.c) — the offset validation for arithmetic and move instructions, the signed division range helpers, and the sign-extending move coercion.
- [Kernel `kernel/bpf/fixups.c`](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/fixups.c) — the rewrites that remove the divide-by-zero and minimum-divided-by-minus-one exceptions before the JIT runs, and the dead-code trap that an unprivileged program hits when it takes a branch the verifier proved unreachable.
- [Kernel `arch/arm64/net/bpf_jit_comp.c`](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/arch/arm64/net/bpf_jit_comp.c) — a JIT that emits a signed divide and a sign-extending move from the same offset field.
- [Kernel `arch/parisc/net/bpf_jit_comp64.c`](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/arch/parisc/net/bpf_jit_comp64.c) — a JIT that reads the offset for division and modulo but not for the move.
- [Series cover letter, `bpf, x86: Fix i386 JIT signed ALU32 DIV/MOD and MOVSX`](https://lore.kernel.org/bpf/20261113-i386_bpf_jit_wrong_code_20261028-v2-0-1dd0b03072a3@gmail.com/) — the two instruction shapes, the three test programs that passed a length beyond the proved bound, and the revision history.
- [Patch 1, signed division and modulo](https://lore.kernel.org/bpf/20261113-i386_bpf_jit_wrong_code_20261028-v2-1-1dd0b03072a3@gmail.com/) — the `cdq`/`idiv` emission, the claim that both exceptions are already removed, and the test counts with and without it.
- [Patch 2, sign-extending moves](https://lore.kernel.org/bpf/20261113-i386_bpf_jit_wrong_code_20261028-v2-2-1dd0b03072a3@gmail.com/) — the per-width `movsx` emission, the high-half handling, and the note that the unprivileged range-check subtest hangs without it.
- [Reply on whether the JIT is worth keeping](https://lore.kernel.org/bpf/CAJZwKkiBQJZofwpps43SUGT05Hb4oOHZtv1+J6n_30wqOO-LCA@mail.gmail.com/) — the answer to a proposal to drop the 32-bit JIT rather than keep patching it, and the argument that two instructions already accepted and miscompiled are the minimum. The same reply records that an earlier revision's new instruction support was turned down.
- [Parisc follow-up series](https://lore.kernel.org/bpf/20261113-parisc_bpf_jit_movsx_20261112-v1-0-7ecc33cdf1b2@gmail.com/) — the same move gap in two more JITs, and the note that the 32-bit half came out of an automated review of that architecture's earlier signed-division series.
- [Signed division selftest](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/tools/testing/selftests/bpf/progs/verifier_sdiv.c) — the expected results for signed division, including the negative operands.
- [Sign-extension selftest](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/tools/testing/selftests/bpf/progs/verifier_movsx.c) — the range-check cases that fail when the move does not sign-extend.

## Community discussion today

This run read two chat workspaces through the approved read-only archive, three allowlisted
channels in total, and the public bpf mailing list as the second, public surface. The archive was
quiet: one workspace had no message in either the 24-hour or the seven-day window, and the other
held ten messages in the seven-day window, none in the last 24 hours. All ten were logistics for a
conference maintainer meeting and for project talks at co-located events, plus a note about a third
party presenting network traces combined with continuous profiles for threat detection. None carried
a technical question, so the answer above comes from a public list thread inside today's window;
that list's rolling index showed 19 thread roots among 49 messages in the window when it was read.
Three surfaces could not be covered and are recorded as inaccessible rather than quiet: the two
chat workspaces, ten allowlisted channels between them, answer their public invite links with an
invalid-invite page, and the practitioner forum returned a human-verification challenge to a normal
browser session.

### Instructions selected by the offset field are the day's recurring defect

Two threads in the window are the same shape. The answered i386 series fixes two instructions that
the JIT already accepts and compiles wrong. Its follow-up, posted hours later for PA-RISC, found
that both of that architecture's JITs emit a plain move for a sign-extending one, adds the
extension to each, and credits an automated review of an earlier signed-division series for the
32-bit half of the finding. That earlier series had already made the division and modulo half of
the same gap work on both PA-RISC JITs, so the pattern repeated inside one architecture: the
division instructions were fixed first, and the move instruction was missed because it is selected
by the same field but read somewhere else. The mechanism is identical throughout: the interpreter
switches on the offset, the verifier proves ranges from it, and a JIT that dispatches on the opcode
alone silently emits a different instruction. The practical diagnostic generalises to any port:
grep the JIT for the offset read, and if the emitter's signature has no instruction pointer, it
cannot be reading it. What remains open is coverage auditing rather than any single bug: nothing in
the verifier rejects a miscompile, so a port's correctness rests on its own selftest sweep.

### When a proposed bounds fix is already somewhere else

A proposal to mask the address of a converted absolute packet load, arguing that a transient
out-of-range offset can dereference a secret, was answered with a flat rejection: the reporter
probably had no reproducer, the patch does nothing, because the conversion of a classic absolute
load inlines its own bounds check in the fast path before the helper fallback. The reasoning is
checkable in one function, and the shape recurs: the fast path compares the remaining header
length against the access size and jumps over the load, and the helper fallback returns a negative
value that the converted program turns into zero. The lesson is procedural and worth keeping: when a
patch claims a load can leave its permitted range, read the conversion that produced the program,
not the program the author wrote.

### Failing calls for functions that were never meant to be attachable

A large submission tags every kernel function that returns an error as a modification-return target,
12633 of them, so a program with the two BPF capabilities can fail the call on the way to a bug and
return one of that function's own errors. The design work is in the rules: a function qualifies only
if a caller could already receive each of those errors without a program, and an error is excluded
when every path to its return releases something through a pointer the function received, because
skipping the function leaves that resource held. The measured claim is stark: in a test that
skipped one call per function over 4401 functions of the networking, filesystem and memory
subsystems, one function in 180 damaged the kernel, which is why the proposal carries a deny list by
name. In review the author agreed to drop 537 tags for functions that are not attachable at all,
since the verifier rejects any attachment to them. The open boundary is whether tagging by data flow
over the return value can ever be complete, and how a deny list is maintained as the tree moves.

### Getting a per-owner value one load closer

A request-for-comments series proposes a map flag that stores a local storage value inside the
owner's storage object instead of in a separate element, removing one dependent load from each
lookup on a cold owner. The numbers make the trade-off visible: the inline area is 840 bytes at the
end of the storage structure, the largest value is 832 bytes on a 64-bit kernel, at most sixteen
inline maps exist per owner type, and map creation claims a slot first-fit and fails with a
no-space error or an argument-too-big error. The correctness work is the interesting part for a
practitioner: a value at a fixed offset can outlive the map that claimed it, so each slot carries a
tag taken from a claim number and compared with an acquire load, an object that can hold inline
values lives until its owner dies and is freed after a tasks-trace grace period, the storage
iterator refuses inline maps, and a delete from a non-maskable context returns an unsupported
error, as it already does for normal maps. Two smaller items ride along: the flag needs a 64-bit
kernel, and the library no longer retries a storage map creation without type information, a retry
that always failed and hid the real error.

### A dump helper that read past the buffer it was given

A library fix closes an out-of-bounds read in the type dumper. The zero-scan that decides whether a
value is all zeros recurses into struct members and array elements without checking the caller's data
end, so a caller passing a size smaller than the type walks past the buffer instead of getting the
buffer-too-small error the interface promises. The fix puts the same overflow check the dump path
already performs at the top of the zero-scan, and drops a direct byte comparison in the array branch
in favour of the now-checked element recursion. The test it adds is the useful part: an all-zero
union passed with a size one byte short of the type must return the error rather than read. The
residual boundary is that the check is per-recursion, so a future recursion added without going
through the checked entry point reintroduces the defect.

### A type tag that has nowhere to live

A request for the per-CPU annotation to stop being emitted on variable declarations rested on a
structural fact: every variant of an aggregate shares one debug entry, so there is nowhere to put a
per-variable type tag on the declaration, and compilers disagree about what to do with it — one
drops it silently, one warns. Reviewers confirmed the effect on two compilers and asked for one
confusing sentence to be dropped, and the measured gain is 280 types and roughly 3 KB of type
information gone from the kernel image. The open question is whether the tag belongs on the pointer
use site instead, which is what a reader of a per-CPU access actually needs to know.

### A private stack that a reentered probe can overwrite

A revision of an earlier series narrows which programs get a private stack, on the argument that
the stack is per program and per CPU, so a preempted invocation of a plain kernel or user probe can
have its stack overwritten by the invocation that preempts it. The current tree gives a private
stack to every probe, tracepoint and perf-event program; the patch keeps it only for the
multi-attach and session forms, whose attach type identifies them as capable of recursion, and adds
coverage that the untagged forms get none. In review the author asked for a stable tag on the
change, since a program that had the protection and loses it changes behaviour. The pattern worth
keeping is the shape of the bug: an optimization keyed on a resource being exclusive per CPU, made
unsound by a program class that can be reentered on one CPU.

### The callback frame, one revision later

Yesterday's question on this site was about a callback argument outliving the frame that owned it;
the series that answers it was rebased onto the current tree in this window. The revision keeps the
design and changes the bookkeeping: it folds in an earlier cleanup of the callback register setup,
pairs the frame identifier in the join path so two joined states cannot disagree about which frame
owns an argument, adds a test for that pairing, and drops both the `Fixes` tags and the stable
request. Dropping the fix tags is the interesting editorial choice: the defect allows arbitrary
kernel read and write from a saved pointer, but the change is large enough that the author no
longer claims it as a backport candidate.

### Smaller items in the same window

Two networking series, one reordering device setup so a program is checked for support before any
state changes and rolling back on error, one bounding a virtual receive queue, each drew the same
automated request for whether the defect was triggered or only read out of the code, a request that
states plainly it was produced by a model and may be wrong. The threads answered on technical merit
both turned on reading the code that runs rather than the code that was written:
one conversion path that already bounds its own load, and one per-CPU annotation that cannot attach
to an aggregate.