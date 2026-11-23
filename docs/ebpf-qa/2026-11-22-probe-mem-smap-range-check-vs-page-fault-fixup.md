# Why does an x86 BPF program pay an address check before every untrusted-pointer load, and what must change in the page fault handler before the check can go?

**Short answer:** the load needs an exception-table entry because the verifier
cannot prove the pointer valid, and on x86 an exception-table entry is only
consulted when the fault landed in the kernel half of the address space. With
SMAP enabled, a kernel-mode access to a *user* address with EFLAGS.AC clear is
treated as a missing `STAC`: `do_user_addr_fault()` calls `page_fault_oops()`
before it looks at the exception table, so the fixup that would have zeroed the
destination register never runs. The JIT therefore filters the address itself
before the load — `lea` of the offset address, a shift, a branch — and zeroes
the destination for every address that is not a kernel one. Removing the check
means letting the page fault handler resolve a BPF program's exception-table
entry on that branch, which is eleven lines in `arch/x86/mm/fault.c` and one
lookup, `is_bpf_text_address()`, on the path that oopses today. Whether that
lookup answers correctly for every program that can take such a fault is the
whole safety argument, and it is where the current review disagreement sits.

## The mechanism

### 1. The verifier marks the loads, the JIT gives each one an extable entry

A load becomes `PROBE_MEM` when the verifier cannot prove the pointer. The
rewrite happens once, after verification, in `bpf_convert_ctx_accesses()`:

```c
if (bpf_is_ptr_to_mem_or_btf_id(ptr_type) &&
    bpf_may_fault_on_deref(ptr_type) &&
    type == BPF_READ) {
        if (BPF_MODE(insn->code) == BPF_MEM)
                insn->code = BPF_LDX | BPF_PROBE_MEM | BPF_SIZE(insn->code);
        else
                insn->code = BPF_LDX | BPF_PROBE_MEMSX | BPF_SIZE(insn->code);
        env->prog->aux->num_exentries++;
}
```

and `bpf_may_fault_on_deref()` is exactly two cases: a plain `PTR_TO_BTF_ID`
(what a walk of a pointer field yields when the field is not marked as
inheriting trust) and any type carrying `PTR_UNTRUSTED`. A store through such
a pointer is rejected earlier, so only loads are rewritten.

The x86 JIT then records, for each of those loads, an exception-table entry
whose `data` is `EX_TYPE_BPF` and whose `fixup` word packs two numbers: the
length in bytes of the x86 load instruction it emitted, and which `pt_regs`
slot holds the destination register (`DONT_CLEAR` for the load-acquire forms
that read into a register without naming it in `dst_reg`). On a fault,
`fixup_exception()` walks the kernel table, the module tables and finally
`search_bpf_extables()`, which finds the program by instruction pointer and
searches the program's own table; `ex_handler_bpf()` then zeroes the
destination slot and adds the recorded instruction length to `regs->ip`. That
is the whole recovery: no re-execution, no signal, the program continues with
a zero.

### 2. On a user address, that recovery is never reached under SMAP

Two different fault entry points reach `fixup_exception()`. A fault on a
kernel address goes through `do_kern_addr_fault()` and ends in
`kernelmode_fixup_or_oops()`, whose first action is `fixup_exception()`. A
fault on a user address goes through `do_user_addr_fault()`, and its first
check on an SMAP machine is:

```c
if (unlikely(cpu_feature_enabled(X86_FEATURE_SMAP) &&
             !(error_code & X86_PF_USER) &&
             !(regs->flags & X86_EFLAGS_AC))) {
        /*
         * No extable entry here.  This was a kernel access to an
         * invalid pointer.  get_kernel_nofault() will not get here.
         */
        page_fault_oops(regs, error_code, address);
        return;
}
```

The comment states the design: ordinary kernel code that wants to survive a bad
address uses the nofault accessors, and those filter user addresses in
software before the access — `copy_from_kernel_nofault_allowed()` rejects
anything below `TASK_SIZE_MAX + PAGE_SIZE` and anything inside the vsyscall
page precisely because the fault handler would oops on it. A JITed BPF program
has no such software filter unless the JIT emits one, which is the check under
discussion.

### 3. Why the check cannot simply compare against TASK_SIZE_MAX

The vsyscall page sits at `0xffffffffff600000`, above `TASK_SIZE_MAX`, but
`fault_in_kernel_space()` special-cases it back into user space:

```c
if (IS_ENABLED(CONFIG_X86_64) && is_vsyscall_vaddr(address))
        return false;
return address >= TASK_SIZE_MAX;
```

so a load from that page takes the user-address path and, under SMAP, oopses.
That is not hypothetical: commit `b599d7d26d6a` ("bpf, x86: Fix PROBE_MEM
runtime load check") was written because a program reading `sk_tsq_flags` off a
`sock` could reach it and crash the kernel. The check therefore bounds the
address from both sides: the accepted range is
`[TASK_SIZE_MAX + PAGE_SIZE, VSYSCALL_ADDR)`, computed as one unsigned compare
of `addr - VSYSCALL_ADDR` against
`TASK_SIZE_MAX + PAGE_SIZE - VSYSCALL_ADDR`.

The current encoding no longer needs the constant in a register. Two JIT
commits merged in October rewrote it: `74309a9` folds the offset address into
a single `lea` (7 bytes, one uop), and `181576e` compares against the next
power of two instead of the limit, turning the compare into a shift and a
`jne`:

```
lea  r11, [src + off - VSYSCALL_ADDR]   ; the whole offset fits in disp32
shr  r11, fls64(limit)                  ; 48 on 4-level, 57 on 5-level
jne  load                                ; nonzero => let the load run
xor  dst, dst                            ; else zero the destination
jmp  done
```

Addresses between the limit and that power of two are non-canonical and used
to be zeroed by the fault handler anyway, so excluding them earlier changes
nothing observable.

### 4. What the check costs, and what the proposal measures

Before those two commits the guard was a `movabs` of `VSYSCALL_ADDR`, a move
and an `add` for the offset address, a `sub`, a `movabs` of the limit, a
`cmp`, a branch, a `xor` and a `jmp` — nine instructions, 39 bytes, 32 when the
offset is zero — in front of a load of four to seven bytes. The series under
review, posted against `bpf-next` at `e1d84a37cba9` on 2026-10-09, measures
what removing it buys, using veristat over every BPF selftest object and over
466 production objects:

- BPF selftests: 3324 programs, 91 with `PROBE_MEM` loads. Mean 35.0%,
  median 37.5%, max 74.4% smaller.
- Production objects: 1710 programs, 228 with `PROBE_MEM` loads. Mean 7.0%,
  median 1.8%, max 69.2% smaller.

No program grows. The largest relative saving is a 1039-to-266 byte program;
the socket and task iterators of the selftests lose about half their code.
Both numbers are code size on a tree that still had the nine-instruction form,
so they are an upper bound on what the current, shorter guard costs.

### 5. The proposed change

Patch 1 of the series adds one branch inside the SMAP path, ahead of the oops:

```c
if (is_bpf_text_address(regs->ip) &&
    fixup_exception(regs, X86_TRAP_PF, error_code, address))
        return;
```

It sits where the code already decides to die, so a fault that is handled
today executes nothing new, and a kernel-mode user access from anything but a
BPF program still oopses. Patch 2 then keeps the JIT check only when SMAP is
absent:

```c
probe_mem = BPF_MODE(insn->code) == BPF_PROBE_MEM ||
            BPF_MODE(insn->code) == BPF_PROBE_MEMSX;
bounds_check = probe_mem && !cpu_feature_enabled(X86_FEATURE_SMAP);
```

The feature bit is fixed before any program is JITed: `setup_smap()` sets CR4
from the boot CPU's capability during early CPU setup, and `clearcpuid=smap`
removes the bit from the capability mask even earlier, so a JIT-time decision
is stable for the life of the kernel. Kernels without SMAP keep the check,
which is what preserves the other half of the contract: an SMAP machine turns
a load from a mapped user page into a fault and
therefore into a zero, while on a kernel without SMAP the same load would
simply succeed and read user memory if the guard were dropped.

### 6. Why the lookup is the safety argument

`is_bpf_text_address()` is a latch-tree lookup in `bpf_tree`, not a check of
the address layout. The tree is populated by `bpf_prog_kallsyms_add()`, which
returns early unless the program is JITed, is not a classic socket filter, and
was loaded by a caller holding `CAP_BPF` (directly or through a token that
grants it):

```c
if (!bpf_prog_kallsyms_candidate(fp) ||
    !bpf_token_capable(fp->aux->token, CAP_BPF))
        return;
```

So the fixup is reachable exactly for programs that are in that tree, and
`search_bpf_extables()` additionally requires a non-zero `num_exentries`.
Anything else that faults on a user address still takes the old path. The
interpreter is not affected either way: `LDX_PROBE_MEM` there goes through
`bpf_probe_read_kernel_common()` and the nofault accessors, which do the
software filter themselves.

## Verifying this on a real kernel

1. Look at the emitted guard, not the BPF assembly. `bpftool prog dump jited`
   (or the program's entry in `/proc/kallsyms`) shows the native code; the
   guard is the `lea` with a displacement of `0xA00000` plus the shift and the
   branch in front of each `PROBE_MEM` load. With the series applied, the load
   stands alone and the exception-table entry is the only thing left.

2. Run the address list, not just NULL. The selftest drives one program with a
   userspace-chosen pointer and checks six addresses: `0`, `4096`, the last
   user page (`(1UL << 47) - 4096`), a non-canonical `1UL << 63`, the vsyscall
   page and an offset into it. Every load must read zero and the kernel must
   survive; the same program reading through the current task pointer must read
   real values, which is what proves the zeroing is conditional rather than
   unconditional.

3. Use the test to tell the two mechanisms apart. The selftest passes both
   with the guard and with the fault-handler change, and its own description
   says the first subtest oopses when only the JIT patch is applied — the oops
   is the SMAP branch, and patch 1 is what converts it into a fixup.

4. Check that the right register was cleared. `ex_handler_bpf()` clears the
   `pt_regs` slot named by the fixup word and advances `regs->ip` by the
   recorded length of the load. A wrong slot index does not fault; the program
   continues and reads a stale value, so compare a field the program reports
   against a known userspace value.

5. Confirm the check really is conditional on SMAP. Boot with
   `clearcpuid=smap` (a testing-only option that taints the kernel), dump the
   JIT again, and the guard is back on a kernel that also runs the series.

## The limits

The fixup is only as good as the tree it searches. `is_bpf_text_address()`
answers for JITed, non-classic programs loaded with `CAP_BPF`; a program
outside that set that takes a user-address fault still oopses, and the fault
handler has no way to tell "a BPF program that is not in kallsyms" from
"ordinary kernel code that forgot a `STAC`". That is why the x86 maintainer's
objection is about the lookup and not about SMAP itself: on this path SMAP is
no longer what stops the read from succeeding, it is only what makes the read
fault, and the recovery then depends on a symbol table.

The measured gain is code size, not time. A guarded load costs a few
instructions on every execution; an unguarded load that does fault costs a full
page fault, an `is_bpf_text_address()` lookup and an exception-table search,
then resumes. The series' numbers only count bytes, so a program whose
untrusted loads almost always miss gets smaller and slower, and the review
that raised this asked for the runtime side to be stated.

The claim in the changelog that arm64, riscv, s390 and loongarch already emit
the bare load is not accurate for three of the four. arm64, riscv and
loongarch all select the arch Kconfig symbol
`ARCH_HAS_NON_OVERLAPPING_ADDRESS_SPACE`, which makes the weak
`bpf_arch_uaddress_limit()` return `TASK_SIZE`; for them the *verifier* — not
the JIT — rewrites every `PROBE_MEM` load into a seven-instruction BPF guard
that shifts the address right by 32 and compares against `limit >> 32`. Those
architectures never let a user address reach the load in the first place.
x86 overrides the hook to return 0 because its JIT emits its own check, and
s390 has separate address spaces and relies on the fault path. The comparison
in the cover letter is therefore one architecture, not four.

The change also does not make user memory readable, and that is the point of
the design rather than a side effect: SMAP still blocks the access, the load
still faults, and the fault still produces a zero. What moves is *who* decides
that the address is not a kernel address — the JIT at load time, or the fault
handler at run time. The x86 side has to accept that the handler now contains a
BPF-specific branch; the series offers to put it behind `CONFIG_BPF_SYSCALL`,
which changes where the code lives but not the dependency. As of this writing
the series has an Ack from a BPF reviewer and an explicit rejection from the
x86 maintainer, who asked for the programs to be massaged instead, so the check
is still there in mainline.

Finally, nothing here changes other fault classes: vmalloc faults, faults in
confidential guests on unaccepted pages, and kernel-address faults keep their
existing handling. The new branch is inside the SMAP condition, so a kernel
without SMAP never evaluates it.

## References

- [BPF mailing list: drop the PROBE_MEM address range check under SMAP, cover letter](https://lore.kernel.org/bpf/20261009024925.3169077-1-memxor@gmail.com/) — the mechanism, the veristat measurements, and the four-architecture claim.
- [BPF mailing list: the fault handler change](https://lore.kernel.org/bpf/20261009024925.3169077-2-memxor@gmail.com/) — the eleven-line hunk in `arch/x86/mm/fault.c` and the `is_bpf_text_address()` gate.
- [BPF mailing list: the JIT change](https://lore.kernel.org/bpf/20261009024925.3169077-3-memxor@gmail.com/) — `bounds_check = probe_mem && !cpu_feature_enabled(X86_FEATURE_SMAP)` and the generated assembly before and after.
- [BPF mailing list: the selftest](https://lore.kernel.org/bpf/20261009024925.3169077-4-memxor@gmail.com/) — the six addresses, the valid-address control, and the note that patch 2 alone oopses.
- [BPF mailing list: x86 maintainer objection](https://lore.kernel.org/bpf/20261009041346.GHashp-hYpLl-o-l3M@fat_crate.local/) — "not at all amused from this adding bpf-specific handling to the #PF handler".
- [BPF mailing list: the review reply that closes the thread](https://lore.kernel.org/bpf/20261123003036.GCawOJLKK1iX47XTDm@fat_crate.local/) — "the only line of defense is for is_bpf_text_address() to really DTRT".
- [BPF mailing list: the 2026-11-22 objection](https://lore.kernel.org/bpf/20261122200848.GBawNL0FcQoK918YDI@fat_crate.local/) — "we basically disable SMAP around bpf programs", and the judgement that the saving is not worth the risk.
- [BPF mailing list: the author's 2026-11-22 answer](https://lore.kernel.org/bpf/CAP01T746W=8FFiJJUM3H-Qv09uRO6-XEi9B-8pjSgtpAPBx4EA@mail.gmail.com/) — "strictly contained to BPF programs", and the offer to gate the branch on `CONFIG_BPF_SYSCALL`.
- [x86 page fault handler](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/arch/x86/mm/fault.c) — the SMAP branch ending in `page_fault_oops()`, `kernelmode_fixup_or_oops()`, and `fault_in_kernel_space()`.
- [x86 exception fixups](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/arch/x86/mm/extable.c) — `fixup_exception()` and the `EX_TYPE_BPF` case.
- [x86 BPF JIT](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/arch/x86/net/bpf_jit_comp.c) — the range check, `ex_handler_bpf()`, the fixup word layout, and `bpf_arch_uaddress_limit()` returning 0.
- [BPF core](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/core.c) — `search_bpf_extables()`, `bpf_prog_ksym_find()`, and the `CAP_BPF` condition in `bpf_prog_kallsyms_add()`.
- [Verifier fixups](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/fixups.c) — the `BPF_LDX` to `BPF_PROBE_MEM` rewrite, and the `bpf_arch_uaddress_limit()` guard used on arm64, riscv and loongarch.
- [Verifier pointer-trust predicates](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/include/linux/bpf_verifier.h) — `bpf_may_fault_on_deref()` and `bpf_is_ptr_to_mem_or_btf_id()`.
- [Nofault accessor gate](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/arch/x86/mm/maccess.c) — `copy_from_kernel_nofault_allowed()` rejecting user addresses and the vsyscall page in software.
- [Commit that added the vsyscall bound](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/commit/?id=b599d7d26d6ad1fc9975218574bc2ca6d0293cfd) — the 2024 crash, and why `TASK_SIZE_MAX` alone was not enough.
- [Fold of the address computation into a lea](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/commit/?id=74309a9294a74975fee75e4d6259eef026495325) — 20 bytes and four uops to 7 bytes and one.
- [Shift instead of a register compare](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/commit/?id=181576ebd87d90aafeedbecf7a2dcc109dcff391) — 15 bytes to 6, and the non-canonical band it absorbs.

## Community discussion today

Two archive-opted Slack workspaces were reviewed through the approved read-only
archive: three allowlisted channels in total, no message in the 24-hour window
and one in the seven-day window. That one message proposed an in-kernel binary
pattern-search kfunc; it is a real open question, but it already has a
dedicated answer here, so today's page is built on the public bpf mailing-list
archive instead, which is where the day's actual design argument ran. r/eBPF was
read as a visible browser page and showed project announcements, not questions.
The two allowlisted Discord workspaces, ten channels in total, were not
reviewed this run: the web client was at a login screen with no session, so
their absence is a coverage gap, not silence.

**Nine instructions versus a page fault.** The day's thread is a revival, after
six weeks of silence, of a three-patch series from early October. The author's
argument has two halves: the guard runs in front of *every* untrusted-pointer
load in every program of that shape, and the fault it prevents is one the
handler already knows how to resolve — it just refuses to look at the exception
table when SMAP is on. The x86 side reads the same patch as subverting SMAP:
after it, "we basically disable SMAP around bpf programs when they are about to
access potentially stale memory, and the only line of defense is for
`is_bpf_text_address()` to really DTRT." The maintainer's conclusion was that
the saving is "simply not worth the risk", and, in a later reply, that the
programs should be massaged so the fault looks like any other userspace fault
rather than teaching the #PF handler about BPF. The author's answer is that the
change is strictly contained — the read still cannot succeed, the destination
register is still zeroed, and the branch could sit behind `CONFIG_BPF_SYSCALL`
— and that the cost being moved is paid on a path that otherwise ends in a
panic.

**Two reviewers found the claim about other architectures wrong.** The
changelog says the patch makes x86 behave like arm64, riscv, s390 and
loongarch. An automated review of that patch pointed out that three of the four
never emit a bare `PROBE_MEM` load for a user address: they select the
non-overlapping address space Kconfig, so the verifier rewrites each load into
a seven-instruction BPF guard before the JIT sees it, and only s390 relies on
the fault path. The same review noted that the veristat numbers measure code
size only, and that a load of NULL plus an offset now takes a full page fault
where it used to take a compare and a branch. The other automated review of the
same series raised the question this page's mechanism section turns on: the
lookup is a kallsyms-tree lookup, so any program that is JITed but not in that
tree would still oops.

**A guard that had already been rewritten twice.** The interesting background
for a reader deciding what the guard costs today is that the sequence the series
quotes is no longer the one in mainline. Two commits merged in the week after
the series was posted replace the address computation with a single `lea` and
the range compare with a shift and a branch, taking the guard from 39 bytes to
about 18. Nobody has re-run the veristat measurement against the
shorter form, so the widely quoted 35% mean reduction is an upper bound. The
same week's JIT work also touched the exception-table metadata for arena
atomic loads, which is the neighbouring case: those loads carry the same
`EX_TYPE_BPF` entry but need the *source* register reported as well, because a
faulting arena access has to name the address it was computing.