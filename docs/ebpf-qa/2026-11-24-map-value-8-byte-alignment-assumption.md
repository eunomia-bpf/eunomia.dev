# Why does the BPF verifier accept a 64-bit atomic or load-acquire on a map value it never proved is 8-byte aligned?

**Short answer:** because the alignment rule is about the offset inside the value,
not about the address the value starts at. `check_ptr_alignment()` routes a
`PTR_TO_MAP_VALUE` register to `check_generic_ptr_alignment()`, which tests the
register's known offset plus the instruction offset against the access size. The
map value pointer itself contributes nothing to that sum: the verifier has no
fact about its low bits, so it assumes them to be zero. For preallocated hash
maps the assumption holds. Three producers break it — `lpm_trie` on 64-bit, the
`bpf_mem_alloc` header on 32-bit, and `devmap` on both — and on arm64 the
resulting 64-bit `LDAR`/`STLR` or atomic raises an alignment fault that nothing
fixes up, so a program that loaded fine dies with an oops inside the BPF program.
A four-patch series posted on 24 November 2026 pads those three layouts instead
of changing the rule, because the rule is what libbpf, BTF and every existing
program already rely on.

## The mechanism

### 1. The verifier checks the offset and never the base

`check_ptr_alignment()` switches on the register type. `PTR_TO_MAP_VALUE` has
no special case beyond `INSN_ARRAY`, so it falls through to
`check_generic_ptr_alignment()`:

```c
reg_off = tnum_add(reg->var_off, tnum_const(off));
if (!tnum_is_aligned(reg_off, size))
        return -EACCES;
```

`reg->var_off` is what the verifier knows about the offset from the start of
the value. It cannot be non-zero for a value returned by a helper; the low bits
of the base address are simply not part of the register state, so no check can
see them. Two consequences follow directly from the source:

- The offset test only runs when `strict` is true. `strict` is
  `env->strict_alignment || strict_alignment_once`, and `env->strict_alignment`
  is the `BPF_F_STRICT_ALIGNMENT` program flag, forced on when the kernel is
  built without `CONFIG_HAVE_EFFICIENT_UNALIGNED_ACCESS` and forced off by
  `BPF_F_ANY_ALIGNMENT`. arm64 and x86-64 both select that config, so plain
  8-byte loads and stores on a map value are not checked at all there.
- Atomics are the exception that still gets checked: `check_atomic_rmw()` and
  `check_atomic_load()` pass `strict_alignment_once=true`, so an 8-byte atomic
  or load-acquire at an odd offset is rejected. The check is still only about
  the offset. At offset 0 it always passes.

The one width rule the verifier does enforce is architectural: `check_atomic()`
rejects a 64-bit load-acquire or store-release with `-EOPNOTSUPP` when
`BITS_PER_LONG != 64`. 64-bit `BPF_ATOMIC` arithmetic is allowed on 32-bit.

### 2. Three producers that hand out a value at 4 mod 8

The offsets below are struct layout, not measured behaviour; each was
recomputed from the current definitions.

- **LPM trie, 64-bit.** `struct lpm_trie_node` is two child pointers, a
  `u32 prefixlen`, a `u32 flags`, then `u8 data[]` — so `data` is at offset 24.
  The value is `found->data + trie->data_size`, and `data_size` is `key_size - 4`.
  An IPv4 trie has key size 8, so `data_size` is 4 and the value lands at
  offset 28 of a node, i.e. 4 mod 8. IPv6 (key 20, data 16) lands at 40 and is
  aligned. Nothing rounds `data_size` up.
- **`bpf_mem_alloc` objects, 32-bit.** `bpf_mem_alloc()` returns
  `unit_alloc(...) + LLIST_NODE_SZ`, where `LLIST_NODE_SZ` is
  `sizeof(struct llist_node)`. That header is 8 bytes on 64-bit and 4 bytes on
  32-bit, so on a 32-bit kernel every object the allocator hands out is 4 mod 8
  from a `kmalloc` address. Non-preallocated hash and rhash entries get their
  values from this allocator, which is why the reviewer's table shows the hash
  value misaligned only on 32-bit. The same applies to `bpf_obj_new()` objects.
  On 32-bit an IPv4 LPM trie value happens to come out aligned (base 4, `data`
  at 20, value at 24) — the one case that works today by accident.
- **devmap, both widths.** `dev_map_lookup_elem()` and
  `dev_map_hash_lookup_elem()` return `&obj->val`, and `val` follows
  `unsigned int idx` in `struct bpf_dtab_netdev`: offset 52 on 64-bit, 28 on
  32-bit. Both are 4 mod 8. The map is read-only to BPF programs, so no atomic
  can target it, but a 64-bit load-acquire at offset 0 is accepted on 64-bit.

### 3. arm64 faults, and the fault is not fixed up

The arm64 JIT turns these instructions into exactly the forms that require
natural alignment: `LDAR` and `STLR` for load-acquire and store-release, and
either an LSE atomic or an `LDXR`/`STXR` loop for `BPF_ATOMIC`. `FEAT_LSE2`
relaxes LSE atomics that stay inside a 16-byte block, and relaxes `LDAR`/`STLR`
only when `SCTLR_EL1.nAA` is set — Linux does not set it, so on a kernel that
boots with `INIT_SCTLR_EL1_MMU_ON` the two load forms keep their alignment
requirement even on hardware that has LSE2.

The fault path then declines to help. `do_alignment_fault()` calls
`do_compat_alignment_fixup()` only when the kernel has
`CONFIG_COMPAT_ALIGNMENT_FIXUPS` and the fault came from a 32-bit compat user
task; everything else goes to `do_bad_area()`, which in kernel mode ends in
`die_kernel_fault()` with "alignment fault". A BPF program runs in kernel mode,
so the outcome is an oops, not a signal. The series carries one:

```
ESR = 0x0000000096000021
FSC = 0x21: alignment fault
Internal error: Oops: 0x96000021 [#1] SMP
pc : bpf_prog_cc9155b98965fc5e+0x64/0x90
```

That trace is a 64-bit atomic add at offset 8 of an IPv4 LPM trie value: the
address is 4 mod 8 and the access crosses a 16-byte block. It was reproduced
on arm64 under QEMU with hardware LSE2, which is the point — hardware support
does not save the `LDAR`/`STLR` case.

On 32-bit arm the failure has a different source and the same ending: the JIT
does not implement `BPF_ATOMIC` at all, so the interpreter executes it with
`atomic64_*()`, and ARMv6K and later implement those with `LDREXD`/`STREXD`,
which require an 8-byte aligned address.

### 4. x86-64 hides it, except when the access is split

x86-64 has no alignment trap for ordinary accesses, and a locked 8-byte access
at 4 mod 8 is only special when it crosses a cache line: that is a split lock,
which raises `#AC` on CPUs with split lock detection. `exc_alignment_check()`
does not consult the mitigation state in kernel mode — if the fault did not come
from user mode it goes straight to `die("Split lock detected")`. The
`split_lock_detect=` state (`warn` by default once the CPU is known to support
detection) only shapes what happens to a user-space fault: `warn` logs and
temporarily disables detection on that CPU, `fatal` sends `SIGBUS`.

So on a machine with detection enabled, one of these loads that happens to
straddle a cache line is fatal; the ones that do not straddle are simply slower.
A test that runs on x86-64 can pass for years and say nothing about arm64.

### 5. What the fix changes, and why two patches must travel together

The series adjusts the three layouts rather than the verifier rule.

- The allocator header becomes `ALIGN(sizeof(struct llist_node), 8)` in
  `memalloc.c`, with the same constant in the hash map's memory-usage
  accounting, and the percpu pointer is stored at `obj + LLIST_NODE_SZ` instead
  of at `obj[1]`. This is Felix Fietkau's patch, rebased.
- `lpm_trie` rounds `data_size` up to a multiple of 8 in the lookup, the node
  fill, the leaf size and the memory-usage count, and adds
  `BUILD_BUG_ON(offsetof(struct lpm_trie_node, data) % 8)` because the value's
  alignment now depends on that offset.
- `devmap` marks `val` as `__aligned(8)`. On 64-bit it moves to offset 56 and
  the struct stays 64 bytes; on 32-bit the struct grows from 36 to 40 bytes,
  where the 64-bit load-acquire is already rejected by the verifier.

The ordering constraint is real and the cover letter states it. The allocator
patch alone moves a 32-bit IPv4 LPM value from 0 to 4 mod 8, breaking the one
case that works today. The trie patch alone, with nodes still at 4 mod 8 on
32-bit, rounds `data_size` to 8 and puts the value at offset 28 of a node whose
base is 4 mod 8. Either one alone is a regression on 32-bit; together they
leave 0 mod 8.

One correction from review is worth keeping: adding `__aligned(8)` to the
`u8 data[]` member changes nothing, because `data` is already at 24 or 16. What
is misaligned is the node, and the trie patch has to fix the offset arithmetic,
not the declaration.

## Checking it on your own kernel

1. Work out the offset from the map implementation, not from the program. For an
   LPM trie that is `offsetof(data) + data_size`; for anything allocated from
   `bpf_mem_alloc` on a 32-bit kernel, add 4 for the header.
2. Measure it instead of reasoning about it if you can. A program that stores the
   pointer returned by `bpf_map_lookup_elem()` into a global variable, and a user
   space check of `addr & 7`, is exactly what the series' selftest does for an
   IPv4 trie, a `BPF_F_NO_PREALLOC` hash and a devmap. Because the program
   executes no alignment-sensitive instruction, an affected kernel fails the
   assertion instead of oopsing. Without the patches, the trie and devmap checks
   fail on 64-bit, and the hash and devmap checks fail on 32-bit.
3. Read the emitted code. `bpftool prog dump jited` shows whether an 8-byte
   load-acquire became `ldar` and whether an atomic became `ldadd` or an
   `ldxr`/`stxr` pair on arm64, and a `lock`-prefixed instruction on x86-64.
4. Run it where the architecture is strict. QEMU on an LSE2 host is enough to
   reproduce the oops; x86-64 will not tell you.
5. On x86-64, confirm whether detection is on before concluding anything: check
   `split_lock_detect` in `/proc/cmdline`, and look for "#AC" or
   "Split lock detected" in the log after the program runs.

## The limits

- The series is v2 and was unmerged on 24 November 2026. Every kernel you can
  boot today still has the layouts above, and the trie case has an oops report
  against 7.3.5.
- It fixes three producers, not the rule. `check_ptr_alignment()` still assumes
  every map value starts 8-byte aligned, so a new map type that returns a value
  pointer at 4 mod 8 reopens the bug; the selftest covers three map types, not
  the invariant.
- The x86-64 failure mode is argued from the split-lock path, not measured in
  the series, and it depends on the CPU supporting split lock detection. A CPU
  without it takes the slow bus lock and survives.
- The 32-bit audit is arm32: no `BPF_ATOMIC` in the JIT, `LDREXD`/`STREXD` in
  the interpreter. Other 32-bit JITs and interpreters were not examined in the
  thread.
- The trie fix changes memory accounting: an IPv4 node grows by 4 bytes and the
  map's `memlock` charge grows with it. The key format users space sees does
  not change.

## References

- [BPF mailing list: the v2 series cover letter](https://lore.kernel.org/bpf/20261124225055.87344-1-tphilipp@gmail.com/) — the three broken layouts, the two-patch ordering constraint, and what the series does not change.
- [BPF mailing list: allocator header on 32-bit](https://lore.kernel.org/bpf/20261124225055.87344-2-tphilipp@gmail.com/) — `LLIST_NODE_SZ`, the percpu pointer move, and the hash-map accounting change.
- [BPF mailing list: lpm_trie value offset](https://lore.kernel.org/bpf/20261124225055.87344-3-tphilipp@gmail.com/) — the node layout, the `round_up` patch, and the arm64 oops trace.
- [BPF mailing list: devmap value offset](https://lore.kernel.org/bpf/20261124225055.87344-4-tphilipp@gmail.com/) — `val` at 52 and the `__aligned(8)` change.
- [BPF mailing list: the alignment selftest](https://lore.kernel.org/bpf/20261124225055.87344-5-tphilipp@gmail.com/) — the XDP program, the three lookups, and the `addr & 7` assertions.
- [BPF mailing list: the original 32-bit allocator patch](https://lore.kernel.org/all/20260717183632.99195-1-nbd@nbd.name/) — the report that opened the 32-bit half.
- [Verifier alignment check](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/verifier.c) — `check_ptr_alignment()`, `check_generic_ptr_alignment()`, the `strict` flags, and `check_atomic()`.
- [LPM trie](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/lpm_trie.c) — `struct lpm_trie_node` and the unrounded `data_size` offsets.
- [Hash map layouts](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/hashtab.c) — `htab_elem_value()`, `rhtab_elem_value()`, and the allocator use.
- [devmap](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/devmap.c) — `struct bpf_dtab_netdev` and the two lookup functions returning `&obj->val`.
- [BPF memory allocator](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/memalloc.c) — `LLIST_NODE_SZ` and the `ret + LLIST_NODE_SZ` return.
- [arm64 BPF JIT](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/arch/arm64/net/bpf_jit_comp.c) — `emit_atomic_ld_st()` and the exclusive-sequence path.
- [arm64 fault handling](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/arch/arm64/mm/fault.c) — `do_alignment_fault()` and `do_bad_area()`.
- [arm64 boot control register](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/arch/arm64/include/asm/sysreg.h) — `INIT_SCTLR_EL1_MMU_ON`, which does not set `nAA`.
- [x86 trap handler](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/arch/x86/kernel/traps.c) — `exc_alignment_check()` dying in kernel mode.
- [x86 split lock state machine](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/arch/x86/kernel/cpu/bus_lock.c) — `sld_state`, the boot default, and the user-mode-only handling.
- [Split lock documentation](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/Documentation/arch/x86/buslock.rst) — the warn/fatal table, including "Kernel OOPs" for both.
- [BPF interpreter](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/core.c) — the `STX_ATOMIC` family and where 32-bit uses `atomic64_*`.
- [ARM atomics](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/arch/arm/include/asm/atomic.h) — `atomic64_*` as `ldrexd`/`strexd` on ARMv6K and later.
- [arm32 BPF JIT](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/arch/arm/net/bpf_jit_32.c) — `BPF_ATOMIC` falling through to the interpreter.

## Community discussion today

Coverage this run: two archive-opted Slack workspaces (three allowlisted
channels) were read through the approved read-only archive and held no message in
either the 24-hour or the 7-day window, so the Slack side is quiet rather than
unread. The two allowlisted Discord workspaces (ten channels) could not be read:
the web client was at its login screen, so their activity is a coverage gap, not
silence. r/eBPF was not readable either — the plain public page returned a
network-security interstitial on every attempt — which is a second coverage gap
this run. The question above comes from the public bpf mailing-list archive,
which is also where the rest of the day's themes come from.

**The alignment assumption, and a review that found the regression.** The
8-byte claim is a promise the verifier makes to the JIT on behalf of every map
type, and the series above is the first time someone checked it against the
implementations rather than the documentation. The technical value of the thread
is the reviewer's table: the same patch that fixes the 32-bit allocator header
moves a 32-bit IPv4 trie value from aligned to misaligned, because the node's
`data` offset and the allocator header were compensating for each other. Any
reader changing a structure that a map type returns as a value should build the
same table for their layout — base alignment, member offset, value offset — and
check it at both widths. The open boundary is the rule itself: the verifier
still has no way to express "this map type's values are 8-byte aligned", so the
next map type has to rediscover the constraint.

**A kfunc prototype change and a resolver that reads BTF.** A follow-up series
lets the kernel keep an old kfunc prototype next to the new one, resolving a BPF
program's declaration by matching the BTF prototype rather than by name. The
author replayed kfunc resolution over 1302 objects against three kernel BTF
images and reported which weak declarations change target and which stay
unresolved; in-tree `compat.bpf.h` now prefers the `___v2` names first. The
mechanism a reader should take away: a `__weak` declaration does not fail
loudly, so the loader is what decides, and a name-only fallback silently binds
to whichever variant the kernel kept. The open question is what happens to a
declaration whose *pointed-to struct* was renamed — the resolver matches on
prototype, so a renamed struct reads as a different function, not an
incompatible one.

**Sleepable struct_ops, and one check copied four times.** A series rejects
sleepable programs in `Qdisc_ops`, `tcp_congestion_ops` and `smc_hs_ctrl` by
adding a `check_member` to each. Review pushed back on both halves: `Qdisc_ops`
`init` and `destroy` run under `rtnl` and can sleep, so a sleepable program
there is valid today, and four copies of the same function means the default is
the bug — the fix belongs once in `check_struct_ops_btf_id()`, where the
struct_ops that already have a `check_member` keep their sleepable programs. The
boundary left open is the split between "this member may sleep" and "this
program may sleep": a struct_ops with `check_member` still has to test
`prog->sleepable` itself.

**Local storage allocation that can now sleep.** Task and inode storage used
only `kmalloc_nolock()`, which fails where a blocking allocation would succeed —
the report describes `ZONE_NORMAL` contention combined with hugepages draining
`ZONE_DMA32` below its watermark, observed in production. The series picks the
allocation at call time: `GFP_KERNEL` when the program is sleepable,
`GFP_NOWAIT` when the caller may spin, `kmalloc_nolock()` otherwise, using the
program's aux metadata and the current preemption state. The property worth
remembering is that "which allocator" became a runtime question with three
answers, so a storage allocation that used to be infallible in one context is
now a different code path in each.

**Cloning a program after the object is loaded.** A reported libbpf bug: an
fentry program targeting a module can be cloned after `bpf_object__prepare()`
but not after `bpf_object__load()`, because load caches the module BTF fd and
type id in the program and the cleanup at the end of loading closes that fd.
Reusing the freed fd number made the clone load against a different module's
type id — a wrong-but-successful load, which is worse than the `-EINVAL`. The
maintainer's answer was that clone was built for loading one program out of an
unloaded object and that load strategies cover the other case; the open
boundary is whether cached attach state should be invalidated at cleanup or kept
alive until object close.