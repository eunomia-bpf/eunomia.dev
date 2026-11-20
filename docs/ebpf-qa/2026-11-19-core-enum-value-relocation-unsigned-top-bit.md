# Why does a CO-RE enum value relocation refuse an object over an unsigned enum member with its top bit set, and what does a fix have to accept?

**Short answer:** BTF keeps every 32-bit enumerator in a signed field and records the enum's
signedness separately, in the type's kind flag. The CO-RE code that resolves an enum value reads
that signed field straight into a 64-bit variable, so a member of an *unsigned* enum at or above
`0x80000000` comes out sign-extended: `0xefffffff` becomes `0xffffffffefffffff`. A compiler emits
the zero-extended form for an unsigned enumerator, the relocation therefore cannot match the
instruction's immediate against the value it computed from the local type, and the whole object is
refused before the verifier ever runs. The target value is also patched in the wrong form, which
matters wherever the program uses it as a 64-bit number. Because clang and GCC disagree about
which form to emit, a fix has to accept both, not just switch the extension.

## The mechanism

### 1. BTF separates the value from its signedness

`BTF_KIND_ENUM` stores each enumerator as a name and a 32-bit signed value, and the type's own
`info` word carries the enum's signedness in bit 31 — the kind flag, documented as "0 for
unsigned, 1 for signed". `BTF_KIND_ENUM64` is the separate kind for values wider than 32 bits.

```c
struct btf_enum {
        __u32   name_off;
        __s32   val;
};
```

So the field is signed no matter what the enum is, and `val` is a bit pattern: the member
`SKB_DROP_REASON_SUBSYS_MASK = 0xffff0000` is stored as `-65536`. A tool that wants the C value
has to decide the extension from the kind flag. libbpf learned this for two other paths before
this one — the data dumper and enum-typed map fields both go through a helper that extends
according to the flag:

```c
static inline __u64 btf_enum_value(const struct btf_type *t, const struct btf_enum *e)
{
        return btf_kflag(t) ? (__s64)e->val : (__u32)e->val;
}
```

### 2. The relocation path skipped that decision

CO-RE enum value relocations are computed in one small function. For the value kind it does this:

```c
t = btf_type_by_id(spec->btf, spec->spec[0].type_id);
if (btf_is_enum(t))
        *val = btf_enum(t)[spec->spec[0].idx].val;
else
        *val = btf_enum64_value(btf_enum64(t) + spec->spec[0].idx);
```

`*val` is `__u64`, the source is `__s32`, and nothing consults the kind flag, so the assignment
sign-extends every 32-bit enumerator. The `enum64` branch takes the full 64-bit value and is
unaffected. This is the whole defect: two helpers in the same library extend the same field two
different ways.

### 3. Where the wrong value becomes a refused object

The computed value is the *expected* value of the local side, and it is validated against the
instruction the compiler produced. For the `ld_imm64` form the check is literal:

```c
imm = (__u32)insn[0].imm | ((__u64)insn[1].imm << 32);
if (res->validate && imm != orig_val) {
        pr_warn("prog '%s': relo #%d: unexpected insn #%d (LDIMM64) value: got %llu, exp %llu -> %llu\n", ...);
        return -EINVAL;
}
```

`got` is the instruction, `exp` is the value libbpf computed, and the third number is what the
target would have been patched to. When the two disagree the relocation fails with an invalid
argument, libbpf refuses the program, and no program is loaded. Nothing in this path consults the
verifier, so the error looks like a build-time or loader-time incompatibility rather than a
kernel-version issue, which is exactly why it is easy to misattribute.

### 4. The kernel runs the same code

`kernel/bpf/Makefile` builds `relo_core.o` from `tools/lib/bpf/relo_core.c`, the identical file,
and the in-kernel relocation entry point walks a program's relocation records one at a time:

```c
err = bpf_core_apply(&ctx, &core_relo, i, &prog->insnsi[core_relo.insn_off / 8]);
```

That path is used when the relocations travel with the load request instead of being applied in
userspace, as with the generated loader a light skeleton carries: the generator copies each
`bpf_core_relo` record into the load attributes. So a fix has to land in the shared file, and the
helper it calls has to exist on both sides — the userspace copy of the header and the kernel's.
The helper above was added for the dumper and map fields in one commit and exists in the BPF tree,
but the kernel header does not have it yet, which is why the relocation series adds it there as
well.

### 5. The two compilers emit different forms

Clang emits the value the way the enumerator's own C type says. Three cases, each compiled here
against a local object and read back from the instruction stream:

- `0x80000000u` in an unsigned enum: the instruction carries `2147483648`, the zero-extended
  `0x0000000080000000`.
- `0xffffffffu` in an unsigned enum: the instruction carries `4294967295`, the zero-extended
  `0x00000000ffffffff`.
- `-1` in an enum that BTF marks unsigned: the instruction carries `-1`, the sign-extended
  `0xffffffffffffffff`.

The first two are zero-extended and cannot equal what the relocation computes today, which is
`0xffffffff80000000` and `0xffffffffffffffff`. The third shows the other half of the problem: it
is an unsigned enum by kind flag, and clang still emitted it sign-extended, because the
enumerator's own type is `int`. So "extend by the kind flag" fixes the first two cases and breaks
the third, in a clang-built object, not only in a GCC-built one.

GCC does not have the choice: the series reports that its builtin returns `int`, so every 32-bit
value is sign-extended in the instruction. An object built by GCC is therefore accepted today, and
would be refused by a fix that only switches the extension. The version of the fix under review
handles both: it extends by the kind flag, and then, when the original value came out
sign-extended and the target value fits in 32 bits, it takes the sign-extended form as a second
accepted original and patches the target in that same form. An original below `0x80000000` is
identical in both forms, so for those the target is extended by the target enum's own flag.

### 6. How much of the kernel is exposed

Counting this host's 7.3 vmlinux BTF: of 2909 32-bit enum types and 23 enum64 types, 29 unsigned
32-bit enums carry a member at or above `0x80000000`. They include `init_ns_ino`
(`IPC_NS_INIT_INO` is `0xefffffff`), the x86 memory-map type (`E820_TYPE_SOFT_RESERVED`, also
`0xefffffff`), the socket drop reasons (`SKB_DROP_REASON_SUBSYS_MASK` is `0xffff0000`), the
network device private flags, the page-fault error code bits, and the ring buffer's busy bit. Any
value relocation on one of those members, from an object compiled against an unsigned local
declaration of the same enum, is the failure. A field-offset or type relocation on the same enum
is not.

## Verification and debugging path

1. Reproduce with one enum and one call:

   ```c
   enum uns_max { M1 = 0xffffffffu };
   __u64 v = bpf_core_enum_value(enum uns_max, M1);
   ```

   Compile with debug info and read the instruction: `llvm-objdump -d` shows
   `r0 = 4294967295 ll`, the zero-extended `0x00000000ffffffff`.

2. Compute what the relocation expects. Take the same member out of the object's own BTF and
   extend it as a signed 32-bit value: `0xffffffffffffffff`. The two differ only in the high
   half.

3. Read the loader's message in that shape. `got` small and zero-extended, `exp` with the high
   half all ones, for an enum whose BTF says `encoding=UNSIGNED`, is this defect rather than a
   target mismatch. Dumping the object's BTF with `bpftool` shows the enum's encoding and the
   value it prints for the member.

4. Separate it from the missing-member case. When the target has no enumerator of that name, the
   relocation is not validated against a value at all: the instruction is poisoned into a call to
   an invalid function number, and the failure surfaces as `invalid func unknown#195896080` if the
   poisoned call is reachable, or not at all if it is dead code. An existence check
   (`bpf_core_enum_value_exists`) never reads the value, so it is unaffected either way.

5. Work around it while the fix is not in your toolchain: mask or compare in 32 bits —
   `(u32)bpf_core_enum_value(...)` or `& 0xffffffffu` — so the sign-extended patch is harmless,
   and keep high-valued unsigned members out of value relocations entirely when the object must
   load on an unpatched libbpf.

## The limitation

The fix is about 32-bit enums only; the enum64 path already reads a full 64-bit value. It cannot
recover a *semantic* mistake: an object that declares the enum signed while the target declares it
unsigned will still produce different values, and the kind flag of the target is what decides how
the patched value is extended. The accepted-forms logic also never distinguishes clang from GCC
for values below `0x80000000`, because the emitted instruction is identical there — so any
reasoning about which compiler built an object from the immediate alone is unsound. And this host
has no BPF capability, so the reproduction here stopped at the instruction bytes, the local BTF
encoding, and the arithmetic of the relocation; a full load was not run.

## References

- [BTF kind encoding for enums, kernel documentation](https://docs.kernel.org/bpf/btf.html) — the kind flag as signedness, the signed 32-bit enumerator field, and enum64 as a separate kind.
- [BTF definitions, uapi header](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/include/uapi/linux/btf.h) — the enumerator structure and the kind-flag bit.
- [CO-RE relocation core, libbpf](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/tools/lib/bpf/relo_core.c) — the enum value computation, the immediate comparison, and the poisoning path.
- [Kernel build of the same relocation file](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/Makefile) — the object built from the libbpf source.
- [In-kernel relocation entry point](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/check_btf.c) — per-record application of a program's relocations.
- [CO-RE helper macros, libbpf](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/tools/lib/bpf/bpf_core_read.h) — the value and existence forms of the enum builtin.
- [Relocation kinds, kernel documentation](https://docs.kernel.org/bpf/llvm_reloc.html) — the enum value kinds and their spec strings.
- [Generated loader relocation records, libbpf](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/tools/lib/bpf/gen_loader.c) — how records travel to the kernel with the load request.
- [Enum extension helper, BPF tree](https://git.kernel.org/pub/scm/linux/kernel/git/bpf/bpf-next.git/plain/tools/lib/bpf/btf.h) — the flag-driven extension used by the dumper and map fields.
- [The same normalization for map fields, libbpf](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/tools/lib/bpf/btf_dump.c) — the dumper comparison that already uses the flag.
- [Commit adding the extension helper](https://github.com/torvalds/linux/commit/89e7a42e3030) — the dumper and map-field change, and why the field was sign-extended before.
- [Series, version 3, cover letter](https://lore.kernel.org/bpf/20261119232647.3392045-1-yonghong.song@linux.dev/) — the two accepted original forms and the target-patch rule.
- [Series, version 3, the libbpf change](https://lore.kernel.org/bpf/20261119232652.3392590-1-yonghong.song@linux.dev/) — the exact error output, the clang and GCC instruction forms, the count of affected enums, and the kernel header addition.
- [Earlier revision of the series](https://lore.kernel.org/bpf/20261119051443.1902163-1-yonghong.song@linux.dev/) — the four-patch shape before the tests were folded in.
- [Namespace inode numbers, kernel source](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/include/linux/dcache.h) — one of the affected enums, with a member at `BIT(31)`.
- [x86 memory-map types, kernel source](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/arch/x86/include/asm/e820/types.h) — the reserved soft-range value at `0xefffffff`.
- [GCC's BPF builtins](https://gcc.gnu.org/onlinedocs/gcc/BPF-Built-in-Functions.html) — the documented enum value builtin the series describes as returning `int`.

## Community discussion today

Two archive-opted Slack workspaces were read through the approved read-only archive, three
allowlisted channels in total: nothing in the 24-hour window, and three messages in the seven-day
window — a meeting-notes notice, a contributor introduction, and a binary-pattern-search question
already answered on this site. The question above therefore comes from the public bpf mailing-list
archive, read as ordinary public pages, which held 61 thread roots at that moment, 78 of its index
rows dated 2026-11-19 in UTC. The two allowlisted Discord workspaces, ten channels between them,
were not covered: the client answered with an unauthenticated login page, so they are recorded as
inaccessible rather than quiet. The public practitioner forum was read and had no new technical
question inside the window; its newest posts were two project announcements and a conference call
for papers from earlier days.

### The same bit, three different owners

The question above is one instance of a pattern that ran through the whole window: a 32-bit value
whose top bit is set, handled by code that assumes something about its width. The enum case puts
the mistake in the relocation, between the signed storage field and the compiler's unsigned
immediate. Two other threads put it in the map and the verifier.

A long-prefix-match map returns its value immediately after the key bytes, with no padding, so an
IPv4 map value starts four bytes into the node and is only four-byte aligned. The verifier checks
the alignment of a map value access against the offset inside the value and assumes the value
itself starts eight-byte aligned — which hash maps guarantee by rounding the key size up to eight
bytes, and this map type does not. A 64-bit atomic add at offset zero inside the value therefore
passes verification and, on an arm64 machine without or across the 16-byte granule of the LSE2
feature, faults in the JITed code and panics the kernel. The proposed repair is the boring one:
align the flexible array, round the value offset up to eight bytes in allocation, lookup and
memory accounting, and match what the hash map already does. The diagnostic generalizes: whenever
a map stores the value after a variable-length key, check whether the value offset is rounded, and
do not trust the verifier's alignment proof to cover the base address.

The third instance is a program, not a value. A helper that rewrites the interrupted instruction
pointer to force a function to fail was already rejected for one probe flavour and still accepted
for the multi-attach user probe. Attach one there and the task returns to user mode with a kernel
address as its instruction pointer: the fault appears in the task's signal frame, and the register
state that produced it belongs to the kernel. The fix rejects such a program at link creation for
both entry and return probes, matching what the single-probe path already does.

### Sample data reused between events, and an address nobody wrote

A three-patch series fixed a sample-data structure built on the stack for software events. The
first patch gives each matching event its own initialization: a sampling event was inheriting the
sample identifier, timestamp, pid and call chain prepared for the event before it, and sometimes a
record size smaller than what was then written, overrunning later records in the ring buffer. The
reported symptom is concrete — two concurrent sampling sessions at a count of one, where the older
one fails, and an unprivileged user's page-fault event can do this to a root-owned CPU-wide
session, broken since 6.3. The second patch stops a BPF program from reading an unset address: the
`addr` field of the perf event data context returned leftover stack, mostly kernel pointers, on
almost all context-switch and clock samples, since 6.1, visible only with the stack-initialization
setting that does not zero. The third removes the 256-byte zeroing of that structure entirely and
marks it uninitialized, on the argument that every field is written before it is read; measured on
one CPU, software counters cost 54 ns on a 627 ns context switch and the change brings that to
44 ns. The residual boundary the author states is real: a privileged program reading the context
through the type-preserving cast can still see unprepared fields.

The same structure produced a second, unrelated thread. Replacing per-CPU register snapshots with
an on-stack copy in the raw tracepoint output helper drew an automated review finding that nested
programs can then hit the warning in the output helper — a panic on kernels configured to panic on
warn — and that the "no more busy" claim in the message is wrong. The maintainers agreed on both
points and asked for the warning to be removed as part of the series, and one of them pointed out
a further hole in the same area: a stack collection with build identifiers can re-enter the same
lock through a program on its completion tracepoint, and the collected identifiers vanish. So the
unresolved part is not the change itself but the recursion contract around it.

### Two boundaries the networking patches state plainly

A series adding socket assignment on the transmit side is explicit about what it does not do:
software segmentation after the assignment is unsupported, decrypted transmit packets are
unsupported, and loopback discards the assignment. It takes a reference on the assigned socket and
installs a destructor that releases it, which is how a lifetime problem becomes a recognisable
packet property. If you plan to use it for transparent proxying, those three exclusions decide
whether the design works before any performance question does.

A receive-side series moves the rule that attached XDP excludes hardware receive coalescing out of
individual drivers and into the network core, because drivers disagreed: some refused the attach,
some cleared the feature in their fixup, at least one kept coalescing while XDP was attached. The
motivation is that the coalescing features should stay in the wanted set and come back after
detach; the practical diagnostic, for anyone debugging a receive workload with XDP attached, is to
read both the wanted and the active feature bits before and after attach and detach rather than
trusting the interface's reported state.

### Smaller items

A symbol-lookup optimization series reached its fourth revision, replacing the linear scan of the
kernel symbol table with a search ordered by name, and added a kernel parameter to change the
density of its index markers. On the tooling side, a fix keeps a memory-cgroup test's page cache
off an overlay filesystem, and a library fix makes a signed 32-bit enum comparison in the dumper
work for unsigned data, which is the same helper the question above depends on.