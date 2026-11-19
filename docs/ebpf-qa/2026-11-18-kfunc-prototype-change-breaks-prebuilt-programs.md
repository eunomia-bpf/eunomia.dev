# When a kfunc prototype changes incompatibly, why do already-built BPF programs stop loading and what does it take to keep them working?

**Short answer:** libbpf resolves a kfunc call at load time by name and then by
prototype. The name it asks the kernel for is the BPF declaration's name with
the last `___<flavor>` suffix removed, and the kernel-side lookup for that name
is exact: there is no fallback to a kernel kfunc that carries a suffix. So when
a namespace gives its plain name to a new prototype and parks the old one under
a suffix, an object built against the old prototype still resolves to the plain
name, fails the compatibility check, and either dies at load or turns into a
poisoned call the verifier rejects if it is reachable.

There are only two ways to keep old objects loading today: keep the *old*
prototype on the plain kernel name and put the suffix on the new one, or have
every BPF source declare the kernel's suffixed name and guard it with a weak
symbol. A libbpf fallback that searches `<name>___<flavor>` on the kernel side
is posted for review, not merged, and would only help libbpf 1.9 and later.

## The mechanism

### 1. The BPF side is lenient about suffixes, the kernel side is not

Name matching for kfuncs uses the *essential name*, the part before the last
`___` separator. `bpf_core_essential_name_len()` scans from the end of the
string for an `X___Y` pattern and returns the prefix length. libbpf records
that prefix for every extern symbol it finds:

```c
ext_essent_len = bpf_core_essential_name_len(ext->name);
ext->essent_name = NULL;
if (ext_essent_len != strlen(ext->name))
        ext->essent_name = strndup(ext->name, ext_essent_len);
```

Resolution then looks up the essential name:

```c
kfunc_id = find_ksym_btf_id(obj, ext->essent_name ?: ext->name, BTF_KIND_FUNC, ...);
```

and `find_ksym_btf_id()` is a plain name-to-BTF-ID search: `btf__find_by_name_kind()`
on vmlinux, then on each module's own BTF. Nothing enumerates the family of
similarly named functions. That asymmetry is the whole problem. A BPF program
may spell `bpf_foo___local`, and libbpf will happily look up the kernel's
`bpf_foo`; but a kernel that renames its old function to `bpf_foo___v1` leaves
no way for libbpf to find it from a declaration of `bpf_foo`.

### 2. What "compatible" actually means

The resolved function's prototype is compared with the declaration by
`bpf_core_types_are_compat()`, the same CO-RE type matcher libbpf already uses
for kfuncs. Its rules are generous in one direction and strict in another:

- a function prototype is compatible if the parameter count matches and each
  parameter and the return value are compatible;
- integers are interchangeable, size and signedness ignored;
- struct, union, enum and forward-declaration names are ignored entirely;
- typedefs, `const`, `volatile` and `restrict` are looked through;
- pointers must point at compatible kinds, so `struct file *` and
  `struct task_struct *` differ while `struct file *` and `struct file___x *`
  do not.

The consequence is that a large class of changes is invisible to the check and
is therefore made *in place*: widening a parameter from `u32` to `u64`, adding
`const`, renaming a type, renaming a parameter. Old objects silently bind the
new definition, with no version signal at all. Only the changes that alter the
shape — adding or removing a parameter, changing the return kind, changing a
callback prototype, or changing what a pointer points at — fail the check.

### 3. What failure looks like, strong and weak

For a strong `__ksym` extern, an incompatible prototype is a hard error:

```
extern (func ksym) 'scx_bpf_dsq_insert': func_proto [..] incompatible with vmlinux [..]
```

and `bpf_object__load()` returns `-EINVAL`.

For a weak extern the failure is deferred and much quieter. libbpf leaves the
extern unset, and during relocation it rewrites the call into a deliberately
invalid helper call:

```c
insn->code = BPF_JMP | BPF_CALL;
insn->imm = POISON_CALL_KFUNC_BASE + ext_idx;
```

If verification reaches that instruction, the verifier reports an invalid call
to an unknown function, and libbpf's verifier-log fixup rewrites the line to
name the missing kfunc:

```
invalid func unknown#2002000000
kfunc 'scx_bpf_dsq_insert___v2' is referenced but wasn't resolved
```

If verification never reaches it, nothing is reported at all: the unreachable
path is replaced by a trap after verification, so the load succeeds with the
call never executed.

Not reaching it is exactly what the compat macros rely on. `bpf_ksym_exists()`
is `!!sym` on a symbol the compiler has checked is `__weak`, and the branch it
guards is decided statically: a resolved typed kfunc is a pointer, and the
verifier hard-wires `pointer != 0` for any pointer it has proven non-null,
while an unresolved weak kfunc is a plain constant zero. The branch that is not
taken becomes dead code, and its poisoned call is never verified.

### 4. The case that produced this question

A scheduler-facing kfunc family changed from returning `void` to returning
`bool`. The first attempt gave the new prototype the plain name and kept the
old one as `<name>___compat`, on the theory that libbpf would match the old
binary to the suffixed function. It does not, so every previously built
scheduler failed to load; the fix reversed the naming, keeping `void` on the
plain name and putting `___v2` on the `bool` version.

A second defect followed from the same asymmetry, and it is the instructive
one. vmlinux.h spells the new function as `<name>___v2`, so a compat header
declared it that way. libbpf stripped the suffix, looked up the plain kernel
name, found the *old* prototype, judged it incompatible, and left the weak
extern unresolved. `bpf_ksym_exists()` then returned false on every kernel
that had the new kfunc, the macro fell through to the older variant, and a
parameter was silently dropped — no warning, because a missing weak symbol is
not an error. The fix was to append a second suffix in the BPF source, so that
stripping one leaves the kernel's exact name. That is the "kernel naming leaks
into BPF programs" cost of the current model: the guard only works when the
stripped name coincides with the kernel's.

### 5. What the pending series changes, and its build-time guard

The posted series makes libbpf recover instead of failing: if the kfunc with
the stripped name is missing, or its prototype is incompatible, libbpf searches
the kfuncs named `<name>___<flavor>` — vmlinux first, then the first module
that has any — among those carrying the `bpf_kfunc` decl tag, and binds the one
whose prototype is compatible. The existing exact lookup runs first, so every
binding that works today is unchanged. More than one compatible flavor is an
error even for a weak declaration, on the principle that ambiguity is not
absence.

That search would be unsafe if a declaration could match two family members,
so `resolve_btfids` gains a kernel-build-time check: after deduplication it
compares every pair of kfuncs sharing an essential name with the same
libbpf function, and fails the build if they are ever compatible. The exception
is an alias — a suffixed kfunc with the verbatim prototype of the unsuffixed
one — because the unsuffixed lookup wins and the alias resolves first anyway.
This check is also why the newest compat macro in that namespace can keep
working: the aliased `___v2` and the plain function are compatible by
construction.

Documentation for the rules ships with the same series as a new lifecycle
section, including the explicit advice to give the new prototype a name that
does not start with `<name>___` when older libbpf or light skeletons must keep
working, and not to rename a `<name>___vN` kfunc in place because programs may
call it by that name.

## Verifying it

1. Enumerate the family in the running kernel and read the real prototypes
   rather than trusting vmlinux.h:

   ```
   bpftool btf dump file /sys/kernel/btf/vmlinux format raw | grep 'FUNC .*<name>'
   ```

   Every suffixed sibling appears with its own `FUNC_PROTO`, so the arity
   difference that breaks the stripped lookup is visible directly.

2. Ask libbpf what it bound. Run the loader with debug logging enabled
   (`LIBBPF_LOG_LEVEL=debug`). A successful resolution names the object that
   supplied the symbol and the BTF ID it picked:

   ```
   extern (func ksym) 'scx_bpf_dsq_insert': resolved to vmlinux [61234]
   ```

   An incompatible prototype instead prints the `func_proto ... incompatible`
   warning, naming the module or vmlinux plus both type IDs. A load that
   succeeds but behaves as if a parameter were missing should be checked here
   first: the debug line shows which member of the family was actually bound.

3. For a weak extern, force the guarded path to run. If the log says the
   kfunc is referenced but was not resolved, the declaration's stripped name
   did not match a kernel prototype and the program took the fallback branch:

   ```
   kfunc 'scx_bpf_dsq_move_to_local___v2' is referenced but wasn't resolved
   ```

## Limits

The compatibility check is structural, so it cannot catch a change that keeps
the shape and changes the meaning: two integer parameters swapped, a widened
offset that the caller computed in 32 bits, a `bool` return whose `false` now
means something new. Those load cleanly against the new kernel and misbehave.

Flavors help only if a kernel developer keeps the old wrapper compiled and
registered with the same flags for the whole deprecation window, and in the
current tree the fallback that finds it is not merged yet; a declaration that
works on this month's libbpf still fails on a distro libbpf, and light skeletons
and the kernel's exact-name lookup never see flavors at all. The alias
exception also puts an obligation on the kernel side that no check can enforce:
two compatible family members must stay semantically identical.

Finally, the documented signal for a kfunc going away is the deprecation flag
plus a kernel-doc note, not an ABI promise; a prototype change is legal at any
time, and the only real protection for an out-of-tree object is to state which
prototype it was built against and to keep the guard cheap.

## References

- [libbpf kfunc resolution, upstream source](https://github.com/torvalds/linux/blob/master/tools/lib/bpf/libbpf.c) — essential-name stripping, the exact kernel lookup, `bpf_core_types_are_compat()`, weak-extern poisoning, and the verifier-log fixup.
- [libbpf CO-RE type compatibility rules](https://github.com/torvalds/linux/blob/master/tools/lib/bpf/relo_core.c) — the exact kind-by-kind comparison, including the integer and struct-name relaxations.
- [Kernel kfunc lifecycle expectations](https://github.com/torvalds/linux/blob/master/Documentation/bpf/kfuncs.rst) — the statement that kfuncs carry no hard stability guarantee and how deprecation is signalled.
- [Public bpf-list series cover letter: kernel-side kfunc flavors](https://lore.kernel.org/bpf/20261117192153.3384031-1-ihor.solodrai@linux.dev/) — the motivation, the sched_ext history, and the measured count of same-name prototype changes between releases.
- [Public bpf-list patch: fall back to kernel flavors](https://lore.kernel.org/bpf/20261117192153.3384031-2-ihor.solodrai@linux.dev/) — the search order, the ambiguity error, and the unchanged exact lookup.
- [Public bpf-list patch: reject ambiguous kfunc families](https://lore.kernel.org/bpf/20261117192153.3384031-4-ihor.solodrai@linux.dev/) — the pairwise check in `resolve_btfids`, the alias exception, and family recognition by decl tag.
- [Public bpf-list patch: document changing a kfunc prototype](https://lore.kernel.org/bpf/20261117192153.3384031-5-ihor.solodrai@linux.dev/) — the rules as they are meant to be documented, including the harmless-change list and the name-it-differently advice.
- [Kernel sched_ext compatibility fix commit](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/patch/?id=2dbbdeda77a6) — the reversed naming that put the old prototype back on the plain name.
- [Kernel sched_ext declaration fix commit](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/patch/?id=56bbc91219c0) — the silently dropped parameter and the double-suffix workaround.
- [sched_ext kfunc definitions](https://github.com/torvalds/linux/blob/master/kernel/sched/ext/ext.c) — both family members registered, the `___v2` name, and the wrapper marked for removal.
- [bpf_ksym_exists definition](https://github.com/torvalds/linux/blob/master/tools/lib/bpf/bpf_helpers.h) — the weak-symbol requirement and the `!!sym` shape the verifier resolves.
- [Kernel verifier, call instruction checks](https://github.com/torvalds/linux/blob/master/kernel/bpf/verifier.c) — pseudo kfunc call encoding, pointer-against-zero branch resolution, and dead-code sanitization.
- [libbpf version header](https://github.com/torvalds/linux/blob/master/tools/lib/bpf/libbpf_version.h) — the 1.9 baseline the fallback would be released with.

## Community discussion today

Two archive-opted Slack workspaces were reviewed through the approved read-only
archive, three allowlisted channels in total: the 24-hour window held no
messages, and the seven-day window held three, one a meeting-notes
announcement, one a contributor introduction, and one already answered
yesterday. So this page's question comes from the public bpf mailing list
rather than from today's chat window. The list was read through its ordinary
public archive pages. The two allowlisted Discord workspaces, ten channels
between them, and r/eBPF were not reviewed this run: all are visible-browser
surfaces and no browser session was opened, so their absence is a coverage gap,
not silence.

**The question above, from the other side.** The list thread is not only about
broken loads. The reviewers pushed on how the fallback should be specified: why
a build-time rejection of ambiguous families is needed at all in the first place,
whether an alias with an identical prototype should be allowed, and whether the
suffixed name is a load-time fallback or a semantic wrapper. The author's
answer is that a per-direction check on every family after deduplication makes
ambiguous bindings impossible, and that the alias is needed exactly because a
namespace already ships a compat macro that declares both the plain and the
suffixed name. What remains open is the obligation the alias creates: two
compatible functions must stay semantically identical, and nothing but review
enforces it.

**Hardware receive offload against attached XDP.** A newer series moves the
"XDP excludes hardware GRO and LRO" rule out of individual drivers into the
network core, because drivers disagree today: some refuse the attach, some clear
the feature from `ndo_fix_features()`, at least one keeps hardware coalescing
running under XDP. The reported failure mode is worth remembering for anyone
debugging a receive workload: a driver that re-adds hardware GRO inside its own
fixup makes the core's warning fire, the attach still succeeds, XDP sees
coalesced frames, and the feature bit is lost for good after detach — which on
one driver also takes header-data split with it, and with it zero-copy receive
paths. The diagnostic is to read the device's wanted and active feature bits
before and after an attach or detach, rather than trusting the interface's
reported state.

**A duplicated caller frame in stack traces.** On arm64 the caller is reported
twice at the top of kernel call chains built from caller registers, because the
register set places the return address in the PC slot while the frame record
already points at that same return address. It affects tracepoint programs,
scheduler software events, and the stack helpers in tracepoint, raw tracepoint,
BTF-enabled tracepoint and fentry programs, and it can be seen plainly in a
short tracepoint stack. x86 draws the distinction by a flags bit; the arm64 fix
marks those register sets with a thread-mode PSTATE value the kernel never
returns to, and skips the leaf PC for them, which also changes what perf-event
programs see in the saved registers. Counting frames from such a stack, or
folding two samples into one, is wrong until the fix lands.

**Read and write APIs for receive metadata.** A large series keeps receive
metadata alive from the descriptor until an skb exists, and the review debate is
about shape rather than need: the reading side must go through per-driver
callbacks because each driver stores metadata in its own descriptor format,
while the writing side always targets the generic frame or skb. The reviewer's
objection is that the write API's separate copy of the information is
unproven, not that the data is unnecessary. Until it settles, treat the
metadata plumbing as a moving API.