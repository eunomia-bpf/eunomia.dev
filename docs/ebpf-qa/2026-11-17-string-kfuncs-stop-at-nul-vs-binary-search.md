# Why does an in-kernel BPF string search stop at the first zero byte, and what would a binary pattern search need?

**Short answer:** every BPF string kfunc is a C-string primitive. `bpf_strnstr`, added in 6.17, and the family around it read the haystack and the needle one byte at a time through a nofault accessor and return as soon as either side yields `'\0'`, so a buffer with an embedded zero is searched only up to that byte. Both sides are also capped at 65536 bytes, and a longer input returns `-E2BIG` instead of a partial answer. There is no `memmem` in the kernel's `lib/string.c` and no kfunc that wraps `memchr` plus `memcmp`; the `bpf_strncmp` helper is not a substitute, because it delegates to `strncmp()`, which leaves its loop at a NUL as well. Binary pattern search today is a hand-written byte loop, and the verifier's iteration bound rules out the nested form that a 64 KB haystack with a 64 KB needle would need.

## The mechanism

### 1. The verifier does not type the string arguments

Every argument of these kfuncs carries the `__ign` suffix, for example `const char *s1__ign`. The verifier classifies such an argument with `is_kfunc_arg_ignore()`, which matches the suffix, and `get_kfunc_arg_type()` turns it into `ARG_IGNORE`. `check_func_arg()` then does one thing with it:

```c
if (arg_type == ARG_IGNORE)
        return 0;
```

The register is read as a source operand and nothing else is checked: no pointer type, no size, no alignment, no opt-in. The safety comes from the implementation, which validates the address once with `copy_from_kernel_nofault_allowed()` and then reads byte by byte under `guard(pagefault)` with `__get_kernel_nofault()`. On x86-64 that first check rejects anything below `TASK_SIZE_MAX + PAGE_SIZE`, the vsyscall page, and non-canonical addresses, which is why the selftests expect `-ERANGE` there for a user pointer, and `-EFAULT` on architectures whose address spaces overlap so the check cannot reject a user address up front. The practical consequence is that a string literal, a read-only map value, and a pointer into a ring buffer sample are all equally acceptable — the kfunc cannot tell them apart, and it does not know how long any of them is.

That is also why the kfuncs are gated twice. They are registered on the common kfunc set, which `btf_kfunc_is_allowed()` consults before the program-type set, so they are reachable from XDP, TC, tracing and syscall programs. Each is flagged `KF_PERFMON`:

```c
BTF_ID_FLAGS(func, bpf_strnstr, KF_PERFMON);
```

and the verifier rejects the call when that flag is set and the loader lacks pointer-leak rights:

```c
if (is_kfunc_perfmon(&meta) && !env->allow_ptr_leaks)
```

That is `CAP_PERFMON` (or `CAP_SYS_ADMIN`) in practice, since the kfunc reads through a pointer the verifier never checked. A generated, unprivileged program cannot use them at all.

### 2. The NUL terminator is part of the algorithm, not an accident of the reader

The body of `__bpf_strnstr` is a two-byte-string scan with two exits on zero. The needle side, inside the inner loop:

```c
__get_kernel_nofault(&c2, s2 + j, char, err_out);
if (c2 == '\0')
        return i;
```

and the haystack side, after the needle byte has been read:

```c
__get_kernel_nofault(&c1, s1 + j, char, err_out);
...
if (c1 == '\0')
        return -ENOENT;
```

Both exits are semantic, not bounds checks. A needle byte of zero means "the needle ended here", so the current position is a match. A haystack byte of zero means "the haystack ended here", so the whole remaining scan is abandoned and the call answers `-ENOENT` even if the pattern sits one byte later. For `{"\x01", "\x00", "A", "B"}` searched for `"AB"` with `len` 4, the scan stops at index 1 and reports no match, although the two bytes are present at offset 2. The `strnstr` in `lib/string.c` has the same shape for a different reason: it calls `strlen()` on the needle and `memcmp()` on candidate windows, so a needle containing zero is truncated before the comparison starts.

### 3. The cap is the extended-attribute limit, applied to both sides

The outer loop bound and the needle bound are both `XATTR_SIZE_MAX`, which is 65536 in `include/uapi/linux/limits.h`:

```c
for (i = 0; i < XATTR_SIZE_MAX; i++) {
        for (j = 0; i + j <= len && j < XATTR_SIZE_MAX; j++) {
```

The `len` argument limits how far into the haystack a match may begin, not how much is read: the comment in the source notes that s2 may be read one byte past `len`, to cover the case where the needle is a suffix of the searched prefix. When the outer loop or the needle loop reaches the cap, the function returns `-E2BIG`, so a haystack larger than 64 KB produces an error rather than "not found in the first 64 KB". The cap is a deliberate bound on how long a kfunc may spin inside the kernel with page faults disabled; it is not related to the size of any particular buffer, and a caller with a 2 MB ring buffer sample has no way to ask for the rest.

### 4. The strncmp helper stops at zero too, and needs a frozen map

`bpf_strncmp` looks like the binary answer, and is not one. Its third argument is a constant string, checked by `check_arg_const_str()`: the register must point into a value of a map created read-only for the program, the value must be frozen, the offset must be a verifier-known constant, and the string must be NUL-terminated inside the value (`strnchr(..., 0)` must succeed). Its body is one line:

```c
BPF_CALL_3(bpf_strncmp, const char *, s1, u32, s1_sz, const char *, s2)
{
        return strncmp(s1, s2, s1_sz);
}
```

and `strncmp()` leaves the loop at a zero:

```c
if (c1 != c2)
        return c1 < c2 ? -1 : 1;
if (!c1)
        break;
```

So two binary buffers that agree up to a shared zero byte and differ afterwards compare equal. A UUID, a hash digest, or a packet header with a zero byte in the middle is exactly the case this misfires on, and it misfires as a false match rather than as an error.

### 5. What a binary search needs, and why the loop bound decides the shape

The obvious userspace workaround is to walk the buffer with `bpf_for(i, start, end)`, which is a `bpf_iter_num` loop, and compare bytes under it. The comparison is cheap; the loop count is not. `bpf_iter_num_new` is inlined by the verifier, and the inlined sequence rejects a range wider than `BPF_MAX_LOOPS`, which is `8 * 1024 * 1024` in `include/linux/bpf.h`:
```
r0 = (u32)end - (u32)start; if (r0 > BPF_MAX_LOOPS) goto e2big;
```

A single pass over a 64 KB buffer is fine. The pattern search itself is two nested passes — for each candidate offset, compare the needle — so a 64 KB haystack and a 64 KB needle is 4 Gi byte comparisons, over 500 times the bound, and no amount of manual unrolling fits. That is the structural reason a search primitive has to live in the kernel rather than in a macro: the inner scan has to be one kfunc call, not a BPF loop.

A kfunc side also has to do the work the dynptr path would otherwise force on the caller. `bpf_dynptr_slice()` takes a length that must be a verifier constant, because the verifier types the returned pointer as `PTR_TO_MEM` with `mem_size` taken from that constant and a `parent_id` tying it to the dynptr; the slice is invalidated when the program stores to its scratch buffer. For an skb or XDP dynptr the slice is a direct pointer only when the requested range is linear, and `NULL` otherwise unless the caller supplied a buffer. So a search kfunc that takes a dynptr has to call `bpf_dynptr_check_off_len()` itself, copy a non-linear region into a scratch buffer, and then run the scan over bytes it owns — which is precisely what the string kfuncs cannot do today, because they take a bare pointer with no length.

## Verifying it

On a kernel that has the kfunc, presence is a BTF question, and a substring search over a buffer with an embedded zero is a one-program question:

```bash
bpftool btf dump file /sys/current/btf/vmlinux format raw | grep -c '^bpf_strnstr$'
```

A `syscall` program with a global `char buf[] = "\x01\x00AB";` and a body of `return bpf_strnstr(buf, "AB", 4);` returns `-ENOENT`, and the same program over `"AABB"` returns 1. The upstream tests for exactly this family are compiled into `test_progs`, so `./test_progs -t string_kfuncs` exercises the found, not-found, empty-needle and user-pointer cases; the `-ERANGE` versus `-EFAULT` split in their expected values is the cheapest way to confirm which architecture path the address check took.

For the absence claim, three greps decide it on any tree: no `bpf_memmem` and no `bpf_memcmp` in `kernel/bpf/helpers.c`, and no `memmem` in either `lib/string.c` or `include/linux/string.h`, while `memchr` and `memcmp` are both present and exported. The tree also explains why no libc fallback exists: LLVM's BPF backend rejects a call to any built-in function other than `memcpy`, `memset`, `memmove`, the 128-bit integer helpers and the trap, unless the experimental libcall flag is set, so `__builtin_memcmp` has to be inlined to survive code generation.

## Limits

The absence is a snapshot, not a law. A `bpf_memcmp` and `bpf_strpbrk` series reached the mailing list as a v2 patchset, and neither kfunc is in `kernel/bpf/helpers.c` at 7.4-rc3, so a page that promises them will age badly; read the tree, not this page, before relying on a name. The same caution applies to the string family's release history: `bpf_strnstr` itself exists in 6.17 but not in 6.16, and the case-insensitive siblings arrived after it, so a "the kfuncs do X" sentence is only true relative to a version.

The 64 KB cap is not going away with a rename. Any search kfunc has to pick a bound for how long it may run inside the kernel with page faults disabled, and picking 65536 means a large buffer is partially searchable at best, with `-E2BIG` as the honest answer for the rest. A caller who needs to search a 1 MB sample will still be splitting it into windows, and a match that straddles a window boundary is still the caller's problem.

Finally, none of this widens what an unprivileged program can do. `ARG_IGNORE` looks loose, but the `KF_PERFMON` gate is what makes it safe: the kfunc can read arbitrary kernel memory, and only a privileged loader is allowed to ask.

## References

- [bpf_strnstr and the string kfunc family, kernel source](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/kernel/bpf/helpers.c) — the NUL exits, the `XATTR_SIZE_MAX` loops, `bpf_strncmp`, and the `KF_PERFMON` registrations.
- [kernel strnstr and memchr, lib/string.c](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/lib/string.c) — the C-string scan the kfunc mirrors, and the absence of `memmem`.
- [XATTR_SIZE_MAX, uapi limits](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/include/uapi/linux/limits.h) — the 65536 value used as both scan caps.
- [BPF_MAX_LOOPS, include/linux/bpf.h](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/include/linux/bpf.h) — the 8 Mi iteration bound that the inlined `bpf_iter_num_new` enforces.
- [kfunc annotation rules](https://docs.kernel.org/bpf/kfuncs.html) — the `__ign`, `__sz` and `__k` suffixes, the `KF_PERFMON` flag, and the per-program-type registration of a kfunc set.
- [bpf_strncmp helper documentation](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/include/uapi/linux/bpf.h) — the constant-string third argument and its stated size semantics.
- [string kfunc selftests](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/tools/testing/selftests/bpf/progs/string_kfuncs_success.c) — the expected results per call, including the empty-needle and suffix cases.
- [Binary comparison gap argued on the list](https://lore.kernel.org/bpf/20260624093638.117062-1-lange_tang@163.com/) — the v2 `bpf_memcmp` and `bpf_strpbrk` cover letter: `bpf_strncmp` stops at null bytes, and the selftest workaround.
- [dynptr copy selftest with a local compare loop](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/tools/testing/selftests/bpf/progs/dynptr_success.c) — the hand-written byte comparison the kernel's own tests fell back to.

## Community discussion today

Two archive-opted Slack workspaces were reviewed through the approved read-only archive, three allowlisted channels in total. The 24-hour window held no messages; the seven-day window held three, of which one is the question answered above and two are introductions without a technical question, so this page's question comes from the wider seven-day window rather than today. The public bpf mailing-list archive was reviewed as a public page. The two allowlisted Discord workspaces, ten channels in total, and r/eBPF were not reviewed this run: both are visible-browser surfaces and no browser session was opened, so their absence here is a coverage gap, not silence.

**Binary search over non-string buffers.** The question above is the same gap the mailing list is arguing, from the other end. The proposed `bpf_memcmp` kfunc is motivated by three concrete workloads — six-byte MAC comparisons for L2 filtering, sixteen-byte UUID and digest allowlists, and hand-rolled loops that the verifier makes awkward — and names the exact defect this page documents: `bpf_strncmp` stops at null bytes, so it corrupts binary comparisons. The companion `bpf_strpbrk` targets the string side of the same problem, delimiter search in mount options and `/proc` entries, where `strspn` plus `strcspn` needs two calls to do what one call should. What remains unresolved is placement rather than need: whether the primitive takes raw pointers like the rest of the string family, or a dynptr so that skb and XDP fragments are handled inside the kfunc instead of by every caller, and what the length argument means when the buffer came from a packet rather than from a frozen map value.

**What "length" means for a length-limited search.** The `bpf_strnstr` length semantics took several revisions before settling, and the reason is a boundary this page's verification relies on. A search bounded by `len` must still accept a needle that ends exactly at `len`, which means reading one byte beyond the searched prefix from the needle and comparing it before deciding; the source keeps a comment explaining that extra read. The resulting behaviour is subtle in both directions: a match that would start within `len` but extend past it is not reported, and a needle longer than the cap yields `-E2BIG` from the inner loop rather than `-ENOENT`. Anyone writing a binary variant inherits the same question and, unlike the string case, cannot use "the haystack ended" as the reason to stop — which is the whole point.

**Failure semantics differ by architecture.** The selftests encode a split that a page like this one has to state carefully: a NULL or user-space pointer handed to a string kfunc returns `-ERANGE` on architectures with non-overlapping address spaces, including x86-64, and `-EFAULT` on the others. On s390 the underlying nofault accessor returns without faulting for NULL, so the whole group of tests is compiled out there. The practical lesson for a caller is not to map the error code to a cause — `-ERANGE`, `-EFAULT` and `-ENOENT` all arrive from the same call — but to treat all three as "the scan did not complete" and to validate the pointer and length on the caller's side first.

**Where the privilege boundary actually sits.** Because the string arguments are ignored by the verifier, the capability check is the only thing standing between a program and an arbitrary kernel read, and the discussion converges on keeping it that way: the kfuncs carry `KF_PERFMON`, and a non-GPL program cannot call a kernel function at all. The open question for a new search kfunc is the same one, one level down: a dynptr-based variant would be strictly safer than a pointer-based one, but it would also be callable from fewer program types, and the packet path is where the need is strongest.