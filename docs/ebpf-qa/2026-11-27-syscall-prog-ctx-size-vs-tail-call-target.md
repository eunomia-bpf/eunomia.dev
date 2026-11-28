# Why can a syscall program read past the context buffer it is run with, and why does a tail call into another syscall program decide how big that buffer must be?

**Short answer:** because for `BPF_PROG_TYPE_SYSCALL` the context is an opaque
caller-sized buffer, and the only size check in the run path looks at the
program the caller names — not at the program a tail call may transfer control
to. `bpf_prog_test_run_syscall()` refuses a context smaller than the entry
program's `max_ctx_offset`, the verifier's record of the furthest byte that
program touches. The compatibility check that guards a program-array insert
compares program type, JIT mode, sleepability and a few cookies, but not that
offset. Two syscall programs are therefore mutually tail-callable while
disagreeing about how much context exists, and at run time the callee inherits
the caller's pointer unchanged: the entry program that touches nothing accepts
an eight-byte context, the callee reads at offset 8192 inside that allocation's
neighbours, and its `retval` — the loaded word — travels back to user space.

The fix pads the copied context with zeros to the largest context the program
type allows — 65535 bytes — whenever the entry program can tail call, and
forbids an extension program from adding a tail call to a target that has
none.

## The mechanism

### 1. Where the size check happens, and what it compares

The syscall program type is the one type whose context is not described by a
type-specific structure. Its access callback accepts any offset below 65535:

```c
static bool syscall_prog_is_valid_access(int off, int size,
					 enum bpf_access_type type,
					 const struct bpf_prog *prog,
					 struct bpf_insn_access_aux *info)
{
	if (off < 0 || off >= U16_MAX)
		return false;
	/* No alignment requirements for syscall ctx accesses. */
	return true;
}
```

Because the offset may be variable, the verifier routes such accesses through
the generic memory-range check with a 65535-byte bound, and records the furthest
byte reached:

```c
	if (is_var_ctx_off_allowed(env->prog)) {
		int err = check_mem_region_access(env, reg, argno, 0, access_size, U16_MAX,
						  zero_size_allowed);
		if (err)
			return err;
		if (env->prog->aux->max_ctx_offset < reg_umax(reg) + access_size)
			env->prog->aux->max_ctx_offset = reg_umax(reg) + access_size;
		return 0;
	}
```

Fixed-offset accesses go through `__check_ctx_access()`, which maintains the
same field by a different route. Either way `max_ctx_offset` is a property of
one program, derived from that program's own code.

The run path then uses exactly that property, and only for the program it was
handed:

```c
	if (ctx_size_in < prog->aux->max_ctx_offset ||
	    ctx_size_in > U16_MAX)
		return -EINVAL;
```

So the accepted range is "at least what this program reads, at most 65535
bytes".
For every other program type the left half of that test is nearly vacuous
because the context layout is fixed by the type or by the attach target; for
the syscall type it is the entire contract, since the context is whatever
buffer the caller passes.

### 2. What a tail call does to that pointer

A tail call transfers control without building a new frame: in the interpreter
the callee's instruction array is adopted and every register is left as it was,
with `R1` still holding the context pointer; the x86 JIT jumps into the
callee's entry with the same `rdi`, and the emitted code says so in its own
comment ("rdi == ctx (1st arg)"). No second size check runs, and no second
buffer exists — the callee reads through the same pointer the entry program
received.

That is by design and normally harmless: for tracepoint, raw tracepoint or
extension programs the context layout is fixed, so a caller and a callee of the
same type interpret the same bytes. The syscall type breaks the assumption
because "how many bytes are there" is a per-program verdict.

### 3. Why the compatibility check does not catch it

A program array enforces compatibility when a program is inserted:

```c
		ret = map->owner->type  == prog_type &&
		      map->owner->jited == fp->jited &&
		      map->owner->xdp_has_frags == aux->xdp_has_frags &&
		      map->owner->sleepable == fp->sleepable;
		if (ret &&
		    map->map_type == BPF_MAP_TYPE_PROG_ARRAY &&
		    map->owner->expected_attach_type != fp->expected_attach_type)
			ret = false;
```

Type, JIT/interpreted parity, fragment capability, sleepability, expected
attach type, cgroup-storage cookies and attach prototypes. `max_ctx_offset` is
not in the list, and for every other type it would be redundant. Two syscall
programs — one that never touches the context, one that loads a word at offset
8192 — are reported as compatible.

Compose the three facts and the run is a straight over-read:

1. the entry program has `max_ctx_offset` 0, so `BPF_PROG_RUN` with an 8-byte
   context succeeds;
2. the callee is in the same program array and is allowed in because the types
   match;
3. the callee loads through the inherited pointer, 8184 bytes past the end of
   the 8-byte allocation, and returns the value, which is copied back to user
   space as `retval`.

With a zero-length context the same path does not even start from an
allocation: `ctx` stays `NULL`, so the callee's displacement becomes a
low-address load instead of a read of a neighbouring slab object.

### 4. What the fix changes

Three changes, each closing one of the three facts above.

`bpf_prog_test_run_syscall()` now sizes the copy from "does this program reach
a tail call", not from `max_ctx_offset`:

```c
	/* the target of a tail call can use that much */
	if (prog->aux->tail_call_reachable)
		size = max_t(u32, size, U16_MAX);

	if (size) {
		ctx = kvzalloc(size, GFP_KERNEL_ACCOUNT | __GFP_NOWARN);
		if (!ctx)
			return -ENOMEM;
		if (copy_from_user(ctx, ctx_in, ctx_size_in)) {
			err = -EFAULT;
			goto out;
		}
	}
```

`tail_call_reachable` is the verifier's summary of the program and its
subprograms, set on the entry program once any reachable path contains a tail
call. It is the right predicate: it is exactly "some program other than this
one may dereference this pointer". `kvzalloc` supplies zero-filled padding, so
a callee reading inside the padded region sees zeros rather than the previous
tenant of that allocation.

The kernel-side caller cannot pad, because it passes the caller's pointer
straight through with no copy. `kern_sys_bpf()` therefore demands the full
size when the program can tail call:

```c
		if (attr->test.ctx_size_in < prog->aux->max_ctx_offset ||
		    (prog->aux->tail_call_reachable && attr->test.ctx_size_in < U16_MAX) ||
		    attr->test.ctx_size_in > U16_MAX) {
			bpf_prog_put(prog);
			return -EINVAL;
		}
```

And an extension program may no longer add a tail call to a target that has
none, because the padding decision is taken from the entry program's flag:

```c
	if (prog->type == BPF_PROG_TYPE_EXT &&
	    tgt_prog->type == BPF_PROG_TYPE_SYSCALL &&
	    prog->aux->tail_call_reachable && !tgt_prog->aux->tail_call_reachable) {
		err = -EINVAL;
		goto out_unlock;
	}
```

That is the same shape as the check directly above it, which already refuses a
freplace program whose context reach exceeds its tracepoint or syscall target's.

## Verifying it

Two probes, both cheap, and neither needs a patched kernel.

1. **Read the effective context size of one program, without any new
   interface.** The smallest `ctx_size_in` that a test run accepts *is* that
   program's `max_ctx_offset`, because the check is a comparison against it.
   Run the entry program with sizes 0, 8, 64 and the target with 0, 8, 8192,
   8196: the entry program starts succeeding at 0, the callee only at 8196 if
   its load is a 4-byte access at offset 8192. The gap between the two
   numbers is the whole bug, and it is visible before anything is run through a
   tail call.
2. **Then observe the transfer.** Insert both programs into one
   `BPF_MAP_TYPE_PROG_ARRAY`, run the entry program with a small context the
   entry program cannot read on its own, and print `retval`. If the entry
   program touches nothing and the run returns non-zero, the callee read
   outside the buffer. Vary the size class between runs so the allocation lands
   at different addresses; identical sizes tend to reproduce the same stale
   bytes and make the effect look deterministic.

A third check is a negative one worth keeping: after the fix, the same entry
program rejects a 4096-byte context outright, and accepts only
`ctx_size_in == 65535`. A run that still accepts 4096 is exercising the old
code path, not the padded one.

## The limits

- **The fix is a padding, not a compatibility rule.** Two syscall programs may
  still disagree about how much context exists; they now always get 65535
  bytes of zero-padded memory, so the disagreement has no consequence for
  reads. The alternative raised in the report — carrying the 16-bit offset in
  the map owner structure — would have failed the insert instead, at the cost
  of rejecting combinations that work fine.
- **Writes past the caller's buffer are dropped.** Only `ctx_size_in` bytes of
  the padded copy are copied back, so a callee that writes beyond the caller's
  size writes into scratch space. Good for safety, invisible in the result.
- **65535 bytes are spent per run, not per program.** A tail-call-reachable
  syscall program zeroes and copies that much on every test run, accounted to
  the memory cgroup; the pre-fix path asked `memdup_user` for the exact size. A
  program with no reachable tail call keeps the exact-size path.
- **`kern_sys_bpf()` callers pay in their own buffer.** Nothing is copied
  there, so the caller must supply the whole 65535 bytes; a light-skeleton
  loader built before this change passes a smaller structure and now fails.
- **The bound comes from the program type, not the kernel's notion of
  "enough".** `U16_MAX` (65535) is what the access callback and the run path
  both use, so that really is the ceiling; a future program type with a larger
  context would need its own padding rule.

## References

- [BPF mailing list: out-of-bounds context access with syscall programs and tail calls](https://lore.kernel.org/bpf/CAAYXoE8ORWnqLVofqZf2AxUvnV_Bd1Gsqb3qdJv+dCM2bVkvAg@mail.gmail.com/) — the report, the two-program reproducer, and the map-owner suggestion.
- [Patch: give a syscall program the largest context when it does tail calls](https://lore.kernel.org/bpf/20261116152626.4032587-4-alexei.starovoitov@gmail.com/) — the message with the commit text and diff.
- [Commit in the BPF next tree](https://git.kernel.org/pub/scm/linux/kernel/git/bpf/bpf-next.git/commit/?id=08c9879049705193e77fd98a8103148b620ef0f1) — applied form of the three changes.
- [Run path for syscall programs](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/net/bpf/test_run.c) — `bpf_prog_test_run_syscall()`, the size check, the padding, and the copy-back.
- [Syscall and test-run entry points](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/syscall.c) — `kern_sys_bpf()`, the extension-versus-target context check, and the `U16_MAX` test-run gate.
- [Verifier context accounting](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/verifier.c) — `max_ctx_offset` updates, the variable-offset range check, and `is_var_ctx_off_allowed()`.
- [Tail-call compatibility and transfer](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/core.c) — the map-owner comparison, and the interpreter's tail-call case.
- [Program map insertion](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/arraymap.c) — where compatibility is enforced for a program array.
- [x86 JIT tail call](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/arch/x86/net/bpf_jit_comp.c) — the indirect jump that reaches the callee entry with the context in the first argument register.
- [Program and attribute declarations](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/include/uapi/linux/bpf.h) — the syscall program type, the test-run attributes, and the documented context-size contract.

## Community discussion today

Coverage: two archive-opted Slack workspaces (three allowlisted channels) were
read through the approved read-only archive and held no message in either the
24-hour or the 7-day window, so that side is quiet rather than unread. The two
allowlisted Discord workspaces (ten channels) could not be read — the web
client sat at its sign-in screen — so their activity is a coverage gap, not
silence. The public r/eBPF page was read as an ordinary web page and carried a
practical AF_XDP question, summarized below. The public bpf list archive
supplied the question above and the rest of this section.

**A context description that only one side of a tail call sees.** The anchor
thread is the cleanest recent example of a general shape: whenever a program
type lets the caller decide the size of an input, some record of "how much did
this program read" exists, and every path that hands the same pointer to
another program has to consult that record. Here the record exists
(`max_ctx_offset`), the run path consults it, and the tail-call path does not,
because for every other program type the answer is a constant. The other
mechanism worth carrying away is `tail_call_reachable` as the predicate: the
cost of the fix is paid per run by exactly the programs that can hand their
context to someone else, and a program that cannot is untouched.

**A mac header that outlives the frame in generic XDP.** A v2 fix recomputes
`skb->tail` after `bpf_xdp_adjust_tail()` in generic mode by subtracting
`mac_len`, because the XDP frame starts `mac_len` bytes before `skb->data`.
Review pointed at the boundary: the helper only guarantees a floor of
`ETH_HLEN` bytes after `xdp->data`, while `mac_len` can be larger — a GRE
device whose header operations have no remote leaves the mac header on the
outer IP header, making `mac_len` at least 24. Shrink a frame to 14 bytes on
such a device and the subtraction puts the tail before the data and wraps
`skb->len`; a later tail adjustment then trusts the wrapped length. The same
gap exists on the `adjust_head` side. The diagnostic that generalizes is to
compare the two constants before trusting either: the helper's floor is a
property of the helper, `mac_len` is a property of the device's header
operations, and generic XDP is the place where both are in scope.

**Classic BPF in the seccomp path, and where the lock time goes.** A
seccomp filter cache proposal was re-aimed at a measured symptom: profiling
showed many short-lived instances collapsing while queueing on one global lock
in the program-pack allocator. The follow-up analysis separates the allocation
path from the free path: allocation may run an indirect-branch predictor flush
when a pack that previously held classic BPF is reused, while the free path
runs the architecture's text invalidation, which on arm64 is a
`memset`-style patch of the code region followed by an inter-processor
interrupt to every CPU before the cache flush completes. Because the pack lock
is held across that, every loader queues behind a free. The useful reading is
that a lock's hold time can be dominated by something the lock does not
protect — the invalidated region cannot be handed out again until the bitmap is
cleared, so the IPI could move outside the critical section, while the
allocator's pack-walking remains a real serialization point.

**Interpreter-unsupported instructions, backported.** The series that makes a
program refuse interpreter fallback when it uses instruction forms only a JIT
implements was queued for both the 6.18 and 6.12 stable trees this week. For a
reader already running one of those kernels, that is the version to watch: the
failure it prevents is a load that succeeds on a JIT-enabled host and faults
later on a host where the JIT was disabled or exhausted, which no load-time log
distinguishes. The check to run before such a rollout is the same one this
column described earlier: read whether the program is JITed
(`bpf_prog_info`'s jited lengths, or the test-run behaviour) rather than
assuming the interpreter is a fallback that always works.

**CO-RE relocation and the sign bit of an immediate.** A one-line libbpf fix
addresses a relocation whose local value is zero-extended but whose
instruction immediate is sign-extended: since 64-bit relocation values were
allowed, the comparison compares a sign-extended 32-bit immediate against a
64-bit value, so a zero-extended unsigned value with bit 31 set — an
enumerator of `0x80000000` compiled as a 32-bit move — is rejected as an
unexpected instruction. The fix compares the low 32 bits, which is what an
immediate holds. The practical consequence: an object built with a newer clang
can fail to load with an older libbpf, and the error names an
instruction rather than the enum value that produced it.

**Copy-mode AF_XDP transmit on a virtual link.** A public question asked
whether a send-heavy scanner of 64-byte packets should shard AF_XDP sockets per
queue with a shared UMEM, or keep transmit on raw sockets; the measured ceiling
was about 24k frames/s through one transmit socket on a veth pair, attributed
to a per-call budget of 32 descriptors. The budget is a socket option with a
floor of 32, a default of 32, and a maximum of the transmit ring size, so 32 is
the smallest value a user may request rather than a cap the kernel imposes.
Copy mode also means the generic transmit path: per descriptor an skb is built
and the frame copied into it, under the socket mutex, so raising the budget
amortizes call overhead and leaves the copy and the serialization in place.
Whether a device can do zero-copy is an advertised feature flag, which is why a
veth measurement is necessarily copy mode; that is the argument for one
transmit socket per queue with per-core senders.