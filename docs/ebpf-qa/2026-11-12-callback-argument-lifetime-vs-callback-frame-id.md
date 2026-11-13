# Why can a pointer handed to a BPF callback outlive the callback invocation, and how does a callback frame id revoke it?

Because the verifier records what the argument points at, but nothing records how long it lives.
Two callback-calling helpers pass their callback a pointer to a local variable on the helper's own
stack: the user ring buffer drain helper passes the address of a descriptor it declares in its
sample loop and releases the sample as soon as the callback returns, and the array map iteration
helper passes the address of its loop counter. A callback can copy that pointer into the caller's
frame, and at callback return the verifier scrubs only the state that belongs to the callback
frame itself. The copy keeps its type and its identifier, reloads as the same pointer type, and
stays dereferenceable after the storage it names has been reused. A posted series fixes the shape
at the lifetime rather than at the access: give the callback frame an id, make that id the id and
the parent id of the argument, and invalidate the id and everything derived from it when the
frame is popped.

## The mechanism

1. **The argument is a pointer into the helper's stack.** The user ring buffer drain helper
   declares `struct bpf_dynptr_kern dynptr;` inside its per-sample loop, initialises it over the
   peeked sample, invokes the callback with its address, and only then releases the sample back to
   the producer. The array map iteration helper declares `u32 i, key;` for the whole loop, assigns
   the index to the key, and invokes the callback with the key's address. In both cases the
   callback's first or second argument is an address in a kernel stack frame that is reused by the
   next invocation.

2. **The verifier records a type, and for the dynptr an id, but no owner.** The user ring buffer
   callback setup marks the first argument with the local dynptr type through a helper that
   assigns a fresh identifier from the verifier's id counter and sets the parent identifier to
   zero. The map iteration setup assigns the second argument the map-key pointer type with a
   zeroed known scalar offset, the map pointer, and a map uid, and assigns no identifier at all;
   the value register in the third slot does get a fresh identifier, because that identifier
   tracks map value reads, not lifetime. So the dynptr argument carries an id that no other state
   refers to, and the key argument carries none.

3. **Return scrubs the callback frame, not the caller's copies.** On callback exit the verifier
   destroys dynptrs held in the callback frame's own stack slots, then invalidates stack slices
   whose recorded frame number is the callback's. Both filters are frame-local: the slice check
   skips any register whose recorded frame number differs, and the dynptr check walks only the
   slot range of the frame being popped. The callback's register file is then freed. A spilled
   copy on the caller's stack, or a copy written into a map, is in neither set, so it survives
   with its type and its id.

4. **Spilling is what turned this into a reachable bug.** Until mid-2026 the verifier decided
   whether a spill preserves a pointer by consulting an allow-list of pointer types that did not
   include the constant dynptr pointer. A dynptr pointer written to a stack slot therefore fell
   through to the branch that marks the slot as miscellaneous bytes, and a later load produced a
   scalar rather than a pointer, so a saved descriptor could not be dereferenced at all. The
   commit that replaced that allow-list with "anything that is not a scalar and not uninitialised"
   made the spill a real spilled pointer, preserving the type and the id across the store and the
   load. The posted fix carries a second `Fixes` tag for exactly that commit, because the leak
   became usable when the spill did.

5. **The fix is a frame id owned by the callback state.** The series adds one field to the
   callback's function state, assigns it from the verifier's id counter the first time a
   frame-scoped argument is marked, and sets the argument's id and parent id to that value. It
   marks only arguments that point into the helper's frame: the dynptr for the drain helper, and
   the key for array and per-CPU array maps. On exit, after the existing stack work, it calls the
   same invalidation routine the verifier already uses when a reference is released. That routine
   walks every register and spilled slot in the current state, matches on the identifier or on the
   parent identifier, pushes the matched register's own id so descendants are visited too,
   invalidates a spilled dynptr in its two-slot pair, and marks everything else as not
   initialised.

6. **Why the argument is its own parent.** The invalidation graph is keyed on parent identifiers:
   a slice returned by `bpf_dynptr_data` records the dynptr's identifier as its parent, and that
   helper call clobbers the argument register under the normal calling convention. If the frame had
   a separate identifier from the dynptr, the slice's parent chain would end at the dynptr id,
   which nothing invalidates on frame exit, so a parked slice would still be usable. Making the
   argument's own id equal to the frame id closes the chain: the argument is its own parent, and
   every slice or clone that descends from it descends from the frame.

7. **State comparison needs the same identity, or the fix is undone by merging.** Two verifier
   states are considered equivalent by comparing registers, spills, stack arguments, and the loop
   stack, with scalar and reference identifiers mapped through a bijection so that "the same value"
   is preserved structurally. A callback frame id is one more identifier in that comparison: two
   states that arrive at the same instruction from different callback invocations must not merge
   when their frame ids differ, and the series adds that mapping to the frame comparison. Without
   it, a joined state could carry a value attributed to the wrong callback frame.

8. **What is deliberately not scoped.** The array key is callback-local because it is a `u32` on
   the helper's frame. A hash map's key is not: the iteration helper hands out a pointer into the
   stored element, which lives in the map, and every map value pointer is map storage as well. The
   series leaves both unscoped and says so, which is the useful boundary to remember: the property
   being fixed is "the storage is the helper's stack", not "the helper called a callback".

## Verifying it

1. **Read the two producer sites.** The lifetime claim is a local-variable declaration and a call
   order, so read them rather than trusting a summary: in the ring buffer helper, the descriptor
   declaration is inside the sample loop and the release call follows the callback call; in the map
   iteration helper, the counter and key are declared before the loop and the callback receives
   the key's address.

2. **Confirm the verifier records no owner.** The dynptr callback setup calls the marking helper
   whose identifier argument is a pre-increment of the verifier's id counter and whose parent
   argument is zero; the map callback setup assigns the key's pointer type, map pointer, and map
   uid, and assigns no id. Grep for those two setup functions and for the two-argument marking
   helper's definition.

3. **Confirm return only scrubs the callback frame.** Read the callback exit path: the dynptr scan
   is bounded by the callee's allocated stack, and the slice invalidation takes the callee's frame
   number and skips registers whose recorded frame number differs. Nothing iterates the caller's
   frame afterwards.

4. **Compile the parked-descriptor program and let the verifier decide.** A callback that stores
   its dynptr argument into the context it was given, and a caller that calls `bpf_dynptr_read`
   through the stored pointer afterwards, is the whole reproducer. On an unfixed kernel the program
   loads; with the series applied it fails with the messages the new selftests assert. For a
   parked descriptor the verifier reports that the expected type for the argument is a frame
   pointer or a dynptr pointer and that it saw a scalar; for a parked slice, and for a parked
   array or per-CPU array key, the message is `invalid mem access 'scalar'`. The hash-key and
   map-value cases stay successful.

   ```console
   $ clang -O2 -g -target bpf -D__TARGET_ARCH_x86 \
       -I/usr/include/x86_64-linux-gnu -c park.c -o park.o
   $ bpftool prog load park.o /sys/fs/bpf/park type raw_tracepoint
   ```

5. **Ask the running kernel whether it carries the field.** The fix adds a field to the verifier's
   function state, so a kernel's own BTF answers the version question without a build:

   ```console
   $ bpftool btf dump file /sys/kernel/btf/vmlinux format c \
       | grep -A6 "struct bpf_func_state {"
   ```

   A kernel that prints only the diagnostic frame identifier and no second one predates the series,
   whatever its release number says about the surrounding verifier code.

## Limits

- The series is unmerged. Its first revision drew a "the fixes look good" from a maintainer, then a
  respin request: the set no longer applied, and both of the author's series were asked to be
  merged into one against the bpf-next tree. Read the shape as the intended direction, not as
  shipped behaviour, and re-check the patch before citing line numbers.

- The fix is opt-in per helper. Each callback setup function has to mark its own frame-scoped
  arguments, so a new helper that passes a stack address to a callback inherits the same bug until
  it is marked. Nothing in the mechanism detects the pattern generically.

- Neither helper in this shape protects the value; both protect a pointer. A callback that copies
  the eight bytes the key points at, instead of the key pointer, is unaffected, and a program that
  wants a descriptor after the helper returns has to keep using it inside the callback or copy the
  bytes out.

- The array key is only a four-byte leak because key loads are read-only: the verifier rejects a
  write through the map-key pointer type outright. The dynptr descriptor is the more dangerous
  half, since the memory it describes is released to the producer, and the series describes the
  consequence as arbitrary kernel read and write.

- The added identifier participates in state comparison, so callback states whose frame ids differ
  no longer merge. That is the point, and it also means a state that would have merged before can
  now be explored twice; the only cost a reader would notice is a load that takes longer on a
  program already near a state or complexity limit.

- The interaction with a pending verifier rewrite that replaces state equality with a join routine
  in its own file is unresolved: the author sent the maintainer a hunk that teaches the join
  comparison about the frame id, and the respin is what settles where it lands. A reader on a
  bpf-next tree needs that hunk, and a reader on a release tree has neither.

- The environment used to check this page could compile the reproducer but not load it: the process
  had no `CAP_BPF`, could not raise its locked-memory limit, and the kernel refused even a trivial
  program object, so the verifier verdicts here are read from the series' own selftest assertions
  rather than observed.

## References

- [Kernel `kernel/bpf/ringbuf.c`](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/ringbuf.c) — the user ring buffer drain helper: stack-local descriptor, callback call, sample release, and the helper prototype.
- [Kernel `kernel/bpf/arraymap.c`](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/arraymap.c) — the array and per-CPU array iteration callback and the stack-local key it passes.
- [Kernel `kernel/bpf/verifier.c`](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/verifier.c) — callback argument setup for both helpers, the dynptr register marking, reference invalidation, and the callback exit path.
- [Kernel `kernel/bpf/states.c`](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/states.c) — function state comparison and the identifier mapping it uses.
- [Kernel `include/linux/bpf_verifier.h`](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/include/linux/bpf_verifier.h) — the callback function state that gains the frame id.
- [Kernel `include/linux/bpf.h`](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/include/linux/bpf.h) — the per-CPU instrumentation counter used as the recursion check, and the per-CPU program miss counters.
- [Series cover letter, `Scope callback arguments to their frame`](https://lore.kernel.org/bpf/20261106231157.2840635-1-ihor.solodrai@linux.dev/) — the two affected helpers, the frame id design, and the revision history.
- [Patch 1, scoping the drain dynptr](https://lore.kernel.org/bpf/20261106231157.2840635-2-ihor.solodrai@linux.dev/) — the frame id field, the marking helper, both exit hooks, and the two `Fixes` tags.
- [Patch 2, the user ring buffer selftests](https://lore.kernel.org/bpf/20261106231157.2840635-3-ihor.solodrai@linux.dev/) — the parked descriptor, parked slice, and clone cases with their exact rejection strings.
- [Patch 3, scoping the array key](https://lore.kernel.org/bpf/20261106231157.2840635-4-ihor.solodrai@linux.dev/) — the exported marking helper and the statement that hash keys and map values stay unscoped.
- [Patch 4, the map key selftests](https://lore.kernel.org/bpf/20261106231157.2840635-5-ihor.solodrai@linux.dev/) — parked array and per-CPU array keys failing and the hash key and array value cases passing.
- [Maintainer reply accepting the direction](https://lore.kernel.org/bpf/DMOO099NDMIT.3A6PF31D90IZ2@gmail.com/) — the tree question and the verdict on the fixes.
- [Maintainer reply requesting the respin](https://lore.kernel.org/bpf/DMTJ704LM94F.W2LQRVNGZAJJ@gmail.com/) — the no-longer-applies verdict, the merged-respin request, and the join comparison hunk it quotes.
- [Commit that added the drain helper](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/commit/?id=205715673844) — the first `Fixes` tag of patch 1.
- [Commit that added array map iteration](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/commit/?id=06dcdcd4b9e8) — the `Fixes` tag of patch 3.
- [Commit that removed the spill allow-list](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/commit/?id=1f737e46ca6a) — replacing the list of spillable pointer types with a check for any pointer type, which made the parked descriptor reloadable.

## Community discussion today

This run reviewed the private read-only archive for two opt-in sources, three allowlisted channels
in total, and read the public bpf mailing list as the second public surface. The archive was quiet
in the strict sense: neither source held a message in the 24-hour window, and the seven-day window
held ten messages in one source, none in the other, all of them logistics for a conference
maintainer meeting and the project talks at co-located events, with a note that a third party was
presenting network traces combined with profiles for threat detection. None of them was a technical
question, so the answer above comes from a public thread inside today's list window, which carried
45 threads. Two community surfaces could not be covered at all and are recorded as inaccessible
rather than quiet: both project chat workspaces answer their public invites with an invalid-invite
page, and the practitioner forum blocked this environment with a network-security page.

### Callback argument lifetime, and why the verifier cannot see it

The answered thread is the best example this window of a class the verifier has no vocabulary for:
a value whose type is known, whose storage is somebody else's stack, and whose valid range is one
callback invocation. The useful part for anyone writing callback code is the second half of the
series, which says explicitly which cases are *not* affected. A hash key pointer points into the
stored element and outlives the callback, a map value pointer points into the map and outlives it,
and a callback that copies bytes rather than keeping the pointer is unaffected. That gives a
concrete rule of thumb when reading your own program: a pointer received in a callback is
callback-local if and only if the helper had to build the object to call you. The other half of
the thread is the interaction with a pending verifier rewrite, where state equality becomes a join
routine and the callback frame identity has to be taught to it; the two sets now have to be respun
as one, which is a reminder that a lifetime property is only as good as every place that merges
states.

### Scalar evolution moved into the tree, and the tradeoff is stated in instructions

A twelve-patch verifier series was applied to the bpf-next tree in this window. It targets a cost
users feel rather than a correctness bug: stack liveness loses the offset of a pointer that is
passed to a helper, the whole frame stays live, and spilled pointers nothing loads again keep
states apart. The cover quantifies the two production programs it was measured on, a Falco receive
path and a Tracee security hook, at roughly a sevenfold and twofold instruction reduction, and one
patch is there to stop a spilled pointer being retained for a read that only sees bytes. The part to
carry forward is the number rather than the mechanism: this is a verifier representation change
with an instruction-count payoff, so the diagnostic when it misbehaves is a load that gets slower
or hits a complexity limit, not a wrong answer at runtime.

### Per-task recursion for preemptible programs, and a lost-event class that predates it

A five-patch revision argues that the recursion check for tracing programs is per CPU while the
programs themselves are preemptible, so a task preempted inside a program makes the next task on
that CPU look like a recursive call and its event is dropped and counted as a miss. The risk is
concrete for paired entry and exit programs: a dropped half desynchronises stateful parsing. The
series reports a measured drop rate on a stock kernel under HTTP load and links the change from
voluntary to lazy and full preemption to why it started mattering. The diagnostic to remember is
the miss counter rising without nesting in your own program, and the boundary is that the check
runs before the first instruction, so no program can work around it. A separate patch in this
window makes the same family of counters CPU-aware for per-CPU helpers and map callbacks, after a
maintainer reply pointed out that all three callback paths already validate the CPU and that one
of the added checks was dead code.

### Iterators, drained state, and the sentinel that keeps the ABI

A one-patch fix for a data-buffer iterator makes the drained state a distinct sentinel instead of
the null pointer, because the underlying list walk also returns null at the end, and the verifier's
iterator contract requires that once a next method returns null it keeps returning null. A
maintainer reply suggested the simpler form the neighbouring iterator already uses, and a follow-up
message added a selftest. The reason the state cannot simply grow is worth keeping: the iterator
state struct is part of the generated kernel headers, so its size is an ABI, and the sentinel is
the cheapest way to keep it one pointer wide.

### Small fixes with large blast radii

A handful of patches this window are ordinary bug fixes whose boundaries are worth noting. A push
into a full message ring read the free slot's length, which is zero, so nothing was copied and the
ring looked empty; it was applied to the bpf tree, and it needs both the BPF and network-admin
capabilities, which is why the reproducers ran with file capabilities. A map free path asserted its
entries were empty in a loop bounded only by the configured maximum, which on a large map ran long
enough under a deferred work item to trip a soft-lockup watchdog; the fix adds a scheduling point.
A cgroup link update could write a replacement program into the wrong effective-program slot after
an out-of-memory detach left a dummy entry in place, which is a lifetime bug: the array holds no
reference, so the old program could be freed while still published. And a socket-map receive path
that slept while holding the socket lock could deadlock with the delivery worker and stop all
receive traffic on that worker; the revision releases the lock around the wait and routes the
non-waiting states, zero-length reads, and out-of-band reads back through the unlocked helper.