# Why does a valid BPF program now fail to load with ENOSPC from a 64 KB verifier log, and why does the level 1 log no longer show the path to the error?

Because the level 1 verifier log stopped being a path and became an accumulation of
every basic block the verifier visited, and the kernel decides `-ENOSPC` from the
largest the log ever got, not from what is left in the buffer at the end. On a
kernel where verification follows the control flow graph and joins the states that
arrive at a block, a program that verifies successfully still fails the load if the
accumulated text ever exceeded the caller's buffer, because the finalize step runs
after the verdict and its `-ENOSPC` overwrites it. A follow-up series, applied to
`bpf-next`, keeps one basic block at a time in the level 1 log again and prints the
way to a rejected block in front of it.

## The mechanism

1. **What the log used to be.** At level 1 the verifier printed each instruction as
   it verified it. At a conditional jump `push_stack()` put the other branch aside
   and recorded the current end of the log in the pending entry, and `pop_stack()`
   called `bpf_vlog_reset()` back to that recorded position when it resumed the
   branch. That pair is what made the log a path: no longer than the longest path
   through the program, and, for a rejected program, exactly the path that reached
   the failing instruction. Both statements hold in the current mainline sources.

2. **What it is after the join-states merge.** Once the verifier verifies basic
   blocks in the order of the control flow graph and joins the states arriving at a
   block, there are no paths to return to. `pop_stack()` is gone from the verifier,
   nothing records a log position per branch, and nothing moves the log back while a
   function is verified. Each block appends its text every time it is verified, so
   the log holds every block, both arms of each branch, in verification order. For a
   large real program the difference is not a percentage: the series reports
   51,434 bytes for one path against 774,446 for all blocks on one Cilium program,
   and 103,567 against 3,668,559 on one Katran program.

3. **Why a valid program then fails.** The text is dropped when the program turns out
   to be safe — the log is reset after a successful pass. That does not help, because
   finalization compares a running maximum, not the current length: every write
   updates `len_max` to the largest total the log has ever had, and finalization
   returns `-ENOSPC` when the buffer exists and that maximum exceeded the buffer
   size. The counter exists precisely so that a caller can learn the true size of a
   truncated log, and it is the same number reported back as the true size.

4. **Why that becomes a load failure rather than a warning.** The finalize call sits
   on the load path after the runtime has been selected, and its return value is
   assigned over the verification result: if finalization fails, the error replaces
   whatever the verifier concluded. So a program the verifier accepted is reported to
   the caller as a failed load, with `ENOSPC` as the errno. The reported count is
   large: 2,841 programs of Cilium, the selftests, the schedulers and a few Rust
   crates crossed a 64 KB buffer before the change, 5,503 after it, and 45 with the
   follow-up series.

5. **Why the buffer cannot simply be raised.** `bpf(2)` shows the pattern that
   breaks: a static buffer, `log_level` one, one call, no retry. A caller that owns
   the buffer gets no second chance from the kernel, and the documented contract
   only says the call fails with `ENOSPC` when the buffer is too small — it does not
   distinguish "program rejected" from "program accepted, log truncated". libbpf
   does distinguish it for the buffers it owns: on `ENOSPC` it doubles its own
   buffer and retries, up to half of the 32-bit range, so the regression landed on
   callers that pass their own fixed buffer.

6. **How the follow-up series fixes the log.** Two changes. The first adds a pair of
   helpers that take the tail of the log out into a temporary buffer and append it
   later, counting the bytes the buffer could not hold so that the reported size and
   the truncation decision describe a log written in its final order. The allocation
   is made to fail rather than trigger the OOM killer, and if it fails the log is
   left as it is. The second moves the log back in front of each basic block, so the
   level 1 log again holds one block at a time, and when a block is rejected it
   prints the way to that block before the block and the message: the block is
   stashed, the way is printed, the block is appended. Kept states carry the edge
   they arrived by — after a join, the edge that changed them last — so the way is
   reconstructed from state, without verifying anything twice. Two 32-bit stores per
   state are the cost on the success path, and the longest way in the corpus takes
   about 0.4 ms.

7. **What the level 2 log keeps.** Nothing changes at level 2, which is why six
   selftests that read a state behind an instruction of the way were changed to
   request level 2. Level 1 does not show the states behind the instructions of the
   way, and does not show the body of a subprogram verified on its own; level 2 has
   both.

## How to verify and debug

1. **Separate the two failures by rerunning without a log.** Load the same program
   with `log_level` zero and take the log out of the picture. If it loads, the
   program was accepted and the `ENOSPC` came from the log, not from verification.
   This is the check that distinguishes the regression from a genuinely overlarge
   program.

2. **Read the reported true size.** The program-load attribute carries an output
   field for the true log size, documented as possibly larger than the buffer when
   the log was truncated and smaller when it was not filled. Read it before and after
   a change rather than guessing from the buffer length: it is the only number that
   says how far over the buffer the log went.

3. **Check which buffer you are passing.** A buffer libbpf allocated itself is grown
   and retried on `ENOSPC`. A buffer supplied through the load options, or a static
   array as in the `bpf(2)` example, is not: libbpf marks it as not owned and
   returns the error after post-processing the text. If a load that used to work now
   fails with `ENOSPC` and your loader passes a fixed buffer, that is the surface.

4. **Read the tail of the log after the fix.** On a kernel with the follow-up
   series, a rejected program's log ends with the failing block and its message, and
   the way to it appears in front, reconstructed from the kept state edges. Tooling
   that matches on line positions must expect the reordering; the text of a block is
   the same as before.

5. **Measure the peak, not the end.** If you are instrumenting the verifier, the
   number to watch is the running maximum the finalization compares against, and the
   byte count of the buffer at exit is a different and much smaller number on a
   successful load.

## The limitation

- The regression and its fix are in the `bpf-next` tree, where verification already
  follows the control flow graph. Current mainline still walks paths with the
  push/pop log positions, so a kernel from mainline behaves as documented in `bpf(2)`.
  The numbers above describe the joined-states tree and the series applied to it.
- The fix restores the size and the shape of the level 1 log; it does not restore
  the exact old text. A block is on the way once, so the way goes around a loop
  once and continues from the loop's dominator, and a function verified inside its
  caller's frames appears once, for its last call. A log consumer that expects the
  pre-join output to be reproduced byte for byte will still see differences.
- A dropped round of verification produces no way at all, and a pending signal ends
  the printing, so the way can be shorter than the full explanation.
- The stash is bounded by the buffer, which may be as large as the kernel accepts,
  and the allocation can fail; the log is then left in place rather than failing the
  load, which means the reordering is best-effort under memory pressure.
- The corpus numbers are the author's, over a fixed set of programs with a fixed
  buffer size; read them as an order of magnitude, not a guarantee for a specific
  binary.

## References

- [Patch series: one basic block and the way to it in the level 1 log](https://lore.kernel.org/bpf/20261108053937.4134138-1-alexei.starovoitov@gmail.com/) — the cover letter with the path-versus-all-blocks table, the buffer counts, and the two changes.
- [Patch: take the end of the verifier log out and append it later](https://lore.kernel.org/bpf/20261108053937.4134138-2-alexei.starovoitov@gmail.com/) — the stash and unstash helpers, the counted lost bytes, and the allocation that is allowed to fail.
- [Patch: keep one basic block in the level 1 log and print the way to it](https://lore.kernel.org/bpf/20261108053937.4134138-3-alexei.starovoitov@gmail.com/) — the log reset per block, the way reconstructed from state edges, and what level 1 stops showing.
- [Kernel `kernel/bpf/log.c`](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/log.c) — the running maximum, the truncation test that returns `-ENOSPC`, and the fixed versus rolling buffer write paths.
- [Kernel `kernel/bpf/verifier.c`](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/verifier.c) — the pending-branch log position recorded at the push and restored at the pop, and the finalization call whose error replaces the verification result.
- [`bpf-next` `kernel/bpf/log.c`](https://git.kernel.org/pub/scm/linux/kernel/git/bpf/bpf-next.git/plain/kernel/bpf/log.c) — the applied stash implementation with the counted start and end bytes and the allocation flags.
- [`bpf-next` `kernel/bpf/flow.c`](https://git.kernel.org/pub/scm/linux/kernel/git/bpf/bpf-next.git/plain/kernel/bpf/flow.c) — the applied way printer: kept-state lookup, dominator walk, stash, print, unstash.
- [`bpf(2)` man page](https://git.kernel.org/pub/scm/docs/man-pages/man-pages.git/plain/man/man2/bpf.2) — the documented load pattern with a static buffer and `log_level` one, and the statement that an undersized buffer fails with `ENOSPC`.
- [uapi `linux/bpf.h`](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/include/uapi/linux/bpf.h) — the true-size output field and its documented relation to the buffer size.

## Community discussion today

This run reviewed six allowlisted communities. Two are Slack workspaces reached
through the approved private read-only archive, covering three allowlisted
channels; all three returned no message in the 24-hour window and none in the
seven-day window either, so the archive contributed no theme. The two chat
workspaces with ten allowlisted channels in total were not covered: this
environment has no authenticated session and the public invite pages answer with
an invalid-invite page, so they are recorded as inaccessible, not quiet. The
practitioner forum returned a network-security block page to the visible browser
throughout the run and is also recorded as inaccessible. The kernel mailing list
was read through its public archive pages, and the question above plus the themes
below come from it.

The dominant theme is what a verifier change does to the things around it rather
than to the verdict. The level 1 log is the clearest case: the verification
result stayed correct while the diagnostic channel silently acquired a new size
class and lost its central property, and the failure surfaced to callers as a
load error on a program the verifier had accepted. Its fix had to keep the
reversed order of a truncated rolling buffer, the true-size report, and the
truncation decision consistent while reordering the text — which is why the
series counts the bytes the buffer could not hold instead of reconstructing the
log. Two sibling reports on the same tree show the same shape: a counter that
wraps just in front of the jump that counts a loop gets a residue computed from
the unwrapped value, so the verifier checks numbers the register never holds, and
the fix declines to set a range when the sum can wrap; and a program that throws
one of several error codes is rejected because the join merges the throw cookie
with the flag, which the fix addresses in the argument checker rather than in the
join.

The second theme is a check that holds only for the inputs its author had in
mind. `btf_struct_access()` rejects direct access to special fields of a
program-allocated object, but for a variable offset it tests only the minimum
offset. Compiler-generated BTF makes that sufficient: the verifier keeps the
access inside one array with a step that is a multiple of the element size, and
the special fields are repeated per element, so every reachable offset selects
the same member. Hand-written BTF with overlapping members breaks the reasoning —
an array member ending past the offset of a kptr pointer means a store to a
well-formed array index overwrites the pointer, and the object's release path
frees the corrupted value. The fix tests only the reachable offsets, the minimum
plus a multiple of the step; a reviewer's smaller version rests on the invariant
that the step is never zero, which the register sanity check already enforces, so
a constant offset is the first case of the same check. The boundary the patch
must not cross is stated by its own test: a store that stays on a non-special
member of every element must keep loading.

The third theme is map and iterator permissions, where two notions of read-only
had been conflated. An iterator's value pointer is writable for all non-per-CPU
map types, which lets a program write into a map created read-only for programs
and corrupt a map whose contents the verifier treats as constants. The proposed
predicate tested both the program-read-only flag and the frozen bit; the
correction is that freezing stops writes from the system-call side only, programs
may still write into a frozen map, so the flag alone is the right test — and
because that flag never changes, there is no race with the freeze call to
consider. The same review trimmed the fix tags, dropped a commit that would have
rejected every map, and asked for a selftest that writes through the iterator's
value pointer on a map created read-only for programs, since the earlier
reproducer passed on a fixed kernel.

A fourth theme concerns what a batch traversal reports when the map changes under
it. A hash map batch lookup can lose its cursor when a key is deleted during the
walk and then report end-of-map, which tells a caller its walk finished when keys
remain. The patch reports the loss as a retryable condition instead, so a walker
resumes rather than concluding, and adds a regression test around deletion during
the traversal. The practical reading for a userspace dumper is that a short batch
is not proof of an empty tail.

A fifth theme is lock ownership in a BPF-replaced receive path. A transport's
receive hook that the sockmap redirects through takes the socket lock and then
sleeps waiting for data, while the transport's delivery worker needs that same
lock to queue the data — a deadlock between the reader and the producer it is
waiting for. The fix releases the lock around the wait and adopts the native
path's exit conditions, and handles separately the reads that should never enter
the sockmap wait: a zero-length read, the error queue, and out-of-band data. The
implementation in the current tree shows the lock taken before the wait loop and
released only at the end, which is what makes the report credible; the author
also lists what is still unfixed, notably data sitting only in the socket's
ingress queue and the lockless reads that lack explicit accessors.

Two smaller signals. A 24-patch arm64 JIT series measures itself by rebuilding
the BPF objects of several real projects and reporting size and modelled cycles
per microarchitecture, which is the kind of evidence a JIT change needs and rarely
gets. And an attachment check now refuses to attach to a subprogram that is
called recursively, because the load-time stack-depth check can see the recursion
but not what gets attached to it later — an example of an invariant that only
holds if both ends of it are enforced at the same point in time.