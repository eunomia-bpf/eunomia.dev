# Why must releasing a BPF dynptr invalidate every derived slice and clone?

**Short answer:** because a dynptr is a verifier-tracked view of backing memory, not an owning buffer that can be copied freely. A slice returned by `bpf_dynptr_slice()` or `bpf_dynptr_slice_rdwr()` aliases that same memory, and a clone represents the same underlying lifetime through another stack object. Once a release operation such as `bpf_ringbuf_submit_dynptr()` or `bpf_ringbuf_discard_dynptr()` ends that lifetime, every reachable alias must become unusable at the same verifier state transition.

Invalidating only the dynptr passed to the release helper is therefore insufficient. A stale slice would still be a verifier-approved `PTR_TO_MEM`, and a clone in another BPF call frame could still appear live. Either gap can turn a legal-looking load, store, or second release into a use-after-free. The safe rule is transitive: revoke the released dynptr, all of its clones, and every slice derived from any member of that lifetime group.

## The lifetime belongs to the backing object, not the C variable

Consider a ring-buffer reservation:

```c
struct bpf_dynptr ptr;
struct event *event;

if (bpf_ringbuf_reserve_dynptr(&events, sizeof(*event), 0, &ptr))
    return 0;

event = bpf_dynptr_slice_rdwr(&ptr, 0, NULL, sizeof(*event));
if (!event) {
    bpf_ringbuf_discard_dynptr(&ptr, 0);
    return 0;
}

event->kind = 1;
bpf_ringbuf_submit_dynptr(&ptr, 0);
/* event and ptr are both dead here. */
```

Before submission, `event` is not an independent allocation. It is an alias into the reserved record. Submission publishes the record to the consumer and ends the BPF program's ownership of the reservation. Discard also ends that ownership, even though the consumer will skip the record. Allowing `event->kind` to be read or written afterwards would let the program access memory whose lifetime has already changed.

The same reasoning applies when a dynptr is cloned. A clone provides another view with its own verifier identity, but it does not create another backing object. Releasing one referenced dynptr ends the shared lifetime, so the original, every clone, and their slices must all be invalidated together.

## One logical lifetime can have several verifier identities

The bug discussed for stable kernels came from treating two kinds of identity as if they were interchangeable.

In the affected stable verifier representation:

- a dynptr stack object has an ID;
- a slice is a `PTR_TO_MEM` register carrying the source `dynptr_id`;
- referenced objects also use `ref_obj_id` to group values that share an acquired lifetime; and
- clones may occupy dynptr stack slots in any active BPF call frame.

`release_reference(ref_obj_id)` can invalidate registers associated with the referenced lifetime, but slices returned by `bpf_dynptr_slice()` and `bpf_dynptr_slice_rdwr()` carry `dynptr_id` while leaving `ref_obj_id` unset. A release path that only walks `ref_obj_id` therefore misses those slices. By contrast, some `bpf_dynptr_data()` paths already carry the reference ID, which is why a test that exercises only that API can conceal the missing `dynptr_id` invalidation.

The stable fix adds an explicit register walk for `PTR_TO_MEM` values whose `dynptr_id` matches the dynptr being released. Checking the register's base type first matters because the metadata field shares storage with metadata used by other register types. The check must still accept the dynptr-specific type flags rather than require one exact type value.

## Clone invalidation must cross BPF call frames

A second gap appears when a subprogram handles a clone while the original or another clone lives in a different frame. Scanning only the current `bpf_func_state` is not enough:

```text
caller frame:  original dynptr + derived slice
                    |
                    +---- shared lifetime ----+
                                              |
callee frame:                              clone
                                              |
                                           release
```

After the callee releases the referenced dynptr, the caller must not resume with a verifier-valid original or slice. The invalidation pass therefore has to inspect the dynptr stack slots in every active frame, not just the frame that executed the helper.

That scan also needs a structural guard. A normal spilled register may have partially overwritten stack bytes that resemble old dynptr metadata. Reading those bytes as a live dynptr can reject a valid program or corrupt verifier bookkeeping. The stable repair first confirms that the slot is actually marked `STACK_DYNPTR`, then reads its dynptr fields.

## A verifier rejection must remain an ordinary error

There is a related control-flow edge case: a callback can try to release a referenced dynptr owned by its caller. That operation should be rejected because the callback cannot consume a reference outside the lifetime context it owns.

The release function can already return an error for that case. Converting the error into `WARN_ON_ONCE()` is wrong for two reasons: it hides the useful verifier rejection, and a system configured with `panic_on_warn=1` can turn an invalid BPF program into a kernel panic during verification. The stable series instead propagates the error to the normal verifier failure path.

This distinction is operationally important. A rejected program is untrusted input handled as designed; a kernel warning says an internal invariant failed. Verifier code must not escalate a foreseeable program error into an invariant failure.

## Mainline and stable kernels can enforce the same rule differently

Mainline commit `308c7a0ae885` replaced the older relationship bookkeeping with explicit parent-child tracking. Each object has an identity, while `parent_id` connects derived objects to the object that governs their lifetime. Releasing a reference walks the object tree and invalidates descendants. Referenced dynptrs use an intermediate lifetime anchor so clones share the release boundary while retaining distinct identities for their own slices.

That is a broad verifier refactor spanning multiple files. A stable branch that still uses `dynptr_id` and `ref_obj_id` cannot safely cherry-pick only fragments of it. The current stable proposal instead implements equivalent safety properties using the old representation:

1. invalidate slice registers by matching `dynptr_id`;
2. return referenced-dynptr release errors through the verifier's normal error path; and
3. scan verified dynptr stack slots across every active call frame to invalidate clones.

The implementation is different, but the contract is the same: no verifier-visible descendant may outlive the object that authorizes access to its backing memory.

## How to write dynptr code that respects the boundary

For BPF program authors, use a lexical ownership pattern:

1. reserve or construct the dynptr;
2. derive slices only inside the block that owns it;
3. check every slice-returning call for `NULL`;
4. submit, discard, or otherwise release exactly once on every path; and
5. do not read, write, return, or pass any dynptr-derived pointer after that release.

Avoid storing a slice in state that survives a callback or subprogram boundary unless the API explicitly proves that lifetime. A clone may help pass a dynptr through structured code, but it is not an ownership split: releasing the referenced lifetime invalidates all aliases.

For kernel and verifier backports, a useful regression matrix is:

| Case | Expected verifier result |
| --- | --- |
| Use a `bpf_dynptr_slice()` result after submit | Reject |
| Use a `bpf_dynptr_slice_rdwr()` result after discard | Reject |
| Release a clone in a callee, then use the original in the caller | Reject |
| Release a caller-owned referenced dynptr from a callback | Reject without warning |
| Scan a partially overwritten ordinary spill with stale-looking metadata | Accept the valid control program |
| Use a live dynptr and slice before the single release | Accept |

Both negative and positive controls matter. The negative tests prove that stale aliases are closed; the positive tests prove that the invalidation walk is not confusing unrelated registers or stack slots with descendants.

## References

- [Linux mainline commit: refactor verifier object relationships and fix dynptr use-after-free](https://github.com/torvalds/linux/commit/308c7a0ae8859b34d9d90a3dff953b2d14242145)
- [Linux kernel documentation: BPF ring-buffer reservation, commit, discard, and verifier reference tracking](https://docs.kernel.org/bpf/ringbuf.html)
- [Linux BPF selftests for dynptr invalidation and failure paths](https://github.com/torvalds/linux/blob/308c7a0ae8859b34d9d90a3dff953b2d14242145/tools/testing/selftests/bpf/progs/dynptr_fail.c)
- [Linux verifier source at the mainline lifetime-tracking fix](https://github.com/torvalds/linux/blob/308c7a0ae8859b34d9d90a3dff953b2d14242145/kernel/bpf/verifier.c)
- [Linux 6.12.y verifier source: stable-branch dynptr and reference representation](https://github.com/gregkh/linux/blob/linux-6.12.y/kernel/bpf/verifier.c)
- [Linux 6.12.y dynptr failure selftests](https://github.com/gregkh/linux/blob/linux-6.12.y/tools/testing/selftests/bpf/progs/dynptr_fail.c)
- [Linux BPF UAPI: dynptr ring-buffer helper definitions](https://github.com/torvalds/linux/blob/master/include/uapi/linux/bpf.h)

## Community discussion today

Today's ordinary visible-browser review covered all 6 approved communities and all 15 allowlisted channels or public pages. Every target was accessible. The selected question came from the strict 24-hour window, so the seven-day fallback was not used. Names, accounts, employers, workspace and channel identities, message links, exact times, private topology, raw logs, and searchable wording have been removed. No raw transcript was retained, and no social interaction was performed.

### Lifetime tracking was the clearest correctness concern

The strongest technical discussion concerned stable-kernel fixes for dynptr release. Review moved beyond the obvious stale slice and tested the full alias closure: read-only and writable slices, clones in another BPF call frame, callbacks that attempt an invalid release, and ordinary spill slots that retain stale-looking metadata. The important engineering lesson is that a backport should reproduce the mainline safety property, not mechanically transplant code whose object model does not exist on the target branch.

### Review dependencies dominated instrumentation chat

The current instrumentation discussion was mostly coordination around prerequisite patches and follow-up work rather than a new user question. Replies continued inside the daily window, but the visible request was for review and merge ordering. That signals concern about keeping dependent schema and runtime changes small enough to review in sequence; it did not justify inventing a separate troubleshooting question.

### Public kernel work concentrated on invalidation and verifier boundaries

Beyond dynptrs, the public archive showed active work on use-after-free prevention in BPF test execution, concurrent lifetime handling, arena pointers returned by value, JIT memory checking, bounded map iteration, and safer BTF dumping. These topics share a pattern: an object crosses a subsystem boundary, and the implementation must preserve ownership, type, and failure information across that boundary. Focused selftests were repeatedly used to distinguish a real stale reference from an overly broad verifier rejection.

### Several user-facing surfaces were quiet

The project-specific help and feature areas, scheduler support surfaces, the general eBPF chat, and the public forum had no new technical question in the strict daily window. One general chat had a new member introduction, while project feeds showed automated repository activity rather than human support requests. Those targets were still checked and counted as accessible; quiet or automated activity was not repackaged as community demand.
