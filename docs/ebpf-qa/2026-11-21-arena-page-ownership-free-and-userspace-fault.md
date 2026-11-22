# Why does a freed BPF arena page stay mapped until the arena is told, and what decides whether user space gets a page or a SIGBUS?

**Short answer:** an arena is not a heap with one owner. A page in it has three
independent parties that can put it there or take it away, and the kernel keeps
one range bitmap to remember which slots are free. BPF side allocation
(`bpf_arena_alloc_pages`) and user-space fault-in (`arena_vm_fault`) both take a
slot out of that bitmap and install a page; `bpf_arena_free_pages` is the only
call that walks back, and it must clear the vmalloc PTE, flush the kernel TLB,
zap the page out of every user VMA that mapped it, and only then put the range
back in the bitmap. So a page that user space faulted in stays resident and
counted in `memlock` until either a BPF program frees that range or the arena is
destroyed. What user space gets on a missing page is decided by one map flag:
without `BPF_F_SEGV_ON_FAULT` the fault installs a fresh zero page, and with it
the fault returns a signal instead of allocating.

## The mechanism

### 1. The bitmap records free slots, and both producers clear it

The arena's range tree is a bitmap of *unallocated* slots. At map creation the
whole range is set, and each allocation clears the slots it takes:

```c
range_tree_init(&arena->rt);
err = range_tree_set(&arena->rt, 0, attr->max_entries);
```

The polarity is visible in the helper names: `is_range_tree_set()` returns 0
when the whole requested range is still free, and `range_tree_clear()` marks a
range as taken. `arena_adjust_tree()` is the single place that decides which
slots a BPF-side allocation consumes — the caller's offset if one was given,
otherwise `range_tree_find()` for a best-fit free run:

```c
if (uaddr) {
        ret = is_range_tree_set(&arena->rt, *pgoff, page_cnt);
        if (ret)
                return ret;
        return range_tree_clear(&arena->rt, *pgoff, page_cnt);
}
ret = range_tree_find(&arena->rt, page_cnt);
```

The user-space fault path is the second producer, and it takes one slot at a
time under the same spinlock, clearing the range-tree bit before it sets the
PTE. That is why the two producers cannot hand out the same page: the bitmap
entry is the arbiter, not a lock on the page.

A reservation is the third state. `bpf_arena_reserve_pages()` clears the bitmap
without installing anything, which is how a producer tells the other side "do
not put a page here"; a slot reserved this way fails a later
`bpf_arena_alloc_pages()` at that address and makes the user-space fault take
the error path instead of allocating.

### 2. Free is four steps, and the order is the whole point

`arena_free_range()` does not free a page when the caller says so. It walks
the range in this order:

1. `apply_range_clear_cb()` clears the kernel PTE and collects the `struct page`
   for each slot, skipping the arena's own scratch page.
2. It waits for any in-flight `arena_vm_fault()` that already found the page —
   `lock_page()`/`unlock_page()` on each collected page — before the PTE can be
   reused.
3. It flushes the kernel TLB with `flush_tlb_kernel_range()`.
4. Only then does it `zap_pages()` every user VMA that mapped the range, and
   `__free_page()` the pages. The bitmap entry is restored at the very end.

The comment in the source states the contract the order protects: "The range
stays allocated in the range tree until the pages are unmapped from vmalloc area
and from all user space vma-s, so that nobody can populate it while stale TLB
entries or user ptes are still around." Reversing any two of those steps gives a
window where a slot is free in the bitmap but still reachable through a stale
mapping — exactly the double-allocation the bitmap is meant to prevent.

The single-page case is optimised, not different: `arena_free_range()` calls
`zap_pages()` inline for one page when its refcount shows a user mapping, and
otherwise skips the walk.

### 3. Two free entry points, one deferred

`bpf_arena_free_pages()` is a kfunc, so it can be called from a program that
cannot sleep. The free path is long — it takes a spinlock, walks page tables,
flushes TLB and takes `mmap_read_lock()` in `zap_pages()` — so the non-sleepable
form defers: it allocates a small `arena_free_span` record, appends it to a
lockless list and queues `irq_work`, which schedules a work item that runs
`arena_free_range()` in process context. The sleepable form runs it in place.

That deferral has an honest cost, documented in the source: if the small record
cannot be allocated from a non-sleepable context, the pages are "intentionally
left inaccessible (leaked) until the arena is destroyed". A program that frees
from an atomic or spinlock-held context under memory pressure can therefore
observe the free as a no-op for the rest of its run.

Note also what `bpf_arena_free_pages()` does with its address argument: it keeps
only the low 32 bits, masks to a page boundary, and clamps the range to the
registered user VMA. A free of an address outside the VMA, or of a range that
starts before it, is silently truncated rather than rejected.

### 4. What user space sees is one map flag

`arena_vm_fault()` has three outcomes, and which one applies is decided by
`BPF_F_SEGV_ON_FAULT` and the bitmap, not by the address alone.

- A page already present (allocated by BPF, or already faulted in) is handed
  back locked.
- If the slot is free in the bitmap and the flag is absent, a new zeroed page is
  allocated, installed in both the kernel and the user mapping, and the
  allocation is charged to the map's memcg and counted in `nr_pages`. This is
  the lazy path: user space populating a page is an allocation, indistinguishable
  from a BPF-side one in `memlock`.
- If the slot is *reserved* in the bitmap (`is_range_tree_set()` true for it),
  the fault returns `VM_FAULT_SIGBUS` — the range exists but was claimed
  without a page.
- With `BPF_F_SEGV_ON_FAULT` set and no page present, the fault returns
  `VM_FAULT_SIGSEGV` and allocates nothing.

The fault handler can also fail closed for reasons that have nothing to do with
the address: a failed spinlock acquisition returns `SIGBUS` rather than
retrying, because the handler never held `mmap_lock` and a retry would re-take
it.

### 5. The BPF side's own missing page is a scratch page, not a fault

A BPF program that dereferences an arena address whose page was never allocated
does not take a page fault in the usual sense. `bpf_arena_handle_page_fault()`
finds the running program from the stack, checks the address is inside the
4 GiB band plus upper guard, installs the arena's scratch page into the empty
PTE, and reports a violation to the program's stderr stream:

```c
bpf_stream_printk(ss, "ERROR: Arena %s access at unmapped address 0x%lx\n",
                  write ? "WRITE" : "READ", addr);
```

The access is allowed to complete against the scratch page. That is a
diagnostic, not a correctness mechanism: two different bad offsets can write
the same scratch page, and the value read is not the program's data. The
scratch page is never freed by a range free (both clear paths skip it) and is
not counted in `memlock`.

Because the scratch page occupies the slot's PTE, the real allocator has to
break it: `apply_range_set_cb()` retries the install in a loop, and on
architectures without an atomic PTE installer it expects a `none` slot. The
source notes that the architectures which lack `ptep_try_set()` also do not wire
up `bpf_arena_handle_page_fault()`, so they never have a scratch page to break.

## Verifying and debugging this on a real kernel

1. Read `memlock` before and after, not the map's RSS. The number the kernel
   charges is `arena->nr_pages << PAGE_SHIFT`, surfaced through
   `/proc/<pid>/fdinfo/<map_fd>`. A freshly created arena reports 0 — the
   scratch page is excluded — and the selftest asserts exactly that sequence:
   0, then `17 * PAGE_SIZE` after a 17-page BPF allocation, then
   `16 * PAGE_SIZE` after freeing one page from the start, then 6, then 0, then
   `8 * PAGE_SIZE` after user space touches eight pages, then 0 after BPF frees
   them.

2. Distinguish the three fault outcomes with one program pair. Touch an
   unallocated page and watch `memlock` grow by a page: that is the lazy path.
   Reserve the same range with `bpf_arena_reserve_pages()` and touch it again:
   the process gets `SIGBUS` and `memlock` does not move. Create the map with
   `BPF_F_SEGV_ON_FAULT` and touch a fresh page: `SIGSEGV`. Any of the three
   that produces a different one tells you which branch the map flags took.

3. Confirm a free really completed, rather than becoming a deferred record.
   `bpf_arena_free_pages()` from a sleepable program (a `syscall` program is the
   easiest harness) returns only after `zap_pages()` ran; from a non-sleepable
   one it returns as soon as the span is queued, or even before that if the
   record could not be allocated. Check `memlock`, then check it again after
   the workqueue has run.

4. Confirm the 32-bit offset contract. BPF code addresses arena memory as
   `kern_vm_start + lower32(user_pointer)`, and the free path keeps only the
   low 32 bits after masking to a page boundary. An arena whose user VMA
   crosses a 4 GiB boundary is refused at mmap time for that reason:
   `map_extra` must keep the whole range inside one 32-bit window.

5. Expect a load-time rejection, not a runtime one, when the license is wrong.
   A program that is not GPL compatible cannot call any kernel function at all;
   the verifier rejects it with "cannot call kernel function from non-GPL
   compatible program" while the calls are being resolved, so the kfunc never
   runs and no arena page is touched.

## The limits

The bitmap arbitrates slots, not users. Two BPF programs can hold pointers into
the same arena, and nothing in the free path checks that the caller still owns
the range: a free of a range another program still uses unmaps it under that
program, and the next access is a scratch-page violation. Ownership is a
convention between the programs that share the arena, not an enforced property.

A free with a page count that runs past the VMA is clamped, and a free whose
start address is outside the VMA returns silently. That is forgiving in use and
unhelpful in debugging: a wrong pointer computed from a stale offset produces no
error, only a smaller free than intended, and the difference shows up later as
an allocation that lands where the caller did not expect.

The scratch page also means that a wrong arena pointer is not reliably fatal. A
program that writes through an unallocated offset corrupts no real data, but it
also gets no error at the access site — only an stderr line — so a program that
ignores its own log will keep running with the violation unrecorded in its
results.

Finally, the mechanism is bounded by the 4 GiB addressable band and by the fact
that the user VMA is chosen once, on the first `mmap()`. A second process must
map at the same address with `MAP_FIXED`, and the arena refuses a VMA of a
different size, so multi-process sharing is a deployment-time decision rather
than something the map can negotiate later.

## References

- [BPF arena implementation, kernel source](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/arena.c) — the range bitmap, `arena_adjust_tree()`, `arena_free_range()`, `zap_pages()`, the deferred free, and the scratch-page violation path.
- [Range tree, kernel source](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/range_tree.c) — the header comment stating that the tree records *unallocated* arena slots, and the split/merge rules for `range_tree_set` and `range_tree_clear`.
- [Arena kfunc declarations, kernel source](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/helpers.c) — `bpf_arena_alloc_pages`, `bpf_arena_free_pages` and `bpf_arena_reserve_pages`, with their deferral to the non-sleepable form.
- [Arena map types and flags, uapi header](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/include/uapi/linux/bpf.h) — `BPF_MAP_TYPE_ARENA`, `BPF_F_SEGV_ON_FAULT`, `BPF_F_MMAPABLE`, and the `map_extra` address contract.
- [Arena memory accounting selftest](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/tools/testing/selftests/bpf/prog_tests/arena_mem_usage.c) — the exact `memlock` sequence across BPF allocation, single and bulk frees, user fault-in, and freeing faulted-in pages.
- [Arena allocation and free test programs](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/tools/testing/selftests/bpf/progs/arena_mem_usage.c) — the `syscall`-program harness that drives `bpf_arena_alloc_pages()` and `bpf_arena_free_pages()` with a caller-supplied offset and count.
- [Arena mmap contract selftest](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/tools/testing/selftests/bpf/prog_tests/arena_mmap.c) — the refusals for a VMA longer than the map, an address of 0, a range crossing 4 GiB, and a wrapping range.
- [Arena kfunc argument forms selftest](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/tools/testing/selftests/bpf/progs/arena_kfunc.c) — page 0 occupied by an arena global, allocation and free of one page, and the 32-bit-scalar and full-user-address argument forms.

## Community discussion today

Two archive-opted Slack workspaces were reviewed through the approved read-only
archive: three allowlisted channels in total, one message in the daily window.
The public bpf mailing-list archive and r/eBPF were reviewed as visible browser
pages; the mailing list was the substantive source today, and r/eBPF's recent
posts were project announcements rather than questions. The two allowlisted
Discord workspaces, ten channels in total, were not reviewed this run: no
Discord session was opened, so their absence here is a coverage gap, not
silence. The one archive message was a proposal for a binary pattern-search
kfunc; it is a real open question, but this page answers the arena-ownership
question below because that search topic already has a dedicated answer and the
arena discussion is where the day's actual design disagreement was.

**Who owns a shared page, and what happens the moment that owner stops.** The
day's clearest design argument was over sharing arena pages between a host and
its guests. The first attempt exposed a pinned arena as a sized file and gave a
guest its share as a NUMA node; review found the two structural problems the
mechanism above predicts. The node is ordinary guest RAM, so when the guest's
scheduler exits its arena pages go back to the guest's buddy allocator while the
host can still write them; and block granularity means 16 MB of guest memory
carried one shared page, which caps the design at a few hundred guests per host
scheduler. The replacement reverses the ownership question: the arena gains one
primitive that maps pages it did not allocate, each import tracked in an
interval tree owned by the *supplier* rather than the arena, released as a whole
by one call, excluded from arena memory accounting, and faulting with `SIGBUS`
if a guest touches pages after its producer went away. Both designs are worth
reading as a pair because the second's list of properties is exactly the list
of things the free path above has to get right for pages the arena does own.

**"Allocation" means two different things for the same page.** A related and
still unresolved confusion surfaced around the user-space side. For a page that
BPF never touched, a user-space store is an allocation: the fault handler takes
a bitmap slot, allocates a page and counts it. For an arena the program has
declared with `BPF_F_SEGV_ON_FAULT`, the same store is an error signal. The two
behaviours differ only by a map flag, and the failure mode is easy to misread —
a `SIGBUS` from a *reserved* range looks identical to a `SIGBUS` from a range
whose allocation failed inside the fault handler, and the second is a
memory-pressure symptom rather than a design error. Checking `memlock` before
and after is the cheap way to tell them apart: an error signal leaves the number
unchanged, and a lazy allocation moves it by exactly one page.

**Verifier work that makes no claim about program safety.** The largest single
thread in the window was a 49-patch rewrite of the verifier's main analysis,
replacing the path-following core and the scalar-evolution analysis with
equations carried in the state. The interesting part for a reader deciding
whether to adopt it is what its own follow-up series says: all seven fixes in
the follow-up address bugs that "do not make the verifier accept an unsafe
program" — a build break without `CONFIG_NET`, path-sensitive runs inventing a
failing path and rejecting a valid program, a log that shows only the last basic
block instead of the path to the error, and an offloaded program being refused
for a precision reason. The stated numbers matter more than the design: over
about 6000 programs from other projects, mean instructions per program drop from
roughly 18k to 1.7k and mean verification time from about 23 ms to 5 ms, with
peak states falling from a mean of 562 to 18. A companion set of seven fixes
measured 385k programs of production verifier workloads with an unchanged
verdict and unchanged processed-instruction count, which is the number that
makes the rewrite safe to ship rather than the speedup.

**Loops are still the boundary where the two sides disagree.** A related fix
series settled three separate loop-interaction bugs in one pass: a skipped trip
count has to be marked as a guess when the bound moves and when a 32-bit value
wraps, and a path that is only out of frames has to be dropped rather than
counted as a failure. The pattern is the same in all three — the analysis had
been treating a value it could not prove as though it had proved it. That is the
same class of mistake as reading a scratch-page result as data, and it is the
reason the arena page above insists that the bitmap, not the page, is the
arbiter of who owns a slot.