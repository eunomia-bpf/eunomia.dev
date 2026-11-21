# Why can a BPF ring buffer consumer sleep with records already queued, and what ordering makes the wakeup handshake complete?

**Short answer:** the kernel does not notify on "new data". It notifies on
"the consumer is exactly caught up with the record just committed", and that
decision is a load of the consumer position. A userspace consumer does the
mirror image: it publishes its consumer position and then loads the producer
position to decide whether to wait. Both sides can read the other's position
from before its latest store unless a full StoreLoad barrier sits between the
publish and the probe. A release store does not order a later load, so with
the barrier missing the consumer concludes the ring is empty while a record is
already committed, the producer concludes nobody is waiting, and the consumer
sleeps. libbpf's consumer loop has exactly that gap, and the fix in review is
a single full fence between publishing a consumer position and the next
producer-position load, plus one retry of a busy record behind that fence.

## The mechanism

### 1. The notification is edge-triggered on an exact position

`bpf_ringbuf_commit()` clears the record's busy bit and then decides whether
to wake the consumer:

```c
xchg(&hdr->len, new_len);

rec_pos = (void *)hdr - (void *)rb->data;
cons_pos = smp_load_acquire(&rb->consumer_pos) & rb->mask;

if (flags & BPF_RB_FORCE_WAKEUP)
        irq_work_queue(&rb->work);
else if (cons_pos == rec_pos && !(flags & BPF_RB_NO_WAKEUP))
        irq_work_queue(&rb->work);
```

The comparison is equality, not "producer ahead of consumer". Only the record
that sits exactly at the consumer's current position raises a notification,
and only if the consumer has not fallen behind. The kernel documentation calls
this self-pacing: "send a notification ... after commit only if consumer has
already caught up right up to the record being committed. If not, consumer
still has to catch up and thus will see new data anyways without needing an
extra poll notification." That reasoning holds only as long as the consumer
really does see the data on its next pass. The load above is the producer's
half of a handshake, and it can be answered with a stale value.

### 2. The consumer's half is a store followed by a load

The current libbpf consumer loop, in outline:

```c
cons_pos = smp_load_acquire(r->consumer_pos);
do {
        got_new_data = false;
        prod_pos = smp_load_acquire(r->producer_pos);
        while (prod_pos - cons_pos > 0) {
                len_ptr = r->data + (cons_pos & r->mask);
                len = smp_load_acquire(len_ptr);
                if (len & BPF_RINGBUF_BUSY_BIT)
                        goto done;
                /* invoke the callback, then */
                smp_store_release(r->consumer_pos, cons_pos);
                if (cnt >= n)
                        goto done;
        }
} while (got_new_data);
```

Every record is published with a release store to `consumer_pos`, and the very
next thing the loop does is load `producer_pos` again to decide whether there
is more to consume. That store-load pair is the consumer's half of the
handshake: it is the value the producer's equality test reads.

Release stores and acquire loads form a release-acquire pair, and that pair
orders the record data before the position that advertises it. It does not
order a *later load* after the store. The kernel memory-barrier documentation
states it directly: the weak instructions behind `smp_store_release()` and
`smp_load_acquire()` "are not required to order prior stores against
subsequent loads in all cases." A StoreLoad barrier is what orders a store
against a load that follows it.

### 3. Which window actually loses a record

Both stores can be in flight at once. The consumer's release store of the new
consumer position has not become visible to the producer's CPU; the producer's
reservation has advanced the producer position but the consumer's own load of
it still returns the old value. Then:

- the producer commits, loads `consumer_pos`, gets the position from *before*
  the consumer's store, and correctly concludes per its own rule that the
  consumer has not caught up, so it queues no `irq_work`;
- the consumer finishes its pass believing the ring is empty and blocks in
  `epoll_wait()`.

Neither side is wrong about what it read, and nothing remains to wake the
consumer. That is the whole defect: not a lost record, a lost notification.

libbpf registers the map fd level-*triggered* (`e->events = EPOLLIN`), so
`ring_buffer__poll()` re-evaluates readiness when it enters the kernel, which
narrows the window to the stretch between that readiness check and the sleep:
the record must be committed after the check and the notification must be
skipped. A rare window is still a hang, and the ring-buffer selftests in the
series add an edge-triggered readiness case that consumes one record per call
so the handshake is exercised on every bounded return instead of only at
startup.

### 4. The fix is a full fence between publish and probe

Patch 3 of the series converts the loop to publish, fence, then probe:

```c
if (!got_new_data)
        return cnt;

/*
 * Order the published consumer position before the next
 * producer-position load, whether below or in a later invocation.
 */
__atomic_thread_fence(__ATOMIC_SEQ_CST);
```

and, on the early exits, executes the same fence before returning so that the
load in the *next* invocation completes the handshake. It also changes the
busy-bit case from "bail out" to "publish, fence, retry once": a producer can
commit a record that the consumer has just read as busy, and skip the wakeup
because it read the old consumer position; retrying the load behind the fence
closes that variant.

Two details make this more than style. First, the barrier has to be full, not
another acquire. Aya hit the same bug and documented why the pair is not
portable: on x86-64 LLVM emits `xchg` for the store and the ordering is
accidental, on AArch64 `STLR`/`LDAR` give RCsc, but on powerpc64 "LLVM places
`sync` before the store and `isync` after the load, leaving no full barrier
between them." Their fix is the same shape: a release store, then an explicit
sequentially consistent fence before reading the producer position or retrying
a busy record. Second, the fix belongs to the consumer, not the producer. The
producer already has the full barrier it needs in the record-committing
`xchg` before it reads the consumer position.

### 5. Stopping early is the second way to miss a notification

Because the notification is edge-triggered, a consumer that has already been
told "ready" and then stops before draining has been told everything it will
be told. The bound in `ring_buffer__consume_n()`, or a callback that returns a
negative value to abort, both end the invocation with records still queued, and
the producer that committed those records saw a consumer that was not caught
up. The series therefore adds the rule to the two consume functions and to the
callback typedef: "Reaching the requested bound does not establish that the
ring is empty. Records can remain queued without a new readiness
notification. Before waiting on the map fd, call consume until it returns 0."
With the level-triggered registration, `ring_buffer__poll()` itself still
re-reports queued data; a program that waits on a condition fed by the
callbacks, or that registers the map fd as edge-triggered in its own epoll
set, does not.

The same patch also records the positive-return contract. A callback return of
zero or more continues consumption and is otherwise ignored; only a negative
return stops it. An earlier proposal to let a positive return request early
termination was declined because positive values were already ignored, and
changing that would break existing users. The caller-owned iterator in the
same series is the API that answers that need instead: it returns one borrowed
record per call, so ordinary loop control replaces the "how do I stop" trick,
and a destroyed iterator that has not been exhausted preserves the wakeup
handshake.

## Verification and debugging path

1. Prove a queued-but-silent ring from userspace. With one ring, compare
   `ring__producer_pos(r)` and `ring__consumer_pos(r)` and then call
   `ring_buffer__poll(rb, -1)`. If the positions differ by at least one whole
   record and the poll never returns, the record is committed and no wakeup is
   coming. Both accessors are acquire loads in libbpf, so the numbers are real
   even when the wakeup path is broken.
2. Separate a busy record from a missed notification. Read the 8-byte length
   word at `data + (consumer_pos & mask)`; if its busy bit is set, the record
   is reserved but not committed, and no notification is due yet by design.
   Drain past it, or wait: the commit will notify if the consumer is still at
   that exact position.
3. Separate the race from an early stop. Instrument the callback to count
   records and log the bound or the negative return that ended the last
   `consume_n` call. If the stall happens only after a bounded return or a
   failing callback, it is the documented drain rule, not the ordering bug.
4. Confirm the ordering. Build the same consumer with a full fence inserted
   between the consumer-position store and the next producer-position load (or
   apply the series) and run the load test that reproduced the stall. A memory
   model run of the four accesses — consumer store then load, producer store
   then load, all to the two position words — shows the outcome where each
   side reads the other's old value, and shows that removing the fence admits
   it while a `SEQ_CST` fence removes it.
5. If patching libbpf is not an option, do not block on the notification
   alone. Poll with a bounded timeout and drain with `ring_buffer__consume()`
   until it returns 0 before sleeping, or treat "producer ahead of consumer"
   from the two position accessors as the readiness condition. Both cost a
   syscall or a spin but do not depend on the missing barrier.

## The limitation

- The fix is a patch series in review, not released API. A consumer built
  against an older libbpf has the gap; upgrading is the only real repair,
  because the fence has to sit inside the consumption loop.
- The handshake assumes one consumer. Multiple threads or processes consuming
  the same ring share `consumer_pos` and will break the equality test between
  them regardless of barriers; shard with a ring per consumer or consume from
  one thread only.
- The kernel's notification path is an `irq_work`, not a direct wakeup, so it
  adds latency even when it fires. Benchmarking "wakes per record" under load
  measures `BPF_RB_NO_WAKEUP`/`BPF_RB_FORCE_WAKEUP` policy and this handshake
  together, not the consumer's own speed.
- Overwrite-mode maps still do not support callback-based consumption, and the
  caller-owned iterator does not extend to them in this series.

## References

- [Ring buffer consumer implementation, libbpf](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/tools/lib/bpf/ringbuf.c) — the consume loop, the release store of the consumer position, the level-triggered epoll registration, and the position accessors used for diagnosis.
- [Ring buffer producer and commit path, kernel](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/kernel/bpf/ringbuf.c) — the record-committing exchange, the equality test against the consumer position, the `irq_work` wakeup, the reserve-side release store of the producer position, and the user ring buffer peek that widens positions.
- [BPF ring buffer design and notification policy](https://docs.kernel.org/bpf/ringbuf.html) — self-pacing notifications sent only when the consumer has caught up, the wrap-around note on the logical counters, and the no-wakeup and force-wakeup flags.
- [Kernel memory barriers](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/Documentation/memory-barriers.txt) — release-acquire pairs are "not required to order prior stores against subsequent loads", and general barriers are what order all operations for every CPU.
- [Consume-bounds documentation, libbpf header](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/tools/lib/bpf/libbpf.h) — the current text of the consume entry points and the callback typedef that the series extends.
- [Series cover letter, version 3](https://lore.kernel.org/bpf/20261120-bpf-ringbuf-fixes-v3-0-ff7ca15d6c37@kernel.org/) — the ring buffer fixes, the caller-owned iterator, the callback-return decision, and the two fixes already applied from earlier revisions.
- [Patch adding the missed-wakeup barrier](https://lore.kernel.org/bpf/20261120-bpf-ringbuf-fixes-v3-3-ff7ca15d6c37@kernel.org/) — the exact fence placement, the early-exit fence, the single busy-record retry, and the new drain-before-waiting documentation.
- [Patch fixing 32-bit position wrap in the user ring buffer](https://lore.kernel.org/bpf/20261120-bpf-ringbuf-fixes-v3-1-ff7ca15d6c37@kernel.org/) — positions kept at word width so the available span wraps with the counters, and the advertised window checked against the ring size.
- [Patch adding the caller-owned iterator](https://lore.kernel.org/bpf/20261120-bpf-ringbuf-fixes-v3-4-ff7ca15d6c37@kernel.org/) — borrowed records, explicit publication per step, and the preserved handshake on early destruction.
- [Consumer ordering fix in a Rust ring buffer reader](https://github.com/aya-rs/aya/commit/7277a57ea8cdb74918d3096a4b22b6d814481973) — the same store-load gap, and the per-architecture evidence that a release store followed by an acquire load is not a full barrier.
- [Original RingBuf reader commit](https://github.com/aya-rs/aya/commit/e2cf734490bc188bcedb1eac92d23d81123e42cd) — equality comparison instead of ordering because the 32-bit positions wrap, and the note on the kernel's own wrap handling at the time.

## Community discussion today

Coverage this run: two archive-opted Slack workspaces through the approved
read-only archive, three allowlisted channels in total — nothing in the
24-hour window, one message in the seven-day window, a binary pattern search
question already answered on this site. The two allowlisted Discord
workspaces, ten channels between them, answered with an unauthenticated login
page and are recorded as inaccessible rather than quiet. The public bpf
mailing-list archive was read as ordinary public pages and held 46 distinct
thread roots dated 2026-11-20, which is where the question above and most of
the themes below come from. The public practitioner forum had no new technical
thread inside the window; its newest posts were project announcements and a
conference call for papers.

### The wakeup handshake, and why the API grew a second shape

The question above is one instance of a theme that ran through several
threads: an edge-triggered notification is a promise about the *next* check,
so every early return has to leave the other side able to make progress. The
ring buffer series states the same boundary from three directions — the
handshake across a bounded return, the callback that may stop consumption, and
the "already committed but not yet visible" record whose producer skipped the
wakeup based on a stale position. Practitioners who have seen a reader hang
under load and blamed the producer, the ring size, or the callback have
usually been looking at the last of those three.

The same series also shows where the API was thin. A callback can only stop by
returning a negative error, which conflates "I have what I need" with "the ring
is broken", and the positive-return shortcut was available but already
meaningless because callers had been ignoring positive values for years. The
answer in review is a caller-owned iterator: one borrowed record per call, a
`next` that ends the previous borrow, and an explicit choice about whether to
publish consumed records immediately. For anyone writing a reader today, the
practical takeaway is to keep the publication of the consumer position and the
decision to wait in the same code path, and to drain before blocking whenever
the path can end early.

### A verifier that decides after the error instead of before

The largest series in the window replaces the main verifier analysis with a
join-first pass and a backward slice from the error. The motivation is
concrete: the current design guesses at each join point whether to merge two
states or keep them apart, and the guess is final, so joining can lose a fact
and reject a valid program while keeping apart can grow the state count until a
limit is hit. The proposed order is the opposite — join all states, run the
pass, and only if an error appears, use equations in the state to keep apart
exactly the states the error depends on.

The cover letter quantifies it over 331763 builds of 6005 programs from other
projects: mean instructions 18110 on the 7.4 candidate, 2716 on bpf-next,
1750 with the series; mean verification time 23.2 ms, 10.3 ms, 5.2 ms; mean
peak states 562, 143, 18. It also deletes the scalar-evolution file outright,
because per-counter ranges become equations attached to the state. The
practitioner-relevant part is the diagnostic direction: with the current
verifier a rejection tells you which pattern the guess misjudged, and with the
proposed one a rejection comes with the path and the numbers the error
actually depends on.

### A per-CPU scratch slot is not safe in a preemptible context

Three helpers shared three per-CPU `pt_regs` slots selected by a nesting
counter. Raw tracepoint and tracing programs can be preempted, so tasks leave
those slots out of order and one task's sample receives another task's
registers. The reported measurement is unambiguous: on a preemptible kernel
with two raw tracepoint programs and a real-time thread forcing preemption,
5 runs of 10 seconds each produced 7 to 18 samples with the wrong return
address per program, 362 to 535 empty kernel stacks, and 168 to 364 empty
stack ids; with the fix, all three are zero. The fix builds a `pt_regs` on the
stack of each helper invocation and removes the slots.

Removing the slots also removed a nesting limit, so the series restores it
explicitly and shows why it exists: `bpf_perf_event_output()` warns on a
fourth entry on one CPU, and build-id collection in the stack helpers takes
`mmap_read_trylock()`, which fires the mmap-lock tracepoints, so programs on
those tracepoints nest into each other as deep as there are programs. With
only the slot removal, measured nesting depth for the build-id stack helper
rose from 4 to 6 and its call count from 1.5M to 5.9M; with the explicit
limit restored the same run is back at depth 4 and 1.5M calls, and the
`bpf_perf_event_output()` warning no longer fires from either a kprobe or a
raw tracepoint program. For anyone debugging a stack trace
that shows another task's frames, the diagnostic is the same: check whether
the kernel context came from a per-CPU scratch buffer that a preemption could
have overwritten.

### Two boundaries stated plainly by their authors

A seccomp filter-cache proposal to JIT a classic BPF filter once and share it
between processes was rejected on two specific points rather than on
performance: "Immutable is not enough. The image depends on the caps of the
task that loads the filter", and any task can time `seccomp()` and learn that
somebody else has that filter installed, while reading a filter back requires
`CAP_SYS_ADMIN`. The reviewer's alternative is the boring one — `fork()`
already shares the filter, so a launcher installs the common filter once and
forks the sandboxes. The general boundary: a JIT image is only as portable as
the credentials and the pointer layout it was generated under.

An arena series replaced its first attempt with a single primitive that maps
pages the arena did not allocate, each import tracked in an interval tree owned
by the supplier rather than by the arena, freed as a whole by one call,
excluded from arena memory accounting, and faulting with `SIGBUS` if a guest
touches pages after its producer went away. The first design had shared a
pinned arena through a guest NUMA node, and the review found the exact
problems that shape predicts: the pages return to the guest's buddy allocator
while the host can still write them, and block granularity costs 16 MB of guest
memory per shared page. Both designs are worth reading as a pair, because they
are the same question asked twice: who owns the page, and what happens the
moment that owner stops.

### Scheduler entry paths that do not agree with each other

Two `sched_ext` fixes came from watching counters that should have moved and
did not. On the fork path, `wake_up_new_task()` consults CPU selection but
activates the new task without the flag that tells the scheduler a CPU was
chosen, so an ext scheduler never sees the "CPU selected" case and the fallback
counter never increments on forks. A probe forcing selection fallbacks counted
0 fallback events over 40k forks on the baseline while the wakeup path counted
normally. The second is a state contract: `rt_mutex_setprio()` is the only
entry into sched_ext that does not block a retained donor first, so a task
boosted by an RT waiter and blocked on a second mutex is routed back to ext
carrying the queued-and-blocked state that the scheduler's own transition code
promises never happens. A driver-model trigger produced about 39966 such donors
across 40000 priority-inheritance cycles on the baseline and 0 with the fix.
The practical lesson for scheduler authors is to treat every entry path as a
separate producer of state, and to check the counters per path rather than in
aggregate.

### Smaller items

The unsigned 32-bit enum relocation series reached its fourth and fifth
revisions within the window, still iterating on which instruction forms a fix
must accept. A 49-patch verifier rewrite appeared at two revisions on the same
day. A perf-core series fixed software-event sample data being reused between
events and stopped a BPF program from reading an unset sample address, then
removed the 256-byte zeroing of that structure: with software counters open the
counters cost 54 ns on a ~627 ns context switch, and the change brings that to
44 ns. A symbol-lookup series reached its third revision with a name-ordered
index and a tunable marker density. On the networking side, a receive-side
series moved the "XDP attach excludes hardware receive coalescing" rule out of
individual drivers and into the core, because drivers disagreed about whether
to refuse the attach or to clear the feature — and so that the features return
from the wanted set after detach.