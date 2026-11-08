# Why does the verifier let any BPF iterator program update a sockmap, and why is only the sockmap iterator safe?

Because the verifier answers the question once, at load time, from the program's type
and expected attach type. `may_update_sockmap()` returns true for every tracing
program whose expected attach type is an iterator, and the comment above it states the
assumption that made that breadth defensible: a program in these contexts "cannot get
access to a locked struct sock". That assumption is about lock context, not about
program type, and the two stopped agreeing. An iterator hands its program a socket
that may be locked by the same task, or unlocked but mid-transition, and the update path
then takes a spinlock with no ownership check, allocates a socket-private state object,
and rewrites the socket's protocol callbacks. The upstream response, in review now, is to
narrow the verifier gate to the one iterator that already guarantees both properties:
the socket is a member of a sockmap, and the iterator does not hold its lock.

## The mechanism

1. **Where the decision is made.** For a `SOCKMAP` or `SOCKHASH` map the verifier
   accepts a fixed list of helpers, and `map_update_elem` and `map_delete_elem` are
   admitted only through `may_update_sockmap()`. That function checks two things: the
   resolved program type is `BPF_PROG_TYPE_TRACING`, and the expected attach type is
   the iterator one. It never looks at *which* iterator the program attaches to, so the
   set of programs that pass is "all of them". The comment records why: the author
   believed no iterator could reach a socket whose lock was already held.

2. **What an update actually does at run time.** The update is not a pointer store.
   `sock_map_update_elem()` checks that the socket is a full socket and that its
   protocol has a psock update hook, then disables bottom halves and takes
   `bh_lock_sock()`, which is a plain `spin_lock()` on the socket's lock word with no
   ownership test, then checks the socket state, then calls the common path. That path
   allocates a link object with `GFP_ATOMIC`, calls `sock_map_link()`, which may
   allocate and install a fresh psock under the socket's callback lock, and finally calls
   the protocol's psock update hook. For a stream Unix socket that hook reads the peer
   pointer and takes a reference on it. So a single helper call changes the socket's
   protocol, its callbacks, and its peer reference count, while packet processing may be
   running on it.

3. **Why the "no locked socket" premise failed.** The Unix iterator ran the program
   under `lock_sock_fast()`. On the fast path that function leaves the socket's spinlock
   held and bottom halves disabled, with the ownership flag still clear. The update path
   then takes the same spinlock through `bh_lock_sock()`, and because `spin_lock()` is
   not recursive, the task spins on a lock it already owns. The kernel reports it as a
   possible recursive locking, with the same socket lock listed as both held and
   requested, and the task never finishes the read of the iterator file. The fix moved
   the iterator to the slow `lock_sock()` form, which drops the spinlock after setting
   the ownership flag, and added the Unix state lock around the program run so the peer
   pointer cannot be freed underneath the hook in the update path.

4. **Why the answer is still "only the sockmap iterator".** Those two fixes make the
   Unix iterator survivable, but they do not make every iterator a sound place to
   mutate a sockmap. The general problem is stated in the review thread: an update that
   arrives from a context that does not hold the socket lock gets no exclusive access to
   the socket, and `bh_lock_sock()` explicitly does not care whether the socket is owned
   by a task. A file-based iterator can hand the program a socket taken from a file
   descriptor, with no lock and no guarantee of state. The sockmap iterator is different
   in two concrete ways: the socket it yields is already a map member, so it has a psock
   and is already in a state the update accepts, and the iterator takes only the RCU read
   lock around the program, so it holds no socket lock and the update's own spinlock is
   the first acquisition rather than a nested one.

5. **The same boundary shows up in the destroy kfunc.** `bpf_sock_destroy()` is
   registered for tracing programs and filtered so it is callable only when the expected
   attach type is the iterator one. Its TCP implementation decides whether to take the
   socket lock by asking whether a BPF program is currently running: inside a BPF context
   it assumes the caller already holds it, and the comment says so. That assumption is
   true for the TCP and UDP iterators, which lock the socket before running the program,
   and false for the sockmap iterator. The same series therefore restricts the kfunc to
   the TCP and UDP iterators by BTF id, and marks the sockmap iterator's socket pointer
   trusted only in the patch that comes after that restriction. Marking it trusted is
   what lets a plain pointer from that context be passed to helpers at all, so doing it
   first would have reopened the hole the restriction closes.

6. **Why the gate is by name, not by a flag.** The patch compares the attach function
   name, after stripping the iterator prefix, against the string for the sockmap target,
   and the author notes having tried a BTF id set first and abandoned it because of the
   configuration guards a single-entry list would need. The name is the same key the
   iterator machinery itself uses to bind a program to a target, so the check matches
   how attachment already works, but it means the gate inherits the string-keyed
   nature of iterator identification.

## How to verify and debug

1. **Read the verdict, not the intent.** Load the same program body twice, once from an
   `iter/sockmap` section and once from `iter/unix`, with a `SOCKMAP` map and one
   `bpf_map_update_elem()` call. On a kernel without the restriction both load; the
   in-tree selftest for this area carries exactly that pair, with the sockmap case
   expected to succeed and, after the patch, the Unix case expected to fail with the
   cannot-update message.

2. **Find the lock context of the iterator you are using.** Read the iterator's show
   callback in the kernel source. If it calls `lock_sock()` before running the program,
   the program runs with the socket owned by the current task; if it only takes the RCU
   read lock, as the sockmap iterator does, it does not. That single difference decides
   whether the update's spinlock is a first acquisition or a nested one on old kernels,
   and whether the peer pointer you are about to reference is protected.

3. **Recognise the two failure signatures.** A deadlock from the lock path shows up as a
   possible recursive locking report naming the same socket lock twice, with the
   iterator's fast lock function in the stack and the update path immediately below it.
   A state race instead shows up as a slab use-after-free inside the Unix psock update
   hook, reached from the map update through the link step, with the free coming from a
   concurrent close. The first is a lock nesting bug, the second is a lifetime bug, and
   they need different fixes.

4. **Check what your program can actually do with a yielded socket.** Dump the vmlinux
   BTF for the iterator function and look at the context argument types, then check
   which of them are marked trusted for your program type. A pointer that is not
   trusted cannot be handed to a helper that expects one, which is why the trusted-marking
   patch and the helper restriction have to land in that order.

5. **Do not infer the runtime contract from a successful load.** A program that loads
   proves only that the verifier's coarse check passed. Enumerate the contexts your
   update can arrive from, and treat any context that does not lock the socket or that
   hands out a socket not already in the map as unsupported.

## The limitation

- The restriction is a patch under review, not released behaviour. On current release
  kernels the update still verifies from any iterator, so a program that works today can
  start failing to load on a kernel that carries the narrowing. Pin the kernel you test
  against, and treat the verifier message as version-dependent.
- Narrowing to one iterator does not make sockmap updates cheap or lock-free. The update
  still allocates, still takes the map's spinlock and the socket's callback lock, and
  still runs inside the iterator's RCU read section with no ability to sleep, so it is a
  control-path operation that belongs at a low rate.
- The gate is keyed on the attach function name, so it shares the fragility of every
  name-keyed binding: a target that is renamed, or a program attached through a path
  that binds by a different name, is judged by that string rather than by the property
  the check means to express.
- The destroy kfunc keeps its own separate restriction, to the TCP and UDP iterators,
  because it needs the socket lock held by the caller. The two restrictions are
  independent and can drift apart in a later change.
- None of this makes an update from an iterator atomic with respect to the socket's data
  path; it only makes the socket's state transition safe from the contexts that are
  allowed to start it.

## References

- [Patch series: restrict iter-context sockmap updates to the sockmap iterator](https://lore.kernel.org/bpf/20261107-sockmap-update-limit-iter-v3-0-7c0bad1ad036@rbox.co/) — the cover letter, the review history, and the stated shape of the change.
- [Patch: gate sockmap updates on the attach target name](https://lore.kernel.org/bpf/20261107-sockmap-update-limit-iter-v3-2-7c0bad1ad036@rbox.co/) — the verifier change, the name comparison, and the note about the alternative that was abandoned.
- [Patch: restrict the destroy kfunc to the TCP and UDP iterators](https://lore.kernel.org/bpf/20261107-sockmap-update-limit-iter-v3-5-7c0bad1ad036@rbox.co/) — why the sockmap iterator does not hold the socket lock, and the BTF id set that replaces the type-only filter.
- [Patch: negative selftest for an update from a non-sockmap iterator](https://lore.kernel.org/bpf/20261107-sockmap-update-limit-iter-v3-3-7c0bad1ad036@rbox.co/) — the expected failure message and the program shape used to produce it.
- [Kernel commit: fix the Unix iterator deadlock](https://github.com/torvalds/linux/commit/4d328dd69538) — the recursive locking report, the fast lock path that holds the spinlock, and the switch to the slow form.
- [Kernel commit: take the state lock for the Unix iterator](https://github.com/torvalds/linux/commit/64c2f93fc325) — the peer use-after-free in the psock update hook and the lock that closes it.
- [Review message: which contexts get exclusive access to a socket](https://lore.kernel.org/bpf/87echn2fhd.fsf@cloudflare.com/) — the argument that a file-based iterator gets no exclusive access, and the proposal to allow updates only from the sockmap iterator.
- [Kernel `kernel/bpf/verifier.c`](https://github.com/torvalds/linux/blob/master/kernel/bpf/verifier.c) — `may_update_sockmap()`, the map-helper compatibility switch that calls it, and the contextual comment.
- [Kernel `net/core/sock_map.c`](https://github.com/torvalds/linux/blob/master/net/core/sock_map.c) — the update path, the link and psock setup, the state and suitability checks, and the iterator registration.
- [Kernel `net/unix/af_unix.c`](https://github.com/torvalds/linux/blob/master/net/unix/af_unix.c) — the Unix iterator show callback with its locks, and the stream psock update hook that reads the peer.
- [Kernel `net/ipv4/tcp.c`](https://github.com/torvalds/linux/blob/master/net/ipv4/tcp.c) — `tcp_abort()`, which skips the socket lock when a BPF context is running.
- [Kernel `include/linux/skmsg.h`](https://github.com/torvalds/linux/blob/master/include/linux/skmsg.h) — the atomic link allocation that makes the update usable from a non-sleepable context.
- [Kernel `kernel/bpf/bpf_iter.c`](https://github.com/torvalds/linux/blob/master/kernel/bpf/bpf_iter.c) — the RCU section around a non-sleepable iterator program, and the feature flags that decide whether a sleepable one is allowed.
- [Kernel `Documentation/bpf/bpf_iterators.rst`](https://github.com/torvalds/linux/blob/master/Documentation/bpf/bpf_iterators.rst) — the iterator registration fields; the no-sleep feature flag is used in the code but still undocumented there.
- [Kernel selftest for sockmap mutation from iterators](https://github.com/torvalds/linux/blob/master/tools/testing/selftests/bpf/progs/verifier_sockmap_mutate.c) — the per-section expectations that encode which contexts may mutate a sockmap.

## Community discussion today

This run reviewed six allowlisted sources. Two are Slack workspaces reached through the
approved read-only archive, covering three allowlisted channels; both were readable, one
carried no message in the 24-hour window and the other carried none in 24 hours and one
in the seven-day window. The kernel mailing list and the public forum were read through
their ordinary public pages and both were reachable; the forum's newest post is three
days old, so its 24-hour window is genuinely empty rather than inaccessible. The two
Discord workspaces, ten allowlisted channels in total, were not covered: this
environment has no authenticated session and their visible pages land on a login form.
They are recorded as inaccessible, not quiet. That is four sources reviewed and two not
covered.

The mailing list window holds two themes. The first is exactly the one this page answers:
which lock a BPF program actually holds, and what code assumes about it. Beyond the
sockmap series, the same window carries a virtio transport fix where the BPF receive path
held the socket lock while sleeping for data, deadlocking with the delivery worker that
needed the same lock to queue that data; the fix releases the lock around the wait and
adopts the native path's exit conditions, and separately handles the reads that should
never enter the sockmap wait at all. A second instance is the destroy kfunc restriction
described above. A third is a review reply on iterator writes to read-only maps: the
proposed predicate tested both the program-read-only flag and the frozen bit, and the
correction is that freezing stops writes from the system-call side only, so programs can
still write into a frozen map and the flag alone is the right test, with the further
point that the flag never changes and therefore needs no race consideration. The thread
also corrects the fix tags, drops a commit a bot had proposed, and asks for a selftest
that writes through the iterator's value pointer on a map created read-only for programs.

The second theme is that a verifier which joins states changes what a program's value
ranges are, and both directions of that change showed up in one day. One series of nine
fixes follows a large verifier rework: a counter that wraps just in front of the jump
that counts a loop gets a step and a residue computed from the unwrapped value, so the
range handed to the next instruction both excludes numbers the register really holds and
includes numbers it never holds, and the fix declines to set a range at all when the sum
can wrap. The same series fixes two selftests that fail only under one compiler or one
architecture, where an unrolled inner loop walks a map value with pointers and the
verifier can no longer find a counter for it, and fixes a case where a program that
throws one of several error codes is rejected because the join merges the flag with the
throw cookie; the fix teaches the argument checker to read the throw argument too, so
states that differ in the cookie are not joined. A companion series tests short loops
that carry what they load, including a loop whose slot is filled and only copied later,
which the verifier walks in a handful of instructions instead of the full trip count.
Elsewhere on the list, a patch refuses to attach to a subprogram that is called in a
recursion, because the load-time stack-depth check sees the recursion but cannot see what
gets attached to it later, and a trampoline patch that would have disabled preemption for
all non-sleepable programs drew the objection that programs with bounded loops can run for
a quarter of a second and preemption should be disabled per program, by an explicit
helper, rather than globally.

The archive-backed observability channels were quiet on the day; their single seven-day
message is quoted nowhere here because it is a status remark rather than a technical
question. The question answered above came from the public mailing list, and every
mechanism claim in it was checked against the current kernel sources for the verifier, the
sockmap map, the Unix socket and the TCP abort path, not against the patch descriptions
alone.