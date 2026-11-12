# When does a BPF LRU hash map evict an entry, and what changes when its LRU list is per-CPU instead of shared across CPUs?

Eviction happens only on the insert or update path, and only once the table has already reached `max_entries`. A lookup never directly frees a slot: a read does not release a node. What a read does is set a lazy recently-used bit on the node it hit; the actual list movement and the actual slot release happen later, when an update is forced to free a node and the list rotation scans run. The LRU is a two-phase active and inactive list, not a single least-used queue, so "least recently used" is a heuristic the kernel maintains incrementally, not a value you can read off at any instant.

The `BPF_F_NO_COMMON_LRU` flag does not change when eviction fires. It changes which entry survives and the lock profile. Without the flag every CPU shares one global LRU list, so all rotation and free-node accounting goes through shared state; with it, each CPU keeps its own LRU list and evicts from its own list before stealing from another CPU. The trigger stays exactly the same in both cases: an insert at capacity. The flag trades the globally-least-used property for per-CPU locality and a cheaper lock, not for fewer evictions.

## The mechanism

1. **Eviction is on the insert path.** `htab_lru_map_update_elem` first calls `prealloc_lru_pop`, which asks the LRU list for a free node (`bpf_lru_pop_free`). While the table is below `max_entries` there are already free nodes in the preallocated pool and nothing is evicted. When the table is at capacity, `bpf_lru_pop_free` has no free node, so it forces one out: it scans the inactive list (`__bpf_lru_list_shrink_inactive`), and for a node whose recently-used bit is clear it calls the map delete hook and moves the node onto the free list. That is the moment a live key is actually dropped from the hash table.

2. **Recency is a lazy bit, not a position.** The in-BPF lookup helper sets the node's recently-used bit (`bpf_lru_node_set_ref`); the user-space lookup helper does not. Nothing is moved on a read. The lists are touched only during the rotation scans (`__bpf_lru_list_rotate_active` and `__bpf_lru_list_rotate_inactive`): a scanned node with the bit set goes back to the head of the active list for one more chance to survive, and a node without it falls from active to inactive. Nodes are freed only from the tail of the inactive list.

3. **The `BPF_F_NO_COMMON_LRU` switch.** `bpf_lru_pop_free` dispatches on the per-CPU flag. Without it, `bpf_common_lru_pop_free` keeps one shared LRU list plus per-CPU pending lists and takes a shared lock to account for them. With it, `bpf_percpu_lru_pop_free` gives every CPU its own list guarded by its own lock, and a CPU only steals a node from another CPU, in round-robin, when its own list is exhausted. The batch size to pull per update, `target_free`, is computed at map init with a minimum of one node and a budget of 128, with the aggregate reservation across all CPUs capped at half the map size.

4. **OBI trace context map as a concrete case.** OBI's pinned `traces_ctx_v1` is a `BPF_MAP_TYPE_LRU_HASH` with `max_entries` of `1 << 14` and no `BPF_F_NO_COMMON_LRU`, so it uses the shared global LRU list. OBI does not reserve a slot for any thread, so when the number of live stamped threads is high the least-recently-used entry is evicted on the next insert, and a currently active thread can miss its own context.

## Verification and debugging path

1. **Check the live element count.** `bpf_map_get_info` or `bpftool map show` on the pinned map reports the element count against `max_entries`. Eviction only becomes possible once the count is at `max_entries`; below that, inserts never drop a key.

2. **Watch the update return code, not the lookup.** A full non-LRU hash returns `-E2BIG` on a new-key insert. A full LRU hash instead evicts the LRU victim and returns `0`, or rarely `-ENOMEM` when it cannot secure a free node under lock contention. If a map you declared LRU is failing inserts with `-E2BIG` at capacity, it is a plain `BPF_MAP_TYPE_HASH`, not an LRU variant.

3. **Trace the victim with the delete hook.** Probe the map update path, or attach to the map delete, and watch which key disappears when you push one element past capacity. The key that vanishes is the LRU victim the list selected.

4. **For the flag question, measure victim locality.** With the shared LRU the evicted key is the globally least-recently-used one. With `BPF_F_NO_COMMON_LRU` it is the least-recently-used key on the CPU that did the insert, which on a hot multi-CPU workload can be a different, still-active key than the global one.

## The limitation

- The LRU property is a heuristic. The kernel maintains two lists and rotates them in bounded scans; it is not a strict least-recently-used queue, and the kernel documents these internals as subject to change.
- Recency is not symmetric. An in-BPF read marks the node; a user-space read does not. Two identical workloads that differ only in who reads the map will evict different keys.
- The flag changes victims and lock cost, not the trigger. `BPF_F_NO_COMMON_LRU` does not stop eviction and does not keep the globally least-used entry; it makes eviction per-CPU.
- A miss on a full LRU map is not proof the key was never written. It means the key was the LRU victim at the last insert, so the value may have existed and then been evicted.

## References

- [kernel Documentation/bpf/map_hash.rst](https://github.com/torvalds/linux/blob/master/Documentation/bpf/map_hash.rst) - `BPF_MAP_TYPE_LRU_HASH` automatically evicts the least-recently-used entry when the table reaches capacity; the `BPF_F_NO_COMMON_LRU` flag requests a per-CPU LRU list instead of the shared one.
- [kernel kernel/bpf/hashtab.c](https://github.com/torvalds/linux/blob/master/kernel/bpf/hashtab.c) - `htab_lru_map_update_elem` and `prealloc_lru_pop` are the insert path where eviction happens; the lookup helper sets the recently-used bit in-BPF and skips it from user space.
- [kernel kernel/bpf/bpf_lru_list.c](https://github.com/torvalds/linux/blob/master/kernel/bpf/bpf_lru_list.c) - the two-phase active and inactive LRU: `__bpf_lru_list_rotate_active`, `__bpf_lru_list_shrink_inactive`, and `bpf_lru_pop_free` with the per-CPU versus common dispatch.
- [kernel commit 3a08c2fd7634 ("bpf: LRU List")](https://github.com/torvalds/linux/commit/3a08c2fd7634) - the original LRU list implementation and the eviction steps taken during a map update.
- [OBI bpf/shared/obi_ctx.h](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/blob/main/bpf/shared/obi_ctx.h) - `traces_ctx_v1`, a pinned `BPF_MAP_TYPE_LRU_HASH` with `max_entries` of `1 << 14` and no `BPF_F_NO_COMMON_LRU`.

## Community discussion today

The two opt-in archive channels (the two CNCF OpenTelemetry instrumentation channels) produced zero messages this run; the 24-hour window advanced to an empty snapshot. With no new question in the archive and the recent days already covering the FUSE and OBI trace context map miss axes, this entry takes the natural continuation of that same map: `traces_ctx_v1` is an LRU hash, so the follow-on question is what actually evicts from it and what the per-CPU flag would change. The answer is grounded in the kernel LRU map source and the OBI map spec rather than in a new archive question.

Channel coverage for this run: the two opt-in archives supplied zero messages. The visible-browser-only sources - the eunomia-bpf and sched-ext Discord servers, the bpf mailing list, and r/eBPF - could not be reviewed this run because no visible-browser session was available, so they are marked not-covered, not quiet.
