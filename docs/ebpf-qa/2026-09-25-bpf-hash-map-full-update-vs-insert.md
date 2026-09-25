# Why does a full BPF hash map reject new keys with E2BIG, while updating an existing key still succeed?

**Short answer:** For `BPF_MAP_TYPE_HASH`, `max_entries` is a hard cap on distinct keys, and the cap is enforced at element allocation time: a *new* key needs a fresh element, and when no element is available the update returns `E2BIG` (documented: "the number of elements in the map reached the *max_entries* limit"). An *update of an existing key* does not need a new element: the preallocated path swaps it with a per-CPU spare slot, and the lazy path explicitly permits allocation "when map is full and update() is replacing old element, it's ok to allocate, since old element will be freed immediately" — so both keep working at capacity. `BPF_MAP_TYPE_LRU_HASH` does not fail at capacity the same way: it evicts the least recently used entry to make room instead of returning `E2BIG`, which is why a key you never deleted can suddenly read back `NULL`.

## Where the E2BIG comes from

The uapi contract is explicit. The `bpf_map_update_elem(2)` man page (generated from `include/uapi/linux/bpf.h`) lists `E2BIG` with the description "the number of elements in the map reached the *max_entries* limit specified at map creation time", alongside `EEXIST` (when `BPF_NOEXIST` is given for an existing key) and `ENOENT` (when `BPF_EXIST` is given for a missing key).

Two map families reach that limit in different code:

- **Preallocated hash (the default).** `prealloc_init` allocates all `max_entries` elements in one contiguous pool plus a per-CPU freelist. When the freelist is empty, `__pcpu_freelist_pop` in `alloc_htab_elem` fails and returns `ERR_PTR(-E2BIG)`.
- **Lazy hash (`BPF_F_NO_PREALLOC`).** `alloc_htab_elem` calls `is_map_full` before allocating, and `is_map_full` compares the element count against `max_entries` (a `percpu_counter` with batch 32 when the map is large, otherwise a plain `atomic_t`). Full map + no old element to replace → `ERR_PTR(-E2BIG)`. A genuinely out-of-memory allocation in the lazy path is a different error: `ENOMEM`, not `E2BIG`.

`E2BIG` also appears at map *creation* time — `key_size + value_size` too large to be a kmalloc-able element, per-CPU value size over `PCPU_MIN_UNIT_SIZE`, or `max_entries > 1UL << 31` — so the errno by itself does not tell you which of these you hit.

## Why updating an existing key does not hit the cap

`htab_map_update_elem` looks up the key first. If `l_old` is found, the full-capacity question only matters for the *allocation* step:

- **Preallocated:** `alloc_htab_elem` takes the per-CPU `extra_elems` slot, stashes `l_old` into it (`*pl_new = old_elem`), and swaps the new element in at the head of the bucket. No freelist pop, no capacity check, no `E2BIG` — the total element count does not grow.
- **Lazy:** `is_map_full` returns true, but the comment in `alloc_htab_elem` states the rule: when the map is full and the update is replacing an old element, allocation is allowed because the old element is freed immediately (`free_htab_elem` after the bucket-list swap, after `check_and_cancel_fields` cancels any embedded BPF object fields). The count stays flat: `inc_elem_count` for the new element, `dec_elem_count` for the old one.

The capacity check `is_map_full` is only consulted for *new* keys on a full map. That is the whole asymmetry: at `max_entries`, "new key" → `E2BIG`, "existing key" → success.

## The spin-lock value exception

For a map whose value embeds `bpf_spin_lock` (updates pass the `BPF_F_LOCK` flag — the map's BTF record must carry `BPF_SPIN_LOCK` or the update is rejected with `-EINVAL`), `htab_map_update_elem` takes a shortcut: if the key already exists, it grabs the element's spin lock and calls `copy_map_value_locked` in place, returning before any allocation. Updating an existing spin-locked value therefore never allocates and cannot be blocked by capacity — `bpf_spin_lock` (kernel 5.1+) is the documented way to keep a value consistent while programs on different CPUs mutate it concurrently. New keys on such a map still fall through to the normal allocation path and are subject to the same `E2BIG` at capacity.

## How the LRU variants instead

`BPF_MAP_TYPE_LRU_HASH` / `LRU_PERCPU_HASH` replace the `E2BIG` at capacity with eviction. The kernel docs state it directly: "An LRU hash will automatically evict the least recently used entries when the hash table reaches capacity." In the update path (`htab_lru_map_update_elem` / `__htab_lru_percpu_map_update_elem`), the element is obtained from the LRU before taking the bucket lock, because the eviction work it may need acquires the bucket lock itself: `prealloc_lru_pop` → `bpf_lru_pop_free`.

`bpf_lru_pop_free` (in `kernel/bpf/bpf_lru_list.c`) escalates in steps:

1. Pop a node from this CPU's local free list (`free_llist`); pending nodes are not popped here — they are flushed into the global lists during the fetch below.
2. If that is empty, fetch up to `target_free` free nodes from the global lists into the local free list — this first flushes this CPU's pending nodes into the global free/active/inactive lists by ref bit, then drains the global free list — and re-pop the local free list.
3. If the global free list is short, shrink: move ref-bit-cleared nodes from the global inactive list into the free list via `lru->del_from_htab` (unlink from the hash table — the same unlink-then-free path as `delete`, without zeroing value bytes).
4. If the inactive list yields nothing, do a *force shrink* that removes a node ignoring the ref bit, preferring the inactive list over the active list, so that even a hot map can be made to give up an element.
5. If still nothing, steal a free (then pending) node from a local list — round-robin over CPUs starting at `next_steal`, per the kernel comment: "the current CPU and remote CPU in RR" — skipping any victim whose lock cannot be acquired.
6. If all of that fails, `prealloc_lru_pop` yields nothing and the update returns `-ENOMEM` — the LRU variant fails at capacity as an allocation failure, not the plain hash's explicit `E2BIG` full check.

The docs summarize the escalation as: use CPU-local state, fetch free nodes from the global lists, pull any node from a global list and remove it from the hash map, then pull any node from any CPU's list. One consequence worth remembering: an LRU map's "full" means *evict now*, so a lookup for a key you never deleted can return `NULL` after capacity pressure — a capacity eviction, not a delete. The map `type` field in `bpf_map_info` is how user-space confirms which family it is dealing with.

## How to verify it

1. **Watch the errno, not the message.** Fill a plain `BPF_MAP_TYPE_HASH` to `max_entries`, then issue two updates: a new key must fail with `E2BIG` (`strerror(-ret)` from `bpf_map_update_elem` / libbpf's `bpf_map__update_elem`), and an update to an existing key must return 0. Repeat after one `bpf_map_delete_elem`: a new key succeeds again.
2. **Confirm the configured cap.** `bpf_map_info` reports `max_entries` and the map type, so a failure you expected at 1000 keys may actually be a map created with a smaller `max_entries` — check the fd's info before assuming capacity.
3. **Distinguish LRU eviction from a bug.** On a `BPF_MAP_TYPE_LRU_HASH` at capacity, lookups for keys you never deleted can start failing after other updates evict them; verify with `bpf_map_lookup_elem` returning `NULL` plus a new update that would have been `E2BIG` on a plain hash.
4. **Count elements without assuming a counter.** For plain hash maps there is no stable user-space "current count" field in `bpf_map_info`; estimate by iterating keys with `bpf_map_get_next_key` (batched variants for a stable sweep), and remember that iteration restarts from the first key if the `cur_key` you passed was just deleted.
5. **Check which allocation family.** If the map was created with `BPF_F_NO_PREALLOC`, an update failure at capacity can be `ENOMEM` (allocator exhaustion) rather than `E2BIG` — the `E2BIG` path there is the explicit full-check, and `ENOMEM` is `bpf_mem_cache_alloc` running out of memory.

## Where the answer stops

- The `E2BIG` on *create* (oversized `key_size + value_size`, per-CPU value over `PCPU_MIN_UNIT_SIZE`, `max_entries > 1UL << 31`) shares the errno with the full-map update; the return code alone does not say which.
- `per-CPU LRU` maps with `BPF_F_NO_COMMON_LRU` give each CPU its own LRU list, which changes *which* element evicts per CPU — not the eviction-instead-of-E2BIG behavior.
- The element count used by `is_map_full` is a `percpu_counter` with batch 32 on large maps, so the "full" comparison is approximate by design; the eviction guarantee on LRU maps is the LRU property, not an exact-size promise.
- Whether a recycled element's value bytes persist is a separate boundary (documented in the 2026-09-23 entry): unlinking and evicting never zero the value region; `E2BIG` and eviction are capacity decisions, not memory-scrubbing guarantees.

## References

- [BPF_MAP_TYPE_HASH, with PERCPU and LRU Variants (kernel docs)](https://docs.kernel.org/bpf/map_hash.html) — preallocation by default, `BPF_F_NO_PREALLOC`, "automatically evict the least recently used entries when the hash table reaches capacity", the LRU update escalation steps, `bpf_spin_lock` synchronization.
- [kernel/bpf/hashtab.c](https://raw.githubusercontent.com/torvalds/linux/master/kernel/bpf/hashtab.c) — `alloc_htab_elem` (freelist pop → `E2BIG`; the "when map is full and update() is replacing old element" comment; the `extra_elems` swap); `htab_map_update_elem` (lookup-first, `BPF_F_LOCK` in-place shortcut); `htab_lru_map_update_elem` / `__htab_lru_percpu_map_update_elem` (alloc-before-lock); `is_map_full` (`percpu_counter` batch 32 vs `atomic_t`); the create-time `E2BIG` checks.
- [kernel/bpf/bpf_lru_list.c](https://raw.githubusercontent.com/torvalds/linux/master/kernel/bpf/bpf_lru_list.c) — `bpf_lru_pop_free` (local free lists, cross-CPU steal, inactive shrink via `del_from_htab`, force shrink ignoring the ref bit).
- [include/uapi/linux/bpf.h](https://raw.githubusercontent.com/torvalds/linux/master/include/uapi/linux/bpf.h) — the `bpf_map_update_elem(2)` `E2BIG` contract text, `BPF_F_LOCK`/`BPF_F_NO_PREALLOC`/`BPF_F_NO_COMMON_LRU` flags, `struct bpf_map_info` (`max_entries`, the map `type` field).

## Community discussion today

Honest coverage note: the two watchlist-opted Slack archives were reachable this run but returned **zero messages** in the rolling window (snapshot ok, 0 bytes, 0 messages), and the allowlisted Discord channels, the public bpf mailing list, and r/eBPF remain visible-browser-only, with no visible-browser session available in this run. No private community material was available for 2026-09-25, so the question above is a **fallback selection**: a genuine, recurring practitioner boundary of the monitored eBPF development community ("my map rejects new keys at capacity, yet updating the same key works" / "a key disappeared from my LRU map without a delete"), grounded entirely in the public primary sources above (kernel docs and upstream `kernel/bpf/hashtab.c`, `bpf_lru_list.c`, and the uapi header) rather than in any thread.

The practitioner pattern that recurs around this boundary: a monitoring tool or eBPF program keyed on connection state (tuple → counter/state structs) silently stops recording once the key space outgrows `max_entries`, because every new key is dropped at the update with `E2BIG` while the hot existing keys keep working — the symptom reads like "new sessions are missing but old ones keep counting", and the fix is sizing `max_entries` (or switching to `BPF_F_NO_PREALLOC` or an LRU variant, accepting eviction), not a retry loop. The LRU side recurs as its own surprise: an evicted key that was never deleted reads back `NULL`, which people misdiagnose as a delete bug when it is capacity eviction. The public-doc-anchored rule that closes both: a plain hash map's `max_entries` is a hard cap enforced at allocation (new keys `E2BIG`, existing-key updates exempt, spin-locked values in-place); an LRU hash map turns that same capacity into an eviction decision, so "full" means "something older leaves now", not "the next insert fails". No private text, identity, channel, or link is reproduced here.
