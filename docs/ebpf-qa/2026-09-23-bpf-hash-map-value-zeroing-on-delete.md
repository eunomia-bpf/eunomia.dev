# Does deleting a key from a BPF hash map zero out the value memory, and can a reused slot still hold the old value bytes?

**Short answer:** No. `bpf_map_delete_elem` unlinks the element from the hash buckets and hands its memory back to the per-CPU freelist (preallocated hash) or to the RCU-deferred BPF allocator (a `BPF_F_NO_PREALLOC` hash); it does **not** clear the value bytes. You cannot read a deleted value back *through the map API* — `bpf_map_lookup_elem` returns `NULL` and `bpf_map_get_next_key` skips it — but the physical bytes stay in the reused slot until a later update overwrites them, and a normal full update *does* rewrite the whole value region, so there is no cross-key value leak through the API. The rule: never rely on `delete` (or LRU eviction) to erase value contents; zero the value yourself, and do not assume embedded kernel objects (kptr/dynptr) survive, because the kernel cancels those pointers when the element is freed.

## What delete actually does

`htab_map_delete_elem` computes the bucket for the key, takes the bucket lock, and — if the element is present — calls `hlist_nulls_del_rcu(&l->hash_node)` to unlink it, then `free_htab_elem`. There is no `memset` or equivalent on the value region anywhere in that path. What happens to the memory next depends on how the map was created:

- **Preallocated hash (the default).** At map creation, `prealloc_init` allocates all `max_entries` elements in one contiguous `htab->elems` pool and fills a per-CPU freelist. On delete the element goes back onto that per-CPU freelist (`pcpu_freelist_push`). It is immediately reusable — the next same-CPU update pops it back (`__pcpu_freelist_pop`, LIFO) — but the value bytes were never touched.
- **Lazy hash (`BPF_F_NO_PREALLOC`).** Elements are allocated on demand from the per-CPU BPF memory cache, and on delete `htab_elem_free` returns them to that allocator. The physical free is **RCU-deferred** (the allocator's `free_by_rcu` / `call_rcu_tasks_trace` path), so the page is not recycled until a grace period completes; the value bytes persist even longer.

The element *count* is decremented synchronously in both paths (`bpf_map_dec_elem_count` / `dec_elem_count`), which is why a delete immediately makes room for a new element even though the underlying *memory* may be recycled a little later.

## Why a full update cannot leak one key's value into another

When a new key is inserted into a recycled element, `alloc_htab_elem` copies the full `key_size` bytes and then copies the **entire** value region (`copy_map_value`, or a rounded-`value_size` `memcpy`). A well-formed, fully-written value therefore completely replaces the previous occupant's bytes. There is no path where you `bpf_map_lookup_elem` key B and read key A's value through the map API. The residual is a *physical-memory* fact (the old bytes sit in the slot until overwritten or scrubbed), not a *logical* leak you can trigger through the API.

## The corners that are not safe

- **The value bytes are not erased.** Delete contains no scrub step, and the lazy allocator recycles its objects without a scrub a BPF program can rely on. If a value ever held secret data, `delete` does not wipe it — the bytes linger in the preallocated pool or the allocator's freelist until reuse. The safe pattern is to zero the value explicitly before releasing the entry.
- **Embedded kernel objects are cancelled, not preserved.** A value that carries a kptr/dynptr/fd has its embedded pointers released on free via `check_and_cancel_fields` / `bpf_obj_cancel_fields`. A program that reads the value after the element is freed or recycled cannot assume the pointer is still valid.
- **LRU maps evict on their own.** `BPF_MAP_TYPE_LRU_HASH` (and `LRU_PERCPU_HASH`) evict the least-recently-used element when the map is full, through the same unlink-and-free path. A key you never called `delete` on can read back `NULL` after eviction.

## How to verify it

1. **Confirm the unlink.** `bpf_map_lookup_elem(key)` returns `NULL` right after `bpf_map_delete_elem(key)`, and `bpf_map_get_next_key` / `bpf_map_get_next_key_batch` never return the deleted key.
2. **Mind the iteration restart.** The docs note that if you pass a `cur_key` that has just been deleted, `bpf_map_get_next_key` restarts from the *first* key in the table; use the batched lookup for a stable sweep.
3. **Contrast with arrays.** `BPF_MAP_TYPE_ARRAY` values are *zero-initialized at creation time* (documented), whereas hash values are neither zeroed at creation nor zeroed at delete — that asymmetry is the whole boundary.
4. **Guarantee clean values.** The only reliable way to erase contents is to write zeros before `delete`/eviction; there is no scrub flag on `bpf_map_delete_elem`.

## Where the answer stops

- This is about value *memory*, not the map API's contract: a deleted or evicted key still reads as `NULL` and is not enumerated, so the *logical* view is clean even when the *physical* bytes are not.
- Whether the slab later scrubs a recycled page is not a BPF-visible guarantee; a program must not depend on it.
- The per-CPU LRU flag (`BPF_F_NO_COMMON_LRU`) and the LRU variants change *which* element evicts, but not the unlink-and-free-without-zeroing mechanism.

## References

- [BPF_MAP_TYPE_HASH, with PERCPU and LRU Variants (kernel docs)](https://docs.kernel.org/bpf/map_hash.html) — default preallocation, `BPF_F_NO_PREALLOC`, delete semantics, the `bpf_map_get_next_key` restart caveat, per-CPU slots.
- [BPF_MAP_TYPE_ARRAY (kernel docs)](https://docs.kernel.org/bpf/map_array.html) — "all array elements are pre-allocated and zero initialized when created."
- [kernel/bpf/hashtab.c](https://raw.githubusercontent.com/torvalds/linux/master/kernel/bpf/hashtab.c) — `htab_map_delete_elem` → `free_htab_elem` → `pcpu_freelist_push` / `htab_elem_free`; `prealloc_init`; the full-value overwrite in `alloc_htab_elem`; `check_and_cancel_fields`.
- [kernel/bpf/memalloc.c](https://raw.githubusercontent.com/torvalds/linux/master/kernel/bpf/memalloc.c) — the lazy BPF memory cache and its RCU-deferred free path.

## Community discussion today

Honest coverage note: the two watchlist-opted Slack archives returned **zero messages** in the rolling window today, and the allowlisted Discord channels plus the public bpf mailing list and r/eBPF remain visible-browser-only, with no visible-browser session available in this run — so no private community material was available for 2026-09-23. The question above is the fallback selection: a genuine, recurring practitioner boundary of the monitored eBPF development community, grounded entirely in the public primary sources above (kernel BPF docs and upstream source) rather than in any thread.

The recurring practitioner theme in that space — "my hash-map values go stale, a deleted key still seems to show old data" — resolves to exactly this boundary: deletion is an *unlink*, not a *zeroing*, and the element's memory is recycled through a per-CPU freelist (prealloc) or an RCU-deferred allocator (lazy). Two public-doc-anchored follow-ups recur here. First, the **LRU** surprise: a key that was never deleted can read back `NULL` once the map is full and the least-recently-used element evicts, which people misread as a delete bug when it is actually capacity eviction. Second, the **secret-handling** implication: because delete does not scrub the value bytes, a map that once held sensitive data still holds them physically until reuse, so the safe pattern is to zero the value (and drop embedded kernel objects explicitly) before releasing the entry. The boundary the docs leave open is that there is no scrub-on-delete API; keeping value memory clean is a program responsibility, not a kernel one.
