# Why does BPF hash map key iteration return keys in an order that looks random instead of insertion order?

Short answer: a BPF hash map stores keys in open-addressed buckets keyed by a hash of the key bytes, not in any order visible to the caller. The kernel hashes the key with jhash using a per-map random seed chosen at map creation, and new keys are pushed to the head of that bucket's collision chain. `bpf_map_get_next_key` walks the buckets by ascending bucket index and hands back the first element of each nonempty chain, so the order is determined by hash scattering plus chain-head insertion, and the random seed makes it differ between maps and between boots. The kernel guarantees coverage of every key, not any particular sequence; the API has no cursor, so a walk always begins with a `NULL` key.

## The mechanism

**Buckets are chosen by hash, with a random per-map seed.** Map allocation hashes keys with jhash, fed a seed that is random unless the map was explicitly created test-only:

```c
htab->n_buckets = roundup_pow_of_two(htab->map.max_entries);
...
if (htab->map.map_flags & BPF_F_ZERO_SEED)
	htab->hashrnd = 0;
else
	htab->hashrnd = get_random_u32();
```

The lookup path reduces that to a bucket index with a mask:

```c
static inline u32 htab_map_hash(const void *key, u32 key_len, u32 hashrnd)
{
	if (likely(key_len % 4 == 0))
		return jhash2(key, key_len / 4, hashrnd);
	return jhash(key, key_len, hashrnd);
}
```

`jhash` and `jhash2` mix the seed into their initial state, so two maps holding identical keys land in different bucket layouts unless both are created with the zero-seed flag.

**Insertion places new elements at the chain head.** An update allocates the element and pushes it before the old one in the same bucket's null-terminated RCU list, with the comment spelling out why:

```c
/* add new element to the head of the list, so that
 * concurrent search will find it before old elem
 */
hlist_nulls_add_head_rcu(&l_new->hash_node, head);
```

So within one bucket, the most recently inserted keys for that bucket come first. The order across buckets is not an insertion order at all.

**The iterator walks bucket indices, not a history list.** `htab_map_get_next_key` locates the bucket of the given key, returns the next element in that same bucket if one exists, and otherwise scans bucket indices upward until it finds a nonempty chain; a `NULL` key skips straight to the first nonempty bucket:

```c
static int htab_map_get_next_key(struct bpf_map *map, void *key, void *next_key)
{
	...
	if (!key)
		goto find_first_elem;

	hash = htab_map_hash(key, key_size, htab->hashrnd);
	head = select_bucket(htab, hash);

	l = lookup_nulls_elem_raw(head, hash, key, key_size, htab->n_buckets);
	if (!l)
		goto find_first_elem;

	/* key was found, get next key in the same bucket */
	next_l = hlist_nulls_entry_safe(rcu_dereference_raw(hlist_nulls_next_rcu(&l->hash_node)),
				  struct htab_elem, hash_node);
	if (next_l) {
		memcpy(next_key, next_l->key, key_size);
		return 0;
	}

	/* no more elements in this hash list, go to the next bucket */
	i = hash & (htab->n_buckets - 1);
	i++;

find_first_elem:
	for (; i < htab->n_buckets; i++) {
		head = select_bucket(htab, i);
		next_l = hlist_nulls_entry_safe(rcu_dereference_raw(hlist_nulls_first_rcu(head)),
					  struct htab_elem, hash_node);
		if (next_l) {
			memcpy(next_key, next_l->key, key_size);
			return 0;
		}
	}

	/* iterated over all buckets and all elements */
	return -ENOENT;
}
```

Three properties combine: the hash seed is random per map instance, so keys scatter into buckets in an order that is unpredictable from the key values alone; insertion always lands at the head of the chain, so recency inside a bucket is visible but never across buckets; and the walk itself advances by bucket index, which is a function of the hash, not of when anything was inserted. The same function is registered in the `bpf_map_ops` blocks for the plain, per-CPU, LRU, and hash-of-maps variants through the `map_ops` registration, so all of those iterate the same way.

## Verification and debugging path

1. Insert a known key sequence and record the iteration order. Insert, say, 16 integer keys in ascending order, then iterate with `bpf_map__get_next_key` from `NULL` until `-ENOENT`, logging each key. The order will almost certainly not be the insertion order.
2. Prove the seed is the source of the variation. Create a second map of the same size, insert the identical keys, and iterate. The two sequences should differ. In a test-only environment, creating the maps with `BPF_F_ZERO_SEED` makes both sequences identical, which isolates the random seed as the distinguishing factor.
3. Exercise the contract. Confirm that starting from `NULL` returns some key, that stepping from a returned key eventually covers every inserted key exactly once before `-ENOENT`, and that the batch `BPF_MAP_GET_NEXT_KEY` syscall path shows the same unordered behavior.
4. Cross-check the variants. `BPF_MAP_TYPE_LRU_HASH`, the per-CPU hash variants, and `BPF_MAP_TYPE_HASH_OF_MAPS` are all registered through the same `htab_map_get_next_key` walk, so they iterate with the same unordered semantics; LRU eviction is layered on the elements and does not add any ordering. The one hash-family type with a genuinely different walk is `BPF_MAP_TYPE_RHASH`, which is backed by rhashtable and iterates through `rhashtable_next_key`, which is equally unordered.

A minimal walk in libbpf. The high-level helper takes the `bpf_map` object; the low-level syscall wrapper takes the file descriptor:

```c
#include <bpf/bpf.h>
#include <linux/bpf.h>

__u32 cur = 0, next;
int map_fd;
struct bpf_map *map; /* the libbpf object for the same map */

while (1) {
	int err = bpf_map__get_next_key(map, cur ? &cur : NULL, &next, sizeof(next));
	/* the low-level form is bpf_map_get_next_key(map_fd, &cur, &next);
	 * it returns -ENOENT when the walk is exhausted */
	if (err)
		break; /* -ENOENT at the end of the iteration */
	cur = next;
	/* record the key; do not assume it arrived in insertion order */
}
```

## The limitation

- The walk is stateless. Each `bpf_map_get_next_key` call is a fresh lookup of the previous key plus a bucket scan; there is no opaque cursor that survives map mutations. If entries are deleted or added while you walk, the sequence you see is whatever the hash table happens to look like at that moment.
- `BPF_F_ZERO_SEED` is documented for testing only. It is not a supported way to make application-level iteration reproducible; it zeroes the jhash seed so the bucket layout is deterministic given the same keys, and the kernel uapi comment says it should only be used for testing.
- The unordered contract holds across every htab-family type. `BPF_MAP_TYPE_HASH`, `PERCPU_HASH`, `LRU_HASH`, `LRU_PERCPU_HASH`, and `HASH_OF_MAPS` are all bound to the same `htab_map_get_next_key`, so none of them can promise an order; the only distinct hash-family walk is `BPF_MAP_TYPE_RHASH` (rhashtable-backed), which is likewise unordered.

## References

- `htab_map_get_next_key`, bucket allocation, `get_random_u32` seed, chain-head insertion: https://elixir.bootlin.com/linux/latest/source/kernel/bpf/hashtab.c
- `jhash` / `jhash2` seed mixing: https://elixir.bootlin.com/linux/latest/source/include/linux/jhash.h
- `BPF_F_ZERO_SEED` uapi flag and `BPF_MAP_GET_NEXT_KEY` command documentation: https://elixir.bootlin.com/linux/latest/source/include/uapi/linux/bpf.h
- `bpf_map__get_next_key` userspace helper (NULL key = first key, `-ENOENT` on exhaustion): https://github.com/libbpf/libbpf/blob/master/src/libbpf.h
- `bpf` syscall page: https://man7.org/linux/man-pages/man2/bpf.2.html

## Community discussion today

Coverage: **0** watchlist-opted-in community archives were reachable in this run's rolling window — no 2026-09-29 snapshot file existed at run start, and the browser-only communities in the allowlist had no visible-browser session available and are marked **unavailable, not quiet**. No workspace, channel, participant, or message URL is reproduced below.

The question for this entry instead comes from the running Q&A series itself: the 2026-09-25 entry already covered why a full hash map rejects new keys with `E2BIG` while an update still succeeds, and that answer made the hash-table internals the standing context. The follow-up a reader of that page would naturally ask next is what order the map's keys come back in when you walk it. The answer above is grounded in the public primary sources in the References section rather than in a monitored community thread; no discussion window was available to summarize, so nothing is invented to fill the gap.
