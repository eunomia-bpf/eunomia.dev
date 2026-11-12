# Why does a BPF LPM-trie lookup return a less specific stored prefix, and how does the key prefixlen field decide which entry wins?

**Short answer:** Set the lookup key `prefixlen` to the map's maximum prefix length (32 for a 4-byte IPv4 data field, 128 for a 16-byte IPv6 field). The kernel caps the match at `min(node->prefixlen, key->prefixlen)` (`kernel/bpf/lpm_trie.c`), so a lookup key carrying a smaller `prefixlen` tells the walk it may match no more bits than that, and the lookup stops at — or falls back to — a less specific stored entry even when a more specific one also matches. The kernel doc states the rule plainly: for a longest-prefix lookup, `key->prefixlen` should equal `max_prefixlen`. A lookup key whose `prefixlen` exceeds the map maximum returns `NULL` (while `update`/`delete` return `-EINVAL`), and intermediate nodes never carry a value.

## How a lookup walks the trie

`trie_lookup_elem` walks from the root node. Each node holds its `prefixlen` and its `data` (stored big-endian, `data[0]` most-significant), and two child pointers lead to more specific matches. At each node:

- `__longest_prefix_match` counts how many leading bits of the node agree with the key, but the count is capped at `limit = min(node->prefixlen, key->prefixlen)` (line 172). The compare runs over 64/32/16/8-bit big-endian chunks (`be64_to_cpu`, `be32_to_cpu`, …), so a key's `data` byte 0 is the MSB.
- If the match reaches the map's maximum prefix (`matchlen == trie->max_prefixlen`), that node is a full-length exact match and is returned immediately (lines 258–262); this fires only when a stored prefix of the maximum length (e.g. a /32) matches the key.
- If the match is shorter than the node's own `prefixlen` (`matchlen < node->prefixlen`, lines 268–269), the walk breaks and the last *non-intermediate* node it has seen is returned — the longest stored prefix that fully matched on the path so far.
- A node flagged `LPM_TREE_NODE_FLAG_IM` is an artificial intermediate node with no value; it is never returned and exists only so two more specific children can share a parent (lines 271–275).
- Otherwise the last-visited value-carrying node becomes the candidate, and the next bit of the key (`extract_bit(key->data, node->prefixlen)`) selects which child to descend into (lines 281–283).

`max_prefixlen` is `data_size * 8` bits (line 596) — 8× the key's data length, so 32 for a 4-byte key and 128 for a 16-byte key.

## Why the lookup key prefixlen caps the match

The cap is `limit = min(node->prefixlen, key->prefixlen)`. `__longest_prefix_match` returns as soon as its running count reaches `limit` (each chunk ends in `if (prefixlen >= limit) return limit;`). The walk is therefore told that the *key* matches at most `key->prefixlen` bits, no matter how many more matching bits actually exist in a more specific node.

Worked example, using the doc's own numbers: store `192.168.0.0/16` value 1 and `192.168.0.0/24` value 2. The /24 is a more specific prefix on the same path, so it is attached directly as `child[0]` of the /16 (no intermediate node is needed until two more specific children of the same bit range have to share a parent, as in the doc's `192.168.1.0/24` / `192.168.128.0/24` split through a /23).

- Lookup `192.168.0.5` with `prefixlen = 32`: the walk descends to the /24 node, whose own `prefixlen` is 24; the match reaches the node's 24 bits, the node is the last value-carrying node on the path, and there is no deeper child to descend into — so the /24 is returned, value 2, the longest stored match. (The exact-match branch only fires if a stored prefix of the full 32-bit length, a /32, matched the key.)
- Lookup `192.168.0.5` with `prefixlen = 16`: the walk reaches the /24 node, but the match is capped at 16; now `matchlen (16) < node->prefixlen (24)`, so the walk breaks and returns the last found node — the /16, value 1. The more specific /24 entry is skipped silently.
- Lookup with `prefixlen = 33` (above the 32 maximum): `trie_lookup_elem` returns `NULL` at line 244; `update`/`delete` return `-EINVAL` (lines 338, 469).

The doc's one-line rule: "The `key` should have `prefixlen` set to `max_prefixlen` when performing longest prefix lookups."

## The corners that are not safe

- **A lookup with a `prefixlen` below the map maximum caps the result.** Capping the key at a stored prefix's length makes the trie stop at the longest *uncapped* match, so a more specific stored prefix can be bypassed. Always pass the maximum (32/128).
- **`prefixlen` above the map maximum is a no-match, not an error, on lookup.** The lookup returns `NULL` (line 244); only `update`/`delete` surface it as `-EINVAL`.
- **Key data is big-endian.** The compare is `be32_to_cpu` over the stored and lookup arrays and `data[0]` is the MSB; a little-endian host that stores the address as a native integer will misroute. The canonical key struct is `struct bpf_lpm_trie_key_u8` (`prefixlen` + flexible `data[]`); the older `struct bpf_lpm_trie_key` is deprecated.
- **Intermediate nodes carry no value.** They are internal structural nodes (the `LPM_TREE_NODE_FLAG_IM` bit) and are never returned; a trie with many splits returns the closest non-intermediate ancestor's value even when only the leaf holds a value.
- **The map must be created with `BPF_F_NO_PREALLOC`.** The kernel doc requires the flag; the maximum prefix length is a multiple of 8 in the range 8 to 2048 bits (`LPM_DATA_SIZE_MAX` is 256 bytes, line 557).

## How to verify it

1. **Exact match wins.** Store `192.168.0.0/16` (value 1) and `192.168.0.0/24` (value 2). `bpf_map_lookup_elem` with key `192.168.0.5` and `prefixlen = 32` returns value 2.
2. **The cap.** Same two entries; key `192.168.0.5` with `prefixlen = 16` returns value 1 — the /24 is skipped because the match is capped at 16.
3. **Above max.** Key `prefixlen = 33` makes the lookup return `NULL`; `bpf_map_update_elem`/`bpf_map_delete_elem` return `-EINVAL`.
4. **Iteration order.** `bpf_map_get_next_key` walks leftmost-leaf-first, so more specific keys come before less specific ones; pass `NULL` for the first call. The upstream selftest `tools/testing/selftests/bpf/test_lpm_map.c` exercises these paths.

## Where the answer stops

- This is about *lookup* semantics: storage is unaffected — a stored /24 is still there and reachable by a correctly-capped lookup. The cap only limits how many bits the *lookup* is allowed to match.
- The trie is unbalanced; a fully-populated IPv4 trie has height 32 (one level per bit). The cap is a bound on the match a given lookup can express, not a storage-correctness bug.
- Other map types have no notion of a prefix; this boundary is specific to `BPF_MAP_TYPE_LPM_TRIE`.

## References

- [BPF_MAP_TYPE_LPM_TRIE (kernel docs)](https://docs.kernel.org/bpf/map_lpm_trie.html) — the `prefixlen`/`max_prefixlen` rule, big-endian `data`, the `BPF_F_NO_PREALLOC` requirement, the `bpf_map_lookup_elem`/`update`/`delete`/`get_next_key` contracts, IPv4/IPv6 data lengths, the `192.168.0.0` worked example, and the selftest pointer.
- [kernel/bpf/lpm_trie.c](https://raw.githubusercontent.com/torvalds/linux/master/kernel/bpf/lpm_trie.c) — `trie_lookup_elem` (the walk, the `matchlen` cap, the intermediate-node flag); `__longest_prefix_match` (`limit = min(node->prefixlen, key->prefixlen)` at line 172, big-endian word compare); `trie->max_prefixlen = trie->data_size * 8` (line 596); `LPM_DATA_SIZE_MAX 256` (line 557); the `key->prefixlen > max_prefixlen` guards (lines 244, 338, 469).
- [include/uapi/linux/bpf.h](https://raw.githubusercontent.com/torvalds/linux/master/include/uapi/linux/bpf.h) — `struct bpf_lpm_trie_key` (deprecated), `struct bpf_lpm_trie_key_hdr`, and `struct bpf_lpm_trie_key_u8`; `BPF_MAP_TYPE_LPM_TRIE` in `enum bpf_map_type`.

## Community discussion today

Honest coverage note: the two watchlist-opted Slack archives were **inaccessible** this run — the read-only snapshot reader refused to overwrite a pre-existing 0-byte snapshot file, so the Step 0 snapshot returned `output_exists` and no archive content was read. The allowlisted Discord channels and public lists remain visible-browser-only, with no visible-browser session available in this run — so no private community material was available for 2026-09-24. The question above is the fallback selection: a genuine, recurring practitioner boundary of the monitored eBPF development community, grounded entirely in the public primary sources above (kernel BPF docs and upstream source) rather than in any thread.

The recurring practitioner theme in that space — building a CIDR/route table in an XDP or TC datapath and asking why the less specific route wins — resolves to exactly this boundary: the lookup key `prefixlen` caps the longest-prefix match, and the doc's rule is to set it to `max_prefixlen` (32 for IPv4, 128 for IPv6) so the more specific stored prefix actually wins. Two public-doc-anchored follow-ups recur here. First, the **big-endian** surprise: the `data` array is interpreted in network byte order, so a host that stores the address as a native little-endian integer silently misroutes, and people misread it as "the prefix did not match" when it is a byte-order bug. Second, the **above-maximum** corner: passing a `prefixlen` above the map's maximum makes the lookup return `NULL` (not an error), which reads as "no match" when it is a key-shape bug, while `update`/`delete` return `-EINVAL` on the same out-of-range value, so the two paths disagree on the identical key.
