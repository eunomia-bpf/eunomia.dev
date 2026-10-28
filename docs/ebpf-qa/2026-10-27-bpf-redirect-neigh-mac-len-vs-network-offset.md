# Why does the BPF redirect neighbor helper trip a kernel BUG on a packet whose Ethernet header is short, and how does the fix read the true header length differently on the receive path than on the transmit path?

Because `skb->mac_len` is only populated on the receive path. A `bpf_redirect_neigh()` fix that guards the pull with `mac_len` under `ETH_HLEN` is safe on ingress, but on the transmit path — locally generated packets through tc egress, or netkit forwarding — `mac_len` is 0 for every skb, so the same guard would silently drop all valid redirect-neighbor traffic from those paths. The header-length field that is valid on both paths is `skb_network_offset()`, the distance from `skb->data` to `network_header`; the corrected fix reads the length from that instead.

## The mechanism

1. The bug. `bpf_redirect_neigh()` runs through `skb_do_redirect()` into `__bpf_redirect_neigh()` in `net/core/filter.c`. The original code did `ethh = eth_hdr(skb)`, checked only `mac_header` below `network_header`, then called `skb_pull(skb, sizeof(*ethh))` — that is a 14-byte `ETH_HLEN` pull. `__skb_pull()` does `skb->len -= len` and then, if the linear part underflows, `BUG()`. After a VLAN pop-eth, a privileged tc program can leave the skb with a 1-byte effective MAC header and one byte of linear headroom (the reproducer is `bpf_skb_change_head(skb, 1, 0)`), then call `bpf_redirect_neigh()`; the old check passed and the 14-byte pull tripped the BUG.

2. The v1 fix and its regression. The fix adds a guard before the pull and re-reads `ethh` after it: drop when `mac_len` is under `ETH_HLEN` or when `pskb_may_pull()` cannot make the header linear, then `ethh = eth_hdr(skb);`. `pskb_may_pull()` linearizes without panicking, and the `mac_len` check drops genuinely short packets. But `skb->mac_len` is written only on the receive path, by `__netif_receive_skb_core()` calling `skb_reset_mac_len()`, which sets it to `network_header` minus `mac_header`. Nothing on the transmit path writes `mac_len`; `__dev_queue_xmit()` and `dev_loopback_xmit()` call `skb_reset_mac_header()` but not `skb_reset_mac_len()`, so a locally generated packet carries `mac_len == 0`. On tc egress and netkit, `0 < ETH_HLEN` is always true, so every valid redirect-neighbor skb falls through to the drop path (`kfree_skb`, `-ENOTSUPP`) — a regression that would also reach stable.

3. The path-independent fix. When `__bpf_redirect_neigh()` runs, `skb->data` sits at the start of the L2 header on both paths: on ingress, `sch_handle_ingress()` pushed `mac_len` bytes before `skb_do_redirect()`; on egress, `dev_hard_header()` already pushed the L2 header and `__dev_queue_xmit()` set `mac_header` to that offset. So the true L2 header length is `network_header` minus `data`, which is exactly `skb_network_offset()`. The corrected guard drops when `skb_network_offset()` is under `ETH_HLEN` or when `pskb_may_pull()` cannot linearize; it still rejects the 1-byte header from `bpf_skb_change_head`, and it is correct on egress and netkit, where `mac_len` is 0.

4. The kernel already knows this. `act_mirred.c` picks `mac_len` at ingress and `skb_network_offset()` on every other path — it already treats `mac_len` as unreliable outside ingress. The `bpf_redirect_neigh` selftests attach at tc ingress, where `mac_len` is valid, so they cannot catch an egress or netkit regression.

## How to verify and debug

1. See which field is zero. On the egress or netkit path, inspect a packet's `mac_len` (0) against `skb_network_offset()` (the real L2 header length) and note which one matches the header. The egress path never calls `skb_reset_mac_len`, so `mac_len` is 0 by construction.

2. Reproduce the original BUG. Attach a privileged SCHED_CLS program at tc ingress, do a VLAN pop-eth, call `bpf_skb_change_head(skb, 1, 0)` then `bpf_redirect_neigh()`; on an unfixed kernel the `skb->len` underflow check inside `__skb_pull()` fires.

3. Prove the regression. Attach the same redirect-neighbor program at tc egress on a locally generated socket packet, or forward it through netkit, and watch the v1 `mac_len` guard drop the packet with `-ENOTSUPP`; the `skb_network_offset()` version keeps it.

## The limitation

- The guard still drops any skb whose L2 header is shorter than `ETH_HLEN`; that is intentional, not a defect.
- `pskb_may_pull()` may reallocate `skb->head`, which is why the fix re-reads `eth_hdr(skb)` after it instead of reusing a pointer taken before the guard.
- A `mac_header` above `network_header` is a distinct failure mode from a too-short header; `skb_network_offset()` covers the too-short case, but the two are not the same check.

## References

- [Kernel `net/core/filter.c`](https://github.com/torvalds/linux/blob/master/net/core/filter.c) — `__bpf_redirect_neigh` and the original `skb_pull(ETH_HLEN)` path.
- [Kernel `include/linux/skbuff.h`](https://github.com/torvalds/linux/blob/master/include/linux/skbuff.h) — `skb_reset_mac_len`, `skb_reset_mac_header`, `skb_network_offset`, `__skb_pull`.
- [Kernel `net/sched/act_mirred.c`](https://github.com/torvalds/linux/blob/master/net/sched/act_mirred.c) — the pick `mac_len` at ingress, `skb_network_offset()` elsewhere.
- [Kernel `include/uapi/linux/if_ether.h`](https://github.com/torvalds/linux/blob/master/include/uapi/linux/if_ether.h) — `ETH_HLEN` (14) and `ETH_ALEN` (6).
- [Commit `b4ab31414970` ("bpf: Add redirect_neigh helper as redirect drop-in")](https://github.com/torvalds/linux/commit/b4ab31414970) — the `Fixes:` target for the redirect-neighbor helper.
- [Public bpf mailing-list patch thread: fix `skb_pull()` BUG in `__bpf_redirect_neigh()`](https://lore.kernel.org/bpf/bpf_redirect_neigh_short_linear.v1.r0.20261028033717-1@mojatatu.com/) — the public patch and the egress/netkit regression review.

## Community discussion today

A public bpf-list patch fixes a `skb_pull()` BUG in `__bpf_redirect_neigh()` that a privileged tc program can trigger by shrinking the MAC header after a VLAN pop-eth. The first version guarded the pull with `mac_len` under `ETH_HLEN`, but an automated review caught that `mac_len` is only set on the receive path and is 0 on tc egress and netkit, so the guard would drop all valid redirect-neighbor traffic from those paths; the thread agreed and a follow-up reads the length from `skb_network_offset()` instead, which is valid on both paths. The discussion so far is narrow: the boundary is which field carries the L2 header length per path, and the existing redirect-neighbor selftests all attach at ingress, so they cannot catch an egress regression.

This run's archive-backed sources produced 0 messages in the 7-day window (the two opt-in archive sources were empty at snapshot time), so the question above was sourced from the permitted public bpf mailing list, reviewed through its public atom feed, and the patch thread was used as the primary reference. The Discord channels for eunomia-bpf and sched-ext and r/eBPF were not covered this run (no visible-browser session was available), so they are marked not-covered rather than quiet.
