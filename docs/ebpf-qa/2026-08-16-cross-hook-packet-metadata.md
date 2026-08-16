# How should eBPF programs carry per-packet metadata across networking hooks?

**Short answer:** there is no universal upstream BPF scratch area that follows an `skb` through every networking hook today. Use `data_meta` when the producer is XDP and the consumer is TC ingress, use a map keyed by a stable flow or socket identifier when the state is not truly packet-local, and treat `skb->cb` as short-lived layer-owned storage rather than a cross-stack ABI. A proposed BPF `skb` extension would add lifecycle-bound per-packet storage, but it is still under review and must not be treated as an available production interface yet.

The first design decision is therefore not which helper to call. It is how long the metadata must live and what object owns it.

## Choose storage by lifetime, not convenience

| Required scope | Practical upstream choice | Important boundary |
| --- | --- | --- |
| One program invocation | Registers and BPF stack | Gone when the invocation returns |
| XDP to TC ingress on the same packet path | Reserve bytes with `bpf_xdp_adjust_meta()` and read them through `data_meta` | Not a general facility for later socket, LSM, tracing, or netfilter hooks |
| Several packets in one flow or socket | Hash, LRU, socket storage, or another map keyed by a stable identifier | State is no longer tied automatically to one packet's allocation and lifetime |
| A tightly controlled, adjacent `skb` processing stage | A deliberately owned part of the control buffer | `skb->cb` is reused by kernel layers and is not a durable cross-stack contract |
| Arbitrary `skb`-based hooks | No universal upstream BPF interface | Requires an explicit correlation design or a kernel feature that is not upstream yet |

For XDP-to-TC handoff, the documented path is narrow but useful. An XDP program moves `data_meta` backward with `bpf_xdp_adjust_meta()`, writes a fixed application-defined structure before packet data, and a TC program validates the metadata bounds before reading it. The kernel documentation explicitly says TC-BPF can access this area after `XDP_PASS`. The same documentation also describes important limits around redirects, AF_XDP descriptors, and driver-provided metadata. Do not generalize this contract to every later `skb` hook.

Maps solve a different problem. A map can share state among programs and user space, but it needs an identity and a cleanup policy. A socket cookie, connection tuple plus generation, or application-created correlation ID can work when the desired state belongs to a flow. Such a key does not make the value packet-local: retransmission, fragmentation, NAT, tunnel transitions, and concurrent packets may all change what “the same flow” means.

Keying a map by an `skb` address is especially fragile. Cloning, copying, segmentation, and object reuse can break the assumed identity. Cleanup must cover every release path before the address can be reused, which is why designs based on `consume_skb` or `kfree_skb` tracing need careful loss and race analysis. Use that pattern only in a controlled experiment where its lifecycle assumptions are measured, not as a portable ABI.

The kernel's own `sk_buff` control buffer is also not a general solution. Its source documentation says that each layer may put private data there and that the current queue owner owns it. A value written by one controlled stage can be useful to the next stage, but carrying it through unrelated networking layers risks collision or overwrite. BPF-visible `cb` fields should therefore have an explicit, local ownership contract and a short path.

## What the proposed BPF `skb` extension changes

The active proposal adds a dedicated `skb` extension for BPF metadata. Its first patch describes a build-time-sized buffer, exposed through a new `bpf_dynptr_from_skb_ext()` kfunc. A caller that requests creation gets storage and a writable dynptr; creating on an already shared extension performs copy-on-write. A reader that does not request creation receives a read-only dynptr or `-ENOENT` when the packet has no extension.

That model addresses three weaknesses of an address-keyed map:

- storage is attached to the packet object rather than reconstructed from an external key;
- the kernel frees it with the `skb`; and
- clone handling can share data until a writer requests an unshared copy.

The series also proposes access from a broad set of `skb`-based program types and tests survival across clones, virtual Ethernet traversal, tunneling, cgroup and socket paths, LWT, netfilter, tracing, LSM, and stream-verdict processing. A later revision deliberately keeps the BPF extension across packet scrubbing rather than silently deleting it during tunnel or virtual-device transitions.

Those are proposal semantics, not an upstream guarantee. At the time of this review, the series is still receiving mailing-list feedback, and the current upstream UAPI header does not expose either `bpf_dynptr_from_skb_ext()` or its creation flag. Production software should feature-detect the exact kfunc and verify that a minimal object loads. A kernel version string alone is insufficient, especially for distribution backports and experimental kernels.

## Design a portable metadata record

Whether the transport is `data_meta`, a map, or a future `skb` extension, make the record self-describing and bounded:

```c
struct packet_meta_v1 {
    __u16 version;
    __u16 length;
    __u32 flags;
    __u64 correlation_id;
};
```

The producer should initialize every byte, set `length` to the bytes actually written, and publish the version last when ordering matters. The consumer should reject unknown versions, lengths smaller than the required prefix, or flags outside the supported mask. Do not put pointers, credentials, packet payload copies, or mutable userspace addresses in the record.

Keep fields semantic rather than hook-specific. A correlation ID or classification result can survive a change in attachment point; a raw pointer or parser offset usually cannot. If multiple programs can write the metadata, allocate explicit field ownership or use one producer and read-only consumers. Hidden last-writer-wins behavior is difficult to diagnose even if the underlying storage is race-safe.

## Verify the actual packet paths

A useful test matrix should prove both retention and loss boundaries:

1. Write at XDP, pass the packet, and read the record at TC ingress.
2. Redirect through the devices and maps that production uses; verify which metadata remains and which hardware metadata is unavailable.
3. Exercise clone and copy paths, then mutate one copy and confirm whether copy-on-write isolation is present.
4. Test encapsulation, decapsulation, namespace traversal, GRO/GSO, segmentation, retransmission, and error drops separately. Do not infer one from another.
5. Confirm that a packet without metadata produces an explicit miss rather than zero-filled data that looks valid.
6. Run repeated attach, detach, queue reset, and failure injection while tracking allocation balance.
7. Benchmark with the real fraction of packets that carry metadata. An opt-in allocation may be inexpensive when sparse and costly when nearly every packet uses it.

For an experimental `skb` extension kernel, add a load-time capability probe and record the feature in telemetry. If the kfunc is absent, fail closed or select a documented map-based mode whose weaker lifetime semantics are visible to the operator. Never silently switch implementations while reporting identical guarantees.

The durable rule is simple: use the narrowest storage whose lifetime matches the data. `data_meta` is an XDP handoff mechanism, maps are correlation state, and `skb->cb` is locally owned scratch space. A BPF `skb` extension could become the missing per-packet cross-hook primitive, but only after its ABI and lifecycle semantics land upstream.

## References

- [Linux kernel documentation: XDP RX metadata and the XDP-to-TC `data_meta` path](https://docs.kernel.org/networking/xdp-rx-metadata.html)
- [Linux kernel source: `sk_buff` layout, control-buffer ownership, cloning, and extensions](https://github.com/torvalds/linux/blob/master/include/linux/skbuff.h)
- [Linux kernel UAPI header: current BPF contexts and interfaces](https://github.com/torvalds/linux/blob/master/include/uapi/linux/bpf.h)
- [Linux kernel documentation: BPF maps](https://docs.kernel.org/bpf/maps.html)
- [Linux kernel documentation: BPF kfunc and dynptr annotations](https://docs.kernel.org/bpf/kfuncs.html)
- [BPF mailing-list proposal: `skb` extension for per-packet BPF metadata](https://lore.kernel.org/bpf/20260814-bpf-meta-inside-skb-ext-v1-0-767edd862656%40cloudflare.com/T/#t)
- [Proposal implementation: `bpf_dynptr_from_skb_ext()` and lifecycle semantics](https://lore.kernel.org/bpf/20260814-bpf-meta-inside-skb-ext-v1-1-767edd862656%40cloudflare.com/T/#t)
- [BPF mailing list: follow-up corrections to structured verifier diagnostics](https://lore.kernel.org/bpf/20260816015746.2632990-1-memxor%40gmail.com/T/#t)
- [BPF mailing list: rejecting one atomic instruction reached through incompatible pointer types](https://lore.kernel.org/bpf/20260816-bpf-next-038-mixed-atomic-v1-v2-0-4644c1886dbc%40mails.tsinghua.edu.cn/T/#t)
- [BPF mailing list: XDP and AF_XDP lifecycle fixes in a network driver](https://lore.kernel.org/bpf/CALuQH%2BWN%2BxftDONAk%3DT8zeB1qF5y%3DyaJ4F23DAg4W6puP%2BjToA%40mail.gmail.com/T/#t)

## Community discussion today

Today's ordinary visible-browser review covered all 6 approved communities and all 15 allowlisted channels or public pages. Every target was accessible. The public forum was reviewed through its ordinary visible legacy interface. The selected question came from the 24-hour window, so the seven-day fallback was not used. Names, accounts, employers, channel identities, message links, exact times, private topology, original logs, and searchable phrasing have been removed. No raw transcript was retained.

### Packet metadata needs an owner and a lifetime

The strongest discussion examined how BPF programs at different networking hooks can share data about one packet without maintaining an external address-keyed cache. The proposed answer is a BPF-specific `skb` extension accessed through a dynptr. Its value is not merely extra bytes: allocation, copy-on-write, packet cleanup, and read-only lookup make the metadata follow the kernel object that owns it.

The unresolved boundary is compatibility. The proposal is still being reviewed, its selftests are receiving correctness feedback, and current upstream kernels do not provide the new kfunc. Practitioners who need this today should first classify the data as packet-, flow-, or socket-scoped. XDP-to-TC metadata and stable-key maps cover many real deployments, but neither should be described as a universal cross-hook packet store. For experimental kernels, test clone, scrub, tunnel, namespace, segmentation, and free paths rather than relying on a single loopback success case.

### Verifier diagnostics must preserve the reason, not just the rejection

Another active series corrected cases where structured verifier output attributed a failure to the wrong mechanism. A read of verifier-managed stack state could be mislabeled as uninitialized memory, while a variable-offset atomic stack access could be described as a helper call. The practical effect is wasted remediation: initializing bytes or changing capabilities cannot fix an attempt to read an opaque dynptr or iterator representation.

The proposed corrections classify the stack slot first, retain ordinary uninitialized-memory advice only for genuinely invalid bytes, and identify atomic accesses as atomic. A related fix records the destination pointer type for every atomic read-modify-write path. Without that record, an arena pointer and a normal stack pointer can converge on one instruction while post-verification fixup selects an encoding based on only one path. The diagnostic path and the safety path therefore share the same requirement: retain type and source attribution at the exact instruction where control-flow states merge.

For debugging, preserve the first structured rejection, instruction index, program type, and target-kernel BTF. Reproduce on the kernel that will run the program, because verifier acceptance and diagnostics can change together. What remains open is how quickly these richer diagnostics will reach stable and distribution kernels; tools should keep parsing the traditional verifier log as a fallback without inventing a root cause.

### XDP correctness includes budgets, ownership, and teardown ordering

The day's driver fixes showed why a dataplane can pass packet tests and still fail under sustained load or reconfiguration. If XDP-consumed packets are not counted against the NAPI budget, a poll loop can run far longer than intended, monopolize a CPU, and delay AF_XDP transmit work. If XSK buffers are not released on error descriptors or ring shutdown, queue cycling leaks memory. Disabling queues or unmapping DMA in the wrong order can instead deadlock or expose a stale pool pointer.

The practical validation path is broader than throughput: run a workload dominated by non-`XDP_PASS` actions and observe NAPI rescheduling, repeatedly bind and unbind XSK pools, reset queues under traffic, inject RX errors, and verify that DMA unmap occurs only after queues stop. Track buffer accounting before and after every failure path. These fixes are driver-specific, but the engineering rule generalizes: every descriptor needs one budget decision, one owner, and one terminal release path.

The chat and project-focused surfaces were otherwise quiet in the daily window or contained automated build notifications rather than new practitioner questions. The public forum's newest unresolved technical post fell outside the 24-hour window, so it was not used to replace the stronger same-day upstream discussion.
