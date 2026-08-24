---
date: 2026-08-24
title: "Who Owns a Packet Buffer in a Zero-Copy eBPF Datapath?"
description: "AF_XDP, io_uring ZC Rx, and DPDK recycle buffers differently. This report develops ownership capabilities and policy provenance for zero-copy eBPF paths."
tags:
  - Daily Report
  - eBPF
  - Networking
  - Security
  - XDP
  - Zero Copy
research_question: "How should a zero-copy eBPF networking path represent buffer ownership, DMA reachability, recycling, and policy provenance across kernel, userspace, and NIC handoffs?"
source_cutoff: 2026-08-24
status: daily-report
---

# Who Owns a Packet Buffer in a Zero-Copy eBPF Datapath?

A packet enters an XDP program, gets redirected into AF_XDP, is inspected by a userspace dataplane, and later goes back to a NIC. The application sees the same UMEM address several times. The bytes may never be copied. That is the performance win.

It also creates a less obvious systems question: **at each moment, who is allowed to touch that buffer, which device may DMA into it, when may it be recycled, and which policy decision still applies to the bytes now occupying that address?**

<!-- more -->

Linux already has strong local answers for parts of this problem. [AF_XDP](https://docs.kernel.org/networking/af_xdp.html) uses explicit FILL and COMPLETION rings to transfer ownership of UMEM frames between userspace and the kernel. [io_uring zero-copy Rx](https://docs.kernel.org/networking/iou-zcrx.html) receives TCP payload directly into registered userspace memory and uses a refill ring to return consumed buffers. The kernel [page_pool](https://docs.kernel.org/networking/page_pool.html) tracks in-flight packet pages, DMA mapping, synchronization, and recycling. DPDK has its own packet-buffer model built around `rte_mbuf`, mempools, reference counts, and external-buffer lifetime callbacks.

Each mechanism is reasonable inside its own path. The problem appears when a high-performance dataplane crosses several paths and expects one security or provenance story to survive them.

Consider AF_XDP first. An XDP program can redirect an ingress frame through an `XSKMAP` into userspace. The AF_XDP documentation is unusually explicit about ownership: the FILL ring transfers a UMEM frame from userspace to the kernel; the COMPLETION ring transfers a processed TX frame back to userspace. Shared UMEM is possible across sockets and queues, but its producer/consumer rules need explicit synchronization. The documentation also warns that placing one frame into conflicting ownership paths can corrupt packets because the NIC may receive or transmit through the same memory concurrently.

Now compare io_uring ZC Rx. It deliberately keeps TCP header processing in the kernel while delivering payload into userspace memory without the kernel-to-user copy. The NIC must support header/data split, RSS, and flow steering. The current kernel interface does not configure that NIC state; the user sets it up out of band. Consumed buffers are returned through the refill ring and become available to the kernel again.

DPDK moves the boundary farther into userspace. Its current programmer documentation describes a userspace packet-processing framework whose `mbuf` objects come from memory pools. Its external-buffer API uses a separate reference count and a free callback so the buffer can be released only after all attached mbufs detach. That is another valid lifetime protocol, but it is not the AF_XDP protocol and it is not the io_uring ZC Rx protocol.

The common mental model is that zero copy is mainly a data-movement optimization. If every subsystem already has a ring, reference count, or page lifetime rule, then correctness should be a local implementation detail.

That model is incomplete for programmable networking. A BPF program may make a security decision before an AF_XDP handoff. NIC steering may then move a flow to a queue configured outside the BPF control plane. A userspace runtime may apply another policy. The same physical or virtual address can later hold an unrelated packet after recycling. Local lifetime rules prevent many use-after-free bugs, but they do not automatically preserve **policy generation, provenance, or authority across the handoff**.

This report argues for treating a zero-copy buffer handoff as a typed capability transition, not only as a queue operation. The capability should be generation-scoped, cheap enough for fast paths, and rich enough to answer three questions after an incident: who owned the buffer, which policy authorized the transition, and whether the address had already been recycled into a different logical packet.

This is distinct from the earlier [io_uring eBPF programmability report](https://eunomia.dev/research/io-uring-bpf-programmability/), which asked what control surface BPF should have inside an io_uring execution loop. It is also distinct from the [heterogeneous eBPF placement report](https://eunomia.dev/research/heterogeneous-ebpf-execution-placement/), which asked where a BPF program should execute. Here the execution location may already be chosen. The missing property is a common lifetime and provenance contract for the memory crossing those locations.

## Zero copy already means explicit ownership transfer

The first useful observation is that zero copy does not remove ownership. It makes ownership more important because the same memory is reused aggressively.

### AF_XDP makes the transfer visible in rings

An AF_XDP socket connects an XDP program to a UMEM region supplied by userspace. RX and TX descriptors refer to offsets in that UMEM. The kernel documentation says the FILL and COMPLETION rings transfer ownership of frames between userspace and the kernel.

That wording matters. A FILL entry is not just a free-list hint. Once userspace publishes the address, the kernel may use that frame for ingress. A COMPLETION entry gives the frame back to userspace after the kernel has finished processing the corresponding TX descriptor. The documentation also notes that completion does not prove successful transmission, only that the frame can be reused.

Shared UMEM makes the ownership topology more interesting. Several sockets can reference the same memory, including sockets on different queues or devices. The rings are single-producer/single-consumer, so applications need their own synchronization. AF_XDP therefore already exposes a small distributed ownership protocol among an XDP program, kernel socket machinery, userspace workers, and the NIC.

The protocol is path-local, however. It does not say that a frame was admitted under policy generation 84, redirected by rule 19, then inspected by userspace policy generation 52 before reuse.

### io_uring ZC Rx splits control and data paths

io_uring ZC Rx exposes a different boundary. Packet headers stay in the normal kernel TCP stack while payload is placed directly in registered userspace memory. The application later returns consumed areas to the kernel through the refill ring.

The NIC setup is currently out of band. Users configure header/data split, RSS, and flow steering before registering the io_uring queue. This makes a useful counterexample to a pure BPF-centric model: the packet's actual path can depend on hardware queue state that is not itself represented in the BPF policy objects.

A policy debugger that sees only the XDP or socket decision can therefore miss a queue-steering change that moved traffic into or out of the zero-copy path. The bytes are correct and the buffer lifetime may be correct, while the explanation of *why this flow followed this path* is incomplete.

### page_pool protects a kernel-local recycling contract

The kernel page_pool allocator is optimized for pages and fragments used by skb and XDP frames. Its documentation describes in-flight accounting, reference handling, DMA mapping, device synchronization, and direct recycling in safe contexts such as NAPI.

This is useful evidence that high-performance packet memory already needs more than `malloc` and `free`. The allocator tracks whether a page can be recycled and when DMA synchronization must happen. But page_pool intentionally solves a kernel networking allocator problem. It does not define a portable logical packet identity across AF_XDP, io_uring, a userspace BPF runtime, or DPDK.

### DPDK uses a different userspace lifetime model

DPDK's `rte_mbuf` model puts packet metadata in userspace and uses mempools for reuse. External buffers have their own shared information, reference count, and free callback. The callback runs when attached mbufs no longer reference the buffer.

That protocol can express sharing that would be awkward with a single owner bit. It also illustrates the abstraction mismatch: AF_XDP communicates ownership through rings, page_pool through kernel references and recycle APIs, io_uring ZC Rx through completion/refill state, and DPDK through mbuf ownership and reference counts.

Trying to replace all of them with one allocator would be the wrong goal. The more useful abstraction is a **common handoff contract above the allocator**, with adapters that preserve the native fast path.

## The hard bug is stale meaning, not only stale memory

Suppose UMEM frame `0x4000` first carries a packet from tenant A. An XDP program running policy generation 84 redirects it to an AF_XDP socket. Userspace consumes the packet, the frame returns through a completion or refill path, and later the same address carries a packet from tenant B under policy generation 85.

A trace that stores only `addr=0x4000` cannot safely join the two events. The address is a storage location, not a stable packet identity.

This is the same kind of mistake that appears when a PID is treated as a process identity across reuse, or when a map entry is interpreted without its configuration generation. Zero-copy networking amplifies it because address reuse is intentional and frequent.

The minimal logical identity needs at least a generation or lease epoch:

```text
buffer_ref = {
  region_id,
  offset,
  lease_generation
}
```

A handoff can then carry the policy and execution context that was true for that lease:

```text
handoff = {
  buffer_ref,
  from_owner,
  to_owner,
  queue_or_device,
  dma_domain,
  policy_generation,
  decision_id,
  transition_seq
}
```

This metadata should not travel in every packet header. It can live in a bounded side table, a userspace manifest, sampled transition events, or a compact token encoded in metadata that a particular path already carries. The research question is how little state is sufficient to prove the important invariants.

## Where current work is still weak

### 1. Path-local ownership protocols do not compose into one cross-boundary contract

AF_XDP, io_uring ZC Rx, page_pool, and DPDK each define when their own buffer can be reused. They do not share one notion of owner, lease generation, DMA reachability, or handoff authority.

The missing capability is not a universal memory allocator. It is a machine-readable transition model that can say, for example, "userspace relinquished this AF_XDP frame to kernel RX", "the NIC may DMA into this lease", and "this lease was later recycled and must no longer inherit the earlier packet's policy witness."

A decisive experiment would connect two or more native zero-copy paths, inject ownership mistakes at their boundary, and compare a common contract against the unmodified path-local APIs. If all injected faults are already rejected at the native boundary with equally useful diagnostics, the extra abstraction is unnecessary.

### 2. Buffer addresses are reused faster than policy provenance

Fast paths want to recycle addresses. Incident analysis wants stable identities. Those goals conflict if observability joins events by address alone.

The missing property is a generation-scoped buffer identity that survives long enough to relate a BPF decision to the exact logical packet lease, then expires when the memory is recycled. A stable physical address should be allowed to produce many distinct logical identities over time.

The test is straightforward: recycle a small UMEM aggressively while changing policy generations and interleaving tenants. An analyzer should never attribute a generation-84 decision to generation-85 bytes merely because the address matches.

### 3. Hardware steering can change the zero-copy boundary outside the BPF control plane

io_uring ZC Rx currently requires out-of-band NIC configuration for header/data split, RSS, and flow steering. AF_XDP can also share UMEM across queues and devices. DPDK may bypass the normal kernel networking path entirely.

The missing capability is a compact description of the *realized path* that can be joined with BPF policy state: NIC queue, steering generation, memory region, and the handoff that moved the packet into userspace. Without it, a security policy may be correct at one hook while the operator's model of which traffic reaches that hook is stale.

A discriminating evaluation would mutate steering rules while keeping BPF policy constant. If the system's explanation cannot tell that traffic moved to a different queue or userspace path, the policy observability boundary is incomplete.

### 4. Zero-copy failures are hard to diagnose without copying the payload

The obvious debugging response is to capture packets. That can destroy the performance property being debugged, expose sensitive payloads, and still fail to explain buffer ownership.

The missing mechanism is metadata-first evidence: ownership transitions, lease generations, queue identities, policy witnesses, and drop/recycle reasons without retaining packet contents by default.

The test should compare full packet capture, ordinary counters, and metadata-only witnesses under the same fault set. A useful witness design should diagnose double reuse, stale policy attribution, and wrong-path steering with much less data than payload capture.

## Promising directions with academic and production value

### 1. A generation-scoped buffer capability for zero-copy handoffs

**Gap.** Native APIs expose path-specific lifetime rules but no portable representation of ownership and policy provenance across them.

**Mechanism.** Give every active zero-copy buffer lease a compact capability:

```text
cap = {
  region_id,
  offset,
  generation,
  owner,
  access_mode,
  dma_target,
  policy_generation
}
```

Transitions are explicit operations such as `USER_TO_RX`, `RX_TO_USER`, `USER_TO_TX`, `TX_COMPLETE`, `USER_TO_IOURING_REFILL`, and adapter-specific DPDK attach/detach transitions. The capability is invalidated on recycle and recreated with a new generation. The native ring or mbuf remains the fast-path source of truth; the capability layer mirrors only the state needed for cross-path checking and provenance.

For eBPF, a verifier-friendly representation could keep immutable region descriptors in maps and use bounded per-CPU or per-queue state for active generations. Userspace eBPF runtimes such as [bpftime](https://github.com/eunomia-bpf/bpftime) could consume the same schema at their attach boundary rather than inventing a different packet-lifetime vocabulary.

**Delta.** Existing APIs protect their own object lifetime. The new property is a typed capability whose generation and authority survive a transition between APIs without requiring them to use the same allocator.

**Artifact.** A small UAPI-neutral schema, AF_XDP and io_uring adapters, an optional DPDK adapter, a BPF-side checker for selected transitions, and a userspace validator that can reconstruct the lease state machine.

**Evaluation.** Run line-rate forwarding with AF_XDP zero-copy, io_uring ZC Rx, and DPDK variants. Measure packets per second, CPU cycles per packet, cache misses, memory overhead, and tail latency. Inject double-fill, premature recycle, use after completion, stale policy generation, worker crash, and NIC reset faults. Measure prevention rate and diagnostic precision.

**Academic value.** This asks whether linear or capability-like ownership can be made practical across kernel, userspace, and device packet-memory protocols without putting a general-purpose type system on the fast path.

**Production value.** A runtime can fail closed on invalid handoffs and explain the exact lease transition rather than reporting generic packet corruption.

**Failure condition.** If the capability shadow state costs enough cache traffic to erase the zero-copy benefit, or if native APIs already catch the cross-path failures with equivalent provenance, the mechanism should be rejected.

### 2. Preserve a compact handoff witness with each policy generation

**Gap.** A security decision and the buffer lifetime often live in separate evidence streams. Recycled addresses make timestamp-only joining unsafe.

**Mechanism.** Assign each relevant BPF policy decision a compact `decision_id` scoped to a policy generation. At zero-copy handoffs, emit or retain a witness containing the buffer lease generation, decision ID, source and destination owner classes, device/queue, and transition sequence. Userspace keeps the reverse mapping from `decision_id` to policy object or rule.

The witness does not contain packet payload by default. It can be emitted only for denies, anomalous transitions, policy-generation changes, and a sampled fraction of ordinary allows. The goal is not tracing every packet. The goal is to make later evidence unambiguous when a buffer address has been reused hundreds of times.

**Delta.** Existing packet traces can show addresses, queue events, or policy verdicts. The new property is an explicit join key between the *logical buffer lease* and the *policy generation that authorized the handoff*.

**Artifact.** A witness format, BPF map/event helpers, an AF_XDP userspace library wrapper, and a command that reconstructs one packet lease without reading payload contents.

**Evaluation.** Drive a tiny UMEM at high reuse rates while rotating policies and queue steering. Compare address-only correlation, timestamp-based correlation, and generation-scoped witnesses. Score false joins, missed joins, evidence volume, and root-cause accuracy. Include adversarial timing where two generations reuse the same frame within one scheduler tick.

**Academic value.** This creates a measurable provenance problem at the boundary between systems security and high-performance I/O rather than treating provenance as an unlimited logging problem.

**Production value.** Operators can answer "which policy admitted these bytes before this userspace handoff?" without full packet capture or preserving old control-plane state forever.

**Failure condition.** If generation-scoped witnesses do not materially reduce false attribution compared with cheaper queue-local counters and timestamps, they are unnecessary metadata.

### 3. Build a cross-path zero-copy fault benchmark before standardizing the contract

**Gap.** It is easy to design a beautiful ownership schema without proving that real mixed datapaths need it.

**Mechanism.** Build a benchmark that composes native paths rather than testing them in isolation. Scenarios should include XDP to AF_XDP, AF_XDP forwarding across shared UMEM, TCP payload delivery through io_uring ZC Rx, DPDK external buffers, and an optional userspace eBPF policy stage. Each scenario declares the expected ownership and policy-generation sequence, then a fault injector violates one boundary at a time.

Faults should include:

- the same AF_XDP frame published to conflicting ownership paths;
- a recycled address reused under a new tenant or policy generation;
- a queue-steering change that moves a flow outside the expected BPF path;
- userspace exit while buffers are outstanding;
- delayed completion followed by premature reuse;
- NIC reset or queue reconfiguration during a policy update;
- a DPDK external buffer detached while another logical packet reference remains.

The benchmark should compare three designs: unmodified native APIs, metadata-only witness collection, and active capability checking.

**Delta.** Existing API selftests verify individual mechanisms. The proposed benchmark makes *composition failure* the unit of evaluation and gives the contract proposal a way to fail.

**Artifact.** Reproducible traffic generators, fault injectors, a reference ownership trace, and graders for safety, provenance correctness, diagnosis time, and performance overhead.

**Evaluation.** Besides throughput and latency, report fault-detection recall, false positives, false policy joins after buffer reuse, recovery correctness, and bytes of evidence per million packets. Run across at least two NIC/driver combinations because zero-copy and DMA behavior is hardware-sensitive.

**Academic value.** The benchmark can distinguish a real cross-boundary abstraction problem from a collection of implementation bugs.

**Production value.** Networking runtimes get regression tests for failures that otherwise appear only under load, queue reconfiguration, or rapid buffer reuse.

**Failure condition.** If realistic compositions do not produce failures beyond what each native API already detects, the right result is to improve local diagnostics instead of standardizing a new contract.

## What would change this conclusion?

The strongest alternative is that zero-copy ownership should stay intentionally path-specific. AF_XDP rings, io_uring refill state, page_pool references, and DPDK mbuf reference counts are optimized around different execution models. A common capability layer could add cache pressure, duplicate state, and new failure modes while providing little more than a tracing convention.

That alternative wins if experiments show three things together: native APIs already reject the important cross-path misuse cases; address and timestamp correlation is reliable enough under realistic reuse; and hardware steering changes can be reconstructed cheaply from existing control-plane state.

The conclusion changes in the other direction if mixed datapaths repeatedly produce failures that are individually legal inside each subsystem but collectively violate buffer ownership or policy provenance. In that case, the missing abstraction is not another zero-copy transport. It is a small contract that says what one buffer lease means as bytes move through BPF, kernel networking, userspace, and the NIC.
