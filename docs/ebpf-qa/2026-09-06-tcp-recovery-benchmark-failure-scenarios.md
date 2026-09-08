# Which failure scenarios are mandatory before a transient-TCP-recovery eBPF benchmark is meaningful?

**Short answer:** the boundary is between the layer where you redirect packets and the layer where connection state actually lives. An eBPF XDP or TC program moves *frames* between netdevs or into user space; it does not move, copy, or rebuild a socket's Transmission Control Block, and it has no way to reconstruct the application's in-flight request/response state. A benchmark is therefore meaningful only if it injects the failure at a specific layer, asserts recovery at that layer *and one layer above it*, and explicitly distinguishes three outcomes that a naive "packets flow again" check collapses into one: the datapath re-routed, the kernel TCP connection survived, and the application operation completed.

The four quantities typically reported for this kind of prototype—recovery latency, packet loss during failover, socket/connection state, and CPU overhead—each bind to a different layer. The decisive design decision is which layer each metric is asserted at. If every metric is sampled only on the datapath, the benchmark can read as a pass while the connections it claims to have recovered were in fact reset by the kernel.

## The layer where connection state actually lives

TCP connection state is not a property of the packet stream. RFC 9293 defines the connection record as the **Transmission Control Block (TCB)**, which holds the local and remote addresses and ports, the send and receive sequence numbers (`SND.UNA`, `SND.NXT`, `RCV.NXT`), the window values, and pointers to the send/receive buffers and the retransmit queue (see [RFC 9293, §3.3.1 Key Connection State Variables](https://www.rfc-editor.org/rfc/rfc9293.html#name-key-connection-state-variables)). The TCB lives in the kernel socket, not in the wire format.

This is the boundary that matters for the benchmark. An eBPF datapath hook operates on `sk_buff` or `xdp_buff` data and can return a `redirect`/`pass`/`drop` action. The [XDP redirect mechanism](https://docs.kernel.org/bpf/redirect.html) looks up a target (`DEVMAP`, `DEVMAP_HASH`, `CPUMAP`, or `XSKMAP`), enqueues the frame into a per-CPU bulk queue, and flushes it inside the NAPI poll loop. The AF_XDP documentation makes the asymmetry explicit: a UMEM completion "returns ownership of a frame to user space, but does not by itself guarantee that the packet was successfully transmitted" ([AF_XDP documentation](https://docs.kernel.org/networking/af_xdp.html)). Moving a frame to a new interface or a user-space ring says nothing about whether the TCB that owns the flow still exists.

Consequently, three statements are *not* equivalent, and a benchmark must not score them as one:

1. **Datapath recovered** — packets now traverse the intended netdev or XSK. This is the only thing an XDP/TC redirect can directly prove.
2. **Connection recovered** — the kernel TCP socket for the flow is still in `ESTABLISHED` (or a resumption state) and its sequence/window space is intact. This is a property of the TCB, not the wire.
3. **Application recovered** — a request that was in flight completed, or a new request succeeded, without a visible gap, duplicate, or reset reaching the caller.

A datapath redirect can deliver on (1) while (2) and (3) are false: the TCB was reset by the kernel for an unrelated reason, or the application's outstanding request was dropped and its client-side retry produced a fresh, unrelated operation.

## The failure-scenario matrix

Which scenarios are mandatory follows from the claim a benchmark makes: a
scenario is required when the claim covers the layer its failure mode lives
in. Each scenario below isolates a failure mode that a different layer owns;
omitting one means that layer's defects can hide behind a green datapath
counter, and a benchmark whose claim spans several layers needs the scenarios
that span the same layers.

**Fault at the datapath (interface, queue, or redirect target).**
Kill or reprogram the forwarding path while the flows are live. When the local addresses and socket ownership remain valid and routing converges, the kernel's TCP stack should keep the TCB and retransmit on the new path. The benchmark must assert that (a) packets resume on the intended netdev/XSK, and (b) the socket's sequence counters did *not* roll back and no `RST` was emitted for the flow. This is the case where "recovery latency" is a well-defined number: it is the time from the last packet on the dead path to the first accepted packet on the live path.

**Fault at the kernel socket (the TCB disappears).**
Force the socket to be reset—drop the owning process, let the kernel exhaust its retransmission budget for the flow, or trigger an `ABORT`—and then re-establish. Here the datapath can be healthy the whole time, yet the connection is gone. The benchmark must show that a new handshake (or the application's reconnect) is what restored service, and it must measure the *reconnect* latency rather than claiming the redirect recovered the connection. The kernel controls live in [the IP sysctl reference](https://docs.kernel.org/networking/ip-sysctl.html) and are subtler than a single cutoff: given a value of N, `tcp_retries2` describes a *hypothetical* connection under exponential RTO backoff that retransmits N times and is killed at the (N+1)th RTO; the default N of 15 corresponds to 924.6 s, which the documentation explicitly calls a *lower bound for the effective timeout* — Linux aborts at the first RTO that exceeds the hypothetical timeout, so the real cutoff tracks the live connection's RTO (bounded by `tcp_rto_min_us` and `tcp_rto_max_ms`), not one universal number. Keepalive is a separate, optional regime: [RFC 9293, §3.8.4 TCP Keep-Alives](https://www.rfc-editor.org/rfc/rfc9293.html#name-tcp-keep-alives) makes keep-alives optional, per-connection, and default-off, and Linux counts `tcp_keepalive_time`, `tcp_keepalive_probes`, and `tcp_keepalive_intvl` only "when keepalive is enabled" on the socket. An idle connection with keepalive off and no unacked data is therefore not killed by `tcp_retries2` at all — nothing is retransmitting. The benchmark must name the regime it tests: with unacked data in flight, straddle the effective `tcp_retries2` boundary; with a quiet connection, either enable keepalive and straddle the keepalive budget or state that the socket survives the outage by design. Mixing the two regimes is exactly how a benchmark reports "recovery" while the liveness rules differed per scenario.

**Fault at the application (in-flight operation is lost).**
Different losses at this layer have different owners. Segments that were already transmitted but not yet acknowledged, on a TCB that survives, are transport loss: the standard recovery is TCP's own retransmission, not application retry — the bytes stay in the sender's retransmission queue and are repaired below the socket boundary ([RFC 9293, §2.2](https://www.rfc-editor.org/rfc/rfc9293.html#name-key-tcp-concepts) defines reliability as "correction via retransmission," and [§3.8.1](https://www.rfc-editor.org/rfc/rfc9293.html#name-retransmission-timeout) with the R1/R2 failure thresholds of [§3.8.3](https://www.rfc-editor.org/rfc/rfc9293.html#name-tcp-connection-failures) define when that repair gives up). What the transport cannot repair is the boundary above the wire: an application that crashed after committing an operation but before the response reached its caller cannot tell, from TCP, whether the operation completed, and retrying there is the application's own decision whose exactly-once behavior comes from application semantics (idempotency keys, dedup, transaction markers) — not from TCP or the datapath. The benchmark must therefore inject a *synthetic, sequence-marked* operation and assert on the application-visible outcome — completion of the marked operation according to the application's declared retry semantics — rather than on packet counts, and it must distinguish losses absorbed by retransmission from losses visible to the caller. This is the layer where "packet loss during failover" can be nonzero yet still consistent with a correct run, because the loss was absorbed by the transport below the application boundary.

**Fault at the redirect target (stale or empty map entry).**
The XSKMAP and DEVMAP entries are the redirect's targets. The [AF_XDP documentation](https://docs.kernel.org/networking/af_xdp.html) notes that on redirect XDP validates that the XSK at the map index was bound to that device and ring, and that a mismatched XSK or an empty map slot means the packet is dropped rather than delivered. The benchmark must exercise an empty or stale target and assert that the drop is *counted*, not silently absorbed. This is the failure mode that a pure latency histogram will miss entirely, because nothing ever reaches the "recovered" path.

**Fault that outlives the failover (sustained outage longer than any single recovery).**
Hold the outage open past the kernel's retransmission budget for unacked in-flight data — and past the connection's keepalive budget, when the application actually enables keepalive on that socket. This is the scenario that proves the benchmark is measuring *recovery* and not merely *the absence of a transient blip*. If the sustained-outage case is not in the matrix, a fast transient recovery can mask a design that has no behavior at all for a long-lived fault.

## How to assert each scenario without overclaiming

For each scenario, the assertion must be placed at the layer one above the mechanism being tested, using a signal that layer owns:

- **Datapath:** count on the redirect tracepoints. The [redirect documentation](https://docs.kernel.org/bpf/redirect.html) lists `xdp_redirect`, `xdp_redirect_err`, `xdp_redirect_map`, `xdp_redirect_map_err`, and `xdp_devmap_xmit`; the devmap transmit failure is carried in `xdp_devmap_xmit`'s `err` field — there is no separate `xdp_devmap_xmit_err` tracepoint — and its bpftrace examples build errno histograms from `xdp_redirect*_err` and `xdp_devmap_xmit` to debug silent drops. A probe that counts `xdp_redirect_err` distinguishes "the redirect failed" from "the redirect succeeded and the frame was delivered," which a throughput number cannot.
- **Connection:** sample the socket state and sequence counters, not just that the socket exists. The TCB's `ESTABLISHED` state, `SND.NXT`/`RCV.NXT` monotonicity, and the absence of a `RST` for the flow are the signals. A socket that exists but whose counters reset is a new connection, not a recovered one.
- **Application:** complete the marked operation end-to-end and check for gaps, duplicates, and errors at the caller. Packet counters are not a substitute for this.

The CPU-overhead metric should be reported *per scenario and per layer*, not as a single aggregate. Redirect processing, TCP recovery or reconnection, and application replay exercise different code paths and may shift work between kernel and user space. Reporting one blended number hides which layer is actually costing you.

## What makes the benchmark meaningless

The benchmark becomes meaningless when the failure injection and the recovery assertion live in the same layer. Concretely:

- Injecting the fault at the datapath and asserting recovery only at the datapath. The datapath reads as "recovered" as soon as the interface returns, regardless of what happened to the connections.
- Reporting "recovery latency" without stating which layer's first-successful event it measures. A latency number without a layer is a free variable.
- Using packet-loss-during-failover as the only loss metric, without an application-level outcome check per the application's declared retry/idempotency semantics. The kernel can lose and retransmit packets indefinitely; the application may still be correct. Conversely, the application can be correct while the wire shows heavy loss.
- Not naming the liveness regime under test. With unacked data in flight the effective `tcp_retries2` boundary governs; an idle connection survives indefinitely unless keepalive is enabled on that socket. Without naming the regime, the benchmark cannot tell a surviving connection from a re-established one.

The decisive rule is the same as for any cross-layer recovery test: **the recovery assertion must be one layer above the fault injection, and the metric must be named for the layer it measures.**

## References

- [RFC 9293: Transmission Control Protocol, §3.3.1 Key Connection State Variables](https://www.rfc-editor.org/rfc/rfc9293.html#name-key-connection-state-variables)
- [RFC 9293, §2.2 Key TCP Concepts](https://www.rfc-editor.org/rfc/rfc9293.html#name-key-tcp-concepts)
- [RFC 9293, §3.8.1 Retransmission Timeout](https://www.rfc-editor.org/rfc/rfc9293.html#name-retransmission-timeout)
- [RFC 9293, §3.8.3 TCP Connection Failures](https://www.rfc-editor.org/rfc/rfc9293.html#name-tcp-connection-failures)
- [RFC 9293, §3.8.4 TCP Keep-Alives](https://www.rfc-editor.org/rfc/rfc9293.html#name-tcp-keep-alives)
- [Linux kernel documentation: BPF XDP redirect mechanism and packet-drop debugging tracepoints](https://docs.kernel.org/bpf/redirect.html)
- [Linux kernel documentation: AF_XDP, XSKMAP, and UMEM completion semantics](https://docs.kernel.org/networking/af_xdp.html)
- [Linux kernel documentation: BPF map types including `DEVMAP`, `SOCKMAP`, and `XSKMAP`](https://docs.kernel.org/bpf/maps.html)
- [Linux kernel documentation: TCP-related IP sysctls (`tcp_retries2`, `tcp_rto_min_us`, `tcp_rto_max_ms`, keepalive knobs)](https://docs.kernel.org/networking/ip-sysctl.html)

## Community discussion today

The question this page answers is a real question raised by eBPF practitioners
in the communities this site monitors, republished here in anonymized form.
The page carries no participant identity, account, employer, project, channel
name, message link, exact timestamp, private topology, raw log, or wording
that could be searched back to a single person, and no reply, reaction, direct
message, follow, invitation, or moderation action accompanies this
publication.

The retained question came from readable archive material for one monitored
community and one allowlisted channel. A second archive-backed community was
inaccessible because its expected channels could not be matched. The two
browser-only chat communities, the public mailing-list archive, and the public
forum were not reviewed in this attempt. Those five communities are therefore
unavailable coverage, not quiet sources; no claim about their activity is made.

One verified cross-link to prior coverage stays relevant. An earlier daily Q&A
explained [why `bpf_tail_call()` cannot expose a return value](/ebpf-qa/2026-09-02-bpf-tail-call-return-value/):
a helper whose success path never returns makes the failure continuation, not
a return code, the observable event. That is the same design lesson this page
applies to benchmarks — the observable outcome must live at the layer that
owns it, one step above the mechanism under test. How quickly
transient-recovery benchmarks converge on per-layer assertions, separating
datapath recovery, connection survival, and application-visible completion
instead of reporting a single "packets flow again" number, remains the open
question the monitoring pipeline keeps tracking.
