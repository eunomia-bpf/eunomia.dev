# Why do TLS-encrypted service-to-service traces not stitch together in zero-code eBPF tracing, and which component actually bridges the encrypted hop?

**Short answer:** because the trace context travels in the HTTP headers, and TLS encrypts exactly that payload. A zero-code eBPF tracer cannot decrypt in kernel space, so it can neither read nor inject `traceparent` inside ciphertext. The bridge is not the payload but the transport: OpenTelemetry eBPF Instrumentation (OBI) rides the context on a **custom TCP option (kind 25) attached to the connection before TLS**, which the kernel can see regardless of payload encryption, plus — for Go applications only — a userspace uprobe that writes the header *before* the TLS layer encrypts it. The cost is that this channel is OBI-specific (only other OBI-instrumented endpoints understand it) and connection-scoped, so it breaks at L7 proxies that discard and replay packets, and it cannot represent multiple concurrent streams on one HTTP/2 or gRPC connection.

## Why encrypted hops break header-based propagation

Zero-code distributed tracing works by serializing the W3C `traceparent` value across the boundary where the request crosses: the eBPF program reads the incoming context, carries it through the process, and writes it onto the outgoing request. In plaintext HTTP/1, the request bytes (headers included) flow through the kernel socket buffer in the clear. OBI's `kprobe` on `tcp_sendmsg`/`tcp_recvmsg` sees them, and the `sk_msg` program (tpinjector) can even extend the packet to add the header where the application did not. Any W3C-SDK peer will also understand it, because `traceparent` is a standard header.

For TLS the situation inverts. The application's TLS library encrypts the request in user space before it ever reaches the kernel socket buffer. From the kernel's point of view the bytes are ciphertext:

- a `kprobe` on `tcp_sendmsg`/`tcp_recvmsg` cannot parse HTTP headers out of it;
- the `sk_msg` injector cannot splice a `traceparent` header into ciphertext;
- and the receiving side cannot extract context the sender's application encrypted.

So header injection, the primary W3C-compatible channel, is simply unavailable on an encrypted hop. This is a property of where TLS terminates (user space, above the eBPF probes), not a configuration gap: enabling more probes does not let the kernel decrypt.

## What actually carries the context across TLS

OBI's [context propagation architecture](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/blob/main/devdocs/context-propagation.md) documents three injection methods, and each TLS case lands in a different one:

1. **HTTP/1 headers (L7)** — `traceparent` in plaintext HTTP. W3C-compatible, readable by any SDK-instrumented peer. Unavailable on TLS.
2. **TCP options (L4)** — a custom TCP option, kind 25, written onto the connection's segments. It exists *before* TLS (in the IP packet), so it is unaffected by payload encryption. The sender's tpinjector schedules the option; the receiver's `BPF_SOCK_OPS` program parses it into the `incoming_trace_map` on ingress. Two hard limits: the option is connection-scoped, and only OBI-instrumented endpoints know kind 25, so the context does not reach an SDK-only peer. The OBI docs state the consequence directly: for TLS, OBI "injects the information at TCP/IP packet level" and "is only able to send the trace information to other OBI instrumented services".
3. **Per-stream HPACK injection (L7, multiplexed protocols)** — for HTTP/2 and gRPC, one connection carries N concurrent streams, so a single connection-scoped TCP option cannot represent N distinct trace contexts ("a connection-scoped option cannot represent N concurrent stream contexts"). OBI instead writes a per-stream `traceparent` HPACK field into outgoing HEADERS frames via `bpf_msg_push_data`, which is the only network mechanism for multiplexed HTTP/2.

The per-stream HPACK path meets TLS again: for generic (non-Go) TLS HTTP/2 the HPACK splice would land in ciphertext, so it cannot work — the documented boundary is that "Generic non-gRPC HTTP/2 context propagation remains limited to Go library instrumentation." The Go exception works because a userspace uprobe into Go's HTTP/TLS layer writes the context into Go's plaintext request buffer *before* the TLS layer encrypts it (Go's `persistConnRoundTrip` path uses `bpf_probe_write_user` into the application buffer), so the encrypted wire carries a header the receiver's TLS layer will eventually expose.

On ingress the layers use "last one wins": `BPF_SOCK_OPS` parses TCP options first, the `kprobe`/`protocol_http` layer parses HTTP headers later and overwrites, so the most reliable method (standard headers) naturally takes priority when both are present. Notably, receiving-side parsing of incoming `traceparent` from OpenTelemetry SDK-instrumented services "still work[s]" even when the outgoing TCP-option channel is disrupted — a peer that sends standard headers in the clear can always be read.

The middlebox caveat is the operational detail that surprises people most. A custom TCP option on an established connection is not preserved end-to-end through many network paths: L7 proxies and load balancers discard the original packets and replay them on new connections, and some middleboxes or managed endpoints drop or reset the segment carrying the unknown option — the client then sees `connection reset by peer`, typically on the first request after connect, and the effect is path-dependent, so it appears intermittently. The documented remedy: where instrumented services talk through such intermediaries, use `headers` and do not enable `tcp`; TCP options are safe only when OBI is on both ends and the path preserves them (a direct L2/L3 network with no option-stripping middlebox).

## How to verify which case you hit

1. **Check the config state.** Network-level context propagation in OBI is disabled by default. Enable it via `OTEL_EBPF_BPF_CONTEXT_PROPAGATION=all` (or a subset) or the `context_propagation` key in the OBI configuration (`all`, `headers`, `tcp`, or `headers,tcp`; the former `http` alias is removed and the deprecated `ip` value has no effect).
2. **Read the wire.** A `tcpdump` of the connection between two instrumented TLS endpoints should show the kind-25 TCP option on the sender's segments. If the option disappears after an L7 proxy or load balancer, the proxy is the boundary: the hop after it cannot carry L4 context, and the trace will split at that point.
3. **Watch for the reset signature.** Intermittent `connection reset by peer` on the first request after connect, on TLS paths with `tcp` enabled, is the option-stripping middlebox symptom described in the docs; dropping `tcp` from the propagation mode removes it.
4. **Confirm the receiving side.** OBI parses incoming `traceparent` automatically on both header and TCP-option channels, so an SDK-instrumented upstream sending standard headers will still stitch correctly even when the L4 channel is dead — check that the server span carries the upstream trace ID.
5. **Check the protocol and runtime.** For HTTP/2 or gRPC over TLS, the trace only stitches when the peer is Go-instrumented (uprobe-before-encryption); a generic-language TLS HTTP/2 client is at the documented limit, and its per-stream contexts cannot be injected by OBI over the encrypted hop.

## Where the answer stops

- The kind-25 TCP option is OBI-specific. It is not a W3C mechanism: an SDK-only peer cannot read it, and the context is silently dropped at that boundary.
- L7 proxies and load balancers that terminate and re-establish TCP connections break the L4 channel; the trace then splits exactly at the proxy.
- Option-stripping middleboxes and managed endpoints can turn the channel into connection resets rather than clean non-propagation.
- Multiplexed TLS HTTP/2 and gRPC contexts are injectable only through Go's user-space uprobe path; other runtimes have no TLS-capable per-stream mechanism in the zero-code layer.
- None of this changes the receiving side's ability to read standard headers: incoming `traceparent` from any W3C-compliant sender (SDK or OBI) is parsed in the clear wherever the bytes are visible to the probes.

## References

- [OBI context propagation architecture (devdoc)](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/blob/main/devdocs/context-propagation.md)
- [OBI gRPC/HTTP2 context propagation (devdoc)](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/blob/main/devdocs/grpc-context-propagation.md)
- [OBI distributed traces documentation (TLS limitations and `context_propagation` config)](https://opentelemetry.io/docs/zero-code/obi/distributed-traces/)
- [W3C Trace Context specification](https://www.w3.org/TR/trace-context/)

## Community discussion today

Honest coverage note: the two watchlist-opted Slack archives returned **zero messages** in the rolling 7-day window today, and the allowlisted Discord channels are visible-browser-only, with no visible browser session available in this run — so no private community material was available for 2026-09-22. The question above is the fallback selection: a genuine, recurring, still-open boundary of the monitored OpenTelemetry eBPF community, grounded entirely in the public primary sources above rather than in any thread.

The recurring practitioner symptom in that space — "my service-to-service traces stop at the TLS hop" — resolves to exactly the boundary documented here: header-based propagation dies where TLS terminates, the L4 TCP-option channel is the only zero-code bridge for TLS, and that bridge has a hard compatibility surface (OBI-only peers, option-preserving paths, and per-stream limits on multiplexed protocols). Two themes from the public documentation recur in this space: (1) the intermittent first-request `connection reset by peer` that option-stripping middleboxes produce, which is easy to misread as application flakiness when `tcp` propagation is enabled behind an L7 proxy; and (2) the silent split at any non-OBI endpoint, where the TCP option is simply ignored rather than rejected, so traces look complete on one side and orphaned on the other. The open question the documentation itself leaves unresolved is what happens to the L4 channel on paths with stateful load balancers and NAT devices that rewrite segment headers; the documented workaround is to fall back to `headers` and accept that non-OBI or proxied peers break the chain.
