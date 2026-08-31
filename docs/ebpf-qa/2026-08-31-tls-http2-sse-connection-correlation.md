# Why are PID and TID insufficient to correlate concurrent TLS, HTTP/2, and SSE traffic?

**Short answer:** a process or thread identifies where a probe ran, not which connection or request owns the bytes. Keep thread-scoped state for pairing a function entry with its return, but give protocol reconstruction its own connection-lifetime identity. Within that connection, HTTP/2 needs stream-scoped request state, while its compression state belongs to the connection and direction. Server-sent events (SSE) must then be parsed inside the correct HTTP response body.

This distinction matters even in a single-threaded application. One event loop can alternate between several connections. Conversely, an application can hand a connection between workers. Grouping everything by PID merges unrelated traffic; grouping by TID can both merge different connections and split one connection into several incomplete histories. Increasing a parser timeout cannot repair that identity error.

## Separate the probe call from the protocol object

The Linux helper `bpf_get_current_pid_tgid()` returns the current task's thread-group ID in the upper half and task ID in the lower half. In usual user-space terminology, these are the process ID and thread ID. Neither field describes a TLS object. See the [kernel helper definition](https://github.com/torvalds/linux/blob/master/include/uapi/linux/bpf.h).

For a conventional synchronous TLS-library call, entry/return pairing can use a thread-scoped slot to retain arguments until the return probe runs. If the selected hooks can nest or observe wrapper and underlying calls, add a bounded invocation/depth distinction or choose one non-overlapping hook layer. Otherwise, one invocation can overwrite another's arguments or the same bytes can be counted twice.

That temporary slot is not a reassembly buffer. A useful conceptual separation is:

| State | Identity and scope |
| --- | --- |
| Pending function invocation | Process instance, TID, operation, and invocation/depth when needed |
| Plaintext byte reconstruction | Connection lifetime and direction |
| HTTP/2 request and response | Connection lifetime and stream ID |
| HPACK decoder | Connection lifetime and observed direction; shared across its streams |
| SSE parser | The particular HTTP response body |

These are design scopes, not a ready-to-copy ABI. A collector still has to define how it creates each identity, orders events, detects gaps, and retires state.

## A TLS handle needs a lifetime, not just an address

For OpenSSL TLS-over-TCP hooks, the `SSL *` argument is a useful connection discriminator. Preserve it at entry along with the buffer and operation metadata; do not throw it away and later try to infer the connection from TID. The [`SSL_read` interface](https://docs.openssl.org/3.5/man3/SSL_read/) makes that object argument explicit.

An address alone is not a durable identifier. Different processes can have the same virtual address, an allocator can reuse freed storage, and OpenSSL can reset an existing object for another connection with [`SSL_clear`](https://docs.openssl.org/master/man3/SSL_clear/). A collector-local model could therefore be:

```text
process_instance = collector_scope + process_lifetime
connection       = process_instance + tls_object + connection_epoch
request          = connection + http2_stream_id
```

The epoch must follow observed connection creation or reuse, not merely the first byte seen after an arbitrary timeout. Account for process exit and `exec`, object reset, final destruction, and collector restart according to the library and hooks you actually support. A file descriptor or socket cookie can help connect library events to socket observations, but neither replaces a proven mapping between the two layers. File descriptors also get reused, and not every TLS BIO is a socket BIO.

If lifecycle events were missed, mark the association uncertain and bound or retire the stale state. Do not silently merge a new connection into an old parser because its address happens to match. Similarly, attaching in the middle of a live connection is partial coverage, not proof that its earlier protocol state was empty.

## A successful TLS call is still not an HTTP message

The [`SSL_read` documentation](https://docs.openssl.org/3.5/man3/SSL_read/) distinguishes requested capacity from bytes actually returned. For `SSL_read_ex`, success is a status and the byte count comes through the output argument. Read the returned data only after success, and never treat the requested buffer size as valid payload length.

Writes have a similar distinction. [`SSL_write` and `SSL_write_ex`](https://docs.openssl.org/3.5/man3/SSL_write/) can require retries, and partial-write mode changes how much a successful operation accepts. A reconstruction pipeline should commit the successful byte count, not append the entire entry buffer again on every attempted write. An entry-side observation is an attempted write until its return establishes otherwise; successful acceptance by the TLS library is also not proof of remote application delivery.

The parser should consume an ordered byte stream for each connection direction, retaining incomplete input between calls. It must not use a TLS-call boundary as an HTTP frame or event boundary. Capturing both a wrapper and its implementation, losing a return probe, truncating a buffer, or dropping an emitted event all need explicit accounting. A timestamp-only sort cannot establish correctness if the instrumentation has not preserved the relevant per-connection order.

## HTTP/2 and SSE add different kinds of state

[HTTP/2](https://www.rfc-editor.org/rfc/rfc9113.html) multiplexes streams inside a connection. A stream number can appear again on another connection, so `(TID, stream_id)` is not a safe substitute for `(connection, stream_id)`. Request and response belong to the same stream identity; direction selects which bytes and parser state are being processed.

[HPACK](https://www.rfc-editor.org/rfc/rfc7541.html) has separate directional compression contexts. A passive decoder needs the corresponding context for each connection direction, shared across that direction's streams. A process-global decoder contaminates unrelated connections. A fresh decoder for every stream loses shared header-table history. If a capture gap hides a table update, later decoded headers are not trustworthy just because some bytes still resemble valid frames; stop claiming complete decoding until a new captured connection or a protocol-proven recovery restores the state.

SSE is a format within an HTTP response body, not another thread-level transport. The [HTML standard](https://html.spec.whatwg.org/multipage/server-sent-events.html) defines line processing and blank-line event boundaries. Preserve partial lines and events across input chunks, and feed only the bytes belonging to that response. An SSE `id` is application-provided reconnection state, not a globally unique connection or request key. Reconnecting creates a new HTTP response association even when the application resumes a logical event sequence. It can reuse an existing HTTP/2 connection through a new stream; advance the transport epoch only when that transport lifetime actually changes.

## Verify identity before expanding parsing coverage

Use a synthetic local workload with distinct, non-sensitive markers. The following is a proposed regression checklist, not a claim that these experiments were run here:

| Test case | Required observation |
| --- | --- |
| One worker alternates between two TLS connections | Payload and parser state never cross connections |
| One connection is handed between workers | Its connection identity survives the TID change |
| Two HTTP/2 connections use the same stream number | The requests remain distinct |
| Multiple streams use dynamic header compression | Shared history works within one direction, without cross-connection contamination |
| SSE lines cross several read boundaries | Each complete event belongs to exactly one response |
| An object is reused, or capture begins late | Old parser state is retired, or incomplete coverage is reported |
| A retry, truncation, or event drop is injected | No duplicate append or fabricated complete message |

Compare expected request ownership at the application boundary with the collector's output. First inspect anonymous connection IDs, directions, byte counts, gap flags, and state transitions. Payload inspection should be an explicit, narrowly scoped option: TLS uprobes can see plaintext, and any temporary payload copies in maps, buffers, logs, or exports need a defined access and retention policy. Downstream redaction does not undo earlier collection.

The connection model above is specific to the observation layer, not a universal tracing identity. Other TLS libraries, language runtimes, QUIC, and unsupported I/O paths need their own mappings and coverage tests. An application request ID or trace context can provide a stronger semantic association when available. Missing context should remain missing; PID/TID proximity is useful diagnostic metadata, not permission to invent a request relationship.

## References

- [Linux BPF UAPI: current process and thread identifier helper](https://github.com/torvalds/linux/blob/master/include/uapi/linux/bpf.h)
- [OpenSSL: `SSL_read` and `SSL_read_ex`](https://docs.openssl.org/3.5/man3/SSL_read/)
- [OpenSSL: `SSL_write` and `SSL_write_ex`](https://docs.openssl.org/3.5/man3/SSL_write/)
- [OpenSSL: resetting a TLS object with `SSL_clear`](https://docs.openssl.org/master/man3/SSL_clear/)
- [RFC 9113: HTTP/2 streams, framing, and compression state](https://www.rfc-editor.org/rfc/rfc9113.html)
- [RFC 7541: HPACK compression contexts and dynamic tables](https://www.rfc-editor.org/rfc/rfc7541.html)
- [HTML Standard: server-sent event parsing](https://html.spec.whatwg.org/multipage/server-sent-events.html)
- [RFC 9293: TCP connection state and the transmission control block](https://www.rfc-editor.org/rfc/rfc9293.html)
- [Linux kernel documentation: RCU removal and reclamation](https://docs.kernel.org/RCU/whatisRCU.html)
- [GitHub documentation: changing a pull request's comparison base](https://docs.github.com/en/pull-requests/how-tos/create-pull-requests/changing-the-base-branch-of-a-pull-request)

## Community discussion today

The visible-browser review covered all 6 approved communities and all 15 allowlisted channels or public pages. Every target was accessible. The selected question appeared within the strict 24-hour window; no seven-day fallback was needed. This synthesis removes participant and channel identities, message links, exact times, private deployment details, and original wording. No raw transcript was retained and no community interaction was performed.

### Correlation errors can survive successful capture

The clearest debugging question concerned concurrent encrypted application traffic that was observable but not reliably attributable. The failure boundary is between capturing bytes and assigning ownership: correct probes do not compensate for a parser keyed by the wrong object. The first useful diagnostic is therefore a two-connection isolation test, followed by connection reuse and loss tests, before adding more protocol heuristics. The OpenSSL interfaces and HTTP/2 state model above support that separation. The available discussion does not establish which failure dominates in every collector, so incomplete hook coverage, lost events, and identity collisions must remain separate hypotheses.

### Network recovery needs an application-level success criterion

A separate question asked how to evaluate an eBPF-assisted network recovery design convincingly. Restored packet forwarding, a recovered TCP connection, and a successfully completed application operation are different outcomes. [TCP's transmission control block](https://www.rfc-editor.org/rfc/rfc9293.html) carries connection state; redirecting packets alone does not demonstrate that a replacement endpoint has that state or the application's state.

Our recommended test plan separates a temporary path outage from an endpoint restart. For each, compare an unmodified baseline and the proposed mechanism under the same load, record recovery-time distributions and CPU cost, and check application-visible gaps, duplicates, and failed operations using synthetic sequence markers. Record the exact kernel builds rather than claiming a universal minimum from one successful run. The observed discussion was a request for evaluation guidance, not independently verified evidence that transparent recovery already works.

### Detaching a hook and reclaiming its code are separate events

Kernel review continued to examine lifetime safety around programs reached through generated trampolines, with requests for a reproducible concurrency test. The general mechanism is the distinction between preventing new entry and waiting for execution already in flight. The [RCU documentation](https://docs.kernel.org/RCU/whatisRCU.html) explains removal versus reclamation, but a proof must cover the actual execution path and its synchronization domain, including intermediate code that can still reach another object.

A useful test overlaps repeated attachment and detachment with busy hooks, then checks both stale execution and resources that never become reclaimable. Review activity also covered alias invalidation, bounded iteration, and type-validation edges. These are proposals or review signals, not a statement that a particular distribution kernel has all fixes. Deployment conclusions require the target source revision, backport contents, and appropriate selftests.

### Dependent instrumentation changes need explicit review order

The active instrumentation replies focused on splitting interdependent changes without presenting the same prerequisite diff repeatedly. The practical issue is the comparison base and merge order, not a new telemetry field requirement. A dependency list, incremental comparison where the repository permits it, and a combined integration check can make the intended sequence clear. [GitHub's base-branch documentation](https://docs.github.com/en/pull-requests/how-tos/create-pull-requests/changing-the-base-branch-of-a-pull-request) also warns that changing the base can invalidate review context. Branch permissions and release coordination remain repository-specific; the discussion did not establish a generally available workflow.

### Quiet targets were still checked

Most project help and feature areas and the scheduler support surfaces had no new substantive question in the window. General eBPF chat contained review coordination rather than a fresh troubleshooting problem, and the eBPF instrumentation area had no new in-window technical exchange. Automated development feeds and article sharing were not counted as additional user demand. These are accessible-but-quiet or coordination-only observations, not gaps hidden as zero activity.
