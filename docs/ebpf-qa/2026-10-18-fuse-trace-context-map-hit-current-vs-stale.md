# When a FUSE daemon lookup of the OBI trace context map returns an entry, is that entry the trace context of the current request, and when can the hit be stale?

Not necessarily. A hit is not a promise that the entry is the trace context of the FUSE request you are about to serve. The OBI pinned trace context map `traces_ctx_v1` holds, per OS thread, the trace context of the innermost span that OBI is *currently* tracking on that thread. OBI stamps the thread at span start and clears or rolls the entry back when the span ends, so the value you read on the caller thread is whichever active span OBI last wrote there. It can still be the previous request's span on a reused worker thread, or the parent server span that a just-finished client span rolled back onto. The hit is a signal of "this thread is inside some active OBI span right now", not "this request is that span". The daemon must treat the returned `trace_id` / `span_id` as the thread's current OBI span, and must not assume it is the FUSE request's span.

## The mechanism

1. **Map spec (external surface).** OBI defines `traces_ctx_v1` as a `BPF_MAP_TYPE_LRU_HASH` keyed by the raw `u64` `bpf_get_current_pid_tgid` packed as `(tgid << 32) | tid`, with a value of `obi_ctx_info_t` holding `trace_id` and `span_id`. It is pinned by name (`LIBBPF_PIN_BY_NAME`) and sized `max_entries` to `1 << 14`. The spec is part of an OpenTelemetry OTEP, so other components read it too, and OBI does not reserve slots for your thread.

2. **Stamp at span start.** OBI's `server_or_client_trace` writes the thread entry when a server or client span begins, using `obi_ctx__set` to store that span's `trace_id` and `span_id`. That is the moment the thread entry becomes valid for the span.

3. **Virtual-thread carriers are skipped on purpose.** When a request is served by a Java virtual thread, OBI keys its internal `server_traces` map by the virtual-thread logical id, but it deliberately does *not* write a carrier-keyed `traces_ctx_v1` entry. The reason is in the source: a carrier-keyed entry would attribute the virtual-thread's context to whatever runs on the carrier next. So on a virtual-thread carrier the pinned map either misses or holds a different span, and the hit does not resolve to the virtual-thread span.

4. **Server span end deletes the entry.** When the server span ends, OBI calls `delete_server_trace`, which removes the thread's `traces_ctx_v1` entry via `obi_ctx__del` (unless the key is a Java virtual-thread TID). After that the caller thread has no entry of its own.

5. **Client span end rolls back to the parent.** When a *client* span ends inside a server span, OBI does not delete the entry. It calls `obi_ctx__restore_server`, which looks up the parent in `server_traces` and rewrites the thread entry to the parent's `trace_id` and `span_id`. The restore is guarded: it only happens when the entry still points at the just-ended client span (a span-id equality check) and the parent is present in `server_traces`. If either guard fails, the entry is left as it is, possibly still pointing at the old client span.

## Verification and debugging path

1. **Compare the hit against the request you know.** If the daemon derived the expected `trace_id` for this request out of band (from the caller's metadata or a side channel), compare it to the `trace_id` you read from the map. A mismatch means the entry is another, still-open span on that thread, not this request.

2. **Time the read against the span's lifetime.** The entry exists only while OBI's span on the thread is open. If you read after the caller's request span has ended, the hit (if any) is the rolled-back parent server span or nothing, never the request's own span.

3. **Check the thread model.** If the caller is a Java virtual thread, a hit on the carrier does not resolve to the virtual-thread span, by design. Correlate the virtual-thread logical id out of band before you read the map.

4. **Treat LRU eviction as an additional miss axis.** `max_entries` is `1 << 14`, and OBI does not pin a slot for your thread. Under a high number of live stamped threads the LRU list evicts the least-recently-used entry, so an active thread can still miss. A miss is therefore not proof that OBI never stamped the thread.

## The limitation

- A hit is not proof of "current request". It is the innermost active span OBI is tracking on the OS thread, and it can belong to the previous request on a reused worker thread, or to a parent server span.
- The entry is transient. OBI writes it at span start, removes it at server-span end, and rolls it back to the parent at client-span end. Read it inside the span window, or you read the wrong span or nothing.
- The rollback is conditional. The client-span-end restore only fires when the entry still points at the ended client span and the parent is present. If another write intervened, the stale value stays, and the map can hold a span that no longer matches the thread's current work.
- Virtual-thread carriers have no entry by design, and LRU pressure can evict an active thread's entry, so a hit and a miss are each only advisory on their own; they make sense only when you know what OBI is tracking on that thread.

## References

- [OBI bpf/shared/obi_ctx.h](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/blob/main/bpf/shared/obi_ctx.h) - the pinned `traces_ctx_v1` map: `BPF_MAP_TYPE_LRU_HASH`, `u64` key, `obi_ctx_info_t` value (`trace_id` / `span_id`), `max_entries` of `1 << 14`, pinned by name; plus the `obi_ctx__get` / `obi_ctx__set` / `obi_ctx__del` helpers.
- [OBI bpf/common/trace_lifecycle.h](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/blob/main/bpf/common/trace_lifecycle.h) - `server_or_client_trace` stamps the thread at span start and skips virtual-thread carriers; `delete_server_trace` deletes at server-span end; `obi_ctx__restore_server` rolls back to the parent server span at client-span end, guarded by a span-id check.
- [kernel Documentation/bpf/map_hash.rst](https://github.com/torvalds/linux/blob/master/Documentation/bpf/map_hash.rst) - `BPF_MAP_TYPE_LRU_HASH` automatically evicts the least-recently-used entry when the table reaches capacity.
- [OBI trace context read path (10-14 entry)](/ebpf-qa/2026-10-14-fuse-otel-trace-propagation-via-obi-trace-map/) - how a FUSE daemon reads the pinned map and the first two failure axes (population and runtime affinity).
- [PID namespace key mismatch (10-17 entry)](/ebpf-qa/2026-10-17-fuse-trace-context-map-lookup-misses-in-pid-namespace/) - the third, namespace-based miss axis this page complements.

## Community discussion today

Two opt-in archive channels (the two CNCF OpenTelemetry instrumentation channels) produced eleven messages this run. The 24-hour window did not advance from the 10-16 and 10-17 runs, so the substantive threads are the same already-answered ones: the FUSE and OBI trace propagation thread is the 10-14 published entry and its 10-16 and 10-17 follow-ups, and the Cilium eBPF tracepoint thread is the 10-13 published entry. The archive also carried a maintainer review-status update on an OpenTelemetry semantic-conventions pull request (a review update, not a question) and a maintainer caveat on thread affinity, which is the 10-16 axis. With the window frozen and no new question having entered, this entry takes the continuation axis of the same FUSE and OBI thread: not another miss cause, but what a *hit* on the map actually means and when it goes stale, answered from the OBI source and the kernel LRU map documentation.

Channel coverage for this run: the two opt-in archives supplied eleven messages. The visible-browser-only sources - the eunomia-bpf and sched-ext Discord servers, the bpf mailing list, and r/eBPF - could not be reviewed this run because no visible-browser session was available, so they are marked not-covered, not quiet.
