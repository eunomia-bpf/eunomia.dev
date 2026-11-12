# How can a FUSE userspace filesystem daemon inherit OpenTelemetry trace propagation by reading the OBI trace context map keyed by pid and tid?

Feasible, but only when two independent sides both hold. The kernel fills `fuse_in_header.pid` for every FUSE request, so a FUSE userspace daemon can always tell which calling thread issued the request. OpenTelemetry's eBPF instrumenter (OBI) already keeps a pinned map, `traces_ctx_v1`, that holds the trace id and span id of the request a thread is currently serving, keyed by that thread. A FUSE daemon can read that pin and inherit the W3C trace context. But the map is written only when OBI has captured a trace context for that thread and population is enabled; an untracked protocol or an unsupported runtime leaves it empty, and the FUSE lookup misses. The header also carries the thread id, while OBI keys on the process-and-thread pair, so the daemon has to resolve the thread group before it can build the key.

## The mechanism

The boundary is between what the kernel gives the daemon and what OBI has already stored:

1. **FUSE side (kernel).** Every request a FUSE daemon reads from `/dev/fuse` opens with a `fuse_in_header`, and the kernel fills its `pid` field in `fuse_args_to_req` from `fuse_args.pid`, which `fs/fuse/req.c` sets to `task_pid(current)` mapped into the mount's pid namespace. So the daemon receives the calling thread's thread id, in the mount's pid namespace, per request. That is a TID, not a process id: the flock path deliberately uses `task_tgid` instead, and that contrast is what tells you the generic header pid is thread-scoped.
2. **OBI side (trace context map).** OBI pins a `BPF_MAP_TYPE_LRU_HASH` named `traces_ctx_v1`, pinned by name to `/sys/fs/bpf/` by default, with a `u64` key and a two-field value: `trace_id` (16 bytes) and `span_id` (8 bytes), the W3C Trace Context pair. OBI writes an entry at the moment a server span starts, keyed by `bpf_get_current_pid_tgid()`, which packs the thread group and the thread id into one 64-bit value. The same map feeds OBI's own log enricher, and it is the public surface an external eBPF program or a FUSE daemon can read.
3. **The glue (daemon).** To inherit the context the daemon (a) reads `fuse_in_header.pid` to get the calling TID, (b) resolves the thread group id from that TID by reading the `Tgid` field of the caller's `/proc` status page, (c) builds the 8-byte key `(tgid << 32) | tid`, and (d) does a `bpf_map_lookup_elem` on the `traces_ctx_v1` pin to recover the trace id and span id, then seeds or extends the span it creates for the FUSE operation from them.

## Verification and debugging path

1. **Confirm the daemon can see the caller.** Read a request header from `/dev/fuse` and print `fuse_in_header.pid`. Compare it to the thread id of the process doing the write; on a multi-threaded caller it should track the specific worker thread, not just the process.
2. **Confirm OBI actually populated the map.** Turn on OBI's `populate_trace_context` option, then read the `traces_ctx_v1` pin (for example open `/sys/fs/bpf/traces_ctx_v1` and run `bpf_map_lookup_elem` with a known key). A hit returns `trace_id` and `span_id`; a miss means no context was captured for that thread.
3. **Build the key the way OBI does.** Pack the resolved thread group and thread id into the `u64` key before looking up; a TID-only key will not match OBI's `bpf_get_current_pid_tgid()` key.
4. **Isolate the two failure modes.** A lookup miss with population on points at the OBI side (the thread's runtime or protocol produced no W3C context, or its language is not instrumented); a match that is stale points at the thread-affinity side (the FUSE work ran on a different thread than the one OBI stamped).

## The limitation

- `traces_ctx_v1` is opt-in. OBI leaves it off unless something reads it; enable `populate_trace_context` or your external FUSE reader has nothing to look up.
- OBI must have a trace context first. The map holds the context OBI already captured on that thread. An untracked protocol, or a runtime OBI does not instrument (the maintainers flagged C++ as not covered, against Go, Java, Node.js, and Python with asyncio), leaves the entry absent, so the FUSE lookup misses and there is nothing to inherit.
- The key is a thread-group-and-thread pair, but the header only carries the thread. Resolve the thread group from the TID; without it you cannot build the key.
- Thread affinity matters. OBI stamps the context on the thread serving the request. If the FUSE work moves to another thread, that per-thread entry is empty; widening the correlation to the whole thread group is possible but ambiguous when one process serves several concurrent requests.
- The map is `BPF_MAP_TYPE_LRU_HASH`, so under load an entry for a busy thread can be evicted; read it promptly after the request lands.

## References

- [kernel fs/fuse/req.c](https://github.com/torvalds/linux/blob/master/fs/fuse/req.c) - `fuse_args.pid` is set to the caller thread's TID, `task_pid(current)` mapped into the mount's pid namespace.
- [kernel include/uapi/linux/fuse.h](https://github.com/torvalds/linux/blob/master/include/uapi/linux/fuse.h) - the `fuse_in_header.pid` field a FUSE daemon reads.
- [OBI bpf/shared/obi_ctx.h](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/blob/main/bpf/shared/obi_ctx.h) - the pinned `traces_ctx_v1` map, its `u64` key and `trace_id` / `span_id` value, and the OTEP that owns the spec.
- [OpenTelemetry OTEP 4855](https://github.com/open-telemetry/opentelemetry-specification/pull/4855) - the external trace-context map spec OBI implements.
- [W3C Trace Context](https://www.w3.org/TR/trace-context/) - the `traceparent` layout (`trace_id` / `span_id`) the map carries.

## Community discussion today

Two opt-in archive channels (the two CNCF OpenTelemetry instrumentation channels) produced eleven messages this run. The thread this page answers: a practitioner asked how feasible it is to add OpenTelemetry trace propagation to a FUSE program by exporting OBI's trace information through its trace-context map and collecting it in the FUSE daemon using the pid and tid. The archive carried the question and the maintainers' follow-ups - that FUSE is a userspace filesystem that can read the caller's thread, that a thread-group id is reachable through the caller's status page under `/proc`, that Go and Python with asyncio are instrumented while C++ is not, and that the OBI map is populated only when a trace context already exists on that thread. The other two threads in the window had already been answered by an earlier post, or were a review-status update rather than a question, so this run selected the non-duplicative FUSE and OBI trace-propagation question.

Channel coverage for this run: the two opt-in archives supplied eleven messages. The visible-browser-only sources - the eunomia-bpf and sched-ext Discord servers, the bpf mailing list, and r/eBPF - could not be reviewed this run because no visible-browser session was available, so they are marked not-covered, not quiet.
