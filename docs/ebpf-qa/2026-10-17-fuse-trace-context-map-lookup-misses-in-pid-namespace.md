# Why do FUSE daemon lookups in the OBI trace context map miss when the FUSE filesystem runs inside a PID namespace?

Because the two sides of the lookup are keyed in two different PID namespaces. The kernel fills `fuse_in_header.pid` with the calling thread's thread id re-based into the mount's PID namespace, while OBI keys its `traces_ctx_v1` trace context map with `bpf_get_current_pid_tgid`, which is the thread's value in the init PID namespace. When the FUSE daemon (or the caller) lives in a PID namespace that differs from the one OBI sees, the two numbers describe the same thread but are not equal, so the key the FUSE daemon builds from the header finds no entry and the lookup misses. This is a third failure axis that is independent of whether OBI population is enabled and of which thread OBI associated with the context: the 10-14 and 10-16 entries cover those two axes, and this one shows up whenever the PID numbers themselves do not line up.

## The mechanism

1. **FUSE side (kernel).** The kernel fills `fuse_in_header.pid` for every request. In `fuse_fill_creds` it sets `args->pid` to `task_pid(current)` re-based by `fc->pid_ns`, which was captured at mount time from the mounting process's active PID namespace. So the header pid is the calling thread's thread id translated into the mount PID namespace. If the mount and the caller share that namespace the number equals the global thread id, but if the caller is in a child PID namespace the number is re-based and is not the init namespace value.
2. **OBI side (trace context key).** OBI stamps `traces_ctx_v1` at span start using `bpf_get_current_pid_tgid`. That helper reads the global thread id (`current->tgid` and `current->pid`), which is the value in the init PID namespace, and packs it into the `u64` key. OBI's own log enricher and any external reader key the map the same way, so the map is keyed in the init PID namespace regardless of the namespaces any individual workload runs in.
3. **The mismatch.** The FUSE daemon takes `fuse_in_header.pid`, resolves the calling thread group id, and packs `(tgid << 32) | tid` as the key it looks up in `traces_ctx_v1`. But that `(tgid, tid)` pair is expressed in the mount PID namespace, while every entry OBI wrote was keyed in the init PID namespace. Unless the two namespaces are the same, the packed key does not match OBI's key and the lookup returns no entry even though OBI did stamp a trace context for that very thread.

## Verification and debugging path

1. **Confirm the two namespaces differ.** Compare `fuse_in_header.pid` against the same thread's id as seen from the root PID namespace (for example by reading the thread's global pid from the host, or by resolving the mount PID namespace against the root namespace). If they differ, the FUSE header pid and OBI's key are in different namespaces and the lookup is guaranteed to miss for that key.
2. **Confirm OBI actually has an entry for the global thread.** Look up `traces_ctx_v1` with the init namespace packed key. A hit confirms OBI stamped a context for that thread; combined with a miss on the re-based key it isolates the namespace mismatch as the cause rather than an empty map.
3. **Align the namespaces before the lookup.** Run the FUSE daemon so its PID namespace matches the one OBI keys in (the init namespace, or a namespace OBI resolves to), or resolve the host visible thread id for the FUSE operation before building the key. Only when the FUSE reported number and OBI's key are in the same PID namespace does the lookup succeed.

## The limitation

- The mismatch is silent. Neither side reports an error when the namespaces differ; the lookup simply returns no entry, so the daemon sees a miss that looks identical to an empty map or a thread OBI never stamped.
- It is independent of OBI population. Even with OBI fully populating `traces_ctx_v1` and the correct thread associated with the context, a re-based FUSE header pid still produces a key that OBI never wrote.
- The kernel does not bridge the two namespaces for the FUSE daemon. The header carries only the re-based pid; there is no field in `fuse_in_header` that carries the init namespace thread id, so the daemon must resolve it out of band if it needs to key the map OBI's way.
- libfuse documents the zero case: when the FUSE filesystem runs in a PID namespace but is accessed from outside that namespace, there is no valid pid to report and the field is zero, so the lookup misses on a degenerate key as well.

## References

- [kernel fs/fuse/req.c](https://github.com/torvalds/linux/blob/master/fs/fuse/req.c) - `fuse_fill_creds` sets `args->pid` to the caller thread's `task_pid(current)` re-based into the mount PID namespace via `fc->pid_ns`.
- [kernel fs/fuse/inode.c](https://github.com/torvalds/linux/blob/master/fs/fuse/inode.c) - `fuse_conn_init` captures `fc->pid_ns` from the mounting process's active PID namespace.
- [kernel include/uapi/linux/fuse.h](https://github.com/torvalds/linux/blob/master/include/uapi/linux/fuse.h) - the `fuse_in_header.pid` field the FUSE daemon reads.
- [kernel kernel/bpf/helpers.c](https://github.com/torvalds/linux/blob/master/kernel/bpf/helpers.c) - `bpf_get_current_pid_tgid` reads the global thread id; the namespace aware variant `bpf_get_ns_current_pid_tgid` reads a per namespace value.
- [libfuse include/fuse_lowlevel.h](https://github.com/libfuse/libfuse/blob/master/include/fuse_lowlevel.h) - notes that the reported client pid may be zero when the FUSE filesystem runs in a PID namespace but is accessed from outside it.
- [OBI bpf/shared/obi_ctx.h](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/blob/main/bpf/shared/obi_ctx.h) - the pinned `traces_ctx_v1` map, its `u64` key and `trace_id` / `span_id` value.
- [OBI trace context map (10-14 entry)](/ebpf-qa/2026-10-14-fuse-otel-trace-propagation-via-obi-trace-map/) - how a FUSE daemon reads the pinned OBI trace context map keyed by the caller thread, and the two failure axes this page adds to.

## Community discussion today

Two opt-in archive channels (the two CNCF OpenTelemetry instrumentation channels) produced eleven messages this run. The window did not advance from the previous day, so its substantive threads are the same ones already answered: the FUSE and OBI trace propagation thread is the 10-14 published entry, and the Cilium eBPF tracepoint thread is the 10-13 published entry. The archive also carried a maintainer review-and-bump of an OpenTelemetry semantic-conventions pull request for a unified agent to agent interaction model (a review status update, not a question) and a maintainer caveat that the FUSE plan works only if the thread that sets the trace context is the same thread doing the FUSE work. Because that caveat names the thread affinity axis already covered by the 10-16 entry, this page takes the adjacent axis that the same thread sets up but no entry has answered: how the PID namespace between the FUSE header pid and OBI's key breaks the lookup even when population and thread association are both correct.

Channel coverage for this run: the two opt-in archives supplied eleven messages. The visible-browser-only sources - the eunomia-bpf and sched-ext Discord servers, the bpf mailing list, and r/eBPF - could not be reviewed this run because no visible-browser session was available, so they are marked not-covered, not quiet.
