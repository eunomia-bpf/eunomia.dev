# For a Node.js service instrumented by OBI, why can't I just turn off the per-callback trace-context sentinel because the service doesn't use manual spans or log enrichment?

**Short answer:** because the sentinel is not a feature of OBI's; it is the write side of a shared, pinned kernel map whose consumer set extends beyond OBI's own configuration. The injected Node agent's `async_hooks` `before` hook (`fs.existsSync("/dev/null/obi-ctx/<fd>")` fired before every JS callback) keeps `traces_ctx_v1` — a name-pinned LRU hash keyed by pid/tgid whose value is the active request's `{trace_id, span_id}` — aligned with the request currently executing on that thread. OBI's own consumers (manual-span parenting, log trace annotation) are only part of the consumer set: the map's own header declares its spec to be part of an OTEP and warns that changing it "may break other components relying on it," and an out-of-process eBPF profiler reading the pinned map to correlate its profiles to OBI's traces is a consumer that OBI's config selectors cannot see. So a gate computed from "manual spans off + log annotation off" looks complete — and silently drops the external trace/profile correlation channel.

## What the sentinel actually maintains

The cost story (the per-callback sentinel is where the injected agent's overhead lives, not the eBPF probes) is already the answer to the p99 question; this one is about *safety of disabling it*, which is a different boundary.

On the agent side (`pkg/internal/nodejs/fdextractor.js`), the trace-context propagation machinery is deliberately the expensive part: "net prototype wraps + an async_hooks before hook firing on every callback," and the agent is templated with `TRACES_ENABLED` so that metrics-only injections "skip [it] entirely." The `before` hook fires before every JS callback and signals the kernel through a `fs.existsSync` on a fixed path — safe inside `async_hooks` because a synchronous fs operation does not create an `AsyncWrap`, so it cannot re-trigger the hook. Two sentinel forms exist:

- `/dev/null/obi-ctx/<fd>` — refresh: the 4-digit incoming fd of the current async context, so the kernel map reflects the active request;
- `/dev/null/obi-noreqctx` — clear: emitted on the request-to-no-request transition (a background timer, a callback after its request finished), so a later span is not parented into the previous request's trace.

On the kernel side (`bpf/generictracer/nodejs.c`), those paths are decoded by uprobe handlers on the synchronous fs call: `handle_async_switch` refreshes the map, `handle_ctx_clear` deletes the entry, and `handle_node_span` reads it. The map itself (`bpf/shared/obi_ctx.h`) is:

```c
struct {
    __uint(type, BPF_MAP_TYPE_LRU_HASH);
    __type(key, u64);                 // pid/tgid
    __type(value, obi_ctx_info_t);    // { trace_id, span_id }
    __uint(max_entries, 1 << 14);
    __uint(pinning, LIBBPF_PIN_BY_NAME);
} traces_ctx_v1 SEC(".maps");
```

with the decisive comment above it: "NOTE: this map spec is part of an OTEP (open-telemetry/opentelemetry-specification#4855). Changing its spec may break other components relying on it."

## Who actually reads `traces_ctx_v1`

Three consumer classes matter:

1. **Manual spans.** `spanbridge.js` emits a manual span's end via `/dev/null/obi-span/<json>`; the kernel's `handle_node_span` stamps the event's parent ids from `obi_ctx__get(pid_tgid)` so the manual span "can be parented under OBI's automatic server span." Without the per-callback refresh, a manual span that ends inside an async callback would carry a stale or missing parent.
2. **Log trace annotation.** OBI's config schema exposes a log trace-annotation block carrying `trace_id`/`span_id` field names (`internal/config/schema/correlation.go`). For an instrumented Node.js process, the trace context a log line is attributed to for that thread is exactly the per-thread context the sentinel maintains in the map — so the annotation only stays correct while the sentinel keeps the map aligned with the active request.
3. **External trace/profile correlation.** The OTEP that standardizes correlating OBI traces to profiles (open-telemetry/opentelemetry-specification#4855; PoC at OBI PR #1184 and the Coralogix eBPF profiler) is precisely the OTEP the map header declares the spec to be part of. An out-of-process eBPF profiler reading the pinned map to join its own profile samples to OBI's trace ids is this third consumer. The merged process-context OTEP (`oteps/profiles/4719-process-ctx.md`) is the broader spec-level channel for external eBPF profilers to read per-process context.

## Why the OBI config gate cannot see the third one

The gate that people reach for is expressed entirely in OBI's feature vocabulary: "manual spans disabled, log annotation disabled, so the trace-context machinery is unused — skip it." That is exactly the `TRACES_ENABLED` switch, and it is a reasonable *cost* gate. But it is under-complete by construction, because the map's consumer set is not defined by OBI's feature toggles. The pinning (`LIBBPF_PIN_BY_NAME`) is the signal that out-of-process readers exist: a component that does not participate in OBI's configuration reads the map through the pin, and no OBI config selector says "but keep the context alive for the external correlation integration." When you compute the gate from OBI's own consumers only, the result *looks* complete — OBI's manual spans and log annotation are off, so nothing in OBI complains — and the external trace/profile correlation channel goes dark with no error anywhere in OBI.

## How to gate it safely

1. **Enumerate the map's consumers, not just OBI's features.** The pinned `traces_ctx_v1` has manual-span, log-annotation, and external-correlation consumers. Any gate that drops the sentinel must account for every consumer that reads the map in your deployment.
2. **Keep the sentinel if any out-of-process profiler correlates to OBI's traces.** If you run an eBPF profiler that joins its profile samples to OBI trace ids (the trace/profile correlation OTEP's integration), the context the profiler reads is what the sentinel maintains — disable the gate regardless of OBI's feature toggles.
3. **Use the metrics-only path only when nothing reads the map.** The `TRACES_ENABLED=false` metrics-only injection skips trace-context propagation wholesale — the right choice when no consumer (manual spans, log annotation, external correlation) needs the map, and the wrong one when any of them does.
4. **Verify the channel, not the absence of errors.** After a gating change, check that manual spans are still parented under the automatic server span and that the external tool's profile-to-trace join still works. The correlation consumer failing produces no error in OBI; the symptom is a gap in the external tool's correlation.

## The limitation that decides it

The boundary is *who owns the consumer set of a pinned, spec-designated map*. Because the map is name-pinned and its spec is declared part of an OTEP — a cross-component contract — its consumers extend past OBI's configuration surface. A gate computed only from OBI's feature toggles can express "no OBI consumer," but it cannot express "no consumer at all," and the third consumer is precisely one that lives outside OBI's config. The practical rule: gate on the consumer set, not on the feature toggles; treat the map's pinning as the signal that external readers exist, and keep the sentinel alive whenever one does.

## References

- [OBI — `bpf/shared/obi_ctx.h` (the pinned `traces_ctx_v1` map: LRU hash keyed by pid/tgid, value `{trace_id, span_id}`, `LIBBPF_PIN_BY_NAME`; "this map spec is part of an OTEP … Changing its spec may break other components relying on it")](https://raw.githubusercontent.com/open-telemetry/opentelemetry-ebpf-instrumentation/main/bpf/shared/obi_ctx.h)
- [OBI — `bpf/generictracer/nodejs.c` (the sentinel decoders: `handle_async_switch` refreshes the map, `handle_ctx_clear` deletes the stale entry, `handle_node_span` parents a manual span via `obi_ctx__get`)](https://raw.githubusercontent.com/open-telemetry/opentelemetry-ebpf-instrumentation/main/bpf/generictracer/nodejs.c)
- [OBI — `pkg/internal/nodejs/fdextractor.js` (the agent-side `async_hooks` `before` hook, the `obi-ctx`/`obi-noreqctx` sentinels, and the `TRACES_ENABLED` cost gate "skipped entirely for metrics-only injections")](https://raw.githubusercontent.com/open-telemetry/opentelemetry-ebpf-instrumentation/main/pkg/internal/nodejs/fdextractor.js)
- [OBI — `internal/config/schema/correlation.go` (the log trace-annotation config block carrying `trace_id`/`span_id` field names)](https://raw.githubusercontent.com/open-telemetry/opentelemetry-ebpf-instrumentation/main/internal/config/schema/correlation.go)
- [OTEP: correlating OBI traces to profiles (spec PR #4855, closed unmerged; PoC at OBI PR #1184 and the Coralogix eBPF profiler)](https://github.com/open-telemetry/opentelemetry-specification/pull/4855)
- [OTEP — Process Context: Sharing Resource Attributes with External Readers (the merged spec-level channel for external eBPF profilers; OBI applicability noted)](https://raw.githubusercontent.com/open-telemetry/opentelemetry-specification/main/oteps/profiles/4719-process-ctx.md)
- [Node.js — `async_hooks` API (the `before` hook fires before the corresponding callback of each async operation; synchronous fs operations create no `AsyncWrap`)](https://nodejs.org/api/async_hooks.html)

## Community discussion today

The monitored window is a rolling week across two allowlisted read-only Slack archives (both OpenTelemetry instrumentation channels); the two visible-browser chat workspaces and the public mailing-list and subreddit surfaces were not reviewed in this run, and that gap is noted rather than treated as quiet. Four threads recur from the prior days' windows, each of which was already published: the OBI config v1-to-v2 migration contract, the GenAI agent span shape, the hosted agent-harness observability boundary, and the Node.js sentinel cost and agent lifecycle. The one materially new, source-groundable boundary in this run's slice is the Node.js sentinel gating question above.

**Node.js sentinel gating and the hidden consumer (the question above).** The recurring Node.js cost thread — eBPF probes are cheap, the injected agent's per-callback sentinel is the cost — raised the follow-up that this page answers: whether the sentinel can be gated off when a deployment does not use manual spans or log enrichment. The working answer is the consumer-set boundary: the sentinel is the write side of a pinned, OTEP-designated map, and its consumer set extends past OBI's own feature toggles. A maintainer-confirmed read (anonymized) is that client-span parenting comes from the fd-pair map, the per-callback sentinel keeps the trace-context map aligned with the active request, and a third consumer (external trace/profile correlation) also reads the pinned map — so gating the sentinel on manual spans and log enrichment alone could silently break that integration. The public source ground for that read is the map header's OTEP contract note, the `handle_node_span` parent lookup, and the trace-to-profile OTEP and its profiler PoC; the safe rule is to gate on the map's consumer set, with the pinning as the signal that external readers exist.
