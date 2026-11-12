# Why do OBI HTTP spans carry high-cardinality names, and how does the OpenTelemetry semantic convention bound span name cardinality?

The span name is the single highest-cardinality field a trace system carries, so the OpenTelemetry HTTP semantic convention pins the name to `{method} {target}` and requires a **low-cardinality** `target`: `http.route` on the server side, `url.template` on the client side. The convention is explicit that an instrumentation **must not** default to using the raw URI path as the target. OpenTelemetry eBPF Instrumentation (OBI) builds its HTTP span names with exactly this shape - a method plus a route - and the only input that makes the name high-cardinality is whatever fills that route slot. The high-cardinality symptom is not the default; it is what the `unmatched: path` option produces when it is selected, and it is precisely the one setting the convention forbids as a default.

A span name that grows with the number of unique URLs you serve is a cardinality problem, not a correctness one: it is the route value that decides whether two requests collapse into one name or fan out into one name each.

## The mechanism

1. **The convention's name rule.** The HTTP semantic convention says an HTTP span name SHOULD be `{method} {target}` when a low-cardinality `target` is available, and SHOULD be just `{method}` when it is not. The `{method}` is `http.request.method`, or the literal `HTTP` when the method is outside the known-methods set. The `target` SHOULD be `http.route` for a server span or `url.template` for a client span, and it notes that `url.template` is the low-cardinality template of a path (for example `/users/{id}` rather than `/users/42`). The hard guardrail is the last sentence of that section: instrumentation must not default to using the URI path as the target.

2. **OBI names an HTTP span as `{method} {route}`.** In `pkg/appolly/app/request/span.go`, the `TraceName` builder for HTTP event types starts from the method - emitting the method as-is, or `HTTP` when the method is not in the known-methods list - and appends the route only when a route has been resolved. A span with no resolved route is therefore just `{method}`. The resolved route is the same value OBI exports as `http.route` on a server span and `url.template` on a client span, so the span name and the exported attribute stay consistent.

3. **Where the route comes from.** In `pkg/transform/routes.go` the route is matched off the URL path, in order: the user-defined `routes.patterns` matcher, then any per-service custom matchers, then a harvested route matcher. When none of those matched, the `unmatched` policy decides the target. That is the knob that sets cardinality: `unset` leaves the target empty so the name is just `{method}`; `wildcard` collapses the target to a single token; `heuristic` clusters similar paths into a low-cardinality template; `low-cardinality` does the same clustering and additionally bounds it with a path-segment cap; and `path` sets the route to the raw URL path.

4. **Why `path` is the high-cardinality case.** When `unmatched` is `path`, the code assigns the raw path directly to the route, so every distinct URL path becomes a distinct span name: one name per `/users/42`, one per `/orders/1934`, and so on. That is an unbounded set that grows with your traffic, which is the cardinality blow-up. The shipped default is the `heuristic` clustering instead, which keeps the name set bounded; the `low-cardinality` option tightens that further with a per-service segment cap. The convention's "must not default to the URI path" is the reason `path` is an opt-in, never the default.

## Verification and debugging path

1. **Read the `routes.unmatched` value first.** The default is `heuristic`, so a fresh OBI deployment already clusters paths into bounded names. If span names are one-per-URL, look for an explicit `routes` block that sets `unmatched` to `path` (or an empty value that fell through to the wildcard in an older layout) - that is the field to change.

2. **Watch the span name and the route attribute together.** For a busy path, list the distinct `http.route` / `url.template` values, not the full `url.full`. Bounded naming shows a handful of template values; raw-path naming shows one value per distinct URL. The span name and that attribute should agree, because they are built from the same route value.

3. **Cap the metric series independently.** Even with bounded names, the span-name metric series is capped by the metric span-name aggregation limit, so a run of pathological names degrades to a truncated metric set rather than an unbounded one. The limit bounds the damage but does not fix the naming.

## The limitation

- The convention's cardinality rule is a mix of SHOULD and MUST: the "must not default to the URI path" is the one hard constraint; choosing route over template, and picking the low-cardinality option, are SHOULD-level guidance, so an operator can still opt into raw-path names with `unmatched: path`.
- `heuristic` and `low-cardinality` are OBI-internal clustering strategies, not a spec: their output depends on the URL-clusterer and the per-service segment cap, so the exact template set for a given workload is OBI behavior, not something the semantic convention pins down.
- `unset` removes the target entirely (name is just `{method}`), the lowest-cardinality option, but it also drops the per-route distinction from the name, so you trade cardinality for detail.
- This answer describes OBI's `main`-branch behavior for HTTP spans. The route-matching pipeline and the `unmatched` option names are OBI specifics; the span-name cardinality rule itself is the shared semantic convention that any HTTP instrumentation follows.

## References

- [OpenTelemetry semantic conventions, HTTP span name rule](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/http/http-spans.md) - the `Name` section: the name is `{method} {target}`, the target SHOULD be `http.route` or `url.template`, and instrumentation must not default to using the URI path as the target.
- [OBI `pkg/appolly/app/request/span.go`](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/blob/main/pkg/appolly/app/request/span.go) - `TraceName` builds the HTTP span name as the method plus the resolved route, with the method reduced to `HTTP` when it is unknown.
- [OBI `pkg/transform/routes.go`](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/blob/main/pkg/transform/routes.go) - where the route is matched from the URL path and where the `unmatched` policy, including the raw-path case, assigns the target.
- [OBI `pkg/obi/config.go`](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/blob/main/pkg/obi/config.go) - the shipped defaults: `unmatched` set to the heuristic clustering, a `*` wildcard character, and a path-segment cardinality cap.
- [OBI span attribute catalog, `spans.md`](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/blob/main/site/docs/spans.md) - the per-span-type attribute reference showing OBI emits HTTP client and server spans and carries the URL and route attributes.

## Community discussion today

The two opt-in CNCF OpenTelemetry instrumentation channels supplied a three-message frozen window this run. The substantive thread, and the one this entry answers, is a practitioner flagging that, as OBI's HTTP span naming is currently proposed, the span name can end up high-cardinality - in short, that a span whose target is the raw per-request URL path is a distinct name for every unique path, which is exactly the cardinality blow-up the HTTP semantic convention's "must not default to the URI path" guardrail exists to prevent. The working follow-up in that thread is to bound the target to a route or template rather than the raw path.

The same window carried one separate, unresolved line - a request for a pointer to a "skill refinement" item that was mentioned - which is its own thread and is not folded into this answer.

Channel coverage for this run: the two opt-in archives supplied three messages. The visible-browser-only sources - the eunomia-bpf and sched-ext Discord servers, the bpf mailing list, and r/eBPF - could not be reviewed this run because no visible-browser session was available, so they are marked not-covered, not quiet.
