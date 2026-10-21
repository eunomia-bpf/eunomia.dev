# Why can an OpenTelemetry OBI metric stay in the store up to twice as long as its TTL, and does the Prometheus exporter drop the same label set on every scrape?

The answer splits on the exporter. OBI caps per-label-set metric cardinality in both its OpenTelemetry and Prometheus exporters with one shared TTL, but the two exporters delete at two different moments. The OTEL exporter deletes **lazily**: a sweep runs only when the record path next touches the exporter, and because that sweep is itself gated to at most once per TTL, a dead label set can remain visible for up to **two times the TTL** after its last update. The Prometheus exporter deletes on **every scrape**: its collector calls the expiry sweep inside `Collect`, so a dead label set disappears on the first scrape that lands after the TTL has elapsed. Both default to the same five-minute TTL.

A metric series that outlives the label set that produced it is a cardinality leak, not a correctness bug: nothing is wrong with the value, only with how long the dead series keeps occupying memory and scrape payload.

## The mechanism

1. **One shared expiry store, keyed by label set.** `pkg/export/expire/expiry_map.go` defines `ExpiryMap`, a synchronized map from a label-set key to an entry holding the value plus a `lastAccess` time. `GetOrCreate` is the only mutator on the hot path: a miss inserts a new entry, a hit refreshes `lastAccess`. `DeleteExpired` sweeps the map and returns every entry whose `lastAccess` is strictly older than `now - ttl` (the cutoff is `now.Sub(e.lastAccess) > ex.ttl`, so an entry dies just after one TTL of quiet), and `ttl == 0` disables expiration entirely. The key is built by `LabelsKey`, which encodes each label value as its length followed by the value, so `["a", "bc"]` and `["ab", "c"]` cannot collide.

2. **The OTEL expirer is record-driven and gated to once per TTL.** `pkg/export/otel/expirer.go` wraps an `ExpiryMap[attribute.Set]` and holds a `lastExpiration` clock. `ForRecord` is where deletion happens: before returning a data point it checks `now.Sub(ex.lastExpiration) >= ex.ttl`, and only then calls `removeOutdated`, which sweeps the map and, for each expired attribute set, calls `metric.Remove` with that set as the removal option to actually detach the data point from the SDK metric. Because the sweep is skipped whenever the previous sweep ran less than one TTL ago, an entry that expired `TTL` ago can only be found by the next sweep, which no earlier than `TTL` after that - hence up to `2*TTL` total, which is exactly what the code comment states: "an expired metric might stay visible after 2*TTL time after not being updated". The sweep is not a background timer: it runs inside the record paths that call `ForRecord`, the RED duration and size metrics in `pkg/export/otel/metrics.go` and the service-graph totals in `pkg/export/otel/metrics_svc_graph.go`. `RemoveAllMetrics`, the explicit shutdown hook, deletes every entry immediately.

3. **The Prometheus expirer is scrape-driven.** `pkg/export/prom/expirer.go` wraps the same `ExpiryMap` around a `prometheus.MetricVec` and exposes it as a `Collector`. Its `Collect` calls `ex.entries.DeleteExpired()` first, then deletes each returned label set from the wrapped vector with `DeleteLabelValues` before emitting the survivors. There is no `lastExpiration` gate: every scrape is a sweep, so the maximum staleness is one TTL plus the scrape interval. `DeleteLabelValues` on the expirer gives immediate manual deletion of a single label set, which is what the per-process runtime-metric wrappers such as `prom_nodejs.go` use when a Node.js process disappears.

4. **The TTL value and where it comes from.** `pkg/obi/config.go` sets `defaultMetricsTTL = 5 * time.Minute` and assigns it to both `OTELMetrics.TTL` and `Prometheus.TTL`. The OTEL knob is `TTL` on the OTEL metrics config, exposed as YAML `ttl` and environment `OTEL_EBPF_METRICS_TTL`; the Prometheus knob is the same YAML `ttl` and environment `OTEL_EBPF_PROMETHEUS_TTL` on the Prometheus config. One TTL value, two deletion schedules.

## Verification and debugging path

1. **Check which exporter is configured and read the effective TTL.** The OTEL exporter reads `OTEL_EBPF_METRICS_TTL` (YAML `ttl` under the OTEL metrics block); the Prometheus exporter reads `OTEL_EBPF_PROMETHEUS_TTL` (YAML `ttl` under the Prometheus block). Either one unset falls back to the five-minute default, so a "stale series" that is younger than five minutes is not yet a configuration question.

2. **Observe the difference directly on the Prometheus endpoint.** For a label set that stops producing traffic - a route or a service that goes idle - the Prometheus `/metrics` output drops it on the first scrape after one TTL of quiet, and the next scrape is already clean. On an OTEL collector, the same series can survive two scrapes after the TTL; the sweep that removes it only happens when a record for *some* label set passes through the exporter again. If OTEL metrics look consistently "fresher-boundary" while the Prometheus endpoint looks tight, that is the expected shape, not a bug.

3. **Watch the debug logs.** Both expirers log at debug level: the OTEL expirer logs `deleting old OTEL metric` with the attribute set, the Prometheus expirer logs `deleting old Prometheus metric` with the label values, and `GetOrCreate` logs `storing new metric label set`. Raising log level to debug on one exporter at a time turns the sweep boundary into something you can read off the timeline.

4. **Bound the worst case instead of waiting.** When churn is high - many short-lived routes or processes - the two-TTL OTEL tail plus the scrape interval is the effective retention for dead series. Shortening the TTL is the only operator control on both sides; the gate itself is not configurable.

## The limitation

- The OTEL expirer's sweep is gated on the record path, not on a wall-clock timer: when the exporter is silent, the sweep does not run, so "up to two TTLs" is the bound, not a guarantee. A fully idle OTEL pipeline keeps its last label sets indefinitely because nothing calls `ForRecord` to trigger the sweep.
- The two exporters do not share one semantic for the same TTL value: OTEL is bounded by `2*TTL` of quiet, Prometheus by `TTL` plus the scrape interval. Operators reading one number for both will misjudge which series a dead service leaves behind.
- `ttl: 0` disables OBI expiration completely on the side that is set to zero, which trades the cardinality leak for unbounded label-set growth; it is an explicit opt-in, not a default.
- `RemoveAllMetrics` only exists on the OTEL expirer, and it is the shutdown path for one service instance's reporter; it does not run on ordinary quiet periods, so it is not a substitute for the TTL.
- This answer describes OBI's `main`-branch behavior. The `ExpiryMap` shape, the once-per-TTL gate, and the scrape-driven `Collect` sweep are OBI implementation details; the semantic - label sets should expire after a quiet period - is generic cardinality management shared with any per-label metric store.

## References

- [OBI `pkg/export/expire/expiry_map.go`](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/blob/main/pkg/export/expire/expiry_map.go) - the `ExpiryMap` store: `GetOrCreate` refreshes last access on a hit, `DeleteExpired` cuts at `now.Sub(lastAccess) > ttl`, `LabelsKey` makes the label-set key unambiguous.
- [OBI `pkg/export/otel/expirer.go`](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/blob/main/pkg/export/otel/expirer.go) - `ForRecord`'s once-per-TTL sweep gate and its "up to 2*TTL" comment, `removeOutdated` calling `metric.Remove`, and `RemoveAllMetrics` for reporter shutdown.
- [OBI `pkg/export/prom/expirer.go`](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/blob/main/pkg/export/prom/expirer.go) - the Prometheus expirer: `Collect` calls `DeleteExpired` on every scrape and deletes the returned label sets from the wrapped `MetricVec`.
- [OBI `pkg/obi/config.go`](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/blob/main/pkg/obi/config.go) - `defaultMetricsTTL = 5 * time.Minute` assigned to `OTELMetrics.TTL` and `Prometheus.TTL`.
- [OBI `pkg/export/prom/prom.go`](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/blob/main/pkg/export/prom/prom.go) - the Prometheus config's `TTL` field, YAML `ttl`, environment `OTEL_EBPF_PROMETHEUS_TTL`.

## Community discussion today

There was no community discussion to summarize today: the monitored window covered by this run returned no messages at all - the two opt-in archives came back empty for the 24-hour window (`result=ok`, zero messages) - so this entry answers a standing question drawn from the public primary sources above instead of a live thread, and no anonymized discussion is attached.

Channel coverage for this run: the two opt-in archive channels returned zero messages in the monitored window. The visible-browser-only sources - the eunomia-bpf and sched-ext Discord servers, the bpf mailing list, and r/eBPF - could not be reviewed this run because no visible-browser session was available, so they are marked not-covered, not quiet.
