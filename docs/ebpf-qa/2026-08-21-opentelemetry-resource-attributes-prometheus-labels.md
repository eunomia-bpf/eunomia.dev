# Should an OpenTelemetry metrics producer copy service identity into every data point?

**Short answer:** not in its canonical OTLP output. `service.name`, `service.namespace`, and `service.instance.id` describe the entity producing telemetry, so they belong on the OpenTelemetry `Resource`. Repeating them in every metric point creates a second source of truth and turns a resource-to-backend mapping problem into producer-specific telemetry.

Prometheus is the important exception at the **export boundary**, not at the producer boundary. Prometheus has no native OpenTelemetry Resource object. An exporter may therefore translate service identity into the conventional `job` and `instance` target labels, expose other selected resource attributes through `target_info`, or deliberately promote a small allowlist to metric labels. That translation should be explicit and backend-specific. An OTLP producer should remain canonical even when the same process also exposes a Prometheus endpoint.

## Resource attributes and point attributes answer different questions

OTLP metrics are not a flat bag of labeled samples. The protocol groups `ScopeMetrics` beneath `ResourceMetrics`, and each `ResourceMetrics` message carries one Resource. OpenTelemetry's metrics data model includes the originating Resource in a metric stream's identity, alongside instrumentation scope, metric name, and the attributes attached to data points.

The distinction is semantic:

- **Resource attributes** answer “what entity produced this telemetry?” Examples include service identity, host identity, cloud region, container identity, and deployment environment.
- **Data-point attributes** answer “which dimension of this measurement is this point about?” Examples include HTTP method, status code, RPC method, device, or queue.

The service semantic conventions define `service.name`, `service.namespace`, and `service.instance.id` as the identity triplet for a service instance. Moving or copying that triplet into every data point does not add information to OTLP. It changes the metric stream, increases repeated payload and label work, and lets the resource and point disagree.

A useful invariant is:

```text
one observed service identity -> one canonical Resource
measurement dimensions       -> data-point attributes
backend compatibility        -> exporter translation
```

If a data point already carries a key with the same name as a Resource attribute, define precedence before deployment. Silently accepting whichever value a backend happens to retain makes queries depend on exporter order rather than telemetry semantics.

## Why Prometheus needs translation

Prometheus stores labeled time series and scrape-target metadata; it does not carry an OTLP Resource beside every group of metrics. The OpenTelemetry Prometheus/OpenMetrics compatibility specification therefore defines a mapping rather than requiring producers to flatten the Resource themselves.

For service identity, the conventional mapping is:

```text
service.namespace + service.name -> job
service.instance.id               -> instance
other resource attributes         -> target_info or selected labels
```

`target_info` is an info metric whose labels describe the target. With one such series for a `(job, instance)` pair, PromQL can join resource metadata onto application metrics when needed. This keeps resource metadata out of every stored series while retaining it for queries.

The Collector's Prometheus exporter also offers `resource_to_telemetry_conversion.enabled`. When enabled, it copies **all** Resource attributes to metric labels; its documented default is `false`. This is a compatibility lever, not a generally safe default. Broad promotion can multiply series, expose attributes that were never intended as metric dimensions, create conflicts with point attributes, and make dashboards accidentally depend on backend-specific flattening. The exporter documentation recommends selectively copying commonly needed attributes with a transform when that is the desired contract.

## Keep two output paths independent

A producer that supports both OTLP push and a Prometheus scrape endpoint should model them as two encoders over one internal Resource, not as one flattened metric representation reused everywhere.

### OTLP path

1. Put service identity and other entity metadata on the Resource.
2. Put only measurement dimensions on data points.
3. Send canonical `ResourceMetrics` to the Collector or backend.
4. Let the receiving pipeline decide whether its storage model needs resource promotion.

### Prometheus path

1. Derive `job` and `instance` from the same internal service identity.
2. Produce one consistent `target_info` series where the selected compatibility profile requires it.
3. Promote only an explicit allowlist of additional Resource attributes.
4. Resolve label-name collisions deterministically and document the rule.

This separation matters even if both exporters live in the same binary. A current upstream design issue for eBPF-based OpenTelemetry instrumentation documents how independently evolved OTLP and Prometheus exporters can drift: names can map differently, target identity can be omitted, and a producer-created info metric can collide with an exporter-generated `target_info`. The issue is still open, so it is evidence of a compatibility problem and proposed direction, not proof of shipped behavior.

## A migration plan that does not break dashboards

Before removing duplicated service labels, capture the actual contract of each path.

1. **Inspect OTLP structurally.** Verify that service identity appears once on the Resource and that point attributes contain only intended measurement dimensions.
2. **Inspect the Prometheus endpoint.** Record the labels on representative counters, histograms, and `target_info`. Confirm that there is one coherent target identity.
3. **Inventory queries and recording rules.** Separate queries that group directly by copied service labels from queries that use `job`, `instance`, or an info-metric join.
4. **Choose a stable Prometheus profile.** Define the `job`/`instance` mapping, the exact Resource-attribute allowlist, collision precedence, and whether `target_info` is emitted.
5. **Canary the exporter change.** Compare series count, label cardinality, scrape size, missing-series alerts, and representative dashboard results.
6. **Remove producer duplication last.** Keep a bounded compatibility window if consumers cannot migrate atomically, then delete the old point labels rather than maintaining two permanent identities.

Test absence as well as presence. A conformance fixture should fail if a service identity key unexpectedly reappears on every OTLP point, if two different Resources collapse onto one Prometheus `(job, instance)`, or if two `target_info` producers create conflicting series.

## Where should the fix live?

Use this ownership rule:

- If the output is OTLP, fix the producer so it emits a correct Resource and does not duplicate resource identity into points.
- If a Prometheus consumer needs resource fields as labels, configure or fix the Prometheus exporter.
- If different backends need different projections, perform the transformation in backend-specific Collector pipelines.
- If a dashboard depends on accidental duplication, migrate the dashboard; do not redefine the canonical telemetry model around that accident.

The goal is not to prohibit labels. It is to keep one authoritative service identity and make every flattening step visible, testable, and reversible.

## References

- [OpenTelemetry service Resource semantic conventions](https://opentelemetry.io/docs/specs/semconv/resource/service/)
- [OTLP metrics protocol: `ResourceMetrics` groups a Resource with scope metrics](https://github.com/open-telemetry/opentelemetry-proto/blob/main/opentelemetry/proto/metrics/v1/metrics.proto)
- [OpenTelemetry Metrics Data Model](https://opentelemetry.io/docs/specs/otel/metrics/data-model/)
- [OpenTelemetry Prometheus and OpenMetrics compatibility specification](https://github.com/open-telemetry/opentelemetry-specification/blob/main/specification/compatibility/prometheus_and_openmetrics.md)
- [OpenTelemetry Collector Prometheus exporter configuration](https://github.com/open-telemetry/opentelemetry-collector-contrib/blob/main/exporter/prometheusexporter/README.md)
- [Open eBPF instrumentation issue on aligning OTLP and Prometheus metric resources](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/issues/2974)
- [Current OpenTelemetry GenAI span conventions](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-spans.md)
- [Open GenAI semantic-conventions issue on cost attributes](https://github.com/open-telemetry/semantic-conventions-genai/issues/101)
- [BPF thread: validate attach type when updating a cgroup BPF link](https://lore.kernel.org/bpf/1ab678ef-6349-4374-9ebf-22f857211ca7@linux.dev/T/#t)
- [BPF thread: make arena page faults reclaim-capable under memory cgroup limits](https://lore.kernel.org/bpf/20260821050250.35112-1-jiayuan.chen@linux.dev/T/#t)
- [BPF thread: preserve AF_XDP transmit-metadata ABI and batching state](https://lore.kernel.org/bpf/20260819160535.1472459-1-sdf@fomichev.me/T/#t)
- [RFC 1982: serial-number arithmetic](https://www.rfc-editor.org/rfc/rfc1982.html)

## Community discussion today

Today's ordinary visible-browser review covered all 6 approved communities and all 15 allowlisted channels or public pages. Every target was accessible. The selected question came from the 24-hour window, so the seven-day fallback was not used. Names, accounts, employers, workspace and channel identities, closed-chat links, exact times, private topology, raw logs, and searchable chat wording have been removed. No raw transcript was retained.

### Metric producers and backends need a single owner for service identity

The strongest discussion concerned an eBPF-based metrics producer that represented the same service fields both as an OTLP Resource and as point attributes. Participants were trying to preserve useful Prometheus labels without making OTLP consumers pay for duplicated metadata. The mechanism is the model mismatch above: OTLP has `ResourceMetrics`, while Prometheus needs target labels or an info metric.

The immediate diagnostic is to compare the OTLP envelope, the scrape output, and the Collector's resource-conversion setting. The compatible path is to retain canonical Resource attributes, map service identity to `job` and `instance` for Prometheus, and selectively promote only proven query dimensions. The remaining design question is how much of that mapping should be configurable in a built-in exporter versus delegated to the Collector; the [upstream alignment issue](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/issues/2974) is open and should not be described as implemented behavior.

### GenAI cost telemetry still lacks a released common shape

A separate observability question asked whether model-call cost should have a standard span attribute. Some instrumentation already emits vendor-specific cost fields, but the [current GenAI span convention](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-spans.md) does not define a stable common cost attribute. An [open semantic-conventions issue](https://github.com/open-telemetry/semantic-conventions-genai/issues/101) still asks whether cost belongs in the core convention or an extension and how its unit should be represented.

For now, keep vendor-emitted fields namespaced, record currency and pricing-table version when deriving cost, and normalize them only in a versioned internal schema. Do not relabel a vendor field as a standard OpenTelemetry attribute before the convention resolves its unit and ownership. The unresolved item is not merely the attribute name: cross-provider comparison also needs a currency, price effective date, and a rule for cached or discounted tokens.

### Kernel review focused on compatibility checks at replacement and fault boundaries

Public BPF work exposed three distinct cases where a successful fast path did not prove that replacement or pressure paths were safe. A cgroup BPF-link update checked the broad program type but could accept an incompatible attach flavor; the proposed fix validates the attach type during replacement. Arena page faults under a memory-cgroup limit could reach a non-reclaiming allocation path and report a valid, pressure-induced fault as `SIGSEGV`; the proposed direction moves reclaim-capable allocation outside the locked region. AF_XDP transmit-metadata review found both a cross-ABI padding concern and state that must remain correct when zero-copy batching reuses descriptors.

The public evidence is the [cgroup-link update discussion](https://lore.kernel.org/bpf/1ab678ef-6349-4374-9ebf-22f857211ca7@linux.dev/T/#t), [arena memory-pressure series](https://lore.kernel.org/bpf/20260821050250.35112-1-jiayuan.chen@linux.dev/T/#t), and [AF_XDP metadata series](https://lore.kernel.org/bpf/20260819160535.1472459-1-sdf@fomichev.me/T/#t). These are review-stage upstream changes, not a claim about every released kernel. A useful test matrix includes same-type replacement with incompatible attach flavors, a valid fault at `memory.max`, 32/64-bit metadata layout, the metadata-enabled flag, and consecutive batched packets with different metadata state.

### Reliable telemetry streams must bind ordering to producer identity

A project-maintenance feed surfaced review questions about gap markers, replay, resume tokens, capability advertisement, and frame ceilings. They share one protocol rule: control information must be ordered and replayed in the same domain as the data it describes, and a sequence number is meaningful only within an identified producer incarnation. [RFC 1982](https://www.rfc-editor.org/rfc/rfc1982.html) supplies bounded serial-number arithmetic, but an application protocol must still define restart identity, replay retention, and negotiation limits.

The practical checks are to inject a queue gap while older messages remain buffered, resume after a producer restart, start with one capture path unavailable, and send a frame exactly at the negotiated ceiling. Review findings are not released-product behavior; the unresolved work is to turn those invariants into protocol tests before advertising continuity guarantees.

The remaining scheduler, project-support, networking, and public-forum targets had no substantive new technical exchange inside the daily window, or contained only routine introductions and automated notices. They were all accessible and were counted as quiet, not skipped.
