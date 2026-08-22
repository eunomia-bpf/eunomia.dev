# Why can an unused OBI Prometheus endpoint keep growing memory?

**Short answer:** because “not scraped” is not the same as “disabled.” OpenTelemetry eBPF Instrumentation (OBI) can continue discovering services and updating labeled Prometheus metric children while its HTTP endpoint receives no requests. The current upstream Helm chart enables `prometheus_export` on port `9090` in its default OBI configuration, but does not create the metrics Service or ServiceMonitor by default. An open chart fix reports that OBI expires those metric children while serving a scrape. With no scrape to drive that path, stale series can remain in the process and memory can grow as label sets turn over.

If a deployment exports metrics only through OTLP, disable the built-in Prometheus exporter rather than leaving an unreachable endpoint running. If Prometheus is supposed to scrape OBI, make the entire path explicit and test it: exporter, container port, discovery or Service, ServiceMonitor or scrape configuration, and target health. Lowering the Prometheus TTL is not a reliable fix for an endpoint that is never collected.

## Why the TTL does not necessarily bound memory

OBI's Prometheus exporter is enabled when `prometheus_export.port` is nonzero. Its documented `ttl` says when an inactive metric instance should stop being reported; the default is five minutes. That sounds like a wall-clock retention bound, but the important implementation detail in the current chart discussion is **where expiration runs**: the reported cleanup is performed as part of serving a scrape.

That creates this failure mode:

```text
instrumented traffic keeps arriving
        -> new or changing label sets update metric children
Prometheus endpoint is configured
        -> the in-process Prometheus exporter retains those children
no scraper reaches the endpoint
        -> collection-time expiration is not driven
        -> stale children accumulate and resident memory can rise
```

The growing object is exporter state in user space, not an eBPF map merely because OBI uses eBPF for collection. A long-lived DaemonSet makes the effect easier to see: application instances, routes, status codes, peer identities, or other dimensions may change while the exporter process stays alive.

TTL still matters when collection occurs. It determines which inactive metric instances the exporter omits or removes during its normal collection path. It should not be treated as a background garbage-collection guarantee unless the implementation explicitly schedules expiration independently of scrapes.

## How the Helm defaults create an unreachable exporter

At the time of writing, the chart's default values contain all three of these settings:

```yaml
service:
  enabled: false

config:
  data:
    prometheus_export:
      port: 9090
      path: /metrics

serviceMonitor:
  enabled: false
```

The rendered OBI configuration therefore starts the Prometheus exporter even though the chart does not create its own discovery path. That may be intentional when another system scrapes pods directly, adds annotations, or supplies a separate Service. It is wasted state when the deployment uses OTLP only and nothing ever scrapes port `9090`.

An open Helm-chart pull request proposes deleting `prometheus_export` from the rendered configuration unless the chart-managed Service is enabled. Its tests cover the default, Service-enabled, and ServiceMonitor-only cases. The pull request is still open, so the behavior must not be described as released. It also exposes a compatibility decision: `service.enabled: false` does not prove that no scraper exists, because direct pod discovery and separately managed Services are valid topologies.

The safe invariant is stronger than “Service off means exporter off”:

```text
Prometheus exporter enabled <=> an intentional, tested scrape consumer exists
```

The chart should make that invariant explicit, either with a dedicated exporter switch or with clear precedence rules for user-supplied `prometheus_export` configuration.

## Choose one of three deliberate deployment modes

### 1. OTLP-only metrics

Keep `otel_metrics_export` configured and turn off the built-in scrape endpoint. With the current OBI configuration model, port `0` or an unset port means no Prometheus endpoint is opened:

```yaml
config:
  data:
    otel_metrics_export:
      endpoint: "http://${HOST_IP}:4318"
    prometheus_export:
      port: 0

service:
  enabled: false

serviceMonitor:
  enabled: false
```

Render the exact chart version before rollout. Helm map merging, a values wrapper, or a future chart revision may otherwise reintroduce the default stanza. Assert that the rendered OBI configuration has a zero or absent Prometheus port and that the DaemonSet does not declare the application-metrics port.

### 2. Chart-managed Prometheus scraping

Enable a consistent path from Prometheus to the exporter:

```yaml
service:
  enabled: true

serviceMonitor:
  enabled: true

config:
  data:
    prometheus_export:
      port: 9090
      path: /metrics
```

Creating objects is not enough. Confirm that the Service selector matches the OBI pods, the Service port resolves to the exporter container port, the ServiceMonitor selector is selected by the Prometheus instance, and the target is healthy. A ServiceMonitor that exists but is ignored by Prometheus leaves the same no-scrape condition.

### 3. Direct pod discovery or a separately managed Service

Keep the exporter explicitly enabled, but do not assume the chart's Service is the only valid consumer. Record the external ownership in values and tests:

```yaml
service:
  enabled: false

serviceMonitor:
  enabled: false

config:
  data:
    prometheus_export:
      port: 9090
      path: /metrics
```

Then verify the separate scrape configuration against the rendered pod. This mode is the reason a chart must not silently remove an explicitly configured exporter solely because its own Service is disabled. Test upgrades against the pending chart behavior before adopting a release that changes the rendering rule.

## Do not confuse application metrics with OBI internal metrics

`prometheus_export` exposes application, network, and other observed metrics selected by OBI's metrics features. `internal_metrics` reports OBI's own behavior and can use Prometheus or OTLP independently. They may share an HTTP server when configured on the same port, but they have distinct configuration and metric families.

Disabling an unused application Prometheus exporter does not require losing OTLP application metrics or OBI self-observability. Choose each signal path independently:

- application metrics through OTLP, Prometheus, or both;
- internal OBI metrics disabled, sent through OTLP, or exposed for Prometheus;
- one reachable consumer for every enabled pull endpoint.

This distinction also improves diagnosis. A flat application-series count with rising RSS suggests looking beyond exporter cardinality, while rising application-series churn plus no scrape requests strongly supports the unreachable-exporter mechanism.

## A bounded production diagnosis

Use a canary and compare configured intent with runtime evidence.

1. **Render the release.** Inspect the generated ConfigMap, DaemonSet ports, Service, and ServiceMonitor. Do not infer the runtime configuration from a values fragment alone.
2. **Prove whether scraping occurs.** Check Prometheus target health and scrape counters. If internal metrics are enabled, `obi_prometheus_http_requests_total` can show requests reaching an OBI scrape endpoint.
3. **Separate process and kernel memory.** Track container RSS or working set alongside eBPF map memory and map entry counts. Exporter retention should appear primarily in the OBI process.
4. **Track churn, not only traffic.** Count active services and the label combinations that can create series. Stable request volume can still create unbounded historical label turnover.
5. **Change one variable.** On one canary, disable only `prometheus_export` while retaining the OTLP path and workload. A stopped memory slope is stronger evidence than a restart, which clears all process state.
6. **Test the opposite canary.** If Prometheus output is required, keep the exporter on and make a real scraper collect it at a normal interval. Confirm both series expiration and target continuity.
7. **Set a memory limit as containment.** A limit protects the node but is not the fix; an OOM restart can hide a persistent configuration mismatch.

Also compare after a full TTL plus several scrape intervals. An immediate flat line does not prove stale children were evicted, and an immediate drop may only reflect a restarted process.

## What should be fixed upstream?

The durable solution needs two properties:

1. **Lifecycle cleanup must not depend accidentally on consumer activity.** Exporter state should have a bounded expiration mechanism even when a scrape is delayed or absent.
2. **Packaging must not enable unused components implicitly.** The chart should render the Prometheus exporter only when the operator intentionally selects a scrape mode, while preserving explicit direct-scrape configurations.

Until both are true, operators should explicitly disable unused exporters and add render tests for the selected mode. The current pull request is useful evidence and a proposed packaging mitigation, but it is not yet a release guarantee and does not by itself replace runtime expiration.

## References

- [OBI data-export configuration: Prometheus endpoint, port, TTL, and instrumentations](https://opentelemetry.io/docs/zero-code/obi/configure/export-data/#prometheus-exporter-component)
- [Current OpenTelemetry eBPF Instrumentation Helm values](https://github.com/open-telemetry/opentelemetry-helm-charts/blob/main/charts/opentelemetry-ebpf-instrumentation/values.yaml)
- [Open Helm-chart pull request to omit an unscraped `prometheus_export`](https://github.com/open-telemetry/opentelemetry-helm-charts/pull/2360)
- [OBI internal-metrics reporter configuration](https://opentelemetry.io/docs/zero-code/obi/configure/internal-metrics-reporter/)
- [OBI exported metrics, including Prometheus scrape-request telemetry](https://opentelemetry.io/docs/zero-code/obi/metrics/)
- [Open OBI design issue on aligning its OTLP and Prometheus export paths](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/issues/2974)
- [Open GenAI semantic-conventions issue for operation cost](https://github.com/open-telemetry/semantic-conventions-genai/issues/287)
- [Open GenAI semantic-conventions pull request for `gen_ai.usage.cost.*`](https://github.com/open-telemetry/semantic-conventions-genai/pull/443)
- [BPF thread: add KASAN checks to JITed programs](https://lore.kernel.org/bpf/20260822-kasan-v7-0-99afee6ef7fd@bootlin.com/T/#t)
- [BPF thread: dedicated keyring and ML-DSA support for signed loaders](https://lore.kernel.org/bpf/20260821214111.1120748-1-daniel@iogearbox.net/T/#t)
- [BPF thread: correct verifier non-null inference for conditional jumps](https://lore.kernel.org/bpf/20260821-bug-029-bad-non-null-inference-v1-1-45ddc0f7c308@gmail.com/T/#t)

## Community discussion today

Today's ordinary visible-browser review covered all 6 approved communities and all 15 allowlisted channels or public pages. Every target was accessible. The selected question came from the 24-hour window, so the seven-day fallback was not used. Names, accounts, employers, workspace and channel identities, closed-chat links, exact times, private topology, raw logs, and searchable chat wording have been removed. No raw transcript was retained.

### Pull exporters need an explicit consumer and an independent lifecycle

The strongest new operational question concerned a small Helm change intended to reduce memory use in eBPF instrumentation pods whose built-in Prometheus endpoint is not scraped. Public configuration confirms the mismatch: the chart enables the exporter by default while its Service and ServiceMonitor default to off. The [open chart change](https://github.com/open-telemetry/opentelemetry-helm-charts/pull/2360) removes the exporter when the chart-managed Service is absent and documents collection-time series expiration.

The immediate action is to classify each deployment as OTLP-only, chart-managed scrape, or externally managed scrape, then render and test that exact mode. The unresolved design issue is whether a packaging switch alone is sufficient: direct pod scraping must remain possible, and exporter cleanup should not rely on a consumer calling the endpoint.

### GenAI cost semantics are now concrete but still proposed

Another observability thread moved from “should cost exist?” to the harder question of what the value means. An [open tracking issue](https://github.com/open-telemetry/semantic-conventions-genai/issues/287) and an [active pull request](https://github.com/open-telemetry/semantic-conventions-genai/pull/443) propose `gen_ai.usage.cost.*` fields, but the convention is not merged or stable. The discussion distinguished provider-billed values from locally estimated values and raised currency and pricing-source provenance as necessary parts of any interoperable schema.

For production telemetry, keep experimental fields versioned and namespaced, record whether a value is billed or estimated, and retain the currency and pricing revision used for calculations. Do not make alerts or cross-provider dashboards depend on the proposed names until the convention is released.

### Kernel work is pushing observability into generated-code and trust boundaries

Public BPF review concentrated on failure domains that normal verifier success does not cover. A [KASAN-for-JIT series](https://lore.kernel.org/bpf/20260822-kasan-v7-0-99afee6ef7fd@bootlin.com/T/#t) proposes adding memory-access checks while x86 BPF instructions are translated to native code. The current revision deliberately covers generic KASAN on x86 and excludes BPF-stack and potentially faulting access classes, so it is a review-stage diagnostic capability with explicit limits, not universal JIT memory safety.

A separate [signed-loader series](https://lore.kernel.org/bpf/20260821214111.1120748-1-daniel@iogearbox.net/T/#t) proposes a dedicated BPF-only keyring, sealed empty by default, plus ML-DSA tooling and end-to-end tests. The useful security boundary is scope: a loader-controlled keyring proves little by itself, while a boot-provisioned and restricted BPF keyring can represent operator policy without extending that key to the kernel's broader trust hierarchy. Both series remain upstream proposals.

The project-support, scheduler-support, networking, and public-forum targets were quiet in the daily window or contained only routine automated notices and non-technical coordination. They were accessible and counted as checked, not skipped.
