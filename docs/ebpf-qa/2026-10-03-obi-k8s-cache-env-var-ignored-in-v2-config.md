# Why does the OBI Kubernetes cache address environment variable get ignored when the Helm chart renders a Config v2 document, and where must the cache address be written instead?

The variable is not broken — it is a Config v1 runtime override, and Config v2 no longer applies `OTEL_EBPF_*` variables as an overlay on the document. From OBI v0.11.0, both the standalone binary and the OBI Collector receiver load a single Config v2 document, and that loader does not read a legacy `OTEL_EBPF_KUBE_META_CACHE_ADDRESS` (or any other `OTEL_EBPF_*` override) as configuration. The v2 document has its own canonical field — the `metadata_cache.address` field under the Kubernetes enricher block — and only that field wires an OBI pod to the shared `k8s-cache`. The Helm chart kept exporting the v1-style environment variable on the OBI daemonset when `k8sCache.replicas` was set, but the Config v2 document it rendered had no field pointing at the cache service, so the variable was inert: every OBI pod fell back to its own in-process informer cache, the cache service received zero subscriptions, and each pod kept opening its own `LIST`/`WATCH` streams against the Kubernetes API server. The fix is to write the address into the Config v2 document, not to keep exporting the variable.

## The mechanism

OBI centralizes Kubernetes metadata behind an optional `k8s-cache` service. Instead of every OBI pod opening its own informer `LIST`/`WATCH` streams against the API server, each pod opens one gRPC stream to the cache, which runs the `Pod`/`Node`/`Service` informers once and then replays and streams metadata events (`informer.EventStreamService/Subscribe`, with a `SYNC_FINISHED` event marking the end of the initial replay). The switch between "local in-process informers" and "subscribe to the remote cache" is a single config field:

- Config v1: `meta_cache_address` under the `attributes.kubernetes` block, overridable at runtime via the `OTEL_EBPF_KUBE_META_CACHE_ADDRESS` environment variable.
- Config v2: that field moved and was renamed to `metadata_cache.address` under the Kubernetes enricher block. The v1 path is not read; only the v2 path is.

Config v2 is a single-document configuration model. The migration guide is explicit that runtime environment overrides such as `OTEL_EBPF_*` are not read as additional v1 configuration in a v2 deployment: "Merely carrying a legacy variable into the v2 deployment does not preserve its override." A variable has any effect in v2 only when the document references it, and even that is loader-dependent — the upstream `otelconf/x.ParseYAML` path expands `${VAR}`, `${env:VAR}`, `${VAR:-fallback}`, and `${env:VAR:-fallback}` before decoding, while OBI's internal standalone parser decodes the document bytes directly, so a substitution token only survives the loader that performs it.

That is exactly the chart gap. The daemonset template emits:

```yaml
{{- if .Values.k8sCache.replicas }}
- name: OTEL_EBPF_KUBE_META_CACHE_ADDRESS
  value: {{ .Values.k8sCache.service.name }}:{{ .Values.k8sCache.service.port }}
{{- end }}
```

… but the Config v2 document the chart renders into its configmap has no `metadata_cache.address` field referencing the cache service. The environment variable is a v1 overlay, and the v2 document ignores it. The practical result is that `k8sCache.replicas: 1` deploys the cache service and the daemonset env var, yet no OBI pod actually connects. The k8s-cache design note makes the fallback explicit: "If k8s cache address is not provided, OBI will initiate its own local in-process cache" — the per-pod `LIST`/`WATCH` scaling pattern the cache service exists to avoid. The same gap applies when a user supplies their own v2 document (for example via `config.data`): the cache address is never wired in unless it is written into that document.

## Verification and debugging path

Confirm whether the address is actually in the document OBI loads, and whether the cache sees any subscribers:

1. Render or read the Config v2 document the daemonset loads (the chart configmap, or `helm template` with `k8sCache.replicas: 1`). Before the chart fix, there is no `metadata_cache:` block under `enrich.enrichers.kubernetes`; grep the configmap body for the cache service address — it is present only in the daemonset `env`, not in the document.
2. Confirm the pod is running a Config v2 document (schema `version: "2.0"`). If it is still a v1 file, the environment variable does apply, and a cache receiving no connections points at a missing or unreachable cache service rather than at an ignored variable.
3. Check the cache side. The `k8s-cache` exposes internal metrics on a Prometheus `/metrics` endpoint and logs reconnect/subscription behavior; a running cache with zero OBI subscriptions is the signature of this gap. OBI logs whether it is using the local cache or subscribing to the remote one.
4. Cross-check with `obi config migrate`: migrate the v1 file and read the report. A v1 environment overlay that was not materialized into a v2 field is not preserved — "A variable that is not referenced by the v2 document has no migration effect unless the target release explicitly documents it as a separate runtime input."

## The limitation

The environment-variable form is a v1 overlay and is intentionally not part of the v2 document model. There are two correct fixes, and the choice is deployment-driven:

- Upgrade the chart. The chart fix (open-telemetry/opentelemetry-helm-charts pull request 2440, "fix(obi): set the k8s cache address in the Config v2 document") is merged in chart 0.14.2: when `k8sCache.replicas` is set, the rendered v2 configmap now includes the `metadata_cache.address` field under the Kubernetes enricher block, pointing at the cache service. On a chart carrying that change the address is wired automatically.
- Until then, write the address into the Config v2 document yourself — through the chart `config.data` (the `metadata_cache.address` field under the Kubernetes enricher block) — or keep the deployment on the frozen-but-supported v1 config, where the variable still applies.

Do not rely on exporting `OTEL_EBPF_KUBE_META_CACHE_ADDRESS` alone in a v2 deployment. The variable is inert unless the document references it, and the substitution that would make a referenced token do so is loader-dependent. The supported, reproducible form is the `metadata_cache.address` field in the v2 document — hardcoded by the fixed chart or set explicitly in values.

## References

- [OBI — `devdocs/k8s-cache.md`](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/blob/main/devdocs/k8s-cache.md) — k8s-cache design; the `meta_cache_address` / env-var switch; "no address → local in-process cache"; internal metrics.
- [OBI — Config v2 migration guide](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/blob/main/devdocs/config/version-2.0/migration.md) — "OTEL_EBPF_* runtime overrides are not read as additional v1 configuration"; "Rewire v1 environment overrides"; `obi config migrate` behavior.
- [OBI — Config v2 guide](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/blob/main/devdocs/config/version-2.0/config-v2.md) — the v1 field `meta_cache_address` moves and renames to `metadata_cache.address` under the Kubernetes enricher block; environment-substitution loader caveat.
- [OBI Helm chart — daemonset template](https://github.com/open-telemetry/opentelemetry-helm-charts/blob/main/charts/opentelemetry-ebpf-instrumentation/templates/daemonset.yaml) — `OTEL_EBPF_KUBE_META_CACHE_ADDRESS` injection when `k8sCache.replicas` is set.
- [open-telemetry/opentelemetry-helm-charts pull request 2440](https://github.com/open-telemetry/opentelemetry-helm-charts/pull/2440) — "fix(obi): set the k8s cache address in the Config v2 document" (merged 2026-10-02, chart 0.14.2), wires `metadata_cache.address` into the rendered v2 configmap.

## Community discussion today

The selected question came from an opt-in archive: a practitioner noticed that, in the official OBI Helm chart, `OTEL_EBPF_KUBE_META_CACHE_ADDRESS` is automatically injected into the OBI daemonset pods when the k8s cache is enabled, yet a Config v2 document does not accept environment variables as configuration, and it was not clear whether that variable is supported in v2. A follow-up in the same thread confirmed the concern against the real chart: a values file that sets only `k8sCache.replicas: 1` gets a v2 config by default that omits the cache address, so OBI ignores the exported variable and the cache receives no connection; the same is true when a user supplies their own v2 document via the config data. The thread's interim workaround was to wire the address into the v2 document by hand (the `metadata_cache.address` field under the Kubernetes enricher block, pointing at the cache service and its gRPC port), with a note that the chart should set it for the v2 document the way it still sets the v1 variable. This answer confirms that mechanism against the public OBI config docs and the chart: the v1 env var is not a v2 overlay, the field was moved and renamed in v2, and the chart fix that wires the address into the v2 document (pull request 2440) is the durable resolution.

Other threads this day: a continued Cilium LoadBalancer shared-VIP frontend-ownership thread (a re-post of the question published two days ago, now with a maintainer pointing to a tracked upstream issue and a fix-in-review), a GnuTLS HTTP/2 per-connection HPACK decoder thread (published as a question a day earlier), a Hubble flow-to-rule misattribution thread answered as an already-tracked upstream issue with the fix in review (the datapath is correct; only the Go-side copy is off), a high-frequency socket-layer drop-latency versus user-space context-switch benchmark request that was too thin to publish, and an OpenTelemetry GenAI semantic-conventions thread about capturing which skills were available to an agent at invocation start (a semantic-conventions pull request, outside this page's scope).

Channel coverage for this run: the two opt-in archives provided eleven messages, all covered above. The visible-browser-only sources (Discord, the eunomia-bpf and sched-ext communities, the bpf mailing list, and r/eBPF) could not be reviewed in this run — no visible-browser session was available — so they are marked uncovered, not quiet.
