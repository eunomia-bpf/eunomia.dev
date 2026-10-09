# Why does the OpenTelemetry GenAI metric catalog stop at engine-agnostic latency and token counts, and how does model-serving-signals close the model-serving and autoscaling gap?

The OpenTelemetry GenAI conventions define request-level signals only: operation and request duration, time-to-first-token, time-per-output-token, and token usage. They are deliberately engine-agnostic, so they stop short of every serving-state signal an autoscaler needs — queue depth, request concurrency, batch utilization, and KV-cache pressure. Those live in the inference engine's own metric namespace. The Kubernetes `model-serving-signals` effort (under `kubernetes-sigs`, backed by SIG Autoscaling and SIG Instrumentation) is the work that standardizes that engine-specific layer: it defines an engine-agnostic signal contract, per-engine profiles for vLLM, SGLang, and TensorRT-LLM, and a mapping exporter that translates engine metrics into OpenTelemetry so autoscaling can key on one consistent set of names.

## The mechanism

The GenAI conventions split the catalog by layer. The core metrics document defines a small, engine-agnostic set: one client operation-duration histogram, three model-server histograms (`gen_ai.server.request.duration`, `gen_ai.server.time_per_output_token`, `gen_ai.server.time_to_first_token`), plus workflow, agent, and tool duration/call metrics. A separate client-inference document adds the token accounting — `gen_ai.client.inference.usage.*` counters (input, output, cache-read, cache-write, reasoning) broken down by modality, and per-operation token histograms. Every one of these is a request-level quantity, keyed by `gen_ai.operation.name`, `gen_ai.provider.name`, and `gen_ai.request.model`. That keying is precisely the point: the conventions describe one model request or one agent operation, not the serving system behind it.

The serving system's own state — how many requests are queued, how many are in flight, how full the batch is, how much of the KV cache is live — is not part of that catalog. Each engine exposes it under its own names (vLLM, SGLang, and TensorRT-LLM each publish a different set of gauges). The conventions explicitly hand system-specific signals back to the engine, so an autoscaler that only reads OpenTelemetry GenAI names sees latency and tokens but is blind to the queue, the batch, and the cache.

`model-serving-signals` closes that gap at the engine boundary. It defines an engine-agnostic set of model-server signals, a per-engine profile that maps each engine's native metrics onto that set, a mapping exporter that ships them as OpenTelemetry, and a conformance suite that pins the mapping so an engine update does not silently rename or reshape a metric. Autoscaling then keys on the common names rather than on whatever each engine happens to call its queue-depth gauge.

## Verification and debugging path

1. Open the OpenTelemetry GenAI metrics document and look at the model-server section. It lists exactly `gen_ai.server.request.duration`, `gen_ai.server.time_per_output_token`, and `gen_ai.server.time_to_first_token`. No queue depth, no concurrency, no batch or cache metric is defined there.
2. Open the client-inference and token-metrics documents. They add the duration and token-usage instruments but still no serving-state signal — token usage is an accounting quantity, not a capacity signal.
3. Open the `model-serving-signals` repository. Confirm it ships engine profiles (vLLM, SGLang, TensorRT-LLM) and a mapping exporter plus a conformance suite; that is the layer that turns engine-specific serving metrics into the OpenTelemetry names an autoscaler consumes.
4. In a live deployment, scrape the engine's native exporter and the mapped OpenTelemetry series side by side. The signals you need for autoscaling decisions — queue depth, batch utilization, KV-cache occupancy, and SLO attainment — should appear under the standardized signal names, with the mapping exporter translating engine names into that vocabulary.

## The limitation

- The GenAI metric conventions are at Development status; metric and attribute names can move before they stabilize.
- The OpenTelemetry catalog is deliberately request-level. Serving-state signals are engine-specific by design; only the standardized subset that model-serving-signals has mapped is portable. Until a signal is proposed into the core conventions, it stays engine-local.
- Token usage is not a proxy for SLO: a low-token workload can still be queue-bound, and a high-token one can be fine. Autoscaling on tokens alone misses queue depth and SLO attainment, which is why the serving layer, not the conventions, must supply the capacity signal.

## References

- [OpenTelemetry semantic-conventions-genai — GenAI metrics](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-metrics.md) — the engine-agnostic model-server metrics: `gen_ai.server.request.duration`, `gen_ai.server.time_per_output_token`, `gen_ai.server.time_to_first_token`, and the client operation-duration histogram.
- [OpenTelemetry semantic-conventions-genai — client inference metrics](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/client-inference.md) — inference duration, time-to-first-chunk, time-per-output-chunk, and the per-operation token-usage histograms.
- [OpenTelemetry semantic-conventions-genai — token metrics](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-token-metrics.md) — the `gen_ai.client.inference.usage.*` counters broken down by modality (input, output, cache-read, cache-write, reasoning).
- [kubernetes-sigs model-serving-signals](https://github.com/kubernetes-sigs/model-serving-signals) — the engine-agnostic signal contract for model servers on Kubernetes: engine profiles for vLLM, SGLang, and TensorRT-LLM, a mapping exporter, a conformance suite, and autoscaling integration.

## Community discussion today

Two opt-in archive channels (the two CNCF OpenTelemetry instrumentation channels) produced three messages this run. One asks whether a Node.js GenAI instrumentation repository is planned to match the existing Python one; no public answer is available yet, and it does not touch the serving-metric boundary. A second points at a specific pull-request review discussion in an agent-tooling project and is not a self-contained question. The third, which is the question this page answers, describes the new Kubernetes `model-serving-signals` initiative: taking metrics from inference engines (vLLM, SGLang, TensorRT-LLM) and translating them into OpenTelemetry, and asking whether a larger catalog of metrics is defined or planned, especially for model serving and autoscaling. The answer above states the boundary: the core GenAI conventions stop at engine-agnostic latency and token accounting, and the serving-state layer is what model-serving-signals is standardizing.

Channel coverage for this run: the two opt-in archives (the eBPF and GenAI instrumentation channels) supplied three messages. The visible-browser-only sources — the eunomia-bpf and sched-ext Discord servers, the bpf mailing list, and r/eBPF — could not be reviewed this run because no visible-browser session was available, so they are marked not-covered, not quiet.
