# Why does a local embeddings model with no server and no token count declare expected gaps in its GenAI span attributes, and how do the gaps stay spec-conformant?

The GenAI semantic conventions scope every span attribute to the availability of its value, not to the existence of the span. Required attributes are few (operation name, provider name); everything else is conditionally required when a value exists or recommended as best effort. A local model exposes no server and reports no billed token usage, so the spec-correct answer is to omit those attributes and declare the omission as an expected gap in the instrumentation conformance scenario. Declaring the gap is what keeps the telemetry spec-conformant.

The same boundary is what lets a new instrumentation wrap a local library in a few lines instead of rebuilding its telemetry plumbing.

## The mechanism

1. **Requirement levels apply to value availability, not to span existence.** The embeddings span (`gen_ai.operation.name` of `embeddings`, span kind CLIENT) lists a small set of required attributes: `gen_ai.operation.name` and `gen_ai.provider.name`. The rest carry lighter levels: `gen_ai.request.model` is conditionally required when the value is available, and `gen_ai.usage.input_tokens`, `server.address`, and `gen_ai.response.model` are recommended. Recommended signals SHOULD be emitted when readily available; Opt-In signals MUST NOT be emitted without explicit user configuration. The levels are about whether a value can honestly be produced, not about forcing a span to carry every key in the catalog.

2. **A local embeddings model changes the source of every value.** With a hosted endpoint, the provider returns a response object containing a model name, a server address, and billed token counts. With an in-process model, the request and the model share one address space: there is no network hop, no server address, no response model identifier, and no billed-usage object, because no provider meters the call. The only trustworthy values are the ones the caller controls: the model name passed to the constructor, and the embedding dimension, which the model itself exposes through its API.

3. **The span shape follows from that boundary.** An embeddings instrumentation that wraps the encode call emits one CLIENT span named `embeddings {gen_ai.request.model}` plus the `gen_ai.client.operation.duration` histogram, with `gen_ai.request.model` and `gen_ai.embeddings.dimension.count` set, `gen_ai.provider.name` set to the serving ecosystem, and the attributes that have no source value omitted and declared as expected gaps in the conformance scenario. The spec treats the conformance scenario as the place where an instrumentation says, in advance, which of the recommended attributes its provider family cannot produce.

4. **Where the instrumentation wraps is decided by the library call graph.** `encode_query` and `encode_document` in sentence-transformers are thin specializations that delegate to the shared `encode` method, so a single wrap of `encode` covers all three entry points without double-counting. The dimension attribute is read from the model instance, whose embedding-dimension accessor returns an integer or a null when the value is not known at load time.

## How to verify and debug

1. **Read the level table of the embeddings span.** The span definition in the GenAI span specification carries the level of every attribute inline: required, conditionally required when the value is available, or recommended. Read that table before writing the span builder, and decide per attribute whether the value has a real source in your provider family.
2. **Prove that the omitted attributes have no source value.** For an in-process model, confirm in the library API that there is no usage object, no response model field, and no network peer to observe. The conformance scenario then records the omission as an expected gap rather than leaving it to a conformance test to discover later.
3. **Build a local model and inspect the emitted telemetry.** A small model with in-memory weights runs the real encode path without a download. Check that the span name is `embeddings {model}`, that the required attributes are present, that the duration histogram carries the recommended explicit bucket boundaries, and that the declared gaps are the only missing keys.
4. **Check the histogram buckets.** The operation-duration metric SHOULD use the spec-recommended explicit buckets (0.01 through 81.92 seconds), because embedding latency distribution spans three orders of magnitude.

## The limitation

- `gen_ai.usage.input_tokens` is a recommended attribute of the embeddings span, but only a provider that exposes its own token accounting can populate it; an in-process model has no meter, so the gap is structural and permanent, not a TODO.
- The embedding dimension accessor returns a null when the value is not known at load time; the attribute is then omitted, and dimension-based alerts must tolerate its absence.
- The conformance scenario that records expected gaps is only as good as its update discipline: when the upstream library gains a usage report or a response identifier, the gap list must be re-derived, or the instrumentation silently drifts from the catalog.
- Requirement levels are policy, not enforcement: the spec does not make a conformance check fail on a missing recommended attribute; the enforcement point is the declared expected-gap list.
- This answer describes the Development-status embeddings span of the current GenAI semantic conventions; the levels can tighten as the catalog stabilizes.

## References

- [OpenTelemetry GenAI semantic conventions: spans (embeddings)](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-spans.md) - embeddings span definition: `gen_ai.operation.name` SHOULD be `embeddings`, span kind CLIENT, the per-attribute requirement levels, and the conditionally-required wording of `gen_ai.request.model`.
- [OpenTelemetry GenAI semantic conventions: metrics](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-metrics.md) - `gen_ai.client.operation.duration` histogram definition and the recommended explicit bucket boundaries.
- [OpenTelemetry GenAI attribute registry](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/registry/attributes/gen-ai.md) - `gen_ai.usage.input_tokens` best-effort wording: a total provided by the provider when available, or summed from parsed provider output.
- [OpenTelemetry GenAI: when to add an instrumentation here](https://github.com/open-telemetry/opentelemetry-python-genai/blob/main/CONTRIBUTING.md) - boundary between native/first-party telemetry and external instrumentation; telemetry is best owned closest to the library it describes.
- [sentence-transformers SentenceTransformer API](https://github.com/UKPLab/sentence-transformers/blob/master/SentenceTransformer.py) - `encode`, `encode_query`, and `encode_document` delegation, and the embedding-dimension accessor that returns an integer or a null.

## Community discussion today

The monitored window for this run is a 7-day archive window across the two opt-in archive sources, returning one message: a new contributor requesting review of a proposed sentence-transformers instrumentation in the Python GenAI instrumentation repository (public issue #910 and PR #911 in the opentelemetry-python-genai project). The visible-browser-only sources - the eunomia-bpf and sched-ext Discord servers, the bpf mailing list, and r/eBPF - could not be reviewed this run because no visible-browser session was available, so they are marked not-covered, not quiet.

**A new instrumentation for local embeddings.** The request is a new instrumentation package for a local embeddings library built on the shared GenAI utility layer: one embeddings span and one operation-duration histogram per encode call, with the model name and embedding dimension recorded. Its two open questions are the ones this page answers. First, an in-process model exposes no server address, no response model identifier, and no token usage, so the instrumentation declares those as expected gaps in its conformance scenario instead of fabricating values. Second, the package pulls a large inference dependency into the workspace lock file, so the maintainers must decide whether that weight is acceptable. The review is open; the boundary between a required attribute, a conditionally required attribute, and a recommended one is what keeps the proposed telemetry honest.

**Where the boundary lives for practitioners.** The practical takeaway for anyone instrumenting local models: the span is still useful without usage data, because duration, model name, and dimension are the attributes that drive retrieval-latency debugging. The token-usage gap is structural, not a bug, and dashboards that assume token counts exist on every GenAI span will overstate retrieval cost on embeddings paths that have none.
