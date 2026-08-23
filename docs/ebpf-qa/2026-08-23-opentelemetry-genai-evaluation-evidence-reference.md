# Should an OpenTelemetry GenAI evaluation result carry a verifiable-evidence reference?

**Short answer:** a digest-bound evidence reference is a useful, in-scope extension for `gen_ai.evaluation.result`, but it should be optional and it must not imply that OpenTelemetry verified the evidence. The current GenAI convention records the evaluation name, score or label, explanation, error class, and response identifier. It does **not** define an evidence URI, digest, media type, signature, or verification result.

A good experimental design keeps the evidence outside telemetry and records only a stable reference to the exact bytes: a non-secret URI, an algorithm-qualified digest, and a media type. That lets a consumer retrieve and integrity-check an evaluation receipt without putting a large or sensitive artifact in every event. Authenticity, signer identity, policy, and trust remain the responsibility of the referenced evidence format and its verifier.

## The current event answers “what was the result?”, not “where is the proof?”

The current `gen_ai.evaluation.result` event is a recommended, development-status convention. It requires `gen_ai.evaluation.name`; conditionally records `gen_ai.evaluation.score.value`, `gen_ai.evaluation.score.label`, and `error.type`; recommends `gen_ai.evaluation.explanation`; and uses `gen_ai.response.id` when the evaluated operation span is unavailable.

Those fields are enough to render a result such as “relevance: pass” and correlate it with a model response. They are not enough to reproduce the evaluator, inspect a signed receipt, or prove which bytes were reviewed. A free-form explanation is human-readable context, not an integrity anchor.

That distinction should remain explicit:

```text
evaluation event     -> result and correlation
evidence reference   -> identity of exact external bytes
evidence envelope    -> claims, signer, signature, and verification rules
policy decision      -> whether a verifier accepts those claims
```

Adding a reference is therefore additive. It does not change the meaning of the score and should not make evidence mandatory for lightweight or online evaluations.

## What the reference needs to bind

The smallest interoperable shape has three logical fields:

| Field | Contract |
| --- | --- |
| URI | Identifies the evidence object without embedding credentials. The specification must say whether this is an identity, a retrieval location, or both. |
| Digest | Binds the reference to exact bytes and includes the algorithm, for example `sha256:<hex>`. |
| Media type | Tells a consumer how to parse the retrieved bytes, without claiming that the content is valid or trusted. |

The in-toto `ResourceDescriptor` is a useful public precedent: it separates `uri`, `downloadLocation`, a map of algorithm-qualified digests, and `mediaType`. OpenTelemetry attributes are flatter, so an initial proposal could use one digest string whose value includes the algorithm. It should not create dynamic attribute keys such as `...digest.sha256`, and it should not use parallel arrays of URIs, digests, and media types whose indexes can drift.

For one result with several supporting files, point to a manifest or signed envelope that lists the files and their digests. That gives the event one atomic reference and lets the evidence format own cardinality and structure.

## Prototype without claiming a standard

Until the GenAI semantic convention accepts and releases names, an implementation should use its own vendor namespace. A prototype could express this logical shape:

```text
event.name                                   = "gen_ai.evaluation.result"
gen_ai.evaluation.name                      = "policy_compliance"
gen_ai.evaluation.score.label               = "pass"
example.evaluation.evidence.uri             = "urn:example:evidence:01J..."
example.evaluation.evidence.digest          = "sha256:7f83b165..."
example.evaluation.evidence.media_type      = "application/json"
```

The `example.*` keys above are placeholders, not OpenTelemetry attributes. A production prototype should document:

- exactly which byte representation is hashed, including decompression and canonicalization rules;
- the allowed digest algorithms and encoding;
- whether the URI is resolvable directly or through an internal resolver;
- whether one evaluation result can reference only one evidence object;
- the retention relationship between telemetry and evidence; and
- what a missing, expired, unauthorized, or digest-mismatched object means to the consumer.

OpenTelemetry's convention-authoring guidance recommends prototyping new attributes across implementations before standardization. This case especially needs evidence from at least an online evaluator and an asynchronous evaluation pipeline, because their correlation and retention paths differ.

## In-flight and post-hoc evaluation need different trace relationships

### In-flight evaluation

When the evaluator runs while the GenAI operation is active, emit `gen_ai.evaluation.result` with that operation's trace context. This follows the current recommendation to parent the evaluation event to the operation being evaluated. The external evidence may be written before or immediately after the event, but the event should not be exported until the system has the final digest of the stored bytes.

If evidence creation fails, record the evaluation result only if it is independently known. Do not emit a URI and fill in the digest later: telemetry is append-only in many pipelines, and a partially populated reference becomes ambiguous.

### Post-hoc evaluation

Do not reopen or append to an ended span. OpenTelemetry span operations cannot record new data after the span has ended. Instead:

1. create telemetry for the later evaluation operation;
2. link its span to the original `SpanContext` when that context was retained legitimately;
3. set `gen_ai.response.id` when the response identifier is available; and
4. emit the result and evidence reference from the later evaluation context.

OpenTelemetry links can point to a `SpanContext` in the same or a different trace, which is a better model for asynchronous work than fabricating a parent-child lifetime. The response identifier is a correlation fallback, not a replacement for evidence integrity.

## A digest is not an attestation

A digest answers one question: “Did I retrieve the same bytes that the producer referenced?” It does not answer:

- who produced those bytes;
- whether the signer was authorized;
- whether the evaluator executed the claimed procedure;
- whether the inputs were complete or untampered before hashing; or
- whether the result satisfies the consumer's policy.

If those guarantees matter, the referenced object should use an authenticated envelope such as DSSE/in-toto or another documented attestation format. The verifier must check the signature, signer identity, predicate type, subject, freshness, and local policy. The telemetry event should still report only that a reference exists unless verification itself is a separately modeled result.

This also prevents a dangerous naming mistake. An attribute named `verified=true` would collapse several different facts—digest match, signature validity, trusted signer, and policy acceptance—into one boolean with no portable meaning.

## Privacy and operational limits

Evidence references are high-cardinality and can be sensitive even when they contain no prompt text.

- Never put pre-signed URLs, bearer tokens, user names, prompt fragments, tenant names, or private object-store topology in the URI.
- Prefer an opaque identifier or content-addressed URI resolved inside the authorized environment.
- Treat the URI as opt-in telemetry and apply the same export, access, and retention policy as the evidence.
- Remember that hashing low-entropy or guessable content can create a confirmation oracle; a digest is not anonymization.
- Keep the evidence bytes out of span attributes and event bodies. Large content increases telemetry cost and may bypass the evidence store's access controls.
- Define behavior when the trace outlives the evidence or the evidence outlives the trace. A dangling reference must be distinguishable from a failed evaluation.

Media type is also only a parsing hint. Consumers must verify the digest over the agreed byte representation before parsing, then validate the payload's schema and, where applicable, its authenticated envelope.

## A bounded adoption plan

1. **Prototype in a vendor namespace.** Use one reference per evaluation result and keep standard `gen_ai.evaluation.*` fields unchanged.
2. **Exercise both timing modes.** Test an evaluator running inside the request and another running after the original span has ended.
3. **Verify byte identity.** Store evidence, compute the digest from the stored representation, retrieve it through the consumer path, and recompute the digest before parsing.
4. **Test failure states.** Cover not found, unauthorized, expired, wrong media type, unsupported digest algorithm, and digest mismatch.
5. **Measure telemetry impact.** Confirm that URI cardinality is not promoted into metrics or indexed indiscriminately.
6. **Separate integrity from trust.** Add signature and policy verification in the evidence verifier, not as an assumed property of the telemetry reference.
7. **Bring interoperable evidence upstream.** Document at least two independent producers and consumers, the exact privacy warning, and the one-versus-many artifact rule before proposing standard names.

The useful contract is modest: the evaluation result can tell a consumer where the evidence is and which bytes to expect. It should stop before claiming that those bytes are true.

## References

- [OpenTelemetry GenAI event conventions, including `gen_ai.evaluation.result`](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-events.md)
- [OpenTelemetry GenAI attribute registry](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/registry/attributes/gen-ai.md)
- [OpenTelemetry semantic conventions for events](https://opentelemetry.io/docs/specs/semconv/general/events/)
- [OpenTelemetry guidance for defining and prototyping semantic conventions](https://opentelemetry.io/docs/specs/semconv/how-to-write-conventions/)
- [OpenTelemetry Trace API: span lifetime, events, and links](https://opentelemetry.io/docs/specs/otel/trace/api/)
- [in-toto `ResourceDescriptor`: URI, digest, download location, and media type](https://github.com/in-toto/attestation/blob/main/spec/v1/resource_descriptor.md)
- [in-toto envelope specification and authenticated payload guidance](https://github.com/in-toto/attestation/blob/main/spec/v1/envelope.md)
- [Current OpenTelemetry GenAI agent and framework span conventions](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-agent-spans.md)
- [OCI Image Manifest Specification and artifact guidance](https://github.com/opencontainers/image-spec/blob/main/manifest.md)
- [Linux BPF stream implementation](https://github.com/torvalds/linux/blob/master/kernel/bpf/stream.c)
- [Linux BPF token UAPI](https://github.com/torvalds/linux/blob/master/include/uapi/linux/bpf.h)
- [Linux BPF selftests](https://github.com/torvalds/linux/tree/master/tools/testing/selftests/bpf)

## Community discussion today

Today's ordinary visible-browser review covered all 6 approved communities and all 15 allowlisted channels or public pages. Every target was accessible. The selected question came from the 24-hour window, so the seven-day fallback was not used. Names, accounts, employers, workspace and channel identities, message links, exact times, private topology, raw logs, and searchable wording have been removed. No raw transcript was retained.

### Evaluation telemetry needs an integrity boundary without becoming an evidence store

The strongest question asked whether a GenAI evaluation result should point to a signed or otherwise offline-checkable receipt. The mechanism and safe shape are the main answer above: the [current event convention](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-events.md) can report a score and correlate it to a response, but it has no evidence-reference fields. A URI, algorithm-qualified digest, and media type can bridge that gap if they remain opt-in and if the digest is not mislabeled as authenticity or policy approval.

The next useful engineering step is a two-producer prototype: one evaluator emits while the model operation is active, and one runs asynchronously. Both should produce the same logical reference, and a consumer should retrieve the object, verify the digest, validate the media type, and then hand any signature to the appropriate verifier. The unresolved boundary is standard naming and cardinality: a single manifest reference is easier to query and evolve than parallel arrays for multiple artifacts.

### Artifact packaging and trace rendering both fail when two schemas claim ownership

A project-maintenance feed exposed two related compatibility problems. One involved changing an OCI configuration descriptor so orchestration tooling could recognize an artifact, while existing consumers may still depend on the older media type. The [OCI manifest specification](https://github.com/opencontainers/image-spec/blob/main/manifest.md) treats `config.mediaType`, `artifactType`, digest, and content as a contract; silently rewriting one field can make the same bytes mean different things to different pullers. The safe diagnostic is to inspect the pushed manifest and test registry storage, pull, and runtime consumption separately. A migration should publish a new, versioned artifact shape or use the standard empty-config/artifact-type pattern, then preserve old consumers for a bounded window.

The other problem was an OTLP exporter that could reconstruct model-call spans but did not necessarily emit the surrounding agent or workflow spans expected by a visualization. A successful export therefore did not prove a complete trace hierarchy. Compare the emitted span names, kinds, parents, and `gen_ai.operation.name` values with the [current agent-span conventions](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-agent-spans.md), then distinguish “model calls are visible” from “agent lifecycle is modeled.” The unresolved work is to decide which layer can reliably observe the agent boundary; an exporter should not invent an `invoke_agent` root from HTTP traffic alone when it cannot prove that semantic operation.

### Kernel review concentrated on failure-path accounting and delegated capability limits

Public kernel work in the daily window repeatedly tested invariants outside the success path. BPF stream changes examined capacity reservation after allocation failure, staged-write accounting, partial progress when a userspace read faults, and rejection of output larger than the backing buffer. The [current stream implementation](https://github.com/torvalds/linux/blob/master/kernel/bpf/stream.c) is the primary code boundary. The practical test rule is transactional: after every failed allocation or copy, reported capacity must match allocated storage, reserved bytes must be returned exactly once, and a read that already copied bytes must not erase that progress. Failure injection and oversize cases belong in [BPF selftests](https://github.com/torvalds/linux/tree/master/tools/testing/selftests/bpf), not only in review reasoning.

A separate review questioned whether delegated BPF-token programs should reach connection-tracking kfuncs whose capability checks assume a more privileged network context. The [BPF token UAPI](https://github.com/torvalds/linux/blob/master/include/uapi/linux/bpf.h) delegates selected commands, map types, program types, and attach types inside a user namespace; that does not automatically make every kfunc safe for token-loaded programs. The next diagnostic is to test both token and non-token loading for every exposed program-type/kfunc pair and confirm that capability checks occur in the intended namespace. The unresolved boundary is per-kfunc delegation policy, so a review-stage restriction should not be described as a guarantee in released kernels until it lands.

### Quiet targets were still checked

The scheduler help surfaces, public practitioner forum, and most project-specific support channels had no substantive new engineering exchange inside the daily window. One networking surface contained only community scheduling, and several project channels contained introductions or automated maintenance notices. They were accessible and counted as quiet rather than skipped. A recent public post about crash-safe BPF loader reconciliation remained outside the 24-hour window and was not needed as a fallback.
