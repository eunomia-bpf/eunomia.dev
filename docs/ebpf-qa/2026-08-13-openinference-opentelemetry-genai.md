# How should OpenInference coexist with OpenTelemetry's GenAI semantic conventions?

**Short answer:** use OpenTelemetry as the transport and lifecycle foundation, treat the OpenTelemetry GenAI semantic conventions as the target interoperability contract, and treat OpenInference as a donated library and compatibility source whose instrumentations are being integrated incrementally. The donation does not make every existing OpenInference span semantically identical to an OpenTelemetry GenAI span overnight.

OpenInference already emits normal OpenTelemetry data and can export through OTLP. Its value is broad framework coverage and mature AI-specific instrumentation. OpenTelemetry's GenAI project defines the vendor-neutral names and structures for model calls, agents, MCP, metrics, and events. These are complementary layers, but their attribute names and modeling choices can differ. During migration, choose one canonical schema at the collector boundary and test the resulting telemetry rather than enabling two instrumentations for the same library.

## What the donation actually changes

The official OpenTelemetry donation repository says it received the OpenInference code grant. Its documented process is deliberately incremental: the original code is received in a one-time contribution, the receiving repository is archived, and the GenAI SIG cherry-picks instrumentations one by one.

That establishes governance and a path for code reuse. It does not imply that:

- every package has already moved to a final OpenTelemetry namespace;
- all OpenInference attributes already match the current OTel GenAI schema;
- users should install both instrumentations for the same SDK; or
- existing stored traces are automatically rewritten.

The OpenTelemetry GenAI semantic conventions now live in their own official repository. That repository extends the core semantic conventions and covers spans, metrics, events, agents, MCP, and provider-specific conventions. It is the public contract to target for new interoperable telemetry. OpenInference remains a real, maintained instrumentation ecosystem, and its repository explicitly supports any OTLP-compatible collector.

## Separate wire compatibility from semantic compatibility

OTLP answers “can the collector receive this telemetry?” Semantic conventions answer “do different producers mean the same thing by these fields?” Two libraries can both emit valid OTLP while representing the same concept with different keys, span names, or value encodings.

For example, a compatibility layer may need to translate an OpenInference span-kind attribute into an OTel `gen_ai.operation.name`, not merely rename a package. The accepted collector-component discussion describes this as in-flight attribute normalization because the sources already use the standard OTLP receiver. That is a useful architectural boundary:

```text
application instrumentation
        ↓ OTLP
collector normalization and redaction
        ↓ canonical OTel GenAI schema
backend, dashboards, and alerts
```

Keep normalization explicit and versioned. A mapping should state its source profile, target semantic-convention version, whether original fields are retained, and how values are transformed. Blind key renaming can corrupt meaning when one source value maps to several OTel operations or when event payloads carry different privacy risks.

## A safe migration plan

First inventory which library instruments each SDK and whether the application, framework, or vendor already emits telemetry. Duplicate instrumentation is more damaging than a temporary schema mismatch: it creates nested or repeated spans, double-counted tokens, and ambiguous error attribution.

Then select a canonical target version and build a small conformance corpus:

1. one successful model call;
2. one streaming response;
3. one tool call and result;
4. one agent handoff or nested invocation;
5. one failure; and
6. one case with prompts and outputs disabled or referenced externally.

For each trace, verify operation name, provider and model identity, request and response token usage, finish reasons, error status, tool-call linkage, and conversation-content policy. Compare by trace semantics, not serialized JSON order.

During a transition, prefer one of three modes:

- keep OpenInference instrumentation and normalize at the collector;
- replace it with an adopted OpenTelemetry instrumentation after conformance tests pass; or
- keep source-specific fields alongside canonical fields for a bounded compatibility window.

Do not run both producers on the same SDK merely to compare them in production. If a shadow comparison is necessary, isolate it in a test workload or suppress export from one pipeline.

## Privacy and rollout limits

GenAI telemetry can contain prompts, responses, tool arguments, retrieved documents, and identifiers. Schema convergence does not make those fields safe. Apply content capture controls at instrumentation, enforce redaction and size limits in the collector, and configure backend retention separately. Preserve references rather than bodies when the application already stores content securely.

The remaining uncertainty is package-level timing. The donation process explicitly allows instrumentations to move one by one, so users must check the current package repository and release notes for each SDK. “OpenInference was donated” is not a sufficient upgrade instruction by itself.

## References

- [OpenTelemetry: OpenInference donation repository and integration process](https://github.com/open-telemetry/donation-openinference)
- [OpenTelemetry GenAI semantic conventions repository](https://github.com/open-telemetry/semantic-conventions-genai)
- [OpenTelemetry documentation: GenAI conventions moved to the dedicated repository](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [OpenInference repository: OpenTelemetry instrumentation and supported OTLP destinations](https://github.com/Arize-ai/openinference)
- [OpenTelemetry Collector proposal: normalize GenAI attributes emitted over OTLP](https://github.com/open-telemetry/opentelemetry-collector-contrib/issues/46069)
- [BPF mailing list: redesigned verifier diagnostics](https://lore.kernel.org/bpf/a4e7eebf34507bf3041f232561e6f0a8acd47d7f.camel@gmail.com/T/#t)
- [BPF mailing list: loader file-descriptor proposal](https://lore.kernel.org/bpf/20260813002618.3755631-1-tweek@google.com/T/#t)

## Community discussion today

Today's visible review covered all 6 approved communities and all 15 allowlisted channels or public pages; every target was accessible. The selected question came from the 24-hour window, so no seven-day fallback was used. Names, accounts, employers, channel identities, message links, exact times, private topology, original logs, and searchable phrasing have been removed. No raw transcript was retained.

### GenAI observability is now a migration problem, not only a coverage problem

The strongest discussion asked whether OpenTelemetry was building a competing GenAI framework after receiving OpenInference. The underlying confusion comes from mixing three layers: OpenTelemetry's telemetry APIs and OTLP transport, the GenAI semantic contract, and SDK-specific instrumentation libraries. The donation primarily changes ownership and supplies mature instrumentation code; the dedicated OTel repository remains the canonical place for shared GenAI semantics.

A related daily discussion exposed the operational consequence. Older Python instrumentations were being deprecated while an upstream SDK made a new major release. The immediate question was whether the deprecated package should temporarily cap the supported SDK range before its final release. This is exactly where package ownership, schema ownership, and compatibility policy must remain separate. A deprecation notice does not protect users from a breaking dependency, and a semantic-convention mapping does not repair an incompatible monkey patch. Maintainers need a bounded version constraint, an explicit successor package, and tests against both the last supported and first unsupported SDK versions.

For practitioners, the diagnostic path is concrete: identify the actual instrumentation package loaded in the process, record its version and the target SDK version, capture one minimal trace, and inspect both span duplication and attribute shape. If a collector normalizer is used, test its output independently from the instrumentation. What remains uncertain is the adoption schedule for each donated library; the official process intentionally moves them one at a time.

### Runtime eBPF instrumentation was quiet while project automation was active

The eBPF instrumentation group had no new substantive technical message in the daily window. Its visible recent material still centered on choosing between runtime eBPF instrumentation and compile-time language instrumentation. The practical boundary remains attachability versus source-level fidelity: eBPF can observe unmodified processes and broad fleets, while compile-time instrumentation can expose language and framework context that kernel-visible signals cannot reconstruct reliably.

Several project-specific channels were quiet. A development-notification channel was active with automated tests, builds, and preview deployments rather than practitioner questions. The general project area contained no new technical request, and the scheduler support area had no daily activity after the earlier command-line argument incident. These surfaces were checked and counted as quiet; they were not used to manufacture a topic.

### Upstream BPF work concentrated on diagnostics, ownership, and failure paths

The public BPF archive was highly active. A verifier-diagnostics series proposed structured categories and source or instruction context for register type safety, memory bounds, resource lifetime, call arguments, execution context, control-flow structure, policy, and verifier limits. This directly addresses a recurring usability gap: the rejection is often correct, but the log does not clearly separate the root cause from later propagated failures. The practical path remains to preserve the first diagnostic event, correlate it with source and instruction context, and avoid treating the final register-state dump as the only explanation.

Other threads covered loading BPF objects from an already opened file descriptor, a possible out-of-bounds relocation at a terminal `LDIMM64`, trampoline-image lifetime after multi-detach failure, stack-depth accounting, lazy population of mmap-able array maps, and 16-byte aggregate returns. Across them, the common mechanism was ownership at a boundary: who owns the file, relocation pair, trampoline image, verification root, or returned register pair when an operation partially succeeds or fails. Tests need to exercise cancellation and rollback, not only the successful path.

The public forum's newest visible item described an XDP/TC DDoS project rather than asking a new troubleshooting question. Its architecture separated host and router modes and included dry-run operation, which is a useful deployment pattern: observe counters and decisions before enabling drops. The Cilium/eBPF discussion area had no new daily technical question; its most recent substantive socket-map thread had already been answered in an earlier Q&A. Together, today's discussions point to the same systems lesson across agent telemetry and kernel BPF: interoperability requires explicit ownership, versioned contracts, and failure-path tests, not merely a shared transport or a successful load call.
