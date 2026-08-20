# How can you tell whether an OpenTelemetry GenAI attribute is stable enough to depend on?

**Short answer:** do not infer stability from an attribute appearing in a generated registry, an instrumentation package emitting it, or a pull request receiving approval. Check the status of the exact semantic-convention document, then check whether the definition is part of a published release and whether the producing instrumentation promises a stable telemetry shape.

At the time of writing, the official OpenTelemetry GenAI semantic-conventions overview is marked **Development**. The dedicated repository has no published release, its changelog contains only an `Unreleased` section, and its top-level schema URL is still unset. That means current GenAI names and signal shapes are useful for experiments and bounded production adoption, but they are not a general-availability compatibility contract. Pin the profile you consume, isolate it behind a translation layer, and expect migration work.

## Four different signals are easy to confuse

OpenTelemetry has several independent kinds of stability. They answer different questions.

### 1. Document status governs the convention being described

OpenTelemetry's document-status rules say that a status applies to the individual document, not to the entire specification. `Development` means the component is incomplete, may change frequently, should not be treated as production-ready, and may even be removed without prior notice. `Release Candidate` sharply limits breaking changes; `Stable` is the general-availability level.

The GenAI overview currently says `Status: Development`. Its generated registry lists many `gen_ai.*` keys, but the stability cells are blank. A populated table proves that a key is defined in the current model; it does not silently promote that key to Stable. If a narrower GenAI document or section later declares a different status, that narrower declaration can be evaluated on its own. Until then, the Development status is the controlling signal for the GenAI convention described there.

Core attributes reused by a GenAI span need two checks. An attribute such as a general network or error field may already have a stable definition in the core registry, while the GenAI span's name, requirement level, placement, or interpretation remains Development. Attribute stability does not automatically stabilize the whole signal.

### 2. Repository state is not release state

The core semantic-conventions repository moved its GenAI definitions to a dedicated repository in its v1.42.0 release and deprecated the old copies. That was a change of ownership and source of truth, not a promotion of GenAI to Stable.

The dedicated repository describes a future development release channel using `vX.Y.Z-dev` tags and `gen-ai-dev` schema URLs. Its current releases page contains no release, the changelog has no released version, and the repository README does not yet publish a schema URL. Therefore:

- `main` is a moving development snapshot;
- an open or merged pull request is not a released convention;
- the old core schema version is not proof that the moved definition is stable; and
- inventing an official-looking GenAI schema URL would make migrations harder, not safer.

Use an actual tag and published schema artifact once they exist. Before then, record the exact source revision or an internal profile version in deployment metadata, without claiming that it is an official OpenTelemetry schema release.

### 3. Instrumentation stability is separate from semantic stability

OTLP can carry arbitrary attributes. A stable SDK, exporter, or instrumentation package can emit Development conventions, and two packages can emit valid OTLP while disagreeing about span names, field requirements, or value semantics.

OpenTelemetry's telemetry-stability rules explicitly distinguish stable and unstable producers. An unstable producer gives no guarantee that its emitted shape will remain compatible across versions. A stable producer must label that promise and, under the current schema-transformation moratorium, cannot rely on a future schema file to excuse breaking its emitted telemetry.

When evaluating a library, inspect its own release notes and stability declaration. Do not transfer the maturity of the OpenTelemetry API, SDK, or OTLP protocol to the GenAI attributes it happens to emit.

### 4. Proposal maturity is not convention maturity

Agent lifecycle events provide a current example. A public proposal defines pause, resume, checkpoint, and pause-resolution events for long-running agents, with correlation attributes that can span process and trace boundaries. The design is concrete and has a reference scenario, but the pull request remains open. Those names are proposals, not a published contract.

For experimentation, emit them behind a feature flag or map internal events to the proposed shape at a collector boundary. For dashboards, alerts, or stored-query APIs that must survive upgrades, keep an internal versioned schema and translate only after a proposal lands in a release you have chosen to adopt.

## A production-safe adoption pattern

Treat every telemetry producer as implementing an explicit profile. A useful profile records:

```text
instrumentation package and version
semantic-convention repository revision or released schema URL
enabled signals and content-capture policy
collector normalization version
backend query/dashboard version
```

Then build a small conformance corpus with one successful model call, one streaming call, one tool invocation, one agent or workflow operation, and one failure. For each fixture, assert the span name and kind, operation name, model/provider identity, token accounting, error representation, and whether sensitive content is absent by default. Store expected normalized output, not just a screenshot of a dashboard.

At the collector boundary:

1. accept the producer profile you have tested;
2. normalize it into an internal versioned schema;
3. reject, quarantine, or separately label unknown profiles rather than silently mixing them;
4. preserve the source profile as low-cardinality metadata; and
5. apply redaction and cardinality controls before export.

When upgrading, replay the corpus through both versions and diff the normalized result. Use a bounded dual-read or shadow pipeline if a dashboard migration needs overlap. Avoid running two instrumentations against the same client library: duplicate spans and token counts obscure whether a schema migration is correct.

Content-bearing fields require an additional gate. The GenAI registry warns that input and output message attributes can contain sensitive information. A field becoming Stable would not make its contents safe, low-cardinality, or appropriate for default capture. Stability, privacy, and cost are three separate acceptance decisions.

## What would make a dependency reasonable?

There are two valid meanings of “depend on”:

- **Experimental or bounded production use:** reasonable now if the producer version and convention revision are pinned, compatibility tests exist, content capture is controlled, and a migration layer owns changes.
- **A public, long-lived contract with no expected field migration:** wait for the relevant GenAI document or section to reach at least Release Candidate, prefer Stable, and adopt a published repository release and schema artifact rather than `main`.

There is no authoritative per-attribute graduation calendar in the current sources. The reliable indicators are repository status, document status, released artifacts, and the actual instrumentation's stability promise. An issue, meeting plan, or accepted design can suggest direction, but it cannot provide a compatibility date.

## References

- [OpenTelemetry GenAI semantic conventions: current Development status](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/README.md)
- [OpenTelemetry GenAI generated attribute registry](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/registry/attributes/gen-ai.md)
- [OpenTelemetry definitions of Development, Release Candidate, Stable, and Deprecated](https://opentelemetry.io/docs/specs/otel/document-status/)
- [OpenTelemetry telemetry-stability requirements for producers](https://opentelemetry.io/docs/specs/otel/telemetry-stability/)
- [OpenTelemetry semantic-conventions v1.42.0: GenAI definitions moved to the dedicated repository](https://github.com/open-telemetry/semantic-conventions/releases/tag/v1.42.0)
- [Dedicated GenAI repository: schema URL is not yet published](https://github.com/open-telemetry/semantic-conventions-genai)
- [Dedicated GenAI repository: no published releases](https://github.com/open-telemetry/semantic-conventions-genai/releases)
- [Dedicated GenAI repository: current changelog is unreleased](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/CHANGELOG.md)
- [Dedicated GenAI repository: planned development release and schema process](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/RELEASING.md)
- [Open agent-lifecycle event proposal](https://github.com/open-telemetry/semantic-conventions-genai/pull/445)
- [Git submodule initialization and recursive update](https://git-scm.com/docs/git-submodule)
- [Linux source for the `bpf_strncmp` helper](https://github.com/torvalds/linux/blob/master/kernel/bpf/helpers.c)
- [GNU C Library dynamic-linker guidance](https://sourceware.org/glibc/manual/latest/html_node/Dynamic-Linker.html)
- [BPF thread: preserve programs until their trampoline image is released](https://lore.kernel.org/bpf/c145f1ec-a4fc-42e4-a267-0667775bf5f8@linux.dev/T/#t)
- [BPF report: task hang while freeing a socket map](https://lore.kernel.org/bpf/87ecfsvnb3.fsf@cloudflare.com/T/#t)
- [BPF fix: one-CPU per-CPU freelist infinite loop](https://lore.kernel.org/bpf/178725180892.429815.3721455388690610183.git-patchwork-notify@kernel.org/T/#t)
- [BPF fix: reject oversized array maps before signed iteration overflows](https://lore.kernel.org/bpf/20260820084643.35489-1-meishaoming@xiaomi.com/T/#t)
- [bpftool fix: preserve every type in a sorted C dump](https://lore.kernel.org/bpf/478fbefa108d7da8eb28897998857ab72467b276.camel@gmail.com/T/#u)
- [cgroup proposal: expose CPU statistics through BPF kfuncs](https://lore.kernel.org/bpf/20260818002450.3071325-1-ziyang.meme@gmail.com/T/#t)

## Community discussion today

Today's ordinary visible-browser review covered all 6 approved communities and all 15 allowlisted channels or public pages. Every target was accessible. The selected question came from the 24-hour window, so the seven-day fallback was not used. Names, accounts, employers, workspace and channel identities, closed-chat links, exact times, private topology, raw logs, and searchable chat wording have been removed. No raw transcript was retained.

### GenAI users want a compatibility signal, not just a list of attributes

The leading observability discussion asked where to find the current status and expected timeline for individual GenAI attributes. A related thread proposed lifecycle events for agents that pause, checkpoint, and resume. The shared concern is operational: dashboards, metric aggregation, and stored queries need to know whether a field is a durable contract, while instrumentation authors need room to correct an incomplete model.

The immediate answer is to treat the current GenAI group as Development and an open proposal as experimental. The [official status definitions](https://opentelemetry.io/docs/specs/otel/document-status/) and the [current GenAI overview](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/README.md) provide the authoritative maturity signal; the [open lifecycle proposal](https://github.com/open-telemetry/semantic-conventions-genai/pull/445) shows direction but does not establish a released field. The practical diagnostic is to inventory the producer package and emitted profile, then compare it with a pinned registry revision. What remains unresolved is the graduation schedule: the current public sources provide no per-attribute dates.

### Build failures were often dependency or host-loader failures before eBPF began

Several project-support signals looked like eBPF loader failures but belonged to earlier layers. Missing libbpf or bpftool headers after cloning point first to an uninitialized repository submodule; the [Git submodule contract](https://git-scm.com/docs/git-submodule) explains that `update --init --recursive` populates the commits recorded by the superproject. A verifier report naming an unavailable helper needs to be checked against the running kernel's helper set, not only the source tree used to compile; the current [kernel helper implementation](https://github.com/torvalds/linux/blob/master/kernel/bpf/helpers.c) confirms that `bpf_strncmp` is a kernel-provided helper rather than a userspace fallback.

Another warning originated in the ELF dynamic loader because of a system-wide preload entry, while the BPF program itself ran. The [GNU C Library documentation](https://sourceware.org/glibc/manual/latest/html_node/Dynamic-Linker.html) makes the ownership boundary clear: the dynamic linker loads dependencies before application startup. The right diagnostic sequence is therefore repository dependencies, executable loader, kernel/helper availability, libbpf open/load, and only then verifier or attach behavior. The remaining uncertainty is version-specific packaging: a source build can change the launcher path, but it does not repair a host-wide preload configuration.

### Kernel discussions concentrated on lifetime, teardown, and bounded iteration

Public kernel work repeatedly asked what must remain alive while another kernel object can still reference it. One fix keeps BPF programs alive until the trampoline image that calls them is freed; a separate report found a task stuck while socket-map teardown waited inside socket locking. These are failure-path problems: successful attach or insertion does not prove that detach, replacement, and destruction use the same ownership order. The upstream [trampoline lifetime fix](https://lore.kernel.org/bpf/c145f1ec-a4fc-42e4-a267-0667775bf5f8@linux.dev/T/#t) and [socket-map teardown report](https://lore.kernel.org/bpf/87ecfsvnb3.fsf@cloudflare.com/T/#t) are the public evidence.

Two smaller fixes exposed the same lesson through numeric boundaries. A per-CPU freelist could loop forever when only one CPU was possible, and an array map accepted a size that later overflowed a signed iterator. The practical regression matrix must include the minimum topology, maximum accepted sizes, partial initialization, rollback, and concurrent teardown. The [single-CPU freelist fix](https://lore.kernel.org/bpf/178725180892.429815.3721455388690610183.git-patchwork-notify@kernel.org/T/#t) and [array-size guard](https://lore.kernel.org/bpf/20260820084643.35489-1-meishaoming@xiaomi.com/T/#t) remain active kernel changes, so operators should verify the exact kernel they ship instead of assuming that an upstream patch is already present.

### Introspection is expanding, but its output still needs correctness tests

Tooling work fixed a sorted bpftool C dump that could silently omit a type. That failure is more dangerous than a syntax error because generated output can remain plausible while becoming incomplete. Tests should compare the set of emitted type identities before and after sorting, not merely compile one generated header. The [bpftool patch](https://lore.kernel.org/bpf/478fbefa108d7da8eb28897998857ab72467b276.camel@gmail.com/T/#u) adds coverage around that exact boundary.

A separate proposal exposes CPU-cgroup statistics to BPF through kfuncs. This would let in-kernel policy read accounting data without reconstructing it from unrelated events, but a read API does not itself define scheduling or enforcement policy. Callers still need to handle hierarchy, counter semantics, and kernel-version availability. The [public cgroup series](https://lore.kernel.org/bpf/20260818002450.3071325-1-ziyang.meme@gmail.com/T/#t) is a proposal under review, not a portable released interface.

The remaining project, scheduler, networking, and public-forum targets had no substantive new technical item inside the daily window, or only older discussion and routine announcements. They were all accessible and were counted as quiet, not skipped.
