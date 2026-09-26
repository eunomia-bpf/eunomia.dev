---
date: 2026-09-23
slug: ebpf-map-reuse-semantic-compatibility
title: "Does a Reused eBPF Map Still Mean the Same Thing?"
description: "libbpf can reject pinned-map reuse when map parameters change, but equal byte sizes do not prove that an upgraded program interprets persistent state with the same schema or meaning."
tags:
  - Daily Report
  - eBPF
  - Linux
  - libbpf
  - BTF
  - Compatibility
research_question: "When an eBPF application reuses a pinned map across software upgrades, what evidence should prove that the old bytes still have the schema and semantics expected by the new program?"
source_cutoff: 2026-09-23
status: daily-report
---

# Does a Reused eBPF Map Still Mean the Same Thing?

A daemon upgrades from version N to N+1. Both versions define a pinned hash map with the same map type, 16-byte key, 32-byte value, entry count, and flags. libbpf opens the pinned map, sees compatible map parameters, and reuses it.

The new program loads. The verifier is satisfied. Every lookup returns the expected number of bytes.

That still does not prove the state means the same thing.

Version N might interpret the first eight bytes of the value as `last_seen_ns` and the next four as a policy generation. Version N+1 can preserve the total value size while reordering those fields, changing signedness, reusing a field for a different unit, or replacing an identifier with another identifier that happens to have the same width. The kernel map object remains structurally loadable while the application silently reads old state under a new schema.

This report asks a narrower question than [transactional eBPF application upgrade](https://eunomia.dev/research/stateful-ebpf-transactional-upgrade/). That earlier report covered prepare/migrate/commit/retire across programs, links, maps, and controllers. Here the focus is the **admission decision for one existing map**: before a new artifact is allowed to reuse old state, what should establish that the state representation is compatible?

It also continues the current deployment-compatibility series after [host capability evidence](https://eunomia.dev/research/ebpf-kernel-capability-evidence/), [cross-kernel semantic compatibility](https://eunomia.dev/research/ebpf-kernel-upgrade-semantic-compatibility/), and [typed interface negotiation](https://eunomia.dev/research/ebpf-kernel-interface-negotiation/). Those boundaries concern whether code can run and which interface variant should run. This one concerns **whether state created by an older application generation can safely be interpreted by the new one**.

<!-- more -->

## Pinning extends object lifetime, not application-schema lifetime

Linux exposes `BPF_OBJ_PIN` and `BPF_OBJ_GET` so BPF objects can be referenced through bpffs paths. A pin holds a reference to the object after the creating process exits; a later process can open the same kernel object and continue using its contents.

That is an object-lifetime mechanism. It is not a declaration that a future program has the same key or value semantics.

A second boundary matters here: ordinary bpffs pinning is not durable storage across a machine reboot. The pinned kernel object disappears with the kernel unless some separate application mechanism serializes and restores state. So the direct problem is most common across controller restarts, daemon upgrades, program reloads, or in-place application generations within one boot. The same schema question also applies to separately restored state, but restoration is an additional mechanism rather than a property of `BPF_OBJ_PIN` itself.

This distinction prevents two different claims from being mixed together:

```text
object lifetime
    "Can another process still get an fd for this map?"

representation compatibility
    "Will the new producer and consumer interpret existing key/value bytes correctly?"
```

The first can be true while the second is false.

## libbpf already checks useful map parameters, but not the whole state contract

Current libbpf has an automatic pinned-map reuse path. Its `map_is_reuse_compat()` helper retrieves `bpf_map_info` for the existing map and compares the map type, key size, value size, maximum entries, flags, and `map_extra` against the new map definition. Some map-specific normalization is applied before the comparison.

Those checks are necessary. Reusing a hash map where the new object expects an array, or reusing a 24-byte value where the new program expects 32 bytes, should fail early.

But they deliberately operate at the map-definition level. Consider two value types:

```c
/* version N */
struct flow_state {
    __u64 last_seen_ns;
    __u32 policy_generation;
    __u32 verdict;
    __u64 bytes;
    __u64 packets;
};

/* version N+1: same total size, incompatible interpretation */
struct flow_state {
    __u64 bytes;
    __u32 verdict;
    __u32 policy_generation;
    __u64 last_seen_ns;
    __u64 packets;
};
```

Both definitions can produce the same `value_size`. The map type, key size, entry count, flags, and `map_extra` can also remain identical. A parameter-level reuse check therefore cannot distinguish them.

This is not a libbpf bug. libbpf cannot infer arbitrary application semantics from byte counts. The missing layer belongs to the application/tooling contract around persistent map state.

## BTF provides type evidence, but raw BTF IDs are not a schema version

BTF makes the problem more tractable because a BPF map can carry key/value type information. Linux exposes `btf_id`, `btf_key_type_id`, and `btf_value_type_id` in `bpf_map_info`, and tooling can retrieve the associated BTF blob and inspect the full type graph.

That is much richer evidence than `value_size == 32`.

However, simply comparing numeric type IDs is not a portable compatibility rule. A type ID identifies a type inside one particular BTF object. Rebuilding an application can renumber types without changing their meaning, and two independently loaded BTF blobs can assign different IDs to structurally identical types. Conversely, preserving a type name is not enough when field offsets or meanings changed.

A useful reuse decision therefore needs a stable representation of the relevant type graph, not just the kernel-local integer ID.

Even structural identity is not the end of the problem. These definitions have the same layout:

```c
struct token_bucket_v1 {
    __u64 last_refill_ns;
    __u64 tokens;
};

struct token_bucket_v2 {
    __u64 last_refill_ns;
    __u64 tokens; /* now measured in milli-tokens */
};
```

BTF can prove the shape. It cannot infer the unit change in the comment or the application invariant that one token used to mean one request.

The state contract therefore has at least two layers:

1. **structural schema** — type graph, field offsets, widths, signedness, arrays, nested types, enum representation, and map parameters;
2. **semantic schema** — units, identifier namespaces, lifecycle epochs, valid ranges, ownership rules, and invariants that are not represented by the C layout alone.

## Explicit `bpf_map__reuse_fd()` is even more reason to make the contract visible

libbpf also exposes `bpf_map__reuse_fd()`, which lets a loader explicitly associate an existing map fd with a map in a BPF object. This is useful when userspace deliberately manages map lifetime or shares state across objects.

It should not be interpreted as a compatibility proof. The API is a mechanism for selecting the existing kernel object. The application still owns the decision that the object is appropriate for the new artifact.

That separation is healthy: low-level libraries should not pretend to know domain semantics. But it means production loaders need an explicit policy rather than treating a successful fd reuse as evidence that state is safe.

## Where current practice is still weak

### Equal map parameters can admit a wrong schema

The first gap is mechanical. A loader can have an existing map and a new map definition whose kernel-visible parameters agree, while their BTF structures differ in ways that preserve total key/value sizes.

The direct experiment is simple: generate map-schema mutations that keep all parameters checked by `map_is_reuse_compat()` constant. Reorder equal-width fields, change nested structs while preserving size, move bitfields, change enum interpretation, or alter key composition. Then measure which mutations are admitted by parameter-only reuse and which produce silent behavioral errors rather than load failures.

A compatibility mechanism is useful only if it rejects the wrong-schema cases without rejecting ordinary rebuilds whose effective schema is unchanged.

### Structural compatibility is not semantic compatibility

A stronger schema checker still cannot infer application meaning. A 32-bit field can change from milliseconds to microseconds without any BTF delta. An integer can move from process ID to cgroup ID. A generation counter can wrap under a new policy. A cache value may remain readable but no longer be valid under the new algorithm.

So the contract needs an application-declared semantic revision or invariant set in addition to structural evidence.

The evaluation should include deliberately same-layout semantic mutations. If the mechanism only detects layout changes, it protects against accidental ABI drift but not persistent-state semantic drift.

### Live migration has a concurrency boundary

Even when N and N+1 know how to transform one entry, migrating a live hash map is not equivalent to converting an offline file. BPF programs can update entries while userspace iterates them. LRU maps can evict entries. Per-CPU maps have multiple value slots. Maps can contain other maps or references to kernel objects. Timers and spin locks impose additional constraints on some values.

A migration mechanism therefore needs a cut between old and new state generations. Otherwise the result can contain some entries from before migration and some updates from after it without a well-defined ordering.

This is where the narrow reuse problem meets the earlier transactional-upgrade work: schema admission can say whether direct reuse is legal; if it is not, a migration protocol still needs generation control and rollback.

## Direction 1: derive a stable structural fingerprint from the BTF type graph

**Gap.** Map-parameter equality catches size and map-shape changes but not many key/value layout changes. Raw BTF type IDs are not stable across independent BTF objects.

**Mechanism.** Compute a canonical fingerprint for the effective key and value schemas. Starting from each map's BTF key/value type, recursively normalize the reachable type graph:

```text
kind
  + resolved size/alignment
  + member name
  + member bit/byte offset
  + signedness / integer encoding
  + array length and element schema
  + enum representation
  + nested structural fingerprints
        -> canonical schema digest
```

The normalization should ignore BTF-local numeric IDs and other build artifacts that do not change the representation. The resulting digest is stored in an application manifest or metadata sidecar and compared before a pinned map is reused.

The hard part is defining *compatible* rather than merely *identical*. Renaming a C typedef should not force a migration. Adding explicit padding may be harmless. A producer/consumer pair that only reads a prefix might intentionally allow an append-only extension. On the other hand, ignoring field names can hide reordering of same-width semantic fields.

So the artifact should support at least two policies: exact structural identity and declared compatible evolution with explicit rules.

**Artifact.** A libbpf-adjacent `map-schema` tool that extracts canonical BTF fingerprints from an object and an existing map, prints a human-readable structural diff, and emits a machine-readable compatibility result.

**Evaluation.** Build a mutation corpus across key/value structs, nested types, arrays, enums, padding, CO-RE-friendly source changes, compiler versions, and BTF deduplication changes. Measure false accepts, false rejects, digest stability under semantically neutral rebuilds, and time added to startup.

**Production value.** A loader can explain "reuse rejected because `flow_state.last_seen_ns` moved from offset 0 to 16" instead of discovering the mismatch through corrupted state later.

**Failure condition.** If canonicalization is unstable across ordinary toolchain changes, or if most real applications do not preserve BTF for maps, the fingerprint cannot be a mandatory fleet-wide gate. It would need a fallback manifest generated from source definitions.

## Direction 2: make map state versioned and migrate explicitly when semantics change

**Gap.** Structural identity cannot represent changes in units, identifier namespaces, validity epochs, or algorithm invariants.

**Mechanism.** Give each state-bearing map a logical schema contract separate from its bpffs path:

```text
map_identity: flow_state
structural_fingerprint: sha256:...
semantic_revision: 4
lifecycle: must-preserve
migration_from: [2, 3]
reset_policy: forbidden
```

On startup, the loader classifies each existing map into one of four outcomes:

```text
direct reuse
compatible read / migrate before write
explicit reset
hard refusal
```

If migration is required, create a new generation map rather than mutating the old representation in place. Quiesce or generation-gate writers, transform entries, validate counts and domain invariants, switch the program/controller to the new map generation, then retire the old map only after the new generation is accepted. Keep the old pin long enough for rollback.

This is intentionally narrower than inventing another full application transaction system. The new contribution is that **schema evidence determines which maps require migration and which may be reused directly**.

**Artifact.** A versioned map manifest plus a migration runner with generated structural adapters for simple transformations and application callbacks for semantic transformations.

**Evaluation.** Test rolling controller upgrades with concurrent map updates, process crashes at every migration phase, LRU eviction, per-CPU values, and maps whose entry count exceeds memory available for a full duplicate. Measure lost/duplicated updates, rollback success, downtime, and the fraction of migrations that can be generated safely.

**Production value.** Operators can distinguish "old state is intentionally reusable" from "the loader happened to accept the old fd." They also get an explicit reset policy instead of accidentally clearing security or accounting state when compatibility is uncertain.

**Failure condition.** If applications usually treat pinned state as disposable caches and can cheaply rebuild it, explicit migration machinery may cost more than resetting the map. The lifecycle declaration should allow `reset-safe` rather than forcing persistence everywhere.

## Direction 3: validate a new decoder against old state before granting write authority

**Gap.** A manifest can be wrong. Two versions can declare the same semantic revision while the new code interprets a field incorrectly. A migration callback can also produce values that are structurally valid but violate application invariants.

**Mechanism.** Add a shadow admission phase. Before N+1 gains write authority, let it read a bounded sample or snapshot of N's map state and evaluate declared invariants:

- ranges and monotonicity of counters;
- identifier resolution against the control plane;
- cross-field invariants such as `packets <= bytes` where applicable;
- generation/epoch membership;
- old-decoder versus new-decoder results for the same raw entry;
- aggregate parity for metrics or policy decisions derived from the map.

For maps where reads have no side effects, N and N+1 can decode the same raw bytes and compare normalized logical records. For active maps, the comparison needs an epoch or snapshot boundary so concurrent updates are not misclassified as decoder disagreement.

When shadow validation passes, the loader records a reuse receipt tying the existing map identity, structural fingerprint, semantic revision, new object build identity, and validation result. When it fails, the system migrates, resets only if policy permits, or refuses the upgrade.

**Artifact.** A reusable state-compatibility harness with map-type-specific sampling/snapshot adapters and invariant plugins.

**Evaluation.** Inject same-layout semantic bugs, stale schema declarations, corrupted entries, partial migrations, and concurrent updates. Compare manifest-only admission with shadow validation. Measure silent bad reuses caught, false alarms, startup delay, and coverage under large maps.

**Production value.** The compatibility decision gains behavioral evidence instead of depending entirely on metadata authored before deployment.

**Failure condition.** If the application has no cheap side-effect-free decoder or meaningful invariants, shadow validation can become expensive ceremony. In that case it should remain a risk-based option for `must-preserve` state rather than a universal requirement.

## Practical deployment guidance today

A production loader does not need to wait for new kernel APIs to become safer. It can enforce a state contract in userspace now:

1. **Classify every pinned map.** Mark it `ephemeral`, `reset-safe`, `must-preserve`, or `migrate`. Do not infer lifecycle from the bpffs path.
2. **Check the kernel-visible definition first.** Type, key/value sizes, maximum entries, flags, `map_extra`, and map-type-specific constraints remain the first gate.
3. **Record schema metadata next to the artifact.** Prefer a BTF-derived structural fingerprint when BTF is available, plus an explicit application semantic revision.
4. **Do not use `bpf_map__reuse_fd()` as a compatibility verdict.** Treat it as the mechanism applied *after* compatibility has been established.
5. **Refuse ambiguous `must-preserve` state.** A failed or missing schema comparison should not silently become a reset or blind reuse for security, billing, policy, or accounting maps.
6. **Migrate into a new generation.** Preserve the old map until validation and cutover succeed; define rollback before starting migration.
7. **Log the evidence.** Record expected and observed map parameters, structural digest, semantic revision, migration decision, and final map identity so an incident can reconstruct why a particular state object was accepted.

This is also a useful packaging boundary. Program/interface compatibility and state compatibility can be evaluated separately:

```text
artifact + target kernel
        -> interface/capability admission

artifact + existing map state
        -> representation/semantic admission

both pass
        -> load, attach, and grant state authority
```

A loader should not let success in the first path imply success in the second.

## What would change this conclusion?

The need for a richer contract would weaken if production data showed that long-lived pinned maps are almost always either immutable ABI-stable structures or disposable caches that are reset on every application upgrade. In that environment, parameter equality plus explicit reset could be sufficient.

A structural fingerprint would also be less useful if BTF-derived schemas prove too unstable across normal compiler/toolchain rebuilds or if deployed applications routinely strip the information needed to recover key/value types. A source-generated schema manifest might then be a better artifact.

The shadow-validation direction fails if real map workloads cannot provide a consistent, low-cost observation point and the invariants are too weak to discriminate correct decoding from wrong decoding.

There is also an important counterexample: if a map value is intentionally treated as an opaque fixed-size byte string by both generations and no code assigns meaning to its internal layout, a structural change inside some source-language wrapper does not matter. Compatibility should be defined by the actual consumers, not by a reflexive rule that every type spelling must match.

Finally, if an upgrade intentionally resets the state and that reset is safe under the application's policy, there is no reason to preserve old semantics. The compatibility gate matters only when old state is carried forward.

The current Linux/libbpf boundary still leaves a clear gap. Kernel-visible map parameters establish that two objects can share one storage shape. BTF can expose much of the representation. Neither, by itself, proves that a new application generation assigns the same meaning to the old bytes. A safe reuse decision should therefore combine **map-definition compatibility, stable structural schema evidence, and an explicit application semantic contract**, with migration or refusal when those layers disagree.

## Sources

- [Linux kernel documentation: BPF maps](https://docs.kernel.org/bpf/maps.html)
- [Linux kernel documentation: BPF Type Format (BTF)](https://docs.kernel.org/bpf/btf.html)
- [Linux kernel source: libbpf map reuse implementation](https://github.com/torvalds/linux/blob/master/tools/lib/bpf/libbpf.c)
- [eBPF Docs: `bpf_map__reuse_fd`](https://docs.ebpf.io/ebpf-library/libbpf/userspace/bpf_map__reuse_fd/)
- [eBPF Docs: `BPF_MAP_CREATE`](https://docs.ebpf.io/linux/syscall/BPF_MAP_CREATE/)
- [Eunomia Daily Report: Can a Stateful eBPF Application Upgrade Atomically?](https://eunomia.dev/research/stateful-ebpf-transactional-upgrade/)
- [Eunomia Daily Report: Can an eBPF Loader Trust the Kernel Version?](https://eunomia.dev/research/ebpf-kernel-capability-evidence/)
- [Eunomia Daily Report: Can an eBPF Object Keep Its Meaning After a Kernel Upgrade?](https://eunomia.dev/research/ebpf-kernel-upgrade-semantic-compatibility/)
- [Eunomia Daily Report: Can an eBPF Loader Treat a kfunc as Just Present or Missing?](https://eunomia.dev/research/ebpf-kernel-interface-negotiation/)
