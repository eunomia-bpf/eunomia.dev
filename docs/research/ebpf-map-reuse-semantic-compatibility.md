---
date: 2026-09-30
slug: ebpf-map-reuse-semantic-compatibility
title: "Does a Reused eBPF Map Still Mean the Same Thing?"
description: "Pinned eBPF maps can pass libbpf reuse checks while upgraded programs may interpret the same bytes under a different structural or application schema."
tags:
  - Daily Report
  - eBPF
  - Linux
  - libbpf
  - BTF
  - Compatibility
research_question: "When an eBPF application reuses a pinned map across software upgrades, what evidence should prove that the old bytes still have the schema and semantics expected by the new program?"
source_cutoff: 2026-09-30
status: daily-report
---

# Does a Reused eBPF Map Still Mean the Same Thing?

An eBPF daemon upgrades from version N to N+1. Both generations define a pinned hash map with the same map type, 16-byte key, 32-byte value, entry count, and flags. libbpf can accept the existing object for reuse, the new program can load, and every lookup can return the expected number of bytes. None of those facts proves that version N+1 gives those bytes the same meaning as version N.

A field can move while total size stays constant. An identifier can change from PID to cgroup ID without changing width. A counter can change units while its C type remains `__u64`. The storage object is reusable, yet the state contract is not.

This report asks a narrower question than [transactional eBPF application upgrade](https://eunomia.dev/research/stateful-ebpf-transactional-upgrade/). That work covers prepare, migrate, commit, rollback, and retirement across a whole application. Here the decision happens earlier: **may this one existing map be interpreted directly by this new artifact, or must it be migrated, reset, or rejected?**

It extends the current deployment-compatibility series after [capability evidence](https://eunomia.dev/research/ebpf-kernel-capability-evidence/), [cross-kernel semantic compatibility](https://eunomia.dev/research/ebpf-kernel-upgrade-semantic-compatibility/), and [typed interface negotiation](https://eunomia.dev/research/ebpf-kernel-interface-negotiation/).

<!-- more -->

## Why eBPF map reuse is not a schema proof

Linux BPF maps are kernel objects shared between BPF programs and userspace. Pinning an object in bpffs gives it a filesystem reference, so a later process can reopen the same live kernel object after the creator exits. That is useful for daemon restarts and rolling application generations.

Pinning is an object-lifetime mechanism, not an application-schema mechanism. Ordinary bpffs pinning also does not turn a kernel object into reboot-durable storage: after a host reboot, preserving state requires a separate serialization and restoration mechanism.

Current upstream libbpf makes the distinction visible. Its automatic pinned-map reuse path retrieves `bpf_map_info` and compares the existing map with the new definition. As of the source cutoff, `map_is_reuse_compat()` checks map type, key size, value size, `max_entries`, map flags, and `map_extra`, with a map-type-specific normalization for devmap flags. These are useful and necessary storage-shape checks.

They are not a complete representation check. Consider:

```c
/* N */
struct flow_state {
    __u64 last_seen_ns;
    __u32 policy_generation;
    __u32 verdict;
    __u64 bytes;
    __u64 packets;
};

/* N+1 */
struct flow_state {
    __u64 bytes;
    __u32 verdict;
    __u32 policy_generation;
    __u64 last_seen_ns;
    __u64 packets;
};
```

Both values have the same size. The map can have the same type, key size, capacity, flags, and `map_extra`. Parameter equality therefore cannot detect the changed interpretation.

This is not a libbpf bug. The low-level loader cannot infer arbitrary application meaning from byte counts.

## BTF gives structural evidence, not an application semantic version

BTF makes the problem more tractable. Linux exposes `btf_id`, `btf_key_type_id`, and `btf_value_type_id` through `bpf_map_info`; tooling can recover the associated BTF blob and inspect the reachable type graph. That graph can describe integer encoding, member offsets, arrays, nested structs, unions, enums, and other representation details.

A raw BTF type ID is still not a portable schema version. The ID is meaningful inside one BTF object. Rebuilding can renumber types without changing the representation, while two independently loaded BTF blobs can assign different IDs to equivalent structures.

Even a perfect structural comparison cannot recover application semantics that are absent from the type graph:

```c
struct token_bucket {
    __u64 last_refill_ns;
    __u64 tokens;
};
```

If version N measures `tokens` in requests and N+1 measures it in milli-tokens, BTF can prove that the layout is unchanged while the application contract has changed.

A state reuse decision therefore needs at least three layers:

1. **map-definition compatibility**: kernel-visible storage properties match;
2. **structural compatibility**: key/value representations are identical or follow an explicitly allowed evolution rule;
3. **semantic compatibility**: units, identifier namespaces, epochs, ownership, validity rules, and invariants still match.

`bpf_map__reuse_fd()` belongs below these layers. It selects an existing map FD; it is not evidence that reuse is correct.

## Where current work is still weak

### Equal map parameters can silently admit a different structure

The simplest missing test is a mutation corpus that preserves every parameter checked by libbpf while changing the key or value layout. Equal-width field reorderings, nested-struct changes, enum reinterpretation, bitfield moves, and key composition changes should all be exercised.

The useful metric is not “did the object load?” It is the false-accept rate: how often does a parameter-only admission decision permit state that the new program interprets incorrectly?

### Structural identity still misses semantic drift

A stronger BTF comparison can catch representation changes and still miss same-layout changes in units, identifier meaning, lifecycle epoch, or algorithm invariants. That makes an application-declared semantic revision necessary for state that must survive upgrades.

The discriminating experiment should deliberately keep BTF identical while changing meaning. If a proposed mechanism cannot reject or migrate those cases, it protects ABI drift but not semantic drift.

### Migration has a concurrency boundary

When direct reuse is not legal, transforming a live map is not the same as converting an offline file. BPF programs can update entries during userspace iteration; LRU maps can evict; per-CPU maps have multiple value slots; maps-of-maps introduce another identity layer.

A migration therefore needs a defined cut between old and new generations. Otherwise the resulting map can combine pre-migration entries with post-migration writes without a coherent ordering rule.

## Promising directions with academic and production value

### Direction 1: canonicalize the BTF type graph into a structural fingerprint

**Gap.** Kernel-visible parameter equality misses layout changes, while numeric BTF IDs are local to one BTF object.

**Mechanism.** Starting from a map's BTF key and value types, recursively canonicalize the reachable representation: kind, resolved size, member names and offsets, integer encoding and signedness, array lengths, enum representation, and nested structural fingerprints. Ignore BTF-local numeric IDs and build artifacts that do not affect representation. Hash the normalized graph.

The tool should support both exact identity and explicitly declared compatible evolution. A typedef rename should not force migration; a field move should. Append-only compatibility should be allowed only when the actual consumers declare that they read a stable prefix.

**Delta from related work.** BTF already transports type information; this mechanism turns it into a stable cross-build comparison artifact rather than treating kernel-local IDs as versions.

**Artifact.** A `map-schema` tool for libbpf applications that prints a machine-readable fingerprint and a human-readable old/new schema diff.

**Evaluation.** Generate a mutation corpus across structs, arrays, enums, padding, compiler versions, BTF deduplication, and semantically neutral rebuilds. Measure false accepts, false rejects, fingerprint stability, and startup overhead.

**Academic value.** The question is how to define representation equivalence over independently generated BTF graphs.

**Production value.** A loader can say “reuse rejected because `flow_state.last_seen_ns` moved from offset 0 to 16” instead of discovering corrupted behavior later.

**Failure condition.** If canonical fingerprints are unstable under ordinary toolchain rebuilds, or deployed artifacts lack usable BTF, a source-generated schema manifest is the simpler baseline.

### Direction 2: version state semantics and make migration an explicit outcome

**Gap.** Structural equality cannot describe units, namespaces, epochs, or application invariants.

**Mechanism.** Give each state-bearing map a logical contract separate from its bpffs path:

```text
map_identity: flow_state
structural_fingerprint: sha256:...
semantic_revision: 4
lifecycle: must-preserve
migration_from: [2, 3]
reset_policy: forbidden
```

At startup, classify existing state into four outcomes: direct reuse, read-and-migrate before write, explicit reset, or refusal. If migration is required, create a new generation map, gate writers, transform and validate entries, switch authority to the new generation, and retain the old pin until rollback is no longer needed.

**Delta from related work.** The earlier transactional-upgrade protocol coordinates a whole application. Here schema evidence determines the per-map admission outcome that feeds such a protocol.

**Artifact.** A versioned map manifest plus a migration runner with generated structural adapters and application callbacks for semantic changes.

**Evaluation.** Exercise rolling upgrades with concurrent writers, process crashes at every phase, LRU eviction, per-CPU values, and maps too large to duplicate cheaply. Measure lost/duplicated updates, rollback success, downtime, and migration cost.

**Academic value.** The research problem is whether state-schema evolution can expose a small compatibility algebra that is useful across map types.

**Production value.** Operators can distinguish intentionally reusable state from a map that merely happens to pass a low-level FD reuse check.

**Failure condition.** If production maps are overwhelmingly disposable caches that can be rebuilt cheaply, explicit migration should lose to a declared `reset-safe` policy.

### Direction 3: validate the new decoder on old state before granting write authority

**Gap.** A manifest can be stale or wrong, and a migration callback can emit structurally valid but semantically invalid values.

**Mechanism.** Before N+1 receives write authority, let it read a bounded snapshot or sample and evaluate declared invariants: counter ranges, identifier resolution, generation membership, cross-field constraints, and old-decoder versus new-decoder logical records. For active maps, bind the comparison to an epoch or snapshot boundary so normal concurrent updates are not mistaken for decoder disagreement.

A successful admission produces a reuse receipt binding the existing map identity, structural fingerprint, semantic revision, new build identity, and validation result.

**Delta from related work.** Metadata-only compatibility decides from declarations. Shadow validation adds evidence from the actual state about to be reused.

**Artifact.** A compatibility harness with map-type-specific sampling/snapshot adapters and invariant plugins.

**Evaluation.** Inject same-layout semantic bugs, stale revision declarations, corrupted entries, partial migrations, and concurrent updates. Compare manifest-only admission with shadow validation for silent bad reuses caught, false alarms, and startup cost.

**Academic value.** This asks how much sampled behavioral evidence is needed to distinguish representation compatibility from semantic compatibility.

**Production value.** High-value policy, accounting, and security state gets a final guard before a new writer can mutate it.

**Failure condition.** If an application has no cheap side-effect-free decoder or meaningful invariants, shadow validation should remain optional rather than becoming universal ceremony.

## A practical eBPF map reuse gate today

No new kernel API is required for a useful first version:

1. Classify pinned maps as `ephemeral`, `reset-safe`, `must-preserve`, or `migrate`.
2. Run the existing kernel-visible map-definition checks first.
3. Record a stable structural fingerprint with the artifact when BTF is available.
4. Record an explicit semantic revision for state whose meaning must survive.
5. Treat `bpf_map__reuse_fd()` as the action after admission, not the admission proof.
6. Refuse ambiguous `must-preserve` state rather than silently resetting or blindly reusing it.
7. Migrate into a new generation and keep the old generation until validation and rollback conditions are satisfied.
8. Log expected and observed schema evidence so an incident can reconstruct why reuse was allowed.

This separates two deployment questions:

```text
artifact + target kernel
    -> program/interface admission

artifact + existing map state
    -> representation/semantic admission

both pass
    -> attach and grant state authority
```

## What would change this conclusion?

The richer contract is unnecessary when the application intentionally treats a map value as an opaque fixed-size byte string and both generations use the same opaque protocol. A source-language wrapper change then has no behavioral meaning.

The case also weakens if production evidence shows that long-lived pinned maps are almost always either ABI-frozen or disposable caches that are safely rebuilt on every upgrade. Parameter checks plus an explicit reset policy could then be enough.

BTF fingerprints lose value if normal compiler/toolchain variation makes a canonical representation unstable in practice. In that case a source-generated schema manifest is a better artifact. Shadow validation loses value if real maps lack a consistent observation point or useful invariants.

The boundary is therefore specific: current libbpf checks can establish that an existing map has a compatible kernel-visible storage shape; BTF can expose much of its representation. Neither alone proves that the next application generation assigns the old bytes the same meaning. State that matters across upgrades needs an explicit bridge from **storage compatibility** to **structural schema** to **application semantics**.

## Sources

- [Linux kernel documentation: BPF maps](https://docs.kernel.org/bpf/maps.html)
- [Linux kernel documentation: BPF Type Format](https://docs.kernel.org/bpf/btf.html)
- [Linux kernel source: libbpf map reuse implementation](https://github.com/torvalds/linux/blob/master/tools/lib/bpf/libbpf.c)
- [libbpf API: `bpf_map__reuse_fd`](https://docs.ebpf.io/ebpf-library/libbpf/userspace/bpf_map__reuse_fd/)
- [Eunomia Daily Report: Can a Stateful eBPF Application Upgrade Atomically?](https://eunomia.dev/research/stateful-ebpf-transactional-upgrade/)
- [Eunomia Daily Report: Can an eBPF Loader Trust the Kernel Version?](https://eunomia.dev/research/ebpf-kernel-capability-evidence/)
- [Eunomia Daily Report: Can an eBPF Object Keep Its Meaning After a Kernel Upgrade?](https://eunomia.dev/research/ebpf-kernel-upgrade-semantic-compatibility/)
- [Eunomia Daily Report: Can an eBPF Loader Treat a kfunc as Just Present or Missing?](https://eunomia.dev/research/ebpf-kernel-interface-negotiation/)
- [eunomia-bpf organization](https://github.com/eunomia-bpf)
