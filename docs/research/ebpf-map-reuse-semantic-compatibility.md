---
date: 2026-09-29
slug: ebpf-map-reuse-semantic-compatibility
title: "Does a Reused eBPF Map Still Mean the Same Thing?"
description: "libbpf checks pinned-map parameters before reuse, but equal sizes do not prove that a new eBPF program interprets old map state with the same schema or meaning."
tags:
  - Daily Report
  - eBPF
  - Linux
  - libbpf
  - BTF
  - Compatibility
research_question: "When an eBPF application reuses a pinned map across upgrades, what evidence should prove that old state still has the schema and meaning expected by the new program?"
source_cutoff: 2026-09-29
status: daily-report
---

# Does a Reused eBPF Map Still Mean the Same Thing?

An eBPF daemon upgrades from N to N+1. Both versions define a pinned hash map with the same type, key size, value size, entry count, and flags. libbpf accepts the existing object for reuse, the new program loads, and every lookup returns the expected number of bytes.

That does not prove the bytes mean the same thing.

N can store `last_seen_ns`, a policy generation, and counters while N+1 keeps the same total size but reorders fields, changes a unit, or reuses an integer for another identifier namespace. The kernel object is still usable, yet the application can silently decode old state under a new schema.

This is narrower than [transactional eBPF upgrade](https://eunomia.dev/research/stateful-ebpf-transactional-upgrade/). That report covers prepare, migrate, commit, and retire across an application. Here the question is one admission decision: **may this new artifact inherit this existing map?** It extends the current series after [capability evidence](https://eunomia.dev/research/ebpf-kernel-capability-evidence/), [cross-kernel semantics](https://eunomia.dev/research/ebpf-kernel-upgrade-semantic-compatibility/), and [interface negotiation](https://eunomia.dev/research/ebpf-kernel-interface-negotiation/).

<!-- more -->

## Pinning preserves an object, not an application schema

A bpffs pin keeps a reference to a live BPF object after the creating process exits. Another process can reopen the same map and continue using its contents.

That is an object-lifetime guarantee, not a schema guarantee:

```text
object lifetime:        can another process still open this map?
state compatibility:    will new code interpret the old keys and values correctly?
```

The first can be true while the second is false. Pinning also is not host-reboot durability: the old kernel objects disappear at reboot unless another mechanism serializes and restores state.

## What libbpf currently checks

Current libbpf's automatic pinned-map reuse path uses `map_is_reuse_compat()`. At the 2026-09-29 source cutoff, it compares the existing map's type, key size, value size, `max_entries`, map flags, and `map_extra` with the new definition, with map-specific normalization where needed.

These checks are necessary. They reject obvious storage mismatches. They do not describe field-level representation.

```c
/* N */
struct flow_state {
    __u64 last_seen_ns;
    __u32 generation;
    __u32 verdict;
    __u64 bytes;
};

/* N+1: same total size, different interpretation */
struct flow_state {
    __u64 bytes;
    __u32 verdict;
    __u32 generation;
    __u64 last_seen_ns;
};
```

A parameter-only check cannot distinguish these layouts if the sizes and other map attributes match. This is not a libbpf bug. The missing contract is application-level state compatibility.

## BTF adds structure, not meaning

Linux can attach BTF key and value types to a map, and `bpf_map_info` exposes the associated BTF IDs. Tooling can retrieve the BTF graph and compare member offsets, widths, nested types, arrays, and enums.

Raw BTF type IDs are not stable schema versions because they are local to one BTF object. A rebuild can renumber types without changing layout.

More importantly, identical layout can still carry different semantics. A `__u64 tokens` field can change from whole tokens to milli-tokens with no BTF change. Persistent-state compatibility therefore has two layers:

1. **Structural schema:** map parameters and the reachable key/value type graph.
2. **Semantic schema:** units, identifier namespaces, epochs, valid ranges, ownership, and other application invariants.

The explicit `bpf_map__reuse_fd()` API reinforces this separation. It selects an existing object; it does not prove that reusing that object is correct.

## Where current practice is still weak

### Equal parameters can admit a wrong schema

Same-sized field reordering, nested-layout changes, or key-composition changes can preserve all parameters checked by automatic reuse. A discriminating test should generate such mutations and measure silent wrong-state acceptance, not merely load failures.

### Structural compatibility is not semantic compatibility

BTF cannot infer a unit, namespace, generation policy, or cache-validity rule. Any proposal that only catches layout drift solves ABI accidents, not semantic drift. Evaluation therefore needs deliberate same-layout semantic mutations.

### Migration needs a consistency boundary

If direct reuse is unsafe, entry conversion alone is insufficient. BPF programs may update maps while userspace iterates them, LRU maps may evict, and per-CPU maps contain multiple value slots. Migration needs an explicit generation cut and rollback rule rather than a best-effort copy.

## Promising directions with academic and production value

### Direction 1: canonical BTF schema fingerprints

**Gap.** Parameter equality misses layout drift, while raw BTF IDs are local identifiers.

**Mechanism.** Canonicalize the reachable key/value BTF graph using kind, size, member offsets, integer encoding, array length, enum representation, and nested fingerprints. Ignore build-local IDs, then hash the canonical form. Support exact identity plus explicitly declared compatible evolution.

**Artifact.** A libbpf-adjacent `map-schema` tool that compares an ELF definition with a live map, prints a structural diff, and returns a machine-readable verdict.

**Evaluation.** Mutate nested structs, arrays, enums, padding, compiler versions, and BTF deduplication. Measure false accepts, false rejects, digest stability, and startup cost.

**Academic and production value.** This defines a persistent-state ABI across independent builds and gives operators an explainable reason for refusing reuse.

**Failure condition.** If fingerprints are unstable across normal rebuilds or real deployments lack usable BTF, a source-generated manifest is a better mandatory artifact.

### Direction 2: explicit semantic revisions and migration contracts

**Gap.** Structural identity cannot encode units, namespaces, epochs, or algorithm invariants.

**Mechanism.** Give each state-bearing map a logical identity, structural fingerprint, semantic revision, lifecycle class, allowed migration sources, and reset policy. Startup classifies state as direct reuse, migrate-before-write, explicit reset, or refusal. Migration writes a new-generation map, validates it, switches ownership, and keeps the old generation for rollback.

**Artifact.** A versioned map manifest plus a migration runner with generated adapters for structural changes and application callbacks for semantic changes.

**Evaluation.** Test rolling upgrades with concurrent writes, injected crashes, LRU eviction, per-CPU values, and memory pressure. Measure lost or duplicated updates, rollback success, downtime, and automatically generated migrations.

**Academic and production value.** The mechanism composes static schema evidence with failure-atomic state evolution while giving operators an explicit policy instead of accidental reuse.

**Failure condition.** If a map is a cheap disposable cache, migration can cost more than reset. The contract must allow `reset-safe` state.

### Direction 3: shadow validation before write authority

**Gap.** Manifests and migration callbacks can be wrong even when their metadata looks valid.

**Mechanism.** Before N+1 can write, let it decode a bounded sample or snapshot of N's state and check declared invariants: ranges, identifier resolution, generation membership, cross-field relations, and old-versus-new decoder parity. Bind validation to an epoch so concurrent updates are not mistaken for decoder disagreement.

**Artifact.** A state-compatibility harness with map-type sampling adapters, invariant plugins, and a reuse receipt that records the evidence used for admission.

**Evaluation.** Inject same-layout semantic bugs, stale revisions, corrupted entries, partial migrations, and concurrent updates. Compare manifest-only admission with shadow validation on silent errors caught, false alarms, and startup latency.

**Academic and production value.** This studies when bounded runtime evidence can compensate for incomplete semantic specifications and provides a practical gate for policy, security, or accounting state.

**Failure condition.** If the application has no cheap side-effect-free decoder or useful invariants, shadow validation should remain optional.

## Practical deployment guidance

A loader can apply the boundary today:

1. classify pinned maps as `ephemeral`, `reset-safe`, `must-preserve`, or `migrate`;
2. keep libbpf's kernel-visible definition checks as the first gate;
3. add a stable structural fingerprint when BTF is available;
4. declare an application semantic revision separately;
5. call `bpf_map__reuse_fd()` only after compatibility is established;
6. migrate into a new generation when direct reuse is unsafe;
7. retain enough evidence to explain why a map was reused, migrated, reset, or refused.

The same split is useful for tooling such as [eunomia-bpf](https://github.com/eunomia-bpf/eunomia-bpf):

```text
artifact + target kernel       -> interface/capability admission
artifact + existing map state  -> representation/semantic admission
both pass                      -> grant state authority
```

## What would change this conclusion?

The richer contract is unnecessary if production evidence shows that persistent maps are almost always either ABI-stable structures or disposable caches safely reset at every upgrade. Parameter equality plus explicit reset could then be enough.

BTF fingerprints also lose value if canonicalization is unstable across ordinary toolchains or deployed applications routinely omit usable key/value type information. A source-generated schema manifest would be more practical in that environment.

Shadow validation fails if real maps provide no consistent, low-cost observation point and application invariants are too weak to distinguish correct decoding from wrong decoding.

There is also a valid counterexample: if both generations intentionally treat a value as an opaque fixed-size byte string and no consumer assigns meaning to its internal source-language layout, a wrapper-struct change is irrelevant.

The current Linux/libbpf boundary is therefore specific: map parameters can establish storage-shape compatibility and BTF can expose structural representation, but neither proves that a new application assigns the same meaning to old state. Safe reuse needs **map-definition compatibility, stable structural evidence, and an explicit semantic contract**, with migration or refusal when they disagree.

## Sources

- [Linux kernel documentation: BPF maps](https://docs.kernel.org/bpf/maps.html)
- [Linux kernel documentation: BPF Type Format](https://docs.kernel.org/bpf/btf.html)
- [Linux kernel source: libbpf map reuse implementation](https://github.com/torvalds/linux/blob/master/tools/lib/bpf/libbpf.c)
- [eBPF Docs: `bpf_map__reuse_fd`](https://docs.ebpf.io/ebpf-library/libbpf/userspace/bpf_map__reuse_fd/)
- [Eunomia: Stateful eBPF Application Upgrade](https://eunomia.dev/research/stateful-ebpf-transactional-upgrade/)
