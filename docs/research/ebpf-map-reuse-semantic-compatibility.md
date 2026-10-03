---
date: 2026-10-03
slug: ebpf-map-reuse-semantic-compatibility
title: "Does a Reused eBPF Map Still Mean the Same Thing?"
description: "libbpf can reuse a pinned eBPF map when its parameters match, yet equal sizes do not prove that a new program gives old state the same schema or meaning."
tags:
  - Daily Report
  - eBPF
  - Linux
  - libbpf
  - BTF
  - Compatibility
research_question: "When an eBPF application reuses a pinned map across software upgrades, what evidence should prove that old state still has the schema and semantics expected by the new program?"
source_cutoff: 2026-10-03
status: daily-report
---

# Does a Reused eBPF Map Still Mean the Same Thing?

A daemon upgrades from version N to N+1. Both versions define a pinned hash map with the same type, key size, value size, entry count, and flags. libbpf can accept the old map for reuse. The new program can load and every lookup can return the expected number of bytes.

That still does not prove that the bytes mean the same thing.

A field can move while the total value size stays constant. Two versions can also keep the exact same layout while changing a field from milliseconds to microseconds or from one identifier namespace to another. The kernel can see a compatible storage object while the application silently misinterprets old state.

This report is narrower than [transactional eBPF application upgrade](https://eunomia.dev/research/stateful-ebpf-transactional-upgrade/). That report covers prepare, migrate, commit, and retire across programs, links, maps, and controllers. Here the question is one admission decision: **before a new artifact receives authority over an existing map, what proves that direct reuse is safe?**

<!-- more -->

## Pinning preserves a kernel object, not an application schema

Linux `BPF_OBJ_PIN` retains a filesystem reference to a live BPF object, allowing it to outlive the file descriptor and process that created it. This is an object-lifetime mechanism. It does not say that a future producer or consumer assigns the same meaning to the key and value bytes.

Pinning is also not a serialization format. Cross-reboot durability requires a separate persistence and restore mechanism. The direct reuse problem therefore appears most clearly during controller restarts, daemon upgrades, and in-boot program generations.

## What eBPF map reuse proves today

Current upstream libbpf at commit `85c0d575a447651bbeb8e2376eacd64c30dce9a6` implements automatic pinned-map reuse through `map_is_reuse_compat()`. The check compares map type, key size, value size, `max_entries`, map flags, and `map_extra`. If those fields match, libbpf can hand the pinned file descriptor to the new object.

An independent loader reaches nearly the same boundary. `cilium/ebpf` at commit `4b8b24942ca3e53a1aa6c9152d8ab0fade7dc3be` calls `MapSpec.Compatible()` before reusing a pinned map and compares type, key size, value size, `max_entries`, and flags. These checks are necessary because they reject obvious storage-definition mismatches. Neither loader treats the existing entries' application meaning as part of that compatibility verdict.

Linux exposes stronger structural evidence when BTF is available. At Linux commit `8623551a424d34a311bab3fb9243535c98e08c26`, `bpf_map_info` includes `btf_id`, `btf_key_type_id`, and `btf_value_type_id`, so tooling can retrieve the associated BTF and inspect the reachable type graph. A numeric BTF type ID, however, is scoped to one BTF object rather than serving as a stable application schema version across builds.

Even a stable structural comparison is incomplete. BTF can show that two layouts are identical; it cannot infer that a field changed units, namespace, validity rules, or application meaning.

The state contract therefore has two layers: **structural schema**, covering map parameters and representation, and **semantic schema**, covering units, identifier namespaces, epochs, validity rules, ownership, and application invariants.

`bpf_map__reuse_fd()` reinforces the separation. It selects an existing map for a BPF object; it does not certify that the old state is correct for the new artifact.

Production experience also shows that migration is its own lifecycle surface. Cilium issue #24013 documents an interrupted bpffs map migration that left a stale `:pending` map and blocked a later installation. That bug is not semantic-schema drift, but it shows that migration, naming, rollback, and cleanup can fail independently of verifier acceptance.

## Where current work is still weak

### Equal parameters can admit the wrong representation

A loader can satisfy every field checked by `map_is_reuse_compat()` while key or value layouts differ. A useful mechanism should be tested with size-preserving mutations: reorder equal-width fields, alter nested structs, move bitfields, or change enum representation while keeping the kernel-visible map definition constant.

### Structural identity cannot prove semantic identity

A field can change from PID to cgroup ID with no representation change. A timeout can change units. A cache entry can remain readable while becoming invalid under a new algorithm. Evaluation must therefore include same-layout semantic mutations, not only layout drift.

### Migration needs a concurrency boundary

If direct reuse is rejected, converting a live map is not like converting an offline file. Programs can update entries during userspace iteration, LRU maps can evict entries, and per-CPU maps expose multiple value slots. A migration protocol needs an old/new generation cut and rollback rule.

## Promising directions with academic and production value

### 1. Canonical BTF-derived structural fingerprints

**Mechanism.** Canonicalize the reachable key/value BTF graph and hash representation-relevant facts such as kind, size, member name and offset, integer encoding, array length, enum representation, and nested fingerprints. Ignore BTF-local numeric IDs.

**Artifact and evaluation.** Build a `map-schema` tool that compares an ELF object with an existing map and emits a machine verdict plus human-readable diff. A prototype could also extend the [eunomia-bpf package metadata](https://github.com/eunomia-bpf/eunomia-bpf) with an explicit persistent-state schema field. Test it across compilers, BTF deduplication, nested types, padding, and deliberate size-preserving mutations. Measure digest stability, false accepts, false rejects, and startup cost.

**Failure condition.** If ordinary toolchain rebuilds cannot produce stable canonical schemas or deployed maps lack sufficient BTF, use a source-generated schema manifest instead of making BTF a mandatory gate.

### 2. Versioned state contracts with explicit migration policy

**Mechanism.** Give each state-bearing map a logical identity, structural fingerprint, semantic revision, lifecycle class, allowed migration sources, and reset policy. The loader classifies old state into `direct-reuse`, `migrate-before-write`, `explicit-reset`, or `refuse`.

If migration is required, create a new-generation map, gate writers, transform and validate state, switch authority, and retire the old map only after acceptance.

**Artifact and evaluation.** Implement a versioned map manifest plus migration runner. Inject crashes at each phase, concurrent updates, LRU eviction, per-CPU state, and memory pressure. Measure lost or duplicated updates, rollback success, downtime, and how many migrations can be generated safely.

**Failure condition.** If a workload treats almost all pinned maps as cheap disposable caches, migration can cost more than reset. The contract should permit `reset-safe`.

### 3. Shadow validation before granting write authority

**Mechanism.** Metadata can be wrong. Before N+1 writes old state, let it decode a bounded sample or snapshot and compare application invariants or old/new logical records. A successful check yields a reuse receipt binding map identity, structural fingerprint, semantic revision, new build identity, and validation result.

**Artifact and evaluation.** Build a state-compatibility harness and inject same-layout semantic bugs, stale declarations, corrupted entries, partial migrations, and concurrent updates. Compare manifest-only versus shadow admission for silent bad reuses caught, false alarms, coverage, and startup delay.

**Failure condition.** If the application has no cheap side-effect-free decoder or useful invariant, shadow validation should remain a risk-based option for `must-preserve` state.

## Practical deployment guidance today

No new kernel API is required. A loader can classify every pinned map as `ephemeral`, `reset-safe`, `must-preserve`, or `migrate`; run the existing parameter gate; attach a structural fingerprint and semantic revision to the artifact; and refuse ambiguous `must-preserve` state.

Treat `bpf_map__reuse_fd()` as the mechanism used after compatibility has been established, not as the verdict itself. If migration is required, keep the old generation until validation and cutover succeed.

## What would change this conclusion?

A richer contract would be unnecessary if production evidence showed that long-lived pinned maps are almost always either permanently ABI-stable or disposable and explicitly reset on every upgrade.

The BTF fingerprint idea weakens if canonical schemas are unstable across normal toolchain rebuilds or map type information is routinely unavailable. Shadow validation weakens if workloads have no consistent, low-cost observation point and no invariants strong enough to distinguish correct from incorrect decoding.

There is also a direct counterexample: if both generations intentionally treat a value as an opaque fixed-size byte string and no consumer depends on its internal layout, a wrapper-struct change is irrelevant. Compatibility should follow the real consumer contract, not type spelling.

Current Linux/libbpf leaves a clear boundary. Kernel-visible map parameters can establish compatible storage shape, and BTF can expose much of the representation. Neither proves that a new application generation gives old bytes the same meaning. Safe direct reuse therefore needs **map-definition compatibility, stable structural evidence, and an explicit semantic contract**, with migration or refusal when those layers disagree.

## Sources

- [Linux kernel documentation: eBPF syscall and object pinning](https://docs.kernel.org/userspace-api/ebpf/syscall.html)
- [Linux kernel documentation: BPF Type Format](https://docs.kernel.org/bpf/btf.html)
- [libbpf at 85c0d575: pinned-map reuse compatibility](https://github.com/libbpf/libbpf/blob/85c0d575a447651bbeb8e2376eacd64c30dce9a6/src/libbpf.c)
- [Linux at 8623551a: `bpf_map_info` BTF fields](https://github.com/torvalds/linux/blob/8623551a424d34a311bab3fb9243535c98e08c26/include/uapi/linux/bpf.h)
- [`cilium/ebpf` at 4b8b2494: `MapSpec.Compatible`](https://github.com/cilium/ebpf/blob/4b8b24942ca3e53a1aa6c9152d8ab0fade7dc3be/map.go)
- [Cilium issue #24013: stale state during bpffs map migration](https://github.com/cilium/cilium/issues/24013)
- [Eunomia Daily Report: Can a Stateful eBPF Application Upgrade Atomically?](https://eunomia.dev/research/stateful-ebpf-transactional-upgrade/)
