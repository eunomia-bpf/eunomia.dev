---
date: 2026-09-25
slug: ebpf-map-reuse-semantic-compatibility
title: "Does a Reused eBPF Map Still Mean the Same Thing?"
description: "Equal eBPF map parameters can permit reuse even when a new program interprets old state with a different schema or meaning."
tags:
  - Daily Report
  - eBPF
  - Linux
  - libbpf
  - BTF
research_question: "What evidence should prove that existing pinned-map state is safe for a new eBPF application generation to reuse?"
source_cutoff: 2026-09-25
status: daily-report
---

# Does a Reused eBPF Map Still Mean the Same Thing?

Suppose version N and N+1 both define a pinned hash map with the same map type, key size, value size, entry count, and flags. The new loader can reopen the old map and the new program can load. That proves the kernel object is mechanically reusable. It does not prove that N+1 gives the old bytes the same meaning.

A 32-byte value can keep the same size while equal-width fields move, an identifier changes namespace, or a counter changes units. The failure is especially dangerous because every lookup still succeeds.

This report is narrower than [transactional eBPF application upgrade](https://eunomia.dev/research/stateful-ebpf-transactional-upgrade/). That report covered whole-application cutover. Here the question is the admission decision for one existing map: **when is direct state reuse legal?** It extends the deployment-compatibility sequence after [capability evidence](https://eunomia.dev/research/ebpf-kernel-capability-evidence/), [cross-kernel semantic compatibility](https://eunomia.dev/research/ebpf-kernel-upgrade-semantic-compatibility/), and [typed interface negotiation](https://eunomia.dev/research/ebpf-kernel-interface-negotiation/).

<!-- more -->

## What current mechanisms prove

BPF_OBJ_PIN keeps a filesystem reference to an in-kernel BPF object so its lifetime can extend beyond the creating process. BPF_OBJ_GET lets another process obtain a file descriptor for that object. This is an object-lifetime guarantee, not an application-schema guarantee.

Current libbpf makes the distinction visible. Its automatic pinned-map reuse path calls map_is_reuse_compat(), reads bpf_map_info, and compares map type, key size, value size, max entries, map flags, and map_extra. Those checks correctly reject many incompatible objects.

But all of those fields can remain equal while the value layout changes. A map whose value used to mean {timestamp, generation, verdict, bytes} can keep the same total size after those fields are reordered. Kernel-visible compatibility does not detect that semantic mismatch.

The explicit bpf_map__reuse_fd() interface is even more mechanical: current libbpf reads information from the supplied fd and adopts the existing map's definition and BTF key/value type IDs into the libbpf map object. It chooses an existing object; it is not a verdict that the state belongs to the new artifact.

BTF provides stronger evidence. bpf_map_info can expose btf_id, btf_key_type_id, and btf_value_type_id, and the BTF blob describes member offsets, integer encodings, arrays, structs, unions, enums, and bitfields. Yet numeric BTF type IDs are local to one BTF object. They are not stable cross-build schema versions.

Even an identical BTF layout cannot encode every application invariant. A 64-bit field can stay structurally identical while its unit changes from requests to milli-requests. An integer can change from PID to cgroup ID. Structural identity is therefore evidence, not the whole contract.

A useful reuse decision needs three layers: **kernel-visible map definition, structural schema, and application semantic schema**.

## Where current work is still weak

### Equal parameters can hide a wrong decoder

A loader can accept a map even when key/value structures change while preserving every field checked by map_is_reuse_compat(). A useful benchmark should mutate equal-width fields, nested structs, enums, bitfields, and key composition while holding the checked map parameters constant. The metric is silent wrong reuse, not load success.

### Type compatibility cannot prove meaning

BTF can catch many representation changes, but not same-layout changes in units, identifier domains, policy epochs, or ownership. Production state therefore needs an explicit semantic revision or invariant set. Otherwise a perfect type checker can still admit a wrong policy or accounting interpretation.

### Migration needs a consistency cut

Once direct reuse is rejected, migration is not an offline file conversion. BPF programs can update a map while userspace iterates it; LRU maps can evict; per-CPU maps have multiple value slots. Migration needs a generation or quiescence boundary, or it can combine entries from incompatible logical epochs.

## Promising directions with academic and production value

### Direction 1: BTF-derived structural fingerprints

**Gap.** Map parameters miss layout changes, while raw BTF IDs are unstable identifiers.

**Mechanism.** Canonicalize the reachable key/value BTF graph into a digest containing type kind, resolved size, member name and bit offset, integer encoding, arrays, enums, and nested structural digests. Ignore BTF-local numbering and build artifacts that do not change representation. Keep a human-readable diff beside the digest.

**Delta.** This is stronger than map-definition equality but deliberately stops short of claiming application semantics.

**Artifact.** A libbpf-adjacent map-schema tool that compares an ELF object's expected schema with a live map and explains mismatches.

**Evaluation.** Rebuild a mutation corpus across compiler versions, padding, typedef-only edits, field reorderings, nested types, arrays, enums, and BTF deduplication. Measure false accepts, false rejects, digest stability, and startup cost.

**Academic value.** It defines cross-build representation compatibility over type metadata whose local identifiers are unstable.

**Production value.** A loader can explain exactly why reuse was rejected before state is corrupted.

**Failure condition.** If normal toolchain changes destabilize the digest or production maps usually lack usable BTF, a source-generated schema manifest is a better fallback.

### Direction 2: version state separately from the bpffs path

**Gap.** Structural equality cannot express units, ownership, reset safety, or semantic epochs.

**Mechanism.** Give every state-bearing map a logical contract: map identity, structural fingerprint, semantic revision, lifecycle class, accepted predecessor revisions, migration path, and reset policy. Startup chooses one explicit outcome: direct reuse, migrate, policy-approved reset, or refusal.

If migration is required, create a new map generation. Gate or quiesce writers, transform entries, validate invariants, switch consumers, and keep the old generation until rollback is no longer needed.

**Delta.** Whole-application transactional upgrade supplies the cutover protocol; this mechanism supplies the evidence that determines which maps can bypass migration.

**Artifact.** A versioned state manifest and migration runner with generated structural adapters plus explicit callbacks for semantic changes.

**Evaluation.** Test rolling upgrades with concurrent writes, crashes at every phase, LRU eviction, per-CPU maps, and memory pressure. Measure lost/duplicated logical updates, rollback success, downtime, and migration coverage.

**Academic value.** State compatibility becomes a compositional upgrade property rather than a convention attached to a pathname.

**Production value.** Must-preserve maps get explicit behavior while disposable caches remain reset-safe.

**Failure condition.** If most pinned state is cheaply reconstructed, migration machinery may cost more than resetting it.

### Direction 3: shadow validation before write authority

**Gap.** A manifest can be wrong even when its version numbers match.

**Mechanism.** Before N+1 can write, let it decode a bounded sample or consistent snapshot of existing state and check declared invariants. Old and new decoders can normalize the same raw entries and compare results. Active maps need an epoch or snapshot boundary so concurrent updates are not mistaken for decoder disagreements.

A passing check produces a reuse receipt binding map identity, structural fingerprint, semantic revision, new artifact identity, and validation result.

**Delta.** Static metadata says what should be compatible; shadow validation adds evidence from the actual state about to be reused.

**Artifact.** A state-compatibility harness with map-specific sampling adapters and invariant plugins.

**Evaluation.** Inject same-layout semantic bugs, stale declarations, corrupt entries, partial migrations, and concurrent updates. Compare manifest-only and shadow admission for silent bad-reuse detection, false alarms, and startup delay.

**Academic value.** It tests how much runtime evidence is needed when type evidence is incomplete.

**Production value.** High-value maps gain a last gate before a new version can mutate them.

**Failure condition.** If applications lack cheap side-effect-free decoders or meaningful invariants, this gate should remain risk-based rather than universal.

## What can a loader do today?

No kernel change is required to separate these decisions. A loader can classify pinned maps as ephemeral, reset-safe, must-preserve, or migrate; run the existing map-definition check; compare a structural schema when BTF is available; require an explicit semantic revision for must-preserve state; treat bpf_map__reuse_fd() as the mechanism after compatibility has been established; and migrate or refuse when evidence is missing.

The useful mental model is two independent admissions:

- artifact plus target kernel -> capability/interface admission;
- artifact plus existing state -> representation/semantic admission.

Successful loading in the first path should not imply success in the second.

## What would change this conclusion?

The richer contract is unnecessary when old state is deliberately disposable and cheaply rebuilt. It is also less useful when all consumers intentionally treat values as opaque fixed-size byte strings.

A BTF fingerprint is the wrong primitive if ordinary compiler changes make it unstable or deployed maps usually lack enough BTF. Shadow validation is not worth its cost if real applications have no strong invariants or consistent observation points.

But when state must survive an application upgrade, current parameter checks answer only whether the storage object can be reused mechanically. BTF can provide structural evidence, yet neither storage shape nor type layout alone proves application meaning. A safer boundary combines **map-definition compatibility, stable structural schema evidence, and an explicit semantic contract**, with migration, reset, or refusal when those layers disagree.

## Sources

- [Linux kernel documentation: BPF syscall](https://docs.kernel.org/userspace-api/ebpf/syscall.html)
- [Linux kernel documentation: BPF maps](https://docs.kernel.org/bpf/maps.html)
- [Linux kernel documentation: BPF Type Format](https://docs.kernel.org/bpf/btf.html)
- [Linux kernel source: libbpf map reuse implementation](https://github.com/torvalds/linux/blob/master/tools/lib/bpf/libbpf.c)
- [eBPF Docs: bpf_map__reuse_fd](https://docs.ebpf.io/ebpf-library/libbpf/userspace/bpf_map__reuse_fd/)
- [Eunomia Daily Report: Can a Stateful eBPF Application Upgrade Atomically?](https://eunomia.dev/research/stateful-ebpf-transactional-upgrade/)
