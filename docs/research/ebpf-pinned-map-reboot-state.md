---
date: 2026-09-23
slug: ebpf-pinned-map-reboot-state
title: "Can a Pinned eBPF Map Survive a Host Reboot?"
description: "Pinned eBPF maps survive process restarts, not host reboots. This report defines state contracts, consistent checkpoints, and restore gates for reboot-safe recovery."
tags:
  - Daily Report
  - eBPF
  - Linux
  - BPF Maps
  - State Recovery
  - BTF
research_question: "What state contract is needed to preserve or reconstruct an eBPF application's map state across a host reboot when bpffs pinning only preserves a live kernel object reference?"
source_cutoff: 2026-09-23
status: daily-report
---

# Can a Pinned eBPF Map Survive a Host Reboot?

A daemon creates an eBPF map, fills it with policy state, pins it under `/sys/fs/bpf`, and exits. A replacement daemon can reopen the same map and continue. That makes the map look persistent.

Then the host reboots.

The old map is gone. The path may exist again after bpffs is mounted, but bpffs pinning is a reference-lifetime mechanism for live kernel objects, not a durable storage format. The [eBPF pinning documentation](https://docs.ebpf.io/linux/concepts/pinning/) states the boundary directly: pins are useful across process restarts, but are ephemeral across a system restart.

That distinction creates a deployment problem for stateful eBPF applications. A controller restart, a kernel reboot, a host replacement, and an application upgrade are often grouped under "recovery," even though they preserve very different pieces of state. A pin can bridge the first case. It cannot reconstruct a map after the kernel that owned the map has disappeared.

The harder question is therefore not how to reopen a pin. It is how an eBPF application should decide which state must be checkpointed, which state should be rebuilt from another source of truth, which state must be reset, and how to prove that the restored state still means the same thing on the new kernel and host.

This report argues for an explicit **reboot state contract**. The contract should separate kernel-visible map compatibility from application semantics, define a consistency boundary for checkpoints, and make restoration a staged promotion rather than a best-effort byte replay.

This is narrower than the earlier [transactional eBPF upgrade report](https://eunomia.dev/research/stateful-ebpf-transactional-upgrade/). That report keeps an old live object graph available while a new generation is prepared and activated. Here the reboot destroys the old kernel objects. It is also distinct from [cross-kernel semantic compatibility](https://eunomia.dev/research/ebpf-kernel-upgrade-semantic-compatibility/), which asks whether an artifact behaves the same on two kernels. Reboot recovery first has to establish that the reconstructed **input state** is valid at all.

<!-- more -->

## Pinning extends object lifetime, not storage lifetime

BPF objects are reference counted. `BPF_OBJ_PIN` creates a filesystem reference in bpffs so a program, map, link, or BTF object can stay alive after the process that created it closes its file descriptor. Another process can later use `BPF_OBJ_GET` to obtain a new descriptor for the same live object.

That is exactly why pinning is useful for controller restarts and cross-process sharing. The lifetime relation is roughly:

```text
userspace process lifetime
        < pinned BPF object lifetime
        <= kernel instance lifetime
```

A reboot crosses the last boundary. Mounting bpffs again recreates the filesystem used to name BPF objects, not the old kernel objects themselves. A recovery design that treats `/sys/fs/bpf/foo` as durable state is confusing an object reference with the object contents.

The distinction matters in production because some BPF maps are close to caches while others are authoritative policy or accounting state. Restarting with an empty cache may be fine. Restarting with an empty authorization map can create an outage or, depending on fallback behavior, weaken policy. A single "persistent map" flag cannot express those differences.

## Structural compatibility is not application-state compatibility

A live pinned map can be reused when the new loader expects a compatible kernel object. libbpf exposes APIs such as `bpf_map__reuse_fd()`, and the kernel exposes map metadata through `BPF_OBJ_GET_INFO_BY_FD`. Current [Linux BTF documentation](https://docs.kernel.org/bpf/btf.html) also lets a map expose BTF identifiers for its key and value types.

Those mechanisms answer an important question: can this userspace loader and BPF object safely bind to this live kernel map?

They do not answer another question: do the stored values still mean what the application thinks they mean?

Consider two values with the same binary layout:

```c
struct policy_state_v1 {
    __u32 verdict;
    __u32 generation;
};

struct policy_state_v2 {
    __u32 verdict;
    __u32 lease_seconds;
};
```

Both are eight bytes. A structural check can accept the size. BTF can describe the field names and types, but an application upgrade could also preserve the same field shape while changing a field's interpretation, units, validity epoch, or ownership rule. Raw BTF type IDs are scoped to a BTF object, so they are also poor durable identifiers by themselves.

A reboot image therefore needs two layers of compatibility evidence:

```text
kernel-visible compatibility
  map type + sizes + flags + target support + structural BTF shape

application-state compatibility
  semantic schema + validity epoch + ownership + migration/rebuild rule
```

The target kernel and verifier remain authoritative for whether the recreated BPF objects are legal. The application owns the second layer because the kernel cannot infer whether `generation = 7` and `lease_seconds = 7` are semantically interchangeable.

## A map dump is not automatically a checkpoint

Even for an ordinary hash map, copying every key does not necessarily produce a state that existed at one logical instant.

Linux documents that BPF hash-map values can be accessed concurrently from different CPUs. Userspace can iterate maps or use batch lookup APIs, but those interfaces do not define a transaction spanning many keys, many maps, and controller-side state. The [hash-map documentation](https://docs.kernel.org/bpf/map_hash.html) also notes concurrency-sensitive behavior around iteration and deletion.

Suppose a policy application stores:

```text
map A: active policy generation = 42
map B: principals allowed by generation 42
```

A checkpoint process reads A. The controller then installs generation 43 in both maps. The checkpoint process reads B afterward. Every individual read can succeed, yet the resulting file combines state from two generations.

Per-value `bpf_spin_lock` does not solve this. It protects a value, not a logical transaction across an application state graph. A faster batch lookup does not solve it either. Throughput and consistency are different properties.

This leads to a more useful checkpoint question:

> Which application invariants must hold across the restored set, and what protocol proves that the exported state belongs to one declared recovery cut?

For telemetry counters, "approximately recent" may be a sufficient answer. For policy, ownership, billing, replay protection, or recovery cursors, it usually is not.

## Some maps should be rebuilt, not serialized

BPF maps do not all represent portable key-value state. [Map-of-maps](https://docs.kernel.org/bpf/map_of_maps.html) contains references to other live map objects. Per-CPU maps make state depend on CPU topology. LRU maps deliberately embed eviction policy. Other map types can contain socket, program, queue, ring-buffer, or kernel-managed relationships that do not make sense as opaque bytes after reboot.

A practical reboot policy should therefore classify each map:

| Strategy | When it fits | Reboot action |
| --- | --- | --- |
| Checkpoint | Durable learned or policy state has no better authority | Restore through a versioned state image and validation gate |
| Reconstruct | A controller DB, config, routing table, or other system is authoritative | Rebuild from that source and verify convergence |
| Reset | Cache, telemetry, transient queue, boot-local epoch state | Start empty and record the reset |
| Reject restore | Old state cannot be safely mapped to the new kernel/topology/schema | Fail closed or use an explicit fallback |

This classification is more useful than asking whether a map is "persistent." It also makes failure behavior reviewable before maintenance begins.

A February 2026 [Cilium issue](https://github.com/cilium/cilium/issues/44277) shows why lifecycle state is already operationally significant even without a reboot. Endpoint regeneration entered a recovery loop because a global BPF-map pin already existed at the path where a map was being committed. The failure was about object identity, pin namespace, and controller recovery ordering, not verifier safety. Reboot recovery adds another transition where object identities disappear completely while the control plane still needs to reconstruct a coherent state.

## Where current work is still weak

### The persistence boundary is rarely machine-readable

Linux pinning precisely defines object lifetime, but application manifests usually do not say whether a map must survive a controller restart, kernel reboot, host replacement, or only one deployment generation. Operators then infer semantics from a pin path or a loader option.

The missing piece is a per-map lifecycle declaration that names the survival boundary, durable source of truth, and fallback when recovery evidence is missing. This matters because two pinned maps in the same object may require opposite reboot behavior.

A useful falsification test is to inspect production eBPF applications. If their existing map declarations already encode restart boundary, recovery source, semantic schema, and validation policy in a reusable form, another contract adds little value.

### Structural type evidence does not prove semantic state compatibility

BTF and map metadata can describe structure. They cannot establish application meaning, time validity, authority, or migration semantics. Equal-sized state is a particularly dangerous case because a simple compatibility test can accept it silently.

The missing piece is an application-owned semantic state version tied to explicit migration, reconstruction, or reset rules.

The discriminating test is equal-layout semantic drift. Change a field's meaning without changing its byte size. A correct restore mechanism must reject or migrate the old image even though a structural loader check still succeeds.

### Live export lacks a general application-level consistency cut

Map iteration and batch operations make extraction possible, but they do not define a transaction across concurrently changing BPF maps and userspace state.

The missing piece is a quiescence, epoch, copy-on-write, or delta protocol that tells the recovery system what one checkpoint means.

The test should inject writes throughout the checkpoint. If an implementation is only correct after pausing all mutation, that is a valid design, but it should advertise a stop-the-world checkpoint instead of implying a live consistent snapshot.

### Restore success is usually easier to test than restore correctness

A loader can recreate maps, insert entries, load programs, and attach successfully while the resulting policy or ownership state is wrong.

The missing piece is a post-restore semantic gate with workload-specific invariants and a fallback path.

The metric that matters is **silent bad recovery**. Restore latency matters too, but a fast recovery that accepts stale authorization or an impossible state graph is worse than an explicit failure.

## Promising directions with academic and production value

### 1. A reboot state contract beside each BPF artifact

**Gap.** Map definitions describe how to create kernel objects. They do not describe how application state should cross a reboot after those objects vanish.

**Mechanism.** Add a small manifest beside each artifact. For every stateful map, declare the survival boundary, recovery strategy, semantic schema, structural fingerprint, consistency requirement, source of truth, and validation gate:

```yaml
map: policy_cache
survival: host-reboot
strategy: checkpoint
semantic_schema: policy-state/v3
structural_schema: sha256:<canonical-btf-shape>
consistency: generation-cut
source_of_truth: controller-db
restore_gate: policy-canary/v2
```

The structural fingerprint should be computed from a canonicalized BTF shape instead of raw type IDs. The semantic schema is application-owned and changes whenever meaning changes, even when the C layout does not.

The loader combines this manifest with the target-kernel capability evidence developed in the earlier [capability report](https://eunomia.dev/research/ebpf-kernel-capability-evidence/). A `reset` map starts empty. A `reconstruct` map is rebuilt from the named authority. A `checkpoint` map requires compatible state evidence. A restore with missing or ambiguous evidence takes the declared fallback rather than guessing.

**Delta.** Existing object metadata mainly describes the kernel object and loader expectations. This contract describes state after the old object no longer exists.

**Artifact.** A libbpf-compatible manifest library, canonical BTF-shape fingerprinting, and adapters for hash, array, per-CPU, and map-of-maps state.

**Evaluation.** Use maps from networking, tracing, policy, and profiling applications. Inject structural drift, equal-size semantic drift, CPU-topology changes, missing sources of truth, and corrupted images. Measure unsafe restores, unnecessary resets, startup delay, annotation burden, and operator diagnosis time.

**Academic value.** The experiment asks whether durable BPF state can be described by a small type-and-lifecycle contract instead of application-specific recovery code.

**Production value.** An operator can answer "what survives this reboot, and why?" before taking a host down.

**Failure condition.** If almost all production maps are either disposable caches or trivially reconstructed from an external database, a general manifest may cost more than it saves.

### 2. A consistency-aware checkpoint protocol

**Gap.** A long map walk can copy all entries while still mixing multiple application generations.

**Mechanism.** Give the application a checkpoint epoch. When checkpointing begins, the controller advances the epoch. Mutations after that point are either delayed behind a quiescence barrier or recorded in a bounded delta log. Userspace copies the base state, drains deltas through a declared cut, and seals the image with evidence describing the cut.

For workloads that can tolerate a short pause, the same interface should support a simpler stop-the-world mode. The protocol should expose which mode produced the image instead of hiding the consistency cost.

A sealed image might carry:

```text
checkpoint_epoch: 8841
base_copy_complete: true
delta_through_epoch: 8863
controller_generation: 8863
consistency_mode: live-delta
```

**Delta.** This is not an optimized `bpftool map dump`. The new property is a declared application-consistency cut under concurrent mutation.

**Artifact.** A userspace checkpoint library plus small BPF-side epoch/delta helpers, with a stop-the-world fallback for maps or hooks where live logging is impractical.

**Evaluation.** Run high-update-rate hash, LRU, and per-CPU workloads while checkpointing. Inject writer bursts, deletion and reinsertion, CPU hotplug, controller crashes, and delta-buffer pressure. Compare naive iteration, batch lookup, quiescence, and delta logging. Measure invariant violations, runtime overhead, checkpoint pause, lost updates, image size, and recovery point objective.

**Academic value.** This exposes which consistency mechanisms from databases and checkpoint systems transfer cleanly to shared BPF/userspace state and where map semantics require specialized treatment.

**Production value.** Policy and accounting systems can checkpoint without pretending that a multi-second map walk represented one instant.

**Failure condition.** If a short quiescence pause satisfies almost all reboot workflows at lower complexity and acceptable downtime, live delta logging should remain optional.

### 3. A staged restore gate before reattachment

**Gap.** Successfully loading a program and inserting saved entries does not prove that reconstructed state is valid for the new kernel, topology, or application version.

**Mechanism.** Treat restore like a deployment promotion:

1. create fresh target maps without exposing them to production hooks;
2. verify kernel-visible map shape and target capabilities;
3. migrate, reconstruct, reset, or reject each map according to the reboot state contract;
4. run state invariants and small semantic canaries against the prepared generation;
5. attach or switch production only after the gate passes;
6. retain failed checkpoint evidence long enough for diagnosis.

This composes with the September 18 cross-kernel semantic gate. That earlier gate asks whether the program behaves correctly on the new kernel. The restore gate asks whether the state supplied to that program is itself coherent and valid.

**Artifact.** A VM-based reboot harness, restore controller, and corpus of valid, stale, partially written, old-schema, topology-dependent, and deliberately corrupted state images.

**Evaluation.** Reboot across same-kernel and changed-kernel cases while varying CPU count, map limits, BTF, application versions, and controller versions. Kill the checkpoint writer and restore controller at every phase. Compare pin-only restart logic, naive byte replay, external-source reconstruction, and gated restore. Primary metrics are silent semantic corruption, false rejection, recovery time, state loss, and diagnosis time.

**Academic value.** Recovery becomes a falsifiable correctness property rather than a collection of loader conveniences.

**Production value.** Kernel maintenance and host replacement gain an explicit preflight and recovery boundary, with fail-closed behavior where state validity matters.

**Failure condition.** If ordinary load and application health checks already catch every injected bad-state case with negligible delay, a specialized restore gate is unnecessary.

## What should an operator do today?

A production system does not need to wait for a new kernel API to improve this boundary.

First, classify every stateful BPF map as checkpoint, reconstruct, reset, or reject-on-restore. Do not use the presence of a bpffs pin as the classification.

Second, make semantic state versions explicit. BTF is valuable structural evidence, but application validity needs an application-owned version and migration rule.

Third, choose the checkpoint consistency level deliberately. If maintenance can afford a short quiescence window, use it. If live checkpointing is required, add an epoch or delta mechanism and test it under concurrent mutation.

Finally, stage restoration before attaching the recovered generation to production traffic. Validate both the program on the target kernel and the reconstructed state it will consume.

## What would change this conclusion?

This proposal assumes that some eBPF applications carry state whose correctness matters across host maintenance and cannot always be regenerated cheaply from an external authority. If most production deployments treat BPF maps only as rebuildable caches, the general reboot contract should stay lightweight and application-specific mechanisms may be enough.

The consistency machinery would also be unnecessary if a short stop-the-world window is operationally acceptable for nearly every reboot. In that case, an explicit quiescence protocol plus semantic schema checking is the simpler design.

The strongest counterevidence would be a mature, reusable recovery system that already combines per-map durability intent, semantic versioning, a consistent checkpoint cut, topology-aware reconstruction, and post-restore semantic validation across multiple unrelated eBPF applications. That would show the gap is mostly documentation and adoption rather than a missing systems abstraction.

Until then, calling a pinned map "persistent" hides the boundary operators most need to reason about. A pin preserves a live kernel object. Reboot-safe state requires a separate contract for what survives after that object is gone.

## References

- [eBPF Docs: Pinning](https://docs.ebpf.io/linux/concepts/pinning/)
- [Linux kernel documentation: BPF maps](https://docs.kernel.org/bpf/maps.html)
- [Linux kernel documentation: Hash maps](https://docs.kernel.org/bpf/map_hash.html)
- [Linux kernel documentation: BTF](https://docs.kernel.org/bpf/btf.html)
- [Linux kernel documentation: Map of maps](https://docs.kernel.org/bpf/map_of_maps.html)
- [Linux kernel documentation: Array and per-CPU array maps](https://docs.kernel.org/bpf/map_array.html)
- [libbpf API and source](https://github.com/libbpf/libbpf)
- [Cilium issue #44277: endpoint recovery loop while committing BPF pins](https://github.com/cilium/cilium/issues/44277)
