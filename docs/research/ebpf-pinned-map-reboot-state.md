---
date: 2026-09-19
slug: ebpf-pinned-map-reboot-state
title: "Can a Pinned eBPF Map Survive a Host Reboot?"
description: "A bpffs pin keeps a BPF map alive across process restarts, not host reboots. This report asks how to checkpoint and restore state without corruption."
tags:
  - Daily Report
  - eBPF
  - Linux
  - BPF Maps
  - State Recovery
  - BTF
research_question: "What state contract is needed to preserve or reconstruct an eBPF application's map state across a host or kernel reboot, when bpffs pinning itself only preserves an in-kernel object reference?"
source_cutoff: 2026-09-19
status: daily-report
---

# Can a Pinned eBPF Map Survive a Host Reboot?

A daemon creates an eBPF hash map, fills it with policy state, pins it under `/sys/fs/bpf`, and exits. A replacement daemon can reopen the same map and continue. It is tempting to call that state "persistent."

Then the host reboots.

The distinction matters because a BPF pin is a lifetime reference to a live kernel object, not a durable storage format. Linux defines `BPF_OBJ_PIN` as creating a filesystem reference that prevents a BPF object from being deallocated when the original file descriptor closes. That extends the object's lifetime beyond one userspace process. It does not turn the object's contents into a disk-backed database that can be reconstructed after the kernel and its BPF objects disappear.

The result is a practical lifecycle gap. Process restart, agent upgrade, kernel reboot, and host replacement are often grouped under "restart recovery," but they require different mechanisms. A pin is enough for the first case. It can participate in the second. It cannot by itself solve the third or fourth.

This report asks what a production eBPF runtime must add when map state actually has to cross a reboot boundary. The answer is not simply "dump the map to a file." A live map may be changing while it is copied, libbpf's pinned-map reuse test is mostly a structural compatibility check, BTF type identity does not define application semantics, and some map state is better reconstructed than serialized.

This is deliberately narrower than the earlier [stateful eBPF transactional-upgrade report](https://eunomia.dev/research/stateful-ebpf-transactional-upgrade/). That report studied atomic cutover among live programs, links, maps, and controllers. Here the old kernel object graph is gone. The question is how to produce a trustworthy new graph from durable evidence.

<!-- more -->

## Pinning preserves an object reference, not durable state

The Linux UAPI description of `BPF_OBJ_PIN` is precise. A pathname in BPF filesystem retains a reference to a BPF object, so closing the original file descriptor does not deallocate it. Removing the pin removes that reference, and the object disappears after its remaining references are gone.

That is an in-kernel lifetime rule.

It gives production systems useful properties:

- a loader can restart while programs and maps remain alive;
- another process can reopen a pinned object with `BPF_OBJ_GET`;
- several tools can share the same map without one process owning its lifetime;
- a control plane can separate object lifetime from controller-process lifetime.

But reboot destroys the kernel objects those references point to. The common eBPF pinning documentation states the operational consequence directly: pins are ephemeral and do not persist across a system restart.

A useful mental model is:

```text
process lifetime < pinned BPF object lifetime < kernel/host lifetime
```

Calling the second term "persistent" without naming the boundary is therefore dangerous. It can be persistent across a daemon crash and still be volatile across a reboot.

The same distinction applies when `/sys/fs/bpf` is mounted automatically after boot. Recreating the bpffs mount point does not recreate the old map objects or their entries. The path namespace and the kernel objects behind those paths are separate concerns.

## Reopening a pinned map proves much less than restoring one

libbpf already knows how to reuse a live pinned map. Its loader opens the pin, reads map information with `BPF_OBJ_GET_INFO_BY_FD`, checks whether the existing map is compatible with the map declared by the new object, and then reuses that file descriptor instead of creating a new map.

The compatibility check is intentionally concrete. Current libbpf source compares properties such as map type, key size, value size, maximum entries, map flags, and `map_extra`. That catches many obvious mismatches. It is exactly what a loader needs before wiring a new program to an existing live kernel map.

It is not a durable-state schema system.

Two maps can have the same key and value sizes while the application changes what a field means. For example, an eight-byte value can move from:

```c
struct state_v1 {
    __u32 verdict;
    __u32 generation;
};
```

to:

```c
struct state_v2 {
    __u32 verdict;
    __u32 lease_seconds;
};
```

The binary layout still fits. Replaying old bytes would be structurally valid and semantically wrong.

BTF improves introspection. Linux can associate map key and value type IDs with a map, and `BPF_OBJ_GET_INFO_BY_FD` can expose the map's BTF metadata. That lets tooling recover a rich structural type description rather than only byte sizes. It still does not say whether `generation == 7` can be interpreted as `lease_seconds == 7`, whether an old counter should be reset, or whether a cached authorization is valid after boot.

A reboot restore path therefore needs two compatibility layers:

```text
kernel/map compatibility
    type + sizes + flags + map-specific constraints + target support

application-state compatibility
    schema version + field meaning + ownership + validity + migration rule
```

Only the first is naturally visible to a generic BPF loader.

## A map dump is not automatically a consistent checkpoint

Even for a plain hash or array map whose values are serializable, there is another problem: live BPF programs can update the map while userspace is copying it.

Linux explicitly permits concurrent access to hash-map values from programs running on different CPUs. Userspace can iterate a hash map with `bpf_map_get_next_key()` or use batch lookup APIs, but those APIs are iteration mechanisms. They do not turn an actively mutating map into a multi-entry transactional snapshot.

This creates familiar checkpoint anomalies. Suppose a policy map stores two related records:

```text
A: policy generation = 42
B: allowed principals for generation 42
```

A checkpoint process might read `A`, then the datapath or controller advances both records to generation 43, then the checkpoint reads `B`. The file on disk can describe a state that never existed at one logical instant.

Per-entry locking does not solve the whole problem. `bpf_spin_lock` can protect fields inside one map value, but a checkpoint usually needs consistency across many keys, multiple maps, or controller-side state. Batch lookup improves throughput, not transaction scope.

For counters or best-effort telemetry, an approximate copy may be fine. For policy, connection ownership, resource accounting, or a recovery cursor, the consistency boundary has to be declared.

The important design question is not "Can I dump every key?" It is:

> What invariant must hold across the restored collection, and what protocol establishes that invariant while the old system is still running?

## Some BPF state should be reconstructed instead of serialized

A generic checkpoint format also needs to resist the urge to treat every map as a bag of portable bytes.

BPF maps have different semantics. Per-CPU maps split values by CPU. Map-of-maps stores references to other map objects. Networking maps can represent live kernel or device relationships. LRU maps intentionally make retention policy part of the map behavior. Other map types expose queues, stacks, ring buffers, socket references, program references, or kernel-managed state.

The correct reboot policy can therefore differ even within one application:

- **checkpoint:** durable policy or learned state whose exact values should survive;
- **reconstruct:** state that can be regenerated from a controller database, configuration, routing table, or external source of truth;
- **reset:** cache, telemetry, transient queue, or epoch-local state that is valid only inside one boot;
- **reject restore:** state whose old representation cannot be safely mapped onto the new kernel or topology.

A production loader should make this decision explicit per map. Silently serializing everything produces an attractive backup file without proving that restoring it is meaningful.

The recent Cilium pinning failure reported in February 2026 is useful operational evidence even though it is not a reboot case. Endpoint regeneration entered a recovery loop because a global BPF map pin already existed at the path where a new pin was being committed. The incident shows that the bpffs namespace is part of a real control-plane state machine: object identity, pin ownership, create/reuse decisions, and cleanup ordering can all affect recovery. Reboot recovery adds another state transition where the paths may return but the old object identities cannot.

## Where current work is still weak

### "Persistent map" usually does not name the persistence boundary

Linux clearly defines reference-counted object lifetime, and mature loaders understand how to reopen or reuse pinned maps. The weak point is operational vocabulary and machine-readable intent. A deployment often says a map is persistent without distinguishing process restart, agent replacement, kernel reboot, or host replacement.

The missing element is a map-level restart contract that states how far the state is expected to survive and where its durable source of truth lives.

A falsifying test is simple: inspect several production eBPF control planes. If their existing map declarations already encode reboot durability, reconstruction source, semantic schema version, and validation policy in a reusable form, a new contract adds little.

### Current compatibility checks are mostly structural

libbpf can reject a pinned map whose basic kernel-visible properties differ. BTF can expose key and value types. Neither proves that replayed values retain the same application meaning.

The missing element is a semantic state version tied to migration or reconstruction logic, not just a byte layout.

The discriminating test is equal-sized semantic drift. If a restore system detects only size or BTF-shape differences but accepts a field whose meaning changed, it has not solved the hard case.

### Live map export has no general application-level snapshot boundary

Map lookup and batch APIs make extraction possible, but a collection of maps can be mutating concurrently. The kernel does not know which entries must represent one application epoch.

The missing element is a quiescence, epoch, or delta protocol that turns many reads into one declared recovery cut.

A useful test must inject writes during checkpoint. A scheme that works only after stopping all BPF programs is still useful, but it should advertise "stop-the-world checkpoint" rather than claiming live consistency.

### Restore success is usually tested as loadability, not state correctness

A recreated map can accept all entries and a program can load successfully while the restored policy, ownership graph, or counters are wrong.

The missing element is a post-restore semantic gate with workload-specific invariants and a clear fallback to reconstruction or reset.

The right metric is silent bad recovery, not only restore latency.

## Promising directions with academic and production value

### 1. A restart contract for every stateful BPF map

**Gap.** Pinning says that a live object outlives one file descriptor. It does not tell an operator whether map contents should survive reboot, where they should come from afterward, or which semantic version they use.

**Mechanism.** Add a small state manifest next to the BPF application artifact. For each map, declare:

```text
map: policy_cache
survival: reboot
strategy: checkpoint | reconstruct | reset
schema: policy-state/v3
kernel_shape:
  type: HASH
  key_btf_digest: ...
  value_btf_digest: ...
consistency: generation-cut
source_of_truth: controller-db
restore_gate: policy-canary-v2
```

The BTF digest should be based on a canonicalized structural description, not raw BTF type IDs, because IDs are local to a BTF object. The semantic `schema` is application-owned and changes when meaning changes even if the C layout does not.

The loader combines this manifest with target-kernel capability evidence. A map marked `reset` is recreated empty. A `reconstruct` map is repopulated from its declared source. A `checkpoint` map requires a compatible durable image and a successful restore gate.

**Delta.** Existing map definitions describe how to create an in-kernel object. The new contract describes how state crosses lifecycle boundaries after that object no longer exists.

**Artifact.** A libbpf-compatible manifest parser, a small library that produces canonical BTF schema fingerprints, and adapters for hash, array, per-CPU, map-of-maps, and selected non-serializable map classes.

**Evaluation.** Take 20 to 30 maps from real networking, tracing, and policy applications. Introduce structural drift, equal-size semantic drift, topology changes, and missing source-of-truth data. Measure unsafe restores, unnecessary resets, operator annotations required, startup latency, and how often the manifest can select the correct recovery mode automatically.

**Academic value.** The question is whether BPF state durability can be expressed as a small type-and-lifecycle contract instead of application-specific recovery code.

**Production value.** Operators can answer "what survives this reboot?" before maintenance begins and can audit why one map was restored while another was rebuilt.

**Failure condition.** If most production maps are either trivial caches or already governed by external databases, a general manifest may be more machinery than the durable state justifies.

### 2. A quiescence-aware checkpoint protocol for maps that stay live

**Gap.** Batch lookup can copy map entries efficiently, but it does not define a logically consistent cut across concurrent BPF and userspace updates.

**Mechanism.** Introduce an application checkpoint epoch. When a checkpoint starts, the controller advances an epoch visible to participating BPF programs. Mutations after that point are tagged with the new epoch or mirrored into a bounded delta log. Userspace copies the base maps with batch lookup, records the checkpoint epoch, then drains and applies the delta up to a declared cut before sealing the image.

For applications that can tolerate a brief pause, the same API can use a simpler quiescence barrier: stop new mutations, wait for in-flight operations according to the hook's semantics, copy, then resume. The contract should expose which mode it used rather than hiding the consistency cost.

The checkpoint file stores both the map-state image and evidence about the cut:

```text
checkpoint_epoch: 8841
base_copy_complete: yes
delta_range: 8841..8863
delta_complete: yes
controller_state_generation: 8863
```

**Delta.** This is not a faster map dump. It is a consistency protocol that makes concurrent mutation part of the checkpoint model.

**Artifact.** A libbpf userspace checkpoint library plus BPF-side epoch/delta helpers, with a mode that uses ring-buffer or side-map logging and a stop-the-world fallback.

**Evaluation.** Run high-update-rate hash, LRU, and per-CPU workloads while checkpointing. Inject writer bursts, CPU hotplug, entry deletion/reinsertion, controller crashes, and delta-buffer pressure. Measure invariant violations, checkpoint pause, runtime overhead, image size, lost updates, and recovery point objective.

**Academic value.** This tests where consistency mechanisms borrowed from database/checkpoint systems fit BPF's shared kernel/userspace state model and where map-specific semantics defeat a generic protocol.

**Production value.** Stateful policy and accounting systems can checkpoint without pretending that a long map walk was one instant.

**Failure condition.** If a short stop-the-world pause is cheaper and adequate for nearly all reboot workflows, the live delta protocol should remain optional rather than becoming default infrastructure.

### 3. A reboot restore gate that treats reconstruction as a tested deployment step

**Gap.** After reboot, a loader can recreate maps, insert saved entries, load programs, and attach them, but those successful operations do not prove that the application state is valid on the new kernel and topology.

**Mechanism.** Make restore a staged promotion:

1. create fresh target maps without exposing them to production hooks;
2. verify kernel-visible shape and canonical BTF schema evidence;
3. migrate, reconstruct, reset, or reject each map according to the restart contract;
4. run state invariants and small semantic canaries against the prepared program/map generation;
5. attach or switch production only after the gate passes;
6. keep the checkpoint and failure evidence long enough for diagnosis instead of immediately overwriting it.

This is complementary to the September 18 [cross-kernel semantic-compatibility report](https://eunomia.dev/research/ebpf-kernel-upgrade-semantic-compatibility/). That report asks whether the same artifact preserves behavior across kernels. This gate asks whether the **reconstructed state** is a valid input to that artifact after the original kernel state disappeared.

**Artifact.** A VM-based reboot harness, restore controller, and state-image corpus containing valid, stale, partially written, old-schema, and deliberately corrupted checkpoints.

**Evaluation.** Reboot across same-kernel and changed-kernel cases. Vary CPU count, map limits, feature support, schema versions, and controller versions. Kill the checkpoint writer and restore controller at every phase. Compare pin-only restart logic, naive byte dump/replay, external-database reconstruction, and gated restore. Primary metrics are silent semantic corruption, false rejection, recovery time, state loss, and operator diagnosis time.

**Academic value.** It turns BPF restart recovery into a falsifiable correctness problem rather than a best-effort loader feature.

**Production value.** Kernel maintenance can have an explicit preflight and recovery contract instead of discovering after boot which "persistent" state was actually volatile.

**Failure condition.** If external authoritative stores plus ordinary program startup already reconstruct state with negligible correctness risk and acceptable recovery time, application-specific reboot gates are sufficient and a shared BPF mechanism is unnecessary.

## A practical recovery ladder should say what it knows

A production implementation does not need to checkpoint every map.

A better policy is layered:

```text
process restarted, kernel unchanged?
    -> reopen/reuse live pins when structural + semantic contract matches

kernel or host restarted?
    -> pins are gone as object references
    -> recreate map graph

map strategy = reconstruct?
    -> rebuild from external source of truth

map strategy = checkpoint?
    -> validate image + schema + recovery cut
    -> restore into fresh maps

map strategy = reset?
    -> start empty and record that state continuity was intentionally lost

prepared generation passes invariants/canaries?
    -> attach/promote
    -> otherwise fail closed, reconstruct, or use declared degraded mode
```

This model also keeps claims honest. "Pinned" means the object survives controller-process lifetime while its kernel remains alive. "Checkpointed" means a durable image exists for a named consistency cut. "Restored" means that image or source of truth produced a new map graph. "Validated" means the application checked the invariants that make that state useful.

Those are different guarantees and should have different evidence.

For an eBPF toolchain such as [eunomia-bpf](https://eunomia.dev/eunomia-bpf/) and its [GitHub repository](https://github.com/eunomia-bpf/eunomia-bpf), the useful integration point is the application package or loader manifest: ship the BPF object together with the declared state lifecycle, not just map creation metadata.

## What would change this conclusion?

Three findings would weaken the case for a first-class reboot-state contract.

First, if a survey of production eBPF applications shows that almost all state is either disposable or already reconstructed from an external database, reboot durability belongs in those controllers rather than a reusable BPF layer.

Second, if structural map properties plus canonical BTF shape predict restore correctness across real application upgrades with almost no equal-layout semantic failures, the separate semantic schema layer can be simplified.

Third, if live checkpoint consistency is rarely needed because maintenance workflows can cheaply quiesce BPF mutations before reboot, a generic epoch-and-delta protocol would add overhead without much value.

The current mechanisms support a narrower conclusion. Linux pinning is excellent for decoupling BPF object lifetime from a userspace process. It is not durable storage. Once recovery must cross the kernel-lifetime boundary, a trustworthy system needs an explicit answer for **what state is durable, how it is captured, how its meaning is versioned, and what proves the reconstructed state is safe to activate.**

## Sources

- Linux UAPI `BPF_OBJ_PIN` / `BPF_OBJ_GET` lifetime semantics: <https://github.com/torvalds/linux/blob/master/include/uapi/linux/bpf.h>
- Linux kernel BTF documentation, including map key/value BTF metadata and `BPF_OBJ_GET_INFO_BY_FD`: <https://docs.kernel.org/bpf/btf.html>
- Linux BPF hash-map documentation on concurrent access and userspace iteration: <https://docs.kernel.org/bpf/map_hash.html>
- Linux bpftool map documentation on pinning, map creation, batch-style operations, and map inspection: <https://github.com/torvalds/linux/blob/master/tools/bpf/bpftool/Documentation/bpftool-map.rst>
- libbpf source and pinned-map reuse compatibility logic: <https://github.com/libbpf/libbpf/blob/master/src/libbpf.c>
- eBPF Docs pinning concept, including the system-restart boundary: <https://docs.ebpf.io/linux/concepts/pinning/>
- Cilium issue #44277, February 10, 2026, showing production recovery failure around an existing global BPF map pin: <https://github.com/cilium/cilium/issues/44277>
