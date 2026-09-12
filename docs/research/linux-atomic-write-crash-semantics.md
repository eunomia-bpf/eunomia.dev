---
date: 2026-09-12
slug: linux-atomic-write-crash-semantics
title: "Does an Atomic Linux Write Mean the Data Survives a Crash?"
description: "Linux atomic writes prevent torn data ranges, but crash-safe applications also need persistence, ordering, metadata, and recovery guarantees."
tags:
  - Daily Report
  - Linux
  - Storage
  - Atomic I/O
  - Crash Consistency
research_question: "How should Linux applications compose atomic-write, persistence, ordering, filesystem-metadata, and recovery guarantees so that RWF_ATOMIC is not mistaken for a complete crash-consistency contract?"
source_cutoff: 2026-09-12
status: daily-report
---

# Does an Atomic Linux Write Mean the Data Survives a Crash?

Linux now has a real atomic-write interface for regular files. With `pwritev2(..., RWF_ATOMIC)`, a supported block filesystem can ask the storage stack for torn-write protection: after a power failure or hardware failure, the target range must contain either the old data or the new data, not a mixture of both.

That sounds close to the property a database wants from a page update. It is also easy to over-read it.

An untorn write answers one narrow question: **can one eligible data range be observed half old and half new after failure?** It does not, by itself, answer whether the new data was durable when the syscall returned, whether two atomic writes reached stable storage in the intended order, whether a directory entry or file-size update is consistent with the data, or whether several files form one recoverable application transaction.

The distinction matters because Linux exposes these properties through different mechanisms. `RWF_ATOMIC` controls torn-write protection. `O_SYNC`, `O_DSYNC`, `RWF_SYNC`, `fsync()`, and filesystem journaling address different persistence and metadata boundaries. An application that compresses all of these into one Boolean called `atomic_write_supported` can still recover into a state that no application-level transaction ever intended.

<!-- more -->

This report is an adjacent Linux/storage detour from the active eBPF deployment-compatibility roadmap. The current rolling Daily Report window is already at the normal seven-of-ten eBPF ceiling, so another eBPF-centered report would violate the repository's topic-mix contract. The storage question is nevertheless directly relevant to systems software: it asks how a new kernel capability should be exposed without letting a narrow lower-layer guarantee silently expand into a broader application guarantee.

## Linux atomic writes solve torn writes, not every crash-consistency problem

The current Linux manual page defines `RWF_ATOMIC` as torn-write protection for regular files in block-based filesystems. If the storage path supports the operation, a hardware failure must leave all or none of the written data, never a mixture of old and new bytes. The same manual page also says the write must use `O_DIRECT`, obey the atomic-unit and alignment limits reported through `statx()`, and use synchronized I/O such as `O_SYNC`, `O_DSYNC`, or `RWF_SYNC` when the application needs consistency between in-core state and the storage device.

That last sentence is the important one. Atomicity and synchronization are separate dimensions in the API.

The `statx()` interface exposes `stx_atomic_write_unit_min`, `stx_atomic_write_unit_max`, `stx_atomic_write_segments_max`, and `stx_atomic_write_unit_max_opt`. Those fields tell a program what range shapes the current file and filesystem can submit atomically. They do not describe an application transaction, nor do they say that two separately submitted writes form one atomic unit.

Ext4 makes the lower-layer dependencies visible. Its current atomic-write documentation requires Direct I/O on regular extent-based files and hardware atomic-write support from the underlying block device. Single-filesystem-block atomic writes have been supported since Linux 6.13; multi-block atomic writes use bigalloc and are bounded by filesystem and device atomic-write units. For a mixed mapped/unwritten region, ext4 first normalizes the target into a single contiguous extent and can force the current journal transaction to commit before issuing the data I/O. That work exists precisely because an apparently simple range guarantee crosses allocation metadata, unwritten-extent conversion, the journal, iomap, and the device.

Ext4 journaling is another different contract. JBD2 primarily protects filesystem metadata from ending up halfway through a metadata transaction. In the default `data=ordered` mode, file data is not generally journaled even though related data is ordered before metadata commit. `data=journal` is stronger and more expensive. Therefore "the filesystem has a journal" is no more equivalent to "my application transaction is durable" than `RWF_ATOMIC` is.

A useful way to reason about crash behavior is to separate at least five properties:

| Property | Example mechanism | Question it answers |
| --- | --- | --- |
| Range atomicity | `RWF_ATOMIC` + supported storage | Can this one write tear? |
| Completion/persistence | `O_SYNC`, `O_DSYNC`, `RWF_SYNC`, `fsync()` | What must be stable before completion is reported? |
| Ordering | flush/barrier and application protocol | If A must precede B, can recovery observe B without A? |
| Filesystem metadata consistency | ext4/JBD2 modes and metadata rules | Are allocation, size, rename, and directory state recoverable? |
| Application transaction consistency | WAL, commit record, copy-on-write, recovery logic | Which combinations of several objects are legal after a crash? |

A system can satisfy the first property and fail the fifth. For example, imagine a database that atomically overwrites data page P and then atomically overwrites commit record C, but does not establish the intended persistence order. Neither write can tear, yet a crash may still leave an application-invalid combination if C is durable while P is not. Conversely, a WAL protocol may deliberately tolerate a missing data-page write because recovery knows how to replay it. The application's legal crash states come from the protocol, not from the word "atomic" on one syscall.

This is similar to the distinction in the earlier [GPU checkpoint recovery report](https://eunomia.dev/research/gpu-checkpoint-recovery-consistency/): making one component restorable does not prove the whole application resumes from one legal cut. The storage problem is different in mechanism, but the systems lesson is the same: a local guarantee must not be silently promoted into a global recovery guarantee.

## Where current work is still weak

The first gap is **capability description**. Linux gives applications strong per-file facts through `statx()`, and filesystems document their requirements, but the application usually has to reconstruct a larger crash contract from open flags, mount/filesystem behavior, device properties, and its own recovery protocol. "Atomic write available" is too small a summary for a storage engine choosing a safe layout.

The second gap is **composition**. There is no general object that says, for one application commit, "these two atomic data ranges, this metadata update, this directory operation, and this commit record must produce only these post-crash states." Filesystem journals and database WALs each implement composition internally, but the boundary between them is still mostly encoded in bespoke code and assumptions.

The third gap is **failure taxonomy in evaluation**. A test that detects torn sectors can validate an atomic-write mechanism while missing lost whole writes, reordering, metadata/data disagreement, or a recovery protocol that accepts an impossible generation. Linux has mature filesystem and block-layer test frameworks, including blktests, but an application-level atomic-I/O evaluation needs an oracle above "the device did not tear this range."

The fourth gap is **portable deployment evidence**. Atomic-write support is intentionally capability-based rather than a simple kernel-version check. The same application binary can encounter different `statx()` limits, filesystem configurations, and storage hardware. A storage engine needs to explain why it selected an atomic fast path on one host and a WAL/fsync fallback on another, and it should be able to reproduce that decision after an incident.

## Promising directions with academic and production value

### 1. Build a compositional crash-semantics descriptor

Instead of exposing atomic-write support to the application as a Boolean, generate a machine-readable descriptor for the actual storage path. It could bind together:

```text
file = inode/mount identity
atomic_range = statx min/max/segments/alignment
io_mode = O_DIRECT + sync semantics
filesystem = ext4 + relevant feature/mount mode
metadata_scope = what rename/size/allocation updates require
storage_path = device/controller capability identity
application_protocol = expected recovery contract version
fallback = WAL/fsync or copy-on-write path
```

The descriptor would not claim to replace filesystem or device specifications. Its job is to make the application's *assumed* crash semantics explicit and versioned at the integration point where an I/O engine chooses a protocol.

A research prototype could compare descriptor predictions with observed outcomes under controlled crash injection across several kernels, ext4 configurations, and device capability profiles. The primary metric should be **false confidence**: cases where the descriptor says a protocol is safe but fault injection reaches a forbidden state. Secondary metrics include false rejection, descriptor-generation cost, and how often the selected fast path differs from a conservative baseline.

The academic contribution is a compositional model for crash guarantees that normally live in separate API and filesystem documents. The production user is a database, object store, or storage runtime; the integration boundary is its I/O-backend initialization and protocol-selection path.

This direction should be rejected if existing `statx()` fields, open flags, ordinary filesystem discovery, and current storage-engine configuration already determine every relevant recovery state without ambiguity. If the descriptor never changes a safe/unsafe decision or catches an invalid assumption, it is only another manifest.

### 2. Turn each application commit into a crash-cut witness

A second direction is to describe an application commit as a small ordered graph of persistence obligations rather than as a sequence of syscalls whose intended semantics exist only in code comments.

For a page-oriented database, a witness could say:

```text
WAL record W must be durable before page P may become authoritative
P may use one RWF_ATOMIC write if its range is eligible
commit marker C may become durable only after W's persistence edge
metadata operation M is required only when allocation/layout changes
recovery may accept cuts {old, W-only, W+P, W+P+C}
recovery must reject every other visible combination
```

The runtime could emit this witness in debug/test builds and a checker could enumerate crash cuts at each persistence edge. The point is not to put a transaction manager into the kernel. It is to give application-level tests a precise oracle for what lower-level guarantees are supposed to compose into.

Evaluation should compare an atomic-write-aware protocol against a conventional WAL plus `fsync()` baseline. Inject failures before submission, after device completion, around synchronization operations, and around metadata changes. Measure forbidden recovered states first, then barrier count, write amplification, commit latency, and recovery work. The strongest result would show that atomic writes safely remove some logging or copy cost without widening the legal recovery-state set.

The academic contribution is a concrete semantics for composing untorn ranges with persistence ordering and application recovery. The production integration point is the commit path of a database or storage engine, with the witness primarily used in CI, qualification, and incident reproduction rather than on every production I/O.

The idea loses if a fixed WAL/`fsync()` protocol remains simpler while matching or beating the witness-guided design in latency and write amplification with zero forbidden recovery states. Atomic I/O is not automatically worth extra protocol machinery.

### 3. Benchmark crash guarantees by failure class, not one pass/fail bit

A useful benchmark should deliberately distinguish at least these failures:

- torn data inside one requested range;
- an entire completed-but-unsynchronized write being absent after power loss;
- two individually valid writes becoming persistent in an application-invalid order;
- data and filesystem metadata describing different generations;
- namespace operations such as create/rename reaching a different cut from file data;
- process crash, kernel crash, controller reset, and full power-loss models.

The benchmark can reuse existing kernel infrastructure where it fits. `blktests` already provides a framework for Linux block-layer and storage-stack testing, and filesystem test tooling can exercise atomic-write paths. The missing piece is an application-level oracle that labels each observed recovered state as allowed or forbidden under a declared protocol.

Run the same protocol across multiple atomic-unit sizes, extent layouts, sync modes, filesystem modes, and emulated or physical failure models. Report torn-write escapes, forbidden recovered states, false accepts/rejects by the capability selector, recovery time, and performance overhead separately. A single "crash test passed" number hides exactly the distinctions this report is trying to preserve.

The academic value is a measurement methodology for separating atomicity, durability, ordering, metadata consistency, and transaction recovery. The production user is a storage-engine or platform qualification team; the integration boundary is the kernel/filesystem/device matrix used before enabling a fast path in a fleet.

This benchmark is unnecessary if an existing block/filesystem test suite already predicts application recovery across the same failure classes and target matrix at comparable cost. The experiment should explicitly try to demonstrate that the application-level oracle finds no additional failures.

## A practical deployment rule

For now, an application should treat `RWF_ATOMIC` as a capability to simplify one part of a crash protocol, not as permission to delete the protocol.

First query the actual file capability with `statx()` rather than infer it from the kernel version. Respect Direct I/O, alignment, range, and segment constraints. Decide separately whether completion needs synchronized I/O. Keep ordering requirements explicit between writes. Treat filesystem metadata and namespace operations according to the filesystem's own persistence rules. Finally, validate the application's recovery state machine with crash injection before replacing a WAL, copy-on-write step, or barrier.

The earlier [stateful eBPF transactional-upgrade report](https://eunomia.dev/research/stateful-ebpf-transactional-upgrade/) used generation-gated commit and recovery to avoid interpreting a partially applied multi-object update as a valid new generation. Storage software faces the same class of composition mistake even though the primitives are different: one locally atomic operation does not make a multi-object transition atomic.

## What would change this conclusion?

The argument would weaken if Linux grew one end-to-end interface that explicitly combined untorn writes, persistence completion, cross-write ordering, relevant filesystem metadata, and a multi-object transaction boundary with documented recovery semantics. In that world, the application would not need to compose as many independently scoped guarantees.

It would also weaken if real storage engines using `RWF_ATOMIC` showed that a much smaller contract is sufficient in practice: for example, if their application state truly fits inside one eligible atomic range and no other metadata or object participates in the commit. Then a single atomic-and-synchronized write can be the transaction boundary, and additional composition machinery is unnecessary.

Finally, fault-injection evidence could falsify the proposed research directions. If current capability discovery plus conventional WAL/fsync practice already predicts every observed post-crash state across diverse kernels, filesystems, and devices, a new descriptor or witness language would add process rather than safety.

The useful mental model is therefore narrow: **Linux atomic writes can eliminate torn writes for a supported range. Crash consistency remains a protocol property whose persistence, ordering, metadata, and recovery boundaries must be stated separately.**

## References

- Linux kernel documentation, [Atomic Block Writes](https://www.kernel.org/doc/html/latest/filesystems/ext4/atomic_writes.html), accessed 2026-09-12.
- Linux man-pages, [`pwritev2(2)` / `RWF_ATOMIC`](https://man7.org/linux/man-pages/man2/pwritev2.2.html), man-pages 6.19, accessed 2026-09-12.
- Linux man-pages, [`statx(2)` / `STATX_WRITE_ATOMIC`](https://man7.org/linux/man-pages/man2/statx.2.html), accessed 2026-09-12.
- Linux kernel documentation, [ext4 Journal (jbd2)](https://www.kernel.org/doc/html/latest/filesystems/ext4/journal.html), accessed 2026-09-12.
- linux-blktests, [blktests](https://github.com/linux-blktests/blktests), accessed 2026-09-12.
