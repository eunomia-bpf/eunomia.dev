---
date: 2026-08-10
title: "Can a Stateful eBPF Application Upgrade Atomically?"
description: "One BPF link can be replaced atomically, but stateful apps span programs, maps, pins, and controllers. This report examines upgrade and rollback semantics."
tags:
  - Daily Report
  - eBPF
  - Runtime Systems
  - State Migration
  - Linux
research_question: "What transaction semantics are needed to upgrade a stateful eBPF application across programs, links, maps, pinned objects, and its userspace controller without exposing a partially upgraded configuration?"
source_cutoff: 2026-08-10
status: daily-report
---

# Can a Stateful eBPF Application Upgrade Atomically?

Suppose a production policy application has two eBPF programs, one attached to ingress traffic and another attached to a cgroup hook. Both read a pinned policy map. A userspace controller keeps that map synchronized with external policy. Version 2 changes the program logic and also changes the map value layout.

Replacing either program is straightforward. Linux exposes `BPF_LINK_UPDATE`, so one BPF link can switch from an old program to a new one without a detach-and-reattach gap. The hard part is the rest of the application. If the controller migrates the map first, version 1 may observe version 2 state. If it replaces the programs first, version 2 may read the old layout. If one replacement succeeds and the second fails, the host can run a combination that was never tested. A controller crash between these steps makes the correct rollback target even less obvious.

<!-- more -->

This is the boundary between **program replacement** and **stateful eBPF application upgrade**. The kernel already gives us useful atomic operations on individual objects. libbpf separates load from attachment so state can be prepared before programs run. Pins make object lifetime independent of one controller process. Map indirection can switch one map reference. Production systems such as Cilium already maintain their own regeneration, migration, revert, and finalization logic.

The evidence points to a narrower missing mechanism: **there is no common application-level commit protocol that binds those object-level operations into one upgrade generation**. A useful transaction layer should let an operator prepare a complete new generation, migrate or reuse state under explicit rules, validate cross-object invariants, expose one logical commit point, retain the old generation long enough to drain or roll back, and recover after the controller dies midway.

This report continues the [userspace eBPF runtime contract](https://eunomia.dev/research/userspace-ebpf-runtime-contract/) and [hook-composition contract](https://eunomia.dev/research/ebpf-hook-composition-contract/). Those reports argued that an eBPF runtime needs explicit lifetime and composition semantics above bytecode execution. Stateful upgrade is where lifetime, composition, and persistent state meet.

## The kernel already makes several individual transitions safe

A transactional upgrade proposal is only useful if it starts from what Linux already provides rather than rebuilding existing primitives.

### A BPF link can replace one program without an attachment gap

The Linux `bpf()` syscall documentation defines `BPF_LINK_UPDATE` as updating the eBPF program associated with a specified link to a new program. At the libbpf layer, the corresponding update operation is used when an application wants to replace the program behind a link instead of destroying and recreating the link.

That solves an important lifecycle problem. A tracer can replace one attached program without deliberately creating an interval where nothing is attached. A policy program can preserve the attachment object's lifetime and ownership while changing the code it executes.

But the operation names one link and one new program. It does not say that two links, three maps, a pinned object namespace, and a userspace controller all changed as one unit. Object-level replacement is therefore a building block, not yet an application transaction.

### libbpf deliberately separates load and attach

The kernel's libbpf overview describes a BPF application as one or more programs, maps, and global variables. Its lifecycle has distinct open, load, attachment, and teardown phases. During load, maps are created and programs are verified and loaded, but the programs have not yet executed. The documentation explicitly notes that this gives userspace a chance to establish initial map state without racing with program execution.

That separation is almost a prepare phase. Version 2 can be opened, relocated, loaded, and populated while version 1 still serves traffic. The missing decision is what makes the prepared collection become the active application, especially when several hook points must move together.

### Pinned objects survive controller process lifetime

The BPF syscall documentation also makes object lifetime reference-based. Maps and programs can be shared between processes, and BPF objects can be pinned in bpffs. An object is deallocated only after file descriptors, pins, and attachment references are gone.

This is useful for upgrades because the controller does not need to own every object's lifetime through one process. It is also why crash recovery becomes a real design problem. After a controller restarts, both the old and partially prepared new objects may still exist. A correct upgrader must determine which generation is active and which objects are safe to garbage-collect rather than assuming process exit restored a clean state.

### Map indirection can stage a new state object

Linux map-of-maps support allows an outer map to hold references to inner maps. Userspace can update an outer-map entry, while BPF programs perform lookups through the outer map. This provides a useful form of indirection: userspace can build a new inner map off the data path, populate it, and then replace the outer reference for future lookups.

This still operates at one map entry and one indirection relationship. It does not simultaneously update program links. It also does not solve arbitrary schema evolution: the outer map constrains the inner-map type and layout, and multi-level nesting is not supported. The primitive is valuable precisely because it shows the shape of a possible commit point while also exposing its boundary.

## Reusing state and replacing state are different upgrade modes

A large class of eBPF upgrades is simpler than the opening scenario. If version 2 uses exactly the same map schema and state semantics, the safest operation may be to reuse the existing map rather than migrate it.

A libbpf maintainer described this as a typical code-upgrade workflow in the libbpf-rs project: open the new object, reuse the file descriptor of the existing pinned map before load, then load the new programs so they reference the old state. That is a strong counterexample to any claim that every eBPF upgrade needs a transaction layer. For a single program with stable state, map reuse plus one link update may be enough.

The problem appears when compatibility is semantic rather than merely structural.

Consider a hash map whose value changes from:

```c
struct policy_v1 {
    __u32 verdict;
    __u32 flags;
};
```

to:

```c
struct policy_v2 {
    __u32 verdict;
    __u32 flags;
    __u64 epoch;
};
```

The object loader can notice a size mismatch. It cannot infer whether `epoch` can be initialized to zero, derived from another source, or requires a coordinated controller action. Even equal-sized structs can be semantically incompatible if a field changes from a counter to a lease expiration time or if one version changes who is allowed to reset an entry.

Cilium provides production evidence for this distinction. In an older issue involving pinned map property mismatches, the loader logged that incompatible maps were being removed to permit the property upgrade and explicitly expected map data loss. The important point is not that this behavior is wrong. It is that schema incompatibility forces a lifecycle decision that is outside the verifier's normal memory-safety question.

A stateful upgrade protocol therefore needs at least three declared modes:

1. **Reuse** when old and new programs intentionally share the same state semantics.
2. **Transform** when state must move through a versioned conversion before cutover.
3. **Replace** when old state is disposable or can be reconstructed independently.

Treating all three as "load the new object" hides the decision that determines correctness.

## Partial failure is not hypothetical in production BPF control planes

The strongest reason to care about upgrade transactions is not aesthetic API consistency. It is that multi-step datapath regeneration already contains failure paths that can expose the wrong state.

A Cilium issue from 2025 documented endpoint regeneration retries that could leave an endpoint associated with an empty policy map, causing policy-denied traffic. The investigation is especially useful because maintainers traced the failure through concrete regeneration phases: policy state, BPF collection loading, attachment, policy-map synchronization, and deferred revert behavior. One maintainer summarized the desired ordering as creating the policy map, populating it, wiring it into the program, loading the program, and only then attaching it. Another traced how an early return could execute revert logic before later synchronization restored the expected map contents.

Cilium is not evidence that Linux BPF updates are generally unsafe. It is evidence that **a real BPF application has side effects whose ordering and rollback semantics extend beyond one syscall**. The project already has a revert stack and finalization actions because successful regeneration is not a single operation.

This is also a useful novelty check. "Add rollback to an eBPF loader" is too weak as a research direction; production loaders already do it. The more specific question is whether these bespoke state machines can be reduced to a portable generation protocol with properties we can test across applications.

## What should an application-level eBPF transaction guarantee?

Calling an upgrade "atomic" can easily overpromise. A kernel cannot rewind packets already processed by version 1, and two independent events may execute on different CPUs while a commit occurs. The useful guarantee is narrower.

For a declared application generation `G`, an upgrader should define these properties:

- **Prepared completeness.** Every program, map, link target, and controller-side dependency required by `G+1` exists and passes validation before it can become active.
- **State compatibility.** Each state object is explicitly reused, transformed, or replaced, with a declared version and migration rule.
- **Single logical commit.** The system has one authoritative generation decision. Observers should not infer activation from whichever individual link happened to update first.
- **No unsupported mixed generation.** If the application declares that two components must move together, the runtime must not intentionally expose a steady state containing one old and one new component.
- **Recoverable ownership.** After controller failure, persistent metadata is sufficient to distinguish active, prepared, retiring, and orphaned objects.
- **Bounded retirement.** The old generation is retained until the runtime can establish that rollback is no longer required and in-flight work that depends on it has drained according to the hook's semantics.

These are application-level properties. Linux already gives us lower-level reference counting, RCU-backed datapath mechanisms in many map types, verifier checks, link lifetime, and per-object updates. A transaction layer should compose those mechanisms rather than pretend to replace them.

## A generation protocol can make the lifecycle explicit

One practical model is a four-phase lifecycle: **prepare, migrate, commit, retire**.

During **prepare**, the controller opens and loads version 2 without attaching it to production hooks. It creates versioned state objects or binds compatible existing maps. It records the expected old generation and all object identities in a transaction record.

During **migrate**, the controller copies or reconstructs state into the new generation. If live updates continue, the protocol must choose a synchronization strategy: pause writes briefly, capture a snapshot plus delta, dual-write through a controller, or let old and new programs share a compatibility map. The choice is workload-specific and should be measurable rather than hidden in the loader.

During **commit**, the runtime changes the authoritative active generation. On hook families that can afford a stable dispatcher, each dispatcher can read a shared generation selector and route to the prepared program set. That reduces a multi-link application cutover to one control-plane generation decision, at the cost of an extra indirection on the hot path. Where no suitable stable dispatch path exists, a stronger kernel primitive or a weaker documented guarantee may be required.

During **retire**, the old generation stays pinned or otherwise referenced until the runtime knows it is safe to remove. A grace period may be enough for a stateless hook. Stateful applications may need acknowledgements from the userspace controller, completion of outstanding asynchronous work, or evidence that no state references point to the old maps.

This is not necessarily a universal kernel ABI. It is first a model that allows us to compare implementations and say precisely where a backend cannot provide the requested guarantee.

## Where current work is still weak

### There is no portable object that describes an upgrade generation

The interfaces inspected here expose programs, maps, links, pins, and object files. Production systems add their own regeneration context and rollback state. What is missing is a portable manifest that says which objects constitute one application generation, which old objects they supersede, and which compatibility relationships must hold at commit.

Without that object, a crash-recovering controller has to reconstruct intent from pin names, process state, loader-specific conventions, or external databases. The consequence is operational ambiguity: leaked objects are annoying, but deleting the wrong "old" map can be a correctness failure.

A decisive test is whether several existing loaders can express their upgrade state machines with the same small set of generation states and dependency edges. If each requires unrelated lifecycle semantics, a common transaction manifest is the wrong abstraction.

### Map schema compatibility is still mostly structural

Loader checks can compare map type, key size, value size, flags, and other properties. BTF provides rich type information. Neither by itself says whether a semantic migration is correct.

The missing element is a migration contract that connects an old BTF/schema version to a transformation and an invariant. The consequence is that sophisticated projects hand-code migration behavior while generic loaders can only reuse, reject, or recreate state.

A useful experiment should include schema changes that preserve byte size but change meaning. If a BTF-aware migration checker only catches changes that ordinary loader property checks already reject, it adds little value.

### Per-link atomic replacement does not define multi-hook commit semantics

`BPF_LINK_UPDATE` is a strong primitive for one link. The gap appears only when an application has a cross-hook invariant such as "ingress classifier and cgroup policy must use the same policy epoch."

The missing element is either a common generation gate or a kernel-level multi-object commit primitive. The consequence of sequential updates is a bounded but real mixed-generation interval. Whether that interval matters depends on the workload.

The right test is not microbenchmarking update syscalls alone. Inject traffic and faults while repeatedly upgrading coupled hooks, then measure whether any event is processed under a forbidden generation combination. If no realistic workload can observe a harmful mixed state, the extra transaction machinery is unnecessary.

### Crash recovery is underspecified for persistent BPF object graphs

Pins deliberately outlive one controller process. That is a feature, but it means a partially prepared generation can survive the process that created it.

The missing element is a durable transaction journal with idempotent recovery rules. The consequence is that cleanup and rollback become application-specific after a crash at exactly the time when in-memory intent is gone.

The test is straightforward: kill the controller after every side effect in the upgrade state machine, restart it, and check whether it always converges to either the old committed generation or the new committed generation without deleting live state.

## Promising directions with academic and production value

### A generation-gated upgrade runtime for libbpf applications

**Gap.** Existing primitives can prepare programs and replace individual links, but a multi-hook application lacks one logical activation decision.

**Mechanism.** Keep a small, stable dispatch program at each supported hook. The dispatcher reads an application-wide generation selector and routes to the program slot for that generation. New programs and maps are loaded into a versioned namespace while the old generation runs. After validation and state preparation, the controller performs one generation-selector update. The old generation remains reachable until a retirement condition is satisfied. For map state, a parallel versioned indirection can select the state object associated with the same generation where the map type permits it.

This design must not hide its portability boundary. Tail calls and dispatcher techniques differ by program type and hook. The runtime should advertise which hook families support a true generation gate, which require sequential `BPF_LINK_UPDATE`, and which cannot meet the requested cross-hook guarantee without kernel support.

**Delta.** The delta from libxdp-style dispatch or ordinary program arrays is the lifecycle protocol, not the existence of a dispatcher. The new property is that program routing, state version, recovery metadata, and retirement all refer to one application generation.

**Artifact.** A libbpf-based controller and manifest format, initially supporting XDP/TC and one tracing or cgroup family, plus an adapter for [bpftime](https://eunomia.dev/bpftime/) to test the same generation protocol in userspace.

**Evaluation.** Use stateless and stateful networking, policy, and observability applications with 1, 2, 4, and 8 coordinated hooks. Repeatedly upgrade under packet/event load. Inject failure before and after every prepare, migration, selector, and retirement step. Measure forbidden mixed-generation observations, lost events or packets, cutover latency, steady-state dispatcher overhead, retained memory during rollback windows, and recovery time. Compare against sequential link updates, stop-and-restart, and application-specific hand-coded orchestration.

**Academic value.** The research question is whether one generation selector plus explicit retirement can provide a useful cross-object consistency property over heterogeneous BPF hook lifecycles, and where the abstraction breaks.

**Production value.** Platform teams get a reusable update state machine rather than embedding one in every BPF controller. The same manifest can drive health checks, rollback, and garbage collection.

**Failure condition.** If realistic applications rarely have cross-hook invariants, or if the dispatcher overhead and hook-coverage limitations exceed the cost of a short sequential update window, this runtime should not replace simpler link updates.

### BTF-aware state migration with explicit invariants

**Gap.** Reusing a compatible pinned map is efficient, but structural compatibility does not cover semantic schema evolution.

**Mechanism.** Assign every persistent state object a schema version derived from BTF plus application metadata. Before load, compare the old and new schemas and classify changes as reusable, mechanically transformable, or requiring an application-supplied converter. For transformations, populate a shadow map, validate invariants such as key preservation, monotonic counters, or policy-epoch consistency, then bind the new generation to that map. Under write-heavy workloads, add a snapshot-plus-delta protocol or a short quiescence window and make its cost explicit.

**Delta.** This is narrower than a general serialization system. It targets BPF map schemas and the exact transition between loaded generations. It also goes beyond existing property checks by testing semantic invariants and by making migration success a prerequisite for activation.

**Artifact.** A BTF schema differ, migration-plan generator, converter API, and fault-injection harness integrated with the generation runtime. Publish a corpus of real map-schema changes from open-source BPF projects after removing project-specific secrets or operational data.

**Evaluation.** Include key/value additions, field renames, widening and narrowing, equal-sized semantic changes, map-type changes, per-CPU state, LRU maps, and maps too large to copy cheaply. Compare full copy, lazy migration, snapshot-plus-delta, and explicit reset. Measure migration throughput, pause time, memory amplification, update loss, invariant violations, and developer effort. The key correctness metric is whether the system rejects unsafe automatic migrations rather than maximizing the number it accepts.

**Academic value.** This asks which semantic properties of long-lived eBPF state can be inferred from types and which require explicit application invariants, a boundary not addressed by verifier memory safety.

**Production value.** Operators can distinguish a safe zero-downtime upgrade from one that requires planned state loss or a maintenance window before deployment begins.

**Failure condition.** If most production map changes are either exactly reusable or intentionally disposable, a migration framework is unnecessary overhead. A corpus study should establish the frequency before a large implementation effort.

### A crash-consistent journal for pinned BPF object graphs

**Gap.** Pins preserve objects across controller restarts, but the controller's in-memory upgrade intent can disappear while both generations remain alive.

**Mechanism.** Store a small durable transaction record that names the expected old generation, prepared new generation, object identities, state-migration status, commit decision, and retirement status. Every phase is idempotent. Recovery follows a deterministic rule: before commit, discard or resume the prepared generation; after commit, restore the new controller view and finish retirement; if the active generation cannot be established, fail closed and preserve both object sets for diagnosis instead of guessing.

The journal can live outside bpffs if the deployment already has durable control-plane storage. The important property is that pinned object references and transaction metadata are reconciled explicitly.

**Delta.** Production projects already have revert stacks and finalizers. The contribution would be a minimal, portable recovery state machine with a tested crash-consistency property rather than another project-specific rollback callback collection.

**Artifact.** A transaction library, recovery checker, and deterministic fault injector that kills the controller after each externally visible operation. Expose the object graph through `bpftool`-compatible IDs and pins so failures can be inspected with normal tooling.

**Evaluation.** Run thousands of randomized upgrades while killing the controller at each transition, including repeated crashes during recovery. Verify that every run converges to a valid committed generation, does not delete the active map, and eventually collects unreachable objects. Compare with a simple desired-state reconciler that has no explicit journal.

**Academic value.** The question is whether reference-counted kernel objects plus a small userspace journal are sufficient for crash-consistent application upgrades, or whether stronger kernel transaction support is required.

**Production value.** This targets the failure mode that is hardest to debug operationally: an agent restart leaves a datapath running, but nobody knows whether its persistent state belongs to the old or new software generation.

**Failure condition.** If an idempotent desired-state reconciler can recover every tested object graph with the same correctness and less metadata, the journal is redundant and the simpler reconciler should win.

## A kernel multi-object transaction should be a result, not the starting assumption

It is tempting to propose a new syscall that takes a list of links and maps and atomically publishes all replacements. That could eventually be useful, particularly for hook families where a stable dispatcher is impossible or too expensive.

But it is premature as the first implementation. The kernel would need to define what happens when different hook types use different synchronization and lifetime rules, how map state migration participates, how expected revisions prevent concurrent controllers from committing stale state, and how failure is reported without leaving references half-installed. A syscall can make a pointer swap atomic without making the application's state transformation correct.

A better research path is to first implement the transaction model in userspace on top of existing primitives, collect the cases where it cannot provide the required guarantee, and then use those cases to justify the smallest kernel primitive. The resulting kernel proposal might be a multi-link expected-generation commit, a reusable generation handle, or something narrower than a general BPF transaction syscall.

## What would change this conclusion?

The strongest alternative is the simple one: most eBPF applications may not need transactional upgrade at all. A stateless tracer with one link can use `BPF_LINK_UPDATE`. A program with a stable map schema can reuse the existing map. An application that can tolerate a brief maintenance window can stop, migrate, and restart with much less machinery.

Three results would materially weaken the argument for an application transaction layer.

First, a corpus of production BPF deployments could show that coupled multi-hook upgrades and semantic map migrations are rare. Second, fault-injection experiments could show that sequential link updates plus ordinary desired-state reconciliation never expose harmful mixed generations in realistic workloads. Third, generation dispatch could impose enough steady-state cost or hook-specific complexity that operators prefer a short explicit disruption.

The opposite evidence would strengthen the case: repeated real incidents involving partially applied program/state updates, several projects independently implementing similar prepare/revert/finalize state machines, or experiments showing that a single generation protocol removes observable policy and state inconsistencies at low cost.

The useful boundary is therefore not "eBPF needs database transactions." It is more specific: **when several persistent BPF objects jointly define one correctness invariant, the upgrade mechanism needs a commit concept at the same scope as that invariant**. Linux already gives us most of the object-level pieces. The open systems question is how small the layer above them can be.

## References

- [Linux kernel: eBPF syscall reference](https://docs.kernel.org/userspace-api/ebpf/syscall.html)
- [Linux kernel: libbpf overview and BPF application lifecycle](https://docs.kernel.org/bpf/libbpf/libbpf_overview.html)
- [Linux kernel: map of maps](https://docs.kernel.org/bpf/map_of_maps.html)
- [libbpf-rs discussion: reusing an existing map during BPF code upgrades](https://github.com/libbpf/libbpf-rs/issues/52)
- [Cilium issue #38998: empty policy map after endpoint regeneration failure](https://github.com/cilium/cilium/issues/38998)
- [Cilium issue #19091: pinned map property mismatch and expected data loss](https://github.com/cilium/cilium/issues/19091)
- [bpftime: userspace eBPF runtime](https://github.com/eunomia-bpf/bpftime)
