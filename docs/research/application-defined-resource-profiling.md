---
date: 2026-08-20
title: "How Should a Profiler Discover Application-Defined Resources?"
description: "Application-defined resources hide behind normal CPU and memory metrics. This report asks how profilers can discover their semantics and detect stale models safely."
tags:
  - Daily Report
  - Profiling
  - Observability
  - Performance
  - Linux
research_question: "How can a profiler discover the identity, lifetime, capacity, and usage semantics of application-defined resources, validate that model at runtime, and detect when an inferred or declared model has become stale?"
source_cutoff: 2026-08-20
status: daily-report
---

# How Should a Profiler Discover Application-Defined Resources?

Suppose a database suddenly loses throughput while CPU utilization, resident memory, and lock contention still look ordinary. One request has filled an internal buffer pool with temporary data. Later requests now trigger expensive eviction and I/O, but the operating system only sees memory that the database allocated long ago. The important resource is not a page allocator or kernel lock. It is a buffer pool whose meaning exists inside the application.

This is the visibility gap behind **application-defined resource profiling**. Recent work shows that a profiler can recover much of this hidden structure from source code and runtime behavior. The harder next problem is keeping that recovered model correct as software changes. A profiler needs to know not only that a variable or function looks resource-related, but what identifies one resource instance, when that identity can be reused, what units are acquired or released, which operations indicate pressure, and whether the model still matches the binary that is running.

<!-- more -->

The useful design target is therefore a **versioned resource semantics contract**. Applications may declare part of it explicitly. Offline analysis may infer missing pieces. Runtime observation should then validate the contract and downgrade confidence when the observed behavior no longer matches it. This is a general profiling problem. Linux `user_events`, static trace markers, uprobes, compiler instrumentation, and eBPF can all provide evidence, but none of them alone defines what an application resource means.

This report follows the [asynchronous causal profiling report](https://eunomia.dev/research/async-ebpf-causal-profiler/) from a different direction. That report asks how to connect work after a logical operation moves across threads and queues. Here the question comes one step earlier: **how does a profiler obtain a trustworthy identity and lifecycle for an application resource in the first place?**

## Why system metrics can be correct and still miss the bottleneck

The OSDI 2026 paper *Diagnosing Performance Issues in Application-Defined Resources* gives a concrete example. MySQL can fill its InnoDB buffer pool with temporary table pages. Later requests then pay for eviction and reload. The operating system does not report memory pressure because the buffer pool is already allocated inside MySQL. CPU, memory, lock, and I/O profilers can expose symptoms, but the causal resource abstraction is defined by the application.

The paper studies 45 real performance issues and identifies 38 distinct application-defined resources across databases and search systems. These include memory-like pools, shared state such as logs and indexes, and internal queues. Its analytical model maps 39 of the 45 cases to recurring resource pathologies. The important point is not the exact taxonomy. It is that application logic creates performance-relevant capacity and ownership rules that do not necessarily correspond to a kernel object.

[gigiprofiler](https://homes.cs.washington.edu/~baris/public/gigiprofiler.pdf) attacks this gap with a hybrid pipeline. An LLM proposes candidate resources and resource-use sites from names, comments, documentation, and surrounding code. Static analysis validates candidates against control and data flow. The runtime then instruments four kinds of interactions, `WAIT`, `ACQUIRE`, `USE`, and `RELEASE`, and correlates them with requests.

That combination matters because neither semantic inference nor program analysis is sufficient by itself. In the paper's MySQL study, LLM-only candidate identification has false-positive rates between 45% and 60% across the tested models. Static validation reduces false positives, and runtime validation reduces them further. Even after the full discovery pipeline, an exhaustive buffer-pool study finds that the tested models miss about 23% of resource-use events on average. The system still diagnoses all 15 reproduced issues in its evaluation and finds two additional MariaDB bugs confirmed by developers, which shows that an incomplete resource map can still be useful when downstream analysis is robust to missing events.

This is a strong result, but it also exposes the next systems question. A discovered event site is not yet a durable semantic contract.

## A resource event is not yet a resource model

Consider a profiler that has correctly identified these calls:

```text
get_page(pool, key)
release_page(pool, page)
evict(pool)
```

Several facts remain ambiguous.

First, **identity**. Is `pool` a singleton, a per-tenant pool, a sharded pool, or a wrapper whose backing object can change? Is `page` itself the resource, or one unit borrowed from `pool`?

Second, **lifetime**. If an address is freed and reused, the same pointer can name two different logical resources over a long-running profile. A profiler that treats raw addresses as permanent identities can create false joins.

Third, **units and capacity**. `ACQUIRE(pool, 1)` only means something if one unit has a stable interpretation. A buffer page, token slot, connection, queue credit, and cache byte are different accounting units. Some resources have a hard capacity; others grow until a policy threshold triggers cleanup.

Fourth, **state transitions**. A call named `get` may acquire capacity, look up existing state, or merely return a borrowed reference. A `release` may make capacity immediately reusable or only enqueue deferred reclamation. Function names are evidence, not semantics.

Finally, **version**. An application upgrade can inline a wrapper, change an allocator, split one pool into shards, rename an event, or alter the meaning of a counter without changing the high-level feature. The profiler needs a way to detect when yesterday's resource model is no longer valid for today's binary.

These details determine whether an apparent bottleneck is real. If the profiler joins two generations of an object, counts borrowed references as ownership, or assumes the wrong capacity unit, a convincing resource flamegraph can be semantically wrong.

## Linux already offers several ways to expose evidence

A resource contract does not require one instrumentation mechanism.

Linux [`user_events`](https://docs.kernel.org/6.18/trace/user_events.html) lets a process register typed trace events that existing tracing tools such as ftrace and perf can consume. The application can also observe whether a tool has enabled an event and avoid emitting data when nobody is listening. Current Linux documentation includes a multi-format registration mode so different payload formats for the same logical event name can coexist. That solves an important transport and schema-evolution problem.

But a trace-event schema still does not say that `pool_id` is unique only until a pool is destroyed, or that `units` means pages rather than bytes. Those are semantic properties above the event format.

Linux [uprobes](https://docs.kernel.org/6.18/trace/uprobetracer.html) provide the opposite trade-off. A profiler can attach to existing user-space code without modifying the application, fetch arguments or return values, and count probe hits. This is useful for retrofitting observability. The attachment point is still a code location in a particular executable or library, and the tracer needs to understand which value at that location represents the logical resource.

Static user-space trace markers and compiler-inserted probes provide more stable semantic intent because application developers can name events deliberately. gigiprofiler shows another path: infer events automatically and inject instrumentation with an LLVM pass. These approaches are complementary. The missing layer is a representation that lets a profiler compare and combine their semantic claims.

## The profiler should carry a versioned resource semantics contract

A compact contract can make the assumptions explicit without requiring every application to adopt a large tracing framework. Conceptually, one resource class might expose:

```text
ResourceClass {
    name: "buffer_pool"
    schema_version: 3
    build_identity: <binary or module identity>
    instance_key: <expression or declared field>
    generation_rule: <creation/destruction boundary>
    unit: "page"
    capacity: <fixed, dynamic, or unknown>
    events: {
        acquire: ...
        use: ...
        wait: ...
        release: ...
    }
    scope: <process, tenant, shard, request, ...>
    evidence: <declared, statically-validated, inferred>
}
```

This is not intended as a universal ontology for all software resources. It is a minimum contract for answering profiling questions safely: which instance did an event affect, how long is that identity valid, what quantity changed, and how confident are we that the mapping still applies?

The profiler can build the contract from three sources.

1. **Declared semantics.** An application or library can publish a small descriptor next to `user_events`, USDT-style probes, or another stable instrumentation API.
2. **Inferred semantics.** Source or binary analysis can propose resource classes and event mappings when no descriptor exists.
3. **Observed invariants.** Runtime evidence can test whether the declared or inferred model behaves consistently, for example whether object generations overlap, capacity accounting becomes impossible, or expected release paths disappear.

The third source is what keeps the system from treating an old contract as truth forever.

## Where current work is still weak

### Resource discovery is evaluated mostly as event-site accuracy, not semantic identity accuracy

The gigiprofiler evaluation carefully measures false positives and missed resource-use events. That is necessary, but a profiler can find the right function and still assign the wrong logical identity or lifetime to the value observed there.

The missing evidence is an evaluation that separates **event-site correctness** from **resource-instance correctness**. A benchmark should deliberately reuse addresses, shard pools, hand off borrowed references, and change ownership rules between versions. It should measure false joins and split identities in addition to event precision and recall.

This matters in production because a false join can transfer cost from one tenant, request, or generation to another. A resource model that is 95% accurate at finding probe sites can still produce misleading attribution if the remaining semantic mistakes occur at high-fanout resources.

### Explicit trace schemas describe payloads better than lifecycle semantics

`user_events` can register typed fields and even support multiple formats for one event name. Static markers can provide stable named probe points. Neither interface defines a common language for resource generation, capacity units, ownership, borrowing, or delayed reclamation.

The missing capability is not another event transport. It is a small semantic layer that can state those properties and bind them to a particular binary or module version.

A decisive test is cross-tool reuse. If perf, an eBPF-based profiler, and an application-specific debugger cannot consume the same descriptor and agree on resource identities for a workload, the contract has not separated semantics from one profiler implementation.

### Automatic inference lacks a strong stale-model detector

The OSDI 2026 results show why hybrid validation is useful: model-only discovery has high false positives, and later static and dynamic checks remove many of them. The paper also reports missed events when comments are absent or validation rules do not match an access pattern.

Software evolution creates the same problem over time. A resource mapping inferred from version N may silently become incomplete in version N+1. A profiler needs a negative signal that says, "this contract no longer explains the observed execution," rather than continuing with a clean-looking but stale profile.

Useful signals could include impossible capacity balances, unmatched object generations, event distributions moving to previously unseen call sites, or a build identity mismatch. The important property is explicit confidence degradation, not perfect automatic repair.

### There is no shared benchmark for resource semantics across instrumentation strategies

Today it is difficult to compare manual annotations, static markers, compiler instrumentation, automatic inference, uprobes, and hybrid designs on the same ground truth. Each technique pays a different engineering and runtime cost.

A useful benchmark needs resource definitions whose true identity, lifetime, capacity, and ownership are known. It should include source and binary changes that preserve behavior but perturb instrumentation, as well as semantic changes that must invalidate an old descriptor.

Without this benchmark, a low runtime-overhead number says little about the maintenance cost or false-confidence risk of the resource model itself.

## Promising directions with academic and production value

### 1. A portable resource semantics manifest

**Gap.** Existing event formats can carry values, while automatic profilers can infer resource-use sites, but neither produces a small reusable description of resource identity and lifecycle.

**Mechanism.** Define a versioned manifest that names resource classes, instance keys, generation boundaries, units, capacity semantics, ownership scope, and event mappings. Bind the manifest to build IDs or module identities. Let declared descriptors and offline inference produce the same representation, with every field carrying an evidence source and confidence state.

**Delta.** The difference from gigiprofiler is not another resource detector. The detector becomes one producer of a durable artifact that can be reused by multiple profilers and checked across upgrades. The difference from `user_events` is that the manifest describes resource meaning rather than only event payload layout.

**Artifact.** A schema, compiler/runtime adapters for a few representative applications, and readers for perf-style or pprof-style analysis tools.

**Evaluation.** Use databases, web servers, runtimes, and model-serving caches with manual ground truth. Compare manual instrumentation, `user_events` or static markers, gigiprofiler-style inference, and the manifest pipeline. Measure descriptor authoring effort, event precision/recall, resource-instance precision/recall, false joins, attribution error, runtime overhead, and reuse across software versions.

**Academic value.** The general question is whether semantic observability can be expressed as a portable contract independent of one tracing mechanism.

**Production value.** Operators could keep a stable profiling interface for internal pools, queues, caches, and credits while changing the low-level collector.

**Failure condition.** If per-application descriptors require nearly as much maintenance as hand-written diagnostic code, or inferred models remain equally accurate across upgrades without explicit semantics, the manifest adds little value.

### 2. Runtime contract validation with explicit confidence loss

**Gap.** A descriptor can be wrong even when it parses and its probe points still exist.

**Mechanism.** Add lightweight validators that watch semantic invariants rather than only event delivery. Examples include generation uniqueness, acquire/release balance ranges, capacity bounds when known, legal transition ordering, and expected coverage of high-level operations. When evidence contradicts the contract, mark affected resource classes as degraded or unknown instead of silently producing normal attribution.

The validator can use whichever observer is cheapest for the application: declared events, compiler hooks, uprobes, eBPF, or a combination. Expensive checks can activate only after a cheap invariant fails.

**Delta.** Post-profiling validation in gigiprofiler removes candidate false positives from observed workloads. This direction treats validation as a persistent production property and makes stale semantics visible to downstream consumers.

**Artifact.** A validation runtime, a confidence model attached to profile records, and fault-injection tests that mutate resource implementations without updating the descriptor.

**Evaluation.** Measure how quickly the validator detects stale contracts, false alarm rate, percentage of corrupted attribution prevented, overhead, and recovery after a descriptor update. Include versions that only rename functions, versions that reorganize code without changing semantics, and versions that truly change ownership or lifecycle semantics.

**Academic value.** This asks how an observability system can know when its own semantic model has stopped being trustworthy.

**Production value.** A stale profiler integration becomes an explicit health problem instead of a hidden source of bad performance conclusions.

**Failure condition.** If invariant checks either miss most semantic drift or alarm on ordinary workload changes, they are not useful as a trust signal.

### 3. A ground-truth benchmark for application-resource profiling

**Gap.** Current profiler evaluations usually know the root cause of selected bugs, but not every resource identity and transition throughout execution.

**Mechanism.** Build a benchmark harness where resource truth is generated alongside the workload. Include fixed pools, elastic caches, bounded queues, reusable object slots, borrowed references, deferred reclamation, sharded ownership, and cross-request interference. Then create controlled transformations such as inlining, wrapper insertion, allocator changes, object-address reuse, and schema-version changes.

**Delta.** The benchmark would test the semantic layer that ordinary CPU-profile and bug-reproduction suites omit. It would also distinguish a detector that finds useful hot sites from one that reconstructs correct resource lifecycles.

**Artifact.** Open workloads, truth traces, mutation/version pairs, and a scoring harness for event and resource semantics.

**Evaluation.** Compare system-only profiling, manual annotations, static user events, dynamic uprobes, automatic inference, and hybrid approaches under the same overhead budget. Report event-site accuracy, identity/lifetime accuracy, root-cause ranking, attribution error, false confidence, maintenance effort, and runtime cost.

**Academic value.** It provides a reproducible measurement problem for semantic observability rather than another collection of profiler anecdotes.

**Production value.** Tool builders can tell whether a new collector or inference model preserves resource meaning before deploying it on production binaries.

**Failure condition.** If performance diagnoses remain correct even when identity and lifecycle reconstruction is deliberately degraded, then the extra semantic machinery is solving a problem that does not materially affect the target decisions.

## What would change this conclusion?

This argument assumes that application-defined resources matter often enough, and evolve often enough, that stale semantic mappings are a practical source of profiling error. The OSDI 2026 evidence establishes that hidden application resources can cause serious real performance problems and that automated discovery can diagnose them. It does not establish that every production profiler needs a persistent resource contract.

A simpler design should win if automatic inference remains accurate across substantial software evolution, if applications already expose stable resource events with sufficient lifecycle semantics, or if operators only need one-off diagnosis against a fixed source revision. In those cases, a versioned manifest and online validator would add maintenance cost without changing the decision.

The strongest experiment is therefore longitudinal. Take several evolving applications, freeze profiler knowledge at version N, and measure how quickly attribution and diagnosis degrade across later commits. Compare no adaptation, rediscovery from scratch, explicit descriptors, and descriptor-plus-validation. If stale models rarely create wrong decisions, the contract is unnecessary. If they do, the profiler should treat semantic model health as part of observability itself.

## References

- Yigong Hu et al., [*Diagnosing Performance Issues in Application-Defined Resources*](https://homes.cs.washington.edu/~baris/public/gigiprofiler.pdf), OSDI 2026.
- Linux kernel documentation, [`user_events`: User-based Event Tracing](https://docs.kernel.org/6.18/trace/user_events.html).
- Linux kernel documentation, [Uprobe-based Event Tracing](https://docs.kernel.org/6.18/trace/uprobetracer.html).
- SystemTap documentation, [Static user-space probe points](https://sourceware.org/systemtap/langrefse4.html#x33-330004.5.7).
- Eunomia Daily Report, [What Must an eBPF Profiler Track Beyond Threads?](https://eunomia.dev/research/async-ebpf-causal-profiler/).
- Eunomia Daily Report, [When Does Profiler Sampling Become Biased?](https://eunomia.dev/research/profiler-sampling-bias/).
