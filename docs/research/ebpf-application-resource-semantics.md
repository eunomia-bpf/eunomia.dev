---
date: 2026-08-21
title: "Can eBPF Understand Application-Defined Resources?"
description: "eBPF can trace internal pools, queues, and caches without rebuilding apps, but raw probes lack semantics. This report develops a versioned resource contract."
tags:
  - Daily Report
  - eBPF
  - Profiling
  - Observability
  - Uprobes
  - USDT
research_question: "How can an eBPF profiler observe application-defined resources without hard-coding stale application semantics, while preserving resource identity, lifetime, and cross-layer effects?"
source_cutoff: 2026-08-21
status: daily-report
---

# Can eBPF Understand Application-Defined Resources?

A MySQL buffer pool can be full of the wrong pages while Linux still reports normal process memory. A work queue can be overloaded while CPU utilization looks modest. A query cache, connection pool, token bucket, temporary table, or application-level credit can decide whether a request is fast even though none of those objects is a first-class OS resource.

That is the uncomfortable boundary for a system profiler: eBPF is very good at observing *what code and the operating system do*, but it does not automatically know *what an application object means*.

<!-- more -->

The OSDI 2026 paper [Diagnosing Performance Issues in Application-Defined Resources](https://www.usenix.org/conference/osdi26/presentation/hu-yigong) makes the visibility gap concrete. Its study found 38 distinct application-defined resources across 45 real performance issues. The paper reduces many resource interactions to four semantic events: `WAIT`, `ACQUIRE`, `USE`, and `RELEASE`. Its profiler, gigiprofiler, combines LLM-based semantic discovery with static validation, instruments the selected locations with an LLVM pass, then uses runtime evidence to diagnose 15 reproduced issues and two previously unknown MariaDB bugs.

The paper also provides an unusually useful warning for any dynamic profiler. LLM-only event discovery has 45–60% false-positive rates in its evaluation. Static validation improves precision, and post-profiling validation reduces false positives again. Yet an exhaustive MySQL buffer-pool study still finds that the resulting pipeline misses about 23% of usage events on average. The measured runtime overhead averages 3.7% and never exceeds 7.8% in the nine cases where end-to-end comparison is meaningful.

This is strong evidence that application-resource semantics are both useful and fallible. It also exposes a narrower eBPF research question: **can the semantic model be separated from the instrumentation mechanism, then checked continuously at runtime rather than compiled into one profiler build?**

This report argues for a **versioned application-resource contract consumed by eBPF**. Static analysis, an LLM, developer annotations, or existing USDT/user-events metadata may propose resource semantics. eBPF then becomes the deployment and validation substrate: attach to existing binaries, carry a typed descriptor into each probe, correlate resource events with scheduler/I/O/memory effects, and explicitly downgrade confidence when the observed runtime no longer satisfies the descriptor.

The target property is not merely “trace four events.” It is **dynamic, no-rebuild semantic instrumentation plus cross-boundary validation against the running system**. If eBPF cannot provide that property more reliably than compile-time instrumentation, the proposed direction should fail.

## What current systems already give us

### gigiprofiler gives a useful semantic event model

gigiprofiler's most reusable idea is not one specific detector. It is the observation that diverse application resources can often be described through a compact interaction vocabulary.

- `WAIT(resource_id, wait)` records a slow acquisition path.
- `ACQUIRE(resource_id, units)` records obtaining capacity or an object.
- `USE(resource_id, use_type, target_id)` records work performed through the resource.
- `RELEASE(resource_id, units)` records capacity or an object returning to the resource.

Those events are expressive enough to expose several pathologies. Long `WAIT` intervals reveal contention. Frequent acquire/release without useful work can reveal an inefficient policy. An unmatched acquire/release balance can reveal unbounded growth or leakage. A resource held by one request can explain why later requests are forced onto a slow path.

The important point is that these are **semantic events**. A function entry is not an `ACQUIRE` event merely because it returns a pointer. A mutex wait is not necessarily waiting for the application resource that matters. The event meaning has to be derived from application behavior.

### Linux already exposes multiple transport mechanisms

Linux has several ways to transport application-level evidence.

[`user_events`](https://docs.kernel.org/6.18/trace/user_events.html) lets a process register typed trace events that existing tools such as ftrace and perf can consume. Current kernels support multiple formats for one logical event name through `USER_EVENT_REG_MULTI_FORMAT`, which is useful when an application's event schema evolves.

[uprobes](https://docs.kernel.org/trace/uprobetracer.html) can observe an existing executable or shared library at selected offsets without requiring the application to emit a trace event. In libbpf, [`bpf_program__attach_uprobe_multi`](https://docs.ebpf.io/ebpf-library/libbpf/userspace/bpf_program__attach_uprobe_multi/) can attach one BPF program to many functions or offsets and assign a different attach cookie to each site. [`bpf_program__attach_usdt`](https://docs.ebpf.io/ebpf-library/libbpf/userspace/bpf_program__attach_usdt/) provides the corresponding path for USDT probes.

The BPF ring buffer adds another useful property: events emitted from BPF programs preserve ordering across CPUs in one shared FIFO. BPF maps can retain compact lifetime and correlation state while a userspace analyzer consumes only the evidence needed for diagnosis.

These interfaces solve **where to observe and how to transport state**. They do not solve **what a pool, queue, cache entry, lease, or credit means**.

### Dynamic attachment is not the same as dynamic semantics

It is tempting to say that uprobes make the problem solved: find the resource's functions and attach BPF programs to them. That only moves the instrumentation point.

Suppose version A of an application uses `pool_get()` to return one reusable object. Version B changes the function so it sometimes returns a borrowed object that must not be counted as an acquisition. The symbol may still exist. A uprobe may still fire. The trace may still look internally consistent. The *semantic contract* is now stale.

A raw address is equally dangerous. If a pointer is reused after an object is destroyed, a profiler that treats the address as a stable resource identifier can merge two lifetimes. The resulting trace is precise at the machine level and wrong at the application level.

So the problem is not just probe placement. It is **semantic versioning and runtime validation**.

## Where current work is still weak

### The semantic model is usually coupled to one build or one analysis run

gigiprofiler performs static analysis for an application version and injects lightweight probes through an LLVM pass. That is reasonable for on-demand diagnosis, but the discovered resource model is tied to the code version that was analyzed and instrumented.

Production observability often has a different constraint: the profiler may need to attach to a binary that was already built, packaged, or supplied by another team. Re-running a compiler pass is not always possible. Even when source is available, fleets can contain several builds at once.

The missing artifact is a portable contract that says *which build this semantic claim applies to*, *where the event lives*, and *how to derive stable resource identity and lifetime from runtime values*.

### A probe can remain valid syntactically after becoming wrong semantically

Symbol resolution, attach success, and argument decoding are weak health checks. They prove that a probe can execute, not that the event still means `ACQUIRE`, `USE`, `WAIT`, or `RELEASE`.

A production profiler therefore needs a semantic health signal. For example, if an `ACQUIRE` descriptor says that a returned object should later appear in `USE`, but a new build produces thousands of acquisitions that are never used and never released, the collector should not silently publish the same interpretation with high confidence.

The correct response may be “model stale,” not “resource leak.”

### System effects and application-resource effects are still separate evidence planes

An application-defined resource matters because it eventually changes real execution. A buffer-pool miss may trigger I/O. A full queue may delay runnable work. A depleted credit may stall a request. A cache eviction may cause page faults or CPU work elsewhere.

The site's earlier [async eBPF profiler report](https://eunomia.dev/research/async-ebpf-causal-profiler/) argued that causal topology must survive thread boundaries. The [page-level memory report](https://eunomia.dev/research/page-level-ebpf-memory-attribution/) argued that memory attribution needs lifetime-aware provenance. Application-resource profiling needs both ideas at once: resource identity must survive its own lifetime, and its events must be connectable to scheduler, I/O, memory, and request-level consequences.

A compiler-inserted event stream can describe application semantics accurately while still missing those system-level effects. A pure system profiler can see the effects while not knowing the resource semantics. The open problem is the join.

### Evaluation usually rewards the final diagnosis, not semantic-contract correctness

A system can produce the correct bottleneck label for the wrong reason. If an event site is stale but correlated with the same workload phase, aggregate diagnosis may still look good.

For a reusable semantic instrumentation layer, evaluation should score at least four separate properties:

1. event-site precision and recall;
2. resource-instance and lifetime identity accuracy;
3. stale-contract detection;
4. end-to-end diagnosis accuracy under a fixed overhead budget.

Without these dimensions, it is hard to tell whether a portable profiler learned a real application resource or merely a workload-specific proxy.

## Promising directions with academic and production value

### 1. A versioned resource-semantics manifest compiled into eBPF attachments

**Gap.** Resource discovery tools can identify candidate semantics, and eBPF can attach dynamically, but there is no common artifact connecting a semantic claim to a specific binary and probe plan.

**Mechanism.** Define a compact `resource.manifest` with entries such as:

```yaml
resource: mysql.buffer_pool.page
build_id: 9f2c…
event: ACQUIRE
site: buf_LRU_get_free_block+0x1a4
instance_key: arg0
unit_key: retval
generation: allocation_epoch
confidence: validated
```

The real schema should also encode resource class, units/capacity, optional `target_id`, expected lifetime transitions, symbol/offset provenance, and the evidence used to derive the descriptor.

A loader resolves the manifest against the target build. Many sites can share one `uprobe_multi` program; attach cookies index the per-site descriptor. USDT sites can use the same semantic descriptor while libbpf handles their argument locations. BPF maps store only the compact state needed to turn raw arguments into stable `(resource_class, instance, generation)` identities.

The loader rejects a build-ID mismatch by default. An operator may allow a best-effort symbol re-resolution mode, but that mode begins with degraded confidence rather than silently inheriting the old semantic status.

**Delta.** Existing semantic profilers discover and instrument events; existing eBPF loaders dynamically attach programs. The new object is a versioned boundary between the two, so resource semantics can be regenerated, audited, distributed, attached, and revoked independently from the application binary.

**Artifact.** A manifest compiler, libbpf loader, reusable BPF event program, and a small analyzer that emits the four canonical resource events with generation-scoped identity.

**Evaluation.** Build manifests for MySQL, PostgreSQL, Apache, and one runtime-heavy application such as llama.cpp. Compare manifest attach coverage and event correctness with compiler-inserted instrumentation and with manually written USDT/user-events probes. Measure load time, event overhead, probe count, and event-site precision/recall.

**Academic value.** This asks whether application semantics can become a versioned interface between program analysis and dynamic instrumentation rather than being embedded in one profiler implementation.

**Production value.** A fleet can deploy resource-aware profiling to existing binaries and can roll semantic descriptors forward or back without rebuilding the target application.

**Failure condition.** If realistic resource events cannot be represented without arbitrary per-application BPF code, the manifest is only a thin configuration wrapper and does not provide a useful abstraction.

### 2. Runtime semantic validation with explicit confidence loss

**Gap.** A dynamically attached probe can continue firing after its meaning changes. Existing attach success and event counts cannot distinguish a real pathology from a stale semantic model.

**Mechanism.** Compile validation invariants alongside each manifest entry. Examples include:

- an acquired instance should belong to an active resource generation;
- a released unit should not remain in the active ownership set;
- a `USE` event should usually refer to an instance previously acquired or otherwise declared externally owned;
- instance keys should not be reused across overlapping generations;
- value distributions and transition ratios should stay within broad training envelopes, with envelopes used only as warning evidence rather than correctness proof.

BPF maps maintain the bounded online state needed for these checks. Violations increment typed counters and optionally emit compact samples through the ring buffer. The userspace controller turns repeated violations into a confidence transition such as `validated -> suspect -> stale`.

The important design choice is **not** to mutate the semantic definition automatically when a violation appears. Runtime evidence says the current contract no longer explains execution; it does not tell us the correct replacement. A new static/LLM analysis or human review must produce the next candidate manifest.

Because the same eBPF program can also observe scheduler, block-I/O, page-fault, and process events, the validator can ask whether a resource event has the cross-layer consequence its model predicts. That makes eBPF more than a transport: it is an independent runtime checker spanning application and OS boundaries.

**Delta.** gigiprofiler already uses post-profiling validation to remove false event candidates. The proposed mechanism promotes that idea into a persistent, versioned health state for each deployed semantic descriptor and combines it with independent OS evidence.

**Artifact.** A BPF validation library plus a controller that records per-descriptor confidence, evidence counts, and the exact invariant that failed.

**Evaluation.** Create version mutations that preserve symbols while changing semantics: borrowed versus owned returns, changed queue capacity units, handle reuse, moved release paths, inlined or split functions, and changed resource ownership. Measure stale-model detection latency, false alarms, and how often the system avoids a wrong diagnosis compared with an attach-only baseline.

**Academic value.** The general systems problem is how to validate inferred semantic instrumentation against a running program without assuming that successful observation implies semantic correctness.

**Production value.** Operators get a visible distinction between “resource contention observed” and “the profiler's model of this resource is no longer trustworthy.”

**Failure condition.** If useful invariants require so much per-resource state that they erase eBPF's deployment/overhead advantage, validation belongs in a heavier application-specific runtime instead.

### 3. A mutation benchmark for semantic profiling

**Gap.** Existing profiler evaluations rarely separate event correctness from diagnosis correctness, and they rarely test how semantic instrumentation survives software evolution.

**Mechanism.** Build a ground-truth benchmark around real application resources. For each resource, define a correct event/lifetime trace and then introduce controlled mutations:

- rename or move the operator function;
- preserve the symbol but change ownership semantics;
- change the unit from objects to bytes;
- reuse handles after destruction;
- add an asynchronous handoff before `USE`;
- create a system-level side effect, such as I/O or scheduler delay, that is caused by one resource event but appears on another thread.

Run four instrumentation strategies against the same workload: compiler-inserted semantic events, explicit `user_events`/USDT, versioned eBPF manifests, and system-only tracing. The benchmark scores event-site accuracy, lifetime identity, stale-model detection, cross-layer causal attribution, final diagnosis, and overhead.

**Delta.** A normal benchmark asks whether a profiler finds known bugs. This one asks whether the profiler knows when its semantic assumptions have stopped matching the program.

**Artifact.** A reproducible suite of application versions, workload generators, ground-truth event traces, and scoring scripts. It can begin with a small set of MySQL buffer-pool mutations and then expand to queues, caches, connection pools, and runtime schedulers.

**Evaluation.** The benchmark is itself the evaluation artifact. Report per-mutation confusion matrices and confidence calibration, not only aggregate diagnosis accuracy.

**Academic value.** It creates a measurable target for semantic observability under software evolution, a property current profiling benchmarks largely hide.

**Production value.** Teams can test whether an observability rule survives a release before rolling it across a fleet.

**Failure condition.** If the benchmark's mutations do not predict failures seen across real version upgrades, it becomes a synthetic stale-probe test rather than a useful semantic-profiler benchmark.

## What would change this conclusion?

The strongest conclusion here is deliberately narrower than “eBPF should replace source instrumentation.” It should not.

If an application already exports stable, well-versioned `user_events`, USDT probes, metrics, or tracing spans with explicit resource identities and lifetimes, those interfaces are better semantic sources than reverse-engineering raw function calls. eBPF can still correlate them with OS effects, but it does not need to rediscover semantics the application already publishes.

The proposed eBPF layer is most valuable when three conditions hold together:

1. the application-defined resource materially controls performance;
2. the running binary does not expose a sufficient stable semantic interface;
3. operators need dynamic attachment and independent system-level correlation without rebuilding the application.

A decisive experiment would compare the versioned eBPF manifest against gigiprofiler's compiler instrumentation across several real software upgrades. If the eBPF path has lower semantic precision, fails to detect stale contracts, or costs similar engineering effort while providing no better cross-layer diagnosis, then compile-time or source-defined instrumentation is the better design.

But if a small semantic manifest can survive deployment boundaries while eBPF verifies its assumptions against real execution, application-defined resources become something system observability can reason about without pretending that a pointer or a function name is the resource itself.