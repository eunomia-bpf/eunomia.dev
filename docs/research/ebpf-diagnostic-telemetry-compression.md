---
date: 2026-08-22
title: "Can eBPF Compress Telemetry Without Losing the Diagnosis?"
description: "Always-on eBPF telemetry can overwhelm storage or discard the evidence needed for diagnosis. This report develops diagnostic-preserving semantic compression."
tags:
  - Daily Report
  - eBPF
  - Observability
  - Profiling
  - Telemetry
research_question: "How can an always-on eBPF observability system reduce telemetry volume before export while preserving the evidence required for later diagnosis?"
source_cutoff: 2026-08-22
status: daily-report
---

# Can eBPF Compress Telemetry Without Losing the Diagnosis?

An always-on eBPF profiler can see a syscall, scheduler transition, allocation, page fault, queue operation, or application probe every time it happens. That is useful until the workload becomes busy. Export every event and the collector spends bandwidth, memory, storage, and analyst attention on repeated detail. Aggregate too aggressively and the incident that happens an hour later may ask a question the retained counters can no longer answer.

This is the hard part of eBPF telemetry compression: the system has to discard information *before it knows which future diagnosis will matter*.

<!-- more -->

Linux already gives eBPF two strong building blocks for reducing export volume. [BPF maps](https://docs.kernel.org/bpf/maps.html) keep state close to the event source, including arrays, hashes, per-CPU variants, queues, stacks, Bloom filters, and other specialized structures. The [BPF ring buffer](https://docs.kernel.org/bpf/ringbuf.html) transports variable-length records efficiently and preserves reservation order across CPUs in one shared buffer. A BPF program can therefore aggregate thousands of events into a few map entries and emit only selected records.

But those primitives do not define which reductions are safe for diagnosis. A counter can tell us that 8,000 waits occurred without retaining which resource generation was held when the long wait began. A histogram can preserve a latency distribution while erasing the state transition that made one tail request different. A top-k map can retain the busiest keys while deleting a rare key that later turns out to be the failed dependency.

This report argues for **diagnostic-preserving semantic compression**: an eBPF collector should declare which questions its compact representation can still answer, keep bounded raw exemplars for transitions that summaries cannot reconstruct, and attach explicit coverage metadata to every summary. Compression ratio is a secondary metric. The primary metric is whether an investigator reaches the same correct diagnosis under a fixed collection budget.

That is narrower than the earlier [agent trace evidence-budget report](https://eunomia.dev/research/agent-trace-evidence-budget/). That report asked how a whole agent-observability system should divide retention across representative sampling, incidents, causal anchors, and a flight recorder. Here the question sits lower in the stack: **before high-rate system telemetry leaves an eBPF collector, what can be summarized, what must remain as an exemplar, and how can downstream analysis know what was lost?**

## Why eBPF telemetry compression is not ordinary compression

There are at least three different ways to make observability data smaller, and they preserve different properties.

### Lossless representation compression preserves records

[μSlope](https://www.usenix.org/conference/osdi24/presentation/wang-rui) attacks redundancy in semi-structured logs. Its evaluation reports 21.9:1 to 186.8:1 compression, up to 2.34 times the compression ratio of Zstandard, while allowing search without full decompression. [Tracezip](https://arxiv.org/abs/2502.06318) applies a similar observation to distributed traces: repeated span structure can be represented once and reconstructed at the backend through a Span Retrieval Tree.

These systems are valuable because they preserve the original logical records. The backend can still reconstruct fields that were compressed away on the wire or disk.

That is different from source-side eBPF aggregation. If a BPF program replaces 100,000 `(pid, resource, latency)` events with one histogram, no codec can later recover which individual event belonged to which resource generation. The information was not encoded more efficiently. It was removed.

### Sampling preserves examples, not complete coverage

[OpenTelemetry](https://opentelemetry.io/docs/concepts/sampling/) distinguishes head sampling from tail sampling. Head sampling decides early and cheaply but cannot use the whole trace to select important cases. Tail sampling can inspect most or all spans and retain errors or other interesting traces, but the collection pipeline still handles the spans before the decision.

For high-rate eBPF events, moving the decision earlier is exactly what saves overhead. It is also what makes the decision dangerous. Once a kernel-side program drops an event or collapses it into a summary, a later tail sampler cannot request the missing fields.

Sampling also has a statistical contract that aggregation may not. If every event has a known independent sampling probability, population estimates can carry uncertainty. A hand-written map that keeps "interesting" keys and silently evicts others has no such simple interpretation.

### Workload-aware tracing preserves selected structure

[StriaTrace](https://www.usenix.org/conference/osdi26/presentation/wu-haonan) shows how much can be saved when the system knows which execution structure matters. For online LLM inference it traces key synchronization points, follows critical paths, and enables detailed tracing during abnormalities. The paper reports a 97.8% tracing-overhead reduction relative to alternatives and hundreds of diagnosed abnormalities across 19 root causes.

The result is important because it optimizes *what to observe*, not just how to encode observations. It also depends on a workload model. Synchronization points and inference critical paths are meaningful because StriaTrace knows the serving architecture.

A general eBPF observability layer sees a wider range of programs and questions. It needs a way to express such workload-specific retention choices without hard-coding one diagnosis into every probe.

## The kernel primitives already expose the tension

The Linux interfaces make the engineering trade-off concrete.

A BPF ring buffer is non-blocking. When there is no room, reservation fails. In NMI context, reservation can also fail because the internal lock is unavailable even when the buffer is not full. `bpf_ringbuf_query()` can expose available data and producer/consumer positions, but the kernel documentation describes those values as momentary snapshots useful for debugging, reporting, or heuristics rather than stable truth.

That means an exporter can be both fast and incomplete. If the consumer only sees committed records, it needs separate accounting to distinguish "nothing happened" from "the producer could not reserve evidence."

Maps have a different failure mode. They are excellent for compact state, but their semantics depend on the map type and the program that updates it. Per-CPU maps avoid some cross-CPU synchronization; LRU variants can evict entries when capacity is reached. A map value can be exact for the state it represents while still being incomplete for the future question.

The missing abstraction is therefore not another map type or another compression codec. It is a contract between **diagnostic questions** and **retained telemetry**.

## A diagnostic contract for source-side compression

Consider an operator who wants an always-on eBPF collector for application queues and scheduler effects. Raw events might include:

```text
enqueue(queue_id, item_id, generation, ts)
dequeue(queue_id, item_id, generation, ts)
sched_in(pid, cpu, ts)
sched_out(pid, cpu, state, ts)
complete(item_id, status, ts)
```

The collector cannot keep every record indefinitely. Before writing BPF code, it should state what later questions must remain answerable:

1. Which queue generations experienced long residence time?
2. Did a slow item wait in the queue, wait to be scheduled, or execute slowly?
3. How many items were dropped or observed incompletely?
4. Can one rare failed item be reconstructed far enough to inspect its transition sequence?

Those are **diagnostic obligations**. They can drive a compact representation:

- per-generation counters for enqueue/dequeue/completion balance;
- latency histograms for queue residence and runnable delay;
- bounded maps for active item generations;
- a small set of raw exemplars for first occurrence, state transitions, outliers, and invariant violations;
- coverage counters for eligible events, successful updates, evictions, ring-buffer reservation failures, and probe/schema generation.

The key point is that the representation comes with a machine-readable answerability statement. A query may be `exact`, `estimated with a declared sampling rule`, `bounded by retained exemplars`, or `unavailable`. A dashboard should not present all four as equally certain numbers.

## Where current work is still weak

### 1. Compression systems optimize bytes, while profilers need to optimize answerability

Lossless systems such as μSlope and Tracezip can measure compression ratio because the logical input remains reconstructible. Source-side aggregation is different. The important question is which diagnoses survive the reduction.

Most eBPF collectors define this informally in code. One program stores a histogram, another a counter keyed by PID, another emits a record only above a threshold. The implementation may be efficient, but there is no common artifact saying which future queries remain valid after the reduction.

The missing capability is an explicit mapping from diagnostic obligations to retained state. A useful evaluation would replay incidents with known root causes and compare diagnoses from the raw trace against diagnoses from each compact representation under the same CPU, memory, and export budget.

### 2. Loss is often recorded separately from the summary it invalidates

Ring-buffer reservation failure, map eviction, unavailable probes, schema mismatches, and collector restarts all change what a summary means. Yet telemetry pipelines often expose the resulting counter or histogram without attaching those coverage conditions to the value.

Suppose a per-resource latency histogram contains 95% of events but the missing 5% occurred exactly when the ring buffer was saturated. Treating that histogram as an unbiased view can be worse than showing no value at all.

The missing mechanism is **coverage-carrying telemetry**: every compact result should identify the collection generation and the relevant evidence-loss counters. An experiment can inject controlled ring-buffer pressure, map eviction, probe removal, and process restarts, then measure whether diagnosis confidence degrades in proportion to actual error.

### 3. A trigger cannot recover context that was already summarized away

Detailed tracing only during abnormalities is attractive because normal execution dominates most fleets. StriaTrace shows that workload-aware escalation can work very well.

The boundary appears when the trigger occurs after the decisive transition. A five-second latency anomaly might have been caused by a queue ownership change thirty seconds earlier. Enabling raw tracing after the tail request becomes slow does not restore the old ownership event.

The missing capability is a small, bounded **pre-trigger exemplar history** attached to semantic state, not simply a time-based full-event buffer. It should retain the transitions most likely to explain a future change of state while allowing repetitive steady-state events to collapse into aggregates.

### 4. Hand-written aggregation does not tell us when a different representation would have been better

BPF maps make it easy to build one efficient summary. They do not tell the collector that its current key cardinality has exploded, its top-k set has become unstable, or an invariant is failing often enough that raw exemplars are now more valuable than another counter update.

This is the boundary with the next research question in the series. Before an adaptive collector can safely change fidelity, it needs a stable contract describing what each representation preserves. Otherwise "adaptive observability" is just a collection of heuristics with no way to tell whether the new mode still supports the same diagnosis.

## Promising directions with academic and production value

### 1. Compile diagnostic obligations into an eBPF retention plan

**Gap.** Today an operator normally chooses BPF maps, filters, thresholds, and emitted fields by hand. The resulting program has no machine-readable statement of which diagnoses its summaries support.

**Mechanism.** Define a small diagnostic contract that lists required entities, transitions, joins, accuracy bounds, and allowed approximations. A compiler maps that contract onto a retention plan:

```yaml
question: queue_delay_attribution
entities:
  - queue_generation
  - item_generation
required_transitions:
  - enqueue
  - dequeue
  - sched_in
  - complete
outputs:
  queue_latency:
    mode: histogram
    exact_count: true
  slow_item_examples:
    mode: bounded_exemplars
    retain: first,last,outlier,invariant_violation
coverage:
  track:
    - eligible_events
    - map_evictions
    - ringbuf_reservation_failures
```

The compiler chooses BPF map layouts, per-CPU versus shared state, export records, and userspace reconstruction logic. It also emits a query manifest describing whether each supported answer is exact, estimated, exemplar-backed, or unavailable.

**Delta.** Tracezip and μSlope compress records while preserving reconstructibility. Existing eBPF tools aggregate at the source but leave the diagnostic contract implicit. This direction makes *answerability after reduction* the explicit compiled property.

**Artifact.** A small compiler, reusable libbpf/BPF templates, a query manifest, and a replay harness that can compare raw and compact traces.

**Evaluation.** Use scheduler, network, memory, and application-resource incidents with ground-truth causes. Give raw export, hand-written eBPF aggregation, probabilistic sampling, and the compiled plan the same CPU, map-memory, and byte budget. Measure root-cause accuracy, false attribution, query coverage, export bytes, BPF runtime cost, and map pressure. Ablate the contract compiler by replacing it with manually selected aggregates.

**Academic value.** The general question is whether observability reduction can preserve a declared set of diagnostic properties rather than merely minimizing data volume.

**Production value.** Teams could state what their always-on monitor must still answer and generate a bounded collector instead of maintaining a separate ad hoc BPF program for every dashboard.

**Failure condition.** If a small contract cannot express realistic diagnoses without embedding most application logic, hand-written collectors remain simpler and the abstraction does not earn its complexity.

### 2. Keep state-transition exemplars beside compact summaries

**Gap.** Aggregates preserve steady-state statistics well but can erase the sequence that explains a rare transition. Triggered tracing may start too late.

**Mechanism.** Maintain two levels of evidence in the eBPF data path. The first level is the normal compact map state: counters, histograms, active generations, and low-cardinality summaries. The second is a bounded exemplar cache keyed by semantic entity or generation.

An exemplar is retained only when it adds transition information: the first event in a generation, the last event before retirement, an invariant violation, a category change, a threshold crossing, or a statistically rare outlier. Repeated steady-state events update aggregates without allocating another raw record. When userspace requests escalation or detects an incident, the collector exports the retained exemplars plus current summaries.

The cache can use bounded maps for entity-local state and the ring buffer for committed capsules. Retention policy must be explicit about eviction so an absent exemplar is never interpreted as proof that a transition did not occur.

**Delta.** A conventional flight recorder keeps a recent time window of raw events. This design keeps a sparse history of **state changes**, so old but semantically important transitions can survive while repetitive newer events disappear.

**Artifact.** An exemplar library for common eBPF observability patterns plus a userspace capsule format that joins summaries with retained transitions.

**Evaluation.** Build incidents where the causal transition occurs 1, 10, and 60 seconds before the visible symptom. Compare full raw export, a fixed-size time ring, tail-triggered tracing, and transition exemplars at equal memory and export budgets. Measure whether the true transition survives, root-cause accuracy, false explanations, and overhead.

**Academic value.** This tests whether semantic change is a better retention unit than recency for online system diagnosis.

**Production value.** Always-on monitors could retain enough pre-incident context to explain rare failures without storing a continuous raw event stream.

**Failure condition.** If a simple time-based ring preserves the same diagnostic context at comparable memory across diverse workloads, semantic exemplar selection is unnecessary.

### 3. Make every compressed result carry its evidence coverage

**Gap.** A compact value often looks exact even when the evidence feeding it was incomplete.

**Mechanism.** Give every summary a collection generation and a compact coverage record. Depending on the plan, that record can include:

- number of eligible events;
- number incorporated into the summary;
- known probabilistic sampling rate;
- ring-buffer reservation failures;
- map insert failures or evictions relevant to the key space;
- probe/schema generation;
- collector restart epoch;
- counts of invariant violations or unknown entity identities.

Userspace joins the value with this coverage record before reporting it. A diagnosis can then say "queue residence increased with 99.8% observed event coverage" or "attribution unavailable because the active-item map evicted 18% of generations" instead of publishing a precise-looking number with hidden loss.

The first version should remain deliberately non-adaptive. It records when the representation became less trustworthy. A later controller can use that signal to decide whether and how to increase fidelity.

**Delta.** Sampling systems record probability so aggregate estimates can reason about uncertainty. eBPF monitoring needs a broader notion of coverage because loss can come from buffer pressure, bounded state, missing probes, restarts, and semantic mismatch rather than one sampling probability.

**Artifact.** A common coverage schema, BPF-side accounting helpers, and downstream libraries that propagate coverage into queries and alerts.

**Evaluation.** Inject controlled loss through ring-buffer saturation, LRU eviction, probe disabling, restarts, and schema changes. Measure calibration between reported coverage and actual query error, plus whether coverage-aware diagnosis abstains from false root causes more often than an identical collector without coverage metadata.

**Academic value.** This turns observability completeness into an explicit measurable property of a compressed representation.

**Production value.** Operators can distinguish "the metric is normal" from "the collector lacks enough evidence to decide," which is especially important for rare incidents.

**Failure condition.** If coverage metadata does not predict diagnosis error or only repeats information downstream systems already infer reliably, the extra accounting is not worth the hot-path cost.

## A benchmark should score diagnosis retention, not compression ratio

A useful benchmark needs raw ground truth and an equal resource budget. It should include workloads where different representations win.

For each incident, collect a full reference trace outside the budgeted path. Then run the candidate collectors with fixed limits on BPF CPU time, map memory, exported bytes per second, and userspace processing. Ask a set of diagnosis queries that require different information:

| Incident | Required evidence | Easy summary | Information that can be lost |
| --- | --- | --- | --- |
| queue buildup | enqueue/dequeue generations | residence histogram | rare ownership transition |
| scheduler delay | runnable and scheduled intervals | per-process delay total | which queue item was blocked |
| memory regression | allocation/page generations | bytes per stack | COW/reclaim transition |
| network retry storm | connection/request generations | retry counter | first failed dependency |
| stale semantic probe | probe generation and invariants | event rate | evidence that meaning changed |

The benchmark should report at least:

- root-cause accuracy and false attribution;
- fraction of required diagnosis queries still answerable;
- calibration between stated coverage and actual error;
- raw-event bytes and exported bytes;
- BPF execution overhead and map memory;
- time to retrieve enough evidence for diagnosis.

A collector that compresses 1000:1 but fails the one query that distinguishes queue delay from scheduler delay is not better than a 20:1 design that preserves the needed transition. Conversely, a design that retains elaborate exemplars but gives no diagnosis benefit over a histogram has spent complexity without earning it.

## What would change this conclusion?

The proposal assumes that source-side reduction is necessary because raw event export is expensive enough to constrain always-on deployment, and that future diagnosis questions have a stable core that can be declared in advance.

Two results would weaken that assumption.

First, if lossless systems can encode and export the complete relevant eBPF event stream at similar CPU, memory, and bandwidth cost to semantic aggregation, preserving raw evidence is simpler. Tracezip and μSlope show why this is a serious baseline rather than a straw man.

Second, if real incidents routinely require fields and relationships that no practical diagnostic contract predicts, source-side semantic reduction may be too brittle. The safer design would retain a larger probability-known raw sample or more continuous flight-recorder state.

The decisive experiment is an equal-budget incident replay across several domains. If hand-written aggregates, probability sampling, or lossless compression match the proposed contract on root-cause accuracy and query coverage, there is no reason to add a compiler or exemplar layer.

If the contract consistently preserves diagnosis at lower export cost, then eBPF observability has a useful target beyond "emit fewer events": **discard repetition while making the loss itself inspectable**.

## References

- [Linux kernel documentation: BPF maps](https://docs.kernel.org/bpf/maps.html)
- [Linux kernel documentation: BPF ring buffer](https://docs.kernel.org/bpf/ringbuf.html)
- [OpenTelemetry: Sampling](https://opentelemetry.io/docs/concepts/sampling/)
- [Tracezip: Efficient Distributed Tracing via Trace Compression](https://arxiv.org/abs/2502.06318)
- [StriaTrace: Efficient Tracing and Diagnosis for Online LLM Inference](https://www.usenix.org/conference/osdi26/presentation/wu-haonan)
- [μSlope: High Compression and Fast Search on Semi-Structured Logs](https://www.usenix.org/conference/osdi24/presentation/wang-rui)
- [Can eBPF Understand Application-Defined Resources?](https://eunomia.dev/research/ebpf-application-resource-semantics/)
- [What Must an eBPF Profiler Track Beyond Threads?](https://eunomia.dev/research/async-ebpf-causal-profiler/)
- [When Does Profiler Sampling Become Biased?](https://eunomia.dev/research/profiler-sampling-bias/)
- [What Should an AI Agent Trace Keep?](https://eunomia.dev/research/agent-trace-evidence-budget/)
