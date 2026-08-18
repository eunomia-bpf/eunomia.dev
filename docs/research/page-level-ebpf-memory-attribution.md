---
date: 2026-08-18
title: "Can eBPF Attribute Memory to the Pages That Actually Matter?"
description: "Memory allocation, RSS, and page hotness answer different questions. This report asks how eBPF can connect allocation provenance to the pages actually used."
tags:
  - Daily Report
  - eBPF
  - Memory Profiling
  - Linux Memory Management
  - Observability
  - Performance
research_question: "How can an eBPF-based memory profiler connect application allocation provenance to the virtual and physical pages that are actually touched, faulted, reclaimed, migrated, or responsible for memory traffic without pretending that every page has one stable owner?"
source_cutoff: 2026-08-18
status: daily-report
---

# Can eBPF Attribute Memory to the Pages That Actually Matter?

A service reserves a 64 GiB virtual arena at startup. During a normal request burst it touches only 3 GiB, keeps 2 GiB resident, faults some cold pages back in, shares file-backed pages with helper processes, and periodically moves pages between NUMA nodes. A heap profiler says which call sites requested bytes. `/proc/<pid>/smaps` says how much of each mapping is resident. DAMON can estimate which address ranges are hot. Hardware sampling can show expensive memory accesses.

Those answers are all useful, but they answer different questions.

When an operator sees high RSS, reclaim pressure, or memory bandwidth, the question is often more specific: **which application allocation or logical resource is responsible for the pages that the machine is actually paying for now?** An allocation stack may reserve bytes that are never touched. A resident page may have been faulted by a different thread than the one that created the region. A physical page can move, split, become shared through copy-on-write, or belong to the page cache. A sampled load can identify an address without explaining which allocator object or runtime resource created that address.

<!-- more -->

This report argues that eBPF memory profiling needs a provenance layer between allocation tracing and page activity. The useful unit is not simply "bytes allocated" or "pages resident." It is a **lifetime-aware chain from an application allocation to a virtual-region generation, then to page or folio generations and observed access or lifecycle events**.

Linux already exposes most of the ingredients separately. The missing mechanism is a reliable join across them, plus an evaluation that distinguishes exact attribution from sampling and inference.

This is the first report in the **eBPF Observability and Profiling** series. It builds on the existing [eBPF memleak tutorial](https://eunomia.dev/tutorials/16-memleak/), which traces allocation and free call sites, but asks a different question: after memory is allocated, which part of it becomes real working-set cost?

## Allocation, residency, and use are different measurements

An allocation tracer observes intent. The Eunomia `memleak` example uses uprobes around `malloc`, `calloc`, `realloc`, `mmap`, and their release paths, records addresses and sizes, and groups outstanding allocations by stack. That is a good way to find leaked or unreleased allocations.

It cannot tell whether all those bytes were faulted into memory.

Linux exposes residency from another layer. The [`/proc` documentation](https://docs.kernel.org/filesystems/proc.html) distinguishes virtual size from `VmRSS`, and `smaps` provides per-mapping memory accounting. The same documentation notes that ordinary RSS accounting is asynchronous and that scanning `smaps` can provide a more precise snapshot at higher cost. Residency therefore answers "what is mapped into the resident set now," not "which allocation site made these pages hot" or "which page caused bandwidth."

Linux also has mechanisms that observe use rather than allocation. [Idle page tracking](https://docs.kernel.org/admin-guide/mm/idle_page_tracking.html) marks pages idle and clears that state when the kernel observes references; it can estimate a workload's working-set size. [DAMON](https://docs.kernel.org/mm/damon/design.html) samples access frequency and age, grouping neighboring pages into regions so overhead stays bounded. DAMON explicitly trades spatial precision for controllable cost, and its documentation warns that region-based sampling loses quality when pages grouped into one region do not share the same access pattern.

At the hardware level, `perf mem` and memory-mode reporting can sample data addresses, memory latency, TLB and cache information, and on supported configurations physical data addresses. These samples get closer to the cost of actual loads and stores, but remain samples and are architecture-dependent.

The resulting measurement stack looks like this:

| Layer | Useful question | What it does not establish by itself |
| --- | --- | --- |
| allocator / `mmap` tracing | Who requested this address range and size? | whether the bytes became resident or hot |
| `/proc/<pid>/smaps` | Which mappings are resident and how are they accounted? | allocation-site provenance and access intensity |
| idle page tracking | Which pages were referenced during an interval? | which application allocation or object should own the cost |
| DAMON | Which address regions are frequently or recently accessed? | exact per-page provenance under every access pattern |
| `perf mem` / PMU sampling | Which sampled accesses were expensive and where? | complete coverage or a stable allocation identity |
| page lifecycle tracepoints | What did the kernel allocate, reclaim, or migrate? | the application object that ultimately caused the lifecycle |

A profiler that reports only one column from this table will mislead some workloads.

## Physical page identity is not application ownership

It is tempting to join everything by PFN. Linux exposes PFNs through several diagnostic interfaces, and the `mm_page_alloc` tracepoint records a PFN for page allocation. The kernel's [page_owner](https://docs.kernel.org/mm/page_owner.html) facility can store the stack and allocation metadata for physical pages.

But page-owner information describes who allocated the **kernel page**, not necessarily which userspace `malloc` call should receive the cost. A file-backed page can be shared by several processes. Anonymous memory can be shared after `fork` until copy-on-write. Transparent huge pages combine many base pages into a folio and can later split. Reclaim can remove a resident page and a later fault can create a different physical backing for the same virtual address. Migration can move data to another page while the application allocation remains the same.

Current tracepoints expose parts of those transitions but not one universal identity stream. [`mm_page_alloc`](https://github.com/torvalds/linux/blob/master/include/trace/events/kmem.h) reports PFN, order, GFP flags, and migration type. [`mm_vmscan_write_folio`](https://github.com/torvalds/linux/blob/master/include/trace/events/vmscan.h) exposes a folio PFN during reclaim writeback, while other reclaim events are aggregate. [`mm_migrate_pages`](https://github.com/torvalds/linux/blob/master/include/trace/events/migrate.h) reports counts, mode, and reason for a migration batch rather than a per-page old-to-new mapping.

So a useful attribution model should not make PFN the root identity. The stable application-side identity is closer to:

```text
(process or address-space identity,
 allocation or mapping identity,
 generation,
 virtual interval)
```

Physical backing is an edge attached to that identity for a bounded lifetime.

This matters for correctness. If a profiler says "allocation stack A owns 8 GiB of bandwidth" because a sampled physical address once belonged to A, it can silently misattribute after unmap, reuse, copy-on-write, or migration. Generation boundaries are not bookkeeping detail; they prevent stale identities from surviving address reuse.

## The join should preserve ambiguity instead of inventing ownership

Memory can have more than one legitimate consumer. Shared libraries, shared memory, page cache, deduplicated pages, and copy-on-write all break the idea that every page has one owner.

A better model is a small provenance graph:

```text
allocation / resource
        |
        v
virtual interval generation
        |
        +---- exact mapping/fault edge ----> page or folio generation
        |                                    |
        |                                    +---- reclaim / migration lifecycle
        |
        +---- sampled access edge ---------> access weight / latency / bandwidth
```

Edges should carry their evidence type. An allocator uprobe can establish an exact address interval at one instant. A page fault hook can establish that a virtual address acquired backing. DAMON can contribute sampled access weight for a region. PMU memory sampling can contribute sampled loads or stores. A shared mapping can attach one page generation to several virtual intervals rather than force a single owner.

The final report can then distinguish:

- **reserved bytes**, attributed to allocation or mapping intent;
- **resident bytes**, attributed to current mappings;
- **touched working-set bytes**, based on access evidence over a time window;
- **reclaim and migration activity**, attached to page lifecycle;
- **sampled memory cost**, such as latency or access weight;
- **unattributed or multiply attributed cost**, kept explicit when evidence is incomplete.

This is more useful than forcing every metric into a single "memory usage" total.

## Where current work is still weak

### Allocation profilers stop before the VM lifecycle

Heap and eBPF allocation profilers can explain outstanding allocations very well. Their natural key is an address plus size and stack. Once a page is faulted, reclaimed, migrated, remapped, or shared, that allocation record is no longer enough to explain machine cost.

The missing capability is a maintained relation between allocation intervals and VM/page lifecycle. The relation has to survive address reuse and represent shared backing without inventing one owner.

A direct test is a reserve-versus-touch workload: reserve tens of gigabytes, touch a controlled subset, reclaim part of it, then reuse the virtual range for a different allocation. A correct profiler should keep reserved, resident, touched, and later-reused bytes separate.

### Page activity tools lack application allocation provenance

Idle page tracking and DAMON answer important working-set questions. DAMON in particular gives a tunable overhead/accuracy trade-off and can observe access frequency and age over address regions. These tools intentionally focus on memory behavior, not the userspace call site or logical runtime object that created each byte.

For production debugging, the missing join is often the important part. "This 2 GiB region is hot" is less actionable than "this hot region came from cache shard creation at stack X, and 70% of its touched pages belong to shards that have not served a request in ten minutes."

The test is whether adding allocator/runtime provenance changes an optimization decision compared with DAMON-only region statistics. If it does not, the extra tracing is unnecessary.

### Reclaim and migration evidence is not a complete per-page lineage API

Linux tracepoints provide valuable VM events, but they were not designed as one normalized page-lineage protocol. Some events carry PFNs, some carry aggregate counts, and internal hooks can change across kernel versions.

An eBPF profiler can attach to additional BTF-visible functions when necessary, but a production design must label those hooks by stability level and degrade gracefully when exact lineage is unavailable. Otherwise a profiler that works on one kernel may silently produce a different attribution model on another.

The discriminating test is cross-version replay: run the same memory workload across supported kernels and require the profiler to report either comparable attribution or an explicit loss of evidence, never a fabricated exact result.

### Memory bandwidth attribution is sampled and hardware-dependent

Hardware memory sampling is powerful because it observes actual accesses rather than allocation intent. It is also incomplete by construction. Supported load/store events, physical-address availability, sampling skid, and memory-source fields vary by architecture.

A useful profiler therefore needs confidence and coverage metadata for bandwidth attribution. A sampled byte or latency weight should not be presented as an exact counter unless the underlying event provides exact accounting.

The test is to compare sampled attribution with a controlled memory generator whose address ranges and access counts are known, across sequential, random, NUMA-local, and NUMA-remote patterns.

## Promising directions with academic and production value

### 1. A lifetime-aware allocation-to-page provenance ledger

**Gap.** Existing tools observe allocation, residency, and VM events separately.

**Mechanism.** Build an eBPF collector plus a userspace join engine. Uprobes or runtime probes record allocator objects and their virtual intervals. System-call and VM hooks record `mmap`, `brk`, `munmap`, faults, and relevant mapping transitions. Every virtual interval receives a generation so address reuse creates a new identity. When exact physical backing is observable, the join engine creates a bounded page/folio-generation edge. Shared and copy-on-write mappings create multiple or successor edges instead of overwriting ownership.

The data model should distinguish evidence that is exact from evidence that is inferred or sampled. It should also expire page identities aggressively after unmap, migration, reclaim, and process exit.

**Delta.** This extends allocation tracing beyond "allocated but not freed" without trying to replace DAMON or page_owner. DAMON remains the access monitor; page_owner remains useful for kernel page-allocation provenance. The new artifact is the cross-layer join that attaches those facts back to application allocations.

**Artifact.** An open schema, a CO-RE BPF collector for stable tracepoints and selected BTF hooks, allocator adapters for glibc and one managed runtime, and a userspace reconstruction library that outputs folded stacks or pprof labels for reserved, resident, touched, reclaimed, and migrated bytes.

**Evaluation.** Use deterministic microbenchmarks for reserve-versus-touch, `malloc` arenas, file mappings, `fork` plus copy-on-write, transparent huge pages, reclaim, swap, and NUMA migration. Compare against allocation-only eBPF tracing, `smaps`, page_owner, and DAMON. Measure attribution precision/recall against ground truth, event loss, CPU overhead, BPF-map memory, and reconstruction cost.

**Academic value.** The general question is how to maintain correct resource provenance when virtual identity is stable longer than physical backing.

**Production value.** An SRE could move from "this process has high RSS" to the allocator stack or logical cache/resource that produced the working set.

**Failure condition.** If DAMON regions plus ordinary allocation stacks make the same decisions with much lower overhead across realistic workloads, a page-lineage ledger is not worth operating.

### 2. Access-weighted attribution with explicit confidence

**Gap.** Residency assigns equal weight to a cold page and a page generating millions of memory accesses.

**Mechanism.** Join the provenance ledger with access evidence from DAMON and architecture-supported `perf mem` sampling. DAMON contributes region-level frequency and age. PMU samples contribute address, latency, memory-source, or physical-address information when available. The join engine attributes weights to allocation/resource identities while carrying sampling rate, unsupported fields, lost samples, and ambiguity.

The output should be a distribution, not false precision. For example, "cache shard allocation stack X accounts for 31% of sampled remote-NUMA load latency, with the corresponding sampling uncertainty" is a better contract than "X used 31% of memory bandwidth" when only sampled loads were observed.

**Delta.** Conventional heap profiles weight by allocated bytes. DAMON weights address regions by observed access. `perf mem` weights sampled data accesses. The new part is provenance-preserving fusion with explicit coverage semantics.

**Artifact.** A DAMON adapter, a perf-event/BPF sampling adapter, and a profiler view that can pivot the same logical allocation by reserved bytes, resident bytes, access frequency, sampled latency, and NUMA locality.

**Evaluation.** Use STREAM-like bandwidth workloads, random pointer chasing, mixed hot/cold caches, and NUMA placement workloads. Compare attribution against known address-range generators and hardware counters where they provide a suitable aggregate baseline. Vary sampling rates and DAMON region limits to draw accuracy-overhead curves.

**Academic value.** This creates a measurable model for when sampled access evidence changes memory attribution enough to justify its cost.

**Production value.** It can separate a large cold cache from a smaller hot structure that is actually causing remote-memory or bandwidth pressure.

**Failure condition.** If sampling bias or missing hardware support makes attribution unstable across machines, the system should fall back to region-level working-set reporting rather than claim portable bandwidth attribution.

### 3. A ground-truth benchmark for memory attribution

**Gap.** Memory profilers are usually evaluated on overhead and whether a known leak or hot region appears. That is not enough to compare provenance accuracy across allocation, residency, access, reclaim, and migration.

**Mechanism.** Build workloads that expose the true mapping from logical resource to virtual interval and controlled page activity. The harness records a private oracle while the profiler sees only normal tracing interfaces. Cases should include untouched reservation, sparse touch, free/reuse, shared file mappings, `fork` and copy-on-write, THP split/collapse, reclaim/refault, NUMA migration, and mixed resources in one allocator arena.

**Artifact.** A reproducible benchmark suite and trace corpus with a common output schema for allocation identity, page-state transitions, and attributed cost.

**Evaluation.** Score byte-level or page-level precision and recall where exact ground truth exists; for sampled access metrics, score estimation error and interval coverage. Include event-loss injection and kernel-version variation. The simplest baselines should be allocation-only tracing, `smaps`, idle-page working-set measurement, and DAMON.

**Academic value.** The benchmark turns "better memory attribution" into a falsifiable systems property rather than a more detailed visualization.

**Production value.** Profiling tools can state which workloads they attribute correctly and where they degrade to estimates.

**Failure condition.** If the benchmark shows that page-level lineage adds little diagnostic accuracy beyond VMA-level provenance plus DAMON, future work should optimize the cheaper VMA-level design instead.

## The first implementation should avoid tracking every page all the time

A naive design would record every allocation, every fault, every reclaim decision, every migration, and every memory sample. That would recreate the observability problem inside the profiler.

A practical first implementation should keep exact metadata at coarse semantic boundaries and spend page-level budget only where evidence says it matters:

1. track allocation and mapping generations continuously because they define provenance;
2. use `smaps`, DAMON, or pressure signals to identify interesting regions;
3. enable finer page/fault/access capture for those regions or during a bounded diagnostic window;
4. preserve lost-event and sampling metadata in the result;
5. retire fine-grained state when the region cools or the investigation ends.

This follows the same principle as the earlier [async causal profiling report](https://eunomia.dev/research/async-ebpf-causal-profiler/): topology-defining evidence deserves a different budget from expensive context. Here the topology is allocation-to-region identity; page activity is the expensive detail.

## What would change this conclusion?

The strongest counterargument is that Linux already exposes enough at VMA granularity. If allocator stacks can be joined to VMAs and DAMON region statistics, and that combination diagnoses real memory incidents as well as per-page lineage, then page-level provenance is unnecessary complexity.

Another counterexample is a workload dominated by file cache or globally shared memory. There may be no useful single application allocation to charge. In those cases the right identity may be inode, memcg, shared-memory object, or another resource rather than an allocator stack. A general system must allow those identities instead of forcing heap ownership.

Finally, hardware access sampling may be too platform-specific for portable bandwidth attribution. The core provenance ledger still has value if it can explain reserved, resident, touched, reclaimed, and migrated memory, but sampled bandwidth should remain an optional evidence source with an explicit support matrix.

The proposal is worth building only if a ground-truth benchmark shows that the cross-layer join changes diagnoses or optimization decisions enough to justify its overhead. Until then, the right design is incremental: preserve allocation and mapping identity first, attach page-level evidence when it is available, and keep uncertainty visible.
