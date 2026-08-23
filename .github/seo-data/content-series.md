# Daily Report content series

This file is the authoritative thematic roadmap for scheduled Daily Report work.
Daily Report is eBPF-first and series-driven. The goal is to build durable
technical ownership around eBPF and adjacent systems topics rather than publish
unrelated daily news.

## Editorial mix

Apply these rules to the rolling window of the most recent 10 published Daily
Reports:

- **5–7 of 10 must explicitly contain eBPF** as a central mechanism, runtime,
  measurement substrate, comparison point, or systems boundary.
- **Pure AI-agent topics are capped at 1–2 of 10.** Agent topics that primarily
  study eBPF instrumentation, policy enforcement, profiling, or runtime
  mechanisms count as eBPF-centered when eBPF is essential to the technical
  question rather than a decorative mention.
- The remaining reports may cover directly adjacent systems areas such as Linux
  kernels, observability, profiling, networking, security, runtimes, GPU and
  heterogeneous systems, distributed systems, compilers, or storage.
- Until the archive reaches 10 reports, apply the same proportions to the
  available set as closely as possible.

Record the rolling mix in the daily operating record before selecting a topic.
Do not manipulate classification to satisfy the ratio; classify by the report's
actual central technical question.

## Daily publication rule

Every scheduled daily run must publish **one new Daily Report page**. `no-report`
is not an allowed completion outcome.

This is a publication requirement, not permission to lower the quality bar. If
the first candidate fails the evidence, gap, novelty, or usefulness gates, reject
that candidate and continue research on another question in the approved roadmap
until one passes. Prefer changing the question over padding weak evidence or
inventing novelty.

A daily report must still provide:

- a concrete reader problem;
- primary-source evidence;
- a non-trivial gap or unresolved mechanism;
- a reasoned technical conclusion;
- a small number of implementable research directions with academic and
  production value;
- a discriminating evaluation or evidence that would change the conclusion;
- a thesis that does not duplicate an existing report.

## Series rules

- Keep one **active series** for normal daily publication. A series normally
  contains 4–6 substantial reports that build on one another.
- Search inside the active series first. Each new report must answer a distinct
  question and materially advance the series argument.
- Reuse valid evidence maps, unresolved questions, experiments, terminology, and
  counterexamples from earlier reports, but never copy the thesis with a new
  example.
- If the active series cannot yield a publishable report on a given day, move to
  another approved eBPF or adjacent-systems series for that run rather than
  publishing weak material.
- A material external development may justify an out-of-series report when it
  has durable systems consequences and still respects the rolling topic mix.
- Search Console, GA4, GitHub activity, reader pathways, and technical releases
  can influence ordering inside the roadmap, but short-lived popularity does not
  override the editorial mix or technical quality standard.
- After a series has at least three strong reports, consider a public series hub
  and stronger internal linking. Do not create thin hub pages in advance.

## Active series — eBPF Networking and Security

Working question: **Where are eBPF networking and security mechanisms still
missing deployable abstractions or correctness guarantees?**

This series became active after the `2026-08-22` report completed the normal
six-report boundary for eBPF Observability and Profiling.

The roadmap initially preferred transactional policy updates across programs,
maps, links, and userspace control planes. Fresh novelty review on `2026-08-23`
rejected that candidate for this run because
`/research/stateful-ebpf-transactional-upgrade/` already develops a prepare /
migrate / commit / retire generation protocol across programs, links, maps,
pinned state, controller recovery, and rollback. Repeating the same update
transaction with a networking example would not materially advance the archive.

The `2026-08-23` report therefore starts the active series with a distinct
security-policy question: how additive Kubernetes `NetworkPolicy`, tiered
`ClusterNetworkPolicy`, and Cilium L3-L7 policy can compose for multiple owners
without losing authority, delegation, or source-policy provenance after the
result is compiled into an eBPF datapath. It develops an authority-aware
composition IR, generation-stable verdict witnesses, and a counterexample-driven
multi-tenant policy benchmark.

### Published progress

1. `2026-08-23`: `/research/ebpf-network-policy-composition/` separates
   policy-language semantics from BPF hook composition and update transactions.
   It asks how several legitimate policy owners can produce one effective
   network verdict while keeping the deciding owner, tier, delegation path, and
   source rule inspectable across policy generations. The report is
   eBPF-centered because the proposed provenance contract is evaluated against
   the realized BPF policy datapath, not only against Kubernetes objects.

### Preferred next questions

1. zero-copy and programmable I/O paths across XDP, AF_XDP, io_uring, DPDK, and
   userspace eBPF, with a networking/security question that does not repeat the
   earlier io_uring control-surface report;
2. information-flow enforcement across process, file, socket, and encrypted
   application boundaries;
3. verifier and runtime interfaces for richer stateful policies;
4. portable policy execution between kernel, userspace, NIC, and DPU targets;
5. revisit transactional network-policy rollout only when new evidence supports
   a mechanism materially different from the published stateful-upgrade
   generation protocol.

Before and after the August 23 publication, the newest ten contain **7
eBPF-centered / 0 pure Agent / 3 adjacent systems**. The incoming eBPF-centered
report ages the `2026-08-10` eBPF-centered transactional-upgrade report out of the
rolling ten.

## Completed series — eBPF Observability and Profiling

Historical roadmap state: **Active series — eBPF Observability and Profiling**
through the `2026-08-22` report. The series is completed after that publication;
Networking and Security is the active series for the next normal run.

Working question: **Which important performance and correctness questions remain
unanswerable with today's eBPF observability stack?**

The `2026-08-18` report started this series with **page-level memory attribution**:
how to distinguish allocation from pages that were actually touched, faulted,
reclaimed, migrated, or responsible for sampled memory cost while preserving
provenance across `mmap`, `brk`, allocator, runtime, and process boundaries.

The `2026-08-19` report was a deliberate **adjacent profiling detour** on sampling
bias. Its central mechanism is general profiler measurement design, not eBPF, so
it is classified as adjacent systems.

The first `2026-08-20` report was a second adjacent profiling detour required by
the rolling-window arithmetic. It asks whether a late CUDA kernel start came from
host scheduling, runtime/command-buffer work, a dependency, or device
availability. CUPTI and Nsight Systems are the primary mechanisms; Linux/eBPF
host tracing can be an optional evidence source but is not essential, so this
report is adjacent systems rather than eBPF-centered.

The second `2026-08-20` report follows the same adjacent-systems boundary from a
causality angle. It asks how host API work, runtime handoffs, CUDA Graph replay,
and device execution can retain a trustworthy causal identity. CUPTI/CUDA
dependency semantics are central while eBPF is one possible host observer.

The `2026-08-21` report returns to an eBPF-essential question: how dynamic probes
can observe application-defined pools, queues, caches, and credits without
silently turning stale program semantics into confident diagnoses. It develops a
versioned semantics manifest, runtime semantic validation, and a mutation
benchmark for software evolution.

The `2026-08-22` report closes the series with another eBPF-essential question:
how always-on collectors can reduce high-rate telemetry before export without
silently deleting the evidence required for diagnosis. It develops a compiled
diagnostic contract, bounded state-transition exemplars, coverage-carrying
summaries, and an equal-budget diagnosis-retention benchmark.

### Published progress

1. `2026-08-18`: `/research/page-level-ebpf-memory-attribution/` separates
   allocation intent, residency, working-set evidence, page lifecycle, and
   sampled memory cost, then develops a lifetime-aware provenance ledger,
   access-weighted attribution with explicit confidence, and a ground-truth
   benchmark spanning reserve-versus-touch, COW, THP, reclaim, refault, and NUMA
   migration.
2. `2026-08-19`: `/research/profiler-sampling-bias/` asks when sampling itself is
   untrustworthy. It compares randomized sampling history, current Linux perf
   interfaces and kernel profile-collection guidance, and OSDI 2026 Blink, then
   develops a sampling-schedule contract with aliasing diagnostics, independent
   profile epochs with uncertainty and rank stability, and uncertainty-triggered
   selective instrumentation. Because the mechanism applies to profilers in
   general and does not require eBPF, this report is adjacent systems rather than
   eBPF-centered.
3. `2026-08-20`: `/research/gpu-kernel-launch-latency/` separates CUDA API time,
   command-buffer queue/submission, dependency readiness, device availability,
   and kernel execution. It develops an explicit launch-state ledger, a
   cross-domain launch identity that survives host handoffs and graph replay, and
   a ground-truth launch-delay attribution benchmark. The central question is GPU
   profiling, so it is adjacent systems.
4. `2026-08-20`: `/research/gpu-host-device-causality/` develops
   generation-scoped host/device causal identity, dependency-aware critical-path
   reasoning, explicit unknown/loss states, and a ground-truth causality
   benchmark. CUPTI/CUDA dependency semantics are essential while eBPF is one
   possible host observer, so this report is adjacent systems.
5. `2026-08-21`: `/research/ebpf-application-resource-semantics/` asks how eBPF
   can dynamically observe application-defined resources without hard-coding
   stale semantics. It develops a versioned resource-semantics manifest compiled
   into eBPF attachments, runtime semantic validation with explicit confidence
   loss, and a software-mutation benchmark. Dynamic no-rebuild instrumentation
   and independent cross-layer validation are central, so this report is
   eBPF-centered.
6. `2026-08-22`: `/research/ebpf-diagnostic-telemetry-compression/` asks how an
   always-on eBPF collector can reduce telemetry volume before export while
   preserving later root-cause diagnosis. It develops a diagnostic-contract
   compiler for BPF retention plans, state-transition exemplars, explicit
   coverage accounting, and an equal-budget incident benchmark. Source-side BPF
   aggregation and exemplar retention are central, so this report is
   eBPF-centered.

Before and after the August 22 publication, the newest ten contain **7
eBPF-centered / 0 pure Agent / 3 adjacent systems**. The incoming eBPF-centered
report ages an eBPF-centered report out of the rolling ten.

A related unresolved question remains: online confidence and adaptive collection
when trace loss, missing probes, or stale schemas make the current representation
uncertain. It is intentionally not a seventh report in this series. The August 22
report first defines what a compact representation promises to preserve; adaptive
fidelity can be revisited later when fresh evidence supports a distinct mechanism.

## Completed series — eBPF Runtime, Extensibility, and Composition

Working question: **What mechanisms are still missing if eBPF is treated as a
programmable runtime substrate rather than only a kernel observability feature?**

This series connected kernel and userspace execution, composition, state,
upgrade, safety, asynchronous attribution, programmable I/O, and heterogeneous
execution placement. It reached the normal six-report series boundary on
`2026-08-17` and is no longer the default topic source for scheduled publication.

### Published progress

1. `2026-08-08`: `/research/userspace-ebpf-runtime-contract/` established that
   first-class userspace eBPF needs explicit attachment, capability, state,
   lifetime, and attribution contracts above ISA compatibility.
2. `2026-08-09`: `/research/ebpf-hook-composition-contract/` narrowed the
   multi-program problem beyond dispatch and ordering to effect visibility,
   outcome resolution, shared-state ownership, and versioned hook composition.
3. `2026-08-10`: `/research/stateful-ebpf-transactional-upgrade/` separated
   object-level replacement from application-level upgrade and developed an
   explicit prepare / migrate / commit / retire generation protocol for programs,
   links, maps, pinned state, controller recovery, and rollback.
4. `2026-08-12`: `/research/async-ebpf-causal-profiler/` separated thread
   execution from logical-work causality and developed typed, lifetime-aware
   handoff edges, an edge-versus-context measurement budget, and a ground-truth
   causal-attribution benchmark across `io_uring`, workqueues, runtime tasks, and
   application-defined resources.
5. `2026-08-15`: `/research/io-uring-bpf-programmability/` separated the current
   per-opcode cBPF admission path from the eBPF `io_uring_bpf_ops`
   execution-control path, then developed a typed capability contract, versioned
   ring policy generations, explicit provenance, and a comparative
   control-boundary benchmark.
6. `2026-08-17`: `/research/heterogeneous-ebpf-execution-placement/` separated
   backend compatibility from execution placement and developed a target
   manifest, generation-scoped state ownership, and a ground-truth
   placement/provenance benchmark across kernel, userspace, NIC/DPU, and
   GPU-side targets.

The Daily Report index already exposes the sequence. Report-level acquisition and
navigation evidence is still too young to justify a dedicated public series hub.
Revisit that decision only when evidence shows a retrieval benefit.

## Queued series — GPU and Heterogeneous Runtime Systems

Working question: **What runtime and observability abstractions are missing at
CPU/GPU and host/device boundaries?**

Candidate topics include GPU/CPU causal profiling, memory movement, megakernel
observability, programmable device-side instrumentation, distributed GPU
coordination, utilization versus allocatability, host-side scheduling noise, and
eBPF-like programmable monitors near GPU or DPU execution.

Reports in this series count toward the eBPF share only when eBPF or an eBPF-like
runtime is central to the mechanism being evaluated.

The two August 20 reports are focused adjacent-systems contributions to this
roadmap. They do not make this queued series active.

## Queued series — Agent Systems (limited)

Pure Agent systems work is intentionally a minority topic. Use this series only
for questions with unusually strong systems consequences that are not better
framed through eBPF, Linux, runtime, observability, or security mechanisms.

Existing anchors:

- `/research/agent-trace-evidence-budget/`
- `/research/parallel-agent-effect-serializability/`

Neither pure-Agent report remains inside the newest ten after the August 22
publication. Pure-Agent publication is allowed by the cap but is not required;
prefer the technically stronger question after applying the evidence, novelty,
and editorial-mix gates.

Future Agent reports should preferentially connect back to eBPF or systems
infrastructure, for example OS-level effect tracing, eBPF policy enforcement,
sandbox escape visibility, syscall/tool causality, or runtime resource control.

## Choosing the next report

Each daily run should:

1. calculate the current rolling topic mix from the actually published index;
2. start inside the active series when the mix permits it;
3. research multiple candidate questions if necessary;
4. reject candidates that do not pass the evidence and novelty gates;
5. choose one question that both passes quality review and keeps the rolling mix
   compliant;
6. publish exactly one new Daily Report;
7. record the chosen series, topic classification, rejected candidates when
   useful, and why the report materially advances the roadmap.

A temporary out-of-series detour should be recorded in the daily operating record
and in this file so the repository, not chat history, remains authoritative.
