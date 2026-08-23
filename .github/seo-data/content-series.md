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

The initial roadmap placed transactional policy update first, but the August 23
novelty review found that `/research/stateful-ebpf-transactional-upgrade/`
already develops the relevant prepare/migrate/commit/retire generation protocol
across programs, maps, links, pinned state, controller recovery, and rollback.
The generic multi-tenant composition candidate also overlaps materially with
`/research/ebpf-hook-composition-contract/`, which already studies outcome
algebra, mutation visibility, shared-map ownership, effect inference, ordering,
and versioned composition. Do not revisit either topic without a materially new
mechanism or new boundary.

The first published Networking and Security report instead narrows the
information-flow question to the precision frontier of process-level taint. It
asks how eBPF can preserve confidentiality when a process mixes sensitive and
public data without claiming that OS events can reconstruct arbitrary userspace
byte flow. The report keeps conservative process taint as the fallback and
recovers precision only through a trusted declassification boundary or a
coverage-aware application boundary.

### Published progress

1. `2026-08-23`: `/research/ebpf-information-flow-declassification/` compares
   BPF-LSM/cgroup mediation, whole-system provenance, TLS plaintext boundaries,
   kTLS/sendfile paths, and current capture-integrity limitations. It develops a
   trusted release proxy with explicit authority, a coverage-aware egress
   manifest that retains `unknown` for unobserved release paths, and a mixed-flow
   DLP benchmark that scores leaks and benign blocking together. eBPF is central
   to the mediation, provenance, boundary correlation, and sink-enforcement
   mechanism, so the report is eBPF-centered.

### Preferred next questions

1. zero-copy and programmable I/O paths across XDP, AF_XDP, io_uring, DPDK, and
   userspace eBPF, especially where ownership and security policy change across
   memory domains;
2. verifier and runtime interfaces for richer stateful policies whose correctness
   depends on temporal or cross-object state rather than one event;
3. portable policy execution between kernel, userspace, NIC, and DPU targets,
   including which security semantics survive placement changes;
4. a narrower policy-conflict question only if it goes beyond the existing hook
   composition contract with a distinct network-policy authority or resolution
   mechanism;
5. transactional policy updates only if fresh evidence exposes a boundary not
   already handled by the existing stateful-upgrade report.

Before and after the August 23 eBPF-centered publication, the newest ten contain
**7 eBPF-centered / 0 pure Agent / 3 adjacent systems** because the incoming
report ages the eBPF-centered August 10 transactional-upgrade report out of the
window.

## Completed series — eBPF Observability and Profiling

Working question: **Which important performance and correctness questions remain
unanswerable with today's eBPF observability stack?**

The series completed at the six-report boundary on `2026-08-22`:

1. `2026-08-18`: `/research/page-level-ebpf-memory-attribution/` separates
   allocation intent, residency, working-set evidence, page lifecycle, and
   sampled memory cost, then develops a lifetime-aware provenance ledger,
   access-weighted attribution with explicit confidence, and a ground-truth
   memory benchmark.
2. `2026-08-19`: `/research/profiler-sampling-bias/` studies sampling bias,
   independent profile epochs, uncertainty, and selective instrumentation. The
   central mechanism applies to profilers generally, so it is adjacent systems.
3. `2026-08-20`: `/research/gpu-kernel-launch-latency/` separates CUDA API time,
   command-buffer work, dependency readiness, device availability, and kernel
   execution. The central question is GPU profiling, so it is adjacent systems.
4. `2026-08-20`: `/research/gpu-host-device-causality/` develops
   generation-scoped host/device causal identity, dependency-aware critical-path
   reasoning, explicit unknown/loss states, and a ground-truth causality
   benchmark. It is adjacent systems.
5. `2026-08-21`: `/research/ebpf-application-resource-semantics/` develops a
   versioned resource-semantics manifest compiled into eBPF attachments, runtime
   confidence loss for stale contracts, and a software-mutation benchmark. It is
   eBPF-centered.
6. `2026-08-22`: `/research/ebpf-diagnostic-telemetry-compression/` develops a
   diagnostic-contract compiler for BPF retention plans, state-transition
   exemplars, explicit coverage accounting, and an equal-budget incident
   benchmark. It is eBPF-centered.

A related unresolved question remains: online confidence and adaptive collection
when trace loss, missing probes, or stale schemas make the current representation
uncertain. Revisit it later only if fresh evidence supports a mechanism distinct
from the completed series.

## Completed series — eBPF Runtime, Extensibility, and Composition

Working question: **What mechanisms are still missing if eBPF is treated as a
programmable runtime substrate rather than only a kernel observability feature?**

The series reached its normal six-report boundary on `2026-08-17`:

1. `2026-08-08`: `/research/userspace-ebpf-runtime-contract/` established
   explicit attachment, capability, state, lifetime, and attribution contracts.
2. `2026-08-09`: `/research/ebpf-hook-composition-contract/` developed effect
   visibility, outcome resolution, shared-state ownership, and versioned hook
   composition.
3. `2026-08-10`: `/research/stateful-ebpf-transactional-upgrade/` developed an
   explicit prepare/migrate/commit/retire protocol across programs, links, maps,
   pinned state, controller recovery, and rollback.
4. `2026-08-12`: `/research/async-ebpf-causal-profiler/` developed typed,
   lifetime-aware handoff edges and a ground-truth causal-attribution benchmark
   across asynchronous execution boundaries.
5. `2026-08-15`: `/research/io-uring-bpf-programmability/` separated cBPF
   admission from eBPF `io_uring_bpf_ops` execution control and developed typed
   capability, policy-generation, provenance, and comparative control-boundary
   ideas.
6. `2026-08-17`: `/research/heterogeneous-ebpf-execution-placement/` developed a
   target manifest, generation-scoped state ownership, and a ground-truth
   placement/provenance benchmark across kernel, userspace, NIC/DPU, and GPU-side
   targets.

The Daily Report index already exposes the sequence. Report-level acquisition and
navigation evidence is still too young to justify a dedicated public series hub.

## Queued series — GPU and Heterogeneous Runtime Systems

Working question: **What runtime and observability abstractions are missing at
CPU/GPU and host/device boundaries?**

Candidate topics include GPU/CPU causal profiling, memory movement, megakernel
observability, programmable device-side instrumentation, distributed GPU
coordination, utilization versus allocatability, host-side scheduling noise, and
eBPF-like programmable monitors near GPU or DPU execution.

Reports in this series count toward the eBPF share only when eBPF or an eBPF-like
runtime is central to the mechanism being evaluated. The two August 20 reports
are focused adjacent-systems contributions to this roadmap. They do not make this
queued series active.

## Queued series — Agent Systems (limited)

Pure Agent systems work is intentionally a minority topic. Use this series only
for questions with unusually strong systems consequences that are not better
framed through eBPF, Linux, runtime, observability, or security mechanisms.

Existing anchors:

- `/research/agent-trace-evidence-budget/`
- `/research/parallel-agent-effect-serializability/`

Neither pure-Agent report remains inside the newest ten after the August 23
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
