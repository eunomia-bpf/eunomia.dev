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
  available set as closely as possible. The two existing Agent-centered reports
  mean the next reports should strongly favor eBPF before another pure Agent
  report is considered.

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

A daily report must still provide a concrete reader problem, primary-source
evidence, a non-trivial gap or unresolved mechanism, a reasoned technical
conclusion, implementable research directions, and a discriminating evaluation
or evidence that would change the conclusion.

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

## Active series — eBPF Observability and Profiling

Working question: **Which important performance and correctness questions remain
unanswerable with today's eBPF observability stack?**

The `2026-08-18` report starts this series with **page-level memory attribution**.
It distinguishes allocator intent, resident memory, working-set evidence, page
lifecycle, and sampled memory cost, then asks how provenance can survive
`mmap`/`brk`, allocator boundaries, page reuse, reclaim, migration, shared
mappings, and hardware-dependent sampling.

### Published progress

1. `2026-08-18`: `/research/page-level-ebpf-memory-attribution/` develops a
   lifetime-aware allocation-to-page provenance ledger, access-weighted
   attribution with explicit confidence, and a ground-truth benchmark spanning
   reserve-versus-touch, COW, THP, reclaim, refault, and NUMA migration.

### Preferred next questions

1. sampling theory for profilers: phase locking, randomized sampling, bias,
   variance, and confidence intervals;
2. always-on semantic compression that preserves diagnostic evidence instead of
   raw event volume;
3. application-defined resource profiling that combines static discovery,
   runtime eBPF evidence, and online validation;
4. causal profiling for GPU host-side bottlenecks and megakernel execution;
5. revisit async or syscall causal profiling only when a new mechanism or
   evaluation materially extends the published causal-profiler report.

## Completed series — eBPF Runtime, Extensibility, and Composition

Working question: **What mechanisms are still missing if eBPF is treated as a
programmable runtime substrate rather than only a kernel observability feature?**

This series reached its normal six-report boundary on `2026-08-17`:

1. `2026-08-08`: `/research/userspace-ebpf-runtime-contract/` established explicit
   attachment, capability, state, lifetime, and attribution contracts above ISA
   compatibility.
2. `2026-08-09`: `/research/ebpf-hook-composition-contract/` developed effect
   visibility, outcome resolution, state ownership, and versioned composition.
3. `2026-08-10`: `/research/stateful-ebpf-transactional-upgrade/` developed
   generation-based prepare / migrate / commit / retire semantics.
4. `2026-08-12`: `/research/async-ebpf-causal-profiler/` developed typed
   lifetime-aware handoff edges and a causal-attribution benchmark.
5. `2026-08-15`: `/research/io-uring-bpf-programmability/` separated cBPF request
   admission from eBPF `io_uring_bpf_ops` execution control and developed a
   versioned ring policy model.
6. `2026-08-17`: `/research/heterogeneous-ebpf-execution-placement/` separated
   backend compatibility from execution placement and developed target manifests
   and generation-scoped state ownership across kernel, userspace, NIC/DPU, and
   GPU-side targets.

The Daily Report index already exposes this sequence. A dedicated public series
hub remains evidence-gated rather than cadence-driven.

## Queued series — eBPF Networking and Security

Working question: **Where are eBPF networking and security mechanisms still
missing deployable abstractions or correctness guarantees?**

Candidate topics include transactional policy updates, multi-tenant policy
composition, zero-copy programmable I/O across XDP/AF_XDP/io_uring/DPDK,
information-flow enforcement, richer stateful verifier/runtime interfaces, and
portable policy execution across kernel, userspace, NIC, and DPU targets.

## Queued series — GPU and Heterogeneous Runtime Systems

Working question: **What runtime and observability abstractions are missing at
CPU/GPU and host/device boundaries?**

Candidate topics include GPU/CPU causal profiling, memory movement, megakernel
observability, programmable device-side instrumentation, distributed GPU
coordination, utilization versus allocatability, host-side scheduling noise, and
eBPF-like programmable monitors near GPU or DPU execution.

Reports in this series count toward the eBPF share only when eBPF or an eBPF-like
runtime is central to the mechanism being evaluated.

## Queued series — Agent Systems (limited)

Pure Agent systems work is intentionally a minority topic. Existing anchors are
`/research/agent-trace-evidence-budget/` and
`/research/parallel-agent-effect-serializability/`. Because these already occupy
the pure-Agent budget in the current small archive, do not schedule another pure
Agent report until the rolling mix is compliant.

Future Agent reports should preferentially connect back to eBPF or systems
infrastructure, for example OS-level effect tracing, eBPF policy enforcement,
sandbox escape visibility, syscall/tool causality, or runtime resource control.

## Choosing the next report

Each daily run should:

1. calculate the current rolling topic mix;
2. start inside the active series;
3. research multiple candidate questions if necessary;
4. reject candidates that do not pass the evidence and novelty gates;
5. choose one question that both passes quality review and keeps the rolling mix
   compliant;
6. publish exactly one new Daily Report;
7. record the chosen series, topic classification, rejected candidates when
   useful, and why the report materially advances the series.

A series switch should be recorded in the daily operating record and in this
file so the repository, not chat history, remains authoritative.
