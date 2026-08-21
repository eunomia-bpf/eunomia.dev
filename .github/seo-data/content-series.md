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

A daily report must still provide a concrete reader problem, primary-source
evidence, a non-trivial gap, a reasoned conclusion, a small number of implementable
research directions, and a discriminating evaluation or boundary condition.

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

## Active series — eBPF Observability and Profiling

Working question: **Which important performance and correctness questions remain
unanswerable with today's eBPF observability stack?**

### Published progress

1. `2026-08-18`: `/research/page-level-ebpf-memory-attribution/` separates
   allocation intent, residency, working-set evidence, page lifecycle, and
   sampled memory cost, then develops a lifetime-aware provenance ledger,
   access-weighted attribution with explicit confidence, and a ground-truth
   benchmark spanning reserve-versus-touch, COW, THP, reclaim, refault, and NUMA
   migration. **Classification: eBPF-centered.**
2. `2026-08-19`: `/research/profiler-sampling-bias/` asks when sampling itself is
   untrustworthy and develops sampling-schedule diagnostics, independent profile
   epochs, and uncertainty-triggered selective instrumentation. The mechanism is
   general profiler design rather than eBPF. **Classification: adjacent systems.**
3. `2026-08-20`: `/research/gpu-kernel-launch-latency/` separates CUDA API time,
   command-buffer queue/submission, dependency readiness, device availability,
   and kernel execution. CUPTI/CUDA are central; eBPF host tracing is optional.
   **Classification: adjacent systems.**
4. `2026-08-20`: `/research/gpu-host-device-causality/` develops generation-scoped
   host/device causal identity, dependency-aware critical-path reasoning, explicit
   unknown/loss states, and a ground-truth causality benchmark. CUPTI/CUDA
   dependency semantics are essential while eBPF is one possible host observer.
   **Classification: adjacent systems.**
5. `2026-08-21`: `/research/ebpf-application-resource-semantics/` asks how eBPF can
   dynamically observe application-defined pools, queues, caches, and credits
   without hard-coding stale semantics. It develops a versioned resource-semantics
   manifest compiled into eBPF attachments, runtime semantic validation with
   explicit confidence loss, and a software-mutation benchmark. Dynamic no-rebuild
   eBPF instrumentation and independent cross-layer validation are central to the
   mechanism. **Classification: eBPF-centered.**

Before the August 21 publication, the newest ten already contained **7
eBPF-centered / 0 pure Agent / 3 adjacent systems** because the August 20
host-device causality report had landed after the earlier operating-state update.
Publishing the August 21 eBPF-centered report ages the oldest eBPF report out of
the rolling ten, so the newest ten remain **7 eBPF / 0 pure Agent / 3 adjacent**.
The publication therefore remains within the required 5–7 eBPF ceiling without
reclassification.

### Preferred next questions

The active series remains the default source. The next publication may still be
eBPF-centered if it materially advances the series, because another eBPF report
would again age an eBPF report out of the newest ten and keep the count at seven.
Prefer, in order after fresh novelty review:

1. always-on semantic compression that preserves diagnostic evidence instead of
   raw event volume;
2. online confidence and adaptive collection for semantic eBPF profilers when
   trace loss, missing probes, or stale schemas make an explanation uncertain;
3. GPU host-side and megakernel profiling only when eBPF is genuinely essential
   to the mechanism rather than an optional evidence source;
4. revisit async and syscall causal profiling only when new mechanism or
   evaluation evidence materially extends the published causal-profiler report.

## Completed series — eBPF Runtime, Extensibility, and Composition

Working question: **What mechanisms are still missing if eBPF is treated as a
programmable runtime substrate rather than only a kernel observability feature?**

This series connected kernel and userspace execution, composition, state,
upgrade, safety, asynchronous attribution, programmable I/O, and heterogeneous
execution placement. It reached the normal six-report series boundary on
`2026-08-17` and is no longer the default topic source for scheduled publication.

### Published progress

1. `2026-08-08`: `/research/userspace-ebpf-runtime-contract/`
2. `2026-08-09`: `/research/ebpf-hook-composition-contract/`
3. `2026-08-10`: `/research/stateful-ebpf-transactional-upgrade/`
4. `2026-08-12`: `/research/async-ebpf-causal-profiler/`
5. `2026-08-15`: `/research/io-uring-bpf-programmability/`
6. `2026-08-17`: `/research/heterogeneous-ebpf-execution-placement/`

The Daily Report index already exposes the sequence. Report-level acquisition and
navigation evidence is still too young to justify a dedicated public series hub.

## Queued series — eBPF Networking and Security

Working question: **Where are eBPF networking and security mechanisms still
missing deployable abstractions or correctness guarantees?**

Candidate topics include transactional policy updates; multi-tenant policy
composition; zero-copy programmable I/O across XDP, AF_XDP, io_uring, DPDK, and
userspace eBPF; information-flow enforcement across process/file/socket/encrypted
boundaries; richer verifier/runtime interfaces; and portable policy execution
across kernel, userspace, NIC, and DPU targets.

## Queued series — GPU and Heterogeneous Runtime Systems

Working question: **What runtime and observability abstractions are missing at
CPU/GPU and host/device boundaries?**

Candidate topics include GPU/CPU causal profiling, memory movement, megakernel
observability, programmable device-side instrumentation, distributed GPU
coordination, utilization versus allocatability, host-side scheduling noise, and
eBPF-like programmable monitors near GPU or DPU execution.

Reports here count toward the eBPF share only when eBPF or an eBPF-like runtime
is central to the mechanism being evaluated.

## Queued series — Agent Systems (limited)

Pure Agent systems work is intentionally a minority topic. Use this series only
for questions with unusually strong systems consequences that are not better
framed through eBPF, Linux, runtime, observability, or security mechanisms.

Existing anchors:

- `/research/agent-trace-evidence-budget/`
- `/research/parallel-agent-effect-serializability/`

Neither pure-Agent report is currently inside the newest ten after the August 21
publication. Pure-Agent work remains allowed by the cap but is not required; the
active eBPF series remains stronger unless fresh evidence establishes a better
Agent-systems question.

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

A temporary out-of-series detour should be recorded here so the repository, not
chat history, remains authoritative.
