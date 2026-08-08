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

## Active series — eBPF Runtime, Extensibility, and Composition

Working question: **What mechanisms are still missing if eBPF is treated as a
programmable runtime substrate rather than only a kernel observability feature?**

This series should connect kernel and userspace execution, composition, state,
upgrade, safety, performance, and deployment boundaries.

### Preferred sequence

1. **What is still missing for first-class userspace eBPF?**
   - Compare kernel eBPF, uprobes, DBI, language runtimes, and bpftime-style
     execution.
   - Focus on attach semantics, state, compatibility, safety, and which workloads
     genuinely benefit from moving execution out of the kernel.

2. **How should multiple independent eBPF extensions share one hook safely?**
   - Study ordering, isolation, attribution, state sharing, resource accounting,
     and conflicting policies.
   - Compare dispatcher chains, tail calls, trampolines, vBPF-style approaches,
     and explicit composition contracts.

3. **Can a stateful eBPF application be upgraded transactionally?**
   - Go beyond replacing one program pointer.
   - Analyze coordinated changes to programs, links, maps, pinned state, userspace
     controllers, and rollback after partial failure.

4. **What would an asynchronous profiler built around modern eBPF look like?**
   - Cover syscalls, async runtimes, io_uring, work queues, task handoff, causality,
     sampling bias, and application-defined resources.
   - Ask which causal edges can be reconstructed online with acceptable overhead.

5. **Which new Linux I/O hooks make previously impractical eBPF mechanisms
   possible?**
   - Study recent io_uring and related programmable I/O interfaces, file access,
     policy, caching, scheduling, and fast-path control.
   - Prefer concrete mechanisms that could not be implemented cleanly with older
     hook sets.

6. **Where should eBPF execution live in heterogeneous systems?**
   - Compare kernel, userspace, DPU/NIC, GPU-adjacent, and device-side execution.
   - Investigate state placement, verifier assumptions, memory visibility,
     cross-device coordination, and observability/control tradeoffs.

These are research questions, not fixed titles. Final titles and claims must come
from the day's evidence.

## Queued series — eBPF Observability and Profiling

Working question: **Which important performance and correctness questions remain
unanswerable with today's eBPF observability stack?**

Candidate topics:

1. async and syscall causal profiling across thread and process handoff;
2. page-level memory attribution: RSS, touched versus untouched allocation,
   mmap/brk provenance, page faults, reclaim, and memory bandwidth;
3. sampling theory for profilers: phase locking, randomized sampling, bias, and
   confidence intervals;
4. always-on semantic compression that preserves diagnostic evidence instead of
   raw event volume;
5. application-defined resource profiling that combines static discovery,
   runtime eBPF evidence, and online validation;
6. causal profiling for GPU host-side bottlenecks and megakernel execution.

## Queued series — eBPF Networking and Security

Working question: **Where are eBPF networking and security mechanisms still
missing deployable abstractions or correctness guarantees?**

Candidate topics:

1. transactional policy updates across programs, maps, links, and control planes;
2. multi-tenant network-policy composition and conflict resolution;
3. zero-copy and programmable I/O paths across XDP, AF_XDP, io_uring, DPDK, and
   userspace eBPF;
4. information-flow enforcement across process, file, socket, and encrypted
   application boundaries;
5. verifier and runtime interfaces for richer stateful policies;
6. portable policy execution between kernel, userspace, NIC, and DPU targets.

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

Pure Agent systems work is intentionally a minority topic. Use this series only
for questions with unusually strong systems consequences that are not better
framed through eBPF, Linux, runtime, observability, or security mechanisms.

Existing anchors:

- `/research/agent-trace-evidence-budget/`
- `/research/parallel-agent-effect-serializability/`

Because these already occupy the pure-Agent budget in the current small archive,
do not schedule another pure Agent report until the rolling mix is compliant.

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
