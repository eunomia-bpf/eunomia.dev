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

## Active series — eBPF Observability and Profiling

Working question: **Which important performance and correctness questions remain
unanswerable with today's eBPF observability stack?**

The `2026-08-18` report starts this series with **page-level memory attribution**:
how to distinguish allocation from pages that were actually touched, faulted,
reclaimed, migrated, or responsible for sampled memory cost while preserving
provenance across `mmap`, `brk`, allocator, runtime, and process boundaries.

The `2026-08-19` report is a deliberate **adjacent profiling detour** on sampling
bias. Its central mechanism is general profiler measurement design, not eBPF, so
it is classified as adjacent systems. It closes the first ten-report window at
7 eBPF / 2 Agent / 1 adjacent without relabeling an eBPF-essential question.

The `2026-08-20` report is a second adjacent profiling detour required by the
rolling-window arithmetic. It asks whether a late CUDA kernel start came from
host scheduling, runtime/command-buffer work, a dependency, or device
availability. CUPTI and Nsight Systems are the primary mechanisms; Linux/eBPF
host tracing can be an optional evidence source but is not essential, so this
report is also adjacent systems rather than eBPF-centered.

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

After the August 20 report, the rolling ten-report window is **7 eBPF-centered /
1 pure Agent / 2 adjacent systems**. The oldest report still inside that window
is the remaining pure-Agent parallel-effect report. Therefore the next
publication must again be non-eBPF: immediately adding an eBPF-centered report
would create 8 eBPF reports in the rolling ten and violate the 5–7 requirement.
Once the oldest eBPF reports start aging out, an eBPF-centered report can return
without exceeding the cap.

### Preferred next questions

The active eBPF-series questions remain valuable but are deferred while the
rolling window is at its eBPF ceiling:

1. always-on semantic compression that preserves diagnostic evidence instead of
   raw event volume;
2. application-defined resource profiling that combines static discovery,
   runtime eBPF evidence, and online validation;
3. causal profiling for GPU host-side bottlenecks and megakernel execution where
   eBPF is genuinely essential to the mechanism;
4. revisit async and syscall causal profiling only when new mechanism or
   evaluation evidence materially extends the published causal-profiler report.

For the next run, choose a genuine adjacent-systems question or an unusually
strong pure-Agent systems question after fresh evidence and novelty review. Do
not relabel one of the eBPF-essential candidates just to continue the active
series one day earlier.

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
   per-opcode cBPF admission path from the eBPF `io_uring_bpf_ops` execution-control
   path, then developed a typed capability contract, versioned ring policy
   generations, explicit provenance, and a comparative control-boundary benchmark.
6. `2026-08-17`: `/research/heterogeneous-ebpf-execution-placement/` separated
   backend compatibility from execution placement and developed a target manifest,
   generation-scoped state ownership, and a ground-truth placement/provenance
   benchmark across kernel, userspace, NIC/DPU, and GPU-side targets.

The Daily Report index already exposes the sequence. Report-level acquisition and
navigation evidence is still too young to justify a dedicated public series hub.
Revisit that decision only when evidence shows a retrieval benefit.

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

The August 20 launch-latency report is a focused adjacent-systems contribution to
this roadmap. It does not promote this queued series to active; the active series
remains eBPF Observability and Profiling once the rolling mix permits another
eBPF-centered report.

## Queued series — Agent Systems (limited)

Pure Agent systems work is intentionally a minority topic. Use this series only
for questions with unusually strong systems consequences that are not better
framed through eBPF, Linux, runtime, observability, or security mechanisms.

Existing anchors:

- `/research/agent-trace-evidence-budget/`
- `/research/parallel-agent-effect-serializability/`

After the August 20 report, only the parallel-effect report remains inside the
rolling ten-report window. Pure-Agent publication is allowed by the cap, but is
not required; prefer a technically stronger adjacent-systems question when one
passes the evidence and novelty gates.

Future Agent reports should preferentially connect back to eBPF or systems
infrastructure, for example OS-level effect tracing, eBPF policy enforcement,
sandbox escape visibility, syscall/tool causality, or runtime resource control.

## Choosing the next report

Each daily run should:

1. calculate the current rolling topic mix;
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
