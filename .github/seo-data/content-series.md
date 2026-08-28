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

## Active series — GPU and Heterogeneous Runtime Systems

Working question: **What runtime, programmability, and observability abstractions
are missing at CPU/GPU and host/device boundaries?**

This roadmap becomes active after the `2026-08-28` report reaches the normal
six-report boundary for eBPF Networking and Security.

Candidate topics include GPU/CPU causal profiling, memory movement, megakernel
observability, programmable device-side instrumentation, distributed GPU
coordination, utilization versus allocatability, host-side scheduling noise, and
eBPF-like programmable monitors near GPU or DPU execution.

Reports in this series count toward the eBPF share only when eBPF or an eBPF-like
runtime is central to the mechanism being evaluated. A report whose core
mechanism is CUPTI, CUDA, GPU scheduling, or another general accelerator
abstraction without essential eBPF machinery is adjacent systems.

The two August 20 GPU reports are useful existing anchors but do not count as
progress in the newly active sequence. They establish two boundaries that new
reports should not simply repeat:

- `/research/gpu-kernel-launch-latency/` separates queueing/readiness delay from
  kernel execution time;
- `/research/gpu-host-device-causality/` develops host/device causal identity and
  explicit unknown dependency edges.

### Preferred next questions

1. a device-side programmability or policy boundary where an eBPF-like runtime
   materially changes what can be enforced or adapted without rebuilding the
   workload;
2. memory-placement or movement semantics where CPU and GPU observers disagree
   about ownership, residency, or cost and a deployable cross-domain contract is
   missing;
3. fine-grained GPU execution observability only when the question is distinct
   from launch-latency attribution and host/device causal tracing;
4. distributed GPU coordination when there is a concrete runtime invariant to
   test rather than a generic scaling survey;
5. revisit heterogeneous placement only when fresh evidence exposes a mechanism
   beyond the already published target-manifest and generation-scoped ownership
   contract.

After the August 28 publication, the newest ten remain **7 eBPF-centered / 0 pure
Agent / 3 adjacent systems**. The next report may therefore be either eBPF-centered
or adjacent, but evidence and thesis quality decide the classification.

## Completed series — eBPF Networking and Security

Working question: **Where are eBPF networking and security mechanisms still
missing deployable abstractions or correctness guarantees?**

This series became active after the `2026-08-22` report completed eBPF
Observability and Profiling. It reaches the normal six-report boundary on
`2026-08-28`.

### Published progress

1. `2026-08-23`: `/research/ebpf-network-policy-composition/` separates
   policy-language semantics from BPF hook composition and update transactions.
   It asks how several legitimate policy owners can produce one effective
   verdict while keeping the deciding owner, tier, delegation path, and source
   rule inspectable across policy generations.
2. `2026-08-24`: `/research/ebpf-zero-copy-buffer-ownership/` asks how packet
   buffer leases preserve owner, DMA reachability, recycle generation, and BPF
   policy provenance across AF_XDP, io_uring ZC Rx, DPDK, userspace eBPF, and NIC
   handoffs.
3. `2026-08-25`: `/research/ebpf-stateful-policy-verification/` separates
   verifier-safe bytecode from legal temporal transitions of persistent policy
   state shared across hooks, CPUs, maps, and userspace updates.
4. `2026-08-26`: `/research/ebpf-authorization-revocation/` separates rollout
   from revocation effectiveness and asks for a measurable bound on stale
   authority across conntrack, auth maps, socket-local state, and other derived
   datapath objects.
5. `2026-08-27`: `/research/ebpf-complete-mediation-offload/` separates backend
   placement from global security coverage and asks whether every reachable
   host/SmartNIC/DPU packet path still crosses a policy-equivalent enforcement
   point for the current generation.
6. `2026-08-28`: `/research/ebpf-l7-proxy-policy-identity/` separates path
   coverage from authorization identity continuity. It asks how a principal,
   policy generation, and authorization provenance survive when eBPF redirects a
   request into an L7 proxy that terminates the downstream connection and emits
   or reuses a different upstream socket. It develops generation-scoped handoff
   capabilities, policy-safe multiplexing, and an authorization-lineage
   benchmark across kernel fast paths and proxy slow paths.

The sixth report was selected only after rejecting a broad information-flow
candidate that still overlapped ActPlane effect labels and a narrower fast/slow
verdict-equivalence candidate that overlapped the complete-mediation report. The
proxy identity question is distinct because the request can cross every intended
enforcement point and still be attributed to the wrong principal after the
semantic handoff.

Future networking/security work should return only when fresh evidence supports
a mechanism beyond these six published boundaries. In particular, do not repeat
transactional rollout, zero-copy ownership, revocation, complete mediation, or
proxy identity continuity with a new product example alone.

## Completed series — eBPF Observability and Profiling

Working question: **Which important performance and correctness questions remain
unanswerable with today's eBPF observability stack?**

Published progress:

1. `2026-08-18`: `/research/page-level-ebpf-memory-attribution/` develops
   lifetime-aware page provenance and access-weighted attribution.
2. `2026-08-19`: `/research/profiler-sampling-bias/` is an adjacent profiling
   detour on aliasing, skid, coverage, and uncertainty.
3. `2026-08-20`: `/research/gpu-kernel-launch-latency/` is an adjacent GPU
   profiling report on launch-state attribution.
4. `2026-08-20`: `/research/gpu-host-device-causality/` is an adjacent GPU
   report on host/device causal identity and dependency uncertainty.
5. `2026-08-21`: `/research/ebpf-application-resource-semantics/` develops a
   versioned semantics manifest and mutation benchmark for application-defined
   resources.
6. `2026-08-22`: `/research/ebpf-diagnostic-telemetry-compression/` develops a
   diagnostic-contract compiler, state-transition exemplars, coverage-carrying
   summaries, and equal-budget diagnosis-retention evaluation.

A related question about online confidence and adaptive collection remains valid
but should be revisited only when fresh evidence supports a mechanism distinct
from the compression contract.

## Completed series — eBPF Runtime, Extensibility, and Composition

Working question: **What mechanisms are still missing if eBPF is treated as a
programmable runtime substrate rather than only a kernel observability feature?**

Published progress:

1. `2026-08-08`: `/research/userspace-ebpf-runtime-contract/` establishes
   attachment, capability, state, lifetime, and attribution contracts above ISA
   compatibility.
2. `2026-08-09`: `/research/ebpf-hook-composition-contract/` develops typed
   composition semantics for effect visibility, outcomes, shared state, and
   versioned hooks.
3. `2026-08-10`: `/research/stateful-ebpf-transactional-upgrade/` develops a
   prepare / migrate / commit / retire generation protocol across programs,
   links, maps, pinned state, controller recovery, and rollback.
4. `2026-08-12`: `/research/async-ebpf-causal-profiler/` develops typed,
   lifetime-aware handoff edges and a ground-truth cross-thread attribution
   benchmark.
5. `2026-08-15`: `/research/io-uring-bpf-programmability/` separates the cBPF
   admission gate from the eBPF `io_uring_bpf_ops` control surface and develops
   policy generations, provenance, and capability contracts.
6. `2026-08-17`: `/research/heterogeneous-ebpf-execution-placement/` develops a
   target manifest, generation-scoped state ownership, and a placement/provenance
   benchmark across kernel, userspace, NIC/DPU, and GPU-side targets.

The Daily Report index already exposes these sequences. Revisit dedicated public
series hubs only when report-level acquisition or navigation evidence shows a
retrieval benefit beyond the current index.

## Queued series — Agent Systems (limited)

Pure Agent systems work is intentionally a minority topic. Use this series only
for questions with unusually strong systems consequences that are not better
framed through eBPF, Linux, runtime, observability, or security mechanisms.

Existing anchors:

- `/research/agent-trace-evidence-budget/`
- `/research/parallel-agent-effect-serializability/`

Neither pure-Agent report remains inside the newest ten after the August 28
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