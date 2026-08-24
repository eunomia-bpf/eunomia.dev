# Daily Report content series

This file is the authoritative thematic roadmap for scheduled Daily Report work.
Daily Report is eBPF-first and series-driven. The goal is to build durable
technical ownership around eBPF and adjacent systems topics rather than publish
unrelated daily news.

## Editorial mix

Apply these rules to the rolling window of the most recent 10 published Daily
Reports:

- **5–7 of 10 must explicitly contain eBPF** as a central mechanism, runtime, measurement substrate, comparison point, or systems boundary.
- **Pure AI-agent topics are capped at 1–2 of 10.** Agent topics that primarily study eBPF instrumentation, policy enforcement, profiling, or runtime mechanisms count as eBPF-centered when eBPF is essential to the technical question rather than a decorative mention.
- The remaining reports may cover directly adjacent systems areas such as Linux kernels, observability, profiling, networking, security, runtimes, GPU and heterogeneous systems, distributed systems, compilers, or storage.
- Until the archive reaches 10 reports, apply the same proportions to the available set as closely as possible.

Record the rolling mix in the daily operating record before selecting a topic.
Do not manipulate classification to satisfy the ratio; classify by the report's
actual central technical question.

## Daily publication rule

Every scheduled daily run must publish **one new Daily Report page**. `no-report`
is not an allowed completion outcome. A weak candidate is rejected and replaced
with another approved question rather than padded into a report.

A daily report must still provide a concrete reader problem, primary-source
evidence, a non-trivial gap, a reasoned technical conclusion, a small number of
implementable directions with academic and production value, a discriminating
evaluation or falsifier, and a thesis that does not duplicate an existing report.

## Series rules

- Keep one active series for normal daily publication. A series normally contains 4–6 substantial reports that build on one another.
- Search inside the active series first. Each report must answer a distinct question and materially advance the series argument.
- Reuse valid evidence maps, unresolved questions, experiments, terminology, and counterexamples from earlier reports, but never copy the thesis with a new example.
- If the active series cannot yield a publishable report, move to another approved eBPF or adjacent-systems series for that run rather than publishing weak material.
- A material external development may justify an out-of-series report when it has durable systems consequences and still respects the rolling topic mix.
- Search Console, GA4, GitHub activity, reader pathways, and technical releases can influence ordering inside the roadmap, but short-lived popularity does not override the editorial mix or technical quality standard.
- After a series has at least three strong reports, consider a public series hub and stronger internal linking. Do not create thin hub pages in advance.

## Active series — eBPF Networking and Security

Working question: **Where are eBPF networking and security mechanisms still
missing deployable abstractions or correctness guarantees?**

This series became active after the `2026-08-22` report completed the normal
six-report boundary for eBPF Observability and Profiling.

The initial transactional-policy-update candidate was rejected because
`/research/stateful-ebpf-transactional-upgrade/` already develops a prepare /
migrate / commit / retire generation protocol across programs, links, maps,
pinned state, controller recovery, and rollback. Repeating that protocol with a
networking example would not materially advance the archive.

### Published progress

1. `2026-08-23`: `/research/ebpf-network-policy-composition/` asks how additive Kubernetes `NetworkPolicy`, tiered `ClusterNetworkPolicy`, and Cilium L3-L7 policy can compose for several legitimate owners without losing authority, delegation, or source-policy provenance after compilation into an eBPF datapath. It develops an authority-aware composition IR, generation-stable verdict witnesses, and a counterexample-driven multi-tenant policy benchmark.
2. `2026-08-24`: `/research/ebpf-cross-boundary-information-flow/` asks how eBPF information-flow enforcement can stay precise when one long-running process multiplexes public and sensitive work and TLS moves semantic data across userspace, kernel, proxies, and hardware offload. It develops generation-scoped sub-process flow identity, a trusted semantic-to-kernel binding, an explicit TLS-path coverage contract, and a benchmark that scores false allows, false denies, and unknown coverage. The central mechanism is eBPF enforcement, so it is eBPF-centered.

The August 24 report is materially different from the August 23 report. The first
is about **policy-owner authority and composition**; the second is about
**data-flow identity and enforcement coverage across shared execution and
cryptographic boundaries**.

### Preferred next questions

1. verifier and runtime interfaces for richer stateful policies, especially when policy state evolves across maps, helpers, kfuncs, and multiple hooks;
2. portable policy execution between kernel, userspace, NIC, and DPU targets while preserving enforcement semantics and authority;
3. zero-copy and programmable I/O paths across XDP, AF_XDP, io_uring, DPDK, and userspace eBPF only when fresh evidence identifies a security/networking thesis that does not repeat the earlier io_uring control-surface, heterogeneous-placement, or application-resource reports;
4. revisit transactional network-policy rollout only when new evidence supports a mechanism materially different from the published stateful-upgrade generation protocol.

Before and after the August 24 publication, the newest ten contain **7
eBPF-centered / 0 pure Agent / 3 adjacent systems**. The incoming eBPF-centered
report ages another eBPF-centered report out of the rolling ten.

## Completed series — eBPF Observability and Profiling

Working question: **Which important performance and correctness questions remain
unanswerable with today's eBPF observability stack?**

Published sequence:

1. `2026-08-18`: `/research/page-level-ebpf-memory-attribution/` — eBPF-centered page lifecycle and memory-cost provenance.
2. `2026-08-19`: `/research/profiler-sampling-bias/` — adjacent profiling measurement and sampling uncertainty.
3. `2026-08-20`: `/research/gpu-kernel-launch-latency/` — adjacent GPU launch-delay attribution.
4. `2026-08-20`: `/research/gpu-host-device-causality/` — adjacent host/device causal identity and critical paths.
5. `2026-08-21`: `/research/ebpf-application-resource-semantics/` — eBPF-centered dynamic semantics for application-defined resources.
6. `2026-08-22`: `/research/ebpf-diagnostic-telemetry-compression/` — eBPF-centered source-side compression that preserves diagnostic evidence.

A related unresolved question remains online confidence and adaptive collection
when trace loss, missing probes, or stale schemas make the current representation
uncertain. It can be revisited later when fresh evidence supports a distinct
mechanism.

## Completed series — eBPF Runtime, Extensibility, and Composition

Working question: **What mechanisms are still missing if eBPF is treated as a
programmable runtime substrate rather than only a kernel observability feature?**

Published sequence:

1. `2026-08-08`: `/research/userspace-ebpf-runtime-contract/`
2. `2026-08-09`: `/research/ebpf-hook-composition-contract/`
3. `2026-08-10`: `/research/stateful-ebpf-transactional-upgrade/`
4. `2026-08-12`: `/research/async-ebpf-causal-profiler/`
5. `2026-08-15`: `/research/io-uring-bpf-programmability/`
6. `2026-08-17`: `/research/heterogeneous-ebpf-execution-placement/`

The Daily Report index already exposes the sequence. Report-level acquisition and
navigation evidence is still too young to justify a dedicated public series hub.

## Queued series — GPU and Heterogeneous Runtime Systems

Working question: **What runtime and observability abstractions are missing at
CPU/GPU and host/device boundaries?**

Candidate topics include GPU/CPU causal profiling, memory movement, megakernel
observability, programmable device-side instrumentation, distributed GPU
coordination, utilization versus allocatability, host-side scheduling noise, and
eBPF-like programmable monitors near GPU or DPU execution. Reports count toward
the eBPF share only when eBPF or an eBPF-like runtime is central to the mechanism.

## Queued series — Agent Systems (limited)

Pure Agent systems work is intentionally a minority topic. Existing anchors are
`/research/agent-trace-evidence-budget/` and
`/research/parallel-agent-effect-serializability/`. Neither remains inside the
newest ten after the August 22 publication.

Future Agent reports should preferentially connect back to eBPF or systems
infrastructure, for example OS-level effect tracing, eBPF policy enforcement,
sandbox escape visibility, syscall/tool causality, or runtime resource control.

## Choosing the next report

Each daily run should:

1. calculate the current rolling topic mix from the actually published index;
2. start inside the active series when the mix permits it;
3. research multiple candidate questions if necessary;
4. reject candidates that do not pass the evidence and novelty gates;
5. choose one question that both passes quality review and keeps the rolling mix compliant;
6. publish exactly one new Daily Report;
7. record the chosen series, topic classification, rejected candidates when useful, and why the report materially advances the roadmap.
