# Daily Report content series

This file is the authoritative thematic roadmap for scheduled Daily Report work.
Daily Report is eBPF-first and series-driven. The goal is durable technical
ownership around eBPF and adjacent systems topics rather than unrelated daily
news.

## Editorial mix

Apply these rules to the rolling window of the most recent 10 published Daily
Reports:

- **5–7 of 10 must explicitly contain eBPF** as a central mechanism, runtime,
  measurement substrate, comparison point, or systems boundary.
- **Pure AI-agent topics are capped at 1–2 of 10.** Agent topics count as
  eBPF-centered only when eBPF is essential to the question.
- Remaining reports may cover adjacent Linux, observability, profiling,
  networking, security, runtime, GPU, distributed-systems, compiler, or storage
  questions.
- Record the rolling mix before topic selection. Never relabel old reports or add
  an extra report merely to repair the ratio.

Before the 2026-09-25 publication, the newest ten actually published reports
contain **7 eBPF-centered / 1 pure Agent / 2 adjacent systems**. The oldest report
rotating out is the eBPF-centered 2026-09-09 native-operation trust-boundary
report. Today's eBPF-centered map-reuse report replaces that eBPF slot, so
successful publication preserves **7 / 1 / 2**.

Open, draft, closed-unmerged, and branch-only work never counts as published
state.

## Daily publication rule

Every scheduled daily run publishes **exactly one new bilingual Daily Report**.
A weak candidate must be replaced rather than turned into a no-report day.

Every report must provide a concrete reader problem, primary-source evidence, a
non-trivial unresolved mechanism, a reasoned conclusion, a small number of
implementable research directions, discriminating evaluation, and evidence that
would change the conclusion. The thesis must not duplicate an existing report.

## Series rules

- Keep one active series for normal publication. A series normally contains 4–6
  substantial reports that build on one another.
- Search the active series first when the rolling mix permits it.
- If the active series cannot yield a publishable report or the mix blocks its
  classification, use a technically strong approved detour.
- Material external developments may justify an out-of-series report when they
  have durable systems consequences.
- After at least three strong reports, add stronger internal linking or a public
  hub only when acquisition/navigation evidence supports it.

## Completed series — eBPF Runtime, Extensibility, and Composition

Working question: what mechanisms are missing if eBPF is treated as a
programmable runtime substrate rather than only a kernel observability feature?

This series reached six reports:

1. 2026-08-08 — /research/userspace-ebpf-runtime-contract/
2. 2026-08-09 — /research/ebpf-hook-composition-contract/
3. 2026-08-10 — /research/stateful-ebpf-transactional-upgrade/
4. 2026-08-12 — /research/async-ebpf-causal-profiler/
5. 2026-08-15 — /research/io-uring-bpf-programmability/
6. 2026-08-17 — /research/heterogeneous-ebpf-execution-placement/

Return only with a mechanism beyond these boundaries.

## Completed series — eBPF Observability and Profiling

Working question: which important performance and correctness questions remain
unanswerable with today's eBPF observability stack?

This series reached six reports:

1. 2026-08-18 — /research/page-level-ebpf-memory-attribution/
2. 2026-08-19 — /research/profiler-sampling-bias/ (adjacent)
3. 2026-08-20 — /research/gpu-kernel-launch-latency/ (adjacent)
4. 2026-08-20 — /research/gpu-host-device-causality/ (adjacent)
5. 2026-08-21 — /research/ebpf-application-resource-semantics/
6. 2026-08-22 — /research/ebpf-diagnostic-telemetry-compression/

Return only with a mechanism beyond these boundaries.

## Completed series — eBPF Networking and Security

Working question: where are eBPF networking and security mechanisms still
missing deployable abstractions or correctness guarantees?

This series reached six reports:

1. 2026-08-23 — /research/ebpf-network-policy-composition/
2. 2026-08-24 — /research/ebpf-zero-copy-buffer-ownership/
3. 2026-08-25 — /research/ebpf-stateful-policy-verification/
4. 2026-08-26 — /research/ebpf-authorization-revocation/
5. 2026-08-27 — /research/ebpf-complete-mediation-offload/
6. 2026-08-28 — /research/ebpf-l7-proxy-policy-identity/

Return only with a mechanism beyond these boundaries.

## Completed series — GPU and Heterogeneous Runtime Systems

Working question: what runtime and observability abstractions are missing at
CPU/GPU and host/device boundaries?

This series reached six reports:

1. 2026-08-29 — /research/gpu-memory-placement-evidence/
2. 2026-08-30 — /research/gpu-instrumentation-safety-contract/
3. 2026-08-31 — /research/gpu-utilization-allocatability/
4. 2026-09-02 — /research/gpu-membership-generation-continuity/
5. 2026-09-03 — /research/ebpf-gpu-megakernel-observability/
6. 2026-09-04 — /research/gpu-checkpoint-recovery-consistency/

Return only with a mechanism beyond these boundaries.

## Completed series — eBPF Optimization and Execution Specialization

Working question: how can eBPF specialize to hardware and workload behavior
without silently changing verifier-approved semantics, portability,
debuggability, trust, or the validity of performance claims?

This series reached six reports:

1. 2026-09-05 — /research/ebpf-runtime-profile-specialization/
2. 2026-09-06 — /research/ebpf-portable-architecture-specialization/
3. 2026-09-07 — /research/ebpf-specialization-debug-provenance/
4. 2026-09-09 — /research/ebpf-native-operation-trust-boundary/
5. 2026-09-10 — /research/ebpf-cross-backend-operation-semantics/
6. 2026-09-11 — /research/ebpf-optimization-evidence-contract/

Return only with a mechanism beyond these boundaries.

## Active series — eBPF Deployment Compatibility and Lifecycle

Working question: **How can one eBPF application remain loadable, semantically
correct, and operationally explainable across real kernel, distribution,
backport, toolchain, BPF-interface, and persistent-state evolution?**

Published boundaries before the September 25 run:

1. **2026-09-15 — /research/ebpf-kernel-capability-evidence/**: replace
   version-string assumptions with artifact-bound capability evidence, bounded
   semantic canaries, and replayable support envelopes.
2. **2026-09-18 — /research/ebpf-kernel-upgrade-semantic-compatibility/**:
   separate CO-RE/load-time admission from application behavior using
   cross-kernel semantic witnesses, drift localization, and promotion gates.
3. **2026-09-22 — /research/ebpf-kernel-interface-negotiation/**: treat kfunc,
   iterator, struct_ops, and provider-specific interfaces as typed and scoped
   contracts, with target verifier admission remaining authoritative.

The September 25 report, once the complete merge/deployment acceptance path
finishes, becomes boundary four:

4. **2026-09-25 — /research/ebpf-map-reuse-semantic-compatibility/**: decide
   whether an existing pinned map's state can be reused by a new application
   generation. It separates libbpf map-definition compatibility from
   BTF-derived structural schema and application semantic schema, then develops
   canonical structural fingerprints, versioned state contracts/migration, and
   shadow validation before write authority.

Remaining candidate boundaries include:

- restored-state identity/provenance across host replacement or explicit
  serialize/restore, but only if it develops a mechanism beyond in-boot map reuse
  and the August transactional-upgrade protocol;
- package/controller ownership of persistent BPF links only if fresh primary
  evidence supports a contract materially distinct from existing unmerged
  attempts;
- another lifecycle boundary only when it cannot be reduced to capability
  evidence, cross-kernel behavior, interface negotiation, or map-state reuse.

Novelty guards:

- do not repeat the September 15 version/backport capability-evidence boundary;
- do not repeat the September 18 post-admission cross-kernel behavioral boundary;
- do not repeat the September 22 typed/scoped interface-negotiation boundary;
- after successful September 25 publication, do not rename structural
  fingerprints, semantic revisions, migration, or shadow validation into another
  map-state reuse report;
- do not repeat September 6 architecture-specific specialization/fallback;
- do not repeat the August 10 whole-application transactional-upgrade protocol;
- do not repeat the August 8 userspace-runtime capability/lifetime contract;
- require current primary evidence before selecting the next boundary.

### Mix-driven adjacent detour — 2026-09-17

The CXL memory-tier isolation report is adjacent systems work. It was selected
because an eBPF report on that day would have moved the rolling window above the
configured 7-of-10 eBPF ceiling. It preserved 7 / 1 / 2.

### Material external-development detour — 2026-09-20

/research/ebpf-exception-cleanup-unwind/ used the material-external-development
escape hatch. It is eBPF-centered but does not count as a deployment-
compatibility series boundary. It also preserved 7 / 1 / 2.

Unmerged daily attempts, including September 23 map-reuse and reboot-state work,
do not establish published roadmap state.

## Queued series — Agent Systems (limited)

Pure Agent systems work remains intentionally a minority topic. Existing anchors
include:

- /research/agent-trace-evidence-budget/
- /research/parallel-agent-effect-serializability/
- 2026-09-14 — /research/agent-tool-retry-effect-idempotency/

Any future pure-Agent detour must remain inside the 1–2 of 10 cap and clear the
same evidence and novelty gates as eBPF work.

## Choosing the next report

Each daily run should:

1. calculate the actual rolling topic mix from the published index;
2. start inside the active series when the mix permits it;
3. research multiple candidate questions when necessary;
4. reject candidates that fail evidence, novelty, or usefulness gates;
5. choose one question that preserves the editorial mix without relabeling old
   reports;
6. publish exactly one new bilingual Daily Report;
7. record the chosen series, classification, rejected candidates, and why the
   report materially advances the roadmap.

Temporary detours must be recorded here or in the daily operating record so the
repository, not chat history, remains authoritative.
