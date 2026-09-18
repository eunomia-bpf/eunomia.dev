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
- **Pure AI-agent topics are capped at 1–2 of 10.** Agent topics that primarily
  study eBPF instrumentation, policy enforcement, profiling, or runtime
  mechanisms count as eBPF-centered only when eBPF is essential to the question.
- Remaining reports may cover directly adjacent Linux, observability, profiling,
  networking, security, runtime, GPU, distributed-systems, compiler, or storage
  questions.
- Record the rolling mix before topic selection. Never relabel old reports or add
  an extra report merely to repair the ratio.

## Daily publication rule

Every scheduled daily run publishes **exactly one new bilingual Daily Report**.
`no-report` is not an allowed normal completion outcome. A weak first candidate
must be rejected and replaced with another approved question rather than padded
with weak evidence.

Every report must provide a concrete reader problem, primary-source evidence, a
non-trivial unresolved mechanism, a reasoned conclusion, a small number of
implementable research directions, a discriminating evaluation, and evidence
that would change the conclusion. The thesis must not duplicate an existing
report.

## Series rules

- Keep one active series for normal publication. A series normally contains 4–6
  substantial reports that build on one another.
- Search the active series first when the rolling mix permits it. Each report
  must answer a distinct question.
- If the active series cannot yield a publishable report or the mix temporarily
  blocks its classification, use a technically strong approved detour rather
  than violate the quality or mix gates.
- A material external development may justify an out-of-series report when it
  has durable systems consequences.
- After a series has at least three strong reports, consider stronger internal
  linking or a public hub only when acquisition/navigation evidence supports it.

## Completed series — eBPF Runtime, Extensibility, and Composition

Working question: **What mechanisms are missing if eBPF is treated as a
programmable runtime substrate rather than only a kernel observability feature?**

This series reached its six-report boundary on `2026-08-17`:

1. `2026-08-08` — `/research/userspace-ebpf-runtime-contract/`: first-class
   userspace attachment, capability, state, lifetime, and attribution.
2. `2026-08-09` — `/research/ebpf-hook-composition-contract/`: effect visibility,
   outcome resolution, shared-state ownership, and versioned hook composition.
3. `2026-08-10` — `/research/stateful-ebpf-transactional-upgrade/`: prepare /
   migrate / commit / retire generations for programs, links, maps, pinned state,
   controller recovery, and rollback.
4. `2026-08-12` — `/research/async-ebpf-causal-profiler/`: typed lifetime-aware
   causal handoff edges across `io_uring`, workqueues, runtimes, and application
   resources.
5. `2026-08-15` — `/research/io-uring-bpf-programmability/`: cBPF admission versus
   eBPF `io_uring_bpf_ops`, capability, policy generation, provenance, and
   resource ownership.
6. `2026-08-17` — `/research/heterogeneous-ebpf-execution-placement/`: target
   manifests and generation-scoped state ownership across kernel, userspace,
   NIC/DPU, and GPU-side execution.

Return only when fresh evidence supports a mechanism beyond these boundaries.

## Completed series — eBPF Observability and Profiling

Working question: **Which important performance and correctness questions remain
unanswerable with today's eBPF observability stack?**

The series reached six reports on `2026-08-22`:

1. `2026-08-18` — `/research/page-level-ebpf-memory-attribution/`.
2. `2026-08-19` — `/research/profiler-sampling-bias/` (adjacent systems).
3. `2026-08-20` — `/research/gpu-kernel-launch-latency/` (adjacent systems).
4. `2026-08-20` — `/research/gpu-host-device-causality/` (adjacent systems).
5. `2026-08-21` — `/research/ebpf-application-resource-semantics/`.
6. `2026-08-22` — `/research/ebpf-diagnostic-telemetry-compression/`.

Do not add a seventh report merely by renaming one of these boundaries.

## Completed series — eBPF Networking and Security

Working question: **Where are eBPF networking and security mechanisms still
missing deployable abstractions or correctness guarantees?**

The series reached six reports on `2026-08-28`:

1. `2026-08-23` — `/research/ebpf-network-policy-composition/`.
2. `2026-08-24` — `/research/ebpf-zero-copy-buffer-ownership/`.
3. `2026-08-25` — `/research/ebpf-stateful-policy-verification/`.
4. `2026-08-26` — `/research/ebpf-authorization-revocation/`.
5. `2026-08-27` — `/research/ebpf-complete-mediation-offload/`.
6. `2026-08-28` — `/research/ebpf-l7-proxy-policy-identity/`.

Return only with a mechanism beyond policy composition, zero-copy ownership,
temporal state correctness, revocation, complete mediation, or proxy identity.

## Completed series — GPU and Heterogeneous Runtime Systems

Working question: **What runtime and observability abstractions are missing at
CPU/GPU and host/device boundaries?**

The active sequence reached its six-report boundary on `2026-09-04`:

1. `2026-08-29` — `/research/gpu-memory-placement-evidence/`.
2. `2026-08-30` — `/research/gpu-instrumentation-safety-contract/`.
3. `2026-08-31` — `/research/gpu-utilization-allocatability/`.
4. `2026-09-02` — `/research/gpu-membership-generation-continuity/`.
5. `2026-09-03` — `/research/ebpf-gpu-megakernel-observability/`.
6. `2026-09-04` — `/research/gpu-checkpoint-recovery-consistency/`.

Return only with a mechanism beyond these six boundaries.

## Completed series — eBPF Optimization and Execution Specialization

Working question: **How can eBPF programs and runtimes specialize to hardware and
workload behavior without silently changing verifier-approved semantics,
portability, debuggability, trust, or the validity of the performance claim?**

This series reached its six-report boundary on `2026-09-11`:

1. `2026-09-05` — `/research/ebpf-runtime-profile-specialization/`: verifier
   acceptance, optimizer equivalence, profile assumptions, and invalidation.
2. `2026-09-06` — `/research/ebpf-portable-architecture-specialization/`:
   architecture capability, proof-linked native emits, and fallback.
3. `2026-09-07` — `/research/ebpf-specialization-debug-provenance/`: exact
   specialization generation and execution provenance.
4. `2026-09-09` — `/research/ebpf-native-operation-trust-boundary/`: delegated
   native-code TCB, effect scope, assurance, and artifact identity.
5. `2026-09-10` — `/research/ebpf-cross-backend-operation-semantics/`: one
   observable state-transition contract across host/native/NIC/DPU backends.
6. `2026-09-11` — `/research/ebpf-optimization-evidence-contract/`: scoped
   performance evidence, holdouts/counterexamples, and production promotion.

Future reports must not repackage these six optimization boundaries with a new
example.

## Active series — eBPF Deployment Compatibility and Lifecycle

Working question: **How can one eBPF application remain loadable, semantically
correct, and operationally explainable across real kernel, distribution,
backport, toolchain, and BPF-interface evolution?**

Published boundaries:

1. `2026-09-15` — `/research/ebpf-kernel-capability-evidence/`: kernel and
   distribution version metadata versus direct capability evidence. The report
   develops an artifact-bound capability receipt, side-effect-bounded semantic
   canaries for ambiguous edges, and replayable artifact-to-kernel support
   envelopes. The central mechanism is deployment admission on real
   distribution/backport/configuration/privilege combinations, not
   architecture-specific native code generation.
2. `2026-09-18` — `/research/ebpf-kernel-upgrade-semantic-compatibility/`: what
   must be re-proven when the same artifact relocates, verifies, and attaches on
   both sides of a kernel upgrade. The report separates CO-RE structural
   adaptation and load-time admission from application behavior, then develops
   artifact-specific cross-kernel semantic witnesses, a compatibility dependency
   graph for drift localization, and a semantic kernel-upgrade promotion gate.
   Recent Linux 7.2/Cilium probe failures provide direct evidence that even the
   loader's interpretation of a verifier result can be a compatibility surface.

Remaining candidate boundaries include:

- version/capability negotiation for kfunc, `struct_ops`, iterator, and other
  rapidly evolving BPF-facing interfaces;
- pinned-map and persistent-state lifecycle when kernel capabilities, BTF, or
  object layouts evolve across host upgrades;
- reproducible capability and artifact manifests across distributions so a
  loader can explain why a program chose, rejected, or downgraded one path;
- a narrower CO-RE structural-versus-semantic boundary only if it develops a
  mechanism materially distinct from the September 18 cross-kernel behavior
  contract, rather than merely restating that successful relocation is not a
  semantic proof.

Novelty guards:

- do not repeat the `2026-09-15` version/backport capability-evidence boundary;
- do not repeat the `2026-09-18` post-admission cross-kernel behavioral-compatibility
  boundary with a different example;
- do not repeat September 6 architecture-specific specialization and fallback;
- do not repeat the August 10 application-level transactional-upgrade protocol;
- do not repeat the August 8 userspace-runtime capability/lifetime contract;
- require current primary evidence from Linux/BPF tooling, distributions, CI, or
  production compatibility systems before selecting the next boundary.

### Mix-driven adjacent detour — 2026-09-17

Before the September 17 publication, the newest ten actually published reports
contained **7 eBPF-centered / 1 pure Agent / 2 adjacent systems**. The oldest
report rotating out that day was the adjacent `2026-09-04` GPU-checkpoint report.
Publishing an eBPF-centered report would have moved the window to **8 / 1 / 1**
and violated the configured 5–7 eBPF-centered range, so the active series paused
for one run.

The selected adjacent report was:

- `2026-09-17` — `/research/cxl-memory-tier-isolation/`: Linux CXL/NUMA/cgroup
  lifetime residency semantics. It separates initial allocation eligibility from
  reclaim demotion, migration, shared-page residency, and pressure failure
  behavior, then develops a tier-residency hardwall, multi-owner shared-page
  policy, and adversarial conformance benchmark.

That report is adjacent rather than eBPF-centered because Linux memory tiering is
the central mechanism. It preserved the newest-ten mix at **7 / 1 / 2**. On
September 18 the oldest report rotating out became the eBPF-centered September 5
runtime-profile report, so the active eBPF series became mechanically eligible
again. The September 18 eBPF report replaces that eBPF slot and keeps the mix at
**7 / 1 / 2**.

The abandoned September 16 CXL draft and the still-open September 16 `io_uring`
draft are not published-state boundaries and must not be counted in this roadmap.

## Queued series — Agent Systems (limited)

Pure Agent systems work remains intentionally a minority topic. Existing anchors:

- `/research/agent-trace-evidence-budget/`
- `/research/parallel-agent-effect-serializability/`
- `2026-09-14` — `/research/agent-tool-retry-effect-idempotency/`: durable effect
  identity, ambiguous post-dispatch reconciliation, and post-commit fault
  injection for one intended external mutation.

The September 14 report is distinct from parallel-agent serializability: it
addresses duplicate materialization of one intended mutation rather than global
composition of several workers' effects. Any future pure-Agent detour must remain
inside the 1–2 of 10 cap and clear the same evidence and novelty gates as eBPF
work.

## Choosing the next report

Each daily run should:

1. calculate the actual rolling topic mix from the published index;
2. start inside the active series when the mix permits it;
3. research multiple candidate questions when necessary;
4. reject candidates that fail evidence, novelty, or usefulness gates;
5. choose one question that preserves the editorial mix without relabeling old
   reports;
6. publish exactly one new bilingual Daily Report;
7. record the chosen series, classification, useful rejected candidates, and why
   the report materially advances the roadmap.

Temporary detours must be recorded here or in the daily operating record so the
repository, not chat history, remains authoritative.
