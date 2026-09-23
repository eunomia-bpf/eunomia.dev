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

Before the `2026-09-23` publication, the newest ten actually published reports
contain **7 eBPF-centered / 1 pure Agent / 2 adjacent systems**. The oldest report
rotating out is the eBPF-centered `2026-09-09` native-operation-trust report.
Today's eBPF-centered reboot-state report replaces that eBPF slot, so successful
publication preserves **7 / 1 / 2**.

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
3. `2026-09-22` — `/research/ebpf-kernel-interface-negotiation/`: how a loader
   selects among artifact variants when kfunc, iterator, `struct_ops`, and
   provider-specific interfaces are typed and context-scoped rather than simple
   present-or-missing features. The report develops typed artifact interface
   requirements, scoped capability-negotiation receipts, and dependency-driven
   compatibility CI while keeping trial loading and the target verifier as the
   final admission authority.
4. `2026-09-23` — `/research/ebpf-pinned-map-reboot-state/`: what state contract
   is required when a host reboot destroys the old pinned map objects. The
   report separates live-object pin lifetime from durable state semantics, then
   develops per-map reboot state contracts, consistency-aware checkpoint epochs,
   and staged restore gates before production reattachment. The central boundary
   is reconstruction after the old kernel object graph has disappeared, not
   live transactional cutover.

Remaining candidate boundaries include:

- a narrower CO-RE structural-versus-semantic boundary only if it develops a
  mechanism materially distinct from the September 18 cross-kernel behavior
  contract, rather than merely restating that successful relocation is not a
  semantic proof;
- controller restart and persistent-link ownership only if future evidence keeps
  it materially distinct from the September 23 host-reboot state boundary.

Novelty guards:

- do not repeat the `2026-09-15` version/backport capability-evidence boundary;
- do not repeat the `2026-09-18` post-admission cross-kernel behavioral-compatibility
  boundary with a different example;
- do not repeat the `2026-09-22` typed/scoped interface-negotiation boundary with
  a host-global capability manifest or ordered trial-loading wrapper;
- do not repeat the `2026-09-23` reboot-state contract with a generic pinned-map
  persistence tutorial or live-upgrade example;
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

### Material external-development detour — 2026-09-20

The September 20 publication used the roadmap's material-external-development
escape hatch rather than adding a deployment-compatibility boundary:

- `2026-09-20` — `/research/ebpf-exception-cleanup-unwind/`: compiler-generated
  cleanup landing pads for `bpf_throw()` unwind, verifier resource tracking,
  cleanup metadata, and cross-JIT differential conformance.

The report is eBPF-centered and replaced the eBPF-centered September 6 report in
the rolling window, preserving **7 / 1 / 2**. It does not count as a published
boundary of the active deployment-compatibility series.

The abandoned September 16 CXL draft and the still-open September 16 `io_uring`
draft are not published-state boundaries and must not be counted in this roadmap.
The unmerged September 19 reboot-state attempt `#207` and controller-restart
attempt `#210` are likewise not published boundaries; today's fresh September 23
run supersedes `#207` from current `main`.

## Queued series — Agent Systems (limited)

Pure Agent systems work remains intentionally a minority topic. Existing anchors:

- `/research/agent-trace-evidence-budget/`
- `/research/parallel-agent-effect-serializability/`
- `2026-09-14` — `/research/agent-tool-retry-effect-idempotency/`: durable effect
  identity, ambiguous post-dispatch reconciliation, and post-commit fault
  injection for one intended external mutation.
