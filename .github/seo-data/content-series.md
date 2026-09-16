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

This roadmap became active after the optimization series closed. Its first
published boundary is:

1. `2026-09-15` — `/research/ebpf-kernel-capability-evidence/`: kernel and
   distribution version metadata versus direct capability evidence. The report
   develops an artifact-bound capability receipt, side-effect-bounded semantic
   canaries for ambiguous edges, and replayable artifact-to-kernel support
   envelopes. The central mechanism is deployment admission on real
   distribution/backport/configuration/privilege combinations, not
   architecture-specific native code generation.

Remaining candidate boundaries include:

- verifier-acceptance and behavior drift across kernels/toolchains, including how
  a deployment records and diagnoses “same object, different verifier outcome”;
- CO-RE relocation compatibility versus semantic compatibility of helpers, maps,
  kfuncs, program types, and attachment behavior;
- version/capability negotiation for kfunc, `struct_ops`, iterator, and other
  rapidly evolving BPF-facing interfaces;
- pinned-map and persistent-state lifecycle when kernel capabilities, BTF, or
  object layouts evolve across host upgrades;
- reproducible capability and artifact manifests across distributions so a
  loader can explain why a program chose, rejected, or downgraded one path.

Novelty guards:

- do not repeat the `2026-09-15` version/backport capability-evidence boundary;
- do not repeat September 6 architecture-specific specialization and fallback;
- do not repeat the August 10 application-level transactional-upgrade protocol;
- do not repeat the August 8 userspace-runtime capability/lifetime contract;
- require current primary evidence from Linux/BPF tooling, distributions, CI, or
  production compatibility systems before selecting the next boundary.

### September 16 mix-driven detour

After the September 15 publication the newest ten actually published reports are
**7 eBPF-centered / 1 pure Agent / 2 adjacent systems**. On September 16, the
oldest report rotating out is the adjacent `2026-09-04` GPU checkpoint report.
Therefore another eBPF-centered report would create **8 / 1 / 1** and violate the
5–7 eBPF bound.

The approved detour is `2026-09-16` —
`/research/io-uring-cancel-terminal-state/`, classified **adjacent systems**. It
asks when an `io_uring` operation is truly terminal after cancellation races with
normal completion. The report develops terminal-state receipts,
generation-aware resource retirement fences, and an adversarial race/effect
benchmark. This is distinct from the August 15 eBPF/io_uring programmability
report because eBPF is not part of the mechanism under study, and distinct from
the September 14 Agent retry report because kernel completion channels are
available here even though application lifetime handling can still be wrong.

Publishing the detour preserves the newest-ten mix at **7 eBPF-centered / 1 pure
Agent / 2 adjacent systems**. The active eBPF deployment series remains active
and should resume only when the actual published window mechanically permits the
next eBPF-centered report.

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