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

This series reached its normal six-report boundary on `2026-08-17`:

1. `2026-08-08` — `/research/userspace-ebpf-runtime-contract/`: first-class
   userspace eBPF attachment, capability, state, lifetime, and attribution.
2. `2026-08-09` — `/research/ebpf-hook-composition-contract/`: effect visibility,
   outcome resolution, shared-state ownership, and versioned hook composition.
3. `2026-08-10` — `/research/stateful-ebpf-transactional-upgrade/`: prepare /
   migrate / commit / retire upgrade generations for programs, links, maps,
   pinned state, controller recovery, and rollback.
4. `2026-08-12` — `/research/async-ebpf-causal-profiler/`: typed lifetime-aware
   causal handoff edges across `io_uring`, workqueues, runtimes, and application
   resources.
5. `2026-08-15` — `/research/io-uring-bpf-programmability/`: cBPF admission
   versus eBPF `io_uring_bpf_ops`, capability, policy generation, provenance,
   and resource ownership.
6. `2026-08-17` — `/research/heterogeneous-ebpf-execution-placement/`: target
   manifests and generation-scoped state ownership across kernel, userspace,
   NIC/DPU, and GPU-side execution.

Return only when fresh evidence supports a mechanism beyond these boundaries.

## Completed series — eBPF Observability and Profiling

Working question: **Which important performance and correctness questions remain
unanswerable with today's eBPF observability stack?**

The series reached six reports on `2026-08-22`:

1. `2026-08-18` — `/research/page-level-ebpf-memory-attribution/`: allocation,
   residency, page lifecycle, and sampled-memory provenance.
2. `2026-08-19` — `/research/profiler-sampling-bias/`: adjacent-systems sampling
   aliasing, skid, uncertainty, and selective instrumentation.
3. `2026-08-20` — `/research/gpu-kernel-launch-latency/`: adjacent host/runtime/
   queue/dependency/device launch-delay attribution.
4. `2026-08-20` — `/research/gpu-host-device-causality/`: adjacent host/device
   causal identity and dependency-aware critical paths.
5. `2026-08-21` — `/research/ebpf-application-resource-semantics/`: versioned
   resource-semantics manifests and confidence under software evolution.
6. `2026-08-22` — `/research/ebpf-diagnostic-telemetry-compression/`: diagnostic
   contracts, bounded exemplars, coverage, and equal-budget diagnosis retention.

Do not add a seventh report merely by renaming one of these boundaries.

## Completed series — eBPF Networking and Security

Working question: **Where are eBPF networking and security mechanisms still
missing deployable abstractions or correctness guarantees?**

This series became active after Observability and Profiling and reached six
reports on `2026-08-28`:

1. `2026-08-23` — `/research/ebpf-network-policy-composition/`: authority-aware
   composition and generation-stable verdict provenance.
2. `2026-08-24` — `/research/ebpf-zero-copy-buffer-ownership/`: generation-scoped
   packet-buffer leases and policy-linked handoff witnesses.
3. `2026-08-25` — `/research/ebpf-stateful-policy-verification/`: temporal policy
   contracts for persistent map-backed security state.
4. `2026-08-26` — `/research/ebpf-authorization-revocation/`: scoped revocation
   epochs and a measurable bound on stale authorization.
5. `2026-08-27` — `/research/ebpf-complete-mediation-offload/`: path coverage and
   generation-continuous policy enforcement across host/offload/fallback.
6. `2026-08-28` — `/research/ebpf-l7-proxy-policy-identity/`: policy/principal
   identity continuity across proxy termination, pooling, retry, and fallback.

Return only with a mechanism beyond policy composition, zero-copy ownership,
temporal state correctness, revocation, complete mediation, or proxy identity.

## Completed series — GPU and Heterogeneous Runtime Systems

Working question: **What runtime and observability abstractions are missing at
CPU/GPU and host/device boundaries?**

The August 20 launch and causality reports predate activation and remain adjacent
background boundaries. The active sequence reached its six-report boundary on
`2026-09-04`:

1. `2026-08-29` — `/research/gpu-memory-placement-evidence/`: evidence-carrying
   placement under HBM oversubscription.
2. `2026-08-30` — `/research/gpu-instrumentation-safety-contract/`: probe-effect
   manifests, resource budgets, and explicit observation coverage.
3. `2026-08-31` — `/research/gpu-utilization-allocatability/`: candidate-specific
   admission rather than retrospective utilization.
4. `2026-09-02` — `/research/gpu-membership-generation-continuity/`: application
   state generation continuity across communicator membership changes.
5. `2026-09-03` — `/research/ebpf-gpu-megakernel-observability/`: semantic task
   hooks and coverage-carrying device-side eBPF aggregation.
6. `2026-09-04` — `/research/gpu-checkpoint-recovery-consistency/`: application-
   consistent recovery cuts across CPU, GPU, communication, and external effects.

Return only with a mechanism beyond these six boundaries.

## Completed series — eBPF Optimization and Execution Specialization

Working question: **How can eBPF programs and runtimes specialize to hardware and
workload behavior without silently changing verifier-approved semantics,
portability, debuggability, trust, or the validity of the performance claim?**

This series became active after the GPU/runtime series closed and reaches its
normal six-report boundary with the `2026-09-11` publication.

The six boundaries are intentionally different:

1. `2026-09-05` — `/research/ebpf-runtime-profile-specialization/` separates
   verifier acceptance, optimizer semantic equivalence, profile assumptions, and
   guarded invalidation/deoptimization.
2. `2026-09-06` — `/research/ebpf-portable-architecture-specialization/` separates
   portable BPF semantics from architecture-specific implementation eligibility,
   proof-linked native emits, and deterministic fallback.
3. `2026-09-07` — `/research/ebpf-specialization-debug-provenance/` makes the
   exact specialization generation, assumptions, transforms, JIT image, and
   activation interval durable for postmortem attribution.
4. `2026-09-09` — `/research/ebpf-native-operation-trust-boundary/` makes the
   delegated native-code TCB, effect scope, assurance evidence, and artifact
   identity explicit rather than treating verifier success as a blanket trust bit.
5. `2026-09-10` — `/research/ebpf-cross-backend-operation-semantics/` asks whether
   eligible and trusted host, native, NIC, and DPU implementations refine one
   observable state-transition contract under concurrency, failures, and handoff.
6. `2026-09-11` — `/research/ebpf-optimization-evidence-contract/` separates
   semantic admissibility from performance generalization and production
   promotion. Kops, BPF CI/veristat, and the current bpf-bench corpus show why one
   instruction win, verifier statistic, application result, or tuned search set
   cannot serve as a universal performance oracle. The report develops scoped
   evidence envelopes, frozen holdout/counterexample evaluation for adaptive
   optimizers, and profitability-aware promotion/rollback gates.

### Rolling mix at closure

Before September 10 the newest ten contained **5 eBPF-centered / 0 pure Agent /
5 adjacent systems**. The September 10 eBPF-centered report rotated the August
29 adjacent GPU memory-placement report out, producing **6 / 0 / 4**.

Before September 11 the mix is therefore **6 / 0 / 4**. The September 11
optimization-evidence report is eBPF-centered and rotates the August 30 adjacent
GPU-instrumentation report out. After publication the newest ten contain **7
eBPF-centered / 0 pure Agent / 3 adjacent systems**. No classification changes
were made to obtain that result.

The series is closed. Future reports must not repackage verifier/equivalence,
profile invalidation, architecture capability/fallback, execution provenance,
native-operation trust accounting, cross-backend state-transition semantics, or
performance-evidence/promotion scope with a different optimizer example.

## Active series — eBPF Deployment Compatibility and Lifecycle

Working question: **How can one eBPF application remain loadable, semantically
correct, and operationally explainable across real kernel, distribution,
backport, toolchain, and BPF-interface evolution?**

This roadmap becomes active after the optimization series reaches six reports on
`2026-09-11`. It is deliberately about deployment compatibility across evolving
systems, not architecture-specific code generation and not application-level
transactional upgrade.

Candidate boundaries include:

- real-kernel and distribution/backport feature evidence versus kernel-version
  heuristics for deciding which BPF features are actually available;
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

- do not repeat September 6 architecture-specific specialization and fallback;
- do not repeat the August 10 application-level transactional-upgrade protocol;
- do not repeat the August 8 userspace-runtime capability/lifetime contract;
- require current primary evidence from Linux/BPF tooling, distributions, CI, or
  production compatibility systems before selecting a boundary.

The newest-ten mix after September 11 sits at the allowed maximum of **7
eBPF-centered / 0 pure Agent / 3 adjacent systems**. Therefore the next scheduled
run must not mechanically publish another eBPF-centered report if doing so would
push the rolling window above 7. When the active series is temporarily blocked by
that arithmetic, select a strong approved adjacent-systems detour (or, if it
passes the quality bar, a limited pure-Agent systems question) and record the
detour. Resume this active series as soon as the rolling window permits it.

## Queued series — Agent Systems (limited)

Pure Agent systems work remains intentionally a minority topic. Existing anchors:

- `/research/agent-trace-evidence-budget/`
- `/research/parallel-agent-effect-serializability/`

Pure-Agent publication is allowed by the cap but is never required. Prefer Agent
questions with strong systems consequences, especially OS-level effect tracing,
eBPF policy enforcement, sandbox visibility, syscall/tool causality, or runtime
resource control.

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