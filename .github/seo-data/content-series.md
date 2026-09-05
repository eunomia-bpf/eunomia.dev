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

## Completed series — eBPF Networking and Security

Working question: **Where are eBPF networking and security mechanisms still
missing deployable abstractions or correctness guarantees?**

This series became active after the `2026-08-22` report completed the normal
six-report boundary for eBPF Observability and Profiling and reached its own
normal six-report boundary on `2026-08-28`.

The roadmap initially preferred transactional policy updates across programs,
maps, links, and userspace control planes. Fresh novelty review on `2026-08-23`
rejected that candidate because `/research/stateful-ebpf-transactional-upgrade/`
already develops a prepare / migrate / commit / retire generation protocol across
programs, links, maps, pinned state, controller recovery, and rollback. Repeating
the same update transaction with a networking example would not materially
advance the archive.

The `2026-08-23` report starts the series with a distinct security-policy
question: how additive Kubernetes `NetworkPolicy`, tiered `ClusterNetworkPolicy`,
and Cilium L3-L7 policy can compose for multiple owners without losing authority,
delegation, or source-policy provenance after the result is compiled into an
eBPF datapath. It develops an authority-aware composition IR, generation-stable
verdict witnesses, and a counterexample-driven multi-tenant policy benchmark.

The `2026-08-24` report advances a second, materially different boundary. AF_XDP,
io_uring ZC Rx, page_pool, and DPDK all recycle packet memory through native
ownership protocols, while BPF policy decisions, NIC steering, and userspace
processing can cross those boundaries. The report develops a generation-scoped
buffer capability, policy-linked handoff witnesses, and a cross-path zero-copy
fault benchmark. It is distinct from the earlier io_uring programmability report:
the missing property is buffer lease ownership and provenance across APIs, not
BPF execution control inside the ring.

The `2026-08-25` report advances a third boundary: the Linux verifier can prove
that each loaded BPF program executes safely while the security policy encoded in
persistent map state still follows an invalid temporal trace. Production policy
state can be shared across hooks, CPUs, programs, and userspace and can change
under eviction, map pressure, revocation, restart, or stale control-plane writes.
The report develops a small temporal policy contract, verifier-cooperative runtime
transition guards, and an adversarial state-fault benchmark. This is distinct
from verifier-error diagnosis, transactional upgrade, and policy composition:
the target property is legal runtime state transition after safe bytecode has
already been admitted.

The `2026-08-26` report narrows one consequence of persistent state into an
incident-response property. A policy or identity can be revoked while an old
allow still survives in connection tracking, authentication caches, socket-local
storage, or another derived datapath object. The report asks for a measurable
upper bound on that stale authority rather than another general state-machine
contract. It develops scoped revocation epochs, a cross-layer completion barrier,
and a benchmark whose primary outcome is the last stale allow after a revoke.

The `2026-08-27` report advances a fifth boundary by narrowing heterogeneous
execution placement into a complete-mediation property. Linux representors and
XDP offload expose host slow paths and device fast paths that can change under
misses, updates, and faults. The report asks whether every reachable
policy-relevant packet path still crosses an enforcement point for the current
policy generation. It develops a path-coverage plan, generation-continuous
offload/fallback, and a benchmark whose primary outcome is policy escape rather
than offload throughput.

A broad cross-boundary information-flow candidate was considered again on
`2026-08-28` but not selected. Existing ActPlane and Eunomia material already
covers process/file/network effect labels and layered enforcement, so a broad
version has higher thesis-overlap risk than a narrower cross-boundary property.
A second candidate limited to fast-path versus proxy allow/deny equivalence was
also rejected because it overlaps complete mediation and placement without
capturing what identity the accepted request carries.

The `2026-08-28` report closes the series with a sixth distinct boundary: an L7
proxy can terminate a policy-bound downstream connection and emit or reuse a new
upstream socket, so every intended enforcement point can still be present while
the request loses the principal and policy generation that justified it. The
report develops generation-scoped handoff capabilities, policy-safe
multiplexing, and an authorization-lineage benchmark across kernel fast paths
and proxy slow paths.

### Published progress

1. `2026-08-23`: `/research/ebpf-network-policy-composition/` separates
   policy-language semantics from BPF hook composition and update transactions.
   It asks how several legitimate policy owners can produce one effective
   network verdict while keeping the deciding owner, tier, delegation path, and
   source rule inspectable across policy generations. The report is
   eBPF-centered because the proposed provenance contract is evaluated against
   the realized BPF policy datapath, not only against Kubernetes objects.
2. `2026-08-24`: `/research/ebpf-zero-copy-buffer-ownership/` separates
   allocator-specific lifetime rules from a cross-path ownership contract. It
   asks how packet-buffer leases can preserve owner, DMA reachability, recycle
   generation, and BPF policy provenance across AF_XDP, io_uring ZC Rx, DPDK,
   userspace eBPF, and NIC handoffs. It is eBPF-centered because policy identity
   and BPF/user-runtime transitions are part of the correctness contract rather
   than optional instrumentation.
3. `2026-08-25`: `/research/ebpf-stateful-policy-verification/` separates
   per-program verifier safety from temporal correctness of persistent security
   state. It asks how map-backed policy states can restrict legal transitions,
   transition authority, generations, expiry, capacity failure, and userspace
   writes without forcing the Linux verifier to model unbounded event history.
   It is eBPF-centered because BPF maps, hooks, verifier boundaries, and in-kernel
   transition enforcement are the central mechanism rather than optional probes.
4. `2026-08-26`: `/research/ebpf-authorization-revocation/` separates policy
   rollout from revocation effectiveness. It asks how a previously admitted
   authorization can be made unusable within a measurable bound across conntrack,
   auth maps, socket-local state, endpoint policy, and userspace-managed state.
   It is eBPF-centered because the proposed epoch checks and completion barrier
   are evaluated on persistent BPF datapath state and hot-path policy reuse.
5. `2026-08-27`: `/research/ebpf-complete-mediation-offload/` separates backend
   placement from global security coverage. It asks how host software, SmartNIC
   fast paths, representor fallback, and DPU/offload execution can preserve one
   current enforcement point for every reachable policy-relevant packet path.
   It is eBPF-centered because BPF/XDP attachment, offload capability, policy
   generations, and cross-backend verdict continuity are the core mechanism.
6. `2026-08-28`: `/research/ebpf-l7-proxy-policy-identity/` separates path
   coverage from authorization identity continuity across a semantic proxy
   boundary. It asks how the original principal, policy generation, and
   authorization provenance survive proxy termination, connection pooling,
   multiplexing, retry, and fast/slow-path fallback. It is eBPF-centered because
   the handoff starts and ends at BPF policy boundaries and the proposed
   capability/lineage contract is evaluated against the realized BPF datapath.

Future work should return to this series only when fresh evidence supports a
mechanism beyond these six boundaries. In particular, do not repeat policy
composition, zero-copy ownership, temporal state verification, revocation,
complete mediation, or proxy identity continuity with a different product
example alone.

A direct recount on `2026-08-29` corrected the previous rolling-window arithmetic.
Immediately before the August 29 publication, the newest ten contain **8
eBPF-centered / 0 pure Agent / 2 adjacent systems**, not `7 / 0 / 3`. This is an
operating-record correction only; no report classification changed.

## Completed series — eBPF Observability and Profiling

Historical roadmap state: **Active series — eBPF Observability and Profiling**
through the `2026-08-22` report. The series is completed after that publication;
Networking and Security became the active series for the next normal run.

Working question: **Which important performance and correctness questions remain
unanswerable with today's eBPF observability stack?**

The `2026-08-18` report started this series with **page-level memory attribution**:
how to distinguish allocation from pages that were actually touched, faulted,
reclaimed, migrated, or responsible for sampled memory cost while preserving
provenance across `mmap`, `brk`, allocator, runtime, and process boundaries.

The `2026-08-19` report was a deliberate **adjacent profiling detour** on sampling
bias. Its central mechanism is general profiler measurement design, not eBPF, so
it is classified as adjacent systems.

The first `2026-08-20` report was a second adjacent profiling detour required by
the rolling-window arithmetic. It asks whether a late CUDA kernel start came from
host scheduling, runtime/command-buffer work, a dependency, or device
availability. CUPTI and Nsight Systems are the primary mechanisms; Linux/eBPF
host tracing can be an optional evidence source but is not essential, so this
report is adjacent systems rather than eBPF-centered.

The second `2026-08-20` report follows the same adjacent-systems boundary from a
causality angle. It asks how host API work, runtime handoffs, CUDA Graph replay,
and device execution can retain a trustworthy causal identity. CUPTI/CUDA
dependency semantics are central while eBPF is one possible host observer.

The `2026-08-21` report returns to an eBPF-essential question: how dynamic probes
can observe application-defined pools, queues, caches, and credits without
silently turning stale program semantics into confident diagnoses. It develops a
versioned semantics manifest, runtime semantic validation, and a mutation
benchmark for software evolution.

The `2026-08-22` report closes the series with another eBPF-essential question:
how always-on collectors can reduce high-rate telemetry before export without
silently deleting the evidence required for diagnosis. It develops a compiled
diagnostic contract, bounded state-transition exemplars, coverage-carrying
summaries, and an equal-budget diagnosis-retention benchmark.

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
4. `2026-08-20`: `/research/gpu-host-device-causality/` develops
   generation-scoped host/device causal identity, dependency-aware critical-path
   reasoning, explicit unknown/loss states, and a ground-truth causality
   benchmark. CUPTI/CUDA dependency semantics are essential while eBPF is one
   possible host observer, so this report is adjacent systems.
5. `2026-08-21`: `/research/ebpf-application-resource-semantics/` asks how eBPF
   can dynamically observe application-defined resources without hard-coding
   stale semantics. It develops a versioned resource-semantics manifest compiled
   into eBPF attachments, runtime semantic validation with explicit confidence
   loss, and a software-mutation benchmark. Dynamic no-rebuild instrumentation
   and independent cross-layer validation are central, so this report is
   eBPF-centered.
6. `2026-08-22`: `/research/ebpf-diagnostic-telemetry-compression/` asks how an
   always-on eBPF collector can reduce telemetry volume before export while
   preserving later root-cause diagnosis. It develops a diagnostic-contract
   compiler for BPF retention plans, state-transition exemplars, explicit
   coverage accounting, and an equal-budget incident benchmark. Source-side BPF
   aggregation and exemplar retention are central, so this report is
   eBPF-centered.

Before and after the August 22 publication, the newest ten contain **7
eBPF-centered / 0 pure Agent / 3 adjacent systems**. The incoming eBPF-centered
report ages an eBPF-centered report out of the rolling ten.

A related unresolved question remains: online confidence and adaptive collection
when trace loss, missing probes, or stale schemas make the current representation
uncertain. It is intentionally not a seventh report in this series. The August 22
report first defines what a compact representation promises to preserve; adaptive
fidelity can be revisited later when fresh evidence supports a distinct mechanism.

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
   per-opcode cBPF admission path from the eBPF `io_uring_bpf_ops`
   execution-control path, then developed a typed capability contract, versioned
   ring policy generations, explicit provenance, and a comparative
   control-boundary benchmark.
6. `2026-08-17`: `/research/heterogeneous-ebpf-execution-placement/` separated
   backend compatibility from execution placement and developed a target
   manifest, generation-scoped state ownership, and a ground-truth
   placement/provenance benchmark across kernel, userspace, NIC/DPU, and
   GPU-side targets.

The Daily Report index already exposes the sequence. Report-level acquisition and
navigation evidence is still too young to justify a dedicated public series hub.
Revisit that decision only when evidence shows a retrieval benefit.

## Completed series — GPU and Heterogeneous Runtime Systems

Working question: **What runtime and observability abstractions are missing at
CPU/GPU and host/device boundaries?**

The two August 20 reports are focused adjacent-systems contributions to this
roadmap, but they predate activation and do not count as progress in the normal
sequence. They establish two boundaries that should not simply be repeated:
kernel launch-latency attribution and host/device causal tracing.

The series became active after the `2026-08-28` Networking and Security report
reached that series' normal six-report boundary. It reached its own six-report
post-activation boundary on `2026-09-04`.

The `2026-08-29` report asks what evidence a GPU runtime needs before migrating,
evicting, prefetching, replicating, or remotely mapping Unified Memory pages
under HBM oversubscription. It develops evidence-carrying placement decisions,
placement intent with observable compliance, and a counterexample benchmark that
measures decision regret while progressively revealing richer evidence. It is
adjacent systems because eBPF is optional instrumentation rather than the central
mechanism.

The `2026-08-30` report asks what contract a dynamic GPU instrumentation runtime
needs before an inserted device-side probe can be treated as a faithful observer.
It develops a verified probe-effect manifest, resource-budgeted instrumentation
with explicit coverage, and a counterexample benchmark for observer-induced
failures. It is adjacent systems because the property applies across NVBit,
GTPin, vendor profiling interfaces, native instrumentation, and eBPF-like device
monitors.

The `2026-08-31` report separates retrospective GPU activity from
**allocatability**, the counterfactual question of whether a particular incoming
workload can safely co-reside now. It develops an allocatability certificate,
bounded two-stage admission, and paired counterexamples that hold headline
utilization or occupancy similar while changing the true co-residency result. It
is adjacent systems because eBPF is not required by the admission mechanism.

The `2026-09-02` report separates communicator liveness from application-state
generation consistency. It develops a generation-scoped reconfiguration
certificate, ownership-aware state reconstruction, and a membership-transition
counterexample benchmark whose oracle is semantic consistency rather than
recovery latency. It is adjacent systems because eBPF/GPU instrumentation is
optional evidence and fault-injection machinery.

The `2026-09-03` report returns the series to an eBPF-essential mechanism.
Megakernel compilers deliberately fuse many operator and dependency boundaries
into one persistent GPU kernel, so an external kernel timeline or PC sample no
longer automatically identifies the logical task or request generation. It
develops a versioned semantic task-hook ABI, coverage-carrying on-device eBPF
aggregation, and a counterexample benchmark that holds the outer megakernel
similar while changing the internal task-level cause. It is eBPF-centered
because the device-side eBPF runtime is central to the proposed monitor.

The `2026-09-04` report closes the series with a sixth distinct boundary:
transparent GPU checkpoint/restore can reconstruct CUDA and process state while
the resulting image still crosses application epochs among CPU state, GPU state,
peer communication, persistent-kernel state, and externally visible effects. It
develops a machine-checkable recovery-cut certificate, narrow semantic
quiescence/effect-fence adapters, and an adversarial benchmark whose oracle is
whether recovery corresponds to an allowed application prefix plus legitimate
replay. It is adjacent systems because the consistency mechanism does not require
eBPF.

### Published progress

1. `2026-08-29`: `/research/gpu-memory-placement-evidence/` separates address
   validity and demand faults from the evidence needed to make a placement
   decision under oversubscription. It proposes an evidence-to-decision contract
   and fixed-budget evaluation. It is adjacent systems.
2. `2026-08-30`: `/research/gpu-instrumentation-safety-contract/` separates
   logical probe-state preservation from whole-execution non-interference. It
   develops a probe-effect manifest, explicit resource budgets and observation
   coverage, and adversarial cases around occupancy, timing, synchronization,
   control behavior, and unsupported sites. It is adjacent systems.
3. `2026-08-31`: `/research/gpu-utilization-allocatability/` separates observed
   utilization from candidate-conditioned admission. It distinguishes hard fit,
   interference risk, and unknown evidence, then proposes an inspectable
   allocatability certificate, a bounded online interference probe, and a
   spare-capacity counterexample benchmark. It is adjacent systems.
4. `2026-09-02`: `/research/gpu-membership-generation-continuity/` separates
   communicator liveness from application-state generation consistency. It binds
   member incarnations, a committed application frontier, old-generation
   quiescence/rollback evidence, state manifests, and ownership maps into an
   activation contract, then tests stale completions, rank reuse, partial state
   transfer, and repartitioning as semantic counterexamples. It is adjacent
   systems.
5. `2026-09-03`: `/research/ebpf-gpu-megakernel-observability/` separates
   persistent-kernel identity and PC identity from compiler/runtime logical task
   identity. It proposes a versioned semantic task-hook ABI, a device-side eBPF
   program type for late-bound bounded aggregation with explicit coverage, and a
   fault-injection benchmark comparing low-level evidence, compiler-native task
   profiling, and semantic eBPF hooks. It is eBPF-centered.
6. `2026-09-04`: `/research/gpu-checkpoint-recovery-consistency/` separates
   process/GPU restorable state from an application-consistent recovery cut. It
   proposes a recovery-cut certificate, semantic quiescence and external-effect
   fences, and an adversarial benchmark for duplicate effects, missing commits,
   mixed epochs, pointer errors, communication deadlocks, and unsupported
   recovery coverage. It is adjacent systems.

Before the September 4 publication, the newest ten contain **6 eBPF-centered / 0
pure Agent / 4 adjacent systems**. The incoming adjacent report rotates the
`2026-08-24` eBPF-centered report out, so after publication the mix becomes **5 /
0 / 5**, still inside the normal 5–7 eBPF target band.

Future work should return to this GPU/runtime series only when fresh evidence
supports a mechanism beyond these six boundaries. Do not repeat memory placement,
instrumentation non-interference, allocatability, membership/generation
continuity, launch/host-device causality, megakernel semantic observability, or
checkpoint recovery-cut consistency with a different product example alone.

## Active series — eBPF Optimization and Execution Specialization

Working question: **How can eBPF programs and runtimes specialize to hardware and
workload behavior without silently changing verifier-approved semantics,
portability, or debuggability?**

This series becomes active after the GPU/runtime series reaches its normal
six-report boundary on `2026-09-04`. The rolling mix is then **5 / 0 / 5**, so a
genuinely eBPF-centered question is preferred, but the normal evidence and
novelty gates still apply.

Candidate boundaries include:

- profile-guided or workload-guided BPF re-JIT that can prove semantic
  equivalence while changing machine code or helper lowering;
- specialization contracts that make architecture-specific assumptions explicit
  rather than hiding them behind one nominal BPF program;
- safe delegation of high-level operations to kernel, NIC, DPU, or other
  hardware-specific implementations without repeating the completed execution-
  placement thesis;
- optimization evidence that distinguishes a portable semantic contract from
  one lucky microbenchmark or one JIT backend;
- debugging and provenance for dynamically specialized BPF code so operators can
  explain which version and optimization decision actually executed.

The first report must establish one concrete optimization boundary with current
primary evidence and measurable ground truth. Do not restate the completed
userspace-runtime contract, hook-composition contract, heterogeneous execution
placement report, or transactional-upgrade report under a new optimization name.

The `2026-09-05` report establishes that first boundary. It separates kernel
verifier safety from optimizer equivalence and from the lifetime of profile-based
assumptions. K2 and EPSO show that semantics-preserving BPF rewrites can carry
explicit equivalence checking; Kops shows proof-structured hardware
specialization; the public BpfReJIT design shows how configuration and workload
facts can drive a userspace re-JIT while every candidate still returns through
the stock verifier/JIT. The report develops an optimization-equivalence
certificate, guarded specialization dependencies with bounded deoptimization,
and a phase-shift/rare-path benchmark whose correctness oracle is observable
divergence from the portable source program.

### Published progress

1. `2026-09-05`: `/research/ebpf-runtime-profile-specialization/` asks when a
   workload- or deployment-guided BPF rewrite is still justified as the same
   program. It distinguishes unconditional profile hints, conditional semantic
   assumptions, verifier acceptance, and optimizer equivalence; proposes a
   generation-scoped certificate and invalidation registry; and evaluates
   correctness under adversarial profile staleness before treating speedup as a
   win. It is eBPF-centered because BPF bytecode semantics, verifier/JIT
   acceptance, BPF effects, and link generations are the object of the contract.

Before the September 5 publication, the newest ten contain **5 eBPF-centered / 0
pure Agent / 5 adjacent systems**. The incoming eBPF-centered report rotates the
`2026-08-25` eBPF-centered report out, so after publication the mix remains **5 /
0 / 5**. No classification was changed to obtain that result.

A later report should not repeat "verifier accepted is not equivalent" or stale
profile invalidation with a different optimizer example. The next distinct
boundaries remain architecture-specific specialization contracts, delegated
native operations and their trust boundary, portable optimization evidence
across JIT backends, and debugging/provenance mechanisms that go beyond the
source-to-generation certificate introduced here.

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
