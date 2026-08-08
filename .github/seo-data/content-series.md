# Daily Report content series

This file is the authoritative thematic roadmap for scheduled Daily Report work.
The daily task should advance a coherent series rather than choose unrelated
trending topics. Publication remains quality-gated; a series is a research
constraint, not a quota.

## Series rules

- Keep one **active series** at a time. A normal series contains roughly 4–6
  substantial reports developed over multiple daily research runs.
- Search inside the active series first. A new report must answer a distinct
  question and materially advance the series argument rather than restate an
  earlier report with different examples.
- Prefer continuity: reuse evidence maps, unresolved questions, experiments, and
  counterexamples from earlier reports when they remain valid.
- An out-of-series report is allowed only for a material external development
  with durable systems consequences. Record why it justified interrupting the
  active series.
- Data analysis may influence which series becomes active next. Search demand,
  landing-page behavior, GitHub activity, reader pathways, and technical changes
  are evidence, but they do not override the research quality gate.
- Do not publish merely to maintain cadence. A daily research run may update the
  evidence map, revise an existing report, or record a no-report result.
- After a series has at least three strong reports, consider a public series hub
  and stronger internal linking. Do not create thin hub pages in advance.

## Active series — Agent Runtime Correctness

Working question: **What runtime guarantees are needed when an AI agent's work
spans multiple model calls, tools, processes, checkpoints, permissions, and
external side effects?**

This series should connect correctness mechanisms that are usually discussed
separately: concurrency, recovery, authority, effect control, and outcome
verification.

### Existing anchor

1. **When Several AI Agents Work at Once, Who Makes Sure the Final Result Is
   Right?**
   - Existing page: `/research/parallel-agent-effect-serializability/`
   - Role in the series: shows why isolated branches and locally successful
     workers are insufficient when effects must be committed together.

### Preferred next questions

2. **What state has to survive checkpoint and restore together?**
   - Investigate local workspace, external object versions, credentials,
     delegated processes, pending effects, policy state, and replay safety.
   - Look for a concrete recovery contract rather than another checkpoint format.

3. **When should an old approval stop authorizing a long-running agent?**
   - Study changes in repository state, task intent, policy, credentials, tool
     implementations, and delegation relationships between approval and effect.
   - Compare time-based expiry with state-bound revalidation at real effect
     boundaries.

4. **Which agent side effects can be rolled back, compensated, or must be
   serialized?**
   - Build a practical effect taxonomy across files, Git, databases, cloud APIs,
     messages, payments, deployments, and credentials.
   - Identify where transaction semantics end and compensation or prevention must
     begin.

5. **What should the durable unit of agent work contain?**
   - Ask whether one work unit should bind execution state, authority, effects,
     recovery, observability, accounting, and placement.
   - Compare request-, process-, session-, and trajectory-scoped abstractions.

6. **How should an agent prove that a task succeeded after real side effects?**
   - Separate tool return codes from outcome verification.
   - Explore independent oracles, state receipts, postconditions, and failure
     cases where every individual step reports success but the user goal is not
     satisfied.

Do not force these exact titles. They are research questions and continuity
anchors. The final title and thesis must come from evidence.

## Queued series — Agent Observability and Evidence

Working question: **Under bounded overhead and privacy budgets, what evidence is
needed to explain and verify long-running agent behavior?**

Existing anchor:

- `/research/agent-trace-evidence-budget/`

Candidate sequence:

1. cross-layer causal tracing from agent actions to processes, files, sockets,
   repository versions, and external objects;
2. causal flight recorders that retain decisive evidence instead of fixed time
   windows;
3. diagnosis benchmarks that measure whether retained evidence distinguishes
   competing root causes, not merely whether events were collected;
4. asynchronous and syscall-level profiling for tool-heavy agents;
5. trace compression that preserves semantic dependencies and uncertainty;
6. binding trajectories to changing workspace and environment state.

## Queued series — eBPF as Runtime Infrastructure

Working question: **Which missing runtime mechanisms become possible when eBPF
is treated as a programmable systems substrate rather than only an observability
feature?**

Candidate areas include userspace eBPF, multi-tenant composition, stateful and
transactional upgrades, async profiling, new I/O hooks, policy enforcement, and
portable execution across heterogeneous runtime boundaries. Each report must
identify a mechanism that is not reducible to simply “add eBPF.”

## Queued series — GPU Runtime and Observability

Working question: **What runtime abstractions are missing for diagnosing,
controlling, and composing modern GPU workloads?**

Candidate areas include GPU/CPU causal profiling, memory movement, megakernel
observability, programmable device-side instrumentation, distributed GPU
coordination, utilization versus allocatability, and how host/runtime state
changes the interpretation of device traces.

## Choosing the next series

Finish or deliberately pause the active series before switching. A switch should
be justified by at least one of:

- the active series has answered its main question well enough;
- remaining questions lack evidence or testability;
- a queued series has materially stronger technical evidence and reader demand;
- a major systems development changes the research frontier;
- the site's search, reader, or GitHub data shows a durable opportunity that
  matches Eunomia's technical ownership.

Record a series switch and its reason in the daily operating record and update
this file so the repository, not chat history, remains authoritative.
