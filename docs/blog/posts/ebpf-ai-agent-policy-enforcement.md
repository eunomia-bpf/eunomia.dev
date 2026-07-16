---
date: 2026-07-15
slug: ebpf-ai-agent-policy-enforcement
description: eBPF AI agent enforcement blocks system-level effects but can't judge intent; ActPlane adds a contextual policy plane linking task, history, and authority.
---

# eBPF AI Agent Enforcement Needs a Contextual Policy Plane

An AI coding agent runs `git commit`, and the kernel sees nothing unusual: a
familiar process writing familiar files. A syscall-level allowlist waves it
through, even though the agent edited a source file after its last test run and
the repository forbids committing in that state. ARMO recently framed the gap
well: what can eBPF AI agent enforcement catch in a real workload, and what
does it miss? Its answer is largely right, because eBPF can observe and mediate
process, file, and network effects, yet a kernel event never explains the
task-level reason behind an action.

Closing that gap is the job of a policy plane: AI agent security needs one that
turns project intent, event history, and authority into a decision the kernel
can enforce. The kernel supplies complete mediation for the effects it covers,
and the policy plane supplies the meaning behind them.

The [ActPlane paper](https://arxiv.org/abs/2606.25189) makes this separation
concrete. Its results show why static allowlists and behavioral baselines remain
useful, yet cannot independently express rules such as “allow a commit only
after the correct tests have passed since the latest source edit.”

<!-- more -->

## Observation, Policy, and Enforcement Are Different Jobs

Discussions of AI agent runtime security often combine three separate jobs:

| Job | Question | System role |
|---|---|---|
| Observation | What did the agent and its descendants actually do? | Correlate agent sessions with process, file, network, and resource evidence |
| Policy | Is this effect allowed now, given the task, history, and rule authority? | Resolve context and maintain the state needed for a decision |
| Enforcement | Can the operation be stopped before it takes effect? | Mediate covered OS operations with eBPF and BPF-LSM |

Separating the jobs shows why each shortcut fails. More kernel telemetry never
closes the semantic gap on its own, and application-level context never reaches
the shell, generated script, or compiled helper that produces the eventual
effect.

A complete design connects the layers without confusing their guarantees.
[AgentSight](https://eunomia.dev/blog/2025/08/26/agentsight-keeping-your-ai-agents-under-control-with-ebpf-powered-system-observability/)
provides system-level profiling and monitoring, while ActPlane takes a concrete,
authority-scoped rule and enforces it over the real process tree.

## What Generic eBPF Enforcement Misses

ARMO's article, [eBPF for AI Agent Enforcement: What Kernel-Level Security
Catches and What It
Misses](https://www.armosec.io/blog/ebpf-based-ai-agent-enforcement/), identifies
two important problems with applying traditional runtime security directly to
agents.

One problem is that system events carry little intent. An eBPF program can watch
the agent open a connection to an unfamiliar endpoint, yet the event never says
whether the agent is deploying to a service the user approved or exfiltrating
credentials after an indirect prompt injection.

The other problem is that agent behavior changes from task to task. A server
usually has a stable process tree and a bounded set of peers. A coding agent creates scripts, discovers
tools, and touches different files and endpoints across tasks. A static
allowlist can become too broad to protect the system or too narrow to preserve
useful autonomy.

Behavioral baselines improve detection because they distinguish familiar from
unusual behavior. Authorization asks a different question: is this operation
permitted under the current task and policy? A familiar `git commit` can still
violate a repository rule when the agent edited a source file after its last
test run. An unfamiliar deployment endpoint can be legitimate when the user
explicitly selected it. Frequency and anomaly scores cannot determine either
answer on their own.

## The ActPlane Study Reveals the Missing Policy Inputs

ActPlane studies the instructions developers already place in `CLAUDE.md` and
`AGENTS.md`. Across 64 popular repositories, 84 instruction files, and 2,116
statements, the paper reports:

- 64.3% of statements are behavioral directives.
- 83% of those directives concern system-observable behavior.
- 81% of repositories contain at least one cross-event directive.
- 74% of system-observable directives need project or task context before they can become concrete rules.

The findings identify four policy inputs that generic per-event rules usually
do not carry:

| Policy input | Example question | Why it matters |
|---|---|---|
| Task context | Which command is “the full test suite” in this repository? | Natural-language instructions must resolve to concrete commands, paths, and endpoints |
| Event history | Did tests exit successfully after the most recent relevant write? | Many rules describe freshness, ordering, lineage, or information flow across events |
| Authority | Was the rule defined by an administrator, repository owner, or the task agent itself? | A compromised task agent must not weaken inherited constraints |
| Recovery feedback | What state must the agent repair after a denial? | A semantic reason helps the agent comply instead of retrying through another path |

ActPlane represents these inputs in a compact policy DSL, tracks labels and
temporal state across the process tree, and compiles enforcement state into
eBPF. Higher-authority policy domains are loaded before the task agent starts;
child domains may add or narrow constraints, while inherited rules remain in
force.

That contextual layer is the key addition to ARMO's behavioral-baseline framing:
a baseline estimates what is normal, while a contextual policy states what is
allowed, by whom, and at which point in a task. Production systems can use both,
because they answer different questions.

For the DSL, label propagation, temporal gates, and deployment architecture,
read the existing mechanism guide: [ActPlane: Pushing Agent Harness Enforcement
Down to Kernel eBPF](https://eunomia.dev/blog/2026/05/31/actplane-pushing-agent-harness-enforcement-down-to-kernel-ebpf/).

## How the Control Models Compare

No single control covers every layer of an agent system, so the question worth
asking is which decision each control can make reliably.

| Control | Task meaning | Cross-event state | Indirect OS effects | Primary use |
|---|---:|---:|---:|---|
| Prompt or tool-call guard | Strong at the proposed action | Limited | Limited after a shell-out | Pre-action guidance and tool screening |
| Static OS allowlist | None unless encoded manually | Limited | Strong for covered hooks | Stable resource boundaries |
| Behavioral baseline | Inferred from prior behavior | Statistical | Strong when fed system telemetry | Detection and investigation |
| Contextual OS policy | Explicit task and project context | Explicit | Strong for covered hooks | Deterministic enforcement of loaded rules |

This table does not make Falco, Tetragon, seccomp, sandboxes, or anomaly
detection obsolete. Each remains valuable within its decision boundary. The
architectural mistake is expecting a kernel event stream to infer task intent,
or expecting a tool-call guard to mediate effects it never sees.

## What the ActPlane Results Support

The paper evaluates the full path from natural-language directive to policy
translation, runtime intervention, feedback, and agent recovery.

- Across 190 direct, script, hidden, and compliant traces, ActPlane reaches 75.8% Decision Compliance Rate, 22 to 31 percentage points above the evaluated prompt-filter, tool-regex, tool-level IFC, and feedback-free kernel IFC baselines.
- The same kernel rules produce 86 correct violation-trace outcomes with semantic feedback and 27 without it.
- One policy revision raises violation detection from 77.2% to 94.7%, showing that policy translation quality remains a first-class dependency.
- No-hit overhead is 1.9% on an agent-trace replay and 6.5% on a Linux kernel build at 32 active rules; both workloads remain below 8.4% at 100 rules.
- On 361 OpenAgentSafety tasks, ActPlane prevents 78 of 106 baseline-unsafe effects. Policies also activate on 16% of baseline-safe tasks, exposing the cost of overly broad rules.

These results support a bounded claim: eBPF is a strong enforcement substrate
for observable OS effects, including effects reached through indirect process
paths, but the quality and authority of the loaded policy still determine
whether the enforced decision is correct.

## A Practical Runtime Security Architecture

AgentSight and ActPlane fit into a reviewable control loop:

```text
agent and task context
        ↓
AgentSight runtime evidence and audit
        ↓
operator or trusted policy-agent review
        ↓
concrete, authority-scoped policy
        ↓
ActPlane eBPF enforcement
        ↓
semantic feedback to the agent
```

AgentSight provides evidence for profiling, detection, investigation, and
candidate-rule review. It does not automatically authorize or block actions.
ActPlane enforces loaded rules over covered process, file, and network events.
It does not identify every harmful prompt or understand arbitrary generated
content.

System-level runtime safety therefore remains one layer of a larger design,
where isolation limits blast radius, identity and authorization constrain
available capabilities, content and protocol checks cover semantics outside
syscall events, and a trusted exception path handles high-impact changes. Our
broader guide,
[Runtime Observability and Enforcement for Opaque AI Agents with
eBPF](https://eunomia.dev/blog/2026/05/25/runtime-security-for-ai-agents/),
places these controls into a three-layer security model.

## Common Questions

### Is eBPF enough for AI agent security?

eBPF provides strong observation and mediation for covered OS events. It does
not independently supply task intent, policy authority, content semantics,
identity, or isolation. Those inputs and controls must surround the kernel
enforcement layer.

### Can behavioral baselines replace policy?

Behavioral baselines detect deviations from prior behavior. Policy defines
permission for the current task. A mature system can use anomaly detection to
propose or prioritize rules, followed by authority-aware review before
enforcement.

### Does AgentSight enforce ActPlane policies?

No. AgentSight is a system-level profiler and monitor. ActPlane is the policy
enforcement component. Evidence can inform policy review, but the projects keep
observation and enforcement as distinct responsibilities.

### What remains outside ActPlane's coverage?

Chat-only harm, unsafe generated content, service-side effects beyond covered
OS hooks, missing hooks, kernel compromise, and incorrectly generated policy
require additional controls. ActPlane's guarantee applies to loaded rules at
the OS events its enforcement engine mediates.
