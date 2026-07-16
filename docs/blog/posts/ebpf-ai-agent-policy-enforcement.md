---
date: 2026-07-15
slug: ebpf-ai-agent-policy-enforcement
description: AI agent safety rules live in CLAUDE.md and AGENTS.md. A study of 2,116 statements shows 64% are policies, most needing OS-level enforcement with context.
---

# What 2,116 CLAUDE.md and AGENTS.md Rules Reveal About AI Agent Safety

An AI coding agent runs `git commit`, and the kernel sees nothing unusual: a familiar process writing familiar files. The repository's CLAUDE.md says "Run the full test suite before committing," and the agent edited source code after its last test run. That rule, drawn verbatim from the study's dataset, is not enforced by any layer in the stack.

The [ActPlane paper](https://arxiv.org/abs/2606.25189) quantifies this gap. It measures the policies developers actually write in CLAUDE.md and AGENTS.md files, classifies what enforcing them requires, and validates that OS-level enforcement combined with semantic feedback works. The paper's own summary of the landscape: 64% of statements are policies, 83% involve system actions, and 74% depend on context that cannot be pre-defined statically.

<!-- more -->

## What Developers Actually Write in CLAUDE.md and AGENTS.md

Most discussions of AI agent safety start from threat models or attack surfaces. ActPlane starts from a different question: what do developers already tell their agents to do and not do, and what would it take to enforce those instructions?

The study examines 64 popular repositories containing CLAUDE.md and AGENTS.md files (median 20K GitHub stars, snapshot from 2026-05-23), covering 84 instruction files and 2,116 individual statements. Unlike prior work that analyzed instruction files at the file or section-heading level, ActPlane classifies every statement independently.

Across those 2,116 statements, 64% are policies: they require, forbid, or condition a specific agent action. The remaining 36% are descriptive context, such as architecture notes or project background. Policy density varies widely across repositories, from 0% to 97%, with 70.1% of repositories containing more policy statements than descriptive ones. Development Process and Implementation Details are the heaviest policy topics at 87% and 85%; Architecture is mostly descriptive, with only 23% policies.

![Policy fraction per repository across 64 repos with CLAUDE.md or AGENTS.md](imgs/actplane-empirical_rq1_policy_fraction.png)

Four real statements from the dataset illustrate the range of enforcement requirements:

| Statement | Enforcement level | Context |
|---|---|---|
| S4: "Never push to main directly." | per-event | self-contained |
| S6: "Run the full test suite before committing." | cross-event | project |
| S7: "Data read from .env must not reach the network." | cross-event | project |
| S8: "Do not update dependencies without approval." | per-event | task |

## Most Rules Are System-Observable, and the Hard Ones Are Cross-Event

The paper classifies each policy into the first matching tier of an enforcement waterfall: semantic-only covers reasoning, communication, or output style; content covers predicates over file contents; per-event covers a single command, file access, or network connection; and cross-event covers policies that depend on temporal ordering or data lineage across operations. The union of content, per-event, and cross-event tiers is called system-observable.

Of the 1,361 policies in the dataset, only 17% are semantic-only. The remaining 83% are system-observable, meaning a kernel-level monitor could in principle evaluate them: 38% require content inspection, 29% match a single OS event, and 16% require cross-event state.

![Enforcement waterfall showing semantic-only, content, per-event, and cross-event distribution across 1,361 policies](imgs/actplane-empirical_waterfall_enforcement.png)

These cross-event policies follow four recurring patterns: temporal ordering constrains sequencing ("run tests before committing"); cross-file consistency links changes across artifacts ("update docs when behavior changes"); multi-step workflows enforce release checklists with verification gates; and conditional triggers couple operations ("if you change specs, also update the SDK"). None can be decided from a single event: enforcement must record what ran, in what order, and what has changed since. Such policies are widespread, with 81% of repositories containing at least one cross-event policy and 43% spanning all four enforcement tiers.

Context dependence compounds the enforcement challenge. Of the 1,127 system-observable policies, only 26.4% are self-contained. The majority, 64.2%, require project context: "the test suite" or "upstream source" must be resolved against a specific repository before the policy becomes a concrete rule. Another 9.4% require task context, such as "unless explicitly requested" or "without approval." The policies that require tracking state across events are also the ones that rarely specify the concrete commands and paths needed to write the rule: cross-event policies are 95% context-dependent (77% project, 19% task), compared to 58% for content policies.

![Context waterfall showing self-contained, project context, and task context distribution across 1,127 system-observable policies](imgs/actplane-empirical_waterfall_context.png)

A fixed set of static rules can cover only the self-contained fraction. Instantiating the rest requires reading the repository and interpreting the current task before any check can run.

## Why No Existing Layer Enforces These Rules

Prompt instructions rely on the model's own compliance, but they are vulnerable to prompt injection and compete with the user's task prompt for attention in a long context window. Separate agents or LLM guards can check prompts, responses, or action trajectories at runtime, but these checks are inherently probabilistic.

Tool-call guardrails and application-level IFC systems intercept at the harness boundary deterministically, but they observe only harness-mediated requests, not system-level effects once a tool starts executing. An indirect subprocess, shell-out, or compiled binary can bypass the tool boundary.

OS-level mechanisms like seccomp, AppArmor, Landlock, and Tetragon control resource access, not actions in the sense developers write about. They expect statically pre-written policies and return opaque errors that confuse the agent: a bare EPERM with no explanation of what rule was violated or how to recover.

The core insight from the paper ties these layers together: most rules need project or task context that resides with the agent, so the agent itself must be able to turn policies into concrete rules; yet many policies define event ordering or data flow that is invisible to tool-call guardrails, so the rules must be concrete enough for deterministic OS-level enforcement. Bridging that gap is what ActPlane addresses.

## Inside ActPlane: Rules Agents Write, the Kernel Enforces

![ActPlane overview: the agent closest to the task writes concrete policy DSL, compiled and enforced inside the OS kernel](imgs/actplane-illustration.png)

Each ActPlane rule has five components: a source that identifies what is being governed, a target operation (such as exec, write, or connect), an effect, an optional temporal gate, and a reason string for semantic feedback. The paper's running example makes this concrete:

```
kill exec "git" "commit" unless after exec "go" "test" exits 0
```

This rule kills any `git commit` unless `go test` has exited successfully since the most recent relevant source edit. The reason field, omitted here for brevity, provides the agent with a structured explanation when the rule fires.

Effects form a gradient matching the distinction between instructions and constraints. Block is a pre-operation synchronous denial with no TOCTOU gap: the kernel intercepts the system call before it executes, and the agent can reroute. Kill terminates the process after the operation has begun, preventing the agent from switching to an alternate channel after a block. Notify delivers guidance without stopping the action. Constraints use block or kill; instructions use notify.

Temporal gates let rules express ordering rather than point-in-time predicates. The `after ... since ...` construct encodes that one event must have occurred after another: tests must have run after the most recent edit, not merely at some earlier point. The `exits N` qualifier distinguishes successful from failed exits. A lineage gate checks process ancestry, allowing rules to restrict operations to specific process trees.

Information-flow labels propagate along fork, exec, read, write, and connect and are monotonic: once a process reads a labeled object, the label cannot be removed. When a process reads `.env`, it acquires that file's source label. If it later attempts to connect to an external endpoint, the rule matching that label fires and blocks the connection. This is how S7 from the study ("Data read from .env must not reach the network") becomes an enforceable cross-event rule.

Policy authority relies on a temporal trust boundary. Rules loaded before the agent starts are higher-authority and immutable to the agent. The agent and its sub-agents can add new rules or narrow existing ones within child domains, but they cannot weaken, remove, or disable inherited constraints. Runtime deltas arrive through a ring buffer and pass through an in-kernel authority checker that validates each change against the domain hierarchy before activation.

Because labels are monotonic, long-running sessions risk over-tainting: after many reads, a process can accumulate so many labels that every subsequent operation triggers a rule. ActPlane mitigates this by clearing inherited labels when a fresh subprocess is spawned, bounding taint accumulation to the lifetime of each process rather than the entire session.

The implementation is compact. The userspace compiler and runner are roughly 3.2K lines of Rust. The eBPF enforcement engine is roughly 1.8K lines of BPF C. BPF-LSM hooks handle pre-operation decisions (block), while tracepoints handle observation and post-operation termination (kill). Labels are stored as 64-bit bitmasks in per-object BPF maps, and propagation reduces to a single bitwise OR. The engine supports up to 128 concurrent rules; the largest repository in the study had 66 policies. For deeper coverage of the deployment architecture and mechanism details, see [ActPlane: Pushing Agent Harness Enforcement Down to Kernel eBPF](https://eunomia.dev/blog/2026/05/31/actplane-pushing-agent-harness-enforcement-down-to-kernel-ebpf/).

## Does It Work? What the Evaluation Shows

Policy translation is no longer a bottleneck. A Codex agent compiled all 607 OS-enforceable policies from the dataset into ActPlane rules on the first or second attempt, with only 2 of 607 needing a syntax-error retry. The cost was roughly $0.028 per policy, compared to approximately $11 per rule when written manually.

Contextual enforcement resolves far more violations than any baseline. On the decision-compliance benchmark (190 traces, 38 rules drawn from the empirical study), ActPlane achieves a 75.8% Decision Compliance Rate. The baselines trail well behind: prompt-filter at 48.4%, tool-regex at 45.3%, FIDES (tool-level IFC) at 48.9%, and kernel IFC without feedback at 53.7%. The gap concentrates on violation traces, where ActPlane correctly resolves 86 of 114, compared to 27 to 44 for baselines, a 2.0 to 3.2 times improvement. The advantage comes primarily from indirect execution paths that tool-call interception cannot observe.

Semantic feedback is the dividing line between compliance and retry loops. Full ActPlane produces three times more correct violation-trace outcomes than the same engine without feedback, 86 versus 27. Recovery rates tell the same story: 97.7% with semantic feedback versus 31.4% without. When the agent learns why an action was blocked and what state it needs to repair, it reroutes and completes the task. When it receives only a bare denial, it retries the same forbidden action through alternative paths.

Overhead fits comfortably in a development workflow. End-to-end overhead on agent traces is 1.9%, and a Linux kernel build with 32 active rules adds 6.5%. Even at 100 rules, overhead stays below 8.4%.

External safety benchmarks confirm breadth beyond the paper's own dataset. On 361 OpenAgentSafety personal-assistant tasks, ActPlane loads agent-generated safety policies as higher-authority rules before the agent begins, preventing 74% of baseline-unsafe behaviors (78 of 106 unsafe outcomes blocked).

The [ActPlane source code](https://github.com/eunomia-bpf/ActPlane) is available on GitHub.

## Common Questions

### Is eBPF enough for AI agent safety?

eBPF provides deterministic enforcement over OS events: file writes, process launches, network connections, and similar system actions. That covers the 83% of policies that are system-observable, but the remaining 17%, which cover reasoning, communication style, or output quality, require harness-level controls outside the kernel. Task intent, policy authority, content semantics, and isolation are all concerns that the layers around kernel enforcement must handle.

### Can behavioral baselines replace policy?

Behavioral baselines answer "is this unusual?" by flagging deviations from historical patterns. Policy answers "is this permitted under the current task?" A routine `git commit` can be perfectly normal by baseline standards and still violate a project rule. An unfamiliar deployment endpoint can trigger an anomaly alert and still be legitimate because the user explicitly requested it. Detection and authorization are two different problems, and a mature system uses both: anomaly signals can propose candidate rules, which then go through authority-aware review before becoming enforced policy.

### Does AgentSight enforce ActPlane policies?

No. [AgentSight](https://eunomia.dev/blog/2025/08/26/agentsight-keeping-your-ai-agents-under-control-with-ebpf-powered-system-observability/) is a system-level profiler and monitor that captures what agents do at runtime. ActPlane is the enforcement engine that blocks or redirects actions violating loaded policies. The two serve complementary roles: runtime evidence from AgentSight can feed into policy review and refinement, but observation and enforcement remain separate responsibilities.

### What remains outside ActPlane's coverage?

The paper's threat model is explicit about boundaries. Semantically equivalent operations can bypass string-based matching: an agent could use a custom Git client instead of calling `exec git`, though the underlying `connect` and `write` system calls remain visible. File-content semantics, kernel compromise, CAP_BPF compromise, and side channels are all out of scope. The 17% of policies that are semantic-only, covering reasoning quality, communication tone, or output formatting, require harness-layer handling rather than kernel enforcement.

The dataset itself is this paper's most distinctive contribution. Before ActPlane, no one had measured what developers actually ask their agents to obey or how those rules distribute across enforcement requirements. Most of these rules are already written down in CLAUDE.md and AGENTS.md files across thousands of repositories; what has been missing is an enforcement layer that can read the project context, understand the current task, and turn those natural-language policies into concrete, kernel-level rules. The [ActPlane repository](https://github.com/eunomia-bpf/ActPlane) contains the implementation, and a broader three-layer security model placing kernel enforcement alongside isolation, identity, and content controls appears in [Runtime Observability and Enforcement for Opaque AI Agents with eBPF](https://eunomia.dev/blog/2026/05/25/runtime-security-for-ai-agents/).
