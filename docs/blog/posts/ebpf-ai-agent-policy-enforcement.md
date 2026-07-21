---
date: 2026-07-15
slug: ebpf-ai-agent-policy-enforcement
description: AI agent rules look simple in CLAUDE.md, but ActPlane's 2,116-statement study shows why context and layered OS enforcement decide what can be checked.
---

# An Empirical Study: AI Agent Rules Need Context and Layered Enforcement

A rule like "run the full test suite before committing" looks simple until an AI Agent edits a source file after the last test run and then calls `git commit`. The kernel sees an ordinary process writing a commit object, while the harness, the software layer managing the agent's tool calls and session state, sees one more tool call. Deciding whether to allow the commit still depends on which test result remains valid and which edit invalidated it.

The [ActPlane paper](https://arxiv.org/abs/2606.25189) measures the gap between the behavioral rules developers write and the subset a system can actually check. Its statement-level analysis of 2,116 instructions shows that developers are not short of rules; the difficulty lies in turning natural-language requirements into state that a system can observe and evaluate over time. Many rules concern files, processes, or network activity but still depend on repository structure, task progress, or prior events, so a single OS hook can cover only part of the policy set. This article draws on our ActPlane empirical study, then follows the gap into the system design and evaluation.

<!-- more -->

## Developers Have Already Written the Policies

Most discussions of AI agent safety start from threat models or attack surfaces. ActPlane starts from a different question: what do developers already tell their agents to do and not do, and what would it take to enforce those instructions?

The study examines 64 popular repositories containing CLAUDE.md and AGENTS.md files (median 20K GitHub stars, snapshot from 2026-05-23), covering 84 instruction files and 2,116 individual statements. Unlike prior work that analyzed instruction files at the file or section-heading level, ActPlane classifies every statement independently. The study asks three questions: are instruction files primarily behavioral policies or descriptive context? Which policies require OS-level enforcement, and what kinds of OS-level checks do they need? What context is needed to instantiate these policies into concrete, enforceable rules?

Statements were extracted through a two-pass LLM agent-assisted pipeline that recorded source line ranges and four labels per statement: content type, topic, enforcement level, and context requirement. A validation script verified full source coverage and verbatim span matching, then two independent agents (Claude and Codex) cross-checked the results. A stratified sample of 100 statements went through independent human review, which confirmed the labels were correct.

Across those 2,116 statements, 64% are policies that require, forbid, or condition a specific agent action. The remaining 36% are descriptive context, such as architecture notes or project background. Policy density varies widely across repositories, from 0% to 97%, with 70.1% of repositories containing more policy statements than descriptive ones. File- or heading-level studies do not report this statement-level distribution, which is why the finer classification matters.

To understand how policies distribute across concerns, the study assigns each statement to one of 12 topic categories adapted from prior instruction-file research, applied at statement granularity rather than file granularity. Development Process and Implementation Details dominate the policy landscape at 87% and 85% respectively. Architecture is mostly descriptive at 23% because directory layouts and design summaries make up the bulk of those sections. The imported source figures call policy statements directives and call the system-observable policy subset system-level directives. The prose follows the paper's policy and system-observable terminology.

![Policy fraction per repository across 64 repos with CLAUDE.md or AGENTS.md](imgs/actplane-empirical_rq1_policy_fraction.png)

Five real statements from the dataset illustrate the range of enforcement requirements:

| Statement | Enforcement level | Context |
|---|---|---|
| S4: "Never push to main directly." | per-event | self-contained |
| S5: "Never modify upstream source code." | per-event | project |
| S6: "Run the full test suite before committing." | cross-event | project |
| S7: "Data read from .env must not reach the network." | cross-event | project |
| S8: "Do not update dependencies without approval." | per-event | task |

## The Enforcement Gap Begins with Context

Each policy exits at the first matching tier of an enforcement waterfall. Semantic-only covers reasoning, communication, or output style; content covers predicates over file contents; per-event covers a single command, file access, or network connection; and cross-event covers policies that depend on temporal ordering or data lineage across operations. The union of content, per-event, and cross-event tiers is called system-observable.

Of the 1,361 policies in the dataset, only 17% are semantic-only. The remaining 83% are system-observable, comprising 38% that require content inspection, 29% that match one OS event, and 16% that require cross-event state. Only the per-event and cross-event classes, 45% together, form the OS-enforceable subset. Cross-event policies concentrate in Development Process, which accounts for 39.5% of all cross-event policies.

![Enforcement waterfall showing semantic-only, content, per-event, and cross-event distribution across 1,361 policies](imgs/actplane-empirical_waterfall_enforcement.png)

"Run tests before committing" belongs to temporal ordering because one event must happen after another, not merely at some earlier point. Three other relationships recur in the dataset. Cross-file consistency couples a source edit to a documentation update when behavior changes. Multi-step workflows require each release gate to complete before the next begins. Conditional triggers activate only when a precondition is met, as in "if you change specs, also update the SDK."

None of these can be decided from a single event, so enforcement must record what ran, in what order, and what has changed since. Such policies are widespread, with 81% of repositories containing at least one cross-event policy and 43% spanning all four enforcement tiers.

Context dependence compounds the enforcement challenge. Of the 1,127 system-observable policies, only 26.4% are self-contained. The majority, 64.2%, require project context: "the test suite" or "upstream source" must be resolved against a specific repository before the policy becomes a concrete rule. Even a per-event policy like S5, "Never modify upstream source code," requires resolving which paths constitute "upstream source" before a file-write check can fire. Another 9.4% require task context, such as "unless explicitly requested" or "without approval."

The two difficulties compound, because the policies that require tracking state across events are also the ones that rarely specify the concrete commands and paths needed to write the rule. Cross-event policies are 95% context-dependent (77% project, 19% task), compared to 58% for content policies. A policy that says "run tests before commit" sounds simple until the enforcement engine needs to know which test command to watch for, which source directories count as "relevant edits," and whether the test passed or merely ran.

![Context waterfall showing self-contained, project context, and task context distribution across 1,127 system-observable policies](imgs/actplane-empirical_waterfall_context.png)

A fixed set of static rules can cover only the self-contained fraction. Instantiating the rest requires reading the repository and interpreting the current task before any check can run.

> **Agent policy enforcement begins by compiling repository and task context into concrete state that deterministic checks can evaluate.**

## One Rule Crosses Several Enforcement Layers

Prompt instructions rely on the model's own compliance, but they are vulnerable to prompt injection and compete with the user's task prompt for attention in a long context window. Separate agents or LLM guards can check prompts, responses, or action trajectories at runtime, but these checks are inherently probabilistic.

Tool-call guardrails and application-level information-flow control (IFC) systems intercept at the harness boundary deterministically, but they observe only harness-mediated requests, not system-level effects once a tool starts executing. An indirect subprocess, shell-out, or compiled binary can bypass the tool boundary. Consider an agent that writes a Python script containing `subprocess.run(["git", "push"])` and then executes it: the tool-call layer sees "run python script.py," not the `git push` inside it.

OS-level mechanisms like seccomp, AppArmor, Landlock, and Tetragon control resource access, not actions in the sense developers write about. They expect statically pre-written policies and return opaque errors that confuse the agent: a bare EPERM with no explanation of what rule was violated or how to recover.

Those layers still leave a structural split between who holds policy context and who can see every execution path. Most rules need project or task context that resides with the agent, so the agent itself must turn policies into concrete rules, yet many policies define event ordering or data flow that tool-call guardrails never see, so the rules must still be concrete enough for deterministic OS-level enforcement. Bridging that gap is what ActPlane addresses.

Two design requirements follow. The policy specification must be agent-writable yet OS-enforceable, so the agent can produce concrete rules from natural-language policies with minimal expertise and receive semantic feedback to understand violations and recover. Enforcement must also stay safe, isolated, and efficient, meaning agent-authored policy must not weaken constraints set by higher authority, must not affect other agents' policies, and must not slow the agent's normal workload.

## Compiling Intent into Enforceable State

![ActPlane overview: the agent resolves policy context, ActPlane compiles the DSL, and the kernel enforces it](imgs/actplane-illustration.png)

Each ActPlane rule has five components: a source that identifies what is being governed, a target operation (such as exec, write, or connect), an effect, an optional temporal gate, and a reason string for semantic feedback. The paper's running example makes this concrete:

```
kill exec "git" "commit" unless after exec "go" "test" exits 0 since write "**/*.go"
```

This rule kills any `git commit` unless `go test` has exited successfully since the most recent relevant source edit. The reason field, omitted here for brevity, provides the agent with a structured explanation when the rule fires.

Effects form a gradient matching the distinction between instructions and constraints. Block is a pre-operation synchronous denial with no TOCTOU gap: the kernel intercepts the system call before it executes, and the agent can reroute. Kill terminates the process after the operation has begun, preventing the agent from switching to an alternate channel. Notify delivers guidance without stopping the action. Constraints use block or kill; instructions use notify.

Temporal gates let rules express ordering rather than point-in-time predicates. The `after ... since ...` construct encodes that one event must have occurred after another: tests must have run after the most recent edit, not merely at some earlier point. The `exits N` qualifier distinguishes successful from failed exits. A lineage gate checks process ancestry, allowing rules to restrict operations to specific process trees.

Information-flow labels propagate along fork, exec, read, write, and connect and are monotonic: once a process reads a labeled object, the label cannot be removed. When a process reads `.env`, it acquires that file's source label. If it later attempts to connect to an external endpoint, the rule matching that label fires and blocks the connection. This is how S7 from the study ("Data read from .env must not reach the network") becomes an enforceable cross-event rule.

Policy authority relies on a temporal trust boundary. Rules loaded before the agent starts are higher-authority and immutable to the agent. The agent and its sub-agents can add new rules or narrow existing ones within child domains, but they cannot weaken, remove, or disable inherited constraints. Runtime deltas arrive through a ring buffer and pass through an in-kernel authority checker that validates each change against the domain hierarchy before activation. The trusted computing base consists of the kernel enforcement engine and the higher-authority policy, and everything below this boundary is untrusted execution. A compromised userspace agent therefore cannot modify the active rule set beyond what its domain hierarchy permits.

Because labels are monotonic, long-running sessions risk over-tainting: after many reads, a process can accumulate so many labels that every subsequent operation triggers a rule. In a typical coding session, a process might read dozens of configuration and source files, and without mitigation each read adds a label, so after enough reads every subsequent write or connect would match some rule. ActPlane mitigates this by clearing inherited labels when a fresh subprocess is spawned, bounding taint accumulation to the lifetime of each process rather than the entire session.

The 607-policy dataset exercises most DSL features and validates the language's expressiveness. Effects skew toward observation: 66% of clauses are notify, 29% are block, and only 5% are kill, reflecting that most policies monitor rather than prevent. Hooks concentrate on code execution (60% exec) and file mutation (37% write), with network and cleanup operations under 1% each. Cross-event features see substantial use, with 28% of policies using an `after/since` temporal gate and 214 using `unless` to encode exceptions.

The implementation stays compact enough to reason about. The userspace compiler and runner are roughly 3.2K lines of Rust, and the eBPF enforcement engine is roughly 1.8K lines of BPF C. BPF-LSM hooks handle pre-operation decisions (block), while tracepoints handle observation and post-operation termination (kill). Labels live as 64-bit bitmasks in per-object BPF maps, so propagation reduces to a single bitwise OR, and the engine can support up to 128 concurrent rules, comfortably above the largest observed repository's 66 policies. For deeper coverage of the deployment architecture and mechanism details, see [ActPlane: Pushing Agent Harness Enforcement Down to Kernel eBPF](https://eunomia.dev/blog/2026/05/31/actplane-pushing-agent-harness-enforcement-down-to-kernel-ebpf/).

## Recovery Reveals What Enforcement Alone Misses

Before recovery can matter, the DSL has to accept the policies developers already wrote. A Codex agent compiled all 607 OS-enforceable policies into ActPlane rules on the first or second attempt, with only 2 of 607 needing a syntax-error retry. The translation completed in 34 minutes using 7 subagents with 4 running in parallel, expanding 607 policies into 1,283 rule lines. The cost was roughly $0.028 per policy using 1.7M input tokens and 177K output, compared to approximately $11 per rule at typical software-engineer rates. Most policies are structurally simple, with 74% having at most two enforcement clauses and even the 95th percentile staying under 152 tokens.

Contextual enforcement resolves far more violations than any baseline. On the decision-compliance benchmark (190 traces, 38 rules drawn from the empirical study), ActPlane achieves a 75.8% Decision Compliance Rate under the primary Qwen3.6-27B setting. In this setting, the tested agent, prompt-filter classifier, and trajectory judge all use Qwen3.6-27B; the benchmark traces were generated earlier by Codex CLI under GPT-5.5, and policy artifacts for the non-prompt systems were translated separately with Claude Opus 4.6:

| System | DCR |
|---|---|
| ActPlane | 75.8% |
| ActPlane-opaque (no feedback) | 53.7% |
| FIDES (tool-level IFC) | 48.9% |
| Prompt-filter | 48.4% |
| Tool-regex | 45.3% |

The benchmark generates five traces per rule: two compliant traces that stress false-positive detection (one exercising permitted actions near the boundary, one performing technically compliant operations that resemble violations) and three violation traces reflecting execution-path categories observed in deployed coding agents. Direct violations trigger the prohibited operation via a tool call, script violations split authoring and execution into a subprocess, and hidden violations embed the side effect in an auxiliary artifact behind a neutral entrypoint.

The gap concentrates on violation traces, where ActPlane correctly resolves 86 of 114, compared to 27 to 44 for baselines, a 2.0 to 3.2 times improvement. All systems perform comparably on compliant traces and direct violations, so ActPlane's advantage opens on script and hidden trace families, where the prohibited effect occurs inside a subprocess or behind a neutral entrypoint and tool-call interception cannot observe it.

Semantic feedback is the dividing line between compliance and retry loops. Full ActPlane produces three times more correct violation-trace outcomes than the same engine without feedback, 86 versus 27. Of the violations ActPlane detects, recovery rate, the fraction where the agent then complies, is 97.7% with semantic feedback versus 31.4% without. When the agent learns why an action was blocked and what state it needs to repair, it reroutes and completes the task. When it receives only a bare denial, it retries the same forbidden action through alternative paths.

That recovery loop only helps if enforcement stays cheap enough for everyday coding. End-to-end overhead was measured on two workloads under no-hit configurations where policies are loaded but no rule fires. The first workload is an agent trace suite that replays 68 tool actions with 20 Bash subprocesses. The second is a Linux kernel build (`defconfig` + `vmlinux`, `make -j24`). At 32 active rules, ActPlane adds 1.9% on the agent trace and 6.5% on the kernel build. Even at 100 rules, overhead stays below 8.4%.

Microbenchmarks isolate where per-syscall cost concentrates. Across the one- through 100-rule configurations, the absolute additions on `fork` and `exec` range from 3.12 to 68.73 microseconds. At 100 rules specifically, `fork` adds 20.39 microseconds and `exec` adds 68.73 microseconds over native latencies of 48.94 and 248.30 microseconds.

Under the same 100-rule load, absolute latencies reach 13.4 microseconds for `open`, 0.84 microseconds for `write`, and 3.17 microseconds for `connect`, so path lookups and rule scans dominate these otherwise sub-microsecond file and network calls. The cumulative ActPlane overhead of an entire tool-call's syscall sequence is five to six orders of magnitude smaller than a single LLM inference turn of 2 to 10 seconds. Policy updates propagate quickly: a one-rule hot reload submitted through the userspace ring buffer reaches the kernel drain path in 26.3 microseconds on average, and an immediate exec violation is detected at p50 176.4 microseconds including process launch and event delivery.

ActPlane's advantage replicates under a second model. A DeepSeek-Pro V4 end-to-end replication preserves the system ranking with ActPlane highest at 77.4% DCR, and per-cell agreement between the two model settings yields a Cohen's kappa of 0.822.

Translation quality drives both detection and recovery rates, because rules that are too narrow miss violations while rules that are too broad match compliant actions. To measure improvability, the paper feeds each false-negative trace's evidence and corrective feedback to the translation agent and lets it revise the rule once. Rerunning the 28 false-negative traces with revised rules recovers 26 (93%), showing that the DSL supports iterative refinement.

Results on a selected set of real-world coding tasks suggest the pattern may extend beyond synthetic traces. On a 21-task subset of OctoBench with 61 OS-enforceable rules spanning seven repositories, ActPlane improves user-query reward by 9.9 points and implementation/test reward by 9.7 points over the no-enforcement baseline. The gains extend beyond compliance-typed checks, suggesting that OS-level enforcement with semantic feedback can help agents follow rules and complete tasks more effectively on this subset.

A separate safety benchmark extends the evidence beyond the paper's own dataset. On 361 OpenAgentSafety personal-assistant tasks, ActPlane loads agent-generated safety policies as higher-authority rules before the agent begins, preventing 74% of baseline-unsafe behaviors (78 of 106 unsafe outcomes blocked). Those policies were generated only from task descriptions, without human tuning. That deployment-like constraint also reveals a cost: ActPlane activated on 16% of tasks whose baseline was already safe when a description-only policy matched a benign operation near the prohibited boundary.

The 28 unblocked cases fall into three categories: chat or semantic harm where the unsafe behavior is a message with no OS-observable artifact, unsafe file content that falls outside ActPlane's primary scope, and service-side artifacts where the effect is a WebDAV upload or database mutation inside a service container that the current hook set does not observe.

The [ActPlane source code](https://github.com/eunomia-bpf/ActPlane) is available on GitHub. The `policies/` directory contains the full set of 607 translated rules across all 64 repositories, ready to serve as starting points for your own instruction files.

## Where Layered Enforcement Stops

### Is eBPF enough for AI agent safety?

eBPF provides deterministic enforcement over OS events such as file writes, process launches, and network connections. The per-event and cross-event classes form the directly OS-enforceable 45% of policies. The broader 83% system-observable set also includes 38% content policies, which need linters or static analyzers, while the remaining 17% cover reasoning, communication style, or output quality and require harness-level controls. Task intent, policy authority, content semantics, and isolation remain responsibilities of the layers around kernel enforcement.

### Can behavioral baselines replace policy?

Behavioral baselines answer "is this unusual?" by flagging deviations from historical patterns. Policy answers "is this permitted under the current task?" A routine `git commit` can be perfectly normal by baseline standards and still violate a project rule. An unfamiliar deployment endpoint can trigger an anomaly alert and still be legitimate because the user explicitly requested it. Detection and authorization are two different problems, and a mature system uses both, letting anomaly signals propose candidate rules that go through authority-aware review before becoming enforced policy.

### Does AgentSight enforce ActPlane policies?

No. [AgentSight](https://eunomia.dev/blog/2025/08/26/agentsight-keeping-your-ai-agents-under-control-with-ebpf-powered-system-observability/) is a system-level profiler and monitor that captures what agents do at runtime. ActPlane is the enforcement engine that blocks or redirects actions violating loaded policies. The two serve complementary roles: runtime evidence from AgentSight can feed into policy review and refinement, but observation and enforcement remain separate responsibilities. An organization might use AgentSight to discover that agents frequently write to production config files, then encode a corresponding ActPlane rule to block that pattern with an explanation of why it is forbidden.

### What remains outside ActPlane's coverage?

The paper's threat model is explicit about boundaries. Semantically equivalent operations can bypass string-based matching: an agent could use a custom Git client instead of calling `exec git`, though the underlying `connect` and `write` system calls remain visible. Service-side effects behind protocol boundaries, such as WebDAV uploads or database mutations inside service containers, also escape the current hook set. File-content semantics, kernel compromise, CAP_BPF compromise, and side channels are all out of scope. The 17% of policies that are semantic-only, covering reasoning quality, communication tone, or output formatting, require harness-layer handling rather than kernel enforcement.

For a team governing coding agents, the practical decision is narrower than "add more eBPF." Start from the statements already sitting in CLAUDE.md and AGENTS.md, resolve the project and task context they omit, and only then compile the OS-enforceable subset into kernel checks with feedback the agent can act on. Prior instruction-file studies worked at file or section granularity, while the ActPlane dataset measures individual policy statements and maps them to enforcement and context requirements. The [ActPlane repository](https://github.com/eunomia-bpf/ActPlane) contains the implementation, and a broader three-layer security model placing kernel enforcement alongside isolation, identity, and content controls appears in [Runtime Observability and Enforcement for Opaque AI Agents with eBPF](https://eunomia.dev/blog/2026/05/25/runtime-security-for-ai-agents/).

## References

- [ActPlane: Programmable OS-Level Policy Enforcement for Agent Harnesses](https://arxiv.org/abs/2606.25189)
- [ActPlane source code and policy artifacts](https://github.com/eunomia-bpf/ActPlane)
- [Agent READMEs: An Empirical Study of Context Files for Agentic Coding](https://arxiv.org/abs/2511.12884)
- [Securing AI Agents with Information-Flow Control](https://arxiv.org/abs/2505.23643)
- [Landlock: Unprivileged Access Control](https://www.kernel.org/doc/html/latest/userspace-api/landlock.html)
- [Tetragon: eBPF-based Security Observability and Runtime Enforcement](https://tetragon.io/)
- [OctoBench: Benchmarking Scaffold-Aware Instruction Following in Repository-Grounded Agentic Coding](https://arxiv.org/abs/2601.10343)
- [OpenAgentSafety: A Comprehensive Framework for Evaluating Real-World AI Agent Safety](https://openreview.net/forum?id=xggSxCFQbA)
- [AgentSight: System-Level Observability for AI Agents Using eBPF](https://doi.org/10.1145/3766882.3767169)
