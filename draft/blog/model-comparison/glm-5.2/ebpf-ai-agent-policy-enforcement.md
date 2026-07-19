---
date: 2026-07-15
slug: ebpf-ai-agent-policy-enforcement
description: A study of 2,116 statements in real CLAUDE.md and AGENTS.md files shows 64% are policies, 83% are OS-observable, and 74% need project or task context to enforce.
---

# 2,116 Real AI Agent Rules Need OS Enforcement With Context

An AI coding agent runs `git commit`. The kernel sees a familiar process writing familiar files, nothing unusual. The repository's CLAUDE.md says "Run the full test suite before committing," and the agent edited source code after its last test run. That sentence, copied verbatim from the [ActPlane paper's](https://arxiv.org/abs/2606.25189) dataset, is enforced by no layer in the stack today.

The paper compresses the landscape into one sentence. 64% of statements are policies, 83% involve system actions, and 74% depend on context that cannot be pre-defined statically. Behind those three numbers is the first large-scale measurement of what developers actually write in CLAUDE.md and AGENTS.md files, what enforcing those sentences requires, and whether a working OS-level engine can close the gap. This post walks the empirical findings, why each existing enforcement layer misses them, and what the ActPlane validation shows.

<!-- more -->

## What 2,116 Statements in Real Instruction Files Look Like

[ActPlane](https://github.com/eunomia-bpf/ActPlane) does not start from a threat model or an attack surface. It starts from a simpler question: what have developers already told their agents to do, and what would it take to make those instructions enforceable?

The corpus spans 64 popular repositories carrying CLAUDE.md or AGENTS.md files (median 20K GitHub stars, snapshot 2026-05-23), covering 84 instruction files and 2,116 individual statements. Earlier work on agent instruction files classified at file or section-heading granularity, which buries behavioral rules inside long architecture descriptions. ActPlane labels every statement on its own, so a six-line directory map and a one-line "do not push to main" stop being the same unit of analysis.

Extracting statements at that resolution calls for a careful pipeline. A two-pass LLM agent-assisted process records the source line range and four labels per statement (content type, topic, enforcement level, context requirement). A validation script checks full source coverage and verbatim span matching against the original files, and two independent agents (Claude and Codex) cross-check the labels. A stratified sample of 100 statements then goes through independent human review and confirms the labels.

Counting by statement instead of by line changes the picture. 64% of the 2,116 statements are policies, meaning they require, forbid, or condition a specific agent action. The remaining 36% are descriptive context such as architecture notes or project background. Policy density varies widely, from 0% in some repositories to 97% in others, and 70.1% of repositories hold more policies than descriptions. Counting by lines instead tells a different story, because descriptions average 6.8 lines while policies average only 3.6, so line-based metrics see just 49% policies. That gap is the reason statement-level analysis is worth the extra work.

Twelve topic categories adapted from prior instruction-file research, applied per statement, show where those policies concentrate. Development Process and Implementation Details dominate at 87% and 85% policy share respectively. Architecture sits at the other end with 23%, because directory layouts and design summaries fill most of those sections with description rather than rules.

![Policy fraction per repository across 64 repos with CLAUDE.md or AGENTS.md](imgs/actplane-empirical_rq1_policy_fraction.png)

Five real statements from the dataset give a feel for the range of what enforcement has to handle:

| Statement | Enforcement level | Context |
|---|---|---|
| S4: "Never push to main directly." | per-event | self-contained |
| S5: "Never modify upstream source code." | per-event | project |
| S6: "Run the full test suite before committing." | cross-event | project |
| S7: "Data read from .env must not reach the network." | cross-event | project |
| S8: "Do not update dependencies without approval." | per-event | task |

## 83% of Policies Are System-Observable, and 16% Are Cross-Event

The paper sorts every policy into the first tier of an enforcement waterfall it can satisfy. Semantic-only covers reasoning, communication, or output style. Content covers predicates over file contents. Per-event covers a single command, file access, or network connection. Cross-event covers policies that depend on temporal ordering or data lineage across operations. The union of content, per-event, and cross-event is called system-observable, because a kernel-level monitor can in principle evaluate it.

Of the 1,361 policies in the dataset, 83% land in the system-observable region, with 38% needing content inspection, 29% matching a single OS event, and 16% requiring cross-event state. The remaining 17% are semantic-only. Cross-event policies cluster inside Development Process topics, which alone account for 39.5% of all cross-event policies.

![Enforcement waterfall showing semantic-only, content, per-event, and cross-event distribution across 1,361 policies](imgs/actplane-empirical_waterfall_enforcement.png)

Cross-event policies fall into four recurring shapes. Temporal ordering constrains sequencing, so "run tests before committing" needs one event to have happened after another, not merely at some earlier point. Cross-file consistency links changes across artifacts, so "update docs when behavior changes" couples a source edit to a documentation update. Multi-step workflows enforce release checklists with verification gates, where each step must complete before the next begins. Conditional triggers couple operations, so "if you change specs, also update the SDK" fires only when a precondition is met. None of those can be decided from a single event. The engine has to record what ran, in what order, and what has changed since. They are also widespread, with 81% of repositories holding at least one cross-event policy and 43% spanning all four enforcement tiers.

Context dependence compounds the difficulty. Of the 1,127 system-observable policies, only 26.4% are self-contained. 64.2% need project context, because phrases like "the test suite" or "upstream source" have to be resolved against a specific repository before the policy becomes a concrete rule. Even a per-event policy such as S5 ("Never modify upstream source code") needs someone to resolve which paths count as "upstream source" before a file-write check can fire. Another 9.4% need task context, such as "unless explicitly requested" or "without approval."

The two difficulties land on the same policies. Cross-event policies are 95% context-dependent (77% project, 19% task), against 58% for content policies. "Run tests before commit" sounds simple until the engine needs to know which test command to watch for, which source directories count as "relevant edits," and whether the test passed or merely ran.

![Context waterfall showing self-contained, project context, and task context distribution across 1,127 system-observable policies](imgs/actplane-empirical_waterfall_context.png)

A fixed set of static rules can cover only the self-contained slice. Everything else needs the engine to read the repository and interpret the current task before any check can run.

## Why Prompt, Tool, and Sandbox Layers Each Miss These Rules

Three existing layers all fail against the rules the corpus contains, and each fails for a different reason.

Prompt instructions depend on the model's own compliance, but they are vulnerable to prompt injection and lose weight as the conversation grows. A rule written in CLAUDE.md competes with the user's task prompt for attention inside a long context window, and indirect violations are easy. An agent told "don't delete files" can write a Makefile target containing `rm -rf` and call `make clean`, never technically breaking the instruction.

Tool-call guardrails and application-level information-flow control systems intercept at the harness boundary deterministically, but they observe only harness-mediated requests. An indirect subprocess, shell-out, or compiled binary bypasses the tool boundary entirely. Imagine an agent that writes a Python script containing `subprocess.run(["git", "push"])` and then executes it. The tool-call layer sees "run python script.py", not the `git push` inside it.

OS-level mechanisms such as seccomp, AppArmor, Landlock, and Tetragon control resource access, not the actions developers actually describe. They expect statically pre-written policies, and when a check fails they return opaque errors. A bare `EPERM` with no explanation of what rule was violated or how to recover leaves the agent retrying the same forbidden action through alternative paths.

These three failures point to the same conclusion. Most rules need project or task context that lives with the agent, so the agent itself must be able to turn policies into concrete rules. Many of those same policies define event ordering or data flow that is invisible to tool-call guardrails, so the rules must be concrete enough for deterministic OS-level enforcement. The ActPlane paper turns that conclusion into two design requirements. The policy specification has to be agent-writable yet OS-enforceable, with semantic feedback so the agent can understand violations and recover. Enforcement has to be safe, isolated, and efficient, so agent-authored policy cannot weaken higher-authority constraints, cannot affect other agents, and cannot slow the workload.

## How ActPlane Validates Those Requirements

ActPlane is the engine the paper builds to test whether the requirements hold up against real rules. The full mechanism design lives in a [sibling post on pushing agent harness enforcement down to kernel eBPF](https://eunomia.dev/blog/2026/05/31/actplane-pushing-agent-harness-enforcement-down-to-kernel-ebpf/); the pieces that matter here are the ones the empirical study exercises.

![ActPlane overview: the agent closest to the task writes concrete policy DSL, compiled and enforced inside the OS kernel](imgs/actplane-illustration.png)

Each rule has five parts. A source identifies what is being governed, a target operation such as exec, write, or connect names the action, an effect sets the consequence, an optional temporal gate encodes ordering, and a reason string carries semantic feedback. The running example from the paper reads:

```
kill exec "git" "commit" unless after exec "go" "test" exits 0
```

That rule kills any `git commit` unless `go test` has exited successfully since the most recent relevant source edit. The reason field, omitted here for brevity, gives the agent a structured explanation when the rule fires.

Effects form a gradient that mirrors the distinction between instructions and constraints. Block is a pre-operation synchronous denial with no TOCTOU gap, because the kernel intercepts the system call before it executes and the agent can reroute. Kill terminates the process after the operation has begun, preventing the agent from switching to an alternate channel after a block. Notify delivers guidance without stopping the action. Constraints use block or kill, and instructions use notify.

Temporal gates let rules express ordering rather than point-in-time predicates. The `after ... since ...` construct encodes that one event must have happened after another, so tests must have run after the most recent edit, not merely at some earlier point. The `exits N` qualifier distinguishes successful from failed exits. A lineage gate checks process ancestry and lets rules restrict operations to specific process trees.

Information-flow labels propagate along fork, exec, read, write, and connect, and the propagation is monotonic. Once a process reads a labeled object, the label cannot be removed. When a process reads `.env`, it acquires that file's source label, and if it later tries to connect to an external endpoint, the rule matching that label fires and blocks the connection. That is how S7 from the dataset, "Data read from .env must not reach the network," becomes an enforceable cross-event rule.

Policy authority rests on a temporal trust boundary. Rules loaded before the agent starts are higher-authority and immutable to the agent. The agent and its sub-agents can add new rules or narrow inherited ones inside child domains, but they cannot weaken, remove, or disable constraints set above them. Runtime deltas arrive through a ring buffer and pass through an in-kernel authority checker that validates each change against the domain hierarchy before activation. Because that checker runs entirely in kernel space, a compromised userspace agent cannot modify the active rule set beyond what its domain hierarchy permits.

Long-running sessions risk over-tainting because labels are monotonic. In a typical coding session a process might read dozens of configuration and source files, and without mitigation each read adds a label until every subsequent write or connect matches some rule. ActPlane clears inherited labels when a fresh subprocess is spawned, bounding taint accumulation to the lifetime of each process instead of the entire session.

The 607-policy dataset exercises most of the DSL and confirms the language covers what real instruction files contain. Effects skew toward observation, with 66% notify, 29% block, and only 5% kill, which matches the intuition that most policies monitor rather than prevent. Hooks concentrate on code execution (60% exec) and file mutation (37% write), while network and cleanup operations sit under 1% each. Cross-event features see real use, with 28% of policies relying on an `after/since` temporal gate and 214 using `unless` to encode exceptions.

Implementation stays compact because the rule language keeps the kernel work simple. The userspace compiler and runner are roughly 3.2K lines of Rust, and the eBPF enforcement engine is roughly 1.8K lines of BPF C. BPF-LSM hooks handle pre-operation decisions (block), while tracepoints handle observation and post-operation termination (kill). Labels are stored as 64-bit bitmasks in per-object BPF maps, so propagation reduces to a single bitwise OR, which is why the engine can support up to 128 concurrent rules against the largest repository in the study (66 policies) without strain.

## Detection, Recovery, and Overhead on Real Policies

Policy translation scales to the full dataset. A Codex agent compiled all 607 OS-enforceable policies into ActPlane rules on the first or second attempt, with only 2 of 607 needing a syntax-error retry. The full run finished in 34 minutes using 7 subagents (4 in parallel), expanding 607 policies into 1,283 rule lines, and cost about $0.028 per policy on 1.7M input tokens and 177K output tokens. Compared with roughly $11 per rule at typical software-engineer rates, automatic translation is cheap enough to run on every repository. Most policies are structurally simple, with 74% having at most two enforcement clauses and the 95th percentile staying under 152 tokens.

On the decision-compliance benchmark (190 traces, 38 rules drawn from the empirical study), contextual enforcement resolves far more violations than any baseline, with ActPlane reaching a 75.8% Decision Compliance Rate (DCR):

| System | DCR |
|---|---|
| ActPlane | 75.8% |
| ActPlane-opaque (no feedback) | 53.7% |
| FIDES (tool-level IFC) | 48.9% |
| Prompt-filter | 48.4% |
| Tool-regex | 45.3% |

The benchmark generates five traces per rule. Two are compliant traces that stress false-positive detection. One exercises permitted actions near the boundary, and the other performs technically compliant operations that resemble violations. Three are violation traces reflecting execution-path categories observed in deployed coding agents. Direct violations trigger the prohibited operation through a tool call. Script violations split authoring and execution into a subprocess. Hidden violations embed the side effect in an auxiliary artifact behind a neutral entrypoint.

ActPlane's advantage opens on the violation traces, where it correctly resolves 86 of 114, against 27 to 44 for the baselines, a 2.0 to 3.2 times improvement. All systems perform comparably on compliant traces and direct violations. The gap appears on script and hidden trace families, where the prohibited effect happens inside a subprocess or behind a neutral entrypoint that tool-call interception cannot observe.

Semantic feedback is the dividing line between compliance and retry loops. Full ActPlane produces three times more correct violation-trace outcomes than the same engine without feedback, 86 versus 27, and recovery rates track the same pattern at 97.7% with semantic feedback versus 31.4% without. When the agent learns why an action was blocked and what state it needs to repair, it reroutes and completes the task. When it receives only a bare denial, it retries the same forbidden action through alternative paths.

End-to-end overhead stays inside a development workflow's budget. The paper measures overhead on two workloads under no-hit configurations, where policies are loaded but no rule fires. The first workload replays 68 tool actions with 20 Bash subprocesses. The second builds a Linux kernel (`defconfig` + `vmlinux`, `make -j24`). At 32 active rules, ActPlane adds 1.9% overhead on the agent trace and 6.5% on the kernel build, and even at 100 rules overhead stays below 8.4%.

Microbenchmarks isolate where per-syscall cost concentrates. Fork and exec carry the highest absolute additions at 3 to 69 microseconds across rule counts, but those stay modest relative to their native latency of 49 and 248 microseconds. File and network operations stay lightweight, with open reaching 13.4 microseconds at 100 rules while write and connect stay sub-microsecond. The cumulative ActPlane overhead of an entire tool-call's syscall sequence is five to six orders of magnitude smaller than a single LLM inference turn of 2 to 10 seconds, which is why harness overhead effectively disappears inside an agent loop. Policy updates propagate quickly: a one-rule hot reload submitted through the userspace ring buffer reaches the kernel drain path in 26.3 microseconds on average, and an immediate exec violation is detected at p50 176.4 microseconds including process launch and event delivery.

The ranking holds under a second model. A DeepSeek-Pro V4 end-to-end replication preserves the system ordering with ActPlane highest at 77.4% DCR, and per-cell agreement between the two model settings yields a Cohen's kappa of 0.822.

Translation quality drives both detection and recovery, because rules that are too narrow miss violations while rules that are too broad match compliant actions. To measure improvability, the paper feeds each false-negative trace's evidence and corrective feedback to the translation agent and lets it revise the rule once. Rerunning the 28 false-negative traces with revised rules recovers 26 (93%), showing that the DSL supports iterative refinement.

Real-world coding tasks confirm the pattern beyond synthetic traces. On a 21-task subset of OctoBench with 61 OS-enforceable rules spanning seven repositories, ActPlane improves user-query reward by 9.9 points and implementation/test reward by 9.7 points over the no-enforcement baseline. The gains reach beyond compliance-typed checks, which suggests that OS-level enforcement with semantic feedback helps agents both follow rules and finish their tasks more effectively.

External safety benchmarks confirm breadth beyond the paper's own dataset. On 361 OpenAgentSafety personal-assistant tasks, ActPlane loads agent-generated safety policies as higher-authority rules before the agent begins, preventing 74% of baseline-unsafe behaviors (78 of 106 unsafe outcomes blocked). The 28 unblocked cases fall into three categories: chat or semantic harm where the unsafe behavior is a message with no OS-observable artifact, unsafe file content that falls outside ActPlane's primary scope, and service-side artifacts where the effect is a WebDAV upload or database mutation inside a service container that the current hook set does not observe.

The [ActPlane source code](https://github.com/eunomia-bpf/ActPlane) is available on GitHub. The `policies/` directory contains the full set of 607 translated rules across all 64 repositories, ready to serve as starting points for your own instruction files.

## Common Questions

### Is eBPF enough for AI agent safety?

eBPF provides deterministic enforcement over OS events: file writes, process launches, network connections, and similar system actions. That covers the 83% of policies that are system-observable, but the remaining 17% cover reasoning, communication style, or output quality and require harness-level controls outside the kernel. Content policies, the 38% that require reading file contents, are better served by linters and static analyzers and fall outside eBPF's per-event and cross-event enforcement scope. Task intent, policy authority, content semantics, and isolation are all concerns that the layers around kernel enforcement must handle.

### Can behavioral baselines replace policy?

Behavioral baselines answer "is this unusual?" by flagging deviations from historical patterns. Policy answers "is this permitted under the current task?" A routine `git commit` can be perfectly normal by baseline standards and still violate a project rule. An unfamiliar deployment endpoint can trigger an anomaly alert and still be legitimate because the user explicitly requested it. Detection and authorization are different problems, and a mature system uses both. Anomaly signals can propose candidate rules, which then go through authority-aware review before becoming enforced policy.

### Does AgentSight enforce ActPlane policies?

No. [AgentSight](https://eunomia.dev/blog/2025/08/26/agentsight-keeping-your-ai-agents-under-control-with-ebpf-powered-system-observability/) is a system-level profiler and monitor that captures what agents do at runtime. ActPlane is the enforcement engine that blocks or redirects actions violating loaded policies. The two serve complementary roles, since runtime evidence from AgentSight can feed into policy review and refinement while observation and enforcement remain separate responsibilities. An organization might use AgentSight to discover that agents frequently write to production config files, then encode a corresponding ActPlane rule to block that pattern with an explanation of why it is forbidden.

### What remains outside ActPlane's coverage?

The paper's threat model is explicit about boundaries. Semantically equivalent operations can bypass string-based matching, so an agent could use a custom Git client instead of calling `exec git`, though the underlying `connect` and `write` system calls remain visible. Service-side effects behind protocol boundaries, such as WebDAV uploads or database mutations inside service containers, also escape the current hook set. File-content semantics, kernel compromise, CAP_BPF compromise, and side channels are all out of scope. The 17% of policies that are semantic-only, covering reasoning quality, communication tone, or output formatting, require harness-layer handling rather than kernel enforcement.

## What Changes Once the Dataset Exists

Before ActPlane, nobody had measured what developers actually ask their agents to obey or how those rules distribute across enforcement requirements. The rules themselves have been sitting in CLAUDE.md and AGENTS.md files across thousands of repositories, waiting for an enforcement layer that can read project context, interpret the current task, and turn those natural-language policies into concrete kernel-level rules. The [ActPlane repository](https://github.com/eunomia-bpf/ActPlane) contains the implementation and the full rule set, and a broader three-layer security model placing kernel enforcement alongside isolation, identity, and content controls appears in [Runtime Observability and Enforcement for Opaque AI Agents with eBPF](https://eunomia.dev/blog/2026/05/25/runtime-security-for-ai-agents/).
