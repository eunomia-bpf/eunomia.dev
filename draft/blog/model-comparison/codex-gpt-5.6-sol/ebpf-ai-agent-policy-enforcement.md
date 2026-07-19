---
date: 2026-07-15
slug: ebpf-ai-agent-policy-enforcement
description: An empirical study of 2,116 AI agent instruction file statements finds 64% are policies, 83% are system-observable, and 74% require context to enforce.
---

# 64% of AI Agent Instructions Are Policies

An AI coding agent runs `git commit`, and the kernel sees an ordinary process writing ordinary files. Yet the repository's CLAUDE.md says, "Run the full test suite before committing," and the agent changed source code after its last test run. The instruction came verbatim from a real AI agent instruction file, but nothing in the stack guarantees that the agent will follow it.

The [ActPlane paper](https://arxiv.org/abs/2606.25189) measures how often developers write rules like this in CLAUDE.md and AGENTS.md, then asks what enforcing them would require. Across 2,116 statements, 64% are policies, 83% of those policies involve system-observable actions, and 74% of the system-observable policies depend on context that cannot be defined statically.

<!-- more -->

## What 2,116 Statements Reveal About AI Agent Instructions

AI agent safety discussions often begin with a threat model. This study begins with the instructions developers have already written. What do repository maintainers ask agents to do or avoid, and how much machinery would it take to make those requests enforceable?

The dataset covers 64 popular repositories containing CLAUDE.md or AGENTS.md files, with a median of 20K GitHub stars in a snapshot taken on 2026-05-23. Those repositories contribute 84 instruction files and 2,116 individual statements. Earlier instruction-file research often analyzed whole files or section headings. ActPlane instead classifies each statement independently.

That statement-level view supports three questions. The first asks whether instruction files contain behavioral policies or descriptive context. The second asks which policies require OS-level enforcement and what kind of checks they need. The third asks what context must be supplied before a natural-language policy can become a concrete rule.

The researchers extracted statements with a two-pass LLM agent-assisted pipeline. For every statement, the pipeline recorded its source line range and assigned four labels covering content type, topic, enforcement level, and context requirement. A validation script checked complete source coverage and verbatim span matching. Two independent agents, Claude and Codex, then cross-checked the results, while independent human reviewers examined a stratified sample of 100 statements and confirmed that the labels were correct.

The first result changes how these files should be understood. Of the 2,116 statements, 64% require, forbid, or condition a specific agent action and therefore count as policies. The other 36% provide descriptive context such as architecture notes or project background.

Repositories differ sharply in how they use instruction files. Policy density ranges from 0% to 97%, and 70.1% of repositories contain more policies than descriptive statements. A line-based analysis obscures this result because descriptions average 6.8 lines while policies average 3.6 lines. Counting lines would report only 49% policy content, which is why the statement-level denominator matters.

The study also assigns every statement to one of 12 topic categories adapted from prior instruction-file research. Development Process and Implementation Details contain the highest policy fractions at 87% and 85%. Architecture is mostly descriptive, with only 23% policy content, because those sections tend to document directory layouts and design structure.

![Policy fraction per repository across 64 repos with CLAUDE.md or AGENTS.md](imgs/actplane-empirical_rq1_policy_fraction.png)

Five statements from the dataset show how quickly familiar repository guidance turns into different enforcement problems:

| Statement | Enforcement level | Context |
|---|---|---|
| S4: "Never push to main directly." | per-event | self-contained |
| S5: "Never modify upstream source code." | per-event | project |
| S6: "Run the full test suite before committing." | cross-event | project |
| S7: "Data read from .env must not reach the network." | cross-event | project |
| S8: "Do not update dependencies without approval." | per-event | task |

S4 can be evaluated when one operation occurs. S6 needs a history of edits and test executions. S7 needs data lineage, while S8 changes according to the current user's approval. They may all appear as short imperative sentences in the same file, but they do not describe the same enforcement problem.

## Most Policies Are Observable, but Context Makes Them Hard

The study places each policy into the first matching tier of an enforcement waterfall. Semantic-only policies govern reasoning, communication, or output style. Content policies require predicates over file contents. Per-event policies can be checked against a single command, file access, or network connection. Cross-event policies depend on ordering or data lineage across several operations. The study calls the union of content, per-event, and cross-event policies system-observable.

Only 17% of the 1,361 policies are semantic-only. The remaining 83% are system-observable in principle, including 38% that need content inspection, 29% that match one OS event, and 16% that require cross-event state. Development Process contributes 39.5% of all cross-event policies, which fits the way repositories express rules about testing, generation, review, and release sequences.

![Enforcement waterfall showing semantic-only, content, per-event, and cross-event distribution across 1,361 policies](imgs/actplane-empirical_waterfall_enforcement.png)

Cross-event policies repeatedly take four forms. Temporal ordering requires one action to occur after another, as in running tests before committing. Cross-file consistency connects changes in different artifacts, as when a behavior change also requires a documentation update. Multi-step workflows add verification gates to release checklists. Conditional triggers activate an obligation only after a relevant change, as in updating an SDK after changing its specification.

A single event cannot settle any of these questions. An enforcement layer must remember which commands ran, whether they succeeded, what order they ran in, and what changed afterward. These requirements are common rather than exceptional. At least one cross-event policy appears in 81% of repositories, and 43% of repositories contain policies from all four enforcement tiers.

Even observable actions rarely arrive with all their operands filled in. Among the 1,127 system-observable policies, only 26.4% are self-contained. Project context is required by 64.2%, while another 9.4% depend on task context.

The distinction becomes concrete in S5. "Never modify upstream source code" describes a per-event file-write decision, but an enforcement layer still needs the repository-specific paths that count as upstream source. Task-qualified instructions such as "unless explicitly requested" or "without approval" require information about the current request before the rule can be instantiated.

Cross-event policies combine both difficulties. They are 95% context-dependent, with 77% requiring project context and 19% requiring task context, compared with 58% of content policies. "Run tests before commit" sounds complete to a developer, but a rule engine still needs to identify the test command, decide which directories contain relevant source, and distinguish a successful test from one that merely started.

![Context waterfall showing self-contained, project context, and task context distribution across 1,127 system-observable policies](imgs/actplane-empirical_waterfall_context.png)

Static rules can directly cover only the self-contained fraction. Most repository instructions must first be interpreted against the project and the current task, then lowered into specific commands, paths, events, and conditions.

## What the Instruction Files Imply for Enforcement

Prompt instructions remain useful because they put project knowledge in the model's context. They also rely on the model to comply. Prompt injection can compete with those instructions, and a long task prompt can dilute their influence. Separate agents and LLM guards can inspect prompts, responses, or action trajectories, but their decisions remain probabilistic.

Tool-call guardrails and application-level information-flow control systems make deterministic decisions at the harness boundary. Their view ends at harness-mediated requests. Once a tool launches a subprocess, shells out, or runs a compiled binary, the resulting system effects may no longer resemble the original tool request.

For example, an agent can write a Python program containing `subprocess.run(["git", "push"])` and then execute that program. The tool layer authorizes "run python script.py." The consequential `git push` occurs inside the new process and never appears as a tool call.

OS mechanisms such as seccomp, AppArmor, Landlock, and Tetragon see lower-level resource access. Their policies are normally written statically, however, and resource permissions do not directly represent the actions developers describe in instruction files. When one of these mechanisms rejects an operation, the agent may receive only an opaque `EPERM`, with no indication of which repository rule fired or how to recover.

The measurements point to a split responsibility. Project and task context live near the agent, so the agent must help instantiate natural-language policies. Event ordering and data flow live below the tool boundary, so the resulting rules need deterministic OS-level enforcement.

Two design requirements follow from that split. Policy specifications must be simple enough for an agent to write while remaining concrete enough for the OS to enforce, and violations must return semantic feedback that helps the agent recover. Enforcement must also remain isolated, safe, and efficient. Agent-authored rules cannot weaken higher-authority constraints, interfere with another agent's policy domain, or impose unacceptable overhead on ordinary work.

## How the Empirical Categories Become Concrete Rules

The study does not stop after classifying instruction files. It uses [ActPlane](https://github.com/eunomia-bpf/ActPlane) to test whether the observed categories can be translated into rules and enforced against actual execution paths.

![ActPlane overview: the agent closest to the task writes concrete policy DSL, compiled and enforced inside the OS kernel](imgs/actplane-illustration.png)

An ActPlane rule contains five components. A source identifies what is governed, a target names an operation such as exec, write, or connect, an effect defines the response, an optional temporal gate adds history, and a reason string supplies semantic feedback. The paper uses the following running example:

```
kill exec "git" "commit" unless after exec "go" "test" exits 0
```

The rule kills a `git commit` unless `go test` has exited successfully since the most recent relevant source edit. The omitted reason field gives the agent a structured explanation when enforcement fires.

The available effects form a gradient. Block is a synchronous pre-operation denial with no TOCTOU gap because the kernel intercepts the system call before it executes. The agent can then choose another route. Kill terminates the process after an operation begins, preventing the agent from switching channels after a block. Notify supplies guidance without stopping the operation. Constraints use block or kill, while instructions use notify.

Temporal gates represent ordering rather than a point-in-time predicate. The `after ... since ...` construct requires one event to occur after another, so a test from before the latest edit does not satisfy a test-before-commit rule. The `exits N` qualifier distinguishes successful and failed executions. A lineage gate can inspect process ancestry and limit an operation to a designated process tree.

Information-flow labels cover another class from the empirical taxonomy. Labels propagate across fork, exec, read, write, and connect, and they are monotonic. Once a process reads a labeled object, it cannot remove that label. A process that reads `.env` acquires the file's source label, so a later external connection can match the label and be blocked. This turns S7, "Data read from .env must not reach the network," into a cross-event rule.

Policy authority is protected by a temporal trust boundary. Rules loaded before the agent starts have higher authority and remain immutable to the agent. Agents and sub-agents can add rules or narrow existing ones inside child domains, but they cannot weaken, remove, or disable inherited constraints.

Runtime policy deltas travel through a ring buffer to an in-kernel authority checker. The checker validates every change against the domain hierarchy before activation. The trusted computing base therefore consists of the kernel enforcement engine and the higher-authority policy, while execution below that boundary remains untrusted. A compromised userspace agent cannot change the active rule set beyond the authority granted to its domain because the checker runs entirely in kernel space.

Monotonic labels introduce an over-tainting risk in long sessions. A process that reads dozens of configuration and source files can accumulate enough labels to make many later writes or connections match a rule. ActPlane clears inherited labels when it spawns a fresh subprocess, bounding accumulation to each process lifetime rather than the entire session.

The translated dataset also reveals which language features real instructions exercise. Across 607 policies, 66% of clauses use notify, 29% use block, and 5% use kill, so observation is more common than prevention. Hooks concentrate on code execution and file mutation, with 60% using exec and 37% using write. Network and cleanup operations each account for less than 1%. Cross-event constructs remain substantial, with 28% of policies using an `after/since` temporal gate and 214 using `unless` to encode exceptions.

The implementation consists of roughly 3.2K lines of Rust for the userspace compiler and runner and roughly 1.8K lines of BPF C for the eBPF enforcement engine. BPF-LSM hooks make pre-operation block decisions, while tracepoints support observation and post-operation kill effects. Per-object BPF maps store labels as 64-bit bitmasks, reducing propagation to one bitwise OR. The engine supports up to 128 concurrent rules, while the largest repository in the study contains 66 policies. [ActPlane: Pushing Agent Harness Enforcement Down to Kernel eBPF](https://eunomia.dev/blog/2026/05/31/actplane-pushing-agent-harness-enforcement-down-to-kernel-ebpf/) covers the deployment architecture and mechanisms in greater depth.

## What Happens When the Rules Meet Real Execution Paths

The first practical question is whether an agent can translate repository instructions at the scale observed in the dataset. A Codex agent compiled all 607 OS-enforceable policies into ActPlane rules on its first or second attempt. Only 2 of 607 required a retry after a syntax error.

Translation took 34 minutes using 7 subagents, with 4 running in parallel, and expanded the 607 policies into 1,283 rule lines. The process used 1.7M input tokens and 177K output tokens at a cost of roughly $0.028 per policy. The paper compares that result with approximately $11 per rule at typical software-engineer rates. Most policies remain structurally small. A total of 74% contain at most two enforcement clauses, and the 95th percentile remains below 152 tokens.

The second question is whether context-aware OS enforcement changes agent outcomes. On a decision-compliance benchmark containing 190 traces and 38 rules drawn from the empirical study, ActPlane reaches a 75.8% Decision Compliance Rate:

| System | DCR |
|---|---|
| ActPlane | 75.8% |
| ActPlane-opaque (no feedback) | 53.7% |
| FIDES (tool-level IFC) | 48.9% |
| Prompt-filter | 48.4% |
| Tool-regex | 45.3% |

Each rule contributes five traces. Two compliant traces test false positives, with one exercising permitted actions near the policy boundary and another performing technically compliant operations that resemble violations. Three violation traces reproduce execution paths observed in deployed coding agents.

A direct violation invokes the prohibited operation through a tool call. A script violation separates authoring from execution by placing the action in a subprocess. A hidden violation places the side effect in an auxiliary artifact behind a neutral entrypoint.

The systems perform similarly on compliant traces and direct violations. The difference appears when execution crosses the tool boundary. ActPlane correctly resolves 86 of 114 violation traces, while the baselines resolve between 27 and 44, producing a 2.0 to 3.2 times improvement. Script and hidden traces account for the advantage because their prohibited effects occur inside subprocesses or behind neutral entrypoints that tool-call interception cannot see.

Semantic feedback determines whether enforcement produces compliance or a retry loop. Full ActPlane generates three times as many correct violation-trace outcomes as the same engine without feedback, with 86 compared with 27. Recovery reaches 97.7% with semantic feedback and 31.4% without it. An explanation lets the agent identify the missing state, revise its plan, and finish the task. An opaque denial often leads it to retry the same forbidden operation through another path.

The paper measures overhead under no-hit configurations, where policies are loaded but no rule fires. One workload replays an agent trace suite containing 68 tool actions and 20 Bash subprocesses. The other builds the Linux kernel with `defconfig` plus `vmlinux` using `make -j24`. With 32 active rules, ActPlane adds 1.9% to the agent trace and 6.5% to the kernel build. Overhead remains below 8.4% with 100 rules.

Microbenchmarks show where the per-syscall cost falls. Fork and exec have the largest absolute additions, ranging from 3 to 69 microseconds across rule counts, compared with native latencies of 49 and 248 microseconds. Open reaches 13.4 microseconds at 100 rules, while write and connect remain below one microsecond.

Across an entire tool call, the accumulated syscall overhead is five to six orders of magnitude below one LLM inference turn of 2 to 10 seconds. Policy updates are also fast. A one-rule hot reload submitted through the userspace ring buffer reaches the kernel drain path in 26.3 microseconds on average. An immediate exec violation is detected at p50 176.4 microseconds, including process launch and event delivery.

A second-model replication preserves the result. DeepSeek-Pro V4 reaches the highest DCR at 77.4%, and per-cell agreement between the two model settings produces a Cohen's kappa of 0.822.

Rule quality still matters. A translation that is too narrow misses violations, while one that is too broad catches compliant behavior. To test whether errors can be repaired, the paper gives the translation agent evidence and corrective feedback for each false-negative trace and allows one revision. Revised rules recover 26 of the 28 false negatives, or 93%, showing that the DSL supports iterative refinement.

Real coding tasks follow the same pattern. On a 21-task OctoBench subset containing 61 OS-enforceable rules from seven repositories, ActPlane improves user-query reward by 9.9 points and implementation/test reward by 9.7 points over the no-enforcement baseline. Improvements extend beyond compliance-typed checks, suggesting that OS-level enforcement with semantic feedback can help an agent complete its task as well as follow its rules.

The paper also evaluates 361 OpenAgentSafety personal-assistant tasks. ActPlane loads agent-generated safety policies as higher-authority rules before execution begins and prevents 74% of baseline-unsafe behavior, blocking 78 of 106 unsafe outcomes.

The 28 unblocked cases fall into three categories. Some involve chat or semantic harm with no OS-observable artifact. Others depend on unsafe file content, which sits outside ActPlane's primary scope. The remaining cases involve service-side artifacts such as WebDAV uploads or database mutations inside a service container that the current hooks cannot observe.

The [ActPlane source code](https://github.com/eunomia-bpf/ActPlane) is available on GitHub. Its `policies/` directory contains all 607 translated policies from the 64 repositories and can serve as a starting point for other instruction files.

## Common Questions

### Is eBPF enough for AI agent safety?

eBPF provides deterministic enforcement over OS events such as file writes, process launches, and network connections. That capability covers the 83% of policies that are system-observable. The remaining 17% concern reasoning, communication style, or output quality and require harness controls outside the kernel.

Content policies account for the 38% that require file-content inspection. Linters and static analyzers are better suited to those checks, which fall outside eBPF's per-event and cross-event enforcement scope. Task intent, policy authority, content semantics, and isolation also remain responsibilities of the layers surrounding kernel enforcement.

### Can behavioral baselines replace policy?

Behavioral baselines ask whether an action is unusual by comparing it with historical patterns. Policy asks whether the action is permitted for the current task.

A routine `git commit` can look completely normal and still violate a project rule. An unfamiliar deployment endpoint may trigger an anomaly alert even when the user explicitly requested it. Detection and authorization solve different problems. A mature system can use anomaly signals to propose candidate rules, then require authority-aware review before those rules become enforced policy.

### Does AgentSight enforce ActPlane policies?

No. [AgentSight](https://eunomia.dev/blog/2025/08/26/agentsight-keeping-your-ai-agents-under-control-with-ebpf-powered-system-observability/) is a system-level profiler and monitor that records what agents do at runtime. ActPlane is the enforcement engine that blocks or redirects operations that violate loaded policies.

The two systems have complementary roles. AgentSight evidence can inform policy review and refinement, while observation and enforcement remain separate responsibilities. An organization could use AgentSight to discover frequent writes to production configuration files, then encode an ActPlane rule that blocks those writes and explains why they are forbidden.

### What remains outside ActPlane's coverage?

The paper defines several explicit boundaries. Semantically equivalent operations can evade string-based matching. An agent could use a custom Git client instead of invoking `exec git`, although the underlying `connect` and `write` system calls remain visible. Service-side effects behind protocol boundaries, including WebDAV uploads and database mutations inside service containers, also escape the current hook set.

File-content semantics, kernel compromise, CAP_BPF compromise, and side channels remain out of scope. The 17% of semantic-only policies covering reasoning quality, communication tone, or output formatting require harness-layer handling rather than kernel enforcement.

The dataset remains the paper's most distinctive contribution. Before ActPlane, no study had measured what developers ask agents to obey or mapped those instructions to their enforcement requirements. Most of these rules are already present in CLAUDE.md and AGENTS.md files across thousands of repositories. Enforcing them requires a layer that can combine project and task context with concrete, deterministic checks. The [ActPlane repository](https://github.com/eunomia-bpf/ActPlane) contains the implementation, while [Runtime Observability and Enforcement for Opaque AI Agents with eBPF](https://eunomia.dev/blog/2026/05/25/runtime-security-for-ai-agents/) places kernel enforcement within a broader three-layer model covering isolation, identity, and content controls.
