---
date: 2026-05-31
description: ActPlane is an eBPF-based policy engine that observes and enforces AI agent behavior at the OS kernel level. This post analyzes the systemic blind spots of prompt constraints, tool-layer guards, and sandboxes, and explains how ActPlane uses label propagation and temporal predicates to implement a deterministic agent harness.
---

# Introducing ActPlane: AI Agent Harnesses Need a System-Level Policy Engine Based on eBPF Information Flow Control

You wrote a rule in CLAUDE.md: "do not run `git push`." The agent obeyed. Then it wrote a Python script that called `subprocess.run(["git", "push"])`. The prompt constraint was never violated, but the code was already pushed to the remote.

This is not a hypothetical scenario. Claude Code, Codex, OpenHands — these agents are already running in real projects. Dozens to hundreds of tool calls per task, generating and executing code, modifying files, making network connections — all routine. We try to tell them "don't do that" via CLAUDE.md, intercept dangerous tool calls with MCP gateways, and lock them inside container sandboxes. Each layer helps, but each layer has holes. And agents don't need malicious intent to find paths through those holes — they're just trying to finish the task you gave them.

We built a tool to close these holes.

[ActPlane](https://github.com/eunomia-bpf/ActPlane) is an eBPF-based policy engine that sits below all tool layers and sandboxes, operating directly at the OS kernel level. Prompt constraints are probabilistic — the agent might follow them, or it might not. ActPlane is deterministic: when a rule matches, it always fires. But it's not another sandbox. A sandbox gives you a wall, and all you get when you hit it is `Permission denied`. ActPlane is a harness: when an agent hits a constraint, it receives a human-readable reason explaining why it was stopped and what to do instead. The agent reads the reason, takes a different path, and continues the task.

Each rule chooses its own enforcement level: **notify** only reminds the agent without intervening; **block** stops the operation before it commits; **kill** terminates the process immediately. All three can coexist in the same policy file, and in every mode the reason for the match is fed back to the agent so it can self-correct.

<!-- more -->

## What It Can Express

Before diving into the problem analysis, here are a few constraints we actually want to express. They share one thing in common: you can't say them clearly enough in a prompt, you can't enforce them at the tool layer, and you can't express them in a container sandbox.

First scenario: you want to prevent codex from running `git push` and from writing files outside `/src`. Sounds like an ordinary sandbox rule. But the question is how far "prevent from running" actually reaches. The agent might call git directly, or write a shell script that calls it, or generate a Python program with nested subprocess calls. ActPlane tracks along process lineage: no matter how many layers of indirection the agent goes through before reaching git, as long as its ancestor process is codex, the rule fires. This isn't checking "did the agent call the git tool" at the tool API layer — it's checking "did any process in the codex process tree execute the git binary" at the kernel layer. The two questions look similar but have completely different coverage.

Second scenario: after modifying `specs/*`, the server, SDK, and docs must be updated. ActPlane won't block the agent from editing spec files — that's part of its normal work. But when the agent tries to `git commit` after changing a spec file, ActPlane uses notify mode to remind it: "protocol schema changed, generated code may be stale, run `make proto` to regenerate." Note the choice of notify rather than block. The intent is to remind, not to prevent. The agent decides whether to run code generation.

Third scenario: tests must pass before committing. This sounds simple, but the key is that it's not a one-time check. Every time the agent modifies a file under `src/`, previous test results are automatically invalidated — it must re-run tests before committing. This "automatic invalidation" works through the `since` clause in the rule: it doesn't track "whether tests have ever been run," but "whether tests have been run since the last source file modification." Run tests then change one line of code? Test status resets, run them again.

Fourth scenario: don't delete the build cache unless the user explicitly asks or you're debugging. This constraint is bypassable — it can be lifted through specific arguments or context. It's not a rigid prohibition, but a conditional default. Sandboxes can't express this kind of "usually forbidden but allowed in specific situations."

Fifth scenario: the production database `prod.db` can only be accessed through the migration tool — the agent can't open it directly. This is a mandatory mediation rule: no matter how the agent reaches the file-open call, if its ancestor process chain hasn't executed the `migrate` tool, the operation is blocked. ActPlane uses the `lineage-includes` condition to express this, checking not "does the agent have a certain label" but "has the agent's process lineage passed through a specified gate program."

Looking back at these five scenarios, they share an obvious characteristic: none of them are static allow/deny lists. They all involve process lineage, temporal ordering, mandatory mediation, or conditional bypass. To understand why a kernel-level approach is needed, let's look at where the existing three layers of constraints each fail.

## Blind Spots in Three Layers of Constraints

| Approach | What it does | What it can't cover |
|----------|-------------|---------------------|
| **Prompt constraints** (CLAUDE.md, AGENTS.md) | Tell the agent what to do and not do | Probabilistic: agents forget or non-maliciously route around them in long contexts |
| **Tool-layer guards** (MCP gateway, AgentSpec) | Intercept and authorize at the tool API | Completely bypassed when agents shell out, link SDKs, or spawn subprocesses |
| **Sandboxes** (containers, VMs, E2B, Daytona) | Isolate the entire execution environment | All-or-nothing: can't express "file A may only be accessed via script A" or "run tests before committing" |

### Prompt Constraints

CLAUDE.md, AGENTS.md, system prompts are the most common way to constrain agents today. They're essentially natural-language instructions that depend on the model "remembering and obeying" during inference.

The problem is in "remembering." When conversations exceed tens of thousands of tokens, early constraints get diluted in attention allocation by subsequent user instructions and tool outputs. The model hasn't forgotten the rule, but the rule's weight in decision-making has dropped. A constraint strictly followed at turn 3 might be creatively reinterpreted by turn 30. In our [AgentCgroup characterization study](agentcgroup-characterization.md), we observed that agents average dozens of tool calls per task, and the conversation context can grow very large. The earlier a constraint is set and the longer the conversation, the lower the probability of effective compliance.

Harder to defend against than forgetting is indirect violation. The agent is told "don't delete files," so it writes a Makefile target containing `rm -rf` and calls `make clean`. From the agent's perspective, it executed a build command, not a delete operation. This isn't malicious circumvention. The agent is genuinely trying to complete the task — it just found a path that doesn't cross the constraint surface. In multi-step reasoning, each decision point is locally reasonable, but the global behavior chain may violate the originally stated policy.

The fundamental problem: prompt constraints constrain the agent's intent expression, not its actual behavior. An agent can produce policy-violating side effects without "violating" any prompt rule.

### Tool-Layer Guards

MCP gateways, AgentSpec, tool-level permission controls operate closer to the actual operations. If the agent calls `file_write("/etc/passwd", ...)`, the tool layer can reject it at the API entry point. This is much more reliable than prompt constraints because it checks the operation itself rather than intent.

But agents don't only operate the system through registered tools. The problem is the shell.

Nearly every major agent framework provides some form of shell or terminal tool. This is necessary: agents need to run build commands, execute tests, call third-party CLIs. But once the agent has bash, tool-layer guards are undermined. The agent writes a Python script and executes it; inside the script, `import subprocess; subprocess.run(["curl", "-X", "POST", ...])` sends data to an external network. The tool layer sees the agent called the `Bash` tool to execute `python script.py` — it has no idea what the script does internally, what subprocesses it spawns, or what those subprocesses' subprocesses do. The call chain can be three or four layers deep; the tool layer only sees the first.

There's an even more hidden bypass: code generated by the agent directly links database drivers, HTTP client libraries, or filesystem APIs. These operations happen at runtime of agent-generated code and never pass through any tool call path. Tool-layer guards are completely transparent to them.

The fundamental problem is coverage. The tool layer can only see operations the agent initiates through registered tools. Side effects produced through code execution are entirely outside the tool layer's view.

### Sandboxes

Containers, VMs, E2B, Daytona wrap isolation boundaries around the entire execution environment. This is currently the most reliable security boundary: processes inside the sandbox cannot access resources outside. For "preventing agent escape to the host," sandboxes are the right answer.

But the constraints agents need in real work go far beyond "can or can't access a resource."

"After modifying `.proto` files, the agent must run `protoc` before committing" is a temporal constraint: operation B must happen after operation A and before operation C. Sandboxes have no concept of time — they only know which resources are accessible at the current instant. "Sensitive data read from the database must not be written to log files" requires tracking where data came from and where it flows. Sandboxes operate at process or container granularity — they don't know what a process read or wrote internally. The same `git commit` command should have different policies depending on whether tests just passed or haven't been run yet. Sandboxes can't distinguish based on historical operation context.

There's also an underestimated problem: feedback. When a sandbox rejects an operation, the agent typically receives `Permission denied` or `EPERM`. It doesn't know why it was rejected or what to do to satisfy the constraint. The typical result we observe is the agent retrying the same operation three to five times, then giving up on the entire task. Or worse: the agent tries to bypass via another path, which may introduce new problems. A sandbox is an opaque wall — after hitting it, the agent has no information beyond backing off.

The fundamental problem is expressiveness. Sandboxes answer "can this process access this resource." Agents need to answer far richer questions: under what conditions, in what order, based on what data-flow history, with what context, is this operation permitted?

## Enforcing Policy at the Kernel Boundary

The blind spots of all three layers point in the same direction: the enforcement mechanism needs to sit on the path that every operation must travel. No matter what tool the agent uses, what script it writes, or how many layers of subprocesses it spawns, all side effects eventually go through the OS kernel. Every exec, every file open, every network connect, every fork — the kernel is always there. ActPlane works at this layer: it installs lightweight eBPF programs in the kernel that hook every path an agent might produce side effects through, then makes decisions based on labeled information-flow policies.

Specifically, ActPlane hooks the entire process lifecycle (`sched_process_fork`, `sched_process_exec`, `sched_process_exit`), file operations (`sys_enter_openat`, `sys_enter_unlinkat`, `sys_enter_renameat2`), and network connections (`sys_enter_connect`). These hooks cover four categories of side effects: process creation, binary execution, file access/deletion/rename, and network connections. No matter what path an agent takes to produce side effects, it will pass through these syscalls.

The eBPF programs are precompiled into CO-RE (Compile Once, Run Everywhere) format and embedded in ActPlane's Rust binary. No clang, llvm, libbpf, or any compilation toolchain is needed at install time. The deployment path is `cargo install actplane`, then `actplane init` to generate a starter config, `actplane check` to validate rules, and `sudo actplane run <command>` to execute the agent under the harness. eBPF programs are checked by the kernel verifier, guaranteeing they won't crash the kernel or loop infinitely — the most important safety guarantee eBPF has over traditional kernel modules.

When the kernel has BPF-LSM (Linux Security Module) enabled, ActPlane can stop an operation before it executes — this is block mode. In block mode, `git commit` hasn't committed yet when ActPlane intercepts it and returns an error to the agent. Without BPF-LSM, ActPlane uses kill mode to terminate the process after the operation completes, or notify mode to record and notify the agent without intervening. Three modes correspond to three enforcement strengths, and different rules in the same policy file can choose different modes. "Must run tests before committing" uses block to intercept before commit; "please update docs after modifying spec" uses notify to remind without blocking; "absolutely no git branches" uses kill to terminate immediately.

How does this differ fundamentally from the three layers above? Go back to the opening scenario. The agent wrote a Python script that spawned a subprocess calling git push. The tool layer only saw `python script.py`; ActPlane sees all syscalls across the entire process tree, including that git three layers deep. This is the coverage difference. And constraints aren't attached to "the git tool" but to the process tree. "Codex's entire subprocess tree cannot touch git" is one rule — no need to guard every possible tool entry point.

But ActPlane doesn't just push tool-layer guards down to the kernel. It introduces two capabilities that neither tool layers nor sandboxes have: data-flow tracking and temporal reasoning. Labels propagate across fork/exec and file read/write boundaries, making "data read from A must not flow to B" an expressible policy. The `since` clause lets rules dynamically update on the event timeline — "have tests been run since the last source modification" isn't a static query but a predicate that invalidates and rebuilds as new events occur. And in every mode, when a rule fires the agent receives a human-readable reason and suggestion, not an error code. This is the fundamental difference between a harness and a sandbox.

## Harness, Not Just a Sandbox

A sandbox draws an isolation boundary. Everything inside is allowed, everything outside is forbidden. For untrusted code this is the right model: you don't trust the code, so you lock it in a cage. But an agent isn't untrusted code. An agent is your collaborator — you want it to complete the task, you just want it to follow certain constraints along the way.

Sandboxes answer "can this process access this resource." A harness answers a broader set of questions. Not just security ("sensitive data must not reach the network"), but software engineering discipline. "Run tests before committing" isn't a security constraint, it's an engineering workflow. "Don't mix data from independent tasks in one commit" isn't a permissions problem, it's a work habit. "Access prod.db through the migration tool, not directly" isn't an isolation issue, it's an operational standard. These workflow constraints are exactly the kind of rules agents need when operating autonomously in real codebases. Sandboxes can't express them because they're not allow/deny resource access problems.

At the same time, a harness subsumes sandbox capabilities. When an agent spawns a sub-agent or runs an untrusted command, you can write a rule confining the entire subtree to read-only, no-network, or a specific directory. This is exactly what traditional sandboxes do, but in ActPlane it's just a subset of the rules. You can have sandbox-style rules ("this subprocess tree can't access the network") and workflow rules ("must run tests before committing") in the same policy file.

Cross-vendor scenarios make this distinction sharper. When Claude Code calls Codex, or Codex calls a custom tool chain, each vendor's framework-level guards only know about their own registered tools. Claude Code's hooks don't know Codex's permission config, and vice versa. Framework-level guards assume "I know what paths the agent will use to operate the system" — cross-vendor calls break that assumption immediately. But OS-level rules propagate along process lineage regardless of which vendor's runtime is underneath. One rule constrains the entire cross-vendor execution tree from top-level agent to deepest subprocess.

The feedback loop is the most critical aspect of harness design. When a rule fires, ActPlane delivers the reason to the agent through its framework's hook system:

```
🚫 KILLED: process 'git' (pid 4213, ppid 4210) — /usr/bin/git
   effect: kill
   reason: no git under the agent; use the review workflow
```

The agent receives the reason, understands the constraint, and takes a different path to complete the task. It doesn't need to "remember" that it can't use git — it can try, then be told why it can't and what to do instead. This "deterministic constraints + probabilistic decisions" combination forms an interesting architectural pattern: the agent's reasoning is still probabilistic (that's what makes LLMs useful), but critical constraints are enforced deterministically by the kernel, and the feedback on violations lets the agent self-correct rather than hitting a wall.

## Core Mechanism: Label Propagation

ActPlane's policy is not a static allow/deny list. It uses labeled information-flow policies: processes and files get labels, labels propagate automatically along fork/exec edges and file read/write edges, and rules make decisions based on labels. This sounds abstract, but walking through one example makes it concrete.

Say we want to constrain Claude Code's entire process tree. First, declare a label source: `source AGENT = exec "claude"`. This means whenever any process in the system executes a binary named `claude`, that process gets the `AGENT` label. Labels aren't generated by default — only behaviors you explicitly declare as sources produce labels. The system doesn't track all data flows, only the ones you care about. Similarly, `source SCHEMA_CHANGED = file "src/protocol/**/*.proto"` declares that when these files are written, the writer gets the `SCHEMA_CHANGED` label.

Once a label is created, it propagates automatically. Along process lineage: a parent process with the `AGENT` label forks a child — the child inherits it. The child execs a new program — it inherits it too. The agent starts bash, bash starts Python, Python calls subprocess to start git — every layer carries the `AGENT` label. Labels don't get lost at process boundaries. Along file I/O: a process with the `AGENT` label writes to file `f` — `f` gets the `AGENT` label too. Later, a process that didn't originally have that label reads `f` — it also gets marked `AGENT`. When data crosses process boundaries through files, labels follow the data.

With labels in place, rules are straightforward. `kill exec "git" "push" if AGENT`: if a process with the `AGENT` label tries to execute `git push`, terminate it. Back to that headache-inducing opening scenario:

```
Claude Code (AGENT) 
  → bash (inherits AGENT)
    → python script.py (inherits AGENT) 
      → subprocess: git push (inherits AGENT) ← rule fires here
```

No matter how deep the nesting or how many process boundaries are crossed, labels propagate along process lineage all the way, and the rule fires at the final exec call point. This is how ActPlane solves the shell escape problem: not checking at the tool entry, but checking at the kernel's exec call point, regardless of what path the agent took to get there.

File-direction label propagation solves a different class of problems. Suppose an agent exports sensitive data from a production database to `data.csv` (the process gets a `DB_SENSITIVE` label, and the file gets it too), then another agent or a later step of the same agent reads `data.csv` (the reader gets `DB_SENSITIVE`), then writes the content to a log or sends it over the network. At that point the rule `block connect if DB_SENSITIVE` blocks the process from making a network connection. The entire data flow crossed process boundaries and file boundaries, but labels stayed with the data throughout.

It's worth noting that kernel-level tracking granularity is at the syscall level. ActPlane knows process P wrote to file F, but not what content was written. This means over-tainting can occur: a process reads the first line (the header) of a sensitive file, then writes completely unrelated data to another file — the second file still gets marked sensitive because the writer process once read the sensitive file. In academic literature this is called label creep. In practice three factors mitigate this: different agent tasks typically run in different process trees so labels are naturally isolated; rule matching checks operation-type-plus-label combinations (`block connect if DB_SENSITIVE` only checks on network connections, not on every file read/write just because the label exists); and only explicitly declared sources produce labels. This is a clear design tradeoff: better to over-label than to miss real data flows.

## Temporal Constraints: The `since` Clause

Label propagation solves "who did it" and "where did the data come from." But many of the constraints mentioned earlier also involve temporal ordering: "run tests before committing," "run protoc after changing specs." These constraints need to know "has event X happened after event Y." ActPlane uses the `since` clause for this: rules can reason over the event timeline, expressing "after X happens and before Y, Z is not allowed."

Combining labels and temporal logic, you can write highly expressive policy files. Here's a complete example with four rules showing four different constraint patterns:

```yaml
# actplane.yaml
version: 1
policy: |
  source AGENT = exec "claude"

  # Track when protocol schema files are modified
  source SCHEMA_CHANGED = file "src/protocol/**/*.proto"

  rule no-git-branch:
    kill exec "git" "branch"   if AGENT
    kill exec "git" "worktree" if AGENT
    because "This workspace forbids creating git branches or worktrees.
             Use other git commands, or ask the user to manage branches."

  rule regenerate-after-schema:
    notify exec "git" "commit"
      if SCHEMA_CHANGED unless after exec "protoc" since write "src/protocol/**"
    because "Protocol schema changed — generated code may be stale.
             Run `make proto` to regenerate, then commit."

  rule test-before-commit:
    block exec "git" "commit"
      if AGENT unless after exec "pnpm" "test" since write "src/**"
    because "Source files changed since last test run.
             Run `pnpm test:changed`, then commit."

  rule mediate-proddb:
    block open file "**/prod.db"
      unless lineage-includes exec "**/migrate"
    because "prod.db is reachable only through the migration tool.
             Run `./migrate` to access it."
```

Four rules, four constraint patterns.

`no-git-branch` is the simplest per-event rule, using kill. Any process in the agent's tree that tries `git branch` or `git worktree` is terminated immediately. No conditions, no temporal logic. After termination the agent receives the reason: "This workspace forbids creating git branches or worktrees. Use other git commands, or ask the user to manage branches." The agent reads this and knows to use other git commands or ask the user to manage branches.

`regenerate-after-schema` is a cross-event conditional rule using notify. If a process modifies `src/protocol/**/*.proto` (triggering the `SCHEMA_CHANGED` label) and then tries `git commit`, ActPlane checks `unless after exec "protoc" since write "src/protocol/**"`: has `protoc` been executed since the last write to the protocol directory? If yes, commit proceeds. If no, notify fires and the agent gets a reminder: "generated code may be stale, run `make proto`." The key is the `since` clause: whenever the protocol directory is written to again, the "already ran protoc" state resets. This is a dynamically updating predicate on the event timeline, not a static check.

`test-before-commit` is a temporal rule with dynamic invalidation, using block. The semantics are similar to above but stricter: have tests been run since the last write to `src/**`? If not, block the commit before it executes. Every modification to a file under `src/` resets the test status. The agent runs tests then changes one line of code — it must run tests again before committing.

`mediate-proddb` is a mandatory mediation rule. Unlike the previous three, it doesn't rely on labels but uses `lineage-includes` to check process lineage. Any process wanting to open `prod.db` is checked: has the `migrate` tool been executed somewhere in its ancestor chain? If yes, proceed. If no, block. This rule expresses "the only legitimate access path" — it doesn't check who you are or what labels you carry, it checks whether you came in through the right gate. The agent directly calling `open("prod.db")` gets blocked; the agent calling `./migrate`, which internally opens `prod.db`, succeeds.

Traditional sandboxes have no concept of "time" and no concept of "path." They only know the current instant's state — not what happened before, not what route you took to arrive here. ActPlane maintains both an event timeline for temporal reasoning and process lineage for path checking. This lets it express workflow constraints and mandatory mediation, not just access control.

## Policy Language and Agent Self-Maintenance

Why let agents write their own rules? This isn't a nice-to-have feature — it's demanded by the structure of the problem itself.

Look back at the rules above. Every single one requires information that only the agent (or a developer who knows the project) has.

The "run tests before committing" rule in ActPlane DSL is `unless after exec "pnpm" "test" since write "src/**"`. But why `pnpm test` rather than `pytest` or `make check`? Why monitor `src/**` instead of `lib/**` or `app/**`? Because this particular project happens to use pnpm for package management with source code in src/. For a Python project, the same constraint becomes `unless after exec "pytest" since write "*.py"`. The intent is identical, but every concrete value in the rule comes from project context.

"prod.db can only be accessed through the migration tool" becomes `unless lineage-includes exec "**/migrate"`. The `migrate` here is this project's own database migration tool — it might be called `alembic`, `prisma migrate`, `flyway`, or a custom script `./scripts/db-migrate.sh`. The rule structure is generic (mandatory mediation), but the gate program's name and path are project-specific.

"Run protoc after modifying specs" requires knowing spec files are at `src/protocol/**/*.proto` and the code generation tool is called `protoc`. For an OpenAPI project, the same constraint becomes "run `openapi-generator` after modifying `api/openapi.yaml`." Same rule pattern, completely different paths and tool names.

This information isn't in the kernel, isn't in the eBPF program, and isn't in any generic security policy template. It's in the project's README, in the Makefile, in the `scripts` field of `package.json`, in the code directory structure the agent has already read. A sysadmin can write "deny all non-root processes access to /etc/shadow" — a generic policy that needs no project context — but can't write any of the rules above, because they don't know whether this project uses pnpm or pytest, whether source is in src/ or lib/, or what the database migration tool is called.

Prompt constraints have this context — CLAUDE.md literally says "run tests with pnpm test" and "run make proto after modifying proto files." But prompts are probabilistic: the agent might follow them, or it might not. Traditional OS-level mechanisms (SELinux, AppArmor, seccomp) are deterministic, but they expect pre-defined static policies written by sysadmins who don't know project workflows. Each side has half. ActPlane's design choice is to expose the policy engine as a programmable interface: let the agent — which understands the project context — define system-level policies, then let the kernel execute them deterministically. The agent provides context, the kernel provides enforcement.

You've probably noticed the `because` clause in the YAML above. It's not a comment. It's a first-class member of the rule, and when a rule fires, the `because` content is fed back to the agent verbatim. This design forces rule writers to think from the agent's perspective: what can the agent do after receiving this message? "Security policy violation" is a useless feedback — the agent gets as much from it as from `Permission denied`. "Source files changed since last test run. Run `pnpm test:changed`, then commit." is actionable — the agent reads it and goes straight to running tests. The `because` clause is the most concrete embodiment of the harness vs. sandbox design philosophy: constraints aren't meant to stop the agent, but to guide it toward the right path.

This agent-facing design extends to the entire rule lifecycle. `actplane check` validates rule files for syntactic and semantic correctness without needing root privileges — no eBPF programs are loaded, pure static analysis. This means agents themselves can participate in writing and maintaining rules. Imagine the workflow: the agent analyzes project structure and workflows, determines what constraints the project should have, generates `actplane.yaml`, validates its own rules with `actplane check`, self-corrects if there are errors, and hands the result to a human for review. The rule language is structured clearly enough for agents to reliably generate valid rule files, and the `check` command provides immediate feedback for self-correction. This creates an interesting closed loop: agents participate in defining the constraints that govern themselves.

ActPlane's rule language takes a different path from Rego (OPA), SELinux policy, and iptables. It's declarative — each rule describes "under what conditions is an operation disallowed" rather than imperatively specifying execution steps. Rules have no ordering dependencies and can be composed independently — adding a new rule never requires modifying existing ones. This DSL is purpose-built for AI agent constraint scenarios, optimizing not for generality but for being readable, writable, and verifiable by agents.

## Agent Integration

ActPlane delivers rule match reasons to agents through their framework's hook systems. The kernel remains the sole authority for observation and enforcement. Hooks only relay match events into the agent's decision context.

Claude Code integration via `.claude/settings.local.json`, configuring `PostToolUse` and `PostToolUseFailure` hooks:

```json
{
  "hooks": {
    "PostToolUse": [{ "matcher": "*", "hooks": [{ "type": "command", "command": "actplane feedback-hook" }] }],
    "PostToolUseFailure": [{ "matcher": "*", "hooks": [{ "type": "command", "command": "actplane feedback-hook" }] }]
  }
}
```

Codex integration via `.codex/hooks.json`:

```json
{
  "hooks": {
    "PostToolUse": [{ "matcher": ".*", "hooks": [{ "type": "command", "command": "actplane feedback-hook" }] }]
  }
}
```

The `actplane feedback-hook` adapter does something simple: it checks whether any new rule match events have occurred since its last invocation, and if so, forwards the event's reason (the rule's `because` content) to the hook output. After every tool call, the agent runs this hook — if a rule was triggered, the reason enters the agent's context for its next decision. The entire integration requires no modifications to the agent framework itself, just hook configuration.

## Runtime Architecture

ActPlane's runtime splits into kernel-space and user-space.

```
actplane.yaml ─▶ collector (Rust) ─▶ .rodata config ─▶ eBPF kernel engine
 policy: |        parse + lower DSL    (set_global)      propagate labels,
                                                          match rules,
 matches ◀─────── ring buffer (in-process, via aya) ◀─── emit on match only
```

The kernel part (the `bpf/` directory) hooks fork, exec, exit, open, unlink, rename, and connect syscalls, maintaining a per-node label set (for processes, files, and network endpoints), executing label propagation, evaluating compiled rules, and emitting events to user-space via ring buffer only when rules match. Unmatched operations produce zero user-space overhead. This matters for performance: in our [AgentCgroup characterization](agentcgroup-characterization.md) we saw that an active agent can trigger hundreds of file operations and process creations per second. If every operation had to notify user-space for a decision, latency would become unacceptable. ActPlane completes label propagation and rule matching entirely in kernel space; user-space only participates when a rule fires.

The user-space part is the `actplane` Rust binary. It discovers and parses `actplane.yaml`, compiles the DSL into kernel configuration (written into the eBPF program's `.rodata` section), loads the precompiled eBPF object in-process via [aya](https://github.com/aya-rs/aya) (no libbpf or clang dependency), seeds the target process's lineage (telling the kernel "start tracking from this process"), and listens on the ring buffer for rule matches and their policy reasons.

On permissions: `actplane run` and `actplane watch` need root or `CAP_BPF` + `CAP_SYS_ADMIN` to load the eBPF engine. But once loaded, the target command is dropped back to the current user. The agent itself doesn't run as root. `actplane check` needs no privileges at all — it loads no eBPF programs, only performs static rule validation.

## Use Cases and Limitations

ActPlane's strongest use cases share a common characteristic: single-layer constraints aren't enough.

Cross-vendor multi-agent collaboration is the most typical. When Claude Code calls Codex and Codex calls a custom tool chain, each vendor's framework-level guards only know their own registered tools. Claude Code's hooks don't know Codex's permission config, and vice versa. Framework-level guards assume "I know what paths the agent will use to operate the system" — cross-vendor calls immediately break that assumption. But OS-level rules don't care which vendor's runtime is underneath, propagating along process lineage. One rule governs the entire cross-vendor execution tree.

CI/CD agent governance is another strong scenario. Agents running in CI environments need stricter constraints: can't push code, can't modify CI config, must pass tests before building artifacts. These temporal constraints are exactly what `since` clauses do. Agents deployed in sensitive environments need data-flow-level policies like "data read from prod.db must not flow to the network." Sandboxes can't track at this granularity; label propagation can.

But ActPlane isn't universal. It's built on eBPF, so it only runs on Linux and requires kernel 5.8+ with BTF support (`/sys/kernel/btf/vmlinux`). macOS and Windows agent development scenarios aren't covered, though most production deployments are on Linux. Loading eBPF programs requires root or `CAP_BPF` + `CAP_SYS_ADMIN` — some shared servers and cloud containers won't grant this. Kernel-level tracking only reaches syscall granularity; in-process memory operations and encryption/decryption are out of scope. Block mode requires BPF-LSM, which not all distributions enable by default.

## Related Work

ActPlane builds on a rich lineage of prior work in information-flow control and OS-level provenance. The in-kernel label propagation mechanism is not itself new — [CamQuery](https://dl.acm.org/doi/10.1145/3243734.3243776) (Pasquier et al., CCS 2018) already demonstrated cross-channel taint propagation and enforcement inside the kernel over the [CamFlow](https://dl.acm.org/doi/10.1145/3127479.3129249) provenance graph. What ActPlane contributes is bringing this mechanism to the modern eBPF/BPF-LSM substrate (no kernel module needed), targeting a cooperative-but-forgetful AI agent threat model rather than a remote adversary, and closing the loop with corrective semantic feedback to the agent.

On the eBPF enforcement side, [Cilium Tetragon](https://tetragon.io/) provides `matchBinaries` + `followChildren` which propagates a binary lineage flag to descendants — the closest OSS feature to ActPlane's lineage tracking — but only along fork/exec, not across file/network edges, and without semantic feedback. [OAMAC](https://arxiv.org/abs/2601.14021) (2026) propagates execution-origin labels via BPF-LSM but only across process creation, not file I/O. Neither is agent-aware.

In the agent guardrail space, [AgentSpec](https://arxiv.org/abs/2503.18666) (Wang et al., ICSE 2026) is the closest analog to ActPlane's corrective-feedback idea, with "corrective invocation" and "self-reflection" mechanisms. [Progent](https://arxiv.org/abs/2504.11703) (2026) provides deterministic per-tool-call privilege control via symbolic rules. Both enforce at the tool-call API layer inside the agent framework, making them bypassable by shell-out or direct SDK calls. [SAFEFLOW](https://arxiv.org/abs/2506.07564) (2025) brings information-flow control to the agent protocol layer with provenance tracking, but at the orchestration level rather than the OS kernel. ActPlane complements all of these by enforcing below the tool layer where bypass isn't possible.

The observability foundation comes from [AgentSight](https://arxiv.org/abs/2508.02736) (2025), which uses eBPF to capture both intent-level and action-level agent behavior, framing the "semantic gap" between what agents intend and what they actually do at the system level. ActPlane turns that observability stream into an enforcing policy engine.

## Conclusion

An agent's value lies in flexibility and creativity; deploying agents requires predictability and safety guarantees. There's tension between these two. Prompts are suggestions not rules, tool-layer guards are bypassed by a single shell-out, sandboxes can only do allow/deny resource isolation.

ActPlane adds a layer of deterministic constraints at the kernel. The agent still reasons freely, but critical operations are adjudicated by information-flow rules. When constraints trigger, the agent gets actionable feedback rather than error codes. It doesn't replace the first three layers, but closes each of their blind spots. In complex systems, every single-layer constraint has holes, and agents will naturally find paths through them. Layered constraints may be a necessary architectural component for agents heading toward production deployment.

---

> **GitHub**: [github.com/eunomia-bpf/ActPlane](https://github.com/eunomia-bpf/ActPlane) — MIT License
>
> ActPlane is an open-source project from the [eunomia-bpf](https://github.com/eunomia-bpf) community. Built on [AgentSight](https://github.com/eunomia-bpf/agentsight/)'s eBPF observability foundation.
