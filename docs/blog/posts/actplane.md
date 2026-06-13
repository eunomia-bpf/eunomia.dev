---
date: 2026-05-31
description: ActPlane is an eBPF-based policy engine that observes and enforces AI agent behavior at the OS kernel level. This post analyzes the systemic blind spots of prompt constraints, tool-layer guards, and sandboxes, and explains how ActPlane uses label propagation and temporal predicates to implement a deterministic agent harness.
---

# ActPlane: an IFC Policy Engine for AI Agent Harnesses in eBPF

You wrote a rule in CLAUDE.md: "do not run `git push`." The agent obeyed. Then it wrote a Python script that called `subprocess.run(["git", "push"])`. The prompt constraint was never violated, but the code was already pushed to the remote.

This is not a hypothetical scenario. Claude Code, Codex, OpenHands — these agents are already running in real projects. Dozens to hundreds of tool calls per task, generating and executing code, modifying files, making network connections — all routine. We try to tell them "don't do that" via CLAUDE.md, intercept dangerous tool calls with MCP gateways, and lock them inside container sandboxes. Each layer helps, but each layer has holes. And agents don't need malicious intent to find paths through those holes — they're just trying to finish the task you gave them.

We built a tool to close these holes.

[ActPlane](https://github.com/eunomia-bpf/ActPlane) is an eBPF-based policy engine that sits below all tool layers and sandboxes, operating directly at the OS kernel level. Prompt constraints are probabilistic — the agent might follow them, or it might not. ActPlane is deterministic: when a rule matches, it always fires. But it's not another sandbox. A sandbox gives you a wall, and all you get when you hit it is `Permission denied`. ActPlane is a harness: when an agent hits a constraint, it receives a human-readable reason explaining why it was stopped and what to do instead. The agent reads the reason, takes a different path, and continues the task.

Each rule chooses its own enforcement level: **notify** only reminds the agent without intervening; **block** stops the operation before it commits; **kill** terminates the process immediately. All three can coexist in the same policy file, and in every mode the reason for the match is fed back to the agent so it can self-correct.

<!-- more -->

## What It Can Express

Before diving into the problem analysis, here are a few constraints we actually want to express. They share one thing in common: you can't say them clearly enough in a prompt, you can't enforce them at the tool layer, and you can't express them in a container sandbox.

First scenario: you want to prevent codex from running `git push` and from writing files outside `/src`. Sounds like an ordinary sandbox rule. But how far does "prevent from running" actually reach? The agent might call git directly, or write a shell script that calls it, or generate a Python program with nested subprocess calls. What we need isn't checking "did the agent call the git tool" at the tool API layer — it's checking "did any process in the codex process tree execute the git binary" across the entire process tree. The two questions look similar but have completely different coverage.

Second scenario: after modifying `specs/*`, the server, SDK, and docs must be updated. We don't want to block the agent from editing spec files — that's part of its normal work. But when the agent tries to `git commit` after changing a spec file, it should receive a reminder: "generated code may be stale, run `make proto` to regenerate." The intent is to remind, not to prevent. The agent decides whether to run code generation.

Third scenario: tests must pass before committing. This sounds simple, but the key is that it's not a one-time check. Every time the agent modifies a file under `src/`, previous test results should be automatically invalidated — it must re-run tests before committing. The check isn't "whether tests have ever been run," but "whether tests have been run since the last source file modification." Run tests then change one line of code? Test status resets, run them again.

Fourth scenario: the production database `prod.db` can only be accessed through the migration tool — the agent can't open it directly. This is a mandatory mediation constraint: no matter how the agent reaches the file-open call, if its process ancestor chain hasn't executed the `migrate` tool, the operation should be blocked. The check isn't "does the agent have a certain permission" but "did the agent come in through the right gate."

Looking back at these four scenarios, they share an obvious characteristic: none of them are static allow/deny lists. They all involve process lineage, temporal ordering, or mandatory mediation. To understand why a kernel-level approach is needed, let's look at where the existing three layers of constraints each fail.

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

MCP gateways, [AgentSpec](https://arxiv.org/abs/2503.18666), [Progent](https://arxiv.org/abs/2504.11703), and other tool-level permission controls operate closer to the actual operations. If the agent calls `file_write("/etc/passwd", ...)`, the tool layer can reject it at the API entry point. This is much more reliable than prompt constraints because it checks the operation itself rather than intent.

But agents don't only operate the system through registered tools. The problem is the shell.

Nearly every major agent framework provides some form of shell or terminal tool. This is necessary: agents need to run build commands, execute tests, call third-party CLIs. But once the agent has bash, tool-layer guards are undermined. The agent writes a Python script and executes it; inside the script, `import subprocess; subprocess.run(["curl", "-X", "POST", ...])` sends data to an external network. The tool layer sees the agent called the `Bash` tool to execute `python script.py` — it has no idea what the script does internally, what subprocesses it spawns, or what those subprocesses' subprocesses do. The call chain can be three or four layers deep; the tool layer only sees the first.

There's an even more hidden bypass: code generated by the agent directly links database drivers, HTTP client libraries, or filesystem APIs. These operations happen at runtime of agent-generated code and never pass through any tool call path. Tool-layer guards are completely transparent to them.

The fundamental problem is coverage. The tool layer can only see operations the agent initiates through registered tools. Side effects produced through code execution are entirely outside the tool layer's view.

### Sandboxes

Containers, VMs, [E2B](https://github.com/e2b-dev/E2B), [Daytona](https://github.com/daytonaio/daytona) wrap isolation boundaries around the entire execution environment. This is currently the most reliable security boundary: processes inside the sandbox cannot access resources outside. For "preventing agent escape to the host," sandboxes are the right answer.

But the constraints agents need in real work go far beyond "can or can't access a resource."

"After modifying `.proto` files, the agent must run `protoc` before committing" is a temporal constraint: operation B must happen after operation A and before operation C. Sandboxes have no concept of time — they only know which resources are accessible at the current instant. "Sensitive data read from the database must not be written to log files" requires tracking where data came from and where it flows. Sandboxes operate at process or container granularity — they don't know what a process read or wrote internally. The same `git commit` command should have different policies depending on whether tests just passed or haven't been run yet. Sandboxes can't distinguish based on historical operation context.

There's also an underestimated problem: feedback. When a sandbox rejects an operation, the agent typically receives `Permission denied` or `EPERM`. It doesn't know why it was rejected or what to do to satisfy the constraint. The typical result we observe is the agent retrying the same operation three to five times, then giving up on the entire task. Or worse: the agent tries to bypass via another path, which may introduce new problems. A sandbox is an opaque wall — after hitting it, the agent has no information beyond backing off.

The fundamental problem is expressiveness. Sandboxes answer "can this process access this resource." Agents need to answer far richer questions: under what conditions, in what order, based on what data-flow history, with what context, is this operation permitted?

## Enforcing Policy at the Kernel Boundary

The blind spots of all three layers point in the same direction: the enforcement mechanism needs to sit on the path that every operation must travel. [AgentSight](https://arxiv.org/abs/2508.02736) (2025) used eBPF to capture both intent-level and action-level agent behavior, framing the "semantic gap" between what agents intend and what they actually do at the system level. ActPlane builds on that observability foundation and adds enforcement. No matter what tool the agent uses, what script it writes, or how many layers of subprocesses it spawns, all side effects eventually go through the OS kernel. Every exec, every file open, every network connect, every fork — the kernel is always there. ActPlane works at this layer: it installs lightweight eBPF programs in the kernel that hook every path an agent might produce side effects through, then makes decisions based on labeled information-flow policies.

Specifically, ActPlane hooks the entire process lifecycle (`sched_process_fork`, `sched_process_exec`, `sched_process_exit`), file operations (`sys_enter_openat`, `sys_enter_unlinkat`, `sys_enter_renameat2`), and network connections (`sys_enter_connect`). No matter what path an agent takes to produce side effects, it will pass through these syscalls.

How does this differ fundamentally from the three layers above? Go back to the opening scenario. The agent wrote a Python script that spawned a subprocess calling git push. The tool layer only saw `python script.py`; ActPlane sees all syscalls across the entire process tree, including that git three layers deep. And constraints aren't attached to "the git tool" but to the process tree. "Codex's entire subprocess tree cannot touch git" is one rule — no need to guard every possible tool entry point.

But ActPlane doesn't just push tool-layer guards down to the kernel. It introduces two capabilities that neither tool layers nor sandboxes have: **data-flow tracking** and **temporal reasoning**. Labels propagate across fork/exec and file read/write boundaries, making "data read from A must not flow to B" an expressible policy. The `since` clause lets rules dynamically update on the event timeline — "have tests been run since the last source modification" becomes a predicate that invalidates and rebuilds as new events occur. The next two sections unpack these mechanisms. But first, a note on what kind of tool ActPlane is — it's not just a deeper sandbox.

## Harness, Not Just a Sandbox

A sandbox draws an isolation boundary. Everything inside is allowed, everything outside is forbidden. For untrusted code this is the right model: you don't trust the code, so you lock it in a cage. But an agent isn't untrusted code. An agent is your collaborator — you want it to complete the task, you just want it to follow certain constraints along the way.

Sandboxes answer "can this process access this resource." A harness answers a broader set of questions. Not just security ("sensitive data must not reach the network"), but software engineering discipline. "Run tests before committing" isn't a security constraint, it's an engineering workflow. "Don't mix data from independent tasks in one commit" isn't a permissions problem, it's a work habit. "Access prod.db through the migration tool, not directly" isn't an isolation issue, it's an operational standard. These workflow constraints are exactly the kind of rules agents need when operating autonomously in real codebases. Sandboxes can't express them because they're not allow/deny resource access problems.

At the same time, a harness subsumes sandbox capabilities. When an agent spawns a sub-agent or runs an untrusted command, you can write a rule confining the entire subtree to read-only, no-network, or a specific directory. This is exactly what traditional sandboxes do, but in ActPlane it's just a subset of the rules. You can have sandbox-style rules ("this subprocess tree can't access the network") and workflow rules ("must run tests before committing") in the same policy file.

The feedback loop is the most critical aspect of harness design. When a rule fires, ActPlane delivers the reason to the agent through its framework's hook system:

```
🚫 KILLED: process 'git' (pid 4213, ppid 4210) — /usr/bin/git
   effect: kill
   reason: no git under the agent; use the review workflow
```

The agent receives the reason, understands the constraint, and takes a different path to complete the task. It doesn't need to "remember" that it can't use git — it can try, then be told why it can't and what to do instead. This "deterministic constraints + probabilistic decisions" combination forms an interesting architectural pattern: the agent's reasoning is still probabilistic (that's what makes LLMs useful), but critical constraints are enforced deterministically by the kernel, and the feedback on violations lets the agent self-correct rather than hitting a wall.

## Core Mechanism: Label Propagation

ActPlane's policy is not a static allow/deny list. It uses labeled information-flow policies: processes and files get labels, labels propagate automatically along fork/exec edges and file read/write edges, and rules make decisions based on labels. The academic roots of this model trace back to [CamQuery](https://dl.acm.org/doi/10.1145/3243734.3243776) (CCS 2018) and [CamFlow](https://dl.acm.org/doi/10.1145/3127479.3129249) (SoCC 2017), which implemented cross-channel taint propagation and enforcement on an in-kernel provenance graph. ActPlane brings the same idea to the modern eBPF/BPF-LSM substrate — no custom kernel module needed — and targets cooperative-but-forgetful AI agents rather than remote adversaries. This sounds abstract, but walking through one example makes it concrete.

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

The user-space part is the `actplane` Rust binary. The eBPF programs are precompiled into CO-RE (Compile Once, Run Everywhere) format and embedded in the binary — no clang, llvm, libbpf, or any compilation toolchain is needed at install time. The deployment path is `cargo install actplane`, then `actplane init` to generate a starter config, `actplane check` to validate rules, and `sudo actplane run <command>` to execute the agent under the harness. eBPF programs are checked by the kernel verifier, guaranteeing they won't crash the kernel or loop infinitely.

At runtime, the binary loads the precompiled eBPF object in-process via [aya](https://github.com/aya-rs/aya), parses `actplane.yaml` and compiles the DSL into kernel configuration (written into the `.rodata` section), seeds the target process's lineage (telling the kernel "start tracking from this process"), and listens on the ring buffer for rule matches and their policy reasons. Compared with [Cilium Tetragon](https://tetragon.io/), which provides `matchBinaries` + `followChildren` to propagate a binary lineage flag along fork/exec — the closest OSS feature to ActPlane's lineage tracking — ActPlane additionally propagates labels across file and network edges and provides semantic feedback to the agent.

On permissions: `actplane run` and `actplane watch` need root or `CAP_BPF` + `CAP_SYS_ADMIN` to load the eBPF engine. But once loaded, the target command is dropped back to the current user. The agent itself doesn't run as root. `actplane check` needs no privileges at all — it loads no eBPF programs, only performs static rule validation.

## Use Cases and Limitations

ActPlane's strongest use cases share a common characteristic: single-layer constraints aren't enough.

Cross-vendor multi-agent collaboration is the most typical. When Claude Code calls Codex and Codex calls a custom tool chain, each vendor's framework-level guards only know their own registered tools. Claude Code's hooks don't know Codex's permission config, and vice versa. Framework-level guards assume "I know what paths the agent will use to operate the system" — cross-vendor calls immediately break that assumption. But OS-level rules propagate along process lineage regardless of which vendor's runtime is underneath. One rule governs the entire cross-vendor execution tree — this is the harness-over-sandbox distinction in its sharpest form.

CI/CD agent governance is another strong scenario. Agents running in CI environments need stricter constraints: can't push code, can't modify CI config, must pass tests before building artifacts. These temporal constraints are exactly what `since` clauses do. Agents deployed in sensitive environments need data-flow-level policies like "data read from prod.db must not flow to the network." Sandboxes can't track at this granularity; label propagation can.

But ActPlane isn't universal. It's built on eBPF, so it only runs on Linux and requires kernel 5.8+ with BTF support (`/sys/kernel/btf/vmlinux`). macOS and Windows agent development scenarios aren't covered, though most production deployments are on Linux. Loading eBPF programs requires root or `CAP_BPF` + `CAP_SYS_ADMIN` — some shared servers and cloud containers won't grant this. Kernel-level tracking only reaches syscall granularity; in-process memory operations and encryption/decryption are out of scope. Block mode requires BPF-LSM, which not all distributions enable by default.

## Conclusion

An agent's value lies in flexibility and creativity; deploying agents requires predictability and safety guarantees. There's tension between these two. Prompts are suggestions not rules, tool-layer guards are bypassed by a single shell-out, sandboxes can only do allow/deny resource isolation.

ActPlane adds a layer of deterministic constraints at the kernel. The agent still reasons freely, but critical operations are adjudicated by information-flow rules. When constraints trigger, the agent gets actionable feedback rather than error codes. It doesn't replace the first three layers, but closes each of their blind spots.

Go back to the opening scenario: the agent wrote a Python script that called `subprocess.run(["git", "push"])`. Under ActPlane, the `AGENT` label propagates along process lineage from Claude Code to bash to Python to that git three layers deep — the rule fires, the operation is intercepted, the agent receives the reason and an alternative path. What the prompt couldn't stop, the kernel did.

In complex systems, every single-layer constraint has holes, and agents will naturally find paths through them. Layered constraints may be a necessary architectural component for agents heading toward production deployment.

---

> **GitHub**: [github.com/eunomia-bpf/ActPlane](https://github.com/eunomia-bpf/ActPlane) — MIT License
>
> ActPlane is an open-source project from the [eunomia-bpf](https://github.com/eunomia-bpf) community. Built on [AgentSight](https://github.com/eunomia-bpf/agentsight/)'s eBPF observability foundation.
