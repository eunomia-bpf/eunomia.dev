---
date: 2026-05-31
description: ActPlane is an eBPF-based policy engine that observes and enforces AI agent behavior at the OS kernel level. This post analyzes the systemic blind spots of prompt constraints, tool-layer guards, and sandboxes, and explains how ActPlane uses label propagation and temporal predicates to implement a deterministic agent harness.
---

# ActPlane: Pushing Agent Harness Enforcement Down to Kernel eBPF

You wrote a rule in CLAUDE.md: "do not run `git push`." The agent obeyed, and it never called the git tool. But it wrote a Python script that called `subprocess.run(["git", "push"])`, and the code was pushed to the remote. The prompt constraint was never violated.

This scenario exposes a structural problem: **agent constraints and agent side effects live on different planes.** Prompt constraints operate at the reasoning layer, tool guards at the API layer, sandboxes at the container layer, but every side effect an agent produces ultimately passes through the OS kernel. Every exec, every file open, every network connect: no matter what path the agent takes, the kernel is there. If constraints aren't enforced at this layer, the agent can always find a way around them, and it doesn't even need to be malicious; it's just trying to finish the task you gave it.

[ActPlane](https://github.com/eunomia-bpf/ActPlane) is built on this insight. It installs a policy engine in the kernel via eBPF, observing and enforcing agent constraints at the syscall layer. When a rule matches, it fires with no dependency on the agent "remembering" anything. But ActPlane does more than block: when a constraint triggers, the agent receives human-readable feedback explaining why it was stopped and what to do instead. The agent understands the reason, takes a different path, and continues the task. This is the fundamental difference between a harness and a sandbox: a sandbox gives you a wall and a `Permission denied`; a harness gives you a rule and an alternative.


<!-- more -->

## What We Actually Want

Before dissecting the problem, consider a few constraints that agents genuinely need in day-to-day work. None of them can be said clearly enough in a prompt, enforced by an MCP gateway, or expressed by a container sandbox.

Preventing an agent from running `git push` sounds like an ordinary sandbox rule, until you realize that the coverage of "prevent" depends entirely on the layer doing the checking. The agent might call git directly, write a shell script that calls it, or generate a Python program with nested subprocess calls. The tool API layer can only check "did the agent call the git tool," but the real question is "did any process in this agent's process tree execute the git binary," and those two questions have completely different coverage.

Now consider a temporal constraint: after modifying `specs/*`, the agent must run `protoc` before committing, otherwise it should get a reminder. We don't want to block the agent from editing spec files, since that's part of its normal work, but if it edits a spec and goes straight to commit, it skipped a step. The goal is not to prevent the commit, but to remind the agent before it happens so it can decide whether to run code generation.

"Tests must pass before committing" follows similar logic but adds dynamic invalidation: every time the agent modifies a file under `src/`, previous test results should automatically expire. The check isn't "have tests ever been run," but "have tests been run since the last source file modification." Run tests, then change one line of code? The status resets and you must run them again.

Finally, a mandatory mediation constraint: the production database `prod.db` may only be accessed through the migration tool, and the agent can't open it directly. No matter how the agent reaches the file-open call, if its process ancestor chain hasn't executed the `migrate` tool, the operation is blocked. What matters isn't whether the agent has some abstract permission, but which path it took to get here: pass through the designated gate program and you're allowed in; bypass it and you're stopped.

Together, these four constraints involve process lineage tracking, temporal ordering, dynamic invalidation, and mandatory mediation, all beyond the scope of static allow/deny. To see why meeting them requires a kernel-level approach, let's look at where the existing three layers of constraints each fall short.

## Three Layers, Three Blind Spots

| Approach | What it does | What it can't cover |
|----------|-------------|---------------------|
| **Prompt constraints** (CLAUDE.md, AGENTS.md) | Tell the agent what to do and not do | Probabilistic: agents forget or non-maliciously route around them in long contexts |
| **Tool-layer guards** (MCP gateway, AgentSpec) | Intercept and authorize at the tool API | Completely bypassed when agents shell out, link SDKs, or spawn subprocesses |
| **Sandboxes** (containers, VMs, E2B, Daytona) | Isolate the entire execution environment | All-or-nothing: can't express "file A may only be accessed via script A" or "run tests before committing" |

CLAUDE.md, AGENTS.md, and system prompts are the most common constraint mechanism today: natural-language instructions that depend on the model "remembering and obeying" during inference. The trouble is with "remembering." When conversations exceed tens of thousands of tokens, early constraints get diluted in attention allocation by subsequent user instructions and tool outputs. The model hasn't technically forgotten the rule, but the rule's weight in decision-making drops as the context expands. In our [AgentCgroup characterization study](https://eunomia.dev/blog/2026/02/17/agentcgroup-characterization/), we observed that agents average dozens of tool calls per task and the conversation context grows very large. The earlier a constraint is set and the longer the conversation, the lower the probability of effective compliance. Even harder to defend against is indirect violation: the agent is told "don't delete files," so it writes a Makefile target containing `rm -rf` and calls `make clean`. From the agent's perspective it just executed a build command, and each decision point is locally reasonable, but the resulting behavior chain violates the original policy. Prompt constraints govern what the agent intends; actual behavior is outside their jurisdiction.

What if we push down one layer and intercept at the tool call API? MCP gateways, [AgentSpec](https://arxiv.org/abs/2503.18666), and [Progent](https://arxiv.org/abs/2504.11703) are indeed far more reliable than prompts: if the agent calls `file_write("/etc/passwd", ...)`, the tool layer rejects it outright, checking the operation itself rather than intent. But nearly every agent framework provides a shell tool, because agents need to run builds, execute tests, and call third-party CLIs. Once the agent has bash, it can write a Python script whose internals call `subprocess.run(["curl", "-X", "POST", ...])` to send data to an external network. The tool layer sees `python script.py` and knows nothing about the subprocesses spawned inside; the call chain can be three or four layers deep, but the tool layer only sees the first. Less obviously, agent-generated code can directly link database drivers or HTTP client libraries, producing side effects at runtime that never pass through any tool call path. The tool layer's vision stops at the registered tool entry point, and side effects from code execution are entirely outside its view.

Pushing down one more layer, we reach full environment isolation. Containers, VMs, [E2B](https://github.com/e2b-dev/E2B), and [Daytona](https://github.com/daytonaio/daytona) are the most reliable security boundary available today. For preventing agent escape to the host, they are the right answer. But the constraints agents actually need are far richer than "can or can't access a resource." "After modifying proto files, the agent must run protoc before committing" is a temporal constraint, and sandboxes have no concept of time and only know which resources are accessible at the current instant. "Sensitive data read from the database must not be written to logs" requires tracking data flow, but sandboxes operate at process granularity and don't know what a process read or wrote internally. And the same `git commit` should have different policies depending on whether tests just ran or haven't, a distinction sandboxes can't make because they have no sense of historical context.

A frequently underestimated problem compounds all of this: feedback quality. When a sandbox rejects an operation, the agent receives only `Permission denied` or `EPERM` and doesn't know why it was rejected or what to do to satisfy the constraint. The typical result is the agent retrying three to five times and then giving up on the entire task, or worse, trying another path to bypass the rejection and introducing new problems. Sandboxes answer "can this process access this resource," but agents need far richer questions answered: under what conditions, in what order, and based on what data-flow history is this operation permitted?

## The Path All Side Effects Must Travel

All three layers' blind spots point in the same direction: the constraint mechanism must sit on the path that every operation travels through. [AgentSight](https://arxiv.org/abs/2508.02736) (2025) used eBPF to capture both intent-layer and action-layer agent data, coining the concept of a "semantic gap" between the two. ActPlane builds enforcement on that observability foundation. No matter what tool the agent uses, what script it writes, or how many layers of subprocesses it spawns, all side effects ultimately pass through OS kernel syscalls. ActPlane installs lightweight eBPF programs in the kernel, hooking the process lifecycle (`sched_process_fork`, `sched_process_exec`, `sched_process_exit`), file operations (`sys_enter_openat`, `sys_enter_unlinkat`, `sys_enter_renameat2`), and network connections (`sys_enter_connect`), then makes decisions based on labeled information-flow policies. Regardless of what path an agent takes, it will pass through these syscalls.

Return to the opening scenario to see the difference: the agent wrote a Python script that spawned a subprocess calling git push. The tool layer only saw `python script.py`, but ActPlane sees all syscalls across the entire process tree, including that git call three layers deep. Because constraints attach directly to the process tree, "Codex's entire subprocess tree cannot touch git" is a single rule, with no need to guard every possible tool entry point.

ActPlane goes beyond simply pushing tool-layer guards down to the kernel, introducing two capabilities that neither tool layers nor sandboxes have: **data-flow tracking** and **temporal reasoning**. Labels propagate across fork/exec and file read/write boundaries, making "data read from A must not flow to B" an expressible policy. The `since` clause lets rules dynamically update on the event timeline, turning "have tests been run since the last source modification" into a predicate that invalidates and rebuilds as new events occur. The next two sections unpack these mechanisms, but first let's clarify what ActPlane is, because it's not just a deeper sandbox.

## Harness, Not Just a Sandbox

A sandbox draws an isolation boundary: everything inside is allowed, everything outside is forbidden. For untrusted code this is the right model: you don't trust it, so you lock it in a cage. But an agent isn't untrusted code; it's your collaborator, and you want it to complete the task while following certain constraints along the way.

Many of these constraints have nothing to do with security permissions, yet they're exactly the kind of rules agents need when operating autonomously in real codebases. "Run tests before committing" is an engineering workflow, "access prod.db through the migration tool" is an operational standard, "don't mix independent tasks in one commit" is a work habit. Sandboxes can't express them because their semantics go beyond resource access. A harness, however, also subsumes sandbox capabilities: when the agent runs an untrusted command, you can write a rule confining the entire subtree to read-only, no-network, or a specific directory. In ActPlane this is just a subset of the rules, coexisting with workflow rules in the same policy file.

The feedback loop is the most critical part of harness design. When a rule fires, ActPlane delivers the reason to the agent through its framework's hook system:

```
🚫 KILLED: process 'git' (pid 4213, ppid 4210) — /usr/bin/git
   effect: kill
   reason: no git under the agent; use the review workflow
```

The agent reads the reason, understands the constraint, and takes a different path to complete the task. It doesn't need to "remember" that it can't use git beforehand; it can try, get told why it can't and what to do instead, and self-correct. This produces an interesting architectural pattern: the agent's reasoning remains probabilistic (that's what makes LLMs useful), while critical constraints are enforced deterministically by the kernel. Deterministic constraints and probabilistic decisions, joined by a feedback loop, yield an architecture that is both flexible and controllable.

## Core Mechanism: Label Propagation

ActPlane's policies are not static allow/deny lists. Instead, they are labeled information-flow policies: processes and files get labels, labels propagate automatically along fork/exec edges and file read/write edges, and rules make decisions based on those labels. The academic roots trace back to [CamQuery](https://dl.acm.org/doi/10.1145/3243734.3243776) (CCS 2018) and [CamFlow](https://dl.acm.org/doi/10.1145/3127479.3129249) (SoCC 2017), which implemented cross-channel taint propagation and enforcement on an in-kernel provenance graph. ActPlane brings the same idea to the modern eBPF/BPF-LSM substrate, requiring no custom kernel module, and targeting cooperative-but-forgetful AI agents rather than remote adversaries.

Walking through an example makes this concrete. Say we want to constrain Claude Code's entire process tree. First, declare a label source: `source AGENT = exec "claude"`, meaning whenever a process executes a binary named `claude`, it gets the `AGENT` label. Labels aren't generated by default: only behaviors explicitly declared as sources produce them, so the system tracks only the data flows you care about. Similarly, `source SCHEMA_CHANGED = file "src/protocol/**/*.proto"` declares that when these files are written, the writer gets the `SCHEMA_CHANGED` label.

Once a label is created, it propagates automatically along two dimensions. Along process lineage: a parent process with the `AGENT` label forks a child, and the child inherits it. The child execs a new program and inherits it too. The agent starts bash, bash starts Python, Python calls subprocess to start git, and every layer carries the `AGENT` label. Labels don't get lost at process boundaries. Along file I/O: a process with the `AGENT` label writes to file `f`, and `f` gets the `AGENT` label too. Later, a process that didn't originally have that label reads `f` and also gets marked `AGENT`. When data crosses process boundaries through files, labels follow the data.

With labels in place, rules are straightforward. `kill exec "git" "push" if AGENT` means any process with the `AGENT` label that tries to execute `git push` gets terminated. Back to the opening scenario:

```
Claude Code (AGENT) 
  → bash (inherits AGENT)
    → python script.py (inherits AGENT) 
      → subprocess: git push (inherits AGENT) ← rule fires here
```

No matter how deep the nesting or how many process boundaries are crossed, labels propagate along lineage all the way and the rule fires at the final exec call point. The checkpoint has moved from the tool entry down to the kernel's exec call, covering every path the agent might take.

File-direction label propagation solves a different class of problems. Suppose an agent exports sensitive data from a production database to `data.csv`: the process gets a `DB_SENSITIVE` label, and the file does too. A later step reads `data.csv`, acquiring the `DB_SENSITIVE` label, then tries to send the content over the network. At that point `block connect if DB_SENSITIVE` blocks the network connection. The data flow crossed both process and file boundaries, but labels followed it throughout.

Kernel-level tracking operates at syscall granularity: ActPlane knows process P wrote to file F, but not what content was written. This means over-tainting can occur: a process reads one line from a sensitive file, then writes completely unrelated data to another file, and the second file still gets marked sensitive. In academic literature this is called label creep. In practice three factors mitigate it. Different agent tasks typically run in different process trees, so labels are naturally isolated. Rule matching checks operation-type-plus-label combinations (`block connect if DB_SENSITIVE` only fires on network connections, not on every file write). And only explicitly declared sources produce labels in the first place. This is a deliberate design tradeoff: better to over-label than to miss real data flows.

## Temporal Constraints: The `since` Clause

Label propagation answers "who did it" and "where did the data come from." But many of the constraints mentioned earlier also involve temporal ordering, such as "run tests before committing," "run protoc after changing specs," which require knowing "has event X happened after event Y." ActPlane uses the `since` clause to reason over the event timeline: "after X happens and before Y, Z is not allowed."

Combining labels with temporal logic yields highly expressive policy files. The following four rules demonstrate four distinct constraint patterns:

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

`no-git-branch` is the simplest: any process in the agent's tree that tries `git branch` or `git worktree` is terminated immediately, with no conditions and no temporal logic. The agent receives the `because` content and knows to use other git commands or ask the user to manage branches.

`regenerate-after-schema` is a cross-event conditional rule using the notify effect. Its `unless` clause asks a specific temporal question: since the last time a process wrote to the protocol directory, has `protoc` been executed? If yes, the commit proceeds; if no, the agent gets a reminder. The key is the `since` clause's dynamic nature: whenever the protocol directory is written to again, the "already ran protoc" state resets and must be re-established, a predicate that dynamically updates on the event timeline rather than a one-time static check.

`test-before-commit` has similar semantics but is stricter, using the block effect: have tests been run since the last write to `src/**`? If not, the commit is blocked before it executes. Every source file modification resets the test status, so changing even one line after a passing test run means running them again.

`mediate-proddb` takes a different approach: instead of relying on labels, it uses `lineage-includes` to check process ancestry. Any process wanting to open `prod.db` must have `migrate` somewhere in its ancestor chain. This rule expresses "the only legitimate access path": the agent directly calling `open("prod.db")` gets blocked, but calling `./migrate`, which internally opens the file, succeeds. What matters isn't whether the agent holds some abstract permission, but whether its lineage passed through the designated gate program.

Traditional sandboxes have no concept of "time" or "path" and only know the current instant's state. By maintaining both an event timeline for temporal reasoning and process lineage for path checking, ActPlane can express workflow constraints and mandatory mediation, not just access control.

## Agent Integration

ActPlane delivers rule match reasons to agents through their framework's hook systems, with the kernel remaining the sole authority for observation and enforcement. Hooks only relay match events into the agent's decision context.

Claude Code integration is configured via `.claude/settings.local.json` with `PostToolUse` and `PostToolUseFailure` hooks:

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

`actplane feedback-hook` is straightforward: it checks whether any new rule match events have occurred since its last invocation and, if so, forwards the `because` content to the hook output. Because the agent runs this hook after every tool call, any triggered rule's reason enters the next decision's context, and the entire integration requires no modifications to the agent framework itself.

## Runtime Architecture

ActPlane's runtime splits into kernel-space and user-space:

```
actplane.yaml ─▶ compiler (Rust) ─▶ .rodata config ─▶ eBPF kernel engine
 policy: |        parse + lower DSL    (set_global)      propagate labels,
                                                          match rules,
 matches ◀─────── ring buffer (in-process, via aya) ◀─── emit on match only
```

The kernel part (`bpf/` directory) maintains a per-node label set for processes, files, and network endpoints, executes propagation, evaluates compiled rules, and emits events to user-space via ring buffer only when rules match. Unmatched operations produce zero user-space overhead, which matters because an active agent can trigger hundreds of file operations and process creations per second. If every operation notified user-space for a decision, latency would be unacceptable. Label propagation and rule matching complete entirely in kernel space, and user-space participates only when a rule fires.

The user-space part is the `actplane` Rust binary, with eBPF programs precompiled into CO-RE (Compile Once, Run Everywhere) format and embedded in the binary. Installation needs no clang, llvm, libbpf, or any compilation toolchain. The deployment path is `cargo install actplane` → `actplane init` to generate a starter config → `actplane check` to validate rules → `sudo actplane run <command>` to execute the agent under the harness. eBPF programs are checked by the kernel verifier, guaranteeing they won't crash the kernel or loop infinitely.

At runtime the binary loads precompiled eBPF objects in-process via [aya](https://github.com/aya-rs/aya), parses `actplane.yaml`, compiles the DSL into kernel configuration (written into `.rodata`), seeds the target process's lineage, and then listens on the ring buffer. Compared with [Cilium Tetragon](https://tetragon.io/), whose `matchBinaries` + `followChildren` can propagate a lineage flag along fork/exec (the closest OSS feature to ActPlane's lineage tracking), ActPlane additionally propagates labels across file and network edges and provides semantic feedback to the agent.

On permissions: `actplane run` and `actplane watch` need root or `CAP_BPF` + `CAP_SYS_ADMIN` to load the eBPF engine, but once loaded the target command drops back to the current user. `actplane check` needs no privileges at all, as it only performs static rule validation.

## Use Cases and Limitations

Kernel-level constraints prove most valuable when multiple agents from different vendors collaborate. When Claude Code calls Codex and Codex calls a custom tool chain, each vendor's framework-level guards only know their own registered tools: Claude Code's hooks don't know Codex's permission config, and vice versa. Framework-level guards assume "I know what paths the agent will use to operate the system," and cross-vendor calls immediately break that assumption. OS-level rules, by contrast, propagate along process lineage regardless of which vendor's runtime is underneath, so a single rule governs the entire cross-vendor execution tree.

CI/CD environments impose even stricter requirements: agents in build pipelines can't push code, can't modify CI config, and must pass tests before producing build artifacts, all temporal constraints that `since` clauses are designed to express. In deployments involving sensitive data, agents also need data-flow-level policies like "data read from prod.db must not flow to the network." Traditional sandboxes can't track cross-process data flow at this granularity, but label propagation can.

ActPlane also has clear boundaries. Because it is built on eBPF, it only runs on Linux 5.8+ with BTF support (`/sys/kernel/btf/vmlinux`), leaving macOS and Windows agent development scenarios uncovered, though most production deployments are on Linux. Loading eBPF programs requires root or `CAP_BPF` + `CAP_SYS_ADMIN`, which some shared servers and cloud containers won't grant. Kernel-level tracking reaches only syscall granularity, so in-process memory operations and encryption/decryption are out of scope. Block mode depends on BPF-LSM, which not all distributions enable by default.

## Can the Agent Recover and Keep Working?

Blocking the nested `git push` from the opening example is only half the job. If the agent sees an unexplained `EPERM`, retries the same action, and abandons the task, the policy is safe but the harness is not useful. The paper therefore evaluates policies collected from an empirical study on both coding tasks and safety benchmarks, including generated scripts and nested subprocesses that bypass tool-call interception.

Across the evaluated configurations, ActPlane improves policy compliance with 1.9% to 8.4% overhead. The experiments also follow what happens after a rule fires: actionable feedback gives the agent a reason and a recovery path, allowing it to change its plan instead of treating enforcement as an opaque system failure.

## A Kernel Checkpoint the Agent Can Understand

Agent reasoning remains useful because it can find unexpected ways to complete a task. Production policy needs the opposite property at a few critical checkpoints: every path to a protected side effect must reach the same decision. ActPlane places those decisions at the kernel boundary, then returns enough context for the agent to revise its plan. The larger idea is a feedback loop in which flexible reasoning proposes actions, deterministic enforcement judges their effects, and the agent can continue after learning why one path was rejected.

---

> **Paper**: [arXiv:2606.25189](https://arxiv.org/abs/2606.25189)
>
> **GitHub**: [github.com/eunomia-bpf/ActPlane](https://github.com/eunomia-bpf/ActPlane), MIT License
>
> ActPlane is an open-source project from the [eunomia-bpf](https://github.com/eunomia-bpf) community. Built on [AgentSight](https://github.com/eunomia-bpf/agentsight/)'s eBPF observability foundation.
