# ActPlane: eBPF-Based IFC Policy Engine for AI Agent Harnesses

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2606.25189-b31b1b.svg)](https://arxiv.org/abs/2606.25189)

**Runtime `enforcement` and `observability` for AI agent harnesses and sandboxing: declare information-flow policies for safety, security and compliance, and ActPlane enforces them in the kernel with eBPF.**

Prompt constraints and model guardrails are probabilistic. ActPlane is deterministic. Tool call check cannot see indirect system behavior, e.g. a sh script.

**What you can express:**

- **"No `codex` may run `git push` or write outside `/src`"**: fine-grained sandboxing rules follow process lineage, no bypass via bash scripts or python.
- **"Never remove the build cache in makefile unless explicitly asked or debugging"**: bypassable with a specific argument when necessary, not just sandbox.
- **"When changing `specs/*`, also update the server, SDK, and docs"**: ActPlane never blocks the edit, it notifies the agent that downstream outputs are now stale.
- **"Run  `make check` & `npm tests` before committing"**: causal ordering, not just per-operation checks.


## Quickstart

Install with one command. The eBPF program ships prebuilt (CO-RE, architecture
independent), so there is **no clang/llvm/libbpf to install** — just a Rust
toolchain:

```bash
cargo install actplane
```

Write a policy and run an agent (or any command) under the harness:

```bash
actplane init                                  # write a starter actplane.yaml
actplane compile                               # validate rules (no privileges)
actplane doctor                                # diagnose hooks, MCP, kernel support

codex                                         # MCP auto-attach tries passwordless sudo
sudo -E actplane run claude -p "review this repo"
```

When a rule matches, ActPlane kills the action and tells the agent why:

```
🚫 KILLED: process 'git' (pid 4213, ppid 4210) — /usr/bin/git
   effect: kill
   reason: no git under the agent; use the review workflow
```

The agent receives this reason through its hook integration, understands the
constraint, and takes a different path to complete the task.

**Requirements:** Linux kernel 5.10+ with BTF (`/sys/kernel/btf/vmlinux`). Linux
5.10-6.0 supports static exec/file/IPv4 policies (file suffixes up to 16 bytes); the full runtime requires 6.1+. `run`
and `watch` load the eBPF engine, so they need root (or `CAP_BPF` +
`CAP_SYS_ADMIN`); Linux 5.10 also needs `CAP_SYS_RESOURCE` when its memlock hard
limit is finite. ActPlane drops the target command back to your user. In the
supported operations, BPF-LSM lets rules `block` before they commit; otherwise
they `notify` (report) or `kill`.

## Why an OS-level harness?

Agent constraints today come in three forms. Each solves a real problem but
leaves a gap that the next layer down needs to cover.

| Approach | What it does | What it can't cover |
|----------|-------------|---------------------|
| **Prompt constraints** (`CLAUDE.md`, `AGENTS.md`) | Tell the agent what to do and not do | Probabilistic: long-context agents forget or route around them, often non-maliciously |
| **Tool-layer guards** (MCP gateways, AgentSpec) | Intercept and authorize at the tool API | Bypassed the moment the agent shells out, links an SDK, or spawns a subprocess |
| **Sandboxes** (containers, VMs, E2B, Daytona) | Isolate the entire execution environment | All-or-nothing: can't express "file A must only be accessed via script A" or "run tests before committing" |

ActPlane sits below all three, at the OS level. Every `exec`, file open, and
network connect goes through the kernel, so a rule like *"nothing descended from
`codex`, however many hops, may run `git` or modify files outside `/work`"*
holds regardless of which tool path the agent takes.

The key differences:

- **OS-level coverage**: observation and enforcement happen at the kernel, not the tool API. Bash, Python subprocess, direct SDK calls, all covered.
- **Call-chain granularity**: rules follow process lineage, not just single operations. "Codex's entire subprocess tree cannot touch git" is one rule.
- **Data-flow constraints**: rules express "data read from A must never flow to B", tracked across arbitrary fork/exec and file read/write edges, not just at a boundary.
- **Causal ordering**: rules express "run tests before committing" via `since` clauses and gate invalidation, not just per-operation checks.
- **Corrective feedback, not just blocking**: rule matches feed a human-readable reason back to the agent, so it can retry a different way. This is what makes it a harness, not a sandbox.
- **Agent-maintained rules**: the rule language is designed so agents can write, validate (`actplane check`), and evolve their own policies.

## Harness, not just a sandbox

A sandbox draws an isolation boundary: everything inside is allowed, everything
outside is denied. That works for untrusted code, but agents need something
richer — the data-flow, causal-ordering, and corrective-feedback properties
above are things no isolation boundary can express.

Sandboxes answer "can this process access this resource?" A harness answers a
broader set of questions: not just security ("secret data must not reach the
network") but also software engineering discipline ("run tests before
committing", "don't mix data from independent tasks in one commit", "use the
migration tool to access prod.db"). These are workflow constraints, not access
control, and they are exactly the kind of rules agents need to operate
autonomously in real codebases.

A harness also subsumes sandboxing when you need it. When an agent spawns a
sub-agent or runs an untrusted command, you can write a rule that confines the
entire subtree to read-only, no-network, or a specific directory. This is
especially important when agents cross vendor boundaries: Codex calling Claude
Code, or the other way around. Framework-level guards from different vendors
don't compose, but OS-level rules follow process lineage regardless of which
runtime is underneath.

## How rules work

Rules are **labeled information-flow policies**, not static allow-lists.
Labels propagate along fork/exec edges and file read/write edges, so
constraints follow derived data across processes and files.

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
    because "This workspace forbids creating git branches or worktrees. Use other git commands, or ask the user to manage branches."

  rule regenerate-after-schema:
    notify exec "git" "commit"
      if SCHEMA_CHANGED unless after exec "protoc" since write "src/protocol/**"
    because "Protocol schema changed — generated code may be stale. Run `make proto` to regenerate, then commit."

  rule test-before-commit:
    block exec "git" "commit"
      if AGENT unless after exec "pnpm" "test" since write "src/**"
    because "Source files changed since last test run. Run `pnpm test:changed`, then commit."
```

Three rules, three effects, three patterns:

- **`no-git-branch`** (kill): per-event rule — anything in the agent's
  process tree that tries `git branch` is terminated immediately.
- **`regenerate-after-schema`** (notify): cross-event conditional — if
  the agent modified a `.proto` file, ActPlane reminds it to run `protoc`
  before committing. The `since` clause re-arms the gate whenever the
  schema changes again.
- **`test-before-commit`** (block): cross-event temporal with staleness —
  the agent must run tests before committing, and editing any `src/`
  file invalidates the previous test run.

See [`docs/rule-language.md`](rule-language.md) for the full rule language and
worked examples.

## Agent integration

ActPlane feeds rule-match reasons back to agents via their hook systems.

`actplane init --with-codex` writes a ready-to-use Codex hook at
`.codex/hooks.json`. It runs `actplane feedback-hook` after each tool call and
injects any new `.actplane/last-violation.txt` content into the next model turn.

**Claude Code** (`.claude/settings.local.json`):

```json
{
  "hooks": {
    "PostToolUse": [{ "matcher": "*", "hooks": [{ "type": "command", "command": "actplane feedback-hook" }] }],
    "PostToolUseFailure": [{ "matcher": "*", "hooks": [{ "type": "command", "command": "actplane feedback-hook" }] }]
  }
}
```

**Codex** (`.codex/hooks.json`):

```json
{
  "hooks": {
    "PostToolUse": [{ "matcher": ".*", "hooks": [{ "type": "command", "command": "actplane feedback-hook" }] }]
  }
}
```

The adapter forwards new rule matches as hook context. The kernel remains the sole
authority for observation and enforcement. See [`script/CLAUDE.snippet.md`](https://github.com/eunomia-bpf/ActPlane/blob/master/script/CLAUDE.snippet.md)
for the agent instruction snippet.

ActPlane also ships an MCP server:

```bash
actplane mcp --auto-attach-parent
```

It exposes `actplane:///policy` for live policy validation and
`actplane:///feedback` for the latest corrective feedback. When started by Codex,
`--auto-attach-parent` tries passwordless sudo, loads the eBPF engine, and seeds
the parent Codex process so directly running `codex` is protected.

Attach an already-started agent with a foreground engine:

```bash
sudo -E actplane attach --pid <pid>
```

Once an MCP auto-attach or `actplane watch` engine is already running, bind an
already-started subagent into a child runtime domain with:

```bash
actplane attach --pid <pid> --child-domain --domain-id <domain-id>
actplane attach --pid <pid> --child-domain --delta child-policy.dsl
```

`attach` is post-hoc: future events from that process tree enter ActPlane, but
ActPlane does not reconstruct labels or file/network history from before the
attach. For strict launch-time enforcement, start the process with
`actplane run ... -- <cmd>` or `actplane control launch-child ... -- <cmd>`.

`actplane init --with-mcp` also writes project `.mcp.json`:

```json
{
  "mcpServers": {
    "actplane": {
      "type": "stdio",
      "command": "actplane",
      "args": ["mcp", "--auto-attach-parent"]
    }
  }
}
```

Prefer the project `.mcp.json` that `actplane init --with-mcp` writes. If your Codex build
does not read project MCP config, use a global `codex mcp add actplane -- actplane
mcp --auto-attach-parent` entry instead, but do not keep both or auto-attach can
start twice.

## How it works

```
actplane.yaml ─▶ policy compiler ─▶ runtime/control ─▶ eBPF kernel engine
 policy: |        parse + lower DSL   load + seed       propagate labels,
                                      domains           match rules,
 matches ◀─────── feedback/report ◀── ring buffer ◀──── emit on match only
```

- **Kernel** (`bpf/`): hooks `fork / exec / exit / open / unlink / rename / connect`,
  keeps a per-node label set (process / file / endpoint), propagates labels,
  evaluates compiled rules, emits only match events.
- **Policy compiler** (`crates/actplane-ifc-compiler/`): parses the ActPlane IFC
  policy language and lowers it to the fixed kernel config ABI.
- **Runtime library** (`crates/actplane-runtime/`): resolves `actplane.yaml`,
  loads the prebuilt eBPF object in-process via
  [`ebpf-ifc-engine`](https://github.com/eunomia-bpf/ActPlane/tree/master/bpf/) (aya) — no libbpf/clang at runtime — seeds the target
  process lineage, and reports rule matches with policy reasons.
- **CLI frontend** (`crates/actplane-cli/`): provides the `actplane` command,
  project setup, policy review, MCP, and command dispatch.

## Build from source

`cargo install actplane` is all most users need. To hack on ActPlane:

```bash
git clone --recurse-submodules https://github.com/eunomia-bpf/ActPlane
cd ActPlane
cargo build --release -p actplane
```

Editing the kernel eBPF (`bpf/*.bpf.c`) requires the BPF toolchain
(clang/llvm, libelf, zlib) and the `libbpf`/`bpftool` submodules. Rebuild and
refresh the committed object with:

```bash
ACTPLANE_REBUILD_BPF=1 cargo build -p ebpf-ifc-engine   # regenerates bpf/prebuilt/process.bpf.o
```

Run the tests:

```bash
make test                          # bpf C unit tests + Rust workspace unit tests
sudo bash script/e2e_examples.sh   # live E1–E12 enforcement
```

## LICENSE

MIT License. See [LICENSE](https://github.com/eunomia-bpf/ActPlane/blob/master/LICENSE).

## Cite the Project

If ActPlane helps your research, please cite the arXiv preprint:

```bibtex
@misc{zheng2026actplane,
  title = {ActPlane: Programmable OS-Level Policy Enforcement for Agent Harnesses},
  author = {Yusheng Zheng and Tianyuan Wu and Quanzhi Fu and Tong Yu and Wenan Mao and Wei Wang and Dan Williams and Andi Quinn},
  year = {2026},
  eprint = {2606.25189},
  archivePrefix = {arXiv},
  primaryClass = {cs.OS},
  url = {https://arxiv.org/abs/2606.25189}
}
```
