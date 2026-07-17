---
date: 2026-07-21
slug: monitor-claude-code-ebpf
description: "Monitor Claude Code with zero code instrumentation using AgentSight and eBPF. See every prompt, tool call, file operation, and subprocess in real time."
---

# Monitor Claude Code with eBPF: Full Runtime Visibility

You gave Claude Code a task: refactor the authentication module. Fifteen minutes later it's done, the tests pass, and the diff looks clean. But between the prompt and the commit, what actually happened? How many API calls did it make? Which files did it read beyond the ones it changed? Did it spawn subprocesses you didn't expect?

Claude Code's built-in hooks let you intercept specific tool calls, but only the ones the agent framework exposes. If a subprocess shells out to `curl` or writes to a file through a child process, hooks never fire. Traditional observability tools like Langfuse, LangSmith, or OpenTelemetry-based solutions all require SDK integration into the application code, and Claude Code is a closed-source compiled binary. You can't instrument what you can't modify.

[AgentSight](https://github.com/eunomia-bpf/agentsight) takes a different approach. It uses eBPF to intercept TLS-encrypted LLM traffic and kernel-level system events from outside the process, with zero code changes. This tutorial walks through installing AgentSight, recording a Claude Code session, and exploring the captured data.

<!-- more -->

## Why Hooks and SDK-Based Tracing Fall Short

Claude Code provides a hooks system that runs shell commands before or after events like tool calls. Hooks work for policy enforcement ("block `git push`"), but they only see what the agent framework chooses to expose. A subprocess spawned by a tool call, a file opened through a child process, or an unexpected network connection all happen below the hook layer.

SDK-based observability tools take a different approach: they instrument the application code to emit traces and spans. Tools like Langfuse, SigNoz, and Arize Phoenix work well when you own the source and can add `@trace` decorators or callback handlers. Claude Code is a compiled Bun binary with BoringSSL statically linked and symbols stripped. Adding an SDK isn't an option.

| Approach | What it sees | Blind spot |
|----------|-------------|------------|
| **Claude Code hooks** | Tool calls the framework exposes | Subprocesses, file I/O, network calls below the hook layer |
| **SDK/OTel instrumentation** | Whatever the SDK wraps | Requires source access; can't instrument closed-source CLIs |
| **AgentSight (eBPF)** | All TLS traffic, process trees, file ops, network activity | Requires Linux with eBPF support and sudo for probes |

AgentSight operates at the kernel level. It attaches uprobes to the SSL library functions inside the Claude Code binary, captures plaintext request and response data at the `SSL_read`/`SSL_write` boundary, and correlates that traffic with process execution events from kernel tracepoints. The agent runs unmodified.

## Install AgentSight

AgentSight ships as a single binary. Install with Cargo or download the prebuilt release:

```bash
cargo install agentsight
# or download the release binary directly:
wget https://github.com/eunomia-bpf/agentsight/releases/latest/download/agentsight && chmod +x agentsight
```

**Prerequisites:** Linux kernel 4.1+ with eBPF support (5.0+ recommended) and sudo access for the eBPF probes.

## Record a Claude Code Session

Launch Claude Code under AgentSight with a single command:

```bash
sudo agentsight record -- claude
```

This one command handles everything automatically: it resolves the `claude` binary via `$PATH`, follows symlinks to the real executable (e.g. `claude` -> `~/.local/share/claude/versions/2.1.150`), discovers the statically linked BoringSSL library through byte-pattern matching, and attaches uprobes for SSL capture alongside process and system monitoring. Claude Code's TUI works normally in your terminal while AgentSight records in the background.

When you exit Claude Code, AgentSight saves the entire session to an `agentsight-*.db` SQLite file in your current directory.

## Explore the Captured Data

### Live Web UI

While recording, open [http://127.0.0.1:7395](http://127.0.0.1:7395) in your browser for a real-time dashboard with four views:

- **Timeline** (`/timeline`): every LLM call, process spawn, file operation, and network event on a single timeline
- **Process Tree** (`/tree`): the full process tree rooted at Claude Code, showing child processes, arguments, and exit status
- **Event Log** (`/logs`): raw event stream with filtering
- **Metrics** (`/metrics`): CPU and memory usage over time

### Command-Line Reports

After the session ends, query the saved database:

```bash
agentsight report                    # high-level session summary
agentsight report token              # token usage breakdown
agentsight report prompts --json     # full prompt/response payloads
agentsight report audit --json       # process spawns, file opens, API calls
agentsight report serve              # reopen the web UI for a saved session
```

### Live Top View

For ongoing monitoring across multiple agent sessions, `agentsight top` provides a ranked live view similar to Unix `top`:

```bash
sudo agentsight top
```

Sessions appear ranked by model, token usage, health, process family, tool calls, file activity, and network activity.

## How eBPF Makes This Work

Claude Code is a Bun application with BoringSSL compiled in and symbols stripped. There's no shared `libssl.so` for traditional interception tools to hook. AgentSight's `record` command handles this transparently: it detects the embedded BoringSSL via byte-pattern matching, attaches uprobes to the correct function offsets, and captures plaintext HTTP payloads before encryption.

Process monitoring uses kernel tracepoints to track every subprocess Claude Code spawns, including deeply nested child processes. File monitoring captures opens, writes, truncations, renames, and deletes. All of these kernel-level events are correlated with the LLM traffic timeline, so you can trace which API call triggered which system actions.

The overhead stays low: our evaluation reports less than 3% CPU overhead for typical traced agent workloads. The monitored agent runs unprivileged as your normal user; only the eBPF probes require sudo.

For a deep dive into how the BoringSSL interception works, see [Reverse-engineering Claude Code's SSL traffic with eBPF](https://eunomia.dev/blog/2026/02/13/reverse-engineering-claude-codes-ssl-traffic-with-ebpf/).

## FAQ

**Can I see what Claude Code is doing in real time?**
Yes. Run `sudo agentsight record -- claude` and open `http://127.0.0.1:7395/timeline`. You'll see prompts, model responses, tool calls, subprocess executions, and file operations as they happen.

**Does AgentSight slow Claude Code down?**
Our evaluation reports less than 3% CPU overhead for typical traced workloads. The eBPF probes operate at the kernel level and don't inject code into the agent process.

**Does AgentSight work with other AI coding agents?**
AgentSight works with any process that makes TLS-encrypted API calls. `record` auto-discovers binaries for Gemini CLI, Codex, Python-based agents (aider, open-interpreter), Node.js tools, and Docker-containerized agents. Run `sudo agentsight record -- <command>` for any agent.

**Can I export data to my existing observability stack?**
AgentSight exports captured LLM calls as OpenTelemetry GenAI (`gen_ai.*`) spans over OTLP/HTTP, so you can send agent telemetry to any OTel-compatible backend without in-process instrumentation:

```bash
sudo agentsight debug trace --otel --otel-endpoint http://localhost:4318
```

## What's Next

AgentSight turns Claude Code from a black box into an observable system, with zero changes to the agent itself. To go further:

- **[AgentSight on GitHub](https://github.com/eunomia-bpf/agentsight)**: installation, docs, and source
- **[Profiling AI agents with semantic flamegraphs](https://eunomia.dev/blog/2026/06/24/profiling-ai-agents-with-semantic-flamegraphs/)**: aggregate multi-session agent behavior into pprof-compatible flamegraphs
- **[AgentSight: keeping your AI agents under control](https://eunomia.dev/blog/2025/08/26/agentsight-keeping-your-ai-agents-under-control-with-ebpf-powered-system-observability/)**: the original announcement and architecture overview
