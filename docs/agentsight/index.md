# AgentSight: System-wide AI agent tracing and monitoring with eBPF

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/eunomia-bpf/agentsight/actions/workflows/ci.yml/badge.svg)](https://github.com/eunomia-bpf/agentsight/actions/workflows/ci.yml)
[![arXiv:2508.02736](https://img.shields.io/badge/arXiv-2508.02736-b31b1b.svg)](https://arxiv.org/abs/2508.02736)
[![DOI:10.1145/3766882.3767169](https://img.shields.io/badge/DOI-10.1145%2F3766882.3767169-blue.svg)](https://dl.acm.org/doi/10.1145/3766882.3767169)

**English** | [中文](https://github.com/eunomia-bpf/agentsight/blob/master/README.zh-CN.md)

Your local first `perf`/`top`/`strace`/`nsight`-like tool for AI agents. See what agents actually do
to your machine, and connect those actions back to the prompts, model calls, and
tool decisions that triggered them.

Run `agentsight` around Claude Code, Codex, Gemini CLI,
OpenCode, OpenClaw, or any command. AgentSight records a local trace of:

- processes and child processes, shell commands, cwd, argv, exit status, and duration
- files created, written, truncated, renamed, or deleted
- network destinations contacted
- prompts, responses, tool intent, and LLM/model/token

No SDK, no proxy, no vendor integration. AgentSight observes with eBPF and TLS traffic tracing, so it works even when the agent is a
closed-source CLI. **✨ Zero Instrumentation Required**

## Quick Start

```bash
cargo install agentsight
# or: wget https://github.com/eunomia-bpf/agentsight/releases/latest/download/agentsight && chmod +x agentsight
sudo agentsight top
```

<div align="center">
  <img src="docs/top-mode-demo.png" alt="AgentSight top live session view" width="1000">
  <p><em>Live sessions ranked by model, session tokens, health, process family, tool calls, file activity, and network activity</em></p>
</div>

If you downloaded the binary into the current directory, run `sudo ./agentsight top`.
`top` loads eBPF probes, discovers local agents, and connects system activity to
agent behavior in real time. See the [Usage](#usage) section for more examples
and details.

## 🚀 Why AgentSight?

### Traditional Observability vs. System-Level Monitoring

Application-level tools such as [LangSmith](https://docs.langchain.com/langsmith/observability-concepts), [Langfuse](https://langfuse.com/docs/observability/overview), and [Phoenix](https://arize.com/docs/phoenix/) are great for traces, prompts, tokens, evals, and latency when you own the application code. Gateway/proxy tools such as [Helicone](https://docs.helicone.ai/getting-started/integration-method/gateway) are useful when you can route provider traffic through a managed endpoint.

AgentSight focuses on the layer those tools often miss: what the agent actually does at the system boundary. It observes existing binaries and CLI agents without SDKs or proxies, then correlates LLM traffic with process execution, file access, and system activity.

| **Challenge** | **Application-Level Tools** | **AgentSight Solution** |
|---------------|----------------------------|------------------------|
| **Framework Adoption** | ❌ SDK, callback, or gateway integration per app | ✅ Drop-in system tracer, no code changes |
| **Closed-Source CLIs** | ❌ Limited to what the tool exposes or logs | ✅ Observes existing binaries and CLI agents from outside |
| **Agent-Controlled Logs** | ❌ Logs can be incomplete, disabled, or modified | ✅ Kernel-level events independent of app logging |
| **TLS LLM Traffic** | ❌ Visible when routed through SDKs/proxies | ✅ Captures plaintext at SSL/TLS calls without a proxy |
| **System Actions** | ❌ Often misses subprocesses and local file activity | ✅ Tracks process execution, file access, and resource use |
| **Cross-Boundary Behavior** | ❌ Traces usually stop at framework/process boundaries | ✅ Correlates LLM traffic with process and file events |

AgentSight captures critical interactions that application-level tools miss:

- Subprocess executions that bypass instrumentation
- Plaintext LLM payloads at SSL/TLS call boundaries
- File operations and system resource access  
- Cross-boundary behavior across LLM, process, and file events

## Usage

### Prerequisites

- **Linux kernel**: 4.1+ with eBPF support (5.0+ recommended)
- **sudo access**: eBPF probes are auto-elevated; your agent stays unprivileged

For source builds, see [docs/build.md](https://github.com/eunomia-bpf/agentsight/blob/master/docs/build.md).

### Installation

#### Cargo or Release Binary

For local use, install with `cargo install agentsight` or download the latest
release binary, then start with `sudo agentsight top`. Use the examples below when
you want to record a specific command or inspect saved sessions.

#### Docker

Docker is useful for container, CI, or isolated Linux environments, but it still needs privileged host access for eBPF. See [docs/docker.md](https://github.com/eunomia-bpf/agentsight/blob/master/docs/docker.md).

#### Build from Source

Build requirements and source build commands live in [docs/build.md](https://github.com/eunomia-bpf/agentsight/blob/master/docs/build.md).

### Querying Past Sessions

Every `stat -- <command>` or `record` session is automatically saved to SQLite. Start with the perf-style commands, then use `agentsight db` for structured queries:

```bash
agentsight stat                       # counters for the latest saved session
sudo agentsight top                   # live ranked view of current agent sessions
agentsight top --db run.db --once     # ranked view of a saved session
sudo agentsight record -- claude      # record a command
agentsight report                     # high-level run summary
agentsight list                       # all recorded sessions
agentsight prompts --json             # full LLM request/response JSON
agentsight db token                   # token usage (auto-finds latest session)
agentsight db audit --json            # process spawns, file opens, API calls
agentsight db export -o snapshot.json # export for web dashboard
```

### Web Interface

During a session, visit [http://127.0.0.1:7395](http://127.0.0.1:7395) for live traffic, process trees, and metrics:
- **Timeline View**: http://127.0.0.1:7395/timeline
- **Process Tree**: http://127.0.0.1:7395/tree
- **Raw Logs**: http://127.0.0.1:7395/logs

<div align="center">
  <img src="https://github.com/eunomia-bpf/agentsight/raw/master/docs/demo-tree.png" alt="AgentSight Demo - Process Tree Visualization" width="800">
  <p><em>Process tree visualization for agent subprocesses and file activity</em></p>
</div>

<div align="center">
  <img src="https://github.com/eunomia-bpf/agentsight/raw/master/docs/demo-timeline.png" alt="AgentSight Demo - Timeline Visualization" width="800">
  <p><em>Timeline visualization for LLM, process, file, and network events</em></p>
</div>

<div align="center">
  <img src="https://github.com/eunomia-bpf/agentsight/raw/master/docs/demo-metrics.png" alt="AgentSight Demo - Metrics Visualization" width="800">
  <p><em>Metrics visualization for memory and CPU usage</em></p>
</div>

<div align="center">
  <p><strong>Try the <a href="https://agentsight.us">live demo</a></strong> to explore a real recorded Claude Code session in the browser.</p>
</div>

### Agent Discovery and Adapters

> **Privileges:** eBPF probes need root. Use `sudo` for live capture commands.
> AgentSight can auto-elevate if you forget, but that is a fallback. Your agent
> still runs as your normal user.

**Discover what agents are installed locally:**

```bash
./agentsight discover
```

**Attach to a running agent with `record`:**

```bash
sudo ./agentsight record -c claude
sudo ./agentsight record -c python
sudo ./agentsight record -c node --binary-path docker://openclaw
```

Built-in SQL adapters cover Anthropic, Claude Code, Gemini CLI, and OpenClaw sessions. Use `--no-adapters` to disable, or `agentsight db adapters list --json` to inspect.

### Usage Examples

#### Zero-Config: `record`

`record` is the simplest way to trace an agent. Put the command you want to run
after `record --`; AgentSight handles everything else:

```bash
# Launch and trace Claude Code — no --binary-path or --comm needed
sudo ./agentsight record -- claude

# Works for any agent: pass the command exactly as you'd normally run it
sudo ./agentsight record -- claude -p "review my last commit"
sudo ./agentsight record -- python my_agent.py
sudo ./agentsight record -- node ./cli.js
```

What `record -- <command>` does automatically:

1. **Discovers the SSL binary** — resolves the command via `$PATH`, follows
   symlinks (e.g. `claude` → `~/.local/share/claude/versions/2.1.150`), and
   chases shebang wrappers (e.g. a `#!/usr/bin/env node` script → the real
   `node` ELF) so uprobes attach to the correct executable.
2. **Derives the `--comm` process filter** from the command name.
3. **Launches the agent** with your terminal attached (its TUI/REPL works
   normally) while SSL + process + system monitoring runs quietly in the
   background.
4. **Stops automatically** when the agent process exits.

> **`sudo` note**: under `sudo`, `record` still finds *your* user-local installs
> (it reads `$SUDO_USER`'s home for `~/.local/bin`, `~/bin`, and `~/.nvm`), so
> `sudo ./agentsight record -- claude` traces the claude in your home directory,
> not a different one on root's `$PATH`.

Useful flags: `--binary-path <path>` to override auto-discovery, `--no-server`
to disable the web UI, `--server-port <port>`, `-o <log-file>`.

#### Monitoring Claude Code

Claude Code is a Bun-based application with BoringSSL statically linked and
symbols stripped. AgentSight auto-detects BoringSSL functions via byte-pattern
matching when `--binary-path` is provided:

```bash
# Find the Claude binary version
CLAUDE_BIN=~/.local/share/claude/versions/$(claude --version | head -1)

# Record all Claude activity with web UI
sudo ./agentsight record -c claude --binary-path "$CLAUDE_BIN"
# Open http://127.0.0.1:7395 to view timeline

# Advanced: full trace with custom filters
sudo ./agentsight debug trace --ssl true --process true --comm claude \
  --binary-path "$CLAUDE_BIN" --server true --server-port 8080
```

This captures:
- **Conversation API**: `POST /v1/messages` requests with full prompt/response SSE streaming
- **Telemetry**: heartbeat, event logging, Datadog logs
- **Process activity**: file operations, subprocess executions

> **Note**: All SSL traffic in Claude flows through an internal "HTTP Client"
> thread, not the main "claude" thread. When `--binary-path` is specified,
> the `--comm` filter is automatically skipped for SSL monitoring (but still
> applied for process monitoring) to ensure traffic is captured correctly.

#### Monitoring Python AI Tools

```bash
# Monitor aider, open-interpreter, or any Python-based AI tool
sudo ./agentsight record -c "python"

# Custom port and log file
sudo ./agentsight record -c "python" --server-port 8080 --log-file /tmp/agent.log
```

#### Monitoring Node.js AI Tools (Gemini CLI, etc.)

> **Important**: Node.js (both NVM and system installs) **statically links
> OpenSSL into the `node` binary** — there is no system `libssl.so` to hook.
> SSL capture therefore requires pointing sslsniff at the `node` binary itself.

The easiest way is `record -- <command>`, which discovers the `node` binary automatically:

```bash
# Gemini CLI runs on Node — record finds the right binary and traces it
sudo ./agentsight record -- gemini
```

With `record`, AgentSight now auto-discovers the Node binary from `-c node`
(it detects that Node embeds OpenSSL and attaches to the binary instead of a
system library), so this just works without `--binary-path`:

```bash
# Monitor Gemini CLI or other Node.js AI tools — binary auto-discovered
sudo ./agentsight record -c node

# Pin the binary explicitly if auto-discovery picks the wrong Node install
sudo ./agentsight record -c node --binary-path ~/.nvm/versions/node/v20.0.0/bin/node
```

> **Behind an HTTP/HTTPS proxy?** Traffic is still TLS-encrypted inside the
> Node process (the proxy only tunnels it), so AgentSight captures it the same
> way — at the `SSL_read`/`SSL_write` calls before encryption.

#### Monitoring Agents in Docker Containers (OpenClaw, etc.)

For an agent running inside a Docker container, pass the container to
`--binary-path` with the `docker://` scheme. AgentSight resolves the container's
process tree and attaches sslsniff to the right binary automatically:

```bash
# OpenClaw is a Node.js agent that runs in a container — works out of the box
sudo ./agentsight record -c node --binary-path docker://openclaw

# Accepts a container name or ID; supported by record / trace / ssl
sudo ./agentsight debug trace --binary-path docker://openclaw --server
```

`docker inspect` reports the container's *init* process (often `tini`), which
has no SSL code. AgentSight walks the descendant process tree and attaches to the
first process whose binary actually embeds SSL (the `node` process). See
[docs/openclaw.md](https://github.com/eunomia-bpf/agentsight/blob/master/docs/experiment/openclaw.md) for the full walkthrough.

#### Advanced Monitoring

```bash
# Combined SSL and process monitoring with web interface
sudo ./agentsight debug trace --ssl true --process true --server true

# Custom port and log file
sudo ./agentsight record -c "python" --server-port 8080 --log-file /tmp/agent.log
```

#### Export to OpenTelemetry (GenAI semantic conventions)

AgentSight can export captured LLM calls as OpenTelemetry **GenAI**
(`gen_ai.*`) spans over OTLP/HTTP — standards-compliant agent telemetry for any
agent, with zero in-process instrumentation. Send them to an OpenTelemetry
Collector and on to Jaeger, Grafana Tempo, Datadog, Honeycomb, etc.

```bash
# Export gen_ai.* spans to a collector (defaults to http://localhost:4318)
sudo ./agentsight debug trace --otel --otel-endpoint http://localhost:4318

# Include prompt/completion content (opt-in; off by default for privacy)
sudo ./agentsight debug trace --otel --otel-capture-content
```

Each LLM request/response pair becomes a `chat {model}` span with
`gen_ai.provider.name`, `gen_ai.request.model`, `gen_ai.usage.{input,output}_tokens`,
`gen_ai.response.finish_reasons`, and more. See [docs/otel.md](https://github.com/eunomia-bpf/agentsight/blob/master/docs/otel.md) for
collector setup and backend integration.

#### Browser Plaintext Capture

For browser-specific plaintext capture, use the standalone `browsertrace` BPF
tool instead of `sslsniff`:

```bash
# Chrome / Chromium
sudo ./bpf/browsertrace --binary-path /opt/google/chrome/chrome

# Firefox on Ubuntu Snap
sudo ./bpf/browsertrace --binary-path /snap/firefox/current/usr/lib/firefox/firefox
```

> **Note**: On Ubuntu, `/usr/bin/firefox` is often a wrapper script rather than
> the real browser ELF. Point `browsertrace` at the actual Firefox binary.

#### Local MCP over stdio

For local MCP servers that communicate over `stdio` instead of HTTP/TLS, use
the standalone `stdiocap` BPF tool:

```bash
# Capture stdin/stdout/stderr payloads for a local MCP server process
sudo ./bpf/stdiocap -p <mcp_server_pid>
```

AgentSight also includes a minimal MCP fixture for local testing under
[`docs/mcp-test/README.md`](https://github.com/eunomia-bpf/agentsight/blob/master/docs/experiment/mcp-test/README.md). It provides both `stdio`
and HTTP test modes so you can generate predictable MCP traffic before wiring
it into the Rust collector.

#### Direct eBPF Program Usage

```bash
# Run sslsniff directly on Claude binary
sudo ./bpf/sslsniff --binary-path ~/.local/share/claude/versions/2.1.39

# Run sslsniff on NVM Node.js
sudo ./bpf/sslsniff --binary-path ~/.nvm/versions/node/v20.0.0/bin/node --verbose

# Run browsertrace directly on Chrome
sudo ./bpf/browsertrace --binary-path /opt/google/chrome/chrome

# Run stdiocap directly on a local MCP server PID
sudo ./bpf/stdiocap -p 12345

# Run process tracer
sudo ./bpf/process -c python
```

## ❓ Frequently Asked Questions

**Q: What permissions does AgentSight need?**
A: eBPF probes need root privileges, so AgentSight may prompt for `sudo`. With `record -- <command>` or `stat -- <command>`, the monitored agent still runs as your normal user; only the probes are elevated.

**Q: What's the performance impact?**
A: Our evaluation reports less than 3% CPU overhead for typical traced agent workloads.

**Q: Where does captured data go?**
A: `record` and `stat -- <command>` store sessions locally in SQLite by default. Use `agentsight stat`, `agentsight top`, `agentsight report`, `agentsight list`, `agentsight db audit --json`, and `agentsight db token` to inspect prior runs. Captured data can include prompts, responses, paths, headers, and network targets, so treat logs and DBs as sensitive.

**Q: Why doesn't AgentSight capture traffic from Claude Code, Node.js, or Gemini CLI?**
A: These applications statically link their SSL library (BoringSSL for Claude/Bun, OpenSSL for **all** Node.js — both NVM and system installs) into their own binary instead of using system `libssl.so`, so there's nothing for sslsniff to hook by default. AgentSight handles this for you: `record -- <command>` always discovers the binary, and `record -c node` now auto-discovers the Node binary too. For Claude attach mode, pass `--binary-path`. See the "Zero-Config: record" and "Monitoring Node.js AI Tools" sections.

**Q: What should I check if tracing fails?**
A: Verify you are on Linux with eBPF support, have `sudo` or `CAP_BPF`/`CAP_SYS_ADMIN`, and are using `record -- <command>` or the correct `--binary-path` for statically linked agents.

## 🤝 Contributing

We welcome contributions! After cloning and building (see [docs/build.md](https://github.com/eunomia-bpf/agentsight/blob/master/docs/build.md)), you can:

```bash
# Run tests
make test

# Frontend development server
cd frontend && npm run dev

# Build debug versions with AddressSanitizer
make -C bpf debug
```

### Key Resources

- [CLAUDE.md](https://github.com/eunomia-bpf/agentsight/blob/master/CLAUDE.md) - Project guidelines and architecture
- [docs/design/README.md](https://github.com/eunomia-bpf/agentsight/blob/master/docs/design/README.md) - archived design notes and research drafts

## 📄 License

MIT License - see [LICENSE](https://github.com/eunomia-bpf/agentsight/blob/master/LICENSE) for details.

---

**💡 The Future of AI Observability**: As AI agents become more autonomous and capable of self-modification, traditional observability approaches become insufficient. AgentSight provides independent, system-level monitoring for safe AI deployment at scale.
