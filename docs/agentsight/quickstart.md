---
title: AgentSight Quick Start
description: Install AgentSight, record an agent process, and inspect the session in the local UI.
---

This guide records one local agent process and opens the built-in web UI.

## Prerequisites

- Linux with eBPF support. Kernel 5.x or newer is recommended.
- Root privileges for loading eBPF programs.
- A target process such as `claude`, `node`, or `python`.
- If building from source: Rust, Node.js, clang/LLVM, and libelf development headers.

## Install a Release Binary

```bash
wget https://github.com/eunomia-bpf/agentsight/releases/download/v0.1.1/agentsight
chmod +x agentsight
```

## Record an Agent

Choose the process command name used by the agent.

```bash
# Claude Code.
sudo ./agentsight record -c claude

# Gemini CLI often appears as node.
sudo ./agentsight record -c node

# Python-based agent applications.
sudo ./agentsight record -c python
```

When the agent uses a Node.js binary with statically bundled OpenSSL, pass the binary path explicitly:

```bash
sudo ./agentsight record --binary-path /usr/bin/node -c node
```

## Open the UI

After recording starts, open:

```text
http://127.0.0.1:8080
```

The UI shows the process tree, event timeline, logs, and resource metrics for the captured session.

## What to Check First

1. Confirm the target process appears in the process tree.
2. Inspect child processes and command execution under the agent.
3. Open timeline events around each LLM request or tool invocation.
4. Use the log view for raw event payloads when debugging parser behavior.

## Stop Recording

Use `Ctrl-C` in the terminal running AgentSight. If eBPF programs remain attached after an interrupted run, restart AgentSight cleanly or unload stale probes with the project cleanup scripts from the repository.
