---
title: AgentSight
description: Zero-instrumentation observability for LLM and AI agent behavior with eBPF.
---

AgentSight observes LLM and AI agent activity from the system boundary. It connects process activity, TLS/network traffic, file access, tool execution, and resource behavior without adding SDKs to the agent application.

Use it when application logs are not enough: coding agents that spawn subprocesses, CLI agents that call remote LLM APIs, multi-agent workflows, or production pilots that need auditable evidence of what an agent actually did.

Project links:

- GitHub: <https://github.com/eunomia-bpf/agentsight>
- Paper: <https://arxiv.org/abs/2508.02736>
- Legacy page: [/GPTtrace/agentsight/](/GPTtrace/agentsight/)

## What It Captures

- LLM API requests and responses before TLS encryption, where supported by the traced library.
- Process trees, subprocess execution, command arguments, and lifecycle events.
- File reads, writes, and metadata activity relevant to agent behavior.
- Network and HTTP-level signals that can be correlated with process activity.
- Runtime metrics and event timelines for replaying an agent session.

## Why System-Level Observability

Application-level tracing sees what the application chooses to report. AgentSight runs below that layer. This makes it useful for closed-source tools, agent subprocesses, shell commands, and workflows where prompts, tools, or file access can bypass framework instrumentation.

AgentSight is not a replacement for application tracing. It is the ground-truth layer that complements traces from LangSmith, OpenTelemetry, proxy logs, or custom SDK instrumentation.

## Documentation

- [Quick start](quickstart.md): install the binary, record Claude Code, Gemini CLI, or Python-based agents, and open the UI.
- [Architecture](architecture.md): how eBPF probes, the Rust collector, analyzers, storage, and UI fit together.
- [Visualization](visualization.md): how to read the process tree, event timeline, log view, and metrics.
- [Operational notes](operational-notes.md): permissions, privacy, supported environments, and deployment guidance.
- [Troubleshooting](troubleshooting.md): common issues when events are missing or TLS hooks do not attach.

## Minimal Workflow

```bash
wget https://github.com/eunomia-bpf/agentsight/releases/download/v0.1.1/agentsight
chmod +x agentsight

# Record Claude Code.
sudo ./agentsight record -c claude

# Record Gemini CLI, which commonly runs as node.
sudo ./agentsight record -c node

# Record Python-based AI tools.
sudo ./agentsight record -c python
```

Then open <http://127.0.0.1:8080> to inspect the captured session.

## Current Scope

AgentSight focuses on Linux agent processes and common user-space TLS/process observability paths. It provides evidence and visibility; policy enforcement, sandboxing, and runtime guardrails should be handled by a dedicated control layer such as ActPlane or another security system.
