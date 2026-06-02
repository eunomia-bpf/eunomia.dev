---
title: AgentSight Architecture
description: The data path from eBPF probes to analysis, storage, and visualization.
---

AgentSight is organized as a capture pipeline. eBPF programs collect low-level events, a Rust collector normalizes and analyzes them, and the frontend renders the session as process, timeline, log, and metric views.

## Pipeline

```text
Agent process
  -> eBPF probes
  -> JSON event stream
  -> Rust runners
  -> analyzer chain
  -> storage and web UI
```

## eBPF Capture

The eBPF layer observes activity below the agent framework:

- TLS read/write paths for supported user-space libraries.
- Process lifecycle events.
- File and command execution activity.
- System-level timing and resource signals.

This layer is useful because it can observe subprocesses and closed-source tools that do not emit application traces.

## Collector

The Rust collector runs the capture programs and turns raw events into a consistent stream. It is responsible for:

- Starting and stopping runners.
- Normalizing timestamps and process metadata.
- Parsing HTTP or LLM-related payloads where possible.
- Filtering or redacting events before output.
- Serving the local web interface.

## Analyzer Chain

Analyzers are small processing stages over the event stream. Typical analyzer work includes:

- HTTP parsing.
- Server-sent event chunk merging.
- Authentication header removal.
- Timestamp normalization.
- File logging or OpenTelemetry export.

This keeps capture code separate from interpretation logic, which makes it easier to add a new parser or output path without changing the eBPF programs.

## Frontend

The frontend reads the normalized event stream and presents several operational views:

- Process tree for parent/child process relationships.
- Timeline view for ordering requests, tool calls, file access, and command execution.
- Log view for raw event inspection.
- Resource metrics for CPU, memory, and runtime behavior.

## Limits

AgentSight is an observability tool. It can provide evidence about what happened, but enforcement should be implemented by a policy layer. It also depends on the target process, TLS library, kernel capabilities, and the availability of required eBPF hooks.
