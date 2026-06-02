---
title: AgentSight Visualization
description: How to read the AgentSight UI and connect events into an agent session.
---

AgentSight's UI is built for session investigation. Start from the process tree, then use the timeline and raw logs to inspect the exact events around a prompt, tool call, or subprocess execution.

## Process Tree

The process tree shows the agent process and its children. It is the best first view when you need to answer:

- Which command started the agent?
- Which subprocesses did the agent spawn?
- Did a tool call run in a shell, Python, Node.js, or another binary?
- Which process performed a file or network action?

![AgentSight process tree](https://github.com/eunomia-bpf/agentsight/raw/master/docs/demo-tree.png)

## Timeline

The timeline orders events by time. Use it to correlate:

- LLM request/response events.
- Tool calls and subprocess execution.
- File access before or after a model response.
- Network activity around a specific agent step.

![AgentSight timeline](https://github.com/eunomia-bpf/agentsight/raw/master/docs/demo-timeline.png)

## Logs

The log view exposes normalized and raw event payloads. It is useful when:

- A parsed event looks incomplete.
- You need the exact HTTP payload, command argument, or file path.
- You are debugging a new analyzer or filter.

## Metrics

The metrics view helps explain runtime behavior around an agent session, including CPU and memory changes while an agent reasons, calls tools, or spawns processes.

![AgentSight metrics](https://github.com/eunomia-bpf/agentsight/raw/master/docs/demo-metrics.png)

## Investigation Pattern

1. Find the root agent process in the process tree.
2. Expand child processes and identify tool execution.
3. Jump to the corresponding time range in the timeline.
4. Inspect raw log payloads for the events that need evidence.
5. Use metrics to explain performance or resource anomalies.
