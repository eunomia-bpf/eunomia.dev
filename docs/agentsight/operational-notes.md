---
title: AgentSight Operational Notes
description: Permissions, privacy, supported environments, and deployment guidance for AgentSight.
---

AgentSight captures sensitive runtime data. Treat its output as audit evidence and handle it with the same care as logs containing prompts, API payloads, file paths, and command arguments.

## Permissions

AgentSight needs privileges to load eBPF programs. In local development this usually means running with `sudo`. In production-like environments, run it only on hosts where the operator is allowed to capture process and network metadata.

## Privacy and Redaction

Captured data can include:

- Prompt and response text.
- HTTP headers and request metadata.
- File paths and command arguments.
- Tool outputs and subprocess behavior.

Use analyzer filters to remove authentication headers and avoid storing data that is not needed for the investigation. When sharing traces, redact prompts, secrets, tokens, and private file paths.

## Environment Fit

AgentSight is most useful on Linux hosts where agent processes run directly or in containers with enough privileges for eBPF capture. It is less suitable when:

- The host does not allow eBPF program loading.
- The target process uses unsupported TLS paths.
- All traffic is hidden behind a separate proxy process that is not being traced.
- The operator cannot legally or organizationally capture prompt payloads.

## Deployment Shape

For development and incident reproduction, run AgentSight locally around one agent command. For pilot deployments, scope the capture to a small set of hosts or processes, define a retention policy, and decide which fields should be redacted before storage.

## Evidence Handling

For audit or incident use:

1. Record the command used to start AgentSight.
2. Record the target process name and binary path.
3. Preserve timestamps and host metadata.
4. Keep raw event logs when parser output is disputed.
5. Store the trace in a location with access control.
