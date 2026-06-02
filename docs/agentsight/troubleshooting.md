---
title: AgentSight Troubleshooting
description: Common fixes for missing events, TLS hook failures, and UI issues.
---

This page covers the most common issues when running AgentSight locally.

## The Target Process Does Not Appear

Check that the command name matches the actual process name:

```bash
ps -eo pid,comm,args | grep -E 'claude|node|python'
```

Then pass the value from the `comm` column:

```bash
sudo ./agentsight record -c node
```

## TLS Events Are Missing

Possible causes:

- The target process uses a different TLS library than the attached probe expects.
- The runtime bundles OpenSSL statically.
- The process is not the one performing the network request.
- Traffic is routed through a separate proxy process.

For Node.js with a known binary path, try:

```bash
sudo ./agentsight record --binary-path /usr/bin/node -c node
```

## Permission Errors

Run AgentSight with root privileges and confirm that the host allows eBPF program loading. Some locked-down containers, kernels, and security profiles block the required operations.

## The UI Does Not Load

Confirm that the local server is running and that port `8080` is not already in use:

```bash
ss -ltnp | grep 8080
```

If another process owns the port, stop it or configure AgentSight to use another port if the current build supports that option.

## Events Are Too Noisy

Narrow the target process and add analyzer filters where possible. For sensitive sessions, reduce capture scope before recording instead of relying only on after-the-fact redaction.

## Data Looks Out of Order

Use normalized timestamps from the collector when comparing events. Raw kernel, userspace, and frontend timestamps can differ depending on buffering and clock source.
