# How should a Linux-VM eBPF backend support macOS and Windows without mislabeling host coverage?

**Short answer:** run the existing Linux eBPF collector inside a Linux guest, keep host-native session discovery outside the guest, and merge both streams through the tool's existing event model. Label every event with its capture backend and execution boundary. A Lima or WSL 2 integration provides **Linux guest capture on a macOS or Windows machine**; it is not native eBPF visibility into the host kernel, host processes, or host user-space TLS.

This distinction is architectural, not cosmetic. Lima launches Linux virtual machines, while WSL 2 runs a Linux kernel inside a managed lightweight virtual machine. The eBPF program therefore sees the guest kernel and workloads that actually execute or pass traffic there. It cannot infer that a host process was observed merely because the guest runs on the same laptop.

## Define three kinds of support separately

A cross-platform observability tool should publish a capability matrix rather than one broad “supported” badge:

| Backend | What it can observe | What it must not claim |
| --- | --- | --- |
| Linux eBPF | Linux guest or host kernel events, subject to enabled hooks and privileges | Events from a different host kernel or process namespace |
| Host-native session source | Agent session files and APIs intentionally exposed by the host integration | Kernel provenance, syscall coverage, or decrypted traffic that was never captured |
| Platform-native telemetry | Only the hooks implemented by that operating system's backend | Linux hook parity or source compatibility without a verified mapping |

The distinction matters on Windows in particular. eBPF for Windows is a real native implementation, but its own documentation describes a Windows-specific hosting environment and says source compatibility is intended only for hooks and helpers that apply across operating systems. That is a separate backend from running Linux eBPF under WSL 2. One cannot silently substitute for the other.

On macOS, a Lima backend similarly gives the project a controlled Linux execution environment. It does not turn host processes into Linux tasks. Host-only context must come from a documented native source, or be reported as unavailable.

## Put the VM boundary behind the existing collector contract

For AgentSight, the current repository already exposes a useful seam. The reusable capture crate separates `sources/`, `runners/`, sinks, and a common event model. A first VM integration should preserve that contract:

```text
macOS or Windows host
├── native session source ───────────────┐
├── VM lifecycle and transport adapter  │
└── Linux VM                            │
    └── existing Linux eBPF runner ─────┤
                                        ↓
                         common collector/event model
                                        ↓
                              view, report, or sink
```

The host adapter should detect an explicitly configured Lima instance or WSL 2 distribution, start the existing Linux capture command inside it, and carry normalized events back over a narrow local transport. It should not fork the event schema for each operating system. Transport framing, reconnect behavior, and lifecycle state belong in the adapter; eBPF loading and Linux-specific feature checks remain in the guest runner.

Every event needs immutable provenance sufficient to prevent accidental blending. At minimum, record:

- `capture_backend`, such as `linux-ebpf`, `native-session`, or a future platform-native backend;
- the observed operating system and kernel, not merely the UI host;
- an execution-environment identifier that distinguishes host and guest;
- the source clock domain and collection timestamp; and
- a correlation key whose meaning is explicit.

Do not use a Linux guest PID as a host process identifier. PIDs, namespaces, paths, network interfaces, and clocks cross the VM boundary with different meanings. If a host-side agent session launches work inside the guest, create a correlation token at launch or transport time. If no trustworthy token exists, show two adjacent timelines rather than asserting a causal join.

## Make partial coverage visible

The most dangerous failure mode is a plausible but incomplete trace. The CLI and report should therefore say which backend ran and what it could see. Good states include:

- `linux-vm capture active` with the guest identity and enabled probes;
- `host session source active` with no kernel capture;
- `degraded` when the guest is reachable but a required hook or privilege is missing; and
- `unavailable` when the configured guest cannot start or the transport cannot authenticate.

Do not silently fall back from guest eBPF to host session files while keeping the same “capturing” status. A report assembled from session files can still be valuable, but its evidence class is different.

The same rule applies to network and TLS observations. A guest probe can observe traffic generated inside the guest and traffic deliberately routed through it. It does not automatically see host-local sockets, and it cannot recover plaintext that was encrypted in a host process before reaching the guest. File sharing and port forwarding make integration convenient, but they do not erase the capture boundary.

## Test boundaries before adding features

A small first implementation is easier to trust than a broad compatibility layer. Test the same focused capture command in three environments: native Linux, a Linux VM on macOS, and WSL 2. The acceptance tests should verify that:

1. the guest runs the unchanged Linux capture path;
2. events retain the same common schema while exposing different provenance;
3. host-native session context can correlate through an explicit token;
4. stopping or restarting the guest produces a visible lifecycle transition;
5. a host-only process is never reported as guest-observed; and
6. missing privileges, unsupported hooks, clock drift, and transport loss fail closed rather than creating a complete-looking trace.

Also test version skew. Record the host integration version, guest collector version, guest kernel, and event-schema version. Negotiate the schema before streaming; reject an incompatible major version instead of guessing. This keeps a VM image update from becoming an invisible data-model change.

## Treat the guest as a security boundary

An observability guest commonly receives shared files, forwarded ports, elevated BPF privileges, and sensitive agent events. Use the smallest required mounts and capabilities, authenticate the local transport, avoid exposing the collector beyond the host, and make retention explicit. Do not copy prompts, responses, credentials, or complete traffic payloads merely to improve correlation.

The practical roadmap is therefore incremental: first add an honest Linux-VM backend with explicit provenance and negative tests; then improve host-to-guest correlation; only later add truly native platform telemetry behind separate capability declarations. “Runs on macOS or Windows” is a packaging statement. “Observes the macOS or Windows host” is an evidence statement and requires a different backend.

## References

- [AgentSight issue: cross-platform support and the proposed Linux-VM capture boundary](https://github.com/eunomia-bpf/agentsight/issues/17)
- [AgentSight capture crate source layout](https://github.com/eunomia-bpf/agentsight/tree/master/agentsight-capture/src)
- [AgentSight README: Linux eBPF requirements and native-session fallback behavior](https://github.com/eunomia-bpf/agentsight)
- [Lima documentation: Linux virtual machines with file sharing and port forwarding](https://lima-vm.io/docs/)
- [Microsoft documentation: WSL 2 runs a Linux kernel in a lightweight utility VM](https://learn.microsoft.com/en-us/windows/wsl/about#what-is-wsl-2)
- [eBPF for Windows: architecture, hooks, helpers, and source-compatibility limits](https://github.com/microsoft/ebpf-for-windows)
- [`scx_lib_init_probe` fentry probe in the public sched-ext source](https://github.com/sched-ext/scx/blob/558aa09863e7bddb09101e4b242cc6efaee3dd5f/scheds/include/scx/common.bpf.h#L522-L540)
- [BPF mailing list: verifier diagnostics redesign](https://lore.kernel.org/bpf/178682023625.53386.10978136746024990805.git-patchwork-notify@kernel.org/T/#t)
- [OpenTelemetry Python contrib: security-only patch policy for deprecated GenAI instrumentations](https://github.com/open-telemetry/opentelemetry-python-contrib/pull/4955)
- [An in-kernel rolling-window design for measuring an eBPF cache](https://naveensrinivasan.com/posts/2026-08-02-measuring-an-ebpf-cache-without-leaving-the-kernel/)

## Community discussion today

Today's ordinary visible-browser review covered all 6 approved communities and all 15 allowlisted channels or public pages. Every target was accessible; one public forum's current interface presented a human-verification gate, so its ordinary visible legacy interface was used instead. The selected question came from the 24-hour window, so the seven-day fallback was not used. Names, accounts, employers, channel identities, message links, exact times, private topology, original logs, and searchable phrasing have been removed. No raw transcript was retained.

### Cross-platform support needs an evidence boundary

The strongest daily question asked where a contributor should connect a VM-based macOS or Windows prototype to an existing eBPF observability tool. The important result was not a platform-detection snippet; it was the separation between portable sources, Linux capture runners, and the shared event model. That seam permits a small adapter while preserving the meaning of existing events.

The broader concern was whether a Linux VM can be advertised as host support. It can provide a useful product experience: install on the host, execute probes in the guest, and correlate guest events with host-native session context. But the report must retain provenance. This is the difference between “the tool runs here” and “the tool observed this kernel.” Practitioners should demand negative tests showing that host-only work is not attributed to the guest.

### Attach failures still require isolating the layer that failed

A scheduler support discussion described a newer-kernel failure while attaching non-`struct_ops` BPF programs. Reproducing the failure by starting the scheduler directly ruled out the service loader. The remaining public code path includes a weak fentry probe attached to the scheduler-registration function, so the useful next step is a minimal version matrix and a public issue containing the first libbpf attach error, kernel version, scheduler version, architecture, and whether the same object loads with that optional probe disabled.

This incident was still unresolved during the review. The evidence supports narrowing the fault to program attachment, not claiming a kernel root cause. The reusable lesson is to remove orchestration layers one at a time, identify the exact BPF program and attach type, and preserve the earliest verifier or attach diagnostic instead of only the final aggregate error.

### Release policy and runtime compatibility remain separate

A GenAI instrumentation discussion focused on deprecated packages that remain eligible for security patches but no longer appear in the major/minor release workflow. The public release-policy change confirms that this asymmetry is intentional. It does not guarantee compatibility with future upstream SDK majors. Security-only maintenance, dependency constraints, and semantic-convention migration are three separate contracts; release automation and documentation must state each one explicitly.

Other project-specific areas were quiet or carried build notifications rather than new practitioner questions. The general eBPF discussion area likewise had no fresh daily troubleshooting thread, and its most recent socket-map topic had already been answered in an earlier Q&A.

### Upstream discussion emphasized diagnosability and low-overhead measurement

The public BPF archive was highly active. The most relevant series proposed structured verifier diagnostic categories, source and instruction context, and pruning diagnostics when the verifier abandons a path. That work reinforces the same operational principle seen in the scheduler incident: retain the earliest specific failure and its execution context instead of asking users to interpret a cascade of register-state consequences.

The newest public-forum technical post asked how to keep rolling usage metrics for an eBPF cache without continuously sending events to user space, adding shared locks, or using a general LRU. The design uses per-CPU state and bounded in-kernel buckets, trading exact global ordering for predictable overhead. Before adopting that pattern, measure update cost, aggregation error, bucket rollover behavior, memory per CPU, and what happens when CPUs are hot-plugged. Across today's topics, the common demand was honest scope: a VM backend must state where it observes, a loader must state which attachment failed, and a metric must state the approximation it introduces.
