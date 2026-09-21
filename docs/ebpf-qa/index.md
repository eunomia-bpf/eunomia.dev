# eBPF Q&A

Daily, source-grounded answers to recurring questions about eBPF, Linux
observability, profiling, runtime extension, and security. Each answer ends with
an anonymized summary of the wider technical discussion seen that day.

Questions may begin in public issues, mailing lists, forums, or technical chat
communities. Before publication, we rewrite them to remove identities, private
deployment details, and wording that could identify the original participant.
The answer is then checked against public primary sources. One question is
published each day; when part of the monitored window cannot be covered, the
gap is reported honestly on the page instead of stopping publication, and no
invented material is ever used to fill it.

## Latest Answers

- [For a Node.js service instrumented by OBI, why can't I just turn off the per-callback `async_hooks` trace-context sentinel because the service doesn't use manual spans or log enrichment?](/ebpf-qa/2026-09-21-nodejs-traces-ctx-sentinel-gating-consumers/)
- [How do you get observability out of a hosted agent harness when the agent loop runs in a vendor service rather than your process?](/ebpf-qa/2026-09-20-hosted-agent-harness-otel-export-boundary/)
- [Why does the OBI config migration from v1 to v2 refuse to write a file when some fields have no mapping, and what is the middle ground?](/ebpf-qa/2026-09-19-obi-config-migration-v1-v2-partial-fields/)
- [Why does an injected Node.js agent from an eBPF tracer stay inside a process after the service is excluded, until a pod restart?](/ebpf-qa/2026-09-18-nodejs-agent-lives-past-service-exclusion-until-pod-restart/)
- [Should GenAI agent spans nest each tool execution under the model call that requested it, or keep model calls and tool executions as siblings under one agent span?](/ebpf-qa/2026-09-17-genai-agent-invoke-agent-chat-execute-tool-sibling-tree/)
- [Why did enabling an eBPF tracer on a Node.js service triple p99 on the hot path when the eBPF probes are cheap?](/ebpf-qa/2026-09-16-nodejs-tracer-async-hooks-p99-cost/)
- [Can a BPF program safely read user-space memory that belongs to another process?](/ebpf-qa/2026-09-15-bpf-read-other-task-user-memory-zeros/)
- [Why can't a BPF program sleep or block, and what should you use instead in a non-sleepable context?](/ebpf-qa/2026-09-14-bpf-sleep-non-sleepable-context-alternatives/)
- [Why does a kprobe on a function never fire when the compiler inlined it?](/ebpf-qa/2026-09-13-kprobe-inlined-function-never-fires/)
- [When should I collect eBPF samples in a BPF ring buffer instead of a perf event array?](/ebpf-qa/2026-09-12-bpf-ringbuffer-vs-perf-event-array/)
- [Why can't SIGKILL stop a BPF program load after verification succeeds?](/ebpf-qa/2026-09-11-bpf-post-verification-rewrite-sigkill/)
- [What eBPF/kernel limitations should a process behavior reconstruction tool account for in its architecture?](/ebpf-qa/2026-09-10-process-behavior-reconstruction-event-loss/)
- [Which technical boundaries actually constrain malicious eBPF payloads in practice?](/ebpf-qa/2026-09-08-malicious-ebpf-payload-boundaries/)
- [Which failure scenarios are mandatory before a transient-TCP-recovery eBPF benchmark is meaningful?](/ebpf-qa/2026-09-06-tcp-recovery-benchmark-failure-scenarios/)
- [Why can the eBPF verifier lose a relationship that is true for the low 32 bits?](/ebpf-qa/2026-09-04-ebpf-verifier-low-32-bit-scalar-equality/)
- [Why can reverse-path filtering drop return traffic in an eBPF Kubernetes datapath?](/ebpf-qa/2026-09-03-ebpf-kubernetes-rp-filter-return-traffic/)
- [Why can't an eBPF program read `bpf_tail_call()`'s return value?](/ebpf-qa/2026-09-02-bpf-tail-call-return-value/)
- [Why are PID and TID insufficient to correlate concurrent TLS, HTTP/2, and SSE traffic?](/ebpf-qa/2026-08-31-tls-http2-sse-connection-correlation/)
- [Why must releasing a BPF dynptr invalidate every derived slice and clone?](/ebpf-qa/2026-08-29-bpf-dynptr-release-slice-clone-lifetime/)
- [Can an unprivileged container create its own BPF token?](/ebpf-qa/2026-08-28-unprivileged-container-bpf-token/)
- [Why can a single compare-and-swap lose atomic min/max updates under contention?](/ebpf-qa/2026-08-26-atomic-min-max-cas-contention/)
- [How can you start a `sched_ext` scheduler at boot without systemd?](/ebpf-qa/2026-08-25-sched-ext-boot-without-systemd/)
- [Why can a syscall-rewriting trampoline crash threads created by `clone` or `clone3`?](/ebpf-qa/2026-08-24-clone-syscall-trampoline-child-stack/)
- [Should an OpenTelemetry GenAI evaluation result carry a verifiable-evidence reference?](/ebpf-qa/2026-08-23-opentelemetry-genai-evaluation-evidence-reference/)
- [Why can an unused OBI Prometheus endpoint keep growing memory?](/ebpf-qa/2026-08-22-obi-unused-prometheus-exporter-memory-growth/)
- [Should an OpenTelemetry metrics producer copy service identity into every data point?](/ebpf-qa/2026-08-21-opentelemetry-resource-attributes-prometheus-labels/)
- [How can you tell whether an OpenTelemetry GenAI attribute is stable enough to depend on?](/ebpf-qa/2026-08-20-opentelemetry-genai-attribute-stability/)
- [Why can libbpf load a BPF object slowly on a host with many kernel modules?](/ebpf-qa/2026-08-19-libbpf-selective-kmod-btf-loading/)
- [Does cgroup v2 `cpu.max` still limit CPU time under BPF extensible scheduling?](/ebpf-qa/2026-08-18-sched-ext-cgroup-cpu-max/)
- [Why can classic uprobe BPF programs crash preemptible kernels that use private BPF stacks?](/ebpf-qa/2026-08-17-uprobe-private-stack-preemption-crash/)
- [How should eBPF programs carry per-packet metadata across networking hooks?](/ebpf-qa/2026-08-16-cross-hook-packet-metadata/)
- [How should a Linux-VM eBPF backend support macOS and Windows without mislabeling host coverage?](/ebpf-qa/2026-08-15-cross-platform-ebpf-linux-vm-backend/)
- [How should OpenInference coexist with OpenTelemetry's GenAI semantic conventions?](/ebpf-qa/2026-08-13-openinference-opentelemetry-genai/)
- [Why can `scxctl` accept a scheduler switch while the service still does not start as intended?](/ebpf-qa/2026-08-11-scxctl-scheduler-arguments/)
- [Why can inserting a socket into `SOCKHASH` from TC egress soft-lock the kernel?](/ebpf-qa/2026-08-10-tc-egress-sockhash-soft-lock/)
- [Why can a `sched_ext` scheduler fail until `pahole` is upgraded?](/ebpf-qa/2026-08-09-sched-ext-pahole-version/)
- [Can eBPF detect secrets in network traffic without collecting them?](/ebpf-qa/2026-08-08-detect-secrets-with-ebpf/)
