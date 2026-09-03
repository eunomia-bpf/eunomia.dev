# eBPF Q&A

Daily, source-grounded answers to recurring questions about eBPF, Linux
observability, profiling, runtime extension, and security. Each answer ends with
an anonymized summary of the wider technical discussion seen that day.

Questions may begin in public issues, mailing lists, forums, or technical chat
communities. Before publication, we rewrite them to remove identities, private
deployment details, and wording that could identify the original participant.
The answer is then checked against public primary sources. One question is
published after each successful daily review; access or evidence failures are
reported as failures rather than filled with invented material.

## Latest Answers

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
