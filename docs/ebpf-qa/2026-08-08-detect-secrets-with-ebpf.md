# Can eBPF detect secrets in network traffic without collecting them?

**Short answer:** yes, but only where plaintext is visible, and the detector must be designed so that raw values never leave that point.

An eBPF program attached at TC, XDP, or to a packet socket sees the bytes carried on the wire. For HTTPS, those bytes are normally ciphertext, so the program can classify connections and measure flows but cannot recognize an API key inside the encrypted request. OpenTelemetry eBPF Instrumentation (OBI) reflects this boundary by separating network observability from application observability. Its documentation also notes that TLS requests rely on user-space uprobes, which may require additional privileges, rather than ordinary packet inspection alone.

At a plaintext boundary, such as a supported TLS library before encryption or after decryption, an eBPF-based sensor can inspect enough data to classify a possible credential. That still means the sensor can see sensitive bytes. Privacy comes from what it does next: match locally, emit only a category or counter, and discard the value instead of sending payloads through a ring buffer, log, trace, or collector.

## How to verify the design

Test the whole data path, not only the eBPF program:

1. Send a synthetic credential through an encrypted test request.
2. Confirm that packet-level hooks see ciphertext and that the plaintext-aware hook emits only the expected classification.
3. Inspect BPF maps, ring-buffer events, debug logs, traces, and exported telemetry for the synthetic value.
4. Repeat with fragmented requests, retries, unsupported TLS libraries, and non-TLS traffic so an apparent privacy guarantee does not depend on one happy path.

Downstream redaction remains useful as a second layer, but it solves a different problem. For example, Grafana Alloy's `loki.secretfilter` redacts secrets from log entries before forwarding them to Loki; it does not make an upstream payload capture private.

The practical rule is simple: classify as close as possible to the plaintext boundary, export the smallest non-sensitive result that answers the operational question, and verify that no raw value survives elsewhere in the pipeline. If the required TLS boundary is unsupported, fall back to metadata-based detection or application instrumentation rather than capturing broad payloads.

## References

- [OpenTelemetry eBPF Instrumentation: security and operation modes](https://opentelemetry.io/docs/zero-code/obi/security/)
- [OpenTelemetry eBPF Instrumentation: troubleshooting TLS visibility](https://opentelemetry.io/docs/zero-code/obi/troubleshooting/)
- [Grafana Alloy `loki.secretfilter`](https://grafana.com/docs/alloy/latest/reference/components/loki/loki.secretfilter/)
- [Linux kernel documentation: libbpf application lifecycle and CO-RE](https://docs.kernel.org/bpf/libbpf/libbpf_overview.html)
- [Linux kernel documentation: `BPF_MAP_TYPE_SOCKMAP` and `BPF_MAP_TYPE_SOCKHASH`](https://docs.kernel.org/bpf/map_sockmap.html)
- [Linux kernel documentation: `sched_ext`](https://docs.kernel.org/scheduler/sched-ext.html)
- [Linux kernel documentation: BPF Type Format](https://docs.kernel.org/bpf/btf.html)
- [OpenTelemetry: code-based and zero-code instrumentation](https://opentelemetry.io/docs/concepts/instrumentation/)
- [OpenTelemetry: Go compile-time instrumentation](https://opentelemetry.io/docs/zero-code/go/compile-time/)

## Community discussion today

Today's visible review covered seven approved public channels across four technical communities. Names, channel identities, message links, timestamps, and deployment-specific details have been removed from this summary.

### Detecting secrets without exporting them

The most concentrated question was how to identify credentials moving between services without turning the observability system into another store of sensitive data. The decisive issue is the observation point. Packet hooks can describe encrypted flows but cannot inspect an HTTPS credential; a supported user-space TLS hook can see plaintext and must therefore be treated as sensitive code. A workable design performs classification at that boundary, exports a category or counter rather than the matched bytes, and then searches maps, ring buffers, logs, traces, and collector output for a synthetic test value. The OBI security and TLS troubleshooting documentation describes the visibility and privilege boundary, while the `loki.secretfilter` documentation is useful for understanding why downstream redaction remains only a second layer.

### Recovering a dataplane after its loader exits

Another discussion asked what an eBPF dataplane controller should do when its user-space loader restarts while kernel objects may still be alive. The recovery path should begin by treating the contents of `bpffs` as observed state, not proof that the desired deployment is healthy. Reopen the pinned maps, programs, and links; compare their identifiers, map schemas, attachment points, and expected version with the controller's desired state; reuse compatible objects; and replace stale or partial state in a controlled order. The libbpf lifecycle documentation explains the open, load, attach, and teardown phases, which makes a useful checklist for this reconciliation.

The control loop also needs a concurrency boundary. BPF syscalls, link attachment, and cleanup can block or fail independently of an asynchronous event loop, so they belong in a bounded worker path with explicit timeouts, cancellation, and idempotent retries. Health should be based on a successful reconciliation and a functioning data path, not merely on the presence of files under `bpffs`. What remains workload-specific is whether maps preserve valuable live state across an upgrade; that decision requires a compatibility rule for each map rather than a blanket reload policy.

### A `SOCKHASH` hang from the TC path

A separate networking question concerned a hang observed while adding an established socket to `SOCKHASH` from a TC program. The first diagnostic step is to reduce the setup to the upstream sockmap selftests and verify the exact kernel version, program type, attach type, socket state, and whether another parser or verdict program is already inherited by the socket. Inserting a socket changes its callbacks and attaches a `sk_psock`; the kernel documentation also notes that conflicting parser or verdict programs can make the update fail. Those lifecycle effects make a generic TC packet path a poor place to assume that any observed socket is immediately safe to enroll.

The supported insertion paths should drive the design: user space can add a socket by file descriptor, while a `BPF_PROG_TYPE_SOCK_OPS` program can use `bpf_sock_hash_update()` with the socket context. Stream processing then belongs to the documented `SK_MSG` or `SK_SKB` verdict hooks. If the minimal selftest-compatible arrangement still hangs instead of returning an error, preserve the reproducer, collect the kernel stack and verifier or trace output, and report it as a kernel regression. The public sockmap documentation and linked selftests provide the reference behavior without relying on the original community deployment.

### Matching `sched_ext` to the running kernel

The scheduler discussion began with a `sched_ext` object that could not load because the expected scheduler kfunc types did not match the running kernel's BTF. Rebuilding the same source or toggling runtime flags does not answer that mismatch. A useful comparison starts with the running kernel configuration, especially `CONFIG_SCHED_CLASS_EXT` and `CONFIG_DEBUG_INFO_BTF`, then dumps `/sys/kernel/btf/vmlinux` and checks whether the scheduler object and its generated `vmlinux.h` came from the same kernel and `sched_ext` interface revision. The kernel's libbpf and BTF documentation describes `/sys/kernel/btf/vmlinux` as the authoritative runtime type source used for CO-RE relocation.

For a patched or distribution kernel, record the exact kernel build, patch set, compiler, pahole version, and source revision that produced the image. That evidence separates three different failures that otherwise look alike: the feature is absent, the kfunc exists with a different signature, or the object was built against another interface revision. The `sched_ext` documentation also provides the required configuration and runtime state files, so the next step is a small known-good scheduler built against the same source tree before returning to the larger program.

### Choosing an instrumentation boundary

The final theme compared eBPF observation with compile-time language instrumentation. The practical choice depends on which context the question requires and which part of the deployment can change. eBPF is useful when operators need broad coverage without rebuilding applications and when process, network, and library boundaries contain enough information. Compile-time instrumentation is a better fit when the build pipeline can change, no privileged runtime agent is acceptable, or hooks must cover library calls that the eBPF instrumentor does not support. Code-based OpenTelemetry remains the route for application-specific intent and custom spans.

OpenTelemetry's instrumentation guidance explicitly treats code-based and zero-code approaches as complementary, while its Go compile-time documentation spells out the build-pipeline tradeoff. A small comparison should therefore use the same service and ask whether each method preserves trace context, reaches the required libraries, survives upgrades, and stays within the deployment's privilege budget. The project-owned channels contained automated development activity during the review window but no new substantive user question; that is reported separately from the technical discussions above rather than counted as community engagement.
