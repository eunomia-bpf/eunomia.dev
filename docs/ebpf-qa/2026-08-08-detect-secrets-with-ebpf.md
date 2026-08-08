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

## Community discussion today

Today's visible review covered seven approved public channels across four technical communities. Names, channel identities, message links, timestamps, and deployment-specific details have been removed from this summary.

The clearest theme was the boundary between observing sensitive activity and collecting sensitive data. Practitioners want to identify credentials moving between services while keeping raw values out of telemetry. That concern produced today's question and points toward local classification at a plaintext boundary, minimal exported results, and end-to-end privacy tests.

Reliability was nearly as prominent. One discussion examined how an eBPF dataplane loader should recover after its user-space process dies, reconcile pinned objects in `bpffs`, and keep blocking kernel operations away from an asynchronous control loop. Another raised a kernel hang observed while placing an established socket into a `SOCKHASH` from a TC path, showing that lifecycle and hook compatibility remain practical debugging problems rather than abstract API details.

Kernel and toolchain compatibility formed a third cluster. A `sched_ext` program failed to load because the BTF description of scheduler kfuncs did not match what the loader expected, and rebuilding or changing runtime flags did not immediately isolate the cause. The useful diagnostic question is therefore not only which package version is installed now, but which toolchain and patches produced the BTF embedded in the running kernel.

The remaining discussion focused on choosing instrumentation mechanisms. Contributors are trying to explain when zero-code eBPF observation is preferable to compile-time language instrumentation, and where each approach loses context or portability. The project-owned channels had automated development activity but no new substantive user question during the reviewed window.
