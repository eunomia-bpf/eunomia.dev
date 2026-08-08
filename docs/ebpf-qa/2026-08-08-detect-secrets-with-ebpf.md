# Can eBPF detect secrets in network traffic without collecting them?

**Short answer:** yes, but only where plaintext is visible, and the detector must be designed so that raw values never leave that point.

An eBPF program attached at TC, XDP, or to a packet socket sees the bytes carried on the wire. For HTTPS, those bytes are normally ciphertext, so the program can classify connections and measure flows but cannot recognize an API key inside the encrypted request. OpenTelemetry eBPF Instrumentation (OBI) reflects this boundary by separating network observability from application observability. Its documentation also notes that TLS requests rely on user-space uprobes, which may require additional privileges, rather than ordinary packet inspection alone. ([OBI security and operation modes](https://opentelemetry.io/docs/zero-code/obi/security/), [OBI troubleshooting](https://opentelemetry.io/docs/zero-code/obi/troubleshooting/))

At a plaintext boundary, such as a supported TLS library before encryption or after decryption, an eBPF-based sensor can inspect enough data to classify a possible credential. That still means the sensor can see sensitive bytes. Privacy comes from what it does next: match locally, emit only a category or counter, and discard the value instead of sending payloads through a ring buffer, log, trace, or collector.

## How to verify the design

Test the whole data path, not only the eBPF program:

1. Send a synthetic credential through an encrypted test request.
2. Confirm that packet-level hooks see ciphertext and that the plaintext-aware hook emits only the expected classification.
3. Inspect BPF maps, ring-buffer events, debug logs, traces, and exported telemetry for the synthetic value.
4. Repeat with fragmented requests, retries, unsupported TLS libraries, and non-TLS traffic so an apparent privacy guarantee does not depend on one happy path.

Downstream redaction remains useful as a second layer, but it solves a different problem. For example, Grafana Alloy's `loki.secretfilter` redacts secrets from log entries before forwarding them to Loki; it does not make an upstream payload capture private. ([`loki.secretfilter` documentation](https://grafana.com/docs/alloy/latest/reference/components/loki/loki.secretfilter/))

The practical rule is simple: classify as close as possible to the plaintext boundary, export the smallest non-sensitive result that answers the operational question, and verify that no raw value survives elsewhere in the pipeline. If the required TLS boundary is unsupported, fall back to metadata-based detection or application instrumentation rather than capturing broad payloads.
