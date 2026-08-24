---
date: 2026-08-24
title: "Can eBPF Enforce Information Flow Through TLS and Shared Processes?"
description: "eBPF can track process, file, and socket flows, but TLS and shared runtimes blur which bytes carry which labels. This report develops flow-scoped enforcement."
tags:
  - Daily Report
  - eBPF
  - Security
  - Networking
  - Information Flow
research_question: "How can eBPF enforce information-flow policy when one process handles differently labeled data and TLS or multiplexing breaks the correspondence between application data and kernel-visible sockets?"
source_cutoff: 2026-08-24
status: daily-report
---

# Can eBPF Enforce Information Flow Through TLS and Shared Processes?

Suppose one long-running service reads both a public configuration file and a customer secret. Later it sends two HTTPS requests over the same connection pool. A process-level policy can tell that the service has touched secret data. A socket-level policy can tell where the service connects. Neither fact alone tells the kernel which request actually contains the secret.

That distinction decides whether information-flow control is useful or merely conservative. If the whole process stays labeled forever, a safe policy may block every later network request. If the label is dropped too early, the secret can leave through a buffer, TLS library, reused socket, proxy, or asynchronous I/O path without a matching policy decision.

<!-- more -->

Linux already gives eBPF several strong enforcement points. [BPF LSM programs](https://docs.kernel.org/bpf/prog_lsm.html) can implement system-wide mandatory access-control and audit policy at LSM hooks. Cgroup and socket program types can constrain network operations. [Sockmap and sockhash](https://docs.kernel.org/bpf/map_sockmap.html) can attach message verdict programs and even apply a verdict to a specified number of bytes in a socket message. These mechanisms are powerful because they act below application tool APIs.

Whole-system provenance research shows why this layer matters. [CamFlow](https://camflow.org/publications/socc-2017.pdf) used LSM and networking hooks to capture provenance across kernel objects, while [CamQuery](https://camflow.org/publications/ccs-2018.pdf) moved provenance analysis into the runtime path for applications including data-loss prevention. More recently, [ActPlane](https://github.com/eunomia-bpf/ActPlane) uses eBPF and BPF LSM to propagate labels across process, file, and network edges. Its public [rule-language model](https://eunomia.dev/actplane/rule-language/) makes the conservative choice explicit: reading a labeled file labels the process, and later writes or connects propagate that label onward.

That is a sound starting point, but it exposes the next boundary. A modern server is rarely one process per security principal or one socket per logical data object. Event loops, language runtimes, worker pools, HTTP/2, connection reuse, RPC multiplexing, and async I/O intentionally share processes and transport channels. Once one process handles several independent flows, a label attached only to the process stops identifying the data that caused the label.

TLS makes the mismatch sharper. Linux [kernel TLS](https://docs.kernel.org/networking/tls.html) can move the TLS record layer into the kernel after userspace has established the session. [TLS device offload](https://docs.kernel.org/networking/tls-offload.html) can move cryptographic work again toward the NIC. Other applications keep the complete TLS record path in userspace. The same logical HTTPS request can therefore cross very different observation and enforcement boundaries depending on the library and offload mode. A policy that assumes plaintext is visible at one socket hook is not a portable information-flow policy.

This report argues that eBPF information-flow enforcement needs one more abstraction between semantic data and kernel objects: a **flow-scoped identity with explicit coverage**. Process, file, and socket labels remain useful, but precise enforcement should be able to say which logical flow a label belongs to, how that flow became associated with a particular kernel operation, and when the association is unknown.

This question is narrower than the previous [multi-tenant network-policy report](https://eunomia.dev/research/ebpf-network-policy-composition/), which focused on policy-owner authority and verdict provenance after several policy languages are compiled into one datapath. It is also different from [BPFflow](https://vtechworks.lib.vt.edu/items/c88095ab-6dd3-4589-bd40-d0f79939a18f), which prevents an eBPF program itself from leaking sensitive kernel data by adding information-flow constraints to eBPF verification. Here the protected data belongs to the application, and the hard part is preserving its identity while execution crosses application, process, TLS, and socket boundaries.

## Kernel-object labels work until one object carries several flows

A label on a process answers a useful question: has information from a sensitive source reached this process? It does not answer which later computation depends on that source.

Consider a server with one event loop:

```text
read public.json  ─┐
                   ├─> server process ─> shared TLS connection ─> api.example
read secret.key   ─┘
```

A conservative process label turns the server into `SECRET` after `secret.key` is read. If the rule is "SECRET must not reach api.example", every later request to that endpoint is denied, including requests formed entirely from `public.json`. That is safe but can make a persistent service unusable.

The opposite design, clearing the process label after one request, is difficult to justify from kernel events alone. The secret may have been copied into a heap object, queued into another task, retained in a cache, passed through a pipe, or written into a buffer consumed later. The kernel sees operations on memory and objects, not the language-level dependence that says which bytes derive from which input.

ActPlane documents this trade-off as intentional over-tainting. For a harness around short-lived agent process trees, conservative propagation is often a good decision. The same rule becomes more expensive for multiplexed servers, database clients, language runtimes, and proxies whose process lifetime is much longer than one protected flow.

The research problem is therefore not "add more hooks." The missing property is **sub-process flow identity that remains connected to an enforceable kernel event**.

## TLS changes where useful data exists

Network policy often talks about destinations, identities, methods, paths, or data classes. The kernel may see only some of those at any one point.

With ordinary userspace TLS, application plaintext is transformed before the TCP socket sees it. Packet-layer XDP or tc programs see encrypted records, which preserve transport metadata but not the application content that motivated a data-loss policy. Uprobes can observe library calls before encryption, but a tracing hook by itself is not a hard pre-operation enforcement point, and library implementations differ.

Kernel TLS changes the path: userspace performs the handshake and installs crypto state, while the kernel TLS ULP can handle record encryption and decryption. Hardware offload can move record crypto further into the NIC path. This is useful for performance, but it means "the TLS boundary" is not one stable location that a portable eBPF policy can assume.

Sockmap message verdict programs show that eBPF can reason about byte ranges at the socket layer. The kernel documentation explicitly supports applying a verdict to the next number of bytes and corking until enough bytes exist for a decision. That is a useful primitive, but it still needs a trustworthy answer to a higher-level question: **which security label belongs to these bytes?** Parsing the bytes is not enough when encryption has already happened, when several requests share a stream, or when the relevant classification came from an earlier file or database read.

A robust design therefore needs to separate two facts:

- **semantic provenance:** which logical data flow carries a label such as `CUSTOMER_SECRET`;
- **enforcement binding:** which kernel-visible operation or byte range is currently carrying that flow.

The first fact may originate in application or runtime semantics. The second has to be checked at a boundary where eBPF can enforce or audit it.

## Where current work is still weak

### 1. Process-wide taint loses precision in multiplexed runtimes

Whole-system provenance and current eBPF IFC systems can propagate labels across processes, files, and endpoints. This is enough to prove that a sensitive source may have influenced a process, and it is deliberately conservative.

The missing element is a bounded identity for concurrent logical flows inside one process. Without it, a process that handles one secret request and one public request cannot express that only one of the two outbound operations is sensitive. The result is either persistent over-tainting or an unsafe heuristic for clearing labels.

A decisive test should run a single event-loop process with interleaved public and secret requests, shared worker threads, connection pooling, and deliberate buffer reuse. A design succeeds only if it blocks the ground-truth secret-derived egress while allowing independent public traffic without relying on one process per request.

### 2. TLS and transport offload make hook placement part of policy correctness

Linux can keep TLS record processing in userspace, move it into kTLS, or offload parts of it to hardware. The application-visible plaintext and the packet-visible ciphertext therefore live at different boundaries across deployments.

The missing capability is a machine-readable coverage contract that says where a flow label is introduced, where it is bound to transport state, which enforcement hooks are active, and which path transitions are unsupported. A policy should report `unknown` rather than silently assume that a library-level label followed data into the socket.

The test should execute the same application policy with userspace TLS, kTLS software mode, supported TLS hardware offload, plaintext TCP, and a proxy that terminates and re-establishes TLS. If the policy result changes because the instrumentation boundary moved rather than because the data flow changed, the enforcement contract is incomplete.

### 3. There is no general trusted handoff from application semantics to BPF enforcement

Kernel hooks are strong enforcement points, but they do not automatically know that a byte range is "customer secret A" or "public metrics response B." Application instrumentation can know that semantic distinction, but a compromised process must not be able to mint arbitrary labels or declassification claims that bypass policy.

The missing mechanism is a narrow, freshness-aware handoff: a flow identity, label set, generation, and allowed transformation that can be bound to a kernel object or message without trusting arbitrary application metadata. The trust model needs to say which component may introduce labels, which component may remove them, and how reuse of file descriptors, buffers, sockets, or request IDs is detected.

A useful adversarial test should attempt stale-token replay, fd reuse, buffer-address reuse, child-process handoff, proxying, and unauthorized declassification. If a stale semantic label can be rebound to a new socket or a process can self-declare its sensitive output as public, the interface does not provide enforceable provenance.

### 4. Existing evaluations rarely distinguish safe over-taint from precise enforcement

A system can look perfectly safe by marking an entire long-running service sensitive forever. It can also look fast by omitting paths where provenance is hard to recover. Neither result tells an operator whether the mechanism is deployable.

The missing benchmark needs byte- or request-level ground truth and should score both false allows and false denies. It should also expose unsupported paths explicitly, because an `unknown` decision is different from a verified allow.

The benchmark should contain shared runtimes, HTTP/2 or RPC multiplexing, async queues, fd passing, `sendfile`, user-space TLS, kTLS, proxies, and a simple one-process-per-flow baseline where coarse labels should win on simplicity and cost.

## Promising directions with academic and production value

### 1. Bind security labels to flow generations instead of process lifetime

**Gap.** Process labels preserve provenance but collapse all concurrent work in one process into one security state.

**Mechanism.** Introduce a `flow_id` whose lifetime is shorter than a process and whose reuse is protected by a generation number. A trusted source adapter creates the flow when classified input enters a runtime. Runtime instrumentation propagates the ID through known task, queue, and buffer transitions. At a kernel boundary, the flow is joined to stable kernel identity such as process lineage plus socket cookie and a generation-scoped message sequence. BPF maps retain only the active binding needed for enforcement; detailed semantic metadata remains in userspace.

The important design rule is that a raw pointer, fd number, or request ID is not a flow identity. All can be reused. The binding must include lifetime evidence and expire when the corresponding task, buffer ownership, or socket generation ends.

**Delta.** CamFlow-style provenance versions kernel objects, and ActPlane propagates labels across process/file/network objects. The new property is a versioned identity for **concurrent sub-process flows** that can be joined to those kernel objects without permanently tainting their shared owner.

**Artifact.** A small flow-label ABI, adapters for one async runtime and one TLS stack, a BPF LSM/socket enforcement backend, and a debugger that explains the flow-to-socket binding used for each verdict.

**Evaluation.** Use event-loop HTTP clients, worker pools, connection reuse, async queues, and one-process-per-request controls. Measure false allows, false denies, unknown-coverage rate, flow-binding lifetime bugs, CPU cost, map memory, and tail latency. Ablate generation tracking and force fd/buffer reuse to test whether stale bindings cause incorrect decisions.

**Academic value.** The general question is whether whole-system IFC can recover useful sub-process precision with bounded runtime metadata rather than full language-level dynamic taint tracking.

**Production value.** Long-running services and agent runtimes could enforce data-egress policy without becoming permanently unable to use the network after touching one sensitive object.

**Failure condition.** If correct flow propagation requires instrumenting most memory operations or application-specific code paths, the abstraction has moved too close to full dynamic taint analysis and loses the deployment advantage of eBPF.

### 2. Make TLS-path coverage an enforceable contract

**Gap.** The same logical request can cross userspace TLS, kTLS, or hardware-offloaded paths, so a policy tied to one plaintext hook can silently lose its meaning after a deployment change.

**Mechanism.** Define a path manifest for each protected connection class. It records the semantic-label source, expected TLS mode, the hook that binds a flow to socket/message state, the enforcement hook, and the fallback when any step is unavailable. A loader verifies the required kernel features and attachments before marking the policy active. Runtime mode changes invalidate the binding generation and force re-establishment rather than inheriting stale trust.

For one deployment the contract might bind labels before a userspace TLS write and enforce an endpoint rule at connect/send. For another, kTLS may expose a different usable path. The design does not require every mode to expose the same bytes; it requires each mode to declare what is and is not proved.

**Delta.** Existing BPF attachment APIs describe where programs run. The new property is a **policy-level proof of observation/enforcement coverage across a changing cryptographic path**, with explicit unknown states when the path cannot support the intended claim.

**Artifact.** A feature detector, path-manifest schema, adapters for userspace TLS and kTLS, and a conformance suite that intentionally changes TLS/offload modes while the application stays constant.

**Evaluation.** Run identical labeled request traces across plaintext TCP, OpenSSL or another userspace TLS stack, kTLS software mode, available device offload, and TLS-terminating proxies. Compare policy decisions and coverage reports against ground truth. Measure setup cost, per-request overhead, and the fraction of flows that degrade to `unknown`.

**Academic value.** This turns hook placement from an implementation detail into a first-class correctness condition for cross-layer security policy.

**Production value.** Operators can change TLS libraries or enable offload without silently invalidating the meaning of their eBPF data-egress rules.

**Failure condition.** If one stable kernel hook already provides equivalent semantic coverage across all relevant TLS modes, the manifest adds complexity without a new guarantee.

### 3. Build a counterexample benchmark for cross-boundary IFC

**Gap.** Safety-only evaluation rewards coarse tainting, while throughput-only evaluation rewards missing hard paths.

**Mechanism.** Construct workloads with ground-truth labeled payloads and deliberately confusing execution paths. Each test records the source label, transformations, expected sink decision, and the exact point where provenance may be lost. The harness injects concurrency, multiplexing, fd reuse, buffer reuse, proxy hops, async handoff, and TLS-mode changes. Every system must return `allow`, `deny`, or `unknown`; an unobserved path cannot be counted as a correct allow.

**Delta.** Whole-system provenance benchmarks usually score capture or query capability, while policy tests often check whether a known rule fires. This benchmark measures **decision precision under shared objects and changing enforcement boundaries**.

**Artifact.** A reproducible Linux workload suite, ground-truth trace format, adapters for coarse process IFC, flow-scoped IFC, and application-only policy, plus a scorer for false allow, false deny, unknown coverage, and overhead.

**Evaluation.** Compare a process-wide label baseline, an application-instrumented baseline, ActPlane-style object propagation, and the proposed flow-scoped mechanism. Include a simple short-lived process workload where process-wide taint should be the preferred design. Report correctness with confidence intervals and resource cost under the same workload rate.

**Academic value.** The benchmark makes the precision-versus-coverage trade-off measurable instead of treating conservative over-taint as an unqualified success.

**Production value.** Security teams can choose the simplest enforcement model that meets their false-deny budget and required path coverage rather than adopting a more complex system by default.

**Failure condition.** If real production policies rarely need sub-process distinctions, and coarse labels achieve acceptable false-deny rates on representative traces, the benchmark should show that the simpler design is sufficient.

## What would change this conclusion?

The case for flow-scoped identity becomes weaker if production workloads mostly align security principals with short-lived processes or dedicated sockets. In that environment, conservative process/file/socket labels are easier to audit and may already provide the right security boundary.

It also weakens if a stable BPF hook can recover the needed application label across userspace TLS, kTLS, multiplexing, and offload without extra runtime cooperation. The current interfaces show strong enforcement and byte-level socket primitives, but they do not by themselves establish that semantic correspondence.

The strongest evidence would be a cross-boundary benchmark with real services. If process-wide labels block little legitimate traffic while catching the same leaks as a flow-scoped design, the extra metadata is unnecessary. If the flow-scoped design materially reduces false denies without increasing false allows or leaving large `unknown` regions, then sub-process flow identity is a useful next layer for eBPF security policy.

## References

- [Linux kernel: LSM BPF Programs](https://docs.kernel.org/bpf/prog_lsm.html)
- [Linux kernel: BPF sockmap and sockhash](https://docs.kernel.org/bpf/map_sockmap.html)
- [Linux kernel: Kernel TLS](https://docs.kernel.org/networking/tls.html)
- [Linux kernel: Kernel TLS offload](https://docs.kernel.org/networking/tls-offload.html)
- [Pasquier et al., Practical Whole-System Provenance Capture, SoCC 2017](https://camflow.org/publications/socc-2017.pdf)
- [Pasquier et al., Runtime Analysis of Whole-System Provenance, CCS 2018](https://camflow.org/publications/ccs-2018.pdf)
- [Dimobi et al., BPFflow: Preventing information leaks from eBPF, eBPF 2025](https://vtechworks.lib.vt.edu/items/c88095ab-6dd3-4589-bd40-d0f79939a18f)
- [ActPlane source repository](https://github.com/eunomia-bpf/ActPlane)
