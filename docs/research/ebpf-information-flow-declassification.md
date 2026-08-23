---
date: 2026-08-23
title: "Can eBPF Prevent Secret Egress Without Tainting the Whole Process?"
description: "Process-level eBPF taint can block real leaks but overtaint mixed workloads. This report develops trusted declassification boundaries and precision escalation."
tags:
  - Daily Report
  - eBPF
  - Security
  - Information Flow
  - Linux
research_question: "How can an eBPF information-flow policy stop sensitive data from reaching unauthorized network sinks without treating every later output from a process that once read a secret as sensitive?"
source_cutoff: 2026-08-23
status: daily-report
---

# Can eBPF Prevent Secret Egress Without Tainting the Whole Process?

A build service reads a signing credential, a public manifest, and a large set of source files. Later it uploads a public test report over HTTPS. A process-level information-flow policy sees the credential read and marks the service as sensitive. From that point on, a conservative rule has two choices: block every network write from the process, including the harmless report, or allow some destinations and accept that the same process could also send the credential there.

This is a precision problem, not a missing-hook problem. Linux already exposes strong mediation points for files, processes, sockets, and network connections. eBPF programs can attach to LSM and cgroup hooks, and whole-system provenance systems have shown for years that kernel object flows can support data-loss prevention. The difficult boundary appears after sensitive and non-sensitive data have been mixed inside one userspace address space. The kernel can see that a process once read a secret and later wrote to a socket. It usually cannot prove which bytes in that write came from the secret.

<!-- more -->

The practical conclusion is that high-assurance eBPF information-flow control should not promise byte-precise taint from OS events alone. A deployable design needs an explicit **declassification boundary** for the cases where coarse provenance would otherwise block useful work, plus a coverage contract that says when the runtime can and cannot observe the relevant release path. Process-level taint remains the safe default. Precision is earned only at a boundary whose transformation and authority can be checked.

This question starts the active **eBPF Networking and Security** series. It is deliberately narrower than the earlier [hook-composition report](https://eunomia.dev/research/ebpf-hook-composition-contract/), which studied how multiple BPF programs share one hook, and the [stateful-upgrade report](https://eunomia.dev/research/stateful-ebpf-transactional-upgrade/), which studied how whole BPF applications change generation. Here the invariant is about information released to an external sink.

## Kernel provenance is already strong at object boundaries

The Linux BPF LSM interface lets privileged BPF programs attach to LSM hooks and implement system-wide MAC and audit policies. The LSM hook set includes operations around file access, task transitions, IPC, and socket activity. Cgroup BPF program types add policy points at network boundaries such as connect and socket-option handling. These interfaces are useful because a descendant process, generated shell script, or closed-source binary still reaches the same kernel boundary.

This object-level view is not new research. [CamFlow](https://camflow.org/publications/socc-2017.pdf) used Linux Security Modules and networking hooks to capture whole-system provenance, and demonstrated applications including data-loss prevention. [CamQuery](https://camflow.org/publications/ccs-2018.pdf) moved provenance analysis into the runtime path so information-flow queries could drive real-time security decisions instead of only post-hoc forensics. The important lesson is that process, file, IPC, and network objects can form a useful provenance graph without application rewrites.

Modern eBPF makes that approach easier to deploy and compose with other kernel controls, but it does not change the granularity of the evidence. A read edge from file `secret.key` to process `P` says that information from the file may now influence `P`. A later socket write from `P` says that `P` emitted bytes. There is no kernel object representing every intermediate `memcpy`, parser field, string concatenation, compression step, or redaction inside the address space.

This is why conservative information-flow systems propagate labels broadly. The current [ActPlane rule-language documentation](https://eunomia.dev/actplane/rule-language/) states this trade-off explicitly: after a process reads a labeled file, the process inherits that label, and later writes propagate the label further. Its [support matrix](https://eunomia.dev/actplane/support-matrix/) describes over-tainting as intentional because preserving provenance is safer than silently dropping a derived flow. That is a reasonable security default. It also identifies the production pain this report focuses on.

## HTTPS creates a second semantic boundary

Encrypted traffic makes the same gap easier to see. At the kernel socket layer, TLS application data may already be ciphertext. A network hook can still enforce destination, cgroup, socket, address, or connection policy, but it cannot infer the plaintext field that produced one record.

At a userspace TLS API, the situation is different. OpenSSL documents `SSL_write()` and `SSL_write_ex()` as taking an application buffer and writing those bytes to a TLS connection. An uprobe at that call can therefore observe plaintext before encryption. The [Eunomia sslsniff tutorial](https://eunomia.dev/tutorials/30-sslsniff/) demonstrates this mechanism with eBPF uprobes.

But treating `SSL_write` as a universal plaintext checkpoint would be another overclaim. OpenSSL also exposes `SSL_sendfile()` when kernel TLS is enabled, allowing file data to move through a zero-copy path. Other applications use BoringSSL, GnuTLS, rustls, language runtimes, custom framing, QUIC, or static linking. The [Claude Code TLS investigation](https://eunomia.dev/blog/2026/02/13/reverse-engineering-claude-codes-ssl-traffic-with-ebpf/) shows the operational work needed even when a single BoringSSL path ultimately explains the traffic: stripped binaries, runtime-specific call paths, and timing all affect whether a probe actually covers the relevant requests.

So application-boundary probes can improve semantic precision, but they introduce a **coverage obligation**. A security system must distinguish "the inspected TLS call contained no sensitive data" from "the egress took a path we did not inspect."

A recent preprint, [A Study of Kernel Telemetry Options for Security-Oriented Provenance](https://arxiv.org/abs/2608.11418), reinforces the broader point from another direction. It finds eBPF promising for provenance capture, while reporting that many evaluated capture stacks still fail security-oriented integrity or availability requirements for their event streams. For enforcement, missing evidence is not a neutral result. Unknown coverage has to remain visible in the policy decision.

## The real choice is where declassification becomes trustworthy

Suppose the service from the opening example reads a secret signing key and then produces a public statement saying only `signature verified`. A useful policy should allow the statement while forbidding the key itself from leaving the host.

There are three broad ways to justify that release.

One is full byte-level dynamic taint tracking inside the application. This can be precise, but it requires instruction- or language-level instrumentation and carries a very different deployment and overhead model from an eBPF policy plane. It is not something an LSM hook can reconstruct after the fact.

A second is to trust the whole process as a declassifier. That is simple but weak. Once the process is allowed to clear its own sensitive label, a bug or compromise in the same process can release anything.

The third is to make declassification an explicit boundary with narrower authority. A small trusted component receives labeled input, performs one declared transformation, and produces or transmits only the released result. Kernel provenance then has a new object or subject boundary it can mediate. This approach gives up transparent byte-level inference in exchange for a property that is much easier to state and test.

The design question is therefore not "how many more eBPF hooks can we add?" It is "what evidence lets a coarse kernel label become less restrictive without letting the original process simply declare itself clean?"

## Where current work is still weak

### Process labels cannot distinguish mixed buffers

Whole-system provenance and lineage-scoped IFC are intentionally conservative. Once a process consumes sensitive and public inputs, a single process label represents their union. That is safe for confidentiality but can turn a long-running compiler, database, agent runtime, or web service into a permanently tainted subject.

The missing element is a deployable way to recover precision at selected release points without instrumenting every userspace instruction. The consequence is a familiar DLP trade-off: coarse policies either block too much or create broad destination exceptions that weaken the original guarantee.

A decisive experiment should run the same mixed workload under process-level taint, byte-level taint, and explicit declassification. If coarse taint rarely blocks benign output in realistic workloads, the extra mechanism is unnecessary.

### Application-boundary probes have no universal coverage contract

Uprobes can observe `SSL_write`, parser functions, serializers, and application-defined gates without recompiling the kernel. They do not automatically prove that every relevant execution path used those functions.

The missing element is a machine-readable coverage contract tied to binary identity, function or offset identity, runtime/library version, and the sink that the probe is supposed to explain. Without it, a successful attach can be mistaken for complete mediation.

The test should deliberately switch among OpenSSL `SSL_write`, OpenSSL `SSL_sendfile` with kTLS, BoringSSL, a Rust TLS stack, direct plaintext sockets, and a helper subprocess. The system should report either a covered release path or an explicit unknown state, never silently classify an unobserved path as clean.

### Declassification authority is usually implicit

Information-flow systems need some way to remove or transform labels, otherwise useful systems eventually become saturated with taint. A rule such as "clear SECRET after running sanitizer" is only strong if the sanitizer itself is a trusted release boundary.

The missing element is an authority and identity model for declassifiers. The runtime should know which binary or service instance is allowed to release which label, under which input/output relationship, and for which policy generation. A userspace event emitted by the untrusted process itself is not sufficient evidence under an adversarial threat model.

A useful fault test should replace the sanitizer binary, inject a sibling process with the same name, replay a stale release token, and crash the policy controller during a release. The policy must fail closed or retain an explicit unknown state.

### Security evaluation rarely measures benign blocking and leak resistance together

A policy can look strong by blocking every network operation, or look usable by allowing every common endpoint. Neither result measures whether the system separates sensitive from non-sensitive releases correctly.

The missing benchmark is a mixed-flow workload with known ground truth about which outputs depend on protected inputs. It should score false-negative leaks, false-positive blocks, unknown/coverage states, recovery behavior, and overhead under the same workload.

This matters because a mechanism that reduces false positives by introducing an unmeasured bypass is not an improvement.

## Promising directions with academic and production value

### A trusted release proxy with kernel-visible provenance

**Gap.** Process-level taint cannot safely clear one output from a process that still holds sensitive state.

**Mechanism.** Move declassification into a small, separately confined release service. The tainted application sends candidate output and a requested release class over a kernel-visible IPC boundary. The service performs the declared transformation, such as redaction, aggregation, signature verification, or schema projection. BPF-LSM policy restricts which executable identity may act as that declassifier and which sinks it may use. The service either transmits the released result itself or returns it through a dedicated object whose provenance is tied to the declassifier generation. The original tainted process never receives authority to clear its own label.

The important property is separation, not the word "proxy." A trusted helper process, a dedicated local service, or a sandboxed release worker can all implement the boundary if the OS can mediate its inputs, outputs, executable identity, and destination authority.

**Delta.** CamFlow and CamQuery show that whole-system provenance can support DLP, while existing eBPF IFC engines propagate labels conservatively. The new artifact would treat declassification as a first-class cross-domain protocol with explicit authority and generation state, rather than a label-removal rule executed inside the original subject.

**Artifact.** A libbpf/BPF-LSM controller plus release-service SDK, initially supporting file-to-network and IPC-to-network flows. The policy manifest would declare source labels, allowed declassifier identities, release classes, permitted sinks, and fail-closed behavior for missing evidence.

**Evaluation.** Run build systems, data-processing jobs, and agent workloads that intentionally mix secret and public inputs. Compare process-level taint, destination allowlists, the release proxy, and an application-instrumented baseline. Measure actual secret leaks, benign network operations blocked, release latency, throughput, CPU cost, policy-state memory, and recovery after helper/controller crashes. Include an adversarial workload that calls the release service with malformed or replayed requests.

**Academic value.** The general question is how much precision an OS-level IFC system can recover by introducing a small trusted declassification boundary instead of tracking arbitrary in-process memory flow.

**Production value.** Operators can keep conservative kernel enforcement around opaque or complex applications while carving out narrow, reviewable release paths for telemetry, reports, redacted logs, and approved exports.

**Failure condition.** If the IPC and restructuring cost is higher than ordinary application instrumentation, or if most workloads already isolate sensitive handling in dedicated services, the proxy adds complexity without enough precision benefit.

### A coverage-aware egress manifest for opaque runtimes

**Gap.** Application probes can recover plaintext or semantic context, but attachment success does not prove that every egress path is covered.

**Mechanism.** Define an egress manifest that maps a runtime build identity to expected plaintext boundaries and downstream socket paths. Entries can name shared-library build IDs, statically linked function fingerprints, USDT probes, or supported runtime adapters. eBPF uprobes attach to those boundaries and associate release events with socket identity. Kernel network hooks independently observe the actual connection. If a socket emits traffic with no matching covered release path, the policy marks the flow `unknown` and applies a configured fail-closed, isolate, or audit action.

The manifest should explicitly include alternative paths such as OpenSSL `SSL_sendfile`/kTLS and helper subprocesses. Coverage is a runtime property that can decay after an upgrade, so the controller records the validated build generation rather than assuming yesterday's offsets remain correct.

**Delta.** This extends ordinary TLS tracing by making *absence of a correlated semantic event* part of the enforcement state. It also differs from the earlier application-resource semantics report: here the contract protects a confidentiality decision at an egress sink, and incomplete semantic coverage changes whether release is allowed.

**Artifact.** A coverage validator and eBPF correlation runtime with adapters for OpenSSL, BoringSSL, one Rust TLS stack, direct sockets, and kTLS. The artifact should expose a machine-readable coverage report before policy activation.

**Evaluation.** Mutate library versions, strip symbols, switch TLS implementations, toggle kTLS, fork helper processes, and replay real client/server workloads. Measure coverage detection, false `unknown` states, missed egress paths, attach/update time, and steady-state overhead. Compare against "probe attached successfully" as the baseline acceptance rule.

**Academic value.** The research question is whether heterogeneous application probes and kernel sinks can provide a sound enough *coverage contract* for security decisions without requiring full application instrumentation.

**Production value.** Security teams get an enforceable distinction between inspected encrypted egress and traffic whose plaintext path is not understood.

**Failure condition.** If runtime variation makes manifests too fragile, or if unobserved paths cannot be detected reliably from kernel-side socket evidence, the mechanism belongs in observability rather than hard enforcement.

### A mixed-flow benchmark for eBPF DLP precision

**Gap.** Existing provenance and policy evaluations often demonstrate that a forbidden flow can be detected, but deployment decisions also need to know how much useful work is blocked by conservative taint.

**Mechanism.** Build a benchmark where each workload produces a labeled dependency graph as ground truth. Cases should include secret and public file reads in one process, forks and execs, pipes and Unix sockets, file descriptor passing, mmap, temporary files, compression, redaction, HTTPS through several TLS stacks, kTLS/sendfile, direct sockets, and approved declassification. Each output is labeled by the harness according to whether protected source bytes semantically influence it.

**Artifact.** A reproducible workload suite, trace generator, attack/bypass corpus, and scorer. The scorer reports leak rate, benign-block rate, unknown coverage, provenance continuity, policy recovery after faults, and overhead under a fixed request mix.

**Evaluation.** Compare path and destination allowlists, process-level label propagation, provenance-query enforcement, the trusted release proxy, and application-assisted approaches. Include ablations that remove TLS coverage validation, declassifier identity checks, and policy-generation freshness.

**Academic value.** The benchmark turns "more precise IFC" into a measurable trade-off between confidentiality, availability, coverage, and cost.

**Production value.** Teams can decide whether coarse eBPF IFC is sufficient for their workload or whether they need an explicit release boundary before enabling blocking mode.

**Failure condition.** If the benchmark's dependency oracle cannot represent realistic transformations or attackers can trivially exploit unmodeled semantics, it should remain a stress suite rather than a security ground truth.

## What would change this conclusion?

The argument here assumes a common production setting: sensitive and public data share long-lived userspace processes, operators want OS-level mediation, and full byte-level dynamic taint is too invasive or expensive to deploy everywhere.

Several results would weaken the case for explicit declassification. If measurements show that conservative process-level taint causes almost no benign blocking in representative services, the simpler model wins. If practical whole-process byte-level taint becomes deployable with comparable overhead and language/runtime coverage, a release proxy may be unnecessary. If kernel or hardware interfaces expose trustworthy fine-grained provenance for userspace buffers, the boundary between object provenance and in-process flow would move.

The opposite result would strengthen the case: if mixed workloads show high false-positive blocking under coarse taint, while narrow trusted release boundaries preserve confidentiality with modest cost, then declassification should become a first-class part of eBPF security runtimes rather than an application-specific exception.

The key constraint remains simple. A system may be conservative when it lacks evidence. It should not claim that sensitive information became safe merely because the path that would prove otherwise was invisible.

## References

- [Linux kernel documentation: LSM BPF Programs](https://docs.kernel.org/bpf/prog_lsm.html)
- [Linux kernel documentation: BPF program types and cgroup network attach points](https://docs.kernel.org/bpf/libbpf/program_types.html)
- [CamFlow: Practical Whole-System Provenance Capture, SoCC 2017](https://camflow.org/publications/socc-2017.pdf)
- [CamQuery: Runtime Analysis of Whole-System Provenance, CCS 2018](https://camflow.org/publications/ccs-2018.pdf)
- [A Study of Kernel Telemetry Options for Security-Oriented Provenance, 2026](https://arxiv.org/abs/2608.11418)
- [OpenSSL documentation: SSL_write, SSL_write_ex, and SSL_sendfile](https://docs.openssl.org/master/man3/SSL_write/)
- [Eunomia: eBPF TLS plaintext tracing tutorial](https://eunomia.dev/tutorials/30-sslsniff/)
- [Eunomia: ActPlane support matrix](https://eunomia.dev/actplane/support-matrix/)
- [ActPlane source](https://github.com/eunomia-bpf/ActPlane)
