---
date: 2026-08-27
title: "Can eBPF Preserve Security Semantics Across Backends?"
description: "eBPF security policies can run in kernels, userspace, and NICs. This report develops contracts and tests for preserving verdict semantics across backends."
tags:
  - Daily Report
  - eBPF
  - Security
  - Networking
  - Offload
research_question: "What must a system establish before an eBPF network security policy can execute in the kernel, userspace, or a NIC/DPU backend without changing allow, drop, redirect, metadata, state, or failure semantics?"
source_cutoff: 2026-08-27
status: daily-report
---

# Can eBPF Preserve Security Semantics Across Backends?

Suppose the same XDP security program can run in three places: the Linux kernel, a userspace runtime, and a NIC or DPU fast path. All three accept the bytecode. All three return familiar XDP verdicts. The NIC version is much faster.

Are they enforcing the same policy?

Not necessarily.

One backend may expose an RX hash that another does not. A redirect may preserve custom metadata but lose the original hardware descriptor. One runtime may implement a helper with slightly different error behavior. A hardware offload may support only a subset of maps, metadata, parser depth, or side effects. An exception path may send some packets to software while the accelerated path handles the rest. The program can be instruction-compatible and still make a different security decision because the environment around those instructions changed.

<!-- more -->

This report starts after placement has already been chosen. The earlier Daily Report, [Where Should eBPF Run in a Heterogeneous System?](https://eunomia.dev/research/heterogeneous-ebpf-execution-placement/), asks how to choose among kernel, userspace, SmartNIC, and accelerator execution based on event access, state, authority, and cost. Here the question is narrower: **once a security policy is moved, what evidence is enough to say that its enforcement semantics survived the move?**

That distinction matters because security is asymmetric. A false drop is an availability problem. A false allow can be a security failure. A backend should therefore not silently reinterpret an unsupported field, helper, map operation, or fallback as permission.

## Instruction compatibility is only the first contract

The BPF ISA now has an explicit interoperability vocabulary. [RFC 9669](https://www.rfc-editor.org/rfc/rfc9669.html) defines named conformance groups such as `base32`, `base64`, `atomic32`, `atomic64`, `divmul32`, `divmul64`, and the historical `packet` group. That is useful: a runtime and compiler can agree on which instructions exist instead of treating “eBPF compatible” as an undefined phrase.

But a network policy is more than its instructions.

Current Linux makes this visible in several places. The kernel's [`kernel/bpf/offload.c`](https://github.com/torvalds/linux/blob/master/kernel/bpf/offload.c) limits device-bound program initialization to `BPF_PROG_TYPE_SCHED_CLS` and `BPF_PROG_TYPE_XDP`. The same path gives offload devices explicit verifier preparation, instruction hooks, finalization, translation, and map operations. Current core offloaded-map allocation accepts array and hash maps, rather than every Linux BPF map type.

There is an even sharper boundary around packet metadata. The current offload code rejects device-bound metadata kfuncs for fully offloaded programs with the verifier message `metadata kfuncs can't be offloaded`. A program can therefore be valid as a device-bound kernel program while not being valid as a hardware-offloaded program, even before considering whether a third-party userspace runtime exposes the same metadata surface.

Linux's [XDP RX metadata documentation](https://docs.kernel.org/networking/xdp-rx-metadata.html) also treats metadata as capability-dependent. A driver may return `-EOPNOTSUPP` when a metadata kfunc is not implemented and `-ENODATA` when that metadata is unavailable for a particular frame. Custom metadata placed in `data_meta` requires an out-of-band format contract between producer and consumer. After `bpf_redirect_map()` to another device, the next consumer cannot access the original hardware descriptor; only metadata explicitly copied by the first XDP program survives.

Those are not obscure implementation details. A policy that says “drop packets from VLAN X,” “rate-limit by RX queue or hash,” or “redirect suspicious traffic and retain provenance for a second-stage decision” can change meaning if a backend exposes different context or preserves different metadata.

The userspace case has the same shape. [bpftime](https://github.com/eunomia-bpf/bpftime) is not only an instruction VM. Its runtime includes loaders, verifiers, helpers, maps, event sources, and attachment mechanisms, precisely because executing BPF instructions is not enough to reproduce an execution environment. Its XDP support and userspace maps make it a useful concrete target for asking which parts of a kernel security contract can be reproduced and which require a declared adaptation.

Hardware research reached a similar conclusion from the other direction. OSDI 2020 [hXDP](https://www.usenix.org/conference/osdi20/presentation/brunella) ran unmodified XDP programs on an FPGA NIC, but doing so required an optimizing compiler, a purpose-built eBPF processor, and FPGA implementations of XDP maps and helper functions. The system did not obtain XDP semantics from instruction decoding alone; it recreated the surrounding XDP substrate.

Modern DPU pipelines add another split. Current [NVIDIA DOCA Flow](https://docs.nvidia.com/doca/sdk/doca-flow/) documentation describes fully hardware-accelerated pipe entries while packets that miss hardware entries can be sent to Arm cores for exception handling and reinjected. Such a design can be entirely reasonable, but a security policy then needs an explicit answer for overload, unsupported matches, exception-path failure, and which path owns the final verdict.

## Where current work is still weak

### 1. “This backend supports the program” is weaker than “this backend preserves the policy”

A loader can answer whether bytecode verifies, translates, or executes. It rarely states a security-level contract for every input the policy depends on.

The missing element is an **executable semantic contract** above the ISA. For a network policy it should name at least:

```text
program / hook type
packet and context fields that may be read
metadata items and their missing-value behavior
helper / kfunc return domains
map types and update semantics
state freshness and generation requirements
terminal effects: pass, drop, redirect, modify
unsupported / unknown behavior
```

The consequence is that a backend can pass a compatibility test while silently weakening an enforcement assumption. A parser that cannot reach an encapsulated header, a missing metadata field treated as zero, or a redirect whose provenance disappears can all produce a valid program result that is not the intended security result.

The direct test is differential: seed the same policy state, feed the same packet and context corpus to a trusted reference and the candidate backend, then compare verdict, packet mutation, state delta, metadata observations, and explicit error state.

### 2. Unsupported capability is often represented too late

Linux metadata kfuncs already distinguish unsupported (`-EOPNOTSUPP`) from unavailable-on-this-frame (`-ENODATA`). That distinction is valuable because the policy can choose what each condition means. Cross-backend deployment needs the same idea before traffic is admitted.

The missing element is a **coverage-aware activation gate**. A backend should publish the capabilities needed by the compiled policy, and activation should fail if a security-relevant requirement is unsupported unless the policy explicitly defines a safe fallback.

The consequence is dangerous ambiguity. If a hardware path cannot implement one match or state operation and the system discovers that only after packets arrive, the fallback may become a hidden fail-open path or an unbounded exception queue.

The test is to deliberately remove one required capability at a time and verify that the policy either refuses activation or follows a declared exception path. No mutation should silently turn `unknown` into `allow`.

### 3. Backend differences hide in state and side effects, not only return codes

Two executions can both return `XDP_DROP` for a packet while updating different counters, connection state, rate-limit buckets, or policy generations. The divergence may affect a later packet.

The missing element is a conformance model for **observable state transition**, not just the immediate verdict. Security tests need to compare the policy-relevant state that survives one event and influences the next.

The consequence is delayed semantic drift. The first packet appears equivalent, but a different map atomicity rule, eviction behavior, update ordering, or freshness boundary causes later packets to diverge.

The test is a sequence, not a single packet: replay controlled flows with state mutations, concurrent updates, resource pressure, and backend resets, then compare both event verdicts and the policy state reachable after each step.

### 4. Exception paths need security semantics under overload

A hardware pipeline can accelerate common cases and send misses or unsupported cases to software. That is often the right architecture. The security problem appears when the exception path is slow, unavailable, or saturated.

The missing element is an explicit rule for **what the fast path may do while the authoritative exception path cannot answer**. Depending on the policy, the correct behavior may be fail closed, use a short-lived cached decision, rate-limit, quarantine, or return an explicit unknown state to another enforcement layer.

The consequence is that overload becomes a policy bypass. A benchmark that measures only steady-state throughput will not reveal it.

The test is to saturate or stop the exception processor while generating exactly the traffic that requires it. Measure false allows, false denies, queue growth, recovery, and whether the system ever reports the degraded state as fully enforced.

## Promising directions with academic and production value

### 1. Compile a security-semantics contract and differential conformance harness

**Gap.** ISA conformance and successful loading do not establish equality of network-security decisions across execution environments.

**Mechanism.** Extend the policy build artifact with a machine-readable contract generated from the program, attach configuration, and policy compiler. The contract declares the required hook, context fields, metadata, helper/kfunc behavior, map operations, persistent-state invariants, allowed terminal effects, and failure policy.

A test harness then runs a bounded corpus against a reference Linux implementation and each target backend. Linux already provides a useful building block: [`BPF_PROG_RUN`](https://docs.kernel.org/bpf/bpf_prog_run.html) can execute XDP and several other program types against userspace-provided data and contexts and return the program result. Regular test-run mode intentionally suppresses real packet side effects, so it is not a complete oracle, but it provides a repeatable reference for pure packet/context behavior. Side-effectful cases can use controlled live execution and inspect declared state deltas.

The comparison should be richer than `retval == retval`:

```text
input packet + context + policy generation
        |
        +-- reference kernel ------> verdict, mutation, state delta, evidence
        |
        +-- userspace backend -----> verdict, mutation, state delta, evidence
        |
        +-- NIC/DPU backend -------> verdict, mutation, state delta, evidence
```

Every mismatch is classified. A candidate may be equivalent, explicitly unsupported, intentionally adapted under a named rule, or wrong. “It ran” is not a classification.

**Delta from related work.** RFC 9669 gives instruction-level conformance groups. hXDP demonstrates that an FPGA target can recreate XDP maps and helpers. The proposed layer asks a different question: which environment assumptions are part of the *security decision*, and can a target show bounded behavioral conformance to those assumptions?

**Artifact.** A contract schema, a libbpf-side extractor/compiler, adapters for Linux, bpftime-like userspace runtimes, and at least one NIC/DPU target, plus a reusable packet/state corpus.

**Evaluation.** Use XDP firewall, redirect, rate-limit, load-balancing, and stateful allow-list policies. Vary metadata availability, map behavior, parser depth, redirection, state generations, and concurrent updates. Primary metric is false allow relative to the reference policy. Secondary metrics include false deny, explicit unsupported/unknown rate, state divergence, test coverage, activation latency, steady-state cycles, throughput, and memory.

**Academic value.** It separates ISA interoperability from environment-dependent security semantics and gives a concrete equivalence target that can be studied across runtimes and accelerators.

**Production value.** A deployment pipeline can reject a backend before traffic reaches it instead of discovering semantic incompatibility in production.

**Failure condition.** If real policy compilers cannot extract a useful contract without manually restating the entire program, or if the reference itself is too backend-specific to define portable semantics, the approach becomes documentation rather than executable assurance.

### 2. Make backend admission coverage-aware and fail explicit

**Gap.** A target may implement 95% of a policy and still be unsafe if the missing 5% contains the deciding security condition.

**Mechanism.** Each backend exposes a capability manifest with versions and precise failure behavior. The policy contract is matched against that manifest at activation. The result is not a percentage. It is a set relation over required semantics:

```text
required(policy) ⊆ provided(backend)
```

If the relation does not hold, the loader can reject deployment or install a declared split path. A split path carries a small witness describing why the packet left the accelerated path and which authority owns the final decision. Unsupported and overloaded cases remain observable states; neither is silently mapped to `PASS`.

The manifest should distinguish stable capability from frame-specific absence. Linux's `-EOPNOTSUPP` versus `-ENODATA` distinction is a useful precedent. It should also include limits such as tunnel depth, map capacity or type restrictions, metadata preservation across redirect, and exception-path dependencies.

**Delta from related work.** Device feature discovery already exists in several subsystems, and DOCA-style pipelines already support hardware and software exception paths. The proposed contribution is to bind those capabilities to a compiled security-policy requirement set and to make activation/fallback part of the policy's enforcement semantics.

**Artifact.** A backend capability schema, policy-to-capability matcher, fail-closed admission gate, and a per-packet exception witness for split execution.

**Evaluation.** Mutate one capability at a time: remove RX hash or timestamp support, truncate parser depth, disable a required map operation, exhaust exception queues, reset the device, or disconnect the software handler. Compare silent fallback, reject-on-mismatch, and witness-carrying split execution. Measure false allows first, then availability loss, exception volume, offload coverage, and performance.

**Academic value.** This turns partial acceleration from an implementation detail into a compositional security contract between the policy compiler and execution substrate.

**Production value.** Operators can know before activation whether “hardware offload enabled” means full enforcement, partial enforcement with a named fallback, or no safe deployment.

**Failure condition.** If capability manifests change too quickly or cannot describe semantic limits at useful granularity, conservative admission may disable acceleration so often that a simpler single-backend system is preferable.

### 3. Build a semantic-mutation benchmark for backend boundaries

**Gap.** Current performance evaluations can show that an offloaded or userspace implementation is fast while never stressing the assumptions most likely to change a security verdict.

**Mechanism.** Construct a benchmark corpus in which every test mutates one environmental assumption while preserving a known ground-truth policy result. Examples include:

- RX hash, timestamp, VLAN, queue, or custom metadata present versus absent;
- redirect with and without copied provenance metadata;
- nested tunnel/header depth just inside and outside parser capability;
- map update races, atomic operations, capacity pressure, and generation changes;
- exception handler latency, queue overflow, crash, and recovery;
- userspace/runtime restart while policy state survives or is rebuilt;
- device reset or backend switch during long-lived stateful flows.

A metamorphic test can be especially useful: if a policy does not depend on field X, removing X should not change the verdict; if it does depend on X, the target must either preserve it or report unsupported/unknown according to the contract.

**Delta from related work.** Packet-processing benchmarks usually emphasize throughput, latency, or instruction execution. This benchmark makes semantic mutation the independent variable and false security decisions the primary outcome.

**Artifact.** Open packet/context traces, state snapshots, backend fault adapters, expected verdicts, and a report format that separates false allow, false deny, unknown, and state divergence.

**Evaluation.** Run the corpus across kernel-native XDP, a userspace eBPF runtime such as bpftime, and at least one hardware or DPU backend. Compare three deployment policies: ISA/loadability only, differential conformance only, and conformance plus coverage-aware activation. Keep CPU/offload budget fixed when comparing overhead.

**Academic value.** The benchmark supplies a falsifiable target for claims that a policy is portable across heterogeneous execution substrates.

**Production value.** It can become a regression suite for backend upgrades, driver changes, new offload targets, and policy compiler releases.

**Failure condition.** If the corpus cannot represent proprietary hardware semantics or real production state, it may certify only a narrow common subset. In that case the useful result is still to name that subset rather than advertise universal compatibility.

## What would change this conclusion?

The strongest counterargument is that current loaders and hardware toolchains already reject every security-relevant unsupported operation, and that the policies operators actually offload use only a small, stable common subset. If a broad differential test across kernel, userspace, and hardware targets shows identical verdicts and policy-state transitions under metadata loss, state pressure, redirects, exception overload, and backend reset, an additional semantic-contract layer may add little value.

A second counterargument is performance and complexity. A capability witness or runtime cross-check that touches every packet would defeat much of the reason to offload. The proposed design therefore tries to move most assurance to build and activation time, leaving only the state or exception evidence that is genuinely required on the hot path.

The practical conclusion is not that every backend must emulate every Linux BPF feature. It is narrower: **a security policy is portable only across the subset of environment semantics that its decision actually depends on, and unsupported semantics must remain explicit rather than being silently translated into a different verdict.**

That is a stronger statement than bytecode compatibility and a narrower one than general heterogeneous placement. It gives kernel, userspace, NIC, and DPU implementations a property that can be tested before performance numbers are allowed to stand in for security equivalence.

## Sources

- IETF: [RFC 9669, BPF Instruction Set Architecture](https://www.rfc-editor.org/rfc/rfc9669.html)
- Linux kernel source: [`kernel/bpf/offload.c`](https://github.com/torvalds/linux/blob/master/kernel/bpf/offload.c)
- Linux kernel documentation: [XDP RX Metadata](https://docs.kernel.org/networking/xdp-rx-metadata.html)
- Linux kernel documentation: [Running BPF programs from userspace (`BPF_PROG_RUN`)](https://docs.kernel.org/bpf/bpf_prog_run.html)
- Brunella et al., OSDI 2020: [hXDP: Efficient Software Packet Processing on FPGA NICs](https://www.usenix.org/conference/osdi20/presentation/brunella)
- NVIDIA: [DOCA Flow](https://docs.nvidia.com/doca/sdk/doca-flow/)
- Eunomia: [bpftime userspace eBPF runtime](https://github.com/eunomia-bpf/bpftime)
