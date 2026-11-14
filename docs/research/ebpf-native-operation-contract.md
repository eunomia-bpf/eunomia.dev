---
date: 2026-09-06
title: "Can eBPF Add Native Operations Without Creating a Second Verifier?"
description: "Native eBPF operations can recover hardware performance, but every backend adds a new trust boundary. This report develops an auditable operation contract."
tags:
  - Daily Report
  - eBPF
  - JIT
  - Program Verification
  - Compilers
  - Kernel Security
research_question: "How can eBPF expose architecture-specific native operations while keeping verifier-visible semantics authoritative and making each JIT backend's extra trust small, testable, and revocable?"
source_cutoff: 2026-09-06
status: daily-report
---

# Can eBPF Add Native Operations Without Creating a Second Verifier?

A 64-bit rotate is a useful stress test for the eBPF compilation pipeline. A modern CPU can execute it with one native instruction. Portable BPF bytecode may need a longer sequence, and a deliberately simple kernel JIT can preserve that inefficiency all the way to machine code.

The obvious fix is to let the JIT recognize the sequence and emit the native instruction. The less obvious question is who is now responsible for proving that the new machine instruction still means what the verifier thought it approved.

That question becomes harder as native operations move beyond arithmetic. An architecture-specific lowering may depend on CPU features, control-flow hardening, memory ordering, verifier metadata, stack layout, or other assumptions that do not appear in the portable BPF instruction sequence. If each backend is allowed to reinterpret those obligations independently, the system has quietly created a second semantic authority after the verifier.

This report argues for a narrower design: **a native operation should be an implementation of a verifier-visible portable contract, not a new BPF semantic primitive.** The contract should state the portable proof sequence, observable effects, architecture preconditions, native implementation identity, and the evidence used to establish conformance. Unsupported or invalidated implementations should fall back to the portable sequence rather than weakening the verifier boundary.

<!-- more -->

This is distinct from yesterday's report on [runtime profile specialization](https://eunomia.dev/research/ebpf-runtime-profile-specialization/). That report asks when a rewritten BPF program remains equivalent as workload assumptions change over time. Here the source program and workload can remain fixed. The problem is whether one architecture-specific native implementation is a trustworthy realization of an already-defined BPF operation.

## The portable BPF ISA is the useful semantic anchor

The IETF [BPF ISA specification, RFC 9669](https://www.rfc-editor.org/rfc/rfc9669.html) gives BPF a portable instruction-level contract. Arithmetic width, signedness, atomic operations, jumps, calls, and memory behavior have meanings independent of one particular x86-64 or arm64 lowering.

That separation is valuable because the Linux verifier reasons about BPF-level state before the architecture JIT emits machine code. The verifier can establish properties about registers, pointers, control flow, helper use, and bounded execution without proving every native instruction sequence emitted by every backend.

The contract becomes weaker if a new optimization bypasses that relation. A backend-only pseudo-operation that the verifier cannot interpret either forces the verifier to trust backend logic it did not check or requires a second verifier for the new operation. Both choices grow the semantic trusted computing base.

A better extension point keeps the verifier's language authoritative. A native operation may have an optimized machine implementation, but it must also have an ordinary BPF representation that states what the operation means.

## Kops shows that proof sequence plus native emit is a practical split

[Kops](https://arxiv.org/abs/2606.24213) implements exactly this basic shape. Each operation has two forms: a proof sequence made from ordinary BPF instructions that the existing verifier checks, and a native emit that the architecture JIT can compile. Its EInsn prototype includes seven hardware idioms such as rotate and conditional select. The paper reports up to 24% improvement on microbenchmarks and up to 12% on production applications, while Lean 4 proofs connect each native emit to its proof sequence.

The important architectural property is not the particular seven instructions. It is that adding an operation does not require teaching the verifier a new opaque semantic rule. The proof sequence remains the verifier-visible meaning, while the native emit is the additional trusted implementation.

The public Linux Plumbers Conference 2026 contribution ["kops and rejit: Safely Optimizing eBPF for Hardware and Workloads"](https://lpc.events/event/20/contributions/2445/) makes the intended trust boundary explicit: module-supplied native emits are the part that needs careful placement in the kernel trust story.

That design still leaves an important systems question open. What exactly must be proved or tested about a native emit before it is allowed to stand in for the proof sequence?

For a pure rotate, equality of the resulting register value is close to the whole story. For future operations that touch memory, interact with control flow, use architecture state, or participate in concurrency, output equality is only one part of the contract.

## Current kernel JIT work shows that backend obligations already exceed instruction translation

Recent Linux BPF changes give concrete examples of information that cannot safely live as private backend knowledge.

A 2026 patch series [moved constant blinding out of architecture-specific JITs](https://lists.openwall.net/linux-kernel/2026/04/15/79). The problem was not that constant blinding computed the wrong arithmetic result. The private rewrite could leave the JIT's transformed instruction copy out of sync with verifier-global auxiliary data. The fix moved the rewrite into generic verifier code so instruction changes and metadata can be updated together.

The same series [passes `bpf_verifier_env` into the JIT](https://lists.openwall.net/linux-kernel/2026/04/16/307). One immediate use is control-flow integrity. On x86 with CET/IBT and arm64 with BTI, indirect jump targets need architecture-specific landing-pad instructions. The verifier already knows which BPF instructions are indirect targets, so that information is carried into the JIT instead of being rediscovered independently by every backend. The series was applied to the BPF tree with x86 ENDBR and arm64 BTI support.

This is a useful precedent for native operations. Architecture-specific machine code is not only a function from BPF opcode to instruction bytes. Correct lowering may depend on verifier facts and platform security mode. A safe extension interface needs a way to state those dependencies explicitly.

JIT hardening adds another dimension. Current Linux documentation exposes `bpf_jit_harden`, and 2026 kernel fixes added stronger protection against JIT-spraying attacks and indirect-branch predictor reuse when JIT memory is recycled. A native operation can be value-equivalent to its proof sequence while still violating a code-generation hardening invariant if it emits the wrong control-transfer shape or bypasses required mitigation behavior.

## Concurrency makes "same result" an even weaker test

The upcoming LPC 2026 [blitmus](https://lpc.events/event/20/contributions/2432/) work points at another boundary. BPF now includes acquire/release operations, atomics, spinlocks, ring buffers, and other concurrent mechanisms across x86, arm64, RISC-V, PowerPC, and additional JIT backends. The project argues that BPF still lacks a formal executable memory model and an end-to-end way to check that requested ordering survives verifier and JIT translation on each architecture.

That matters for extensible native operations. Two implementations can produce the same register value in a single-threaded test and still differ under concurrency because one omitted a barrier or used an instruction with weaker ordering. A useful conformance contract therefore needs an **effect model**, not only an input/output relation.

The same design pressure appears in the LPC 2026 proposal on [proof-carrying verification for eBPF](https://lpc.events/event/20/contributions/2440/). That work separates expensive proof discovery in userspace from a restricted kernel checker that validates explicit proof steps. Its target is verifier safety rather than JIT native operations, but the architecture is relevant: untrusted tooling can produce rich evidence while the kernel keeps a small, deterministic checking surface.

## Where current work is still weak

### Native-operation proofs usually define too little of the observable contract

For an arithmetic idiom, proving that the destination register matches the portable sequence is compelling. A general extension mechanism will eventually want operations involving loads, stores, atomics, address-space properties, or control flow. Then the observable contract includes more than a value: memory effects, ordering, faults, clobbers, helper-visible state, control-flow targets, and possibly speculation or hardening constraints.

If every operation invents its own notion of equivalence, the interface becomes difficult to audit and compose. The missing abstraction is a small set of BPF-specific effect classes that native implementations must preserve.

### Architecture and hardening preconditions are not naturally part of the operation identity

A native emit can be correct only under a particular feature set or kernel configuration. Examples include an instruction-set extension, CET/IBT or BTI mode, JIT hardening, stack-layout constraints, or a verifier fact such as indirect-target identity.

Treating these as comments inside backend code makes fallback and incident response fragile. The operation needs machine-readable preconditions that can be checked when the implementation is registered and again when it is selected.

### Cross-architecture conformance is still mostly backend-specific engineering

The kernel has many BPF JIT backends. One operation may have x86-64 and arm64 emits today, RISC-V later, and no native implementation elsewhere. A paper proof for one emit does not establish the others. Nor does a normal unit test expose weak-memory behavior, control-flow-hardening mistakes, or metadata mismatches.

A native-operation interface therefore needs a repeatable conformance artifact that follows every backend implementation, not one global statement that the operation is "verified."

## Research directions worth building

### 1. Define a verifier-visible native-operation contract with typed effects

The first artifact should be a compact operation descriptor whose semantic center is still ordinary BPF:

```text
operation = rotate64_v1
proof_sequence = [mov, lsh, rsh, or, ...]
effects = {
  regs: [r1 -> r0],
  memory: none,
  ordering: none,
  control_flow: fallthrough,
  faults: same_as(proof_sequence)
}
arch = x86_64
required_features = [ROL]
required_jit_properties = [ibt_safe]
native_emit_hash = sha256(...)
conformance = lean-proof:...
fallback = proof_sequence
```

The existing verifier checks the proof sequence exactly as it checks ordinary BPF. A small registration checker validates that the descriptor is internally consistent, that the effect class is one the kernel understands, and that the selected backend advertises the required features and hardening properties.

The effect vocabulary should stay intentionally small. Pure register transforms, bounded memory transforms, atomics with a named ordering, and control-flow operations can have different proof obligations. Operations that cannot fit one of the supported effect classes should remain ordinary BPF until the contract is extended deliberately.

The academic problem is to find an effect system expressive enough for useful hardware idioms without recreating a full machine-code verifier. The production value is a stable review boundary: maintainers can see exactly what new trust is added by one native operation.

### 2. Build cross-architecture differential and litmus conformance as a release gate

Formal proofs are strong when a precise machine model exists, but a production interface also needs regression evidence across kernels, toolchains, and CPUs.

A native-operation conformance harness should execute the proof sequence and native emit from identical generated states, then compare the declared effect set. For pure operations this can be large-scale differential testing over edge values, random states, and verifier-derived ranges. For memory and atomic operations, the harness should add blitmus-style concurrent litmus tests and compare allowed outcomes rather than only final values.

The same harness can check architecture obligations that are easy to miss in semantic proofs: indirect targets have required landing pads, emitted code respects stack and clobber conventions, JIT hardening is not bypassed, and unsupported CPU features force the portable fallback.

Run this matrix in `selftests/bpf` or an equivalent cross-architecture CI over x86-64, arm64, RISC-V, PowerPC, and other supported backends. Each backend gets its own conformance status. Adding a new native emit is not complete until its backend-specific artifact passes.

The interesting research question is how to combine formal evidence and hardware testing without pretending one substitutes for the other. A Lean proof can establish a modeled semantic relation. Real-hardware litmus and hardening tests can expose missing model assumptions. Disagreement between them is itself a useful result.

### 3. Make native implementation provenance revocable at runtime

The third artifact should record which native implementation actually executed, not just which BPF program was loaded.

A program-info or JIT-info record could expose:

```text
bpf_prog_id = 4182
operation = rotate64_v1
operation_impl = x86_64/3
native_emit_hash = ...
proof_or_test_set = kops-einsn-2026.09
cpu_features = [bmi2, ibt]
jit_hardening = enabled
fallback_available = yes
```

This provenance is useful for both debugging and security response. If a backend implementation later fails a conformance test, a kernel update changes a relevant invariant, or a CPU erratum invalidates an assumption, the implementation can be disabled by operation ID and backend version. Programs then fall back to the verifier-approved proof sequence instead of requiring source rebuilds or continuing to run suspect machine code.

This is not the profile invalidation problem from the previous report. The workload can stay identical. The invalidated object is the **trust claim about one machine implementation**.

A research prototype should measure disable/fallback latency, performance loss under fallback, the amount of provenance needed for incident reproduction, and whether operation-level revocation can remain simpler than invalidating an entire JIT backend.

## What would change this conclusion?

Three results would weaken the case for an explicit native-operation contract.

First, broader evaluation may show that operation-level native specialization has little value once normal compiler improvements and kernel JIT work are included. If only a handful of trivial arithmetic idioms benefit, maintaining a general effect and conformance framework may cost more than upstreaming those idioms directly into each JIT.

Second, a verified or translation-validated BPF-to-machine compiler could become practical across the major Linux architectures and cover the hardening, memory-ordering, and verifier-metadata obligations discussed here. If the whole backend already carries a maintained machine-checked refinement proof, per-operation contracts become less important.

Third, Linux may evolve a generic JIT IR in which verifier metadata, architecture capabilities, hardening constraints, and machine lowering are already represented in one shared pipeline. In that design, native operations could become ordinary transformations inside the generic IR rather than independently registered emits.

Current evidence points to a smaller near-term trust boundary. Kops demonstrates that verifier-visible proof sequences plus native emits can recover real performance without replacing the verifier. Current kernel JIT work shows that verifier metadata and architecture hardening already need explicit coordination. blitmus shows that cross-architecture correctness can depend on behavior invisible to single-threaded value tests. **The useful extension is therefore not "let modules emit arbitrary faster machine code." It is "let a bounded native implementation stand in for a portable BPF contract, with its assumptions, effects, evidence, and fallback all visible."**

## References

- IETF. [RFC 9669: BPF Instruction Set Architecture](https://www.rfc-editor.org/rfc/rfc9669.html), October 2024.
- Linux kernel documentation. [`bpf_jit_harden` and BPF JIT sysctls](https://docs.kernel.org/next/admin-guide/sysctl/net.html), accessed 2026-09-06.
- Yusheng Zheng et al. [Kops: Safely Extending the eBPF Compilation Pipeline with Native Operations](https://arxiv.org/abs/2606.24213), 2026.
- Yusheng Zheng, Hao Sun, Tong Yu. [kops and rejit: Safely Optimizing eBPF for Hardware and Workloads](https://lpc.events/event/20/contributions/2445/), Linux Plumbers Conference 2026 contribution, accessed 2026-09-06.
- Xu Kuohai et al. [bpf: Move constants blinding out of arch-specific JITs](https://lists.openwall.net/linux-kernel/2026/04/15/79), Linux kernel mailing list, April 2026.
- Xu Kuohai et al. [bpf: Pass bpf_verifier_env to JIT](https://lists.openwall.net/linux-kernel/2026/04/16/307), Linux kernel mailing list, April 2026.
- Linux kernel BPF maintainers. [Applied series: emit ENDBR/BTI instructions for indirect jump targets](https://lists.openwall.net/linux-kernel/2026/04/16/938), April 2026.
- Greg Kroah-Hartman. [CVE-2026-64508: bpf: Support for hardening against JIT spraying](https://lists.openwall.net/linux-cve-announce/2026/07/25/203), July 2026.
- Puranjay Mohan. [blitmus: Litmus-testing the eBPF memory model on real hardware](https://lpc.events/event/20/contributions/2432/), Linux Plumbers Conference 2026 contribution, accessed 2026-09-06.
- Martin Fink et al. [Proof-Carrying Verification for eBPF](https://lpc.events/event/20/contributions/2440/), Linux Plumbers Conference 2026 contribution, accessed 2026-09-06.