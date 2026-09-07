---
date: 2026-09-06
title: "Can Architecture-Specific eBPF Optimization Stay Portable?"
description: "Architecture-specific eBPF fast paths need capability negotiation, portable fallbacks, and cross-JIT evidence. This report defines the missing contract."
tags:
  - Daily Report
  - eBPF
  - JIT
  - Compilers
  - Portability
  - Program Verification
research_question: "How can eBPF exploit architecture-specific native instructions and JIT capabilities without turning one portable BPF program into an implicit per-machine contract?"
source_cutoff: 2026-09-06
status: daily-report
---

# Can Architecture-Specific eBPF Optimization Stay Portable?

The same BPF program can be loaded on an x86-64 server, an ARM64 machine, or another Linux target with a BPF JIT. The verifier still reasons about BPF instructions, but the final machine code is not the same. One backend may have a single native instruction for an operation that another backend expands into several instructions, and some JIT capabilities are architecture-specific.

That difference is normally an implementation detail. It becomes a portability problem when an optimizer deliberately depends on it.

If a userspace compiler or extension runtime finds a faster native implementation, it should not force the BPF program itself to become an opaque architecture-specific binary. This report argues for a narrower contract: **keep one portable BPF semantic witness, make every native specialization an optional capability-gated implementation of that witness, and define portability as the ability to reject or fall back safely when a target cannot provide the optimized implementation.**

<!-- more -->

This is different from the previous report on [runtime profile-guided eBPF specialization](https://eunomia.dev/research/ebpf-runtime-profile-specialization/). That report asks whether a rewrite remains semantically justified when workload or deployment assumptions change over time. Here the workload can be perfectly stable. The question is what happens when the same semantic program reaches a different JIT backend or machine.

It is also different from [heterogeneous eBPF execution placement](https://eunomia.dev/research/heterogeneous-ebpf-execution-placement/). That report chooses among kernel, userspace, NIC/DPU, and GPU-side execution environments. This report keeps execution in the Linux BPF pipeline and asks whether architecture-specific lowering can remain a transparent optimization rather than creating a new application ABI.

## The BPF ISA already separates portable semantics from implementation capability

[RFC 9669](https://www.rfc-editor.org/rfc/rfc9669.html) standardizes the BPF instruction set and explicitly defines conformance groups. Every implementation must support the `base32` group and may support additional groups such as `base64`, `atomic32`, `atomic64`, `divmul32`, and `divmul64`. The RFC explains why the grouping exists: a runtime and a compiler can use capability discovery to agree on which portable BPF instructions are available.

That is an important precedent. Portability does not require every runtime to implement every optional feature. It requires the program and runtime to know which semantic instruction contract they share.

Linux adds another layer underneath that ISA. The kernel documentation lists JIT implementations for x86-64, ARM, ARM64, PowerPC, RISC-V, s390, MIPS, SPARC, and other architectures. Those backends translate the same verified BPF semantics into different native instruction streams.

The current Linux [BPF design Q&A](https://github.com/torvalds/linux/blob/master/Documentation/bpf/bpf_design_QA.rst) gives a concrete example. For 32-bit ALU operations, the verifier can insert explicit zero-extension instructions when an architecture declares through `bpf_jit_needs_zext()` that its JIT needs them. A backend with partial hardware support can then remove unnecessary extensions with a local peephole. The semantic requirement is common; the lowering strategy is not.

The kernel's current [`kernel/bpf/core.c`](https://github.com/torvalds/linux/blob/master/kernel/bpf/core.c) exposes the same pattern more broadly through weak architecture hooks such as `bpf_jit_needs_zext()`, `bpf_jit_supports_subprog_tailcalls()`, `bpf_jit_supports_kfunc_call()`, and `bpf_jit_supports_far_kfunc_call()`. The defaults are conservative and architectures override the capabilities they implement.

This means architecture capability already participates in BPF admission and lowering. What is missing is a similarly explicit contract for optimizer-added native fast paths.

## Kops shows that a native fast path can keep a portable semantic witness

[Kops](https://arxiv.org/abs/2606.24213) is useful because it does not require the verifier to understand arbitrary machine code. Each Kops operation has two forms: a proof sequence of ordinary BPF instructions that the existing verifier can inspect, and a native emit that an architecture-specific JIT path can use. Lean 4 proofs connect the native implementation to the proof sequence.

The paper reports seven hardware idioms, evaluated on x86-64 and ARM64, with up to 24% improvement on microbenchmarks and up to 12% on production applications. The repository's public paper artifact also makes the per-architecture nature explicit: each module covers one native instruction on one architecture, with 14 modules on x86-64 and 11 on ARM64 in the evaluated module tree.

That proof-sequence/native-emit split is a strong portability primitive. It says the semantic operation can remain expressible in ordinary BPF even when one target has a better implementation.

But a multi-architecture deployment still needs to answer questions outside the local proof: Which native implementation is eligible on this host? Which kernel/JIT assumptions does it require? What happens when no implementation exists for this architecture? Does the loader fall back to the proof sequence, reject the optimization, or reject the program? Can an operator tell whether two machines are running the same semantic operation through different native implementations?

A proof of one native emit does not by itself define that deployment contract.

## Architecture specialization should be negotiated, not hidden

A portable optimizer needs at least two levels of capability information.

The first level is the BPF semantic contract. It includes the ISA conformance group, program type, helper or kfunc dependencies, verifier-visible effects, and any other platform-level requirement needed to load the portable form.

The second level is the optional implementation contract. It describes which native specialization can replace a particular portable proof sequence on this architecture and under which machine and JIT conditions.

A useful representation could look like this:

```text
semantic_op = rotate64_v1
proof_seq_hash = sha256(portable_bpf_sequence)
required_bpf_groups = [base64]

native_impl = x86_64_rotate_v3
target_arch = x86_64
required_cpu_features = [...]
required_jit_capabilities = [...]
native_emit_hash = sha256(native_emit)
proof = lean4:rotate64_v1_x86_64_v3
fallback = proof_sequence
```

The exact fields are research questions, not a proposed Linux ABI. The important property is that a native implementation is selected because its predicate is satisfied, not because an optimizer happened to run on the same architecture on which the program was later loaded.

This also changes the meaning of failure. An unavailable optimization should usually be an optimization miss, not an application failure. If ARM64 lacks one x86-specific implementation, the loader should still have enough information to execute the portable proof sequence or choose an ARM64 implementation that proves the same semantic operation.

A userspace runtime such as [bpftime](https://github.com/eunomia-bpf/bpftime) could experiment with this negotiation without first changing the Linux userspace ABI. The kernel verifier would still see ordinary BPF semantics, while the runtime or extension layer decides whether a proven native implementation is eligible on the current target.

## Where current work is still weak

### ISA conformance does not describe optimizer-level native capabilities

RFC 9669's conformance groups solve an instruction-set interoperability problem. They do not say that a particular kernel JIT can lower an optimizer-defined semantic operation through a particular native idiom, nor do they identify the proof or fallback associated with that lowering.

Linux architecture hooks expose some backend capabilities internally, but they are implementation interfaces rather than one deployment-level manifest that a portable optimizer can use to explain its decisions.

The result is a gap between `this BPF program is portable` and `this optimized implementation is available here`.

### Per-architecture native modules can drift without a shared portability oracle

Architecture-specific code naturally evolves at different rates. Kops' evaluated module counts already differ between x86-64 and ARM64 because operations may target one architecture only. That is not a defect. The problem appears when a deployment assumes that optimization availability, performance, or even fallback behavior is symmetric across targets when it is not.

A portable system needs a first-class answer for partial coverage. Unsupported native operations should produce an observable fallback or eligibility result rather than silently changing what the program can do on another machine.

### A speedup on two backends is not yet evidence of performance portability

Optimization papers normally report speedup on supported machines. A fleet operator needs a different question answered: if the same semantic BPF artifact moves across architectures and kernel versions, how often does the optimized implementation remain available, how much performance is lost when it is not, and does fallback preserve the same observable behavior?

Without that evidence, architecture specialization can improve a benchmark while making deployment behavior less predictable.

## Promising directions with academic and production value

### 1. Build a two-level specialization capability manifest

The first artifact should make the semantic contract and native implementation contract independently queryable.

At the semantic level, record the portable BPF operation or proof sequence, its conformance requirements, program type, effect scope, and stable identity. At the implementation level, register one or more architecture-specific native implementations with explicit eligibility predicates.

Selection then becomes a small negotiation protocol:

```text
portable semantics accepted?
        |
        +-- no  -> reject program
        |
        +-- yes -> find eligible native implementation
                     |
                     +-- found -> use proven fast path
                     |
                     +-- none  -> execute portable proof sequence
```

The research problem is defining a capability vocabulary that is specific enough to prevent accidental mis-selection without freezing every JIT implementation detail into a permanent ABI. The production value is predictable fallback across mixed fleets and rolling kernel upgrades.

A prototype can start above the kernel ABI: encode manifests in an ELF section or sidecar object, resolve them in a loader or re-JIT runtime, and keep the Linux verifier as the final admission boundary for the portable form.

### 2. Package one semantic operation with multiple proof-linked backends

The second artifact should make a specialization unit explicitly multi-backend.

Instead of publishing `x86 implementation A` and `ARM implementation B` as unrelated optimization modules, publish one semantic operation identity with:

- one portable BPF proof sequence;
- zero or more x86-64 native emits;
- zero or more ARM64 native emits;
- later RISC-V, s390, or other implementations when available;
- a proof or translation-validation result for every native emit;
- an explicit fallback rule when no native implementation matches.

This extends the useful Kops proof/native split from local implementation safety to deployment portability. A new backend can be added without changing the semantic operation ID or requiring applications to ship another source program.

The key experiment is adversarial capability mismatch. Deliberately present the loader with a module compiled for the wrong architecture, a missing CPU feature, an older JIT capability set, and a kernel upgrade that changes backend support. The expected result is deterministic rejection of the native implementation plus successful portable execution, not undefined selection behavior.

### 3. Measure portability as a first-class optimization outcome

The third artifact should be a cross-JIT benchmark whose unit is one semantic BPF workload deployed across a target matrix.

Use the same XDP, tracing, and compute-oriented BPF programs on at least x86-64 and ARM64, plus one target where selected native operations are intentionally unavailable. Run stock JIT, architecture-specific specialization, and specialization-with-negotiated-fallback on several kernel versions.

The primary correctness oracle is semantic equality to the portable form for return values, packet or context writes, map effects, and helper/kfunc-visible behavior. Then measure:

- native specialization coverage by architecture and kernel version;
- fallback rate and reason;
- performance relative to stock BPF on each target;
- cross-target performance regret relative to the best safe implementation available there;
- proof/check and selection overhead;
- the number of native implementations added to the trusted computing base.

This would distinguish two very different results. An optimization that is 20% faster on one machine but disappears unpredictably across the rest of the fleet has weak performance portability. An optimization that is faster where supported, explicitly falls back elsewhere, and preserves one semantic artifact has a much stronger deployment story even if peak speedup is smaller.

## What would change this conclusion?

Three results would weaken the need for a separate architecture-specialization contract.

First, the stock Linux JITs may absorb nearly all profitable architecture-specific idioms while keeping those choices completely internal. If explicit optimizer-added native operations provide no material performance or deployment benefit, capability negotiation above the JIT is unnecessary complexity.

Second, BPF ISA conformance groups or a future standardized capability interface may grow enough to express the relevant operations and runtime support directly. If compilers can already negotiate every useful fast path through a stable standardized mechanism, a second optimizer-specific capability layer would be redundant.

Third, broad cross-kernel evaluation may show that simple module presence plus a portable BPF fallback is sufficient in practice. If architecture, CPU-feature, and JIT-version mismatches never create ambiguous selection or operational failures, a richer manifest may not justify its maintenance cost.

Current evidence points the other way. RFC 9669 already treats capability discovery as part of BPF interoperability. Linux JIT backends already expose architecture-dependent capability hooks. Kops shows that native hardware idioms can recover meaningful performance while retaining a portable BPF proof sequence. **The missing abstraction is a deployment contract that connects those layers: what semantic operation this is, which native implementation this machine may use, and what portable behavior remains when the fast path is unavailable.**

## References

- IETF. [RFC 9669: BPF Instruction Set Architecture](https://www.rfc-editor.org/rfc/rfc9669.html), October 2024.
- Linux kernel documentation. [Linux Socket Filtering / BPF JIT compiler](https://kernel.org/doc/html/latest/networking/filter.html), accessed 2026-09-06.
- Linux kernel documentation. [BPF Design Q&A](https://github.com/torvalds/linux/blob/master/Documentation/bpf/bpf_design_QA.rst), accessed 2026-09-06.
- Linux kernel source. [`kernel/bpf/core.c`](https://github.com/torvalds/linux/blob/master/kernel/bpf/core.c), accessed 2026-09-06.
- Yusheng Zheng et al. [Kops: Safely Extending the eBPF Compilation Pipeline with Native Operations](https://arxiv.org/abs/2606.24213), 2026.
