---
date: 2026-09-09
title: "When eBPF Calls Native Code, What Exactly Has to Be Trusted?"
description: "eBPF can verify the caller while native operations remain trusted. This report asks how to bound that TCB with proof-linked contracts, attestation, and fault tests."
tags:
  - Daily Report
  - eBPF
  - JIT
  - Verification
  - Security
  - Compilers
research_question: "How can an eBPF runtime delegate verified work to native operations without turning each optimization, helper, or backend implementation into an opaque and unbounded trusted computing base?"
source_cutoff: 2026-09-09
status: daily-report
---

# When eBPF Calls Native Code, What Exactly Has to Be Trusted?

Suppose an XDP program contains a hot eight-instruction sequence that implements a rotate. A specialized backend recognizes it and emits one native `ROL` instruction. The ordinary BPF sequence has already passed the verifier, the fast path is much cheaper, and the result looks straightforward.

Now change the example slightly. The replacement is no longer one instruction. It is a native routine that touches memory, depends on a CPU feature, has a compiler-generated prologue, and is updated independently from the BPF program. The verifier can still prove that the original BPF sequence is safe, but it does not execute the native implementation symbolically. The system has moved part of its correctness argument across a trust boundary.

That boundary already exists in several forms. Linux BPF programs call helpers and kfuncs whose implementations live in trusted kernel code. Every JIT backend must faithfully translate verified BPF into machine instructions. New work such as Kops makes the trade-off explicit by pairing a verifier-visible BPF proof sequence with a native emit. The interesting question is therefore not whether native code can be used. It is **how much native code must be trusted, what property each trusted component is supposed to preserve, and how a runtime can detect when that contract no longer matches the implementation that actually executes.**

<!-- more -->

This report follows the previous work on [runtime specialization](https://eunomia.dev/research/ebpf-runtime-profile-specialization/), [portable architecture-specific fast paths](https://eunomia.dev/research/ebpf-portable-architecture-specialization/), and [execution provenance](https://eunomia.dev/research/ebpf-specialization-debug-provenance/). Those reports ask whether a specialization is semantically justified, whether an implementation is eligible on a machine, and which generation actually ran. Here the specialization is eligible and its identity is known. The remaining issue is whether the native implementation behind that identity deserves the trust the runtime gives it.

## The verifier proves properties about BPF, not arbitrary native implementations

The Linux verifier reasons about BPF instructions, register and stack state, pointer types, control flow, helper and kfunc call constraints, and other properties needed to admit a program. Its state exploration and pruning are defined over the BPF program representation, not over arbitrary machine code emitted later by a JIT or supplied by another extension mechanism.

That separation is intentional. A verifier that also modeled every architecture backend, compiler optimization, kernel routine, NIC microcode implementation, and future accelerator would become much harder to maintain and trust. Instead, Linux relies on a pipeline: the verifier establishes properties of the BPF program, then trusted runtime components preserve those properties while executing or translating it.

The cost is that verifier acceptance and native correctness are different claims. A bug in a JIT can violate the semantics of a verifier-safe program. Jitterbug made this concrete: its authors found and fixed 16 previously unknown bugs across deployed BPF JITs while developing a formal specification of JIT correctness. The work also exposes a useful lesson for newer extension mechanisms. Even when the semantic relation is clear, some bridge between the verified model and production implementation remains trusted unless it is checked directly.

## kfuncs already expose a caller-versus-callee trust split

Linux kfuncs show the same structure at a different boundary. BTF and verifier annotations describe what arguments a BPF program may pass and what obligations surround a call. `KF_ACQUIRE`, `KF_RELEASE`, `KF_RET_NULL`, `KF_SLEEPABLE`, `KF_RCU`, and related rules let the verifier reason about reference ownership, nullability, sleepability, and pointer validity.

Those annotations are valuable because they turn part of a kernel function's calling contract into machine-checkable verifier state. But the kernel documentation is also explicit that an exposed function itself must be reviewed for whether it is safe in the contexts where BPF can invoke it. The verifier can ensure that a pointer passed to a kfunc satisfies the declared contract; it does not prove the body of every kfunc correct.

This makes kfuncs a useful comparison point for native optimization. A good interface does not need to prove every implementation from first principles, but it should minimize the undocumented part of the contract. The more effects hidden behind one call, the larger the semantic surface that reviewers and operators must simply trust.

## Kops makes trusted-code size an optimization dimension

Kops proposes native operations that carry two forms. A proof sequence uses ordinary BPF instructions and goes through the existing verifier. A native emit produces the architecture-specific instructions used for execution. For its EInsn operations, the paper reports Lean 4 proofs that each native emit computes the same result as the proof sequence; seven hardware idioms produce up to 24% microbenchmark speedup and up to 12% speedup in evaluated applications.

The same mechanism can replace much more code. Kops reports a whole-program native replacement reaching 2.358x performance, but that point deliberately grows the trusted computing base. This is the useful tension: optimization is not one scalar axis from slow to fast. It moves along at least two axes, performance and the amount of implementation whose correctness is assumed rather than derived from the verifier-visible program.

A one-instruction rotate and a thousand-instruction native replacement should not be described by the same Boolean label, `verified=true`. They require different evidence and different operational risk budgets.

## The missing abstraction is a trust contract for delegated execution

Current mechanisms provide pieces of the answer, but they expose them at different levels. The verifier has instruction semantics and abstract state. kfuncs have BTF signatures and verifier flags. JIT-verification work defines semantic equivalence between BPF and generated instructions. Kops has proof sequences, native emits, and formal proofs for selected operations. Runtime provenance can record which generation executed.

What is still missing is a common contract that answers four questions for one delegated operation:

1. **What does the verifier know?** For example, an ordinary BPF proof sequence, argument types, allowed memory regions, reference ownership, or required constants.
2. **What is additionally trusted?** This should name the native implementation, architecture/backend, compiler or emitter version, and any kernel or firmware code whose semantics are assumed.
3. **What effects are allowed?** Register clobbers, memory reads and writes, helper calls, sleeping, allocation, synchronization, faults, and control transfers need an explicit boundary rather than an informal implementation convention.
4. **What binds the contract to execution?** A proof or review of version A is not evidence about version B unless the runtime can bind the loaded implementation to the checked artifact.

Without that information, operators can count source lines or point to a proof, but they cannot answer the operational question: did the implementation currently executing satisfy the same contract that was reviewed or verified?

## Where current work is still weak

The first gap is **trust granularity**. JIT correctness is often stated for an entire backend, kfunc safety is reviewed function by function, and native-operation systems can range from single-instruction idioms to whole-program replacement. There is no shared way to compare the semantic surface placed outside the BPF verifier.

The second gap is **effect completeness**. Functional equivalence of return values is insufficient for native operations that access memory, change synchronization behavior, call other kernel code, fault, or expose timing-sensitive side effects. A proof can be correct for the modeled state while the implementation contract omits an effect that matters in production.

The third gap is **artifact binding**. A source-level proof, code review, or CI result can become stale after a compiler, kernel, module, firmware, or backend update. Execution provenance identifies what ran, but a trust decision also needs the digest and dependency context of the implementation that was checked.

The fourth gap is **evaluation under a fixed trust budget**. Optimization papers normally compare speed and sometimes code size. They rarely ask whether two systems with the same performance improvement require radically different amounts of trusted native code, or whether a slightly slower design wins because its trusted surface is much smaller and independently checkable.

## Promising directions with academic and production value

### 1. A verifier-linked native-operation contract

A native operation could carry a compact contract beside its BPF proof sequence: input and output register types, bounded memory regions, allowed side effects, clobbers, sleepability, failure behavior, architecture requirements, and a content digest of the native implementation. The verifier does not need to prove the machine code. It needs to check that the BPF-side assumptions match the operation contract and that the runtime only activates an implementation whose identity matches the approved entry.

The artifact would be a small kernel/runtime interface plus tooling that emits and inspects these contracts. The strongest baseline is today's combination of verifier-visible BPF plus manually reviewed native code. Evaluation should inject contract violations such as an undeclared memory write, extra helper call, wrong clobber set, or stale implementation digest and measure whether the system rejects activation before the bad path executes. A useful ablation removes each contract field to show which failure becomes unobservable.

The research question is how much semantic information must cross the verifier/native boundary to make delegation compositional without teaching the verifier every backend. The production value is reviewability: a kernel or runtime maintainer can reason about a 20-line effect contract instead of treating an entire backend as one opaque trust decision.

### 2. Proof-carrying trust tiers instead of one `verified` bit

Different native operations deserve different assurance mechanisms. A simple arithmetic idiom may support exhaustive equivalence checking or a Lean proof. A larger routine may be better served by translation validation over each emitted artifact. A hardware or firmware implementation may only support conformance tests plus a signed version identity.

A runtime could therefore attach a trust tier to each operation generation: for example, verifier-only, verifier plus differential tests, per-artifact translation validation, or machine-checked proof. The tier must describe evidence, not prestige. Scheduling or deployment policy can then require stronger evidence for operations with broader effects or larger trusted code.

The artifact would combine a validation pipeline with a policy engine that selects implementations under both a latency target and an assurance budget. Evaluate it across small instruction idioms, medium native routines, and whole-program replacements. Measure proof/validation latency, trusted code size, performance, escaped semantic faults, and fallback frequency. The mechanism loses if one cheap validation method catches the same fault set at materially lower complexity.

### 3. A trust-budget benchmark with adversarial native faults

The field needs a benchmark where correctness failures are first-class, not accidental. Start with verifier-safe BPF workloads and provide several semantically intended native replacements. Inject controlled faults into the native side: wrong arithmetic for rare inputs, undeclared memory writes, reference leaks, architecture-dependent flag behavior, stale code after an update, and mismatched proof/implementation pairs.

Compare stock JIT execution, helper/kfunc-style delegation, proof-linked native operations, verified JIT variants, and whole-program native replacement. Hold performance targets constant where possible, then measure escaped faults, time to detect a bad artifact, trusted source and binary surface, validation cost, and achievable speedup per unit of trusted code.

The academic contribution would be a measurable notion of trust/performance trade-off rather than a qualitative TCB claim. The production contribution is a release gate: maintainers can see whether a proposed fast path buys 8% throughput by trusting 20 instructions or by silently accepting an entire new compiler/runtime component.

## What would change this conclusion?

The argument would weaken if existing verifier, kfunc, JIT, or native-operation mechanisms already expose a machine-readable and version-bound contract that completely describes native effects and lets operators compare trust surface across implementations. It would also weaken if empirical testing showed that native-operation bugs are caught just as reliably by ordinary regression tests, making the extra contract and validation machinery redundant.

A stronger counterexample would be a translation-validation system that can cheaply prove every production native artifact against the verifier-visible BPF semantics, including relevant memory and side effects, across architectures. In that world, the trusted boundary could shrink to the validator and runtime binding mechanism, and a separate hierarchy of trust contracts might add little value.

Until then, performance specialization should make trust growth visible. The useful question is not simply whether a native fast path is safe enough to ship. It is whether the system can state precisely what it trusts, bind that statement to the code that ran, and show that the performance gain is worth the extra trusted surface.

## References

- Yusheng Zheng et al., [Kops: Safely Extending the eBPF Compilation Pipeline with Native Operations](https://arxiv.org/abs/2606.24213), arXiv:2606.24213v1, 2026.
- Linux kernel documentation, [BPF Kernel Functions (kfuncs)](https://docs.kernel.org/bpf/kfuncs.html).
- Linux kernel documentation, [eBPF verifier](https://docs.kernel.org/bpf/verifier.html).
- Luke Nelson et al., [Specification and verification in the field: Applying formal methods to BPF just-in-time compilers in the Linux kernel](https://www.usenix.org/conference/osdi20/presentation/nelson), OSDI 2020.
- Jitterbug artifact, [Verification of BPF JIT compilers](https://github.com/uw-unsat/jitterbug).
