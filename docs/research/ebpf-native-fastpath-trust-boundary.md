---
date: 2026-09-07
title: "How Small Can the Trusted Computing Base for Native eBPF Fast Paths Be?"
description: "Native eBPF fast paths can bypass verifier-checked semantics at codegen time. This report designs a bounded TCB with validation and typed emitters."
tags:
  - Daily Report
  - eBPF
  - JIT
  - Compilers
  - Program Verification
  - Kernel Security
research_question: "How can eBPF gain architecture-specific native fast paths without making every optimizer and machine-code emitter part of the kernel's trusted computing base?"
source_cutoff: 2026-09-07
status: daily-report
---

# How Small Can the Trusted Computing Base for Native eBPF Fast Paths Be?

Suppose an eBPF optimizer recognizes a rotate, bit-select, checksum idiom, or another operation that the stock JIT lowers inefficiently. It supplies a short proof sequence in ordinary BPF, the verifier accepts that sequence, and an architecture-specific emitter replaces it with a faster native instruction sequence.

The verifier can be completely correct and the program can still fail if the emitter writes one wrong register, uses one wrong stack offset, or emits a control-flow target that does not match the proof sequence. The CPU executes the native bytes, not the verifier's abstract BPF instructions.

That makes native specialization a trust-transfer problem. The previous report on [portable architecture specialization](https://eunomia.dev/research/ebpf-portable-architecture-specialization/) asked when a native implementation is eligible on a machine and how to fall back to one portable semantic witness. This report asks the next question: **once an eligible native implementation exists, how much new code must the system trust for the claim that those bytes still implement the verifier-approved operation?**

The answer is unlikely to be zero. If the kernel accepts arbitrary native bytes without checking them, some code generator or proof chain must be trusted. But the trusted surface does not have to include an entire userspace optimizer, a whole compiler backend, every optimization pass, and every architecture-specific emitter. A more useful target is a **bounded trust envelope**: keep optimization discovery outside the kernel TCB, constrain the native operation interface, and make the final native bytes pass a small independently checkable validation step before they can replace the portable BPF form.

<!-- more -->

## The verifier proves BPF semantics, not arbitrary native bytes

Linux eBPF deliberately separates admission from execution. The verifier checks the BPF program, then the architecture JIT emits native code. This organization keeps the verifier's safety reasoning independent of one CPU instruction set, but it also creates a compiler-correctness boundary: native execution is correct only if the JIT preserves the verified BPF semantics.

That boundary has failed in real systems. The OSDI 2020 Jitterbug work built precise specifications for Linux BPF JITs and found **16 previously unknown bugs in five deployed JITs** while verifying them. The important lesson is not that old JITs were unusually buggy. It is that verifier acceptance and JIT correctness are different properties. A perfectly safe BPF program can be translated incorrectly by a backend.

Current kernel development keeps exposing the same structural issue. In April 2026, Linux moved constant blinding out of architecture-specific JITs into generic verifier-side code. The motivation was not cosmetic duplication: JIT-local instruction rewriting could leave the JIT's private instruction stream out of sync with global verifier auxiliary metadata. The fix centralized the rewrite and adjusted the corresponding verifier state together.

That change is a useful design signal. When a transformation changes the instruction stream but depends on verifier-derived facts, duplicating the transformation independently inside each backend increases the number of places that must preserve hidden invariants. Moving shared semantic work toward one common boundary can shrink both inconsistency risk and per-architecture trusted logic.

## Native operations make the trust transfer explicit

[Kops](https://arxiv.org/abs/2606.24213) makes this boundary unusually clear. A Kops operation carries two forms: a proof sequence of ordinary eBPF instructions that the existing verifier checks, and a native emit that supplies the architecture-specific machine instructions. The paper's EInsn operations use Lean 4 proofs to show that the native implementation computes the same result as the proof sequence. Kops reports up to 24% improvement on microbenchmarks and up to 12% on production applications for these local operations.

The attractive part of this design is that the userspace optimizer does not need to become trusted merely because it discovered an optimization opportunity. The proof sequence remains ordinary BPF, and the native implementation can be kept small.

But there is still an important production boundary. If the kernel loads a native emit because a module registered it, the actual machine-code emitter becomes part of the trusted path. An offline Lean proof is strong evidence about the implementation that was modeled, but an operator also needs to know that the exact bytes selected today were generated by the implementation that was proved, with the expected architecture features, clobbers, calling convention, stack layout, and kernel/JIT assumptions.

Whole-program native replacement makes the trade-off more obvious. Kops can replace an entire BPF program with native code and reports up to 2.358x performance, but explicitly notes the larger TCB. Performance can therefore increase exactly when the semantic boundary becomes harder to audit.

## Fresh KASAN work shows that safety instrumentation can itself become a codegen hazard

A particularly useful counterexample landed in BPF development this week. The September 2026 v9 series adding KASAN checks to JITed BPF programs was applied to `bpf-next`. Its goal is defensive: identify memory-accessing BPF instructions and make the x86 JIT insert KASAN checks around them.

Yet the v9 cover letter records a bug in the instrumentation itself: an unwanted stack access could be instrumented while targeting the **wrong stack offset**. The revision added another guard to avoid instrumentation when the address register is the BPF frame pointer or parameter register, on top of verifier-provided stack-access metadata.

This is not an argument against JIT-side KASAN. It is a clean example of the trust problem. Even code inserted specifically to improve memory-safety diagnosis must preserve register state, stack identity, instruction offsets, patched verifier metadata, and the architecture JIT's calling convention. The more semantic knowledge a backend transformation carries, the more ways a locally reasonable emit can violate a global invariant.

For native eBPF fast paths, the question should therefore not be only "was this operation proved once?" It should be "what is the smallest mechanism that must be correct every time the exact native implementation is selected and emitted?"

## Where current work is still weak

### Offline equivalence does not identify the exact loaded bytes

A proof can establish that an implementation function is equivalent to a BPF proof sequence under a model. Production loading adds another chain: compiler version, target features, relocation, kernel version, JIT state, module version, and the final emitted byte sequence.

If the system cannot bind the proof result to the exact operation identity and native bytes that execute, the proof is valuable but incomplete as a deployment artifact. Reproducible builds help, but they do not by themselves tell the kernel which semantic effects, clobbers, and control-flow properties it should check before admitting those bytes.

### Arbitrary native emitters expose too much machine state

An unconstrained emitter can write registers, stack slots, memory, branch targets, and helper-call state. A small source function can therefore have a much larger semantic effect surface than its line count suggests.

Counting "one native emit function" as the TCB is not enough. A useful boundary should state which registers may change, which memory may be touched, whether control flow may leave the operation, whether helper/kfunc calls are allowed, what stack ranges are legal, and what assumptions about CPU features or calling convention are required.

### Optimization papers rarely measure trusted-code growth against escaped faults

Speedup is easy to graph. TCB quality is usually described qualitatively. That makes it hard to compare a stock JIT optimization, a Kops-style local native operation, a verified compiler backend, and whole-program native replacement.

The missing experiment is adversarial: mutate an emitter, its metadata, or its proof linkage, then measure whether the admission boundary rejects the wrong implementation before execution. Without this, "small TCB" can become another synonym for "few lines of code" rather than an evaluated security and correctness property.

## Promising directions with academic and production value

### 1. Load-time proof-carrying native operation capsules

A native operation should arrive as a capsule whose identity connects five things:

```text
semantic operation ID
        + portable BPF proof sequence hash
        + target/feature predicate
        + declared native effects
        + exact emitted-code hash + validation certificate
```

The userspace optimizer may remain completely untrusted. It can choose candidate operations and request specialization, but the kernel accepts a native implementation only if a small checker validates that the capsule belongs to the approved semantic operation and that its declared machine effects match the operation contract.

There are several possible implementations. A formally proved emitter can produce a compact certificate that a much smaller checker validates. A translation validator can compare a bounded native sequence against the BPF operation semantics at load time. For very small idioms, the kernel could admit only known instruction templates plus relocations rather than arbitrary byte arrays.

The research challenge is to keep the checker smaller and more stable than the compiler it replaces in the TCB. If the validator evolves into another full optimizing compiler, the architecture has only moved the trust problem.

A prototype could start with the same class of operations Kops targets: rotate, conditional select, byte manipulation, and other short register-local idioms. That keeps the semantic state small enough for precise validation while still exercising real architecture-specific instructions.

### 2. An effect-typed native emitter instead of an unrestricted assembler

A second direction is to reduce what an emitter is allowed to express.

Define a small native-operation IR or typed macro-assembler where each primitive carries effects such as:

- reads `r1`, `r2`; writes `r0`;
- clobbers flags but no BPF-visible register outside the declaration;
- touches no memory, or only a declared stack range;
- contains no indirect branch;
- calls no helper or only a named helper with a specified ABI;
- requires `x86_64 + BMI2`, `arm64 + LSE`, or another explicit target feature.

The final architecture backend then has a narrower job: encode already typed primitives into bytes. The loader can validate control-flow shape, stack accesses, clobbers, and feature predicates mechanically.

This does not prove every arithmetic identity automatically, so it complements rather than replaces semantic equivalence proofs. Its value is containing the classes of mistakes that live below the algebraic proof: wrong register allocation, undeclared memory access, wrong stack offset, invalid branch target, or ABI mismatch.

The recent KASAN JIT fixes make this direction concrete. A typed emitter should make "this instrumentation may not touch the BPF stack" a machine-checkable effect rule instead of a convention distributed across verifier metadata and architecture code.

### 3. A TCB-performance frontier benchmark with injected emitter faults

The evaluation should make trust cost a first-class axis rather than an appendix.

Take a common workload set across XDP, tracing, and compute-oriented BPF, and compare at least four configurations:

1. stock Linux JIT;
2. local native operations with manually trusted emitters;
3. the same operations behind load-time validation and effect typing;
4. whole-program native replacement or a substantially more expressive native backend.

Run the matrix on x86-64 and ARM64, across more than one kernel/JIT version. For each configuration, measure throughput or latency, specialization coverage, load-time validation cost, trusted implementation size, and the number of architecture-specific trusted components.

Then inject faults. Mutate a destination register, stack displacement, immediate, branch target, CPU-feature predicate, clobber declaration, operation identity, or proof/code hash. The primary correctness metric is **mutant escape rate**: how many semantically wrong implementations can cross the admission boundary and execute. Also record false rejection of correct native implementations.

This produces a TCB-performance frontier. One design may gain 10% while adding a tiny checker and rejecting nearly every injected fault. Another may gain 30% but require trusting a large backend with broad memory and control-flow authority. That is a more useful systems result than a speedup graph without the trust cost.

## The practical architecture is a hierarchy of trust, not one universal verifier

The goal should not be to force Linux's existing eBPF verifier to reason directly about every x86 or ARM instruction. That would entangle portable BPF admission with architecture semantics and make verifier maintenance much harder.

A cleaner hierarchy is:

```text
portable BPF program
      |
      v
Linux verifier                 <- existing portable safety boundary
      |
      v
semantic native-operation ID
      |
      +--> no eligible implementation -> execute portable BPF form
      |
      v
small native validator/checker <- bounded extra trust boundary
      |
      v
exact native bytes + provenance
```

The architecture-specific optimizer, profile collector, search algorithm, or superoptimizer can sit outside that trusted path. If it proposes nonsense, the consequence should be an optimization miss, not kernel memory corruption or a silent semantic change.

This also creates a better debugging story. When an operator asks which code ran, the runtime can name the portable operation, the selected native implementation, target predicate, checker result, and code hash. That provenance is not the main question of this report, but a bounded trust interface gives the next debugging/provenance work something stable to record.

## What would change this conclusion?

Several results would weaken the case for a separate bounded native-operation trust layer.

First, if production measurements show that almost all useful eBPF speedups can be achieved by ordinary BPF-to-BPF rewrites that return through the stock verifier and JIT, native emits may not justify their added trust surface. The [runtime-profile specialization report](https://eunomia.dev/research/ebpf-runtime-profile-specialization/) already treats that path as preferable when the portable form can express the optimization.

Second, a broadly verified JIT could make per-operation validation unnecessary. Jitterbug demonstrates that verified production JIT components are possible. If Linux eventually obtains machine-checked code generation for the relevant architectures and the native-operation interface can reuse that verified backend without adding new arbitrary emit logic, the smaller capsule checker may be redundant.

Third, the proposed validator could fail its own adversarial benchmark. If realistic wrong-register, stack-offset, control-flow, or proof-linkage mutations routinely escape a compact checker, then the bounded-operation abstraction is too weak. The system would need a stronger proof boundary or should fall back to portable BPF rather than claim safety.

Finally, if load-time proof checking or translation validation costs enough to erase the performance value of short native operations, the right granularity may be larger verified regions rather than one operation at a time.

For now, the evidence points toward a middle ground. Linux history shows that JIT correctness cannot be inferred from verifier correctness. Kops shows that useful native operations can be much smaller than a whole compiler backend. Recent constants-blinding and KASAN work shows that duplicated or backend-local transformations can violate subtle verifier/JIT invariants even when their purpose is security hardening. **The interesting systems problem is therefore not how to trust a faster compiler. It is how to arrange the interface so that the system never has to trust most of that compiler in the first place.**

## References

- Yusheng Zheng et al., [Kops: Safely Extending the eBPF Compilation Pipeline with Native Operations](https://arxiv.org/abs/2606.24213), 2026.
- Luke Nelson et al., [Specification and verification in the field: Applying formal methods to BPF just-in-time compilers in the Linux kernel](https://www.usenix.org/conference/osdi20/presentation/nelson), OSDI 2020.
- Xu Kuohai, [Move constants blinding out of arch-specific JITs](https://lists.openwall.net/linux-kernel/2026/04/15/79), Linux BPF patch discussion, 2026.
- Linux BPF maintainers, [Applied ENDBR/BTI and generic constant-blinding series](https://lists.openwall.net/linux-kernel/2026/04/16/938), 2026.
- Alexis Lothoré, [KASAN checks in JITed BPF programs, v9](https://lkml.iu.edu/2609.0/08000.html), 2026.
- Linux BPF maintainers, [Applied KASAN support for JITed BPF programs](https://lkml.iu.edu/2609.0/11290.html), 2026.
