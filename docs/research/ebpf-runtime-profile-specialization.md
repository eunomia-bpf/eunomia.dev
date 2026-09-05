---
date: 2026-09-05
title: "Can eBPF Use Runtime Profiles Without Changing Program Semantics?"
description: "Runtime profiles can guide faster BPF rewrites, but verifier acceptance alone does not prove equivalence. This report defines guarded, revocable specialization."
tags:
  - Daily Report
  - eBPF
  - JIT
  - Profile Guided Optimization
  - Program Verification
  - Compilers
research_question: "Can an eBPF runtime use workload and deployment profiles to specialize BPF programs after load while proving observable behavior remains equivalent and invalidating stale assumptions safely?"
source_cutoff: 2026-09-05
status: daily-report
---

# Can eBPF Use Runtime Profiles Without Changing Program Semantics?

An XDP program was compiled once, passed the verifier, and has been running safely for weeks. Production profiles now show that one branch handles almost every packet, a configuration value almost never changes, and the machine has instructions that the generic BPF bytecode cannot express directly. There is an obvious temptation: specialize the program for the workload that is actually running.

The difficult part is not producing faster code. It is deciding what evidence makes that faster code still the same program.

Linux gives BPF a strong safety boundary. Every candidate submitted with `BPF_PROG_LOAD` goes through the verifier before the kernel JIT translates it. But verifier acceptance answers a different question from optimization equivalence. A rewritten program can be memory-safe, type-safe, bounded, and verifier-approved while still returning a different verdict, updating a different map entry, calling a helper in a different case, or mishandling a rare input that the profile happened not to observe.

This report argues that runtime specialization needs two contracts instead of one. The kernel should continue to decide whether a candidate is safe to execute. The optimizer must separately justify that the candidate preserves the original observable behavior, or make every extra workload assumption explicit, guarded, and revocable.

<!-- more -->

This is distinct from the earlier report on [stateful eBPF transactional upgrade](https://eunomia.dev/research/stateful-ebpf-transactional-upgrade/). An upgrade intentionally changes program or state semantics and needs an atomic generation transition. Runtime specialization makes the opposite claim: the optimized generation is supposed to preserve the program's declared meaning. It is also different from [heterogeneous eBPF execution placement](https://eunomia.dev/research/heterogeneous-ebpf-execution-placement/), which asks where a program should execute. Here the execution target can stay fixed while the implementation is specialized to the current machine and workload.

## The verifier proves that a candidate is safe, not that it is equivalent

The Linux [`bpf()` syscall documentation](https://docs.kernel.org/userspace-api/ebpf/syscall.html) makes the load boundary explicit. `BPF_PROG_LOAD` verifies and loads a BPF program. `BPF_LINK_UPDATE` can replace the program associated with an existing BPF link, and `BPF_ENABLE_STATS` can enable runtime statistics gathering. These are enough primitives to observe a program, prepare a new candidate, verify it, and switch an attachment without teaching the application a new deployment path.

What they do not provide is a relation between the old and new programs.

The IETF [BPF ISA specification, RFC 9669](https://www.rfc-editor.org/rfc/rfc9669.html), defines instruction semantics such as 32-bit truncation, signed operations, atomic behavior, and helper-related execution constraints. Those details matter because a rewrite is correct only if it preserves BPF semantics, not the optimizer author's intuition about equivalent C source. RFC 9669 also warns that compilers translating verified BPF to machine instructions need careful auditing because compilation itself can introduce vulnerabilities.

This yields a useful separation:

```text
kernel verifier:     may this candidate execute safely?
optimizer checker:   is this candidate equivalent to the original contract?
profile contract:    are any specialization assumptions still valid?
```

Passing the first line does not imply the second or third.

## Existing BPF optimizers show that equivalence checking is practical

The equivalence problem is not hypothetical. Several BPF optimizers already treat semantic preservation as a first-class obligation.

[K2](https://k2.cs.rutgers.edu/) uses program synthesis to optimize BPF bytecode while checking correctness and verifier safety. Its published results show that synthesis can find smaller and faster programs across packet-processing workloads, but equivalence checking is part of the optimizer rather than something delegated to the Linux verifier.

[EPSO](https://arxiv.org/abs/2511.15589), published in 2025, pushes that idea toward a faster online path. It performs expensive superoptimization offline, caches rewrite rules, and applies those rules to new BPF programs. The authors report 795 discovered rules, an average 24.37% program-size reduction relative to Clang output, and an average 6.60% runtime reduction. The important point for this report is architectural: reusable optimization rules still need an equivalence argument before they can be treated as transparent rewrites.

[Kops](https://arxiv.org/abs/2606.24213) attacks a different boundary. The stock BPF JIT intentionally stays simple, commonly translating one BPF instruction at a time in a single pass. Kops lets an operation carry a verifier-visible sequence of ordinary BPF instructions plus a hardware-specific native emit. Lean 4 proofs connect the native emit to the proof sequence. Seven hardware idioms improve microbenchmarks by up to 24% and production applications by up to 12% in the paper.

Together, these systems establish that BPF optimization can carry stronger evidence than "the new program passed the verifier." They mostly optimize from program structure or hardware capability. Runtime profiles add a different kind of fact: evidence that is true about this deployment now, but may stop being true later.

## Runtime specialization turns observations into assumptions

The public Linux Plumbers Conference 2026 abstract for ["kops and rejit: Safely Optimizing eBPF for Hardware and Workloads"](https://lpc.events/event/20/contributions/2445/) describes BpfReJIT, a userspace LLVM path that can intercept unmodified BPF load and attach calls, rewrite bytecode using configuration, workload, and kernel-version information, and send every candidate back through the original verifier and JIT. It explicitly compares runtime speculative optimization with V8 or a JVM.

That design opens a useful systems question that static equivalence alone does not answer: **which profile facts are merely hints, and which facts become semantic preconditions for the generated program?**

A branch-frequency profile is usually safe to use for layout or code placement because the cold path remains present. A runtime configuration value is different. Replacing a map lookup with a constant can be correct only if the optimizer knows that the value cannot change while that specialized generation is active, or if generated code guards the assumption and falls back when it changes. Removing an apparently impossible helper path because it never appeared in a trace is even stronger: absence in a finite profile is not proof of impossibility.

The same distinction appears in mature speculative JITs. Optimization is not just `profile -> faster code`; it is `profile -> assumptions -> optimized code + invalidation path`.

For BPF, the invalidation path has an unusual advantage. The original portable program can remain the reference implementation. If a specialization assumption becomes false, the runtime can fall back to the original program or compile another specialized generation, and the kernel verifier remains the final safety admission point for every candidate.

## The kernel JIT already shows why transformation metadata must stay synchronized

Recent BPF kernel work provides a concrete warning against treating transformed code as an anonymous implementation detail.

A 2026 patch series that [moves constant blinding out of architecture-specific JITs](https://lists.openwall.net/linux-kernel/2026/04/15/79) explains that private instruction rewriting could leave the JIT's transformed instructions out of sync with verifier-global auxiliary data. The proposed fix moves the rewrite into generic verifier code and updates instruction metadata together. The series was later applied with related changes that pass verifier information into JIT backends.

A separate September 2026 series adds [KASAN checks to JIT-compiled BPF programs](https://lkml.iu.edu/2609.0/08000.html). Its v9 cover letter describes fixes for incorrectly instrumented stack accesses and adds another guard against instrumenting the wrong stack offset. KASAN intentionally changes execution behavior to detect memory errors, so it is not an equivalence-preserving optimizer. It is useful evidence for a narrower point: once the JIT starts injecting or rewriting machine-level behavior, architecture-specific transformations, instruction metadata, and debugging identity must stay coordinated.

A workload-specializing re-JIT makes that requirement stronger because several executable generations may be valid at different times.

## Where current work is still weak

### Verifier success does not certify transparent replacement

A runtime can take an original BPF program, produce a different program that passes the verifier, and switch the link successfully. None of those steps proves that packet verdicts, return values, map writes, tail calls, helper effects, or externally visible state transitions are equivalent to the original.

For purely local arithmetic rewrites, K2- or EPSO-style equivalence checking is a plausible answer. Whole-program BPF behavior is harder because helpers, maps, kernel context, concurrency, and program-type-specific effects participate in the semantics. A production re-JIT therefore needs to state the equivalence scope it actually proved rather than presenting one generic "verified" bit.

### A profile has a lifetime, but optimized code usually does not expose it

A workload profile is a historical sample. Branch bias can reverse. A configuration map can change. A CPU can be replaced after migration. A kernel update can alter helper or kfunc availability. A BTF layout can move. If these facts influenced code generation, the running program needs a machine-readable dependency on them.

Otherwise an optimization can be correct at compilation time and unjustified ten minutes later.

### Operators can inspect the loaded program without knowing why this generation exists

Linux exposes BPF program IDs, metadata, links, and JIT-related information, but a dynamic specialization layer can create another identity problem: which original program produced this candidate, which profile epoch triggered it, which assumptions were embedded, which transforms ran, and why was this generation selected instead of another?

Without that provenance, a performance regression or correctness incident becomes difficult to reproduce. The program an operator dumps after the incident may not be the program that handled the problematic workload phase.

## Research directions worth building

### 1. Make specialization produce an optimization-equivalence certificate

The first artifact is a compact certificate attached to every specialized generation. It should identify both the stable semantic source and the concrete candidate:

```text
source_prog_hash = sha256(original_bpf)
candidate_hash = sha256(specialized_bpf)
program_type = XDP
equivalence_scope = [return, packet_writes, map_effects, helper_trace]
transform_set = [branch_layout, alu_rewrite, const_fold]
checker = translation_validation_v3
checker_result = equivalent
kernel_verifier = accepted
profile_epoch = 418
assumptions = [config_generation=92]
```

The certificate deliberately separates three facts. The candidate passed the Linux verifier. The optimizer checked some defined equivalence relation. Any assumption that made the rewrite conditionally valid is listed separately.

The checker can be heterogeneous. Small instruction slices can use SMT-backed translation validation similar to K2 or EPSO. Kops-style hardware operations can use a proof sequence plus machine-level proof. Helper-heavy regions may use a conservative effect summary and refuse optimization when the checker cannot model them. "Unknown" is a valid result and should fall back to the original program.

The academic problem is compositional equivalence across BPF-specific effects. The production value is auditability: a runtime can prove what it checked without asking operators to trust an opaque optimizing compiler.

### 2. Treat profile-derived facts as guarded specialization dependencies

The second artifact is an assumption registry tied to the active specialization generation.

Some transformations require no runtime assumption. Reordering basic blocks while preserving all edges can use branch frequency as a hint without making hotness part of correctness. Other transformations are conditional. Constant-folding a configuration map entry, specializing a helper path for a kernel capability, or eliminating a case because a feature flag is fixed needs a validity condition.

The runtime should classify these dependencies explicitly:

```text
layout_hint(branch_17 = 99.8% taken)          -> no semantic guard
config(map_fd=8,key=3,generation=92,value=1) -> invalidate on generation change
kernel_btf(hash=...)                           -> invalidate on kernel/BTF change
cpu_features(avx2,bmi2)                        -> invalidate on migration
```

A violated dependency should trigger bounded deoptimization: switch the attachment back to the portable original or to another already-verified generation, then recompile if useful. `BPF_LINK_UPDATE` is one possible actuation primitive, but the research contribution is not link replacement itself. It is the rule that connects a profile assumption to the period during which one optimized program is semantically justified.

This is where BPF can borrow from speculative language runtimes without importing their whole execution model. The portable BPF program is already a natural deoptimization target, and the kernel verifier remains a safety gate after every rewrite.

### 3. Benchmark specialization with phase shifts and rare-path counterexamples

A profile-guided optimizer can look excellent on a stationary benchmark and fail on the first workload transition. Evaluation should therefore make profile staleness part of the test rather than noise to average away.

A useful benchmark suite would take the same real BPF programs and run controlled phases that change one assumption at a time: packet mix, configuration-map generations, kernel or BTF versions, CPU capabilities, rare error paths, map pressure, and helper outcomes. Each run records both the reference program and every specialized generation.

Compare at least four classes of implementation:

- ordinary Clang plus the stock kernel verifier/JIT;
- static equivalence-preserving optimization such as K2 or EPSO;
- hardware-specialized operations such as Kops;
- runtime profile-guided re-JIT with guards and deoptimization.

The primary correctness metric should be **observable divergence from the portable reference under adversarial phase changes**, including wrong return values, packet mutations, map effects, and helper-effect traces. Secondary metrics can include speedup, time to detect an invalid assumption, deoptimization latency, equivalence-check cost, verifier cost, specialization churn, and the fraction of proposed optimizations conservatively rejected.

The benchmark should deliberately include rare paths that do not appear in the training profile. If an optimizer turns "not observed" into "impossible," the test should make that mistake visible immediately.

For debugging, each result should carry the source program hash, specialization generation, profile epoch, transform set, assumption set, and resulting JIT identity. Performance numbers without this provenance are not enough to reproduce a dynamic optimizer.

## What would change this conclusion?

Three results would weaken the case for a separate runtime specialization contract.

First, experiments may show that the profitable workload-guided BPF optimizations are almost entirely unconditional transformations. If profiles only choose layout and instruction scheduling while all semantic paths remain intact, static equivalence checking may be enough and an assumption/deoptimization layer adds little value.

Second, production evaluation may show that profile-guided re-JIT provides negligible benefit after accounting for modern Clang, static BPF optimizers, verifier constraints, and the stock JIT. In that case the extra compiler, profile collector, equivalence checker, and generation management are operational complexity without enough performance return.

Third, Linux could eventually provide a standardized optimization or translation-validation interface that binds transformed BPF or machine code to verifier-visible semantics and exposes generation provenance. If that interface also represents conditional assumptions and invalidation, a separate userspace certificate layer would become redundant.

Current evidence supports the opposite direction. K2 and EPSO show that BPF bytecode still contains profitable semantics-preserving rewrites. Kops shows that hardware-specific operations can recover performance while carrying explicit proof structure. The BpfReJIT design shows that deployment and workload information can be introduced without replacing the kernel verifier. **The missing abstraction is not another source of profile data. It is a contract that says exactly why this specialized generation is still the same program, and when that claim expires.**

## References

- IETF. [RFC 9669: BPF Instruction Set Architecture](https://www.rfc-editor.org/rfc/rfc9669.html), October 2024.
- Linux kernel documentation. [`bpf()` syscall reference](https://docs.kernel.org/userspace-api/ebpf/syscall.html), accessed 2026-09-05.
- Q. Xu et al. [K2: Synthesizing Safe and Efficient Kernel Extensions for Packet Processing](https://k2.cs.rutgers.edu/), SIGCOMM 2021.
- Qian Zhu et al. [EPSO: A Caching-Based Efficient Superoptimizer for BPF Bytecode](https://arxiv.org/abs/2511.15589), 2025.
- Yusheng Zheng et al. [Kops: Safely Extending the eBPF Compilation Pipeline with Native Operations](https://arxiv.org/abs/2606.24213), 2026.
- Yusheng Zheng, Hao Sun, Tong Yu. [kops and rejit: Safely Optimizing eBPF for Hardware and Workloads](https://lpc.events/event/20/contributions/2445/), Linux Plumbers Conference 2026 contribution, accessed 2026-09-05.
- Xu Kuohai et al. [bpf: Move constants blinding out of arch-specific JITs](https://lists.openwall.net/linux-kernel/2026/04/15/79), Linux kernel mailing list, April 2026.
- Alexis Lothoré. [bpf: add support for KASAN checks in JITed programs, v9](https://lkml.iu.edu/2609.0/08000.html), Linux kernel mailing list, September 2026.
