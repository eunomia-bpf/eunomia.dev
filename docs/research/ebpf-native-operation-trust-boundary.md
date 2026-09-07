---
date: 2026-09-07
title: "Can eBPF Delegate Native Operations Without Growing Its Trust Boundary?"
description: "Native eBPF operations can bypass verifier-visible bytecode and expand the trusted computing base. This report defines proof-linked delegation and fault tests."
tags:
  - Daily Report
  - eBPF
  - JIT
  - Program Verification
  - Systems Security
  - Compilers
research_question: "How can eBPF delegate a verified semantic operation to native code without making every optimizer, backend, and native implementation part of the trusted computing base?"
source_cutoff: 2026-09-07
status: daily-report
---

# Can eBPF Delegate Native Operations Without Growing Its Trust Boundary?

Suppose an eBPF optimizer recognizes a sequence that can execute as one native CPU instruction. The portable BPF sequence passes the verifier, the target machine supports the fast path, and the loader selects the right implementation. There is still one question left: **what proves that the native code actually does only what the verified BPF sequence was allowed to do?**

This matters because the verifier reasons about BPF instructions and verifier-described calls, not arbitrary machine instructions inserted later by an optimizer. Once a specialization mechanism can replace verifier-visible work with native code, a bug in that replacement can become a kernel bug even when the original BPF program was safe.

This report argues for a narrow design rule: **native delegation should carry a verifier-visible semantic witness and a separately checkable implementation certificate, so adding an optimization does not automatically add its compiler and generator to the trusted computing base.**

<!-- more -->

The previous report on [architecture-specific eBPF specialization](https://eunomia.dev/research/ebpf-portable-architecture-specialization/) asked which native implementation is eligible on a machine and what happens when no fast path exists. That is a portability problem. Here, assume the eligible implementation has already been selected correctly. The problem is whether selecting it silently enlarges the set of code whose correctness the kernel must trust.

The earlier report on [runtime profile-guided specialization](https://eunomia.dev/research/ebpf-runtime-profile-specialization/) asks whether an optimization is still semantically justified when workload assumptions change. Here the assumptions may be timeless. A wrong native implementation can violate semantics on the very first execution.

## The verifier can police a call boundary without proving the callee

Linux already has a useful example in BPF kernel functions, or kfuncs. A kfunc exposes a kernel function to BPF programs. The current Linux documentation makes two properties explicit.

First, the verifier can enforce a rich calling contract. Pointer arguments are trusted by default, BTF types constrain what may be passed, and flags such as `KF_ACQUIRE`, `KF_RELEASE`, `KF_RET_NULL`, `KF_SLEEPABLE`, `KF_RCU`, and `KF_DESTRUCTIVE` tell the verifier about lifetime, nullability, execution context, and unusually dangerous effects. For example, `KF_ACQUIRE` makes the verifier prove that a returned reference is eventually released or transferred into supported map state.

Second, the implementation behind that boundary remains ordinary kernel code. Linux says an existing kernel function can be registered directly as a kfunc, while maintainers must still review the context in which BPF can invoke it and whether doing so is safe. The verifier guarantees valid arguments and tracks declared properties, but those declarations are not a proof of the function body.

That split is productive. The verifier does not need to interpret all of Linux to check every BPF program. But it also exposes the trust question for optimizer-defined native operations: a precise call-site contract can keep BPF safe only if the delegated implementation respects the contract it advertises.

The same distinction appears in the general verifier documentation. The verifier symbolically executes BPF instructions and checks function-call argument constraints. After a call, it updates register state according to the function prototype. This is an interface-level proof boundary, not a machine-code equivalence proof for the callee.

## Kops makes the extra trusted code visible

[Kops](https://arxiv.org/abs/2606.24213) is a useful case because it is explicit about the trade-off. A Kops operation has a proof sequence made from ordinary eBPF instructions and a native emit used by an architecture-specific JIT path. The ordinary sequence remains verifier-visible. The native emit is the per-operation code that must be trusted to implement the same semantics.

For its hardware-idiom operations, Kops uses Lean 4 proofs to establish equivalence between a native emit and its BPF proof sequence. The paper reports up to 24% speedup on microbenchmarks and up to 12% on production applications. It also demonstrates whole-program native replacement, which reaches 2.358x in its evaluation but explicitly comes with a larger trusted computing base.

That result sharpens the engineering choice. Faster native delegation is not one mechanism with one fixed safety cost. The trust cost depends on how much verifier-visible semantics remain and how much native code is accepted on faith.

Linux's JIT controls point in the same direction from another angle. The kernel can enable JIT hardening to mitigate JIT spraying, and it provides privileged mechanisms for inspecting generated JIT code. Those controls do not prove semantic equivalence, but they show that generated native code is already treated as a security-sensitive execution artifact rather than an irrelevant implementation detail.

## The missing contract is between semantic authority and implementation authority

It helps to separate two questions that are often collapsed.

A **semantic authority** says what an operation is allowed to do. For a native BPF operation, that could be an ordinary BPF proof sequence plus an explicit effect summary: which registers are inputs and outputs, which context or map memory may be touched, whether helpers or kfuncs may be called, and whether the operation can sleep, allocate, acquire references, or produce externally visible effects.

An **implementation authority** says why this particular native body may stand in for that semantic operation. It should bind a concrete implementation hash to the semantic witness and to evidence that can be checked independently of the optimizer that generated it.

A minimal package could look like this:

```text
semantic_op = rotate64_v1
proof_seq_hash = sha256(portable_bpf_sequence)
effects = {read: r1,r2; write: r0; memory: none; calls: none}

native_impl_hash = sha256(machine_code)
target = x86_64
certificate = proof-or-translation-validation-result
checker_version = v3
```

The exact representation is not the point. The important property is separation of duties. A userspace optimizer may search aggressively, a backend may generate machine code, and a small checker may decide whether the result is authorized. If the optimizer is wrong, the checker should reject the implementation rather than turn the optimizer into part of kernel safety.

This is different from the capability manifest in the previous report. Capability negotiation answers, "May this implementation run here?" A trust contract answers, "Why is this implementation allowed to have the same authority as the verified semantic operation?"

## Where current work is still weak

### Kfunc-style contracts describe boundary obligations, not implementation equivalence

Linux kfunc metadata is increasingly expressive about pointer trust, ownership, RCU state, sleepability, and destructive behavior. That helps the verifier reject unsafe calls. It still relies on kernel review and implementation correctness behind the boundary.

For stable in-tree kernel functions, that may be the right engineering trade-off. It scales less comfortably to a model where userspace optimizers or third-party modules can introduce many machine-specific operations. Review-only trust grows roughly with every new implementation and backend.

The missing evidence is whether a compact semantic/effect contract plus an independent checker can catch implementation mistakes that call-site typing cannot. A fault-injection study should mutate native bodies while leaving their advertised kfunc-like signature unchanged.

### Proof-linked operations still need a deployable checker boundary

Kops demonstrates that native emits can be related to verifier-visible proof sequences, and its Lean proofs are strong evidence for the evaluated operations. A production extension mechanism still has to decide what is checked at build time, load time, kernel integration time, or not at all.

If every proof must be trusted because it came from a particular compiler pipeline, the trusted base has merely moved. If a tiny proof checker or translation validator can independently reject a malformed implementation, the generator can stay outside the trusted base.

The missing benchmark is not another speed comparison. It is trusted-code growth versus escaped semantic bugs as the operation library and architecture matrix scale.

### Whole-program native replacement changes the failure radius

Replacing a small arithmetic idiom with one native instruction and replacing an entire BPF program with native code are both "native specialization," but their failure modes are not comparable. A small operation may have no memory effects. A whole program can touch packet or context memory, call helpers, update maps, and branch through many states.

A useful trust model therefore needs an effect envelope, not only an equivalence label. If the checker cannot prove the full implementation, the runtime should at least know which classes of effects the native body is permitted to exercise and fail closed when it exceeds that envelope.

## Promising directions with academic and production value

### 1. Make native operations carry independently checkable certificates

The first artifact should define a small certification interface for delegated BPF operations.

Each operation carries a verifier-visible proof sequence, a stable semantic identity, an explicit effect summary, the native implementation, and a certificate checked by code that is substantially smaller than the optimizer or native generator. The certificate could be a proof object for a restricted operation language, translation-validation evidence, or another representation whose checker can be audited independently.

The strongest baseline is not "no optimization." Compare against ordinary BPF/JIT execution, review-only native modules, and native operations whose generator is fully trusted. Measure accepted optimization coverage, checker time, code size, speedup, and trusted source lines or trusted components. Then inject wrong register results, clobbered callee-saved state, unauthorized memory writes, missing side effects, and backend-specific corner cases.

The academic question is whether the trust of an extensible JIT can scale with a small checker rather than with every optimizer/backend implementation. The production value is the ability to accept third-party or rapidly evolving native optimizations without treating their entire toolchain as kernel-trusted code.

The idea loses if the certificate/checker becomes almost as complex as the optimizer or if nearly all useful operations require semantics too broad for compact checking.

### 2. Enforce a runtime effect envelope around partially verified delegation

Some useful native operations may be too complex for a complete proof. A second design can still reduce their failure radius.

Give each operation an effect envelope derived from verifier-visible semantics: permitted register outputs, memory regions, helper/kfunc call classes, reference-lifetime changes, and control-flow return behavior. The JIT or runtime inserts cheap guards only where native code could cross that envelope. High-risk operations can additionally run differential canaries against the portable BPF form during deployment or after a kernel/JIT update.

This is not a portability fallback mechanism. The target is already capable of running the native operation. The mechanism asks whether a buggy implementation can escape the authority it was delegated. On a violated guard or differential mismatch, the runtime disables that implementation generation and records the failure.

Evaluate with deliberately malformed native operations and realistic hardware idioms. Measure semantic escapes, detection latency, guard overhead, false positives, and the percentage of operations that can be contained without full proof. Compare full proof, effect-only containment, shadow execution, and review-only trust.

The production user is an operator running an extensible re-JIT or BPF runtime across frequent kernel and optimizer updates. The design is not worthwhile if effect guards cost more than the native optimization saves or if most important semantic failures occur entirely inside the declared envelope.

### 3. Build a TCB mutation benchmark for eBPF specialization

Current optimization evaluation naturally emphasizes verifier acceptance, code size, and runtime. A trust-boundary benchmark should make the optimizer fail on purpose.

Start from a corpus of verifier-accepted BPF programs and certified native operations. Generate controlled mutations: wrong arithmetic flags, stale architecture assumptions, register clobbers, unauthorized loads/stores, missing reference releases, extra helper calls, incorrect exceptional cases, and malicious certificates. Run them under several trust designs: stock BPF/JIT, kfunc-style typed boundary, review-only native delegation, proof-linked delegation, and proof-linked delegation plus effect containment.

The primary metric is **escaped semantic violation**, not crash rate. A mutation fails the system if native execution produces an observable state that the verifier-approved semantic program could not produce. Secondary metrics include trusted-code growth per new operation/backend, certification latency, optimization coverage, runtime overhead, and debugging evidence after rejection.

Such a benchmark would let systems papers make a stronger claim than "our extension is safe by construction." It would show which faults the construction actually excludes and where trust still accumulates. In production, the same corpus can become regression tests for new JIT backends and operation libraries.

The benchmark is less useful if independent implementations almost never expose failures beyond what normal kernel testing already catches. That outcome would itself argue for simpler review-and-test processes rather than a new certification layer.

## What would change this conclusion?

The need for a separate native-operation trust contract becomes much weaker if profitable specialization can remain entirely in verifier-visible BPF. EPSO and other BPF-level optimizers show why that simpler design is attractive: optimized bytecode can still pass through the existing verifier and stock JIT. If native operations add little performance beyond such rewrites, moving the trust boundary is unnecessary.

The conclusion also weakens if native delegation remains a tiny, stable, in-tree mechanism maintained like other kernel code. Linux already relies on review for kfunc implementations and JIT backends. If an optimization interface exposes only a handful of mature operations and never becomes a third-party extension surface, the existing kernel trust model may be sufficient.

Finally, an empirical mutation study could show that a small checker adds little protection. If almost every realistic implementation fault is already caught by compiler validation, architecture tests, verifier-side contracts, and ordinary kernel review, certificate complexity would not pay for itself.

But an extensible optimizer changes the scaling assumption. Kfuncs show that the verifier can enforce precise obligations at a trusted call boundary. Kops shows that native operations can retain verifier-visible semantic witnesses and that larger native replacement buys more speed at the cost of a larger TCB. **The next abstraction should make that cost explicit and checkable: delegate authority to a native implementation only when independent evidence binds it to the semantics the verifier approved.**

## References

- Linux kernel documentation. [eBPF verifier](https://www.kernel.org/doc/html/latest/bpf/verifier.html), accessed 2026-09-07.
- Linux kernel documentation. [BPF Kernel Functions (kfuncs)](https://www.kernel.org/doc/html/latest/bpf/kfuncs.html), accessed 2026-09-07.
- Linux kernel documentation. [Documentation for /proc/sys/net/](https://kernel.org/doc/html/latest/admin-guide/sysctl/net.html), accessed 2026-09-07.
- Yusheng Zheng et al. [Kops: Safely Extending the eBPF Compilation Pipeline with Native Operations](https://arxiv.org/abs/2606.24213), 2026.
- Qian Zhu et al. [EPSO: A Caching-Based Efficient Superoptimizer for BPF Bytecode](https://arxiv.org/abs/2511.15589), 2025.
