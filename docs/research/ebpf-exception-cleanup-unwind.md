---
date: 2026-09-20
slug: ebpf-exception-cleanup-unwind
title: "Can eBPF Exceptions Unwind Resources Safely?"
description: "A new BPF cleanup-table design lets bpf_throw unwind owned resources. This report examines verifier contracts, JIT boundaries, and tests for safe cleanup."
tags:
  - Daily Report
  - eBPF
  - Linux
  - Verifier
  - Compilers
  - Rust
research_question: "What proof and runtime contract are required for bpf_throw() to unwind frames that own kernel resources without leaking verifier-tracked state or diverging across JITs?"
source_cutoff: 2026-09-20
status: daily-report
---

# Can eBPF Exceptions Unwind Resources Safely?

Consider a BPF function that enters an RCU read-side critical section, constructs a value whose lifetime owns that lock, and then calls another BPF function. The callee discovers a fatal condition and invokes `bpf_throw()`.

The ordinary Rust expectation is straightforward: unwinding runs `Drop`, and `Drop` releases the guard. The current BPF expectation is different. `bpf_throw()` discards intervening BPF frames. If one of those frames still owns a lock, reference, or other verifier-tracked resource, the verifier rejects the program rather than allow an unwind that would leak the obligation.

That restriction is visible in today's user-facing documentation: throwing with lingering resources leads to a verification error. It is not merely an ergonomic inconvenience. It is the verifier enforcing a lifetime invariant when the runtime has no place to execute cleanup code.

A 20-patch `bpf-next` series posted on September 16 changes that design boundary. It proposes a compiler-generated `.bpf_cleanup` table, verifier edges from covered call sites to cleanup landing pads, runtime dispatch of those pads while `bpf_throw()` walks the stack, and a `bpf_unwind_resume()` kfunc that returns control to the unwinder. LLVM 23 already contains the compiler-side support for emitting the cleanup table.

This is a substantial step toward language-level resource management in BPF. It also creates a new systems question: **what does it take to prove that compiler-generated cleanup, verifier reasoning, libbpf transport, and architecture-specific JIT unwinding all describe the same resource-lifetime transition?**

<!-- more -->

## The current restriction is about resource lifetime, not exception syntax

BPF already has an exception mechanism. `bpf_throw()` was introduced in Linux 6.7 and terminates the current execution by unwinding the BPF call stack to an exception boundary. The supplied cookie becomes the program result under the default callback path, or an exception callback can transform it where that mode is supported.

The difficult part is what happens to obligations owned by discarded frames.

The verifier is not only a memory-safety checker for individual instructions. It also tracks resources whose lifetime must be balanced. A reference acquired by one operation must eventually be released. The Linux verifier documentation illustrates the same principle with socket references: exiting with an unreleased reference is rejected. BPF kfunc APIs similarly encode acquire/release relationships for task references, cgroup references, RCU read-side sections, and other objects.

An exception therefore cannot be treated as a control-flow shortcut that simply jumps to the end. Suppose the verifier state at one frame contains:

```text
RCU read lock: held
referenced kptr: owned
preemption: disabled
```

If `bpf_throw()` just drops the frame, those obligations disappear from the program's control flow without disappearing from kernel state. Rejecting the throw is the safe answer when there is no executable cleanup path.

That is why this problem matters beyond Rust. Rust makes the mismatch obvious because RAII and `Drop` express cleanup structurally, but C++ destructors or a compiler-generated cleanup dialect would face the same requirement. Even hand-written BPF benefits if the verifier can represent a non-local exit without pretending resources ceased to exist.

## The September proposal gives cleanup a first-class control-flow path

The new patch series connects four pieces that previously did not share one cleanup model.

First, LLVM's BPF backend can lower an `invoke` with a cleanup landing pad and emit a `.bpf_cleanup` section. Each entry is a 12-byte triple:

```text
(begin, end, landing_pad)
```

The interval identifies a call-site region. If unwinding crosses a frame whose return address falls inside that region, the corresponding landing pad should run before the frame is discarded. LLVM PR #192164, merged in April 2026, added the BPF backend support needed to preserve those cleanup edges and emit the table.

Second, libbpf in the proposed kernel series learns to collect those records, pass them at `BPF_PROG_LOAD`, resolve the compiler's `_Unwind_Resume` to the kernel-side `bpf_unwind_resume()` kfunc, and preserve the information through light skeletons and static linking.

Third, the verifier is taught that a covered call has another possible successor: the cleanup landing pad. It explores the pad and updates the resource state according to the cleanup operations it sees. In the motivating RCU example, the verifier can reason that `bpf_rcu_read_unlock()` actually discharges the lock before unwinding continues.

Fourth, the runtime path mirrors that model. `bpf_throw()` already uses architecture support to walk BPF frames. The proposed design looks up the return PC in each frame's cleanup table, runs a matching pad, and resumes the unwind. The first series supplies x86-64 and arm64 JIT support.

This symmetry is the strongest part of the design. The verifier does not merely trust a side table saying "cleanup happened." It follows the cleanup code. The runtime does not invent a separate unwinding language. It dispatches the same pad whose control-flow effect the verifier analyzed.

## The proposal is deliberately narrower than general-purpose exceptions

It would be easy to read `.bpf_cleanup` as "BPF gets C++ exceptions." That is too broad.

The posted series supports cleanup pads, not catch pads. A cleanup pad is expected to release resources and then resume unwinding. It does not regain ordinary execution of the discarded frame. The proposal also requires a JIT capable of dispatching cleanup pads. The first version covers x86-64, where the existing exception path depends on ORC unwinding, and arm64. Unsupported JIT targets reject the load with `-EOPNOTSUPP` instead of silently omitting cleanup.

The series also rejects combinations and shapes the runtime cannot safely dispatch: offloaded programs, private stacks, combining a cleanup table with an exception callback, and several operations inside pads such as tail calls, indirect jumps, or on-stack call arguments.

Most importantly, the Rust toolchain is not yet a complete production path for BPF exception handling. The patch-series selftests hand-write the equivalent cleanup records and landing pads so the kernel and libbpf paths can be tested without claiming the language integration is finished.

These limitations are healthy. They keep the initial claim testable: **a covered BPF call may unwind through a verifier-checked cleanup pad on supported JITs**. They do not claim arbitrary language exceptions, portable catch semantics, or automatic safety for every form of cleanup.

## The unresolved gap is cross-layer cleanup correctness

The proposal closes the obvious resource-leak hole, but it also creates a correctness boundary spanning the compiler, ELF metadata, libbpf, verifier, JIT, architecture stack walker, and runtime exception path.

For a normal BPF branch, instruction bytes and verifier state largely define what happens. For cleanup unwinding, a call site's behavior depends on side metadata generated by a compiler and transformed by userspace tooling before the kernel consumes it. At runtime, an architecture-specific JIT and stack walker must map a machine return PC back to the same logical region the verifier associated with the landing pad.

A correct unwind therefore needs several statements to be true together:

```text
compiler region
    == libbpf-carried region
    == verifier-covered call region
    == JIT/runtime PC region

and

cleanup resource transition seen by verifier
    == cleanup resource transition executed at runtime
```

The kernel proposal is designed around this equivalence, but there is not yet a general deployment or testing abstraction that makes the equivalence explicit and easy to falsify across toolchain versions and architectures.

This boundary is different from [kernel capability admission](https://eunomia.dev/research/ebpf-kernel-capability-evidence/). Capability admission asks whether an artifact can use a feature on one target. It is also different from [cross-kernel semantic compatibility](https://eunomia.dev/research/ebpf-kernel-upgrade-semantic-compatibility/), which asks whether an admitted application preserves behavior after a kernel upgrade. Here the question is narrower and lower-level: whether a single non-local control-flow operation preserves the verifier's resource-lifetime proof when that proof crosses compiler metadata and JIT runtime machinery.

It also touches the trust boundary discussed in [native eBPF operations](https://eunomia.dev/research/ebpf-native-operation-trust-boundary/). A landing pad is verified BPF code, but correct dispatch still depends on trusted native unwinding machinery. The interesting research target is therefore not another exception syntax. It is a contract that makes this cross-layer transition auditable.

## Research direction 1: give cleanup pads an explicit resource-effect discipline

The first direction is to make the intended role of a landing pad visible as a verifier property rather than only as a collection of syntactic restrictions.

**Gap.** The runtime needs cleanup pads to retire obligations owned by the unwinding frame. A pad that acquires new long-lived resources, creates a new ownership cycle, or transfers an obligation somewhere the unwinder cannot represent would make the semantics much harder to reason about. The initial patch series constrains unsafe control-flow shapes, but the deeper invariant is about resource effects.

**Mechanism.** Define a cleanup-effect discipline over verifier state. For each landing pad, compute the resource set on entry and exit. A conservative first contract would require the pad to be monotonic with respect to unwind obligations: it may discharge resources already owned by the frame and perform bounded temporary operations that are fully balanced inside the pad, but it may not leave a new live reference, lock, preemption-disable state, iterator, or other tracked obligation when it calls `bpf_unwind_resume()`.

Conceptually:

```text
owned_at_resume ⊆ owned_at_pad_entry
```

with explicit rules for temporary acquire/release pairs that cancel before resume.

This does not need a new source-language annotation in the first prototype. It can be derived from the verifier's existing resource state while it explores the landing pad. The verifier could produce diagnostics in terms of the violated cleanup effect: "landing pad introduces reference id N" is more useful than a generic unsupported unwind shape.

**Delta from the current proposal.** The posted series proves individual paths using existing verifier machinery. The proposed discipline names the invariant that all accepted pads should satisfy and could eventually replace some ad hoc restrictions with one resource-state rule.

**Prototype.** Extend the cleanup selftests with a small matrix of owned resources: RCU locks, preemption-disable sections, referenced kptr/task objects, and iterator-like lifetime state where applicable. Generate pads that release correctly, double-release, acquire a fresh resource, transfer ownership through a map or kptr, or conditionally leave an obligation live.

**Evaluation.** Measure verifier acceptance against a hand-labeled expected set. Fuzz nested resource combinations and cleanup order, then compare the effect-based rule with the initial syntactic restrictions. The useful outcome is fewer accidental rejects without any accepted program reaching `bpf_unwind_resume()` with an unmatched obligation.

**Academic value.** This would make non-local BPF control flow a resource-logic problem with a stated invariant rather than a growing list of special cases.

**Production value.** Compiler and language-runtime developers would get a stable target for what generated cleanup is allowed to do, plus diagnostics that map to RAII/`Drop` obligations.

**Failure condition.** If the existing verifier path exploration already enforces the same invariant with equally precise diagnostics and no meaningful restriction pressure, a separate cleanup-effect abstraction adds complexity without value.

## Research direction 2: bind cleanup metadata to an auditable artifact contract

The second direction is about provenance and diagnosis rather than trusting the compiler more.

**Gap.** `.bpf_cleanup` is generated by the compiler, can pass through static linking or light-skeleton generation, and is then translated into kernel load metadata. The kernel must continue to verify the actual code, but when something fails it can be difficult to tell whether the compiler described one region, a linker transformed another, libbpf carried a stale table, or the target kernel rejected the shape for an architecture-specific reason.

**Mechanism.** Produce a normalized cleanup descriptor alongside the loaded artifact. The descriptor is evidence, not authority. It could contain the object digest, compiler and backend identity, subprogram and call-site region IDs, landing-pad offsets before and after linking, the verifier-observed cleanup outcome, and the target JIT capability used for admission.

For diagnostics, the loader could expose a record like:

```text
artifact: sha256:...
compiler: llvm-bpf ...
cleanup_region: subprog=foo callsite=3
object_range: [0x..., 0x...)
landing_pad: 0x...
verifier_result: accepted
resource_delta: rcu_lock 1 -> 0
jit_dispatch: x86_64/orc
```

The descriptor must never let compiler metadata bypass verification. Its purpose is to tie the compiler's intention, userspace transformation, kernel interpretation, and target runtime capability to one reproducible artifact.

**Delta from current tooling.** libbpf's proposed support carries the table correctly; this direction asks for an inspectable record of how the table changed and how the kernel interpreted it. It extends the provenance idea beyond "feature present" to "this exact cleanup region was accepted with this exact resource transition."

**Prototype.** Add a debug/export mode to an experimental libbpf loader that serializes normalized cleanup records before load and combines them with verifier logs or structured verifier-side identifiers after load. Exercise normal ELF loading, static linking, and light skeleton generation.

**Evaluation.** Build the same cleanup corpus with multiple LLVM revisions and optimization levels, run it through different libbpf/linker generations, and intentionally corrupt or stale one stage at a time. Measure whether the descriptor localizes mismatches faster and more accurately than raw verifier logs plus `llvm-objdump` inspection.

**Academic value.** This tests whether proof-relevant compiler metadata can remain auditable across a multi-stage systems toolchain without becoming trusted proof by declaration.

**Production value.** A failed deployment could say whether the mismatch came from compiler generation, object transformation, kernel admission, or JIT support instead of reducing everything to "exception cleanup unsupported."

**Failure condition.** If almost every failure is already local and obvious from existing verifier/libbpf diagnostics, or if the descriptor changes too often to be stable across normal toolchain releases, the provenance layer is not worth maintaining.

## Research direction 3: differential-test the verifier and JIT unwind as one mechanism

The third direction targets the most dangerous class of failure: verifier/runtime disagreement.

**Gap.** Cleanup is accepted because the verifier follows a landing pad and updates resource state. Runtime safety depends on the architecture JIT and stack walker dispatching the corresponding pad for the same frame and call-site region. An architecture-specific off-by-one PC range, unusual prologue, nested exception, or toolchain transformation could turn a verifier-approved cleanup into a runtime skip or misdispatch.

**Mechanism.** Build an unwind conformance harness that injects `bpf_throw()` at every eligible call boundary in a generated program family. Each test records the verifier's expected resource transitions and a runtime witness showing which pads executed and in what order. Compare the final state rather than only the return cookie.

The matrix should vary:

- nested call depth and multiple cleanup regions per subprogram;
- one and several simultaneously owned resources;
- conditional cleanup paths;
- boundary PCs at the start and end of each `[begin, end)` region;
- static-link and light-skeleton transformations;
- x86-64 and arm64, then any later JIT that implements the ABI;
- unsupported combinations that must deterministically fail load.

A kernel test VM can additionally instrument resource-specific invariants, such as proving the RCU lock or reference is actually retired after the throw.

**Delta from ordinary selftests.** The posted series already adds extensive end-to-end and rejection selftests. The proposed harness makes cross-architecture differential behavior and generated boundary coverage the primary test objective, with the verifier's accepted resource transition used as the oracle to challenge the runtime implementation.

**Prototype.** Start from the series' hand-authored cleanup programs, generate region permutations and nested ownership patterns, and run the same corpus in x86-64 and arm64 VMs. Record verifier result, cleanup execution trace, final resource state, and program result.

**Evaluation.** Seed faults in cleanup range lookup, JIT return-PC normalization, pad dispatch, or resume handling. Score detection rate and false positives. Then run the unmodified implementations across kernel/JIT revisions to measure whether the harness catches architecture-specific regressions that ordinary positive tests miss.

**Academic value.** This is a concrete instance of differential testing between a static safety model and the native runtime mechanism that is supposed to realize that model.

**Production value.** Architectures and distribution kernels could gate enabling exception cleanup on a reproducible conformance result rather than assuming that a successful build implies equivalent unwind behavior.

**Failure condition.** If existing BPF selftests already provide equivalent cross-JIT boundary coverage and seeded verifier/runtime mismatches are always caught without the differential harness, the additional test system would be redundant.

## Practical deployment guidance today

This feature should currently be treated as active kernel/toolchain development, not a portable production contract.

For production BPF programs today, explicit cleanup and error-return paths remain the conservative default when a frame owns verifier-tracked resources. Existing kernels intentionally reject `bpf_throw()` when those obligations remain live.

For experiments with cleanup unwinding, pin the exact kernel series, LLVM revision, libbpf revision, architecture, and JIT configuration. Test the actual loaded artifact, including any static-link or light-skeleton path, and include at least one resource-specific runtime witness rather than checking only that the program loads or returns the expected cookie.

A language runtime should also fail closed. If the target does not support the cleanup-table ABI or the required JIT dispatch, it should reject the unwind-capable artifact or select a separately validated error/abort implementation. It should not silently compile RAII cleanup under the assumption that `Drop` will run.

The eventual production boundary should be simple to state: **a target may enable unwind cleanup only when the kernel verifies the cleanup path and the runtime/JIT is known to dispatch the same path for the same call-site region.**

## What would change this conclusion?

Three findings would reduce the need for stronger cleanup contracts and conformance machinery.

First, if the final kernel implementation proves that ordinary verifier path exploration already provides a complete and clean resource-effect invariant for landing pads, with no growing set of special-case restrictions or ambiguous diagnostics, a separate cleanup-effect discipline would be unnecessary.

Second, if compiler, linker, libbpf, and kernel transformations preserve cleanup metadata through one mechanically checked representation whose mismatches are already diagnosed precisely, an additional artifact-level provenance descriptor would add little.

Third, if broad x86-64 and arm64 fault-injection testing shows that cleanup-region dispatch is mechanically tied to JIT code generation in a way that makes verifier/runtime divergence practically impossible, a separate differential promotion gate could be excessive.

The current evidence does not establish those stronger claims yet. The patch series is intentionally new and under review, the Rust BPF exception path is not yet a complete production toolchain, and the feature crosses several layers that historically evolve independently.

## Conclusion

The interesting part of BPF exception cleanup is not that `bpf_throw()` can become friendlier to Rust. It is that non-local control flow is being connected to the same resource accounting that makes verifier-approved BPF safe.

The September proposal has the right basic shape: the compiler marks a cleanup region, libbpf carries it, the verifier explores the landing pad, and the JIT/runtime dispatches that pad while unwinding. That can turn an impossible program, such as throwing while an RCU guard is live, into one whose cleanup is both statically checked and actually executed.

The next problem is making the cross-layer equality falsifiable. A cleanup pad should have a clear resource effect, its metadata should remain auditable through the toolchain, and verifier-approved cleanup should be differential-tested against the architecture runtime that executes it. If those pieces hold, BPF exceptions can gain RAII-style cleanup without weakening the central rule that every owned kernel resource must have a verifier-visible lifetime.

## Sources

- Yonghong Song, `[PATCH bpf-next 00/20] bpf: Run exception cleanup landing pads when bpf_throw() unwinds`, September 16, 2026: https://lwn.net/Articles/1095028/
- LLVM PR #192164, `[BPF] Add exception handling support with .bpf_cleanup section`, merged April 16, 2026: https://github.com/llvm/llvm-project/pull/192164
- eBPF Docs, `bpf_throw`: https://docs.ebpf.io/linux/kfuncs/bpf_throw/
- Linux kernel documentation, eBPF verifier: https://docs.kernel.org/bpf/verifier.html
- Linux kernel documentation, BPF kfuncs: https://docs.kernel.org/bpf/kfuncs.html
