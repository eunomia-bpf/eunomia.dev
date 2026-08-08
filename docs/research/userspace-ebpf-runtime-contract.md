---
date: 2026-08-08
title: "What Is Missing Before Userspace eBPF Becomes a Real Runtime?"
description: "Userspace eBPF can run bytecode outside the kernel, but a real runtime also needs portable attach, safety, state, capability, and lifecycle contracts."
tags:
  - Daily Report
  - eBPF
  - Userspace eBPF
  - Runtime Systems
  - bpftime
research_question: "What runtime contract is still missing if eBPF programs are expected to move between the Linux kernel, userspace runtimes, and other execution backends without losing understandable attach, safety, state, and lifecycle semantics?"
source_cutoff: 2026-08-08
status: daily-report
---

# What Is Missing Before Userspace eBPF Becomes a Real Runtime?

Suppose an operator wants to trace a database process without adding application instrumentation and without paying the cost of sending every event through a kernel probe. A userspace eBPF virtual machine solves one part of the problem: it can execute BPF instructions inside the process or beside it. But the first operational questions arrive before the first instruction runs. What exactly does the program attach to? Which version of the target did it inspect? Which helpers and memory regions may it access? Who owns its maps? What happens when the target library is replaced, the process exits, or the extension must be revoked?

Those questions are not instruction-set questions. They are runtime questions.

The BPF ecosystem has already standardized the instruction set well enough for an IETF specification, [RFC 9669](https://www.rfc-editor.org/rfc/rfc9669.html), to describe the BPF ISA independently of one implementation. Linux eBPF, however, is useful for much more than executing those instructions. Linux also provides program and map objects, verifier rules, helper and kfunc interfaces, attach types, BPF links, pinning, lifetime rules, capability checks, and user-space APIs that give an eBPF application a concrete operating model.

<!-- more -->

This report argues that **first-class userspace eBPF needs a portable execution contract above the ISA and below each runtime backend**. That contract should describe attachment identity, available capabilities, state ownership, lifetime and failure behavior, and resource attribution. It does not need to make every backend identical. Its job is to make the differences explicit enough that an application can know what it is relying on and a runtime can test whether it provides those semantics.

That distinction matters because userspace eBPF is no longer one implementation pattern. [uBPF](https://github.com/iovisor/ubpf) is an embeddable VM with an interpreter, JITs, helper registration, and a newer safe execution profile. [bpftime](https://github.com/eunomia-bpf/bpftime) is a larger userspace runtime that adds loaders, maps, helpers, verification, attach backends, `bpf_link`-style objects, and compatibility with existing libbpf and bpftrace workflows. [eBPF for Windows](https://github.com/microsoft/ebpf-for-windows) reuses the BPF instruction format and familiar libbpf APIs but supplies a Windows-specific verifier, execution environment, hooks, helpers, and native/JIT execution choices. These systems share an instruction language while exposing different runtime contracts.

## A BPF VM is only one layer of the runtime

The easiest way to see the missing layer is to separate three questions that are often collapsed into "does this support eBPF?"

First, **can the backend execute BPF instructions correctly?** RFC 9669 gives implementations a common ISA target. A VM can interpret those instructions or JIT them to native code.

Second, **what execution environment does the program see?** An eBPF program is written against more than registers and opcodes. It expects a context type, helper functions, map operations, memory rules, return-value semantics, and often a specific event source. Two engines can implement the same ISA and still disagree on every one of those interfaces.

Third, **how does an operator manage the program as a live system object?** The answer includes loading, verification, attachment, discovery, replacement, detachment, state ownership, authority, diagnostics, and cleanup. These operations determine whether a runtime is deployable under failures and upgrades rather than merely able to run a function.

uBPF illustrates the boundary clearly. Its own README describes it as a library for executing eBPF programs and provides an assembler, disassembler, interpreter, and JITs. Its safe profile adds pointer provenance, checked dereferences, typed helper metadata, and registered external regions. Those are meaningful execution-safety semantics. The embedding application still decides what an event is, which target should trigger a program, how a program is named and revoked, and how several VM instances share or isolate state.

That is not a weakness in uBPF. It is what an embeddable VM should do. The problem appears when the ecosystem treats VM compatibility as runtime compatibility.

## Linux eBPF already exposes a richer operating model

Linux makes the extra layers easy to overlook because they are part of the platform.

The kernel's [libbpf lifecycle documentation](https://docs.kernel.org/bpf/libbpf/libbpf_overview.html) separates an application into open, load, attach, and tear-down phases. During loading, maps are created, relocations are resolved, programs are verified, and initial state can be established before execution. During attachment, programs become connected to concrete hook points. Tear-down releases those relationships and resources.

The [`bpf()` userspace API](https://docs.kernel.org/userspace-api/ebpf/syscall.html) gives programs and maps file-descriptor-based lifetimes and supports persistent objects through pinning. BPF links make the relationship between a loaded program and an attachment point into an object with its own lifetime. Newer BPF token support goes further: a token can carry allowed BPF commands, map types, program types, and attach types, so delegated loading is tied to an explicit capability set rather than an all-or-nothing privilege assumption.

None of these mechanisms is part of the BPF ISA. Yet applications rely on them every day.

This suggests a useful definition: **a runtime is first-class when an operator can reason about a program before, during, and after execution, not only about the instructions while they execute.**

## Userspace execution changes the attachment problem

Moving eBPF into userspace is attractive because it can remove a kernel crossing from hot event paths, operate where kernel BPF is unavailable, and expose application-local functions or memory more directly. It also removes many assumptions that the kernel previously supplied.

Consider a uprobe. In kernel eBPF, the kernel and tracing stack mediate an attachment to a process or binary location. A userspace runtime based on dynamic binary rewriting may instead patch a function entry, a syscall path, a language runtime, a packet-processing loop, or even device code. The target might be identified by symbol, build ID, module offset, runtime method, or driver-specific event identifier. The target can disappear or be replaced independently of the eBPF program.

A portable attach description therefore needs more than "program type = uprobe." It needs enough identity to answer questions such as:

- Which process, executable, module, function, or event generation was selected?
- What must still be true when the runtime commits the attachment?
- Does attachment survive `exec`, library reload, process fork, or target replacement?
- Can another principal discover, update, or revoke the attachment?
- What cleanup is guaranteed if injection or patching succeeds only partially?

[bpftime's current runtime](https://eunomia.dev/bpftime/documents/attach/) already has to solve many of these problems for uprobes, syscall hooks, XDP-style paths, GPU instrumentation, and pluggable event backends. Its OSDI 2025 work, [Extending Applications Safely and Efficiently](https://www.usenix.org/conference/osdi25/presentation/zheng-yusheng), adds the Extension Interface Model (EIM), which treats required extension features such as memory access or the ability to call an application function as resources that an extension manager can grant. This is a useful step toward an explicit userspace runtime contract because it separates what an extension needs from the mechanism used to execute it.

The remaining question is how much of that contract can become portable across runtimes rather than staying inside one implementation.

## Compatibility needs levels, not a single yes-or-no label

"Compatible with eBPF" can currently mean several different things:

| Compatibility level | What it actually promises | What can still differ |
| --- | --- | --- |
| ISA | The backend executes the same BPF instruction semantics | Context, helpers, maps, attachment, lifecycle |
| Object/toolchain | It can consume ELF/BTF produced by common compilers | Relocations, helper availability, CO-RE target data |
| API | Existing loader code can call familiar libbpf-style operations | Which calls are implemented and their failure semantics |
| Program environment | A program type exposes comparable context and helpers | Target identity, authority, lifetime, performance |
| Operational | Load, attach, update, observe, revoke, and clean up behave predictably | Backend-specific mechanisms and costs |

This layered view explains why eBPF for Windows is interesting evidence. The project intentionally reuses existing eBPF toolchains and libbpf APIs, but its README also documents a Windows-specific hosting layer, hook set, helper surface, verifier path, and execution choices. Source compatibility is valuable, yet it cannot erase differences in what the host operating system can safely provide.

Userspace eBPF should embrace the same reality. A runtime should be able to say, for example, "ISA and ELF compatible, libbpf loader compatible for this subset, uprobe attach semantics at contract version 2, shared-map persistence supported, delegated helper capability supported." That statement is more useful than a broad compatibility badge because it can be tested.

## The runtime contract needs five pieces

A practical contract does not have to standardize every implementation detail. Five pieces carry most of the operational meaning.

### 1. Attachment identity and preconditions

An attachment description should name the target in a form that can be revalidated. For a native process this might include PID namespace identity, executable build ID, module build ID, symbol or offset, and an expected generation. For a packet path it may include interface identity and queue or driver constraints. For GPU instrumentation it may include module and kernel identity.

The runtime should fail closed when those preconditions no longer hold instead of silently attaching to whatever now occupies the same address or name.

### 2. Program capabilities and context

A program needs a declared view of helpers, context fields, maps, memory regions, callable functions, and side effects. Linux encodes much of this through program types, verifier rules, helpers, kfuncs, capabilities, and now BPF tokens. uBPF's safe profile makes helper metadata and external memory regions explicit. EIM makes application extension resources explicit.

A userspace contract can unify the idea without forcing one enforcement mechanism: declare the required capability surface first, then let a backend prove that it can enforce it.

### 3. State ownership and lifetime

Maps and other state need owners and lifetime rules independent of a single VM invocation. A runtime should specify whether state is private to one program, shared between programs, shared across processes, persistent after detach, or imported from another backend. It should also define what happens to state when a program fails validation, attachment fails halfway, or the target exits.

This report stops short of specifying transactional upgrades across programs, links, and maps. That deserves its own analysis. The immediate requirement is simpler: a runtime must expose enough object and lifetime semantics that a transactional design can be built at all.

### 4. Authority and revocation

Who is allowed to attach which program to which target, and who can revoke it later? In an embedded library, the host application may be the authority. In a system runtime, a daemon, container boundary, user namespace, or policy service may delegate only a subset of operations.

The important property is that the authority becomes part of the attachment state. Checking permission only when a loader starts is not enough for a long-lived extension whose target or capability set can change.

### 5. Resource and effect attribution

Kernel eBPF benefits from shared kernel accounting and well-known inspection tools, even though its accounting is not perfect. A userspace runtime may execute JIT code inside another process, allocate shared maps, patch instructions, register callbacks, or consume event buffers. Without per-extension attribution, an operator can observe that the target process became slower but cannot tell which extension consumed CPU time, memory, event bandwidth, or helper calls.

A first-class runtime should expose a stable identity that joins attach state, runtime cost, faults, and effects. This becomes more important when multiple independent extensions share one process, which is the next composition problem in this series.

## When userspace eBPF is actually the right abstraction

Not every instrumentation problem needs this machinery.

Kernel uprobes remain a strong default when the event rate is moderate, the required data can cross the kernel boundary cheaply enough, and Linux provides the needed visibility. Dynamic binary instrumentation frameworks are better when an operator needs arbitrary instruction rewriting and does not need eBPF compatibility. Language runtimes often provide richer semantic hooks when all targets live inside one VM such as the JVM or a managed tracing API.

Userspace eBPF becomes more compelling when several conditions coincide: existing eBPF tooling or programs are worth reusing; events are hot enough that the kernel path is material; the target exposes useful application-local state; kernel BPF is unavailable or too restricted; or the same extension model must span process, networking, GPU, and other backends.

That is also the strongest argument against over-standardizing. If deployments only need one runtime on one platform, an implementation-specific API may be simpler and better. A portable contract earns its complexity only when programs, control planes, or policies need to move across backends or when operators need consistent safety and lifecycle guarantees.

## Where current work is still weak

### Runtime semantics are not part of ISA conformance

RFC 9669 gives the ecosystem a common instruction language, but it deliberately does not define a full host environment. As a result, two implementations can both be correct BPF engines while disagreeing on context layout, helper behavior, map semantics, attach identity, failure handling, and lifetime.

A useful test would run the same contract-focused workload across Linux, uBPF embeddings, bpftime, and eBPF for Windows, then classify which failures are ISA failures and which are host-contract mismatches. Today there is no widely used conformance suite for that second category.

### Compatibility claims are difficult to compare

Projects often report compatibility in terms of accepted object files or unchanged tooling. Those are important adoption properties, but they do not tell an operator whether detachment, target replacement, helper errors, state sharing, or revocation behave the same way.

The missing evidence is a capability and lifecycle matrix backed by executable tests. If most real applications pass unchanged across runtimes once they can load, then the broader contract proposed here is unnecessary. If operational failures cluster after loading succeeds, the missing layer is real.

### Safety is expressed through backend-specific mechanisms

Linux uses verifier rules, program types, helper restrictions, capabilities, LSM hooks, and tokens. uBPF's safe profile uses pointer provenance and typed helper/region metadata. bpftime combines verification with userspace isolation and an EIM resource model. eBPF for Windows combines PREVAIL verification with a Windows-specific execution environment.

These approaches can each be sound within their own threat model. What is missing is a portable way for a program or operator to state required privileges and for a backend to report which guarantees it actually enforces.

### Userspace resource attribution is still secondary

Userspace execution can make probes cheaper, but moving execution into a target process also moves costs into that process. A JIT compiler, shared map, injected trampoline, callback, and event queue all consume resources that ordinary process accounting attributes to the host application.

A benchmark that reports only average probe overhead misses the multi-extension operational question: which extension caused the cost, under what event rate, and can the runtime cap or revoke it without stopping the target? That measurement gap becomes a deployment problem as soon as userspace eBPF is shared infrastructure rather than a single debugging tool.

## Promising directions with academic and production value

### A machine-readable runtime contract and conformance suite

**Gap.** ISA and loader compatibility do not describe attachment, capability, state, and lifecycle semantics.

**Mechanism.** Define a small schema that declares a program's required context version, helpers and callable functions, map/state semantics, attachment identity rules, expected lifetime, and optional authority constraints. Each backend publishes the subset it supports. A conformance harness exercises the declared behavior with deterministic target programs and fault injection rather than checking only whether the bytecode executes.

**Delta.** RFC 9669 standardizes instructions, while libbpf and platform-specific APIs describe one host. This proposal standardizes neither new instructions nor one universal API. It makes the host contract testable across implementations.

**Artifact.** An open schema, reference validator, and adapters for Linux libbpf, bpftime, a minimal uBPF embedding, and eBPF for Windows where equivalent hooks exist.

**Evaluation.** Use observability, policy, packet-processing, and application-extension workloads. Measure the share that can declare one contract unchanged, mismatch detection rate, false compatibility claims caught, adapter complexity, and runtime overhead. Inject target replacement, missing helpers, stale modules, failed attach, and state-loss faults. The simplest baseline is today's documentation plus backend-specific integration tests.

**Academic value.** The research question is whether execution-environment semantics can be factored from implementation mechanisms without collapsing to the least common denominator.

**Production value.** Tool authors and operators gain a machine-checkable answer to "will this program run here with the guarantees it expects?" before touching a production target.

**Failure condition.** If the schema either becomes a backend-specific feature list or excludes most useful applications, the abstraction is not general enough to justify standardization.

### Capability-aware attach handles

**Gap.** Userspace attachment can target mutable process state, while privilege checks and target identity are often handled separately.

**Mechanism.** Make every attachment return a durable handle that binds four things: the exact target generation, the program identity, the granted capability set, and the state objects it owns. Before patching or registering a callback, the runtime revalidates target identity and capabilities. The handle supports query, revocation, and deterministic cleanup. Backends can implement it with kernel BPF links, injected runtime metadata, or host-specific objects.

**Delta.** Linux BPF links provide object lifetime, BPF tokens provide delegated capability restrictions, and EIM describes extension resources. The proposed handle combines these ideas at the userspace attach boundary where the target itself can change independently of the runtime.

**Artifact.** A bpftime prototype plus a small host API for embedders, with adapters that expose comparable metadata for kernel BPF links.

**Evaluation.** Stress PID reuse, `exec`, shared-library reload, concurrent attach/detach, permission revocation, partial injection failure, and target crashes. Compare a best-effort attach API, version checks alone, and the full capability-aware handle. Measure stale-target failures prevented, cleanup completeness, attach latency, steady-state overhead, and metadata size.

**Academic value.** The general question is how to make dynamic code attachment safe when target identity, authority, and lifetime evolve independently.

**Production value.** Long-running observability and policy agents can update or revoke instrumentation without relying on process-wide restarts or undocumented cleanup assumptions.

**Failure condition.** If normal target churn makes handles invalidate so frequently that operators disable the checks, the design is too strict or the chosen target identity is wrong.

### Per-extension resource ledgers

**Gap.** Userspace eBPF cost is usually charged to the host process, which hides interference between independent extensions.

**Mechanism.** Give every attach handle a resource ledger that records CPU/JIT time, map and code memory, event-buffer traffic, helper or host-function calls, faults, and dropped events. Enforce optional budgets at event dispatch or helper boundaries. The ledger should distinguish one-time attach/JIT cost from steady-state execution cost.

**Delta.** Ordinary process accounting observes the host process, while microbenchmarks report aggregate probe overhead. The ledger makes the extension itself the accounting principal.

**Artifact.** Runtime counters and budget hooks in bpftime, an export format, and a benchmark that composes several extensions with different event rates and failure behavior.

**Evaluation.** Run one to dozens of extensions on database, web-server, syscall, and packet-processing targets. Compare process-level accounting, sampling-based attribution, and exact runtime counters. Measure attribution error, enforcement latency, overhead, tail latency on the host, and whether budgets isolate a noisy extension without stopping unrelated ones.

**Academic value.** The broader systems question is whether dynamically injected extensions can become independently accountable tenants while still sharing one process address space.

**Production value.** Operators can answer which extension caused a regression and cap it without disabling the whole instrumentation stack.

**Failure condition.** If accurate attribution requires instrumentation overhead comparable to the probes themselves, sampling or coarse process accounting may remain the better operational tradeoff.

## What would change this conclusion?

The argument depends on userspace eBPF becoming a shared and portable runtime layer rather than remaining a collection of specialized embedded VMs. Several results would weaken it.

If a study of real deployments finds that almost all userspace eBPF workloads run on one backend, use a fixed helper set, keep no persistent state, and never require delegated attachment, then ISA plus local APIs may be enough. If programs that already load successfully through libbpf almost never fail because of lifecycle or capability differences, a cross-runtime contract would solve a small problem. If target-identity checks and per-extension accounting add enough overhead to erase the main benefit of userspace execution, they should remain optional debugging features instead of mandatory runtime semantics.

The opposite evidence would strengthen the case: recurring failures after successful program loading, incompatible state or detach behavior across backends, production incidents caused by stale attachments, and multi-extension interference that process-level accounting cannot attribute.

The near-term engineering test is therefore concrete. Take a small set of eBPF programs that already run on more than one backend, write down the runtime assumptions they currently leave implicit, and turn those assumptions into executable cross-backend tests. If the tests expose meaningful semantic drift, the ecosystem needs a contract above the ISA. If they do not, we should keep the runtime interface smaller.

## References

1. IETF, [RFC 9669: BPF Instruction Set Architecture](https://www.rfc-editor.org/rfc/rfc9669.html), 2024.
2. Linux kernel documentation, [libbpf Overview](https://docs.kernel.org/bpf/libbpf/libbpf_overview.html).
3. Linux kernel documentation, [eBPF Syscall](https://docs.kernel.org/userspace-api/ebpf/syscall.html).
4. IO Visor, [uBPF](https://github.com/iovisor/ubpf), userspace BPF VM and safe execution profile.
5. Eunomia, [bpftime](https://github.com/eunomia-bpf/bpftime), userspace eBPF runtime and extension framework.
6. Zheng et al., [Extending Applications Safely and Efficiently](https://www.usenix.org/conference/osdi25/presentation/zheng-yusheng), OSDI 2025.
7. Microsoft, [eBPF for Windows](https://github.com/microsoft/ebpf-for-windows), Windows hosting environment for eBPF toolchains and APIs.

For a hands-on introduction to the existing implementation space, see the [userspace eBPF tutorial](https://eunomia.dev/tutorials/36-userspace-ebpf/) and the [bpftime runtime documentation](https://eunomia.dev/bpftime/).