---
date: 2026-08-09
title: "eBPF Hook Composition: Sharing One Hook Safely"
description: "eBPF hook composition needs more than execution order. This report compares kernel chains, libxdp, isolation work, and a testable composition contract."
tags:
  - Daily Report
  - eBPF
  - Runtime Systems
  - Composition
  - Linux
research_question: "What semantics should a runtime expose when multiple independently developed eBPF programs share one hook, mutate common state, return competing outcomes, and evolve independently?"
source_cutoff: 2026-08-09
status: daily-report
---

# eBPF Hook Composition: Sharing One Hook Safely

Imagine one ingress path shared by three independently managed eBPF programs. A security program can drop traffic. A telemetry program wants to observe every packet. A traffic-control program rewrites metadata before forwarding. All three are individually valid, all three can attach, and all three may be correct in isolation. The hard question begins after that: if they share one hook, what does the combined system mean?

This is the **eBPF hook composition** problem. Execution order matters, but order alone does not answer whether a drop is final, whether later programs see original or modified data, whether two programs can safely write the same map, or whether replacing one member can expose an unsafe intermediate chain. Linux already has several useful multi-program attachment models, yet those models encode different answers to these questions.

<!-- more -->

The conclusion of this report is that **safe multi-program eBPF needs a composition contract, not only a dispatcher**. Such a contract should make four things explicit: what each program can affect, how intermediate effects are exposed, how outcomes are combined, and which generation of the whole composition is active. Existing work on dispatchers, isolation, static analysis, and virtualization provides important pieces. The missing layer is a machine-checkable protocol that binds those pieces to attachment and update semantics.

This report continues the [Daily Report on first-class userspace eBPF runtimes](https://eunomia.dev/research/userspace-ebpf-runtime-contract/). That report argued that an eBPF runtime needs explicit attachment, capability, state, and lifetime semantics above the ISA. Once several extensions share one attachment point, the same requirement becomes a composition problem.

## eBPF hook composition already exists, but not as one contract

Linux does not have one universal rule for "run several BPF programs here." Different hook families deliberately expose different composition semantics because their underlying operations are different.

The [`BPF_SK_LOOKUP` documentation](https://docs.kernel.org/bpf/prog_sk_lookup.html) is a good example. Multiple programs can attach to the same network namespace and execute in attachment order. But the final result is not simply the return value of the last program. A program may select a socket with `bpf_sk_assign()`. If several programs select sockets and return `SK_PASS`, the last valid selection wins. A `SK_DROP` only determines the result when no program returned `SK_PASS` with a valid socket. The hook therefore has an explicit outcome algebra involving both verdicts and an accumulated selected object.

[`BPF_PROG_TYPE_CGROUP_SOCKOPT`](https://docs.kernel.org/bpf/prog_cgroup_sockopt.html) exposes a different model. In a cgroup hierarchy, programs execute from the child upward to its ancestors. A later program sees modifications made to the shared `bpf_sockopt` context by an earlier program. A return value of `0` rejects the operation, while `1` continues. Here composition includes both ordered context transformation and a gate on continuation.

[HID-BPF](https://docs.kernel.org/hid/hid-bpf.html) makes the mutation issue even clearer. Several programs can attach to one device for most HID-BPF attachment types. Programs execute one after another on the same data buffer, and later programs see the modified data rather than the original input. A negative return discards the event. The API supports inserting a program at the front with `BPF_F_BEFORE`, but that flag does not tell an independently developed program whether it is semantically safe to consume another program's rewritten buffer.

The networking ecosystem has built additional composition machinery above kernel primitives. [`libxdp`](https://github.com/xdp-project/xdp-tools/blob/master/lib/libxdp/README.org) can load multiple XDP programs on one interface through a dispatcher. Component programs have run priorities and chain-call actions. A program's return code can either continue to the next component or terminate the chain. This is much better than accidental replacement, but it still assumes that the programs agree on the meaning of shared packet mutations, maps, and return behavior.

Modern TCX goes further on lifecycle and ordering. The [Eunomia TCX tutorial](https://eunomia.dev/tutorials/50-tcx/) demonstrates BPF link ownership, relative placement with `BPF_F_BEFORE` and `BPF_F_AFTER`, `BPF_F_REPLACE`, chain revision tracking, and return codes such as `TCX_NEXT`, `TCX_PASS`, and `TCX_DROP`. TCX shows that explicit relative ordering and revision-aware attachment can be part of a production interface instead of loader convention.

These mechanisms are all reasonable for their hooks. Their differences are the important evidence.

| Mechanism | Ordering | What later programs observe | How outcomes combine | Shared state / update semantics |
| --- | --- | --- | --- | --- |
| `BPF_SK_LOOKUP` | attachment order | selected socket can be carried forward | last valid selection can win; drop is conditional on no valid selection | hook-specific |
| cgroup sockopt | child to parent | earlier context mutations | return controls rejection/continuation | cgroup/socket local storage available |
| HID-BPF | list order, optional front insertion | same mutable data buffer | negative error discards event | link lifetime; shared mutation is visible |
| libxdp | run priority | packet after previous components | chain-call actions decide continuation | dispatcher-managed component set |
| TCX | relative before/after ordering | packet and `__sk_buff` state | explicit terminal and non-terminal return codes | BPF links, replacement, revision-aware changes |

There is no contradiction here. A socket selector, a packet classifier, a HID event transformer, and a socket-option filter should not necessarily share one return-value rule. The problem is that the rule is mostly implicit in the hook API rather than represented as a composition contract that tools can reason about.

## Ordering is only one axis of eBPF hook composition

A common response to multi-program conflicts is to add priorities or an explicit chain. That solves one real problem: deterministic order. It does not solve four other kinds of interaction.

### The hook needs an outcome algebra

Consider a security filter followed by a telemetry program. If the security filter returns a deny verdict, should telemetry still execute so it can record the denied event? If it executes, may its return value accidentally override the deny? Now replace telemetry with a socket-selection program. "First wins" and "last wins" are both plausible but have very different meaning.

Linux already answers these questions on a per-hook basis. `BPF_SK_LOOKUP` accumulates a selected socket with documented precedence rules. TCX distinguishes `TCX_NEXT` from terminal outcomes. libxdp lets component metadata declare which XDP actions should continue. These are small algebras, not just integer return values.

A general composition interface should therefore identify the hook's outcome categories and the resolver that combines them. A loader should be able to distinguish "terminal deny," "continue," "select resource," "redirect," and "transform then continue" rather than treating every program as a function that returns an opaque integer.

### Mutation visibility needs to be intentional

HID-BPF explicitly documents that every program works on the same data buffer and later programs do not have the original data unless they preserve it themselves. Cgroup sockopt similarly propagates context modifications to later programs.

This is powerful because it enables pipelines. It is also a hidden dependency. A parser written against the original packet or report can silently become wrong when an earlier extension rewrites a field. Conversely, forcing every program to see an immutable snapshot would prevent useful staged transformations and add copying cost.

The composition contract therefore needs a visibility choice. A program may require the original view, the previous program's output, or a named derived view. If the runtime cannot provide the requested view, attachment should fail before production traffic reaches the chain.

### Shared maps are a concurrency contract, not only a data structure

Linux allows maps to be shared deliberately. The [`BPF_MAP_TYPE_CGROUP_STORAGE` documentation](https://docs.kernel.org/bpf/map_cgroup_storage.html) notes that since Linux 5.9 storage can be shared by multiple programs and that there is no implicit synchronization when programs on different CPUs access it. The user must provide synchronization where necessary.

That is appropriate at the primitive level. For independently managed extensions, however, "both programs can open this map" is not enough information. One program may assume single-writer counters, another may reset entries, and a third may treat an entry as a lease. The verifier can prove memory safety without proving that those state-machine assumptions compose.

A composition contract should at least distinguish private state, shared read-only state, shared state with one writer, and shared multi-writer state with an explicit synchronization protocol. Richer state invariants can remain optional, but ownership cannot be invisible.

### Updating one member is a whole-chain event

Link-based attachment makes individual program lifetime manageable. Relative ordering makes insertion predictable. Revision checking, as exposed by TCX, helps detect concurrent modifications. The remaining issue is semantic atomicity.

Suppose a security program changes from version A to version B and the new version expects a companion metadata producer to run first. Replacing only the security program can create a period where B executes against the old composition. Even if each individual `BPF_LINK_CREATE` or replace operation is atomic, the desired multi-program configuration may require several linked changes.

This is why composition state needs a generation. The runtime should be able to validate the complete proposed chain, construct it off the hot path where possible, and commit a new generation only if the currently active generation still matches the expected revision.

## Recent work solves important pieces, not the whole composition protocol

The multi-program problem is active research rather than an empty space.

[vBPF, presented at OSDI 2026](https://www.usenix.org/conference/osdi26/presentation/zhang-jing), targets multi-tenant eBPF contention. It late-binds logical programs to physical hooks, attributes events to tenants, uses an O(1) dispatcher, and adds compiler-assisted state isolation. This is strong evidence that simple linear native attachment is insufficient when independent tenants share infrastructure. Its core problem is virtualization and tenant isolation, not a general contract for the semantic interaction of programs that intentionally operate on the same event.

[KRAKENGUARD, presented at NSDI 2026](https://www.usenix.org/conference/nsdi26/presentation/patel), uses symbolic execution in a trusted userspace manager to enforce policies over helper use, memory access, return values, and cross-program interference. Its XDP-as-a-Service use case demonstrates that load-time analysis can reject unsafe combinations before they share a host interface. This is directly useful for composition. It still leaves the attachment layer to decide what the allowed composition is supposed to mean.

[Yaksha-Prashna](https://arxiv.org/abs/2602.11232) analyzes eBPF bytecode network functions for conformance and dependencies on other bytecodes. That matters because a future composition system should infer effects when source annotations are unavailable. It also means that "statically analyze two eBPF programs for conflicts" is no longer a sufficient research thesis by itself. The open question is how such analysis becomes enforceable runtime state.

The scheduled [BPFChain tutorial at ACM SIGCOMM 2026](https://conferences.sigcomm.org/sigcomm/2026/ttbpfchain/) is another useful signal. Its published program explicitly calls out execution-order conflicts, return-value overrides, shared-map races, a trampoline chainer, and chain monitoring. As of this report's August 9 source cutoff, SIGCOMM 2026 is still upcoming, so this is evidence from the announced tutorial program rather than a completed event. It nevertheless makes one point clear: a new dispatcher alone is not a novel answer to multi-program eBPF.

The residual problem sits between these efforts. We have mechanisms to chain programs, isolate tenants, infer effects, and manage individual links. We do not yet have a widely used object that says, in machine-readable form, **which effects are allowed to compose, which result resolver applies, which state is shared, and which complete generation is active**.

## A composition contract should make effects and dependencies explicit

A useful contract does not need to replace existing hook APIs. It can sit above them and compile down to TCX, libxdp, cgroup attachment, kernel links, or userspace runtimes such as [bpftime](https://github.com/eunomia-bpf/bpftime).

A minimal per-program manifest could describe:

- the hook and attach target it expects;
- context fields or packet regions it may read and write;
- helper classes and externally visible side effects it may invoke;
- maps it reads or writes and the ownership mode of each map;
- whether it requires the original event, the previous program's result, or a named transformed view;
- relative ordering constraints such as `after: parser` or `before: terminal-policy`;
- result categories it may produce and whether any are terminal;
- failure behavior and an optional execution budget.

The hook adapter would add the native outcome algebra. For XDP it might map XDP actions into terminal and continuing categories. For TCX it can use the existing `TCX_NEXT` versus terminal behavior. For `BPF_SK_LOOKUP` it must model both the verdict and the selected socket. For HID-BPF it must model the shared mutable buffer and event-discard rule.

Some manifest facts can be authored. Others should be inferred from bytecode, BTF, map references, helper calls, or program analysis. The important distinction is that analysis is evidence for the contract, not the contract itself. The loader should compare declared and inferred effects, report contradictions, and reject a composition when it cannot establish the requested guarantees.

## Where current work is still weak

### Hook-specific semantics are difficult to compare or reuse

Kernel documentation defines semantics precisely within individual hook families, but an operator managing dozens of BPF components still has to understand each family manually. There is no common vocabulary for "terminal result," "selected object," "shared mutation," or "original-view requirement" that a deployment tool can validate across hook types.

The missing evidence is a corpus of real multi-program deployments showing which semantic dimensions recur across XDP, TCX, cgroup, tracing, HID, and userspace backends. If every hook needs entirely bespoke rules, a common contract may collapse into documentation metadata. If a small set of effect and outcome patterns covers most deployments, the abstraction is useful.

### Isolation analysis is not yet the attachment protocol

KRAKENGUARD can reason about helper use, memory accesses, return values, and interference. Yaksha-Prashna can expose conformance and bytecode dependencies. Those capabilities are stronger than simple declaration files.

What remains weak is the path from analysis result to live composition. An analyzer may determine that program B reads a field written by program A, but the attachment system still needs to decide whether that dependency requires `A before B`, whether B may run after A returns a terminal result, and whether an update can temporarily violate the dependency.

This separation suggests a systems boundary: analysis should produce evidence; a composition protocol should turn that evidence into enforceable attachment and generation rules.

### Revision-aware links do not automatically provide transactional composition

TCX is a strong counterexample to the claim that BPF attachment lacks ordering or revision control. It has relative ordering, replacement, links, and expected revisions. The research gap is narrower: how should a *set* of programs, state contracts, and result-resolution policy change as one semantic unit?

A useful experiment must inject failures between several coordinated updates. If ordinary TCX or another existing mechanism can already preserve every required invariant without a new bundle abstraction, then a versioned composition generation is unnecessary.

### Shared-state correctness remains mostly outside verifier scope

The verifier can establish many safety properties about one program. Map types and locks provide mechanisms for concurrent access. Fine-grained isolation systems can restrict which state a program may touch.

None of those automatically proves that two independently designed state machines agree on ownership, reset behavior, epochs, or update ordering. The composition problem therefore extends beyond memory safety into protocol compatibility. The practical question is how much of that protocol can be captured without recreating a general-purpose formal specification language.

## Promising directions with academic and production value

### A typed composition manifest backed by effect inference

**Gap.** Deployment systems can order programs, and analysis systems can discover some program effects, but the two are not connected by a portable attachment contract.

**Mechanism.** Define a compact manifest with typed reads, writes, map ownership, outcome categories, visibility requirements, and partial-order constraints. Use BTF, ELF metadata, helper-call inspection, and optional symbolic or abstract interpretation to infer an effect summary from the compiled program. The loader checks declared effects against inferred effects and solves the partial order before attaching anything.

**Delta.** This is not another dispatcher and not another standalone analyzer. libxdp or TCX can remain the execution mechanism; KRAKENGUARD- or Yaksha-Prashna-style analysis can remain the evidence producer. The new artifact is the machine-checkable contract between analysis and attachment.

**Artifact.** A schema plus a libbpf-based loader with adapters for TCX and libxdp first, followed by cgroup hooks and a userspace adapter for bpftime. A small compiler plugin could emit optional source-derived effect metadata.

**Evaluation.** Build a corpus of independently developed observability, security, and networking programs. Form valid and invalid pairs and triples. Measure whether the system detects write/write conflicts, missing ordering dependencies, map-ownership mismatches, and undeclared side effects. Report false rejection and false acceptance rates separately. Measure attach-time analysis cost and steady-state overhead, which should be close to the native backend after the composition has been validated.

**Academic value.** The central question is whether an effect system for eBPF composition can be expressive enough for real programs while remaining decidable and backend-independent.

**Production value.** Operators get a pre-deployment answer to "can these two independently shipped programs safely coexist on this hook?" instead of learning through traffic loss or policy bypass.

**Failure condition.** The idea fails if annotations dominate developer effort, inference produces too many false conflicts, or most useful programs require hook-specific escape hatches that erase the common type system.

### Hook adapters with explicit outcome algebras

**Gap.** Every multi-program hook has return values, but those return values do not have one composition meaning. Flattening them into generic priorities can silently change security or routing behavior.

**Mechanism.** Give each hook adapter a typed result model. A small common vocabulary could include `continue`, `deny`, `select`, `redirect`, `transform`, and `terminal`, while the adapter keeps hook-specific payloads such as a selected socket. A composition plan declares which resolver applies. The planner rejects combinations where two terminal or stateful results are ambiguous unless the operator supplies an explicit resolver.

**Delta.** Existing chain mechanisms determine which program runs next. This mechanism makes the *meaning of the combined result* explicit and testable. It should compile to native return conventions rather than introduce a second packet-processing runtime when the kernel already has suitable semantics.

**Artifact.** Hook adapters for `BPF_SK_LOOKUP`, TCX, XDP/libxdp, cgroup sockopt, and HID-BPF, plus a test harness that runs the same abstract composition cases against native kernel behavior.

**Evaluation.** Generate permutations of policy, telemetry, transformation, selection, and redirect programs. Compare the declared outcome against the observed kernel outcome under each permutation. Include malicious and accidental return-value overrides. Measure adapter overhead and, more importantly, whether the model catches semantic ambiguity before attach.

**Academic value.** The research problem is to find a small algebra that preserves enough hook-specific behavior to be useful without pretending that all BPF hooks are identical.

**Production value.** Security and networking teams can inspect why one result wins and can reject a deployment where an observability or routing component would accidentally weaken policy.

**Failure condition.** If preserving native semantics requires a completely unique algebra per hook with no reusable properties, the abstraction should remain hook-local instead of becoming a cross-hook layer.

### Versioned composition generations

**Gap.** Individual BPF links can be replaced safely, and TCX exposes revision-aware chain changes, but a semantic composition may include several programs, ordering constraints, state contracts, and one result resolver that must evolve together.

**Mechanism.** Represent the active composition as a generation object. Build the next generation off-path, validate all members and dependencies, resolve ordering, prepare compatible maps or state views, then commit using compare-and-swap against the expected current generation. If preparation or validation fails, the old generation remains active. If the backend cannot swap a complete chain atomically, the adapter must expose that limitation rather than claiming transactional behavior.

**Delta.** This proposal does not solve arbitrary stateful eBPF application upgrade, which is the next question in this Daily Report series. Its scope is narrower: make the *membership and semantics of one shared hook* a coherent versioned object.

**Artifact.** A userspace composition manager with a TCX prototype first because TCX already exposes ordering and revision concepts, then a libxdp dispatcher adapter. Record generation IDs, member program/link IDs, effect contracts, and resolver choice in an inspectable control-plane object.

**Evaluation.** Repeatedly add, remove, reorder, and replace programs while traffic or events exercise the hook. Inject process crashes and failures between every preparation step. Define safety invariants such as "deny policy is never absent," "parser version and consumer version match," and "no old writer runs with new map layout." Measure any interruption window, rollback time, throughput impact, and the number of intermediate invalid compositions observed.

**Academic value.** The question is whether transactional configuration can be layered over heterogeneous BPF attachment mechanisms without kernel changes, and where kernel support becomes necessary.

**Production value.** Fleet operators can roll out independent eBPF components without turning every chain update into a hand-coordinated maintenance operation.

**Failure condition.** If current link and revision primitives already provide the required multi-object atomicity with a thin userspace wrapper, then this should remain an engineering library rather than a new systems abstraction.

## The practical architecture is a planner over existing mechanisms

The strongest design is probably not a universal mega-dispatcher. It is a planner and contract layer that reuses the best native primitive for each hook.

For TCX, the planner can use native links, relative ordering, and revisions. For XDP, it can compile a validated plan into a libxdp dispatcher. For cgroup hooks, it can respect hierarchy and hook-specific return semantics. For HID-BPF, it can reject a program that requires the original event if an earlier transformer destroys that view. In userspace runtimes such as bpftime, the same contract can drive a local dispatcher and capability model even though the physical attachment mechanism is different.

This architecture also gives observability a stable unit. Instead of merely listing program IDs, tooling can report a composition generation, its declared dependencies, which program terminated an event, which shared-state contract is active, and whether the running chain still matches the validated plan.

That matters because composition failures often look like ordinary application failures. A packet disappears, a syscall is rejected, or an input event changes shape. Without a first-class composition identity, an operator has to reconstruct the chain from several independent loaders and infer which interaction caused the behavior.

## What would change this conclusion?

Three kinds of evidence would weaken the case for a new eBPF hook composition contract.

First, an existing production mechanism could already provide a cross-hook, machine-readable model of program effects, outcome resolution, shared-state ownership, ordering dependencies, and versioned whole-chain updates. In that case the right work is adoption and conformance testing, not another abstraction.

Second, a broad empirical study could show that independently managed eBPF programs almost never have semantic conflicts once deterministic ordering and memory isolation are provided. That would suggest the remaining contract adds complexity for rare failures.

Third, a prototype may show that effect inference is too imprecise or that transactional composition requires so much copying, pausing, or backend-specific code that operators are better served by explicit per-hook managers.

The evidence available today points the other way. Linux already exposes multiple distinct composition semantics; libxdp and TCX make multi-program chains operationally practical; vBPF and KRAKENGUARD show that multi-tenant coexistence needs more than the classic verifier; Yaksha-Prashna shows that bytecode dependencies can be analyzed; and the announced BPFChain material treats multi-program conflict as a production discipline. The next missing step is to connect those pieces into one testable statement of what a shared hook is allowed to mean.