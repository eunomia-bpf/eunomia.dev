---
date: 2026-08-25
title: "Can eBPF Verify a Stateful Security Policy, Not Just Safe Bytecode?"
description: "Stateful eBPF security relies on persistent map transitions across events. This report develops policy-state contracts, runtime guards, and temporal verification."
tags:
  - Daily Report
  - eBPF
  - Security
  - Verification
  - BPF Maps
research_question: "How can eBPF security systems verify temporal invariants over persistent policy state shared across programs, CPUs, and userspace without moving the whole policy into an expensive general-purpose verifier?"
source_cutoff: 2026-08-25
status: daily-report
---

# Can eBPF Verify a Stateful Security Policy, Not Just Safe Bytecode?

A network policy can allow the first packet of a connection, remember that decision, and admit reply traffic later. A syscall policy can allow an operation only after initialization. An authentication cache can turn a previous identity check into a fast-path decision. In all three cases, the security result depends on **what happened before this invocation**.

Linux eBPF is good at building such mechanisms. BPF programs run at many security-relevant hooks, and [BPF maps](https://docs.kernel.org/bpf/maps.html) keep state that is shared across invocations and with userspace. Cilium, for example, implements stateful network policy and maintains connection-tracking, authentication, and endpoint policy state in BPF maps. Its current documentation explicitly describes stateful session enforcement and fixed-capacity connection-tracking and policy maps.

<!-- more -->

The verifier proves a different property. The [Linux eBPF verifier](https://docs.kernel.org/bpf/verifier.html) symbolically executes program paths and tracks register, stack, pointer, and scalar facts so that loaded bytecode cannot perform unsafe kernel operations. That protection is fundamental. It does not, however, mean that a security state machine encoded in map values implements the intended temporal policy.

Consider a simplified connection policy. One hook records `AUTHENTICATED`, another accepts packets while that entry exists, userspace may revoke the identity, and an LRU map may evict an entry under pressure. Each individual BPF program can be memory-safe and verifier-safe while the combined state machine still contains a security bug: a stale state survives revocation, two CPUs race through an invalid transition, a userspace update creates a state no BPF path would create, or capacity pressure silently changes the fail-open/fail-closed behavior.

Production systems already reveal why this distinction matters. Cilium documents that policy enforcement is stateful for session protocols, that BPF map capacities bound datapath state, and that policy-map overflow can optionally force an endpoint into lockdown. These are not bytecode-safety properties. They are properties of **persistent state, transition authority, resource limits, and recovery behavior**.

This report asks whether eBPF needs a stronger contract between static verification and runtime policy state. The goal is not to make the Linux verifier prove arbitrary distributed-system properties. A more practical design would let a policy author declare a small temporal contract, prove the parts that are static, and install cheap runtime guards only where the property depends on state that the verifier cannot know at load time.

The question is distinct from the earlier [stateful eBPF transactional-upgrade report](https://eunomia.dev/research/stateful-ebpf-transactional-upgrade/), which focuses on changing programs and maps atomically across generations. It is also distinct from [multi-tenant network-policy composition](https://eunomia.dev/research/ebpf-network-policy-composition/), which asks whose rule and authority determined an effective verdict. Here the policy generation may already be installed correctly and the authority may already be known. The missing property is whether the **runtime state transitions produced by that policy are themselves valid over time**.

## The Linux verifier proves execution safety inside one program run

The verifier performs abstract interpretation over BPF instructions. It tracks possible values and pointer types for registers and stack slots, explores branches, and rejects operations that it cannot prove safe. A pointer returned by a map lookup, for example, is tracked as a possible map-value pointer and must be checked before dereference.

This model is deliberately local to program execution. A map is a safe object that the program may access through an approved interface. The verifier does not treat the current contents of a generic hash map as a temporal security specification that must remain true across future invocations.

That boundary has made eBPF practical. Loading a program does not require model-checking every future packet, syscall, timer callback, userspace update, or CPU interleaving. Once verified, the fast path avoids a general-purpose runtime safety monitor.

Security applications add a second layer of meaning on top of those safe map accesses. A value such as

```text
{ subject = 42, state = AUTHENTICATED, generation = 17 }
```

is just bytes to a generic map. To the policy, it may mean that a particular subject is allowed to perform an operation until revocation or expiry. The map interface preserves memory safety; the security system must preserve the meaning.

## Stateful policy is already a production requirement

Cilium provides a concrete example. Its policy documentation defines session-based enforcement as stateful: allowing `A => B` also allows reply traffic for that connection, while it does not allow B to initiate a new connection to A. Its eBPF map documentation lists large node-scoped connection-tracking and authentication maps and per-endpoint policy maps with explicit capacity limits.

Capacity is part of semantics when a map stores security state. Insertion failure, eviction, or pressure can change which history remains available. Cilium therefore exposes map-pressure metrics, and its endpoint lifecycle includes an optional lockdown mode when a policy map cannot accommodate all required entries.

The important lesson is not that Cilium has a bug. It is that a real eBPF security datapath already has policy behavior that depends on conditions outside one verified instruction trace: map capacity, endpoint regeneration, identity changes, connection lifetime, and userspace control-plane updates.

System-call filtering shows the same need from another direction. The paper [Programmable System Call Security with eBPF](https://arxiv.org/abs/2302.10366) argues that classic seccomp filters are mostly static because cBPF cannot safely express rich stateful policies. Its Seccomp-eBPF design adds filter state, synchronization, and controlled access to kernel and user state; temporal specialization reduced the exposed syscall surface by up to 55.4% in its evaluation. Stateful policy is therefore useful enough to justify new execution interfaces.

But expressiveness and policy correctness are separate. Giving a program safe mutable state lets it implement a state machine. It does not prove that the state machine permits exactly the intended traces.

## Related verification work points to a split design

Recent work provides useful pieces without closing this exact gap.

[VEP](https://www.usenix.org/conference/nsdi25/presentation/wu-xiwei) uses annotations and a two-stage toolchain to verify broader eBPF-C programs, including memory safety for map-owned memory, and then checks proof-carrying bytecode with a smaller checker. Its target is full safe programmability of a program, not temporal policy invariants across an open-ended sequence of program invocations and control-plane updates.

[ePass](https://ebpf.foundation/research-update-verifier-cooperative-runtime-enforcement-for-ebpf/) takes another useful approach: when a property is difficult for static verification, transform the program to add targeted runtime enforcement while keeping the verifier as the gatekeeper. That verifier/runtime split is a strong fit for policy state because some transition facts are static while others exist only at runtime.

Outside eBPF, [p4tv](https://www.usenix.org/conference/nsdi25/presentation/zhang-delong) demonstrates temporal verification for stateful P4 programs using a temporal specification and a model of packet-level state transitions. It shows that stateful data-plane behavior can be verified as a trace property rather than only as per-packet code safety. eBPF is harder in a different way: its maps are generic, programs attach to heterogeneous hooks, several programs and userspace processes may write the same state, and executions can be concurrent across CPUs.

[BPF-DB](https://www.pdl.cmu.edu/PDL-FTP/Database/butrovich-sigmod2025_abs.shtml) attacks another part of the state problem by adding transactional data management for eBPF applications. It is valuable evidence that single-map operations are not enough for every stateful application. Transactions can make a group of updates atomic, but atomicity alone does not say whether `UNAUTHENTICATED -> ADMIN` is a legal policy transition.

Together these systems suggest a narrower research target: **a small temporal policy contract whose static parts can be checked ahead of time and whose dynamic parts can be enforced through bounded runtime transition APIs**.

## Where current work is still weak

### 1. Map types describe storage, not legal policy transitions

BPF maps specify key/value layout, capacity, lookup/update behavior, and, for some types, concurrency or eviction behavior. Security systems layer meanings such as connection state, authentication status, policy generation, or revocation on top.

What is missing is a machine-checkable declaration such as:

```text
state AUTHENTICATED {
  enter: only auth_hook or trusted_control_plane
  requires: identity_generation == policy_generation
  leave: timeout, revoke, endpoint_delete
}

invariant:
  allow_sensitive_operation => state == AUTHENTICATED
```

A useful test would inject illegal but memory-safe map updates from BPF and userspace and ask whether the system rejects them before they produce an allow decision.

### 2. Verification does not naturally span independent hook invocations

The verifier reasons about one program execution path. A temporal policy can span an LSM hook, a cgroup networking hook, a timer callback, a different CPU, and a userspace controller.

The missing abstraction is a bounded transition relation over persistent policy state. The contract needs to identify which hook classes may perform which transitions and which fields must remain generation-consistent, without making the kernel enumerate an unbounded event history.

A discriminating benchmark should include races, reordered events, delayed revocation, program restart, and multi-hook state sharing. If per-program verification plus ordinary unit tests finds the same violations reliably, an extra temporal layer is not justified.

### 3. Capacity and eviction are semantic events for security state

Generic maps have finite `max_entries`; LRU maps may evict entries. In a cache, eviction may only reduce hit rate. In a security state machine, losing a record can change a future verdict.

The missing property is an explicit failure semantic for each state class: `fail_closed`, `recompute`, `unknown`, or `safe_to_evict`. Cilium's optional policy-map lockdown demonstrates one production answer for overflow, but a general eBPF policy interface does not encode such semantics in the state schema.

The test is to drive maps to capacity under realistic load and measure whether the declared policy invariant still holds, not merely whether the BPF program continues running.

### 4. Userspace can be part of the transition authority

BPF maps are intentionally shared with userspace. That is powerful for dynamic policy, but it means a stateful security proof cannot assume that only verified BPF code writes the state.

The missing capability is a way to distinguish configuration writes, trusted state transitions, observation-only access, migration, and repair. A raw map FD gives a controller a broader operation than many policies actually need.

The test is to replay crashes, stale controllers, and concurrent policy generations. If a stale userspace process can create a state accepted by the fast path, the transition authority is incomplete.

## Promising directions with academic and production value

### 1. Compile a temporal policy contract into map and hook obligations

**Gap.** Current BPF type and verifier information can prove memory and pointer safety, but the policy meaning of map fields remains outside the load-time contract.

**Mechanism.** Define a deliberately small policy-state schema: named states, allowed transitions, authorized hook/program classes, generation relationships, expiry behavior, and capacity failure semantics. A compiler lowers the schema into ordinary BPF map layouts plus per-program proof obligations. Static checks verify that each program requests only transitions allowed for its attachment class and that invariant-relevant fields are updated together.

The compiler should not try to prove arbitrary C. It can restrict transition code to a small generated interface while the surrounding BPF program remains ordinary C or Rust. A policy author could still read state freely but must call generated transition helpers to change security-critical fields.

**Delta.** VEP verifies safe annotated eBPF programs; p4tv verifies temporal behavior in a P4-specific stateful dataplane. This direction keeps Linux's existing verifier for general bytecode and adds a narrow policy-state language for cross-invocation semantics.

**Artifact.** A schema compiler, libbpf integration, generated map value types and transition wrappers, and a checker that emits a compact manifest consumed at load time.

**Evaluation.** Implement stateful network, syscall, and LSM policies. Seed illegal transitions, stale generations, missing expiry, and update-order bugs. Compare handwritten BPF, property-based tests, VEP-style per-program verification where applicable, and the contract compiler. Measure violation detection, false rejects, verifier/build time, code size, and runtime overhead.

**Academic value.** The general question is how much temporal policy semantics can be moved into a small decidable contract before verification becomes intractable.

**Production value.** Security teams get a reviewable state machine instead of inferring policy semantics from scattered map writes and control-plane code.

**Failure condition.** If realistic policies cannot fit the restricted transition model without frequent escapes to unchecked writes, the abstraction is too weak.

### 2. Add verifier-cooperative runtime guards for dynamic transitions

**Gap.** Some facts cannot be known when the program is loaded: current policy generation, whether an entry was evicted, which controller owns a repair lease, or whether a revocation raced with the current event.

**Mechanism.** Following the verifier-cooperative idea used by ePass, transform security-critical state updates into calls to a small guarded transition API. The static verifier proves the arguments, pointer ownership, and bounded execution. The runtime guard checks only dynamic predicates such as `old_state`, `generation`, transition authority, and expiry epoch.

A possible implementation is a kfunc or map wrapper associated with a typed state descriptor. For fast paths, the common legal transition should require a few comparisons and one update. More expensive recovery can stay in userspace. Failed transitions return an explicit reason and can select a declared fail-closed or recompute path.

**Delta.** BPF-DB provides general transactions and ePass adds runtime checks for verifier-related safety. This proposal uses a much narrower runtime path to enforce semantic policy transitions rather than general database transactions or general bytecode safety.

**Artifact.** A prototype transition-map type or kfunc API, verifier annotations for transition authority, and libbpf tooling that attaches a policy-state descriptor to the program/map bundle.

**Evaluation.** Measure packet and syscall throughput, P50/P99 latency, cache misses, map memory, and transition failure cost. Stress concurrent CPUs, map pressure, revocation storms, controller crash/restart, and rolling policy updates. Ablate static checks versus runtime guards to show which violations require each layer.

**Academic value.** This tests a hybrid verification boundary: which security invariants are cheap enough to enforce dynamically while preserving eBPF's fast-path model.

**Production value.** Operators can fail closed on illegal security-state transitions without routing every event through a userspace policy engine.

**Failure condition.** If the runtime guard adds material hot-path overhead or needs policy-specific code as complex as the original program, ordinary testing and specialized implementations are preferable.

### 3. Build a temporal eBPF policy benchmark with adversarial state faults

**Gap.** Existing eBPF verifier suites mainly test bytecode safety and verifier behavior. Policy tests often validate expected packets or syscalls but do not provide a shared ground truth for state-machine violations under concurrency, capacity pressure, and control-plane failure.

**Mechanism.** Define compact policy automata and generate event traces that include legitimate transitions plus adversarial perturbations: duplicate events, reorderings, concurrent writers, stale generations, LRU eviction, full maps, userspace crash, BPF program replacement, and delayed revocation. Every trace carries the expected allow/deny/unknown outcome and the earliest invariant violation.

Include a P4-like temporal subset as one baseline, but add eBPF-specific dimensions: heterogeneous hook types, userspace map writers, per-CPU state, map semantics, and program generations.

**Artifact.** An open corpus, fault injector, replay harness, and adapters for kernel eBPF security systems. The benchmark should run against ordinary BPF maps, generated transition contracts, runtime guards, and transaction-backed state.

**Evaluation.** Primary metrics are false allows, false denies, undetected invalid transitions, time to identify the first bad transition, throughput, and overhead under a fixed state budget. Report capacity and recovery failures separately from ordinary policy logic.

**Academic value.** A shared temporal benchmark makes stateful policy correctness measurable instead of reducing verification work to load acceptance.

**Production value.** CNI, runtime-security, and syscall-policy projects get regression tests for the failures most likely to appear only under state pressure or recovery.

**Failure condition.** If real incidents and production fault models cannot be represented without application-specific semantics that dominate the benchmark, the corpus should remain a collection of system-specific tests rather than claim to be a general standard.

## A practical boundary for the verifier

The Linux verifier should remain responsible for the job it does well: proving that a BPF program can execute safely in the kernel under the BPF execution model. Expanding that engine until it understands every security policy's history would make loading slower, state explosion worse, and the verifier responsible for application semantics it cannot infer reliably.

A better boundary is to make **security-critical persistent state explicit**. The program declares a small transition contract. Static tooling proves what it can. The verifier checks that the program reaches state through the approved interface. Runtime guards handle the few predicates that only exist when an event happens. Tests and temporal model checking cover longer traces and recovery.

That design also improves diagnostics. Instead of a future incident report saying "the BPF program passed verification, so the policy should have been safe," an operator can ask a more precise question: which transition created this state, under which policy generation, and which invariant authorized the allow decision?

## What would change this conclusion?

The strongest counterargument is that stateful eBPF policies already work well with ordinary maps, careful control-plane code, unit tests, fuzzing, and production monitoring. A general temporal contract could add another language, another verifier, and another fast-path check without preventing enough bugs to justify the complexity.

Three results would weaken the proposal substantially:

1. a representative corpus of stateful eBPF security systems shows that security-relevant state transitions are too application-specific to share even a small common schema;
2. property-based and fault-injection tests find essentially all transition bugs at lower engineering and runtime cost;
3. runtime transition guards add enough cache traffic or synchronization to erase the performance advantage of in-kernel enforcement.

Conversely, repeated real failures caused by stale generations, concurrent map transitions, capacity pressure, or control-plane/BPF disagreement would strengthen the case. The most useful next evidence is therefore not another verifier microbenchmark. It is a corpus of **memory-safe, verifier-safe eBPF policies that still make the wrong security decision because persistent state followed an invalid trace**.

## References

- [Linux kernel: eBPF verifier](https://docs.kernel.org/bpf/verifier.html)
- [Linux kernel: BPF maps](https://docs.kernel.org/bpf/maps.html)
- [Cilium: Policy Enforcement](https://docs.cilium.io/en/latest/security/network/policyenforcement/)
- [Cilium: eBPF Maps](https://docs.cilium.io/en/latest/network/ebpf/maps/)
- [Cilium: Endpoint Lifecycle and policy-map lockdown](https://docs.cilium.io/en/stable/security/policy/lifecycle/)
- [Programmable System Call Security with eBPF](https://arxiv.org/abs/2302.10366)
- [VEP: A Two-stage Verification Toolchain for Full eBPF Programmability](https://www.usenix.org/conference/nsdi25/presentation/wu-xiwei)
- [On Temporal Verification of Stateful P4 Programs](https://www.usenix.org/conference/nsdi25/presentation/zhang-delong)
- [BPF-DB: A Kernel-Embedded Transactional Database Management System for eBPF Applications](https://www.pdl.cmu.edu/PDL-FTP/Database/butrovich-sigmod2025_abs.shtml)
- [ePass: Verifier-Cooperative Runtime Enforcement for eBPF](https://ebpf.foundation/research-update-verifier-cooperative-runtime-enforcement-for-ebpf/)
