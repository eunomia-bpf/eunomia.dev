---
date: 2026-09-10
title: "Can One eBPF Operation Mean the Same Thing on CPU, NIC, and DPU?"
description: "eBPF can delegate one logical operation to CPU, NIC, or DPU code. This report asks how atomicity, ordering, failures, and state stay semantically identical."
tags:
  - Daily Report
  - eBPF
  - SmartNIC
  - DPU
  - Offload
  - Verification
research_question: "How can eBPF define one higher-level operation across CPU, kernel, NIC, and DPU implementations while preserving its state transitions, concurrency, ordering, and failure semantics?"
source_cutoff: 2026-09-10
status: daily-report
---

# Can One eBPF Operation Mean the Same Thing on CPU, NIC, and DPU?

Imagine an XDP application that keeps per-flow state and exposes one logical operation: update a flow record and return the new decision. On one machine the operation runs against a normal kernel BPF hash map. On another, the program and map are offloaded to a NIC. On a third, a DPU runtime implements the same operation against device-local state and periodically reconciles it with the host.

All three implementations can pass a simple test. Give them one key, one value, and no concurrency, and each returns success with the expected record. That still does not establish that they implement the same operation.

The harder cases begin when two packets update the same flow, a device queue reorders work, an update fails after partial progress, host fallback races with device execution, or state is copied between backends. Linux documents that a normal BPF hash-map update replaces an existing element atomically. Its BPF offload path, meanwhile, dispatches lookup, update, delete, and iteration through device-specific map operations. Once a higher-level operation is implemented on several execution substrates, **semantic equivalence has to include state transitions, atomicity, visibility, ordering, and failure behavior, not only the value returned by one isolated call.**

<!-- more -->

This report continues the optimization series after [portable architecture specialization](https://eunomia.dev/research/ebpf-portable-architecture-specialization/) and the [native-operation trust boundary](https://eunomia.dev/research/ebpf-native-operation-trust-boundary/). Those reports ask whether an implementation is eligible on a machine, whether a portable fallback exists, and what native implementation must be trusted. Here those questions are assumed to be settled. Both implementations are eligible, their identities are known, and operators are willing to trust them. The remaining question is whether they implement the *same stateful operation* when real concurrency and failures appear.

It is also narrower than [complete mediation across host and offload paths](https://eunomia.dev/research/ebpf-complete-mediation-offload/). Complete mediation asks whether every policy-relevant packet still crosses a current enforcement point. This report assumes the operation is reached everywhere it should be. It asks whether the operation itself has one meaning across the backends that execute it.

## The BPF interface already contains semantics richer than a function result

BPF maps are a useful starting point because their public interface looks simple while their semantics are not uniform.

Linux documents `BPF_MAP_TYPE_HASH` as shared key/value storage and says that `bpf_map_update_elem()` replaces an existing element atomically. Per-CPU hash maps intentionally provide a different state model: each CPU has its own value slot. LRU variants add eviction behavior, and map values that contain a spin lock require explicit locked access. The same family of operations therefore carries assumptions about visibility, ownership, eviction, and synchronization in addition to a return code.

The UAPI makes failure semantics observable as well. `BPF_MAP_UPDATE_ELEM` can distinguish create-or-update, create-only, and update-only behavior through `BPF_ANY`, `BPF_NOEXIST`, and `BPF_EXIST`, and returns errors such as `EEXIST`, `ENOENT`, or capacity-related failures. Batch operations can report that only a prefix of requested elements was processed. Applications can depend on these details even when their source code says only “update a map.”

That matters for delegated operations because a backend that produces the same final value in a happy-path test may still disagree about which concurrent history is legal or what a failure means.

## Linux offload already routes BPF-facing operations into device implementations

The current Linux `kernel/bpf/offload.c` makes the implementation split concrete. For an offloaded map, the kernel's `bpf_map_offload_lookup_elem()`, `bpf_map_offload_update_elem()`, `bpf_map_offload_delete_elem()`, and `bpf_map_offload_get_next_key()` call methods in device-specific `dev_ops`. Program offload similarly invokes device callbacks for verifier preparation, instruction hooks, finalization, translation, and teardown.

This is a clean extensibility boundary. The host keeps a BPF-facing object and lifecycle while the device supplies the implementation. But the dispatch interface itself is not a formal statement that every device implementation has the same concurrent and failure semantics as every host map type or as another accelerator implementation.

The difference is easy to hide when an operation is stateless. A rotate or bit-select instruction can often be specified as a pure mapping from input registers to output registers. Kops exploits exactly this shape: a verifier-visible BPF proof sequence is paired with a native emit, and the EInsn operations have Lean 4 equivalence proofs. The stateful case has a larger observation boundary. Two implementations may compute the same local result and still differ in linearization point, visibility to another worker, rollback after error, or the treatment of an update that crosses a backend transition.

## ISA portability is not enough for semantic-operation portability

[RFC 9669](https://www.rfc-editor.org/rfc/rfc9669.html) gives BPF a platform-neutral instruction-set specification and conformance groups. That is the right level for saying what a BPF instruction means and which instruction groups an implementation supports.

A higher-level operation has a different problem. Consider a hypothetical `flow_update_v1` implemented in four ways:

```text
host_hash_map     -> update under Linux hash-map semantics
host_native_fast  -> optimized kernel/native implementation
nic_table         -> update device-local flow table
DPU_service       -> RPC or shared-memory update to DPU-owned state
```

Capability negotiation can tell the runtime that all four implementations are available. A proof or review can establish that each implementation is individually safe. Execution provenance can tell an operator which one ran. None of those facts defines whether concurrent invocations are linearizable, whether a success is durable across a reset, whether a timeout can mean “committed but reply lost,” or whether a host fallback may observe device state that has not yet become visible.

Those are part of the operation's semantics. If they are left implicit, “same operation” becomes an API name rather than a correctness statement.

## Stateful equivalence needs an observable transition contract

A useful semantic operation needs a reference relation over state, not only an input/output signature.

For one invocation, let the abstract operation be:

```text
(result, S') = OP(args, S)
```

For concurrent invocations, the contract additionally needs to say which histories are allowed. An implementation might promise linearizability, per-key serialization, eventual visibility, or only local ordering within one queue. Those are materially different contracts. The strongest contract is not automatically the right one, but the choice must be explicit.

Failure behavior belongs in the same model. If a device reports `-EIO`, did the abstract state remain unchanged? Can the update have committed even though the caller saw failure? Is retry idempotent? Does backend reset lose accepted operations? If an implementation cannot answer those questions, a runtime cannot safely swap it for another implementation just because both expose the same function signature.

This is the gap between an *implementation capability* and an *operation semantics*. The first says “this backend can run it.” The second says “these are all observable histories that count as correct.”

## Where current work is still weak

The first gap is **state-transition equivalence**. Existing verifier reasoning is excellent at proving safety properties of BPF execution, and native-operation work can prove equivalence for bounded instruction sequences. There is less machinery for saying that two heterogeneous implementations refine the same state machine when their state is stored, synchronized, and failed differently.

The second gap is **concurrency semantics across backend boundaries**. Host BPF maps have map-type-specific synchronization and visibility behavior. Device tables can have their own queues, atomic primitives, batching, and memory domains. A backend-neutral operation needs a declared linearization or visibility model; otherwise a move from host to NIC can change the set of legal concurrent outcomes without changing any BPF bytecode.

The third gap is **failure equivalence**. Return-code compatibility is too weak when an implementation can fail after partial or irreversible progress. Timeout, reset, queue overflow, DMA failure, firmware restart, and host-device disconnect can create uncertainty about whether an operation happened. A portable semantic operation needs to expose that uncertainty rather than map every backend-specific failure into one generic error.

The fourth gap is **transition correctness during fallback or migration**. Even if host and device implementations are each correct in isolation, mixed execution can violate the abstract contract. One request can complete on the device while a fallback path starts from stale host state, or a copied table can omit an in-flight update. This is distinct from asking whether fallback is available. The question is whether the combined history still belongs to the operation's allowed semantics.

## Promising directions with academic and production value

### 1. Define a backend-independent operation transition contract

A semantic operation can carry a small machine-readable contract above its implementation choices. The contract should identify the operation version, argument and result types, abstract state touched, atomicity or linearization guarantee, visibility domain, ordering constraints, failure classes, retry/idempotence rules, and the state ownership expected during execution.

For example:

```text
operation = flow_update_v1
state = flow_table[key]
atomicity = per_key_linearizable
success = new_value_visible_before_return
failure = {no_effect, outcome_unknown}
retry = idempotent_if(request_id_matches)
ownership = one_active_backend_per_generation
```

The artifact could begin as an ELF/BTF sidecar consumed by a loader or userspace runtime rather than a new kernel ABI. A backend registers the contract version it implements. The loader may choose among implementations only when the semantic version and guarantees match the application requirement.

The experiment should deliberately include two implementations that agree on single-threaded results but differ under races. Ground truth is the abstract state machine and its allowed histories. Measure false acceptance of semantically weaker backends, contract-check overhead, and the amount of backend-specific policy that can be removed from application code. An ablation that drops atomicity, failure, or ownership fields should expose which counterexamples become admissible again.

The academic question is what the smallest useful transition language is for heterogeneous BPF operations. The production value is making backend substitution reviewable before a fleet mixes kernel, NIC, and DPU implementations.

### 2. Pair each stateful operation with an executable reference model

Pure instruction equivalence can compare outputs directly. Stateful operations need an oracle for histories. Package each operation with a slow reference implementation or small transition model, then test every backend against the same model.

For sequential operations, differential testing may be enough. For concurrent operations, record invocation, completion, request identity, result, and relevant abstract state, then check whether the observed history can be linearized or otherwise refined to the contract. For failures, inject a cut at each backend-defined commit point and require the implementation to classify the resulting state as committed, not committed, or explicitly unknown.

The artifact is a conformance harness that can run the same operation package against host BPF maps, a native fast path, and device implementations. Kops-style proof can still be used for pure sub-operations; the higher-level checker handles stateful histories that are impractical to reduce to one register-equivalence theorem.

Evaluation should report semantic counterexamples found, test-generation cost, checking time, and coverage over concurrency and failure points. The approach loses if ordinary unit tests catch the same divergence with materially lower complexity.

### 3. Build a mixed-backend continuity benchmark, not another throughput benchmark

The hardest failures appear during transitions, so the benchmark should force them.

Start with a stateful XDP workload whose reference behavior is known. Run it on a host implementation and on one device implementation. While requests are concurrent, trigger backend handoff, queue drain, reset, table copy, stale-state injection, and fallback. Keep request IDs so the oracle can detect lost updates, duplicate effects, impossible reorderings, stale reads, and ambiguous outcomes.

Compare at least four designs: host-only execution, device-only execution, naive fallback that copies state and switches a pointer, and contract-aware handoff that establishes a state frontier before changing the active backend. Measure throughput and latency, but make the primary correctness metric the fraction of observed histories that cannot be explained by the declared operation semantics. Also report transition pause time, state-transfer overhead, and the number of operations that must be marked outcome-unknown.

The research value is an evaluation target that joins heterogeneous execution with formalizable state semantics. The production value is a regression gate for firmware, driver, and runtime updates: a new backend is not interchangeable merely because it is faster and passes single-call tests.

## What would change this conclusion?

The proposed abstraction would be less useful if existing BPF offload and accelerator interfaces already provide a complete, versioned contract for atomicity, ordering, visibility, failure outcomes, retries, and state continuity, and if different backends are routinely tested against that shared contract. In that case, another operation layer would duplicate machinery that already exists.

The problem would also be smaller if profitable heterogeneous specialization stayed almost entirely pure. If NIC, DPU, and native fast paths only replaced stateless instruction sequences while all mutable state remained host-owned behind existing BPF map semantics, Kops-style local equivalence plus ordinary capability negotiation could cover much of the practical need.

A more direct counterexample would be empirical. If a broad mixed-backend test suite found no semantic divergence once return values and final state matched, then explicit concurrency and failure contracts might be unnecessary engineering weight. The proposed benchmark is designed to make that claim testable.

Until such evidence exists, a higher-level eBPF operation should not be called portable merely because several backends implement the same symbol. For stateful delegation, portability means that every accepted implementation, including the histories created while switching between them, refines one explicit observable state-transition contract.

## References

- Linux kernel documentation, [BPF maps](https://docs.kernel.org/bpf/maps.html), accessed 2026-09-10.
- Linux kernel documentation, [BPF_MAP_TYPE_HASH, with PERCPU and LRU Variants](https://docs.kernel.org/bpf/map_hash.html), accessed 2026-09-10.
- Linux kernel source, [`kernel/bpf/offload.c`](https://github.com/torvalds/linux/blob/master/kernel/bpf/offload.c), accessed 2026-09-10.
- Linux kernel source, [`tools/include/uapi/linux/bpf.h`](https://github.com/torvalds/linux/blob/master/tools/include/uapi/linux/bpf.h), accessed 2026-09-10.
- IETF, [RFC 9669: BPF Instruction Set Architecture](https://www.rfc-editor.org/rfc/rfc9669.html), October 2024.
- Yusheng Zheng et al., [Kops: Safely Extending the eBPF Compilation Pipeline with Native Operations](https://arxiv.org/abs/2606.24213), 2026.
