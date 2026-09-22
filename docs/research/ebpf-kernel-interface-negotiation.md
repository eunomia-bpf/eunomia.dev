---
date: 2026-09-22
slug: ebpf-kernel-interface-negotiation
title: "Can an eBPF Loader Treat a kfunc as Just Present or Missing?"
description: "kfuncs, struct_ops, and BPF iterators expose typed, context-specific contracts that make simple present-or-missing feature checks incomplete."
tags:
  - Daily Report
  - eBPF
  - Linux
  - kfunc
  - struct_ops
  - Compatibility
research_question: "When unstable or provider-scoped BPF-facing interfaces evolve, what should a loader negotiate before choosing a compatible program variant?"
source_cutoff: 2026-09-22
status: daily-report
---

# Can an eBPF Loader Treat a kfunc as Just Present or Missing?

An eBPF loader inspects the target kernel BTF, finds the kfunc name it wants, and selects the fast program variant. That sounds like a good replacement for checking `uname -r`.

But the name alone does not describe the contract the verifier and runtime will enforce. A kfunc may be registered only for one BPF program type. Its pointer arguments may carry ownership, nullability, RCU, lock, or lifetime rules. An open-coded iterator is a matched constructor/next/destructor protocol whose state size is part of the user-visible API. An XDP metadata kfunc can exist while a particular device driver still reports `-EOPNOTSUPP` at runtime.

So the useful question is not simply whether an interface exists. It is whether the target kernel, program type, provider, and verifier agree on the particular interface contract that this BPF artifact expects.

This report continues the deployment-compatibility series after [capability-based admission](https://eunomia.dev/research/ebpf-kernel-capability-evidence/) and [cross-kernel semantic compatibility](https://eunomia.dev/research/ebpf-kernel-upgrade-semantic-compatibility/). Those reports ask whether a host can admit an artifact and whether an admitted artifact keeps its application-level meaning. Here the narrower problem is **interface negotiation before choosing the artifact variant**.

<!-- more -->

## A kfunc name is only one dimension of compatibility

Linux deliberately gives kfuncs a different stability contract from classic BPF helpers. Current kernel documentation says kfuncs do not have a stable interface and can change between kernel releases. Visibility is registered per BPF program type, and the verifier interprets BTF types plus kfunc flags such as acquire/release and nullable-return annotations.

That means a loader can observe several different states that all look like "the function exists" at first glance:

```text
name absent
    -> this variant clearly cannot use the kfunc

name + BTF signature present, wrong program type
    -> verifier cannot admit this call from the selected program

name + signature + program type present, different verifier contract
    -> pointer/lifetime/context assumptions may no longer match

call admitted, provider lacks implementation
    -> runtime reports an unsupported operation for this device/context
```

The last case is not hypothetical. Linux's XDP RX metadata API exposes a family of kfuncs for packet timestamp, hash, VLAN, and related metadata. Drivers opt into those operations. The kernel documentation explicitly defines `-EOPNOTSUPP` for a driver that does not implement a metadata operation and provides a netlink feature query for per-netdev support.

A host-global Boolean such as `has_bpf_xdp_metadata_rx_timestamp=true` therefore answers the wrong deployment question. The relevant fact is closer to: "this XDP program on this netdev can use this operation under this argument and return-value contract."

## Newer BPF interfaces increasingly encode protocols, not isolated calls

Open-coded BPF iterators make the problem easier to see because Linux documents the protocol explicitly. One iterator consists of a state structure and a tightly coupled trio of kfuncs: constructor, `next`, and destructor. The verifier protects the iterator state and relies on the guarantee that `next` eventually returns `NULL`. The documentation also warns that the iterator state-structure size is user-visible API, so changing it breaks backwards compatibility.

Compatibility is therefore a relation across several pieces:

```text
iterator state type and size
        + constructor signature and initialization rule
        + next signature, return type, nullability, and verifier semantics
        + destructor signature and lifetime rule
```

Checking only that `bpf_iter_<type>_next` exists cannot establish that relation.

`struct_ops` adds another shape of the same problem. `sched_ext`, for example, is loaded through `struct sched_ext_ops`; current documentation says all operations are optional except `ops.name`, and default behavior can apply when a callback is omitted. A loader selecting between scheduler variants needs to reason about the target `struct_ops` schema and which callbacks or flags its algorithm actually requires, not merely whether `CONFIG_SCHED_CLASS_EXT` is enabled.

These APIs are useful precisely because they let BPF evolve with kernel subsystems. Treating them like frozen helpers would remove much of that flexibility. The deployment mechanism should instead make the moving contract explicit.

## Trial loading is authoritative, but it is not a complete negotiation protocol

There is a strong simple alternative: build several BPF object variants, try to load the preferred one, and let the verifier reject anything incompatible. This has real advantages. The verifier is the final authority for target-specific safety, and a loader does not need to duplicate every kernel rule in userspace.

For a small application with two well-understood variants, trial loading may be the right design.

The limitation appears when the fleet or interface family grows. A failed load can combine multiple reasons: missing BTF type, wrong program type, changed kfunc signature, ownership mismatch, unsupported `struct_ops` field, or an unrelated verifier constraint. Trying variants in sequence tells the loader which object eventually loads, but it does not necessarily explain which interface requirement selected that object. Provider-scoped support can also remain unresolved until runtime.

That distinction matters for operations. A deployment controller wants to answer questions such as:

- Why did this node choose the fallback object?
- Does the same object need a different variant for another netdev on the same host?
- Which kernel-interface change requires the compatibility matrix to be rerun?
- Is the failure an expected interface mismatch, an application bug, or a verifier regression?

Trial loading remains a necessary final check. It should be the last authority in a negotiation path, not the only representation of the path.

## Where current work is still weak

### Interface requirements are distributed across BTF, verifier metadata, docs, and provider state

BTF gives rich type information, and the kernel registers kfunc visibility and semantic flags. Individual subsystems expose more state, such as XDP RX metadata support per netdev. Yet an application artifact has no common machine-readable way to say which subset of that information it requires.

A loader can hard-code the checks, but then the compatibility policy becomes application code. The missing piece is an artifact-level requirement description that can be compared with target evidence before the expensive or destructive parts of deployment.

A useful test is whether the same requirement description can correctly classify objects across multiple upstream kernels, distribution kernels, program types, and devices without encoding kernel version ranges.

### A successful host-level capability probe does not identify the valid context

The previous [capability-evidence report](https://eunomia.dev/research/ebpf-kernel-capability-evidence/) argued for probing the real host rather than trusting version strings. That still leaves a scope problem. A kfunc can be valid for one BPF program type but not another; XDP metadata support can differ between two netdevs on one host.

The missing abstraction is **scoped capability evidence**: evidence bound to the context that makes the interface usable. For some APIs that scope is program type; for others it includes device, module/provider, sleepability, attachment type, or another verifier-visible condition.

A benchmark should intentionally create hosts where global feature presence is identical but valid contexts differ, then measure false admissions from host-global probing.

### Compatibility CI does not know which interface changes deserve which tests

A broad kernel matrix catches failures but is expensive and often opaque. When one kfunc signature, iterator state type, or `struct_ops` field changes, most BPF applications do not need every test repeated; the applications that depend on that contract do.

What is missing is a dependency-aware way to connect kernel-interface deltas to artifact requirements and then select the smallest useful compatibility matrix. Without it, projects either under-test unstable interfaces or repeatedly run large matrices without knowing what each cell proves.

The discriminating experiment is historical replay: given a sequence of kernel interface changes, can the dependency model select the tests that would have caught real compatibility failures while running substantially fewer irrelevant cells?

## Promising directions with academic and production value

### Direction 1: embed typed interface requirements in each BPF artifact variant

**Gap.** BPF objects can contain BTF and CO-RE metadata, but the loader still tends to encode higher-level interface assumptions in source code and variant-selection logic.

**Mechanism.** Add a compact requirement manifest beside each program variant. A requirement names the interface family and the properties the artifact depends on, for example:

```text
kfunc:
  name: bpf_example
  program_type: tracing
  signature: <BTF-derived type identity>
  effects: [acquire, nullable-return]

iterator:
  state_type: bpf_iter_example
  state_size: ...
  protocol: [new, next, destroy]

provider_feature:
  scope: netdev
  operation: rx_timestamp
```

The loader derives a target profile from BTF, kernel-exposed feature state, and provider queries, then performs a structural match before trying to load the object. The verifier still has final authority; the manifest does not attempt to reimplement verification.

**Delta.** CO-RE says how to relocate an object against target types. This manifest says which *interface contract* a particular object variant expects before relocation and verification are attempted.

**Artifact.** A libbpf-side manifest generator and resolver, plus a `bpftool` view that prints artifact requirements and the matched target evidence.

**Evaluation.** Test upstream and distribution kernels across several program types, kfunc families, iterators, `struct_ops`, and XDP devices. Compare kernel-version gates, BTF-name presence, trial-load-only selection, and typed matching. Measure false admits, false rejects, number of failed trial loads, and diagnostic specificity.

**Academic value.** The research question becomes how much of an evolving kernel interface can be described declaratively without duplicating verifier semantics.

**Production value.** Packaging one application with several object variants becomes explainable and testable instead of a chain of opaque load attempts.

**Failure condition.** If typed matching does not predict a better variant than simple ordered trial loading, or if maintaining the manifest requires hand-copying verifier internals, the extra layer is not justified.

### Direction 2: negotiate variants against scoped capabilities, not one host feature bitmap

**Gap.** A machine-wide capability map loses the context that makes some interfaces valid only for one program type, device, module, or attachment environment.

**Mechanism.** Treat variant selection as constrained matching over a scoped profile:

```text
artifact requirements
    x kernel/BTF generation
    x BPF program type
    x attach target
    x provider/device identity
    x verifier-relevant context
        -> selected variant + rejected alternatives + evidence
```

The output is a small negotiation receipt recording the selected variant and the exact scoped facts that made it eligible. The receipt is cached only at the scope where those facts remain valid. A netdev capability does not become a host-global fact; a program-type kfunc registration does not become proof for another program type.

**Delta.** The September 15 report proposed artifact-bound admission receipts. This mechanism is earlier and more specific: it decides *which artifact variant should enter admission* when one application supports several unstable interface shapes.

**Artifact.** A resolver library, a normalized scoped-capability schema, and fixtures for kfunc, iterator, `struct_ops`, and device-specific feature discovery.

**Evaluation.** Construct hosts with identical kernel versions and BTF but different device support or program-type eligibility. Measure incorrect variant choices, fallback rate, probe cost, and cache invalidation errors. Include reboot, module reload, device replacement, and mixed-netdev cases.

**Academic value.** It tests whether capability negotiation should be modeled as a context-dependent relation rather than a flat feature set.

**Production value.** Operators get a concrete answer to "why this variant on this node and device?" and can invalidate only the receipts whose scope changed.

**Failure condition.** If nearly all useful BPF-facing capabilities are effectively host-global in real deployments, scoped negotiation may add complexity without enough avoided failures.

### Direction 3: drive compatibility CI from interface diffs and artifact dependencies

**Gap.** Kernel matrices are expensive, while static support tables become stale. Neither directly says which application variants are affected by a particular interface change.

**Mechanism.** Record each artifact's interface dependencies from the requirement manifest. For every candidate kernel, compute normalized deltas in relevant BTF signatures, kfunc visibility/effects, iterator state/protocols, `struct_ops` schemas, and provider feature inventories. Use the dependency graph to choose the smallest set of artifact/kernel/provider combinations that cover every changed requirement.

This does not replace broad periodic testing. It adds a targeted layer that can run on every kernel update and explain why a test cell exists.

**Artifact.** An interface-diff tool, dependency graph, and CI planner that emits a reproducible test matrix and links each selected cell to a changed contract.

**Evaluation.** Replay historical kernel releases plus synthetic changes. Compare full Cartesian matrices, fixed LTS/current sampling, and dependency-selected matrices. Measure compatibility failures detected, matrix size, time to diagnosis, and missed interactions between independently unchanged interfaces.

**Academic value.** The general problem is test selection for a moving typed kernel/application boundary under incomplete dependency information.

**Production value.** eBPF projects can spend CI budget on interfaces their shipped artifacts actually consume and get early warning when an unstable dependency changes.

**Failure condition.** If most failures arise from interactions outside the recorded interface dependencies, matrix reduction would create dangerous blind spots and the planner should remain advisory only.

## A practical loader can still keep the verifier as final authority

The proposal is not to move the verifier into userspace. A production path can remain simple:

```text
collect scoped target evidence
        -> reject obviously incompatible variants
        -> choose the best matching artifact
        -> run CO-RE relocation
        -> let the kernel verifier make the final admission decision
        -> attach and, where needed, validate provider/runtime behavior
        -> record the negotiation and admission evidence
```

This layered design preserves the kernel's authority while making failures before and around the verifier easier to explain. It also avoids pretending that unstable interfaces are stable. The loader instead knows which parts of the contract it depends on and re-evaluates them when their scope changes.

For applications using only stable helpers and program types, the manifest can stay small or empty. The mechanism earns its cost only for interfaces whose evolution or provider scope already forces projects to maintain several paths.

## What would change this conclusion?

A broad empirical study could show that ordered trial loading already handles unstable BPF interfaces with almost no ambiguous failures, negligible startup cost, and enough diagnostic information for production. In that case, a separate negotiation layer would mostly duplicate the verifier.

The conclusion would also weaken if kfunc, iterator, and `struct_ops` interfaces converge toward practical long-term stability and provider-specific capability differences become rare. The benefit of explicit contracts depends on real interface movement.

Finally, dependency-driven CI is only useful if interface changes predict application risk. If historical failures are dominated by unrelated verifier behavior or cross-subsystem interactions that the dependency graph cannot capture, a smaller targeted matrix would be less safe than broad testing.

Current Linux interfaces point to a more nuanced boundary. Kfuncs explicitly have no hard stability guarantee, program-type visibility matters, open-coded iterators expose a protocol and state layout, and XDP metadata support can vary by device. The deployable contract should therefore be **typed, scoped, and verifier-backed**, rather than a Boolean answer to whether one name exists.

## Sources

- [Linux kernel documentation: BPF Kernel Functions (kfuncs)](https://docs.kernel.org/bpf/kfuncs.html)
- [Linux kernel documentation: BPF Iterators](https://docs.kernel.org/bpf/bpf_iterators.html)
- [Linux kernel documentation: XDP RX Metadata](https://docs.kernel.org/networking/xdp-rx-metadata.html)
- [Linux kernel documentation: Extensible Scheduler Class](https://docs.kernel.org/scheduler/sched-ext.html)
- [Linux kernel documentation: BPF Design Q&A](https://docs.kernel.org/bpf/bpf_design_QA.html)
- [Eunomia Daily Report: Can an eBPF Loader Trust the Kernel Version?](https://eunomia.dev/research/ebpf-kernel-capability-evidence/)
- [Eunomia Daily Report: Can an eBPF Object Keep Its Meaning After a Kernel Upgrade?](https://eunomia.dev/research/ebpf-kernel-upgrade-semantic-compatibility/)
