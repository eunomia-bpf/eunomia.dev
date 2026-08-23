---
date: 2026-08-23
title: "How Should eBPF Compose Multi-Tenant Network Policies?"
description: "Multi-tenant clusters combine admin, namespace, and Cilium policy layers. This report develops explainable eBPF policy composition with conflict witnesses."
tags:
  - Daily Report
  - eBPF
  - Networking
  - Security
  - Kubernetes
research_question: "How can an eBPF network-policy datapath preserve correct multi-tenant precedence while explaining which policy owner and rule determined each effective verdict?"
source_cutoff: 2026-08-23
status: daily-report
---

# How Should eBPF Compose Multi-Tenant Network Policies?

A platform administrator denies cross-tenant traffic. A namespace owner adds a Kubernetes `NetworkPolicy` that allows one service. A security team adds a Cilium L7 rule for the same workload. All three policies may be valid, and all three may eventually influence the eBPF datapath that decides whether a packet is accepted.

The difficult question is no longer whether eBPF can enforce a rule. It is whether the system can preserve **who had authority to decide, which rule won, and why a different rule did not** after several policy languages have been compiled into compact datapath state.

<!-- more -->

Kubernetes deliberately gives its original `NetworkPolicy` simple composition semantics. If several policies select the same pod, their allowed ingress or egress traffic is combined additively. There is no ordering between those policies. The current [Kubernetes NetworkPolicy documentation](https://kubernetes.io/docs/concepts/services-networking/network-policies/) describes the effective allow set as the union of applicable rules.

The newer [Network Policy API](https://network-policy-api.sigs.k8s.io/) adds a different kind of composition. `ClusterNetworkPolicy` v0.2.0 introduces `Admin` and `Baseline` tiers, priorities, and `Accept`, `Deny`, and `Pass` actions. Admin rules can impose cluster-wide decisions that ordinary workload policies cannot override, while `Pass` can deliberately delegate a decision to a lower layer. The latest released `ClusterNetworkPolicy` API is v0.2.0 according to the project's [getting-started guide](https://network-policy-api.sigs.k8s.io/getting-started/).

Cilium then adds another practical dimension. [Cilium 1.20.1](https://docs.cilium.io/en/stable/network/kubernetes/policy/) can enforce Kubernetes `NetworkPolicy`, Kubernetes `ClusterNetworkPolicy`, `CiliumNetworkPolicy`, and `CiliumClusterwideNetworkPolicy` together. Its documentation explicitly warns that using several policy formats at once can make the complete allowed set difficult to understand and can lead to unintended allows if operators do not reason about the combination carefully.

That warning exposes a systems gap. The control plane may know policy objects, owners, tiers, selectors, and source labels. The eBPF datapath needs a compact answer such as an allowed identity, port, and protocol tuple. Cilium documents a per-endpoint [Policy BPF map](https://docs.cilium.io/en/latest/network/ebpf/maps/) whose default capacity is 16,000 allowed identity, port, and protocol pairs. That representation is efficient for enforcement, but the operator's original question is richer than the lookup key: **which authority caused this tuple to exist, which higher-priority rule was considered, and what policy object should be changed if the verdict is wrong?**

This report argues that multi-tenant eBPF policy needs an explicit **composition contract** between policy intent and datapath state. The contract should normalize policies from different owners into one decision model, detect meaningful conflicts before installation, and preserve a compact verdict witness so a datapath decision can be traced back to the policy authority that determined it.

This is distinct from the earlier [eBPF hook-composition report](https://eunomia.dev/research/ebpf-hook-composition-contract/), which asked how several BPF programs sharing one hook should combine mutations and outcomes. It is also distinct from the [stateful eBPF transactional-upgrade report](https://eunomia.dev/research/stateful-ebpf-transactional-upgrade/), which asked how programs, maps, links, and controller state can switch generations safely. Here the BPF programs may stay unchanged. The missing property is the semantics of several **security-policy owners and policy languages** being compiled into one effective network decision.

## Three policy layers already use different composition rules

A composition mechanism should start from the semantics that exist rather than invent a universal deny-overrides-allow rule.

### Kubernetes NetworkPolicy is intentionally additive

The original Kubernetes API is allow-oriented. Once a pod is isolated in one direction, traffic is permitted when at least one applicable rule allows it, and the effective allowed set is the union across policies. This model is useful because namespace owners can add policies without depending on evaluation order.

It also means that the word "conflict" has to be used carefully. Two ordinary `NetworkPolicy` objects do not conflict in the API's formal semantics simply because one is narrower than another. If either allows a flow, the union allows it. An operator can still experience an *intent conflict* when one team believes a narrower rule restricts a broader rule, but the API does not interpret it that way.

That distinction matters for tooling. A useful analyzer should not report "policy conflict" merely because two selectors overlap. It should report that a policy is shadowed, broadens the effective allow set, has no effect, or changes a flow in a way that violates a declared organizational constraint.

### ClusterNetworkPolicy adds authority and delegation

The Network Policy API exists because cluster administrators need controls that the original namespaced model cannot express cleanly. Its [API reference](https://network-policy-api.sigs.k8s.io/reference/spec/) gives higher precedence to lower numeric priorities within an administrator layer, while the [ClusterNetworkPolicy model](https://network-policy-api.sigs.k8s.io/api-overview/) separates administrator and baseline responsibilities.

Cilium's current implementation documents the practical evaluation order as `Admin` tier, `NetworkPolicy` tier, then `Baseline` tier. An Admin decision cannot be overridden by `NetworkPolicy`, while Baseline acts as a lower-priority guardrail. `Pass` is especially interesting because it makes delegation part of the policy language instead of treating every unmatched rule as an implicit fall-through.

Now an explanation such as "this packet was denied" is insufficient. The operator may need to know that an Admin-tier rule made the final decision before a namespace policy could participate, or that an Admin-tier `Pass` deliberately delegated the flow and a workload policy then allowed it.

### Cilium compiles several formats into an eBPF datapath

Cilium provides the concrete eBPF setting for this problem. It can ingest several Kubernetes and Cilium policy formats, calculate endpoint policy, assign security identities, and program BPF state for enforcement. Its [troubleshooting guide](https://docs.cilium.io/en/stable/security/policy/troubleshooting/) shows how an operator can pair endpoint information with policy labels and `cilium-dbg policy get` to reconstruct which source policies apply to an endpoint.

That is useful control-plane introspection, but it is a reconstruction workflow. The packet verdict itself is compact. Hubble can report a `policy-verdict` with source and destination identities, direction, port, and allow/drop state, as shown in Cilium's [policy-creation guide](https://docs.cilium.io/en/latest/security/policy-creation/). The evidence required to answer "which rule owned this decision?" still lives across policy repositories, endpoint state, selector caches, and the datapath representation.

The gap is therefore not that Cilium has no observability. The gap is that **policy provenance is not a portable property of the compiled verdict**.

## Multi-tenancy makes missing provenance more than a debugging inconvenience

A single team can often resolve ambiguous policy behavior by inspecting all objects it owns. Multi-tenancy changes the authority model.

Imagine a shared cluster with three roles:

1. the platform team owns Admin-tier isolation and emergency blocks;
2. tenant teams own namespace-level service connectivity;
3. a security team owns cluster-wide L7 controls through Cilium policy.

Suppose tenant `blue` expects traffic from `blue/frontend` to `blue/api` on TCP 443. The platform policy first checks whether source and destination belong to the same tenant. The tenant policy allows the service pair. The security policy restricts the HTTP method on that connection.

If the connection fails, the correct remediation depends on the deciding layer. Changing the namespace `NetworkPolicy` cannot override an Admin deny. Changing an Admin rule to fix an L7 denial expands authority unnecessarily. A policy engine that only reports the final drop forces the operator to reconstruct the hierarchy manually.

The Network Policy API's own current examples show that tenancy semantics are still evolving. Its [tenant-isolation example](https://network-policy-api.sigs.k8s.io/reference/examples/) marks the same-tenant selector needed by the sample as "currently not implementable" and points to [NPEP-122](https://network-policy-api.sigs.k8s.io/npeps/npep-122/). That proposal describes ambiguity around how tenancy itself should be represented, including strict versus overridable isolation. This is not evidence that multi-tenancy cannot be enforced today. It is evidence that **tenant identity and delegation are first-class semantics that a generic policy compiler cannot safely infer from selector overlap alone**.

## The effective policy needs a representation before it becomes BPF map entries

A practical design can separate policy-language semantics from datapath layout with an intermediate representation. Call one normalized rule:

```text
rule = {
  source_object,
  source_generation,
  owner,
  authority_tier,
  priority,
  action,
  subject_selector,
  peer_selector,
  direction,
  protocol,
  port,
  l7_constraints
}
```

The important fields are not all needed on every packet. They are needed while computing the effective decision.

For a candidate flow, the compiler evaluates rules according to the semantics of the originating API. Ordinary Kubernetes `NetworkPolicy` contributes to an additive allow set. `ClusterNetworkPolicy` contributes tier, priority, and explicit `Accept`, `Deny`, or `Pass`. Cilium L7 constraints add another enforcement stage. The compiler then produces two related outputs:

- **datapath state**, optimized for fast identity/port/protocol and L7 enforcement;
- **decision provenance**, optimized for explaining why that state exists.

The provenance does not need to copy complete YAML into a BPF map. A compact witness can use a generation ID, normalized rule ID, action, and authority tier. Userspace retains the reverse mapping from normalized rule IDs to source objects and owners.

For an allowed L3/L4 tuple, a witness might look like:

```text
policy_gen=1842
verdict=ALLOW
owner=tenant-blue
rule_id=0x31a8
layer=NetworkPolicy
admin_path=PASS:0x9f20
```

For a deny decided by the administrator layer:

```text
policy_gen=1842
verdict=DENY
owner=platform-security
rule_id=0x0d44
layer=Admin
```

This turns a packet verdict into a join key rather than a forensic exercise.

## Where current work is still weak

### 1. Cross-format policy composition is implemented, but its effective intent is hard to inspect

Kubernetes `NetworkPolicy` has additive semantics. `ClusterNetworkPolicy` adds tiered authority, priority, and delegation. Cilium policies add Cilium-specific L3-L7 constructs. Cilium can run these formats together and explicitly warns that the complete allowed set may become confusing.

The missing capability is a common **effective-policy artifact** that says how source policies from several formats contributed to each reachable decision region. A useful artifact should distinguish formal precedence from mere overlap. It should identify broadening rules, unreachable rules, delegated decisions, and rules whose effect is conditional on another policy layer.

A decisive experiment would generate policy sets using all supported formats, enumerate or symbolically sample relevant source/destination identities and ports, and compare the analyzer's effective-policy result with the actual datapath verdict for every generated flow. If a compact normalized model cannot match the implementation without embedding Cilium internals, the proposed abstraction is too generic.

### 2. Tenant identity is often implied by labels rather than declared as an owned object

The current Network Policy API tenancy proposal explains why "same tenant" is harder than one selector expression. A label can group namespaces, but the system still needs a definition of which label establishes tenant identity, who may change it, and whether isolation is strict or can be overridden by namespace owners.

The missing element is an authority-bound tenant identity that policy composition can reference directly. Without it, two policy authors can use the same labels with different assumptions, and an analyzer can only guess which label is organizational identity versus ordinary workload metadata.

The test should include tenant-label mutation and adversarial namespace creation. If changing an untrusted namespace label can make a namespace enter another tenant's policy group before an independent authority check rejects it, the composition model is incomplete.

### 3. Datapath verdicts do not naturally carry the source-policy witness

A BPF policy map has to make packet decisions quickly. Cilium's documented policy map is sized around allowed identity, port, and protocol pairs, not around retaining every source policy object that helped produce those pairs. Existing troubleshooting can recover source policies through endpoint labels and agent state, but that is separate from the verdict path.

The missing property is a **stable witness for the deciding policy generation and rule**. It can be stored in a companion map, attached to policy-verdict events, or reconstructed from an immutable generation manifest, but the relationship must survive policy updates long enough to explain an incident.

The discriminating test is to update policies rapidly while retaining verdict logs. Ask the debugger to explain verdicts from old generations after the control plane has moved on. If explanations silently resolve against the current policy instead of the policy that produced the packet decision, provenance is not strong enough.

### 4. L3/L4 and L7 rules can change effective denial behavior in different places

Cilium's [policy enforcement documentation](https://docs.cilium.io/en/latest/security/policy/intro/) notes that an L7 rule can cause drops even when `EnableDefaultDeny` is disabled unless the L7 policy explicitly allows the traffic. This is a good example of why one Boolean "default deny" flag cannot summarize the full policy stack.

The missing capability is a composition model that exposes which enforcement stage owns the final result. The operator should be able to distinguish "Admin tier denied before workload policy", "L3/L4 policy did not admit the identity", and "L7 policy admitted the connection but rejected this request" without translating each subsystem's internal representation by hand.

A useful evaluation should replay the same source/destination pair with L3, L4, and L7 changes and score whether an explanation identifies the correct deciding layer and source rule.

## Promising directions with academic and production value

### 1. Compile several policy languages into an authority-aware composition IR

**Gap.** Current APIs expose different composition semantics, and production CNI implementations can support several formats simultaneously, but operators lack one inspectable artifact for the effective result.

**Mechanism.** Build a compiler front end for Kubernetes `NetworkPolicy`, `ClusterNetworkPolicy`, `CiliumNetworkPolicy`, and `CiliumClusterwideNetworkPolicy`. Normalize selectors and actions while preserving each source language's semantics. Represent authority tier, owner, priority, explicit delegation, and enforcement layer as first-class fields. The compiler then partitions the relevant identity/port space into decision regions and emits both datapath entries and a provenance manifest.

The compiler must not flatten every format into a global "deny wins" rule. Kubernetes `NetworkPolicy` remains additive; `ClusterNetworkPolicy` tier and `Pass` semantics remain explicit; Cilium L7 constraints remain staged. Conflicts are reported only when an organizational invariant or declared ownership rule is violated, or when one rule's effect is unexpectedly broadened, shadowed, or delegated.

**Delta.** Existing CNI controllers already compile policies. The new property is that the compilation product is a **portable, inspectable composition IR with authority provenance**, rather than an implementation-specific internal policy tree plus BPF state.

**Artifact.** An open composition schema, importers for the four policy formats, a Cilium backend, and a command that can answer "what permits or denies this flow?" from a pinned policy generation.

**Evaluation.** Generate policy suites with overlapping selectors, Admin/Baseline tiers, `Pass`, namespace policies, Cilium L7 rules, and label changes. Compare the IR against Cilium's realized verdicts and the API-defined expected semantics. Measure verdict agreement, conflict-report precision, compile latency, BPF map entry count, and incremental update cost. Ablate the authority fields and show which cases become ambiguous or incorrectly flattened.

**Academic value.** The general question is whether independently authored network policies can be composed through an explicit authority algebra while preserving each language's original semantics.

**Production value.** Platform teams gain one artifact for pre-deployment review, policy diffing, incident explanation, and migration between policy APIs.

**Failure condition.** If the IR needs implementation-specific exceptions for most real policies, or existing policy tracing already exposes the same effective semantics in a stable machine-readable form, a new intermediate representation does not justify another abstraction.

### 2. Preserve a compact verdict witness through the eBPF datapath

**Gap.** Fast policy maps answer the enforcement question, while source-policy ownership and rule identity remain primarily in the control plane.

**Mechanism.** Assign every compiled decision region a generation-scoped witness ID. The normal BPF lookup continues to use compact identity/port/protocol keys. A companion value or companion map associates the resulting entry with the witness ID. Policy-verdict telemetry exports that ID on denies and on a configurable sample of allows. Userspace resolves it against an immutable manifest containing owner, tier, source object, rule, and delegation path.

The design should keep the hot path cheap. It can avoid attaching full provenance to every packet by emitting a witness only on first-seen flows, policy-verdict events, sampled allows, or operator-requested diagnostic mode. The generation must remain part of the identity so old verdict logs cannot be explained using a newer policy manifest.

**Delta.** Hubble already emits policy-verdict events, and Cilium already labels source policy rules. The new property is a **direct generation-stable join from a realized datapath decision to the deciding source-policy witness**.

**Artifact.** A Cilium prototype extending policy-map metadata or adding a companion witness map, plus Hubble decoding and a retained generation manifest.

**Evaluation.** Run connection matrices under policy churn and retain verdict events across several generations. Measure explanation accuracy after old policies have been deleted, additional BPF map memory, packet-path overhead, event bandwidth, and manifest-retention cost. Compare full per-packet provenance, sampled witnesses, and control-plane-only reconstruction.

**Academic value.** This tests how much provenance an enforcement datapath must retain for explanations to remain correct under control-plane evolution.

**Production value.** Security and SRE teams can map a denied or unexpectedly allowed flow directly to the responsible policy owner instead of correlating several live controller views after the fact.

**Failure condition.** If control-plane reconstruction stays exact across policy churn and adds negligible incident latency, or if even a compact witness materially harms datapath scale, the witness should remain outside the packet path.

### 3. Build a counterexample-driven multi-tenant policy benchmark

**Gap.** Policy conformance can show whether one resource is implemented correctly, but it does not necessarily expose operator mistakes created by several legitimate owners composing policies together.

**Mechanism.** Build a benchmark generator with explicit ground-truth tenants, authority roles, and intended connectivity invariants. It emits policy sets across Kubernetes and Cilium formats, then introduces one controlled mutation at a time: a broader namespace selector, a misplaced tier, an accidental `Pass`, a tenant-label change, an L7 restriction, a stale policy generation, or a policy-map capacity stress case.

For each mutation, the benchmark records both the expected datapath verdict and the minimal explanation: which authority and rule should determine the outcome, and which alternative rule should be ignored, delegated, or treated as additive.

**Delta.** Existing API conformance tests validate implementation behavior against resource semantics. This benchmark targets **composition mistakes and explanation quality across owners and formats**, including cases where every individual policy object is syntactically valid.

**Artifact.** A reproducible Kubernetes test suite, policy corpus, expected flow matrix, policy-generation history, and adapters for Cilium first, with other Network Policy API implementations added as their support matures.

**Evaluation.** Score datapath verdict correctness, explanation correctness, conflict detection precision and recall, time to identify the responsible owner, and resource overhead. Include a human/operator study only as a secondary measure; the primary oracle is the generated ground-truth decision graph. Compare ordinary `kubectl` inspection, Cilium's existing policy/debug tooling, the composition IR, and the datapath witness design.

**Academic value.** The benchmark makes policy compositionality and explainability measurable properties instead of relying on anecdotal configuration failures.

**Production value.** CNI maintainers and platform teams can regression-test upgrades, API migrations, and organization-specific guardrails against realistic multi-owner policy interactions.

**Failure condition.** If real production policy sets rarely mix authorities or formats and generated conflicts do not resemble observed incidents, the benchmark overstates a corner case and should narrow to the policy combinations operators actually deploy.

## A practical deployment path does not require replacing Kubernetes policy APIs

The composition contract can be incremental.

A first implementation can run entirely in the Cilium control plane. It imports current policy objects, emits a composition manifest alongside the existing realized policy, and compares predicted decisions with Cilium policy tracing. No packet-path change is required.

A second step can add witness IDs only to policy-verdict telemetry. If that is sufficient for incident explanation, there is no reason to burden every policy-map lookup with more metadata.

Only if policy churn or post-incident reconstruction proves that telemetry-only witnesses are insufficient should the datapath carry a generation-scoped witness directly.

This staged path matters because eBPF is valuable here for enforcement and observability, but the hardest part is semantic. Faster packet lookup cannot repair an authority model that was flattened incorrectly before the BPF map was populated.

## What would change this conclusion?

The case for an explicit composition contract weakens if current `ClusterNetworkPolicy` semantics plus existing Cilium policy tracing already make multi-owner effective policy mechanically explainable across the policy combinations operators actually use. A benchmark that finds no explanation mismatch, no stale-generation errors, and no meaningful operator ambiguity would favor improving existing tooling rather than adding a new IR.

The datapath witness is also conditional. If retaining a rule witness causes measurable policy-map pressure, reduces endpoint scale, or adds packet-path cost that control-plane reconstruction avoids, the witness should stay in sampled verdict telemetry or in a userspace generation manifest.

Finally, tenancy may become simpler as the Network Policy API stabilizes a clearer tenant model. If tenant identity, ownership, delegation, and precedence become explicit enough that every implementation can preserve them without extra metadata, the composition layer can shrink. The goal is not to create another policy language. It is to make the path from several legitimate policy owners to one eBPF verdict **correct, inspectable, and stable across policy generations**.

## References

- [Kubernetes Network Policies](https://kubernetes.io/docs/concepts/services-networking/network-policies/)
- [Kubernetes NetworkPolicy API reference](https://kubernetes.io/docs/reference/kubernetes-api/networking/network-policy-v1/)
- [Network Policy API](https://network-policy-api.sigs.k8s.io/)
- [Network Policy API v0.2.0 getting started](https://network-policy-api.sigs.k8s.io/getting-started/)
- [Network Policy API specification](https://network-policy-api.sigs.k8s.io/reference/spec/)
- [Network Policy API examples](https://network-policy-api.sigs.k8s.io/reference/examples/)
- [NPEP-122: Tenancy API](https://network-policy-api.sigs.k8s.io/npeps/npep-122/)
- [Network Policy API implementations](https://network-policy-api.sigs.k8s.io/implementations/)
- [Cilium 1.20.1 Network Policy](https://docs.cilium.io/en/stable/network/kubernetes/policy/)
- [Cilium eBPF Maps](https://docs.cilium.io/en/latest/network/ebpf/maps/)
- [Cilium policy troubleshooting](https://docs.cilium.io/en/stable/security/policy/troubleshooting/)
- [Cilium policy enforcement modes](https://docs.cilium.io/en/latest/security/policy/intro/)
- [Cilium repository](https://github.com/cilium/cilium)
