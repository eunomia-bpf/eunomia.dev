---
date: 2026-08-26
title: "How Long Can a Revoked Authorization Stay Alive in an eBPF Datapath?"
description: "eBPF datapaths cache authorization in connection, socket, auth, and policy state. This report develops scoped revocation epochs, barriers, and a benchmark."
tags:
  - Daily Report
  - eBPF
  - Security
  - Networking
  - Revocation
research_question: "How can an eBPF security system bound the stale-authorization window after a previously allowed decision has been materialized in persistent datapath state, without flushing all state or consulting userspace on every event?"
source_cutoff: 2026-08-26
status: daily-report
---

# How Long Can a Revoked Authorization Stay Alive in an eBPF Datapath?

Suppose a workload is allowed to open a connection at 10:00. At 10:01 its identity is revoked. The policy object has changed, but packets are still moving through a datapath that already remembers the old decision.

Where is the authorization now?

It may be represented by an endpoint policy revision, a BPF policy-map entry, connection-tracking state, an authentication cache, socket-local BPF storage, or a socket reference held in a sockmap. Some of that state was created because the old policy allowed the flow. Some of it may live as long as the socket or connection. Updating the policy source does not by itself prove that every copy of the old authorization has stopped affecting future packets.

<!-- more -->

This is a different problem from asking whether a state transition is legal. The previous Daily Report, [Can eBPF Verify a Stateful Security Policy, Not Just Safe Bytecode?](https://eunomia.dev/research/ebpf-stateful-policy-verification/), asks whether persistent security state follows a valid transition relation. Here the transition `ALLOWED -> REVOKED` can be perfectly legal. The missing property is **when the old allow stops being usable everywhere that cached it**.

A production system needs more than eventual convergence. Some revocations are routine configuration changes. Others happen because a credential was compromised, a workload changed identity, an endpoint entered quarantine, or a tenant lost access. In those cases the operator wants a bound such as: after revision 42 revokes principal X, no enforcement path may accept X using revision 41 for more than 100 milliseconds.

That sounds like a control-plane synchronization problem, but eBPF makes it a datapath-state problem too. The fast path is fast precisely because it can reuse state instead of re-evaluating every decision in userspace.

## Authorization already outlives one policy lookup

Linux provides several primitives that make cached state practical.

[`BPF_MAP_TYPE_SK_STORAGE`](https://docs.kernel.org/bpf/map_sk_storage.html) stores BPF values directly with a socket. The kernel frees that storage when the socket or map is deleted, and both BPF programs and userspace can create, update, read, or delete it. This is useful for keeping per-socket metadata close to enforcement, but its lifetime is the socket lifetime unless the application defines a shorter validity rule.

[`BPF_MAP_TYPE_SOCKMAP` and `BPF_MAP_TYPE_SOCKHASH`](https://docs.kernel.org/bpf/map_sockmap.html) hold socket references and support BPF verdict and redirection programs. Again, a policy decision can be represented by state that remains after the event that created it.

Cilium shows the same pattern at a larger scale. Its [policy documentation](https://docs.cilium.io/en/latest/security/network/policyenforcement/) explicitly defines session-oriented network policy as stateful. If `A => B` is allowed, reply traffic for that connection is automatically allowed even though a new connection in the reverse direction would require its own rule. Its [BPF-map documentation](https://docs.cilium.io/en/latest/network/ebpf/maps/) lists node-scoped connection-tracking and authentication maps as production datapath state.

Current Cilium source makes the implementation boundary visible. Its endpoint policy code maintains desired and realized policy revisions, can regenerate endpoints asynchronously, and may succeed for some endpoints while failing for others. The CLI even provides [`cilium-dbg policy wait`](https://docs.cilium.io/en/stable/cmdref/cilium-dbg_policy_wait/) to wait for all endpoints to reach a requested revision. Policy convergence is therefore an observable operation rather than an instantaneous property.

There are already narrower expiration mechanisms. Cilium's mutual-authentication design stores authentication expiry with cached authorization, and current Helm configuration exposes authentication garbage-collection intervals. Expiry is valuable, but a certificate timeout and an emergency revocation have different semantics. A five-minute garbage-collection interval does not establish a five-minute upper bound on stale authorization unless the datapath itself checks a validity condition before using the entry.

The useful question is not whether these systems have a revocation bug. It is whether eBPF security systems have a general, inspectable way to state and measure **how long previously admitted authority may remain effective**.

## A policy revision is not yet a revocation proof

A revision number says which policy a component believes it has realized. It does not necessarily enumerate every derived object that still carries authority from an older revision.

Consider four states:

```text
policy map:       principal X is denied in revision 42
endpoint:         realized policy revision = 42
conntrack entry:  flow F was admitted under revision 41
socket storage:   auth_generation = 41, authenticated = true
```

The first two lines can be correct while the last two remain dangerous, depending on how the datapath treats established flows. Conversely, deleting all connection state to force a fresh decision may terminate or perturb legitimate traffic that was unaffected by the revocation.

This creates a classic systems trade-off. Rechecking everything gives fresh authority but destroys the fast-path benefit. Never rechecking keeps the fast path cheap but lets cached decisions outlive the policy that justified them. Periodic timeout sits in between but gives a stale window determined by the timeout rather than by the urgency of the revocation.

A better interface would make that trade-off explicit.

## Where current work is still weak

### 1. Cached authorization rarely carries an explicit stale-allow budget

A connection or socket can remember that an earlier decision succeeded, but the state often does not say how long that success is allowed to remain authoritative after the policy or identity changes.

The missing element is a validity contract attached to authorization-bearing state. It should identify the policy or principal generation that justified the allow and the maximum stale interval after a revocation of that generation.

The consequence is operational ambiguity. An operator can see that a new policy revision is installed without knowing whether an old connection, authentication record, or socket-local cache can still authorize traffic.

A direct test is simple: admit long-lived flows, revoke the relevant identity or rule, then measure the last packet that is still accepted because of pre-revocation state. Repeat across connection tracking, authentication caches, socket-local state, endpoint regeneration, and controller failure.

### 2. Control-plane convergence and datapath invalidation are different completion conditions

Cilium exposes desired and realized policy revisions and provides a command that waits for endpoint updates. This is useful because policy rollout can be asynchronous. But a generic revocation contract needs one more question: which derived authorization caches must be invalidated before the revocation is considered complete?

The missing element is a cross-layer completion barrier. An endpoint reaching revision 42 is not sufficient if an authorization-bearing object outside that endpoint's policy map can still accept revision 41.

The consequence is a false sense of completion. A control plane may report convergence while stale authority remains reachable in a different map, socket, node, or userspace redirection path.

The test is to deliberately delay one enforcement domain while allowing the others to converge, then check whether the system reports success early or keeps the revocation pending.

### 3. Flush-all invalidation is correct only by being unnecessarily destructive

One safe reaction to stale state is to delete everything that might depend on the old policy. That can remove the stale allow, but it can also discard unrelated connection state and load-balancing decisions. Cilium's own configuration comments warn that rebuilding BPF state can disrupt ongoing connections and change decisions for established traffic.

The missing element is selective invalidation: a way to find or cheaply reject only state derived from the revoked subject, policy generation, credential, or authority domain.

The consequence is that urgent revocation can force a bad choice between a broad outage and a long stale window.

The test is to revoke one principal in a workload with many unrelated long-lived connections. Compare stale permits, false drops, reconnects, and convergence time for flush-all, timeout-only, and selective mechanisms.

### 4. Controller downtime can turn an eventual timeout into an unbounded assumption

A userspace controller can garbage-collect map entries and repair state, but a security bound should not silently depend on the controller being scheduled on time. Earlier Cilium mutual-authentication design work called out this exact concern: without datapath-visible expiry, authenticated sessions could remain accepted during prolonged agent downtime.

The missing element is an enforcement rule whose safety condition is checkable in the datapath even when userspace is delayed.

The consequence is that the most stressful failure mode, loss of the controller during an incident, can be the one that weakens revocation.

The test is to stop or partition the controller immediately before revocation and measure whether the stale-allow bound still holds.

## Promising directions with academic and production value

### 1. Put scoped revocation epochs on authorization-bearing state

**Gap.** Cached allows do not have a cheap, uniform way to prove that the authority that created them is still current.

**Mechanism.** Associate each authorization domain with a monotonically increasing revocation epoch. A domain can be an identity, endpoint, policy subject set, credential class, or other bounded unit chosen by the policy compiler. Any cached allow that may bypass a fresh policy lookup stores the epoch that justified it:

```text
cached_allow = {
  principal_id,
  policy_id,
  revocation_epoch,
  optional_deadline,
  decision_metadata
}
```

The fast path compares the cached epoch with the current epoch for the relevant domain before reusing the allow. Revocation increments the epoch. Old entries do not need to be synchronously deleted to become unusable.

The design should avoid one global epoch because one tenant's revocation would invalidate every cached decision. A small hierarchy can keep the hot lookup bounded: for example, endpoint epoch plus principal epoch, with a compile-time rule defining which one each cache entry must carry.

For socket-local storage, the epoch lives with the socket. For connection or authentication maps, it is part of the value or derivable from a compact side map. The mechanism can use a deadline as a fallback when a domain cannot be indexed precisely.

**Delta from related work.** Ordinary policy revisions describe configuration progress, and authentication expiry describes time validity. A scoped revocation epoch instead makes cached authorization depend on a specific invalidation generation and lets the datapath reject stale authority without deleting the object immediately. It is narrower than the temporal state contract in the previous report because it does not describe arbitrary legal transitions. It targets one property: bounded reuse of a previously allowed decision after revocation.

**Artifact.** A libbpf-friendly authorization-state schema, generated lookup helpers, scoped epoch maps, and adapters for socket storage plus connection/authentication maps.

**Evaluation.** Compare no recheck, TTL-only, global epoch, scoped epoch, and synchronous deletion. Use long-lived TCP, UDP request/reply traffic, mutual-authentication caches, socket-local state, and mixed tenants. Measure stale-allow duration, stale packets, false invalidations, per-packet cycles, cache misses, map memory, and update fan-out. Include a workload where no revocation occurs to expose steady-state overhead.

**Academic value.** The general question is how to add revocation consistency to cached in-kernel authorization without turning each event into a remote policy lookup.

**Production value.** Operators get a measurable emergency-revocation bound while preserving unrelated long-lived flows.

**Failure condition.** If the extra epoch lookup costs as much as simply re-running policy for the target workloads, or if real authorization domains cannot be scoped without invalidating most state anyway, the mechanism loses its advantage.

### 2. Make revocation completion a cross-layer barrier, not a controller guess

**Gap.** Updating policy objects and waiting for endpoint revisions does not by itself prove that every authorization-bearing domain has crossed the revocation boundary.

**Mechanism.** Give each revocation a monotonically increasing `revocation_id` and a declared set of required enforcement domains. Domains can include endpoint policy maps, identity caches, authentication maps, connection state, socket policy, and userspace-managed redirection state. Each domain publishes a high-watermark after it can guarantee that decisions older than the revocation are either rejected by an epoch check or removed.

The controller reports the revocation effective only when all required watermarks reach the target ID:

```text
revocation 82 requires:
  endpoint_policy >= 82
  auth_cache      >= 82
  connection_auth >= 82
  socket_policy   >= 82
```

A domain that cannot make progress is explicit. The policy can choose a bounded fail-closed mode, a short lease, or a degraded `unknown` state instead of quietly reporting success.

This resembles rollout barriers used by distributed systems, but the novelty target is the boundary between control-plane policy revision and datapath-derived authority. Cilium's `policy wait` provides a strong baseline for endpoint revision convergence. The experiment must show cases where endpoint convergence alone does not imply revocation completion and where the extra barrier catches them.

**Delta from related work.** Transactional upgrade protocols try to switch a coherent program/map generation. This proposal does not require all state to change atomically. It requires a verifiable upper bound on when an old authorization becomes unusable across independently updated enforcement domains.

**Artifact.** A revocation controller, compact per-domain acknowledgment maps, a status API that separates requested/realized/effective revisions, and failure injection hooks.

**Evaluation.** Inject delayed endpoint regeneration, one stale node, controller crash/restart, map-update failure, socket lifetime extension, and an authentication-cache delay. Compare endpoint-revision waiting, fixed sleep, global flush, and the barrier. Metrics are early-success errors, revocation completion latency, stale permits after reported success, false denies, and recovery time.

**Academic value.** This turns revocation from an eventually consistent operational action into a consistency property that can be tested across heterogeneous enforcement layers.

**Production value.** An incident responder can ask whether revocation is actually effective instead of whether a configuration object was accepted.

**Failure condition.** If all practical authorization paths already share one generation and one update boundary, the barrier is unnecessary plumbing. The benchmark should include such a simple system as a counterexample where the simpler design wins.

### 3. Build a revocation benchmark around the last stale allow

**Gap.** Security benchmarks often measure policy throughput or rule-update rate. Those numbers do not answer the incident-response question: after a revoke request, when is the last stale packet, syscall, or message still accepted?

**Mechanism.** Construct workloads with known authorization lineage and inject revocations at controlled times. Every accepted event carries enough test-only identity to determine whether it was authorized by pre- or post-revocation state. The benchmark records four timestamps:

1. revoke requested;
2. policy source updated;
3. each enforcement domain reports the revocation realized;
4. last event accepted using stale authority.

Faults should include control-plane pause, endpoint regeneration delay, CPU contention, map pressure, long-lived TCP flows, UDP reply state, socket-local caches, program replacement, and node partition. A second class should change an unrelated identity at the same time to measure collateral invalidation.

**Artifact.** An open trace/replay corpus, eBPF fault-injection adapters, ground-truth revocation labels, and a report format for stale-allow distributions.

**Evaluation.** Compare current system behavior with TTL-only expiry, flush-all, scoped epoch, and epoch-plus-barrier designs under the same policy-update workload. Primary metrics are P50/P99/max stale-allow duration, number of stale permits, false denies, connection disruption, CPU overhead, and map update amplification. The maximum matters because a long tail is exactly what an emergency revocation contract is supposed to bound.

**Academic value.** The benchmark makes a security consistency property measurable across systems that otherwise expose incomparable policy-update mechanisms.

**Production value.** CNI, runtime-security, service-mesh, and socket-policy projects get a regression test for the revocation failure mode most likely to hide behind a successful configuration update.

**Failure condition.** If real deployments cannot attach reliable ground truth to which cached decision authorized an event, the benchmark may be useful only for controlled prototypes. In that case the first research contribution should be better authorization provenance rather than a universal score.

## What would change this conclusion?

The strongest counterargument is that existing policy regeneration, connection-state handling, authentication expiry, and targeted map updates already make stale authorization short enough for real deployments. If a ground-truth benchmark shows that current systems consistently revoke within the required bound under long-lived flows, controller failure, map pressure, and mixed tenants, then a new revocation protocol would add complexity without improving security.

The second counterargument is cost. A generation comparison on every cached fast-path allow may add an extra map lookup or cache miss. If that overhead erases the performance benefit of caching, selective synchronous deletion or shorter leases may be a better engineering choice.

The useful result is therefore not “every eBPF system needs epochs.” It is a clearer contract: **a security system that reuses cached authorization should be able to state the maximum stale-allow window, identify which state can carry old authority, and demonstrate that the bound still holds when the control plane is delayed.**

That property can be implemented with epochs, leases, targeted invalidation, or another mechanism. What should no longer be acceptable is treating “the new policy revision is installed” as automatic proof that every old allow is already dead.

## Sources

- Linux kernel documentation: [`BPF_MAP_TYPE_SK_STORAGE`](https://docs.kernel.org/bpf/map_sk_storage.html)
- Linux kernel documentation: [`BPF_MAP_TYPE_SOCKMAP` and `BPF_MAP_TYPE_SOCKHASH`](https://docs.kernel.org/bpf/map_sockmap.html)
- Cilium documentation: [Policy Enforcement](https://docs.cilium.io/en/latest/security/network/policyenforcement/)
- Cilium documentation: [Endpoint Lifecycle](https://docs.cilium.io/en/latest/security/policy/lifecycle/)
- Cilium documentation: [eBPF Maps](https://docs.cilium.io/en/latest/network/ebpf/maps/)
- Cilium CLI documentation: [`cilium-dbg policy wait`](https://docs.cilium.io/en/stable/cmdref/cilium-dbg_policy_wait/)
- Cilium source: [`pkg/endpoint/policy.go`](https://github.com/cilium/cilium/blob/main/pkg/endpoint/policy.go)
- Cilium source: [`bpf/lib/conntrack.h`](https://github.com/cilium/cilium/blob/main/bpf/lib/conntrack.h)
- Cilium source: [`bpf/lib/host_firewall.h`](https://github.com/cilium/cilium/blob/main/bpf/lib/host_firewall.h)
- Cilium Helm configuration: [authentication and BPF map settings](https://github.com/cilium/cilium/blob/main/install/kubernetes/cilium/README.md)
- Cilium design CFP: [Mutual authentication updates](https://github.com/cilium/design-cfps/blob/main/cilium/CFP-28986-mutual-auth-updates.md)
