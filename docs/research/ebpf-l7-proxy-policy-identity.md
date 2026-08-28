---
date: 2026-08-28
title: "Can eBPF Keep Policy Identity Across an L7 Proxy Handoff?"
description: "Cilium already carries source identity across proxy connections, but request lineage and policy generations can still diverge under updates, pooling, retries, and fast-path fallback."
tags:
  - Daily Report
  - eBPF
  - Networking
  - Security
  - Service Mesh
  - Envoy
  - L7
research_question: "How can an eBPF security datapath preserve request-scoped identity, policy generation, and authorization provenance when traffic crosses an L7 proxy and leaves on a different socket or connection?"
source_cutoff: 2026-08-28
status: daily-report
---

# Can eBPF Keep Policy Identity Across an L7 Proxy Handoff?

A network policy can make a correct decision before a request enters a proxy and still lose enough context to make a later decision ambiguous.

The hard case is not that production systems carry no identity across the proxy. Cilium already does more than that. Its Envoy integration performs policy lookups, its proxy datapath has explicit source-identity marks for upstream traffic, and recent bug reports show connection-tracking entries carrying source security identities into L7 policy processing.

That implementation evidence makes the remaining question narrower and more useful: **what must survive when the security property belongs to a logical request, but the implementation state belongs to a connection?**

<!-- more -->

An L7 proxy terminates a downstream connection, parses HTTP or gRPC, then opens or reuses an upstream connection. A single logical request can therefore cross packet, socket, proxy-request, and new-socket representations. Connection pooling can map many requests onto one upstream socket; retry can map one request onto several sockets; policy updates can change the valid identity or generation while an old connection still exists.

This report argues for a **policy-identity handoff contract**. Existing connection-scoped identity propagation should be treated as a strong baseline, not as the end of the problem. The missing property is request-scoped authorization lineage that remains valid across representation changes and is explicitly tied to the policy generation that authorized the request.

This is different from the previous Daily Report on [complete mediation across host and offload paths](https://eunomia.dev/research/ebpf-complete-mediation-offload/). Complete mediation asks whether every reachable path crosses a valid enforcement point. Here we assume the request does cross the intended points and ask whether those points still refer to the same security subject and authorization generation.

It is also different from [multi-owner policy composition](https://eunomia.dev/research/ebpf-network-policy-composition/). Composition determines which policy wins. The handoff problem starts after that decision and asks whether the resulting authority remains bound to the correct logical request.

## Production systems already carry identity, but mostly at connection boundaries

Cilium's current L7 policy documentation states that L7 policy traffic is proxied through a node-local Envoy instance. Its Envoy build contains custom policy-enforcement filters, and the Cilium agent communicates with Envoy over Unix-domain sockets for configuration, access logs, and administration. See the current [Cilium Envoy documentation](https://docs.cilium.io/en/latest/security/network/proxy/envoy/) and [Layer 7 policy documentation](https://docs.cilium.io/en/latest/security/policy/layer7/).

For Ingress and Gateway API, Cilium documents two logical policy-enforcement points around the per-node Envoy proxy. External traffic commonly starts as the `world` identity and traffic leaving the proxy is evaluated through the special `ingress` identity. See [Cilium Ingress and Network Policy](https://docs.cilium.io/en/latest/network/servicemesh/ingress/).

The implementation goes further than the high-level documentation. Cilium's current BPF headers define proxy marks such as `MARK_MAGIC_PROXY_INGRESS` and `MARK_MAGIC_PROXY_EGRESS` as carrying a source identity for upstream traffic. See [`bpf/lib/common.h`](https://github.com/cilium/cilium/blob/main/bpf/lib/common.h). A recent production issue also shows Envoy's BPF metadata path applying a source identity to an upstream connection.

This is important because it rules out a weak thesis. The research gap is not simply "the proxy forgets the source identity." Cilium already has machinery to preserve connection-level identity through the proxy datapath.

The same implementation evidence exposes a more precise failure mode. In [Cilium issue #44912](https://github.com/cilium/cilium/issues/44912), opened in March 2026, an endpoint identity transition can leave an existing conntrack entry with an old `src_sec_id`. After the old identity is garbage-collected, L7 policy processing can fail for the established connection while a new connection uses the new identity. The bug is a concrete example of why identity needs lifetime and generation semantics, not just a numeric value copied across a boundary.

Linux BPF primitives reinforce this distinction. `BPF_MAP_TYPE_SOCKMAP` and `BPF_MAP_TYPE_SOCKHASH` can redirect messages between sockets and attach socket-level verdict programs. Userspace lookup returns a socket cookie rather than a kernel socket pointer. These are useful transport identities, but they identify socket lifetimes, not arbitrary HTTP/2 streams or retry lineages. See the Linux [sockmap and sockhash documentation](https://docs.kernel.org/next/bpf/map_sockmap.html).

The problem becomes more visible when L7 work moves into kernel fast paths. [L7FP](https://arxiv.org/abs/2605.31084), published in May 2026, synthesizes an eBPF fast path for common service-mesh L7 policies and falls back to an existing userspace proxy for unsupported cases. That split is attractive for performance, but it adds a correctness obligation: fast and slow paths should preserve equivalent authorization lineage, not only equivalent allow/deny verdicts.

## Socket identity is not request identity

For simple TCP forwarding, carrying a source identity on an upstream socket may be sufficient. Multiplexing and retries break the one-request-per-socket assumption.

Consider an HTTP/2 connection from a proxy to one backend. Several downstream requests can share that upstream transport. A socket-scoped source identity is meaningful only if every request that can use the connection is equivalent for all policy decisions made after the proxy. If two requests differ in principal, policy generation, or still-relevant L7 authorization, one socket-level label cannot describe both without an additional rule.

Retry creates the inverse mapping. One logical request may use several upstream connections. Authorization should follow the request, but a socket-local record alone follows only one transport instance.

A useful model therefore separates three identifiers:

- **transport identity**, tied to a concrete packet, flow, or socket lifetime;
- **logical request identity**, tied to one HTTP, gRPC, or other application request across retries and scheduling;
- **policy identity**, containing the principal, authorization scope, and policy generation that justify the request.

A production design may compress or coalesce these identifiers when it can prove equivalence. It should not assume they are interchangeable.

## Where current work is still weak

### Connection-scoped source identity lacks an explicit request lineage contract

Cilium's source-identity propagation is a strong baseline. What is not explicit in the documented contract is how that connection-scoped identity relates to a logical request when a proxy pools, multiplexes, retries, or changes upstream transports.

The missing abstraction is a lineage relation that can answer: which downstream principal, L7 decision, and policy generation caused this specific upstream request?

A discriminating test should mix downstream identities through one proxy, force connection reuse and retries, and verify that every accepted upstream request maps to exactly one valid authorization lineage.

### Identity transitions need generation-aware lifetime semantics

Cilium issue #44912 shows that an established connection can retain an obsolete source security identity after the endpoint has transitioned to a new identity. That is not merely a cache invalidation detail. It demonstrates that the authorization context attached to a long-lived transport can outlive the policy or identity state that made it meaningful.

A robust handoff therefore needs an explicit generation or revocation epoch. The consumer of propagated identity must be able to decide whether that identity is still valid for this request, rather than assuming an established transport freezes authorization forever.

### Fast-path fallback needs lineage equivalence, not only verdict equivalence

L7FP makes fast-path and proxy-slow-path execution a concrete modern design. Both paths can return the same allow/deny result in a functional test while producing different provenance for an accepted request.

The missing evaluation is **authorization-lineage equivalence** across fast and slow paths, including pooling, retry, policy updates, identity transitions, and proxy restarts.

## Promising directions with academic and production value

### 1. Generation-scoped proxy handoff capabilities

**Gap.** Existing systems can propagate a source identity on proxy-related connections, but connection-scoped identity does not by itself bind one logical request to the policy generation that authorized it.

**Mechanism.** Mint a compact capability at the eBPF-to-proxy boundary. It binds the principal, policy generation, destination scope, nonce, expiry or revocation epoch, and the proxy trust domain allowed to refine it. After L7 parsing, the proxy derives a request-scoped witness. A backend-side eBPF path validates the witness and current generation before treating the request as an authorized continuation.

The first prototype should build on existing source-identity propagation rather than replace it. A connection mark or conntrack identity can remain the fast common case, while a generation-scoped handle supplies the missing lifetime and request-lineage semantics.

**Delta.** Complete mediation proves that enforcement happened. Existing Cilium source-identity propagation shows that identity can cross a proxy connection. The proposed mechanism adds request scope and policy-generation validity.

**Artifact.** An eBPF redirect component, a small Envoy extension, a generation table, and an egress validator with a machine-readable lineage record.

**Evaluation.** Reproduce identity transitions like the one in issue #44912, then add pooling, HTTP/2 multiplexing, retry, proxy restart, backend changes, and policy churn. Measure stale-generation accepts, principal misattribution, false rejects, state size, lookup cost, and request latency.

**Academic value.** The work extends reference-monitor reasoning across a semantic transformation and across authorization-state lifetime changes.

**Production value.** It provides a concrete path from today's connection-level identity propagation to auditable request-level least privilege.

**Failure condition.** If current production meshes already expose a request-scoped, generation-aware mechanism that survives pooling, retry, restart, and identity transition without ambiguity, the research contribution should shift to formalizing and evaluating that mechanism.

### 2. Policy-safe multiplexing and request-to-socket coalescing

**Gap.** eBPF naturally caches by packet, flow, or socket while modern proxies may multiplex many logical requests over one upstream transport.

**Mechanism.** Treat connection-pool coalescing as a security proof obligation. Requests may share one upstream transport only if their security contexts are equivalent for every policy property enforced after the proxy. Otherwise the proxy preserves request-level witnesses or partitions the pool by a versioned policy-equivalence class.

An equivalence class can include destination identity, principal class, policy generation, and the subset of L7 attributes still relevant downstream. A policy update that changes equivalence invalidates or drains incompatible pooled state.

**Delta.** This is not ordinary pool partitioning for performance. The grouping rule is derived from downstream authorization semantics.

**Artifact.** An Envoy connection-pool extension plus a small policy-equivalence API consumed by the eBPF control plane. A debug view should explain why two requests were allowed to share a transport.

**Evaluation.** Exercise HTTP/1.1 keepalive, HTTP/2, gRPC streams, retries, hedged requests, identity changes, and backend reuse across multiple principals. Compare naive one-identity-per-socket caching, no pooling, and policy-aware coalescing.

**Academic value.** The work asks when many logical principals may safely share one cached physical resource.

**Production value.** It preserves proxy efficiency without silently erasing distinctions that later enforcement still needs.

**Failure condition.** If downstream enforcement never depends on caller identity after L7 authorization, or existing proxies already partition every relevant pool by an equivalent security context, the extra mechanism adds little value.

### 3. A confused-deputy and fast/slow-path lineage benchmark

**Gap.** Network benchmarks usually emphasize throughput, latency, update time, or final allow/deny correctness. They rarely check whether the accepted request still carries the correct authorization lineage after several representation changes.

**Mechanism.** Build a ground-truth harness that assigns each logical request a principal, policy generation, allowed L7 operation, destination scope, and expected backend identity. Force requests through eBPF fast paths, proxy slow paths, mixed fallback, pooling, multiplexing, retries, identity transitions, policy updates, and proxy restarts.

The primary metric is an **authorization-lineage violation**: an accepted upstream request whose observed principal, generation, destination scope, or L7 authorization cannot be matched to the ground-truth request that caused it.

**Delta.** The complete-mediation benchmark finds a packet that escaped enforcement. This benchmark assumes mediation occurred and asks whether the mediated request carried stale or incorrect authority.

**Artifact.** A reproducible Kubernetes testbed, Cilium/Envoy configuration, mixed-principal workload generator, fault injector, reference authorization log, and trace format for comparing fast and slow paths.

**Evaluation.** Start with one-request-per-connection workloads, then progressively add pooling, multiplexing, retry, identity transition, and policy churn. A proposed mechanism earns its complexity only if it removes lineage violations that the existing connection-level baseline exposes.

**Academic value.** The benchmark makes a cross-layer security property measurable across kernel, CNI, service-mesh, and proxy implementations.

**Production value.** Operators can test whether L7 acceleration and proxy integration preserve authorization meaning under real lifecycle changes, not only whether a request ends with HTTP 200 or 403.

**Failure condition.** If several independent implementations maintain zero lineage violations under these adversarial workloads, the gap is likely already solved by existing practice.

## What would change this conclusion?

The strongest evidence against a new handoff mechanism would be a production-grade implementation that already carries request-scoped, tamper-resistant authorization identity and policy generation across downstream termination, upstream pooling, HTTP/2 or gRPC multiplexing, retries, proxy restart, identity transition, policy update, and fast-path fallback.

Cilium's existing source-identity propagation is evidence that the basic connection handoff is already solvable. Issue #44912 is evidence that lifetime changes can still invalidate a connection-scoped identity. The proposed work is justified only if a benchmark finds failures beyond what existing propagation and invalidation mechanisms handle.

A second boundary is enforcement placement. Some deployments intentionally make Envoy the final authority for all L7 decisions and require later policy to trust only the proxy identity. In that model, carrying the original caller into a later eBPF decision may not help. Request-scoped lineage matters only when downstream enforcement, auditing, revocation, rate control, or provenance still depends on the original authorization context.

The practical next step is therefore the benchmark, not another metadata format. Reproduce stale identity across policy or endpoint transitions, add pooling and retry, then determine whether a compact generation-scoped handoff closes a real gap that existing connection-level identity propagation cannot.