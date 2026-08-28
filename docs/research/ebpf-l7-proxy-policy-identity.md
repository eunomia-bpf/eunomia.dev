---
date: 2026-08-28
title: "Can eBPF Keep Policy Identity Across an L7 Proxy Handoff?"
description: "When eBPF redirects a policy-bound flow through an L7 proxy, the upstream connection can lose the identity and policy generation that justified the request."
tags:
  - Daily Report
  - eBPF
  - Networking
  - Security
  - Service Mesh
  - Envoy
  - L7
research_question: "How can an eBPF security datapath preserve the identity, policy generation, and authorization provenance of a request when traffic crosses into an L7 proxy and leaves on a different socket or connection?"
source_cutoff: 2026-08-28
status: daily-report
---

# Can eBPF Keep Policy Identity Across an L7 Proxy Handoff?

A network policy can make a perfectly reasonable decision before a request enters a proxy and still lose the reason for that decision before the request reaches the backend.

The problem appears whenever a datapath crosses a semantic boundary. An eBPF program sees a packet or socket with one source identity and one connection tuple. It redirects the flow to Envoy or another L7 proxy. The proxy terminates the downstream connection, parses HTTP, gRPC, or another application protocol, and opens or reuses a different upstream connection. The backend-side packet now belongs to the proxy's socket and may carry a different tuple, security identity, connection lifetime, and policy context.

The policy has not become less important. The system still needs to answer a simple question: **which principal and which policy generation justified this particular upstream request?**

<!-- more -->

This report argues that a cross-boundary eBPF security design needs more than path coverage. It needs a **policy-identity handoff contract**: when a request crosses from an eBPF enforcement domain into an L7 proxy and then back into another kernel or network path, the authorization context that matters to the request must survive the handoff in a form that cannot be confused with another request, another connection, or an obsolete policy generation.

This is a narrower problem than the previous Daily Report on [complete mediation across host and offload paths](https://eunomia.dev/research/ebpf-complete-mediation-offload/). Complete mediation asks whether every reachable packet path crosses a valid enforcement point. Here we assume the request does cross enforcement points. The question is whether those points are still talking about the **same security subject and authorization decision** after the proxy has terminated and re-originated traffic.

It is also different from [multi-owner policy composition](https://eunomia.dev/research/ebpf-network-policy-composition/). Composition decides which policy owner and rule should win. A proxy handoff asks whether the result of that decision remains bound to the request as representation changes from packet to socket to L7 request and back to a new socket.

## Production systems already expose the boundary

Cilium's current L7 policy documentation makes the basic topology explicit. L7 policy traffic is proxied through a node-local Envoy instance. Cilium's Envoy build contains custom policy-enforcement filters, and the Cilium agent communicates with Envoy over Unix-domain sockets for configuration and access-log exchange. See the current [Cilium Envoy documentation](https://docs.cilium.io/en/latest/security/network/proxy/envoy/) and [Layer 7 policy documentation](https://docs.cilium.io/en/latest/security/policy/layer7/).

The boundary is even clearer in Cilium Ingress and Gateway API. Cilium documents that the per-node Envoy proxy interacts with the eBPF policy engine and performs policy lookups. It also documents two logical policy-enforcement points: traffic can be checked before it reaches Envoy and again after Envoy before it reaches the backend. External traffic commonly begins with the `world` identity and is assigned a special `ingress` identity at the Envoy boundary. See [Cilium's Ingress and Network Policy description](https://docs.cilium.io/en/latest/network/servicemesh/ingress/).

That design is useful and deployable, but it reveals a general systems question. The identity at the second enforcement point is not necessarily the identity that originated the request. A proxy is intentionally a deputy. It accepts one connection and acts on behalf of the caller by creating or reusing another.

Linux BPF primitives operate naturally at packet and socket boundaries. `BPF_MAP_TYPE_SOCKMAP` and `BPF_MAP_TYPE_SOCKHASH` can redirect messages between sockets and attach socket-level verdict programs. The kernel documentation also exposes socket cookies to userspace rather than raw kernel socket pointers. These are useful identities for one socket lifetime, but a proxy handoff can create a new socket whose lifetime no longer identifies the original request. See the current Linux [sockmap and sockhash documentation](https://docs.kernel.org/next/bpf/map_sockmap.html).

The problem becomes more important as L7 work moves into kernel fast paths. [L7FP](https://arxiv.org/abs/2605.31084), published in May 2026, synthesizes an eBPF fast path for common service-mesh L7 policies and falls back to an existing userspace proxy for unsupported policy cases. Its reported performance gains show why such split execution is attractive. But a fast-path/slow-path system creates a new correctness requirement: a request should not silently acquire a different authorization identity simply because one case stayed in eBPF while another crossed the proxy boundary.

## Socket identity is not request identity

For simple TCP forwarding, carrying policy state on a socket can be enough. Once a proxy multiplexes or retries requests, that assumption becomes fragile.

Consider an HTTP/2 proxy connection from Envoy to one backend. The proxy may carry requests from several downstream clients over the same upstream connection. If the eBPF egress path attaches one source identity or one authorization generation to that upstream socket, which request does it describe? The answer can change from stream to stream without the socket changing at all.

Retries create the opposite transformation. One logical request may use several upstream connections over time. A request-level authorization should survive the retry, but a socket-local record alone will not.

Connection pooling makes both effects normal rather than exceptional. A correct system therefore needs at least three distinct identifiers:

- a **transport identity** for the concrete packet/socket lifetime;
- a **logical request or stream identity** that survives proxy scheduling, multiplexing, and retries;
- a **policy identity** containing the security principal, policy generation, and relevant authorization scope.

These identifiers can be related, but they should not be collapsed into one field just because a particular benchmark uses one request per TCP connection.

## The handoff should be a capability, not an informal side channel

A practical design does not need to copy an entire policy object into every request. It needs a compact statement that the next enforcement domain can validate.

Suppose the eBPF datapath decides that downstream principal `A` may send a class of requests toward service `B` under policy generation `g`. Before redirecting to the proxy, the datapath or its control plane can mint a short-lived **handoff capability** containing or referring to:

- the original security identity or principal class;
- the policy generation that authorized the handoff;
- a flow or request nonce that prevents reuse for unrelated traffic;
- the allowed destination or service scope;
- the L7 capability class that the proxy may refine, such as an HTTP method/path rule family;
- an expiry or revocation epoch;
- the proxy instance or trust domain allowed to consume the capability.

The proxy can refine that capability after parsing the request. For example, a packet-level decision may authorize "principal A may enter proxy P for service B," while the L7 filter proves that one HTTP request also matches method/path rule `r`. When Envoy creates or reuses the upstream connection, it produces a derivative witness that binds the L7 decision to the logical request rather than merely to the proxy socket.

The backend-side eBPF path does not need to parse HTTP again. It can validate the derivative witness, current generation, destination scope, and proxy authority before treating the proxy's packet as an authorized continuation of the original request.

This is closer to a capability system than to a trace annotation. A trace label helps explain what happened after the fact. A handoff capability participates in the authorization decision and should fail closed when it is stale, missing, or bound to the wrong request.

## Where current work is still weak

### Two enforcement points do not automatically preserve one principal

Cilium explicitly documents policy enforcement before and after Envoy. That is already stronger than treating the proxy as an invisible middlebox. But the two checks can legitimately use different identities, such as `world`, `ingress`, or workload identities.

The missing abstraction is an explicit **lineage relation** between them: the second decision should be able to prove which original principal and policy generation caused this request to exist, not merely that the packet came from a trusted proxy process.

The gap matters when the proxy is a confused deputy. If one permitted downstream request can cause the proxy to emit an upstream operation outside the original principal's scope, a backend rule that trusts only the proxy identity is weaker than the original policy. Conversely, if the backend repeats the original L3/L4 policy without enough context, it may reject legitimate transformed traffic.

A discriminating test should mix several downstream identities through one proxy, intentionally create overlapping backend destinations, and check whether an upstream request can ever be attributed to the wrong principal or policy generation.

### Per-socket policy state breaks under multiplexing

Socket cookies, socket-local storage, sockmap entries, and connection-tracking state are useful because they avoid repeating expensive work for every packet. They describe transport lifetimes well.

They do not by themselves describe HTTP/2 streams, gRPC requests, retries, or a connection pool shared by several callers. A policy cache that assumes one principal per upstream socket can therefore be correct for HTTP/1.1 without pooling and wrong for a production HTTP/2 workload.

The missing mechanism is a rule for when request-level authorization may be safely **coalesced** into connection-level state. The system should aggregate only if every active request on that connection is equivalent for the policy properties enforced downstream. Otherwise it needs request/stream-level lineage or must keep the decision at the proxy.

### Fast-path fallback needs policy-equivalent identity, not only policy-equivalent verdicts

L7FP shows a useful split: common policies can stay in an eBPF fast path, while unsupported cases fall back to a userspace proxy. A functional test may verify that both paths allow and deny the same requests.

That is necessary but incomplete. The two paths can return the same verdict in a simple test while attaching different provenance to the accepted request. The difference becomes visible later when a backend policy, audit system, rate limiter, or revocation mechanism depends on the original principal.

The missing evaluation is therefore not only verdict equivalence. It is **authorization-lineage equivalence** across fast and slow paths, including retries, pooling, policy updates, and proxy restarts.

## Promising directions with academic and production value

### 1. Generation-scoped proxy handoff capabilities

**Gap.** eBPF can authorize and redirect a flow to an L7 proxy, but the proxy's upstream socket does not inherently carry the original principal or policy generation.

**Mechanism.** Define a compact handoff capability minted at the eBPF-to-proxy boundary and consumed by an authorized proxy instance. The capability binds principal identity, policy generation, destination scope, nonce, expiry/revocation epoch, and allowed refinement class. After L7 parsing, the proxy derives a request-scoped witness that the backend-side datapath can validate without trusting the proxy's socket identity as the original principal.

The capability should be unforgeable by ordinary workloads. Implementation choices could include kernel-managed opaque handles, map-backed generation records with restricted lookup, or authenticated tokens passed through a local trusted channel. The first prototype should optimize for clear trust boundaries, not for cryptographic novelty.

**Delta.** Complete mediation proves that the request crosses enforcement points. This mechanism proves that the security subject and policy generation remain linked across those points. Multi-owner composition determines the winning rule; the handoff capability carries the resulting authority across a proxy transformation.

**Artifact.** A small eBPF redirect component, a Cilium/Envoy extension or filter, a map-backed capability table, and an egress validator. The artifact should expose a machine-readable explanation linking downstream identity, L7 rule, upstream request, and policy generation.

**Evaluation.** Mix identities through one node-local proxy while repeatedly updating policy, restarting Envoy, rotating proxy instances, retrying requests, and changing backend endpoints. Measure unauthorized upstream requests, stale-generation accepts, incorrect principal attribution, capability lookup cost, added request latency, and state size.

**Academic value.** The problem generalizes reference-monitor reasoning across a semantic transformation where one trusted component terminates and re-originates communication.

**Production value.** Service meshes and eBPF policy engines gain a concrete way to preserve least-privilege intent instead of treating the proxy as a universal identity after redirection.

**Failure condition.** If existing production meshes already expose an equivalent request-scoped, generation-aware handoff that survives proxy restart, pooling, and retry without identity confusion, the research contribution should shift to formalizing and measuring that existing mechanism rather than inventing another one.

### 2. Policy-safe multiplexing and request-to-socket coalescing

**Gap.** The natural eBPF cache key is often a packet, flow, or socket, while modern L7 proxies schedule several logical requests over shared upstream connections.

**Mechanism.** Treat coalescing as a proof obligation. The proxy or policy runtime groups requests on one upstream transport only when their downstream security contexts are equivalent for the policies that will be enforced after the proxy. If contexts differ, it preserves stream-level witnesses or separates transport pools by an explicit policy-equivalence class.

A compact equivalence class might include destination identity, current policy generation, downstream principal class, and any L7 attributes still relevant to backend enforcement. The class should be versioned. A policy update that changes equivalence must prevent old pooled state from being reused blindly.

**Delta.** This is not ordinary connection-pool partitioning for performance. The grouping rule is derived from downstream enforcement semantics and is evaluated for authorization lineage. It is also not a demand to expose every HTTP header to eBPF; the proxy can keep protocol details and export only the security equivalence class needed by the next enforcement domain.

**Artifact.** An Envoy connection-pool extension plus a small policy-equivalence API consumed by the eBPF control plane. A debug view should show when two requests were considered safe to share one upstream socket and which policy fields justified that choice.

**Evaluation.** Exercise HTTP/1.1 keepalive, HTTP/2 multiplexing, gRPC streams, retries, hedged requests, and backend connection reuse across multiple source identities. Compare naive one-identity-per-socket caching, no pooling, and policy-aware coalescing. Measure misattribution, false rejects, connection count, latency, CPU, and policy-update convergence.

**Academic value.** The work connects network policy to a classic systems question: when may many logical principals safely share one cached physical resource?

**Production value.** It can preserve proxy efficiency while preventing connection pooling from erasing distinctions the kernel-side policy still needs.

**Failure condition.** If backend enforcement never depends on the original principal after L7 authorization, or if production proxies already partition all relevant pools by an equivalent security context, the extra mechanism would add state without improving correctness.

### 3. A confused-deputy and fast/slow-path lineage benchmark

**Gap.** Existing network benchmarks usually emphasize throughput, latency, policy-update time, or allow/deny correctness. They rarely make the proxy intentionally act on behalf of several principals while checking whether authorization lineage survives every representation change.

**Mechanism.** Build a ground-truth harness that assigns each logical request a principal, policy generation, allowed L7 operation, destination scope, and expected backend identity. The harness then forces requests through direct eBPF fast paths, Envoy slow paths, mixed fallback, connection pooling, HTTP/2 multiplexing, retries, policy updates, and proxy restarts.

The primary correctness metric is an **authorization-lineage violation**: an accepted upstream request whose observed principal, generation, destination scope, or L7 authorization cannot be matched to the ground-truth request that caused it. Secondary metrics include stale accepts after revocation, false rejects, lineage-loss rate, throughput, CPU, and latency.

**Delta.** The previous complete-mediation benchmark looks for a packet that escaped all current enforcement. This benchmark assumes mediation happened and instead asks whether the mediated request carried the wrong identity across the proxy boundary.

**Artifact.** A reproducible Kubernetes testbed, Cilium/Envoy configuration, workload generator with mixed principals, fault injector, reference authorization log, and trace format that can compare kernel fast-path and proxy slow-path execution.

**Evaluation.** Include simple one-request-per-connection workloads where every system should pass, then progressively add pooling, multiplexing, retries, and policy churn. A proposed mechanism should earn its complexity by eliminating lineage violations in cases where ordinary socket or proxy identity does not.

**Academic value.** The benchmark makes a cross-layer security property measurable and provides a common target for kernel, CNI, service-mesh, and proxy research.

**Production value.** Operators can test whether an L7 acceleration or proxy integration preserves the policy subject through real failure and pooling behavior, not only whether a demo request receives HTTP 200 or 403.

**Failure condition.** If several independent implementations produce zero lineage violations under adversarial multiplexing, restart, update, and fallback workloads, the gap may already be solved well enough by existing practice.

## What would change this conclusion?

The argument depends on one assumption: there are production paths where the security identity used before an L7 proxy is not automatically and unambiguously bound to the logical request after the proxy re-originates traffic.

Strong evidence against a new handoff mechanism would be a production-grade implementation that already carries a request-scoped, tamper-resistant authorization identity and policy generation across downstream termination, upstream connection pooling, HTTP/2 or gRPC multiplexing, retries, proxy restart, policy update, and fast-path fallback. If an adversarial benchmark cannot produce a stale, confused, or unattributable accepted request, the right work is to document and standardize that mechanism.

A second boundary is enforcement placement. Some deployments intentionally make Envoy the final authority for all L7 decisions and require backend policies to trust only the proxy identity. In that model, preserving the original principal into a later eBPF enforcement point may not be useful. The proposed capability is valuable only when downstream enforcement, auditing, revocation, rate control, or provenance still depends on the original request identity.

The practical next step is therefore not to put more metadata on every packet. It is to build the benchmark first, make proxy identity confusion observable, and then test whether a compact generation-scoped handoff eliminates failures that current socket and proxy identities cannot.