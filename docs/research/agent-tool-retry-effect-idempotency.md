---
date: 2026-09-14
slug: agent-tool-retry-effect-idempotency
title: "When an AI Agent Retries a Tool Call, How Do You Know It Didn't Do It Twice?"
description: "Agent tool retries can duplicate payments, messages, deployments, or cloud resources when a response is lost after the external effect already happened."
tags:
  - Daily Report
  - AI Agent
  - Tool Use
  - Distributed Systems
  - Reliability
research_question: "How should an AI-agent runtime preserve one logical external effect across timeouts, cancellations, retries, process restarts, and tool-server changes without suppressing legitimate new work?"
source_cutoff: 2026-09-14
status: daily-report
---

# When an AI Agent Retries a Tool Call, How Do You Know It Didn't Do It Twice?

A tool-using agent asks an API to create a cloud task. The server creates it, but the response is lost. The agent sees a timeout.

What should it do next?

If it retries blindly, it may create a second task. If it refuses to retry, it may leave the user's request half-finished. If it asks the model to infer what probably happened from the timeout text, it has replaced a distributed-systems problem with a guess.

This failure mode matters for much more than cloud tasks. The same ambiguity can duplicate a payment, send the same message twice, create two tickets, launch two deployments, reserve two resources, or run a destructive operation again. The hard case is not a clean failure before the action starts. It is **ambiguous completion**: the caller no longer knows whether the external effect committed.

Modern agent protocols already expose pieces of the problem. MCP has an `idempotentHint`, but the specification explicitly says tool annotations are only hints and must be treated as untrusted unless they come from a trusted server. The 2026-07-28 MCP revision is also stateless at the protocol core. Multi Round-Trip Requests can reissue the original `tools/call` with a different JSON-RPC request ID, and a dropped HTTP request can cancel the protocol request even when downstream work is already too late to undo.

The missing property is not another retry counter. It is a durable identity for the **logical effect** the user intended, plus a way to reconcile an unknown outcome before another attempt is allowed to create a second effect.

<!-- more -->

## A request ID is not an effect ID

It helps to separate three identities that are often collapsed into one:

| Identity | Example | What it should mean |
| --- | --- | --- |
| Protocol request ID | JSON-RPC `id: 37` | Correlate one request with one response |
| Attempt ID | `attempt-3` | Identify one execution attempt for tracing and debugging |
| Effect ID | `launch-report-job-2026-09-14` or an opaque UUID | Identify one declared external mutation the workflow intends |

The distinction is visible in current MCP semantics. A normal `tools/call` carries a JSON-RPC request ID. If a tool returns `input_required`, the client retries the call with the additional answers and the specification requires the JSON-RPC ID to be different from the initial request. That is correct for request/response correlation, but it means request identity cannot also serve as the durable identity of the logical operation.

The new stateless MCP core makes this distinction even clearer. Requests no longer rely on a protocol-level session and can land on different server instances. Stateful tools are advised to return explicit handles and accept them on later calls. This is useful for carts, transactions, browser contexts, and other long-lived objects, but it still does not automatically tell a retrying client whether a one-shot side effect already committed.

MCP's `ToolAnnotations` include `idempotentHint`. When true, the intended meaning is that calling the tool repeatedly with the same arguments has no additional environmental effect. But the same schema says every annotation is a hint rather than a guaranteed description of behavior, and clients should not make tool-use decisions from untrusted annotations. Even a trusted `idempotentHint` describes a property of the tool operation. It does not prove that a particular provider still remembers a deduplication token after a restart, that the token remains within its retention window, or that two superficially identical JSON argument objects represent the same business action.

HTTP has the same separation. RFC 9110 defines some methods as idempotent because multiple identical requests have the same intended effect as one request, and this is what makes automatic retry after communication failure safe for those methods. It does not make every `POST` idempotent, and it does not give an application an operation identity for arbitrary external systems.

Production APIs therefore add an extra mechanism. AWS APIs use client tokens for selected mutating operations. For ECS `RunTask`, the same token and parameters suppress another mutation, while parameter changes produce a conflict and the token has a bounded lifetime. Stripe similarly scopes idempotent replay by an idempotency key, API, account or sandbox, and a retention window. These systems are useful evidence because they show that idempotency is not a generic boolean. It has **identity, parameter-binding, scope, retention, and replay semantics**.

An agent runtime that calls many unrelated tools crosses all of these contracts at once.

## The dangerous state is “unknown,” not “failed”

A retry policy often models a tool attempt as success or failure. External effects need at least one more state:

```text
not_started
in_progress
committed
failed_before_effect
unknown_after_dispatch
```

`unknown_after_dispatch` is the important one. It can arise when:

- the TCP connection drops after the provider committed the mutation but before the response arrives;
- the client times out while the server continues processing;
- cancellation races with the irreversible part of the operation;
- the agent process crashes after sending the request but before persisting the result;
- a tool server commits downstream state and then restarts before returning it;
- a gateway returns a 5xx after the backend already completed the work.

AWS's ECS documentation describes exactly this class of failure: a timeout or server problem can occur after resources were already mutated, leaving the caller unsure whether another retry would stack changes. The documentation recommends idempotency tokens because “retry on error” is otherwise not enough.

For an agent, the uncertainty is worse because the model may be asked to recover. A model can generate a new tool call, change an optional field, choose another endpoint, or paraphrase the operation. Those are useful capabilities for semantic repair, but they are dangerous if the first mutation may already exist. A second call that looks “almost the same” to a human may bypass downstream idempotency because its arguments or target differ.

The runtime therefore needs a policy stronger than “retry transient errors.” It needs to know whether it is retrying the **same effect**, repairing the **same effect**, or intentionally creating a **new effect**.

## Where current work is still weak

The first gap is **effect identity across agent layers**. Model tool-call IDs, JSON-RPC IDs, tracing span IDs, workflow-node IDs, provider idempotency tokens, and created resource IDs all exist for different reasons. The protocol and provider contracts examined here define identities at separate layers, but none of those specifications defines one identifier that spans the user-declared mutation, model or protocol retries, tool-server restarts, and downstream API execution. Whether a particular agent runtime already supplies that end-to-end identity is an implementation question to measure rather than an assumption of this report.

The second gap is **reconciliation after ambiguous completion**. Mature payment and cloud APIs often provide idempotency tokens or lookup mechanisms, but generic tools may wrap shell commands, browser actions, email systems, custom APIs, or multi-step workflows. After a timeout, a generic tool contract may not provide a standard way to ask: “Did logical effect E already happen, and if so, what result should I attach to it?”

The third gap is **idempotency-contract discovery**. MCP exposes `idempotentHint`, but it is deliberately advisory. Downstream APIs can have different token scopes, retention periods, parameter-equivalence rules, and failure behavior. A tool can also be internally idempotent while one of its downstream calls is not, or vice versa. A single boolean cannot describe the actual replay boundary.

The fourth gap is **evaluation coverage as an empirical question**. This report does not assume that representative agent benchmarks omit post-commit acknowledgment loss. A useful benchmark should first inventory existing suites, then test whether they inject failures precisely after an external commit but before acknowledgment and verify the resulting external state. If current suites already exercise that boundary across retries and restarts, this proposed gap disappears; if they do not, ordinary task-completion or tool-call scores can hide duplicate real effects.

This question is narrower than the earlier [parallel-agent effect-serializability report](https://eunomia.dev/research/parallel-agent-effect-serializability/). That report asks whether several workers' effects can compose into one valid result. Here there may be only one worker and one declared external mutation. The problem is whether several attempts accidentally materialize that mutation more than its declared multiplicity.

## Promising directions with academic and production value

### 1. Give every external mutation a durable effect identity

An agent runtime should allocate an `effect_id` and durably commit its effect record **before** dispatching the external mutation. Dispatch is allowed only after that write-ahead record can survive a process restart. The identifier then stays stable across retries. It should be generated by the runtime, not improvised by the language model. A workflow that intentionally creates several external effects gets several effect IDs; retry attempts for one effect keep that effect ID stable.

A minimal effect record might look like:

```text
effect_id = random stable UUID
workflow_id = parent user task
intent_hash = canonicalized mutation intent
tool = provider + tool name + contract version
authority = principal + approval/policy generation
target = normalized logical resource
attempts = [a1, a2, ...]
state = not_started | in_progress | committed | failed_before_effect | unknown_after_dispatch
provider_key = downstream idempotency token if supported
receipt = provider result/resource identity if known
retention_deadline = provider dedupe horizon if known
```

The ordering matters. The runtime first persists `not_started`, transitions to `in_progress` before or as it dispatches, and records the terminal outcome when it can prove one. If it restarts with a durable `in_progress` record, recovery must conservatively treat the outcome as unknown, keep the same `effect_id`, and enter reconciliation rather than minting a new effect. `failed_before_effect` is reserved for failures that prove the external mutation did not begin, so replay policy can distinguish those from ambiguous post-dispatch outcomes.

The important part is the split between `effect_id` and `attempts`. A timeout can create a new attempt without creating a new effect. If the model changes the operation enough to alter the normalized intent, the runtime should not silently reuse the old effect identity. It should either reject the change as a conflicting retry or create a new effect only through an explicit workflow transition.

Adapters can map the stable effect ID to downstream mechanisms. An AWS call can use it as a client token where allowed. A Stripe call can derive an idempotency key with the right account and API scope. A database adapter can use a unique constraint or transaction record. A filesystem adapter can use a staged object plus atomic rename. Tools with no deduplication mechanism can still be marked as requiring reconciliation or human confirmation after an ambiguous outcome.

Evaluation should inject response loss after a confirmed provider commit, restart the agent runtime, and retry the workflow. The primary metric is **retry-induced duplicate external effects per declared effect**, not retry success rate. Secondary metrics are false suppression of legitimate new operations, ledger storage cost, and added latency.

The academic contribution is an end-to-end naming model that separates effect identity from transport and execution attempts. The production integration point is the agent tool gateway or workflow runtime, where calls can be assigned identities before they reach arbitrary providers.

This direction should be rejected if existing workflow IDs, MCP request IDs, or provider-native tokens already remain stable through every relevant retry and restart while unambiguously distinguishing a repaired operation from a new one. If an extra effect identity changes no failure outcome, it is redundant metadata.

### 2. Make ambiguous outcomes enter reconciliation, not automatic replay

A runtime also needs a state machine for what happens when the effect ledger says `unknown`.

For idempotent providers, reconciliation can be simple: retry the same operation with the same provider token and parameter binding. For providers that expose a query-by-token or resource lookup, the runtime can ask for the existing result before deciding to issue another mutation. For tools with weaker support, the adapter can define an observation predicate, such as “ticket with this external reference exists,” “deployment with this release ID reached a terminal state,” or “message with this provider id was accepted.”

The key rule is:

> **An ambiguous mutation is not a normal failure. It is a pending fact that must be resolved against the external system or an explicit deduplication contract.**

A reconciliation record should capture more than success/failure. It should record whether the external state proves the effect committed, proves it did not commit, or remains ambiguous. If ambiguity persists and the action is non-idempotent or high-consequence, the safe outcome may be to stop automatic retries and surface a precise decision to the user.

This mechanism also needs retention semantics. An AWS client token or Stripe idempotency key does not live forever. If the deduplication window expires while an agent sleeps for days and then resumes, “same token” may no longer mean “same protected replay.” The runtime should therefore bind the known replay horizon to the effect record and change policy when it expires.

Evaluation should include network partitions, delayed responses, provider 5xx after commit, client and server crashes, cancellation races, long pauses beyond deduplication TTL, and parameter drift introduced by model repair. Measure time spent in unknown state, duplicate effects, unnecessary human escalations, and cases where reconciliation attaches the wrong external object to an effect ID.

The academic contribution is a recovery protocol for agent side effects under incomplete outcome knowledge. The production user is any long-running agent that operates cloud, communication, ticketing, billing, or deployment tools.

This direction loses if blind retry with existing provider APIs already produces zero duplicates and fewer unresolved tasks across the same fault matrix. The point is not to insert a generic “reconcile” step into every call; it is to use it only where outcome uncertainty can create a second real-world effect.

### 3. Build an adversarial benchmark for exactly-once declared effects, not exactly-once execution

“Exactly once” is a dangerous phrase in distributed systems because runtimes cannot generally guarantee that every component executes exactly once. A more useful target is explicit effect multiplicity: **a user-approved intent declares an allowed external outcome set `O(I)`; retries may use several execution attempts, but they must not add external effects beyond that set or duplicate an effect whose declared multiplicity is one**.

A one-to-many intent is therefore valid. Sending one approved message to five recipients can declare five effect slots; creating three resources can declare three. The benchmark judges retry-induced extras against the declared outcome, not against an assumption that every user intent maps to one external object.

A benchmark can make this measurable.

Each workload would declare:

```text
intent I
allowed external outcome set O(I)
declared multiplicity for each effect class
effect observation oracle
provider idempotency/reconciliation contract
failure injection points
```

Then inject faults at boundaries that ordinary unit tests miss:

- before dispatch;
- after provider receipt but before mutation;
- after mutation but before response;
- after response reaches the tool server but before it reaches the agent;
- during cancellation;
- after model/tool timeout but before backend completion;
- after agent-process crash and restart;
- after load-balancing to a different stateless tool-server instance;
- after provider deduplication-token expiry;
- while a model changes one parameter during a repair attempt.

The benchmark should contain both provider-backed adapters with strong idempotency and deliberately weak tools such as “append a line,” “send a message,” or “create a resource with no unique client token.” Ground truth must come from the external state, not from whether the tool returned `success`.

Report at least four metrics separately: retry-induced effects beyond `O(I)` or its declared multiplicities, missing intended effects, false deduplication of legitimate new actions, and unresolved ambiguous outcomes. Latency and token cost are secondary. A system that gets a good task-completion score by occasionally sending a sixth copy of an intended five-recipient message should fail the benchmark.

The academic value is a fault model and correctness metric for agent tool reliability. The production value is regression testing for retry middleware, MCP gateways, workflow engines, and high-consequence tool adapters.

This benchmark is unnecessary if existing agent reliability suites already inject post-commit acknowledgment loss and can prove declared-effect correctness across restarts, cancellation, load balancing, and deduplication-window expiry. The benchmark should first try to demonstrate that current suites already cover these cases rather than assume a gap.

## A practical rule for agent runtimes today

Before automatically retrying a mutating tool call, separate five questions:

1. Is the tool operation actually idempotent, or is that only an annotation or assumption?
2. Does the downstream provider accept a stable idempotency/client token, and what parameters, account, region, or API scope does it bind?
3. How long does the provider remember that identity?
4. If the caller loses the response after dispatch, can it query or reconcile the existing effect?
5. If none of those are true, is duplicating the action safer than leaving the outcome unresolved?

Read-only calls can stay cheap. Truly idempotent mutations can retry aggressively under their documented contract. High-consequence non-idempotent actions should carry durable effect identity and enter reconciliation after an ambiguous failure instead of asking the model to guess.

The current MCP design is compatible with this approach but does not supply it automatically. Stateless requests and different JSON-RPC IDs are transport behavior. Tool annotations are useful hints. Explicit tool handles are useful application state. None of them alone is the end-to-end identity of an externally visible mutation across retries.

## What would change this conclusion?

The argument would weaken if agent protocols standardized a trusted, end-to-end mutation identity whose semantics survived transport retries, tool-server restarts, model repair, and downstream provider calls, and if real implementations enforced parameter binding and replay retention well enough that duplicate external effects disappeared under fault injection.

It would also weaken if production agent tools proved to be overwhelmingly read-only or naturally idempotent at the business level. In that world, a durable effect ledger would impose complexity on a rare corner case rather than protect a common failure boundary.

Finally, the proposed mechanism should be rejected if a simpler adapter policy works just as well. If “use the provider's native idempotency token when present; otherwise do not automatically retry mutations” reaches the same completion rate and zero retry-induced duplicates across realistic workloads, a general effect ledger is unnecessary.

The useful distinction is therefore simple: **a retry is a new execution attempt, not automatically a new user intent. Agent runtimes need enough durable identity and reconciliation to preserve that distinction when the network stops telling them what happened.**

## References

- Model Context Protocol, [Tools — 2026-07-28 specification](https://modelcontextprotocol.io/specification/2026-07-28/server/tools), accessed 2026-09-14.
- Model Context Protocol, [Schema Reference — `ToolAnnotations`](https://modelcontextprotocol.io/specification/2026-07-28/schema), accessed 2026-09-14.
- Model Context Protocol, [SEP-2575: Make MCP Stateless](https://modelcontextprotocol.io/seps/2575-stateless-mcp), accessed 2026-09-14.
- RFC 9110, [HTTP Semantics, Section 9.2.2: Idempotent Methods](https://www.rfc-editor.org/rfc/rfc9110.html#section-9.2.2), accessed 2026-09-14.
- Amazon ECS, [Ensuring idempotency](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/ECS_Idempotency.html), accessed 2026-09-14.
- Amazon EBS, [Ensure idempotency in StartSnapshot API requests](https://docs.aws.amazon.com/ebs/latest/userguide/ebs-direct-api-idempotency.html), accessed 2026-09-14.
- Stripe, [API v2 overview — idempotent requests](https://docs.stripe.com/api-v2-overview), accessed 2026-09-14.
