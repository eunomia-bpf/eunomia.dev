---
date: 2026-05-21
description: ACRFence explains semantic rollback attacks in AI agent checkpoint/restore workflows and shows how intent-aware fencing prevents duplicate irreversible actions and revived authority.
---

# ACRFence: Preventing Semantic Rollback Attacks in Agent Checkpoint-Restore

AI agent frameworks are bringing checkpoint/restore, time travel, and rewind into everyday developer workflows. If an agent makes a mistake, it can go back to a checkpoint. If a user wants to explore another path, the agent can branch from an earlier state. This is useful for debugging and human-in-the-loop control, but it becomes dangerous once the agent has already called external tools.

Traditional checkpoint/restore rolls back local state. It cannot undo side effects that have already happened in the external world. For ordinary programs, the usual answer is idempotency: retry the external call with the same request id, and the server returns the previous result instead of executing the action again. But an LLM agent is not an ordinary deterministic program. After restore, it may synthesize a semantically equivalent tool call with slightly different fields, such as a new UUID, timestamp, nonce, or reference number. The server cannot see that this is a retry of the same intent. It only sees a new valid request.

This post is based on our arXiv paper [**ACRFence: Preventing Semantic Rollback Attacks in Agent Checkpoint-Restore**](https://arxiv.org/abs/2603.20625). We introduce **semantic rollback attacks**: attacks that exploit the gap between rolled-back agent state and non-rolled-back external state to trigger duplicate irreversible actions or revive consumed authority.

<!-- more -->

## A Simple Transfer Example

Suppose a user asks an agent to transfer $500 to Bob. The agent calls a bank API, generates a unique reference id `a1b2c3d4`, and the transfer succeeds. The agent then calls Bob's MCP service to confirm the receipt. Bob's service returns a malformed response that crashes the agent. The framework restores the agent to a checkpoint before the transfer.

After restore, the agent again executes the intent "transfer $500 to Bob." This time, however, it generates a different reference id, `f9a8b7c6`. The bank's duplicate detection logic only sees two different references, so it accepts the second transfer. Bob receives $1000, while the agent's local view remains "I transferred once."

![Action Replay attack flow](https://eunomia.dev/_content-assets/docs/blog/posts/imgs/acrfence/fig-sequence.png)
*Figure 1: Action Replay. A malicious MCP service triggers a crash after a successful transfer. After restore, the agent reissues the transfer with a new reference id, so the bank treats it as a new transaction.*

The key point is not that the transfer API lacks idempotency. The problem is that the precondition for idempotency is broken. Systems such as Stripe and AWS ECS rely on the caller retrying with the same idempotency key or the same critical parameters. An LLM agent rethinks after restore and may produce a different token sequence. Even at temperature 0, byte-identical tool calls are not guaranteed. As a result, traditional server-side deduplication cannot recognize a "semantically same" retry.

## Root Cause: Local Rollback, External Progress

Checkpoint/restore systems can save local process state, conversation context, variables, file descriptors, and related runtime state. They cannot automatically undo committed external effects. Transfers, emails, cloud resource creation, data deletion, and one-time token consumption are all irreversible side effects from the framework's point of view.

In the agent setting, three facts combine badly:

- **The agent state is rolled back.** The agent returns to an old checkpoint and no longer remembers that the transfer succeeded.
- **The external state is not rolled back.** The bank ledger, approval system, or cloud control plane still records the previous successful action.
- **The post-restore tool call may differ.** The LLM may regenerate UUIDs, nonces, timestamps, or even change the target object under user guidance.

![Divergence between agent state and external state](https://eunomia.dev/_content-assets/docs/blog/posts/imgs/acrfence/fig-divergence.png)
*Figure 2: Restore only affects local agent state. External state keeps moving forward. This divergence is the core of semantic rollback attacks.*

This resembles the classic output commit problem in distributed systems: once output has been committed to the outside world, rolling back the local process alone cannot take the whole system back in time. The new twist is that an LLM agent may synthesize a different request after restore, blurring the boundary between "retry" and "new request."

## Attack 1: Action Replay

**Action Replay** targets irreversible tool calls that have already succeeded. The attacker does not need to control the bank or compromise the agent. It is enough to control a later service in the agent's tool chain, such as Bob's invoice-confirmation MCP service or a seemingly harmless callback endpoint.

The attack path is direct:

1. The agent executes an irreversible action after a checkpoint, such as a transfer or cloud resource creation.
2. The external service returns success, and the side effect is committed.
3. An attacker-controlled later tool returns a malformed response, triggering crash or restore.
4. The agent returns to the old checkpoint and repeats the same task.
5. The LLM generates a fresh request id, so the target service cannot recognize the repeated intent and commits again.

![Normal execution compared with attack execution](https://eunomia.dev/_content-assets/docs/blog/posts/imgs/acrfence/fig-comparison.png)
*Figure 3: Normal execution transfers once. In the attack path, crash-induced restore causes the same semantic action to execute twice.*

In our experiments, we used Claude Code CLI backed by Qwen3-32B. External services were simulated as MCP tool servers: a bank service with UUID-based duplicate detection and a malicious payee service that crashes the agent after a successful transfer. Across 10 checkpoint/restore trials, all 10 produced duplicate commits. A no-checkpoint baseline produced none. This confirms that the vulnerability comes from the interaction between restore and external side effects, not from ordinary model randomness alone.

## Attack 2: Authority Resurrection

The second attack class is **Authority Resurrection**, which targets one-time authorization tokens or short-lived credentials.

Consider an enterprise data deletion workflow. The agent first obtains manager approval, and the approval service returns a one-time token. The agent uses that token to delete Alice's data, and the server marks the token as consumed. A user or malicious insider then rewinds the agent to the checkpoint immediately after approval was granted. In the agent's local state, the token appears again. In the external approval system, the token should already be consumed.

If the target service validates tokens statelessly, for example by checking only a signature and expiration time, the agent may reuse the same token on another target, such as Bob's data. The audit log may show that the manager approved deletion for Alice, while Bob's data was also deleted. The discrepancy is visible only by correlating approval and execution logs.

Our experiment simulated two approval services:

| Validation mode | Result |
| --- | --- |
| Stateless validation, checking only token signature | 2/2 reuse attempts succeeded |
| Stateful validation, recording token consumption server-side | All reuse attempts were rejected |

This shows that checkpoint/restore can do more than duplicate financial side effects. It can also break authorization semantics by reviving authority that should have been consumed.

## Why This Is Not One Framework's Bug

The paper surveys reports across multiple frameworks and communities. The concrete symptoms differ, but they point to the same boundary: restore, retry, approval, preemption, and human-in-the-loop flows can cause tool calls to execute more than once, while frameworks generally do not enforce exactly-once semantics at the tool boundary.

| Framework or system | Observed issue type |
| --- | --- |
| LangGraph | Tool nodes may re-execute after resume or interrupt |
| CrewAI | Workflows run twice, causing repeated emails or actions |
| Google ADK | Rewind documentation warns that external side effects are not undone |
| AutoGen / OpenAI Agents | Graph nodes or function calls are triggered repeatedly |
| Claude Code / Cursor | Duplicate tool behavior around approval, checkpoint, or undo flows |
| OpenHands / Vercel AI / LiveKit / n8n | Duplicate messages, repeated tool calls, doubled token cost, or repeated charges |

These cases do not mean every framework has the same bug. They show that "restoring agent state to the past" while "the external world remains in the present" is a systemic issue. Relying on developers to make tools idempotent is not enough, because the post-restore agent request may not be the same request.

## ACRFence: Replay-or-Fork at the Tool Boundary

ACRFence does not try to make every LLM agent deterministic. Instead, it records irreversible effects at the tool boundary and enforces **replay-or-fork** semantics after restore.

ACRFence can be deployed as an MCP proxy or a similar tool-call proxy between the agent and external services. For each irreversible tool call, ACRFence records an effect log that includes:

- thread and branch identifiers, to distinguish execution branches in the same session;
- tool name and arguments;
- return value or error;
- runtime context, such as process, network, and file-access context, which can be enriched by eBPF-based system-level monitors such as [AgentSight](https://github.com/eunomia-bpf/agentsight/);
- consumed credentials or authorization objects, when applicable.

When the agent restores from a checkpoint and issues another tool call, ACRFence does not immediately forward it. It first compares the new call with the historical effect log:

- **Semantically equivalent: replay.** If the new call only changes non-intent fields such as request id or timestamp, while recipient, amount, resource target, and other intent fields are the same, ACRFence returns the previously recorded response without re-executing the external operation.
- **Semantically divergent: fork.** If the new call changes intent-critical fields, such as a different recipient or a different customer deletion target, ACRFence blocks the call, shows the prior effect log, and requires an explicit fork.
- **Credential reuse: reject or inform.** If the call tries to reuse a consumed token, ACRFence informs the agent before the request reaches the target service.

We use an analyzer LLM for semantic comparison instead of requiring every tool to provide a hand-written schema and idempotency rule. For example, two `transfer` calls with different UUIDs but the same amount and recipient should be treated as the same intent. Two `delete_customer_data` calls with the same approval token but different customer ids should be treated as dangerous divergence. The analyzer runs only on the restore path, not on every normal tool call.

ACRFence aims to provide two guarantees:

1. **Replay safety:** semantically equivalent irreversible calls after restore do not execute again; ACRFence returns the cached result.
2. **Divergence detection:** semantically different calls after restore must explicitly fork; they cannot silently inherit external effects or authority from an old branch.

## How This Differs from Idempotency and Durable Execution

Idempotency is still important, but it solves the problem of "the same request is retried." ACRFence works one level higher, at agent intent: request fields may change while intent stays the same, or the fields may look valid while intent has drifted to a new target.

Durable execution systems usually require deterministic orchestrator logic, with nondeterministic values recorded as side effects and replayed on recovery. That works well for traditional workflows. LLM agents, however, generate their next action from context. Rather than assuming post-restore calls will be byte-identical, ACRFence treats divergence as expected and makes replay versus fork explicit at the tool boundary.

In this division of labor, checkpoint/restore lets the agent return to an earlier state. ACRFence ensures that reconnecting that old state to the external world does not duplicate irreversible side effects or revive consumed authority.

## Limitations and Next Steps

The work validates the two attack classes, while ACRFence itself remains a design that still needs a full implementation and system evaluation. Several challenges remain:

- The analyzer LLM may misclassify calls, so false replay and false fork risks need careful evaluation.
- An adaptive attacker who knows the comparison logic may craft ambiguous parameters to evade semantic detection.
- The boundary between "intent fields" and "non-intent fields" is not always obvious for every tool.
- The current experiments cover one model and one framework; more agent frameworks, models, and real tool ecosystems should be evaluated.

The core conclusion is clear: once agent frameworks introduce checkpoint, rewind, time travel, and branch exploration, external tool calls cannot rely only on traditional idempotency keys. The restore path is a new security boundary.

## Conclusion

Checkpoint/restore makes AI agents easier to debug, recover, and steer across multiple execution paths. But once agents can call external tools, local rollback and external non-rollback create a semantic gap. Action Replay can turn one payment, one resource creation, or one email into many. Authority Resurrection can make consumed authorization reappear in local agent state.

ACRFence records irreversible effects at the tool boundary and enforces replay-or-fork after restore: same intent replays the result without re-execution, different intent must explicitly fork, and consumed credentials cannot be silently reused. As more agent frameworks support checkpoint and time travel, this kind of tool-boundary semantics will become part of the reliability and security foundation.

## References

- [ACRFence: Preventing Semantic Rollback Attacks in Agent Checkpoint-Restore](https://arxiv.org/abs/2603.20625)
- [CRIU: Checkpoint/Restore In Userspace](https://criu.org/)
- [LangGraph Persistence and Time Travel](https://docs.langchain.com/oss/python/langgraph/persistence)
- [LangGraph Durable Execution](https://docs.langchain.com/oss/python/langgraph/durable-execution)
- [Model Context Protocol Specification](https://modelcontextprotocol.io/specification/2025-03-26)
- [Stripe API: Idempotent Requests](https://docs.stripe.com/api/idempotent_requests)
- [Temporal Side Effects](https://docs.temporal.io/develop/go/side-effects)
- [Google ADK Session Rewind](https://google.github.io/adk-docs/sessions/session/rewind/)
- [Vault issue #28378: single-use token reappears after snapshot restore](https://github.com/hashicorp/vault/issues/28378)
