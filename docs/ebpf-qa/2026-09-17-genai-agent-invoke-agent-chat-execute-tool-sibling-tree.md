# Should GenAI agent spans nest each tool execution under the model call that requested it, or keep model calls and tool executions as siblings under one agent span?

**Short answer:** keep them as siblings. For an in-process agent, one `invoke_agent` span per customer turn with the model calls (`chat`) and tool executions (`execute_tool`) as **direct children** of that `invoke_agent` is the intended shape, and the tool executions are correlated to the model call that requested them by the `gen_ai.tool.call.id` attribute, not by nesting. The GenAI agent-span conventions say that "the tool or task spans produced from the plan are typically sibling operations under the same `invoke_agent` span." Nesting each `execute_tool` under the `chat` that requested it is a causal-tree idea (it is only worth doing if you can carry a W3C `traceparent` through that hop); it is not required, and a sibling-plus-`gen_ai.tool.call.id` tree is not a mistake.

## The boundary the conventions actually draw

The GenAI semantic conventions define the agent-span vocabulary as `invoke_agent`, `invoke_workflow`, `plan`, `chat`, and `execute_tool`, each with a `gen_ai.operation.name`. The shape decision rests on one sentence in the *Plan span* section:

> A plan span represents the decision phase where an agent formulates a strategy before executing it. The LLM call that generates the plan SHOULD be a child of the plan span, and **the tool or task spans produced from the plan are typically sibling operations under the same `invoke_agent` span.**

That is the decisive boundary. The model call that *requests* a tool and the `execute_tool` span that *runs* it are siblings under one `invoke_agent`, not a parent/child pair. What joins them is the `gen_ai.tool.call.id` attribute (a `Recommended` attribute, "the tool call identifier" used to correlate a model request with the tool that fulfilled it), so the association is carried in the data, not in the tree structure.

## Why the span kind decides the tree you can build

The `invoke_agent` span has two variants and they carry the shape:

- **`invoke_agent` INTERNAL** — "GenAI agent invocation within the same process" (e.g. an in-process LangChain/CrewAI-style agent). Span kind `INTERNAL`. This is the case where you own the loop and emit `chat`/`execute_tool` yourself, so the sibling tree under a single `invoke_agent` is exactly what you control.
- **`invoke_agent` CLIENT** — "GenAI agent invocation over a remote service," with the conventions naming OpenAI Assistants API and AWS Bedrock Agents as the examples. Span kind `CLIENT`.

The CLIENT variant is the limit of the tree: if the agent loop lives in a managed harness on the vendor's machines, the inner `chat`/`execute_tool` tree only exists *if the harness exports it*. What generic HTTP-client instrumentation can still see are the outbound crossings that leave the box as ordinary HTTP calls. So the "one `invoke_agent` per turn with `chat`/`execute_tool` as children" answer applies cleanly to the in-process case; for a hosted harness the inner tree is whatever the vendor chooses to emit.

## How the correlation actually works, and what to verify

Because the association is an attribute and not a parent/child link, the verification path is simple:

1. **One `invoke_agent` per turn, keyed by the conversation.** For an in-process support agent, one `invoke_agent` per customer turn is right *when that turn is the invocation*. Keep the ticket tied together across turns with `gen_ai.conversation.id`; do not stretch a single `invoke_agent` across a whole multi-turn session unless there is genuinely one long-running run.
2. **`chat` and `execute_tool` as direct children of that `invoke_agent`.** They are siblings. Set `gen_ai.tool.call.id` on both the model call that requested the tool and the `execute_tool` span it produced, so a reader can join them even though the tree does not show the causal edge.
3. **Nest only if you can carry context through.** Nesting `execute_tool` under the requesting `chat` is the causal-tree idea. It pays off only when you propagate a W3C `traceparent` through that hop so the nesting is meaningful; if you cannot, sibling-plus-`gen_ai.tool.call.id` is the correct, non-missing shape.
4. **Check the plan span when planning is real.** If the agent does genuine task decomposition, add a `plan` span (kind `INTERNAL`); its LLM call is its child and the tool/task spans it produces stay siblings under the `invoke_agent`. Omit `plan` when you cannot reliably distinguish planning from ordinary inference.

## The limitation that decides it

The intended tree is "one `invoke_agent` per invocation, `chat` and `execute_tool` as siblings, joined by `gen_ai.tool.call.id`." Nesting each `execute_tool` under the `chat` that requested it is an optional causal refinement, not the convention, and it is only worth it when W3C context propagation carries the nesting. And the whole tree only holds while the agent loop is *yours* to instrument: the moment the loop moves into a managed harness, the `chat`/`execute_tool` children are the vendor's to export, and all generic HTTP instrumentation can reconstruct are the outbound crossings. The sibling shape plus the correlation attribute is not a miss — it is the documented default.

## References

- [OpenTelemetry GenAI semantic conventions: agent and framework spans (`invoke_agent` INTERNAL = "within the same process"; plan spans say "the tool or task spans produced from the plan are typically sibling operations under the same `invoke_agent` span")](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-agent-spans.md)
- [OpenTelemetry GenAI semantic conventions: GenAI spans (`chat` span kind `CLIENT` / `INTERNAL`; `execute_tool` span kind `INTERNAL`, span name `execute_tool {gen_ai.tool.name}`)](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-spans.md)
- [OpenTelemetry GenAI semantic conventions (published registry of `gen_ai.*` attributes, including `gen_ai.tool.call.id`)](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [OpenTelemetry W3C Trace Context (propagating `traceparent` through a causal hop)](https://www.w3.org/TR/trace-context/)
- [OpenAI Agents API observability (public beta: "Tracing is enabled by default for new sessions. The public beta API does not expose tracing configuration or external trace exporters.")](https://developers.openai.com/api/docs/guides/agents-api/observability)

## Community discussion today

The monitored window covered two allowlisted Slack archives (both technical) and two browser-only chat workspaces plus public mailing-list/forum surfaces that were not reviewed this run; the technical content came from the two archive-backed workspaces.

**GenAI agent span shape (the question above).** A practitioner building a small open-source support agent in TypeScript wrote its tracing layer against the GenAI conventions before the agent loop: a turn, model calls exported as `chat`, and tool executions exported as `execute_tool`, with model calls and tool executions as siblings under the turn and correlated by `gen_ai.tool.call.id`. The open question was whether that is the intended shape or whether each model round should nest the tool executions it requested. A maintainer reading the current agent-spans doc confirmed the tree is intended: for an in-process agent `invoke_agent` is `INTERNAL`, one `invoke_agent` per customer turn when the turn *is* the invocation, `chat`/`execute_tool` as direct children, and the nesting-under-the-requesting-`chat` idea (a causal tree) only pays off when a W3C `traceparent` is carried through. The practical guidance: keep siblings + `gen_ai.tool.call.id`; add an `invoke_agent` per turn, not per session; carry the ticket across turns with `gen_ai.conversation.id`.

**Managed harnesses and observability lock-in.** A separate thread worried about the OpenAI Agents API as a hosted harness: because the loop runs on the vendor's machines, the inner `chat`/`execute_tool` tree exists only if the vendor exports it, and the public beta does not expose tracing configuration or external trace exporters — so the general pattern of managed cloud services holding o11y data in-house. The boundary is exactly the CLIENT `invoke_agent` case: generic HTTP-client instrumentation can still capture the outbound crossings that leave the box, but it cannot reconstruct the full inner trace the way an in-process SDK that emits spans can. One practitioner proposed metering the model/tool calls as an ordinary HTTP receipt header (a vendor-prefixed mapping on an otherwise ordinary span) so existing HTTP-client instrumentation captures it without inventing a new `gen_ai.*` attribute.

**OBI config v1-to-v2 migration.** A user reported the `migrate` command failing with "fields are outside the supported v1-to-v2 migration contract," asking whether partial migration should happen. The maintainer said the all-or-nothing behavior is intentional (a silently-dropped field can produce a valid-looking but materially different config), but proposed an explicit `--allow-partial` / `--best-effort` mode that migrates what it can and reports the omitted fields rather than making partial migration the default. The related Helm-chart gap (a v1 field still injected into the v2 config schema by the chart's `_helpers.tpl`) was directed to be filed as a separate issue with chart/OBI versions, the relevant values, the rendered config, and the validation error.

**Node.js sentinel performance (continuing from the prior day).** The cost-attribution work on the injected Node agent's `async_hooks` sentinel advanced to a reviewable draft PR, and a user reported that after excluding a service from tracing they had to roll out a pod restart to see latency drop — suggesting the tracing process can hang inside the pod even after exclusion. A maintainer noted the per-callback refresh is the consumer to gate (and should be allowed to turn off when only populating the trace context map), with the direction to separate per-callback refresh from fd-pair correlation. The unresolved boundary remains how to gate the sentinel per consumer without silently breaking external trace/profile correlation that reads the pinned context map.
