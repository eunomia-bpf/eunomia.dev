# Why should the list of available agent skills in GenAI telemetry be a dedicated span attribute instead of a separate span or a span event, and why does the span duration model rule out the span form?

It belongs on the `invoke_agent` span as a structured attribute, not as its own span or as a span event. A span is a duration-bounded unit of work - it carries a start timestamp and an end timestamp and is meant to cover an operation, retries included; a span event is a timestamped occurrence, something that happened at a moment within that span. The offered-skills list is neither: it is a point-in-time description of what was available when the call started, with no duration of its own and no occurrence to timestamp, so it is reference data and attaches as an attribute - the same shape `gen_ai.tool.definitions` already uses, which the open `gen_ai.skill.definitions` proposal mirrors.

## The mechanism

The trace data model is the reason. A Span encapsulates a name, a context, a parent, a kind, a start timestamp, an end timestamp, attributes, a list of links, a list of timestamped events, and a status. The GenAI spans convention adds the semantic intent: GenAI spans SHOULD cover the duration of the operation, starting when it is initiated and ending when the response is fully received or the operation is terminated, with automatic retries folded into the same span. A span therefore asserts a period of work. The offered-skills list is a snapshot of the agent's capability surface at the start of the call; it has no end, no duration, and no operation to measure, so wrapping it in its own span would fabricate a start and end that do not exist and would add a static node to every invocation for data that is input, not a sub-operation.

The event form fails for the opposite reason. Events are timestamped occurrences, recorded in the order they happened; they model "something happened at t" - a skill loaded, a tool executed, a streaming chunk arriving. The available-skills list is not a thing that happened; it is a description of state. Choosing the event form would force an arbitrary timestamp and assert an occurrence that did not occur. It is data, and data on a span is an attribute.

The attribute is where large structured payloads already live in these conventions. `gen_ai.tool.definitions` is an any-typed Development attribute that holds a structured list of tool definitions, recorded in structured form when the backend supports it and as a JSON string where it does not. A skill has no parameter schema, so the tool-definitions schema does not fit; the dedicated complement is `gen_ai.skill.definitions`, recorded on the `invoke_agent` span so a trace UI can show exactly which skills the agent could choose from and let a platform display them as the agent saw them and search traces by them.

## Verification and debugging path

1. Open the OpenTelemetry trace API specification, Span section: a Span encapsulates a start and an end timestamp plus attributes and a list of timestamped events, and the GenAI spans convention states spans SHOULD cover the duration of the operation. That reading is why a span is a bounded operation, not a home for static reference data.
2. Read the Add Events section: an event has a timestamp for the moment it was recorded and events SHOULD preserve their recorded order - the data model for a timestamped occurrence, which a reference list is not.
3. Read the GenAI attributes registry entry for `gen_ai.tool.definitions`: it is an any-typed Development attribute holding a structured list - the precedent for carrying a list of definitions as a span attribute.
4. Read the open `gen_ai.skill.definitions` proposal: it records the available-skills list on the internal `invoke_agent` span as a dedicated structured attribute that complements the tool-definitions attribute, and notes that the tool schema, built around function-call parameters, does not describe a skill.

## The limitation

- The definitions attribute is open and in Development, not stable, so consumers cannot rely on it being present across SDKs yet; the per-skill attributes - `gen_ai.skill.name`, `gen_ai.skill.description`, `gen_ai.skill.resource.name`, `gen_ai.skill.source.uri` - already exist as attributes, which is the shape the definitions attribute will join.
- A dedicated attribute is structured data attached to a single span; it does not by itself create a searchable index. Search-by-skill depends on the backend indexing the attribute value, not on the trace data model.
- The list can be large; the conventions' content-capture opt-in decides whether it is attached at all, and recording it as a JSON string is the fallback where structured attribute values are not supported.
- An event is only right when the interesting fact is "which skill was loaded or executed and when" - the execution side, already covered by the tool and skill operation spans, which is a different signal from the offered list.

## References

- [OpenTelemetry trace API specification](https://opentelemetry.io/docs/specs/otel/trace/api/) - the Span data model (start and end timestamps, attributes, a list of timestamped events) and the Add Events rules that make an event a timestamped, ordered occurrence.
- [semantic-conventions-genai GenAI spans](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-spans.md) - GenAI spans SHOULD cover the duration of the operation, the `invoke_agent` operation, and the pattern for recording structured content on span attributes.
- [semantic-conventions-genai pull request 557](https://github.com/open-telemetry/semantic-conventions-genai/pull/557) - the open `gen_ai.skill.definitions` attribute on internal `invoke_agent` spans, recording which skills were available when the invocation started, complementing the tool-definitions attribute.
- [semantic-conventions-genai GenAI attributes registry](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/registry/attributes/gen-ai.md) - `gen_ai.tool.definitions` as an any-typed attribute holding a structured list, plus the existing `gen_ai.skill.*` attribute family.

## Community discussion today

Two opt-in archive channels (the two CNCF OpenTelemetry instrumentation channels) produced thirteen messages this run. The thread this page answers: a conventions reviewer asked why the available-skills list was being modeled as a span when there is "no logical duration - it's a point in time thing", and in that case "more of a span event"; the reply reframed it - the goal is not a parsed version of the system prompt but a dedicated structured attribute that complements the existing tool-definitions attribute, so a platform can display the skills as the agent saw them and search traces by them. The other threads had already been answered by earlier posts - the model-serving-signals catalog question and the `logger.enabled` opt-in/opt-out question - and the Node.js GenAI repository question had no decisive technical boundary this run, so it was not selected.

Channel coverage for this run: the two opt-in archives supplied thirteen messages. The visible-browser-only sources - the eunomia-bpf and sched-ext Discord servers, the bpf mailing list, and r/eBPF - could not be reviewed this run because no visible-browser session was available, so they are marked not-covered, not quiet.
