## 2026-10-11 eBPF Q&A run-report (eunomia-community-radar)

## Selected candidate

- Slug: `genai-available-skills-model-attribute-vs-span-vs-event`
- Question: why the list of agent skills available in a GenAI agent
  invocation should be recorded as a dedicated span attribute on the
  `invoke_agent` span rather than as a separate span or a span event,
  and why the span duration model rules out the span form.
- Source: the 10-11 snapshot carries thirteen messages, all in the two
  CNCF OpenTelemetry instrumentation channels
  (`otel-ebpf-instrumentation` / `otel-genai-instrumentation`). Three
  distinct threads: (1) the one this page answers - a conventions
  reviewer asked why the available-skills list was being modeled as a
  span when there is "no logical duration - it's a point in time
  thing, no? in which case it's more of a span event"; the reply
  reframed the goal - not a parsed version of the system prompt, but a
  dedicated structured attribute that complements the existing
  `gen_ai.tool.definitions` attribute, so an observability platform can
  present the available skills as the agent saw them and search traces
  by them. The boundary is the trace data model: a span is a
  duration-bounded operation (start + end timestamps, covering
  retries) and an event is a timestamped ordered occurrence; the
  offered list is neither, so it is reference data and belongs as an
  attribute; (2) the `model-serving-signals` catalog question was
  already published as the 10-09 entry, so it is cited and skipped;
  (3) the `logger.enabled` opt-in/opt-out question was already
  published as the 10-10 entry, so it is cited and skipped; and the
  Node.js GenAI instrumentation repository question (line 2-9) was not
  selected because it has no decisive technical boundary this run
  (the answer is that the instrumentation lands in `otel-js-contrib`
  with auto-generated conventions in `otel-js`, with only the
  `semantic-conventions-genai` issue 247 as guidance). The selected
  theme is non-duplicative: 10-05 covered the offered-vs-executed
  blind spot ("why can't a trace show the offered list at all"), and
  this 10-11 entry covers the telemetry-modeling question that a
  reviewer raised on that same attribute ("should the offered list be
  a span / event / attribute and why does span-duration semantics rule
  out the span form") - a different boundary grounded in the OpenTelemetry
  trace API spec.

## Verification against public primary sources

- OpenTelemetry trace API specification
  (`opentelemetry.io/docs/specs/otel/trace/api/`): the Span data
  model (a start timestamp, an end timestamp, attributes, and a list
  of timestamped events) and the Add Events section (an event has a
  timestamp for the moment it was recorded, and events SHOULD
  preserve their recorded order) - the data model for a timestamped
  ordered occurrence, which a reference list is not.
- OpenTelemetry semantic-conventions-genai GenAI spans
  (`github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-spans.md`):
  GenAI spans SHOULD cover the duration of the operation, starting
  when it is initiated and ending when the response is fully received
  or the operation is terminated, with automatic retries folded into
  the same span; and the `invoke_agent` operation name is a
  predefined `gen_ai.operation.name` value.
- OpenTelemetry semantic-conventions-genai pull request 557
  (`github.com/open-telemetry/semantic-conventions-genai/pull/557`,
  open): adds `gen_ai.skill.definitions` to internal `invoke_agent`
  spans, recording which skills were available to the agent when the
  invocation started; it notes that `gen_ai.tool.definitions` does
  not fit (its schema describes function calls and a skill has no
  parameter schema), and that the data already exists at the start of
  every run - the attribute gives it a standard place in a trace.
- OpenTelemetry semantic-conventions-genai GenAI attributes registry
  (`github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/registry/attributes/gen-ai.md`):
  `gen_ai.tool.definitions` is an any-typed Development attribute
  holding a structured list, and the existing `gen_ai.skill.*`
  attribute family (`gen_ai.skill.name`, `gen_ai.skill.description`,
  `gen_ai.skill.resource.name`, `gen_ai.skill.source.uri`) - the shape
  the definitions attribute will join.

## Privacy + content

- No names/handles/Slack-Discord URLs/timestamps/message-sequence
  numbers/employer/team/internal repo names/IPs/private logs in either
  page; anonymized summary only. Public GitHub repository, pull
  request, and doc links appear in `## References` / `## 参考` only.
- Content order per the standard: direct answer, mechanism,
  verification/debugging path, limitation, references, community
  discussion. H1s kept free of underscores, backticks, and
  apostrophes; the `invoke_agent` and `gen_ai.skill.definitions`
  identifiers stay short (the longest backticked token is the 26-
  character `gen_ai.skill.resource.name`, well under the 390px gate).

## Coverage disclosure

Two opt-in archive channels covered (the two CNCF OpenTelemetry
instrumentation channels) with thirteen messages in the 10-11
snapshot. The candidate was selected from the available-skills
modeling thread; the `model-serving-signals` and `logger.enabled`
threads were skipped (already published as the 10-09 and 10-10
entries, respectively), and the Node.js GenAI instrumentation
repository-plans thread was skipped (no decisive technical boundary
this run). Visible-browser-only sources (the eunomia-bpf and sched-ext
Discord servers, the bpf mailing list, and r/eBPF) were not reviewed
this run (no visible-browser session) - marked uncovered-not-quiet on
the page.

## Artifacts

- EN: `docs/ebpf-qa/2026-10-11-genai-available-skills-model-attribute-vs-span-vs-event.md`
- ZH: `docs/ebpf-qa/2026-10-11-genai-available-skills-model-attribute-vs-span-vs-event.zh.md`
- Index links: first items of `docs/ebpf-qa/index.md` (`Latest
  Answers`) and `docs/ebpf-qa/index.zh.md` (`最新回答`).
- Published QA commit on `origin/main`: `c3bda5865`
  (`docs(ebpf-qa): genai-available-skills-model-attribute-vs-span-vs-event (2026-10-11)`),
  containing exactly the four owned paths.
- Receipt: `/workspaces/.agent-state/eunomia-qa/receipt-2026-10-11.json`
  `status=published` (written by the validator only), commit
  `c3bda58650e258f3a76da311e235714a66f6c1f4`.

## eBPF Q&A publication follow-up (deploy + live verification)

- QA commit `c3bda5865` (4 paths) landed on `origin/main`; the
  GitHub Pages `Deploy Static App` run `38184645942` went green on
  that commit (`conclusion: success`), so all four routes returned 200
  with the expected H1s and the new slug present in both indexes.
  `cache-control: public, max-age=0, must-revalidate` and
  `cf-cache-status: DYNAMIC` confirm a fresh deploy, not a cached copy.
- The two live article pages render the verbatim EN and ZH H1s; the new
  slug is linked from both the EN and ZH index pages (slug count = 3
  on each, the expected link + metadata occurrences).
- The validator re-verify pass returned receipt `status=published`,
  commit `c3bda58650e258f3a76da311e235714a66f6c1f4`, with `branch`,
  `candidate_paths`, `index_links`, `privacy`, `remote_contains_commit`,
  and `public` all `ok`, and `content_test` / `build` / `render` /
  `commit_push` `skipped_already_published` (the re-verify path does
  not re-run those because the original commit already passed them).
- One concrete operational note hit this run, recorded durably in the
  QA publication standard: the validator's first invocation timed out
  at its 600s shell cap while polling the live routes - the GitHub
  Pages deploy takes about twenty minutes, longer than the cap. The QA
  commit had already landed on `origin/main` before the timeout, so
  the recovery was to watch the Pages run to green and then re-run the
  same validator, whose re-verify path completed the receipt without
  touching the commit again.

This run-log commits separately as the 10-11 same-day artifact.
