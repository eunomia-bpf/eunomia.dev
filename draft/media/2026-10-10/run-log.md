## 2026-10-10 eBPF Q&A run-report (eunomia-community-radar)

## Selected candidate

- Slug: `genai-inference-event-emit-logger-enabled-optin-vs-optout`
- Question: why the OpenTelemetry GenAI instrumentation gates the
  client-inference log event on the log-pipeline severity instead of a
  dedicated emit flag, and whether `logger.enabled` is an opt-in or an
  opt-out control.
- Source: the 10-10 snapshot carries twelve messages, all in the two
  CNCF OpenTelemetry instrumentation channels
  (`otel-ebpf-instrumentation` / `otel-genai-instrumentation`). Three
  distinct threads: (1) the Node.js GenAI instrumentation repo lead —
  no decisive boundary, the instrumentation is landing in
  `otel-js-contrib` with auto-generated conventions in `otel-js`, and
  the only guidance is the `semantic-conventions-genai` issue 247, so
  it was skipped; (2) the one this page answers — a pull request in a
  Go agent-development toolkit (`adk-go`) was pointed at the Python
  GenAI utility change (`opentelemetry-python-genai` pull request 836)
  and asked whether the inference event should stay opt-in or move to
  opt-out; the thread pins the boundary on `logger.enabled` (which
  defaults to true, so it is not an opt-in gate), opt-out via a custom
  processor `enabled` method or declarative configuration, and
  minimum-severity / sample-flag filtering arriving in
  `opentelemetry-python` pull request 5724; a sub-thread on the skills
  attribute duplicates the published 10-05 entry and was skipped;
  (3) the `model-serving-signals` thread was already published as the
  10-09 entry, so it is cited and skipped. The selected theme is
  non-duplicative: 10-05 covered the GenAI TRACE skill attribute,
  10-09 covered the GenAI METRIC catalog / model-serving boundary,
  08-20 covered attribute stability, and 08-23 covered evaluation
  evidence — none cover inference-event emission gating.

## Verification against public primary sources

- OpenTelemetry semantic-conventions-genai client inference events
  (`github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/client-inference.md`):
  the client-inference operation-details event is a Recommended
  requirement level and says instrumentations SHOULD set the severity
  to DEBUG (severity number 5); it stores input and output details
  independently of traces.
- `opentelemetry-python` pull request 5724
  (`github.com/open-telemetry/opentelemetry-python/pull/5724`, open):
  adds minimum-severity and trace-based filtering to `Logger.enabled`,
  wired through declarative configuration; also disables events by
  instrumentation name, with minimum-severity / trace-sample-flag
  disabling to follow.
- `opentelemetry-python-genai` pull request 836
  (`github.com/open-telemetry/opentelemetry-python-genai/pull/836`,
  open): the `_maybe_create_event` path now checks `logger.enabled`
  with the event's DEBUG severity and a graceful fallback for loggers
  that do not yet accept a severity argument; the unconditional
  `EMIT_EVENT`-style environment flag is removed in favor of the
  content-capture flag plus the logger check.
- OpenTelemetry Python logs API
  (`opentelemetry.io/docs/languages/python/logs/`): the Python Logs
  API surface (`LoggerProvider`, `Logger`, `enabled`).
- Minor gap, non-blocking: the raw `Logger.enabled` source-file path in
  `opentelemetry-python` was not located directly (several
  `raw.githubusercontent` paths 404'd); the two pull requests plus the
  logs API doc fully establish the semantics, and the pages cite the
  pull requests, not the source file.

## Privacy + content

- No names/handles/Slack-Discord URLs/timestamps/message-sequence
  numbers/employer/team/internal repo names/IPs/private logs in either
  page; anonymized summary only. Public GitHub repository, pull
  request, and doc links appear in `## References` / `## 参考` only.
- Content order per the standard: direct answer, mechanism,
  verification/debugging path, limitation, references, community
  discussion. H1s kept free of underscores, backticks, and
  apostrophes; repository words use hyphens.
- Mobile-render fix: the site has no `word-break`/`overflow-wrap` on
  `<code>`, so any unbreakable monospace token wider than the 390px
  viewport overflows. The 43-character event name was written in plain
  words in the flowing body and references prose; only the remaining
  backticked tokens are the short 19-character method name and 14-
  character API names. Both pages pass the 390px and 1440px render
  gates with no horizontal overflow.

## Coverage disclosure

Two opt-in archive channels covered (the two CNCF OpenTelemetry
instrumentation channels) with twelve messages in the 10-10 snapshot.
The candidate was selected from the GenAI inference-event gating
thread; the Node.js GenAI repo-plans thread and the
`model-serving-signals` thread were skipped (no decisive boundary and
already published 10-09, respectively), and the skills-attribute
sub-thread duplicates the 10-05 entry. Visible-browser-only sources
(the eunomia-bpf and sched-ext Discord servers, the bpf mailing list,
and r/eBPF) were not reviewed this run (no visible-browser session) —
marked uncovered-not-quiet on the page.

## Artifacts

- EN: `docs/ebpf-qa/2026-10-10-genai-inference-event-emit-logger-enabled-optin-vs-optout.md`
- ZH: `docs/ebpf-qa/2026-10-10-genai-inference-event-emit-logger-enabled-optin-vs-optout.zh.md`
- Index links: first items of `docs/ebpf-qa/index.md` (`Latest
  Answers`) and `docs/ebpf-qa/index.zh.md` (`最新回答`).
- Published QA commit on `origin/main`: `1cfed31c0`
  (`docs(ebpf-qa): genai-inference-event-emit-logger-enabled-optin-vs-optout (2026-10-10)`).
  The validator's first scoped commit was `9e6be0b` (parent the pre-run
  base), whose push was rejected (`non-fast-forward` — `origin/main` had
  advanced to `362b0c1`). It was recovered with `git reset --soft
  origin/main` plus a scoped re-commit of the four owned paths (the
  concurrent agent's staged set untouched), fast-forwarding to
  `1cfed31c0`; `9e6be0b` is now orphaned and was never pushed.
- Receipt: `/workspaces/.agent-state/eunomia-qa/receipt-2026-10-10.json`
  `status=published` (written by the validator only), commit
  `1cfed31c03acc88088af4954a75f596c7b849cf3`.
- This run-log commits separately as the 10-10 same-day artifact.

## eBPF Q&A publication follow-up (deploy + live verification)

- QA commit `1cfed31c0` (4 paths) landed on `origin/main`; the
  GitHub Pages `Deploy Static App` run `38096466998` went green on
  that commit (`conclusion: success`), so all four routes returned 200
  with the expected H1s and the new slug present in both indexes.
  `cache-control: public, max-age=0, must-revalidate` and
  `cf-cache-status: DYNAMIC` confirm a fresh deploy, not a cached copy.
- The two live article pages render the verbatim EN and ZH H1s; the new
  slug is linked from both the EN and ZH index pages.
- The validator re-verify pass returned receipt `status=published`,
  commit `1cfed31c03acc88088af4954a75f596c7b849cf3`, with `branch`,
  `candidate_paths`, `index_links`, `privacy`, `remote_contains_commit`,
  and `public` all `ok`.
