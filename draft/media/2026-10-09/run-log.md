## 2026-10-09 eBPF Q&A run-report (eunomia-community-radar)

## Selected candidate

- Slug: `otel-genai-metric-catalog-boundary-model-serving-signals`
- Question: why the OpenTelemetry GenAI metric catalog stops at
  engine-agnostic latency and token counts, and how the Kubernetes
  `model-serving-signals` initiative closes the model-serving and
  autoscaling gap.
- Source: the 10-09 snapshot carries three messages, all in the two
  CNCF OpenTelemetry instrumentation channels. Message 1 asks whether a
  Node.js GenAI instrumentation repository is planned to match the
  existing Python one; no public answer is available yet and it does
  not touch the serving-metric boundary, so it was skipped. Message 2
  points at a specific pull-request review discussion in an agent
  tooling project and is not a self-contained practitioner question, so
  it was skipped. Message 3, the one this page answers, describes the
  new Kubernetes `model-serving-signals` initiative (repo under
  `kubernetes-sigs`, backed by SIG Autoscaling and SIG
  Instrumentation): taking metrics from inference engines (vLLM,
  SGLang, TensorRT-LLM) and translating them into OpenTelemetry, and
  asking whether a larger catalog of metrics is defined or planned,
  especially for model serving and autoscaling. It is in-scope and
  non-duplicative of the 10-05 entry, which covered the GenAI TRACE
  attribute `gen_ai.skill.definitions`; this entry is the GenAI METRIC
  catalog plus the K8s model-serving boundary.

## Verification against public primary sources

- OpenTelemetry GenAI metrics
  (`github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-metrics.md`):
  the model-server section defines exactly
  `gen_ai.server.request.duration`, `gen_ai.server.time_per_output_token`,
  and `gen_ai.server.time_to_first_token`, plus a client
  operation-duration histogram. No queue depth, concurrency, batch, or
  cache metric is defined there — the catalog is deliberately
  engine-agnostic and request-level only.
- OpenTelemetry GenAI client-inference metrics
  (`github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/client-inference.md`):
  adds inference duration, time-to-first-chunk, time-per-output-chunk,
  and per-operation token histograms. Still request-level; no
  serving-state signal.
- OpenTelemetry GenAI token metrics
  (`github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-token-metrics.md`):
  the `gen_ai.client.inference.usage.*` counters broken down by modality
  (input, output, cache-read, cache-write, reasoning). Token usage is an
  accounting quantity, not a capacity signal for autoscaling.
- `kubernetes-sigs` model-serving-signals
  (`github.com/kubernetes-sigs/model-serving-signals`): the
  engine-agnostic signal contract for model servers on Kubernetes —
  per-engine profiles for vLLM, SGLang, and TensorRT-LLM, a mapping
  exporter that ships them as OpenTelemetry, a conformance suite that
  pins the mapping, and autoscaling integration. This is the layer that
  translates engine-specific serving metrics into the consistent names
  an autoscaler consumes.

## Privacy + content

- No names/handles/URLs/timestamps/IPs/credentials/private logs in
  either page; anonymized summary only. Public GitHub repository and
  initiative names plus public doc links appear in `## References` /
  `## 参考` only.
- Content order per the standard: direct answer, mechanism,
  verification/debugging path, limitation, references, community
  discussion. Mobile-clean short inline code tokens. H1s kept free of
  underscores, backticks, and apostrophes; the repository word uses
  hyphens.

## Coverage disclosure

Two opt-in archive channels covered (the two CNCF OpenTelemetry
instrumentation channels) with three messages in the 10-09 snapshot.
The candidate was selected from the model-serving-signals message; the
Node.js GenAI repository-plans message and the agent-tooling
pull-request review message were skipped (no public answer / not a
self-contained question, respectively). Visible-browser-only sources
(the eunomia-bpf and sched-ext Discord servers, the bpf mailing list,
and r/eBPF) were not reviewed this run (no visible-browser session) —
marked uncovered-not-quiet on the page.

## Artifacts

- EN: `docs/ebpf-qa/2026-10-09-otel-genai-metric-catalog-boundary-model-serving-signals.md`
- ZH: `docs/ebpf-qa/2026-10-09-otel-genai-metric-catalog-boundary-model-serving-signals.zh.md`
- Index links: first items of `docs/ebpf-qa/index.md` (`Latest
  Answers`) and `docs/ebpf-qa/index.zh.md` (`最新回答`).
- Published QA commit on `origin/main`: `14224eb42`
  (`docs(ebpf-qa): otel-genai-metric-catalog-boundary-model-serving-signals (2026-10-09)`).
  The validator's scoped commit was originally `503b76f3c` (parent the
  pre-run base), whose push was rejected (`fetch first` — `origin/main`
  had advanced to `255e1a74f`). It was recovered with `git reset --soft
  origin/main` plus a scoped re-commit of the four owned paths (the
  concurrent agent's staged set untouched), fast-forwarding to
  `14224eb42`; `503b76f3c` is now orphaned and was never pushed.
- Receipt: `/workspaces/.agent-state/eunomia-qa/receipt-2026-10-09.json`
  `status=published` (written by the validator only), commit
  `14224eb42ea75bb6dce72bbb3d1e72542d792009`.
- This run-log commits separately as the 10-09 same-day artifact.

## eBPF Q&A publication follow-up (deploy + live verification)

- QA commit `14224eb42` (4 paths) landed on `origin/main`; the GitHub
  Pages `Deploy Static App` run `38005019348` went green on that commit
  (`conclusion: success`), so all four routes returned 200 with the
  expected H1s and the new slug present in both indexes.
  `cache-control: public, max-age=0, must-revalidate` and
  `cf-cache-status: DYNAMIC` confirm a fresh deploy, not a cached copy.
- The two live article pages render the verbatim EN and ZH H1s; the new
  slug is linked from both the EN and ZH index pages.
- The first validator re-verify pass ran while the Pages deploy was
  still in progress and recorded a transient 404; re-running the same
  command after the deploy landed (re-verify path, no re-commit)
  returned receipt `status=published`, commit
  `14224eb42ea75bb6dce72bbb3d1e72542d792009`, with `branch`,
  `candidate_paths`, `index_links`, `privacy`, `remote_contains_commit`,
  and `public` all `ok` and the four content gates
  `skipped_already_published`.
