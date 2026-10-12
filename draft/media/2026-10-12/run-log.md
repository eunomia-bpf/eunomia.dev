## 2026-10-12 eBPF Q&A run-report (eunomia-community-radar)

## Selected candidate

- Slug: `nodejs-genai-instrumentation-otel-js-contrib-vs-dedicated-repo`
- Question: why Node.js GenAI instrumentation lives in
  `opentelemetry-js-contrib` rather than a dedicated repository, and why
  there is no Node.js equivalent to the Python GenAI instrumentation repo.
- Source: the 10-12 snapshot carries fourteen messages, all in the two
  CNCF OpenTelemetry instrumentation channels
  (`otel-ebpf-instrumentation` / `otel-genai-instrumentation`). The thread
  this page answers: a practitioner asked whether there were plans for a
  Node.js repository similar to the Python GenAI instrumentation repo; the
  reply was that, as of now, there were no known plans for one, and that the
  GenAI instrumentation is landing in `opentelemetry-js-contrib` with the
  auto-generated conventions landing in `opentelemetry-js`, with
  `semantic-conventions-genai` issue 247 the only visible guidance on which
  artifacts to generate and whether they live in separate repos.
- Decisive boundary: the distribution split. The GenAI work separates into a
  language-neutral conventions layer (`semantic-conventions-genai`) that
  federates into the core `opentelemetry-js` client, and an asymmetric
  instrumentation layer - Python has a dedicated `opentelemetry-python-genai`
  repo, while Node.js has no standalone GenAI repo and its OpenAI
  auto-instrumentation ships from the `opentelemetry-js-contrib` monorepo.
  The absence of a Node.js GenAI repo is the expected result of that
  layout, not an access gap.
- This is the Node.js GenAI repository thread that the 10-11 run deferred
  ("no decisive technical boundary this run"). This run supplies that
  boundary: the conventions/instrumentation distribution split, plus the
  Python dedicated-repo asymmetry. It is non-duplicative against the four
  saturated telemetry-data-model threads - 10-05
  (offered-vs-executed blind spot), 10-09 (model-serving-signals), 10-10
  (logger.enabled opt-in/opt-out), and 10-11 (available-skills-as-attribute).

## Verification against public primary sources

- OpenTelemetry semantic-conventions-genai README
  (`github.com/open-telemetry/semantic-conventions-genai`): the
  language-neutral source of truth for GenAI spans, metrics, events, and the
  attribute registry; it extends the core semantic-conventions repo and
  generates per-language reference material, so it has no per-language home
  and carries no per-language instrumentation code.
- OpenTelemetry semantic-conventions-genai issue 247
  (`github.com/open-telemetry/semantic-conventions-genai/issues/247`): the
  discussion of which artifacts should be generated for a language and
  whether they belong in separate repos - the only visible guidance on the
  repo-organization question.
- OpenTelemetry opentelemetry-python-genai
  (`github.com/open-telemetry/opentelemetry-python-genai`): the dedicated
  Python GenAI instrumentation repo, with per-client packages (OpenAI,
  Anthropic, Bedrock, LangChain, Llama-Index, OpenAI Agents, and others)
  built on a shared GenAI helper package - the layout that has no Node.js
  counterpart.
- OpenTelemetry opentelemetry-js-contrib
  (`github.com/open-telemetry/opentelemetry-js-contrib`), specifically the
  `packages/instrumentation-openai` package: the single monorepo for
  community-maintained JS instrumentation, where the OpenAI GenAI
  auto-instrumentation lives as one package among many.
- OpenTelemetry opentelemetry-js release notes (1.42.0): the GenAI semantic
  conventions federate into the shared `semantic-conventions-genai` repo and
  their auto-generated bindings ship in the core `opentelemetry-js`
  client - why the conventions side is not split per language.

## Privacy + content

- No names/handles/Slack-Discord URLs/timestamps/message-sequence
  numbers/employer/team/internal repo names/IPs/private logs in either
  page; anonymized summary only. Public GitHub repository, issue, and doc
  links appear in `## References` / `## 参考` only.
- Content order per the standard: direct answer, mechanism,
  verification/debugging path, limitation, references, community
  discussion. H1s kept free of underscores, backticks, and apostrophes;
  the body's longest backticked token is the 26-character
  `semantic-conventions-genai`, well under the 390px gate. The one 41-
  character token, `@opentelemetry/auto-instrumentations-node`, was
  rewritten in words in the body ("the standard auto-instrumentations
  bundle for Node") and kept in its exact form only in a References item.

## Coverage disclosure

Two opt-in archive channels covered (the two CNCF OpenTelemetry
instrumentation channels) with fourteen messages in the 10-12 snapshot.
The candidate was selected from the Node.js GenAI repository-organization
thread. Three other threads in the window were skipped because they echo
already-published entries: the `model-serving-signals` catalog question
(10-09), the `logger.enabled` opt-in/opt-out question (10-10), and the
available-skills span-attribute modeling question (10-11). Visible-
browser-only sources (the eunomia-bpf and sched-ext Discord servers, the
bpf mailing list, and r/eBPF) were not reviewed this run (no visible-
browser session) - marked uncovered-not-quiet on the page.

## Artifacts

- EN: `docs/ebpf-qa/2026-10-12-nodejs-genai-instrumentation-otel-js-contrib-vs-dedicated-repo.md`
- ZH: `docs/ebpf-qa/2026-10-12-nodejs-genai-instrumentation-otel-js-contrib-vs-dedicated-repo.zh.md`
- Index links: first items of `docs/ebpf-qa/index.md` (`Latest Answers`)
  and `docs/ebpf-qa/index.zh.md` (`最新回答`).
- Published QA commit on `origin/main`: `5c2bf5ca3`
  (`docs(ebpf-qa): nodejs-genai-instrumentation-otel-js-contrib-vs-dedicated-repo (2026-10-12)`),
  containing exactly the four owned paths.
- Receipt: `/workspaces/.agent-state/eunomia-qa/receipt-2026-10-12.json`
  `status=published` (written by the validator only), commit
  `5c2bf5ca360e4851e45e8b0654c4ab9a89db9c53`.

## eBPF Q&A publication follow-up (deploy + live verification)

- QA commit `5c2bf5ca3` (4 paths) landed on `origin/main`; the GitHub
  Pages `Deploy Static App` run `38322559202` went green on that commit
  (`conclusion: success`), so all four routes returned 200 with the
  expected H1s and the new slug present in both indexes.
  `cache-control: public, max-age=0, must-revalidate` and
  `cf-cache-status: DYNAMIC` confirm a fresh deploy, not a cached copy.
- The two live article pages render the verbatim EN and ZH H1s; the new
  slug is linked from both the EN and ZH index pages (the slug substring
  appears six times in each index page's raw HTML - the link plus its
  surrounding metadata occurrences).
- The validator re-verify pass returned receipt `status=published`, commit
  `5c2bf5ca360e4851e235714a66f6c1f4`, with `branch`, `candidate_paths`,
  `index_links`, `privacy`, `remote_contains_commit`, and `public` all
  `ok`, and `content_test` / `build` / `render` / `commit_push`
  `skipped_already_published` (the re-verify path does not re-run those
  because the original commit already passed them).
- Two concrete operational notes hit this run:
  1. The first validator run was capped at 300s, which is shorter than
     the local `npm test:content` + build + Chromium render pass, so it
     was killed *before* `commit_and_push` - nothing landed. This is the
     pre-commit mirror of the 10-11 post-commit timeout. Because
     `origin/main` had advanced to `f0583c838` in the meantime, the
     recovery was to forward-move with `git reset --soft origin/main`
     (preserving the concurrent staged set and worktree), then re-run the
     validator with a cap long enough to finish build + render + commit
     (it completed in ~1800s, landing `5c2bf5ca3`).
  2. That run's `public` check then 404'd only because the Pages deploy
     (15-17 minutes) had not finished. Watching the green Pages run and
     re-running the same validator produced the `status=published`
     receipt, exactly as the 10-11 recovery prescribes.

This run-log commits separately as the 10-12 same-day artifact.
