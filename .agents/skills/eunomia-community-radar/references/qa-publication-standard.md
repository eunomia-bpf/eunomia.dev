# Public eBPF Q&A Standard

Use this reference when drafting and reviewing the daily public Q&A.

## Reader Outcome

The page should let an engineer understand the answer without knowing the
original conversation. It should explain the decisive boundary, show how to
verify it, and state where the answer stops applying.

## Recommended Shape

1. Use the real practitioner question as the title after anonymization.
2. Start with a direct short answer.
3. Explain the mechanism that makes the answer true.
4. Give a small verification or debugging path.
5. State the important limitation or alternative.
6. Add a References section with the public primary sources used.
7. End with an anonymized analysis of the day's wider community discussion.

The H1 must read as a plain technical question. The publisher renders each
question as the page H1, and the publication validator re-fetches that H1 from
the live page and requires the exact Markdown title string to appear verbatim.
Because the site's H1 pipeline strips underscores from a plain-text title (so
`bpf_probe_read_user` renders as `bprobereaduser`) and HTML-encodes apostrophes,
keep the H1 free of underscores, backticks, and apostrophes; put helper names
inside the H1 as ordinary words or move them into the answer body where
backticks belong. An H1 that fails the live-content check forces an extra
redeploy just to fix the title.

Keep inline code tokens short. The publication gate renders the built page
at a 390px mobile viewport and fails on any horizontal overflow, and the
site does not wrap monospace content, so an unbreakable backticked token
wider than the mobile content box (roughly 40+ characters, e.g. a dotted
identifier) pushes the page past the gate even when it looks fine at desk
width. In flowing body prose, name very long identifiers in words and keep
the exact backticked form only where it fits (a References list item);
measure the longest token before publishing. A 10-10 run tripped this on a
43-character event name and cleared the gate only after rewriting it as
prose in the body and the references lead-in.

Use only the sections the question needs. This is a useful shape, not a required
template.

## Anonymization Test

Before publishing, confirm that the page contains none of the following:

- a person's name, handle, avatar, employer, or team;
- a workspace, server, channel, or message URL;
- an exact timestamp or sequence that identifies the source thread;
- private logs, hostnames, IP addresses, repository names, credentials, or
  deployment topology;
- copied wording that can be searched back to the participant; or
- an assertion that a community member, customer, or organization endorsed the
  answer.

If removing those details changes the technical answer, do not publish it.

## Evidence And Privacy

Prefer kernel documentation, project documentation, upstream source code,
standards, and papers. A community message is a lead, not public evidence.

Do not collect sensitive data merely to demonstrate that it can be detected.
When the subject itself concerns secrets or private payloads, explain which
component can see plaintext, what leaves that component, and how to verify that
raw values are absent from maps, buffers, logs, traces, and exported telemetry.

## Publication Floor

Every daily run publishes one technically useful, independently verifiable,
non-duplicative Q&A that remains safe after anonymization. When no single
message is sufficient, combine related same-day signals, use a real unresolved
question from the monitored communities regardless of age, or select a genuine
question from permitted public community pages and public primary
documentation. Report access and coverage gaps honestly in the run notes; a
coverage gap is always reported and never blocks publication. Never invent a
question, fabricate evidence, or publish a placeholder.

## Daily Community Briefing

The Q&A page also carries the daily community report. After the references:

- state how many communities and allowlisted channels were actually reviewed;
- group discussion by technical theme rather than source or participant;
- explain the concrete question or symptom, likely mechanism, practical next
  diagnostic or resolution step, and any unresolved boundary for each
  substantive theme;
- cite the public primary references used for these concise technical answers;
- preserve technical depth while removing identity and deployment details;
- distinguish quiet channels from inaccessible channels; and
- omit names, handles, employers, channel names, message URLs, exact timestamps,
  raw quotes, and distinctive deployment details.

When several substantive discussions are present, give them enough space to be
useful on their own. A one-sentence trend label is not a community report; the
discussion section should normally be at least twice as detailed as such a
compressed summary.

## Live Public Routes

After the commit lands on `origin/main`, the routes only go live when the
GitHub Pages pipeline deploys it. The validator's 20-minute public check assumes
that pipeline succeeds. If the routes still 404 after that window, do not
read it as CDN lag: inspect the Pages workflow run for the pushed commit. A
failed run — typically a failing `npm run test:content` in the static-export
step — keeps every later route change 404ing until the pipeline is green again
(a 10-05 run 404ed for hours exactly this way after an unrelated commit broke
a hard-coded count assertion in the content tests; the unblock was fixing the
test and getting a later deploy green, not waiting). Check `cache-control`/
`age` on any fetched page to separate stale content, a failed deploy, and a
cache.
A distinct 404 cause is a failing runner image, not a failing content step:
`ubuntu-latest` can rotate to a distro the pinned Playwright has no chromium
build for, so the "Install browser audit dependencies" step dies and every
later step — build, publish, Pages deploy — is skipped. A 10-22 run failed
exactly this way (the runner resolved to `ubuntu26.04-x64`; Playwright 1.58.2
refused to install chromium on it; all routes 404ed until the pipeline was
repaired). Unblock by pinning the deploy job's runner to a distro the pinned
Playwright supports — commit `runs-on: ubuntu-24.04` on `main`, which
re-triggers the pipeline because GitHub runs the workflow file from the
pushed commit — not by waiting, loosening the browser step, or bumping
Playwright blindly.

The validator's public-route poll is bounded by the shell command cap, which is
shorter than a Pages deploy (about twenty minutes), so a first validator run
can time out *after* it has already committed and pushed the QA commit while
it waits for the live routes. A timed-out run does not mean publication
failed: check that the QA commit is on `origin/main` and that the Pages run
for it is green, then re-run the same validator for the date. The re-verify
path skips the commit, build, and render work, re-checks the live routes, and
rewrites the receipt to `status=published` in a couple of seconds (a 10-11 run
hit exactly this: commit `c3bda5865` landed, the first validator timed out
at its 600s cap, the Pages run went green, and the re-verify pass completed
the receipt without touching the commit again).

The cap must also cover the pre-commit work: the validator runs
`npm test:content`, the full production build, and two Chromium renders
*before* it commits. A cap shorter than that local pipeline (e.g. 300s)
kills the run before `commit_and_push` runs at all, so nothing lands and
no receipt is written - the mirror of the post-commit timeout above. Give
the run a cap long enough to finish build, render, commit, push, and the
public poll together; if `origin/main` advanced in the meantime, forward-
move with `git reset --soft origin/main` (which preserves any concurrent
staged set and the worktree) and re-run (a 10-12 run hit this: a 300s cap
killed the first run before the commit, `origin/main` had advanced, the
forward-move + longer-cap re-run landed commit `5c2bf5ca3`, and the
re-verify pass then completed the receipt).

The validator's two Chromium renders can fail at browser launch, not at
render, when the host is missing the system shared libraries the headless
shell needs (`libnspr4`, `libnss3`, `libatk`, `libdbus`, `libgbm`,
`libxkbcommon`, `libasound`, and the X11 extras). The failure surfaces as
`error while loading shared libraries` from the `chrome-headless-shell`
binary, before any commit lands, so the receipt is `failed` with no QA
commit. Fix the environment, not the run: `ldd` the headless-shell binary to
list the missing `.so` files, `apt-get install` the matching packages, confirm
`ldd` reports no `not found` and the binary's `--version` runs, then re-run
the validator from the same cap - the content test, build, and render all
succeed on the retry and the commit lands. A 10-17 run hit this on a fresh
container: fourteen libraries were absent, the render step aborted before the
commit, and the forward-move + re-run only completed after the shared
libraries were installed. On such a fresh host the apt package index is also
often empty or stale, so a plain `apt-get install` fails with
`Unable to locate package` before any of the libraries install; run
`apt-get update` first, then the install. A 10-19 run hit exactly this:
`apt-get install` reported every library as unlocatable, and only after an
`apt-get update` did the fourteen-package install succeed and the re-render
land the commit.

A fresh container's system `python3` may lack `PyYAML`, and the step-0
snapshot tool (`archive_reader.py`) reads the watchlist with `import yaml`
inside a blanket `except Exception` that collapses *any* load failure to the
bare message `watchlist unreadable` and exits `inaccessible` - so a missing
interpreter dependency masquerades as an unreadable watchlist and hides the
real cause. Run the snapshot under the state venv's Python
(`/workspaces/.agent-state/eunomia-qa/venv/bin/python archive_reader.py …`),
where `yaml` (and `psycopg`) are installed, and read the step-0 output's
`opt_in_sources` / message counts before treating the window as frozen: a
`watchlist unreadable` exit means the archive was *not* actually read this
run. A 10-18 run hit this on a fresh host: system `python3` had no PyYAML,
the snapshot aborted as `watchlist unreadable`, and only the venv re-run
produced the real 2-source, 11-message frozen window.
