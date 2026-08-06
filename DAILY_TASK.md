# Daily site operation

This file is the single authoritative entrypoint for the scheduled eunomia.dev
operation. The current default branch of this repository is the source of truth.
Do not copy operational policy into an external scheduler.

## Scheduler contract

The external scheduler should contain only this instruction:

> Open `eunomia-bpf/eunomia.dev`, read `DAILY_TASK.md` from the current default
> branch, and complete the task exactly as the repository instructs. Treat the
> repository as authoritative.

Run once per day in the timezone declared in `.github/seo-data/site.md`. Maintain
exactly one external scheduler for this operation; do not split analytics, SEO,
and content production into independent schedules that can race or diverge. The
scheduler owns only timing and invocation. This repository owns scope, data
sources, quality gates, delivery rules, and operating state.

## Daily responsibilities

Every run performs these three responsibilities in order:

1. **Data analysis.** Collect and analyze the available search, acquisition,
   traffic, repository, and live-site evidence. This step is mandatory even when
   no public change is justified.
2. **Technical SEO and GEO.** Inspect crawlability, indexing signals, canonical
   ownership, language alternates, structured data, internal links, performance,
   and retrievability. Make a change only when evidence supports it.
3. **Technical content production.** Research a Daily Report candidate and
   publish only when it passes the repository's evidence, gap, novelty, and
   usefulness gates. Daily research does not impose a daily publication quota.

Data should determine the work. Do not manufacture an article, SEO edit, or
visible artifact merely to satisfy cadence.

## Required context

Start from the latest remote default branch and read:

- `CLAUDE.md`;
- this file;
- `.github/seo-data/site.md`;
- `.github/seo-data/status.md`;
- `.github/seo-data/plan.md`;
- `.github/seo-data/block.md`;
- the newest record under `.github/seo-data/daily/`, when present;
- `.github/seo-data/daily-task.md` for the technical SEO subtask;
- `.agents/skills/seo-geo/SKILL.md` and the pinned `.github/seo-skills` skills;
- `.agents/skills/eunomia-research-report/SKILL.md` and its research-method
  reference for Daily Report work.

Resolve conflicts in favor of the most specific current repository instruction.
Do not rely on copied scheduler text, an old run summary, or chat history.

## 1. Analyze data first

Use every source marked enabled in `.github/seo-data/site.md`. Report disabled,
missing, stale, partial, or unavailable sources explicitly; never convert missing
coverage into zero traffic or zero demand.

Use the finalization lag and lookback window from `site.md`. At minimum compare:

- the latest complete 7-day window with the preceding 7 days;
- the latest complete 28-day window with the preceding comparable period when
  enough history exists;
- important pages, queries, referrers, repositories, or technical signals
  against their own prior baseline rather than an invented blended score.

Analyze, when available:

- Search Console clicks, impressions, CTR, average position, queries, pages,
  countries, devices, canonical and indexing state;
- GA4 users, sessions, engaged sessions, engagement, landing pages, acquisition
  sources, outbound repository/paper/tutorial movement, and configured outcomes;
- Cloudflare requests, unique visitors, bots, status codes, cache behavior,
  countries, and unusual traffic changes;
- GitHub repository traffic, referrers, clones, stars, forks, releases, issues,
  and relevant community movement when access and semantics are available;
- live-site crawl, HTTP behavior, sitemap, robots, canonical, hreflang,
  structured data, broken links, performance, rendering, and deployment health.

Store no raw private analytics, credentials, account identifiers, or personal
information in Git. Write a compact, source-attributed daily analysis to
`.github/seo-data/daily/YYYY-MM-DD.md` and refresh `status.md`. A daily record
must state coverage, windows, material changes, likely explanations, uncertainty,
and the next evidence that would distinguish competing explanations.

## 2. Choose zero or one coherent public change

After analysis, choose the highest-value action supported by evidence:

- one coherent technical SEO/GEO improvement;
- one strong Daily Report or revision;
- a content change together with directly coupled technical metadata;
- or no public change, with the reason recorded.

Do not ship unrelated SEO and content edits in one pull request. Put durable
future work in `plan.md`; use `block.md` only for real external permission or
human-only blockers.

For technical SEO, follow `.github/seo-data/daily-task.md`. For a Daily Report,
follow `.agents/skills/eunomia-research-report/SKILL.md`. A fluent summary without
a defensible systems gap, mechanism, and testable direction is a no-publish
result.

## 3. Deliver and verify

Use one fresh branch and one non-draft pull request for the daily record and its
single coherent change. A data-only or no-change run may use a metadata-only pull
request. Do not create a second closeout pull request.

Before merge:

1. run the relevant authoritative repository checks;
2. inspect the complete diff and generated output;
3. wait for required and expected CI;
4. fix issues on the same branch;
5. squash-merge only after a clean final self-review.

For a public site change, wait for the production deployment of the exact squash
commit and verify the defined behavior on the public site. Add one compact
closeout comment to the merged pull request with the merge commit, deployment,
public verification, data coverage, and any remaining uncertainty. The pull
request and its closeout comment are part of the repository's operating record.
The next daily run updates `status.md` from those verified facts.

## Completion criteria

A daily run is complete only when:

- the daily data analysis exists with explicit source coverage;
- `status.md`, `plan.md`, and `block.md` reflect current verified state as needed;
- the run selected at most one coherent public change or recorded a defensible
  no-change decision;
- the pull request passed CI, final self-review, and squash merge;
- any public change deployed from the exact merge commit and was verified;
- the merged pull request contains the compact closeout record.

A draft, local commit, issue, queued workflow, HTTP 200 alone, or unverified
publication is not completion.
