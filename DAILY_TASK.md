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
sources, quality gates, delivery rules, content-series strategy, and operating
state.

## Daily responsibilities

Every run performs these three responsibilities in order:

1. **Data analysis.** Collect and analyze the available search, acquisition,
   traffic, repository, and live-site evidence. This step is mandatory.
2. **Technical SEO and GEO.** Inspect crawlability, indexing signals, canonical
   ownership, language alternates, structured data, internal links, performance,
   and retrievability. Make a site change only when evidence supports it.
3. **Technical content production.** Publish exactly **one new Daily Report page
   every run**. Research inside the active content series first and enforce the
   rolling topic mix in `.github/seo-data/content-series.md`. If a candidate
   fails the evidence, gap, novelty, continuity, or usefulness gates, reject that
   candidate and continue researching another approved question until one passes.
   The daily publication requirement never lowers the quality bar.

Data analysis and series strategy should determine which report is worth
publishing. Do not fill the quota with a weak summary, recycled thesis, or
superficial trend piece; change the candidate instead.

## Required context

Start from the latest remote default branch and read:

- `CLAUDE.md`;
- this file;
- `.github/seo-data/site.md`;
- `.github/seo-data/status.md`;
- `.github/seo-data/plan.md`;
- `.github/seo-data/block.md`;
- `.github/seo-data/content-series.md` for the active Daily Report series,
  eBPF/content mix, and thematic roadmap;
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

Before topic selection, calculate the rolling Daily Report mix defined in
`.github/seo-data/content-series.md` and record it in the daily analysis.

## 2. Select today's report and SEO work

The run must publish one new Daily Report. Select it by this sequence:

1. inspect the rolling topic mix;
2. search and research inside the active series first;
3. build evidence for more than one candidate when necessary;
4. reject weak candidates rather than weakening the publication standard;
5. select one distinct question that passes the report skill's evidence, gap,
   mechanism, originality, usefulness, and evaluation gates;
6. keep the rolling mix compliant: normally 5–7 of the most recent 10 reports
   explicitly center eBPF, while pure Agent reports remain at most 1–2 of 10;
7. publish one new bilingual Daily Report and update series continuity records.

For technical SEO, make at most one coherent site change when the day's evidence
supports it. Prefer a change directly coupled to the report or to an important
measured technical issue. If an unrelated SEO change would make the daily PR
hard to review, put it in `plan.md` for a focused follow-up rather than mixing
unrelated work.

For content work, **series continuity is the default**. A new report must advance
a distinct question instead of restating an existing thesis with different
examples. Leave the active series only when another approved series yields a
stronger report or a material external development has durable systems
consequences. Record any switch.

For technical SEO, follow `.github/seo-data/daily-task.md`. For Daily Report,
follow `.agents/skills/eunomia-research-report/SKILL.md` and
`.github/seo-data/content-series.md`.

## 3. Deliver and verify

Use one fresh branch and one non-draft pull request for the daily analysis, the
mandatory new Daily Report, and any directly coupled SEO change. Do not create a
second closeout pull request.

Before merge:

1. run the relevant authoritative repository checks;
2. inspect the complete diff and generated output;
3. wait for required and expected CI;
4. fix issues on the same branch;
5. squash-merge only after a clean final self-review.

Wait for the production deployment of the exact squash commit and verify the new
English and Chinese report pages on the public site, including navigation,
canonical URLs, language alternates, internal links, gap section, idea section,
and conclusion boundary. Verify any coupled SEO behavior as well.

Add one compact closeout comment to the merged pull request with the merge commit,
deployment, public verification, data coverage, selected series and topic class,
and any remaining uncertainty. The pull request and its closeout comment are
part of the repository's operating record. The next daily run updates
`status.md` from those verified facts.

## Completion criteria

A daily run is complete only when:

- the daily data analysis exists with explicit source coverage;
- the current rolling Daily Report topic mix was calculated and recorded;
- `status.md`, `plan.md`, `block.md`, and `content-series.md` reflect current
  verified state as needed;
- exactly one new bilingual Daily Report was added and passes the content quality
  gates;
- the report follows the active or another approved series and keeps the rolling
  topic mix compliant;
- the pull request passed CI, final self-review, and squash merge;
- the exact merge commit deployed successfully;
- the new public English and Chinese report pages were verified;
- the merged pull request contains the compact closeout record.

`no-report`, a revision-only day, a data-only day, a draft, local commit, issue,
queued workflow, HTTP 200 alone, or unverified publication is not completion.
