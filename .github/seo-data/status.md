# SEO status

## Current state

- Authoritative task: `DAILY_TASK.md`
- Technical SEO subtask: `.github/seo-data/daily-task.md`
- Daily Report subtask: `.agents/skills/eunomia-research-report/SKILL.md`
- External daily scheduler: configured and enabled
- Verified raw Google export window: through `2026-08-23`
- Search Console verified latest finalized row: `2026-08-22`; `2026-08-23` is absent
- Latest finalized GA4 weekly organic landing-page aggregate: `2026-08-17` through `2026-08-23`
- Latest completed daily record before the current run: `2026-08-28`
- Last completed Daily Report pull request: `#178`
- Last verified Daily Report squash commit: `201133f68661952596fa5489c621d08ffce685da`
- Last verified production publication from a Daily Report run: static export commit `8a470b4c4510ae09397112335d4d7a94781e6af4`
- Current daily branch: `daily/2026-08-29-gpu-memory-placement-evidence`
- Skill submodule commit: `516e9e2dcf012506a677a749049d64c5914643e9`

PR `#178` is fully closed out. It squash-merged as
`201133f68661952596fa5489c621d08ffce685da`; exact-merge `Validate SEO Operations`
run `33189308226` and `Deploy Static App` run `33189308343` succeeded. Production
published static export commit `8a470b4c4510ae09397112335d4d7a94781e6af4`,
whose commit message binds the export to that squash SHA. Exact EN/ZH production
artifacts and sitemap entries were verified, and PR `#178` has exactly one
merged-PR closeout comment.

## Current Daily Report mix

A direct recount of the actually published Daily Report index on `2026-08-29`
corrects the stale `7 / 0 / 3` operating record. Before today's report, the newest
ten are:

- eBPF-centered: **8 of 10**
- pure Agent-centered: **0 of 10**
- adjacent systems: **2 of 10**

Today's `/research/gpu-memory-placement-evidence/` report is **adjacent systems**.
Its central mechanism is GPU Unified Memory placement under oversubscription: it
compares demand faults, sampled resident-page activity, object/phase semantics,
and scheduling knowledge, then develops evidence-carrying placement records,
placement intent with observable compliance, and a fixed-budget decision-regret
benchmark. eBPF can contribute instrumentation but is not required by the
mechanism, so reclassifying the report as eBPF-centered would be inaccurate.

The incoming adjacent report displaces one of the two August 20 adjacent GPU
reports, so after publication the mix remains **8 eBPF / 0 pure Agent / 2
adjacent**. This is one eBPF report above the repository target band. A single run
cannot repair it without publishing more than one report or misclassifying
content. An adjacent report on the next normal run would displace the remaining
August 20 adjacent report and still leave **8 / 0 / 2**; a second consecutive
adjacent report after that would displace the August 21 eBPF report and restore
**7 / 0 / 3**. Prefer genuinely adjacent GPU/runtime questions over the next two
normal runs when the quality gates permit; do not repair the ratio through
classification.

**GPU and Heterogeneous Runtime Systems** is now the active normal series. The
August 20 GPU launch-latency and host/device-causality reports are useful anchors
but predate activation. Today's memory-placement report is the first substantial
post-activation contribution and is intentionally distinct from those profiling
questions.

## Current signals

- Google Drive configured evidence: directly reverified `2026-08-29`; the newest source-native weekly set remains `2026-08-17..23`
- Fresh automated site brief generated `2026-08-29 13:14 UTC`: homepage `200`, robots `200`, sitemap `200`, and 710 sitemap entries observed
- Public homepage: current live fetch shows expected Eunomia identity and Daily Report navigation
- Canonical homepage: `https://eunomia.dev/`
- Public GitHub repository evidence: available; private repository/account details are not copied into SEO records
- Public web and primary-source evidence: available
- Google Analytics 4: finalized source-native weekly aggregate through `2026-08-23`
- Google Search Console: newest verified finalized date row `2026-08-22`; complete seven-day comparison still unavailable because required source rows are missing
- Cloudflare: disabled by repository configuration

The longest directly supported equal-duration finalized Search Console comparison
remains `2026-08-17..22` versus `2026-08-10..15`: **477 clicks / 59,798
impressions**, weighted CTR about **0.798%**, and impression-weighted average
position about **9.56**, versus **393 / 66,510**, **0.591%**, and position about
**9.91**. Clicks are about **21.4% higher**, impressions about **10.1% lower**,
CTR about **0.207 percentage points higher**, and average position improves by
about **0.35 positions**.

A complete latest-seven-days versus previous-seven-days GSC comparison remains
unavailable because the required windows need the absent `2026-08-16` and
`2026-08-09` rows. The required 28-day comparison also remains unavailable
because verified export history is too short. Missing rows are never interpreted
as zero.

The finalized GA4 `2026-08-17..23` aggregate remains **984 organic landing-page
sessions** at about **49.29%** session-weighted engagement, including **118
`(not set)` sessions**, versus **970 sessions** at about **44.95%** engagement for
`2026-08-10..16`. Sessions are about **1.4% higher** and engagement about **4.34
percentage points higher**. GPU/CUDA tutorial and profiling pages remain visible
among meaningful organic landing pages, which supports the technical relevance
of the active series but does not establish a causal SEO effect. The weekly
aggregate has no date dimension, so it cannot support daily or within-week causal
attribution.

## Current technical baseline

The repository generates sitemap, robots, canonical, `hreflang`, Open Graph,
structured data, legacy redirect stubs, and static audit artifacts. Production
deploys through `Deploy Static App`.

Fresh live-site inspection, the `2026-08-29` automated site brief, and current
Google evidence do not establish a crawl, robots, sitemap, canonical, hreflang,
structured-data, redirect, broken-link, rendering, accessibility, performance,
or deployment defect that justifies a separate technical SEO implementation
change today. The public web crawler's `/research/` snapshot can lag production;
that cache is not treated as a site defect or as deployment evidence.

The SEO skill submodule remains pinned at
`516e9e2dcf012506a677a749049d64c5914643e9`. The repository still requires a
contract migration before a pointer-only update to a newer upstream layout, so no
skill-submodule movement is part of this run.

## Current focus

1. Complete the `2026-08-29` GPU memory-placement Daily Report through one non-draft PR, expected CI, complete final diff/generated-output self-review, squash merge, exact production deployment, bilingual production verification, and one merged-PR closeout comment.
2. Keep **GPU and Heterogeneous Runtime Systems** active. Prefer genuinely adjacent GPU/runtime questions over the next two normal runs when they pass the quality gates: the first would leave the newest-ten mix at `8 / 0 / 2`, while the second would rotate the August 21 eBPF report out and restore the target-compatible `7 / 0 / 3` without dishonest classification.
3. Recheck Drive freshness every run; no weekly set newer than `2026-08-17..23` was observed on `2026-08-29`.
4. Keep complete GSC 7-day and 28-day comparisons unavailable until source history supports them; never fill missing date rows with zero.
5. Use date-by-page or date-by-query evidence before attributing aggregate search movement to a page, title, or topic family.
6. Treat finalized GA4 weekly movement as aggregate behavioral evidence, not a causal content or SEO result.
7. Investigate GA4 `(not set)` and remaining legacy `/en/` traffic only with richer source-native evidence.
8. Migrate the consuming SEO contract before updating the `seo-skills` submodule pointer.
9. Add Cloudflare evidence only when a supported read-only path is enabled in repository configuration.

Detailed history belongs in `.github/seo-data/daily/` and merged daily pull requests.
