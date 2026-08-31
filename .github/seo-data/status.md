# SEO status

## Current state

- Authoritative task: `DAILY_TASK.md`
- Technical SEO subtask: `.github/seo-data/daily-task.md`
- Daily Report subtask: `.agents/skills/eunomia-research-report/SKILL.md`
- External daily scheduler: configured and enabled
- Verified raw Google export window: through `2026-08-23`
- Search Console verified latest finalized row: `2026-08-22`; `2026-08-23` is absent
- Latest finalized GA4 weekly organic landing-page aggregate: `2026-08-17` through `2026-08-23`
- Latest completed daily record before the current run: `2026-08-30`
- Last completed Daily Report pull request: `#181`
- Last verified Daily Report squash commit: `3ede202d350a7d6f728d7b7e7187a3b1abba8145`
- Last verified production publication from a Daily Report run: static export commit `60b41bda401c36e6f595bf012de5eded130ba5f5`
- Current daily branch: `daily/2026-08-31-gpu-allocatability-contract`
- Skill submodule commit: `516e9e2dcf012506a677a749049d64c5914643e9`

PR `#181` is fully closed out. It squash-merged as
`3ede202d350a7d6f728d7b7e7187a3b1abba8145`; exact-merge `Validate SEO
Operations` run `33322348434` and `Deploy Static App` run `33322348476` both
succeeded. Production published static export commit
`60b41bda401c36e6f595bf012de5eded130ba5f5`, whose commit message binds the
export to that squash SHA. Exact EN/ZH generated production artifacts were
verified for canonical URLs, reciprocal language alternates, Article JSON-LD,
Daily Report navigation, and required report structure. PR `#181` now has the
single required closeout comment.

## Current Daily Report mix

After the `2026-08-30` adjacent GPU instrumentation-safety report, the newest ten
published reports remain:

- eBPF-centered: **8 of 10**
- pure Agent-centered: **0 of 10**
- adjacent systems: **2 of 10**

This is one eBPF report above the normal target band. The current run therefore
selects another genuinely adjacent question inside **GPU and Heterogeneous
Runtime Systems**. The report asks whether utilization telemetry is enough to
admit a specific new kernel or task onto a shared GPU. It is adjacent systems
because the core mechanisms are CUDA resource accounting, SM partitioning,
MPS/Green Contexts, synchronization progress, and scheduler admission; eBPF is
not required by the question.

If published, the report rotates the `2026-08-21` eBPF-centered report out of the
newest ten and restores the rolling mix to **7 eBPF / 0 pure Agent / 3 adjacent**.
The classification follows the mechanism rather than being changed to repair the
ratio.

## Current signals

- Google Drive configured evidence: directly rechecked `2026-08-31`; no source-native weekly set newer than `2026-08-17..23` was observed
- Public homepage: fresh `2026-08-31` inspection shows the expected Eunomia identity, project navigation, and Daily Report entry
- The August 30 English Daily Report is independently discoverable and was crawled `2026-08-31`
- Canonical homepage: `https://eunomia.dev/`
- Public GitHub repository evidence: available; private repository/account details are not copied into SEO records
- Public web and primary-source evidence: available
- Google Analytics 4: finalized source-native weekly aggregate through `2026-08-23`
- Google Search Console: newest verified finalized date row remains `2026-08-22`; complete seven-day comparison remains unavailable because required source rows are missing
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
percentage points higher**. The weekly aggregate has no date dimension, so it
cannot support daily or within-week causal attribution.

## Current technical baseline

The repository generates sitemap, robots, canonical, `hreflang`, Open Graph,
structured data, legacy redirect stubs, and static audit artifacts. Production
deploys through `Deploy Static App`.

Fresh live-site inspection, the current generated-site baseline, and available
Google evidence do not establish a crawl, robots, sitemap, canonical, hreflang,
structured-data, redirect, broken-link, rendering, accessibility, performance,
or deployment defect that justifies a separate technical SEO implementation
change today. A stale crawler snapshot is not treated as deployment acceptance
evidence.

The SEO skill submodule remains pinned at
`516e9e2dcf012506a677a749049d64c5914643e9`. The consuming repository still
requires a contract migration before a pointer-only update to a newer upstream
layout, so no skill-submodule movement is part of this run.

## Current focus

1. Complete the `2026-08-31` GPU allocatability Daily Report through one non-draft PR, expected CI, complete final diff/generated-output self-review, squash merge, exact production deployment, bilingual production verification, and one merged-PR closeout comment.
2. Keep **GPU and Heterogeneous Runtime Systems** active. The current report distinguishes retrospective utilization from candidate-conditioned admission and forward-progress guarantees.
3. Keep the new report distinct from the August 20 launch-latency/host-device-causality reports, August 29 memory-placement report, and August 30 instrumentation non-interference report.
4. After this publication, use the restored `7 / 0 / 3` rolling mix as the next-run baseline rather than forcing another adjacent report.
5. Recheck Drive freshness every run and never infer missing Search Console rows as zero.
6. Use date-by-page or date-by-query evidence before attributing aggregate search movement to a page, title, or topic family.
7. Treat finalized GA4 weekly movement as aggregate behavioral evidence, not a causal content or SEO result.
8. Add Cloudflare evidence only when a supported read-only path is enabled in repository configuration.

Detailed history belongs in `.github/seo-data/daily/` and merged daily pull requests.
