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
- Current daily branch: `daily/2026-08-31-gpu-allocatability`
- Skill submodule commit: `516e9e2dcf012506a677a749049d64c5914643e9`

PR `#181` is fully closed out. It squash-merged as
`3ede202d350a7d6f728d7b7e7187a3b1abba8145`; exact-merge `Validate SEO
Operations` run `33322348434` and `Deploy Static App` run `33322348476` passed.
Production published static export commit
`60b41bda401c36e6f595bf012de5eded130ba5f5`, whose commit message binds the
export to that squash SHA. Exact EN/ZH generated production artifacts for
`/research/gpu-instrumentation-safety-contract/` were verified, and PR `#181`
has exactly one closeout comment.

## Current Daily Report mix

After the `2026-08-30` adjacent GPU instrumentation-safety report, the newest ten
published reports remain:

- eBPF-centered: **8 of 10**
- pure Agent-centered: **0 of 10**
- adjacent systems: **2 of 10**

This is one eBPF report above the repository target band. The current run selects
another genuinely adjacent GPU/runtime question: whether headline utilization is
enough evidence that a specific incoming workload can safely co-reside. If this
report is published, the `2026-08-21` eBPF report rotates out and the newest-ten
mix becomes **7 / 0 / 3**, restoring the target band without relabeling any
report or publishing an extra entry.

**GPU and Heterogeneous Runtime Systems** remains the active normal series. The
`2026-08-29` report studies evidence for memory placement under oversubscription;
the `2026-08-30` report studies observer-induced semantic and resource
perturbation from dynamic instrumentation. The current report advances a third,
distinct invariant: separate hard resource fit from soft interference and
measurement uncertainty before admitting another workload. NVIDIA DCGM, CUDA
execution-context resource semantics, ROCm occupancy/resource allocation, SIRIUS,
KRYPTON, AntMan, and the recent Roomie preprint provide the main evidence. eBPF is
not required by the central mechanism, so the report is **adjacent systems**.

## Current signals

- Google Drive configured evidence: directly reverified `2026-08-31`; the newest source-native weekly set remains `2026-08-17..23`
- Fresh automated site brief generated `2026-08-31 15:13 UTC`: homepage `200` in `445 ms`, robots `200`, sitemap `200`, and 718 sitemap entries observed
- Public homepage: live fetch on `2026-08-31` shows expected Eunomia identity and Daily Report navigation
- Canonical homepage: `https://eunomia.dev/`
- Public GitHub repository evidence: available; the fresh public-safe brief reports 99 active non-fork repositories, 9,860 stars, 1,289 forks, and 283 open issue/PR records
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
percentage points higher**. The weekly aggregate has no date dimension, so it
cannot support daily or within-week causal attribution.

Source-native page/query movement provides useful prioritization evidence but not
causal proof. In the latest export versus the previous weekly export,
`/tutorials/1-helloworld/` rises from 15 to 25 Search Console clicks while its
average position worsens, and the Chinese GPU-architecture page falls from 21 to
11 clicks. The `ebpf` query rises from 1 to 4 clicks while impressions fall from
166 to 129. These mixed movements are not evidence for a title, copy, canonical,
or navigation change without date-by-page or date-by-query support.

## Current technical baseline

The repository generates sitemap, robots, canonical, `hreflang`, Open Graph,
structured data, legacy redirect stubs, and static audit artifacts. Production
deploys through `Deploy Static App`.

Fresh live-site inspection, the `2026-08-31` automated site brief, and current
Google evidence do not establish a crawl, robots, sitemap, canonical, hreflang,
structured-data, redirect, broken-link, rendering, accessibility, performance,
or deployment defect that justifies a separate technical SEO implementation
change today. Homepage response time rose from 158 ms in the prior daily brief to
445 ms today, but one synthetic observation is not sufficient evidence of a
persistent performance regression. The public crawler can lag production and is
not treated as deployment acceptance evidence.

The SEO skill submodule remains pinned at
`516e9e2dcf012506a677a749049d64c5914643e9`. The consuming repository still
requires a contract migration before a pointer-only update to a newer upstream
layout, so no skill-submodule movement is part of this run.

## Current focus

1. Complete the `2026-08-31` GPU utilization-versus-allocatability Daily Report through one non-draft PR, expected CI, complete final diff/generated-output self-review, squash merge, exact production deployment, bilingual production verification, and one merged-PR closeout comment.
2. Keep **GPU and Heterogeneous Runtime Systems** active. If the current report publishes, it becomes the third substantial post-activation contribution and restores the rolling newest-ten mix to `7 / 0 / 3`.
3. Keep the current report distinct from August 20 launch-latency/causality, August 29 memory placement, and August 30 instrumentation non-interference. Its mechanisms are an allocatability certificate, two-stage hard-fit/interference admission with bounded probing, and a counterexample benchmark for spare-capacity claims.
4. For the next normal GPU-series report, prefer a fourth distinct runtime invariant such as distributed GPU coordination with an explicit correctness property or another device/runtime boundary. Do not repeat utilization, memory placement, or instrumentation safety with different tools.
5. Recheck Drive freshness every run; no weekly set newer than `2026-08-17..23` was observed on `2026-08-31`.
6. Keep complete GSC 7-day and 28-day comparisons unavailable until source history supports them; never fill missing date rows with zero.
7. Use date-by-page or date-by-query evidence before attributing aggregate search movement to a page, title, or topic family.
8. Treat finalized GA4 weekly movement as aggregate behavioral evidence, not a causal content or SEO result.
9. Migrate the consuming SEO contract before updating the `seo-skills` submodule pointer.
10. Add Cloudflare evidence only when a supported read-only path is enabled in repository configuration.

Detailed history belongs in `.github/seo-data/daily/` and merged daily pull requests.
