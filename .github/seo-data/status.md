# SEO status

## Current state

- Authoritative task: `DAILY_TASK.md`
- Technical SEO subtask: `.github/seo-data/daily-task.md`
- Daily Report subtask: `.agents/skills/eunomia-research-report/SKILL.md`
- External daily scheduler: configured and enabled
- Scheduler timing: daily, flexible around 08:00 in `America/Los_Angeles`
- Last completed daily run: `2026-08-19`
- Last verified data window: Search Console rows through `2026-08-15`; GA4 weekly organic landing-page files through `2026-08-16`, with the newest weekly aggregate finalized under the three-day lag
- Latest daily record: `2026-08-20`
- Last completed public-change pull request: `#163`
- Last verified production publication from a daily run: static export commit `a0ab9aed6af348eb8d85634339f407eca2122616` for squash commit `38d50fd8584293490cf845f2c4ff710c368dbe88`
- Current daily branch: `daily/2026-08-20-application-resource-profiling`
- Skill submodule commit: `516e9e2dcf012506a677a749049d64c5914643e9`

PR `#163` is fully closed out. It squash-merged as
`38d50fd8584293490cf845f2c4ff710c368dbe88`; final PR-head `Deploy Static App`
run `32274536861` and `Validate SEO Operations` run `32274536879` succeeded. The
production branch contains static-export commit
`a0ab9aed6af348eb8d85634339f407eca2122616` with the exact deployment message for
the squash commit, and the generated English and Chinese profiler-sampling pages
were verified for report content, canonical URLs, reciprocal language alternates
plus `x-default`, Article JSON-LD, and Daily Report navigation. The required
closeout comment is present on the merged PR.

## Current Daily Report mix

Before the `2026-08-20` publication, the rolling ten-report window is:

- eBPF-centered: **7 of 10**
- pure Agent-centered: **2 of 10**
- adjacent systems: **1 of 10**

The new application-defined-resource profiling report is a genuine adjacent
systems report. Its central mechanism is a collector-neutral resource-semantics
contract, stale-model validation, and a cross-instrumentation benchmark. eBPF and
uprobes can provide evidence but are not essential to the mechanism. After
publication the window becomes **7 eBPF / 1 pure Agent / 2 adjacent out of 10**.

The next run must remain non-eBPF because the oldest report that would leave the
window is still the remaining pure-Agent report. Another eBPF report would push
the window to 8 eBPF reports and violate the configured 5–7 range.

## Current signals

- Repository-generated public-safe operating brief: refreshed `2026-08-20 08:02 UTC`
- Homepage: HTTP 200 in **189 ms** in the current operating brief
- `robots.txt`: reachable; sitemap: reachable
- Sitemap entries observed: **674**
- Canonical homepage: `https://eunomia.dev/`
- Public GitHub portfolio snapshot: 97 active non-fork repositories, 9,775 stars, 1,280 forks, and 284 open issue/PR records
- DEV publication surface: 60 articles, 43 public reactions, and 4 comments
- Public web and primary-source evidence: available
- Google Analytics 4: weekly organic landing-page exports available through `2026-08-16`; newest aggregate finalized
- Google Search Console: weekly export sets available through the week ending `2026-08-16`; finalized date rows currently end on `2026-08-15`
- Cloudflare: disabled by repository configuration

The valid equal-duration Search Console comparison is unchanged. `2026-08-10`
through `2026-08-15` reports **393 clicks / 66,510 impressions**, weighted CTR
about **0.591%**, and impression-weighted average position about **9.91**. The
comparable `2026-08-03` through `2026-08-08` slice reports **478 clicks / 69,053
impressions**, **0.692%** CTR, and position about **9.42**. Clicks are about
**17.8% lower**, impressions about **3.7% lower**, CTR about **0.101 percentage
points lower**, and average position about **0.49 positions worse**. The older
slice includes the unusual `2026-08-05` impression spike, so movement remains
monitored rather than attributed to one page, query, or report.

A complete latest-seven-days versus previous-seven-days GSC comparison remains
unavailable because the `2026-08-09` date row is absent. The required 28-day
comparison is also unavailable rather than inferred.

The finalized GA4 organic landing-page export for `2026-08-10` through
`2026-08-16` contains **970 sessions** at about **44.95%** session-weighted
engagement, versus **991 sessions** at about **44.90%** for `2026-08-03` through
`2026-08-09`. `(not set)` is **134 versus 116 sessions**; excluding it, sessions
are **836 versus 875**. No newer weekly export set was present in the configured
Drive folder on `2026-08-20`.

## Current technical baseline

The repository generates sitemap, robots, canonical, `hreflang`, Open Graph,
structured data, legacy redirect stubs, and HTTP audit artifacts. Production
deploys through `Deploy Static App`.

The `2026-08-20` public-safe brief and repository evidence do not establish a new
crawl, canonical, hreflang, structured-data, redirect, broken-link, rendering,
accessibility, performance, or deployment defect. The public web crawler can lag
the current generated Daily Report index, so cached search snapshots are not
used as evidence of a production defect without a fresh direct observation. The
appropriate public change is the mandatory bilingual Daily Report and coupled
report-index/series updates, not an unrelated technical SEO patch.

Today's report asks how profilers can discover the identity, lifetime, units,
capacity, and usage semantics of application-defined resources and detect when
that model becomes stale. Primary evidence includes the OSDI 2026 gigiprofiler
paper and current Linux 6.18 `user_events` and uprobe documentation. The report
develops a portable resource-semantics manifest, runtime confidence degradation,
and a ground-truth benchmark spanning multiple instrumentation strategies.

The SEO skill submodule remains pinned at
`516e9e2dcf012506a677a749049d64c5914643e9`; a pointer-only update is not mixed
into this daily delivery because the consuming contract still names pinned-layout
interfaces.

## Current focus

1. Complete the `2026-08-20` application-resource Daily Report through CI, full diff/generated-output self-review, squash merge, exact production deployment, bilingual production verification, and one merged-PR closeout comment.
2. Keep the next report non-eBPF to preserve the rolling ten-report maximum; prefer an adjacent observability/profiling question close to the active series.
3. Keep the complete GSC 7-day and 28-day comparisons unavailable until source history supports them; never treat the missing `2026-08-09` row as zero.
4. Obtain date-by-page or date-by-query evidence before attributing the `2026-08-05` impression spike.
5. Monitor Search Console click/CTR movement across another finalized period before changing search-facing titles, copy, or navigation.
6. Investigate GA4 `(not set)` and remaining legacy `/en/` traffic only with richer source-native evidence.
7. Migrate the consuming SEO contract before updating the skill submodule pointer.
8. Add Cloudflare evidence only when a supported read-only path is enabled in repository configuration.

Detailed history belongs in `.github/seo-data/daily/` and the merged daily pull requests.
