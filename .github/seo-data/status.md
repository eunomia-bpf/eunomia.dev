# SEO status

## Current state

- Authoritative task: `DAILY_TASK.md`
- Technical SEO subtask: `.github/seo-data/daily-task.md`
- Daily Report subtask: `.agents/skills/eunomia-research-report/SKILL.md`
- External daily scheduler: configured and enabled
- Scheduler timing: daily, flexible around 08:00 in `America/Los_Angeles`
- Last completed daily run: `2026-08-19`
- Last verified data window: Search Console rows through `2026-08-15`; GA4 weekly organic landing-page files through `2026-08-16`
- Latest daily record: `2026-08-20`
- Last completed public-change pull request: `#163`
- Last verified production publication from a daily run: static export commit `a0ab9aed6af348eb8d85634339f407eca2122616` for squash commit `38d50fd8584293490cf845f2c4ff710c368dbe88`
- Current daily branch: `daily/2026-08-20-gpu-launch-latency`
- Skill submodule commit: `516e9e2dcf012506a677a749049d64c5914643e9`

PR `#163` is fully closed out. It squash-merged as
`38d50fd8584293490cf845f2c4ff710c368dbe88`; final PR-head `Deploy Static App`
run `32274536861` and `Validate SEO Operations` run `32274536879` succeeded. The
production `new` branch contains static export commit
`a0ab9aed6af348eb8d85634339f407eca2122616` with exact message `deploy static
app for 38d50fd8584293490cf845f2c4ff710c368dbe88`. Its generated English and
Chinese profiler-sampling pages contain the correct titles, canonical URLs,
reciprocal language alternates plus `x-default`, Article JSON-LD, Daily Report
navigation, and report content. The merged PR has one compact closeout comment.

## Current Daily Report mix

Before today's publication, the completed archive contains ten reports:

- eBPF-centered: **7 of 10**
- pure Agent-centered: **2 of 10**
- adjacent systems: **1 of 10**

Today's GPU launch-latency report is a genuine adjacent-systems report. Its
central mechanisms are CUDA/CUPTI launch-state attribution, cross-domain launch
lineage, dependency uncertainty, and a ground-truth launch-delay benchmark.
eBPF can contribute host scheduling evidence on Linux but is not required for
the mechanism, so the report is not classified eBPF-centered.

After publication, the rolling ten-report window becomes **7 eBPF / 1 pure Agent
/ 2 adjacent**. The next run must remain non-eBPF because adding an eBPF report
while the oldest remaining report is the pure-Agent parallel-effect report would
raise the eBPF count to 8 of 10. Keep the active eBPF Observability and Profiling
series but defer its eBPF-essential questions until the rolling window permits
another eBPF-centered publication.

## Current signals

- Repository-generated public-safe operating brief: refreshed `2026-08-20 08:02 UTC`
- Homepage: HTTP 200 in **189 ms** in the current operating brief
- `robots.txt`: HTTP 200; sitemap: HTTP 200
- Sitemap entries observed: **674**
- Canonical homepage: `https://eunomia.dev/`
- Public GitHub portfolio snapshot: 97 active non-fork repositories, 9,775 stars, 1,280 forks, and 284 open issue/PR records
- DEV publication surface: 60 articles, 43 public reactions, and 4 comments
- Public web and primary-source evidence: available
- Google Analytics 4: weekly organic landing-page exports available through `2026-08-16`; the newest complete aggregate is finalized under the three-day lag
- Google Search Console: weekly export sets available through the week ending `2026-08-16`; finalized date rows currently end on `2026-08-15`
- Newer weekly Google set beginning `2026-08-17`: not observed in the configured Drive folder on `2026-08-20`
- Cloudflare: disabled by repository configuration

For the valid equal-duration Search Console comparison, `2026-08-10` through
`2026-08-15` reports **393 clicks / 66,510 impressions**, weighted CTR about
**0.591%**, and impression-weighted average position about **9.91**. The
comparable `2026-08-03` through `2026-08-08` slice reports **478 clicks / 69,053
impressions**, **0.692%** CTR, and position about **9.42**. Clicks are about
**17.8% lower**, impressions about **3.7% lower**, CTR about **0.101 percentage
points lower**, and average position about **0.49 positions worse**. The older
slice includes the unusual `2026-08-05` impression spike, so the movement remains
monitored rather than attributed to one page, query, or Daily Report.

A complete latest-seven-days versus previous-seven-days GSC comparison remains
unavailable because the `2026-08-09` date row is absent across the adjacent
weekly files. The required 28-day comparison is also unavailable rather than
inferred.

The finalized GA4 organic landing-page export for `2026-08-10` through
`2026-08-16` contains **970 sessions** at about **44.95%** session-weighted
engagement, versus **991 sessions** at about **44.90%** for `2026-08-03` through
`2026-08-09`. Sessions are about **2.1% lower** while engagement is effectively
flat. `(not set)` is **134 versus 116 sessions**; excluding it, sessions are
**836 versus 875**. These are source-native acquisition and measurement signals,
not evidence for a speculative title, copy, navigation, or site-structure
change.

## Current technical baseline

The repository generates sitemap, robots, canonical, `hreflang`, Open Graph,
structured data, legacy redirect stubs, and HTTP audit artifacts. Production
deploys through `Deploy Static App`.

The `2026-08-20` public-safe operating brief and current repository evidence do
not establish a new crawl, canonical, hreflang, structured-data, redirect,
broken-link, rendering, accessibility, performance, or deployment defect. The
appropriate public site change today is therefore the mandatory bilingual Daily
Report and directly coupled report-index/series updates, not an unrelated
technical SEO patch.

Today's research asks whether a CUDA kernel was slow or merely started late.
Current CUPTI exposes API timing and correlation plus optional kernel `queued`
and `submitted` latency timestamps; Nsight Systems exposes API/queue/kernel
phases but documents that its queue interval is an approximation. The report
therefore treats timestamps as evidence about states rather than automatic proof
of one cause. It develops an explicit launch-state ledger with unknown states,
a launch identity that survives host handoffs and CUDA Graph replay, and a
controlled benchmark with independently injected delay causes.

The SEO skill submodule remains pinned at
`516e9e2dcf012506a677a749049d64c5914643e9`. Upstream remains at
`f42128a3f05c73cf10c786a2711c488bb3a14839`, while the consuming repository still
names interfaces from the pinned layout. A pointer-only update is therefore not
mixed into this daily delivery.

## Current focus

1. Complete the `2026-08-20` GPU launch-latency Daily Report through CI, complete diff/generated-output self-review, squash merge, exact production deployment, bilingual production verification, and one merged-PR closeout comment.
2. Keep the next publication non-eBPF while the rolling window remains at the seven-report eBPF ceiling; select a genuine adjacent-systems or unusually strong pure-Agent systems question rather than relabeling an eBPF mechanism.
3. Keep the required complete GSC 7-day and 28-day comparisons unavailable until source history supports them; never treat the missing `2026-08-09` row as zero.
4. Obtain date-by-page or date-by-query evidence before attributing the `2026-08-05` impression spike.
5. Monitor Search Console click/CTR movement across another finalized period before changing search-facing titles, copy, or navigation.
6. Investigate GA4 `(not set)` and remaining legacy `/en/` traffic only with richer source-native evidence.
7. Migrate the consuming SEO contract before updating the skill submodule pointer.
8. Add Cloudflare evidence only when a supported read-only path is enabled in repository configuration.

Detailed history belongs in `.github/seo-data/daily/` and the merged daily pull requests.
