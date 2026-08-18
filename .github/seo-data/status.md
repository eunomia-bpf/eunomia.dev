# SEO status

## Current state

- Authoritative task: `DAILY_TASK.md`
- Technical SEO subtask: `.github/seo-data/daily-task.md`
- Daily Report subtask: `.agents/skills/eunomia-research-report/SKILL.md`
- External daily scheduler: configured and enabled
- Last completed daily run: `2026-08-17`
- Last verified data window: Search Console rows through `2026-08-15`; GA4 weekly organic landing-page export through `2026-08-16`
- Latest daily record: `2026-08-18`
- Last completed public-change pull request: `#161`
- Last verified production deployment from a daily run: `Deploy Static App` for squash commit `78e27707fc63065afa7fa19a25f363a39fcba20a`
- Current daily branch: `daily/2026-08-18-page-memory-attribution`
- Skill submodule commit: `516e9e2dcf012506a677a749049d64c5914643e9`

PR `#161` is fully closed out. It squash-merged as
`78e27707fc63065afa7fa19a25f363a39fcba20a`; its exact commit passed the
repository validation and production `Deploy Static App` workflow, and the
merged-PR closeout records bilingual production verification. The published
heterogeneous-placement report completed the six-report **eBPF Runtime,
Extensibility, and Composition** sequence.

## Current Daily Report mix

Before the current change, the completed archive contains eight reports:

- eBPF-centered: **6 of 8**
- pure Agent-centered: **2 of 8**
- adjacent systems: **0 of 8**

The current page-level memory-attribution report is eBPF-centered. After it
merges, the archive becomes **7 eBPF-centered / 2 pure Agent / 0 adjacent systems
out of 9**. The repository has promoted **eBPF Observability and Profiling** to
the active series; page-level memory attribution is its first report.

## Current signals

- Repository-generated public-safe operating brief: refreshed `2026-08-18 07:59 UTC`
- Homepage: HTTP 200 in the current operating brief, observed in **227 ms**
- `robots.txt`: reachable; sitemap: reachable
- Sitemap entries observed: **668**
- Canonical homepage: `https://eunomia.dev/`
- Public GitHub repository evidence: available; current brief observes 97 active non-fork repositories, 9,764 stars, 1,280 forks, and 282 open issue/PR records across the portfolio snapshot
- DEV publication surface: 60 articles, 43 public reactions, and 4 comments in the current public-safe brief
- Public web and primary-source evidence: available
- Google Analytics 4: weekly organic landing-page exports available through `2026-08-16`
- Google Search Console: weekly export sets available through the week ending `2026-08-16`; finalized date rows currently end on `2026-08-15`
- Cloudflare: disabled by repository configuration

The newest Google weekly export is now available and materially changes the
operating evidence. For the valid equal-duration Search Console comparison,
`2026-08-10` through `2026-08-15` reports **393 clicks / 66,510 impressions**,
weighted CTR about **0.591%**, and impression-weighted average position about
**9.91**. The comparable `2026-08-03` through `2026-08-08` slice reports **478
clicks / 69,053 impressions**, **0.692%** CTR, and position about **9.42**. Clicks
are about **17.8% lower**, impressions about **3.7% lower**, CTR about **0.101
percentage points lower**, and average position about **0.49 positions worse**.
The old comparison includes the unusual `2026-08-05` impression spike, so this
movement is monitored rather than attributed to one page or content change.

A complete latest-seven-days versus previous-seven-days GSC comparison remains
unavailable because the `2026-08-09` date row is absent across the adjacent
weekly files. The required 28-day comparison is also unavailable rather than
inferred.

The GA4 organic landing-page export for `2026-08-10` through `2026-08-16`
contains **970 sessions** at about **44.95%** session-weighted engagement, versus
**991 sessions** at about **44.90%** for `2026-08-03` through `2026-08-09`.
Sessions are about **2.1% lower** while engagement is effectively flat. `(not
set)` increased from **116** to **134 sessions**; excluding it, sessions declined
from **875** to **836**, about **4.5%**. This remains a measurement-quality and
traffic-mix signal, not sufficient evidence for a speculative title, copy, or
site-structure change.

## Current technical baseline

The repository generates sitemap, robots, canonical, `hreflang`, Open Graph,
structured data, legacy redirect stubs, and HTTP audit artifacts. Production
deploys through `Deploy Static App`.

The August 18 public-safe operating brief and live homepage inspection do not
establish a new crawl, canonical, hreflang, structured-data, redirect,
broken-link, rendering, accessibility, performance, or deployment defect. The
correct public site change for this run is therefore the mandatory new bilingual
Daily Report and directly coupled index/series updates, not an unrelated
technical SEO patch.

Today's research separates allocation intent, residency, working-set evidence,
page lifecycle, and sampled access cost. Linux `/proc`, Idle Page Tracking,
DAMON, page_owner, VM tracepoints, and perf memory sampling each expose a useful
slice but do not provide one stable application-allocation-to-page provenance
chain. The report develops a lifetime-aware provenance ledger, access-weighted
attribution with explicit confidence, and a ground-truth memory-attribution
benchmark.

The SEO skill submodule remains pinned at
`516e9e2dcf012506a677a749049d64c5914643e9`. The consuming repository still
names interfaces from that pinned layout, so no pointer-only bump is included in
the daily change.

## Current focus

1. Complete the `2026-08-18` page-level memory-attribution Daily Report through final CI, complete diff/generated-output self-review, squash merge, exact production deployment, bilingual public verification, and one merged-PR closeout comment.
2. Continue the active **eBPF Observability and Profiling** series with profiler sampling theory unless that candidate fails the evidence or novelty gate.
3. Keep the required complete GSC 7-day and 28-day comparisons unavailable until source history supports them; never treat the missing `2026-08-09` row as zero.
4. Monitor the current click/CTR decline across another finalized period before changing search-facing titles, copy, or navigation.
5. Obtain date-by-page or date-by-query evidence before attributing the `2026-08-05` impression spike.
6. Investigate GA4 `(not set)` and remaining legacy `/en/` traffic only with richer source-native evidence.
7. Migrate the consuming SEO contract before updating the skill submodule pointer.
8. Add Cloudflare evidence only when a supported read-only path is enabled in repository configuration.

Detailed history belongs in `.github/seo-data/daily/` and the merged daily pull requests.
