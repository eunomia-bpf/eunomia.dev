# SEO status

## Current state

- Authoritative task: `DAILY_TASK.md`
- Technical SEO subtask: `.github/seo-data/daily-task.md`
- Daily Report subtask: `.agents/skills/eunomia-research-report/SKILL.md`
- External daily scheduler: configured and enabled
- Last verified data window: Search Console rows through `2026-08-15`; GA4 weekly organic landing-page files through `2026-08-16`
- Latest daily record: `2026-08-21`
- Last completed Daily Report pull request before the current run: `#165`
- Last verified production publication from a Daily Report run: static export commit `f57e7d96296e36b98aadbe7f6d54e5b88d550e6e` for squash commit `08c1770391e952defacde8e3d4da0aff50c77c8e`
- Current daily branch: `daily/2026-08-21-ebpf-resource-semantics`
- Skill submodule commit: `516e9e2dcf012506a677a749049d64c5914643e9`

PR `#165` is now closed out. It squash-merged as
`08c1770391e952defacde8e3d4da0aff50c77c8e`; the production `new` branch contains
static export commit `f57e7d96296e36b98aadbe7f6d54e5b88d550e6e` with exact message `deploy static
app for 08c1770391e952defacde8e3d4da0aff50c77c8e`. The deployed English and Chinese
GPU host-device causality pages contain locale-correct canonical URLs, reciprocal
`en`/`zh` language alternates plus `x-default`, Article JSON-LD, and Daily Report
navigation. One compact closeout comment records the verification.

The stale unmerged application-resource PR `#166` was based on pre-`#165` state,
used obsolete rolling-mix arithmetic, and was closed rather than reused. The
current run starts from the latest default branch.

## Current Daily Report mix

Immediately before the `2026-08-21` report, the actually published newest ten
reports are:

- eBPF-centered: **7 of 10**
- pure Agent-centered: **0 of 10**
- adjacent systems: **3 of 10**

The earlier operating files were stale because they did not yet include the
August 20 host-device causality report. Today's
`/research/ebpf-application-resource-semantics/` report is genuinely
eBPF-centered: its central property is dynamic no-rebuild semantic instrumentation
plus independent cross-layer runtime validation using eBPF. Publishing it ages an
eBPF report out of the rolling ten, so the new rolling window remains **7 eBPF /
0 pure Agent / 3 adjacent** and stays within the 5–7 rule.

## Current signals

- Repository-generated public-safe operating brief: refreshed `2026-08-21 08:02 UTC`
- Homepage: HTTP 200 in **140 ms**
- `robots.txt`: HTTP 200; sitemap: HTTP 200
- Sitemap entries observed: **682**
- Canonical homepage: `https://eunomia.dev/`
- Public GitHub portfolio snapshot: 98 active non-fork repositories, 9,786 stars, 1,281 forks, and 271 open issue/PR records
- DEV publication surface: 60 articles, 43 public reactions, and 4 comments
- Public web and primary-source evidence: available
- Google Analytics 4: finalized weekly organic landing-page exports through `2026-08-16`
- Google Search Console: weekly export set through the week ending `2026-08-16`; finalized date rows currently end on `2026-08-15`
- Newer weekly Google set beginning `2026-08-17`: not observed in the configured Drive folder on `2026-08-21`
- Cloudflare: disabled by repository configuration

For the valid equal-duration Search Console comparison, `2026-08-10` through
`2026-08-15` reports **393 clicks / 66,510 impressions**, weighted CTR about
**0.591%**, and impression-weighted average position about **9.91**. The
comparable `2026-08-03` through `2026-08-08` slice reports **478 clicks / 69,053
impressions**, **0.692%** CTR, and position about **9.42**. Clicks are about
**17.8% lower**, impressions about **3.7% lower**, CTR about **0.101 percentage
points lower**, and average position about **0.49 positions worse**. The older
slice includes the unusual `2026-08-05` impression spike, so the movement remains
monitored rather than attributed to one page, query, or report.

A complete latest-seven-days versus previous-seven-days GSC comparison remains
unavailable because the `2026-08-09` date row is absent. The required 28-day
comparison is also unavailable because verified export history is too short.

The finalized GA4 organic landing-page export for `2026-08-10` through
`2026-08-16` contains **970 sessions** at about **44.95%** session-weighted
engagement, versus **991 sessions** at about **44.90%** for `2026-08-03` through
`2026-08-09`. Sessions are about **2.1% lower** while engagement is effectively
flat. `(not set)` is **134 versus 116 sessions**; excluding it, sessions are
**836 versus 875**.

## Current technical baseline

The repository generates sitemap, robots, canonical, `hreflang`, Open Graph,
structured data, legacy redirect stubs, and static audit artifacts. Production
deploys through `Deploy Static App`.

The August 21 public-safe brief, current repository output, and available Google
evidence do not establish a new crawl, canonical, hreflang, structured-data,
redirect, broken-link, rendering, accessibility, performance, or deployment
defect. The appropriate public change is therefore the mandatory bilingual Daily
Report and its directly coupled index/series updates, not a speculative technical
SEO patch.

Today's report uses the OSDI 2026 application-defined-resource profiler as the
main research comparison and current Linux `user_events`, uprobe/USDT, libbpf
multi-uprobe cookies, BPF maps, and ring-buffer semantics as implementation
evidence. It develops a versioned resource-semantics manifest, runtime stale-model
validation with explicit confidence loss, and a software-mutation benchmark that
separates semantic correctness from final diagnosis.

The SEO skill submodule remains pinned at
`516e9e2dcf012506a677a749049d64c5914643e9`. Upstream is newer, but the consuming
repository still names interfaces from the pinned layout. A pointer-only update is
not mixed into this daily delivery.

## Current focus

1. Complete the `2026-08-21` eBPF application-resource semantics Daily Report through expected CI, complete final diff/generated-output self-review, squash merge, exact production deployment, bilingual production verification, and one merged-PR closeout comment.
2. Continue **eBPF Observability and Profiling**. The next eBPF-centered report remains permitted by rolling-window arithmetic; prefer always-on semantic compression after fresh evidence/novelty review.
3. Keep the required complete GSC 7-day and 28-day comparisons unavailable until source history supports them; never treat the missing `2026-08-09` row as zero.
4. Obtain date-by-page or date-by-query evidence before attributing the `2026-08-05` impression spike.
5. Monitor Search Console click/CTR movement across another finalized period before changing search-facing titles, copy, or navigation.
6. Investigate GA4 `(not set)` and remaining legacy `/en/` traffic only with richer source-native evidence.
7. Migrate the consuming SEO contract before updating the skill submodule pointer.
8. Add Cloudflare evidence only when a supported read-only path is enabled in repository configuration.

Detailed history belongs in `.github/seo-data/daily/` and merged daily pull requests.
