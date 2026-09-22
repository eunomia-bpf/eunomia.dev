# Site metadata

## Identity

- Canonical URL: `https://eunomia.dev`
- Site name: `Eunomia`
- Timezone: `America/Los_Angeles`

## Repository

- Default branch: `main`
- Authoritative daily task: `DAILY_TASK.md`
- Automation branch prefix: `daily/`
- Skill submodule path: `.github/seo-skills`
- Skill source: `https://github.com/AutoArchive/seo-skill`
- Allowed skill update branch: `main`

## Daily analysis contract

- Daily analysis required: yes
- Lookback days: 28
- Finalization lag days: 3
- Short comparison: latest complete 7 days versus preceding 7 days
- Long comparison: latest complete 28 days versus the preceding comparable period
- Missing-source treatment: report unavailable, stale, partial, or disabled; never convert missing coverage into zero
- Raw private analytics in Git: prohibited

## Google data

- Google Drive enabled: yes
- Google Drive folder name: `eunomia.dev SEO Weekly CSV`
- GA4 export filename pattern: `*_ga4_*.csv`
- Search Console export filename pattern: `*_gsc_*.csv`
- Verified raw export window: through `2026-09-20`
- Search Console newest observed source row: `2026-09-19`; the `2026-09-20` row is absent
- Latest fully finalized GA4 weekly organic landing-page aggregate: `2026-08-24` through `2026-08-30`
- Newest GA4 weekly organic landing-page aggregate: `2026-09-14` through `2026-09-20`, partial because the frozen export was generated while lagged dates were present and has no date dimension
- Expected refresh cadence: weekly; verify freshness and coverage on every run

The configured folder was directly reverified on `2026-09-22`. The newest source family is `2026-09-14..09-20`. Its Search Console date export contains rows for `2026-09-14..09-19` and no `2026-09-20` row. Under the configured three-day finalization lag, all six observed rows are treated as finalized.

The finalized six-day `2026-09-14..19` slice contains **391 clicks / 55,086 impressions / ~0.710% aggregate CTR / ~6.79 impression-weighted average position**. The equal-duration finalized `2026-09-07..12` slice contains **376 / 55,036 / ~0.683% / ~6.46**. Relative to that slice, clicks are about **4.0% higher**, impressions about **0.1% higher**, CTR about **0.027 percentage points higher**, and weighted average position about **0.33 positions worse**.

This is an equal-duration six-day comparison, not a complete seven-day trend. The newest weekly sets omit their final Sunday rows, and older history contains recorded gaps, so complete current seven-day and 28-day comparable-period claims remain unavailable. Missing rows are never synthesized as zero.

The newest weekly GSC page aggregate contains Daily Report routes at **12 clicks / 2,926 impressions** across 74 matching rows, versus **11 / 2,932** across 59 matching rows in the preceding weekly page export. The page set grew and the export has no date dimension, so this is prioritization evidence only, not causal evidence for a metadata or navigation change.

The GA4 `2026-09-14..20` organic landing-page aggregate contains **935 sessions** at about **45.13% session-weighted engagement**. It remains partial because it was generated while dates inside the configured lag were present and has no date dimension for safe finalized subsetting. The `2026-09-07..13` aggregate contains **880 sessions** at about **43.52% engagement**, and `2026-08-31..09-06` contains **913 sessions** at about **47.54% engagement**; both remain partial for the same reason. The latest fully finalized weekly aggregate remains `2026-08-24..30` at **1,007 sessions** and about **45.88% engagement**.

Public repository and live-site evidence supplement these exports but do not replace their source-native meanings.

## Cloudflare data

- Cloudflare enabled: no
- Zone hostname: `eunomia.dev`
- Preferred dataset: `httpRequestsAdaptiveGroups`

## Public and repository data

- Live-site technical collection enabled: yes
- Public GitHub repository evidence enabled: yes
- Public web and primary-source evidence enabled: yes

The public-safe data brief generated on `2026-09-22 12:42 UTC` reports the homepage, `robots.txt`, and sitemap at HTTP 200, **786 sitemap entries**, canonical `https://eunomia.dev/`, **99 active non-fork repositories**, **10,048 stars**, **1,316 forks**, **304 open issue/PR records**, and **63 DEV articles**. These are contextual public observations, not a blended SEO score.

The production robots file allows crawling and points to `https://eunomia.dev/sitemap.xml`. Exact static output from the production `new` branch remains the strongest publication-inspection surface when an independent public crawler has not yet refreshed a newly deployed Daily Report route. Crawler discovery is supplementary retrievability evidence and is never substituted for exact-squash deployment verification.

## Deployment

- Provider: `github-actions`
- Production workflow: `Deploy Static App`
- Production environment: `github-pages`
- Verification URL: `https://eunomia.dev/`

Store only durable public metadata here.