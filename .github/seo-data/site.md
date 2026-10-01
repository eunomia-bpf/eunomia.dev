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
- Verified raw export window: through `2026-09-27`
- Search Console newest observed source row: `2026-09-25`; September 26–27 are absent
- Latest fully finalized GA4 weekly organic landing-page aggregate: `2026-08-24` through `2026-08-30`
- Newest GA4 weekly organic landing-page aggregate: `2026-09-21` through `2026-09-27`, partial because the frozen export was generated before the labelled period was fully outside the lag window and has no date dimension
- Expected refresh cadence: weekly; verify freshness and coverage on every run

The configured folder was directly reverified on `2026-10-01`. The newest source family is `2026-09-21..09-27`. Its Search Console date export contains rows for `2026-09-21..09-25` and no rows for September 26–27. Under the configured three-day finalization lag, the five observed rows are treated as finalized.

The finalized five-day `2026-09-21..25` slice contains **342 clicks / 37,406 impressions / ~0.914% aggregate CTR / ~6.47 impression-weighted average position**. The equal-duration `2026-09-14..18` slice contains **349 / 49,798 / ~0.701% / ~6.82**. Relative to that slice, clicks are about **2.0% lower**, impressions about **24.9% lower**, CTR about **0.213 percentage points higher**, and weighted average position about **0.35 positions better**.

This is an equal-duration five-day source-native comparison, not a complete seven-day trend. The weekly date exports omit their final weekend rows, and older history contains recorded gaps, so complete current seven-day and 28-day comparable-period claims remain unavailable. Missing rows are never synthesized as zero.

The newest weekly GSC page aggregate contains Daily Report routes at **11 clicks / 2,052 impressions** across 82 matching rows, versus **12 / 2,926** across 74 matching rows in the preceding weekly page export. The published report set changed and the page export has no date dimension, so this is prioritization evidence only, not causal evidence for a metadata or navigation change.

The newest GA4 organic landing-page aggregate contains **848 sessions** at **43.75% session-weighted engagement** and remains partial for the source-timing reason above. The preceding `2026-09-14..20` snapshot contains **935 sessions** at about **45.13% engagement** and is also partial. The latest fully finalized weekly aggregate remains `2026-08-24..30` at **1,007 sessions** and about **45.88% engagement**.

Public repository and live-site evidence supplement these exports but do not replace their source-native meanings.

## Cloudflare data

- Cloudflare enabled: no
- Zone hostname: `eunomia.dev`
- Preferred dataset: `httpRequestsAdaptiveGroups`

## Public and repository data

- Live-site technical collection enabled: yes
- Public GitHub repository evidence enabled: yes
- Public web and primary-source evidence enabled: yes

The public-safe data brief generated on `2026-10-01 14:37 UTC` reports homepage, `robots.txt`, and sitemap HTTP 200, **804 sitemap entries**, canonical `https://eunomia.dev/`, **100 active non-fork repositories**, **10,107 stars**, **1,334 forks**, **447 open issue/PR records**, and **63 DEV articles**. These are contextual public observations, not a blended SEO score.

The production robots file allows crawling and points to `https://eunomia.dev/sitemap.xml`. Exact static output from the production `new` branch remains the strongest publication-inspection surface when an independent public crawler has not yet refreshed a newly deployed Daily Report route.

## Deployment

- Provider: `github-actions`
- Production workflow: `Deploy Static App`
- Production environment: `github-pages`
- Verification URL: `https://eunomia.dev/`

Store only durable public metadata here.
