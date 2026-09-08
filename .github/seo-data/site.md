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
- Verified raw export window: `2026-07-27` through `2026-09-06`
- Search Console newest observed source row: `2026-09-05`; under the configured three-day lag, finalized rows are used through `2026-09-04`; `2026-09-06` is absent
- Latest fully finalized GA4 aggregate: `2026-08-24` through `2026-08-30`
- Newest GA4 aggregate: `2026-08-31` through `2026-09-06`, partial under the configured lag and not used as a finalized week-over-week comparison
- Expected refresh cadence: weekly; verify freshness and coverage on every run

The configured folder was directly reverified on `2026-09-07`. It now contains weekly Google export sets through `2026-08-31..09-06`, in addition to the previously verified sets beginning `2026-07-27..08-02`. Missing rows are not converted to zero.

For Search Console, the newest date export contains rows for `2026-08-31..09-05`; `2026-09-06` is absent. Under the configured three-day lag, the finalized contiguous slice available from the new set is `2026-08-31..09-04`. Those five rows contain **368 clicks / 53,341 impressions / about 0.690% aggregate CTR / about 7.45 impression-weighted average position**.

The equal-duration finalized `2026-08-24..28` slice contains **398 clicks / 48,044 impressions / about 0.828% CTR / about 10.04 weighted position**. Relative to that five-day slice, clicks are about **7.5% lower**, impressions about **11.0% higher**, CTR about **0.139 percentage points lower**, and weighted average position about **2.59 positions better**. This is an equal-duration source-native comparison, not a complete seven-day trend.

A complete latest-seven-days versus previous-seven-days GSC comparison remains unavailable because the required history is not contiguous: the prior weekly set omits `2026-08-30`, and older history also contains the previously recorded `2026-08-23` gap. The same historical gaps prevent the required complete 28-day versus preceding-comparable-period comparison. Missing rows are not converted to zero.

The newest weekly GSC page aggregate contains Daily Report routes at **8 clicks / 1,812 impressions**, compared with **6 / 1,017** in the preceding weekly page export. The page export has no date dimension and the newest weekly set includes dates inside the finalization lag, so this is prioritization evidence only, not causal evidence for a title, topic, navigation, or metadata change.

The GA4 `2026-08-24..30` organic landing-page aggregate remains the latest fully finalized weekly aggregate and contains **1,007 sessions** at about **45.88% session-weighted engagement**. The new `2026-08-31..09-06` aggregate contains **913 sessions** at about **47.54% session-weighted engagement**, but it includes dates inside the configured finalization lag and has no date dimension, so it is explicitly partial and is not used as a finalized week-over-week trend. The preceding finalized `2026-08-17..23` aggregate contains 984 sessions at about 49.29% engagement. Weekly aggregates do not support daily or within-week causal attribution.

Public repository and live-site data supplement these exports but do not replace their source-native meanings.

## Cloudflare data

- Cloudflare enabled: no
- Zone hostname: `eunomia.dev`
- Preferred dataset: `httpRequestsAdaptiveGroups`

## Public and repository data

- Live-site technical collection enabled: yes
- Public GitHub repository evidence enabled: yes
- Public web and primary-source evidence enabled: yes

The production robots file allows crawling and points to `https://eunomia.dev/sitemap.xml`. Exact static output from the production `new` branch remains the strongest publication-inspection surface when an independent public crawler has not yet refreshed a newly deployed Daily Report route. Crawler discovery is supplementary retrievability evidence and is never substituted for exact-squash deployment verification.

## Deployment

- Provider: `github-actions`
- Production workflow: `Deploy Static App`
- Production environment: `github-pages`
- Verification URL: `https://eunomia.dev/`

Store only durable public metadata here.
