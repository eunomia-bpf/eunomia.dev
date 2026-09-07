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
- Search Console newest verified row: `2026-09-05`; finalized rows are used through `2026-09-04`; `2026-09-06` is absent
- Latest fully finalized GA4 aggregate: `2026-08-24` through `2026-08-30`
- Newest GA4 aggregate: `2026-08-31` through `2026-09-06`, partial under the configured lag because the export has no date dimension
- Expected refresh cadence: weekly; verify freshness and coverage on every run

The configured folder was directly reverified on `2026-09-07`. It now contains a fresh `2026-08-31..09-06` weekly export set in addition to the earlier weekly sets. Missing rows are never converted to zero.

For Search Console, finalized `2026-08-31..09-04` rows contain **368 clicks / 53,341 impressions / ~0.690% aggregate CTR / ~7.45 impression-weighted average position**. The equal-duration finalized `2026-08-24..28` slice contains **398 / 48,044 / ~0.828% / ~10.04**. Relative to that slice, clicks are about **7.5% lower**, impressions **11.0% higher**, CTR about **0.139 percentage points lower**, and weighted average position about **2.59 positions better**.

This is a five-day source-native comparison, not the configured complete seven-day trend. `2026-08-30` is missing from the previous export and `2026-09-06` is absent from the new one; older gaps also prevent a complete preceding 28-day source window.

The GA4 `2026-08-24..30` organic landing-page aggregate remains the latest fully finalized weekly aggregate: **1,007 sessions** at about **45.88% session-weighted engagement** versus **984 sessions** at about **49.29% engagement** for `2026-08-17..23`. The new `2026-08-31..09-06` aggregate is partial under the lag and is not used as a finalized week-over-week comparison.

Public repository and live-site data supplement these exports but do not replace their source-native meanings.

## Cloudflare data

- Cloudflare enabled: no
- Zone hostname: `eunomia.dev`
- Preferred dataset: `httpRequestsAdaptiveGroups`

## Public and repository data

- Live-site technical collection enabled: yes
- Public GitHub repository evidence enabled: yes
- Public web and primary-source evidence enabled: yes

Public discovery is supplementary retrievability evidence. Exact-SHA deployment and generated production artifacts remain the publication acceptance boundary.

## Deployment

- Provider: `github-actions`
- Production workflow: `Deploy Static App`
- Production environment: `github-pages`
- Verification URL: `https://eunomia.dev/`

Store only durable public metadata here.
