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
- Missing-source treatment: report unavailable, stale, partial, or disabled; never
  convert missing coverage into zero
- Raw private analytics in Git: prohibited

## Google data

- Google Drive enabled: yes
- Google Drive folder name: `eunomia.dev SEO Weekly CSV`
- GA4 export filename pattern: `*_ga4_*.csv`
- Search Console export filename pattern: `*_gsc_*.csv`
- Verified raw export window: `2026-07-27` through `2026-08-23`
- Search Console finalized rows: usable through `2026-08-21` under the three-day lag; the `2026-08-22` row is present but still inside the lag and `2026-08-23` is absent
- GA4 finalized aggregate: `2026-08-10` through `2026-08-16`; the `2026-08-17` through `2026-08-23` aggregate is present but cannot be treated as finalized because it contains lagged days and has no date dimension for trimming
- Expected refresh cadence: weekly; verify freshness and coverage on every run

The Drive folder is discovered by its configured name. Do not store its folder ID,
property IDs, account identifiers, credentials, or private URLs in Git.

## Cloudflare data

- Cloudflare enabled: no
- Zone hostname: `eunomia.dev`
- Preferred dataset: `httpRequestsAdaptiveGroups`

## Public and repository data

- Live-site technical collection enabled: yes
- Public GitHub repository evidence enabled: yes
- Public web and primary-source evidence enabled: yes

The configured Drive folder now contains weekly Google exports for `2026-07-27`
through `2026-08-02`, `2026-08-03` through `2026-08-09`, `2026-08-10` through
`2026-08-16`, and `2026-08-17` through `2026-08-23`. This newer set was first
observed by the daily operation on `2026-08-25`.

For Search Console, the configured three-day lag makes `2026-08-21` the newest
usable date in the latest set. A valid equal-duration finalized comparison is
therefore `2026-08-17` through `2026-08-21` versus `2026-08-10` through
`2026-08-14`. The newer five-day slice reports 443 clicks / 52,433 impressions,
about 0.845% CTR, and impression-weighted position about 9.24; the prior slice
reports 360 / 60,080, about 0.599%, and position about 9.86. Clicks are about
23.1% higher while impressions are about 12.7% lower; CTR is about 0.246
percentage points higher and average position improves by about 0.63 positions.

A complete latest-seven-days versus previous-seven-days GSC comparison remains
unavailable. The latest finalized seven-day window would require comparison
against a predecessor containing the still-missing `2026-08-09` row. The 28-day
comparison also remains unavailable because verified source history is too short.
Missing rows are not converted to zero.

The latest GA4 `2026-08-17` through `2026-08-23` landing-page export contains no
date dimension. Because part of that aggregate is still inside the configured
three-day lag, the whole file is treated as partial for finalized comparison.
The latest usable finalized GA4 weekly aggregate therefore remains `2026-08-10`
through `2026-08-16` with 970 organic landing-page sessions at about 44.95%
session-weighted engagement, versus 991 at about 44.90% for `2026-08-03` through
`2026-08-09`. Public repository and live-site data supplement these exports but
do not replace their source-native meanings.

## Deployment

- Provider: `github-actions`
- Production workflow: `Deploy Static App`
- Production environment: `github-pages`
- Verification URL: `https://eunomia.dev/`

Store only durable public metadata here. Never add property IDs, Drive IDs,
Cloudflare IDs, account identifiers, personal emails, credentials, or private
URLs.
