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
- Verified raw export window: `2026-07-27` through `2026-08-23`
- Search Console verified latest finalized row: `2026-08-22`; `2026-08-23` is absent
- GA4 finalized aggregate: `2026-08-17` through `2026-08-23`
- Expected refresh cadence: weekly; verify freshness and coverage on every run

The configured folder was directly reverified on `2026-08-28`. It contains weekly Google exports for `2026-07-27..08-02`, `2026-08-03..09`, `2026-08-10..16`, and `2026-08-17..23`; no newer weekly set was observed. Missing rows are not converted to zero.

For Search Console, the newest source-native date export contains rows through `2026-08-22`; `2026-08-23` is absent. Under the configured three-day lag, `2026-08-22` is finalized. The longest equal-duration finalized comparison that the available contiguous rows support remains six days: `2026-08-17..22` versus `2026-08-10..15`. The newer slice reports 477 clicks / 59,798 impressions, about 0.798% CTR, and impression-weighted position about 9.56; the prior slice reports 393 / 66,510, about 0.591%, and position about 9.91. Clicks are about 21.4% higher while impressions are about 10.1% lower; CTR is about 0.207 percentage points higher and average position improves by about 0.35 positions.

A complete latest-seven-days versus previous-seven-days GSC comparison remains unavailable. The latest finalized seven-day window would be `2026-08-16..22` versus `2026-08-09..15`, but the available date exports omit both `2026-08-16` and `2026-08-09`. The 28-day comparison also remains unavailable because verified source history is too short. Missing rows are not converted to zero.

The GA4 `2026-08-17..23` landing-page aggregate is fully outside the configured three-day lag and is finalized as a source-native weekly aggregate. It contains 984 organic landing-page sessions at about 49.29% session-weighted engagement, including 118 `(not set)` sessions, versus 970 sessions at about 44.95% engagement for `2026-08-10..16`. Sessions are about 1.4% higher and engagement about 4.34 percentage points higher. These weekly aggregates have no date dimension, so they do not support daily or within-week causal attribution.

Public repository and live-site data supplement these exports but do not replace their source-native meanings.

## Cloudflare data

- Cloudflare enabled: no
- Zone hostname: `eunomia.dev`
- Preferred dataset: `httpRequestsAdaptiveGroups`

## Public and repository data

- Live-site technical collection enabled: yes
- Public GitHub repository evidence enabled: yes
- Public web and primary-source evidence enabled: yes

## Deployment

- Provider: `github-actions`
- Production workflow: `Deploy Static App`
- Production environment: `github-pages`
- Verification URL: `https://eunomia.dev/`

Store only durable public metadata here.