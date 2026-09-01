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
- Verified raw export window: `2026-07-27` through `2026-08-30`
- Search Console newest verified row: `2026-08-29`; finalized rows are used through `2026-08-28`; `2026-08-30` is absent
- Latest fully finalized GA4 aggregate: `2026-08-17` through `2026-08-23`; a newer `2026-08-24..30` aggregate exists but remains partial under the lag
- Expected refresh cadence: weekly; verify freshness and coverage on every run

The configured folder was directly reverified on `2026-09-01`. It contains weekly Google export sets for `2026-07-27..08-02`, `2026-08-03..09`, `2026-08-10..16`, `2026-08-17..23`, and `2026-08-24..30`. The newest set is present as source-native export material; no later weekly set was observed. Missing rows are not converted to zero.

For Search Console, the newest source-native date export contains rows for `2026-08-24..29`; `2026-08-30` is absent. Under the configured three-day lag, rows through `2026-08-28` are treated as finalized while `2026-08-29` remains provisional. The finalized `2026-08-24..28` slice contains 398 clicks / 48,044 impressions, about 0.828% aggregate CTR, and impression-weighted average position about 10.04. This is recorded as an absolute current-window observation rather than forced into a like-for-like trend comparison because the required source history is not contiguous.

The previously established longest equal-duration contiguous finalized comparison remains `2026-08-17..22` versus `2026-08-10..15`: 477 clicks / 59,798 impressions / about 0.798% CTR / about 9.56 impression-weighted position versus 393 / 66,510 / about 0.591% / about 9.91. Clicks are about 21.4% higher, impressions about 10.1% lower, CTR about 0.207 percentage points higher, and average position improves by about 0.35 positions.

A complete latest-seven-days versus previous-seven-days GSC comparison remains unavailable because verified source rows are not contiguous across the required windows: earlier exports omit `2026-08-16` and `2026-08-23`, while the newest set omits `2026-08-30`. The 28-day comparison also remains unavailable because there is no complete preceding 28-day source window. Missing rows are not converted to zero.

The new GA4 `2026-08-24..30` organic landing-page aggregate currently contains 1,007 sessions at about 45.88% session-weighted engagement. Because the interval includes dates inside the configured finalization lag, it is treated only as a partial early signal. The latest fully finalized aggregate remains `2026-08-17..23`: 984 organic landing-page sessions at about 49.29% session-weighted engagement, including 118 `(not set)` sessions, versus 970 sessions at about 44.95% engagement for `2026-08-10..16`. These weekly aggregates have no date dimension, so they do not support daily or within-week causal attribution.

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
