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
- Search Console known latest row: `2026-08-22`, now outside the three-day lag; `2026-08-23` is absent
- GA4 finalized aggregate: `2026-08-17` through `2026-08-23`
- Expected refresh cadence: weekly; verify freshness and coverage on every run

The configured folder was last verified on `2026-08-25` to contain weekly Google exports for `2026-07-27..08-02`, `2026-08-03..09`, `2026-08-10..16`, and `2026-08-17..23`. Direct Drive search in the `2026-08-26` runtime returned no accessible matching files, so no newer set is inferred and no known row is converted to zero.

For Search Console, the verified newest source-native file contains date rows through `2026-08-22`; `2026-08-23` is absent. The `2026-08-22` row is now outside the finalization lag, but its source-native values were not retrievable in this runtime. The last calculated equal-duration finalized comparison remains `2026-08-17..21` versus `2026-08-10..14`: 443 clicks / 52,433 impressions, about 0.845% CTR, and impression-weighted position about 9.24 versus 360 / 60,080, about 0.599%, and position about 9.86. Clicks are about 23.1% higher, impressions about 12.7% lower, CTR about 0.246 percentage points higher, and average position improves by about 0.63 positions.

A complete latest-seven-days versus previous-seven-days GSC comparison remains unavailable because the latest weekly set is missing `2026-08-23` and the predecessor history contains the still-missing `2026-08-09` row. The 28-day comparison also remains unavailable because verified source history is too short. Missing rows are not converted to zero.

The GA4 `2026-08-17..23` landing-page aggregate is now fully outside the configured three-day lag and is finalized as a source-native weekly aggregate. It contains 984 organic landing-page sessions at about 49.29% session-weighted engagement, including 118 `(not set)` sessions, versus 970 sessions at about 44.95% engagement for `2026-08-10..16`. Sessions are about 1.4% higher and engagement about 4.34 percentage points higher. These weekly aggregates have no date dimension, so they do not support daily or within-week causal attribution.

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
