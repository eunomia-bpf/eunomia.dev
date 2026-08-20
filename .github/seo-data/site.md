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
- Verified export window: `2026-07-27` through `2026-08-16`; the newest Search Console date export has finalized rows through `2026-08-15`; the newest GA4 weekly aggregate through `2026-08-16` is finalized under the three-day lag on `2026-08-19`
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

Search Console and GA4 exports provide weekly sets for `2026-07-27` through
`2026-08-02`, `2026-08-03` through `2026-08-09`, and `2026-08-10` through
`2026-08-16`; no newer weekly set was observed in the configured Drive folder on
`2026-08-20`. Search Console rows through `2026-08-15` are usable. The
`2026-08-09` Search Console date row is still absent across the adjacent files,
so a complete latest-7-days versus previous-7-days comparison remains
unavailable. A valid equal-duration six-day comparison is `2026-08-10` through
`2026-08-15` versus `2026-08-03` through `2026-08-08`. The 28-day comparison is
also unavailable because the export history is not long enough.

GA4 weekly landing-page exports have no date dimension for trimming a partial or
unfinalized day within a file. On `2026-08-20`, every day in the newest
`2026-08-10` through `2026-08-16` aggregate is outside the configured three-day
finalization lag, so the whole aggregate remains usable as a finalized
source-native weekly comparison against `2026-08-03` through `2026-08-09`.
Public repository and live-site data supplement these exports but do not replace
their source-native meanings.

## Deployment

- Provider: `github-actions`
- Production workflow: `Deploy Static App`
- Production environment: `github-pages`
- Verification URL: `https://eunomia.dev/`

Store only durable public metadata here. Never add property IDs, Drive IDs,
Cloudflare IDs, account identifiers, personal emails, credentials, or private
URLs.
