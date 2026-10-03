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

The configured source folder was reverified on `2026-10-03`. The newest source family remains `2026-09-21..09-27`. Its Search Console date export contains finalized observed rows for `2026-09-21..09-25` and no rows for September 26–27. Those five observed days contain **342 clicks / 37,406 impressions / ~0.914% CTR / ~6.47 impression-weighted position**, versus **349 / 49,798 / ~0.701% / ~6.82** for the equal-duration `2026-09-14..18` slice. This is not a complete seven-day trend, and current history does not support a complete 28-day comparable-period claim.

The newest GSC page aggregate contains Daily Report routes at **11 clicks / 2,052 impressions**, versus **12 / 2,926** in the preceding weekly page export. Because the report set changed and the page export has no date dimension, this is prioritization evidence only.

The newest GA4 organic landing-page snapshot contains **848 sessions** at **43.75% session-weighted engagement** and remains partial. The latest fully finalized weekly aggregate remains `2026-08-24..30` at **1,007 sessions** and about **45.88% engagement**.

## Cloudflare data

- Cloudflare enabled: no
- Zone hostname: `eunomia.dev`
- Preferred dataset: `httpRequestsAdaptiveGroups`

## Public and repository data

- Live-site technical collection enabled: yes
- Public GitHub repository evidence enabled: yes
- Public web and primary-source evidence enabled: yes

The public-safe data brief generated on `2026-10-03 12:38 UTC` reports homepage, `robots.txt`, and sitemap HTTP 200, **810 sitemap entries**, canonical `https://eunomia.dev/`, **100 active non-fork repositories**, **10,116 stars**, **1,335 forks**, **322 open issue/PR records**, and **63 DEV articles**. These are contextual public observations, not a blended SEO score.

The production robots file allows crawling and points to `https://eunomia.dev/sitemap.xml`. Exact generated production output remains the primary publication-inspection surface; external crawler discovery may lag a fresh deployment.

## Deployment

- Provider: `github-actions`
- Production workflow: `Deploy Static App`
- Production environment: `github-pages`
- Verification URL: `https://eunomia.dev/`

Store only durable public metadata here.
