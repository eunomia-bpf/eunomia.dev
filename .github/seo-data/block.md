# Human-only blockers

## Cloudflare analytics is not configured

- Blocked action: source-native edge request, bot, cache, country, and status-code analysis.
- Evidence: Cloudflare remains disabled in `site.md`.
- Impact: Search Console, GA4, live-site, GitHub, DEV, and public primary-source evidence remain available.

## Current data-history constraint

The configured Google source folder was rechecked on `2026-09-30`. The newest source family is `2026-09-21..09-27`.

Search Console has observed finalized date rows only for `2026-09-21..25`: **342 clicks / 37,406 impressions / ~0.914% CTR / ~6.47 impression-weighted position**. September 26–27 are absent. The equal-duration `2026-09-14..18` slice contains **349 / 49,798 / ~0.701% / ~6.82**.

This is a five-day comparison, not a complete seven-day trend. Older history also contains gaps, so a complete latest-28-day comparison remains unavailable. Missing rows are never converted to zero.

GA4 `2026-09-21..27` contains **848 sessions** at about **43.75% session-weighted engagement** and remains partial. The latest fully finalized weekly aggregate remains `2026-08-24..30` at **1,007 sessions** and about **45.88% engagement**.

These source-history constraints do not justify skipping the daily publication.
