# Human-only blockers

Only unresolved external conditions belong here. The daily ChatGPT scheduler is
already configured and enabled; it is not a blocker.

## Cloudflare analytics is not configured

- Blocked action: source-native edge request, bot, cache, country, and status-code analysis.
- Evidence: Cloudflare remains disabled in `site.md`.
- Impact: daily analysis can use Search Console, GA4, live-site, GitHub, and public primary-source evidence, but cannot make Cloudflare-grounded traffic or cache conclusions.
- Minimal external action: authorize a supported read-only connector or export route without committing zone IDs, credentials, private URLs, raw private data, or personal information.

## Current data-history constraint

Google Drive access is verified and is not a blocker. The configured folder was directly rechecked on `2026-09-11`; no weekly source set newer than `2026-08-31..09-06` is present.

For Search Console, the newest date export contains rows for `2026-08-31..09-05` and no row for `2026-09-06`. Under the configured three-day finalization lag, all currently observed rows through September 5 are finalized. The finalized six-day `2026-08-31..09-05` slice contains **388 clicks / 60,880 impressions / ~0.637% aggregate CTR / ~7.35 impression-weighted average position**.

The equal-duration finalized `2026-08-24..29` slice contains **436 clicks / 55,594 impressions / ~0.784% CTR / ~10.73 weighted position**. Relative to that slice, the current six days have about **11.0% fewer clicks**, **9.5% more impressions**, CTR about **0.147 percentage points lower**, and weighted average position about **3.38 positions better**. This is useful source-native evidence but is not a complete seven-day trend.

A complete latest-seven-days versus previous-seven-days comparison remains unavailable because the newest weekly export omits `2026-09-06` and the preceding weekly export omits `2026-08-30`. Older history includes the recorded `2026-08-23` gap and other incompleteness, so the required complete 28-day versus preceding-comparable-period comparison is also unavailable. Missing rows are never converted to zero.

The newer frozen GA4 organic landing-page aggregate for `2026-08-31..09-06` contains **913 sessions** at about **47.54% session-weighted engagement**, but it was produced while lagged dates were present and has no date dimension for safe finalized subsetting. It therefore remains partial. The latest fully finalized weekly aggregate remains `2026-08-24..30` at **1,007 sessions** and about **45.88% engagement**.

These constraints never justify skipping the daily operation. Each run must use the available Google evidence, live-site evidence, public GitHub evidence, and public primary-source evidence; missing or partial coverage must never be converted into zero. Every run must still publish one new Daily Report under the current repository contract.

Remove or narrow a blocker in the next daily pull request after the external condition is verified as resolved.
