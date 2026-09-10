# Human-only blockers

Only unresolved external conditions belong here. The daily ChatGPT scheduler is
already configured and enabled; it is not a blocker.

## Cloudflare analytics is not configured

- Blocked action: source-native edge request, bot, cache, country, and status-code analysis.
- Evidence: Cloudflare remains disabled in `site.md`.
- Impact: daily analysis can use Search Console, GA4, live-site, GitHub, and public primary-source evidence, but cannot make Cloudflare-grounded traffic or cache conclusions.
- Minimal external action: authorize a supported read-only connector or export route without committing zone IDs, credentials, private URLs, raw private data, or personal information.

## Current data-history constraint

Google Drive access is verified and is not a blocker. The configured folder was directly rechecked on `2026-09-10`; the newest available weekly source set remains `2026-08-31..09-06`.

For Search Console, the newest date export contains rows for `2026-08-31..09-05` and no row for `2026-09-06`. Under the configured three-day finalization lag, all six observed rows through `2026-09-05` are now finalized. They sum to **388 clicks and 60,880 impressions**, with aggregate CTR about **0.637%** and impression-weighted average position about **7.35**.

The equal-duration finalized `2026-08-24..29` slice contains **436 clicks / 55,594 impressions / ~0.784% CTR / ~10.73 weighted position**. Relative to that slice, the current six days have about **11.0% fewer clicks**, **9.5% more impressions**, CTR about **0.147 percentage points lower**, and weighted average position about **3.38 positions better**. This is useful source-native evidence but is not a complete seven-day trend.

A complete latest-seven-days versus previous-seven-days comparison remains unavailable because the previous weekly export omits `2026-08-30` and the current export omits `2026-09-06`. Older history includes the previously recorded `2026-08-23` gap and other incompleteness, so the required complete 28-day versus preceding-comparable-period comparison is also unavailable. Missing rows are never converted to zero.

The GA4 organic landing-page aggregate for `2026-08-31..09-06` is now fully outside the configured finalization lag and is no longer a finalization blocker. It contains **913 sessions** at about **47.54% session-weighted engagement**, compared with the preceding finalized `2026-08-24..30` aggregate at **1,007 sessions** and about **45.88% engagement**. The weekly export still lacks a date dimension, so it cannot support within-week causal attribution.

These constraints never justify skipping the daily operation. Each run must use the available Google evidence, live-site evidence, public GitHub evidence, and public primary-source evidence; missing coverage must never be converted into zero. Every run must still publish one new Daily Report under the current repository contract.

Remove or narrow a blocker in the next daily pull request after the external condition is verified as resolved.