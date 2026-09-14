# Human-only blockers

Only unresolved external conditions belong here. The daily ChatGPT scheduler is
already configured and enabled; it is not a blocker and remains enabled even when
an individual run cannot complete a repository or source operation.

## Cloudflare analytics is not configured

- Blocked action: source-native edge request, bot, cache, country, and status-code analysis.
- Evidence: Cloudflare remains disabled in `site.md`.
- Impact: daily analysis can use Search Console, GA4, live-site, GitHub, DEV, and public primary-source evidence, but cannot make Cloudflare-grounded traffic or cache conclusions.
- Minimal external action: authorize a supported read-only connector or export route without committing zone IDs, credentials, private URLs, raw private data, or personal information.

## Current data-history constraint

Google Drive access is verified and is not a blocker. The configured folder was directly rechecked on `2026-09-14`; a new weekly source set for `2026-09-07..09-13` is now present.

For Search Console, the newest date export contains rows for `2026-09-07..09-12` and no row for `2026-09-13`. Under the configured three-day finalization lag, rows through `2026-09-11` are treated as finalized, while the observed `2026-09-12` row remains partial. The finalized five-day `2026-09-07..11` slice contains **343 clicks / 47,606 impressions / ~0.720% aggregate CTR / ~6.47 impression-weighted average position**.

The equal-duration finalized `2026-08-31..09-04` slice contains **368 clicks / 53,341 impressions / ~0.690% CTR / ~7.45 weighted position**. Relative to that slice, the current five days have about **6.8% fewer clicks**, **10.8% fewer impressions**, CTR about **0.031 percentage points higher**, and weighted average position about **0.98 positions better**. This is useful source-native evidence but is not a complete seven-day trend.

A complete latest-seven-days versus previous-seven-days comparison remains unavailable because the preceding weekly export omits `2026-09-06` and the newest export omits `2026-09-13`. Older history includes the recorded `2026-08-23` and `2026-08-30` gaps and other incompleteness, so the required complete 28-day versus preceding-comparable-period comparison is also unavailable. Missing rows are never converted to zero.

The newly available GA4 organic landing-page aggregate for `2026-09-07..13` contains **880 sessions** at about **43.52% session-weighted engagement**. It is partial because it was generated while lagged dates were present and has no date dimension for safe finalized subsetting. The earlier `2026-08-31..09-06` aggregate contains **913 sessions** at about **47.54% engagement** and remains partial for the same reason. The latest fully finalized weekly aggregate remains `2026-08-24..30` at **1,007 sessions** and about **45.88% engagement**.

These constraints never justify skipping the daily operation. Each run must use the available Google evidence, live-site evidence, public GitHub/DEV evidence, and public primary-source evidence; missing or partial coverage must never be converted into zero. Every run must still publish one new Daily Report under the current repository contract.

Remove or narrow a blocker in the next daily pull request after the external condition is verified as resolved.
