# Human-only blockers

Only unresolved external conditions belong here. The daily ChatGPT scheduler is already configured and enabled; it is not a blocker.

## Cloudflare analytics is not configured

- Blocked action: source-native edge request, bot, cache, country, and status-code analysis.
- Evidence: Cloudflare remains disabled in `site.md`.
- Impact: daily analysis can use Search Console, GA4, live-site, GitHub, and public primary-source evidence, but cannot make Cloudflare-grounded traffic or cache conclusions.
- Minimal external action: authorize a supported read-only connector or export route without committing zone IDs, credentials, private URLs, raw private data, or personal information.

## Current data-history constraint

Google Drive access is verified and is not a blocker. The configured folder was directly rechecked on `2026-09-08` and contains the `2026-08-31..09-06` weekly source set.

For Search Console, date rows are present for `2026-08-31..09-05`; `2026-09-06` is absent. Under the configured three-day lag, rows through September 5 are finalized for this run. Finalized `2026-08-31..09-05` contains **388 clicks / 60,880 impressions / ~0.637% aggregate CTR / ~7.35 impression-weighted average position**.

The equal-duration finalized `2026-08-24..29` slice contains **436 clicks / 55,594 impressions / ~0.784% CTR / ~10.73 weighted position**. Relative to it, current clicks are about **11.0% lower**, impressions **9.5% higher**, CTR about **0.147 percentage points lower**, and weighted position about **3.38 positions better**.

This remains a source-history constraint rather than a collection failure. Missing `2026-08-30` and absent `2026-09-06` prevent the required complete latest-seven versus previous-seven comparison, and older gaps still prevent a complete 28-day comparison. Missing rows are never converted to zero.

A new GA4 organic landing-page aggregate for `2026-08-31..09-06` is present at **913 sessions** and about **47.54% session-weighted engagement**, but the weekly file has no date dimension and includes a date inside the finalization lag, so it is treated as partial. The latest fully finalized GA4 aggregate remains `2026-08-24..30`: **1,007 sessions** at about **45.88% session-weighted engagement**, versus 984 sessions at about 49.29% for `2026-08-17..23`.

These constraints never justify skipping the daily operation. Each run must use the available Google evidence, live-site evidence, public GitHub evidence, and public primary-source evidence; missing or partial coverage must never be converted into zero. Every run must still publish one new Daily Report under the current repository contract.

Remove or narrow a blocker in the next daily pull request after the external condition is verified as resolved.
