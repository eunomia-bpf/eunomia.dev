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

Google Drive access is verified and is not a blocker. The configured folder was
directly rechecked on `2026-09-19`; no weekly source set newer than
`2026-09-07..09-13` is present.

For Search Console, the newest date export contains rows for `2026-09-07..09-12`
and no row for `2026-09-13`. Under the configured three-day finalization lag, all
six observed rows through `2026-09-12` are now treated as finalized. The finalized
six-day `2026-09-07..12` slice contains **376 clicks / 55,036 impressions /
~0.683% aggregate CTR / ~6.46 impression-weighted average position**.

The equal-duration finalized `2026-08-31..09-05` slice contains **388 clicks /
60,880 impressions / ~0.637% CTR / ~7.35 weighted position**. Relative to that
slice, the current six days have about **3.1% fewer clicks**, **9.6% fewer
impressions**, CTR about **0.046 percentage points higher**, and weighted average
position about **0.89 positions better**. This is useful source-native evidence
but is not a complete seven-day trend.

A complete latest-seven-days versus previous-seven-days comparison remains
unavailable because the preceding weekly export omits `2026-09-06` and the newest
export omits `2026-09-13`. Older history contains additional recorded gaps, so the
required complete 28-day versus preceding-comparable-period comparison is also
unavailable. Missing rows are never converted to zero.

The newest GA4 organic landing-page aggregate for `2026-09-07..13` contains **880
sessions** at about **43.52% session-weighted engagement**. It remains partial
because the frozen export was generated while lagged dates were present and has
no date dimension for safe finalized subsetting. The `2026-08-31..09-06`
aggregate contains **913 sessions** at about **47.54% engagement** and remains
partial for the same reason. The latest fully finalized weekly aggregate remains
`2026-08-24..30` at **1,007 sessions** and about **45.88% engagement**.

These constraints never justify skipping the daily operation. Each run must use
the available Google evidence, live-site evidence, public GitHub/DEV evidence,
and public primary-source evidence; missing or partial coverage must never be
converted into zero. Every run must still publish one new Daily Report under the
current repository contract.

Remove or narrow a blocker in the next daily pull request after the external
condition is verified as resolved.