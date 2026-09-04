# Human-only blockers

Only unresolved external conditions belong here. The daily ChatGPT scheduler is
already configured and enabled; it is not a blocker.

## Cloudflare analytics is not configured

- Blocked action: source-native edge request, bot, cache, country, and status-code analysis.
- Evidence: Cloudflare remains disabled in `site.md`.
- Impact: daily analysis can use Search Console, GA4, live-site, GitHub, and public primary-source evidence, but cannot make Cloudflare-grounded traffic or cache conclusions.
- Minimal external action: authorize a supported read-only connector or export route without committing zone IDs, credentials, private URLs, raw private data, or personal information.

## Current data-history constraint

Google Drive access is verified and is not a blocker. The configured folder was
directly rechecked on `2026-09-04`. No export newer than the source-native
`2026-08-24..30` weekly set is present.

For Search Console, the newest date export contains rows for `2026-08-24..29` and
no row for `2026-08-30`. Every row actually present in that six-day slice is
finalized evidence under the configured lag. Those six rows sum to **436 clicks
and 55,594 impressions**, with aggregate CTR about **0.784%** and
impression-weighted average position about **10.73**.

The equal-duration `2026-08-17..22` slice has 477 clicks, 59,798 impressions,
~0.798% CTR, and ~9.56 weighted position. The comparison is useful but is not a
complete seven-day trend because the preceding weekly export omits `2026-08-23`
and the current export omits `2026-08-30`. Other historical gaps also prevent the
required complete 28-day comparison. Missing rows are never converted to zero.

The GA4 organic landing-page aggregate for `2026-08-24..30` is fully finalized.
It contains **1,007 sessions** at about **45.88% session-weighted engagement**.
The preceding finalized `2026-08-17..23` aggregate contains 984 sessions at about
49.29% engagement. Sessions are about **2.3% higher** week over week while
engagement is about **3.41 percentage points lower**.

These constraints never justify skipping the daily operation. Each run must use
the available Google evidence, live-site evidence, public GitHub evidence, and
public primary-source evidence; missing or partial coverage must never be
converted into zero. Every run must still publish one new Daily Report under the
current repository contract.

Remove or narrow a blocker in the next daily pull request after the external
condition is verified as resolved.
