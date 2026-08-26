# Human-only blockers

Only unresolved external conditions belong here. The daily ChatGPT scheduler is
already configured and enabled; it is not a blocker.

## Cloudflare analytics is not configured

- Blocked action: source-native edge request, bot, cache, country, and status-code
  analysis.
- Evidence: Cloudflare remains disabled in `site.md`.
- Impact: daily analysis can use Search Console, GA4, live-site, GitHub, and public
  primary-source evidence, but cannot make Cloudflare-grounded traffic or cache
  conclusions.
- Minimal external action: authorize a supported read-only connector or export
  route without committing zone IDs, credentials, private URLs, raw private data,
  or personal information.

## Current data-history constraint

Google Drive access is verified and is not a blocker. The configured folder was
directly rechecked on `2026-08-26`; it contains weekly export sets through
`2026-08-17..23` and no newer set was observed.

For Search Console, the verified latest date export contains rows through
`2026-08-22`; `2026-08-23` is absent. The `2026-08-22` row is outside the
configured three-day lag and its source-native values are available. The longest
contiguous equal-duration finalized comparison currently supported by the exports
is six days: `2026-08-17..22` versus `2026-08-10..15` has 477 versus 393 clicks,
59,798 versus 66,510 impressions, about 0.798% versus 0.591% CTR, and average
position about 9.56 versus 9.91. Clicks are about 21.4% higher, impressions about
10.1% lower, CTR about 0.207 percentage points higher, and average position
improves by about 0.35 positions.

A complete latest-7-days versus previous-7-days Search Console comparison remains
unavailable because the latest finalized window would be `2026-08-16..22` versus
`2026-08-09..15`, while the available date exports omit both `2026-08-16` and
`2026-08-09`. The 28-day comparison also remains unavailable because verified
export history is not long enough. Missing rows are never converted to zero.

The GA4 landing-page aggregate for `2026-08-17..23` is fully outside the
configured three-day finalization lag and is usable as a finalized source-native
weekly aggregate. It contains 984 organic landing-page sessions at about 49.29%
session-weighted engagement, compared with 970 sessions at about 44.95% for
`2026-08-10..16`. Sessions are about 1.4% higher and engagement about 4.34
percentage points higher. The weekly files have no date dimension, so they still
cannot support within-week attribution or daily causal claims.

These constraints never justify skipping the daily operation. Each run must use
the available Google evidence, live-site evidence, public GitHub evidence, and
public primary-source evidence; missing or partial coverage must never be
converted into zero. Every run must still publish one new Daily Report under the
current repository contract.

Remove or narrow a blocker in the next daily pull request after the external
condition is verified as resolved.
