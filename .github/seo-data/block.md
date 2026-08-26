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

The last source-native Google set verified by the daily operation is
`2026-08-17..23`, observed on `2026-08-25`. Direct Drive search in the
`2026-08-26` runtime returned no accessible matching files, so no newer export
set or row-level values are inferred from that connector result. Missing retrieval
is not converted to zero and does not erase the previously verified coverage.

For Search Console, the verified latest file contains date rows through
`2026-08-22`; `2026-08-23` is absent. The `2026-08-22` row is now outside the
configured three-day lag, but this run could not retrieve its source-native row
values directly. The last calculated equal-duration finalized comparison is
therefore retained rather than extrapolated: `2026-08-17..21` versus
`2026-08-10..14` has 443 versus 360 clicks, 52,433 versus 60,080 impressions,
about 0.845% versus 0.599% CTR, and average position about 9.24 versus 9.86.

A complete latest-7-days versus previous-7-days Search Console comparison remains
unavailable because the latest source-native week is missing `2026-08-23` and the
predecessor history still contains the missing `2026-08-09` row. The 28-day
comparison also remains unavailable because verified export history is not long
enough. Missing rows are never converted to zero.

The GA4 landing-page aggregate for `2026-08-17..23` is now fully outside the
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
