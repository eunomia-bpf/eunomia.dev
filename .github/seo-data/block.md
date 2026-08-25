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

Google Drive access is verified and is not a blocker. The configured folder now
contains weekly export sets through `2026-08-17..23`; the new set was observed on
`2026-08-25`.

For Search Console, the new file contains date rows through `2026-08-22`. Under
the configured three-day finalization lag, `2026-08-21` is the newest usable date
and `2026-08-22` remains provisional; `2026-08-23` is absent. The available data
supports an equal-duration finalized comparison for `2026-08-17..21` versus
`2026-08-10..14`: 443 versus 360 clicks, 52,433 versus 60,080 impressions, about
0.845% versus 0.599% CTR, and average position about 9.24 versus 9.86.

A complete latest-7-days versus previous-7-days Search Console comparison remains
unavailable because the predecessor window contains the still-missing
`2026-08-09` row. The 28-day comparison also remains unavailable because verified
export history is not long enough. Missing rows are never converted to zero.

The new GA4 landing-page aggregate for `2026-08-17..23` is present, but the file
has no date dimension and part of its window remains inside the three-day lag.
The whole aggregate is therefore partial for finalized comparison. The latest
usable finalized GA4 weekly aggregate remains `2026-08-10..16`, compared with
`2026-08-03..09`.

These constraints never justify skipping the daily operation. Each run must use
the available Google exports, live-site evidence, public GitHub evidence, and
public primary-source evidence; missing or partial coverage must never be
converted into zero. Every run must still publish one new Daily Report under the
current repository contract.

Remove or narrow a blocker in the next daily pull request after the external
condition is verified as resolved.
