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

Google Drive access is verified and is not a blocker. The configured folder
contains three adjacent weekly Search Console and GA4 export sets beginning
`2026-07-27`, `2026-08-03`, and `2026-08-10`; no newer weekly set was observed on
`2026-08-20`. The newest Search Console date file has rows through `2026-08-15`,
which are finalized under the configured three-day lag.

A complete latest-7-days versus previous-7-days Search Console comparison remains
unavailable because the `2026-08-09` date row is absent across the adjacent
weekly exports. The available data supports an equal-duration six-day comparison
for `2026-08-10` through `2026-08-15` versus `2026-08-03` through `2026-08-08`.
The 28-day comparison also remains unavailable because the export history is not
long enough.

GA4 landing-page files are weekly aggregates without a date dimension. On
`2026-08-20`, the complete `2026-08-10` through `2026-08-16` aggregate remains
outside the configured three-day finalization lag and is usable as a finalized
source-native weekly comparison against `2026-08-03` through `2026-08-09`. The
landing-page exports still do not provide complete acquisition, conversion, or
outbound behavior coverage by themselves.

These constraints never justify skipping the daily operation. Each run must use
the available Google exports, live-site evidence, public GitHub evidence, and
public primary-source evidence; missing coverage must never be converted into
zero. Every run must still publish one new Daily Report under the current
repository contract.

Remove or narrow a blocker in the next daily pull request after the external
condition is verified as resolved.
