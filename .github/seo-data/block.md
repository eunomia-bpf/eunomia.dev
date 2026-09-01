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
directly rechecked on `2026-08-31` after the weekly exporter completed. It now
contains a fresh source-native set for `2026-08-24..30` in addition to the older
weekly sets.

For Search Console, the new date export contains rows for `2026-08-24..29` and no
row for `2026-08-30`. Under the configured three-day finalization lag, rows
through `2026-08-28` are treated as finalized; `2026-08-29` remains inside the
lag and is not used as finalized evidence. The finalized `2026-08-24..28` rows
sum to 398 clicks and 48,044 impressions, with aggregate CTR about 0.828% and
impression-weighted average position about 10.04.

A complete latest-7-days versus previous-7-days Search Console comparison remains
unavailable because the required source history is not contiguous. Earlier
weekly exports omit `2026-08-16` and `2026-08-23`, and the current export omits
`2026-08-30`. The 28-day comparison also remains unavailable because there is no
complete preceding 28-day source window. Missing rows are never converted to
zero.

The fresh GA4 organic landing-page aggregate for `2026-08-24..30` is available,
but the whole weekly aggregate is not finalized under the configured lag because
it includes `2026-08-29..30`. It currently contains 1,007 organic landing-page
sessions at about 45.88% session-weighted engagement and is treated only as a
partial early signal. The latest fully finalized source-native GA4 weekly
aggregate remains `2026-08-17..23`: 984 sessions at about 49.29% engagement,
versus 970 at about 44.95% for `2026-08-10..16`.

These constraints never justify skipping the daily operation. Each run must use
the available Google evidence, live-site evidence, public GitHub evidence, and
public primary-source evidence; missing or partial coverage must never be
converted into zero. Every run must still publish one new Daily Report under the
current repository contract.

Remove or narrow a blocker in the next daily pull request after the external
condition is verified as resolved.
