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
contains six Search Console exports and one GA4 organic landing-page export for
`2026-07-27` through `2026-08-02`. Only one complete weekly window is currently
available, so previous-week and 28-day comparisons remain unavailable until
additional exports appear. The GA4 export is also limited to organic landing-page
rows and does not by itself provide complete acquisition, conversion, or outbound
behavior coverage.

These constraints never justify skipping the daily operation. Each run must use
the available Google exports, live-site evidence, public GitHub evidence, and
public primary-source evidence; missing coverage must never be converted into
zero. Every run must still publish one new Daily Report under the current
repository contract.

Remove or narrow a blocker in the next daily pull request after the external
condition is verified as resolved.
