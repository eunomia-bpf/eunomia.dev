# Human-only blockers

The entries below are external account or scheduler conditions that the repository
cannot truthfully mark as resolved by itself.

## Daily scheduler is not configured

- Blocked action: automatic once-per-day invocation of `DAILY_TASK.md`.
- Evidence: the repository has no daily agent workflow, and no external scheduler
  has been configured or verified.
- Impact: the repository defines the complete operation, but nothing invokes it
  every day.
- Minimal external action: create one daily task in `America/Los_Angeles` with
  exactly this instruction:

  > Open `eunomia-bpf/eunomia.dev`, read `DAILY_TASK.md` from the current default
  > branch, and complete the task exactly as the repository instructs. Treat the
  > repository as authoritative.

Do not copy the rest of the operating policy into the scheduler.

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

## Current data-history limitation

Google Drive access is verified and is not a blocker. The configured folder
contains six Search Console exports and one GA4 organic landing-page export for
`2026-07-27` through `2026-08-02`. Only one complete weekly window is currently
available, so previous-week and 28-day comparisons must remain unavailable until
additional exports appear. The GA4 export is also limited to organic landing-page
rows and does not by itself provide complete acquisition, conversion, or outbound
behavior coverage.

These limitations do not justify skipping the daily operation once a scheduler
exists. Each run must use the available Google exports, live-site evidence,
public GitHub evidence, and public primary-source evidence; missing coverage must
never be converted into zero.

Remove or narrow a blocker in the next daily pull request after the external
condition is verified as resolved.
