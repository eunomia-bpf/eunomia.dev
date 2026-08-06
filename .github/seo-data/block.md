# External blockers

## Daily scheduler is not configured

- Blocked action: automatic once-per-day invocation of `DAILY_TASK.md`.
- Evidence: the repository has no daily agent workflow, and no external scheduler
  has been configured or verified.
- Impact: the repository now defines the complete operation, but nothing invokes
  it every day.
- Minimal external action: create one daily task in `America/Los_Angeles` with
  exactly this instruction:

  > Open `eunomia-bpf/eunomia.dev`, read `DAILY_TASK.md` from the current default
  > branch, and complete the task exactly as the repository instructs. Treat the
  > repository as authoritative.

Do not copy the rest of the operating policy into the scheduler.

## Private analytics sources are not configured

- Blocked actions: source-native Search Console, GA4, and Cloudflare analysis.
- Evidence: all three private sources are disabled or unconfigured in `site.md`.
- Impact: query, acquisition, engagement, edge-traffic, bot, cache, and some
  outcome analysis remains unavailable.
- Minimal external action: authorize or configure supported read-only access or a
  public-safe export path for each source, without committing IDs, credentials,
  private URLs, raw private data, or personal information.

## Partial operation remains required

These blockers do not justify skipping the daily operation once a scheduler
exists. Each run must still analyze live-site technical evidence, public GitHub
repository evidence, and public primary-source evidence; it must mark private
sources unavailable and must never convert missing coverage into zero.

Remove or narrow a blocker in the next daily pull request after the external
condition is verified as resolved.
