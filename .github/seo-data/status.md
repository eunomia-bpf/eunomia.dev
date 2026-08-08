# SEO status

## Current state

- Authoritative task: `DAILY_TASK.md`
- Technical SEO subtask: `.github/seo-data/daily-task.md`
- Daily Report subtask: `.agents/skills/eunomia-research-report/SKILL.md`
- External daily scheduler: configured and enabled
- Scheduler timing: daily, flexible around 08:00 in `America/Los_Angeles`
- Last completed daily run: none
- Last verified data window: `2026-07-27` through `2026-08-02`
- Latest daily record: `2026-08-06` (baseline recorded before the first scheduled run)
- Last public-change pull request: none from the unified daily operation
- Last verified production deployment from a daily run: none
- Skill submodule commit: `516e9e2dcf012506a677a749049d64c5914643e9`

## Current Daily Report mix

The public archive currently contains two Daily Reports and both are pure
Agent-centered topics:

- `/research/agent-trace-evidence-budget/`
- `/research/parallel-agent-effect-serializability/`

Until the rolling archive becomes compliant with the repository editorial mix,
new reports should strongly favor eBPF. The active series is **eBPF Runtime,
Extensibility, and Composition**. The rolling target is 5–7 eBPF-centered reports
per 10 published reports and at most 1–2 pure Agent reports per 10.

## Current signals

- Live-site technical evidence: available
- Public GitHub repository evidence: available
- Public web and primary-source evidence: available
- Google Analytics 4: weekly Drive export available and verified
- Google Search Console: six weekly Drive exports available and verified
- Cloudflare: not configured

The verified Drive folder contains Search Console exports for dates, search
appearance, devices, countries, pages, and queries, plus a GA4 organic landing-page
export. The current coverage is one complete weekly window, so source-native
seven-day analysis is possible but prior-period and 28-day comparisons are not yet
supported by the available history.

For `2026-07-27` through `2026-08-02`, Search Console reports 486 clicks and
55,021 impressions, for a weighted CTR of 0.883% and an impression-weighted
average position of approximately 8.21. The daily series falls sharply over the
weekend, but a single week cannot distinguish a weekday effect from a real demand
or ranking change.

The GA4 organic landing-page export shows tutorials and CUDA/GPU material among
the strongest known landing pages. Its largest row is `(not set)`, which should be
treated as a measurement-quality problem before drawing strong content or
conversion conclusions.

## Current technical baseline

The repository generates sitemap, robots, canonical, `hreflang`, Open Graph,
structured data, and HTTP audit artifacts. Production deploys through the
`Deploy Static App` workflow. The unified daily operation has not completed its
first scheduled run yet.

## Current focus

1. Merge the repository-owned daily operating contract and verified analytics
   baseline before the first scheduled run relies on it.
2. Publish one new eBPF-centered Daily Report per scheduled run until the rolling
   mix moves toward 5–7 eBPF reports per 10 and pure Agent reports no longer exceed
   the 1–2 per 10 cap.
3. Use the verified weekly GA4 and Search Console exports in every run; never mark
   them unavailable while the configured folder remains readable and fresh.
4. Accumulate enough weekly history for previous-period and 28-day comparisons.
5. Investigate the GA4 `(not set)` landing-page row and remaining legacy `/en/`
   traffic when selecting technical SEO work.
6. Add Cloudflare evidence when a supported read-only path exists.

This file is the current verified summary. Detailed history belongs in
`.github/seo-data/daily/` and the merged daily pull requests.
