# SEO status

## Current state

- Authoritative task: `DAILY_TASK.md`
- Technical SEO subtask: `.github/seo-data/daily-task.md`
- Daily Report subtask: `.agents/skills/eunomia-research-report/SKILL.md`
- External daily scheduler: configured and enabled
- Scheduler timing: daily, flexible around 08:00 in `America/Los_Angeles`
- Last completed daily run: `2026-08-08`
- Last verified data window: `2026-07-27` through `2026-08-02`
- Latest daily record: `2026-08-08`
- Last public-change pull request: recorded in the `2026-08-08` daily record and merged-PR closeout comment
- Last verified production deployment from a daily run: recorded in the merged-PR closeout comment
- Skill submodule commit: `516e9e2dcf012506a677a749049d64c5914643e9`

## Current Daily Report mix

The public archive contains three Daily Reports after the `2026-08-08` run:

- eBPF-centered: **1 of 3**
  - `/research/userspace-ebpf-runtime-contract/`
- pure Agent-centered: **2 of 3**
  - `/research/agent-trace-evidence-budget/`
  - `/research/parallel-agent-effect-serializability/`
- adjacent systems: **0 of 3**

The archive is still outside the long-run target. New reports should continue to
strongly favor eBPF. The active series is **eBPF Runtime, Extensibility, and
Composition**. The rolling target is 5–7 eBPF-centered reports per 10 published
reports and at most 1–2 pure Agent reports per 10.

## Current signals

- Live-site technical evidence: available
- Public GitHub repository evidence: available
- Public web and primary-source evidence: available
- Google Analytics 4: weekly Drive export available and verified
- Google Search Console: six dimension exports for one weekly window available and verified
- Cloudflare: not configured

The verified Drive folder currently covers one complete Search Console and GA4
weekly date window, `2026-07-27` through `2026-08-02`. Search Console has date,
search-appearance, device, country, page, and query dimensions; GA4 has an organic
landing-page export. The configured source cadence is weekly, so `2026-08-03`
through `2026-08-05` are not yet represented even though those dates are beyond
the configured three-day finalization lag. Previous-period and 28-day comparisons
remain unavailable until more weekly windows accumulate.

For `2026-07-27` through `2026-08-02`, Search Console reports 486 clicks and
55,021 impressions, for a weighted CTR of 0.883% and an impression-weighted
average position of approximately 8.21. Desktop accounts for about 92.9% of
impressions in the current window. Search appearance includes translated-result
traffic, and country rows show broad international discovery, which reinforces
the value of maintaining first-class English and Chinese variants.

The GA4 organic landing-page export shows tutorials and CUDA/GPU material among
the strongest known landing pages. Its largest row is `(not set)` with low
engagement, which remains a measurement-quality problem. The current export is
not sufficient to make site-wide conversion or outbound-action claims.

## Current technical baseline

The repository generates sitemap, robots, canonical, `hreflang`, Open Graph,
structured data, legacy redirect stubs, and HTTP audit artifacts. Production
deploys through the `Deploy Static App` workflow.

The `2026-08-08` technical audit did not establish a new crawl, canonical,
hreflang, structured-data, redirect, or performance defect that justified a
separate implementation change. Public search still exposes legacy `/en/` URLs,
but the current build intentionally emits canonical redirect stubs for those
paths; this remains an observation to monitor rather than a proven new defect.

The SEO skill submodule remains pinned at
`516e9e2dcf012506a677a749049d64c5914643e9`. Newer upstream commits restructure
the skill package and remove interfaces that the current consuming contract
still names, including `change-seo-site` and `scripts/validate_seo_data.py`.
Advancing the pointer therefore requires a coordinated consumer migration rather
than an isolated submodule bump.

## Current focus

1. Continue the **eBPF Runtime, Extensibility, and Composition** Daily Report
   series. The preferred next question is how multiple independent eBPF
   extensions should share one hook safely.
2. Publish one new eBPF-centered Daily Report per scheduled run until the rolling
   mix moves toward 5–7 eBPF reports per 10 and pure Agent reports no longer
   exceed the 1–2 per 10 cap.
3. Use the verified weekly GA4 and Search Console exports in every run; never mark
   them unavailable while the configured folder remains readable.
4. Accumulate enough weekly history for previous-period and 28-day comparisons.
5. Investigate the GA4 `(not set)` landing-page row with a richer export before
   making a measurement or content change.
6. Revisit legacy `/en/` indexing only if fresh crawl or analytics evidence shows
   canonical redirect stubs are causing measurable harm.
7. Migrate the consuming SEO contract to the newer `seo-skill` layout before
   updating the submodule pointer.
8. Add Cloudflare evidence when a supported read-only path exists.

This file is the current verified summary. Detailed history belongs in
`.github/seo-data/daily/` and the merged daily pull requests.