# SEO status

## Current state

- Authoritative task: `DAILY_TASK.md`
- Technical SEO subtask: `.github/seo-data/daily-task.md`
- Daily Report subtask: `.agents/skills/eunomia-research-report/SKILL.md`
- External daily scheduler: configured and enabled
- Scheduler timing: daily, flexible around 08:00 in `America/Los_Angeles`
- Last completed daily run: `2026-08-11`
- Last verified data window: Search Console rows through `2026-08-08`; GA4 weekly export through `2026-08-09` is partial for finalized analysis
- Latest daily record: `2026-08-12`
- Last public-change pull request: `#151`, verified in its merged-PR closeout comment
- Last verified production deployment from a daily run: `Deploy Static App` run `31511561365` for squash commit `5e57487b6055008082127129f4b68b8c2d5dc57f`
- Skill submodule commit: `516e9e2dcf012506a677a749049d64c5914643e9`

The `2026-08-11` operation is fully closed out. PR `#151` squash-merged as `5e57487b6055008082127129f4b68b8c2d5dc57f`; the exact commit passed `Validate SEO Operations` run `31511561363` and production `Deploy Static App` run `31511561365`. The deployed Pages artifact contained both language variants of the stateful-upgrade report with the expected report structure, canonical URLs, reciprocal `hreflang` plus `x-default`, Article JSON-LD, and Daily Report index navigation.

The `2026-08-12` async causal-profiling report is the current daily change. Per `DAILY_TASK.md`, its exact merge commit, final CI, deployment, and bilingual public verification belong in one closeout comment on the merged daily PR rather than a second closeout pull request.

## Current Daily Report mix

The current change adds the sixth Daily Report:

- eBPF-centered: **4 of 6**
  - `/research/async-ebpf-causal-profiler/`
  - `/research/stateful-ebpf-transactional-upgrade/`
  - `/research/ebpf-hook-composition-contract/`
  - `/research/userspace-ebpf-runtime-contract/`
- pure Agent-centered: **2 of 6**
  - `/research/agent-trace-evidence-budget/`
  - `/research/parallel-agent-effect-serializability/`
- adjacent systems: **0 of 6**

The current change brings the eBPF-centered share to about **66.7%**, inside the long-run 5–7/10 target proportion. The two pure-Agent reports still represent too large a share of the six-report archive relative to the eventual 1–2/10 cap, so continue to favor eBPF and adjacent systems before another pure-Agent report. The active series remains **eBPF Runtime, Extensibility, and Composition**.

## Current signals

- Live-site technical evidence: available
- Public GitHub repository evidence: available
- Public web and primary-source evidence: available
- Google Analytics 4: two weekly Drive organic landing-page exports available; the newest export includes `2026-08-09` and has no date dimension
- Google Search Console: two adjacent weekly sets of date, search-appearance, device, country, page, and query exports available; current date rows still end on `2026-08-08`
- Cloudflare: not configured

The configured Drive folder contains adjacent weekly Google export sets for `2026-07-27` through `2026-08-02` and `2026-08-03` through `2026-08-09`. The current Search Console date export still has rows only through `2026-08-08`. A complete latest-seven-days versus previous-seven-days comparison remains unavailable because the preceding window would require `2026-07-26`, which is absent. The required 28-day comparison also remains unavailable rather than being inferred.

A valid common-duration finalized six-day comparison is available. Search Console reports **478 clicks / 69,053 impressions** for `2026-08-03` through `2026-08-08`, versus **409 / 46,327** for `2026-07-28` through `2026-08-02`. That is about **+16.9% clicks** and **+49.1% impressions**. Weighted CTR moved from about **0.883%** to **0.692%**, down about **0.191 percentage points**, while impression-weighted average position moved from about **8.18** to **9.42**. `2026-08-05` alone contributed **24,516 impressions**, about **35.5%** of the newer six-day total, at position about **13.30** and CTR about **0.343%**. The current weekly page/query aggregates cannot attribute that spike to one page or query family without a date-by-page or date-by-query export.

The current device export remains desktop-heavy: desktop has **64,355 impressions / 418 clicks**, mobile **4,614 / 59**, and tablet **84 / 1**. Desktop therefore contributes about **93.2%** of impressions. Translated-result search appearance remains present at **1 click / 116 impressions**.

Current page aggregates continue to show broad discovery across the homepage, eBPF tutorials, XDP/TCX, CUDA/GPU material, `GPTtrace`, AgentSight, bpftime, and long-tail systems pages. The current Daily Report pages are too new for the weekly Search Console export to support a report-level traffic conclusion. Raw query rows remain outside Git.

The newest GA4 organic landing-page export covers `2026-08-03` through `2026-08-09` without a date dimension, so it remains directional rather than a clean finalized-period comparison. `(not set)` is the largest row at **116 sessions** with about **6.0%** engagement; the homepage has **43 sessions** at about **53.5%** engagement; the Chinese GPU-architecture page **28** at about **64.3%**; `GPTtrace` **18** at about **66.7%**; and the Chinese Hello World tutorial **18** at about **72.2%**. The export has no usable configured outcome signal for site-wide conversion claims. `(not set)` remains a measurement-quality question, not evidence for a speculative site change.

## Previous-run closeout

The `2026-08-11` operation is fully closed out. PR `#151` squash-merged as `5e57487b6055008082127129f4b68b8c2d5dc57f`. Its final `Validate SEO Operations` and production `Deploy Static App` workflows succeeded for the exact squash commit. The deployed artifact contained `/research/stateful-ebpf-transactional-upgrade/` and `/zh/research/stateful-ebpf-transactional-upgrade/` with required content, canonical URLs, reciprocal language alternates, Open Graph/Article metadata, and Daily Report navigation.

## Current technical baseline

The repository generates sitemap, robots, canonical, `hreflang`, Open Graph, structured data, legacy redirect stubs, and HTTP audit artifacts. Production deploys through the `Deploy Static App` workflow.

The `2026-08-12` technical audit does not establish a new crawl, canonical, hreflang, structured-data, redirect, broken-link, rendering, accessibility, or performance defect that justifies an unrelated implementation change. The public homepage is crawlable and exposes the Daily Report entry point alongside the expected eBPF/runtime and project navigation. Search-visible legacy `/en/` paths remain a monitoring item because the repository intentionally emits canonical redirect stubs for them.

The active eBPF Runtime, Extensibility, and Composition sequence now has four substantial reports in the current change. The existing Daily Report hub already exposes the sequence. A separate series hub remains deferred until report-level acquisition or navigation evidence shows that another public surface would improve retrieval; do not create a thin page only because the series crossed a count threshold.

The SEO skill submodule remains pinned at `516e9e2dcf012506a677a749049d64c5914643e9`. A newer upstream layout removes interfaces the current consuming contract still names, including `change-seo-site` and `scripts/validate_seo_data.py`. Advancing the pointer still requires a coordinated consumer migration rather than an isolated submodule bump.

## Current focus

1. Complete and verify the current async eBPF causal-profiling Daily Report through final CI, squash merge, exact production deployment, and bilingual public verification.
2. Continue the **eBPF Runtime, Extensibility, and Composition** series with the preferred next question: which new Linux I/O hooks and programmable interfaces make eBPF mechanisms practical that older hook sets could not implement cleanly.
3. Continue favoring eBPF and adjacent systems until the rolling archive is comfortably inside the long-run 5–7/10 eBPF and 1–2/10 pure-Agent mix.
4. Use both weekly GA4 and Search Console export sets in every run; keep required 7-day and 28-day comparisons marked unavailable until complete source history exists.
5. Obtain date-by-page or date-by-query Search Console evidence before attributing the `2026-08-05` impression spike.
6. Investigate GA4 `(not set)` with richer dimensions before making a measurement or content change.
7. Revisit legacy `/en/` indexing only if fresh crawl or analytics evidence shows canonical redirect stubs are causing measurable harm.
8. Migrate the consuming SEO contract before updating the skill submodule pointer.
9. Add Cloudflare evidence when a supported read-only path exists.

This file is the current verified summary. Detailed history belongs in `.github/seo-data/daily/` and the merged daily pull requests.
