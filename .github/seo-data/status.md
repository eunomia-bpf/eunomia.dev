# SEO status

## Current state

- Authoritative task: `DAILY_TASK.md`
- Technical SEO subtask: `.github/seo-data/daily-task.md`
- Daily Report subtask: `.agents/skills/eunomia-research-report/SKILL.md`
- External daily scheduler: configured and enabled
- Scheduler timing: daily, flexible around 08:00 in `America/Los_Angeles`
- Last completed daily run: `2026-08-09`
- Last verified data window: Search Console rows through `2026-08-08`; GA4 weekly export through `2026-08-09` is partial for finalized analysis
- Latest daily record: `2026-08-11`
- Last public-change pull request: `#149`, verified in its merged-PR closeout comment
- Last verified production deployment from a daily run: `Deploy Static App` run `31322000113` for squash commit `db383576aab403c63c4192bc8b52d75f300f58ce`
- Skill submodule commit: `516e9e2dcf012506a677a749049d64c5914643e9`

The `2026-08-11` daily record and stateful-upgrade report are in the current daily change. The prior `2026-08-10` attempt was closed without merge or deployment after `main` advanced; it is not counted as a completed daily publication. Per `DAILY_TASK.md`, exact merge/deployment/public verification for the current change belongs in the single merged-PR closeout comment, and the next run will promote those verified facts into this summary.

## Current Daily Report mix

The current change adds the fifth Daily Report:

- eBPF-centered: **3 of 5**
  - `/research/stateful-ebpf-transactional-upgrade/`
  - `/research/ebpf-hook-composition-contract/`
  - `/research/userspace-ebpf-runtime-contract/`
- pure Agent-centered: **2 of 5**
  - `/research/agent-trace-evidence-budget/`
  - `/research/parallel-agent-effect-serializability/`
- adjacent systems: **0 of 5**

The eBPF share is now 60%, which is inside the long-run 5–7/10 target proportion, while the two pure-Agent reports still occupy too much of the current five-report archive relative to the eventual 1–2/10 cap. Continue to favor eBPF and adjacent systems before another pure-Agent report. The active series remains **eBPF Runtime, Extensibility, and Composition**.

## Current signals

- Live-site technical evidence: available
- Public GitHub repository evidence: available
- Public web and primary-source evidence: available
- Google Analytics 4: two weekly Drive organic landing-page exports available; the newest export includes a non-finalized date and has no date dimension
- Google Search Console: two adjacent weekly sets of date, search-appearance, device, country, page, and query exports available; current date rows extend through `2026-08-08`
- Cloudflare: not configured

The configured Drive folder now contains adjacent weekly Google export sets for `2026-07-27` through `2026-08-02` and `2026-08-03` through `2026-08-09`. With the configured three-day finalization lag on `2026-08-11`, Search Console rows through `2026-08-08` are usable. The latest complete finalized seven-day GSC window is therefore `2026-08-02` through `2026-08-08`, but its preceding comparison would require `2026-07-26`, which is absent. The required 7-day versus previous-7-day comparison and the 28-day comparison remain unavailable rather than being inferred.

A valid common-duration finalized six-day comparison is available. Search Console reports **478 clicks / 69,053 impressions** for `2026-08-03` through `2026-08-08`, versus **409 / 46,327** for `2026-07-28` through `2026-08-02`. That is about **+16.9% clicks** and **+49.1% impressions**. Weighted CTR moved from about **0.883%** to **0.692%**, down about **0.191 percentage points**, while impression-weighted average position moved from about **8.18** to **9.42**. `2026-08-05` alone contributed **24,516 impressions**, about **35.5%** of the newer six-day total, at position about **13.30** and CTR about **0.343%**. The current weekly page/query aggregates cannot attribute that spike to one page or query family without a date-by-page or date-by-query export.

The latest device export remains desktop-heavy at about **93.2%** of impressions. Translated-result search appearance is still present. Current page/query aggregates continue to show discovery around eBPF tutorials, XDP, CUDA/GPU material, project names, and long-tail systems questions. Raw query rows remain outside Git.

The newest GA4 organic landing-page export covers `2026-08-03` through `2026-08-09` without a date dimension, so it is directional rather than a clean finalized-period comparison. `(not set)` remains the largest row at **116 sessions** with about **6.0%** engagement; the homepage has **43 sessions** at about **53.5%** engagement; the Chinese GPU-architecture page has **28 sessions** at about **64.3%** engagement; `GPTtrace` has **18 sessions** at about **66.7%** engagement; and the Chinese Hello World tutorial has **18 sessions** at about **72.2%** engagement. The export has no usable configured outcome signal for site-wide conversion claims. `(not set)` remains a measurement-quality question, not evidence for a speculative site change.

## Previous-run closeout

The `2026-08-09` operation is fully closed out. PR `#149` squash-merged as `db383576aab403c63c4192bc8b52d75f300f58ce`. Its final `Validate SEO Operations` and `Deploy Static App` checks succeeded. Production `Deploy Static App` run `31322000113` completed successfully for that exact squash commit, and the deployed artifact contained the English and Chinese hook-composition pages with the required content, canonical URLs, reciprocal language alternates, Open Graph metadata, and Article JSON-LD.

## Current technical baseline

The repository generates sitemap, robots, canonical, `hreflang`, Open Graph, structured data, legacy redirect stubs, and HTTP audit artifacts. Production deploys through the `Deploy Static App` workflow.

The `2026-08-11` technical audit does not establish a new crawl, canonical, hreflang, structured-data, redirect, broken-link, rendering, accessibility, or performance defect that justifies an unrelated implementation change. The public homepage currently exposes the Daily Report entry point and the expected eBPF/runtime navigation. The newest GA4 slice still contains some legacy `/en/` traffic, but the repository intentionally emits canonical redirect stubs for those paths; this remains a monitoring item rather than a proven defect.

The SEO skill submodule remains pinned at `516e9e2dcf012506a677a749049d64c5914643e9`. Upstream `AutoArchive/seo-skill` currently points at `9f0bd4f0b33b28fc22592e5463d95f63cda4d165`, but that newer layout removes interfaces the current consuming contract still names, including `change-seo-site` and `scripts/validate_seo_data.py`. Advancing the pointer still requires a coordinated consumer migration rather than an isolated submodule bump.

## Current focus

1. Complete and verify the current stateful eBPF transactional-upgrade Daily Report from the latest `main`.
2. Continue the **eBPF Runtime, Extensibility, and Composition** series with the preferred next question: what an asynchronous profiler built around modern eBPF needs to reconstruct causality across syscalls, io_uring, work queues, runtime tasks, and application-defined resources without unacceptable overhead or sampling bias.
3. Continue favoring eBPF and adjacent systems until the rolling archive is comfortably inside the long-run 5–7/10 eBPF and 1–2/10 pure-Agent mix.
4. Use both weekly GA4 and Search Console export sets in every run; keep required 7-day and 28-day comparisons marked unavailable until complete source history exists.
5. Obtain date-by-page or date-by-query Search Console evidence before attributing the `2026-08-05` impression spike.
6. Investigate GA4 `(not set)` with richer dimensions before making a measurement or content change.
7. Revisit legacy `/en/` indexing only if fresh crawl or analytics evidence shows canonical redirect stubs are causing measurable harm.
8. Migrate the consuming SEO contract before updating the skill submodule pointer.
9. Add Cloudflare evidence when a supported read-only path exists.

This file is the current verified summary. Detailed history belongs in `.github/seo-data/daily/` and the merged daily pull requests.
