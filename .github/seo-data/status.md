# SEO status

## Current state

- Authoritative task: `DAILY_TASK.md`
- Technical SEO subtask: `.github/seo-data/daily-task.md`
- Daily Report subtask: `.agents/skills/eunomia-research-report/SKILL.md`
- External daily scheduler: configured and enabled
- Scheduler timing: daily, flexible around 08:00 in `America/Los_Angeles`
- Last completed daily run: `2026-08-09`
- Latest available Google export window: `2026-08-03` through `2026-08-09` (partial under the configured finalization lag)
- Latest finalized GSC date analyzed: `2026-08-07`
- Latest daily record: `2026-08-10`
- Last public-change pull request: `#149`, squash-merged as `db383576aab403c63c4192bc8b52d75f300f58ce`
- Last verified production deployment from a daily run: `Deploy Static App` run `31322000113` for `db383576aab403c63c4192bc8b52d75f300f58ce`
- Skill submodule commit: `516e9e2dcf012506a677a749049d64c5914643e9`

The `2026-08-09` closeout is verified and promoted here: final PR-head `Validate SEO Operations` and `Deploy Static App` checks succeeded, the exact squash commit completed the production `Deploy Static App` workflow, and the deployed export contained the English and Chinese hook-composition Daily Report pages with the expected content and metadata. The `2026-08-10` report is the current daily change; its exact merge/deployment/public verification will be recorded in the single merged-PR closeout comment required by `DAILY_TASK.md`.

## Current Daily Report mix

The archive contains five Daily Reports with the `2026-08-10` publication:

- eBPF-centered: **3 of 5**
  - `/research/stateful-ebpf-transactional-upgrade/`
  - `/research/ebpf-hook-composition-contract/`
  - `/research/userspace-ebpf-runtime-contract/`
- pure Agent-centered: **2 of 5**
  - `/research/agent-trace-evidence-budget/`
  - `/research/parallel-agent-effect-serializability/`
- adjacent systems: **0 of 5**

The eBPF share is now 60%, matching the long-run target proportion of 5–7 eBPF-centered reports per 10. The pure-Agent share remains above the eventual 1–2 per 10 cap while the archive is small, so new reports should still favor eBPF and directly adjacent systems. The active series remains **eBPF Runtime, Extensibility, and Composition**.

## Current signals

- Live-site technical evidence: available
- Public GitHub repository evidence: available
- Public web and primary-source evidence: available
- Google Analytics 4: two weekly Drive landing-page exports are available; the newest `2026-08-03` to `2026-08-09` export is partial for finalized analysis because it includes non-finalized days and has no date dimension
- Google Search Console: two six-dimension weekly export sets are available; the newest date export currently has rows through `2026-08-08`, with finalized analysis through `2026-08-07`
- Cloudflare: not configured

A fresh Google export arrived on `2026-08-10` and materially improved coverage. The prior complete weekly set covers `2026-07-27` through `2026-08-02`; the new set is named for `2026-08-03` through `2026-08-09`. Under the three-day finalization lag, GSC dates through `2026-08-07` are treated as finalized today. The latest complete finalized seven-day window is therefore `2026-08-01` through `2026-08-07`, but the preceding seven-day comparison requires `2026-07-25` and `2026-07-26`, which are not available. The 28-day comparison remains unavailable as well.

For a valid common-duration comparison, finalized GSC data for `2026-08-03` through `2026-08-07` shows **435 clicks**, **59,593 impressions**, **0.730%** weighted CTR, and impression-weighted position **9.98**, versus **309 clicks**, **38,511 impressions**, **0.802%** CTR, and position **8.15** for `2026-07-29` through `2026-08-02`. Clicks rose about **40.8%** and impressions **54.7%**, while CTR fell by roughly **0.072 percentage points** and average position worsened by about **1.83 positions**. `2026-08-05` accounts for about **41.1%** of the current five-day impressions, so a date-by-page or date-by-query export is needed before attributing the movement to a specific content family.

The newest partial device export remains about **93.2% desktop by impressions**, and translated-result discovery remains present. Page/query aggregates continue to show eBPF/tutorial, XDP, GPU/CUDA, project, and long-tail systems discovery, but raw query rows remain outside Git.

The latest partial GA4 organic landing-page export still has `(not set)` as its largest row, now at 116 sessions with about 6.0% engagement. The homepage, GPU/CUDA material, GPTtrace, eBPF tutorials, and Chinese pages remain visible landing surfaces. Because the newest export includes non-finalized dates, lacks a date dimension, and has no usable configured outcome signal, it does not support a finalized week-over-week or site-wide conversion claim.

Public GitHub activity on `2026-08-10` includes bpftime runtime work for selectable Frida fuzzy backtracing and the preload host-process contract. These are current engineering signals for the active runtime series, not traffic conversion evidence.

## Current technical baseline

The repository generates sitemap, robots, canonical, `hreflang`, Open Graph, structured data, legacy redirect stubs, and HTTP audit artifacts. Production deploys through the `Deploy Static App` workflow.

The `2026-08-10` technical audit did not establish a new crawl, canonical, hreflang, structured-data, redirect, broken-link, rendering, accessibility, or performance defect that justified an unrelated implementation change. Public search and the latest partial GA4 export still expose some legacy `/en/` traffic, but the current build intentionally emits canonical redirect stubs for those paths; this remains an observation to monitor rather than a proven new defect. The only coupled SEO work in the current daily change is the bilingual Daily Report index and generated page metadata for the new report.

The SEO skill submodule remains pinned at `516e9e2dcf012506a677a749049d64c5914643e9`. Upstream `AutoArchive/seo-skill` has advanced to `9f0bd4f0b33b28fc22592e5463d95f63cda4d165`, but the newer layout remains incompatible with interfaces named by the current consumer contract. Advancing the pointer still requires a coordinated consumer migration rather than an isolated submodule bump.

## Current focus

1. Complete and verify the `2026-08-10` stateful eBPF transactional-upgrade Daily Report through the required one-PR lifecycle, exact production deployment, bilingual public checks, and merged-PR closeout comment.
2. Continue the **eBPF Runtime, Extensibility, and Composition** series. After the stateful-upgrade report, the preferred next question is what an asynchronous profiler built around modern eBPF would need to reconstruct causality across syscalls, io_uring, work queues, runtime tasks, and application-defined resources with bounded overhead and measurable sampling bias.
3. Keep the rolling mix eBPF-heavy and avoid another pure Agent report while the pure-Agent share remains above the eventual 1–2 per 10 cap.
4. Use both weekly GA4 and Search Console export sets in later runs while respecting the three-day finalization lag; do not compare partial and finalized periods as equivalents.
5. Obtain enough older/newer history for a complete previous-7-day and 28-day comparison, and prefer a date-by-page or date-by-query export to explain the `2026-08-05` impression spike.
6. Investigate the GA4 `(not set)` landing-page row with a richer, date-dimensioned export before making a measurement or content change.
7. Revisit legacy `/en/` indexing only if fresh crawl or analytics evidence shows canonical redirect stubs are causing measurable harm.
8. Migrate the consuming SEO contract to the newer `seo-skill` layout before updating the submodule pointer.
9. Add Cloudflare evidence when a supported read-only path exists.

This file is the current verified summary. Detailed history belongs in `.github/seo-data/daily/` and the merged daily pull requests.
