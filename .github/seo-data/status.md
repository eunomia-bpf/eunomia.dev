# SEO status

## Current state

- Authoritative task: `DAILY_TASK.md`
- Technical SEO subtask: `.github/seo-data/daily-task.md`
- Daily Report subtask: `.agents/skills/eunomia-research-report/SKILL.md`
- External daily scheduler: configured and enabled
- Scheduler timing: daily, flexible around 08:00 in `America/Los_Angeles`
- Last completed daily run: `2026-08-09`
- Last verified data window: `2026-07-27` through `2026-08-02`
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
- Google Analytics 4: weekly Drive export available and verified, but the latest configured export lags the current finalizable date
- Google Search Console: six dimension exports for one weekly window available and verified, but the latest configured export lags the current finalizable date
- Cloudflare: not configured

The verified Drive folder still covers one complete Search Console and GA4 weekly date window, `2026-07-27` through `2026-08-02`. Search Console has date, search-appearance, device, country, page, and query dimensions; GA4 has an organic landing-page export. With the configured three-day finalization lag, data through `2026-08-07` is now old enough to be finalizable, so `2026-08-03` through `2026-08-07` are currently unrepresented in the enabled Google exports. Previous-period and 28-day comparisons remain unavailable until more weekly windows accumulate.

For `2026-07-27` through `2026-08-02`, Search Console reports 486 clicks and 55,021 impressions, for a weighted CTR of 0.883% and an impression-weighted average position of approximately 8.21. Desktop accounts for about 92.9% of impressions in the current window. Search appearance includes translated-result traffic, and country rows show broad international discovery, which reinforces the value of maintaining first-class English and Chinese variants.

The GA4 organic landing-page export shows tutorials and CUDA/GPU material among the strongest known landing pages. Its largest row remains `(not set)` with low engagement. The current export is not sufficient to make site-wide conversion or outbound-action claims.

Public GitHub activity on `2026-08-10` includes bpftime runtime work for selectable Frida fuzzy backtracing and the preload host-process contract. These are current engineering signals for the active runtime series, not traffic conversion evidence.

## Current technical baseline

The repository generates sitemap, robots, canonical, `hreflang`, Open Graph, structured data, legacy redirect stubs, and HTTP audit artifacts. Production deploys through the `Deploy Static App` workflow.

The `2026-08-10` technical audit did not establish a new crawl, canonical, hreflang, structured-data, redirect, broken-link, rendering, accessibility, or performance defect that justified an unrelated implementation change. Public search still exposes legacy `/en/` URLs, but the current build intentionally emits canonical redirect stubs for those paths; this remains an observation to monitor rather than a proven new defect. The only coupled SEO work in the current daily change is the bilingual Daily Report index and generated page metadata for the new report.

The SEO skill submodule remains pinned at `516e9e2dcf012506a677a749049d64c5914643e9`. Upstream `AutoArchive/seo-skill` has advanced to `9f0bd4f0b33b28fc22592e5463d95f63cda4d165`, but the newer layout remains incompatible with interfaces named by the current consumer contract. Advancing the pointer still requires a coordinated consumer migration rather than an isolated submodule bump.

## Current focus

1. Complete and verify the `2026-08-10` stateful eBPF transactional-upgrade Daily Report through the required one-PR lifecycle, exact production deployment, bilingual public checks, and merged-PR closeout comment.
2. Continue the **eBPF Runtime, Extensibility, and Composition** series. After the stateful-upgrade report, the preferred next question is what an asynchronous profiler built around modern eBPF would need to reconstruct causality across syscalls, io_uring, work queues, runtime tasks, and application-defined resources with bounded overhead and measurable sampling bias.
3. Keep the rolling mix eBPF-heavy and avoid another pure Agent report while the pure-Agent share remains above the eventual 1–2 per 10 cap.
4. Use the verified weekly GA4 and Search Console exports in every run; explicitly mark their lag relative to the current finalizable date instead of treating missing days as zero.
5. Accumulate enough weekly history for previous-period and 28-day comparisons.
6. Investigate the GA4 `(not set)` landing-page row with a richer export before making a measurement or content change.
7. Revisit legacy `/en/` indexing only if fresh crawl or analytics evidence shows canonical redirect stubs are causing measurable harm.
8. Migrate the consuming SEO contract to the newer `seo-skill` layout before updating the submodule pointer.
9. Add Cloudflare evidence when a supported read-only path exists.

This file is the current verified summary. Detailed history belongs in `.github/seo-data/daily/` and the merged daily pull requests.
