# SEO status

## Current state

- Authoritative task: `DAILY_TASK.md`
- Technical SEO subtask: `.github/seo-data/daily-task.md`
- Daily Report subtask: `.agents/skills/eunomia-research-report/SKILL.md`
- External daily scheduler: configured and enabled
- Verified raw Google export window: through `2026-09-13`
- Search Console newest observed source row: `2026-09-12`; all observed rows through that date are outside the configured three-day lag, while `2026-09-13` is absent
- Latest fully finalized GA4 weekly organic landing-page aggregate: `2026-08-24` through `2026-08-30`
- Newest GA4 weekly organic landing-page aggregate: `2026-09-07` through `2026-09-13`, partial because the frozen export includes lagged dates and has no date dimension
- Last fully reconciled Daily Report run: `2026-09-20`
- Last merged Daily Report pull request: `#209`
- Last Daily Report squash commit: `831291da7f904bfcbc7b207b3f0e8a56e17bca1c`
- Exact-merge `Validate SEO Operations` for `#209`: run `35521744812`, terminal-success
- Exact-merge `Deploy Static App` for `#209`: run `35521744799`, terminal-success
- Production `new` revision accepted for `#209`: `88b778f06ee63d7fdebd8476770068c8157b10cb`
- Merged-PR closeout for `#209`: exactly one compact top-level closeout comment present
- Default-branch tip at the September 21 run start: `728f053cc5b986f14f89d0f3546a594c9e5d4a0b`
- Current daily branch: `daily/2026-09-21-ebpf-link-reconciliation`
- Current daily pull request: `#210`
- Current branch original base: `728f053cc5b986f14f89d0f3546a594c9e5d4a0b`
- SEO skill submodule commit: `516e9e2dcf012506a677a749049d64c5914643e9`
- Shared agent-skills submodule commit used for workflow guidance: `d9791c478f0e251b3f840e82a64e47eed2b43faa`

September 20 is fully reconciled. PR `#209` was squash-merged as `831291da7f904bfcbc7b207b3f0e8a56e17bca1c`; exact-merge validation and deployment passed; deployed bilingual artifacts and sitemap were verified; and exactly one merged-PR closeout comment is present. Subsequent maintenance commits may move `main` and production `new`, so those SHAs are historical evidence for that acceptance event rather than a permanent current-tip assertion.

Open historical Daily Report attempts including `#200`, `#203`, `#207`, and `#208` remain unmerged and are not counted as published state. PR `#204` was closed without merge. PRs `#207` and `#208` reserve, respectively, the host-reboot pinned-state and evolving-interface-negotiation boundaries so the current run does not duplicate them.

## Current Daily Report mix

Before the September 21 publication, the newest ten actually published reports contain:

- eBPF-centered: **7 of 10**
- pure Agent-centered: **1 of 10**
- adjacent systems: **2 of 10**

The oldest report rotating out today is the eBPF-centered `2026-09-07` specialization/provenance report. Today's `/research/ebpf-link-controller-reconciliation/` report is eBPF-centered, so publication preserves the rolling mix at **7 eBPF-centered / 1 pure Agent / 2 adjacent systems**.

The active roadmap is **eBPF Deployment Compatibility and Lifecycle**. Today's boundary is persistent BPF-link ownership after the controller that created the link has crashed or restarted. It is distinct from September 15 capability admission, September 18 post-admission cross-kernel semantics, the open #207 host-reboot state-reconstruction draft, the open #208 evolving-interface negotiation draft, and the August 10 planned transactional-upgrade protocol.

## Current signals

### Google Search Console

The configured Drive folder was directly rechecked on `2026-09-21`. No newer weekly export than `2026-09-07..13` is present, and a direct search for a `2026-09-14`-starting export returned no result. The newest date export contains rows for `2026-09-07..12`, with no `2026-09-13` row. Under the configured three-day lag, all six observed rows through `2026-09-12` are finalized.

The finalized six-day slice `2026-09-07..12` contains **376 clicks / 55,036 impressions / ~0.683% aggregate CTR / ~6.46 impression-weighted average position**. The equal-duration finalized `2026-08-31..09-05` slice contains **388 / 60,880 / ~0.637% / ~7.35**. Relative to that slice, clicks are about **3.1% lower**, impressions about **9.6% lower**, CTR about **0.046 percentage points higher**, and weighted position about **0.89 positions better**.

This remains a six-day source-native comparison, not a complete seven-day trend. The preceding export omits `2026-09-06`, the newest export omits `2026-09-13`, and older history contains recorded gaps. Missing rows are never interpreted as zero. Complete latest-seven-day and 28-day comparable-period analyses remain unavailable.

The newest page export remains dominated by tutorial/reference demand: `/tutorials/1-helloworld/` has 15 clicks / 263 impressions, `/zh/others/cuda-tutorial/04-gpu-architecture/` 13 / 218, and `/zh/tutorials/1-helloworld/` 11 / 147. `/research/gpu-memory-placement-evidence/` has 4 clicks / 231 impressions, so the research archive is discoverable but remains a small share of search demand.

Device traffic remains desktop-heavy: 323 clicks / 50,790 impressions on desktop, 52 / 4,186 on mobile, and 1 / 60 on tablet. Mobile CTR is higher but on much smaller volume, which is not enough evidence for a device-specific site change.

### Google Analytics 4

The finalized `2026-08-24..30` organic landing-page aggregate remains **1,007 sessions** at about **45.88% session-weighted engagement**. The preceding finalized `2026-08-17..23` aggregate contains **984 sessions** at about **49.29% engagement**.

The `2026-09-07..13` aggregate contains **880 sessions** at about **43.52% session-weighted engagement** and remains partial because the frozen export was produced while lagged dates were present and provides no date dimension for safe finalized subsetting. The `2026-08-31..09-06` aggregate contains **913 sessions** at about **47.54% engagement** and remains partial for the same reason. Neither frozen partial aggregate is promoted into a finalized week-over-week claim after the fact.

The newest landing-page aggregate is led by `(not set)` at 103 sessions and unusually low engagement, followed by `/zh/others/cuda-tutorial/04-gpu-architecture/` at 28 sessions, `/` at 25, and `/tutorials/1-helloworld/` at 25. This is retained as a data-quality and demand-shape observation, not treated as proof of a rendering or canonical defect.

### Public and repository technical evidence

The public-safe data brief generated `2026-09-21 14:23 UTC` reports the homepage as HTTP 200 in 196 ms, `robots.txt` and sitemap as HTTP 200, 784 sitemap entries, and canonical `https://eunomia.dev/`. It reports 99 active non-fork repositories, 10,042 stars, 1,313 forks, 302 open issue/PR records, and 63 observed DEV articles. Portfolio counts are contextual evidence, not a blended SEO score or causal acquisition metric.

Current analytics, repository health, and public-site observations do not establish a concrete crawlability, canonical, `hreflang`, structured-data, redirect, broken-link, rendering, accessibility, persistent-performance, or deployment defect that warrants an unrelated technical SEO change today.

Cloudflare remains disabled by repository configuration, so no Cloudflare-grounded traffic, cache, bot, country, or status-code conclusion is made. GitHub traffic/referrer/clone semantics are not exposed by the current public-safe source set and are not inferred.

## Current technical baseline

The repository generates sitemap, robots, canonical, `hreflang`, Open Graph, Article structured data, legacy redirect stubs, and static audit artifacts. Production deploys through `Deploy Static App`.

The SEO skill submodule remains pinned at `516e9e2dcf012506a677a749049d64c5914643e9`. Upstream movement alone is not evidence that a pointer-only update is safe; the consuming contract must be migrated first.

## Current focus

1. Complete PR `#210` through terminal-green final-head CI, complete diff/generated-output self-review, review-thread reinspection, squash merge, exact production deployment, bilingual production verification, sitemap verification, and exactly one compact merged-PR closeout comment.
2. Preserve the mechanical **7 / 1 / 2** newest-ten mix with today's eBPF-centered controller/link-reconciliation report. Future selection must recalculate the actual published window rather than assuming an open or closed-but-unmerged PR was published.
3. Continue **eBPF Deployment Compatibility and Lifecycle** only with a boundary distinct from the September 15, September 18, and September 21 published reports and the reserved open #207/#208 boundaries.
4. Recheck Drive freshness every run. Keep complete GSC 7-day and 28-day comparisons unavailable until source history is contiguous; never fill missing dates with zero.
5. Keep both newer GA4 weekly aggregates explicitly partial until refreshed or date-dimensional evidence supports finalized interpretation.
6. Keep high-impression/low-click candidates as measurement targets rather than automatic metadata-change targets.
7. Keep Cloudflare evidence unavailable until a supported read-only route is enabled.
8. Keep the shared SEO skill pointer unchanged until the consuming-contract migration required by `plan.md` is completed.
9. Do not make unrelated technical SEO changes without a concrete defect.
10. Keep the recurring operations schedule enabled regardless of source or delivery blockers; blockers are recorded rather than used to stop scheduling.
