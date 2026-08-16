# SEO status

## Current state

- Authoritative task: `DAILY_TASK.md`
- Technical SEO subtask: `.github/seo-data/daily-task.md`
- Daily Report subtask: `.agents/skills/eunomia-research-report/SKILL.md`
- External daily scheduler: configured and enabled
- Scheduler timing: daily, flexible around 08:00 in `America/Los_Angeles`
- Last completed daily run: `2026-08-12`
- Last verified data window: Search Console rows through `2026-08-08`; finalized GA4 weekly export through `2026-08-09`
- Latest daily record: `2026-08-16`
- Last completed public-change pull request: `#155`
- Last verified production deployment from a daily run: `Deploy Static App` run `31837125244` for squash commit `66f5a9e83939d228f3ba8c325c83fc1804f5334b`
- Current daily branch: `daily/2026-08-16-io-uring-bpf-programmability`
- Superseded incomplete pull requests: `#158` and `#159`, both closed without merge
- Skill submodule commit: `516e9e2dcf012506a677a749049d64c5914643e9`

PR `#155` is fully closed out. It squash-merged as `66f5a9e83939d228f3ba8c325c83fc1804f5334b`; the exact commit passed `Validate SEO Operations` run `31837125489` and `Deploy Static App` run `31837125244`. The exact Pages artifact contains both language variants of the async causal-profiling report with the expected gap, directions, conclusion-boundary sections, canonical URLs, reciprocal language alternates plus `x-default`, and Article JSON-LD.

The io_uring report was researched on the superseded August 14/15 attempts, but neither PR `#158` nor `#159` merged. They therefore do not count as completed Daily Report publications. The still-valid report is replayed from current `main` in the August 16 run without reverting unrelated work that landed after those branches were created.

## Current Daily Report mix

The completed archive before the current change contains six reports:

- eBPF-centered: **4 of 6**
  - `/research/async-ebpf-causal-profiler/`
  - `/research/stateful-ebpf-transactional-upgrade/`
  - `/research/ebpf-hook-composition-contract/`
  - `/research/userspace-ebpf-runtime-contract/`
- pure Agent-centered: **2 of 6**
  - `/research/agent-trace-evidence-budget/`
  - `/research/parallel-agent-effect-serializability/`
- adjacent systems: **0 of 6**

The current io_uring report is eBPF-centered. After it merges, the archive becomes **5 eBPF-centered / 2 pure Agent / 0 adjacent systems out of 7**. The two pure-Agent reports already consume the intended Agent budget, so continue to favor eBPF and adjacent systems. The active series remains **eBPF Runtime, Extensibility, and Composition**.

## Current signals

- Repository-generated public-safe operating brief: refreshed `2026-08-16 07:51 UTC`
- Homepage: reachable with HTTP 200 in the current operating brief
- `robots.txt`: reachable; sitemap: reachable
- Sitemap entries observed: **660**
- Canonical homepage: `https://eunomia.dev/`
- Public GitHub repository evidence: available; current brief observes 97 active non-fork repositories and 9,752 stars across the portfolio snapshot
- Public web and primary-source evidence: available
- Google Analytics 4: two adjacent weekly Drive organic landing-page exports available and finalized through `2026-08-09`
- Google Search Console: two adjacent weekly export sets available; current date rows still end on `2026-08-08`
- Newer weekly Google export beginning `2026-08-10`: not found on `2026-08-16`
- Cloudflare: disabled by repository configuration

The Google exports are usable historical evidence but stale relative to today. A complete latest-seven-days versus previous-seven-days GSC comparison remains unavailable because the source does not provide both complete adjacent windows; the required 28-day comparison is also unavailable rather than inferred.

The valid common-duration GSC comparison remains **478 clicks / 69,053 impressions** for `2026-08-03` through `2026-08-08`, versus **409 / 46,327** for `2026-07-28` through `2026-08-02`. Clicks are about **16.9% higher** and impressions about **49.1% higher**. Weighted CTR moved from about **0.883%** to **0.692%**, and impression-weighted average position from about **8.18** to **9.42**. `2026-08-05` contributes **24,516 impressions**, about **35.5%** of the newer slice, and still cannot be attributed to one page or query without date-by-page or date-by-query evidence.

The finalized GA4 comparison remains **991 sessions** at about **44.90%** session-weighted engagement for `2026-08-03` through `2026-08-09`, versus **1,021 sessions** at about **46.72%** for `2026-07-27` through `2026-08-02`. `(not set)` fell from 157 to 116 sessions; excluding it, sessions increased slightly from 864 to 875. These are directional measurement signals, not evidence for an unrelated content or SEO rewrite.

## Current technical baseline

The repository generates sitemap, robots, canonical, `hreflang`, Open Graph, structured data, legacy redirect stubs, and HTTP audit artifacts. Production deploys through `Deploy Static App`.

The `2026-08-16` evidence does not establish a new crawl, canonical, hreflang, structured-data, redirect, broken-link, rendering, accessibility, performance, or deployment defect. The correct site change for this run is the mandatory new Daily Report rather than an unrelated SEO patch.

The io_uring research remains current after revalidation against Linux primary source on August 16: `IORING_REGISTER_BPF_FILTER` is a per-opcode classic-BPF admission mechanism, while `io_uring_bpf_ops` is an eBPF `struct_ops` execution-control interface with io_uring-specific kfuncs and bounded ring-region access. Static restrictions and LSM authority remain separate control layers. The report develops a typed capability contract, versioned ring policy generations, explicit provenance/resource accounting, and a comparative control-boundary benchmark.

The SEO skill submodule remains pinned at `516e9e2dcf012506a677a749049d64c5914643e9`. Upstream remains at `f42128a3f05c73cf10c786a2711c488bb3a14839`; the consuming repository still names interfaces from the pinned layout, so a pointer-only upgrade is not made.

## Current focus

1. Complete the `2026-08-16` io_uring BPF programmability Daily Report through final CI, squash merge, exact production deployment, bilingual public verification, and one merged-PR closeout comment.
2. Continue the active series with the next question: where eBPF execution should live across kernel, userspace, NIC/DPU, GPU-adjacent, and device-side targets, including state placement, verifier assumptions, memory visibility, coordination, and observability/control tradeoffs.
3. Keep required GSC 7-day and 28-day comparisons unavailable until complete source history exists; never treat missing exports as zero.
4. Obtain date-by-page or date-by-query Search Console evidence before attributing the `2026-08-05` impression spike.
5. Investigate GA4 `(not set)` only with richer source-native dimensions before making a measurement or content change.
6. Migrate the consuming SEO contract before updating the skill submodule pointer.
7. Add Cloudflare evidence only when a supported read-only path is enabled in repository configuration.

Detailed history belongs in `.github/seo-data/daily/` and the merged daily pull requests.
