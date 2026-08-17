# SEO status

## Current state

- Authoritative task: `DAILY_TASK.md`
- Technical SEO subtask: `.github/seo-data/daily-task.md`
- Daily Report subtask: `.agents/skills/eunomia-research-report/SKILL.md`
- External daily scheduler: configured and enabled
- Scheduler timing: daily, flexible around 08:00 in `America/Los_Angeles`
- Last completed daily run: `2026-08-16`
- Last verified data window: Search Console rows through `2026-08-08`; finalized GA4 weekly export through `2026-08-09`
- Latest daily record: `2026-08-17`
- Last completed public-change pull request: `#160`
- Last verified production deployment from a daily run: `Deploy Static App` run `31958113001` for squash commit `d72d4d4ed89ec4945e0447e3bb1fc079ca4251b0`
- Current daily branch: `daily/2026-08-17-heterogeneous-ebpf-placement`
- Skill submodule commit: `516e9e2dcf012506a677a749049d64c5914643e9`

PR `#160` is fully closed out. It squash-merged as `d72d4d4ed89ec4945e0447e3bb1fc079ca4251b0`; the exact commit passed `Validate SEO Operations` run `31958113036` and `Deploy Static App` run `31958113001`. The merged pull request has one conversation comment and no review comments, consistent with the repository's one-closeout-comment contract.

The published io_uring report moved the archive to seven reports and completed the fifth eBPF-centered entry. The current August 17 branch begins from the latest default branch after that closeout and adds the sixth, final report in the Runtime, Extensibility, and Composition series.

## Current Daily Report mix

Before the current change, the completed archive contains seven reports:

- eBPF-centered: **5 of 7**
  - `/research/io-uring-bpf-programmability/`
  - `/research/async-ebpf-causal-profiler/`
  - `/research/stateful-ebpf-transactional-upgrade/`
  - `/research/ebpf-hook-composition-contract/`
  - `/research/userspace-ebpf-runtime-contract/`
- pure Agent-centered: **2 of 7**
  - `/research/agent-trace-evidence-budget/`
  - `/research/parallel-agent-effect-serializability/`
- adjacent systems: **0 of 7**

The current heterogeneous-placement report is eBPF-centered. After it merges, the archive becomes **6 eBPF-centered / 2 pure Agent / 0 adjacent systems out of 8**. This stays within the editorial policy. It also brings the Runtime, Extensibility, and Composition series to its normal six-report boundary, so the repository roadmap advances the next run to **eBPF Observability and Profiling** rather than extending the current series indefinitely.

## Current signals

- Repository-generated public-safe operating brief: refreshed `2026-08-17 08:05 UTC`
- Homepage: HTTP 200 in the current operating brief, observed in **192 ms**
- `robots.txt`: reachable; sitemap: reachable
- Sitemap entries observed: **664**
- Canonical homepage: `https://eunomia.dev/`
- Public GitHub repository evidence: available; current brief observes 97 active non-fork repositories, 9,756 stars, 1,275 forks, and 278 open issue/PR records across the portfolio snapshot
- DEV publication surface: 59 articles, 43 public reactions, and 4 comments in the current public-safe brief
- Public web and primary-source evidence: available
- Google Analytics 4: two adjacent weekly Drive organic landing-page exports available and finalized through `2026-08-09`
- Google Search Console: two adjacent weekly export sets available; current date rows still end on `2026-08-08`
- Newer weekly Google export beginning `2026-08-10`: not found in the configured Drive folder on `2026-08-17`
- Cloudflare: disabled by repository configuration

The Google exports remain usable historical evidence but are stale relative to today. A complete latest-seven-days versus previous-seven-days GSC comparison remains unavailable because the source does not provide both complete adjacent windows; the required 28-day comparison is also unavailable rather than inferred.

The valid common-duration GSC comparison remains **478 clicks / 69,053 impressions** for `2026-08-03` through `2026-08-08`, versus **409 / 46,327** for `2026-07-28` through `2026-08-02`. Clicks are about **16.9% higher** and impressions about **49.1% higher**. Weighted CTR moved from about **0.883%** to **0.692%**, and impression-weighted average position from about **8.18** to **9.42**. `2026-08-05` contributes **24,516 impressions**, about **35.5%** of the newer slice, and still cannot be attributed to one page or query without date-by-page or date-by-query evidence.

The finalized GA4 comparison remains **991 sessions** at about **44.90%** session-weighted engagement for `2026-08-03` through `2026-08-09`, versus **1,021 sessions** at about **46.72%** for `2026-07-27` through `2026-08-02`. `(not set)` fell from 157 to 116 sessions; excluding it, sessions increased slightly from 864 to 875. These remain directional measurement signals, not evidence for an unrelated content or SEO rewrite.

## Current technical baseline

The repository generates sitemap, robots, canonical, `hreflang`, Open Graph, structured data, legacy redirect stubs, and HTTP audit artifacts. Production deploys through `Deploy Static App`.

The August 17 operating evidence does not establish a new crawl, canonical, hreflang, structured-data, redirect, broken-link, rendering, accessibility, performance, or deployment defect. The current homepage, robots, sitemap, and canonical checks are healthy in the public-safe operating brief. The correct public site change for this run is therefore the mandatory new Daily Report and its directly coupled bilingual index/navigation entries, not an unrelated technical SEO patch.

Today's research separates backend compatibility from execution placement. RFC 9669 provides ISA conformance groups, Linux verifier and hardware-offload code show that context/helper/offload semantics remain target-specific, and primary systems work such as hXDP, gpu_ext, and fabric_ext demonstrates both the value and the specialization cost of moving BPF execution toward NIC, GPU, and fabric-side events. The report therefore develops a placement-aware target manifest, generation-scoped state ownership and migration, and a ground-truth placement/provenance benchmark rather than duplicating the earlier portable-runtime-contract thesis.

The SEO skill submodule remains pinned at `516e9e2dcf012506a677a749049d64c5914643e9`. Upstream is now at `f42128a3f05c73cf10c786a2711c488bb3a14839` and includes a newer operating layout plus off-site visibility collection. This repository still names interfaces from the pinned layout, so an unrelated pointer-only submodule bump is not made in the current daily PR.

## Current focus

1. Complete the `2026-08-17` heterogeneous eBPF execution-placement Daily Report through final CI, complete diff/generated-output self-review, squash merge, exact production deployment, bilingual public verification, and one merged-PR closeout comment.
2. Start the next run inside the active **eBPF Observability and Profiling** series, with page-level memory attribution as the preferred first question unless evidence rejects it.
3. Keep required GSC 7-day and 28-day comparisons unavailable until complete source history exists; never treat missing exports as zero.
4. Obtain date-by-page or date-by-query Search Console evidence before attributing the `2026-08-05` impression spike.
5. Investigate GA4 `(not set)` only with richer source-native dimensions before making a measurement or content change.
6. Migrate the consuming SEO contract before updating the skill submodule pointer.
7. Add Cloudflare evidence only when a supported read-only path is enabled in repository configuration.

Detailed history belongs in `.github/seo-data/daily/` and the merged daily pull requests.
