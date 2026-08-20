# SEO status

## Current state

- Authoritative task: `DAILY_TASK.md`
- Technical SEO subtask: `.github/seo-data/daily-task.md`
- Daily Report subtask: `.agents/skills/eunomia-research-report/SKILL.md`
- External daily scheduler: configured and enabled
- Scheduler timing: daily, flexible around 08:00 in `America/Los_Angeles`
- Last completed daily run: `2026-08-19`
- Last verified data window: Search Console rows through `2026-08-15`; GA4 weekly organic landing-page files through `2026-08-16`
- Latest daily record: `2026-08-20`
- Last completed public-change pull request: `#163`
- Last verified production publication from a daily run: static export commit `a0ab9aed6af348eb8d85634339f407eca2122616` for squash commit `38d50fd8584293490cf845f2c4ff710c368dbe88`
- Current daily branch: `daily/2026-08-20-gpu-causal-profiling`
- Skill submodule commit: `516e9e2dcf012506a677a749049d64c5914643e9`

PR `#163` is fully closed out. It squash-merged as
`38d50fd8584293490cf845f2c4ff710c368dbe88`; final PR-head `Deploy Static App`
run `32274536861` and `Validate SEO Operations` run `32274536879` succeeded.
The production static branch contains commit
`a0ab9aed6af348eb8d85634339f407eca2122616` for that exact merge commit, and the
generated English and Chinese profiler-sampling pages contain the expected
canonical URLs, reciprocal language alternates plus `x-default`, Article JSON-LD,
Daily Report navigation, and bilingual report content. The merged PR has the
required compact closeout comment.

## Current Daily Report mix

Before the `2026-08-20` publication, the first complete ten-report window is:

- eBPF-centered: **7 of 10**
- pure Agent-centered: **2 of 10**
- adjacent systems: **1 of 10**

Publishing another eBPF-centered report today would age out the oldest pure-Agent
report and create an **8 eBPF / 1 Agent / 1 adjacent** rolling window, violating
the repository's 5–7 eBPF rule. The selected GPU host/device causal-profiling
report therefore uses the approved GPU and Heterogeneous Runtime Systems queue
and is classified as adjacent systems: eBPF is a useful host evidence source,
but the mechanism also works with other host tracers and depends equally on
CUPTI and CUDA dependency semantics.

After publication, the rolling ten-report window becomes **7 eBPF / 1 pure Agent
/ 2 adjacent out of 10**. The classification is based on the report's actual
central mechanism rather than editorial quota wording.

## Current signals

- Repository-generated public-safe operating brief: refreshed `2026-08-20 08:02 UTC`
- Homepage: HTTP 200 in **189 ms** in the current operating brief
- `robots.txt`: HTTP 200; sitemap: HTTP 200
- Sitemap entries observed: **674**
- Canonical homepage: `https://eunomia.dev/`
- Public GitHub portfolio snapshot: 97 active non-fork repositories, 9,775 stars, 1,280 forks, and 284 open issue/PR records
- DEV publication surface: 60 articles, 43 public reactions, and 4 comments
- Public web and primary-source evidence: available
- Google Analytics 4: weekly organic landing-page exports available through `2026-08-16`; newest complete aggregate is finalized under the repository lag
- Google Search Console: weekly export sets available through the week ending `2026-08-16`; finalized date rows currently end on `2026-08-15`
- Cloudflare: disabled by repository configuration

For the valid equal-duration Search Console comparison, `2026-08-10` through
`2026-08-15` reports **393 clicks / 66,510 impressions**, weighted CTR about
**0.591%**, and impression-weighted average position about **9.91**. The
comparable `2026-08-03` through `2026-08-08` slice reports **478 clicks / 69,053
impressions**, **0.692%** CTR, and position about **9.42**. Clicks are about
**17.8% lower**, impressions about **3.7% lower**, CTR about **0.101 percentage
points lower**, and average position about **0.49 positions worse**. The older
slice includes the unusual `2026-08-05` impression spike, so the movement remains
monitored rather than attributed to one page, query, or Daily Report.

A complete latest-seven-days versus previous-seven-days GSC comparison remains
unavailable because the `2026-08-09` date row is absent across the adjacent
weekly files. The required 28-day comparison is also unavailable rather than
inferred. No newer weekly export set was observed in the configured Drive folder
on `2026-08-20`.

The finalized GA4 organic landing-page export for `2026-08-10` through
`2026-08-16` contains **970 sessions** at about **44.95%** session-weighted
engagement, versus **991 sessions** at about **44.90%** for `2026-08-03` through
`2026-08-09`. Sessions are about **2.1% lower** while engagement is effectively
flat. `(not set)` is **134 versus 116 sessions**; excluding it, sessions are
**836 versus 875**. These remain source-native acquisition and measurement
signals, not evidence for a speculative title, copy, navigation, or site-structure
change.

## Current technical baseline

The repository generates sitemap, robots, canonical, `hreflang`, Open Graph,
structured data, legacy redirect stubs, and HTTP audit artifacts. Production
deploys through `Deploy Static App`.

The `2026-08-20` public-safe operating brief reports HTTP 200 for the homepage,
robots, and sitemap, a canonical homepage of `https://eunomia.dev/`, and 674
sitemap entries. A fresh public fetch of the homepage also returns the expected
canonical site and Daily Report navigation. Current evidence does not establish
a new crawl, canonical, hreflang, structured-data, redirect, broken-link,
rendering, accessibility, performance, or deployment defect. The appropriate
public site change today is therefore the mandatory bilingual Daily Report and
directly coupled report-index/series updates, not an unrelated technical SEO
patch.

Today's research asks whether a GPU profiler can prove which host action caused a
slow kernel when CUDA execution is asynchronous. Current NVIDIA documentation
shows that CUPTI correlation IDs map CUDA API calls to kernel/memcpy/memset
activity, external correlation can connect higher-level IDs to CUDA activity,
CUDA Graph activity exposes graph/node IDs, and kernel activity can expose
queued/submitted timestamps. CUDA stream and event semantics define a partial
order that timestamp sorting alone cannot recover. The report develops a
generation-scoped host-device causal token, a dependency-aware critical-path
graph with explicit unknown/loss states, and an adversarial ground-truth
causality benchmark.

The SEO skill submodule remains pinned at
`516e9e2dcf012506a677a749049d64c5914643e9`. No pointer-only skill update is
mixed into this daily delivery.

## Current focus

1. Complete the `2026-08-20` GPU host/device causality Daily Report through final CI, complete diff/generated-output self-review, squash merge, exact production deployment, bilingual production verification, and one merged-PR closeout comment.
2. Recalculate the rolling ten-report mix before the next topic. Do not publish an eBPF-centered report if it would push the rolling count above 7 of 10.
3. Keep the required complete GSC 7-day and 28-day comparisons unavailable until source history supports them; never treat the missing `2026-08-09` row as zero.
4. Obtain date-by-page or date-by-query evidence before attributing the `2026-08-05` impression spike.
5. Monitor the Search Console click/CTR movement across another finalized period before changing search-facing titles, copy, or navigation.
6. Investigate GA4 `(not set)` and remaining legacy `/en/` traffic only with richer source-native evidence.
7. Migrate the consuming SEO contract before updating the skill submodule pointer.
8. Add Cloudflare evidence only when a supported read-only path is enabled in repository configuration.

Detailed history belongs in `.github/seo-data/daily/` and the merged daily pull requests.
