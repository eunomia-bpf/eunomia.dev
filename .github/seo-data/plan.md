# SEO plan

## Purpose

Make eunomia.dev a reliable canonical source for eBPF, systems infrastructure,
observability, profiling, networking, security, runtimes, and heterogeneous
systems research. AI-agent infrastructure remains a smaller adjacent topic rather
than the center of the publication program.

Optimize for technically useful discovery and citation without creating shallow,
repetitive, or trend-driven content. `DAILY_TASK.md` is the authoritative
operating entrypoint; this file stores durable goals and constraints.

## Success signals

- Search Console clicks, impressions, CTR, query/page movement, and indexing
  evidence, preserving each metric's source-native meaning.
- Public-safe GA4 acquisition, landing-page, engagement, referral, and configured
  outcome signals used to distinguish discovery from useful follow-through.
- Cloudflare traffic, bot, cache, and status evidence once a supported read-only
  route is enabled.
- Stable crawlability, canonical ownership, language alternates, structured data,
  internal links, rendering, and production verification enforced by repository
  checks and exact deployed artifacts.
- Qualified movement from relevant pages to public repositories, papers,
  tutorials, and demos without inventing a blended SEO score.
- Daily records that explain source coverage, movement, uncertainty, competing
  explanations, and the next discriminating evidence.
- Exactly one new bilingual Daily Report per scheduled run, with 5–7
  eBPF-centered reports per newest 10 and at most 1–2 pure Agent reports.
- Reports that expose concrete systems gaps and develop implementable, testable
  directions rather than summarize a trend.

## Operating constraints

- The external recurring schedule invokes the repository; repository files own
  operational policy and current state.
- Analyze every enabled data source every run. Missing or partial data is marked
  unavailable/partial, never inferred as zero.
- Raw analytics, credentials, private source identifiers, and personal
  information stay outside Git.
- Each run starts from the latest default branch and uses one fresh branch and one
  real non-draft pull request.
- Every run publishes exactly one new bilingual Daily Report. A weak candidate is
  replaced rather than converted into a no-report day.
- Technical SEO changes remain evidence-driven and may be skipped when no concrete
  defect is established.
- Classify reports by the central mechanism, not by keywords, and preserve the
  mechanical rolling mix.
- Required and expected CI must be terminal-green before final automated
  self-review and squash merge.
- The exact squash commit must deploy successfully. Both language pages and the
  sitemap must be verified from the production artifact/public site.
- Do not create a second closeout PR. Put one compact verified closeout comment on
  the merged daily PR, then reconcile `status.md` in the next run.
- `.agents/skills/seo-geo` and `.github/seo-skills` own technical SEO mechanics;
  `.agents/skills/eunomia-research-report` owns Daily Report research and quality.
- The recurring operations schedule remains enabled when a source or repository
  operation is blocked; record the blocker rather than stopping the schedule.

## Current priorities

1. Preserve the rolling newest-ten mix mechanically. Before the `2026-09-20`
   publication the actually published window contains **7 eBPF-centered / 1 pure
   Agent / 2 adjacent systems**. The oldest report rotating out today is the
   eBPF-centered `2026-09-06` architecture-specialization report. Today's
   eBPF-centered `/research/ebpf-exception-cleanup-unwind/` report replaces it and
   therefore preserves **7 / 1 / 2**. Never repair the ratio by relabeling older
   work or counting an open or closed-but-unmerged PR as published.
2. Treat September 20 as a material-external-development detour from the active
   **eBPF Deployment Compatibility and Lifecycle** series. The fresh `bpf-next`
   cleanup-landing-pad series spans compiler metadata, verifier resource state,
   libbpf transport, and architecture JIT unwinding. Today's report develops a
   cleanup-effect discipline, auditable cleanup metadata, and cross-JIT
   differential conformance. It must not be presented as a merged production ABI.
3. Resume **eBPF Deployment Compatibility and Lifecycle** after this detour only
   with a boundary materially distinct from September 15 capability admission and
   September 18 post-admission behavioral compatibility. Open PR `#207` (pinned
   state durability) and PR `#208` (evolving interface negotiation) are unmerged
   attempts and do not count as published boundaries, but a fresh run should not
   duplicate their work without first resolving or superseding it deliberately.
4. Recheck all verified weekly Search Console and GA4 Drive export sets every run.
   As rechecked on `2026-09-20`, no export newer than `2026-09-07..09-13` is
   present. Search Console contains observed rows through `2026-09-12`; all six
   observed rows are outside the configured three-day lag, while `2026-09-13` is
   absent.
5. Record the newest finalized Search Console `2026-09-07..12` slice as **376
   clicks / 55,036 impressions / ~0.683% CTR / ~6.46 impression-weighted
   position**. The equal-duration finalized `2026-08-31..09-05` slice is **388 /
   60,880 / ~0.637% / ~7.35**. Current clicks are ~3.1% lower, impressions ~9.6%
   lower, CTR ~0.046 percentage points higher, and weighted position ~0.89
   positions better. This is a six-day source-native comparison, not a complete
   seven-day trend.
6. Keep complete GSC 7-day and 28-day comparisons unavailable until source
   history is contiguous. The preceding export omits `2026-09-06`, the newest
   export omits `2026-09-13`, and older history includes recorded gaps. Missing
   rows are never zero.
7. Weekly GSC page/query aggregates may prioritize inspection but cannot support
   causal metadata claims without date-dimensional evidence. Continue treating
   high-impression/low-click pages and queries as measurement candidates rather
   than automatic title/description changes.
8. Treat GA4 `2026-08-24..30` as the latest fully finalized weekly organic
   landing-page aggregate: **1,007 sessions** at about **45.88% session-weighted
   engagement**. The `2026-09-07..13` aggregate contains **880 sessions** at about
   **43.52% engagement**, while `2026-08-31..09-06` contains **913 sessions** at
   about **47.54% engagement**. Both newer frozen aggregates remain partial
   because they were generated with lagged dates and have no date dimension for
   safe finalized subsetting.
9. Treat exact-SHA Pages deployment and generated production artifacts as the
   primary publication acceptance evidence; independent crawler/search discovery
   is supplementary and can lag immediately after deployment.
10. The September 20 public-safe data brief reports homepage, robots, and sitemap
    HTTP 200 with 780 sitemap entries, and direct homepage retrieval succeeds.
    Current public/repository evidence does not establish a separate crawlability,
    canonical, `hreflang`, structured-data, redirect, broken-link, rendering,
    accessibility, persistent-performance, or deployment defect. Do not make an
    unrelated technical SEO implementation change without a concrete defect.
11. Keep Cloudflare evidence unavailable until a supported read-only route is
    enabled in repository configuration. Do not infer GitHub traffic/referrer/
    clone semantics from public repository metadata.
12. Keep the shared SEO skill submodule pinned until its consuming contract is
    migrated to the newer upstream layout; do not make a pointer-only update.
13. Do not create a thin public series hub without report-level acquisition or
    navigation evidence that it would improve retrieval.
14. PR `#206` is fully reconciled after exact-merge CI/deployment verification and
    one top-level closeout comment. PRs `#200`, `#203`, `#207`, and `#208` remain
    unmerged attempts; PR `#204` was closed without merge. None is counted as
    current published state.
15. Complete the September 20 daily PR through terminal-green final-head CI, full
    diff and generated-output self-review, review-thread reinspection, squash
    merge, exact-merge deployment, bilingual production/sitemap verification, and
    one compact merged-PR closeout comment.
