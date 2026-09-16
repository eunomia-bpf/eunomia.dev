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
- Classify reports by the central mechanism, not keywords, and preserve the
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

1. Publish the September 16 adjacent-systems report
   `/research/cxl-memory-tier-isolation/`. It asks whether `cpuset.mems` can be
   treated as a lifetime CXL residency boundary once reclaim demotion, incomplete
   migration, shared pages, and multi-tenant tiering are included. The report
   develops a tier-residency hardwall with named pressure semantics, multi-owner
   shared-page placement, and an adversarial tier-isolation conformance benchmark.
2. Preserve the rolling newest-ten mix mechanically. Before today's publication
   the actually published window is **7 eBPF-centered / 1 pure Agent / 2 adjacent
   systems** and the oldest report rotating out is adjacent. An eBPF-centered
   publication would become **8 / 1 / 1** and violate the configured range;
   today's adjacent report keeps **7 / 1 / 2**. Never repair the ratio by
   relabeling older work or counting an open PR as published.
3. Keep **eBPF Deployment Compatibility and Lifecycle** as the active roadmap,
   but pause it for this one mix-driven detour. On the next run an eBPF-centered
   report rotates out, so the series can resume if evidence supports a distinct
   boundary. Remaining candidates include verifier/behavior drift, CO-RE
   structural relocation versus semantic compatibility, evolving interface
   negotiation, persistent-state lifecycle, and reproducible capability/artifact
   manifests.
4. Recheck all verified weekly Search Console and GA4 Drive export sets every run.
   As rechecked on `2026-09-16`, no export newer than `2026-09-07..09-13` is
   present. Search Console contains rows through `2026-09-12`; under the configured
   three-day lag rows through `2026-09-11` are finalized, `2026-09-12` is partial,
   and `2026-09-13` is absent.
5. Keep the finalized Search Console `2026-09-07..11` slice at **343 clicks /
   47,606 impressions / ~0.720% CTR / ~6.47 impression-weighted position** and the
   equal-duration `2026-08-31..09-04` comparison at **368 / 53,341 / ~0.690% /
   ~7.45**. This is a five-day source-native comparison, not a complete seven-day
   trend. Missing rows are never zero.
6. Keep complete GSC 7-day and 28-day comparisons unavailable until source
   history is contiguous. Weekly GSC page/query aggregates may prioritize
   inspection but cannot support causal metadata claims without date-dimensional
   evidence.
7. Treat GA4 `2026-08-24..30` as the latest fully finalized weekly organic
   landing-page aggregate: **1,007 sessions** at about **45.88% session-weighted
   engagement**. The newer `2026-09-07..13` (**880 / ~43.52%**) and
   `2026-08-31..09-06` (**913 / ~47.54%**) aggregates remain partial.
8. Treat exact-SHA Pages deployment and generated production artifacts as the
   primary publication acceptance evidence; independent crawler/search discovery
   is supplementary.
9. The public-safe brief generated `2026-09-16 12:40 UTC` reports
   homepage/robots/sitemap HTTP 200, a **147 ms** one-shot homepage sample, and
   **768** sitemap entries. Current evidence does not establish a separate
   technical SEO defect, so do not make an unrelated site implementation change
   today.
10. Keep Cloudflare evidence unavailable until a supported read-only route is
    enabled.
11. Keep the shared SEO skill submodule pinned until its consuming contract is
    migrated to the newer upstream layout; do not make a pointer-only update.
12. PR `#202` is fully reconciled with exact-merge CI/deployment evidence and
    exactly one top-level closeout comment. PR `#200` remains open and unmerged
    and is not part of published-state arithmetic.
13. Complete the September 16 daily PR through terminal-green final-head CI, full
    diff/generated-output self-review, review-thread reinspection, squash merge,
    exact-merge deployment, bilingual production/sitemap verification, and one
    compact merged-PR closeout comment.