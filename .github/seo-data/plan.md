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

1. Preserve the rolling newest-ten mix mechanically. Before the `2026-09-19`
   publication the actually published window contains **7 eBPF-centered / 1 pure
   Agent / 2 adjacent systems**. The oldest report rotating out today is the
   eBPF-centered `2026-09-06` architecture-specialization report. Today's
   eBPF-centered `/research/ebpf-pinned-map-reboot-state/` report replaces it and
   therefore preserves **7 / 1 / 2**. Never repair the ratio by relabeling older
   work or counting an open or closed-but-unmerged PR as published.
2. Continue **eBPF Deployment Compatibility and Lifecycle** with a boundary
   distinct from both admission and live upgrade. The September 15 report covers
   direct capability evidence for deployment admission. The September 18 report
   covers post-admission application behavior across a kernel upgrade. The
   September 19 report asks what happens when the old kernel object graph is gone:
   a bpffs pin is a live object reference, not durable storage, so reboot recovery
   needs per-map restart policy, a consistent recovery cut, semantic state
   versioning, and post-restore validation.
3. Keep later reports in this active series distinct. Strong remaining boundaries
   include capability negotiation for rapidly evolving kfunc/`struct_ops`/
   iterator interfaces and a sharper reproducible artifact/capability contract
   across distributions. Do not repeat the September 19 reboot-durability,
   checkpoint-consistency, or restore-gate boundary, the August 10 live
   transactional-upgrade protocol, the September 18 behavioral-upgrade boundary,
   or the September 15 version/backport admission boundary.
4. Recheck all verified weekly Search Console and GA4 Drive export sets every run.
   As rechecked on `2026-09-19`, no export newer than `2026-09-07..09-13` is
   present. Search Console contains observed rows through `2026-09-12`; all six
   observed rows are now outside the configured three-day lag, while
   `2026-09-13` is absent.
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
10. Current public retrieval and repository evidence do not establish a separate
    crawlability, canonical, hreflang, structured-data, redirect, rendering,
    accessibility, persistent-performance, or deployment defect. Do not make an
    unrelated technical SEO implementation change without a concrete defect.
11. Keep Cloudflare evidence unavailable until a supported read-only route is
    enabled in repository configuration.
12. Keep the shared SEO skill submodule pinned until its consuming contract is
    migrated to the newer upstream layout; do not make a pointer-only update.
13. Do not create a thin public series hub without report-level acquisition or
    navigation evidence that it would improve retrieval.
14. PR `#206` is fully reconciled after exact-merge CI/deployment verification,
    bilingual production/sitemap verification, and one top-level closeout comment.
    PRs `#200` and `#203` remain open and unmerged; PR `#204` was closed without
    merge. None of those unmerged attempts is counted as published state.
15. Complete PR `#207` through terminal-green final-head CI, full diff and
    generated-output self-review, review-thread reinspection, squash merge,
    exact-merge deployment, bilingual production/sitemap verification, and one
    compact merged-PR closeout comment.
