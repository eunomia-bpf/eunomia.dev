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

## Current priorities

1. Preserve the rolling ten-report mix mechanically. After the `2026-09-11`
   eBPF optimization-evidence publication, the newest ten contain **7
   eBPF-centered / 0 pure Agent / 3 adjacent systems**, which is the allowed eBPF
   maximum. The oldest report rotating out on September 12 is the adjacent
   `2026-08-31` GPU-utilization report, so another eBPF-centered report would
   produce 8 of 10 and is not allowed. Never repair the ratio by relabeling.
2. Keep **eBPF Deployment Compatibility and Lifecycle** as the active normal eBPF
   roadmap. Candidate boundaries remain real-kernel/backport feature evidence,
   verifier behavior drift, CO-RE relocation versus semantic compatibility,
   kfunc/`struct_ops` capability negotiation, persistent-state lifecycle across
   host upgrades, and reproducible capability/artifact manifests across
   distributions. Resume it as soon as the rolling window permits.
3. Use the September 12 publication as an approved adjacent Linux/storage detour:
   `/research/linux-atomic-write-crash-semantics/`. The report separates
   `RWF_ATOMIC` torn-write protection from persistence, ordering, filesystem
   metadata, and application recovery. Because an adjacent report enters while an
   adjacent report rotates out, the newest-ten mix remains **7 / 0 / 3** after
   publication.
4. Keep **eBPF Optimization and Execution Specialization**, **eBPF Networking and
   Security**, **eBPF Observability and Profiling**, **eBPF Runtime,
   Extensibility, and Composition**, and **GPU and Heterogeneous Runtime Systems**
   closed at their normal boundaries unless fresh evidence supports a genuinely
   new mechanism.
5. Recheck all verified weekly Search Console and GA4 Drive export sets every run.
   As rechecked on `2026-09-12`, the newest source set remains
   `2026-08-31..09-06`. Search Console contains rows through `2026-09-05`; under
   the configured three-day lag all observed rows are finalized, while
   `2026-09-06` is absent.
6. Record the newest finalized Search Console `2026-08-31..09-05` slice as **388
   clicks / 60,880 impressions / ~0.637% CTR / ~7.35 impression-weighted
   position**. The equal-duration `2026-08-24..29` slice is **436 / 55,594 /
   ~0.784% / ~10.73**. Current clicks are ~11.0% lower, impressions ~9.5% higher,
   CTR ~0.147 percentage points lower, and weighted position ~3.38 positions
   better. This is a six-day source-native comparison, not a complete seven-day
   trend.
7. Keep complete GSC 7-day and 28-day comparisons unavailable until source
   history is contiguous. The newest export omits `2026-09-06`, the preceding
   export omits `2026-08-30`, and older history includes the recorded
   `2026-08-23` gap. Missing rows are never zero.
8. Weekly GSC page/query aggregates may prioritize inspection but cannot support
   causal metadata claims without date-dimensional evidence. High-impression
   pages with low aggregate CTR are candidates for future measurement, not an
   automatic title/description rewrite.
9. Treat GA4 `2026-08-24..30` as the latest fully finalized weekly organic
   landing-page aggregate: **1,007 sessions** at about **45.88% session-weighted
   engagement**. The newer frozen `2026-08-31..09-06` aggregate contains **913
   sessions** at about **47.54% engagement**, but remains partial because it was
   exported while lagged dates were present and has no date dimension.
10. Treat exact-SHA Pages deployment and generated production artifacts as the
    primary publication acceptance evidence; independent crawler/search discovery
    is supplementary and can lag immediately after deployment.
11. Keep Cloudflare evidence unavailable until a supported read-only route is
    enabled in repository configuration.
12. Keep the shared SEO skill submodule pinned until its consuming contract is
    migrated to the newer upstream layout; do not make a pointer-only update.
13. Do not create a thin public series hub without report-level acquisition or
    navigation evidence that it would improve retrieval.
