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

1. Preserve the rolling ten-report mix mechanically. Before the `2026-09-14`
   publication, the newest ten actually published reports contain **7
   eBPF-centered / 0 pure Agent / 3 adjacent systems**. The oldest actually
   published report rotating out is the adjacent `2026-09-02` GPU-membership
   report, so another eBPF-centered report would produce 8 of 10 and is not
   allowed. Never repair the ratio by relabeling older work.
2. Use the September 14 publication as an approved pure-Agent systems detour:
   `/research/agent-tool-retry-effect-idempotency/`. It separates protocol request
   identity, execution-attempt identity, and logical external-effect identity,
   then develops durable effect records, ambiguous-outcome reconciliation, and a
   post-commit fault benchmark. After publication the rolling mix becomes **7 /
   1 / 2**.
3. Keep **eBPF Deployment Compatibility and Lifecycle** as the active normal eBPF
   roadmap. The next rolling-window rotation removes the eBPF-centered
   `2026-09-03` GPU-megakernel report, so one eBPF-centered compatibility report
   can enter on the next run without exceeding 7 of 10. Candidate boundaries
   remain real-kernel/backport feature evidence, verifier behavior drift, CO-RE
   relocation versus semantic compatibility, kfunc/`struct_ops` capability
   negotiation, persistent-state lifecycle across host upgrades, and reproducible
   capability/artifact manifests across distributions.
4. Keep **eBPF Optimization and Execution Specialization**, **eBPF Networking and
   Security**, **eBPF Observability and Profiling**, **eBPF Runtime,
   Extensibility, and Composition**, and **GPU and Heterogeneous Runtime Systems**
   closed at their normal boundaries unless fresh evidence supports a genuinely
   new mechanism.
5. Recheck all verified weekly Search Console and GA4 Drive export sets every run.
   As rechecked on `2026-09-14`, the newest source set is now
   `2026-09-07..09-13`. Search Console contains rows through `2026-09-12`; under
   the configured three-day lag rows through `2026-09-11` are finalized,
   `2026-09-12` remains partial, and `2026-09-13` is absent.
6. Record the newest finalized Search Console `2026-09-07..11` slice as **343
   clicks / 47,606 impressions / ~0.720% CTR / ~6.47 impression-weighted
   position**. The equal-duration `2026-08-31..09-04` slice is **368 / 53,341 /
   ~0.690% / ~7.45**. Current clicks are ~6.8% lower, impressions ~10.8% lower,
   CTR ~0.031 percentage points higher, and weighted position ~0.98 positions
   better. This is a five-day source-native comparison, not a complete seven-day
   trend.
7. Keep complete GSC 7-day and 28-day comparisons unavailable until source
   history is contiguous. The preceding export omits `2026-09-06`, the newest
   export omits `2026-09-13`, and older history includes the recorded
   `2026-08-23` and `2026-08-30` gaps. Missing rows are never zero.
8. Weekly GSC page/query aggregates may prioritize inspection but cannot support
   causal metadata claims without date-dimensional evidence. In the newest
   partial aggregate, watch the WASI/component-model article at **5,469
   impressions / 3 clicks / ~5.08 average position** and the query `ai large
   language model linux kernel driver development` at **766 impressions / 0
   clicks / ~6.00 average position**. Require finalized date-dimensional evidence
   or a concrete live snippet/technical defect before changing metadata.
9. Treat GA4 `2026-08-24..30` as the latest fully finalized weekly organic
   landing-page aggregate: **1,007 sessions** at about **45.88% session-weighted
   engagement**. The newly available `2026-09-07..13` aggregate contains **880
   sessions** at about **43.52% engagement**, while `2026-08-31..09-06` contains
   **913 sessions** at about **47.54% engagement**. Both newer frozen aggregates
   remain partial because they were generated while lagged dates were present and
   have no date dimension; do not promote either into a finalized trend claim.
10. Treat exact-SHA Pages deployment and generated production artifacts as the
    primary publication acceptance evidence; independent crawler/search discovery
    is supplementary and can lag immediately after deployment.
11. Keep Cloudflare evidence unavailable until a supported read-only route is
    enabled in repository configuration.
12. Keep the shared SEO skill submodule pinned until its consuming contract is
    migrated to the newer upstream layout; do not make a pointer-only update.
13. Do not create a thin public series hub without report-level acquisition or
    navigation evidence that it would improve retrieval.
14. For mutating agent-tool work, distinguish an ambiguous post-dispatch outcome
    from a clean failure. Provider-native idempotency, retention windows, and
    reconciliation evidence should determine retry safety rather than model
    inference from a timeout string.
15. Treat open, unmerged daily PRs as unpublished state for rolling-mix
    arithmetic. PR `#200` is not part of the public index until it is actually
    merged and verified; any future reconciliation of it must recompute the mix
    against then-current published state rather than reuse stale assumptions.
