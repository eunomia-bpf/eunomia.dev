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

1. Preserve the rolling ten-report mix mechanically. Before the `2026-09-11`
   publication the newest ten contain **6 eBPF-centered / 0 pure Agent / 4
   adjacent systems**. Today's optimization-evidence report is genuinely
   eBPF-centered because BPF verifier/JIT/application/workload evidence and BPF
   optimization promotion are the central objects. It rotates the `2026-08-30`
   adjacent GPU-instrumentation report out, producing **7 / 0 / 3**. Never repair
   the ratio through relabeling or an extra report.
2. Close **eBPF Optimization and Execution Specialization** at its normal six-
   report boundary with the September 11 publication. The sequence now covers:
   verifier safety versus optimizer equivalence and profile lifetime; architecture
   eligibility and portable fallback; execution-generation provenance; native-
   operation trust/TCB accounting; cross-backend state-transition semantics; and
   performance-evidence scope plus production promotion.
3. Activate **eBPF Deployment Compatibility and Lifecycle** as the next normal
   eBPF roadmap. Candidate boundaries are real-kernel/backport feature evidence,
   verifier behavior drift, CO-RE relocation versus semantic compatibility,
   kfunc/`struct_ops` capability negotiation, persistent-state lifecycle across
   host upgrades, and reproducible capability/artifact manifests across
   distributions. Keep it distinct from architecture specialization,
   transactional application upgrade, and userspace runtime contracts.
4. The post-September-11 mix sits at the maximum **7 eBPF / 0 Agent / 3
   adjacent**. The next run must not publish an eighth eBPF-centered item while
   the oldest report rotating out is adjacent. If the active eBPF series is
   temporarily blocked by the arithmetic, use a strong approved adjacent-systems
   detour or a quality-qualified limited Agent systems report, then resume the
   active series when the window permits.
5. Keep **eBPF Networking and Security**, **eBPF Observability and Profiling**,
   **eBPF Runtime, Extensibility, and Composition**, and **GPU and Heterogeneous
   Runtime Systems** closed at their normal boundaries unless fresh evidence
   supports a genuinely new mechanism.
6. Recheck all verified weekly Search Console and GA4 Drive export sets every
   run. As rechecked on `2026-09-11`, the newest source set remains
   `2026-08-31..09-06`. Search Console contains rows through `2026-09-05`; under
   the configured three-day lag all observed rows are finalized, while
   `2026-09-06` is absent.
7. Record the newest finalized Search Console `2026-08-31..09-05` slice as **388
   clicks / 60,880 impressions / ~0.637% CTR / ~7.35 impression-weighted
   position**. The equal-duration `2026-08-24..29` slice is **436 / 55,594 /
   ~0.784% / ~10.73**. Current clicks are ~11.0% lower, impressions ~9.5% higher,
   CTR ~0.147 percentage points lower, and weighted position ~3.38 positions
   better. This is a six-day source-native comparison, not a complete seven-day
   trend.
8. Keep complete GSC 7-day and 28-day comparisons unavailable until source
   history is contiguous. The newest export omits `2026-09-06`, the preceding
   export omits `2026-08-30`, and older history includes the recorded
   `2026-08-23` gap. Missing rows are never zero.
9. Weekly GSC page aggregates may prioritize inspection but cannot support
   page-level causal claims without date-by-page evidence. Daily Report routes
   show **8 clicks / 1,812 impressions** in the newest weekly page export versus
   **6 / 1,017** previously, while the report set itself grew.
10. Treat GA4 `2026-08-24..30` as the latest fully finalized weekly organic
    landing-page aggregate: **1,007 sessions** at about **45.88% session-weighted
    engagement**. The newer frozen `2026-08-31..09-06` aggregate contains **913
    sessions** at about **47.54% engagement**, but remains partial because it was
    exported while lagged dates were present and has no date dimension.
11. Use finalized date-by-page or date-by-query evidence before attributing search
    movement to one report, title, or topic family. Current aggregate evidence
    does not justify a title/metadata rewrite.
12. Treat exact-SHA Pages deployment and generated production artifacts as the
    primary publication acceptance evidence; independent crawler/search discovery
    is supplementary and can lag immediately after deployment.
13. Keep Cloudflare evidence unavailable until a supported read-only route is
    enabled in repository configuration.
14. Keep the shared SEO skill submodule pinned until its consuming contract is
    migrated to the newer upstream layout; do not make a pointer-only update.
15. Do not create a thin public series hub without report-level acquisition or
    navigation evidence that it would improve retrieval.