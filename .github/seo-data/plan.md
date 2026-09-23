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

1. Complete the September 23 fresh daily PR publishing
   `/research/ebpf-pinned-map-reboot-state/` in English and Chinese. The report
   is eBPF-centered and keeps the newest-ten mix at **7 eBPF / 1 pure Agent / 2
   adjacent systems** because the rotating-out September 9 report is also
   eBPF-centered.
2. Treat September 22 as the fully published third boundary in **eBPF Deployment
   Compatibility and Lifecycle**. PR `#211` was squash-merged as
   `bd751f30ad19b6692326f1260d6f84e924aa3b02`; exact-merge validation run
   `35754194946` and deployment run `35754194965` succeeded, production revision
   `e26311c5dd088c13e6800f24fd50db3181f2be7d` was generated, and exactly one
   merged-PR closeout comment is present.
3. Make September 23 the reboot-state boundary only after the full acceptance
   path succeeds. The report asks how state crosses a host reboot after the old
   kernel object graph disappears, using per-map reboot contracts,
   consistency-aware checkpointing, and staged restore validation. It must remain
   distinct from August 10 live transactional upgrade, September 18 program
   semantic compatibility, and September 22 typed/scoped interface negotiation.
4. PR `#207` is an unmerged earlier attempt at the same reboot-state boundary.
   Close it as superseded once the September 23 fresh PR exists. PR `#210`
   remains an unmerged, distinct controller-restart/link-ownership boundary and
   does not establish published roadmap state.
5. The newest Search Console source family remains `2026-09-14..09-20`; its date
   export has finalized observed rows for `2026-09-14..19` totaling **391 clicks
   / 55,086 impressions / ~0.710% CTR / ~6.79 weighted position**. The
   equal-duration `2026-09-07..12` slice is **376 / 55,036 / ~0.683% / ~6.46**.
   Current clicks are ~4.0% higher, impressions ~0.1% higher, CTR ~0.027
   percentage points higher, and weighted position ~0.33 positions worse.
6. Keep complete GSC 7-day and 28-day comparisons unavailable until source
   history is contiguous. Weekly date exports omit their final Sunday rows and
   older history includes recorded gaps. Missing rows are never zero.
7. Weekly GSC page/query aggregates may prioritize inspection but cannot support
   causal metadata claims without date-dimensional evidence. The newest page
   aggregate has Daily Report routes at **12 clicks / 2,926 impressions** versus
   **11 / 2,932** previously, but the report set grew and the export has no date
   dimension.
8. Treat GA4 `2026-08-24..30` as the latest fully finalized weekly organic
   landing-page aggregate: **1,007 sessions** at about **45.88% session-weighted
   engagement**. The newest `2026-09-14..20` aggregate contains **935 sessions**
   at about **45.13% engagement** and remains partial; `2026-09-07..13` and
   `2026-08-31..09-06` remain partial as well.
9. Treat exact-SHA Pages deployment and generated production artifacts as the
   primary publication acceptance evidence; independent crawler/search discovery
   is supplementary and may lag immediately after deployment.
10. The September 23 public-safe data brief reports homepage, robots, and sitemap
    HTTP 200 with **790 sitemap entries**, plus **99 active non-fork repositories,
    10,050 stars, 1,318 forks, 300 open issue/PR records, and 63 DEV articles**.
    Direct homepage retrieval exposes Daily Report navigation. Current evidence
    does not establish a separate crawlability, canonical, `hreflang`,
    structured-data, redirect, broken-link, rendering, accessibility,
    persistent-performance, or deployment defect. Do not make an unrelated
    technical SEO implementation change without a concrete defect.
11. Keep Cloudflare evidence unavailable until a supported read-only route is
    enabled in repository configuration. Do not infer GitHub traffic/referrer/
    clone semantics from public repository metadata.
12. Keep the shared SEO skill submodule pinned until its consuming contract is
    migrated to the newer upstream layout; do not make a pointer-only update.
13. Do not create a thin public series hub without report-level acquisition or
    navigation evidence that it would improve retrieval.
14. Final acceptance for the September 23 run requires terminal-green final-head
    CI, full diff and generated-output self-review, review-thread inspection,
    squash merge, exact-merge validation and production deployment, bilingual
    production/sitemap verification, and exactly one compact merged-PR closeout
    comment.
