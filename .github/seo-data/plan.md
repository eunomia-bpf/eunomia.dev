# SEO plan

## Purpose

Make eunomia.dev a reliable canonical source for eBPF, systems infrastructure,
observability, profiling, networking, security, runtimes, and heterogeneous
systems research. AI-agent infrastructure remains a smaller adjacent topic rather
than the center of the publication program.

Optimize for technically useful discovery and citation without creating shallow,
repetitive, or trend-driven content. DAILY_TASK.md is the authoritative operating
entrypoint; this file stores durable goals and constraints.

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
  the merged daily PR, then reconcile status.md in the next run.
- .agents/skills/seo-geo and .github/seo-skills own technical SEO mechanics;
  .agents/skills/eunomia-research-report owns Daily Report research and quality.
- The recurring operations schedule remains enabled when a source or repository
  operation is blocked; record the blocker rather than stopping the schedule.

## Current priorities

1. Complete the September 25 fresh daily run for
   /research/ebpf-map-reuse-semantic-compatibility/ in English and Chinese. The
   report is eBPF-centered; because the rotating-out September 9 report is also
   eBPF-centered, successful publication keeps the newest-ten mix at
   **7 eBPF / 1 pure Agent / 2 adjacent systems**.
2. Treat map-state reuse as the fourth published boundary in **eBPF Deployment
   Compatibility and Lifecycle** only after the complete acceptance path
   succeeds. Keep it distinct from August 10 whole-application transactional
   upgrade: September 25 decides whether one existing map may be reused directly
   or must migrate/reset/refuse.
3. PR #211 is fully reconciled as the September 22 published interface-negotiation
   boundary: squash commit bd751f30ad19b6692326f1260d6f84e924aa3b02;
   exact-merge validation run 35754194946; exact-merge deployment run
   35754194965; one closeout comment. Do not create another closeout for it.
4. PR #212 is an unmerged stale attempt at the same map-reuse boundary and should
   be closed as superseded by the fresh September 25 run. PR #213 is a different
   unmerged reboot-state attempt and does not count as published state.
5. The newest Search Console source family remains 2026-09-14..09-20, but its
   date export has rows only for 2026-09-14..19: **391 clicks / 55,086
   impressions / ~0.710% CTR / ~6.79 weighted position**. The equal-duration
   2026-09-07..12 slice is **376 / 55,036 / ~0.683% / ~6.46**. Treat this only as
   a six-day comparison.
6. Keep complete GSC seven-day and 28-day comparisons unavailable until source
   history is contiguous. Missing dates are never zero.
7. The newest GSC page aggregate has Daily Report routes at **12 clicks / 2,926
   impressions across 74 rows**, versus **11 / 2,932 across 59 rows** previously.
   Because the report set grew and the export has no date dimension, use this for
   prioritization only, not causal metadata claims.
8. Treat GA4 2026-08-24..30 as the latest fully finalized weekly organic
   landing-page aggregate: **1,007 sessions at ~45.88% weighted engagement**.
   The newest 2026-09-14..20 aggregate is **935 sessions at ~45.13%** and remains
   partial; 2026-09-07..13 remains partial as well.
9. The September 25 public-safe brief reports homepage, robots, and sitemap HTTP
   200, **796 sitemap entries**, canonical https://eunomia.dev/, and no evidence
   of a separate crawlability/canonical/hreflang/structured-data/redirect/
   broken-link/rendering/accessibility/persistent-performance defect. Do not make
   an unrelated technical SEO implementation change without concrete evidence.
10. Treat exact-SHA Pages deployment and generated production artifacts as the
    primary publication acceptance evidence; independent crawler/search discovery
    is supplementary and may lag a fresh deployment.
11. Keep Cloudflare unavailable until a supported read-only route is enabled.
    Do not infer GitHub traffic/referrer/clone semantics from public repository
    metadata.
12. Keep the shared SEO skill submodule pinned at
    516e9e2dcf012506a677a749049d64c5914643e9 until its consuming contract is
    deliberately migrated.
13. After successful September 25 publication, continue the active series only
    with a distinct lifecycle boundary supported by fresh primary evidence. Do
    not repeat capability evidence, cross-kernel behavioral compatibility,
    interface negotiation, or map-state reuse under different names.
14. Keep the recurring operations schedule enabled and recurring through source,
    review, CI, merge, or deployment blockers; report blockers instead of
    stopping scheduling.
