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

1. Deliver September 28 `/research/ebpf-attachment-target-identity/` in English and Chinese. It is eBPF-centered and rotates out the eBPF-centered September 9 report, preserving **7 eBPF / 1 pure Agent / 2 adjacent systems**.
2. Treat September 22 PR `#211` as fully reconciled published state: squash commit `bd751f30ad19b6692326f1260d6f84e924aa3b02`, exact-merge validation run `35754194946`, exact-merge deployment run `35754194965`, production revision `e26311c5dd088c13e6800f24fd50db3181f2be7d`, and one closeout comment.
3. Continue **eBPF Deployment Compatibility and Lifecycle** with today's distinct target-generation boundary. Do not count unmerged PRs `#210`, `#212`, or `#213` as published state.
4. Treat the newest GSC family `2026-09-14..09-20` as six observed rows through September 19: **391 clicks / 55,086 impressions / ~0.710% CTR / ~6.79 weighted position** versus **376 / 55,036 / ~0.683% / ~6.46** for the equal six-day September 7–12 slice.
5. Keep complete current GSC seven-day and 28-day comparisons unavailable until source history is contiguous. Missing rows are never zero.
6. Treat GA4 `2026-08-24..30` as the latest fully finalized weekly aggregate: **1,007 sessions** at about **45.88% engagement**. Keep `2026-09-14..20` labelled partial at **935 sessions** and about **45.13% engagement**.
7. The September 28 public-safe brief reports homepage, robots, and sitemap HTTP 200 with **802 sitemap entries** and the correct homepage canonical. Do not make an unrelated technical SEO implementation change without evidence of a defect.
8. Keep Cloudflare evidence unavailable until a supported read-only route is enabled.
9. Keep the SEO skill submodule pinned until the consuming contract migration is complete; do not make a pointer-only update.
10. Treat exact-SHA Pages deployment and generated production artifacts as the primary publication acceptance evidence; crawler/search discovery is supplementary.
