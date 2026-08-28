# SEO plan

## Purpose

Make eunomia.dev a reliable canonical source for eBPF, systems infrastructure,
observability, profiling, networking, security, runtimes, and heterogeneous
systems research. AI-agent infrastructure is a deliberately smaller adjacent
topic rather than the center of the publication program.

Optimize for technically useful discovery and citation by people and software
agents without creating shallow, repetitive, or trend-driven content.

`DAILY_TASK.md` is the authoritative operating entrypoint. It combines daily data
analysis, technical SEO/GEO, and one mandatory new Daily Report. This file stores
durable goals and constraints, not duplicated scheduler instructions.

## Success signals

- Search Console clicks, impressions, CTR, query/page movement, and
  canonical/indexing state; preserve each metric's source-native meaning.
- Public-safe GA4 acquisition, engaged-session, landing-page, referral, and
  configured outcome signals, used to distinguish discovery from useful on-site
  follow-through.
- Cloudflare traffic, bot, status-code, and cache evidence once a supported
  read-only route is available.
- Stable crawlability, canonical ownership, language alternates, structured data,
  internal links, rendering, and production verification enforced by repository
  checks and live inspection.
- Qualified movement from relevant technical pages to public repositories,
  papers, tutorials, and demos, without inventing a blended SEO score.
- Daily analysis records that explain source coverage, movement, uncertainty,
  competing explanations, and the next discriminating evidence.
- Exactly one new Daily Report per scheduled run, with a rolling editorial mix of
  5–7 eBPF-centered reports per 10 and at most 1–2 pure Agent reports per 10.
- Daily Reports that expose a concrete systems gap and develop implementable,
  testable directions rather than summarizing a trend.

## Operating constraints

- The external scheduler is already configured and only invokes the repository;
  the repository owns all operational policy and current state.
- Data analysis runs every day. A missing private source is marked unavailable,
  never inferred as zero.
- Raw analytics, private identifiers, credentials, and personal information stay
  outside Git.
- Each run starts from the latest default branch and uses one fresh branch and one
  real non-draft pull request.
- Every scheduled run must add exactly one new bilingual Daily Report. A weak
  candidate is replaced by another approved question rather than published or
  converted into a no-report day.
- Technical SEO changes remain evidence-driven and may be skipped on a given day;
  the Daily Report may not be skipped.
- Keep the rolling topic mix compliant and classify by the report's actual central
  mechanism, not by superficial keyword mentions.
- Do not combine unrelated technical SEO and content work when that makes the
  daily pull request incoherent; put unrelated durable SEO work in this plan for a
  focused follow-up.
- Required and expected CI must pass before a clean final automated self-review
  and squash merge.
- Every daily report is a public change, so the exact squash commit must deploy
  successfully and both language pages must be verified.
- Do not create a second closeout pull request. Put the verified closeout in one
  compact comment on the merged daily pull request, then refresh `status.md` in
  the next run.
- `.agents/skills/seo-geo` and `.github/seo-skills` own technical SEO mechanics.
- `.agents/skills/eunomia-research-report` owns Daily Report research, quality,
  writing, and publication gates.

Short-term fixes and remediation backlogs belong in GitHub issues. Durable
priorities that can guide a later daily run belong below.

## Current priorities

1. Preserve the rolling ten-report mix mechanically from the actually published
   archive. Before and after the `2026-08-28` report, the newest ten remain **7
   eBPF-centered / 0 pure Agent / 3 adjacent systems** because the incoming
   eBPF-centered proxy-identity report ages another eBPF-centered report out.
2. Treat **eBPF Networking and Security** as complete at its normal six-report
   boundary after the `2026-08-28` proxy handoff report. The series now covers
   multi-owner policy composition, zero-copy buffer ownership, temporal state
   correctness, revocation bounds, complete mediation across offload, and
   request/policy identity across an L7 proxy semantic boundary.
3. Make the already queued **GPU and Heterogeneous Runtime Systems** roadmap the
   next active normal series after the August 28 publication. Start with questions
   where host/device runtime boundaries expose a concrete missing abstraction or
   correctness guarantee. Count a report as eBPF-centered only when eBPF or an
   eBPF-like runtime is central to the mechanism; otherwise classify it as
   adjacent systems.
4. Use all verified weekly Search Console and GA4 Drive export sets in every run.
   The configured folder was directly reverified on `2026-08-28`; the newest set
   remains `2026-08-17..23` and no later weekly set was observed.
5. Treat the `2026-08-17..23` GA4 aggregate as finalized: 984 organic
   landing-page sessions at about 49.29% session-weighted engagement versus 970
   at about 44.95% for `2026-08-10..16`. Sessions are about 1.4% higher and
   engagement about 4.34 percentage points higher. The weekly aggregate still
   cannot support within-week causal attribution.
6. Keep complete GSC 7-day and 28-day comparisons unavailable until source
   history actually supports them. The latest finalized seven-day comparison
   would require `2026-08-16..22` versus `2026-08-09..15`, but the date exports
   omit both `2026-08-16` and `2026-08-09`; verified history is also too short
   for the 28-day comparison.
7. Use the longest equal-duration finalized GSC comparison supported by contiguous
   source rows: `2026-08-17..22` has 477 clicks / 59,798 impressions / ~0.798%
   CTR / ~9.56 weighted position versus 393 / 66,510 / ~0.591% / ~9.91 for
   `2026-08-10..15`. Clicks are about 21.4% higher, impressions about 10.1%
   lower, CTR about 0.207 percentage points higher, and weighted position improves
   by about 0.35 positions.
8. Use finalized date-by-page or date-by-query evidence before attributing current
   search movement to one page or query family. Do not turn aggregate movement
   into a title, copy, navigation, or site-structure change without that evidence.
9. Treat GA4 `(not set)` and remaining legacy `/en/` traffic as measurement and
   technical SEO questions that require richer source-native evidence rather than
   as reasons to steer Daily Report topics.
10. Add Cloudflare coverage only when a supported read-only route is enabled in
    repository configuration.
11. Use search behavior, GitHub activity, primary research, kernel changes, and
    production evidence to order questions inside approved eBPF and adjacent
    systems series.
12. Revisit a dedicated public hub for completed eBPF series only after
    report-level acquisition or navigation evidence shows that it would improve
    retrieval beyond the existing Daily Report index.
13. Migrate the consuming SEO contract before moving the pinned `seo-skills`
    submodule to a newer upstream layout. Upstream movement alone is not evidence
    that a pointer-only bump is safe.
