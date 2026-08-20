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

1. Preserve the rolling ten-report mix mechanically as reports age out. After the
   `2026-08-20` GPU launch-latency report, the most recent ten reports are **7
   eBPF-centered / 1 pure Agent / 2 adjacent systems**. The oldest remaining
   report in that window is the pure-Agent parallel-effect report, so the next
   publication must again be non-eBPF; adding an eBPF-centered report immediately
   would produce 8 eBPF reports in the rolling ten and violate the 5–7 rule.
2. Keep **eBPF Observability and Profiling** as the active series, but defer its
   eBPF-essential questions until the rolling window permits them again. The
   August 20 report is a GPU-profiling adjacent-systems detour about launch-delay
   attribution, distinct from the August 19 sampling-bias detour. For the next
   run, prefer a genuine adjacent systems question or a strong pure-Agent systems
   question rather than relabeling an eBPF mechanism. Once the oldest eBPF report
   starts aging out, return to always-on semantic compression,
   application-defined resource profiling, or GPU host/device causal profiling
   with eBPF as an essential mechanism only when the mix remains compliant.
3. Use all verified weekly Search Console and GA4 Drive export sets in every run.
   On `2026-08-20`, no newer weekly set than `2026-08-10` through `2026-08-16`
   was observed. That GA4 aggregate remains finalized under the three-day lag.
   Search Console still lacks the `2026-08-09` date row, so keep the required
   complete 7-day and 28-day comparisons unavailable until source history really
   supports them.
4. Obtain or generate date-by-page or date-by-query Search Console evidence before
   attributing the `2026-08-05` impression spike to a specific page or query family.
5. Monitor the current Search Console click/CTR decline and homepage/branded
   softness across another finalized period before changing titles, copy, or site
   structure; the current movement is not uniform enough to justify a speculative
   SEO patch.
6. Treat GA4 `(not set)` and remaining legacy `/en/` traffic as measurement and
   technical SEO questions that require richer source-native evidence rather than
   as reasons to steer report topics.
7. Add Cloudflare coverage only when a supported read-only route is enabled in
   repository configuration.
8. Use search behavior, GitHub activity, primary research, kernel changes, and
   production evidence to order questions inside the approved eBPF and adjacent
   systems series.
9. Revisit a dedicated public hub for the completed eBPF runtime series only after
   report-level acquisition or navigation evidence shows that it would improve
   retrieval beyond the existing Daily Report index.
10. Migrate the consuming SEO contract before moving the pinned `seo-skills`
    submodule to a newer upstream layout that removed interfaces this repository
    still calls. Upstream movement alone is not evidence that a pointer-only bump
    is safe.
