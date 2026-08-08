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

1. Complete the first scheduled daily operation using the root repository
   contract and verify that one new eBPF-centered Daily Report reaches production.
2. Keep upcoming reports strongly eBPF-centered until the current two pure-Agent
   reports no longer dominate the rolling archive; maintain the long-term 5–7/10
   eBPF and 1–2/10 pure-Agent mix.
3. Use the verified Search Console and GA4 Drive exports in every run and build
   enough weekly history for previous-period and 28-day comparisons.
4. Investigate GA4 `(not set)` and remaining legacy `/en/` traffic as technical SEO
   evidence rather than letting those issues distort content selection.
5. Add Cloudflare coverage when a supported read-only route is available.
6. Use search behavior, GitHub activity, primary research, kernel changes, and
   production evidence to order questions inside the approved eBPF and adjacent
   systems series.
