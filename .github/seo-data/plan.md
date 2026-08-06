# SEO plan

## Purpose

Make eunomia.dev a reliable canonical source for eBPF, AI-agent runtime
infrastructure, observability, enforcement, and heterogeneous-systems research.
Optimize for technically useful discovery and citation by people and software
agents without creating low-value, mass-generated content.

`DAILY_TASK.md` is the authoritative operating entrypoint. It combines daily data
analysis, technical SEO/GEO, and high-quality Daily Report production. This file
stores durable goals and constraints, not duplicated scheduler instructions.

## Success signals

- Search Console clicks, impressions, CTR, query/page movement, and
  canonical/indexing state once a public-safe read path is configured; preserve
  each metric's source-native meaning.
- Public-safe GA4 acquisition, engaged-session, landing-page, referral, and
  configured outcome signals once available, used to distinguish discovery from
  useful on-site follow-through.
- Cloudflare traffic, bot, status-code, and cache evidence once a supported
  read-only route is available.
- Stable crawlability, canonical ownership, language alternates, structured data,
  internal links, rendering, and production verification enforced by repository
  checks and live inspection.
- Qualified movement from relevant technical pages to public repositories,
  papers, tutorials, and demos, without inventing a blended SEO score.
- Daily analysis records that explain source coverage, movement, uncertainty,
  competing explanations, and the next discriminating evidence.
- Daily Report publications that expose a concrete systems gap and develop
  implementable, testable directions rather than summarizing a trend.

## Operating constraints

- The external scheduler only invokes `DAILY_TASK.md`; the repository owns all
  operational policy and current state.
- Data analysis runs every day. A missing private source is marked unavailable,
  never inferred as zero.
- Raw analytics, private identifiers, credentials, and personal information stay
  outside Git.
- Each run starts from the latest default branch and uses one fresh branch and one
  real non-draft pull request.
- The daily pull request contains the shared daily record and at most one coherent
  public change. A no-change or data-only run is valid.
- Do not combine unrelated technical SEO and content work. Choose the
  highest-value evidence-backed action and keep other durable work here.
- Required and expected CI must pass before a clean final automated self-review
  and squash merge.
- A public change waits for the exact squash commit's production deployment and
  live verification.
- Do not create a second closeout pull request. Put the verified closeout in one
  compact comment on the merged daily pull request, then refresh `status.md` in
  the next run.
- `.agents/skills/seo-geo` and `.github/seo-skills` own technical SEO mechanics.
- `.agents/skills/eunomia-research-report` owns Daily Report research, quality,
  writing, and publication gates.
- Do not manufacture content, site edits, promotion, or workflow artifacts only
  to satisfy cadence.

Short-term fixes and remediation backlogs belong in GitHub issues. Durable
priorities that can guide a later daily run belong below.

## Current priorities

1. Configure and verify the external daily scheduler with the exact minimal
   invocation in `DAILY_TASK.md`.
2. Establish public-safe read-only Search Console and GA4 coverage; add
   Cloudflare coverage when a supported route is available.
3. Complete the first daily baseline using available live-site and public GitHub
   evidence while unavailable sources remain explicit.
4. Let measured reader/search behavior and primary-source research select future
   technical SEO and Daily Report work.
