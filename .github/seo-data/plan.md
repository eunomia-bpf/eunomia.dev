# SEO plan

## Purpose

Make eunomia.dev a reliable canonical source for eBPF, systems infrastructure, observability, profiling, networking, security, runtimes, and heterogeneous systems research. AI-agent infrastructure remains a smaller adjacent topic rather than the center of the publication program.

Optimize for technically useful discovery and citation without creating shallow or repetitive content. `DAILY_TASK.md` is the authoritative operating entrypoint.

## Success signals

- Source-native Search Console and GA4 movement with missing/partial coverage labelled explicitly.
- Stable crawlability, canonical ownership, language alternates, structured data, internal links, rendering, and exact production verification.
- Qualified movement from relevant pages to public repositories, papers, tutorials, and demos without inventing a blended SEO score.
- Exactly one new bilingual Daily Report per scheduled run, with 5–7 eBPF-centered reports per newest 10 and at most 1–2 pure Agent reports.
- Reports that expose concrete systems gaps and develop implementable, testable mechanisms.

## Operating constraints

- Analyze every enabled source every run. Missing or partial data is never converted to zero.
- Raw analytics, credentials, private source identifiers, and personal information stay outside Git.
- Each run starts from the latest default branch and uses one fresh branch and one real non-draft PR.
- Every run publishes exactly one new bilingual Daily Report; weak candidates are replaced rather than creating a no-report day.
- Technical SEO changes remain evidence-driven and may be skipped when no concrete defect is established.
- Required and expected CI must be terminal-green before final automated self-review and squash merge.
- The exact squash commit must deploy successfully; both locale pages and the sitemap must be verified.
- Add one compact verified closeout comment to the merged daily PR. Reconcile durable state in the next run.
- Keep the recurring operations schedule enabled when a source or repository operation is blocked.

## Current priorities

1. Complete the September 30 fresh publication of `/research/ebpf-map-reuse-semantic-compatibility/` and its Chinese counterpart. It is eBPF-centered and preserves the newest-ten mix at **7 eBPF / 1 pure Agent / 2 adjacent systems**.
2. Treat the report as the fourth boundary in **eBPF Deployment Compatibility and Lifecycle** only after the complete acceptance path. Its scope is pinned-map state reuse admission: map-definition compatibility, structural schema compatibility, and application semantic compatibility.
3. The stale unmerged PR `#212` covers an earlier attempt at this same boundary and must not be counted as published state. Other unmerged PRs, including `#210` and `#213`, also do not establish roadmap state.
4. Search Console observed finalized rows for `2026-09-21..25` are **342 clicks / 37,406 impressions / ~0.914% CTR / ~6.47 weighted position** versus **349 / 49,798 / ~0.701% / ~6.82** for the equal-duration `2026-09-14..18` slice. Keep this labelled as a five-day comparison.
5. Keep complete GSC seven-day and 28-day comparisons unavailable until the source history is contiguous; never synthesize missing rows.
6. Treat GA4 `2026-09-21..27` as partial at **848 sessions** and about **43.75% engagement**. The latest fully finalized weekly aggregate remains `2026-08-24..30` at **1,007 sessions** and about **45.88% engagement**.
7. The September 30 public-safe brief reports homepage, robots, and sitemap collection successful with **802 sitemap entries** and canonical `https://eunomia.dev/`. No current evidence supports an unrelated technical SEO/GEO implementation change.
8. Keep Cloudflare evidence unavailable until a supported read-only route is enabled.
9. Keep the shared SEO skill submodule pinned until its consuming contract is deliberately migrated; do not make a pointer-only update.
10. After this boundary, continue the active series only with a materially distinct lifecycle problem, such as explicit restored-state identity/provenance, not a renamed form of schema fingerprinting, migration, or shadow validation.
