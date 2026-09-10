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

- Search Console clicks, impressions, CTR, query/page movement, and canonical/indexing state; preserve each metric's source-native meaning.
- Public-safe GA4 acquisition, engaged-session, landing-page, referral, and configured outcome signals, used to distinguish discovery from useful on-site follow-through.
- Cloudflare traffic, bot, status-code, and cache evidence once a supported read-only route is available.
- Stable crawlability, canonical ownership, language alternates, structured data, internal links, rendering, and production verification enforced by repository checks and live inspection.
- Qualified movement from relevant technical pages to public repositories, papers, tutorials, and demos, without inventing a blended SEO score.
- Daily analysis records that explain source coverage, movement, uncertainty, competing explanations, and the next discriminating evidence.
- Exactly one new Daily Report per scheduled run, with a rolling editorial mix of 5–7 eBPF-centered reports per 10 and at most 1–2 pure Agent reports per 10.
- Daily Reports that expose a concrete systems gap and develop implementable, testable directions rather than summarizing a trend.

## Operating constraints

- The external scheduler is already configured and only invokes the repository; the repository owns all operational policy and current state.
- Data analysis runs every day. A missing private source is marked unavailable, never inferred as zero.
- Raw analytics, private identifiers, credentials, and personal information stay outside Git.
- Each run starts from the latest default branch and uses one fresh branch and one real non-draft pull request.
- Every scheduled run must add exactly one new bilingual Daily Report. A weak candidate is replaced by another approved question rather than published or converted into a no-report day.
- Technical SEO changes remain evidence-driven and may be skipped on a given day; the Daily Report may not be skipped.
- Keep the rolling topic mix compliant and classify by the report's actual central mechanism, not by superficial keyword mentions.
- Do not combine unrelated technical SEO and content work when that makes the daily pull request incoherent; put unrelated durable SEO work in this plan for a focused follow-up.
- Required and expected CI must pass before a clean final automated self-review and squash merge.
- Every daily report is a public change, so the exact squash commit must deploy successfully and both language pages must be verified.
- Do not create a second closeout pull request. Put the verified closeout in one compact comment on the merged daily pull request, then refresh `status.md` in the next run.
- `.agents/skills/seo-geo` and `.github/seo-skills` own technical SEO mechanics.
- `.agents/skills/eunomia-research-report` owns Daily Report research, quality, writing, and publication gates.

Short-term fixes and remediation backlogs belong in GitHub issues. Durable priorities that can guide a later daily run belong below.

## Current priorities

1. Preserve the rolling ten-report mix mechanically from the actually published archive. Before the `2026-09-10` publication the newest ten contain **5 eBPF-centered / 0 pure Agent / 5 adjacent systems**. The split-operation report is genuinely eBPF-centered because BPF/XDP stage boundaries, BPF offload, host/device state, policy generations, and the semantics of jointly implementing one BPF operation are the core objects. It rotates the `2026-08-30` adjacent GPU-instrumentation report out, so the newest-ten mix becomes **6 / 0 / 4** after publication. Never repair the ratio through classification or by publishing an extra report.
2. **eBPF Optimization and Execution Specialization** is the active series. The September 5 report established verifier safety versus optimizer equivalence and profile-assumption lifetime. September 6 established architecture-specific implementation eligibility, portable semantic witnesses, and deterministic fallback. September 7 established postmortem provenance for the exact specialization generation that executed. September 9 established delegated native-code trust, effect scope, assurance, and artifact identity. September 10 advances a distinct fifth boundary: when one high-level operation is split across host and hardware, correctness additionally requires explicit commit, replay, state-ownership, ordering, and generation semantics.
3. Keep later optimization-series reports materially distinct from these five boundaries. The strongest remaining candidate is optimization evidence that distinguishes a portable semantic contract from one lucky microbenchmark or one JIT backend. Do not repackage verifier equivalence, stale-profile invalidation, cross-JIT fallback, execution provenance, native-operation TCB accounting, or split-operation commit/replay with another optimizer name.
4. Treat **eBPF Networking and Security** as complete at its normal six-report boundary after the `2026-08-28` proxy handoff report. Return only when fresh evidence supports a mechanism beyond policy composition, zero-copy ownership, temporal state correctness, revocation, complete mediation, or proxy identity continuity.
5. Treat **GPU and Heterogeneous Runtime Systems** as complete at its normal six-report post-activation boundary after `2026-09-04`. Do not continue with a seventh report merely because another GPU paper exists. Its covered boundaries are memory-placement evidence, instrumentation non-interference, candidate-conditioned allocatability, membership/generation continuity, semantic observability after megakernel fusion, and application-consistent checkpoint/restore.
6. Use all verified weekly Search Console and GA4 Drive export sets in every run. As rechecked on `2026-09-10`, the newest source set remains `2026-08-31..09-06`. Its Search Console date rows are present through `2026-09-05`, which is now outside the configured three-day lag; `2026-09-06` is absent.
7. Record the newest finalized Search Console `2026-08-31..09-05` slice as **388 clicks / 60,880 impressions / ~0.637% CTR / ~7.35 impression-weighted position**. The equal-duration finalized `2026-08-24..29` slice is **436 / 55,594 / ~0.784% / ~10.73**. Current clicks are ~11.0% lower, impressions ~9.5% higher, CTR ~0.147 percentage points lower, and weighted average position ~3.38 positions better. Label this as a six-day source-native comparison, not a complete seven-day trend.
8. Keep complete GSC 7-day and 28-day comparisons unavailable until source history supports them. The previous weekly export omits `2026-08-30`, the current export omits `2026-09-06`, and older history includes the recorded `2026-08-23` gap and other incompleteness. Missing rows are never zero.
9. Weekly GSC page aggregates may prioritize inspection but must not be used for page-level causal claims without date-by-page evidence. Daily Report routes show **8 clicks / 1,812 impressions** in the newest weekly page export versus **6 / 1,017** previously, but the exports lack date-by-page dimensions and span different published report sets.
10. Treat the GA4 `2026-08-31..09-06` organic landing-page aggregate as the latest fully finalized weekly aggregate: **913 sessions** at about **47.54% session-weighted engagement**. The preceding `2026-08-24..30` aggregate contains **1,007 sessions** at about **45.88% engagement**. Sessions are about **9.3% lower** while engagement is about **1.66 percentage points higher**. Preserve these as source-native weekly metrics and do not infer within-week causality from an export without a date dimension.
11. Use finalized date-by-page or date-by-query evidence before attributing search movement to one page, report, title, or topic family. Weekly page/query aggregates without a date dimension are prioritization evidence, not causal attribution.
12. Treat independent crawler/search discovery as supplementary retrievability evidence. Exact-SHA Pages deployment and generated production artifacts remain the stronger publication acceptance evidence, especially immediately after a deployment when crawler refresh can lag.
13. Treat GA4 `(not set)` and remaining legacy `/en/` traffic as measurement and technical SEO questions that require richer source-native evidence rather than as reasons to steer Daily Report topics.
14. Add Cloudflare coverage only when a supported read-only route is enabled in repository configuration.
15. Use search behavior, GitHub activity, primary research, kernel changes, and production evidence to order questions inside approved eBPF and adjacent systems series.
16. The optimization series reaches five reports with the September 10 publication. Do not create a thin public series hub only to mark the count; revisit a dedicated hub when report-level acquisition or navigation evidence shows it would improve retrieval beyond the existing Daily Report index.
17. Migrate the consuming SEO contract before moving the pinned `seo-skills` submodule to a newer upstream layout. Upstream `main` is newer, but upstream movement alone is not evidence that a pointer-only bump is safe.