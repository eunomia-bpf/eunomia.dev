# SEO plan

## Purpose

Make eunomia.dev a reliable canonical source for eBPF, systems infrastructure, observability, profiling, networking, security, runtimes, and heterogeneous systems research. AI-agent infrastructure is a deliberately smaller adjacent topic rather than the center of the publication program.

Optimize for technically useful discovery and citation by people and software agents without creating shallow, repetitive, or trend-driven content.

`DAILY_TASK.md` is the authoritative operating entrypoint. It combines daily data analysis, technical SEO/GEO, and one mandatory new Daily Report. This file stores durable goals and constraints, not duplicated scheduler instructions.

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
- Required and expected CI must pass before a clean final automated self-review and squash merge.
- Every daily report is a public change, so the exact squash commit must deploy successfully and both language pages must be verified.
- Do not create a second closeout pull request. Put the verified closeout in one compact comment on the merged daily pull request, then refresh `status.md` in the next run.
- `.agents/skills/seo-geo` and `.github/seo-skills` own technical SEO mechanics.
- `.agents/skills/eunomia-research-report` owns Daily Report research, quality, writing, and publication gates.

## Current priorities

1. Preserve the rolling ten-report mix mechanically from the published archive. Before the `2026-09-07` publication the newest ten contain **5 eBPF-centered / 0 pure Agent / 5 adjacent systems**. Today's report is genuinely eBPF-centered and rotates the `2026-08-27` eBPF-centered report out, so the window remains **5 / 0 / 5** after publication.
2. **eBPF Optimization and Execution Specialization** is the active series. September 5 established verifier safety versus optimizer equivalence and profile-assumption lifetime. September 6 established architecture-specific implementation eligibility, a portable semantic witness, deterministic fallback, and cross-JIT portability evidence. September 7 advances the separate native-operation trust/TCB boundary: the selected implementation must be independently bound to verifier-approved semantics instead of making every optimizer/backend trusted.
3. Keep later optimization-series reports materially distinct. Good next candidates include safe delegation of higher-level operations with explicit semantic/effect contracts, and debugging/provenance that explains the exact specialized implementation, certificate, optimizer generation, and decision that executed. Do not repackage stale-profile invalidation, architecture capability negotiation, portable fallback, cross-JIT portability, or today's trust/TCB thesis with a new optimizer name.
4. Treat **eBPF Networking and Security** as complete at its six-report boundary after `2026-08-28`; return only for a mechanism beyond policy composition, zero-copy ownership, temporal state correctness, revocation, complete mediation, or proxy identity continuity.
5. Treat **GPU and Heterogeneous Runtime Systems** as complete at its six-report post-activation boundary after `2026-09-04`; do not continue merely because another GPU paper exists.
6. Use all verified weekly Search Console and GA4 Drive export sets in every run. As of `2026-09-07`, a new `2026-08-31..09-06` set is present. GSC rows exist through `2026-09-05`; rows through September 4 are treated as finalized under the configured three-day lag, September 5 as partial, and September 6 is absent.
7. Record finalized GSC `2026-08-31..09-04` as **368 clicks / 53,341 impressions / ~0.690% CTR / ~7.45 impression-weighted position**. Equal-duration `2026-08-24..28` is **398 / 48,044 / ~0.828% / ~10.04**. Current clicks are ~7.5% lower, impressions ~11.0% higher, CTR ~0.139 percentage points lower, and position ~2.59 positions better. Label this as a five-day source-native comparison, not a complete seven-day trend.
8. Keep complete GSC 7-day and 28-day comparisons unavailable until source history is contiguous. Missing `2026-08-30`, absent `2026-09-06`, and older gaps are never converted to zero.
9. The new GA4 `2026-08-31..09-06` weekly landing-page aggregate is partial under the configured lag because it has no date dimension. Do not compare it as a finalized week. The latest fully finalized aggregate remains `2026-08-24..30`: 1,007 sessions at ~45.88% engagement versus 984 at ~49.29% for `2026-08-17..23`.
10. Weekly GSC page/query aggregates may prioritize inspection but must not support page-level causal claims without date-resolved evidence. Use finalized date-by-page or date-by-query evidence before attributing movement to one report, title, metadata change, or topic family.
11. Exact-SHA Pages deployment and generated production artifacts remain stronger publication acceptance evidence than crawler discovery alone.
12. Treat GA4 `(not set)` and remaining legacy `/en/` traffic as measurement or technical SEO questions requiring richer source-native evidence rather than reasons to steer Daily Report topics.
13. Add Cloudflare coverage only when a supported read-only route is enabled in repository configuration.
14. Revisit a dedicated public series hub only after at least three strong reports and report-level acquisition/navigation evidence show retrieval benefit beyond the existing Daily Report index.
15. Migrate the consuming SEO contract before moving the pinned `seo-skills` submodule to a newer upstream layout. Upstream movement alone is not evidence that a pointer-only bump is safe.
