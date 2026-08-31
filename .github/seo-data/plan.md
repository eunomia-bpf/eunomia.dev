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
- Required and expected CI must pass before a clean final automated self-review and squash merge.
- Every daily report is a public change, so the exact squash commit must deploy successfully and both language pages must be verified.
- Do not create a second closeout pull request. Put the verified closeout in one compact comment on the merged daily pull request, then refresh `status.md` in the next run.
- `.agents/skills/seo-geo` and `.github/seo-skills` own technical SEO mechanics.
- `.agents/skills/eunomia-research-report` owns Daily Report research, quality, writing, and publication gates.

Short-term fixes and remediation backlogs belong in GitHub issues. Durable priorities
that can guide a later daily run belong below.

## Current priorities

1. Preserve the rolling ten-report mix mechanically from the actually published archive. After the `2026-08-30` adjacent GPU instrumentation-safety report, the newest ten remain **8 eBPF-centered / 0 pure Agent / 2 adjacent systems**. The `2026-08-31` utilization-versus-allocatability report is also honestly adjacent systems; if published, it rotates the August 21 eBPF report out and restores **7 / 0 / 3**. Never repair the ratio through classification or by publishing an extra report.
2. Treat **eBPF Networking and Security** as complete at its normal six-report boundary after the `2026-08-28` proxy handoff report. Return only when fresh evidence supports a mechanism beyond policy composition, zero-copy ownership, temporal state correctness, revocation, complete mediation, or proxy identity continuity.
3. Keep **GPU and Heterogeneous Runtime Systems** as the active normal series. The `2026-08-29` report covers evidence for GPU memory placement under oversubscription. The `2026-08-30` report advances instrumentation non-interference through a probe-effect manifest, resource-budgeted observation, and explicit coverage. The `2026-08-31` report advances a third boundary: utilization is retrospective activity, while allocatability is a counterfactual admission property. It separates hard resource fit, soft interference, and evidence confidence, then develops an allocatability certificate, bounded two-stage admission, and a spare-capacity counterexample benchmark. All three are adjacent systems because eBPF is optional rather than essential to their central mechanisms.
4. For the next normal GPU/runtime report, prefer a fourth distinct invariant. Strong candidates include **distributed GPU coordination with an explicit correctness property**, megakernel internal observability only if it exposes an execution boundary not already covered by launch/causality reports, or another device/runtime boundary with measurable ground truth. Do not repeat memory placement, instrumentation non-interference, or utilization/allocatability with a different tool example.
5. Use all verified weekly Search Console and GA4 Drive export sets in every run. A fresh source-native set for `2026-08-24..30` appeared on `2026-08-31`. The Search Console date file contains rows through `2026-08-29`, with `2026-08-30` absent; under the configured three-day lag, use finalized rows only through `2026-08-28`.
6. Record the current finalized Search Console slice `2026-08-24..28` as an absolute observation: **398 clicks / 48,044 impressions / ~0.828% CTR / ~10.04 impression-weighted position**. Do not compare it to an incomplete preceding five-day window merely to produce a trend number.
7. Keep complete GSC 7-day and 28-day comparisons unavailable until source history actually supports them. Earlier exports omit `2026-08-16` and `2026-08-23`, and the current set omits `2026-08-30`; there is also no complete preceding 28-day source window. Missing rows are never zero.
8. Retain the previously established longest equal-duration finalized comparison as historical context, not as a substitute for current data: `2026-08-17..22` has 477 clicks / 59,798 impressions / ~0.798% CTR / ~9.56 weighted position versus 393 / 66,510 / ~0.591% / ~9.91 for `2026-08-10..15`.
9. Treat the new GA4 `2026-08-24..30` organic landing-page aggregate as **partial** until the whole interval clears the lag. It currently contains 1,007 sessions at about 45.88% session-weighted engagement. The latest fully finalized aggregate remains `2026-08-17..23`: 984 sessions at about 49.29% engagement versus 970 at about 44.95% for `2026-08-10..16`.
10. Use finalized date-by-page or date-by-query evidence before attributing search movement to one page, report, title, or topic family. Weekly page/query aggregates without a date dimension are prioritization evidence, not causal attribution.
11. Treat GA4 `(not set)` and remaining legacy `/en/` traffic as measurement and technical SEO questions that require richer source-native evidence rather than as reasons to steer Daily Report topics.
12. Add Cloudflare coverage only when a supported read-only route is enabled in repository configuration.
13. Use search behavior, GitHub activity, primary research, kernel changes, and production evidence to order questions inside approved eBPF and adjacent systems series.
14. Revisit a dedicated public hub for completed eBPF series only after report-level acquisition or navigation evidence shows that it would improve retrieval beyond the existing Daily Report index.
15. Migrate the consuming SEO contract before moving the pinned `seo-skills` submodule to a newer upstream layout. Upstream movement alone is not evidence that a pointer-only bump is safe.
