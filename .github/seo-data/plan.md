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

1. Preserve the rolling ten-report mix mechanically from the actually published archive. Before the `2026-09-04` report the newest ten contain **6 eBPF-centered / 0 pure Agent / 4 adjacent systems**. The checkpoint-recovery-consistency report is genuinely adjacent because its central mechanism is a recovery-cut consistency contract, not eBPF instrumentation. Publishing it rotates the `2026-08-24` eBPF-centered report out and produces **5 / 0 / 5**, the lower edge of the normal eBPF band. Never repair the ratio through classification or by publishing an extra report.
2. Treat **eBPF Networking and Security** as complete at its normal six-report boundary after the `2026-08-28` proxy handoff report. Return only when fresh evidence supports a mechanism beyond policy composition, zero-copy ownership, temporal state correctness, revocation, complete mediation, or proxy identity continuity.
3. Treat **GPU and Heterogeneous Runtime Systems** as complete at its normal six-report post-activation boundary after `2026-09-04`. The sequence covers: `2026-08-29` memory-placement evidence under oversubscription; `2026-08-30` instrumentation non-interference and coverage; `2026-08-31` candidate-conditioned allocatability; `2026-09-02` membership/generation continuity; `2026-09-03` semantic observability after megakernel fusion; and `2026-09-04` application-consistent checkpoint/restore. Do not continue with a seventh report merely because another GPU paper exists.
4. The September 4 report must stay distinct from membership/generation continuity. Its invariant is the consistency of a recovery cut across CPU state, CUDA state, peer communication, persistent-kernel state, and externally visible effects. The ground truth is whether recovery corresponds to an allowed application prefix plus legitimate replay, not whether communicator membership is live or state repartitioning succeeds.
5. After the September 4 publication, prefer a genuinely eBPF-centered approved systems question for the next run because the rolling mix is at **5 / 0 / 5**. Apply the full evidence, novelty, usefulness, and evaluation gates; a weak eBPF report is worse than a strong adjacent report that keeps the ratio compliant.
6. Use all verified weekly Search Console and GA4 Drive export sets in every run. As of `2026-09-04`, no export newer than `2026-08-24..30` is present. Its Search Console date file contains rows through `2026-08-29` with `2026-08-30` absent.
7. Record the current Search Console `2026-08-24..29` slice as **436 clicks / 55,594 impressions / ~0.784% CTR / ~10.73 impression-weighted position**. The equal-duration `2026-08-17..22` slice is **477 / 59,798 / ~0.798% / ~9.56**. Current clicks are ~8.6% lower, impressions ~7.0% lower, CTR ~0.013 percentage points lower, and position ~1.17 worse. Label this as a six-day source-native comparison, not a complete seven-day trend.
8. Keep complete GSC 7-day and 28-day comparisons unavailable until source history supports them. The prior weekly export omits `2026-08-23`, the current set omits `2026-08-30`, and other older gaps prevent a complete preceding 28-day source window. Missing rows are never zero.
9. Weekly GSC page aggregates may prioritize inspection but must not be used for page-level causal claims without date-by-page evidence. Daily Report pages show 6 clicks / 1,017 impressions in the current weekly page export versus 5 / 744 previously, but volume is still too small to justify a title, navigation, or metadata change by itself.
10. Treat the GA4 `2026-08-24..30` organic landing-page aggregate as fully finalized. It contains 1,007 sessions at about 45.88% session-weighted engagement. The preceding finalized `2026-08-17..23` aggregate contains 984 sessions at about 49.29% engagement. Sessions are about **2.3% higher** week over week while engagement is about **3.41 percentage points lower**.
11. Use finalized date-by-page or date-by-query evidence before attributing search movement to one page, report, title, or topic family. Weekly page/query aggregates without a date dimension are prioritization evidence, not causal attribution.
12. Treat GA4 `(not set)` and remaining legacy `/en/` traffic as measurement and technical SEO questions that require richer source-native evidence rather than as reasons to steer Daily Report topics.
13. Add Cloudflare coverage only when a supported read-only route is enabled in repository configuration.
14. Use search behavior, GitHub activity, primary research, kernel changes, and production evidence to order questions inside approved eBPF and adjacent systems series.
15. Revisit a dedicated public hub for completed series only after report-level acquisition or navigation evidence shows that it would improve retrieval beyond the existing Daily Report index.
16. Migrate the consuming SEO contract before moving the pinned `seo-skills` submodule to a newer upstream layout. Upstream movement alone is not evidence that a pointer-only bump is safe.
