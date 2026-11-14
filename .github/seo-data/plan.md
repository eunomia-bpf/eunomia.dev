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

1. Preserve the rolling ten-report mix mechanically from the actually published archive. After the `2026-09-05` runtime-profile-specialization report the newest ten contain **5 eBPF-centered / 0 pure Agent / 5 adjacent systems**. The `2026-09-06` native-operation-contract report is genuinely eBPF-centered because portable BPF semantics, verifier/JIT responsibility, architecture-native lowering, and BPF backend conformance are the central mechanisms. It rotates the `2026-08-26` eBPF-centered authorization-revocation report out, so the newest-ten mix remains **5 / 0 / 5** after publication. Never repair the ratio through classification or by publishing an extra report.
2. **eBPF Optimization and Execution Specialization** is the active series. The September 5 report established runtime rewrite equivalence and profile-assumption lifetime. The September 6 report advances a distinct second boundary: a hardware-specific native emit must remain a bounded implementation of a verifier-visible portable BPF operation instead of becoming a second semantic authority. It develops typed effect contracts, cross-architecture differential/litmus conformance, and implementation-level provenance with portable fallback and revocation.
3. Keep later optimization-series reports materially distinct from both published boundaries. Good remaining candidates include portable optimization evidence that separates a real cross-backend benefit from one microbenchmark, a generic verifier/JIT optimization IR that carries metadata and hardening obligations, or debugging/provenance mechanisms whose operator failure is broader than native-operation version revocation. Do not repackage verifier-safety-versus-equivalence, stale-profile invalidation, or proof-sequence/native-emit trust with another optimizer name.
4. Treat **eBPF Networking and Security** as complete at its normal six-report boundary after the `2026-08-28` proxy handoff report. Return only when fresh evidence supports a mechanism beyond policy composition, zero-copy ownership, temporal state correctness, revocation, complete mediation, or proxy identity continuity.
5. Treat **GPU and Heterogeneous Runtime Systems** as complete at its normal six-report post-activation boundary after `2026-09-04`. Do not continue with a seventh report merely because another GPU paper exists. Its covered boundaries are memory-placement evidence, instrumentation non-interference, candidate-conditioned allocatability, membership/generation continuity, semantic observability after megakernel fusion, and application-consistent checkpoint/restore.
6. Use all verified weekly Search Console and GA4 Drive export sets in every run. As of `2026-09-06`, no export newer than `2026-08-24..30` is present. Its Search Console date file contains rows through `2026-08-29` with `2026-08-30` absent.
7. Record the current Search Console `2026-08-24..29` slice as **436 clicks / 55,594 impressions / ~0.784% CTR / ~10.73 impression-weighted position**. The equal-duration `2026-08-17..22` slice is **477 / 59,798 / ~0.798% / ~9.56**. Current clicks are ~8.6% lower, impressions ~7.0% lower, CTR ~0.013 percentage points lower, and position ~1.17 worse. Label this as a six-day source-native comparison, not a complete seven-day trend.
8. Keep complete GSC 7-day and 28-day comparisons unavailable until source history supports them. The prior weekly export omits `2026-08-23`, the current set omits `2026-08-30`, and other older gaps prevent a complete preceding 28-day source window. Missing rows are never zero.
9. Weekly GSC page aggregates may prioritize inspection but must not be used for page-level causal claims without date-by-page evidence. Daily Report pages show 6 clicks / 1,017 impressions in the current weekly page export versus 5 / 744 previously, but volume is still too small to justify a title, navigation, or metadata change by itself.
10. Treat the GA4 `2026-08-24..30` organic landing-page aggregate as fully finalized. It contains 1,007 sessions at about 45.88% session-weighted engagement. The preceding finalized `2026-08-17..23` aggregate contains 984 sessions at about 49.29% engagement. Sessions are about **2.3% higher** week over week while engagement is about **3.41 percentage points lower**.
11. Use finalized date-by-page or date-by-query evidence before attributing search movement to one page, report, title, or topic family. Weekly page/query aggregates without a date dimension are prioritization evidence, not causal attribution.
12. Fresh public discovery exposes the September 5 English runtime-profile-specialization report. Treat crawler/search discovery as supplementary retrievability evidence. Exact-SHA Pages deployment and generated production artifacts remain the stronger publication acceptance evidence.
13. Treat GA4 `(not set)` and remaining legacy `/en/` traffic as measurement and technical SEO questions that require richer source-native evidence rather than as reasons to steer Daily Report topics.
14. Add Cloudflare coverage only when a supported read-only route is enabled in repository configuration.
15. Use search behavior, GitHub activity, primary research, kernel changes, and production evidence to order questions inside approved eBPF and adjacent systems series.
16. Revisit a dedicated public hub for a series only after at least three strong reports and report-level acquisition or navigation evidence show that it would improve retrieval beyond the existing Daily Report index.
17. Migrate the consuming SEO contract before moving the pinned `seo-skills` submodule to a newer upstream layout. Upstream movement alone is not evidence that a pointer-only bump is safe.

1. Complete September 22 PR `#211`, which publishes
   `/research/ebpf-kernel-interface-negotiation/` in English and Chinese from a
   fresh branch based on current `main`. The report is eBPF-centered and keeps the
   newest-ten mix at **7 eBPF / 1 pure Agent / 2 adjacent systems** because the
   rotating-out September 7 report is also eBPF-centered.
2. Treat September 22 as the third published boundary in **eBPF Deployment
   Compatibility and Lifecycle** if and only if PR `#211` completes the full
   acceptance path. It is a pre-admission variant-selection boundary: typed
   artifact interface requirements, scoped capability negotiation, and
   dependency-driven compatibility CI for unstable/context-scoped kfunc,
   iterator, `struct_ops`, and provider contracts. It must remain distinct from
   September 15 host capability admission and September 18 post-admission
   semantic compatibility.
3. PR `#208` is an unmerged earlier attempt at the same interface-negotiation
   boundary. The September 22 run deliberately supersedes it from current `main`;
   close `#208` rather than leaving two competing attempts. PRs `#207` and `#210`
   remain unmerged and do not establish published roadmap state.
4. The newest Search Console source family is `2026-09-14..09-20`; its date export
   has finalized observed rows for `2026-09-14..19` totaling **391 clicks / 55,086
   impressions / ~0.710% CTR / ~6.79 weighted position**. The equal-duration
   `2026-09-07..12` slice is **376 / 55,036 / ~0.683% / ~6.46**. Current clicks are
   ~4.0% higher, impressions ~0.1% higher, CTR ~0.027 percentage points higher,
   and weighted position ~0.33 positions worse.
5. Keep complete GSC 7-day and 28-day comparisons unavailable until source
   history is contiguous. Weekly date exports omit their final Sunday rows and
   older history includes recorded gaps. Missing rows are never zero.
6. Weekly GSC page/query aggregates may prioritize inspection but cannot support
   causal metadata claims without date-dimensional evidence. The newest page
   aggregate has Daily Report routes at **12 clicks / 2,926 impressions** versus
   **11 / 2,932** previously, but the report set grew and the export has no date
   dimension.
7. Treat GA4 `2026-08-24..30` as the latest fully finalized weekly organic
   landing-page aggregate: **1,007 sessions** at about **45.88% session-weighted
   engagement**. The newest `2026-09-14..20` aggregate contains **935 sessions** at
   about **45.13% engagement** and remains partial; `2026-09-07..13` and
   `2026-08-31..09-06` remain partial as well.
8. Treat exact-SHA Pages deployment and generated production artifacts as the
   primary publication acceptance evidence; independent crawler/search discovery
   is supplementary and may lag immediately after deployment.
9. The September 22 public-safe data brief reports homepage, robots, and sitemap
   HTTP 200 with **786 sitemap entries**. Direct homepage retrieval exposes Daily
   Report navigation. Current public/repository evidence does not establish a
   separate crawlability, canonical, `hreflang`, structured-data, redirect,
   broken-link, rendering, accessibility, persistent-performance, or deployment
   defect. Do not make an unrelated technical SEO implementation change without
   a concrete defect.
10. Keep Cloudflare evidence unavailable until a supported read-only route is
    enabled in repository configuration. Do not infer GitHub traffic/referrer/
    clone semantics from public repository metadata.
11. Keep the shared SEO skill submodule pinned until its consuming contract is
    migrated to the newer upstream layout; do not make a pointer-only update.
12. Do not create a thin public series hub without report-level acquisition or
    navigation evidence that it would improve retrieval.
13. After September 22, continue the active series only with a materially distinct
    boundary. Pinned-map/persistent-state lifecycle across host reboot or
    replacement remains a candidate if it stays distinct from the August
    transactional-upgrade report and current unmerged attempts.
14. Final acceptance for `#211` requires terminal-green final-head CI, full diff
    and generated-output self-review, review-thread reinspection, squash merge,
    exact-merge validation and production deployment, bilingual production/sitemap
    verification, and exactly one compact merged-PR closeout comment.
