# SEO status

## Current state

- Authoritative task: `DAILY_TASK.md`
- Technical SEO subtask: `.github/seo-data/daily-task.md`
- Daily Report subtask: `.agents/skills/eunomia-research-report/SKILL.md`
- External daily scheduler: configured and enabled
- Verified raw Google export window: through `2026-08-23`
- Search Console verified latest finalized row: `2026-08-22`; `2026-08-23` is absent
- Latest finalized GA4 weekly organic landing-page aggregate: `2026-08-17` through `2026-08-23`
- Latest daily record before the current run: `2026-08-25`
- Last completed Daily Report pull request: `#173`
- Last verified Daily Report squash commit: `09e2947bcdf4e93f6a07f7769a15e805059378d0`
- Last verified production publication from a Daily Report run: static export commit `f57b7a7a16281a4e488a65a5eaa71823012c364a`
- Current daily branch: `daily/2026-08-26-ebpf-authorization-revocation`
- Skill submodule commit: `516e9e2dcf012506a677a749049d64c5914643e9`

PR `#173` is fully closed out. It squash-merged as
`09e2947bcdf4e93f6a07f7769a15e805059378d0`; the exact merge commit passed
`Validate SEO Operations` run `32875835650` and `Deploy Static App` run
`32875835638`. Production published static export commit
`f57b7a7a16281a4e488a65a5eaa71823012c364a`, whose commit message binds the
export to that squash SHA. The previously missing single merged-PR closeout
comment was repaired on `2026-08-26` before the new public change; exact EN/ZH
static artifacts were verified for canonical, hreflang, Article JSON-LD, Daily
Report navigation, and report content.

## Current Daily Report mix

Before the `2026-08-26` report, the actually published newest ten reports are:

- eBPF-centered: **7 of 10**
- pure Agent-centered: **0 of 10**
- adjacent systems: **3 of 10**

Today's `/research/ebpf-authorization-revocation/` report is **eBPF-centered**.
It asks how long a cached allow may remain usable after policy or identity
revocation when authority has already been materialized in conntrack, auth maps,
socket-local state, endpoint policy, or other datapath objects. It develops scoped
revocation epochs, a cross-layer completion barrier, and a benchmark based on the
last stale allow after revocation. The incoming report ages another eBPF-centered
report out, so the newest ten remain **7 eBPF / 0 pure Agent / 3 adjacent**.

The active series remains **eBPF Networking and Security**. The `2026-08-23`
report established multi-owner policy composition and authority provenance. The
`2026-08-24` report established zero-copy buffer ownership and policy provenance.
The `2026-08-25` report separated verifier safety from legal temporal state
transitions. The `2026-08-26` report advances a fourth distinct boundary by
separating policy rollout from time-bounded invalidation of authorization that has
already been cached in the fast path.

## Current signals

- Google Drive configured evidence: directly reverified `2026-08-26`; the newest source-native weekly set remains `2026-08-17..23`
- Public homepage: fresh generated brief reports `200` in `166 ms`
- Current production `robots.txt`: reachable; current sitemap reachable
- Current sitemap entries observed by the generated brief: `702`
- Canonical homepage: `https://eunomia.dev/`
- Public GitHub repository evidence: available; fresh generated brief reports 99 active non-fork repositories and 9823 stars across the portfolio
- Public web and primary-source evidence: available
- Google Analytics 4: finalized source-native weekly aggregate through `2026-08-23`
- Google Search Console: newest verified finalized date row `2026-08-22`; complete seven-day comparison still unavailable because required source rows are missing
- Cloudflare: disabled by repository configuration

The longest directly supported equal-duration finalized Search Console comparison
is `2026-08-17..22` versus `2026-08-10..15`: **477 clicks / 59,798 impressions**,
weighted CTR about **0.798%**, and impression-weighted average position about
**9.56**, versus **393 / 66,510**, **0.591%**, and position about **9.91**.
Clicks are about **21.4% higher**, impressions about **10.1% lower**, CTR about
**0.207 percentage points higher**, and average position improves by about
**0.35 positions**.

A complete latest-seven-days versus previous-seven-days GSC comparison remains
unavailable because the latest finalized seven-day window would be
`2026-08-16..22` versus `2026-08-09..15`, while the available date exports omit
both `2026-08-16` and `2026-08-09`. The required 28-day comparison also remains
unavailable because verified export history is too short. No missing row is
interpreted as zero.

The GA4 `2026-08-17..23` aggregate is fully outside the configured lag and is
finalized as a source-native weekly aggregate: **984 sessions** at about
**49.29%** session-weighted engagement, including **118 `(not set)` sessions**,
versus **970 sessions** at about **44.95%** engagement for `2026-08-10..16`.
Sessions are about **1.4% higher** and engagement about **4.34 percentage points
higher**. The weekly aggregate has no date dimension, so it does not support
daily or within-week causal attribution.

## Current technical baseline

The repository generates sitemap, robots, canonical, `hreflang`, Open Graph,
structured data, legacy redirect stubs, and static audit artifacts. Production
deploys through `Deploy Static App`.

The August 26 generated brief reports a reachable homepage, robots file and
sitemap, the expected canonical homepage, and 702 sitemap entries. Current
repository output and available Google evidence do not establish a crawl,
canonical, hreflang, structured-data, redirect, broken-link, rendering,
accessibility, performance, or deployment defect requiring a separate technical
SEO implementation change today.

Today's research uses Linux socket-local storage and sockmap semantics plus
current Cilium stateful policy, endpoint revision, connection-tracking,
authentication-map, and policy-wait behavior. It narrows revocation into a
measurable stale-authority property instead of repeating the general temporal
state-transition contract from August 25.

The SEO skill submodule remains pinned at
`516e9e2dcf012506a677a749049d64c5914643e9`; the consuming repository still
requires contract migration before a pointer-only update.

## Current focus

1. Complete the `2026-08-26` authorization-revocation Daily Report through one non-draft PR, expected CI, complete final diff/generated-output self-review, squash merge, exact production deployment, bilingual production verification, and one merged-PR closeout comment.
2. Keep **eBPF Networking and Security** active after this fourth report; prefer a distinct next question around a security property that must survive kernel/userspace/NIC/DPU placement rather than repeating composition, zero-copy ownership, state transition, or revocation.
3. Recheck Drive freshness on the next run. The configured folder is accessible and currently contains no weekly set newer than `2026-08-17..23`.
4. Keep complete GSC 7-day and 28-day comparisons unavailable until source history supports them; never fill the missing `2026-08-16` or `2026-08-09` rows with zero.
5. Use finalized date-by-page or date-by-query evidence before attributing current search movement to one page or query family.
6. Treat the finalized GA4 `2026-08-17..23` improvement as aggregate behavioral evidence, not a causal content or SEO result.
7. Investigate GA4 `(not set)` and remaining legacy `/en/` traffic only with richer source-native evidence.
8. Migrate the consuming SEO contract before updating the skill submodule pointer.
9. Add Cloudflare evidence only when a supported read-only path is enabled in repository configuration.

Detailed history belongs in `.github/seo-data/daily/` and merged daily pull requests.
