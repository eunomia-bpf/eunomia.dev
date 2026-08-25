# SEO status

## Current state

- Authoritative task: `DAILY_TASK.md`
- Technical SEO subtask: `.github/seo-data/daily-task.md`
- Daily Report subtask: `.agents/skills/eunomia-research-report/SKILL.md`
- External daily scheduler: configured and enabled
- Verified raw Google export window: through `2026-08-23`
- Search Console finalized date rows usable through `2026-08-21`; `2026-08-22` is present but still inside the three-day lag and `2026-08-23` is absent
- Latest finalized GA4 weekly organic landing-page aggregate: `2026-08-10` through `2026-08-16`; the `2026-08-17` through `2026-08-23` aggregate is present but partial under the lag
- Latest daily record: `2026-08-25`
- Last completed Daily Report pull request before the current run: `#172`
- Last verified Daily Report squash commit: `3e8cb15798f07343349d79ee1e6fdb3e8b00135a`
- Last verified production publication from a Daily Report run: static export commit `c85b999fed965933da4f147b866adb81eaea0c4a`
- Current daily branch: `daily/2026-08-25-ebpf-stateful-policy-verification`
- Skill submodule commit: `516e9e2dcf012506a677a749049d64c5914643e9`

PR `#172` is now fully closed out. It squash-merged as
`3e8cb15798f07343349d79ee1e6fdb3e8b00135a`; production published static export
commit `c85b999fed965933da4f147b866adb81eaea0c4a`, whose commit message binds the
export to the exact squash commit. The previously missing merged-PR closeout
comment was repaired on `2026-08-25`; it records green final-head checks, exact
production publication, bilingual report verification, source coverage, topic
classification, and remaining uncertainty.

## Current Daily Report mix

Before the `2026-08-25` report, the actually published newest ten reports are:

- eBPF-centered: **7 of 10**
- pure Agent-centered: **0 of 10**
- adjacent systems: **3 of 10**

Today's `/research/ebpf-stateful-policy-verification/` report is **eBPF-centered**.
It asks how persistent security state in BPF maps can preserve legal temporal
transitions across hooks, CPUs, programs, map pressure, and userspace control
without expanding the Linux verifier into a general-purpose policy model checker.
The incoming report ages another eBPF-centered report out of the newest ten, so
the rolling window remains **7 eBPF / 0 pure Agent / 3 adjacent**.

The active series remains **eBPF Networking and Security**. The `2026-08-23`
report established multi-owner policy composition and authority provenance. The
`2026-08-24` report established zero-copy buffer lease ownership and policy
provenance across AF_XDP, io_uring ZC Rx, DPDK, userspace, and NIC handoffs. The
`2026-08-25` report advances a third distinct correctness boundary: verifier-safe
programs can still implement an invalid security-state trace when persistent map
state is stale, concurrently updated, evicted, capacity-limited, or written by a
stale control plane.

## Current signals

- Google Drive configured evidence: rechecked `2026-08-25`; a new weekly set for `2026-08-17..23` is present
- Public homepage: reachable; fresh generated brief reports `200` in `172 ms`
- Current production `robots.txt`: reachable; current sitemap reachable
- Current sitemap entries observed by the generated brief: `698`
- Canonical homepage: `https://eunomia.dev/`
- Public GitHub repository evidence: available; fresh generated brief reports 99 active non-fork repositories and 9813 stars across the portfolio
- Public web and primary-source evidence: available
- Google Analytics 4: newest raw weekly aggregate through `2026-08-23`, but finalized source-native weekly comparison remains through `2026-08-16`
- Google Search Console: newest raw date row `2026-08-22`; finalized use through `2026-08-21`
- Cloudflare: disabled by repository configuration

For the newest valid equal-duration finalized Search Console comparison,
`2026-08-17` through `2026-08-21` reports **443 clicks / 52,433 impressions**,
weighted CTR about **0.845%**, and impression-weighted average position about
**9.24**. The comparable `2026-08-10` through `2026-08-14` slice reports **360
clicks / 60,080 impressions**, **0.599%** CTR, and position about **9.86**.
Clicks are about **23.1% higher**, impressions about **12.7% lower**, CTR about
**0.246 percentage points higher**, and average position improves by about
**0.63 positions**.

A complete latest-seven-days versus previous-seven-days GSC comparison remains
unavailable because the predecessor window still contains the missing
`2026-08-09` row. The required 28-day comparison is also unavailable because
verified export history remains too short. The new page/query exports cover the
whole source-native week including lagged data, so they are not used to attribute
the finalized improvement to a specific page or query family.

The new GA4 `2026-08-17..23` aggregate contains **984 sessions** and about
**49.29%** session-weighted engagement, including **118 `(not set)` sessions**,
but it is partial under the three-day lag and has no date dimension for trimming.
The latest finalized GA4 comparison therefore remains `2026-08-10..16`: **970
sessions** at about **44.95%** engagement versus **991** at about **44.90%** for
`2026-08-03..09`.

## Current technical baseline

The repository generates sitemap, robots, canonical, `hreflang`, Open Graph,
structured data, legacy redirect stubs, and static audit artifacts. Production
deploys through `Deploy Static App`.

The August 25 evidence shows a reachable homepage, robots file and sitemap, the
expected canonical homepage, and no collector coverage gap. The available Google
evidence and repository output do not establish a crawl, canonical, hreflang,
structured-data, redirect, broken-link, rendering, accessibility, performance, or
deployment defect requiring a separate technical SEO implementation change today.

Today's report uses Linux verifier and map semantics, current Cilium stateful
policy/map/lifecycle behavior, Seccomp-eBPF, VEP, temporal verification of stateful
P4, BPF-DB, and ePass. It develops a small temporal policy-state contract,
verifier-cooperative runtime transition guards, and an adversarial temporal policy
benchmark.

The SEO skill submodule remains pinned at
`516e9e2dcf012506a677a749049d64c5914643e9`; the consuming repository still
requires contract migration before a pointer-only update.

## Current focus

1. Complete the `2026-08-25` stateful-policy-verification Daily Report through one non-draft PR, expected CI, complete final diff/generated-output self-review, squash merge, exact production deployment, bilingual production verification, and one merged-PR closeout comment.
2. Keep **eBPF Networking and Security** active after this third report; prefer a distinct next question such as portable policy execution across kernel/userspace/NIC/DPU targets or another correctness boundary that does not repeat policy composition, state transition, upgrade, or zero-copy ownership work.
3. Recheck the `2026-08-17..23` Google set after the finalization lag; never promote its GA4 aggregate or lagged GSC rows into finalized comparisons early.
4. Keep the complete GSC 7-day and 28-day comparisons unavailable until source history supports them; never treat the missing `2026-08-09` row as zero.
5. Use finalized date-by-page or date-by-query evidence before attributing current search movement to a page or query family.
6. Investigate GA4 `(not set)` and remaining legacy `/en/` traffic only with richer source-native evidence.
7. Migrate the consuming SEO contract before updating the skill submodule pointer.
8. Add Cloudflare evidence only when a supported read-only path is enabled in repository configuration.

Detailed history belongs in `.github/seo-data/daily/` and merged daily pull requests.
