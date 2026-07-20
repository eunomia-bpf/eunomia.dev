---
name: eunomia-content-patrol
description: Run the scheduled or manual daily eunomia.dev content patrol. Use when Codex needs to execute the Eunomia daily publishing routine, browse current platform posts, identify 1-2 high-value repost or quote opportunities, prepare or publish planned platform content, produce a concise daily technical news/report brief, update media ledgers and daily logs, or revise the daily automation prompt. This skill is the versioned source of truth for the `eunomia` cron automation.
---

# Eunomia Content Patrol

Run the daily Eunomia content and platform patrol as a real operating loop, not
as a checklist that creates filler artifacts.

## Required Context

Read these before acting:

- `CLAUDE.md`
- `.agents/README.md`
- `draft/plan/README.zh.md`
- current month plan: `draft/plan/YYYY-MM.zh.md`
- current monthly log, if present: `draft/content-daily-log-YYYY-MM.md`
- `.github/publisher/media/README.md`
- `.github/publisher/media/not-published.md`
- relevant `.github/publisher/media/platforms/*.json`

Read `draft/content-platform-strategy.zh.md` only when choosing a new direction
not already covered by the current plan.

## Daily Mission

Each run should do one or more meaningful public-facing actions:

- publish or advance planned owned content from the monthly plan
- browse current platform posts and quote/repost 1-2 unusually valuable items
- create a concise daily technical news/report brief from high-signal sources
- prepare or publish a platform-native post using the matching publisher skill
- update a platform ledger after a real publish, repost, or confirmed status

Do not manufacture a "visible artifact" just to satisfy the scheduler. The
daily log is required for auditability, but it does not count as the day's
substantive output.

## Action Priority

1. Honor the dated monthly plan when it names a concrete source, platform, or
   publication target.
2. If a publish-ready owned item is queued or planned, prioritize real publish
   over another draft.
3. If no owned item is ready, run a light platform radar on LinkedIn,
   Xiaohongshu, X, Zhihu, Juejin, or other relevant visible surfaces.
4. If the radar finds exceptional public posts, quote/repost at most 1-2 with
   concise original context.
5. If platform posting is blocked or no item meets the bar, produce a daily
   technical news/report brief using `references/daily-report-template.md`.
6. If nothing clears the quality bar, log "no publish today" with the reason and
   the next concrete unblocker. Do not create filler files.

## Publishing Authorization

This skill carries the standing authorization for the `eunomia` daily cron to
perform real publishing within these bounds:

- publish owned eunomia.dev/GitHub/project content that is named by the current
  monthly plan, a platform ledger, or a prepared draft path
- quote/repost at most 1-2 public third-party posts when they pass the high-value
  repost gate below
- publish through normal visible browser UI for social/media platforms
- use existing repository-supported Medium/DEV publishing workflow only when the
  queued source and destination are clear

Before a final publish/repost click, verify the account, destination, exact
content, visibility, link/media preview, and absence of private strategy,
customer, pricing, unreleased roadmap, or unverifiable claims. If any of those
are unclear, stop at the editor/preview and record the blocker.

Do not send DMs, connection requests, comments, likes, follows, votes, account
settings changes, or monetization actions. Do not use hidden platform APIs,
internal endpoints, background HTTP interfaces, or scraping datasets for
LinkedIn, Xiaohongshu, Zhihu, Juejin, X, Reddit, Medium, DEV, HN, Lobsters, or
similar platforms.

## High-Value Repost Gate

Quote/repost only when all are true:

- the post is public and visible in the normal browser UI
- the topic fits Eunomia's public pillars: eBPF, AI-agent observability,
  runtime governance, policy enforcement, systems safety, GPU/runtime
  performance, open-source infrastructure, or adjacent research
- the post teaches something concrete, reports useful data, announces a
  genuinely inspectable artifact, or opens a worthwhile technical discussion
- the quote adds original technical context, not only praise
- the action will not amplify confidential, misleading, inflammatory, or
  purely promotional content

Skip reposting if the value is merely topical, if the author/source looks
unreliable, or if the only possible response is generic agreement.

## Platform Work

Use matching publisher skills for platform-specific copy and QA:

- `linkedin-publisher` for LinkedIn posts, quote/reposts, articles, or carousels
- `xiaohongshu-publisher` for Xiaohongshu visual notes and carousel scripts
- `zhihu-publisher` and `juejin-publisher` for Chinese long-form or technical
  posts
- `x-publisher`, `reddit-publisher`, `hackernews-publisher`,
  `lobsters-publisher`, `medium-publisher`, and `devto-publisher` when those
  surfaces fit

When a publisher skill says to stop before final publish, treat this patrol
skill as the explicit standing confirmation only for actions that satisfy the
Publishing Authorization section. Otherwise stop and record the reason.

## News/Report Brief

Load `references/daily-report-template.md` when producing a daily news/report
brief. Keep it short and source-grounded: 3-5 items, why each matters, and one
possible Eunomia response. Use the browser for platform-visible posts and
primary sources for papers, repos, releases, or project pages.

## Records

Every run must update `draft/content-daily-log-YYYY-MM.md` with:

- date and run mode
- real publish/repost/report actions completed
- URLs and local draft/report paths
- ledgers changed
- blocked publish attempts and why they stopped
- next concrete action

After real publishing, update the relevant platform JSON and
`.github/publisher/media/published.md` or `.github/publisher/media/not-published.md`.
For reposts, record the external post URL, platform, our quote text or summary,
and why it passed the high-value gate.
