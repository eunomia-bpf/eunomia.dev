---
name: eunomia-content-patrol
description: Orchestrate the scheduled or manual daily eunomia.dev content operation. Use when Codex needs to read the rolling publication queue, invoke eunomia-research-report when a weekly analysis becomes eligible, invoke eunomia-social-radar for publication performance and conversations, route an explicitly authorized ready item to the matching publisher skill, and consolidate useful results in the dated media workspace. This skill is the versioned source of truth for the `eunomia` cron automation; it coordinates other skills and does not itself browse platforms, research topics, write reports, or draft platform copy.
---

# Eunomia Content Patrol

Run the daily operation as a thin orchestrator. Delegate every substantive task
to the skill that owns it, then consolidate outcomes without creating filler.

## Required Context

Read these before routing work:

- `CLAUDE.md`
- `.agents/README.md`
- `draft/plan/README.zh.md`
- rolling publication queue: `draft/plan/publishing-queue.zh.md`
- today's media workspace and recent run log, if present:
  `draft/media/YYYY-MM-DD/` and `draft/media/YYYY-MM-DD/run-log.md`
- `.github/publisher/media/README.md`
- `.github/publisher/media/not-published.md`
- relevant `.github/publisher/media/platforms/*.json`

Read `draft/content-platform-strategy.zh.md` only when the current plan does not
provide enough direction and a durable strategy decision is actually required.

## Role Boundary

This skill reads operational state, enumerates every task due that day, invokes
the owning skills, checks end-to-end completion, and records a compact run
summary.

It does not:

- search news or choose a research thesis
- browse social platforms or inspect post metrics
- write a radar, blog, deep report, reply, repost, or platform post
- perform platform QA or update platform-specific records directly

Those actions belong to the routed skills below.

## Routing Map

- Invoke `eunomia-research-report` for broad current-news research, source
  verification, topic selection, thesis formation, and a scheduled weekly deep
  report. It may return "no defensible thesis" without creating a report unless
  the rolling queue explicitly marks a public analysis as `排队` and the
  widened research window supports a defensible alternative topic.
- Invoke `eunomia-social-radar` to inspect every relevant published blog/post,
  current performance, external reposts or citations, comments, discussions,
  replies, and response opportunities.
- Invoke the matching platform publisher skill for adaptation, browser QA,
  publishing, reposting, or replying on LinkedIn, Xiaohongshu, Zhihu, Juejin, X,
  Reddit, Medium, DEV, Hacker News, or Lobsters.
- Invoke `content-launch-planner` only when a new multi-platform launch needs a
  plan that the rolling queue does not already supply.

Do not duplicate a child skill's workflow or silently complete its work inside
this orchestrator.

## Daily Orchestration

1. Read the rolling queue, prepared artifacts, platform ledgers, and the
   previous run's next action. A global pause or blocker at the top of the
   queue overrides every item it covers. Only the first eligible unfinished
   task explicitly marked `排队` is due; `待确认` and `阻塞` are not
   publication instructions, while `跳过` records a rejected item.
2. Invoke `eunomia-social-radar` to refresh the observable results and active
   conversations around published content.
3. Invoke `eunomia-research-report` when the weekly-analysis queue item is
   eligible and marked `排队`. Do not infer an extra research report merely
   because no platform item is ready.
   When the queue authorizes an analysis publication, use the reviewed
   `deep-report.zh.md` only as a working source, then move the final Chinese post
   to `docs/blog/posts/<slug>.zh.md`. Keep public `date`, `slug`, `title`,
   `description`, `research_question`, `research_window`, and precise tags;
   remove workflow-only status, cutoff, and working-thesis fields. Once the
   Chinese source is stable, invoke `blog-writer` and `blog-writing-style` to
   produce a high-quality English counterpart at
   `docs/blog/posts/<slug>.md`. Preserve the title's promise, section order,
   facts, examples, numbers, caveats, links, and references. Use idiomatic
   technical English rather than line-locked translation. Limit Chinese edits
   to punctuation, spacing, terminology consistency, and clear language errors.
   Do not substantively rewrite either source. Use `seo-geo` only for technical
   metadata, links, and indexing checks, then run one independent editor pass.
4. Collect the child results and identify the single publication action
   authorized for the current window. Prefer finishing it over creating
   additional drafts.
5. Invoke the matching publisher skill for that action. Let that
   skill own copy adaptation, visible browser interaction, preview, final QA,
   the action itself, and platform-ledger updates.
6. Confirm the observable result returned by each child skill. Update the
   rolling queue and platform ledger first. If a separate run record is useful,
   write completed actions, real URLs, artifact paths, blockers, and next
   actions to `draft/media/YYYY-MM-DD/run-log.md`.

The research and social-radar calls may be run independently when neither needs
the other's output. Do not create a standalone orchestration report.

## Scheduled Execution Authority

The first eligible rolling-queue item explicitly marked `排队` is the
execution instruction and standing authorization for every action needed to
finish that item. Complete it end to end, including the final publish, repost,
quote, reply, or other platform action it calls for. Do not ask for another
confirmation and do not let a child publisher skill's normal confirmation step
stop the run. No other queue status grants this authority.

Resolve routine implementation details from the rolling queue, prepared artifacts,
platform ledgers, repository context, and visible account state. Do not require
the plan to repeat information already available elsewhere, and do not invent
new approval or field-completeness gates. Preview, draft creation, and prepared
copy are intermediate states when the task calls for publication; completion
requires confirming the resulting public URL or observable platform state.

Do not mark a task blocked until practical recovery paths have been attempted
and a real external condition prevents completion, such as unavailable account
access or a platform failure. Record the exact attempted action and external
condition instead of a generic process objection.

Manual patrol runs do not inherit this standing authority unless the user asks
to execute the daily tasks or otherwise authorizes the platform action.

Never infer authorization for DMs, connection requests, follows, likes, votes,
account settings, monetization changes, or deletion. Never use hidden platform
APIs, background endpoints, or scraping datasets.

## No-Filler Rule

Do not manufacture a visible artifact to satisfy the scheduler. Match the
outcome to the task: a publishing task requires a published item, while a
research or monitoring task may produce a report, observation, response
candidate, or evidence-backed no-thesis result. A draft or prepared artifact
does not substitute for a scheduled publication. A run log is an audit record,
not the substantive output, and should not be created only to satisfy cadence.
Do not create per-article figure inventories, platform-hook notes, publish-QA
notes, or other disposable workflow evidence. Keep necessary checks in working
context and put only final platform artifacts, durable skill lessons, or real
exceptions in the repository.

## Run Summary

When a separate run summary is useful, create or update
`draft/media/YYYY-MM-DD/run-log.md` with one compact entry containing:

- date and run mode
- child skills invoked
- published, reposted, or replied URLs
- research report and prepared artifact paths
- social-performance or conversation findings worth acting on
- blocked actions and their exact missing condition
- next concrete action

Do not create a monthly daily-log file. Do not copy full child reports, browsing
transcripts, or raw metric inventories into the dated summary.
