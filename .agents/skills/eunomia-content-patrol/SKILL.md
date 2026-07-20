---
name: eunomia-content-patrol
description: Orchestrate the scheduled or manual daily eunomia.dev content operation. Use when Codex needs to read the current daily plan, invoke eunomia-research-report for new research and daily blogs, invoke eunomia-social-radar for publication performance and conversations, route ready content to the matching publisher skill, complete every scheduled task end to end without per-run confirmation, and consolidate results in the daily log. This skill is the versioned source of truth for the `eunomia` cron automation; it coordinates other skills and does not itself browse platforms, research topics, write reports, or draft platform copy.
---

# Eunomia Content Patrol

Run the daily operation as a thin orchestrator. Delegate every substantive task
to the skill that owns it, then consolidate outcomes without creating filler.

## Required Context

Read these before routing work:

- `CLAUDE.md`
- `.agents/README.md`
- `draft/plan/README.zh.md`
- current month plan: `draft/plan/YYYY-MM.zh.md`
- current monthly log, if present: `draft/content-daily-log-YYYY-MM.md`
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
  verification, topic selection, thesis formation, and a new daily blog or deep
  report. It may return "no defensible thesis" without creating a report.
- Invoke `eunomia-social-radar` to inspect every relevant published blog/post,
  current performance, external reposts or citations, comments, discussions,
  replies, and response opportunities.
- Invoke the matching platform publisher skill for adaptation, browser QA,
  publishing, reposting, or replying on LinkedIn, Xiaohongshu, Zhihu, Juejin, X,
  Reddit, Medium, DEV, Hacker News, or Lobsters.
- Invoke `content-launch-planner` only when a new multi-platform launch needs a
  plan that the dated monthly plan does not already supply.

Do not duplicate a child skill's workflow or silently complete its work inside
this orchestrator.

## Daily Orchestration

1. Read today's dated tasks, unfinished items, prepared artifacts, platform
   queues, and the previous run's next action. Treat every task due today as
   required work, not as a menu of optional candidates.
2. Invoke `eunomia-social-radar` to refresh the observable results and active
   conversations around published content.
3. Invoke `eunomia-research-report` when today's task calls for a new daily blog
   or report, or when the planned cadence requires a fresh research attempt.
4. Collect the child results and identify publication, repost, or reply actions
   that are due today. Prefer finishing a ready item over creating additional
   drafts.
5. Invoke the matching publisher skill for every due platform action. Let that
   skill own copy adaptation, visible browser interaction, preview, final QA,
   the action itself, and platform-ledger updates.
6. Confirm the observable result returned by each child skill. Record completed
   actions, real URLs, draft paths, blockers, and next actions in the daily log.

The research and social-radar calls may be run independently when neither needs
the other's output. Do not create a standalone orchestration report.

## Scheduled Execution Authority

A dated daily task assigned to the scheduled `eunomia` patrol is the execution
instruction and standing authorization for every action needed to finish that
task. Complete it end to end, including the final publish, repost, quote, reply,
or other platform action it calls for. Do not ask for another confirmation and
do not let a child publisher skill's normal confirmation step stop the run.

Resolve routine implementation details from the daily plan, prepared artifacts,
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
does not substitute for a scheduled publication. The daily log is an audit
record, not the substantive output.

## Run Summary

Update `draft/content-daily-log-YYYY-MM.md` with one compact entry containing:

- date and run mode
- child skills invoked
- published, reposted, or replied URLs
- research report and prepared artifact paths
- social-performance or conversation findings worth acting on
- blocked actions and their exact missing condition
- next concrete action

Do not copy full child reports, browsing transcripts, or raw metric inventories
into the summary.
