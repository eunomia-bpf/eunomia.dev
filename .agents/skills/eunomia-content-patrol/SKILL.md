---
name: eunomia-content-patrol
description: Orchestrate the scheduled or manual eunomia.dev content operation. Use when an agent needs to read the rolling publication queue, invoke eunomia-social-radar for public performance and conversations, route explicitly authorized platform actions to publisher skills, and confirm end-to-end completion. This skill coordinates monitoring and publishing but does not own Daily Report, research-report, or Weekly Analysis workflows.
---

# Eunomia Content Patrol

Run the content operation as a thin orchestrator. Delegate substantive work to
the owning skill, then confirm the public result without creating filler.

## Required Context

Read these before routing work:

- `CLAUDE.md`
- `.agents/README.md`
- `draft/plan/README.zh.md`
- `draft/plan/publishing-queue.zh.md`
- today's `draft/media/YYYY-MM-DD/` workspace and recent run log, when present
- `.github/publisher/media/README.md`
- `.github/publisher/media/community-feedback.md`
- `.github/publisher/media/not-published.md`
- relevant `.github/publisher/media/platforms/*.json`

## Role Boundary

This skill:

- reads operational state;
- finds the first eligible authorized task;
- invokes the correct child skill;
- checks that preparation, publication, and public-page verification completed;
- updates the queue, ledger, and compact run record when useful.

It does not:

- search news or choose a thesis;
- write a Blog article, platform post, or reply;
- pretend a draft or open PR is a completed publication;
- perform a child publisher's platform work directly.

Daily Report, Weekly Analysis, and research-report production belong to a
separate workflow. Do not invoke, schedule, draft, or publish them from this
patrol.

## Routing Map

- Invoke `eunomia-social-radar` for public performance, citations, comments,
  discussions, and response opportunities.
- Invoke `blog-writer` and `blog-writing-style` for project articles, tutorials,
  releases, engineering explanations, and explicitly human-editorial Blog work.
- Invoke the matching platform publisher for authorized LinkedIn, Xiaohongshu,
  Zhihu, Juejin, X, Reddit, Medium, DEV, Hacker News, or Lobsters actions.
- Invoke `content-launch-planner` only when a new multi-platform launch needs a
  plan not already represented in the queue.

Do not duplicate a child skill's workflow inside this orchestrator.

## Daily Orchestration

1. Read the rolling queue, prepared artifacts, platform ledgers, and the
   previous run's next action. A global pause overrides every item it covers.
   Scan from the top for the first eligible unfinished task explicitly marked
   `排队`; bypass `待确认` and `阻塞` items without letting them stall unrelated
   platforms, while `跳过` records a permanently rejected item.
2. Invoke `eunomia-social-radar` to refresh the observable results and active
   conversations around published content.
3. Collect the child results and identify the publication actions authorized
   for the current window. The normal target is one publication per local
   calendar day. If recent days with eligible queue work ended without a
   confirmed publication because of an operational blocker, carry one catch-up
   slot per missed day. Use those slots on the next eligible tasks, never more
   than one publication per platform in the same day. Intentional pauses and
   days with no eligible task do not create catch-up slots.
4. Invoke the matching publisher skill for each authorized action. Let that
   skill own copy adaptation, its documented API or visible-browser submission
   path, preview, final public-page QA, the action itself, and platform-ledger
   updates. If one action reaches a real platform blocker, mark it `阻塞` with
   the exact recovery condition and continue to the next eligible task; do not
   consume a publication or catch-up slot until a public result is confirmed.
5. Confirm the observable result returned by each child skill. Update the
   rolling queue and platform ledger first. If a separate run record is useful,
   write completed actions, real URLs, artifact paths, blockers, and next
   actions to `draft/media/YYYY-MM-DD/run-log.md`.
6. Confirm that `eunomia-social-radar` appended today's compact checkpoint to
   `.github/publisher/media/community-feedback.md`. Do not copy that checkpoint
   into the run log.

Do not create a standalone orchestration report.

## Scheduled Execution Authority

Each eligible queue item explicitly marked `排队` within today's normal or
catch-up slots is standing authorization to complete the named platform action
end to end, including preparation, preview, publication, public-page QA, and
ledger updates.

Do not ask for another confirmation or let a child publisher's normal
confirmation step stop an authorized scheduled run. Resolve routine details
from the queue, artifacts, ledgers, publisher conventions, and visible account
state. A draft or preview is not completion when the task calls for publication.

No other queue status grants that authority. Manual patrol runs do not inherit
standing authority unless the user explicitly asks to execute the tasks.

Do not mark a task blocked until practical recovery paths have been attempted
and a real external condition prevents completion. Record the attempted action
and exact external condition rather than a generic process objection.

Never infer authorization for private messages, connection requests, follows,
likes, votes, account settings, monetization changes, or deletion.

Medium and DEV publishers use their documented APIs by default; all other
platform actions use normal visible-browser workflows. Never use hidden
platform APIs, background endpoints, or scraping datasets.

## No-Filler Rule

Do not manufacture a visible artifact to satisfy the scheduler. Match the
outcome to the task: a publishing task requires a published item, while a
monitoring task may produce an observation or response candidate. A draft or
prepared artifact does not substitute for a scheduled publication. A run log
is an audit record, not the substantive output, and should not be created only
to satisfy cadence.
Do not create per-article figure inventories, platform-hook notes, publish-QA
notes, or other disposable workflow evidence. Keep necessary checks in working
context and put only final platform artifacts, durable skill lessons, or real
exceptions in the repository.

## Run Summary

When a separate summary is useful, record one compact entry in
`draft/media/YYYY-MM-DD/run-log.md` containing:

- date and run mode
- child skills invoked
- published, reposted, or replied URLs
- prepared artifact paths
- social-performance or conversation findings worth acting on
- blocked actions and their exact missing condition
- next concrete action

Do not copy full child reports, browsing transcripts, or raw metric inventories
into the run log.
Do not create a monthly daily-log file.
