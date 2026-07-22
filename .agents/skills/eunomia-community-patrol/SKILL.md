---
name: eunomia-community-patrol
description: Inspect, triage, and actively maintain all open GitHub issues and pull requests across public, non-archived, non-fork eunomia-bpf repositories. Use for recurring Eunomia community patrols, organization-wide issue and PR sweeps, follow-up of prior maintenance comments or pull requests, and scheduled maintenance that may comment, fix verified bugs, push, and open pull requests but must never merge.
---

# Eunomia Community Patrol

Treat this skill as the versioned source of truth for the Eunomia community
maintenance task. Keep runtime state, credentials, private logs, and
deduplication records outside the repository.

## Scope and Schedule

- Operate only in the `eunomia-bpf` GitHub organization.
- Inspect every public repository that is neither archived nor a fork.
- Inspect every open issue and open pull request.
- Perform GitHub writes only in `eunomia-bpf` repositories.
- Run in the Linux maintenance workspace. Do not redirect the task to Windows
  or PowerShell.
- Schedule the patrol for 09:00 `America/Vancouver` every two calendar days.
  Let a successful manual patrol reset the next eligible date. Retry a failed
  run at the next daily scheduler wake and prevent overlapping runs with a
  local lock.
- Resume the designated Codex conversation for each eligible run so the final
  report appears in that conversation. If the target session is unavailable,
  fail without starting a second patrol, preserve the local log, and retry at
  the next scheduler wake.

Keep all new and previously tracked actionable items in the queue. Prioritize
security, confirmed bugs, blocked fixes, needs-info, documentation or support,
and stale items in that order when useful, but never permanently skip a lower-
priority item.

## Start Every Run

1. Read the local automation memory completely. Use it to avoid duplicate
   comments and resume every unresolved item.
2. Read and follow `oss-issue-triage` for issue and pull request
   classification.
3. Refresh the organization repository list instead of relying on the previous
   inventory.
4. Before changing a repository, read and follow `oss-change-workflow` and the
   target repository's `AGENTS.md`, `CONTRIBUTING`, `SECURITY.md`, `README`, and
   relevant workflows.

If this skill or a required repository policy cannot be read, do not guess at
the missing authorization. Record the blocker and continue only with safe,
read-only work elsewhere in scope.

## Inspect Every Item

Check:

- title, state, labels, assignees, author, creation time, and update time;
- latest discussion and latest maintainer or author interaction;
- whether a requested response has been missing for a long time;
- pull request review state and unresolved review threads;
- automated review comments, including unresolved Copilot comments;
- CI and check status, runner or environment failures, mergeability, and
  conflicts;
- linked issues and pull requests, duplicates, and dependency relationships;
- whether a previously handled item has new evidence, failures, reviews, CI
  results, or maintainer decisions.

Do not count inspection as handling. Take one concrete action for every
actionable item and record the result.

## Act by Item Type

### Reproducible bugs

For a safely reproducible and verifiable bug:

1. Confirm the narrowest evidence or reproduction.
2. Follow repository policy and `oss-change-workflow`.
3. Create a neutral, repository-conforming branch.
4. Implement the smallest fix and add or update a regression test when a
   practical test layer exists.
5. Run the smallest relevant validation, inspect the worktree, and commit only
   intended files.
6. Push and open a normal, non-draft pull request.
7. Continue through CI, review, and automated-review feedback until the pull
   request is ready for a maintainer to merge or explicitly blocked.

### Existing fix pull requests

Check the latest CI, mergeability, unresolved review threads, automated-review
comments, and linked issues. Fix clear problems, push updates, and reply to the
relevant thread or comment. Continue tracking until the pull request is ready
for a maintainer to merge, explicitly rejected or closed, or blocked only by a
maintainer, reviewer, reporter, runner, or external infrastructure.

### Missing information

Post one concise and specific request for the minimum information needed, such
as a reproduction, version, environment, configuration, command, error log, or
other necessary context. Do not post a generic request for more information.

### Content-only changes

Use `oss-change-workflow`, but follow its content-only lightweight path for
documentation, blog, README, translation, and other prose-only changes. Perform
a focused fidelity self-review, relevant documentation validation, existing
review-comment handling, and CI monitoring. Do not start the mandatory review
subagent or independent cross-agent review solely for content changes.

Switch immediately to the full code-review path when a change touches code,
tests, scripts, dependencies, configuration, routes, builds, deployment,
generated artifacts, or runnable examples.

### Support, features, duplicates, unsupported requests, and stale items

Classify from available evidence and post a concrete reason and next step when
a public response is useful. Point to relevant documentation, existing issues
or pull requests, supported scope, the needed maintainer decision, or the
reporter's next action. Never promise a response or delivery timeline.

### Security-sensitive reports

Follow the target repository's `SECURITY.md`. Never disclose exploits, secrets,
unpublished vulnerability details, directly reusable abuse steps, or attack
payloads publicly. When the matter cannot be handled safely in public, do not
post sensitive details. Alert the user and direct the report to the private
security channel.

## Follow Through Without Spamming

- Recheck every item previously replied to, classified, opened, or updated, and
  every item where information was requested.
- Do not stop tracking after the first comment or pull request.
- Do not repeat a public comment without new evidence, a changed blocker, a new
  fix, a validation result, or a clear request for another party.
- Keep an unchanged item in local memory and report it as continuing follow-up
  with no new public action.
- Count discovery, actionable items, public replies, newly opened pull requests,
  and updated pull requests separately.

Store only the minimum local continuity state, such as item URL, category,
update time, last-seen signature, last-public-action signature, next step,
blocker, priority, and follow-up status. Never write internal state back to
GitHub or commit it.

## Write Public Replies as a Maintainer

- Write every issue comment, pull request comment, and review as a normal
  project maintainer response.
- Start directly with the evidence, decision, action taken, validation result,
  blocker, or requested next step that matters to the contributor.
- Never mention the patrol, sweep, scheduled run, automation, agent process,
  internal queue, memory, or tooling identity in public GitHub text.
- Avoid status-banner or ceremonial preambles. When revisiting an item, explain
  the new evidence or changed blocker rather than the maintenance process that
  caused the recheck.

## Authorized Writes

Without per-item confirmation, and only in `eunomia-bpf`, the task may:

- comment with a specific reproduction request, classification, investigation
  result, CI or review blocker, or contributor response;
- create a branch, fix a well-supported and safely verifiable bug, add tests,
  push, and open a pull request;
- address clear review or automated-review feedback, push corrections, and
  reply with the result;
- update maintenance branches and pull requests created or owned by the task.

Before every write, verify scope, repository policy, and that the action is not
a duplicate. Treat this list as exhaustive. Do not perform other writes such as
changing labels, assignees, or milestones.

## Name Branches, Commits, and Pull Requests Neutrally

- Follow the target repository's branch convention. If none exists, use a
  neutral technical prefix such as `fix/`, `feat/`, `docs/`, or `chore/`.
- Never put the name of an AI tool, model, assistant, or provider in a branch
  name, commit subject or body, or pull request title.
- Never add AI attribution, generated-by statements, or AI co-author trailers
  to a pull request body. Describe only the technical change, validation, and
  linked issue.
- Apply the same restrictions to release text drafted for the user.

## Never Perform These Actions

- Merge a pull request.
- Publish a release.
- Close an issue or pull request.
- Delete a branch.
- Change access permissions, organization or repository settings, branch
  protection, secrets, webhooks, or deployment configuration.
- Write to a repository outside `eunomia-bpf`.
- Expose credentials, tokens, private logs, or sensitive environment details in
  GitHub content, commits, branch names, pull request text, or reports.

## Report and Persist

Write a concise, actionable Chinese report that includes:

- total open items discovered and total actionable items;
- actual public replies;
- fixes and newly opened pull requests;
- updated pull requests;
- tracked items with no new public action;
- items blocked by a reporter, runner, reviewer, maintainer, CI or
  infrastructure, or the task itself;
- exact links for every important item.

For each important item, state the repository, item type, classification,
current status, action taken, next step, and blocking party. Label any item that
was discovered but not handled as `发现但未处理` and explain why. Never present
discovery as completed work.

Update local automation memory with the minimum deduplication and follow-up
state plus the run summary. Do not commit or publish the memory.
