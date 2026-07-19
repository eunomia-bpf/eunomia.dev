# Eunomia Community Issue and PR Patrol

This file is the source of truth for the recurring Eunomia community GitHub
maintenance task. The task must read the complete file at the start of every
run. Runtime state, credentials, private logs, and deduplication records must
remain outside the repository.

## Objective

Continuously inspect and maintain open issues and pull requests across the
`eunomia-bpf` GitHub organization. Do not stop at reporting: every actionable
item must receive a concrete next action and remain tracked until it is closed,
rejected, resolved, ready for a maintainer to merge, or blocked only by an
external party.

## Scope

- Organization: `eunomia-bpf` only.
- Repositories: every public repository that is neither archived nor a fork.
- Items: all open issues and open pull requests.
- Writes: only repositories in the `eunomia-bpf` organization.
- Execution environment: the Linux maintenance workspace. Do not redirect the
  task to a Windows or PowerShell project.

All new and previously tracked actionable items must stay in the queue. Work
may be prioritized as security, confirmed bug, blocked fix, needs-info,
documentation or support, and stale, but lower-priority items must not be
permanently skipped.

## Schedule

Run at 09:00 in the `America/Vancouver` time zone every two calendar days. A
successful manual patrol resets the next eligible date. If a scheduled run
fails, retry at the next daily scheduler wake instead of waiting another two
days. Prevent overlapping or duplicate invocations with a local execution lock.

## Start of Every Run

1. Read the local automation memory completely. Use it to avoid duplicate
   comments and to resume every unresolved item.
2. Read and follow the `oss-issue-triage` skill for issue and PR classification.
3. Before changing a repository, read and follow the `oss-change-workflow`
   skill and the target repository's `AGENTS.md`, `CONTRIBUTING`,
   `SECURITY.md`, `README`, and relevant workflows.
4. Refresh the organization repository list instead of relying only on the
   previous run's inventory.

If this file or a required repository policy cannot be read, do not guess at
the missing authorization. Record the blocker and continue with safe,
read-only work elsewhere in scope.

## Required Inspection

For every open issue and pull request, inspect:

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

## Required Action for Every Actionable Item

### Reproducible bugs

For a bug that can be reproduced and validated safely:

1. Confirm the narrowest evidence or reproduction.
2. Follow the target repository's policies and `oss-change-workflow`.
3. Create a neutral, repository-conforming branch.
4. Implement the smallest fix and add or update a regression test when a
   practical test layer exists.
5. Run the smallest relevant validation, inspect the worktree, and commit only
   intended files.
6. Push and open a normal, non-draft pull request.
7. Continue through CI, review, and automated-review feedback until the pull
   request is ready for a maintainer to merge or is explicitly blocked.

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

Documentation, blog, README, translation, and other prose-only changes still
use `oss-change-workflow`, but follow its content-only lightweight path. Perform
a focused fidelity self-review, relevant documentation validation, existing
review-comment handling, and CI monitoring. Do not start the mandatory review
subagent or independent cross-agent review solely for content changes.

If a change touches code, tests, scripts, dependencies, configuration, routes,
builds, deployment, generated artifacts, or runnable examples, immediately use
the full code-review path required by `oss-change-workflow`.

### Support, features, duplicates, unsupported requests, and stale items

Classify the item from available evidence and post a concrete reason and next
step when a public response is useful. Point to relevant documentation,
existing issues or pull requests, the supported scope, the needed maintainer
decision, or the reporter's next action. Never promise a response or delivery
timeline.

### Security-sensitive reports

Follow the target repository's `SECURITY.md`. Never disclose exploits, secrets,
unpublished vulnerability details, directly reusable abuse steps, or attack
payloads in public issues or pull requests. When the matter cannot be handled
safely in public, do not post sensitive details; alert the user and direct the
report to the repository's private security channel.

## Follow-up and Deduplication

- Continue checking every previously replied-to or classified item, every item
  where information was requested, and every opened or updated item on later
  runs.
- Do not stop tracking after the first comment or pull request.
- Do not post repeated comments without new evidence, a changed blocker, a new
  fix, a validation result, or a clear request for the other party.
- When no public action is warranted, keep the item in the local queue and
  report it as continuing follow-up with no new public action.
- Keep discovery, actionable, public reply, newly opened pull request, and
  updated pull request counts separate.

The local automation may store only the minimum state needed for continuity,
such as item URL, category, update time, last-seen signature, last-public-action
signature, next step, blocker, priority, and follow-up status. Never write this
internal state back to GitHub or commit it to the repository.

## Authorized GitHub Writes

The task may perform the following actions without waiting for per-item user
confirmation, but only inside the `eunomia-bpf` organization:

- comment on issues and pull requests with a specific reproduction request,
  classification, investigation result, CI or review blocker, or contributor
  response;
- create a branch, fix a well-supported and safely verifiable bug, add tests,
  push, and open a pull request;
- address clear review or automated-review feedback, push corrections, and
  reply with the result;
- update maintenance branches and pull requests created or owned by the task.

Before every write, verify that the target is in scope, the comment is not a
duplicate, and repository policies permit the action.

This authorization list is exhaustive. Do not perform other GitHub writes,
including changing labels, assignees, or milestones.

## Naming and Publication Text

- Follow the target repository's branch convention. If none exists, use a
  neutral technical prefix such as `fix/`, `feat/`, `docs/`, or `chore/`.
- Branch names, commit subjects and bodies, and pull request titles must not
  contain the name of any AI tool, model, assistant, or provider.
- Pull request bodies must not contain AI attribution, generated-by statements,
  or AI co-author trailers. Describe only the technical change, validation, and
  linked issue.
- The same restrictions apply to any release text drafted for the user.

## Prohibited Actions

The task must never:

- merge a pull request;
- publish a release;
- close an issue or pull request;
- delete a branch;
- change access permissions, organization or repository settings, branch
  protection, secrets, webhooks, or deployment configuration;
- write to a repository outside the `eunomia-bpf` organization;
- expose credentials, tokens, private logs, or sensitive environment details in
  GitHub content, commits, branch names, pull request text, or reports.

## End-of-Run Report

Write the report in concise, actionable Chinese. It must include:

- total open items discovered;
- total actionable items;
- number of actual public replies;
- fixes and newly opened pull requests;
- updated pull requests;
- tracked items with no new public action in this run;
- items blocked by a reporter, runner, reviewer, maintainer, CI or
  infrastructure, or the task itself;
- exact links for every important item.

For each important item, state the repository, item type, classification,
current status, action taken, next step, and blocking party. Label any item that
was discovered but not handled as `发现但未处理` and explain why. Never present
discovery as completed work.

At the end of the run, update the local automation memory with the minimum
deduplication and follow-up state plus the run summary. Do not commit that
memory or publish it to GitHub.
