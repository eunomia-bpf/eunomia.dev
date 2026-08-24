# Agent Skills Bridge

This directory exposes repository skills to Codex-compatible agents.
Repository-specific content, publishing, SEO, and research skills are tracked
directly under `.agents/skills`. Reusable maintainer and organization-level
skills come from the pinned `eunomia-bpf/agent-skills` submodule under
`.agents/sources/agent-skills` and are linked into the same directory.

Initialize the submodule and rebuild the bridge with
`scripts/sync-agent-skills.ps1` on Windows or
`scripts/sync-agent-skills.sh` on Unix. The linker adds only shared skills,
uses symbolic links where available, and falls back to Windows directory
junctions when link privileges are unavailable. It refuses to overwrite a
real file or directory.

In Git, `.claude/skills` is a symlink pointer to `../.agents/skills` so Claude
and Codex share the combined skill set. On Windows with `core.symlinks=false`,
the pointer may appear as a small text file. Edit repository-specific skill
directories here. For a shared skill, update the canonical `agent-skills`
repository, push its `main`, update the submodule gitlink here, and rerun the
sync script.

Keep platform publishing skills browser-first except for Medium and DEV.to,
which are API-first through their documented publishing endpoints. Their API
credentials stay local and secret, and their resulting public articles still
require normal visible-browser QA. Audits and all other social/media platform
actions use normal browser interactions, not hidden platform APIs or background
endpoints.

A direct request to publish, post, or submit, or an eligible rolling-queue item
marked `排队`, authorizes the matching publisher to complete the final public
action and QA without asking again at the last button. A request limited to a
draft, preview, or preparation does not authorize publication. This rule is
shared by every platform publisher; private messages, follows, likes, votes,
account settings, payments, and deletions remain outside that authorization.

Keep workflow skills procedural. Long-term brand strategy, channel mix,
campaign cadence, and positioning plans belong under `draft/`, while workflow
skills should hold repeatable execution steps, constraints, platform QA,
scripts, and validation.

Separate output standards from execution. Style guides, checklists, and
reference files define what a good result should feel like or contain. The
workflow skill that uses them owns who performs the work, model selection,
step order, tools, edit permissions, retry behavior, and validation. Do not put
pass instructions or model routing in a style guide, and do not duplicate the
same operational rule in both places. Prefer one normal editing pass that
satisfies the stated outcomes over adding mandatory review rounds.

Use `eunomia-content-patrol` as the source of truth for the scheduled daily
content patrol. The cron prompt should stay short and route execution through
that skill rather than duplicating the full daily operating policy.

Use `eunomia-community-radar` for the daily visible-browser review of approved
technical communities and publication of one combined anonymous eBPF Q&A and
community brief after a successful run. Keep it separate from
`eunomia-social-radar`, which follows discussion around content already
published by Eunomia, and from GitHub issue or pull-request patrol.
