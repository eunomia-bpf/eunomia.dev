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

Publishing authorization, the browser-first/API-first split, and credential
handling are defined once in `CLAUDE.md`'s Publishing section — every
platform publisher follows that, not a repeated local copy.

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

Use `eunomia-community-radar` for the daily review of approved technical
communities through watchlist-opted-in read-only Slack archives plus
visible-browser coverage, and publication of one combined anonymous eBPF Q&A
and community brief after a successful run. Keep it separate from
`eunomia-social-radar`, which follows discussion around content already
published by Eunomia, and from GitHub issue or pull-request patrol.
