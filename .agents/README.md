# Agent Skills Bridge

This directory exposes repository skills to Codex-compatible agents.

In Git, `.claude/skills` is a symlink pointer to `../.agents/skills` so Claude
and Codex share the same skill set. On Windows with `core.symlinks=false`, the
pointer may appear as a small text file; edit the real skill folders under
`.agents/skills`.

Keep platform publishing skills browser-first. Social/media platform audits,
drafts, screenshots, and ledger evidence must use normal browser interactions,
not hidden platform APIs or background endpoints.

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
