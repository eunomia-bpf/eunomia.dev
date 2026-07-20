# Agent Skills Bridge

This directory exposes repository skills to Codex-compatible agents.

In Git, `.claude/skills` is a symlink pointer to `../.agents/skills` so Claude
and Codex share the same skill set. On Windows with `core.symlinks=false`, the
pointer may appear as a small text file; edit the real skill folders under
`.agents/skills`.

Keep platform publishing skills browser-first. Social/media platform audits,
drafts, screenshots, and ledger evidence must use normal browser interactions,
not hidden platform APIs or background endpoints.

Keep skills procedural. Long-term brand strategy, channel mix, campaign
cadence, and positioning plans belong under `draft/`, while skills should hold
repeatable execution steps, constraints, platform QA, scripts, and validation.

Use `eunomia-content-patrol` as the source of truth for the scheduled daily
content patrol. The cron prompt should stay short and route execution through
that skill rather than duplicating the full daily operating policy.
