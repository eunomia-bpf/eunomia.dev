---
name: blog-writer
description: Fast, pinned-model workflow for writing or revising bilingual eunomia.dev posts under docs/blog/posts/. Codex drafts, one different model performs a reader-focused review and limited direct edit, then Codex validates and finishes. Style and SEO details live in the companion checklists.
---

# Blog Writer

Write or revise the requested post under `docs/blog/posts/`. English uses `post.md`; Chinese uses `post.zh.md`.

Read `.agents/skills/blog-writing-style/SKILL.md` and `.agents/skills/seo-geo/SKILL.md` before writing. Treat their style advice as guidance that supports judgment, not as a reason to add review rounds. Repository safety rules, source accuracy, public-path stability, and the edit budget below remain constraints.

## Default workflow

Aim to finish an ordinary single-post task within about ten minutes when sources are already available and validation is healthy. Do not sacrifice factual accuracy to meet the target. If source retrieval, a long primary paper, or a failing build makes the task take longer, report that concrete cause instead of adding agents.

1. **Codex drafts.** Read the primary source and relevant sibling posts. Write a one-sentence thesis and identify the target reader, distinctive insight, evidence, mechanism, boundary, and practical decision. Draft or revise the complete EN/ZH pair, then read each paragraph once for information flow, sentence rhythm, and density. The reader should not have to wait for a missing object, reconstruct a broken causal chain, or unpack several new numbers and terms at once. For new work, use `draft/blog/` until the public filename, date, and slug are settled.
2. **One different model reviews and edits.** Select one non-Codex model before drafting. Give it the complete pair, primary source, sibling posts, and both checklists. It reviews from the target reader's perspective and directly edits only sentences or paragraphs that materially improve accuracy, clarity, naturalness, or insight. Ask it to focus on missing objects, unclear referents, broken causal units, and dense clusters that need interpretation or deferral. A no-change verdict is acceptable when no necessary improvement exists.
3. **Codex finishes.** Inspect the external model's diff, reject regressions, and reread every altered passage in order. Verify that claims name their objects when introduced, sentence boundaries mark useful pauses, and dense evidence has enough interpretation and breathing room. Then verify facts and bilingual consistency and run the smallest relevant local validation. Codex is the final editor and reviewer.
4. **Publish once.** When the task is PR-bound, preserve separate local commits for the draft, external edit, and any final Codex fix when those stages changed files. Push once after final validation so intermediate commits do not queue redundant CI runs.

This is the complete default writing workflow. Do not start a second external model, a separate reader subagent, or another review round unless the user explicitly requests it. A normal blog task should use no subagents; invoke the one different model through its local CLI.

## Model selection

Use the exact pinned configuration selected for the post:

- Codex author: `gpt-5.6-sol` with `model_reasoning_effort=xhigh`
- Default independent editor: `grok-4.5`
- Approved alternatives: `kimi-code/k3`, `zai-coding-plan/glm-5.2`, `claude-opus-4-6[1m]`

Run only one approved independent editor. Do not silently change its model ID or add another model. Record the author and editor configurations in the completion report.

## Edit boundary

The independent editor and the final Codex pass each stay within one third of the current EN file and one third of the current ZH file. Count changed prose paragraphs and changed Markdown blocks; metadata fields, headings, tables, figures, and code blocks count as blocks. Whole-file replacement, full-section replacement, broad reordering, and style-only reflow are out of scope.

Every edit should be necessary and local. Preserve verified evidence, caveats, figures, code, filenames, dates, slugs, and public paths. When a factual correction is needed, change the factual token and the minimum surrounding grammar rather than polishing the whole paragraph.

## Content decisions

- High-quality, source-grounded insight is the main goal. A polished paper summary without a reader-facing synthesis is not enough.
- Let the topic determine length. Professional empirical posts may be long when each section contributes evidence, mechanism, interpretation, boundary, or a practical decision. Do not impose a word target.
- Use a compelling, accurate, professional title. Avoid content-farm framing.
- Choose an argument structure that fits the topic rather than copying a paper's RQ order or a house template.
- Select only figures that materially support the thesis. A paper figure is never mandatory merely because it exists.
- Write Chinese from the same facts and argument, not by translating English line by line. Keep claims, examples, figures, and section progression aligned while allowing natural sentence and paragraph boundaries.

## Final validation

Before completion, Codex checks:

- every important claim, number, denominator, condition, and limitation against the current primary source;
- information flow and density on first reading, including concrete measured or compared objects, defined antecedents, complete causal links, useful sentence boundaries, and enough interpretation or breathing room around dense evidence;
- title, description, date, slug, `<!-- more -->`, links, image paths, and EN/ZH section correspondence;
- the independent model's diff and the one-third budget;
- paragraph flow, terminology, Chinese naturalness, sibling overlap, and the ending's practical takeaway, using the style checklist as guidance;
- `git diff --check` and the smallest relevant site validation, with full repository validation when the PR or risk level calls for it;
- a clean worktree containing only intended files before the final push.

Report the files changed, exact models, edit-budget usage, validation performed, and any remaining uncertainty. Do not claim completion while a factual or repository-safety blocker remains.
