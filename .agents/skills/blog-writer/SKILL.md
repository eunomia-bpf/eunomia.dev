---
name: blog-writer
description: Write or substantially revise bilingual eunomia.dev blog posts from primary sources. Use when Codex should prepare source material, have an independent subagent draft the article, have exactly claude-opus-4-5 rewrite it, and validate the result without drafting the prose itself.
---

# Blog Writer

Use this workflow for posts under `docs/blog/posts/` and article drafts under
`draft/blog/`. Read `.agents/skills/blog-writing-style/SKILL.md` for the desired
prose result.

The workflow has exactly three stages. Writing style lives only in
`blog-writing-style`; do not duplicate its rules here. Do not add review rituals
or repository artifacts merely to prove that a stage happened.

## 1. Codex Prepares Sources

- Collect the original paper PDFs, official repositories, official
  documentation, and other necessary primary material.
- Put temporary source material outside the repository. Do not create outlines,
  evidence ledgers, review logs, figure inventories, or publication-QA files
  unless the user explicitly requests one.
- Record the target reader, topic, output paths, and any title or angle the user
  has already chosen in a compact task brief. Do not prescribe an opening,
  section order, or paragraph plan.
- For an existing article, capture the frontmatter, H1, figures, tables, code,
  links, and references that must survive the rewrite. Do not prepare replacement
  prose.

## 2. Subagent Drafts, Opus Rewrites

- Give the source package, compact task brief, and `blog-writing-style` to a
  fresh independent subagent. It writes the complete source-grounded EN/ZH
  draft without reading the old body, an outline, or another candidate.
- Use exactly `claude-opus-4-5`. If it is unavailable, report the blocker and do
  not silently substitute another model.
- Ask Opus to read the primary material and the subagent draft, then rewrite the
  complete article as the final editor. It owns the opening, structure,
  paragraphing, and prose.
- Treat the subagent draft as editable source-grounded prose, not wording or
  structure that Opus must preserve. Do not expose an old body, outline, or
  another candidate to the editing pass.
- Preserve the user's title and content promise. Preserve existing frontmatter,
  filename, slug, date, figures, tables, code, links, and references unless the
  user explicitly requests a change or a source proves an item obsolete.
- Do not let Opus edit unrelated files, create workflow artifacts, stage,
  commit, push, or publish.

## 3. Codex Validates

- Confirm from the run output that the writing model was
  `claude-opus-4-5`.
- Read the complete result as the intended reader. Check it against
  `blog-writing-style` without turning examples into a line-by-line checklist.
- Compare important facts, numbers, conditions, terminology, and limitations
  with the primary sources.
- Check title, frontmatter, links, figures, tables, code, and references. EN/ZH
  versions preserve the same core claims, sections, examples, figures, tables,
  numbers, and caveats, while sentence and paragraph boundaries may differ.
- If the prose is still unclear or mechanical, tell Opus the concrete reader
  problem and have Opus revise it. Codex does not take over the body rewrite.
- Codex may make only bounded factual, metadata, link, typography, or formatting
  corrections after the prose passes.
- Run `git diff --check` and the smallest relevant content or site validation
  before delivery.

Report the changed files, exact Opus model, validation performed, and any
remaining factual uncertainty.
