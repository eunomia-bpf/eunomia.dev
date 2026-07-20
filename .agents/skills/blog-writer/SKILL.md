---
name: blog-writer
description: Fast, pinned-model workflow for writing or revising bilingual eunomia.dev posts under docs/blog/posts/. Codex drafts, one different model performs a reader-focused review and limited direct edit, then Codex validates and finishes. Style and SEO details live in the companion checklists.
---

# Blog Writer

Write or revise the requested post under `docs/blog/posts/`. English uses `post.md`; Chinese uses `post.zh.md`.

Read `.agents/skills/blog-writing-style/SKILL.md` and `.agents/skills/seo-geo/SKILL.md` before writing. Treat their style advice as guidance that supports judgment, not as a reason to add review rounds. Repository safety rules, source accuracy, public-path stability, and the edit budget below remain constraints.

## Default workflow

Aim to finish an ordinary single-post task within about ten minutes when sources are already available and validation is healthy. Do not sacrifice factual accuracy to meet the target. If source retrieval, a long primary paper, or a failing build makes the task take longer, report that concrete cause instead of adding agents.

1. **Codex drafts.** Read the primary source and relevant sibling posts. Write a one-sentence thesis and identify the target reader, distinctive insight, evidence, mechanism, boundary, and practical decision. Before outlining, write a background contract naming what the target reader likely knows, what the post must explain before its first specialized claim, which concrete story, problem, or operational scenario can open the piece, what user value the reader should feel in the first screen, and which sibling link can carry deeper prerequisites. Every blog needs a short ramp from reader context to the new claim, so do not open by jumping straight into a tool, paper result, benchmark, release note, or implementation detail. The frontmatter `description` and the excerpt before `<!-- more -->` must also begin with one compact background clause or sentence that names the problem domain or reader situation before stating the post's contribution, and they should make the problem and practical payoff clear enough for a homepage card or social preview. For paper-based posts, review the main-body figures and tables before outlining, including the claim each item supports, its source, and the include/omit decision. Keep this as working context, not a standalone artifact. Draft or revise the complete EN/ZH pair, then read each paragraph once for information flow, sentence rhythm, and density. The reader should not have to wait for a missing object, reconstruct a broken causal chain, or unpack several new numbers and terms at once. For new work, use `draft/blog/` only for actual article drafts while the public filename, date, and slug are unsettled. Do not create separate figure inventories, workflow notes, platform-hook drafts, review logs, or publish-QA files.
2. **One different model reviews and edits.** Select one non-Codex model before drafting. Give it the complete pair, primary source, sibling posts, and both checklists. It reviews from the target reader's perspective and directly edits only sentences or paragraphs that materially improve accuracy, clarity, naturalness, or insight. Ask it to focus on missing objects, unclear referents, broken causal units, and dense clusters that need interpretation or deferral. A no-change verdict is acceptable when no necessary improvement exists.
3. **Optional reader subagent when requested.** If the user explicitly asks for a subagent or reader-perspective subagent, run one read-only subagent pass after the external model pass or after the current draft is available. Give the subagent file paths, the primary source, and the style/SEO checklists, and ask for actionable reader findings rather than edits. Treat this as reader feedback, not the final style decision.
4. **Codex finishes.** Inspect the external model's diff and any subagent findings, reject regressions, and reread every altered passage in order. Then perform a final Codex-owned pass against `blog-writing-style` and `seo-geo`, including sentence rhythm, Chinese naturalness, figure interpretation, source fidelity, and whether the content is rich enough for the topic. If the draft feels thin, expand with source-grounded examples, first-party numbers, mechanism detail, practical checks, or limitations instead of polishing around the gap. Codex is the final editor and reviewer.
5. **Publish once.** When the task is PR-bound, preserve separate local commits for the draft, external edit, subagent-driven fixes when applicable, and any final Codex fix when those stages changed files. Push once after final validation so intermediate commits do not queue redundant CI runs.

This is the complete default writing workflow. Run one independent reader/editor pass, not a chain of repeated reviews. Prefer the pinned local CLI when it completes a real review, but if the CLI fails, only prints planning text, cannot read files, or returns a tool error, diagnose it with `agent-cli-tools`, try an inline or prompt-file fallback when appropriate, and then switch to one approved alternative if it still cannot produce actionable findings or an explicit no-change verdict. Use a reader subagent when the user asks for one or when local CLI review is unavailable and a reader pass is still required.

## Edit scale routing

- **Small local edits:** Codex can directly handle title changes, metadata fixes, a few sentences, one bounded paragraph, or one bounded section when the change does not require a new article structure, broad background expansion, or full prose re-composition. Afterward, Codex still runs the relevant style, SEO, factual, link, and image checks.
- **Rewrite work:** Use `claude-opus-4-5` for full-post rewrites, multi-section rewrites, substantial background expansion, tone rebuilds, or cases where the current draft needs paragraph-by-paragraph re-composition rather than line edits. Give Opus file paths, primary sources, sibling posts, and the relevant skills/checklists, not inlined article content, then have Codex inspect the diff and perform the final style and source-fidelity pass.
- **Unavailable rewrite model:** If `claude-opus-4-5` is unavailable, report that exact blocker instead of silently substituting another model. Use another approved editor only when the user authorizes the substitution or the task is a non-rewrite review fallback.

## Model selection

Use the exact pinned configuration selected for the post:

- Codex author: `gpt-5.6-sol` with `model_reasoning_effort=xhigh`
- Default independent editor: `grok-4.5`
- Rewrite editor: `claude-opus-4-5`
- Approved alternatives: `kimi-code/k3`, `zai-coding-plan/glm-5.2`, `claude-opus-4-6[1m]`

Run only one approved independent editor. Do not silently change its model ID or add another model. Record the author and editor configurations in the completion report.

A successful smoke test such as `Reply with exactly: ok` proves only that the CLI is authenticated. It does not satisfy the independent editor step. Count the editor pass only when the tool returns concrete findings, a direct limited edit, or an exact no-change verdict after reading the draft and sources it was given.

## Edit boundary

The independent editor and the final Codex pass each stay within one third of the current EN file and one third of the current ZH file. Count changed prose paragraphs and changed Markdown blocks; metadata fields, headings, tables, figures, and code blocks count as blocks. Whole-file replacement, full-section replacement, broad reordering, and style-only reflow are out of scope.

Every edit should be necessary and local. Preserve verified evidence, caveats, figures, code, filenames, dates, slugs, and public paths. When a factual correction is needed, change the factual token and the minimum surrounding grammar rather than polishing the whole paragraph.

## Content decisions

- High-quality, source-grounded insight is the main goal. A polished paper summary without a reader-facing synthesis is not enough.
- Make the post attractive through professional stakes, not hype. A strong post should open with a real story, failure mode, question, or measured tension, then quickly show what the reader can understand, decide, debug, deploy, or avoid after reading it.
- Make byline expertise visible through evidence, mechanism detail, practical
  teaching, and clear judgment. Use eunomia-bpf projects as proof when they
  help the reader inspect code, reproduce results, or continue learning.
- Make the core claim reusable: one clear platform-native hook, one evidence
  sentence with a source, and one contextual project or canonical archive link.
  Broader cadence, channel mix, and brand-positioning decisions live in
  `draft/content-platform-strategy.zh.md`, not in this writing workflow.
- Let the topic determine length. Professional empirical posts may be long when each section contributes evidence, mechanism, interpretation, boundary, or a practical decision. Do not impose a word target.
- Use a compelling, accurate, professional title that makes the article's contract clear before the reader opens it. Let the strongest source-backed insight determine whether the title foregrounds a finding, tension, boundary, mechanism, evidence type, article form, or practical consequence. Preserve high-value title signals such as `An Empirical Study`, `Inside ...`, `Why ...`, `How ...`, concrete subsystem names, and method names when they improve trust or reader pull. Do not shorten a title merely to make a homepage card look cleaner, and do not use SEO/GEO guidance to override the style checklist's title judgment.
- Make the first paragraph cash the title's promise. Open with a concrete scene, failure mode, measurement, or operator decision that shows why the article matters, then let the next paragraph name the paper, tool, or mechanism. Avoid opening with an abstract project summary after a title that promises a story, study, or inside look.
- Choose an argument structure that fits the topic rather than copying a paper's RQ order or a house template.
- Select only figures that materially support the thesis. A paper figure is never mandatory merely because it exists, but an empty figure set for a paper-based post must still be a deliberate, source-grounded working decision, not a skipped step or a separate inventory file.
- Write Chinese from the same facts and argument, not by translating English line by line. Keep claims, examples, figures, and section progression aligned while allowing natural sentence and paragraph boundaries.
- Use canonical absolute HTTPS URLs for non-image internal links, such as `https://eunomia.dev/blog/2026/07/19/example/` or `https://eunomia.dev/zh/blog/2026/07/19/example/`, so copied Markdown still works on Medium and other syndication platforms. Image links may stay relative to the Markdown file, such as `imgs/example.png`, so post-local media remains portable before upload.
- End every blog with a compact `## References` / `## 参考文献` section containing 5–10 distinct primary sources actually used. Keep inline attribution near supported claims, do not pad the list, and align the sources across languages.

## Final validation

Before completion, Codex checks:

- every important claim, number, denominator, condition, and limitation against the current primary source;
- the frontmatter `description`, excerpt, opening, and first section give enough domain background for the target reader before the first specialized result, tool claim, benchmark, or implementation detail, and make the problem, user value, and promised practical payoff explicit without turning promotional;
- the title and first paragraph work together: the title preserves article type, core object, evidence or mechanism, and reader-facing tension, while the first paragraph proves that promise through a concrete scene, failure mode, measurement, or decision;
- paper-based posts have reviewed the relevant figure/table candidates, selected images have descriptive alt text, and omitted figures do not leave a claim unsupported;
- information flow and density on first reading, including concrete measured or compared objects, defined antecedents, complete causal links, useful sentence boundaries, and enough interpretation or breathing room around dense evidence;
- title, description, date, slug, `<!-- more -->`, links, image paths, final references, and EN/ZH section correspondence. Title review comes from `blog-writing-style`; SEO/GEO only checks technical metadata constraints after the title direction is chosen;
- all non-image internal links use canonical absolute HTTPS URLs, external links use full URLs, and only image links use post-local relative paths;
- the independent model's diff and the one-third budget;
- paragraph flow, terminology, Chinese naturalness, sibling overlap, and the ending's practical takeaway, using the style checklist as guidance;
- `git diff --check` and the smallest relevant site validation, with full repository validation when the PR or risk level calls for it;
- a clean worktree containing only intended files before the final push.

Report the files changed, exact models, edit-budget usage, validation performed, and any remaining uncertainty. Do not claim completion while a factual or repository-safety blocker remains.
