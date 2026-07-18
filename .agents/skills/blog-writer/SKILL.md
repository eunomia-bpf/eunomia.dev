---
name: blog-writer
description: Pinned-model process control for producing eunomia.dev blog posts, covering both writing a new post and reviewing/fixing an existing one under docs/blog/posts/. Drives topic-only comparison briefs, architecture-first drafting, paper-to-blog transformation, EN/ZH pairing, severity-ranked review, paragraph-density checks, mandatory different-model-family review, editing discipline, and verification; all style/SEO rules come from the blog-writing-style checklist. Use for "write a blog", "draft a post", "review this post", or "fix the prose in X".
---

# Blog Writer (process)

Produce or improve the blog post given in `$ARGUMENTS`. If no argument is given, ask for the topic (new post) or file path (existing post). Posts live in `docs/blog/posts/` as `post.md` (English) and `post.zh.md` (Chinese) pairs.

This skill is the workflow only. **Read both rulebooks first, every time**: `.claude/skills/blog-writing-style/SKILL.md` (prose mechanics, blog antipatterns, content-farm bans, length/richness, Chinese-English mixing, bilingual consistency) and `.claude/skills/seo-geo/SKILL.md` (metadata, keyword strategy, GEO citation-worthiness, syndication canonical discipline). Do not restate or override their rules here.

## Pinned model roster

Use only these exact model configurations for model-authored blog drafts and external reviews unless the user explicitly changes this roster:

- Kimi: `kimi-code/k3`
- Grok: `grok-4.5`
- OpenCode/GLM: `zai-coding-plan/glm-5.2`
- Codex: `gpt-5.6-sol` with `model_reasoning_effort=xhigh`
- Claude: `claude-opus-4-6[1m]`

Do not silently follow a provider's default alias and do not auto-upgrade a pinned model. A model-ID or reasoning-setting change requires an explicit roster update. Record the exact author and reviewer configurations in the completion report. Model identity controls reproducibility and reviewer independence; it does not replace the rulebooks or verification.

When the caller asks for multiple model versions, give every model the same **topic-only brief**. The brief may name the topic, primary sources, target files, required language pair, and content-preservation boundaries. It must not prescribe a thesis, preferred outline, target conclusion, or ledger of required numbers. Each writer selects source-backed evidence and constructs its own argument. Reviewers verify every selected number against the primary source, including its denominator and scope.

## Sequential direct-edit workflow

For a production post, Codex `gpt-5.6-sol` with `model_reasoning_effort=xhigh` writes the complete EN/ZH first draft. Before drafting, select and record exactly two different non-Codex model families from the pinned roster. Those two models edit the draft in sequence. Do not change the selected pair halfway through a post unless one model is unavailable and the user approves the replacement.

When the user does not choose the pair, use `grok-4.5` followed by `kimi-code/k3` for bilingual technical blogs. The other pinned models remain approved alternatives, not mandatory passes.

Do not assign specialized roles to these models. Give each model the same complete rulebooks, primary sources, relevant sibling posts, figure inventory, and the latest EN/ZH files. Each model reads the entire current draft, audits the previous model's commit, and then **edits only the necessary sentences, paragraphs, headings, metadata fields, or figure placements**. The next model always receives the previous model's committed and pushed files.

Except for Codex creating the first draft when no draft exists, a model pass must not touch more than one third of the prose paragraphs in either language. It must also keep the changed-line footprint within one third of each baseline file, measured as unique original or newly inserted lines touched by the pass rather than double-counting a replacement as one deletion plus one addition. Headings, metadata fields, figures, tables, and code blocks count as touched blocks. Stay comfortably below the limit when the count is ambiguous.

Each pass must preserve verified content and public identity. It may add source-backed evidence or figures that the current draft omitted, but it must not invent facts, silently delete technical content, change the slug, move the public path, reflow unaffected paragraphs, reorder the whole article, or replace either complete file. “Rewrite for consistency” is not sufficient justification for broad edits. After the two direct-edit passes, Codex integrates remaining consistency fixes under the same one-third limit and runs verification. Model passes return a concise change log after editing; they do not stop at a review report.

### Per-model commit and push checkpoint

For every model pass after the initial draft:

1. Start from a clean worktree at the pushed commit produced by the previous pass.
2. Review `HEAD^..HEAD` before proposing new edits. Compare that diff with the primary source, both rulebooks, sibling posts, and EN/ZH counterpart. Confirm explicitly that the previous pass improved the post without losing facts, figures, caveats, or natural phrasing.
3. If the previous pass introduced a regression, correct it before adding new improvements. If the correction plus necessary new edits would exceed the one-third budget, stop and report the checkpoint as failed instead of stacking more change on top.
4. Name the exact sentences, paragraphs, headings, metadata fields, or figures that need work. Apply targeted patches only. Whole-file replacement, full-section replacement, broad paragraph reflow, and article-wide reordering are forbidden.
5. Record baseline and touched-block counts for EN and ZH, the changed-line footprint, and the percentage consumed. A pass over one third fails and must be reduced before commit.
6. Run source-fidelity, bilingual, figure, formatting, and smallest relevant site validation. Inspect the complete diff and confirm unrelated files are absent.
7. Commit the pass separately with the model and exact version in the commit subject, then push the feature branch. Do not begin the next model until the push succeeds and the worktree is clean.

The orchestrating agent, not a nested model process, owns staging, committing, and pushing. Stage only the intended post, image, and workflow files for that checkpoint.

## Paper figure selection

Before drafting from a paper, inventory every main-body figure and table by number, caption, source asset, and the claim it supports. Use that inventory to choose the smallest set of visuals that materially advances the blog's thesis. No figure is mandatory merely because the article retains its source section or because the topic is an empirical study. A style-only rewrite may remove a low-value source figure when the same evidence remains accurate and understandable in prose.

Prefer repository-owned copies of selected source figures under the post's image directory. Keep image payloads identical across EN/ZH, localize alt text and surrounding explanation, and place each figure immediately after the paragraph that introduces its claim. For every selected figure, record which thesis-bearing claim it helps the reader understand. Verify that omitted figures do not leave retained claims unsupported, but do not treat omission itself as a defect.

Git operations are allowed only through the checkpoint protocol above, on a feature branch, when the caller requested the multi-model production workflow. Never commit or push directly to `main`. Outside that protocol, return the files and report without Git operations.

## Flow A: writing a new post

1. Read the rulebooks, then gather the source material: the caller's outline, papers, repo docs, measured data, and relevant existing eunomia.dev posts. Every number must have a source; data posts require real measurements.
2. Write an **architecture brief** before prose: one-sentence thesis, reader promise, unique angle relative to sibling posts, non-goals, opening scenario or measurement, and a 3-6 H2 progression. Generate several truthful title candidates that emphasize different source-backed stakes, then choose the most compelling candidate that remains precise, differentiated, keyword-aware, and faithful. For paper-based posts, state why this outline does not mirror the paper's sections or RQs.
3. Content boundary check: blogs carry arguments, data, design decisions, and war stories. Installation steps, command references, and walkthroughs belong in product docs; if the material is a tutorial, say so and route it to docs instead of writing the post.
4. Draft the English version into `draft/blog/` (never directly into `docs/blog/posts/`): frontmatter (`date`, `slug`, `description` per the rulebooks), title that states the finding, hook before `<!-- more -->`, and the word-count and paragraph-rhythm targets from the style rulebook.
5. **Terminology map first (mandatory before any ZH prose).** Build the post's EN→ZH term map from the rulebook's Chinese terminology discipline: which terms stay English (the four allowed classes), which translate (one rendering each), and which get a first-use gloss like 策略（policy）. Keep the map in your working notes and follow it for the entire ZH draft.
6. Write the Chinese version from the architecture brief and source facts, not by translating English sentences. Keep H2/H3 progression, examples, figures, tables, claims, and caveats aligned, but let sentence, paragraph, and line boundaries differ wherever Chinese reads more naturally.
7. Run Flow B (review), including its whole-post, density, overlap, and terminology passes, on the draft.
8. Pass the reviewed draft through the sequential direct-edit workflow and final verification before reporting completion.

## Flow B: reviewing/fixing an existing post

1. Read the rulebooks, the entire target file, its bilingual counterpart, the primary source material, and 2-3 sibling posts that target the same project or keyword.
2. **Whole-post architecture pass before line editing.** Write the post's thesis in one sentence and label the role of every H2. Flag a paper-order outline, abandoned hook, section with no thesis contribution, trailing summary, or duplicated sibling angle as **Must fix**. Confirm that adjacent H2s form an argument rather than a coverage checklist.
3. **Source-transformation pass.** For a paper-based post, compare its H2 order with the paper's sections and RQs. Mark abstract-cascade paragraph openings, data ledgers, and exhaustive contribution summaries. Keep source fidelity while replacing the paper's structure and voice with practitioner-facing synthesis.
4. Check frontmatter and SEO metadata from the SEO rulebook, including title length, title appeal, description length, primary keyword placement, sibling cannibalization, and internal links. Treat a merely accurate but generic topic label as a title defect; sharpen it around the post's strongest truthful insight without turning it into clickbait.
5. **Length, load, and rhythm pass.** Record `wc -w` and `wc -l`, then identify English paragraphs above 110 words, Chinese paragraphs above 320 characters, and any run of three dense paragraphs above the style rulebook's normal ranges. Distinguish thin content from overlong or over-compressed content; adding material is not the default fix.
6. **Terminology and Chinese-independence pass.** Apply the Chinese terminology discipline paragraph by paragraph: every English token must fall into one of the four allowed classes; concept nouns with standard renderings use one consistent Chinese form; no English-headed Chinese sentences (`grep -nE '^[A-Z][A-Za-z-]+ ' file.zh.md` on prose lines); table headers and quotations are Chinese; punctuation and CJK-Latin spacing are correct. Compare EN and ZH for facts and macro structure, but flag near line-for-line correspondence or translated English rhetoric as evidence that the Chinese needs recomposition. Violations are **Must fix**.
7. **Paragraph and sentence pass.** Assign each paragraph one primary job, test whether its sentences can be reordered without loss, and apply every sentence-level rule. Rewrite overloaded, spec-sheet, and note-like paragraphs without deleting technical content.
8. Report issues grouped by severity: **Must fix** (architecture, source transformation, overlap, clarity/logic, SEO metadata, faithfulness, thin/overlong/over-compressed content, Chinese-independence errors), **Should fix** (antipatterns), **Consider** (style preferences). For each issue give the line number, quote the text, explain the problem, and suggest a concrete rewrite.
9. Apply fixes with minimal targeted edits, Must fix first, then Should fix, then evaluate each Consider item and apply the ones that improve the text. Do not silently discard anything below Must fix. Exception: when invoked read-only, report findings only and make no edits.
10. Send the resulting full draft through the sequential direct-edit workflow. Each pinned model edits the latest files directly, then Codex resolves remaining consistency issues and runs final verification.

## Mandatory multi-model completion gate

- A model-authored production post is unfinished until Codex has written and checkpointed the initial pair and the two selected non-Codex models have directly edited, validated, committed, and pushed the latest pair in sequence.
- A second session, subagent, or temperature setting of the same family does not replace a missing pinned-model pass.
- Every pass receives the whole EN/ZH pair, both rulebooks, primary sources, sibling posts, and figure inventory. Do not feed it a narrow defect list that prevents whole-post judgment.
- Require each model to inspect the complete post and previous commit before editing, make only necessary sub-one-third improvements, preserve verified content, and report the touched-block budget, what changed, and what remains uncertain.
- After all model passes, Codex checks the accumulated diff against the primary source, resolves contradictions or style drift, and verifies that no valid Must-fix issue remains.
- If one selected model is unavailable, stop before claiming completion and name the missing pass. Never silently substitute another pinned model or call a partial chain complete.

## Editing discipline (both flows)

- **Minimal targeted edits, one sentence at a time** when revising existing text. Never overwrite entire sections or paragraphs at once.
- **Do not change technical content, code blocks, YAML examples, CLI output, or architecture diagrams** during prose edits.
- **Preserve the author's meaning.** Do not soften or strengthen claims; flag questionable claims instead of rewriting them.
- **Deep pass on first attempt.** Thorough review, not just mechanical surface fixes.
- **Always diff-check** after multiple edits to ensure no content was lost.
- When a post cites a paper, verify terminology and numbers against the paper's current published version before writing or editing claim sentences; flag any mismatch as Must fix.

## Verification (before claiming done)

- `grep -c '——' file.zh.md` returns 0 and `grep -cE ' — |—' file.md` returns 0 (code blocks excepted).
- `description` present in both files and within length budget (EN 150-160 chars; ZH ~75-85 CJK-width).
- EN and ZH have the same H2/H3 count and order; both have `<!-- more -->`.
- `wc -w` and `wc -l` recorded for both files; full posts outside the rulebook's word-count range are justified, reduced, or flagged.
- Paragraph-load audit recorded: all paragraphs beyond the review tripwires and every run of three dense paragraphs are fixed or justified.
- Architecture audit recorded: one-sentence thesis, role of each H2, hook resolution, source-outline comparison, and unique angle versus sibling posts.
- Paper figure inventory and selection rationale recorded: each included figure advances a thesis-bearing claim, omitted figures leave no retained claim unsupported, and EN/ZH use the same selected images in matching positions.
- Terminology check ran as its own pass on every ZH file: `grep -nE '^[A-Z][A-Za-z-]+ ' file.zh.md` empty on prose lines, table headers Chinese, no untranslated concept nouns outside the four allowed classes.
- EN/ZH macro structure, facts, numbers, examples, figures, and caveats match; sentence, paragraph, and line counts are allowed to differ.
- Spot-check ZH mixing: no half-width commas in Chinese prose, spaces present between CJK and Latin/digits.
- Multi-model chain recorded: exact model configuration, baseline and resulting commit IDs, previous-diff verdict, touched-block and changed-line percentages, validation results, pushed checkpoint, and unresolved concerns for every pass.

## Fix priority

1. **Faithfulness & public identity:** claims diverging from sources, changed slug/URL, missing date
2. **Article architecture & differentiation:** paper-outline structure, missing thesis, abandoned hook, duplicated sibling angle, trailing restatement
3. **SEO metadata:** missing/overlong description, overlong or teasing title, keyword cannibalization, missing internal links
4. **Content scope & rhythm:** thin, overlong, over-compressed, overloaded paragraphs, data ledgers, spec-sheet prose
5. **Clarity & bilingual independence:** vague referents, missing motivation, calques, line-locked translation, terminology inconsistency
6. **Sentence mechanics & punctuation:** weak openings, passive voice, em dashes, colons before non-lists, semicolons joining independent clauses

## Output format

For review findings:
```
L<line>: "<quoted text>"
  Problem: <what's wrong>
  Fix: "<suggested rewrite>"
```

End with a summary: files produced/edited, total issues by severity, the top 3 most impactful changes, the architecture/density/overlap/Chinese-independence audit results, every verification result, every model checkpoint and diff budget, and the final Codex disposition. List any flagged text not changed, with reasons. Model identity documents reproducibility; it is never evidence of writing quality.
