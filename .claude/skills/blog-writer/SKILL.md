---
name: blog-writer
description: Process control for producing eunomia.dev blog posts, covering both writing a new post and reviewing/fixing an existing one under docs/blog/posts/. Drives the workflow (drafting, EN/ZH pairing, severity-ranked review, editing discipline, verification); all style/SEO rules come from the blog-writing-style checklist. Use for "write a blog", "draft a post", "review this post", "fix the prose in X".
allowed-tools: Read Write Edit Bash(grep *) Bash(wc *) Bash(ls *)
---

# Blog Writer (process)

Produce or improve the blog post given in `$ARGUMENTS`. If no argument is given, ask for the topic (new post) or file path (existing post). Posts live in `docs/blog/posts/` as `post.md` (English) and `post.zh.md` (Chinese) pairs.

This skill is the workflow only. **Read both rulebooks first, every time**: `.claude/skills/blog-writing-style/SKILL.md` (prose mechanics, blog antipatterns, content-farm bans, length/richness, Chinese-English mixing, bilingual consistency) and `.claude/skills/seo-geo/SKILL.md` (metadata, keyword strategy, GEO citation-worthiness, syndication canonical discipline). Do not restate or override their rules here.

Model note: the actual writing/editing pass is best run on `claude-opus-4-6[1m]` (Opus 4.6, 1M context); when the calling agent is a different model, delegate this skill's work to an Opus subagent (`.claude/agents/prose-writer.md`).

Do not perform any Git operation. Return the finished files and a report to the caller.

## Flow A: writing a new post

1. Read the rulebook, then gather the source material: the caller's outline (preferred; if none exists, write a detailed outline first and confirm it with the caller), papers, repo docs, measured data. Every number must have a source; data posts require real measurements.
2. Content boundary check: blogs carry arguments, data, design decisions, and war stories. Installation steps, command references, and walkthroughs belong in product docs; if the material is a tutorial, say so and route it to docs instead of writing the post.
3. Draft the English version into `draft/blog/` (never directly into `docs/blog/posts/`): frontmatter (`date`, `slug`, `description` per the rulebook), title that states the finding, hook before `<!-- more -->`, full-post length target from the rulebook.
4. Write the Chinese version with identical section structure and figure/table placement, applying the rulebook's Chinese-English mixing rules.
5. Run Flow B (review) on your own draft before reporting back.

## Flow B: reviewing/fixing an existing post

1. Read the rulebook, then the entire target file and its bilingual counterpart, if one exists.
2. Check frontmatter and SEO metadata first (rulebook's SEO checklist).
3. Check length and richness (rulebook's expectations, ~200 lines for a full post); thin content is a Must fix, resolved by adding substance from primary sources, never padding.
4. For ZH files, run the Chinese-English mixing checks as a dedicated pass: terminology consistency, term-of-art vs ordinary-word language choice, CJK-Latin spacing, full-width punctuation, no English clause splicing.
5. For each paragraph, analyze every sentence against the rulebook.
6. Report issues grouped by severity: **Must fix** (clarity/logic/SEO-metadata/faithfulness/thin-content errors), **Should fix** (antipatterns), **Consider** (style preferences). For each issue give the line number, quote the text, explain the problem, and suggest a concrete rewrite.
7. Apply the fixes with the Edit tool, Must fix first, then Should fix, then evaluate each Consider item and apply the ones that improve the text (note rejected ones with a one-line reason). Do not ask the user which fixes to apply, and do not silently discard anything below Must fix. Exception: when invoked read-only as a review subagent, report findings only and make no edits.

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
- `wc -l` both files; a full post below ~150 lines must have been flagged (Flow B) or justified (Flow A).
- Spot-check ZH mixing: no half-width commas in Chinese prose, spaces present between CJK and Latin/digits.

## Fix priority

1. **SEO metadata & faithfulness:** missing/overlong description, missing date, titles that tease instead of stating the finding, claims diverging from the cited source
2. **Thin content:** sections below the richness bar
3. **Clarity:** dangling modifiers, vague referents, missing motivation
4. **Structure:** note-like prose, weak openings, passive voice, blog antipatterns
5. **Word choice & ZH mixing:** verbose phrases, terminology inconsistency, spacing/punctuation
6. **Punctuation:** em dashes

## Output format

For review findings:
```
L<line>: "<quoted text>"
  Problem: <what's wrong>
  Fix: "<suggested rewrite>"
```

End with a summary: files produced/edited, total issues by severity, the top 3 most impactful changes, and the results of every verification command. List any sentences flagged but NOT changed, with reasons.
