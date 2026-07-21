---
name: blog-writer
description: Write or revise bilingual eunomia.dev blog posts from primary sources by preparing complete drafts, having exactly claude-opus-4-5 edit them paragraph by paragraph with the repository prompt template, and performing a bounded final verification.
---

# Blog Writer

Use this workflow for posts under `docs/blog/posts/` and article drafts under
`draft/blog/`.

1. Collect the original paper PDF or text, the official repository snapshot,
   and other necessary primary material in a temporary directory outside the
   repository. Start from the complete existing EN/ZH articles. If no complete
   article exists, Codex writes the complete source-grounded drafts first.
2. Copy `.github/prompts/blog-edit.prompt.md` into the temporary task and replace
   only its paper, repository, and article paths. The identical skill-local copy
   is `references/blog-edit.prompt.md`; keep both templates synchronized. Keep
   the template's single reader/style sentence. Do not attach
   `blog-writing-style`, an outline, examples, review findings, extra style
   rules, or another checklist to the Opus prompt.
3. Copy the drafts to working output files and run exactly
   `claude-opus-4-5`. Give Opus `Read`, `Edit`, `Grep`, and `Glob`; let it use its
   own judgment about the title, structure, content, and wording. Its only
   editorial operation constraint is the template: work from top to bottom,
   edit one paragraph per `Edit`, and never replace a complete file at once.
4. Confirm the exact model from the run output, then have Codex verify the full
   result against the primary sources and check frontmatter, links, figures,
   tables, code, references, and Markdown integrity. Codex may change punctuation
   or replace one or two words where necessary, such as replacing an em dash.
   It must not rewrite a sentence, paragraph, title, structure, argument, or
   substantive content. Report any larger remaining issue instead of editing it.
5. Run `git diff --check` and the smallest relevant content validation. Report
   changed files, the exact Opus model, validation performed, and unresolved
   factual issues.

Keep source packages, prompts, and run logs outside the repository. Do not
create outlines, evidence ledgers, review logs, figure inventories, or
publication-QA artifacts unless the user explicitly requests them.
