---
name: blog-writer
description: Write or revise bilingual eunomia.dev blog posts from primary sources by preparing complete drafts, running one claude-opus-4-5 paragraph-by-paragraph edit with the unchanged repository prompt template, and performing bounded mechanical verification without editorial loops.
---

# Blog Writer

Use this workflow for posts under `docs/blog/posts/` and article drafts under
`draft/blog/`.

Treat the user's requested workflow as a hard ceiling. Do not add stages,
constraints, review rounds, alternative drafts, comparisons, or retries. A new
Opus pass requires a separate, explicit user request after the previous result
has been presented.

1. Collect the original paper PDF or text, the official repository snapshot,
   and other necessary primary material in a temporary directory outside the
   repository. Start from the complete existing EN/ZH articles. If no complete
   article exists, Codex writes the complete source-grounded drafts first.
2. Copy `.github/prompts/blog-edit.prompt.md` into the temporary task and replace
   only its paper, repository, and article paths. The identical skill-local copy
   is `references/blog-edit.prompt.md`; keep both templates synchronized. Use
   the template verbatim apart from those path substitutions. Do not attach or
   inject Codex's own
   `blog-writing-style`, an outline, examples, review findings, extra style
   rules, content instructions, success criteria, or another checklist. Only an
   explicit user request may replace or extend the template.
3. Copy the drafts to working output files and run exactly
   one `claude-opus-4-5` pass. Give Opus `Read`, `Edit`, `Grep`, and `Glob`; let
   it use its own judgment about the title, structure, content, and wording. Its
   only editorial operation constraint is the template: work from top to
   bottom, edit one paragraph per `Edit`, and never replace a complete file at
   once. Do not rerun or resume Opus, and do not send Codex's findings back to
   Opus within the same request.
4. Confirm the exact model from the run output, then have Codex verify the full
   result against the primary sources and check frontmatter, links, figures,
   tables, code, references, and Markdown integrity. This is a mechanical
   verification step, not another editorial pass. Codex may change punctuation
   or replace one or two words where necessary, such as replacing an em dash.
   It must not rewrite a sentence, paragraph, title, structure, argument, or
   substantive content. Apply this final reminder literally:
   `检查标点符号和一两个词的多样性。除了符号和一两个词语的替换，不能改别的东西。`
   Do not replace every em dash with the same mark; choose a comma, semicolon,
   colon, parentheses, or period according to the existing relationship. Report
   any larger remaining issue instead of editing it. Do not start another review
   or ask Opus to revise it.
5. Run `git diff --check` and the smallest relevant content validation. Report
   changed files, the exact Opus model, validation performed, and unresolved
   factual issues.

Keep source packages, prompts, and run logs outside the repository. Do not
create outlines, evidence ledgers, review logs, figure inventories, or
publication-QA artifacts unless the user explicitly requests them.
