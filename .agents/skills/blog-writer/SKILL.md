---
name: blog-writer
description: Write or revise bilingual eunomia.dev blog posts from primary sources, verify technical claims, and prepare them for publication without requiring a particular editor or model.
---

# Blog Writer

Use this workflow for posts under `docs/blog/posts/` and article drafts under
`draft/blog/`.

Start from the existing English and Chinese articles, or write complete drafts
when they do not exist. Verify claims against the original paper, official
repository, measurements, and other primary sources relevant to the post.
Separate measured results from hypotheses and proposed work.

The editor may be Codex or another available agent. Choose the editing method
and number of passes to fit the article and the user's request; no model,
provider, prompt template, or tool is mandatory. `references/blog-edit.prompt.md`
is an optional prompt, not a publication gate. Apply `blog-writing-style` to
the finished prose while keeping the English and Chinese claims aligned.

Before publication, inspect the complete diff for lost content and check
frontmatter, links, figures, tables, code, references, and Markdown integrity.
Run `git diff --check` and the smallest relevant site validation. Report the
changed files, validation, and any unresolved factual or deployment issue.

Keep temporary source packages, prompts, and run logs outside the repository.
Do not add process artifacts unless the user asks for them.
