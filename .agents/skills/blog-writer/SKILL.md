---
name: blog-writer
description: Write or revise bilingual eunomia.dev posts. Codex prepares the source-grounded pair, claude-opus-4-5 rewrites once for a comfortable human reading experience, and Codex validates facts and publication integrity.
---

# Blog Writer

Use this workflow for posts under `docs/blog/posts/` and article drafts under
`draft/blog/`. English uses `post.md`; Chinese uses `post.zh.md`.

Before editing, read:

- `.agents/skills/blog-writing-style/SKILL.md` for the desired writing result
- `.agents/skills/seo-geo/SKILL.md` for technical metadata and link checks
- the primary source and relevant sibling posts

The style guide defines the result. This skill owns models, steps, permissions,
and validation. Do not turn style recommendations into extra review rounds.

## Workflow

1. **Codex prepares the pair.** Build or revise a complete EN/ZH article from
   verified sources. Establish one clear argument, preserve evidence and
   limitations, and keep both languages aligned in claims, figures, tables,
   code, and references. Use `draft/blog/` only while a new article's public
   filename, date, or slug is unsettled. Do not create process artifacts such
   as figure inventories, review logs, or publish-QA notes.
2. **Opus rewrites once.** Run `claude-opus-4-5` on the complete pair. Ask it
   to make the article comfortable for a technically qualified human reader,
   with natural paragraph rhythm and a clear path from evidence to conclusion.
   It may recompose prose and paragraph boundaries wherever that genuinely
   improves the complete reading experience. It must preserve the factual and
   publication boundaries below. Do not treat shorter sentences as clearer by
   default: preserve a complete condition-action-cause-consequence unit when it
   reads naturally as one sentence, and reserve short sentences for real
   emphasis or conceptual turns. A no-change result is valid.

   Give Opus concrete failure examples in the task prompt. In particular, it
   must not turn one connected sentence such as "The kernel sees an ordinary
   write, while the harness sees one tool call, yet neither can decide whether
   the commit is allowed" into "The kernel sees a write. The harness sees a
   tool call. Neither can decide." The same failure in Chinese is “内核看到一次
   写入。Harness 看到一次调用。两者都无法判断。” Conditions, contrast, and
   consequence belong together when they express one thought. Also reject
   paragraphs that read as interchangeable fact cards, repeated transitions
   such as "The study... The result... This shows...", or background that is
   either absent before the first technical claim or expanded into an unrelated
   general tutorial. Consecutive Chinese paragraphs must not begin with abstract
   outline announcements such as `跨事件策略反复出现为四类关系。`、`上下文依赖让
   强制执行更难落地。`、`两类难点会叠加。`; the next paragraph should continue
   from a concrete result, example, tension, or question whenever the argument
   naturally does so.
3. **Codex validates.** Inspect the complete diff, reject factual or stylistic
   regressions, compare important claims and numbers with their sources, and
   verify EN/ZH correspondence. Make only the bounded corrections needed for
   accuracy, natural Chinese, metadata, links, images, or build health. Do not
   add another general rewrite pass.
4. **Deliver once.** Run the smallest relevant site validation. For PR-bound
   public content, push the final result once after validation rather than
   triggering CI for intermediate drafts.

## Model Routing

- Codex author and final validator: `gpt-5.6-sol` with
  `model_reasoning_effort=xhigh`
- Default and full-rewrite editor: `claude-opus-4-5`

Small metadata fixes, factual token corrections, title changes explicitly
requested by the user, and one bounded paragraph may be handled directly by
Codex. Complete new posts, multi-section revisions, tone rebuilds, and full
language rewrites use Opus.

If `claude-opus-4-5` is unavailable, report the blocker. Do not silently
substitute another model. An alternative editor requires user authorization.

## Edit Permissions

Give Opus only the named article files, source material, sibling posts, and the
two reference skills. It may edit those article files but must not commit,
push, publish, change repository policy, or create workflow artifacts.

Do not impose a paragraph-count or percentage edit quota. Opus may leave good
passages unchanged or rewrite broadly when the user requested a rewrite. Every
change must still preserve:

- published filenames, URLs, dates, slugs, and titles unless the user explicitly
  requested a change
- verified claims, numbers, denominators, conditions, caveats, and terminology
- figures, tables, code blocks, links, references, and useful examples
- the argument and content scope selected for the article
- natural but substantively aligned EN/ZH versions

Never invent surprise, failure, uncertainty, personal experience, or
first-person narrative to make AI-assisted prose sound human.

## Validation

Before completion, Codex checks:

- source fidelity for important claims, numbers, conditions, and limitations
- title, description, date, slug, tags, excerpt marker, links, images, and final
  references
- corresponding EN/ZH argument, evidence, figures, tables, and caveats
- comfortable first-read flow against `blog-writing-style`
- `git diff --check` and the smallest relevant site build or content validation
- browser rendering of both languages after deployment when publication is in
  scope

Report the files changed, the exact Opus model, validation performed, and any
remaining factual uncertainty.
