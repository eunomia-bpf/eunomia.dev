# Juejin Platform Preferences

Use this reference when adapting eunomia.dev articles to Juejin, choosing
category/tags, or deciding whether an item should become a tutorial, series, or
short post.

## Source Notes

- Juejin describes itself as a Chinese developer technical content sharing and
  exchange platform. Treat the default audience as builders who want practical
  technical payoff.
- Juejin community guidance says articles are less likely to be recommended
  when they are non-original, only encyclopedic definitions, personal notes or
  book notes without personal research/insight/practice, or too short in title
  or body.
- Activity rules and platform policies change. Re-check current event pages
  before entering a campaign or contest.

## What Juejin Rewards

- Make the first paragraph useful immediately: what problem, what environment,
  what result, and what the reader can reproduce.
- Prefer tutorials, debugging writeups, architecture explanations with code, and
  measured comparisons over broad thought pieces.
- For new Juejin-native content, keep each article coherent around a useful
  technical question. Do not split syndicated long-form sources by default.
- Keep code fences labeled and command output readable. Explain the output
  enough that readers can compare it with their own environment.
- Choose category and tags based on the reader path, not on every topic the
  project touches.

## Style And Positioning

- Use a practical developer-advisor voice. The article should help the reader
  build, debug, measure, or decide.
- For eunomia.dev long-form posts, default to Chinese canonical syndication.
  Preserve the source title exactly and keep the body substantively unchanged.
  Only fix Markdown/rendering and set category, tags, cover, and summary
  metadata. Rewrite only when the user explicitly requests it for that
  publication.
- Keep the 80% contribution / 20% promotion ratio. GitHub/eunomia.dev links are
  implementation sources and next steps, not the core value.
- Visible eunomia.dev canonical/source notes are optional. Do not add or edit a
  body link solely to satisfy a checklist.
- Prefer "how to reproduce", "what changed in the runtime", "where this helps",
  and "what failed before" over launch copy.

## Audience

- Chinese developers who want working examples, commands, diagrams, and
  implementation details.
- Backend, Linux, cloud-native, security, AI-agent, performance, and
  observability readers.
- OSS practitioners who may star/fork only after the article proves practical
  value.

## Syndication Rules

- Do not rewrite an already polished Chinese eunomia.dev article just to make it
  "Juejin-native." Preserve the body and fix platform formatting.
- Preserve the opening and section order. If either needs a content fix, update
  the source first or skip syndication.
- Do not shorten, expand, localize, reorder, or split syndicated long-form
  material.
- Preserve code, commands, environment assumptions, and expected output.
- Convert project introductions into "why this helps the developer" before
  naming the repo.
- Add category and tags after reading the final draft, not before.

## Adaptation Workflow

- Extract the source facts before drafting: environment, command/code path,
  observed result, failure mode, artifact link, and the developer task.
- Choose the container first: tutorial, debugging note, architecture explainer,
  series entry, or short practical update.
- For a new Juejin-native project post, frame the project as a reproducible
  developer lesson. Do not apply this rewrite to a syndicated article.
- Keep one primary GitHub/eunomia.dev link per short update; use more links only
  in full tutorials where each link supports a step.
- For long-form posts, finish the Juejin-specific Markdown artifact locally
  before opening the editor. Paste/import the final artifact, then use the
  editor for preview, metadata, and publish settings instead of structural
  repair.
- Run the anti-AI pass: remove "干货满满", "全面升级", "深度解析" without depth,
  vague benefit stacks, and slogans that do not help a reader reproduce.

## Short-Form Style

- Treat short Juejin updates as practical developer notes: what failed, what
  command/repo/example helps, and what the reader can try next.
- Shape: problem/environment -> one technical finding -> minimal source link.
- Prefer concrete phrases such as "在 Linux/eBPF/Agent runtime 里怎么复现" over
  "项目上线", "能力升级", or "欢迎关注".
- Add one reproducible detail: command, config, kernel/runtime version,
  screenshot, trace, benchmark, or GitHub example path.
- Avoid AI-tell phrasing: generic "干货满满", abstract benefit stacks, three
  adjectives without code, and polished slogans that do not help a developer
  reproduce or evaluate the idea.
- If the short item cannot stand alone, turn it into an article draft or series
  note instead of forcing a thin post.
- Titles for original short updates should promise a concrete payoff. This does
  not authorize changing a syndicated long-form title.
- Comments should answer with versions, commands, paths, or issue links. Do not
  answer with generic appreciation only.

## Content Ratio

- Keep the 80% contribution / 20% promotion posture.
- Mention GitHub or eunomia.dev when it helps readers reproduce, inspect, or go
  deeper. Do not make the Juejin article depend on leaving the platform.
- Project announcements need at least one practical lesson, example, diagram,
  checklist, or benchmark to earn the reader's time.

## Quality Gate

- The source contains research, engineering insight, implementation detail, or
  practice experience beyond a thin announcement.
- A syndicated long-form title matches the source exactly; an original short
  post uses a concrete technical title.
- The body contains enough detail for a developer to apply or evaluate the
  idea.
- Links are specific and useful; the article still stands alone.

## Browser Checks

- Check the logged-in account, title, category, tags, cover, and publish
  settings.
- Check Markdown preview for headings, code fences, command output, tables,
  images, and links.
- Verify exact image URLs before publishing. Guessed eunomia.dev
  article-relative image URLs may 404; prefer actual rendered URLs, stable
  GitHub raw URLs for public repository images, or editor uploads.
- Check that the intro gives practical value before any project promotion.
- Check campaign/event requirements again when entering a Juejin activity.
- Stop before final publish unless the user explicitly confirms.

### Editor And Review-State Checks

- Populate the locally finished Markdown artifact in the editor, then validate
  the rendered `.markdown-body`. CodeMirror virtualizes lines and its visible
  line counter or DOM line count is not reliable evidence that newlines were
  lost.
- Wait until `图片解析中...` disappears. Verify every preview image has
  non-zero natural dimensions and a Juejin-hosted URL, then repeat the image
  check on the article page after submission.
- Tag search results may exist in a hidden dropdown after text entry. Focus the
  tag field to open the dropdown, then click only the visible `role=button`
  option; do not click hidden mirror text or zero-sized options.
- The generated 100-character summary may end mid-sentence. Replace it with a
  complete standalone summary before submission.
- A successful submission can redirect to `/published` and provide an article
  URL that displays `审核中`. Record the URL and status separately; do not call
  the item fully approved until the review marker disappears.

## Post-Publish Follow-Up

- Check comments, reactions, collections, and private messages only when the
  user asks or follow-up was part of the task.
- Reply with reproducible details, version context, GitHub issues, or docs
  links.
- Record repeated failure reports as docs/tutorial fixes.
- Do not like, follow, repost, send private messages, or make commitments
  without explicit user instruction.
