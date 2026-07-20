# DEV Community Platform Preferences

## Source Notes

- DEV uses a Markdown editor with Jekyll front matter, supports Liquid embeds,
  and shareable unpublished draft links.
- DEV formatting guidance says the post title is the H1, so body headings should
  start at H2. It also emphasizes image descriptions, limited emoji use, and
  checking Markdown before publishing.

## Style And Positioning

- DEV should feel like a practical developer post: concrete setup, code,
  commands, result, and what to try next.
- For eunomia.dev long-form posts, default to English canonical syndication.
  Preserve the body and improve only frontmatter, headings, images, code
  fences, links, tags, and preview rendering unless the user asks for a rewrite.
- Keep the 80% contribution / 20% promotion ratio. The article should stand on
  DEV even though it links back to eunomia.dev or GitHub.
- Treat the project as a reproducible implementation, not a banner ad.

## Audience

- Developers looking for tutorials, examples, debugging notes, and practical
  comparisons.
- For eunomia.dev topics: backend, Linux, cloud-native, AI-agent, security,
  observability, and OSS readers.

## Syndication Rules

- Convert site frontmatter to DEV fields and add `canonical_url` for syndicated
  posts when known and convenient. Do not add a visible body source link solely
  to satisfy a checklist.
- Do not turn a research article into a fake tutorial. Keep the canonical body
  when it already has a developer-relevant problem, artifact, mechanism, or
  implementation lesson.
- Shorten dense academic openings only when they block readability or the user
  asks for a DEV-native rewrite.
- Keep code examples copy-pasteable with expected output when possible.
- Use up to a small set of focused tags; do not tag every adjacent topic.
- If the source is long, make it a series or split into one practical task per
  post.

## Adaptation Workflow

- Extract the source facts before drafting: developer task, prerequisites,
  commands/code, expected result, GitHub path, source URL, and failure mode.
- Decide whether the post is canonical syndication, tutorial, debugging note,
  comparison, series part, or practical announcement. Do not publish thin
  updates without developer value.
- Keep the first screen useful: what the reader can build, inspect, reproduce,
  or avoid.
- Use specific GitHub, docs, paper, or project links when they support the task.
  A visible eunomia.dev original/canonical note is optional.
- For long-form posts, finish the DEV frontmatter and Markdown artifact locally
  before opening the editor. Use the web editor for preview, metadata, and
  publish flow instead of structural repair.
- Run the anti-AI pass: remove "ultimate guide" without scope, "deep dive",
  "unlock", "game changer", abstract benefit stacks, and examples that do not
  run.

## Short-Form Style

- DEV short-form surfaces are titles, descriptions, series blurbs, comments, and
  short practical posts.
- Start with what the developer can reproduce, inspect, or debug.
- Include one concrete artifact: command, code snippet, GitHub example path,
  error message, benchmark result, or environment version.
- Avoid AI-tell phrasing: "ultimate guide" without scope, generic "deep dive",
  "unlock", "game changer", abstract benefit stacks, and examples that do not
  run.
- For comments, answer with code/version context or a link to the exact repo
  issue/docs section. Do not reply with generic appreciation only.
- If a short post lacks runnable detail, save it as a draft idea until there is
  a reproducible example.
- Titles should name the task and boundary, not only the technology category.
- Descriptions should contain one problem, one mechanism/result, and one reason
  the author knows this from implementation work.

## Browser Checks

- Preview Markdown rendering before publishing.
- Use the DEV web editor and visible submit buttons; do not publish through DEV
  APIs or background endpoints.
- Check H2/H3 hierarchy, code highlighting, image rendering, and embeds.
- Check `canonical_url` when configured and draft visibility.
- Check that GitHub links point to specific repos, examples, issues, or docs.
- After publishing, open the public DEV URL and scroll the rendered post from
  top to bottom before marking the post complete.
- Fix public-page issues through the DEV web UI and re-check. Common issues
  include duplicated canonical/source notes, wrong tags, broken images, heading
  artifacts, and code fences rendered as plain prose.
- Verify selected tag chips after every tag change. Typed-but-unaccepted tags
  are not selected tags.
- With `canonical_url` enabled, rely on DEV's built-in "Originally published"
  notice unless a manual source note adds distinct value.
- Verify exact image URLs before saving. Guessed eunomia.dev article-relative
  image URLs may 404; use actual rendered URLs, stable GitHub raw URLs for
  public repository images, or DEV web-editor uploads, then full-scroll the
  public page so lazy-loaded images enter the viewport.

## Post-Publish Follow-Up

- Watch comments for reproducibility failures, version drift, or missing
  context.
- Reply with commands, links, and corrections.
- Turn recurring questions into docs/tutorial updates and record them in the
  publishing ledger.
