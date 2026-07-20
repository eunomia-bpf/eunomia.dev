# Medium Platform Preferences

## Source Notes

- Medium's distribution guidance values thoughtful, nuanced, knowledgeable
  perspectives, direct experience, reader value, respect for the reader, fresh
  perspective, good sourcing, and useful images with alt text/credits.
- As of Medium's June 29, 2026 guidance, English stories are eligible for Boost
  review while non-English stories are not currently Boosted.
- Medium allows republishing from other sites; when syndicating eunomia.dev
  material, preserve the canonical relationship when available. Do not add a
  visible body source link solely to satisfy a checklist.

## Style And Positioning

- Write as a researched technical essay with narrative clarity.
- For eunomia.dev long-form posts, default to English canonical syndication.
  Preserve the source title exactly and keep the body substantively unchanged.
  Only fix rendering and set platform metadata. Rewrite only when the user
  explicitly requests it for that publication.
- Keep the 80% contribution / 20% promotion ratio. Medium explicitly disfavors
  content whose primary purpose is traffic, signups, sales, or PR.
- Use first-hand engineering experience, measurements, implementation decisions,
  and tradeoffs as the reason the author is worth reading.

## Audience

- English-speaking technical readers who want a polished explanation, not a raw
  changelog.
- Broader than Juejin/Lobsters: include enough context for software engineers
  who are not already eBPF specialists.

## Syndication Rules

- Do not paste blindly: remove front matter, verify headings, images, code
  blocks, links, tags, and canonical/import settings.
- Keep the title, opening, section order, examples, claims, and conclusion intact.
- If the source lacks context or has a content problem, fix the source first or
  skip syndication. Do not repair prose in the Medium editor.
- Preserve a source subtitle when one exists. Leave it blank when optional
  rather than inventing new positioning for a syndicated article.
- Use original diagrams/screenshots when they help; avoid generic AI cover art.
- Keep GitHub links in "how to inspect/reproduce" positions.

## Adaptation Workflow

- Extract the source facts first: thesis, narrative tension, primary evidence,
  diagrams/images, canonical URL, GitHub links, and reader outcome.
- Choose whether the piece is an imported canonical story, publication
  submission, or short field note. Medium should not receive every post.
- Preserve the introduction. Missing context belongs in the source article, not
  in a platform-only variant.
- Treat cover images, subtitles, and excerpts as separate writing surfaces. They
  should be specific, not generic "AI infrastructure" packaging.
- Run the anti-AI pass: remove "deep dive", "unlock", "game-changer", inflated
  significance, neat but empty conclusions, and source-free broad claims.

## Short-Form Style

- Medium is mainly long-form, but titles, subtitles, excerpts, and publication
  notes need short-form discipline.
- For original Medium notes, titles and subtitles should state the reader value
  without mystery hooks or tabloid energy. This does not authorize changing a
  syndicated long-form title.
- Excerpts should include one concrete problem, one mechanism/result, and the
  reason this author has first-hand insight.
- Avoid AI-tell phrasing: generic "deep dive", "unlock", "game changer",
  stacked abstractions, and summaries that could fit any AI infrastructure post.
- If publishing a short note instead of a full essay, make it a field note with
  one observation and one source link, not a thin project announcement.
- Use one primary reader promise in the title/subtitle pair. Do not stack every
  possible audience or benefit into the front matter.
- Publication notes should disclose canonical origin when relevant and avoid
  sounding like a press release.

## Browser Checks

- Check canonical URL/import settings when configured.
- Check title, subtitle, cover image, alt text, credits, and topic tags.
- Before publishing, scroll the imported/editor story from top to bottom in the
  browser preview or editor surface.
- After publishing, open the public Medium URL and scroll the rendered story
  from top to bottom before marking the post complete.
- Check image loading, headings, code blocks, tables or readable table
  fallbacks, link targets, and image sizing on desktop and narrow viewport if
  practical.
- Fix public-page issues through the Medium web UI and re-check. Common import
  failures include site suffixes in titles, flattened tables, empty heading
  artifacts, and code-language labels inserted into prose.
- Prefer readable list/prose fallbacks over fragile Markdown tables when Medium
  import flattens table rows into separate paragraphs.
- Check that the story does not read as sponsored content or a PR release.

## Post-Publish Follow-Up

- Watch responses and private notes for clarification requests.
- Record reader questions as future blog/FAQ/tutorial ideas.
- If a publication reviews the piece, track requested edits separately from the
  canonical eunomia.dev source.
