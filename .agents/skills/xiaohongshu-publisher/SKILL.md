---
name: xiaohongshu-publisher
description: Prepare eunomia.dev content for Xiaohongshu posts and visual notes. Use when asked to draft, adapt, paste, QA, or record Xiaohongshu content for blog posts, tutorials, GitHub projects, papers, demos, talks, or public technical updates. Covers platform-native title/cover framing, carousel or single-note structure, browser-only observation scans, safe stop-before-publish behavior, and media ledger updates.
---

# Xiaohongshu Publisher

Prepare Xiaohongshu-native drafts that make technical work approachable through
clear titles, visual evidence, concise Chinese copy, and a useful takeaway. Stop
before final posting unless the user explicitly confirms publication.

## Inputs

- Source Markdown path, project, repo URL, paper, screenshot set, or topic.
- Intended reader, title direction, media assets, or posting account when
  specified.
- Optional canonical link, GitHub link, tags, cover concept, or follow-up plan.

If the source path is missing, inspect `.github/publisher/posts_queue.txt`,
`.github/publisher/media/not-published.md`, recent `docs/blog/posts/`, and
relevant GitHub repos before asking the user.

## Platform Entry Points

- Discovery/search: <https://www.xiaohongshu.com/explore>
- Creator center: <https://creator.xiaohongshu.com/>

Use a browser surface with the logged-in session when UI work is required.
Never bypass authentication with search results or alternate sources.

## References

Load `references/platform-preferences.md` when choosing Xiaohongshu-native
title, cover, note shape, carousel plan, body length, tags, link treatment,
platform-observation scan, or promotion balance.

## Browser-Only Platform Boundary

Do not directly access Xiaohongshu APIs, internal endpoints, background HTTP
interfaces, or scraped datasets. All observation, drafting, QA, screenshots, and
ledger evidence must come from normal browser interactions that a regular
logged-in user can perform: navigating pages, searching, scrolling, opening
rendered notes, reading visible content, using the editor UI, and capturing
screenshots.

## Draft Preparation

1. Read the source and identify the reader problem, visible artifact, practical
   takeaway, proof, and one low-friction next step.
2. Run a short platform-observation scan only when the user asks to improve
   native fit or when the topic has no recent local pattern. Keep it light:
   record the few observations that will change the draft, and do not copy
   wording or hooks from other creators.
3. Choose the format:
   - single note for one sharp observation, screenshot, diagram, or result
   - carousel for a mini walkthrough, comparison, checklist, or debugging story
   - longer body only when the note needs context after the visual hook
4. Draft from Chinese-native structure, not from a translated LinkedIn post.
5. Keep GitHub/eunomia.dev/paper links as sources or next steps after the note
   already gives useful value.

## Draft Archive

Before opening the Xiaohongshu editor, write or update the draft under
`draft/media/YYYY-MM-DD/<source-slug>/xiaohongshu.md` using the local date.
Include the paste-ready title, body, cover concept, image/carousel plan, source
links, tags, first-screen check, browser-observation notes when used, and QA
state.

## Browser QA

Before stopping for user confirmation, verify:

- title and cover communicate the concrete problem or result at first glance
- first screen is understandable without clicking an external link
- carousel order works as a visual story, not just cropped paragraphs
- images, diagrams, code screenshots, and captions are legible on mobile
- tags are relevant and minimal
- account, visibility, and publish settings are intentional
- no confidential customer, partner, roadmap, pricing, or strategy information
  appears
- the visible final publish action has not been clicked

## Follow-Up

After a confirmed publish, capture the public URL and monitor comments, saves,
messages, and repeated reader questions only when the user asks or follow-up was
part of the task. Draft replies that add technical context, source links, or
concrete next steps. Do not send private messages, likes, follows, comments, or
reposts without explicit instruction.

## Ledger Update

After confirmed publish, update `.github/publisher/media/published.md` with
source path, Xiaohongshu URL, date, format, media, link target, tags, and
follow-up notes. Record useful formatting or audience lessons in this skill or
its references so the next Xiaohongshu launch starts from the learned pattern.
