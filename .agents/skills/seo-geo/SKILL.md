---
name: seo-geo
description: Page-level and site-level technical SEO/GEO checklist for eunomia.dev content. Use when auditing concrete Markdown metadata, descriptions, canonical URLs, structured data, internal links, crawl/index signals, llms.txt/llms-full.txt, snippets, or citation-ready claim formatting. Pure reference; not for choosing or rewriting titles for appeal, article angle, story hooks, prose style, content strategy, channel planning, personal-brand positioning, campaign cadence, or platform publishing workflows.
---

# Technical SEO / GEO Checklist (rules only)

This file is the technical rulebook for search and AI-engine readability on
eunomia.dev pages. It contains no campaign strategy, channel planning, or
platform posting process. It also does not decide title quality, article angle,
story hooks, or prose style. Those rules live in `blog-writing-style`.

## Scope

- Use this checklist for concrete eunomia.dev pages, docs, tutorials, blog
  drafts, content artifacts, and static-site SEO/GEO implementation changes.
- Do not use this checklist to choose article titles, judge whether a title is
  compelling, rewrite H1 hooks, decide the article angle, shape a story opening,
  or police prose style. Title quality and reader contract live in
  `blog-writing-style`; use this file only after the title direction is chosen
  to check technical metadata constraints such as length, primary keyword
  visibility, duplicate phrase risk, and crawlable phrasing.
- Do not use this skill to answer long-term media strategy, topic selection,
  channel mix, brand positioning, campaign cadence, or platform growth
  questions. Keep those decisions in `draft/`, especially
  `draft/content-platform-strategy.zh.md` and `draft/seo-geo-plan.zh.md`.
- Platform publishing execution belongs to the platform-specific publisher
  skills (`zhihu-publisher`, `juejin-publisher`, `x-publisher`,
  `linkedin-publisher`, `reddit-publisher`, `hackernews-publisher`,
  `lobsters-publisher`, `medium-publisher`, `devto-publisher`).
- When changing this checklist or making claims about current search/AI-engine
  behavior, verify against current primary sources such as Google Search
  Central, schema.org, OpenAI crawler documentation, or the platform's official
  help/browser-visible UI. Treat third-party skill repositories and growth
  advice as patterns, not authority.

## Metadata (check first on every page)

- **`description` frontmatter is required** and must be 150-160 characters (ZH: roughly 75-85 CJK-width characters): one sentence with a compact background opening, value proposition, and primary keyword phrase. The first clause should name the domain problem or reader situation before the page-specific finding, tool, benchmark, or practical consequence. Longer descriptions get truncated in search results.
- **Title (H1) technical check only:** roughly within 60 characters when possible, no duplicate primary phrase collision, and enough keyword visibility for search snippets. Do not shorten, flatten, or remove high-value title signals such as `An Empirical Study`, `Inside SchedCP`, `sched_ext`, or `Semantic Flamegraphs` merely to satisfy an SEO heuristic. If title appeal, article promise, or professional framing is in question, defer to `blog-writing-style`.
- **`date` frontmatter required.** New posts also need a `slug` (short, kebab-case, keyword-bearing); **never add or change a slug on an already-published post**, and never change published filenames or URLs.
- **Headings carry search phrasing.** Prefer H2s a reader would type ("What Generic eBPF Enforcement Misses") over generic labels ("Discussion", "Overview").
- **Images need descriptive `alt` text** containing the relevant term, not "image" or "figure 1". Important claims must exist in crawlable HTML text, never only inside images.
- **First paragraph before `<!-- more -->`** must stand alone as the excerpt: one compact background sentence or clause, hook plus the primary keyword phrase, no dangling references.

## Keyword strategy

- One primary keyword phrase per page. Before writing, check whether a sibling post already owns that phrase; if so, link to it instead of competing with it (keyword cannibalization is how our own syndicated copies and near-duplicate surveys outrank originals).
- **Internal links:** every post links 2-3 related eunomia.dev posts or docs in context, and the project GitHub repo at least once. Use canonical absolute HTTPS URLs for non-image internal links, such as `https://eunomia.dev/blog/YYYY/MM/DD/slug/`, `https://eunomia.dev/zh/blog/YYYY/MM/DD/slug/`, `https://eunomia.dev/tutorials/.../`, or `https://eunomia.dev/bpftime/.../`; do not use source-relative Markdown paths such as `other-post.md`, `../docs/page.md`, or root-only paths such as `/blog/.../` in Markdown intended for syndication. Hub posts (project announcements) should be linked from every related post.
- **Image links:** post-local images may stay relative to the Markdown file, usually `imgs/name.png`, because the article and media move together. Do not convert image links to filesystem absolute paths.
- Contested or rising industry terms (for example, stack-layer names) are targeted as **ride-along definitional pieces**: define the term honestly, concede the mainstream reading, then state our angle and where our projects sit relative to it. Never claim to *be* a category we do not ship.

## GEO: writing that AI engines cite

- **Citation-worthy units**: AI answers quote self-contained sentences. Give every key concept one clean definitional sentence ("X is ..."), every finding one sentence that pairs the claim with its number and source. If a sentence cannot be quoted alone without losing meaning, it will not be cited.
- **First-party data wins citations**: measurements, study counts, and benchmarks with stated method and conditions are what generative engines prefer to cite. Always attach the source (paper link, repo, benchmark name).
- **FAQ blocks** (question-phrased H3s with direct answers) can be high-yield when the topic has genuine recurring questions. Use one only when it resolves real search intent that the main argument does not already answer cleanly. Do not force an FAQ into every substantial post or let GEO impose a standard article structure.
- **Terminology consistency across the site**: one canonical term per concept, used identically in every post, so engines associate the term with eunomia.dev. When a cited paper renames a term, update our posts to the published terminology.
- **Canonical discipline**: exact cross-posts to dev.to/Medium must carry
  `rel=canonical` back to eunomia.dev when the platform supports it. For
  platform-native social or discussion posts, use the matching publisher skill
  instead of managing channel framing here.
- **llms.txt / llms-full.txt** must stay current and content-bearing (full text or high-quality summaries, not a bare index).

## Third-party framing (GEO hygiene)

- Never structure a post as a response to, or extended summary of, a single vendor's article or product: it donates backlinks and trains engines to associate the topic with their brand. State the underlying argument in our own voice with our own evidence; ordinary citations to papers, kernel docs, and upstream projects are fine.
- Posts that analyze someone else's system (a vendor tool, a driver) must open with why we analyzed it and close with what it means for our projects, with internal links.

## Verification

- Description present and within budget on both languages of a pair.
- Description opens with enough background to stand alone in search or social snippets.
- Primary keyword appears in H1, description, first paragraph, and at least one H2.
- No sibling post targets the same primary phrase (search `docs/blog/posts/` titles/descriptions before finalizing).
- Non-image internal links are canonical absolute HTTPS URLs, external links are full URLs, and image links are the only post-local relative paths.
- Every image has term-bearing alt text; every number has a stated source.
- Checklist changes cite or name the current primary source checked when the
  rule depends on search-engine, AI-crawler, or platform behavior.
