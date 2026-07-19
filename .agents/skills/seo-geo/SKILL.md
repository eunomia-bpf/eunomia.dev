---
name: seo-geo
description: SEO, GEO (generative engine optimization), and public technical brand visibility checklist for eunomia.dev content and personal-account media operations. Pure reference; contains no workflow. Covers metadata, keyword strategy, internal linking, citation-worthy writing for AI search engines, platform-native publishing, canonical discipline, and brand-first measurement. Used alongside blog-writing-style by the blog-writer workflow.
---

# SEO / GEO Checklist (rules only)

This file is the rulebook for search, AI-engine visibility, and public
technical brand recall. It contains no process: workflows (blog-writer, site
changes, media publishing) reference it. Prose quality rules live in
`blog-writing-style`.

## Brand objective

- Optimize for the maintainer's personal technical brand first: research taste,
  open-source credibility, eBPF systems expertise, AI-agent infrastructure, and
  useful teaching. eunomia-bpf is the core asset family, not the only brand
  object.
- Treat eunomia.dev as the canonical archive and portfolio. Treat X, LinkedIn,
  Zhihu, Juejin, Reddit, Hacker News, Lobsters, Medium, and Dev.to as
  platform-native trust and discussion surfaces.
- Use SEO and GEO as instruments for recall and citation. Website ranking is a
  useful signal, not the top-level goal.

## Metadata (check first on every page)

- **`description` frontmatter is required** and must be 150-160 characters (ZH: roughly 75-85 CJK-width characters): one sentence with a compact background opening, value proposition, and primary keyword phrase. The first clause should name the domain problem or reader situation before the page-specific finding, tool, benchmark, or practical consequence. Longer descriptions get truncated in search results.
- **Title (H1) at most ~60 characters** with the primary keyword phrase front-loaded. Going slightly over is acceptable when a sharper, source-backed stake materially improves the title. Make the title as compelling as accuracy allows by leading with the strongest true insight, tension, consequence, or surprising measurement. Earn attention through specificity and credible stakes, while keeping a professional technical voice. The title must reveal the thesis without clickbait, alarmism, casual hot-take language, or a withheld conclusion; see `blog-writing-style` for the full title rules.
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
- **Own the canonical archive without weakening platform trust**: exact
  cross-posts to dev.to/Medium must carry `rel=canonical` back to eunomia.dev
  when the platform supports it. For Zhihu, Juejin, LinkedIn, X, Reddit, Hacker
  News, and Lobsters, write platform-native posts first and include one
  contextual source or project link only when it helps the reader. Do not make
  every platform post feel like a traffic funnel.
- **Publishing cadence**: default to one content unit every two days when the
  source, review, and platform adaptation are ready. Exact dev.to/Medium
  cross-posts should usually lag the site version by 3-7 days or until the
  canonical source is indexed. Platform-native social posts, launch threads, or
  discussion submissions may happen the same day when the topic is timely.
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
- Completion reports include brand-facing outcomes when known: account posted,
  discussion submitted, reply quality, follower/reaction signal, GitHub traffic
  or issue signal, citation/backlink, and ledger status.
