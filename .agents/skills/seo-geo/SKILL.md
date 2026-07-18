---
name: seo-geo
description: SEO and GEO (generative engine optimization) rule checklist for eunomia.dev content, covering blog posts, docs pages, and landing pages. Pure reference; contains no workflow. Covers metadata (description, title, slug), keyword strategy, internal linking, citation-worthy writing for AI search engines, and syndication canonical discipline. Used alongside blog-writing-style by the blog-writer workflow.
---

# SEO / GEO Checklist (rules only)

This file is the rulebook for search and AI-engine visibility. It contains no process: workflows (blog-writer, site changes) reference it. Prose quality rules live in `blog-writing-style`.

## Metadata (check first on every page)

- **`description` frontmatter is required** and must be 150-160 characters (ZH: roughly 75-85 CJK-width characters): one sentence, value proposition first, primary keyword phrase included. Longer descriptions get truncated in search results.
- **Title (H1) at most ~60 characters** with the primary keyword phrase front-loaded. Going slightly over is acceptable when a sharper, source-backed stake materially improves the title. Make the title as compelling as accuracy allows by leading with the strongest true insight, tension, consequence, or surprising measurement. It must reveal the thesis without clickbait or a withheld conclusion; see `blog-writing-style` for the full title rules.
- **`date` frontmatter required.** New posts also need a `slug` (short, kebab-case, keyword-bearing); **never add or change a slug on an already-published post**, and never change published filenames or URLs.
- **Headings carry search phrasing.** Prefer H2s a reader would type ("What Generic eBPF Enforcement Misses") over generic labels ("Discussion", "Overview").
- **Images need descriptive `alt` text** containing the relevant term, not "image" or "figure 1". Important claims must exist in crawlable HTML text, never only inside images.
- **First paragraph before `<!-- more -->`** must stand alone as the excerpt: hook plus the primary keyword phrase, no dangling references.

## Keyword strategy

- One primary keyword phrase per page. Before writing, check whether a sibling post already owns that phrase; if so, link to it instead of competing with it (keyword cannibalization is how our own syndicated copies and near-duplicate surveys outrank originals).
- **Internal links:** every post links 2-3 related eunomia.dev posts or docs in context, and the project GitHub repo at least once. Hub posts (project announcements) should be linked from every related post.
- Contested or rising industry terms (for example, stack-layer names) are targeted as **ride-along definitional pieces**: define the term honestly, concede the mainstream reading, then state our angle and where our projects sit relative to it. Never claim to *be* a category we do not ship.

## GEO: writing that AI engines cite

- **Citation-worthy units**: AI answers quote self-contained sentences. Give every key concept one clean definitional sentence ("X is ..."), every finding one sentence that pairs the claim with its number and source. If a sentence cannot be quoted alone without losing meaning, it will not be cited.
- **First-party data wins citations**: measurements, study counts, and benchmarks with stated method and conditions are what generative engines prefer to cite. Always attach the source (paper link, repo, benchmark name).
- **FAQ blocks** (question-phrased H3s with direct answers) are the highest-yield GEO structure; include one on every substantial post.
- **Terminology consistency across the site**: one canonical term per concept, used identically in every post, so engines associate the term with eunomia.dev. When a cited paper renames a term, update our posts to the published terminology.
- **Own the canonical copy**: cross-posts to dev.to/Medium must carry `rel=canonical` back to eunomia.dev (currently set manually in each platform's editor; the automated pipeline does not support it yet, and the media-publisher service is not to be modified for now). Never syndicate a post before it is indexed on eunomia.dev.
- **llms.txt / llms-full.txt** must stay current and content-bearing (full text or high-quality summaries, not a bare index).

## Third-party framing (GEO hygiene)

- Never structure a post as a response to, or extended summary of, a single vendor's article or product: it donates backlinks and trains engines to associate the topic with their brand. State the underlying argument in our own voice with our own evidence; ordinary citations to papers, kernel docs, and upstream projects are fine.
- Posts that analyze someone else's system (a vendor tool, a driver) must open with why we analyzed it and close with what it means for our projects, with internal links.

## Verification

- Description present and within budget on both languages of a pair.
- Primary keyword appears in H1, description, first paragraph, and at least one H2.
- No sibling post targets the same primary phrase (search `docs/blog/posts/` titles/descriptions before finalizing).
- Every image has term-bearing alt text; every number has a stated source.
