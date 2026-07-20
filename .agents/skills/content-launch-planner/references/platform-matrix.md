# Platform Matrix

Use this reference to decide where a eunomia.dev artifact should go and what
each platform expects.

## Channel Types

- Owned: eunomia.dev, GitHub repos, docs, README, release notes, examples.
  These are the canonical archive and durable technical proof.
- Rented: Zhihu, Juejin, X, LinkedIn, Xiaohongshu, Medium, DEV. These provide
  reach but need platform-native framing.
- Borrowed/community: Reddit, Hacker News, Lobsters, Product Hunt, newsletters,
  other people's threads, and ecosystem lists. These require fit, restraint, and
  active follow-up.

Do not treat "all platforms" as "same short copy everywhere." Short posts need
platform-native framing. Syndicated long-form posts use the exact source title
and substantively unchanged body; platform work is limited to metadata,
rendering checks, and follow-up.

## Default Long-Form Syndication

When a eunomia.dev blog, tutorial, or paper explainer already has a suitable
canonical source, use this default unless the user asks for a rewrite:

Long-form syndication is a distribution default, not a publishing decision. The
artifact must first pass the strategic relevance check: named user pain,
concrete search/community intent, current alternatives, unique public evidence,
and clear brand-pillar fit.

- Medium: publish/syndicate the English version.
- DEV: publish/syndicate the English version; use `canonical_url` when known
  and convenient.
- Zhihu: publish the Chinese version.
- Juejin: publish the Chinese version.
- X: write a short native post or compact thread with one useful observation
  and a share link.
- LinkedIn: write a short professional feed post with one lesson and a share
  link.
- Xiaohongshu: write a visual-first Chinese note or carousel with one concrete
  scenario, screenshot, diagram, checklist, or result. Do not treat it as
  canonical long-form syndication.

For the four long-form surfaces, preserve the selected source title exactly and
keep the body substantively unchanged. Do not alter the opening, section order,
claims, examples, conclusion, or length for platform style, and do not split a
source article into a series by default. Only remove source front matter or a
duplicate H1, repair images/links, convert heading levels, preserve readable
code/tables/formulas, and choose platform metadata such as tags, category, cover,
or description. Visible canonical/source links are optional. If the platform
cannot represent the source faithfully, stop instead of rewriting it in the
editor.

Medium and DEV execution must use their normal web import/editor and visible
submit flows. Do not plan publish-API execution for either platform.

## Default Platform Fit

| Platform | Best for | Default surface | Use when | Skip when | Handoff |
| --- | --- | --- | --- | --- | --- |
| Zhihu | Chinese technical explanation, AI Works, conceptual essays | Canonical Chinese article, answer, idea, AI Works | Needs mechanism, scenario, and reader education | Only a changelog or thin repo link | `zhihu-publisher` |
| Juejin | Chinese developer practice | Chinese article or practical note | Has commands, code, environment, or reproducible details | No practical developer payoff | `juejin-publisher` |
| X | Fast technical observations, threads, build-in-public updates | Short post/thread with share link, reply, quote | One sharp idea, screenshot, benchmark, repo link, or ongoing discussion | Needs long context and cannot fit a clean thread | `x-publisher` |
| LinkedIn | B2B/professional credibility | Short feed post with share link, article/carousel when asked | Engineering leaders, platform/security teams, consulting/research angle | Too niche without professional consequence | `linkedin-publisher` |
| Xiaohongshu | Chinese visual technical entry points | Visual note, carousel, checklist, screenshot/diagram explainer | Has a concrete scenario, visual artifact, beginner-friendly bridge, or saveable checklist | Only a dense paper/blog with no visual proof or short reader payoff | `xiaohongshu-publisher` |
| Reddit | Subreddit-specific discussion | Text post, link, comment | Solves a visible community question or invites technical critique | No subreddit fit or recent drive-by self-promo risk | `reddit-publisher` |
| Hacker News | Technical curiosity | Link, Show HN, Ask HN, comment | Code, paper, demo, measurement, unusual systems behavior | Marketing announcement without substance | `hackernews-publisher` |
| Lobsters | Durable computing discussion | Story with tags, comment | Deep implementation, kernel/runtime/security/compiler work | No tag fit or mostly product news | `lobsters-publisher` |
| Medium | Polished English long-form | English imported/syndicated story | Has a suitable English source and a broader engineering lesson | No suitable English source or better served as docs/tutorial only | `medium-publisher` |
| DEV | Practical developer tutorial | Canonical English article, optional `canonical_url`, series, comment | Developer-relevant article, commands, code, or inspectable artifact | No practical developer payoff | `devto-publisher` |
| Product Hunt | Tryable product/tool launch | Product launch page | Major public tool with clear URL, assets, support capacity, and feedback goal | Paper/blog-only artifact or no demo/product page | Plan only, no publisher yet |

## Platform Mix Patterns

Use a narrow launch when the artifact is niche or fragile:

- Long-form blog/tutorial/paper explainer: default to Medium/DEV English
  syndication, Zhihu/Juejin Chinese syndication, X/LinkedIn short share posts,
  and selective community platforms only when fit is real.
- Paper/research discussion: Zhihu, X thread, LinkedIn, HN/Lobsters only if the
  source is technically surprising.
- Practical tutorial: Juejin, DEV, X, maybe Reddit if a subreddit has a matching
  question; syndicate the complete source rather than splitting it for a
  platform.
- Project/tool launch: GitHub/eunomia.dev first, then X, LinkedIn, Zhihu/Juejin,
  Xiaohongshu when a visual entry point exists, HN/Reddit/Lobsters if community
  fit is real, Product Hunt only for a tryable product moment.
- Follow-up/correction: publish on the original platform first, then update
  canonical docs or GitHub if the issue is durable.

Use a full matrix when the source has:

- a public artifact that readers can inspect or try
- a clear mechanism or lesson that stands alone
- enough evidence for multiple angles
- visual assets or screenshots
- bandwidth for comment follow-up

## Cadence

- Do not dump all derivatives on the same day unless it is a launch-day plan.
- Spread educational derivatives across days so each version can learn from
  earlier comments.
- For community platforms, post only when the author can answer questions.
- After comments reveal confusion, update the canonical source or follow-up post
  before pushing the same angle elsewhere.
