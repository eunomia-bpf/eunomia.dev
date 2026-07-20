# Source And Research

Use this reference to decide whether a piece should be published and what the
platform plan should be grounded in.

## Source Brief

Before planning, extract:

- Artifact: source path, GitHub repo, paper, project, demo, or draft.
- Canonical URL: eunomia.dev, GitHub, paper, docs, or demo page.
- User pain: the concrete problem the target reader already has.
- Search/community intent: what the reader would search, ask, compare, debug,
  or decide.
- Current alternatives: SDK/OTel, MCP proxy, sandbox, Falco/Tetragon, commercial
  security tools, logs, prompt policy, approval workflow, or manual inspection.
- Brand pillar: AI Agent Observability & Harness, eBPF Infrastructure, GPU &
  Systems Research, or a specific paper/tutorial/release.
- Thesis: the one useful idea a reader should understand.
- Evidence: code path, benchmark, screenshot, trace, citation, command, issue,
  paper claim, or implementation detail.
- Target reader: kernel/eBPF developer, AI-agent builder, security engineer,
  SRE/platform team, OSS maintainer, researcher, or engineering leader.
- Reader outcome: learn, reproduce, evaluate, discuss, debug, contribute, or try.
- Promotion role: proof artifact, source link, next step, or not needed.
- Missing assets: screenshots, diagrams, demo video, GitHub README polish,
  install commands, examples, cover image, alt text, or public URL.

If the artifact is a paper explainer, start from `docs/papers/registry.yaml`
and any explicit `Blog pending` markers before inventing topics.

## Repository Checks

Check these before external research when relevant:

- `.github/publisher/media/published.md`
- `.github/publisher/media/not-published.md`
- `.github/publisher/posts_queue.txt`
- `draft/` material related to the artifact
- recent `docs/blog/posts/` and `docs/blogs/`
- matching GitHub repos and README files
- existing platform skill references under `.agents/skills/*-publisher/`

## Duplicate And Discussion Research

Before recommending Reddit, Hacker News, Lobsters, Product Hunt, or repeated
social posting:

- Search the platform for the exact URL, project name, title, and core phrase.
- Search the web for the same title/project plus the platform name.
- Check if an active thread already exists. If yes, prefer a useful comment over
  a new submission.
- Check whether the current artifact is new enough, substantial enough, and
  different enough from earlier posts.
- Record evidence in the plan: searched terms, likely duplicates, related
  discussions, and the chosen route.

For logged-in platform state, use normal visible browser interactions only.
Public web search is fine for duplicate discovery, but do not use private or
hidden platform APIs.

## Publish, Comment, Or Skip

Recommend "publish" when the artifact has a clear reader payoff and platform
fit.

Recommend "comment" when an existing discussion is active or the useful angle is
an answer to a specific question.

Recommend "skip for now" when:

- the source lacks evidence or a public artifact
- the post would be mostly promotion
- platform rules/self-promotion constraints make it risky
- the topic was already posted recently without a new angle
- required assets are missing for a visual/product-first platform
- the content contains private or unverifiable claims

## Link Policy

- Prefer GitHub links when the repo is the artifact.
- Prefer eunomia.dev links when the article/tutorial is the canonical
  explanation.
- Prefer paper links when the argument depends on a primary source.
- Use one primary outbound link in most short-form posts.
- Avoid tracking links for community platforms. Use clean canonical URLs unless
  the user explicitly asks for campaign tracking and the platform allows it.
