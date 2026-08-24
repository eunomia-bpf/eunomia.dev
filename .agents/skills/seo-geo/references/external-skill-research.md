# External Skill Research Notes

Use this reference when evolving eunomia.dev publishing skills from external
agent-skill repositories. Keep this file as patterns and evaluation notes, not
copied third-party instructions.

## Repositories Sampled

- `anthropics/skills`: official example repository showing the standard
  `SKILL.md` plus optional `scripts/`, `references/`, and `assets/` layout.
- `VoltAgent/awesome-agent-skills`: high-star curated index that explicitly
  favors hand-picked, real-world skills over bulk-generated material.
- `alirezarezvani/claude-skills`, `sickn33/agentic-awesome-skills`,
  `composio-community/awesome-codex-skills`, `github/awesome-copilot`,
  `JimLiu/baoyu-skills`, and `Dimillian/Skills`: sampled for workflow,
  publishing, DevRel, social, and content-strategy patterns.
- `megastep/codex-skills`: sampled `blog-repurpose` for source extraction,
  cross-platform adaptation, and platform-native rewrite workflows.
- `blacktwist/social-media-skills`: sampled context, post writer, thread
  writer, hook writer, platform strategy, content repurposer, and content
  strategy skills.
- `aaaronmiller/create-viral-content`: sampled anti-AI-tell, humanize,
  platform-template, and cross-platform adaptation references. Use as a quality
  gate, not as hype guidance.
- `sergebulaev/x-skills`: sampled X-specific voice rules, hook formulas, post
  writer, thread builder, reply drafter, repurposer, and humanizer patterns.
- `jpeggdev/humanize-writing`: sampled multi-pass anti-AI editing rules for
  structure, vocabulary, rhythm, hedging, and promotional language.
- `skainguyen1412/social-media-research-skill`: sampled Reddit/X research,
  discovery, ranking, and evidence-grounded trend analysis workflows.
- `stevenflanagan1/social-ai-team`: sampled social manager, content calendar,
  platform-specific LinkedIn/X writers, manual approval gates, publisher
  handoff, and performance review feedback loops.
- `coreyhaines31/marketingskills`, `sales-skills/sales`, and
  `charlie947/social-media-skills`: sampled for marketing/social process
  patterns, but only contribution-first and evidence-backed pieces fit the
  eunomia.dev posture.
- `amplitude/builder-skills`: sampled launch strategy, distribution, launch
  tweet, launch blog post, and metrics skills for tiering, channel plans, and
  post-launch retros.
- `SpillwaveSolutions/running-marketing-campaigns-agent-skill` and
  `yabasha/copywrite-skill`: sampled campaign decision trees, content calendar
  fields, UTM/measurement conventions, and launch/post-launch phases.
- `borghei/Claude-Skills` and `robertbstillwell/marketing-skills`: sampled
  owned/rented/borrowed channel models, launch readiness, PMM checklists,
  content repurposing, and social-content orchestration.
- `yoanbernabeu/producthunt-skills`: sampled Product Hunt strategy, safe
  messaging, maker comment, launch-day checklist, analytics, and follow-up.
- Attempted `openclaw/skills`, but the repository was not available when
  cloning.

## Patterns Worth Absorbing

- Keep `SKILL.md` as the routing/workflow surface; move detailed platform taste,
  examples, and checklists into `references/`.
- Keep broad strategy, positioning, calendar design, and channel-mix decisions
  outside skills. Put them in `draft/` so they can evolve without making skills
  over-trigger.
- Separate research, drafting, optimization, browser QA, and follow-up stages.
- Require an audience and intent decision before writing.
- Treat platform adaptation as rewriting, not cross-post copy-paste.
- Add quality gates that check evidence, source links, visuals, formatting, and
  whether the reader gets value before promotion.
- Label community platform heuristics as heuristics. Official docs, visible
  browser state, and current campaign rules outrank old growth advice.
- For social platforms, include post-publish community management: comments,
  mentions, replies, messages, and escalation to GitHub issues/docs.
- Absorb anti-AI-tell skills as quality gates, not as growth hacks. Useful
  pieces: remove generic filler, avoid formulaic openings, require one concrete
  artifact/detail, vary sentence rhythm, avoid canned praise, and require
  human/user approval before visible publishing.
- For short-form platform work, every publisher skill should include a
  platform-specific style section covering posts, comments, titles, replies, or
  excerpts. Short-form writing is not just shortened long-form writing.
- Before platform drafting, extract a small source brief: thesis, intended
  reader, proof point, artifact link, reader outcome, and one useful takeaway.
- Pick the platform container before writing: article, answer, idea, thread,
  standalone post, reply, quote, link submission, Show HN, comment, tutorial,
  series entry, excerpt, or first comment.
- Treat first comments, pinned replies, and follow-up comments as separate
  surfaces with their own purpose: source links, extra context, corrections, or
  reproducibility paths.
- Keep one primary outbound link in most short-form posts. Use GitHub links for
  artifacts, eunomia.dev links for full explanations, and paper links for source
  evidence.
- Search before recommending a community submission: duplicates, active
  discussions, local rules, self-promotion constraints, and whether the current
  artifact belongs on that platform.
- Use performance review as feedback when data exists: best-performing posts,
  recurring comments, saves/bookmarks, technical questions, and conversion to
  GitHub issues/docs should change the next content plan.
- For the eunomia.dev matrix, keep promotional content below 20% and make even
  project launches useful without the click.
- Add a separate upstream planning skill for cross-platform launches. It should
  decide publish/comment/skip, run duplicate and community-fit checks, classify
  launch tier, prepare platform-specific briefs, and hand execution to the
  matching publisher skill.
- Include Product Hunt only as an optional product/tool launch channel. Official
  Product Hunt guidance outranks third-party playbooks: do not ask for upvotes,
  use clean product URLs, prepare required assets, and plan comment follow-up.

## Platform-Specific Style Patterns Absorbed

- X: one idea per post, hard 280-character check, hook supported by evidence,
  no link in the opening line, no padding a single idea into a thread, replies
  and quote posts must add a concrete technical detail.
- LinkedIn: first visible lines carry the promise, professional field-note
  voice, specific proof point, short paragraphs, no "excited to share", and
  optional first-comment link placement.
- Reddit: subreddit-local title and context, full value in the post/comment,
  transparent affiliation, duplicate search, and no identical cross-post titles.
- Hacker News: plain titles close to the source, one canonical URL, Show HN
  only for tryable artifacts, and comment restraint.
- Lobsters: durable computing value, strict tag fit, author disclosure when
  relevant, and no product announcement without technical substance.
- Zhihu: question/scenario first, mechanism before project, GitHub as artifact
  evidence, enough context for "想法", and avoid slogan-only Chinese copy.
- Juejin and DEV: practical developer payoff, runnable or inspectable artifact,
  environment/version/code-path detail, specific tags, and canonical/source
  links.
- Medium: narrative technical essay, title/subtitle/excerpt as separate
  surfaces, canonical relationship, and no generic AI-infrastructure packaging.

## Patterns To Avoid

- Do not install or copy large community skill packs wholesale.
- Do not import platform automation that bypasses normal browser UI boundaries.
- Do not encode brittle algorithm claims as fixed rules.
- Do not let marketing skills override the maintainer's research/consulting
  positioning.
