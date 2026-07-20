# Launch Assets

Use this reference when planning project, release, demo, paper, or tutorial
launches.

## Launch Tiers

Tier 1: major public artifact

- Examples: ActPlane/AgentSight public launch, substantial demo, major project
  repositioning, new capability with clear audience change.
- Needs: GitHub/README polish, canonical page or article, screenshots,
  architecture diagram, demo media, short social copy, community plan, follow-up
  owner, and metrics/retro.

Tier 2: significant technical content or feature

- Examples: paper explainer, deep tutorial, major benchmark, new integration,
  meaningful release.
- Needs: canonical post, GitHub/source links, one or two visuals, X/LinkedIn or
  Zhihu/Juejin variants, selective community submissions.

Tier 3: small update or maintenance signal

- Examples: docs update, minor release, small bug fix, queue follow-up.
- Needs: changelog or short post only if it teaches something.

## Asset Inventory

For each launch, check:

- Canonical article/page URL
- GitHub repository and specific paths
- README summary and install/use instructions
- Screenshot or terminal trace
- Architecture diagram or flow diagram
- Demo video/GIF if it is a visual product or Product Hunt candidate
- Benchmark or measurement table with methodology
- Paper/source citation when claims depend on research
- FAQ or expected objections
- Media alt text
- Follow-up destination: GitHub issue, discussion, docs page, or email/DM draft

If an asset is missing, mark it as a blocker, nice-to-have, or not needed.

## Content Atomization

Turn one source into native pieces:

- Pillar: canonical blog/tutorial/project page/README.
- Long-form syndication: Medium/DEV English, Zhihu/Juejin Chinese. Preserve the
  canonical body by default; only adjust metadata, rendering, links, tags, and
  a short source/project note when it helps the reader.
- Short derivatives: LinkedIn post and X post/thread with one useful insight
  plus a share link.
- Micro-content: single X posts, LinkedIn comments, Zhihu ideas, screenshots,
  charts, HN/Lobsters titles, Reddit comments.

Store platform-specific drafts before browser/editor work under
`draft/media/YYYY-MM-DD/<source-slug>/`. Use one file per platform, such as
`medium.md`, `devto.md`, `zhihu.md`, `juejin.md`, `x.md`, `linkedin.md`,
`reddit.md`, `hackernews.md`, or `lobsters.md`. For unchanged long-form
syndication, the file may point to the canonical source body instead of copying
it, but it still records the platform fields, links, tags/categories, source
or project note if useful, media, and QA state. Short posts, comments, and
replies include the full paste-ready copy.

Extract atoms before writing:

- one mechanism
- one surprising result
- one failure mode
- one concrete use case
- one diagram/screenshot
- one reproducible command/path
- one limitation or caveat

Short and community derivatives should add a platform-native angle, not merely
shorten the pillar. Long-form syndication should keep the canonical article
stable unless a real reader or formatting issue requires edits.

## Technical Project Framing

For eBPF/AI-agent safety projects, avoid category-only phrasing. Explain:

- what is observed
- what is enforced
- where the boundary sits, such as runtime, sandbox, kernel, or eBPF hook
- what failure it prevents
- why it improves safety, compliance, reliability, or instruction following
- what a developer can inspect in GitHub

Use project names after the reader understands the problem.

## Post-Launch Feedback Loop

Within 1-7 days after a meaningful launch:

- collect comments, questions, corrections, and reposts
- identify the top 3 confusion points
- convert reproducible issues to GitHub issues or docs tasks
- update the canonical article/README if needed
- record best-performing angles for the next plan
- decide whether a follow-up post is educational, corrective, or unnecessary
