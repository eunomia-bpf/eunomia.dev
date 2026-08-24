# Reddit Platform Preferences

## Source Notes

- Reddit's Reddiquette says to use factual, opinion-free titles, submit original
  sources, post to the most appropriate community, search for duplicates, use
  canonical URLs, and keep self-promotion within reason.
- Reddiquette mentions a widely used 9:1 rule of thumb: most activity should be
  genuine participation rather than links to your own content.
- Subreddit rules override generic Reddit advice.

## Style And Positioning

- Community first, project second. Reddit punishes write-only promotion.
- Use the 80% contribution / 20% promotion ratio as a minimum; many subreddits
  need even less promotion.
- Be transparent: say when the maintainer authored the project, paper, or post.

## Audience

- Varies by subreddit. Identify the local reader before drafting.
- For systems topics, likely readers include kernel developers, security
  engineers, SRE/platform teams, AI-agent builders, and OSS practitioners.

## Rewrite Rules

- For a link submission to an existing long-form source, preserve the source
  title exactly unless a subreddit rule requires a prefix or flair. Native
  titles may be written for original text posts or when the user explicitly
  requests one; do not rewrite for SEO.
- Prefer a question, technical lesson, benchmark, repo artifact, or concrete
  debugging story over a launch announcement.
- Include a short context paragraph when submitting a link, but do not paste a
  marketing abstract.
- If an existing thread is active, draft a comment that answers the thread
  rather than starting a new promotional submission.

## Community Research Workflow

- Before drafting, search the target subreddit and Reddit-wide for the URL,
  project name, paper title, and core technical phrase.
- Classify the best route: new link submission, text post with source link,
  comment in an existing thread, or no post because the fit is weak.
- Extract subreddit-local evidence: recent questions, rule wording, flair,
  recurring pain points, accepted link types, and how maintainers disclose
  affiliation.
- Make the post useful without the click. The GitHub/eunomia.dev link should
  extend the answer, not hold the answer hostage.
- Run the anti-AI pass: remove generic praise, "I thought I'd share", perfect
  benefit stacks, broad hype, and community-agnostic phrasing.

## Short-Form Style

- Reddit short posts and comments must sound like a person participating in a
  specific community, not a generic AI reply.
- Use AI-assisted drafts as structure only. Rewrite in the maintainer's voice
  and reference details from the actual thread before posting.
- Lead with the answer, caveat, or reproducible detail. Save project mention for
  the sentence where it helps.
- Include one subreddit-local detail when possible: the exact question, quoted
  error class, tool name, distro/kernel/runtime context, or tradeoff discussed
  in the thread.
- Avoid AI-tell phrasing: canned praise, "as an AI", obvious marketing copy,
  perfect three-part benefit lists, broad claims, and generic "hope this helps".
- If the post is self-promotional, disclose affiliation plainly and make the
  useful explanation complete without forcing a click.
- Titles work best as factual observations, narrow questions, or concrete
  artifacts. Avoid launch language and identical cross-post titles.
- In comments, answer the question first, then mention the repo only if it adds
  a reproducible path, issue, command, or source.

## Browser Checks

- Read the sidebar, rules, wiki, pinned posts, and flair guidance.
- Search the subreddit and Reddit-wide for the same URL/topic.
- Check account context and recent self-promotion balance if visible.
- Preview Markdown because titles cannot be edited after submission.

## Post-Publish Follow-Up

- Stay available for questions and criticism.
- Acknowledge limitations and corrections quickly.
- Move reproducible bugs, feature requests, or deep technical reports to GitHub.
- Do not argue with moderation decisions in-thread; use modmail only when the
  user asks.
