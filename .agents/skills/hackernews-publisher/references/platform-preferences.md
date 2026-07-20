# Hacker News Platform Preferences

## Source Notes

- HN guidelines say on-topic submissions are things that gratify intellectual
  curiosity for good hackers.
- HN asks users to submit original sources, avoid promotional primary use,
  avoid editorialized titles, and avoid vote/comment solicitation.
- HN guidelines also warn against generated or AI-edited comments. Treat any
  comment draft as material for the user to personally review and own.

## Style And Positioning

- Curiosity first. HN is not a product-distribution channel.
- Use the 80% contribution / 20% promotion ratio, and submit only the subset of
  work that is genuinely interesting to technical readers.
- Prefer artifacts with technical depth: source code, measurements, papers,
  demos, implementation notes, or surprising systems behavior.

## Audience

- Technically curious programmers, founders, researchers, infrastructure
  engineers, and open-source maintainers.
- They dislike hype, linkbait, and obvious launch marketing.

## Rewrite Rules

- Preserve an existing long-form source title exactly except for a
  platform-required marker such as `[pdf]`. Do not rewrite it into a hook.
- Remove site names, gratuitous numbers/adjectives, all-caps, exclamation
  points, and praise from the title.
- For Show HN, title the tryable artifact directly: "Show HN: <thing>".
- For regular submissions, use the canonical URL and let the linked page do the
  explaining.

## Submission Decision Workflow

- Search HN/Algolia and the web for duplicate URLs, prior project submissions,
  paper titles, and near-identical discussions before recommending a post.
- Classify the route: regular link, Show HN, Ask HN, comment in an active
  thread, or no post because it is too promotional or already discussed.
- Only use HN when the artifact gratifies technical curiosity by itself: code,
  measurement, paper, demo, implementation writeup, or surprising system
  behavior.
- Keep one canonical URL. Do not route through campaign pages, tracking links,
  or link shorteners.
- Run the anti-AI pass on any comment draft: remove formulaic concessions,
  generic praise, synthetic humility, future-of-AI abstraction, and engagement
  bait.

## Short-Form Style

- HN short-form work is mostly title selection and comment restraint.
- For original submissions, titles should be plain and technical. Existing
  long-form source titles stay exact; do not add marketing hooks, emojis, hype
  adjectives, or "why everyone should care".
- For Show HN text, state what it is, who it is for, and what is technically
  interesting in a few factual sentences.
- For comments, do not post AI-polished replies verbatim. Draft only for user
  review, then keep the final reply human-owned, concise, and technical.
- Avoid AI-tell phrasing: generic praise, formulaic concessions, broad "future
  of AI" claims, synthetic humility, and engagement bait questions.
- Good HN follow-up answers a real question, names a limitation, links to code,
  or corrects a misunderstanding.
- Show HN text should be a short factual note: what it is, who can try it, what
  is technically unusual, and where the code lives.
- If the best contribution is a comment, make it shorter than the draft feels.
  HN rewards precision and lived technical detail over completeness.

## Browser Checks

- Search HN/Algolia for duplicates and recent same-topic posts.
- Confirm the URL is the original source.
- Check title normalization in the submit form.
- Check the logged-in account if multiple sessions are possible.

## Post-Publish Follow-Up

- Read the full thread before drafting replies.
- Keep replies short, technical, and personally accountable.
- Correct factual errors and disclose affiliation when relevant.
- Move implementation bugs or feature discussions to GitHub links when the
  thread needs durable tracking.
