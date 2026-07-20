# LinkedIn Platform Preferences

## Source Notes

- LinkedIn's surfaced creator guidance emphasizes originality, thoughtful
  engagement, clear expertise, and writing for a target audience rather than
  chasing virality.
- Community guidance on LinkedIn consistently stresses that the first visible
  lines matter, but treat exact timing, length, and algorithm claims as
  changeable heuristics.

## Style And Positioning

- Write like a technical advisor sharing a useful lesson from real work.
- Default positioning: consulting/research/helping solve hard problems, not
  selling a product.
- For long-form blogs, default to a short feed post with a share link. Do not
  republish the full article as a LinkedIn article unless the user asks.
- When the user requests long-form syndication as a LinkedIn article, preserve
  the source title exactly and keep the body substantively unchanged. Feed-post,
  carousel, and short-copy rewrite guidance does not apply to the article body.
- Keep the 80% contribution / 20% promotion ratio. Use the project as proof,
  implementation, or next step.

## Audience

- Engineering leaders evaluating AI-agent safety, observability, and runtime
  governance.
- Infra, SRE, security, and platform engineers looking for practical patterns.
- Researchers and open-source peers who value credible mechanisms and limits.

## Rewrite Rules

- First lines should name the problem, stake, or lesson without throat-clearing.
- Use a professional story arc: problem seen in practice -> insight -> evidence
  -> project/repo link -> discussion question.
- Avoid turning Chinese-platform phrasing or X-thread fragments into LinkedIn
  without rewriting. LinkedIn needs more context and professional consequence.
- Use 1-3 topics per post. Do not overload one post with every project.
- Use a small number of relevant hashtags only when they help discovery.

## Adaptation Workflow

- Extract the source facts before drafting: field observation, reader role,
  consequence, proof point, project/repo link, and one discussion question.
- When the user asks to improve platform fit from current examples, skim a few
  visible LinkedIn posts from adjacent technical creators or topics using
  normal browser interactions. Record only the 2-3 structural observations that
  will change the draft: first-line hook, proof type, post length, media/link
  treatment, comment prompt, or audience framing. Do not copy distinctive
  wording or use platform APIs.
- Write from first principles for LinkedIn. A good X thread or Zhihu idea should
  be re-expanded into professional context, not pasted with line breaks.
- Treat the first visible lines as the real headline. They should work before
  "see more" and state the professional consequence.
- Prefer evidence-backed hooks: a failed assumption, data point, production
  boundary, research finding, or uncomfortable tradeoff.
- Use outbound links deliberately. If link reach is a concern, prepare a first
  comment with the GitHub/eunomia.dev/paper links instead of crowding the body.
- Run the anti-AI pass: remove "excited to share", "proud to announce",
  "game-changer", "deep dive", "unlock", "leverage", corporate abstractions,
  and generic "agree?" closers.

## Short-Form Style

- A short LinkedIn post should read like a technical advisor's field note, not a
  launch banner.
- First visible lines should state the professional consequence: what changed,
  what broke, what became safer, or what the reader can decide differently.
- Include one concrete proof point: repo, benchmark, command, architecture
  detail, customer-safe lesson, or failed assumption.
- Add human rhythm without fake vulnerability. Use "we expected X; the trace
  showed Y" only when true.
- Avoid AI-tell phrasing: generic praise, perfect three-part lists, "not just X
  but Y", "game changer", "deep dive", "unlock", "leverage", and polished
  slogans that no engineer would say in a design review.
- For comments and replies, add one new noun/concept not already in the parent
  post. Never reply with only "great point", "agree", or a restatement.
- End with a precise question when useful, such as asking where the reader puts
  an enforcement boundary. Do not end with generic engagement bait.
- As a heuristic, 150-300 words is enough for most short posts; go longer only
  when the story/evidence earns the space.
- Use whitespace for scanning: short paragraphs, no wall of text, and bullets
  only when the list is genuinely the artifact.
- Evaluate drafts with a scorecard: hook strength, voice fit, value density,
  structure, and publish readiness.

## Platform-Observation Scan

Use this scan when asked to "刷刷" LinkedIn, benchmark native posts, or improve
weak draft fit:

- Browse visible posts around AI agents, eBPF, observability, platform
  engineering, security engineering, open source, and developer tools. Keep it
  lightweight; it is an input to drafting, not a required ritual.
- Record only reusable patterns, not copy: professional consequence in the
  first two lines, evidence shape, media choice, link placement, paragraph
  rhythm, and whether the discussion question is specific.
- Convert the scan into a short edit checklist before rewriting: sharper first
  line, stronger proof point, simpler body, less promotional phrasing, better
  link placement, or more precise reader role.
- Treat algorithm claims, exact timing rules, and engagement folklore as
  unstable. Prefer observed reader fit and professional clarity.
- Keep follow-up lessons in this reference or the draft archive so future
  LinkedIn posts do not restart from taste-memory alone.

## Recent Browser Observations

2026-07-20 visible LinkedIn search results around AI-agent observability and
eBPF/open-source observability showed a useful split:

- Stronger posts led with a real debugging or adoption pain, then named the
  tool/project as evidence. The best shape was: what was hard before -> what
  the tool correlates or reveals -> current limitation -> repo/article link.
- Link cards and short videos worked as proof when the body already explained
  why the reader should care.
- Weaker posts overused broad enterprise claims, long hashtag walls, and
  project-first launch phrasing. For our posts, keep tags few and make the
  first two lines carry the professional consequence.

## Browser Checks

- Preview the collapsed feed version and confirm the first lines still work.
- Check media crop, document preview, link card, and alt text where available.
- Check the posting identity and visibility setting.
- Check that mentions are intentional and not attention-seeking.

## Post-Publish Follow-Up

- Prioritize thoughtful comments over vanity engagement.
- Reply with direct experience, source links, or concrete next steps.
- Treat messages as potential consulting/research/collaboration leads, but do
  not make commitments without user approval.
- Record useful audience questions as future blog/tutorial ideas.
