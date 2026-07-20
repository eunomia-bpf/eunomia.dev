# X Platform Preferences

## Source Notes

- X Business recommends concise copy, conversational tone, clear CTA where
  applicable, no all-caps, and avoiding hashtags in post copy.
- X also recommends media that stands out without heavy text, short videos, and
  active community management through mentions, reusable responses, and context
  from past conversations.

## Style And Positioning

- Be sharp, concrete, and useful. X should surface one technical observation,
  result, question, or artifact at a time.
- For long-form blogs, default to a short native post or compact thread with a
  share link. Do not republish the whole article as an X Article unless the user
  asks.
- Keep the 80% contribution / 20% promotion ratio. The post should be valuable
  even if the reader never clicks the link.
- Position the maintainer as a research-minded systems engineer sharing hard
  lessons from eBPF, AI-agent runtime safety, observability, and open-source
  implementation work.

## Audience

- AI-agent/runtime builders who need security and observability patterns.
- eBPF/kernel/runtime developers who like concrete implementation details.
- OSS maintainers, infra engineers, and researchers scanning for useful links.

## Rewrite Rules

- Lead with the takeaway, not the project name.
- Use one concrete number, trace, repo artifact, or mechanism when available.
- For threads, make each post carry one step: problem, mechanism, example,
  limitation, link.
- Prefer GitHub links for project artifacts and eunomia.dev links for full
  explanations. Use one primary link per post unless a thread needs sources.
- Avoid broad hype terms without mechanism, such as "agent security solved".

## Adaptation Workflow

- Extract the source facts before drafting: thesis, proof point, artifact link,
  intended reader, and one useful takeaway that can stand alone on X.
- Choose the container first: standalone post, short thread, reply, quote, or
  launch note. Do not stretch one compact idea into a thread.
- Use a hook pattern only when the evidence supports it: contrarian take,
  data-point observation, build-in-public note, mini-list, curiosity gap, or
  "how this broke/changed" teardown.
- Keep links out of the opening line. For threads, put the GitHub or article
  link where it helps the argument, often near the end.
- Run the anti-AI pass before browser paste: remove "excited to share",
  "game-changer", "deep dive", "unlock", "leverage", generic transitions,
  padded three-part lists, and "thoughts?"-style bait.

## Short-Form Style

- Treat every short post as one idea, one artifact, one reader benefit.
- Use the first line for the strongest concrete claim or observation. Do not
  spend it on "we just shipped", "big news", or category labels.
- Add one human fingerprint: a real constraint, failed assumption, measured
  result, code path, command, screenshot, trace, or named project detail.
- Avoid AI-tell phrasing: generic hype, perfect three-part lists, "not just X
  but Y", "game changer", "deep dive", "unlock", "leverage", and vague
  adjectives without evidence.
- Vary line length. A good short post can have one compact sentence, one
  explanatory sentence, and one source link.
- End with a specific technical question only when discussion would help. Avoid
  generic engagement bait such as "thoughts?"
- For threads, do not make every item a mini CTA. Build a chain of reasoning
  and put the link where it naturally helps.
- Count every post against the 280-character limit before paste. Shorter is
  fine when the idea lands cleanly.
- Replies should add a new technical detail, caveat, source, or framing. Never
  answer with canned praise, and do not hard-sell a project in a reply.
- Quote posts should make the original more useful: add context, connect it to
  a concrete artifact, or disagree precisely without dunking.

## Browser Checks

- Check character counts and thread ordering in the composer.
- Check link card destination, media crop, upload completion, and alt text.
- Check the account/profile context if multiple sessions could be logged in.
- Check notifications/search before follow-up so replies use context, not canned
  responses.

## Post-Publish Follow-Up

- Watch replies, quotes, and mentions for technical corrections, reproduction
  attempts, security concerns, and collaboration interest.
- Reply with sources, commands, repo links, or concise reasoning.
- Move substantial implementation discussions to GitHub issues/discussions when
  they need durable tracking.
- Treat DMs as private and user-controlled; summarize or draft only after the
  user explicitly asks.
