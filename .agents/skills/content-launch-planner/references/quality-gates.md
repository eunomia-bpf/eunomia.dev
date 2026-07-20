# Quality Gates

Use this reference before finalizing any content launch plan.

## Strategic Relevance Gate

Pass this gate before platform, syndication, or SEO/GEO details.

- The target reader and user pain are named.
- The search or community intent is concrete: what the reader would search,
  ask, compare, debug, or decide.
- Current alternatives are identified, such as SDK/OTel tracing, MCP proxy,
  sandbox, Falco/Tetragon, commercial security tooling, logs, prompt policy, or
  approval workflow.
- The artifact has unique public evidence: GitHub repo, paper, benchmark,
  trace, demo, screenshot, issue, reproducible command, or limitation.
- The launch strengthens a clear brand pillar: AI Agent Observability &
  Harness, eBPF Infrastructure, GPU & Systems Research, or a specific
  paper/tutorial/release.
- The next step is useful and proportionate: read, try, reproduce, compare,
  comment, open an issue, or discuss a scoped technical problem.

Skip, delay, or turn the item into a small note when it has only a platform
slot, only a keyword, or only a project announcement without a real reader
problem.

## Contribution-First Gate

Pass only if:

- The reader receives useful knowledge without clicking.
- Project promotion is evidence, implementation, source, or next step.
- The plan stays close to 80% contribution and 20% promotion.
- The maintainer posture is consulting/research/helping solve problems, not
  product sales.

## Evidence Gate

Every platform angle needs at least one concrete support:

- GitHub repo or file path
- code snippet or command
- benchmark or trace
- paper/source citation
- screenshot or diagram
- real limitation or tradeoff
- observed discussion/question from the target community

Do not invent numbers, customers, quotes, rankings, or adoption claims.

## Platform-Native Gate

Pass only if each platform has:

- a distinct reader and surface
- an opening that fits the platform
- a link placement plan
- media/asset expectations
- follow-up expectations
- a reason to publish or a reason to skip

For long-form canonical syndication, "native" means platform metadata,
rendering, link placement, tags/categories, and follow-up expectations. Do not
force a full article rewrite when the canonical body already works.

Reject unreviewed paste, context-free link drops, and duplicate posts without a
clear reader purpose and publishing ledger entry.

## Anti-AI-Tell Gate

Remove:

- "excited to share", "proud to announce", "game-changer", "deep dive",
  "unlock", "leverage", "transformative", "revolutionary"
- generic three-benefit stacks without mechanisms
- "not just X but Y" as a structural crutch
- vague claims such as "secure, compliant, reliable" without naming what is
  enforced and where
- generic engagement bait: "thoughts?", "agree?", "let me know in the comments"
- canned praise in replies
- press-release rhythm and corporate abstractions

Preserve technical accuracy, named projects, source links, and real caveats.

## Public Boundary Gate

Remove or block:

- private strategy
- customer names or customer conversations
- fundraising/incubator details
- pricing or sales commitments
- unreleased roadmap
- claims that require private evidence
- secrets, tokens, private paths, or private screenshots

This public repo may contain site operations and publishing guidance, but not
private commercial strategy.

## Local Artifact Gate

For long-form posts on every platform, pass this before opening or changing the
platform editor:

- A platform-specific artifact exists in a temporary directory or
  `draft/media/YYYY-MM-DD/<source-slug>/`.
- The artifact has the exact source title, no duplicate body H1, checked image
  URLs or upload assets, table/formula/code fallbacks, links, tags/categories,
  and the intended source/project note when useful.
- The platform editor is used for import/upload, metadata/settings, preview, and
  QA. Do not rely on the platform editor for writing, large rewrites, link-heavy
  tail-note edits, or structural repairs.
- If a substantial body fix is needed after import, regenerate the local
  artifact and re-import into a fresh draft when practical.

## Syndication Hygiene Gate

For Medium/DEV/Zhihu/Juejin long-form posts, check these after the Strategic
Relevance, Contribution-First, Evidence, and Platform-Native gates pass:

- The canonical body is preserved unless the user asked for a rewrite or the
  source was corrected first.
- Medium and DEV use the English source; Zhihu and Juejin use the Chinese
  source when available.
- The source title is preserved exactly. A title change requires an explicit
  user request for that specific publication.
- The opening, section order, claims, examples, conclusion, and length are not
  changed for platform style. Do not shorten, expand, localize, or split the
  source into a series during syndication.
- GitHub, docs, paper/arXiv, and project links are present when relevant.
- Visible canonical/source links are optional on every platform. Dedicated
  canonical fields, when supported, are hygiene settings rather than growth
  strategy; do not edit the article body just to add a canonical/source link.
- Images, code blocks, headings, tables, tags/categories, and previews are
  checked in the target editor.
- Body text or links are added only after the same source change or an explicit
  user request.

This is a publishing QA gate, not a reason to publish by itself.

## Rendered Browser QA Gate

For every platform post that is drafted or published through a logged-in web UI:

- Use visible browser interaction in the target web editor. Medium and DEV must
  use the web import/editor and submit flow, not publish APIs or background
  endpoints.
- Before final publishing, inspect the editor preview or imported story from top
  to bottom, not only the first viewport.
- After publishing, open the public URL and inspect the rendered page or post
  from top to bottom before marking it confirmed.
- Verify title/subtitle, heading levels, image loading, captions/alt text where
  supported, tables or table fallbacks, code blocks, link targets, canonical
  fields when configured, source/project notes when present, tags/categories,
  and mobile/narrow rendering when practical.
- If the public page reveals broken images, flattened tables, empty headings,
  duplicated notes, wrong tags, or mangled code blocks, edit the published item
  in the web UI and re-run the public-page check.
- Record the preview/public-page QA state, fixes, blockers, and final URL in
  the matching `draft/media/YYYY-MM-DD/<source-slug>/<platform>.md` file and
  publishing ledger.

## Publishing Lessons Gate

At the end of every platform publishing session:

- Name any concrete issue found during drafting, preview, publish, public-page
  QA, or post-publish editing.
- Update the matching publisher skill or reference file when the issue could
  recur, including platform-specific checks or fallback behavior.
- If the issue is one-off or account-specific, record it in the matching
  `draft/media/YYYY-MM-DD/<source-slug>/<platform>.md` QA notes instead of
  turning it into generic guidance.
- Do not mark the launch workflow complete until either the skill/reference was
  updated or the reason for not updating it is recorded.

## Browser And Scheduled Execution

- Platform checks must use visible browser interaction when logged-in state is
  involved.
- Do not use hidden platform APIs, background endpoints, or autopost tools.
- A dated daily task executed by `eunomia-content-patrol` is direct
  authorization for every final platform action needed to complete that task.
  Finish it without secondary confirmation or extra field-completeness gates.
- Resolve routine details from the plan, artifacts, ledgers, publisher
  conventions, and visible account state. When the task says to publish, a
  draft or preview is not completion; confirm the public result.
- Outside a scheduled daily task, platform actions still require explicit user
  instruction.

## Scorecard

Score each platform 0-2:

- Fit: target readers are actually there.
- Value: post stands alone without click.
- Evidence: concrete artifact/source exists.
- Native style: platform conventions are respected.
- Risk: self-promotion, duplicate, confidentiality, or rule risk is controlled.
- Follow-up: owner and response path are clear.

Default recommendation:

- 10-12: publish
- 7-9: publish after fixes
- 4-6: comment or narrow angle
- 0-3: skip
