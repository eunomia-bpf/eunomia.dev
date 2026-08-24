# Output Template

Use this template for launch planning. Keep the final user-facing plan concise,
but preserve the fields needed for execution.

## Standard Plan

```markdown
# Content Launch Plan: [Artifact]

## Verdict

- Launch tier: [Tier 1 / Tier 2 / Tier 3]
- Recommendation: [full matrix / narrow launch / comment only / skip for now]
- Reason: [1-3 sentences]

## Source Brief

- Artifact: [path/repo/paper/demo]
- Source/canonical link: [URL if useful]
- GitHub link: [URL/path]
- User pain: [what concrete problem the reader has]
- Search/community intent: [what they search, ask, compare, debug, or decide]
- Current alternatives: [what they use now]
- Brand pillar: [AI Agent Observability & Harness / eBPF Infrastructure / GPU & Systems Research / specific paper/tutorial/release]
- Reader promise: [what the reader learns or can do]
- Evidence: [code, trace, benchmark, screenshot, paper, issue]
- Missing assets: [blockers/nice-to-haves]

## Research Check

- Already published: [yes/no/unknown + evidence]
- Duplicate/community scan: [queries + result]
- Active discussions to join: [links or none]
- Rule risks: [platform-specific risks]

## Draft Archive

- Draft root: `draft/media/YYYY-MM-DD/[source-slug]/`
- Draft files: `[platform].md` for each publish/comment/share decision
- Long-form title/body policy: [exact source title; substantively unchanged
  source path / full copy included]
- Browser QA state: [not started / draft created / editor preview checked /
  published page checked / fixes applied / blocked]
- Submit path: [web editor/import UI only / no publish API]
- Skill lessons: [updated relevant publisher skill / not needed because...]

## Platform Matrix

| Platform | Decision | Surface | Angle | Primary link | Asset | Follow-up |
| --- | --- | --- | --- | --- | --- | --- |
| Zhihu | publish/skip/comment | canonical Chinese article/answer/idea/AI Works | | | | |
| Juejin | publish/skip/comment | canonical Chinese article/series/note | | | | |
| X | publish/skip/comment | long-form link plus 1-2 sentence hook; thread/reply/quote only when asked | | | | |
| LinkedIn | publish/skip/comment | long-form link plus 1-2 sentence hook; article/carousel only when asked | | | | |
| Reddit | publish/skip/comment | subreddit/comment | | | | |
| HN | publish/skip/comment | link/Show HN/Ask HN/comment | | | | |
| Lobsters | publish/skip/comment | story/comment | | | | |
| Medium | publish/skip | canonical English import/syndicated story | | | | |
| DEV | publish/skip | canonical English article with optional canonical_url/series/comment | | | | |
| Product Hunt | include/maybe/skip | launch page | | | | |

## Per-Platform Briefs

### [Platform]

- Target reader:
- Surface:
- Opening/hook:
- Core contribution:
- Project mention:
- Link placement:
- Media:
- CTA or discussion prompt:
- Browser checks:
- Handoff skill:
```

## If The User Says "Select First" Or "Do Not Touch Yet"

Only output:

- recommended artifact
- why it is worth publishing now
- platform shortlist
- risks and missing assets
- exact next command/request to proceed

Do not draft full posts, open platform composers, or edit ledgers.

## If The User Asks For All-Platform Publishing

Still include skip decisions. "All-platform" means every relevant platform is
evaluated, not every platform must receive a post.

For each publish decision, produce a unique angle. Examples:

- Medium/DEV: exact English source title and substantively unchanged body
- Zhihu/Juejin: exact Chinese source title and substantively unchanged body
- X: published long-form link plus a one- or two-sentence core hook
- LinkedIn: published long-form link plus a one- or two-sentence core hook
- HN/Lobsters: plain technical artifact
- Reddit: subreddit-specific answer or discussion
- Product Hunt: tryable product listing, only if fit passes

## Follow-Up Template

```markdown
## Follow-Up

- First 2 hours: [what to monitor]
- First 24 hours: [comments/questions/corrections]
- Week 1: [docs/GitHub issues/follow-up post]
- Retro notes: [best angle, worst angle, repeated questions, next action]
```
