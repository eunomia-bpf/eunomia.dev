# Medium draft

Platform: Medium

Decision: published via Medium import on 2026-07-19.

Published URL: https://medium.com/@yunwei356/an-empirical-study-ai-agent-rules-need-context-and-layered-enforcement-eunomia-423adab48a1b

Source body: `docs/blog/posts/ebpf-ai-agent-policy-enforcement.md`

Title: An Empirical Study: AI Agent Rules Need Context and Layered Enforcement

Subtitle: AI agent rules look simple in CLAUDE.md, but ActPlane's 2,116-statement study shows why context and layered OS enforcement decide what can be checked.

Canonical URL: https://eunomia.dev/blog/2026/07/15/ebpf-ai-agent-policy-enforcement/

GitHub: https://github.com/eunomia-bpf/ActPlane

Paper: https://arxiv.org/abs/2606.25189

Topics: Software Engineering, Artificial Intelligence, Cybersecurity, Open Source, Programming

Body policy:

- Preserve the English canonical article body.
- Remove YAML front matter.
- Keep the article H1 as the Medium title, not inside the body.
- Keep GitHub and paper links as evidence/source links.
- Medium import added the source note and `rel=canonical` points to the eunomia.dev URL.

Browser QA:

- Published from a fresh import draft, not the earlier corrupted title-edit draft.
- Verified published story visible in the in-app browser.
- Verified Medium platform URL and canonical URL separately: Medium `og:url` is the published URL, while `rel=canonical` points to eunomia.dev.
- Post-publish fix: removed the imported `| eunomia` source-site suffix from the
  Medium title through the Medium web editor and saved the published story
  again.
- Post-publish public-page QA verified the cleaned title, canonical URL,
  ActPlane paper link, GitHub links, code block, headings, and article images.
- Known Medium import limitation: Markdown tables were flattened into readable
  prose-like rows on the public page. This was recorded in the Medium publisher
  skill as a recurring import issue; future Medium imports should convert tables
  to readable list/prose fallbacks before publishing when Medium cannot preserve
  table structure.
