# DEV draft

Platform: DEV Community

Decision: published on 2026-07-19 after editor preview QA.

Published URL: https://dev.to/yunwei37/an-empirical-study-ai-agent-rules-need-context-and-layered-enforcement-43on

Source body: `docs/blog/posts/ebpf-ai-agent-policy-enforcement.md`

```yaml
---
title: An Empirical Study: AI Agent Rules Need Context and Layered Enforcement
published: true
description: AI agent rules look simple in CLAUDE.md, but ActPlane's 2,116-statement study shows why context and layered OS enforcement decide what can be checked.
tags: security, ai, ebpf, opensource
canonical_url: https://eunomia.dev/blog/2026/07/15/ebpf-ai-agent-policy-enforcement/
---
```

Links:

- GitHub: https://github.com/eunomia-bpf/ActPlane
- Paper: https://arxiv.org/abs/2606.25189

Body policy:

- Preserve the English canonical article body.
- Remove YAML front matter and H1 from the body because DEV uses the title field as H1.
- Convert relative images under `imgs/` to checked public absolute URLs. The
  guessed eunomia.dev article-relative image path returned 404, so the published
  DEV article now uses stable GitHub raw URLs under
  `https://raw.githubusercontent.com/eunomia-bpf/eunomia.dev/main/docs/blog/posts/imgs/...`.
- Do not add a manual canonical/source note at the end when DEV's
  `canonical_url` notice is visible.

Browser QA:

- Checked `canonical_url`; published page shows "Originally published at eunomia.dev".
- Checked H1 title plus H2/H3 body hierarchy in preview.
- Checked tables, code blocks, first images, GitHub links, and arXiv links in preview.
- Published after explicit user authorization for real publishing.
- Post-publish fix: changed tags from `security`, `ai`, `linux`,
  `opensource` to `opensource`, `ai`, `security`, `ebpf` through the DEV web
  editor after confirming `ebpf` was accepted as a selected tag chip.
- Post-publish fix: removed the manual trailing "Originally published at ..."
  note because DEV already displays its canonical-url notice.
- Post-publish public-page QA verified the cleaned tags, DEV canonical notice,
  no manual source-note tail, article images, tables, code block, GitHub links,
  and arXiv links.
- Post-publish image fix: replaced four 404 eunomia.dev image URLs with GitHub
  raw image URLs through the DEV web editor. Public-page full-scroll QA verified
  4 article images, 4 raw-proxied DEV image URLs, 0 old 404 image URLs, and no
  broken article images.
