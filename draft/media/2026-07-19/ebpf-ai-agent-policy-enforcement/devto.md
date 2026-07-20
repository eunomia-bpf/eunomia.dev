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
tags: security, ai, linux, opensource
canonical_url: https://eunomia.dev/blog/2026/07/15/ebpf-ai-agent-policy-enforcement/
---
```

Links:

- GitHub: https://github.com/eunomia-bpf/ActPlane
- Paper: https://arxiv.org/abs/2606.25189

Body policy:

- Preserve the English canonical article body.
- Remove YAML front matter and H1 from the body because DEV uses the title field as H1.
- Convert relative images under `imgs/` to public absolute URLs:
  `https://eunomia.dev/blog/2026/07/15/ebpf-ai-agent-policy-enforcement/imgs/...`
- Add the canonical/source note at the end if the platform preview does not make canonical visible.

Browser QA:

- Checked `canonical_url`; published page shows "Originally published at eunomia.dev".
- Checked H1 title plus H2/H3 body hierarchy in preview.
- Checked tables, code blocks, first images, GitHub links, and arXiv links in preview.
- Published after explicit user authorization for real publishing.
