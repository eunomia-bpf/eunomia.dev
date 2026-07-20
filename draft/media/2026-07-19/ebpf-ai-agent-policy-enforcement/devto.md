# DEV draft

Platform: DEV Community

Decision: publish after editor preview QA.

Source body: `docs/blog/posts/ebpf-ai-agent-policy-enforcement.md`

```yaml
---
title: An Empirical Study: AI Agent Rules Need Context and Layered Enforcement
published: false
description: AI agent rules look simple in CLAUDE.md, but ActPlane's 2,116-statement study shows why context and layered OS enforcement decide what can be checked.
tags: aiagents, security, ebpf, opensource
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

- Check `canonical_url`.
- Check H2/H3 hierarchy.
- Check tables and code blocks.
- Check images and image descriptions.
- Stop before final publish unless final publishing is explicitly confirmed.
