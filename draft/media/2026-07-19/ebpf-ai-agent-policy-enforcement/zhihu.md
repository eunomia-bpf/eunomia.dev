# Zhihu draft

Platform: Zhihu

Decision: publish after editor QA.

Source body: `docs/blog/posts/ebpf-ai-agent-policy-enforcement.zh.md`

Title: 实证研究：AI Agent 规则需要上下文与分层强制执行

Source URL for ledger: https://eunomia.dev/zh/blog/2026/07/15/ebpf-ai-agent-policy-enforcement/

GitHub: https://github.com/eunomia-bpf/ActPlane

Paper: https://arxiv.org/abs/2606.25189

Body policy:

- Preserve the Chinese canonical article body.
- Remove YAML front matter.
- Keep the source title as the Zhihu article title.
- Convert relative images under `imgs/` to checked public absolute URLs or upload manually if Zhihu strips external images.
- Add a short source/project note near the end only if useful:
  `ActPlane 源码与策略工件见 GitHub；论文见 arXiv。`

Browser QA:

- Check title length and first-screen hook.
- Check images, tables, and code blocks.
- Check GitHub, docs/project, and arXiv links. A visible eunomia.dev original/canonical note is optional.
- Stop before `发布` unless final publishing is explicitly confirmed.
