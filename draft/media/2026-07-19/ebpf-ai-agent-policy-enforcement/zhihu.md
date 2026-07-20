# Zhihu draft

Platform: Zhihu

Decision: published after editor and public-page QA.

Published URL: https://zhuanlan.zhihu.com/p/2062539029892151274

Selected question: [AI Agent（智能体）正在重塑开发范式，但距离真正可靠的生产级应用还有多远？](https://www.zhihu.com/question/2041284524718436489)

Source body: `docs/blog/posts/ebpf-ai-agent-policy-enforcement.zh.md`

Upload artifact: `zhihu-upload.md`

Title: 实证研究：AI Agent 规则需要上下文与分层强制执行

Source URL for ledger: https://eunomia.dev/zh/blog/2026/07/15/ebpf-ai-agent-policy-enforcement/

GitHub: https://github.com/eunomia-bpf/ActPlane

Paper: https://arxiv.org/abs/2606.25189

Body policy:

- Preserve the Chinese canonical article body.
- Remove YAML front matter.
- Keep the source title as the Zhihu article title.
- The final upload artifact omits inline images because repeated Markdown and
  DOCX imports produced partial or complete image failures. The ActPlane
  architecture image was uploaded separately as the article cover.
- Convert the two Markdown tables to lists before import because Zhihu document import may flatten table structure.
- Add a short source/project note near the end only if useful:
  `ActPlane 源码与策略工件见 GitHub；论文见 arXiv。`

Browser QA:

- Check title length and first-screen hook.
- Check images, tables, and code blocks.
- Check GitHub, docs/project, and arXiv links. A visible eunomia.dev original/canonical note is optional.
- Stop before `发布` unless final publishing is explicitly confirmed.

Completed QA:

- Verified the corrected price sentence, 11 section headings, 15 links, two
  code blocks, and zero broken-image placeholders before publishing.
- Verified the GitHub and arXiv links on the public article through Zhihu's
  outbound redirect links.
- Verified the stored cover image in the editor at 1403 x 636 pixels.
