# Medium draft

Platform: Medium

Decision: publish after recreating/overwriting the browser draft.

Source body: `docs/blog/posts/ebpf-ai-agent-policy-enforcement.md`

Title: An Empirical Study: AI Agent Rules Need Context and Layered Enforcement

Subtitle: AI agent rules look simple in CLAUDE.md, but ActPlane's 2,116-statement study shows why context and layered OS enforcement decide what can be checked.

Canonical URL: https://eunomia.dev/blog/2026/07/15/ebpf-ai-agent-policy-enforcement/

GitHub: https://github.com/eunomia-bpf/ActPlane

Paper: https://arxiv.org/abs/2606.25189

Tags: AI Agents, Security, eBPF, Open Source, Systems

Body policy:

- Preserve the English canonical article body.
- Remove YAML front matter.
- Keep the article H1 as the Medium title, not inside the body.
- Keep GitHub and paper links as evidence/source links.
- Add a low-key source note near the end if not already visible through Medium import.

Browser QA:

- Check the Medium import/canonical setting.
- Recreate or overwrite the current Medium draft before publishing; the first import draft title was accidentally duplicated during manual title editing.
- Check title has no `| eunomia` suffix.
- Check images render.
- Check tables, especially the statement/enforcement table and benchmark table, since Medium import may flatten tables.
- Stop before `Publish` unless final publishing is explicitly confirmed.
