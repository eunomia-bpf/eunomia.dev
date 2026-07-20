# LinkedIn draft

Platform: LinkedIn

Decision: published.

Primary link: https://eunomia.dev/blog/2026/07/15/ebpf-ai-agent-policy-enforcement/

```text
A rule like “run the full test suite before committing” looks simple until an AI coding agent edits a file after the last test run and then calls git commit.

The hard part is not the wording of the rule. It is the missing state: which test command counts, which files invalidate the result, and whether the test happened after the relevant edit.

In the ActPlane study, we analyzed 2,116 statements from CLAUDE.md and AGENTS.md files across 64 popular repositories. 64% were behavioral policies, not just background context. Many touched files, processes, or network activity, but most required project or task context before they could become deterministic OS-level checks.

That is the gap prompt rules and tool-call guardrails miss. Useful agent policy enforcement has to compile human intent into concrete state, then enforce the OS-observable subset across subprocesses, files, network, and cross-event ordering.

Full writeup, with the arXiv paper and ActPlane GitHub repo linked inside:
https://eunomia.dev/blog/2026/07/15/ebpf-ai-agent-policy-enforcement/
```

Browser QA:

- Posting identity: Yusheng Zheng.
- Visibility: public.
- First visible lines checked in composer.
- Link card resolved to eunomia.dev, not GitHub.
- Published URL: https://www.linkedin.com/feed/update/urn:li:share:7484770128912465920
