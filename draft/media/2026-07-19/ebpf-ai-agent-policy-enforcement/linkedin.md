# LinkedIn draft

Platform: LinkedIn

Decision: publish as a professional feed post after preview QA.

Primary link: https://eunomia.dev/blog/2026/07/15/ebpf-ai-agent-policy-enforcement/

```text
A rule like “run the full test suite before committing” looks simple until an AI coding agent edits a file after the last test run and then calls git commit.

The hard part is not the wording of the rule. It is the missing state: which test command counts, which files invalidate the result, and whether the test happened after the relevant edit.

In the ActPlane study, we analyzed 2,116 statements from CLAUDE.md and AGENTS.md files across 64 popular repositories. 64% were behavioral policies, not just background context. Many touched files, processes, or network activity, but most required project or task context before they could become deterministic OS-level checks.

That is the gap prompt rules and tool-call guardrails miss. Useful agent policy enforcement has to compile human intent into concrete state, then enforce the OS-observable subset across subprocesses, files, network, and cross-event ordering.

Full writeup: https://eunomia.dev/blog/2026/07/15/ebpf-ai-agent-policy-enforcement/
Paper: https://arxiv.org/abs/2606.25189
GitHub: https://github.com/eunomia-bpf/ActPlane
```

Browser QA:

- Check posting identity and visibility.
- Check the first visible lines before "see more".
- Check link card.
- Stop before `Post` unless final publishing is explicitly confirmed.
