# Reddit draft

Platform: Reddit

Decision: comment or submit only after fresh subreddit rule and duplicate checks.

Candidate communities:

- r/eBPF
- r/netsec
- r/LocalLLaMA
- r/devops, only if the active discussion is about running agents with local/system privileges

Primary link: https://eunomia.dev/blog/2026/07/15/ebpf-ai-agent-policy-enforcement/

Comment option:

```text
One practical issue with agent rules is that many of them are not directly enforceable from a single tool call. In an ActPlane study we split 2,116 CLAUDE.md/AGENTS.md statements from 64 repos into individual rules and found that a lot of the hard cases depend on project/task context or cross-event state, such as whether tests ran after the last relevant edit.

That is where OS-level enforcement helps: not as a replacement for prompts or sandboxing, but as a lower layer that can observe subprocesses, files, network, and ordering after the agent starts executing.

Writeup: https://eunomia.dev/blog/2026/07/15/ebpf-ai-agent-policy-enforcement/
Source: https://github.com/eunomia-bpf/ActPlane
```

Browser QA:

- Check subreddit rules and self-promotion norms.
- Prefer comments on active discussions over new link posts.
- Stop before `Post` unless explicitly confirmed.
