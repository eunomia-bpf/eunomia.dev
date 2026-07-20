# Zhihu AI Works Reference

Use this reference for Zhihu AI Works forms, campaign notes, project material,
and reusable field copy.

## Default Field Strategy

- **应用链接:** Use the GitHub repository link when the repo is the primary
  artifact or there is no stable hosted demo. If a hosted demo exists, use the
  demo as the application link and include GitHub in the description or related
  links.
- **项目简介:** Write a compact but specific description. Name the problem,
  mechanism, and audience. Avoid a bare category label.
- **项目推广/项目说明:** Explain why the project matters, what it does, how it
  works at a high level, concrete scenarios, and where to inspect the code.
- **图片:** Prefer a real artifact image: architecture diagram, UI screenshot,
  trace/terminal result, benchmark chart, or repository social image. Avoid
  generic logo-only or abstract AI art unless there is no better visual.

## Browser Form Checks

- Check every field is filled with the most specific public link available,
  especially GitHub repos, docs, papers, and demos.
- Check the project type/category against how Zhihu currently labels AI Works.
- Check the project intro is understandable without opening GitHub.
- Check image upload/crop and avoid generic visuals when an architecture or UI
  artifact exists.
- Stop before final submit unless the user explicitly confirms.

## Content Ratio

- Hold the 80% contribution / 20% promotion posture even in event fields.
- Give readers enough context to understand the problem before asking them to
  star, try, or share.
- Use GitHub links as proof and continuation, not as the entire value of the
  entry.

## Strong Project Description Pattern

Use this shape for AI/security/systems projects:

1. Problem: what breaks in real AI-agent or systems workflows.
2. Mechanism: what the project observes, controls, enforces, or automates.
3. Use cases: who uses it and in what production/research scenario.
4. Evidence: GitHub repo, paper, docs, demo, examples, or benchmark.
5. Boundary: what it does not claim to solve yet.

## Example Framing Notes

- **actplane:** Do not describe it only as an "eBPF AI Agent 策略引擎".
  Emphasize kernel-level enforcement of information-flow control for AI-agent
  actions, so agents can follow instructions under safety, compliance, and
  reliability constraints. Explain scenarios such as preventing sensitive data
  exfiltration, constraining tool/file/network access, auditing agent actions,
  and enforcing policy outside the model runtime.
- **AgentSight:** Lead with observability and debugging for AI-agent execution.
  Make clear what signals it captures, how it helps diagnose behavior, and how
  developers can inspect or reproduce issues from the GitHub repo.
- **bpftime / eBPF runtime work:** Lead with the developer problem: running,
  testing, or accelerating eBPF workflows in user space. Mention project links
  after the practical value is clear.

## Reusable Checklist

- Name the project, repo URL, and canonical docs when available.
- Explain the reader/user persona in one sentence.
- Include 3-5 concrete use cases instead of broad adjectives.
- Mention security, compliance, reliability, or observability only when the
  project mechanism actually supports the claim.
- Keep the event copy standalone: a reader should understand the project even
  before opening GitHub.

## Follow-Up

- After confirmed submission, record the AI Works URL or project ID when
  visible.
- Check project comments, reactions, ranking/status, and private messages only
  when the user asks or follow-up was part of the task.
- Turn unclear audience reactions into improved project intros, README sections,
  diagrams, or Zhihu answer topics.
