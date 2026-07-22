---
date: 2026-07-20
slug: agent-work-unit
title: "From Tokens to Verifiable Work: Agent Infra Is Rewriting Its Unit of Measure"
description: As AI agents evolve from single model calls into long-running, multi-tool execution systems, infrastructure must shift from tokens toward verifiable work units that bundle acceptance conditions, execution evidence, total cost, and accountability boundaries.
research_question: When an agent grows from a single model call into a long-running, multi-tool execution system, what should infrastructure use as its fundamental unit of observation, evaluation, and billing?
research_window: 2026-07-18 to 2026-07-20, lookback to 2026-07-17 for the most recent workday, 30-day lookback for mechanisms and counterexamples
tags:
  - daily-analysis
  - research
  - AI Agent
  - Agent Infrastructure
  - Observability
---

# From Tokens to Verifiable Work: Agent Infra Is Rewriting Its Unit of Measure

A single model call is easy to account for. Which model, how many tokens, how long, whether it failed. Agents break this simplicity. One user request now sprawls across dozens of calls, tool invocations, pauses for human approval, retries that succeed or don't, state changes in external systems. The calls are still observable. Whether the work got done is another question.

A cluster of releases in the past week makes this mismatch hard to ignore. [OpenAI](https://openai.com/index/a-scorecard-for-the-ai-age/) proposed measuring AI by completed useful work rather than raw capability. [Google Cloud](https://cloud.google.com/blog/products/ai-machine-learning/13-demos-on-gemini-enterprise-agent-platform) demoed its Agent Platform as a full lifecycle: build, run for weeks, govern, evaluate. [GitHub](https://github.blog/changelog/2026-07-17-repository-level-github-copilot-usage-metrics-generally-available/) now reports Copilot activity per repository and PR. Alibaba Cloud's [AgentLoop](https://www.alibabacloud.com/en/notice/commercialization_notice_for_agentloop_795?_p_lc=1) went commercial on July 20, bundling AI Agent sessions, tool calls, tokens, and traces into one billable observability layer.

Different product lines, same direction. **Agent infrastructure is shifting from managing model calls to managing work. The industry can record most of what an agent does. What it cannot yet agree on: when the work is done, whether the result is right, and what a reliable completion actually costs.**

<!-- more -->

## Four layers, each blind to the one above

[OpenTelemetry's GenAI conventions](https://opentelemetry.io/blog/2026/genai-observability/) decompose an agent run into spans such as `invoke_agent`, model calls, and `execute_tool`, with token counts, latency, and results attached. This is pure telemetry: what the system did, where it slowed, where it broke. For debugging retry storms or runaway token spend, nothing else will do.

Agent platforms add a layer above: the session or execution trace. AgentLoop defines a trace as the full path from user input through model, tools, retrieval, and memory to final response. Google's demos add persistent state machines, pause-and-resume, human-in-the-loop approval. This unit tracks real execution better than a call does; a single task can span many calls and run for days.

GitHub's new repository-level Copilot metrics go higher. They count PRs created by coding agents, PRs merged, and review suggestions accepted. These are measures of delivery activity, not inference requests. Useful for spotting where AI is busy in a codebase. Less useful for knowing whether a merged change is correct, necessary, or actually reducing maintenance burden.

OpenAI's scorecard aims highest: completed useful work, full cost per successful task, reliability, value per dollar at scale. The cost explicitly includes retries, human review, and rework, not just token prices. Closest to a business outcome; hardest to measure automatically.

Stack them:

`Call telemetry → Execution trace → Delivery activity → Verifiable result`

Each layer describes the same work from a wider angle, but you cannot substitute one for another. A successful call says nothing about the trace's coherence. A complete trace says nothing about whether the PR was worth merging. A merged PR says nothing about whether the user's problem is solved. Dashboards get richer; decisions do not automatically follow.

## Why now

Agents are getting longer, wider, and more brittle at the same time. Google's 13 demos span a simple ADK agent to persistent workflows that run for weeks, with MCP tools, multi-agent pipelines, identity federation, human approval gates, and an evaluation flywheel fed by OTel traces. A single prompt tweak can improve three demos and break ten. Regression testing stops being optional.

Cost structure changes with form. Short conversations can be priced by token. Long tasks cannot: planning quality, tool failures, permission blocks, retries, human wait time, and rework all enter the bill. A cheap model that needs three attempts can cost more than an expensive model that passes on the first. Model routing becomes optimization over expected completion cost, not price-per-token.

Commercial products are already billing along this chain. AgentLoop's docs put trace-to-dataset conversion, evaluation, experimentation, prompt versioning, canary release, and audit in one closed loop. The AI Agent onboarding guide now ships a local collector with hooks to capture sessions from Claude Code, Codex, Cursor, and others. The signal is not that any single product delivers everything it promises. Rather, product boundaries have stretched from LLM observability into the full agent production lifecycle.

## Traces are not results

The tension shows up in the announcements themselves. GitHub counts PRs created and merged. OpenAI wants to shift from "adopted" to "completed." Google and Alibaba Cloud both place automatic evaluation after observation: generate datasets from production traces, then let an LLM judge catch regressions. Everyone is moving toward outcomes. Everyone is still landing on proxies of varying strength.

The difficulties are not theoretical. A [benchmark audit](https://arxiv.org/abs/2607.01211) published earlier this month replayed 740 performance-optimization tasks across four cloud machine types. Reference patches that pass on one machine fail on another. Rankings shift with scoring rules; 9 of 28 pairwise comparisons reversed order depending on how scores were aggregated. The study targets performance benchmarks specifically, not all agent evaluation. But it marks a real boundary: even with runnable tests and numeric scores, environment and aggregation can flip the conclusion.

[DeepSWE](https://arxiv.org/abs/2607.07946) points in a better direction. The benchmark hand-wrote behavioral verifiers for 113 long tasks across 91 repositories, checking whether the user-requested functionality works in an isolated environment, rather than whether the patch matches a reference. It releases both verifiers and complete traces. An independent LLM judge disagreed with DeepSWE's verifiers 1.4% of the time versus 32.4% for SWE-Bench Pro's inherited tests. Hand-written verifiers are not the same as real user value. But bundling task description, execution environment, verifier, and trace gets closer to a reproducible "done" than reusing whatever tests shipped with a merged PR.

Polished lifecycle diagrams do not eliminate infrastructure failure. On July 20, a user on Google Developer Forums was still reporting that [Gemini Enterprise's custom MCP reload returns 401](https://discuss.google.dev/t/gemini-enterprise-custom-mcp-reload-custom-actions-always-fails-with-401-ui-uses-api-key-instead-of-oauth-token/371907/2). This is one case, not a platform verdict, but it illustrates why traces must bind to external state. The trace may record the call perfectly. The action the user needed may never have taken effect.

Agent systems need more than additional traces, and more than appending a score to each one. A more robust abstraction would define work as a verifiable work unit containing at least:

- Clear intent, target object, and "done" condition.
- Permissions, policies, and human-approval boundaries in effect during execution.
- Evidence chain of model calls, tool calls, state changes, and output artifacts.
- A deterministically checkable outcome oracle such as a test, deployment status, ledger change, or target-system receipt.
- When deterministic judgment is not possible, scoring basis, confidence, counterexamples, and human-review outcome.
- Total cost from tokens, runtime, retries, human wait time, and rework.
- Agent, model, prompt, skill, tool, and environment versions for replay and attribution.

[AgentSight](https://arxiv.org/abs/2508.02736) correlates semantic intent with kernel-visible effects at the system boundary, with overhead under 3%. It detects prompt injection, inference loops, multi-agent bottlenecks. This fills in execution evidence, not business-result oracles. A causal trace proves what the agent did to the system; it cannot prove the user's goal was met.

[ActPlane](https://arxiv.org/abs/2606.25189) fills in permission and policy context. Its study of 1,127 policy statements found that 73.6% require project or task context; for cross-event policies, 95%. Acceptance cannot rely on reviewing which tools the agent called. It must know what policy was in force, what authorization was granted, what environment version applied. AgentSight and ActPlane supply evidence chain and accountability boundary, but neither replaces an outcome oracle, total cost accounting, or human review. I include them to mark which fields a complete work unit still lacks, not to package existing projects as a finished answer.

This kind of work unit would not replace OTel spans or platform traces. It would connect low-level telemetry to high-level acceptance so the system can answer two different questions: why this execution slowed down, and whether it ultimately did the right thing.

## Second-order effects

Once outcomes become the optimization target, observability data stops being just a failure scene. Production failures flow into datasets, datasets become regression tests, regression results gate whether a prompt or model version ships. Google's evaluation flywheel and AgentLoop's Trace2Dataset both walk this path. The valuable flywheel is not "more data is better"; it is "can failures be converted into stable, replayable acceptance conditions that block the same class of regression."

Prompts and skills will increasingly resemble code assets: versioned, reviewed, canary-deployed, and rolled back. A change in agent behavior can come from the model, but it can just as easily come from instructions, tool permissions, or environment setup. Recording only the model version misses most of the causal variables.

Observability and privacy will pull harder against each other. OpenTelemetry does not collect prompt content or tool parameters by default because those fields may contain sensitive data. AgentLoop's local collector explicitly offers credential-masking configuration. Result verification wants more context; fuller recording means larger exposure surface and higher compliance cost. Verifiable work units will need evidence minimization: test summaries, hashes, signed receipts, and controlled references, not full chain-of-thought dumps and business data by default.

## Counterarguments

This analysis draws mainly on vendor docs and product releases. It can show product direction; it cannot show widespread enterprise adoption or that the promised closed loops actually improve agent quality. AgentLoop's figures on diagnosis time and cost traceability are vendor assertions. Independent case studies or public experiments would be needed to verify them.

"Defining done" is not equally tractable across tasks. Compilation, tests, deployment, and ledger state form strong oracles. Research, design, communication, and exploration need multidimensional judgment. Forcing the second class into a single score may reward shallow, measurable outputs and suppress valuable but uncertain work.

Two kinds of evidence would overturn this thesis. First: if later independent deployments show that call-level or session-level metrics already predict real business outcomes reliably, a work-unit layer only adds complexity. Second: if outcome definitions cannot transfer across models, agents, and platforms, the supposedly universal primitive collapses into per-workflow custom schemas. Current evidence supports "a connecting layer is needed." It does not yet show what the final standard will look like.

## What to do now

No need to wait for platform vendors to converge. Pick one recurring, result-checkable agent workflow. Write "done" as a state an external system can verify, not "agent returned success." A code task: tests pass, target diff exists. A publishing task: public URL reachable, content checksum matches, ledger consistent. An operations task: resource state and change receipt both match expectation.

Separate activity metrics from outcome metrics. Tokens, latency, tool errors, traces explain execution. Success rate, regression rate, human review, retry count, cost per success judge value. Do not roll PR count, session count, and judge score into a single "productivity" number.

Build a replayable entry point for failures. Save the minimum necessary environment version, input conditions, artifacts, and acceptance results; add high-value failures to the regression set. Prefer deterministic oracles. Use semantic judges for what cannot be coded. Keep a human gate for low-confidence or high-risk tasks. Only then does observation feed back into improvement rather than stopping at the dashboard.

## Open questions

- Can a portable schema connect task goal, system effect, acceptance result, version, and full cost without forcing all workflows to share one outcome oracle?
- Under what conditions do activity metrics (PRs, sessions, traces) predict user-visible results, and under what conditions do they systematically distort?
- How do verifiers resist drift in machine type, dependency version, external services, and data state? Can they stay replayable as environments change?
- How are automatic judges calibrated, audited, and versioned, especially when the evaluator itself calls tools and reasons over multiple steps?
- How can evidence chains achieve minimal disclosure, so acceptance, attribution, and privacy protection hold simultaneously?
- Which work-unit fields can be reused across agents and platforms, and which must be defined per business workflow?

## References

- [OpenAI, A scorecard for the AI age, 2026-07-17](https://openai.com/index/a-scorecard-for-the-ai-age/)
- [Google Cloud, 13 hands-on demos to build on Gemini Enterprise Agent Platform, 2026-07-17](https://cloud.google.com/blog/products/ai-machine-learning/13-demos-on-gemini-enterprise-agent-platform)
- [GitHub Changelog, Repository-level GitHub Copilot usage metrics generally available, 2026-07-17](https://github.blog/changelog/2026-07-17-repository-level-github-copilot-usage-metrics-generally-available/)
- [Alibaba Cloud, What is AgentLoop, updated 2026-06-22](https://www.alibabacloud.com/help/en/cms/cloudmonitor-2-0/what-is-an-agentloop)
- [Alibaba Cloud, Onboarding AI Coding Agents, updated 2026-07-17](https://www.alibabacloud.com/help/en/cms/ai-application-access-ai-coding-agent)
- [OpenTelemetry, Inside the LLM Call: GenAI Observability with OpenTelemetry, 2026-05-14](https://opentelemetry.io/blog/2026/genai-observability/)
- [DeepSWE: Measuring Frontier Coding Agents on Original, Long-Horizon Engineering Tasks, 2026-07-08](https://arxiv.org/abs/2607.07946)
- [Chen et al., Are Performance-Optimization Benchmarks Reliably Measuring Coding Agents?, arXiv v2, 2026-07-16](https://arxiv.org/abs/2607.01211)
- [AgentSight: System-Level Observability for AI Agents Using eBPF, arXiv v2, 2025-08-15](https://arxiv.org/abs/2508.02736)
- [ActPlane: Programmable OS-Level Policy Enforcement for Agent Harnesses, arXiv v2, 2026-06-30](https://arxiv.org/abs/2606.25189)
