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

For the past several years, AI infrastructure has found it easy to answer what happened during a single model call. Which model ran, how many input and output tokens flowed, how long the request took, whether an error occurred. Agents stretch that question out. A single user request can now span dozens of model calls, multiple tools, long pauses, human approval gates, failed retries, and external system state changes. Individual calls remain observable, yet it becomes harder to say whether the work is actually done.

A recent wave of product and engineering signals has pushed this mismatch to the foreground. On July 17, [OpenAI](https://openai.com/index/a-scorecard-for-the-ai-age/) proposed measuring AI by completed useful work, cost per successful task, and result reliability. [Google Cloud](https://cloud.google.com/blog/products/ai-machine-learning/13-demos-on-gemini-enterprise-agent-platform) presented its Agent Platform as a full lifecycle from build through persistent execution, governance, and evaluation. [GitHub](https://github.blog/changelog/2026-07-17-repository-level-github-copilot-usage-metrics-generally-available/) pushed Copilot metrics down to the repository and pull-request level. Alibaba Cloud began commercial billing for [AgentLoop](https://www.alibabacloud.com/en/notice/commercialization_notice_for_agentloop_795?_p_lc=1) on July 20, folding coding-agent sessions, tool calls, tokens, logs, and traces into one observability system.

These releases come from different product lines, yet they point toward the same shift. **Agent infrastructure is moving from "model-call management" to "work management." The industry can now record most of an execution, but it has not yet reached shared definitions of what counts as done, whether the result is correct, and what one reliable completion actually costs.**

<!-- more -->

## Four units of measure that emerged together but cannot be substituted

[OpenTelemetry's GenAI semantic conventions](https://opentelemetry.io/blog/2026/genai-observability/) can already decompose an agent execution into `invoke_agent`, model-call, and `execute_tool` spans while recording model, token counts, latency, and tool results. That layer solves telemetry problems: what the system did, where it slowed down, where it failed. For diagnosing retry loops, slow tools, and anomalous token consumption, this layer is indispensable.

Agent platforms are adding a second layer above it: the session or execution trace. Alibaba Cloud's AgentLoop documentation defines a complete trace as the full path from user input through model, tools, retrieval, and memory to the final response. Google's Agent Platform demos add persistent state machines, pause-and-resume, human approval, and cross-agent protocols. This unit sits closer to actual execution than a model call, because one task can span many calls and run for days.

GitHub's repository-level Copilot metrics, launched on July 17, move one layer higher still. The new endpoint reports coding-agent PRs created and merged per repository, plus code-review PRs and suggestion counts. The object being measured is no longer an inference request but delivery activity inside a codebase. This metric tells managers where AI is producing visible activity, yet on its own it cannot prove that a merged change is correct, necessary, or reducing maintenance cost.

OpenAI's same-day scorecard goes one step further, asking for completed useful work, full cost per successful task, result reliability, and whether each dollar produces more value at scale. It explicitly includes retries, human review, and rework in the cost rather than comparing token prices alone. This unit comes closest to a business outcome and is the hardest to obtain automatically.

Stacking the four layers yields a chain that approaches outcomes step by step.

`Call telemetry -> Execution trace -> Delivery activity -> Verifiable result`

They describe the same piece of work, but they are not interchangeable metrics. A successful call does not equal a sensible trace. A complete trace does not mean the PR was worth merging. A merged PR does not mean the user's problem is solved. Treating a visible number from a lower layer as if it were a result from a higher layer makes dashboards richer while decisions fail to become more reliable.

## Why this shift is happening now

Agents' engineering shape is simultaneously lengthening execution time, expanding permissions, and multiplying failure paths. Google's 13 demos range from a simple ADK Agent up to persistent workflows running for weeks, MCP tools, A2A multi-agent pipelines, identity and gateway integration, human approval, and an evaluation flywheel built on OTel data. One prompt tweak may improve three examples and break ten others, so a deployment naturally requires historical data, regression evaluation, and failure clustering.

Cost structure changes alongside form. Short conversations can be priced roughly by token, but long tasks are affected by planning quality, tool errors, permission blocks, model retries, human wait time, and rework. If a cheap model needs multiple attempts, the total cost of a successful result may be higher. If an expensive model passes acceptance on the first try, it may actually be cheaper. Model routing therefore becomes optimization over "success probability times full execution cost," not price-per-token comparison.

Commercial products have already started billing and competing along this chain. AgentLoop entered pay-per-use billing on July 20. Its documentation places trace-to-dataset conversion, evaluation, experimentation, prompt and skill versioning, canary release, and audit inside a single closed loop. The July 17 update to the coding-agent onboarding docs adds a local collector and hooks or plugins to capture sessions and calls from Claude Code, Codex, Cursor, and similar tools. The notable signal is not whether any single product has delivered everything it claims, but that product boundaries have expanded from LLM observability into the full agent production lifecycle.

## A trace is not a result, and automatic scoring is not acceptance

The clearest tension comes from the announcements themselves. GitHub's repository-level metrics count PR creation, merge, and review suggestions. OpenAI argues for shifting from "what was adopted" to "what was completed." Google and Alibaba Cloud both place automatic evaluation after observation, hoping to generate datasets from production traces and let AutoRater, LLM-as-a-Judge, or Agent-as-a-Judge catch regressions. Everyone is moving toward outcomes, yet the actual landing points are still proxy metrics of varying strength.

Difficulties with automatic evaluation are not theoretical worries. A [coding-agent benchmark audit](https://arxiv.org/abs/2607.01211) published in early July replayed 740 performance-optimization tasks across four cloud machine types and found that several benchmarks' reference patches are unstable across environments. Rankings also shift with scoring rules: 9 of 28 paired comparisons reversed order. This result targets performance-optimization benchmarks and does not invalidate all agent evaluation, but it clearly marks a boundary. Even with runnable tests and numeric scores, environment, oracle, and aggregation method can change the conclusion.

[DeepSWE](https://arxiv.org/abs/2607.07946) provides a positive contrast that comes closer to a verifiable work unit. It hand-wrote behavioral verifiers for 113 long tasks across 91 code repositories, checking user-requested functionality in isolated environments while allowing implementations that differ from the reference patch. The paper releases both verifiers and complete traces. An independent LLM judge disagreed with DeepSWE verifiers 1.4% of the time, while disagreeing with SWE-Bench Pro's inherited tests 32.4% of the time. This does not prove that hand-written verifiers equal real user value, but it shows that bundling task description, execution environment, verifier, and complete trace gets closer to a reproducible definition of done than reusing a repository's existing tests.

Community friction also reminds us that a polished lifecycle diagram does not eliminate infrastructure failure. Google Developer Forums still listed a user report on July 20 about [Gemini Enterprise custom MCP reload returning 401](https://discuss.google.dev/t/gemini-enterprise-custom-mcp-reload-custom-actions-always-fails-with-401-ui-uses-api-key-instead-of-oauth-token/371907/2). This is a single case and cannot represent overall platform reliability, but it shows why task results must be bound to external state. The trace may have fully recorded the call while the action the user needed never actually took effect.

Agent systems therefore need more than additional traces, and more than appending a summary score to each trace. A more robust abstraction would define a piece of work as a verifiable work unit containing at least:

- Clear intent, target object, and "done" condition.
- Permissions, policies, and human-approval boundaries in effect during execution.
- Evidence chain of model calls, tool calls, state changes, and output artifacts.
- A deterministically checkable outcome oracle such as a test, deployment status, ledger change, or target-system receipt.
- When deterministic judgment is not possible, scoring basis, confidence, counterexamples, and human-review outcome.
- Total cost from tokens, runtime, retries, human wait time, and rework.
- Agent, model, prompt, skill, tool, and environment versions for replay and attribution.

[AgentSight](https://arxiv.org/abs/2508.02736) correlates semantic intent with kernel-visible effects at the agent-system boundary. The paper reports overhead below 3% and demonstrates detection of prompt injection, inference loops, and multi-agent bottlenecks. It fills in execution evidence, not business-result oracles. A causal trace can prove what system actions the agent took, but cannot on its own prove that the user's goal was met.

[ActPlane](https://arxiv.org/abs/2606.25189) fills in permission and policy context. Its study of 1,127 system-observable policy statements found that 73.6% require project or task context, rising to 95% for cross-event policies. This means acceptance cannot rely only on reviewing which tools the agent called. It must also know which policy was resolved at the time, what authorization was granted, and which version the execution environment applied. AgentSight and ActPlane supply evidence chain and accountability boundary respectively, but neither replaces outcome oracle, total cost, or human review. Including them here marks which fields a complete work unit still lacks, not packaging existing projects as a complete answer.

Such a work unit would not replace OTel spans or platform traces. It would connect low-level telemetry to high-level acceptance so the system can answer two distinct questions: why this execution slowed down, and whether it ultimately did the right thing.

## Second-order effects: infrastructure will move from seeing agents to managing change

Once outcomes rather than calls become the optimization target, observability data stops being just a failure scene. Production failures flow into datasets, datasets become regression tests, and regression results decide whether a prompt, skill, model, or tool version ships. Google's evaluation flywheel and AgentLoop's Trace2Dataset both walk this path. The truly valuable flywheel is not "more data is better" but "can failures be converted into stable, replayable acceptance conditions that block the same class of regression."

Prompts and skills will increasingly resemble code assets. They need versioning, review, canary deployment, and rollback, because a change in agent behavior can come from the model, but equally from instructions, tool permissions, or environment-setup steps. Recording only the final model version misses a large set of causal variables.

The tension between observability and privacy will intensify. OpenTelemetry does not collect prompt content and tool parameters by default because those fields may contain sensitive data. AgentLoop's local-collector documentation explicitly provides credential-masking configuration. Result verification often requires more context, but fuller recording means larger exposure surface and compliance cost. Future verifiable work units must allow evidence minimization, keeping only test summaries, hashes, signed receipts, or controlled references rather than defaulting to full chain-of-thought and business data.

## Counterarguments and uncertainty

This set of signals comes primarily from vendor documentation and product releases. It can prove product direction but cannot prove widespread enterprise adoption or that these closed loops reliably improve agent quality. AgentLoop's figures on diagnosis time, anomalous cost, and traceability are vendor assertions and still require independent case studies or public experiments to verify.

"Defining done" is also not equally easy across all tasks. Compilation, testing, deployment, and ledger state can form strong oracles, but research, design, communication, and exploratory work need multidimensional judgment. Forcing the latter class of tasks into a single score may reward shallow, easy-to-measure outputs and suppress valuable but uncertain exploration.

Two classes of evidence would overturn this report's thesis. First, if later independent deployments show that call-level or session-level metrics already predict real business outcomes reliably, adding a work-unit layer only creates complexity. Second, if outcome definitions cannot transfer across models, agents, and platforms, the so-called universal primitive may collapse into a per-workflow custom schema. Current evidence supports "a connecting layer is needed" without yet proving what the final standard will look like.

## What developers can do now

There is no need to wait for platforms to provide a unified answer. Start by choosing one recurring, result-checkable agent workflow and write "done" as a state that an external system can verify, not as "agent returned success." A code task can require tests to pass and the target diff to exist. A publishing task can require a public URL, content checksum, and ledger consistency. An operations task can require resource state and change receipt to match.

Then separate activity metrics from outcome metrics. Use tokens, latency, tool errors, and traces to explain execution. Use success rate, regression rate, human review, retry count, and cost per success to judge value. Do not roll PR count, session count, or judge score into a single "productivity" number that looks unified.

Finally, build a replayable entry point for failures. Save the minimum necessary environment version, input conditions, artifacts, and acceptance results, and add high-value failures to the regression set. Prefer deterministic oracles. Assign semantic judges to handle what cannot be coded. Preserve a human gate for low-confidence or high-risk tasks. Only then does observation actually feed back into improvement, rather than stopping at the dashboard.

## Questions still unanswered

- Whether a portable schema can connect task goal, system effect, acceptance result, version, and full cost without forcing all workflows to share one outcome oracle.
- Under which conditions activity metrics such as PRs, sessions, and traces predict user-visible results, and under which they systematically distort.
- How verifiers resist drift in machine type, dependency version, external services, and data state, remaining replayable after environments change.
- How automatic judges are calibrated, audited, and versioned, especially when the evaluator itself calls tools and runs multi-step reasoning.
- How evidence chains can achieve minimal disclosure so that acceptance, attribution, and privacy protection can hold simultaneously.
- Which work-unit fields can be reused across agents and platforms, and which must be defined by the specific business workflow.

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
