---
date: 2026-07-20
slug: agent-work-unit
description: As AI agents span model calls, tools, and approvals, tokens and traces no longer prove completion. This report argues for a verifiable unit of work.
tags:
  - daily-analysis
  - research
  - AI Agent
  - Agent Infrastructure
  - Observability
report_id: 2026-07-20-agent-work-unit
title: "From Tokens to Verifiable Work: Agent Infrastructure Is Changing Its Unit of Measure"
research_question: When an agent grows from a single model call into a long-running, multi-tool execution system, what should infrastructure treat as the basic unit of observation, evaluation, and billing?
research_window: July 18-20, 2026; recent-workday coverage extends to July 17, with a 30-day lookback for mechanisms and counterevidence
source_cutoff: 2026-07-20 00:35 PDT
thesis: Agent platforms are moving their unit of management upward from tokens and model calls to tasks and execution trajectories, but sessions, pull requests, trajectories, and automated scores remain proxies for work. The missing infrastructure primitive is a verifiable unit of work that combines acceptance criteria, execution evidence, total cost, and accountability boundaries.
---

# From Tokens to Verifiable Work: Agent Infrastructure Is Changing Its Unit of Measure

For the past several years, AI infrastructure has been best at answering what happened during a model call. Which model ran, how many input and output tokens it used, how long it took, and whether it returned an error were all straightforward questions. Agents stretch that problem over time. A single request can now span dozens of model calls, several tools, long pauses, human approvals, failed retries, and changes in external systems. Each call may still be observable while the system becomes less able to say whether the work was actually completed.

A cluster of recent product and engineering signals has brought this mismatch into focus. On July 17, [OpenAI](https://openai.com/index/a-scorecard-for-the-ai-age/) proposed measuring completed useful work, cost per successful task, and outcome reliability. [Google Cloud](https://cloud.google.com/blog/products/ai-machine-learning/13-demos-on-gemini-enterprise-agent-platform) presented its Agent Platform as a lifecycle spanning construction, durable execution, governance, and evaluation. [GitHub](https://github.blog/changelog/2026-07-17-repository-level-github-copilot-usage-metrics-generally-available/) moved Copilot metrics down to repository and pull request activity. Alibaba Cloud began commercial billing for [AgentLoop](https://www.alibabacloud.com/en/notice/commercialization_notice_for_agentloop_795?_p_lc=1) on July 20 while bringing coding-agent sessions, tool calls, tokens, logs, and traces into one observability system.

These releases come from different product lines, but they point in the same direction. **Agent infrastructure is moving from model-call management to work management. The industry can record most of a work process, but it has not agreed on what completion means, whether the result is correct, or how much one reliable completion actually costs.**

<!-- more -->

## Four Units That Appeared Together but Cannot Be Interchanged

[OpenTelemetry's GenAI semantic conventions](https://opentelemetry.io/blog/2026/genai-observability/) can already decompose an agent execution into spans such as `invoke_agent`, model calls, and `execute_tool`, with attributes for models, tokens, latency, and tool results. This is the telemetry layer: what the system did, where it slowed down, and where it failed. It remains indispensable for diagnosing retry loops, slow tools, and abnormal token consumption.

Agent platforms are adding a second layer above it: the session or execution trajectory. AgentLoop documentation defines a complete trajectory as the process that begins with user input and passes through models, tools, retrieval, and memory before reaching a final response. Google's Agent Platform demonstrations add durable state machines, pause and resume, human approval, and inter-agent protocols. This unit is closer to real execution than a model call because one task can cross many calls and run for days.

GitHub's repository-level Copilot metrics, released on July 17, move one layer higher. The new interfaces report pull requests created and merged by coding agents, along with pull requests and suggestions handled by code review, grouped by repository. The measured object is no longer an inference request but a delivery activity in a codebase. It can show managers where AI generated visible activity, but it cannot by itself establish that the merged change was correct, necessary, or helpful to long-term maintenance.

OpenAI's scorecard proposal from the same day moves one step higher again. It asks how much useful work was completed, the full cost of each successful task, whether outcomes are reliable, and whether each dollar produces more value as usage scales. It explicitly puts retries, human review, and rework into the cost instead of comparing only per-token prices. This unit is the closest to a business outcome and the hardest to obtain automatically.

Together, the four layers form a chain that gradually approaches the result:

`call telemetry -> execution trajectory -> delivery activity -> verifiable outcome`

They describe the same work, but they are not interchangeable measures. A successful call does not imply a sensible trajectory. A complete trajectory does not make a pull request worth merging. A merged pull request does not guarantee that the user's problem was solved. Treating a visible number at a lower layer as an outcome at a higher layer produces richer dashboards without making the associated decisions more reliable.

## Why This Shift Is Happening Now

The engineering shape of agents is simultaneously extending execution time, widening permissions, and multiplying failure paths. Google's 13 demonstrations range from simple ADK agents to durable workflows that can run for weeks, MCP tools, A2A multi-agent pipelines, identity and gateways, human approval, and evaluation flywheels built on OTel data. A prompt tweak may improve three examples and break ten others, so deployment naturally begins to require historical data, regression evaluation, and failure clustering.

The cost structure changes with it. Token counts can approximate the cost of a short conversation, but a long task is also shaped by planning quality, tool errors, blocked permissions, model retries, human waiting time, and rework. A cheaper model that needs several attempts may have a higher total cost per successful result. A more expensive model that passes acceptance on its first attempt may be cheaper overall. Model routing therefore becomes an optimization over success probability and full execution cost, not merely price per token.

Commercial products have started competing and charging around this chain. AgentLoop entered usage-based billing on July 20. Its documentation places trace-to-dataset conversion, evaluation, experiments, prompt and skill version management, staged releases, and audit in one loop. Updated coding-agent onboarding documentation from July 17 also provides a local collector that uses hooks or plugins to capture sessions and calls from tools including Claude Code, Codex, and Cursor. The important signal is not whether one vendor has already delivered every claim. It is that the product boundary has expanded from LLM observability to the production lifecycle of agents.

## A Trajectory Is Not an Outcome, and an Automated Score Is Not Acceptance

The clearest tension comes from the releases themselves. GitHub's repository metrics can count pull requests created and merged, along with review suggestions. OpenAI argues for moving from how much AI was adopted to what it completed. Google and Alibaba Cloud both place automated evaluation after observation, with online traces turned into datasets and AutoRater, LLM-as-a-Judge, or Agent-as-a-Judge systems used to identify regressions. All of them move closer to outcomes, but they stop at proxies of different strengths.

The limitations of automated evaluation are not theoretical. A July [audit of coding-agent benchmarks](https://arxiv.org/abs/2607.01211) replayed 740 performance-optimization tasks across four types of cloud machines and found that reference patches in several benchmarks were unstable across machine environments. Rankings among shared submissions also changed with the scoring rule, producing inconsistent ordering in 9 of 28 pairwise comparisons. The study concerns performance-optimization benchmarks and does not invalidate every form of agent evaluation. It does clearly expose the boundary: even with executable tests and numerical scores, the environment, oracle, and aggregation method can change the conclusion.

[DeepSWE](https://arxiv.org/abs/2607.07946) offers a positive comparison that is closer to a verifiable unit of work. It provides behavioral verifiers for 113 long-horizon tasks across 91 code repositories. Each verifier checks the requested behavior in an isolated environment while allowing implementations that differ from the reference patch. The paper also releases the verifiers and complete trajectories. An independent LLM judge disagreed with DeepSWE's verifiers on 1.4% of cases, but disagreed with inherited SWE-Bench Pro tests on 32.4%. This does not prove that a hand-written verifier equals real user value. It does suggest that combining a task description, execution environment, acceptance verifier, and full trajectory gets closer to a reviewable definition of completion than relying on a repository's inherited tests alone.

Community friction also shows why polished lifecycle diagrams do not eliminate infrastructure failures. On July 20, Google Developer Forums still listed a report that [reloading a custom MCP action in Gemini Enterprise returned 401](https://discuss.google.dev/t/gemini-enterprise-custom-mcp-reload-custom-actions-always-fails-with-401-ui-uses-api-key-instead-of-oauth-token/371907/2). One report cannot characterize the reliability of the platform as a whole. It does illustrate why a task result must be tied to external state. A trace may record every call while the action the user needed never takes effect.

Agent systems therefore need more than additional traces or a single aggregate score attached to every trace. A more robust abstraction would define each job as a verifiable unit of work containing at least:

- Explicit intent, target objects, and completion criteria.
- The permissions, policies, and human-approval boundaries that applied during execution.
- An evidence chain of model calls, tool calls, state changes, and produced artifacts.
- A deterministic outcome oracle where possible, such as tests, deployment state, ledger changes, or receipts from a target system.
- Evaluation criteria, confidence, counterexamples, and human-review results when deterministic judgment is impossible.
- Total cost across tokens, runtime, retries, human waiting, and rework.
- Versions of the agent, model, prompt, skill, tools, and environment for replay and attribution.

[AgentSight](https://arxiv.org/abs/2508.02736) connects semantic intent with kernel-visible effects at the boundary between agents and systems. The paper reports less than 3% overhead and demonstrates detection of prompt injection, reasoning loops, and multi-agent bottlenecks. This supplies execution evidence, not a business-outcome oracle. A causal trajectory can show which system actions an agent performed, but it cannot independently prove that the user's goal was achieved.

[ActPlane](https://arxiv.org/abs/2606.25189) supplies permission and policy context. Its study of 1,127 system-observable policies found that 73.6% require project or task context, rising to 95% among policies that span events. Acceptance therefore cannot look only at which tools an agent called. It must also know which policy was resolved, what authority was granted, and which environment version applied at the time. AgentSight and ActPlane contribute an evidence chain and accountability boundaries, respectively, but neither replaces an outcome oracle, total-cost accounting, or human review. Their role here is to identify fields still missing from a complete work unit, not to present existing projects as a finished answer.

Such a work unit would not replace OTel spans or platform traces. It would connect low-level telemetry to high-level acceptance so that a system can answer two distinct questions: why an execution became slower, and whether it ultimately completed the right work.

## Second-Order Effects: Infrastructure Moves from Seeing Agents to Managing Change

Once the outcome rather than the call becomes the optimization target, observability data stops being only a record of failure. Production failures enter datasets. Datasets become regression tests. Regression results determine whether a prompt, skill, model, or tool version can ship. Google's evaluation flywheel and AgentLoop's Trace2Dataset are both moving in this direction. A useful flywheel does not simply collect more data. It turns failures into stable, replayable acceptance conditions that prevent the same class of regression.

Prompts and skills will increasingly behave like code assets. They need versions, reviews, staged releases, and rollback because a behavioral change may originate in the model, an instruction, tool permissions, or environment setup. Recording only the final model version omits many causal variables.

The tension between observability and privacy will also intensify. OpenTelemetry does not collect prompt content and tool arguments by default because those fields may contain sensitive data. AgentLoop's local collection documentation likewise provides credential-masking configuration. Outcome verification often needs more context, but fuller records increase exposure and compliance costs. A future verifiable work unit must support evidence minimization, such as retaining test summaries, hashes, signed receipts, or controlled references instead of storing full chains of thought and business data by default.

## Counterarguments and Uncertainty

Most signals in this report come from vendor documentation and product releases. They establish product direction, but they do not prove broad enterprise adoption or show that these loops consistently improve agent quality. AgentLoop's figures for diagnosis time, anomalous cost, and traceability are vendor claims that still require independent case studies or public experiments.

Completion is not equally easy to define for every task. Compilation, tests, deployment, and ledger state can produce strong oracles. Research, design, communication, and exploratory work require multidimensional judgment. Compressing the latter into a single score may reward shallow output that is easy to measure while discouraging valuable but uncertain exploration.

Two kinds of evidence would overturn this report's thesis. First, independent deployments might show that call-level or session-level metrics already predict real business outcomes reliably, making an additional work unit needless complexity. Second, outcome definitions may fail to transfer across models, agents, and platforms, reducing any supposed common primitive to a custom schema for each workflow. Current evidence supports the need for a connecting layer, but it does not yet show what the final standard will look like.

## What Developers Can Do Now

There is no need to wait for platforms to agree. Start with one recurring agent workflow whose result can be checked, and define completion as a state that an external system can verify rather than as an agent returning success. A coding task might require passing tests and the intended diff. A publishing task might require a public URL, content verification, and a consistent ledger. An operations task might require both the target resource state and a change receipt.

Next, separate activity metrics from outcome metrics. Tokens, latency, tool errors, and traces explain execution. Success rate, regression rate, human review, retry count, and cost per success judge value. Do not compress pull request counts, sessions, or aggregate judge scores into a single productivity number.

Finally, create a replayable entry point for failures. Retain the minimum necessary environment versions, input conditions, artifacts, and acceptance results, then add high-value failures to the regression set. Prefer deterministic oracles, use semantic judges for what cannot be encoded, and preserve a human threshold for low-confidence or high-risk tasks. This is how observability enters an improvement loop instead of stopping at a dashboard.

## Questions That Remain Open

- Can a portable schema connect task goals, system effects, acceptance results, versions, and full cost without forcing every workflow to share one outcome oracle?
- Under what conditions do activity metrics such as pull requests, sessions, and trajectories predict user-visible outcomes, and when do they fail systematically?
- How can verifiers resist drift in machine type, dependency version, external services, and data state while remaining replayable after the environment changes?
- How should automated judges be calibrated, audited, and versioned, especially when the evaluator itself uses tools and multi-step reasoning?
- How can an evidence chain minimize disclosure while still supporting acceptance, attribution, and privacy?
- Which work-unit fields can transfer across agents and platforms, and which must be defined by a specific business workflow?

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
