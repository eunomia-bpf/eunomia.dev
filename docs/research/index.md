---
title: Research
description: "Source-grounded systems research briefs with testable claims, competing evidence, architecture consequences, explicit scope, and falsification conditions."
---

# Research

Eunomia Research publishes source-grounded systems briefs that develop a testable claim rather than summarize a news cycle. Each brief starts from a precise technical question, compares independent primary evidence, identifies tensions or alternative explanations, and states which architecture decision the conclusion would change.

These pages are reviewed research briefs, not peer-reviewed papers unless a page says otherwise. Claims remain bounded by their stated evidence, source cutoff, assumptions, and falsification conditions. When later evidence changes the conclusion, the stable page should be revised rather than replaced by a near-duplicate article.

## Current research

### [Parallel Agents Need a Commit Protocol: From Effect Serializability to Contract-Valid Execution](https://eunomia.dev/research/parallel-agent-effect-serializability/)

Parallel calls, worktrees, reducers, and serializable resource updates can still compose into a result that violates the user's task or stale authority. This brief audits current agent runtimes and proposes contract-valid effect serializability across code, APIs, budgets, approvals, and irreversible actions.

### [What Should an AI Agent Trace Keep? Observability Under a Fixed Evidence Budget](https://eunomia.dev/research/agent-trace-evidence-budget/)

AI agent traces can generate hundreds of system events around each model call while still omitting decisive state, authority, or provenance. This brief develops an evidence-portfolio architecture that combines representative sampling, anomaly capture, a causal flight recorder, and explicit outcome evidence.

## Publication standard

A research brief must provide more than a fluent literature survey. It should contain:

- a question that affects a real systems or engineering decision;
- independent primary evidence, including implementation or measurement evidence where available;
- a genuine tension, contradiction, or alternative explanation;
- a new synthesis that cannot be copied from any single source;
- explicit assumptions, applicability limits, uncertainty, and falsification conditions;
- a concrete experiment, artifact, or observation that could move the claim forward.

The [Blog](https://eunomia.dev/blog/) remains the archive for project writing, tutorials, releases, engineering explanations, and mature editorial articles. Research briefs use a separate section so their evolving evidentiary status remains visible without lowering the editorial contract of the Blog.
