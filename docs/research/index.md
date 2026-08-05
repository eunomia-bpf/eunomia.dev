---
title: Research
description: "Source-grounded systems research briefs written for human technical readers, with testable claims, architecture consequences, explicit scope, and falsification conditions."
---

# Research

Eunomia Research publishes source-grounded systems briefs for engineers, researchers, and maintainers. Each brief starts from a concrete technical problem, develops a testable claim, compares independent primary evidence, and explains which architecture or engineering decision should change.

These pages are reviewed research briefs, not peer-reviewed papers unless a page says otherwise. Claims remain bounded by their evidence, source cutoff, assumptions, and falsification conditions. When later evidence changes a conclusion, the stable page should be revised instead of being replaced by a near-duplicate article.

## Current research

### [When Several AI Agents Work at Once, Who Makes Sure the Final Result Is Right?](https://eunomia.dev/research/parallel-agent-effect-serializability/)

Worktrees, sandboxes, and parallel tool calls can isolate workers while still producing a wrong combined outcome. This brief uses code changes, shared budgets, approvals, and irreversible actions to explain why parallel agents need one validation and commit step before their effects become real.

### [What Should an AI Agent Trace Keep? Observability Under a Fixed Evidence Budget](https://eunomia.dev/research/agent-trace-evidence-budget/)

AI agent traces can generate hundreds of system events around each model call while still omitting decisive state, authority, or provenance. This brief develops an evidence-portfolio architecture that combines representative sampling, anomaly capture, a causal flight recorder, and explicit outcome evidence.

## Publication standard

A research brief must provide more than a fluent literature survey. It should contain:

- a question that affects a real systems or engineering decision;
- an opening that a technically qualified reader can understand without reading the source corpus;
- independent primary evidence, including implementation or measurement evidence where available;
- a genuine tension, contradiction, or alternative explanation;
- a new synthesis that cannot be copied from any single source;
- explicit assumptions, applicability limits, uncertainty, and falsification conditions;
- a concrete experiment, artifact, or observation that could move the claim forward.

The [Blog](https://eunomia.dev/blog/) remains the archive for project writing, tutorials, releases, engineering explanations, and mature editorial articles. Research briefs use a separate section so their evolving evidentiary status remains visible without lowering the editorial contract of the Blog.
