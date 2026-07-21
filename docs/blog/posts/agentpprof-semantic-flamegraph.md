---
date: 2026-06-24
slug: agentpprof-semantic-flamegraph
description: AI agent traces hide budget hotspots inside thousands of prompts, and agentpprof uses semantic flamegraphs to group intent, tokens, time, files, and network.
---

# Where AI Agent Budgets Go: Semantic Flamegraphs for Agent Traces

A $3000 AI-agent bill is not a diagnosis. It does not tell you whether the money went to code review, debugging, documentation, tool retries, or continuation prompts that stretched one task across many turns. The operational question is which recurring kinds of work are expensive enough to redesign.

[agentpprof](https://github.com/eunomia-bpf/agentsight) reads local agent trace history and aggregates prompts and tool calls by semantic intent into flamegraphs. Width represents token consumption, execution time, or operation count, so a team can see which categories dominate before drilling into specific sessions. It is part of the [AgentSight](https://github.com/eunomia-bpf/agentsight) project, which provides eBPF-based observability for AI agent behavior.

The widest bar is not automatically waste. A review-heavy profile may reflect necessary work, repeated inspection of the same files, an overly broad context window, or all three. The profile gives the team a starting point for investigation.

<!-- more -->

## The Aggregation Problem

Most LLM observability views are organized around a single execution. Timelines locate a failing span, span trees show call hierarchy, and waterfall charts reveal parallelism. These views answer what happened in one trace. The cross-session question has a different shape: after tens of thousands of calls, the reader needs stable categories that merge repeated work and carry a weight such as tokens, elapsed time, or effect count.

Topic clustering and structured trace extraction can group similar inputs, but an input distribution is not a budget profile. Knowing that many prompts mention code does not show how much model budget review consumed, which tools those prompts invoked, or which files they affected. agentpprof focuses on this weighted, cross-session aggregation problem.

CPU profilers solved a similar aggregation problem long ago. Flamegraphs compress millions of function calls into one chart, width representing time share. The stack indicates context, and repeated calls to the same function merge into wider bars. This works because function names are deterministic: the same code path produces the same stack, and identical stacks merge directly.

Agent traces break this assumption. Prompts are natural language: non-deterministic, variable-length, multilingual, and often conversational. "Fix the bug" and "修一下这个 error" express the same intent but share no common string. Using raw prompt text as frame labels produces an unreadable flamegraph where each prompt is an isolated bar, defeating aggregation. Raw prompts also often contain sensitive information, making them unsuitable for sharing.

## Semantic Flamegraphs Restore Aggregation

agentpprof restores aggregation by mapping free-form prompts to short, stable semantic labels like `debug`, `review`, `paper`, or `docs`. Once tagged, prompts behave like function names: repeated activities merge and the flamegraph becomes readable.

Flamegraphs provide more than aggregation: they encode causal chains. In a CPU flamegraph, `main → parse → tokenize` means tokenize was called by parse, which was called by main. In a semantic flamegraph, `prompt:debug → call:llm/analysis → tool:bash → file:src/main.rs` means this file modification was triggered by bash, bash was decided by the LLM, and the LLM was responding to a debug-type prompt.

| | Traditional CPU Flamegraph | Semantic Flamegraph |
| --- | --- | --- |
| **Stack meaning** | Function call chain | prompt → LLM → tool → effect causal chain |
| **Aggregation** | Same function name merges | Same semantic tag merges |
| **Width meaning** | CPU time share | token / time / operation count share |
| **Question answered** | Where does the program spend CPU | Where does the agent spend budget by category |

This structure lets you trace in either direction. From a modified file, you can trace back to the tool, LLM decision, and user intent that caused it. From a prompt category, you can see which LLM calls, tool executions, and system effects it triggered.

## Multiple Views, Different Questions

agentpprof exposes several projections over the same data, each answering a different question.

| View | Width means | Primary question |
| --- | ---: | --- |
| `tokens` | reported token count (input/output/cache) | Which prompts consumed the most model budget? |
| `time` | duration in seconds | How long did each prompt/activity take? |
| `files` | file/path effect count | Which prompts touched which parts of the repository? |
| `network` | network/domain effect count | Which prompts contacted which domains? |

Start with `tokens` to find cost hotspots, use `time` to trace where wall-clock time went, and use `files` and `network` for security audits.

### Start with One Repeatable Profile

For a quick browser-openable profile of recent Codex or Claude Code traces associated with the current repository, run:

```bash
agentpprof --project-root . --view tokens --tagger regex --preset -o tokens.svg
```

The preset is a demo starting point, not a production taxonomy. For repeatable comparison, pass explicit `--session-file` inputs and replace the preset with version-controlled rules. Open the SVG, identify the widest meaningful prompt category, then inspect the underlying sessions before changing a workflow.

## Real Examples from AgentSight Development

The examples below come from AgentSight's own Claude Code development traces. They are descriptive profiles, not controlled benchmarks. Category names depend on the tagging rules used for this project; the views demonstrate what becomes inspectable once traces share stable labels.

### Tokens View Shows Where the Model Budget Went

![Tokens flamegraph](imgs/agentsight-tokens.svg)

Code review (`prompt:review`) dominated the model budget, followed by git operations (`prompt:git`), code work (`prompt:code`), editing (`prompt:edit`), and debugging (`prompt:debug`). The stack shows which LLM calls each prompt category triggered: `call:llm/usage` marks token statistics events, `call:llm/code` and `call:llm/test` mark code-related responses, `call:llm/tool` marks tool calls, and `call:llm/edit` marks modification responses.

### Time View Shows Where Wall-Clock Time Went

![Time flamegraph](imgs/agentsight-time.svg)

Wall-clock time follows a similar pattern to token consumption. Review (`prompt:review`) leads, followed by git, edit, docs, and code prompts. Continuation prompts (`prompt:continue`) appear frequently, reflecting complex tasks that required multiple follow-up exchanges. The `prompt:inspect` category captures quick look-at-this requests common in iterative development.

### Files View Shows Which Code Paths Were Touched

![Files flamegraph](imgs/agentsight-files.svg)

File access patterns show heavy activity in `collector/src/` (the Rust codebase) and `collector/Cargo.toml`, consistent with active development. External paths (`external/tmp`, `external/home`, `external/codex`) appear frequently, reflecting tool invocations that touch temporary files, home directory configs, and Codex session data. The flamegraph distinguishes read and write effects, revealing the balance of inspection versus modification across project and external paths.

### Network View Shows Which External Services Were Contacted

![Network flamegraph](imgs/agentsight-network.svg)

Network activity is sparse relative to file operations, confirming that most development work occurred locally. Contacted domains include `anthropic.com` for model inference, `crates.io` for Rust dependencies, `github.com` for version control, and various localhost ports for local servers. Process chains visible in the upper frames show which tools initiated network requests, enabling attribution to specific agent actions.

## Stable Tags Are the Hard Part

The core technical challenge in semantic flamegraphs is mapping natural language prompts to stable, meaningful tags. This is fundamentally harder than CPU profiling, where function names are deterministic symbols. The project has working solutions but not solved ones, and the limitations are worth stating explicitly.

### Why Tagging Is Hard

Consider real prompts from a development session.

```
"fix the 编译 error"          # Mixed language
"嗯"                          # Single character confirmation
"ok"                          # Ambiguous intent
"继续"                        # Context-dependent
"[Session continued...]"      # System-generated
"看看 collector/src/main.rs"  # Inspection request
"为啥 cargo test 失败了"       # Debug question
```

These prompts exhibit properties that break naive classification.

1. **Multilingual mixing** appears when English and Chinese share the same prompt, sometimes in the same sentence
2. **Extreme length variance** ranges from 1 character to multi-paragraph context restorations
3. **Context dependence** makes "继续" (continue) meaningless without knowing what preceded it
4. **Implicit intent** means "嗯" could be confirmation, acknowledgment, or thinking pause
5. **System noise** adds auto-generated session continuations, tool outputs, and error messages

No single approach handles all cases well. agentpprof provides three backends, each with different tradeoffs.

### Current Approaches

**Regex + iteration** is the production default. Rules like `prompt:debug='(?i)fix|error|bug|broken|为啥'` are pattern-matched against prompt text. The workflow is iterative: run agentpprof, observe unmatched samples, write rules, repeat until coverage exceeds 95%. This typically takes 5-10 rounds for a new project.

The advantage is operational predictability: deterministic, reproducible, fast, free of external dependencies, and rules can be version-controlled and run in CI.

The cost is project-specific maintenance. Rules are brittle to prompt style changes and cannot handle semantic similarity ("fix the bug" versus "resolve the issue").

**LLM tagger** uses local inference via llama.cpp, constraining each result to a single tag and caching output for reuse.

The advantage is semantic coverage: it handles similarity and multilingual prompts without rule writing.

The cost is stability and operational setup. The same prompt may receive different tags across runs, the local model must be configured, and tag quality depends on model capability. Cache results when repeatability matters, then convert useful categories into deterministic rules.

**TF-IDF + K-Means clustering** uses unsupervised methods to discover natural groupings, automatically selecting cluster count (5-25) and generating tag names from cluster keywords.

The advantage is discovery: no predefined categories required, and it can reveal unanticipated structure.

The cost is interpretability: cluster boundaries are arbitrary, tag names are keyword-derived rather than semantic, and results still need post-hoc interpretation.

### Open Questions

Several fundamental questions remain.

**Tag adequacy** is unproven. Grammar-constrained output keeps labels syntactically valid, but that does not show one-word tags preserve enough meaning for human decisions. "debug" might conflate bug fixing, error investigation, and performance debugging, each with different cost implications.

**Cross-project transfer** is unknown. Rules developed for one project may not transfer to another. A Rust systems project has different prompt patterns than a React frontend project, and how much rule overlap exists across project types is not yet known.

**Optimal granularity** has no principled answer. Should "code review" be one tag or split into `review:style`, `review:logic`, `review:security`? Finer granularity preserves information but fragments the flamegraph.

**Multilingual normalization** remains difficult. "Fix the bug" and "修一下这个 bug" should probably receive the same tag, but regex rules cannot express this. LLM taggers can, but with stability tradeoffs.

## Part of AgentSight

agentpprof is the offline profiling component of [AgentSight](https://github.com/eunomia-bpf/agentsight), an eBPF-based observability framework for AI agent behavior. AgentSight provides live visibility through SSL/TLS interception and process monitoring; agentpprof provides aggregate analysis of already-recorded traces.

A typical workflow combines both:

1. Record agent activity with `sudo agentsight record -- claude`
2. Generate summary reports with `agentsight report`
3. Profile token consumption with `agentpprof --view tokens`
4. Audit file access patterns with `agentpprof --view files`
5. Check network destinations with `agentpprof --view network`

For installation and detailed usage, see the [AgentSight repository](https://github.com/eunomia-bpf/agentsight) and the [agentpprof documentation](https://github.com/eunomia-bpf/agentsight/blob/master/docs/agentpprof.md).

## From Visibility to Action Is the Harder Problem

Generating a flamegraph is the easy part. The harder question is what to do with it.

CPU profilers lead to clear actions: find the hot function, optimize the algorithm, reduce allocations. Agent cost profiles are different.

- You will not stop doing code review merely because it is the widest token category
- You will not skip debugging because it is expensive
- The flamegraph shows WHERE budget goes, not WHY it goes there or HOW to reduce it

Making the view actionable requires drilling deeper:

1. **Within-category analysis** asks why review is the widest token category. The cause might be repeated reviews of the same file, unnecessarily broad context windows, or verbose prompts. The flamegraph shows the category; understanding the cause requires examining individual sessions.

2. **Workflow pattern detection** looks for repeated interaction shapes. Frequent continuation prompts (`prompt:continue`) may indicate tasks that should be structured differently upfront; high `prompt:unmatched` rates may indicate prompt styles that need standardization.

3. **Cross-session comparison** asks whether this month's token distribution differs from last month's, or whether a workflow change increased debugging costs. Trend analysis requires baselines.

Work is underway to combine agentpprof with interaction analysis to produce reports that recommend specific changes: CLAUDE.md rules to prevent repeated file reviews, prompt templates to reduce context overhead, and workflow restructuring to minimize continuation churn.

## Current Limitations

**Agent coverage** is limited to Codex and Claude Code local traces. Gemini, Cursor, and other agents require parser extensions via the `agent-session` crate.

**Tagging** remains an open challenge. Project-specific rules are required, and there is not yet evidence that one-word tags are semantically adequate.

**Validation** rests on mechanism evidence showing that the flamegraph correctly aggregates by tag, but there is no user evidence yet that developers make better decisions with this view. That requires user studies not yet conducted.

**Cost attribution** depends on agent-reported usage, which may not reflect actual billing due to cached tokens, batch discounts, and model-specific pricing. The flamegraph shows relative distribution, not dollar amounts.

---

agentpprof is open source and part of the [AgentSight project](https://github.com/eunomia-bpf/agentsight). Contributions and feedback are welcome.

## References

- [AgentSight repository](https://github.com/eunomia-bpf/agentsight)
- [agentpprof documentation](https://github.com/eunomia-bpf/agentsight/blob/master/docs/agentpprof.md)
- [AgentSight: System-Level Observability for AI Agents Using eBPF](https://arxiv.org/abs/2508.02736)
- [Brendan Gregg, Flame Graphs](https://www.brendangregg.com/flamegraphs.html)
- [Go diagnostics: profiling](https://go.dev/doc/diagnostics#profiling)
