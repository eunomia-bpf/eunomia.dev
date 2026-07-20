---
date: 2026-06-24
slug: agentpprof-semantic-flamegraph
description: AI agent traces hide budget hotspots inside thousands of prompts, and agentpprof uses semantic flamegraphs to group intent, tokens, time, files, and network.
---

# Where AI Agent Budgets Go: Semantic Flamegraphs for Agent Traces

A $3000 AI-agent bill is not a diagnosis. It does not tell you whether the money went to code review, debugging, documentation, tool retries, or continuation prompts that stretched one task across many turns. The operational question is not only how much the agent cost, but which recurring kinds of work are expensive enough to redesign.

[agentpprof](https://github.com/eunomia-bpf/agentsight) reads local agent trace history and aggregates prompts and tool calls by semantic intent into flamegraphs. In the resulting view, width represents token consumption, execution time, or operation count, so a team can see which categories dominate before drilling into the sessions behind them. It is part of the [AgentSight](https://github.com/eunomia-bpf/agentsight) project, which provides eBPF-based observability for AI agent behavior.

The widest bar is not automatically waste. A review-heavy profile may describe necessary work, repeated inspection of the same files, an overly broad context window, or all three. The profile gives the team a place to start asking which explanation is true.

<!-- more -->

## The Aggregation Problem

Most LLM observability views are organized around one execution. Timelines locate a failing span, span trees show call hierarchy, and waterfall charts reveal parallelism. They are useful for asking what happened in one trace. The cross-session question has a different data shape. After tens of thousands of calls, the reader needs stable categories that can merge repeated work and carry a weight such as tokens, elapsed time, or effect count.

Topic clustering and structured trace extraction can group similar inputs, but an input distribution is not yet a budget profile. Knowing that many prompts mention code does not show how much model budget review consumed, which tools those review prompts invoked, or which files they affected. agentpprof focuses on that weighted, cross-session aggregation problem.

CPU profilers solved a similar aggregation problem long ago. Flamegraphs compress millions of function calls into one chart, width representing time share. The stack indicates context, and repeated calls to the same function merge into wider bars. This works because function names are deterministic, so the same code path produces the same stack, and identical stacks can be directly merged.

Agent traces break this assumption. Prompts are natural language, which makes them non-deterministic, variable-length, multilingual, and often conversational. "Fix the bug" and "修一下这个 error" express the same intent but share no common string. If you use raw prompt text as frame labels, the flamegraph becomes too wide to read, with each prompt as an isolated bar, losing the point of aggregation. And raw prompts often contain sensitive information, making them unsuitable for sharing.

## Semantic Flamegraphs Restore Aggregation

agentpprof restores aggregation by mapping free-form prompts to short, stable semantic labels like `debug`, `review`, `paper`, or `docs`. Once tagged, prompts behave like function names, repeated activities merge, and the flamegraph becomes readable.

The value of flamegraphs is not just aggregation but also stack-based causal linking. Traditional CPU flamegraph stacks encode function call chains, so `main → parse → tokenize` means tokenize was called by parse, which was called by main. Semantic flamegraph stacks encode agent behavior causal chains, so `prompt:debug → call:llm/analysis → tool:bash → file:src/main.rs` means this file modification was triggered by bash, bash was decided by the LLM, and the LLM was responding to a debug-type prompt.

| | Traditional CPU Flamegraph | Semantic Flamegraph |
| --- | --- | --- |
| **Stack meaning** | Function call chain | prompt → LLM → tool → effect causal chain |
| **Aggregation** | Same function name merges | Same semantic tag merges |
| **Width meaning** | CPU time share | token / time / operation count share |
| **Question answered** | Where does the program spend CPU | Where does the agent spend budget by category |

This causal linking lets you trace back or drill down from any layer. From a modified file, you can trace back to the tool, LLM decision, and user intent that caused it. From a prompt category, you can see which LLM calls, tool executions, and system effects it triggered.

## Multiple Views, Different Questions

agentpprof exposes several projections over the same data, with each view answering a different question.

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

The preset is a demo starting point, not a production taxonomy. For a repeatable comparison, pass explicit `--session-file` inputs and replace the preset with version-controlled rules for the project. Open the SVG, identify the widest meaningful prompt category, then inspect the sessions underneath it before changing a workflow.

## Real Examples from AgentSight Development

The examples below were generated from AgentSight's own Claude Code development traces. They are descriptive project profiles, not a controlled benchmark. Their category names depend on the tagging rules used for this project, while the views demonstrate what can be inspected once the traces share stable labels.

### Tokens View Shows Where the Model Budget Went

![Tokens flamegraph](imgs/agentsight-tokens.svg)

The token distribution shows that code review (`prompt:review`) dominated the model budget, followed by git operations (`prompt:git`), code work (`prompt:code`), editing (`prompt:edit`), and debugging (`prompt:debug`). Through the stack, you can trace which LLM calls each prompt category triggered. Here `call:llm/usage` marks token statistics events, `call:llm/code` and `call:llm/test` mark code-related responses, `call:llm/tool` marks tool calls, and `call:llm/edit` marks modification responses.

### Time View Shows Where Wall-Clock Time Went

![Time flamegraph](imgs/agentsight-time.svg)

Wall-clock time distribution follows a similar pattern to token consumption. Review (`prompt:review`) leads, followed by git, edit, docs, and code prompts. Continuation prompts (`prompt:continue`) appear frequently, reflecting a workflow pattern where complex tasks required multiple follow-up exchanges. The `prompt:inspect` category captures quick look-at-this requests that are common in iterative development.

### Files View Shows Which Code Paths Were Touched

![Files flamegraph](imgs/agentsight-files.svg)

File access patterns show heavy activity in `collector/src/` (the Rust codebase) and `collector/Cargo.toml`, consistent with development work. External paths (`external/tmp`, `external/home`, `external/codex`) appear frequently, reflecting tool invocations that touch temporary files, home directory configs, and Codex session data. The flamegraph distinguishes between read and write effects, revealing the balance of inspection versus modification across both project and external paths.

### Network View Shows Which External Services Were Contacted

![Network flamegraph](imgs/agentsight-network.svg)

Network activity is sparse relative to file operations, confirming that most development work occurred locally. The contacted domains include `anthropic.com` for model inference, `crates.io` for Rust dependencies, `github.com` for version control, and various localhost ports for local development servers. Process chains visible in the upper frames show which tools initiated network requests, enabling attribution of network activity to specific agent actions.

## Stable Tags Are the Hard Part

The core technical challenge in semantic flamegraphs is mapping natural language prompts to stable, meaningful tags. This is fundamentally harder than CPU profiling, where function names are deterministic symbols. We have working solutions but not solved solutions, and we are explicit about the limitations.

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

No single approach handles all cases well. We currently provide three backends, each with different tradeoffs.

### Current Approaches

**Regex + Agent Iteration** is the production default. Rules like `prompt:debug='(?i)fix|error|bug|broken|为啥'` are pattern-matched against prompt text. The workflow is iterative: run agentpprof, observe unmatched samples, write rules, repeat until coverage exceeds 95%. This typically takes 5-10 rounds for a new project.

The advantage is operational predictability. It is deterministic, reproducible, fast, and free of external dependencies, and rules can be version-controlled and run in CI.

The cost is project-specific maintenance. Rules are brittle to prompt style changes and cannot handle semantic similarity, such as "fix the bug" versus "resolve the issue."

**LLM Tagger** uses local inference via llama.cpp. It constrains each result to a single tag and caches the output for reuse.

The advantage is semantic coverage. It handles semantic similarity and multilingual prompts without requiring rule writing.

The cost is stability and operational setup. The same prompt may receive different tags across runs, the local model must be configured, and tag quality depends on model capability. Cache the results when repeatability matters, then convert useful categories into deterministic rules.

**TF-IDF + K-Means Clustering** uses unsupervised clustering to discover natural groupings. It automatically selects cluster count (5-25) and generates tag names from cluster keywords.

The advantage is discovery. It needs no predefined categories and can reveal structure you did not anticipate.

The cost is interpretability. Cluster boundaries are arbitrary, tag names are keyword-derived rather than semantic, and the result still needs post-hoc interpretation.

### What We Do Not Know

Several fundamental questions remain open.

**Tag adequacy** remains unproven. Grammar-constrained output can keep labels syntactically valid, but that does not show that one-word tags preserve enough meaning for human decisions. "debug" might conflate bug fixing, error investigation, and performance debugging, each of which has different cost implications.

**Cross-project transfer** is unknown. Rules developed for one project may not transfer to another. A Rust systems project has different prompt patterns than a React frontend project. We do not yet know how much rule overlap exists across project types.

**Optimal granularity** has no principled answer yet. Should "code review" be one tag, or should it split into "review:style", "review:logic", "review:security"? Finer granularity preserves information but fragments the flamegraph.

**Multilingual normalization** remains difficult. "Fix the bug" and "修一下这个 bug" should probably get the same tag, but regex rules cannot express this. LLM taggers can, but with stability tradeoffs.

### Why We Ship Anyway

Despite these limitations, perfect tagging is not required to test whether the aggregation is useful. A profile can still reveal which labeled activities dominate, how token consumption is distributed, and which prompt categories trigger the most tool calls while leaving uncertain fragments explicitly unmatched.

The goal is not ground-truth classification but actionable visibility. If the flamegraph shows "review" consuming 40% of tokens, the exact boundary of what counts as "review" matters less than knowing that review-like activities are the dominant cost driver.

Current work focuses on three directions.
- LLM-assisted rule generation (model proposes rules from unmatched samples)
- Embedding-based similarity for multilingual normalization
- Human evaluation of tag adequacy (currently missing from our evidence base)

## Privacy by Default

Local agent histories can contain prompts, tool outputs, paths, commands, repository names, and model responses. agentpprof is conservative by default.

- SVG, pprof, and folded outputs contain stack labels and weights, not raw prompts or model responses.
- Absolute paths outside the selected project root are grouped into stable buckets such as `external/home`, `external/tmp`, `external/codex`, and `external/claude`.
- Private-looking domains are collapsed instead of exposing user-specific hostnames.

## Part of AgentSight

agentpprof is the offline profiling component of [AgentSight](https://github.com/eunomia-bpf/agentsight), an eBPF-based observability framework for monitoring AI agent behavior. While AgentSight provides live visibility through SSL/TLS interception and process monitoring, agentpprof provides aggregate analysis of already-recorded agent traces.

A typical workflow combines both.

1. Record agent activity with `sudo agentsight record -- claude`
2. Generate summary reports with `agentsight report`
3. Profile token consumption with `agentpprof --view tokens`
4. Audit file access patterns with `agentpprof --view files`
5. Check network destinations with `agentpprof --view network`

For installation and detailed usage, see the [AgentSight repository](https://github.com/eunomia-bpf/agentsight) and the [agentpprof documentation](https://github.com/eunomia-bpf/agentsight/blob/master/docs/agentpprof.md).

## From Visibility to Action Is the Harder Problem

Generating a flamegraph is the easy part. The harder question is what to do with it.

CPU profilers lead to clear actions such as finding the hot function, optimizing the algorithm, and reducing allocations. Agent cost profiles are different.

- You will not stop doing code review merely because it is the widest token category
- You will not skip debugging because it is expensive
- The flamegraph shows WHERE budget goes, not WHY it goes there or HOW to reduce it

Making the view actionable requires drilling deeper.

1. **Within-category analysis** asks why review is the widest token category. The cause might be repeated reviews of the same file, unnecessarily broad context windows, or verbose review prompts. The flamegraph shows the category, while understanding the cause requires examining individual sessions.

2. **Workflow pattern detection** looks for repeated interaction shapes. Frequent continuation prompts (`prompt:continue`) may indicate tasks that should be structured differently upfront, while high `prompt:unmatched` rates may indicate prompt styles that need standardization.

3. **Cross-session comparison** asks whether this month's token distribution differs from last month's, or whether a workflow change increased debugging costs. Trend analysis requires baseline comparison.

We are working on combining agentpprof with interaction analysis to produce reports that recommend specific changes, such as CLAUDE.md rules to prevent repeated file reviews, prompt templates to reduce context overhead, and workflow restructuring to minimize continuation churn.

## Current Limitations

**Agent coverage** is currently limited to Codex and Claude Code local traces. Gemini, Cursor, and other agents require parser extensions via the `agent-session` crate.

**Tagging** remains an open challenge, as discussed above. Project-specific rules are required, and we do not yet have evidence that one-word tags are semantically adequate.

**Validation** currently rests on mechanism evidence showing that the flamegraph correctly aggregates by tag, but we do not yet have user evidence that developers make better decisions with this view. The latter requires user studies we have not yet conducted.

**Cost attribution** depends on agent-reported usage, which may not reflect actual billing because of cached tokens, batch discounts, and model-specific pricing. The flamegraph shows relative distribution, not dollar amounts.

---

agentpprof is open source and part of the [AgentSight project](https://github.com/eunomia-bpf/agentsight). Contributions and feedback are welcome.

## References

- [AgentSight repository](https://github.com/eunomia-bpf/agentsight)
- [agentpprof documentation](https://github.com/eunomia-bpf/agentsight/blob/master/docs/agentpprof.md)
- [AgentSight: System-Level Observability for AI Agents Using eBPF](https://arxiv.org/abs/2508.02736)
- [Brendan Gregg, Flame Graphs](https://www.brendangregg.com/flamegraphs.html)
- [Go diagnostics: profiling](https://go.dev/doc/diagnostics#profiling)
