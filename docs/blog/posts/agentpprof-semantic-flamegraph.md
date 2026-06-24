---
date: 2026-06-24
slug: agentpprof-semantic-flamegraph
description: Your AI agent spent $3000 this month. Which activities consumed that budget? agentpprof applies the flamegraph paradigm to AI agent traces, mapping natural language prompts to semantic tags and aggregating them like a CPU profiler. This post explains why existing observability tools fail at budget attribution and how semantic flamegraphs restore aggregation for agent workloads.
---

# Profiling AI Agents with Semantic Flamegraphs: Where Did the Budget Go?

End of month, the bill shows the agent spent $3000. What types of work consumed that budget? How much went to code review, how much to debugging, how much to documentation? This question seems simple, but none of the existing agent observability tools can answer it directly.

[agentpprof](https://github.com/eunomia-bpf/agentsight) is a profiling tool built for exactly this question. It reads local agent trace history and aggregates prompts and tool calls by semantic intent into flamegraphs: width represents token consumption, execution time, or operation count. At a glance, you can see where the budget went by category. It is part of the [AgentSight](https://github.com/eunomia-bpf/agentsight) project, which provides eBPF-based observability for AI agent behavior.

<!-- more -->

## The Aggregation Problem

LLM observability platforms like LangSmith, Langfuse, and Phoenix can show token counts and latency for each call, but when you have 80,000 calls, they can only arrange them by timestamp into a timeline. You can inspect each one and see "this call used 500 tokens," but you cannot answer "how much did review tasks cost in total." These tools are designed for single-trace debugging: timeline views help you locate the failing span at 14:03, span trees show call hierarchy, waterfall charts reveal parallelism. They excel at answering "what happened" but for the question "where did the budget go by category," inspecting 80,000 spans one by one simply does not scale.

Datadog and Laminar are starting to move in the right direction with semantic classification. Datadog uses topic clustering to group user messages, Laminar uses Signals to extract structured events from traces. But their clustering primarily targets the distribution of user inputs, not "width represents budget share" aggregate views. You can see "30% of users asked about code," but not "code review consumed 40% of the token budget."

CPU profilers solved a similar aggregation problem long ago. Flamegraphs compress millions of function calls into one chart, width representing time share. The stack indicates context, and repeated calls to the same function merge into wider bars. This works because function names are deterministic: the same code path produces the same stack, and identical stacks can be directly merged.

Agent traces break this assumption. Prompts are natural language: non-deterministic, variable-length, multilingual, and often conversational. "Fix the bug" and "修一下这个 error" express the same intent but share no common string. If you use raw prompt text as frame labels, the flamegraph becomes too wide to read, with each prompt as an isolated bar, losing the point of aggregation. And raw prompts often contain sensitive information, making them unsuitable for sharing.

## Semantic Flamegraphs: Restoring Aggregation

agentpprof restores aggregation by introducing semantic tagging: mapping free-form prompts to short, stable labels like `debug`, `review`, `paper`, or `docs`. Once tagged, prompts behave like function names, repeated activities merge, and the flamegraph becomes readable.

The value of flamegraphs is not just aggregation but also stack-based causal linking. Traditional CPU flamegraph stacks are function call chains: `main → parse → tokenize` means tokenize was called by parse, which was called by main. Semantic flamegraph stacks are agent behavior causal chains: `prompt:debug → call:llm/analysis → tool:bash → file:src/main.rs` means this file modification was triggered by bash, bash was decided by the LLM, and the LLM was responding to a debug-type prompt.

| | Traditional CPU Flamegraph | Semantic Flamegraph |
| --- | --- | --- |
| **Stack meaning** | Function call chain | prompt → LLM → tool → effect causal chain |
| **Aggregation** | Same function name merges | Same semantic tag merges |
| **Width meaning** | CPU time share | token / time / operation count share |
| **Question answered** | Where does the program spend CPU | Where does the agent spend budget by category |

This causal linking lets you trace back or drill down from any layer: from a file being modified, trace back to which tool, which LLM decision, which user intent caused it; or from a prompt category, see what LLM calls, tool executions, and system effects it triggered.

## Multiple Views, Different Questions

agentpprof exposes several projections over the same data, each answering a different question:

| View | Width means | Primary question |
| --- | ---: | --- |
| `tokens` | reported token count (input/output/cache) | Which prompts consumed the most model budget? |
| `time` | duration in seconds | How long did each prompt/activity take? |
| `files` | file/path effect count | Which prompts touched which parts of the repository? |
| `network` | network/domain effect count | Which prompts contacted which domains? |

Start with `tokens` to find cost hotspots, use `time` to trace where wall-clock time went, and use `files` and `network` for security audits.

## Real Examples from AgentSight Development

The examples below were generated from AgentSight's own development traces (Claude Code). They demonstrate what insights each view provides.

### Tokens View: Where Did the Model Budget Go?

![Tokens flamegraph](https://github.com/eunomia-bpf/agentsight/raw/master/docs/flamegraph-example/agentsight-tokens.svg)

The token distribution shows that code review (`prompt:review`) dominated the model budget, followed by git operations (`prompt:git`), code work (`prompt:code`), editing (`prompt:edit`), and debugging (`prompt:debug`). Through the stack, you can trace which LLM calls each prompt category triggered: `call:llm/usage` for token statistics events, `call:llm/code` and `call:llm/test` for code-related responses, `call:llm/tool` for tool calls, and `call:llm/edit` for modification responses.

### Time View: Where Did Wall-Clock Time Go?

![Time flamegraph](https://github.com/eunomia-bpf/agentsight/raw/master/docs/flamegraph-example/agentsight-time.svg)

Wall-clock time distribution follows a similar pattern to token consumption: review (`prompt:review`) leads, followed by git, edit, docs, and code prompts. Continuation prompts (`prompt:continue`) appear frequently, reflecting a workflow pattern where complex tasks required multiple follow-up exchanges. The `prompt:inspect` category captures quick look-at-this requests that are common in iterative development.

### Files View: Which Parts of the Codebase Were Touched?

![Files flamegraph](https://github.com/eunomia-bpf/agentsight/raw/master/docs/flamegraph-example/agentsight-files.svg)

File access patterns show heavy activity in `collector/src/` (the Rust codebase) and `collector/Cargo.toml`, consistent with development work. External paths (`external/tmp`, `external/home`, `external/codex`) appear frequently, reflecting tool invocations that touch temporary files, home directory configs, and Codex session data. The flamegraph distinguishes between read and write effects, revealing the balance of inspection versus modification across both project and external paths.

### Network View: Which External Services Were Contacted?

![Network flamegraph](https://github.com/eunomia-bpf/agentsight/raw/master/docs/flamegraph-example/agentsight-network.svg)

Network activity is sparse relative to file operations, confirming that most development work occurred locally. The contacted domains include `anthropic.com` for model inference, `crates.io` for Rust dependencies, `github.com` for version control, and various localhost ports for local development servers. Process chains visible in the upper frames show which tools initiated network requests, enabling attribution of network activity to specific agent actions.

## The Tagging Challenge

Mapping natural language prompts to stable semantic tags is not trivial. Prompts in a single project may mix languages ("fix the 编译 error"), range from single characters ("嗯", "ok") to long paragraphs, and include many fragments that make no sense in isolation ("continue", "ok", system-generated context restoration messages).

agentpprof provides a pluggable tagger framework with multiple backends:

| Backend | Approach | Best for |
| --- | --- | --- |
| Regex + Agent iteration | Pattern matching, rules iteratively refined by AI agent | Production, CI, reproducible analysis |
| LLM tagger | Local LLM inference via llama.cpp | Complex prompts, initial rule discovery |
| Python clustering | TF-IDF + K-Means unsupervised clustering | Exploratory analysis, finding natural groupings |

The recommended workflow is to have an AI agent observe actual prompt samples and iteratively refine regex rules until the unmatched rate drops below 5%. This iteration typically takes 5-10 rounds, and the final rule set is deterministic and reproducible, suitable for version control and CI use.

By default there are no built-in rules, and all prompts are marked `unmatched`. This is an intentional design choice: generic rules are unlikely to match your project's actual prompt distribution, and blindly applying them produces misleading aggregation.

## Privacy by Default

Local agent histories can contain prompts, tool outputs, paths, commands, repository names, and model responses. agentpprof is conservative by default:

- SVG, pprof, and folded outputs contain stack labels and weights, not raw prompts or model responses.
- Absolute paths outside the selected project root are grouped into stable buckets such as `external/home`, `external/tmp`, `external/codex`, and `external/claude`.
- Private-looking domains are collapsed instead of exposing user-specific hostnames.

## Part of AgentSight

agentpprof is the offline profiling component of [AgentSight](https://github.com/eunomia-bpf/agentsight), an eBPF-based observability framework for monitoring AI agent behavior. While AgentSight provides live visibility through SSL/TLS interception and process monitoring, agentpprof provides aggregate analysis of already-recorded agent traces.

A typical workflow combines both:

1. Record agent activity with `agentsight record`
2. Generate summary reports with `agentsight report`
3. Profile token consumption with `agentpprof --view tokens`
4. Audit file access patterns with `agentpprof --view files`
5. Check network destinations with `agentpprof --view network`

For installation and detailed usage, see the [AgentSight repository](https://github.com/eunomia-bpf/agentsight) and the [agentpprof documentation](https://github.com/eunomia-bpf/agentsight/blob/master/docs/agentpprof.md).

## Limitations and Future Work

agentpprof currently reads Codex and Claude Code local trace files. Other agents can be added via the `agent-session` parser. The semantic tagging approach requires project-specific rule development, and we are exploring ways to automate this through LLM-assisted rule generation and clustering-based discovery.

The broader question is whether semantic flamegraphs lead to actionable insights. Knowing "code review consumed 40% of tokens" is interesting, but what do you do with that information? We are working on combining agentpprof with interaction analysis to produce reports that not only show where budget went, but also recommend specific workflow or CLAUDE.md changes to improve efficiency.

---

agentpprof is open source and part of the [AgentSight project](https://github.com/eunomia-bpf/agentsight). Contributions and feedback are welcome.
