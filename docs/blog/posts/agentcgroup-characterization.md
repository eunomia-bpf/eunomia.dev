---
date: 2026-02-17
description: AgentCgroup characterizes resource bursts in AI coding agents and uses eBPF, sched_ext, and cgroup v2 to enforce tool-call-granularity CPU and memory control without discarding agent state.
---

# AgentCgroup: What Happens When AI Coding Agents Meet OS Resources?

An AI coding agent spends several quiet minutes reading and editing files, then launches `pytest`. Memory can rise by hundreds of megabytes in a second as the test process loads dependencies, only to fall again when the command exits. A container-level controller sees one workload cross a limit. It cannot tell that the burst belongs to a short-lived tool process while the long-lived agent runtime holds the conversation, partial diagnosis, and edits that make the task recoverable.

We measured how often that pattern occurs by running 144 SWE-rebench tasks with two LLM backends. OS work such as container setup and tool execution consumes 56% to 74% of end-to-end latency, and memory reaches 15.4 times its average level with sub-second changes up to 3 GB/s. Even two Bash calls can differ by 13.7 times in memory demand because one runs `git status` while another launches a test suite. Token count offers almost no warning of the next peak: its correlation with peak memory is −0.14 for Haiku and +0.02 for GLM.

Static allocation reacts badly to this combination. Reserving the observed peak wastes up to 93% of provisioned capacity during quiet phases, while killing the container at the peak discards minutes of accumulated, non-reproducible agent state. The [AgentCgroup paper](https://arxiv.org/abs/2602.09345) starts from these measurements and develops an eBPF-based controller that can respond at tool-call granularity.

<!-- more -->

> Paper: [*AgentCgroup: Understanding and Controlling OS Resources of AI Agents*](https://arxiv.org/abs/2602.09345)
>
> GitHub: [github.com/eunomia-bpf/agentcgroup](https://github.com/eunomia-bpf/agentcgroup)

## Follow the Agent from Edit to Test

To follow the `pytest` burst back to the process that caused it, we instrumented [Claude Code](https://docs.anthropic.com/en/docs/claude-code) while it ran 144 software engineering tasks from [SWE-rebench](https://github.com/swe-bench/SWE-ReB). The traces preserve tool-call boundaries alongside one-second CPU and memory samples, so a rise in container memory can be matched to the command running at that moment. We used two LLM backends:

- Claude Haiku 4.5 (cloud API): LLM inference runs on Anthropic's cloud; the container only runs the agent framework and tool calls.
- GLM-4.7-Flash (local GPU): LLM inference runs on a local GPU; everything happens on the same machine.

Both use the exact same agent framework (Claude Code, Node.js-based). The only difference is the underlying model and where inference happens. This lets us isolate the effect of model choice on container-level resource dynamics.

### How We Captured the Burst

| Component | Details |
|-----------|---------|
| Platform | Intel Core Ultra 9 285K (24 cores, 5.8 GHz), 128 GB DDR5, Ubuntu 24.04.3 LTS |
| Kernel | Linux 6.15.11 with cgroup v2 enabled |
| Container Runtime | Podman (rootless, isolated containers) |
| Agent Framework | Claude Code (Node.js) |
| Models | Haiku 4.5 (cloud API, 33 tasks) + GLM-4.7-Flash (local GPU, 111 tasks) |
| Benchmark | SWE-rebench (real GitHub issues from open-source projects) |
| Monitoring | 1-second interval CPU/memory sampling via `podman stats` |
| Tracing | Tool call boundaries (type, start/end timestamps) from agent execution traces |

The tasks cover CLI tools, build systems, scientific and medical code, data processing, and web projects at three difficulty levels. We imposed no resource limit during characterization because the trace must show the unconstrained peak before a controller can decide how to handle it.

## The Tool Call Owns the Spike

The opening `pytest` process is not an edge case hidden beneath model inference. Across both backends, most end-to-end time is spent starting the environment and running tools, which puts the operating system directly on the critical path.

### Most Time Is Spent Outside the Model

Contrary to the intuition that "the LLM is the bottleneck," our measurements show that LLM reasoning accounts for only 26–44% of end-to-end task latency. The remaining 56–74% is OS-level overhead:

| Latency Component | Haiku | GLM |
|-------------------|-------|-----|
| Container + agent initialization | 47.7% | 31.0% |
| Tool execution | 25.9% | 25.5% |
| LLM reasoning | 26.4% | 43.5% |

Container startup alone averages 26.5 seconds (median 23.0s, max 97s), driven by Podman's user-namespace ID remapping of overlay layers that scales with image size. Since SWE-rebench container images range from 2.9 GB to 17.3 GB (median 3.5 GB), roughly 7x larger than typical microservice images and 70x larger than serverless functions, this initialization overhead is substantial.

Each task then remains alive for 5 to 11 minutes, keeping LLM context, code edits, and tool results in one stateful process. Bash and sub-agent calls consume more than 90% of tool time, although Haiku offloads more work to sub-agents while GLM performs almost everything through local Bash. A controller therefore has to preserve the long-lived runtime while accounting for the very different commands launched beneath the same Bash tool.

### Follow Bash One Layer Deeper

The tool name still does not reveal the resource demand. Breaking Bash calls down by command semantics shows where the time goes:

| Bash Category | % of Bash Time (Haiku) | % of Bash Time (GLM) |
|---------------|------------------------|----------------------|
| Test execution (pytest, unittest, etc.) | 72.9% | 43.7% |
| Python snippets | n/a | 26.9% |
| Package installation | 10.8% | 10.1% |
| Git operations | <5% | <5% |
| File exploration | <5% | <5% |

Test execution overwhelmingly dominates, and as shown in the next section, it is also the most resource-intensive category.

The calls also move through a recognizable sequence when execution is divided into ten equal phases:

- Understand phase (0–30%): Read operations dominate (code exploration)
- Modify phase (30–70%): Edit operations are distributed throughout; Bash begins rising
- Verify phase (40–100%): Bash peaks (repeated test execution, debugging)

This phase signature mirrors the "understand, modify, verify" workflow of human software engineering, providing a basis for phase-aware resource control.

## The Burst Ends Before a Container Controller Can Adapt

On our 24-core platform, CPU remains below 36% even at the concurrency limit imposed by memory. Peak memory reaches 2 to 4 GB per task, so allocating every container at peak consumes the machine while much of its CPU capacity remains idle. The trace explains why: one stable layer needs protection, while a second layer appears only when tools run.

### Protect the 185 MB Stateful Baseline

Agent memory has a two-layer structure. The opening `pytest` run belongs to the transient layer, while the conversation and framework live in the stable layer beneath it.

Layer 1, the framework baseline (~185 MB): The Node.js runtime, V8 JIT cache, and agent framework state maintain a stable, incompressible memory floor throughout execution, even during LLM reasoning phases with zero tool activity. Across all 144 tasks, early-execution memory averages 183 MB (Haiku) and 188 MB (GLM).

Layer 2, tool-call bursts (500 MB to 2+ GB): Test execution, dependency installation, and data processing operations create transient spikes that last only 1–2 seconds before collapsing back to the ~185 MB baseline.

When we normalize and aggregate memory traces across all 144 tasks by execution progress, the pattern is clear: the first half of execution stays at a stable 185–200 MB baseline, while the second half shows increasing variance with large spikes, corresponding to the Bash-intensive verify phase.

In a multi-tenant deployment, 64 concurrent instances require ~12 GB just for the framework baseline alone. The tool-call bursts layered on top are the real resource management challenge, and they require different treatment from the stable baseline.

Annotating each sample with its active tool call shows that memory bursts concentrate around tools 1.9 times more often than their share of execution time would suggest. CPU bursts are more dispersed, especially when local inference adds background CPU load, so CPU and memory need separate control signals.

### A Static Limit Has No Good Setting

The tool-driven memory layer changes faster than a container-level policy expects:

- Maximum memory change rate: 3 GB/second
- Maximum CPU change rate: >50%/second
- Burst duration: typically 1–2 seconds

The highest case we observed, a pydicom bioinformatics task (Medical_Bio_Hard), reached 4060 MB peak versus 264 MB average, a 15.4x peak-to-average ratio. This 4 GB spike lasted approximately 1–2 seconds before falling back to the 230 MB baseline.

Allocating the pydicom container at its 4060 MB peak leaves 93% of that memory unused during the typical 264 MB phase. Setting the limit near the average lets a one-second tool burst trigger an OOM kill and discard the protected baseline. The policy needs to distinguish the command that created the temporary layer, so we compared memory spikes inside Bash:

| Bash Category | P95 Memory Spike (Haiku) | P95 Memory Spike (GLM) | Avg CPU Spike |
|---------------|--------------------------|------------------------|---------------|
| Test execution (pytest, etc.) | 518 MB | 234 MB | +3.2% |
| Package installation | 233 MB | n/a | moderate |
| Git operations | 13.5 MB | n/a | minimal |
| File exploration | 4.5 MB | n/a | minimal |

Medical and bioinformatics commands average a 4 GB peak, while web and network commands average 291 MB. A Bash-level budget still groups `ls`, `git status`, and `pytest` together, so the useful boundary lies at the actual command process. CPU cannot stand in for memory at that boundary either: their correlation ranges from -0.84 to +0.50 across tasks.

## The Next Pytest Run Does Not Look Like the Last One

A static limit might still work if yesterday's trace predicted tomorrow's peak. Repeating the exact same task, `iterative/dvc#777`, shows why the agent cannot rely on that history:

| Run | Execution Time | Solution Strategy |
|-----|---------------|-------------------|
| 1 | 402 seconds | Strategy A (different file modifications) |
| 2 | 222 seconds | Strategy B (different approach) |
| 3 | 259 seconds | Strategy C (different file count) |

That is a 1.8x variance in execution time, with completely different solution strategies each time. This non-determinism stems from LLM reasoning randomness and decision-path diversity: the agent may choose entirely different code modifications, tool sequences, and debugging approaches on each run.

The model's own activity offers little earlier warning. Correlations between LLM-observable proxies and resource consumption remain weak:

| Proxy to Target | Haiku (r) | GLM (r) |
|-----------------|-----------|---------|
| Output tokens to peak memory | −0.14 | +0.02 |
| Conversation rounds to execution time | +0.57 | +0.82 |
| Conversation rounds to peak memory | +0.02 | +0.11 |

Output token count shows essentially zero correlation with peak memory. Even conversation rounds, which moderately predict execution time, are useless for predicting memory. Resource consumption is driven by what tools execute (e.g., pytest vs. file read), not by the scale of LLM reasoning. This means that even if one can predict how much an agent will "think," one still cannot predict how much memory it will need.

The familiar edit, test, fail, and retry loop adds another source of drift:

| Metric | Haiku | GLM |
|--------|-------|-----|
| Tasks with retry loops (3+ consecutive identical Bash calls) | 85% (28/33) | 97% (108/111) |
| Average retry groups per task | n/a | 3.9 |
| Maximum consecutive retries | n/a | 56 |
| Execution time consumed by retries | 7.4% | 20.5% |

The "execute test, observe failure, modify code, re-test" iteration loop is the agent's behavioral signature. Each retry retains prior memory context without cleanup, leading to progressive memory accumulation, up to 502 MB of unreleased memory in the worst case we observed. This means memory limits that were adequate early in execution may trigger OOM kills later as retries accumulate.

Across all tasks, peak memory spans 197 MB to 4 GB, a 20x range under the same agent framework. Model choice changes the profile again. The controller therefore needs the current command boundary and current pressure; a percentile from prior runs cannot substitute for either signal.

## Give the Tool Call Its Own Resource Domain

Return to the `pytest` burst. The 185 MB agent runtime and its test child currently share one container limit, even though the runtime contains expensive state and the child owns the temporary allocation. AgentCgroup maps the agent workload to a cgroup v2 node and places each tool call in a child node. A `git status` process and a `pytest` process can then receive different constraints while both remain inside the workload's total budget.

This hierarchy also changes recovery. Crossing a soft limit can freeze or throttle the tool subtree while the parent agent remains alive. If termination becomes necessary, cgroup v2 can kill that subtree atomically, preserving the conversation and edits held by the runtime.

## Move the Response into the Kernel

The command boundary solves granularity, but a one-second, 3 GB/s burst still demands a fast response. AgentCgroup executes control logic at kernel cgroup enforcement points through eBPF, removing the user-space signal, decision, and write-back loop:

- On CPU, `sched_ext` maintains per-workload and per-tool-call metadata in BPF maps, prioritizing latency-sensitive tool calls with automatic fail-safe reversion on errors.
- On memory, `memcg_bpf_ops` hooks implement custom throttling delays when a cgroup breaches its soft limit (`memory.high`), with `memory.max` as the hard limit.

The same in-kernel observations replace prediction with current evidence. AgentCgroup traces process creation and memory allocation, detects tool boundaries, and applies graduated responses as pressure changes: `memory.high` delays throttle allocation, `cgroup.freeze` pauses a subtree, and `memory.max` remains the hard boundary. The parent runtime keeps the state needed to decide what to do after the tool returns or fails.

## What Trace Replay Shows

We evaluated AgentCgroup by replaying real agent memory traces at 50x accelerated speed in a multi-tenant setting on a patched Linux 6.19.0-rc5 kernel (bpf-next + memcg struct_ops RFC patches). Three concurrent agent traces share constrained memory:

Tight memory scenario (1100 MB total for ~1233 MB combined demand):

- Baseline: OOM-kills one low-priority process (66% survival)
- AgentCgroup: all processes complete (100% survival), 239 throttle triggers, high-priority agent finishes with only +2.8% overhead

Moderate memory scenario (1300 MB total):

- AgentCgroup reduces high-priority P95 allocation latency by 29% (70.97 to 50.14 ms) through reduced memory contention
- Total completion time: −1.1% (net improvement)

The measured P50 latency overhead is 0.3%, and BPF throttling precision stays within 2.3% relative error. These are trace-replay results from a proof-of-concept kernel path, so they demonstrate the control mechanism under the paper's setup rather than production-scale behavior.

## Reproducing the Results

The fastest inspection path uses the repository's collected traces in user space, before moving on to experiments that load eBPF programs:

```bash
git clone --recurse-submodules https://github.com/eunomia-bpf/agentcgroup.git
cd agentcgroup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python analysis/characterization.py
```

The [reproduction guide](https://github.com/eunomia-bpf/agentcgroup/blob/main/docs/REPRODUCING.md) then maps commands to the CPU scheduling, memory isolation, and overhead experiments. CPU control requires Linux 6.12 or newer with `sched_ext` and cgroup v2. The memory experiments additionally use the `memcg_bpf_ops` kernel path described by the artifact, so that part requires the matching kernel support and root access.

## Where the Evidence Stops

The characterization covers Claude Code on SWE-rebench, and the controller evaluation uses accelerated trace replay with a proof-of-concept implementation. Live concurrent agents, other frameworks such as OpenHands and SWE-agent, additional container runtimes, and the upstream status of `memcg_bpf_ops` remain open work. The [paper](https://arxiv.org/abs/2602.09345) and [eunomia-bpf/agentcgroup](https://github.com/eunomia-bpf/agentcgroup) repository expose the raw experiments, analysis scripts, controller code, and reproduction instructions needed to test those boundaries.
