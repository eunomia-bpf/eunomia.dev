---
date: 2026-02-17
---

# AgentCgroup: What Happens When AI Coding Agents Meet OS Resources?

AI coding agents such as Claude Code, OpenHands, and SWE-agent are increasingly deployed in multi-tenant cloud environments, where they execute diverse tool calls inside sandboxed containers. Despite growing adoption, the OS-level resource dynamics of these workloads remain poorly understood. We present the first systematic characterization, analyzing 144 software engineering tasks from the SWE-rebench benchmark across two LLM backends. Our measurements reveal that OS-level overhead, including container initialization and tool execution, accounts for 56–74% of end-to-end latency, while LLM reasoning contributes only 26–44%. Memory exhibits a 15.4x peak-to-average ratio (compared to ~1.5x for serverless and 2–3x for microservices), with change rates reaching 3 GB/s in sub-second bursts. The same tool type (Bash) varies 13.7x in memory consumption depending on command semantics, and repeated runs of the same task produce 1.8x execution time variance with near-zero correlation (r = −0.14) between token output and peak memory.

These characteristics expose mismatches with existing resource management mechanisms, from kernel cgroup limits and systemd-oomd to Kubernetes VPA, where static allocation either wastes 93% of provisioned capacity or triggers OOM kills that destroy minutes of accumulated, non-reproducible agent state. In this post, we summarize the characterization findings from our [AgentCgroup paper](https://github.com/yunwei37/agentcgroup-paper) and describe how eBPF-based in-kernel enforcement can bridge the gap between agent workload dynamics and OS-level resource control.

<!-- more -->

> Paper: *AgentCgroup: Understanding and Controlling OS Resources of AI Agents*
>
> GitHub: [https://github.com/yunwei37/agentcgroup-paper](https://github.com/yunwei37/agentcgroup-paper)

## What We Did

We instrumented [Claude Code](https://docs.anthropic.com/en/docs/claude-code), a production AI coding agent, running 144 software engineering tasks from the [SWE-rebench](https://github.com/swe-bench/SWE-ReB) benchmark across two LLM backends:

- Claude Haiku 4.5 (cloud API): LLM inference runs on Anthropic's cloud; the container only runs the agent framework and tool calls.
- GLM-4.7-Flash (local GPU): LLM inference runs on a local GPU; everything happens on the same machine.

Both use the exact same agent framework (Claude Code, Node.js-based). The only difference is the underlying model and where inference happens. This lets us isolate the effect of model choice on container-level resource dynamics.

### Experimental Setup

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

No resource limits were imposed during characterization to capture the unconstrained, ground-truth resource behavior.

### Task Coverage

Our dataset spans six task categories at three difficulty levels, covering representative real-world software engineering scenarios:

| Category | Example Projects | Difficulty Levels |
|----------|-----------------|-------------------|
| CLI Tools | faker, click | Easy, Medium, Hard |
| DevOps / Build | pre-commit, dvc | Easy, Medium, Hard |
| ML / Scientific | numba, scikit-learn | Easy, Medium, Hard |
| Medical / Bio | pydicom, biopython | Easy, Medium, Hard |
| SQL / Data | sqlalchemy, pandas | Easy, Medium, Hard |
| Web / Network | streamlink, requests | Easy, Medium, Hard |

Additionally, we curated an 18-task representative subset (6 categories x 3 difficulties) for detailed per-category analysis.

## Characterization Results

We organized our characterization around three axes, each bearing on a different aspect of resource control:

1. Execution model: determines the granularity at which resources vary
2. Resource dynamics: determines how fast controls must react
3. Unpredictability: determines whether demands can be predicted

### The Agent Execution Model

#### OS Infrastructure Dominates Latency, Not LLM Reasoning

Contrary to the intuition that "the LLM is the bottleneck," our measurements show that LLM reasoning accounts for only 26–44% of end-to-end task latency. The remaining 56–74% is OS-level overhead:

| Latency Component | Haiku | GLM |
|-------------------|-------|-----|
| Container + agent initialization | 47.7% | 31.0% |
| Tool execution | 25.9% | 25.5% |
| LLM reasoning | 26.4% | 43.5% |

Container startup alone averages 26.5 seconds (median 23.0s, max 97s), driven by Podman's user-namespace ID remapping of overlay layers that scales with image size. Since SWE-rebench container images range from 2.9 GB to 17.3 GB (median 3.5 GB), roughly 7x larger than typical microservice images and 70x larger than serverless functions, this initialization overhead is substantial.

Optimizing infrastructure, specifically container startup and resource scheduling during tool execution, therefore directly improves more than half of user-perceived completion time. Focusing solely on LLM inference optimization misses the larger fraction of end-to-end latency.

#### Task Duration and Statefulness

Each agent task runs for 5–11 minutes (GLM mean 10.8 min, Haiku mean 5.8 min, overall median 8.1 min), executing stateful multi-round reasoning and tool-call loops within a single container. Agent tasks sit between serverless invocations (100ms–2s) and batch jobs in duration, but are in-process stateful. All LLM context, intermediate code modifications, and tool results live in the process's memory.

#### Tool Execution Composition

Bash and sub-agent (Task) calls dominate tool execution time, accounting for over 90% of tool time across both models. However, the two models adopt quite different strategies.

Haiku distributes work across multiple tool types:

- Sub-agent calls (Task): 47.8% of tool time (avg 100.47s per call)
- Bash: 43.2% (avg 3.76s per call)
- WebSearch/WebFetch: ~5%
- Read, Edit, Grep: <5% combined

GLM concentrates almost everything in Bash:

- Bash: 99.5% of tool time (avg 5.93s per call)
- No sub-agent or web search usage

This divergence has direct resource management implications. Haiku offloads computation to external services (sub-agents, web search), while GLM funnels all computation through local Bash calls, resulting in significantly higher local resource consumption (Bash total time: 19,598s for GLM vs. 1,543s for Haiku).

#### Bash Command Semantics

Not all Bash calls are equal. Breaking down by command semantics:

| Bash Category | % of Bash Time (Haiku) | % of Bash Time (GLM) |
|---------------|------------------------|----------------------|
| Test execution (pytest, unittest, etc.) | 72.9% | 43.7% |
| Python snippets | n/a | 26.9% |
| Package installation | 10.8% | 10.1% |
| Git operations | <5% | <5% |
| File exploration | <5% | <5% |

Test execution overwhelmingly dominates, and as shown in the next section, it is also the most resource-intensive category.

#### The "Understand-Modify-Verify" Temporal Pattern

When we divide execution into 10 equal phases and plot tool distribution over time, a clear pattern emerges:

- Understand phase (0–30%): Read operations dominate (code exploration)
- Modify phase (30–70%): Edit operations are distributed throughout; Bash begins rising
- Verify phase (40–100%): Bash peaks (repeated test execution, debugging)

This phase signature mirrors the "understand, modify, verify" workflow of human software engineering, providing a basis for phase-aware resource control.

### Resource Dynamics

#### Memory Is the Concurrency Bottleneck, Not CPU

Agent CPU utilization is low:

| Metric | Haiku | GLM |
|--------|-------|-----|
| Average CPU utilization | 13.2% | 7.6% |
| Samples exceeding 50% CPU | 8.2% | 0.5% |
| Peak CPU | >175% (multi-core) | >100% (brief spikes) |

On our 24-core platform, CPU stays below 36% even at maximum memory-limited concurrency density. Memory tells a very different story: peak memory reaches 2–4 GB per task, meaning 128 GB of RAM supports only 32–64 concurrent instances when allocated by peak, while CPU remains underutilized.

This CPU-memory imbalance means that dynamic memory management is the key lever for increasing multi-tenant density: elastically expanding during brief memory bursts and reclaiming during idle periods to accommodate more concurrent instances.

#### The "Two-Layer" Memory Structure

Agent memory exhibits a distinctive two-layer pattern that we did not observe in any prior workload characterization.

Layer 1, the framework baseline (~185 MB): The Node.js runtime, V8 JIT cache, and agent framework state maintain a stable, incompressible memory floor throughout execution, even during LLM reasoning phases with zero tool activity. Across all 144 tasks, early-execution memory averages 183 MB (Haiku) and 188 MB (GLM).

Layer 2, tool-call bursts (500 MB to 2+ GB): Test execution, dependency installation, and data processing operations create transient spikes that last only 1–2 seconds before collapsing back to the ~185 MB baseline.

When we normalize and aggregate memory traces across all 144 tasks by execution progress, the pattern is clear: the first half of execution stays at a stable 185–200 MB baseline, while the second half shows increasing variance with large spikes, corresponding to the Bash-intensive verify phase.

In a multi-tenant deployment, 64 concurrent instances require ~12 GB just for the framework baseline alone. The tool-call bursts layered on top are the real resource management challenge, and they require different treatment from the stable baseline.

#### 98.5% of Memory Bursts Are Tool-Call-Driven

We annotated every 1-second resource sample as "during tool call" or "during LLM reasoning" and counted memory bursts exceeding 300 MB (~1.6x the framework baseline):

| Metric | Haiku | GLM |
|--------|-------|-----|
| Tool call time fraction | 50.6% | 35.9% |
| Memory bursts during tool calls | 98.5% | 67.3% |
| Burst concentration ratio | 1.9x | 1.9x |
| CPU bursts during tool calls | 55.3% | 30.2% |

The asymmetry is notable. Memory bursts are almost exclusively tool-call-driven, while CPU bursts are more dispersed (GLM's local GPU inference generates steady CPU load even outside tool calls). This means memory should be managed at tool-call granularity, while CPU requires broader context awareness.

#### Sub-Second Bursts with Large Peak-to-Average Ratios

Resource bursts are not only tool-driven but also very short-lived:

- Maximum memory change rate: 3 GB/second
- Maximum CPU change rate: >50%/second
- Burst duration: typically 1–2 seconds

The highest case we observed, a pydicom bioinformatics task (Medical_Bio_Hard), reached 4060 MB peak versus 264 MB average, a 15.4x peak-to-average ratio. This 4 GB spike lasted approximately 1–2 seconds before falling back to the 230 MB baseline.

For comparison with traditional cloud workloads:

| Workload Type | Typical Peak/Avg Ratio |
|---------------|----------------------|
| Serverless / FaaS | ~1.5x |
| Microservices | 2–3x |
| Batch / HPC | ~1x |
| AI Coding Agent | up to 15.4x |

This ratio makes static resource limits impractical. Allocating by peak (4060 MB) means 98% of the time memory usage is below 264 MB, resulting in 93% waste. Allocating by average (264 MB) means tool bursts trigger OOM kills, destroying all agent state. No single static threshold can accommodate both the low baseline and the transient spikes.

#### Same Tool, Very Different Resources

An interesting finding is that the same tool type (Bash) varies 13.7x in resource consumption depending on what it actually runs. Resource demand is determined by command semantics, not tool type:

| Bash Category | P95 Memory Spike (Haiku) | P95 Memory Spike (GLM) | Avg CPU Spike |
|---------------|--------------------------|------------------------|---------------|
| Test execution (pytest, etc.) | 518 MB | 234 MB | +3.2% |
| Package installation | 233 MB | n/a | moderate |
| Git operations | 13.5 MB | n/a | minimal |
| File exploration | 4.5 MB | n/a | minimal |

Medical/bioinformatics Bash commands average 4 GB peak memory; web/network commands average 291 MB, a 13.7x difference. The same `Bash` tool invocation can range from a trivial `ls` to a full `pytest` suite loading gigabytes of test data. This renders tool-type-based resource policies ineffective; semantic awareness of what is actually being executed is required.

#### CPU-Memory Independence

CPU and memory do not move together. The correlation between CPU and memory usage varies from -0.84 to +0.50 across tasks, with a mean of -0.39. Some tasks show positive correlation (tool execution pulls up both), while others show negative correlation (CPU-intensive phases coincide with lower memory). This task-dependent coupling means resource control strategies cannot assume CPU and memory demands co-vary and must monitor and manage the two dimensions independently.

### Unpredictability

#### Non-Determinism Within the Same Task

Running the exact same task (iterative/dvc#777) three times produced:

| Run | Execution Time | Solution Strategy |
|-----|---------------|-------------------|
| 1 | 402 seconds | Strategy A (different file modifications) |
| 2 | 222 seconds | Strategy B (different approach) |
| 3 | 259 seconds | Strategy C (different file count) |

That is a 1.8x variance in execution time, with completely different solution strategies each time. This non-determinism stems from LLM reasoning randomness and decision-path diversity: the agent may choose entirely different code modifications, tool sequences, and debugging approaches on each run.

#### Token Count Does Not Predict Resource Usage

We analyzed the correlation between LLM-observable proxies and actual resource consumption:

| Proxy to Target | Haiku (r) | GLM (r) |
|-----------------|-----------|---------|
| Output tokens to peak memory | −0.14 | +0.02 |
| Conversation rounds to execution time | +0.57 | +0.82 |
| Conversation rounds to peak memory | +0.02 | +0.11 |

Output token count shows essentially zero correlation with peak memory. Even conversation rounds, which moderately predict execution time, are useless for predicting memory. Resource consumption is driven by what tools execute (e.g., pytest vs. file read), not by the scale of LLM reasoning. This means that even if one can predict how much an agent will "think," one still cannot predict how much memory it will need.

#### Retry Loops and Progressive Memory Accumulation

Retry behavior is a defining characteristic of agent workloads that has no counterpart in traditional containerized applications:

| Metric | Haiku | GLM |
|--------|-------|-----|
| Tasks with retry loops (3+ consecutive identical Bash calls) | 85% (28/33) | 97% (108/111) |
| Average retry groups per task | n/a | 3.9 |
| Maximum consecutive retries | n/a | 56 |
| Execution time consumed by retries | 7.4% | 20.5% |

The "execute test, observe failure, modify code, re-test" iteration loop is the agent's behavioral signature. Each retry retains prior memory context without cleanup, leading to progressive memory accumulation, up to 502 MB of unreleased memory in the worst case we observed. This means memory limits that were adequate early in execution may trigger OOM kills later as retries accumulate.

#### Cross-Task Heterogeneity

Across our dataset, peak memory requirements range from 197 MB to 4 GB (coefficient of variation = 147%):

- Scientific computing tasks (numba, pydicom): 2–4 GB
- CLI tools (faker): ~200 MB
- Network utilities (streamlink): ~300 MB

That is a 20x variation across tasks using the same agent framework. Model choice amplifies this further: Haiku and GLM show a 1.7x CPU utilization difference on the same tasks. Simply swapping the underlying model, without changing the agent framework, produces a completely different resource profile.

### How Agent Workloads Compare to Traditional Cloud Workloads

| Dimension | Serverless | Microservices | Batch/HPC | AI Coding Agent |
|-----------|-----------|--------------|----------|-------------------|
| Duration | 100ms–2s | Long-running | Min–hours | 5–11 minutes |
| Statefulness | Stateless | External state | Stateful | In-process stateful |
| Memory peak/avg | ~1.5x | 2–3x | ~1x | 15.4x |
| CPU pattern | Brief spike | 10–40% steady | 80–100% | <13% avg, >175% peaks |
| Determinism | Deterministic | Mostly | Deterministic | 1.8x variance same task |
| Resource pattern | Flat | Steady + daily cycles | Stable rise | Burst-silence alternating |
| Kill cost | Just retry | Migrate | Lose progress | Lose all LLM context |
| Image size | ~50 MB | ~500 MB | Varies | 3.5 GB median |

In short, agent workloads are too stateful to kill, too spiky to cap, too unpredictable to predict, and too brief to amortize container overhead.

## Three Mismatches

These characterization results point to three mismatches between agent workloads and the existing resource management stack.

### 1. Granularity Mismatch

Container-level policies (cgroup `memory.max`, Kubernetes QoS) set a single threshold for the entire container, but agent resource demands vary at tool-call granularity. A `git status` (13.5 MB spike) and a `pytest` run (518 MB spike) need completely different memory budgets, yet they share the same cgroup. The `memory.high` soft limit cannot distinguish the ~185 MB framework memory (incompressible Node.js heap, V8 JIT cache) from tool subprocess memory (compressible, limitable). When kernel reclaim hits framework pages, it causes V8 GC pressure and JIT cache thrashing, degrading LLM response parsing.

### 2. Responsiveness Mismatch

User-space controllers (systemd-oomd, Meta oomd, Kubernetes VPA) react at millisecond-to-minute timescales. Agent memory bursts last 1–2 seconds with change rates of 3 GB/s. The full PSI signal to user-space daemon to decision to cgroup write-back loop takes tens of milliseconds at best. By then, the burst is already over or has already triggered a kernel OOM kill. VPA adjusts at Pod-restart granularity (minutes); even in-place resize (alpha) operates on minute timescales. Neither can react within a single tool call.

### 3. Adaptability Mismatch

History-based prediction (Google Autopilot, Kubernetes VPA percentile recommendations) assumes workload reproducibility. Agent non-determinism violates this assumption. Same task, 1.8x execution time variance, completely different solution strategies. Zero token-to-memory correlation (r = −0.14). 20x cross-task variance. The P95 of past runs is not a reliable upper bound for future runs. And unlike serverless where kill-restart costs 100ms, killing an agent destroys 5–11 minutes of accumulated stateful context that cannot be deterministically reproduced.

## AgentCgroup: eBPF-Based In-Kernel Resource Control

To address these three mismatches, we propose AgentCgroup, an eBPF-based resource controller with three corresponding design principles.

### Fine-Grained Resource Domains (Granularity Mismatch)

AgentCgroup organizes resources using a hierarchical cgroup v2 structure where each agent workload maps to a cgroup node with tool calls as child nodes. This enables per-tool-call resource constraints while maintaining overall workload budgets. For recovery, it uses cgroup v2 lifecycle primitives: freezing subtrees when tool calls exceed soft limits, and atomically killing subtrees (not the entire agent) when termination is necessary.

### In-Kernel Enforcement (Responsiveness Mismatch)

AgentCgroup executes control logic directly at kernel cgroup enforcement points via eBPF, enabling microsecond-level reaction without user-kernel round trips:

- On CPU, `sched_ext` maintains per-workload and per-tool-call metadata in BPF maps, prioritizing latency-sensitive tool calls with automatic fail-safe reversion on errors.
- On memory, `memcg_bpf_ops` hooks implement custom throttling delays when a cgroup breaches its soft limit (`memory.high`), with `memory.max` as the hard limit.

### Runtime-Adaptive Policies (Adaptability Mismatch)

Instead of history-based prediction, AgentCgroup uses eBPF to trace process creation and memory allocation in-kernel, detecting tool-call boundaries and resource dynamics in real time. When memory pressure rises, the BPF program applies graduated responses (throttling via `memory.high` delays, freezing via `cgroup.freeze`) rather than termination, preserving agent state.

### Preliminary Results

We evaluated AgentCgroup by replaying real agent memory traces at 50x accelerated speed in a multi-tenant setting on a patched Linux 6.19.0-rc5 kernel (bpf-next + memcg struct_ops RFC patches). Three concurrent agent traces share constrained memory:

Tight memory scenario (1100 MB total for ~1233 MB combined demand):

- Baseline: OOM-kills one low-priority process (66% survival)
- AgentCgroup: all processes complete (100% survival), 239 throttle triggers, high-priority agent finishes with only +2.8% overhead

Moderate memory scenario (1300 MB total):

- AgentCgroup reduces high-priority P95 allocation latency by 29% (70.97 to 50.14 ms) through reduced memory contention
- P50 latency overhead: +0.3%
- Total completion time: −1.1% (net improvement)

Enforcement overhead is negligible, with BPF throttling precision within 2.3% relative error.

## Looking Forward

Our current evaluation is based on trace replay with a proof-of-concept prototype, and the characterization covers one agent framework (Claude Code) and one benchmark (SWE-rebench). There is much more to explore:

- Live agent evaluation at production scale with real concurrent workloads
- Diverse agent frameworks (OpenHands, SWE-agent, Cursor) and domains beyond coding
- Fine-grained resource control across diverse container runtimes (Docker, gVisor, microVMs)
- Upstream kernel integration of the memcg_bpf_ops patches currently under review

The code, data, and paper are available at [https://github.com/yunwei37/agentcgroup-paper](https://github.com/yunwei37/agentcgroup-paper).
