---
date: 2026-07-10
slug: schedcp-agentic-linux-scheduler
description: SchedCP separates AI reasoning from system execution to let an LLM agent tune Linux schedulers safely. The paper reports up to 1.79x kernel-build speedup and 13x lower generation cost through a verified sched_ext feedback loop.
---

# Can an AI Agent Tune the Linux Scheduler? Inside SchedCP and sched-agent

Linux makes scheduling decisions based on runnable tasks, wakeups, priorities, and accumulated CPU time. Applications evaluate those decisions in different terms: a kernel build cares about when the last compiler process finishes, an online service watches P99 latency and throughput, and a batch pipeline may be dominated by a single long job among dozens of short ones. The kernel supplies mechanisms and measurements; the workload defines what counts as a good result. This mismatch (the paper calls it a "semantic gap") means the default EEVDF scheduler applies one-size-fits-all policies to workloads that need different tradeoffs.

The [SchedCP paper](https://arxiv.org/abs/2509.01245) asks whether an LLM agent can bridge that gap without unrestricted control of the machine. The core insight is architectural: decouple "what to optimize" (the AI's domain of semantic reasoning) from "how to observe and act" (the system's domain of safe execution). SchedCP implements this separation as an MCP server exposing workload analysis, a scheduler policy repository, and an execution verifier; `sched-agent` uses those operations to observe a workload, choose or generate a policy, test it, and learn from the result. The paper reports promising case studies (up to 1.79x kernel-build speedup and 13x reduction in generation time and cost) while presenting the evaluation as an initial study rather than a production-readiness claim.

<!-- more -->

## Why a naive prompt fails: several problems in one request

The paper begins with a deliberately simple request: Claude Code starts from an empty directory with full shell access and is asked to "write a FIFO scheduler in eBPF." Only one of three attempts produces a working scheduler. Another stops at pseudocode after six minutes; the third builds a scheduler tracer instead after eight minutes. The successful attempt takes 33 minutes, 221 LLM API calls, more than 15 iterations, and about $6. Even then, the resulting policy performs worse than EEVDF on some workloads; the agent required root access with no fallback mechanism when experiments crashed the system.

That prompt quietly combines several jobs. The agent must learn the `sched_ext` interface, write eBPF code that passes the kernel verifier, arrange privileged loading, choose a benchmark, decide which metric matters, and interpret the measurement. A successful compilation settles only one part of the task. Useful tuning also depends on whether the policy improves the intended workload, keeps overhead and starvation under control, and can be reverted when results degrade. The paper identifies three critical challenges: **Performance** (the AI scheduler must outperform existing ones), **Safety** (no crashes, lockups, or starvation, with minimal privileges), and **Efficiency** (the 33-minute, $6 generation cost is impractical).

SchedCP addresses these challenges by decomposing the problem into two stages: **goal inference** and **policy synthesis**. Goal inference analyzes commands, source structure, process behavior, and performance data to produce a workload description with an objective and constraints. Policy synthesis then searches for an existing scheduler in a repository, modifies or composes known components when appropriate, and generates a new policy only when reuse cannot express the required behavior. This ordering matters: selecting a measured policy from a repository is cheaper and easier to validate than writing new kernel code for every workload.

## The SchedCP control plane: a safe interface between Agent and kernel

Rather than giving the agent a general root shell, SchedCP presents a stable systems interface governed by four design principles: (1) **decoupling and role separation** for future-proofing, (2) **safety-first interface design** that treats the AI as a potentially non-cautious actor and prevents catastrophic failures by default, (3) **adaptive context provisioning** that starts with minimal summaries and progressively reveals detail to manage token costs, and (4) **composable tool architecture** following Unix philosophy so agents can construct novel workflows.

The paper groups the interface into three services. The **Workload Analysis Engine** provides tiered access: cost-effective API endpoints with pre-processed summaries (CPU load, memory usage), then secure sandbox access to `perf`, `top`, file reads, application builds, and dynamically attachable eBPF probes when deeper investigation is needed. The **Scheduler Policy Repository** stores executable eBPF programs with metadata (natural language descriptions, target workloads, historical performance metrics), enabling semantic search so the agent can find relevant schedulers or composable primitives before writing new code. The **Execution Verifier** validates all AI-generated code and configurations before deployment.

![SchedCP control-plane design](imgs/schedcp-design.png)

In the paper's design, verification progresses through multiple stages: (1) the kernel's eBPF verifier ensures memory safety and termination, (2) scheduler-specific static analysis detects logic flaws like starvation and unfairness that the standard verifier misses, and (3) dynamic validation in a secure micro-VM tests correctness and performance. Policies that pass receive signed deployment tokens for monitored canary runs with circuit breakers to revert if performance degrades, eliminating the agent's need for root access.

The public [SchedCP repository](https://github.com/eunomia-bpf/schedcp) already exposes concrete MCP operations: `list_schedulers`, `run_scheduler`, `stop_scheduler`, `get_execution_status`, `create_and_verify_scheduler`, `system_monitor`, and `workload` management. The implementation includes custom scheduler compilation using clang with BPF targets and a 10-second kernel verification step. The full micro-VM, signed-token, and circuit-breaker path described in the paper is not yet visible as an end-to-end implementation, so that portion should be read as proposed architecture.

This boundary redefines responsibilities. The agent reasons about workload intent and explores alternatives; deterministic software owns scheduler loading, measurement, and policy checks. The generated policy runs natively through Linux `sched_ext`, so there is no model inference latency in the scheduler hot path, unlike traditional ML approaches that would add unacceptable overhead. The paper notes the framework is implemented in approximately 4,000 lines of Rust and 6,000 lines of Python (including tests).

## sched-agent: in-context reinforcement learning for schedulers

Building on SchedCP, the paper introduces `sched-agent`, a multi-agent system implementing in-context reinforcement learning (ICRL) for scheduler optimization. Using Claude Code's subagent architecture, it decomposes the task into four specialized roles with separate context windows and customized prompts. The framework integrates with container orchestrators (Kubernetes, Docker) to automatically trigger optimization when applications deploy.

The **Observation Agent** builds workload profiles by querying the Workload Analysis Engine strategically, starting with high-level summaries from process names and commands, then requesting deeper profiling (`perf stat`, `top`) only when initial signals are ambiguous. It manages cost-precision tradeoffs explicitly. For a kernel compilation, it produces a profile like: "CPU-intensive parallel compilation with short-lived processes, inter-process dependencies, targeting makespan minimization."

The **Planning Agent** transforms workload profiles into optimization strategies via the Scheduler Policy Repository, following a decision hierarchy: first configure existing schedulers, then generate patches, and finally compose new schedulers from primitives only when simpler options are insufficient. The **Execution Agent** synthesizes code artifacts, submits them to the Execution Verifier, and interprets results to refine code or fix logic issues. The **Learning Agent** completes the ICRL loop by analyzing deployment outcomes (e.g., "45% makespan reduction"), enabling in-session adaptation while updating the repository with refined metrics, deployment contexts, and documented antipatterns for future searches.

The key insight is that this loop improves through session context and stored experience without retraining a scheduler model for each workload. A scheduler name alone says little: `scx_rusty` can be excellent for one workload and poor for another, and the same workload can change as its concurrency or data distribution shifts. By preserving the connection between workload description, policy, and measured result, the repository enables evidence-based policy reuse rather than static recommendations.

## Preliminary evaluation: four research questions

The evaluation addresses four research questions: **RQ1** (configuring existing schedulers), **RQ2** (generating new schedulers for specific workloads), **RQ3** (cost and efficiency of generation), and **RQ4** (iterative refinement improvements). Two machines are used: an 86-core Intel Xeon 6787P with 758 GB RAM on Linux 6.14, and an 8-core Intel Core Ultra 7 258V with 30 GB on Linux 6.13. Claude Code with Opus 4 drives the agent; each case runs three times and results are averaged. The paper explicitly notes that a complete benchmark suite is future work, and all experiments successfully created working custom configurations or eBPF programs.

**Scheduler Configuration (RQ1):** For kernel compilation (`tinyconfig`, `make -j 172` on Linux 6.14 source), SchedCP initially selects `scx_rusty`, reducing average time from EEVDF's 13.57 seconds to 8.31 seconds (1.63x speedup). Iterative refinement then identifies `scx_layered` for an additional 16% gain, reaching 7.60 seconds; 1.79x total improvement over EEVDF. A pre-trained RL-based scheduler shows no improvement (13.79 seconds), likely because it requires costly hardware/workload-specific retraining.

**Learning from Feedback:** The `schbench` experiment demonstrates how feedback corrects a poor initial choice. With EEVDF: 910 requests per second, 40.3 ms P99 latency. The agent's first selection (`scx_bpfland`) underperforms: 741 requests per second (0.81x), 46.1 ms P99 (0.87x). After three refinement iterations, it identifies `scx_rusty` as superior: 1,452 requests per second (1.60x) and 19.1 ms P99 latency (2.11x improvement over EEVDF).

**New Scheduler Synthesis (RQ2):** Eight diverse batch workloads (file compression, video transcoding, software testing, data analytics) each launch 40 parallel tasks with a long-tail distribution: 39 short jobs, one long job. `sched-agent` correctly identifies the optimization goal and workload pattern, implementing Longest Job First (LJF) scheduling to achieve 20% average latency reduction. Claude Opus successfully classifies all 8 workloads at $0.15 per analysis; Claude Sonnet fails this classification step.

**Cost Reduction (RQ3):** With SchedCP's reuse and tooling, generation efficiency improves 13x (from 33 minutes to 2.5 minutes), with $0.45 synthesis cost per workload versus $6 for naive generation.

## What these cases establish and what remains

The experiments demonstrate a control loop completing three kinds of work under the reported conditions: selecting a scheduler that improves a known workload, recovering after measurement exposes a poor selection, and generating a simple policy for a recognizable workload pattern. They also illustrate why the system boundary matters: the agent can vary with the model and prompt, while validation, deployment, measurement, and stored policy history remain explicit parts of surrounding infrastructure that do not change with each model update.

The current evidence covers two machines, a small set of workloads, three-run averages, and Claude Code with Opus 4. Broader claims about production reliability would require longer deployments, mixed and changing workloads, more hardware and kernel versions, failure injection, and measurements of the safety mechanisms the architecture describes. The public repository labels the project experimental and notes that its OS-optimization benchmark is still under construction.

SchedCP's most useful contribution at this stage is a concrete division of labor. An AI agent translates workload evidence into a scheduling hypothesis; Linux `sched_ext` runs the resulting policy with kernel-level safety guarantees; a control plane decides which experiments are admissible and whether their measured results deserve another iteration. That is a more inspectable path to scheduler optimization than asking a general coding agent to produce privileged kernel code and treating successful compilation as success. The paper frames this as a step toward an "Agentic OS" (systems that can drive their own optimization) while being clear that the current results are preliminary.

## References

- Yusheng Zheng, Yanpeng Hu, Wei Zhang, and Andi Quinn. "Towards Agentic OS: An LLM Agent Framework for Linux Schedulers." arXiv:2509.01245v4, September 2025. <https://arxiv.org/abs/2509.01245>
- SchedCP official repository: <https://github.com/eunomia-bpf/schedcp>
- Linux sched_ext documentation: <https://docs.kernel.org/scheduler/sched-ext.html>
- Model Context Protocol: <https://modelcontextprotocol.io/>
