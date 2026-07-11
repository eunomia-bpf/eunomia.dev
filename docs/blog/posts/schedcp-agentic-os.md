---
date: 2026-07-10
slug: schedcp-agentic-linux-scheduler
description: SchedCP gives AI agents a controlled path from workload intent to verified sched_ext policies, achieving up to 1.79x performance improvement and 13x lower optimization cost in the paper's evaluation.
---

# Can an AI Agent Tune the Linux Scheduler? Inside SchedCP

A parallel Linux kernel build wants throughput. An interactive service wants predictable wake-up latency. A mixed workload may want both, depending on which process is on the critical path. The Linux scheduler sees runnable tasks, priorities, and runtime statistics, but it cannot see the operator's actual goal. Two workloads that look similar to the kernel can need different policies.

An AI agent can read a benchmark description, inspect performance counters, and reason about the goal in natural language. Giving that agent a shell and asking it to rewrite a scheduler creates a different problem: generated kernel policy code still needs to compile, pass safety checks, deploy correctly, and prove that it improved the workload rather than one convenient metric.

[SchedCP](https://github.com/eunomia-bpf/schedcp) gives the agent a narrower and more reliable path. It exposes scheduler observation and control through a Model Context Protocol (MCP) server, then uses Linux `sched_ext` to select or synthesize eBPF scheduling policies. The corresponding paper, [**Towards Agentic OS: An LLM Agent Framework for Linux Schedulers**](https://arxiv.org/abs/2509.01245), studies how this control plane separates semantic reasoning from privileged execution.

<!-- more -->

## The Scheduler's Semantic Gap

Linux already provides rich scheduler telemetry, and `sched_ext` lets administrators load custom scheduling classes implemented with eBPF. The missing piece sits between those capabilities. Runtime counters describe what happened, while an optimization request describes what matters. A request such as "reduce the tail latency of the foreground server without starving batch jobs" has to become a measurable objective, a candidate policy, and a safe deployment plan.

A general-purpose agent can attempt all three steps in one conversation. That approach repeatedly spends model tokens rediscovering scheduler interfaces, parsing noisy output, and repairing generated code. It also places privileged actions next to unconstrained reasoning. A hallucinated command or an invalid scheduler configuration can end the experiment before the agent learns anything useful.

SchedCP treats this semantic gap as a control-plane problem. The AI reasons about workload intent and chooses an optimization direction. A stable systems layer owns profiling, policy construction, verification, deployment, and measurement. Each side works with the representation it understands best.

## Two Stages, One Controlled Loop

The paper divides optimization into goal inference and policy synthesis. Goal inference turns workload descriptions and measurements into an explicit objective. Policy synthesis selects an existing scheduler or produces a new `sched_ext` policy for that objective.

That split becomes a closed loop:

```text
workload + goal
      |
      v
profile and infer objective
      |
      v
select or synthesize sched_ext policy
      |
      v
compile, verify, deploy, and measure
      |
      +---------- feedback ----------+
```

The loop matters because scheduler optimization is empirical. A policy that sounds appropriate can still lose on the real workload because cache behavior, wake-up patterns, or contention differs from the agent's initial model. SchedCP feeds measured results back into the next decision and retains useful policies for later workloads.

## From Observation to a Verified Policy

Three services keep the loop grounded. The workload analysis engine runs the target command, collects CPU, memory, scheduler, and application-level measurements, and presents the agent with structured evidence. The policy repository stores existing schedulers, their configurations, and prior results, which gives the agent a starting point before it generates code.

When existing policies fall short, the execution verifier controls the synthesis path. The agent submits scheduler source through a dedicated interface; SchedCP compiles it, applies static and dynamic checks, and only then makes it available for an experiment. Measurement happens through the same control plane, so a policy enters the repository with its workload context and observed result rather than an unsupported claim.

The MCP boundary turns these operations into explicit tools such as scheduler listing, monitored execution, system measurement, and scheduler creation with verification. The agent can plan across them without receiving unrestricted access to every low-level command. MCP also keeps the interface stable as the scheduler implementation and benchmark harness evolve.

![SchedCP control-plane design](https://raw.githubusercontent.com/eunomia-bpf/schedcp/master/document/design.png)

## Why `sched_ext` Fits Agentic Optimization

Traditional scheduler experimentation often requires kernel patches, reboot cycles, or out-of-tree modules. Those costs make iterative search painfully slow and increase the damage from a bad candidate. `sched_ext` moves the scheduling policy into verified eBPF while keeping a kernel-managed safety boundary. If a policy exits or fails, the system can return to the normal scheduler instead of leaving the machine with a permanently modified kernel.

This model gives SchedCP the iteration speed an agent needs. It can compile a policy, load it for one benchmark run, collect the result, unload it, and move to the next candidate. The kernel verifier and the SchedCP execution verifier cover different risks: the kernel checks eBPF safety properties, while SchedCP checks whether the generated scheduler and its deployment fit the experiment it is about to run.

## What the Evaluation Shows

The paper evaluates SchedCP with workloads that include Linux kernel builds, `schbench`, and batch-processing tasks. Within that evaluation, the generated or selected policies improve performance by up to 1.79x. Reusing the control plane, policy repository, and structured tools also reduces optimization cost by 13x compared with the paper's naive agentic baseline.

Those are scoped results rather than a promise that every workload becomes faster. Their stronger implication concerns process: an agent can complete an end-to-end scheduler optimization loop with a much smaller search cost while preserving verification and measured feedback. The repository includes the prompts, workloads, generated schedulers, and benchmark paths needed to inspect how those results were produced.

## Trying the Artifact

After building the repository and its `sched_ext` submodules, `autotune` accepts the workload command as the optimization target. For example, the paper artifact includes a Linux build workload and `schbench`:

```bash
./autotune/target/release/autotune cc \
  "make -C workloads/linux-build-bench/linux -j"

./autotune/target/release/autotune cc \
  workloads/basic/schbench/schbench
```

The [SchedCP repository](https://github.com/eunomia-bpf/schedcp) documents the kernel requirements, build steps, MCP tools, and paper workloads. The [paper](https://arxiv.org/abs/2509.01245) provides the architecture and evaluation methodology. Together they make the result inspectable at three levels: the agent's reasoning interface, the verified `sched_ext` policy, and the workload measurement that decides whether a candidate survives.

SchedCP points toward an operating-system control plane where agents express and refine goals while verified mechanisms retain authority over privileged changes. Scheduler tuning is a useful proving ground because success has a concrete test: the generated policy must run safely, and the target workload must actually improve.
