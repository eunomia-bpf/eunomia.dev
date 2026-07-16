---
date: 2026-07-10
slug: schedcp-agentic-linux-scheduler
description: SchedCP gives AI agents a controlled path from workload intent to verified sched_ext policies, achieving up to 1.79x performance improvement and 13x lower optimization cost in the paper's evaluation.
---

# Can an AI Agent Tune the Linux Scheduler? Inside SchedCP

Load a throughput-oriented scheduler and a parallel Linux kernel build may finish sooner. Keep the same policy for `schbench`, and wake-up latency can move in the wrong direction. Both runs keep the CPUs busy, so the scheduler's counters do not explain which outcome the operator values. The missing input is the workload's goal.

An AI agent can read a request such as "reduce tail latency without starving the batch jobs" and turn it into an experiment. Giving that agent an unrestricted root shell also lets one hallucinated command, broken configuration, or invalid scheduler end the experiment before it produces useful feedback. Generated kernel policy code still needs to compile, pass safety checks, deploy correctly, and improve the target workload under measurement.

[SchedCP](https://github.com/eunomia-bpf/schedcp) turns that open-ended shell session into a controlled optimization loop. A Model Context Protocol (MCP) server exposes specific observation and scheduler-management operations, while Linux `sched_ext` provides the loadable eBPF policies. The paper [**Towards Agentic OS: An LLM Agent Framework for Linux Schedulers**](https://arxiv.org/abs/2509.01245) studies how this split gives the agent room to reason while a systems control plane retains authority over privileged execution.

<!-- more -->

## Faster According to Which Metric?

Consider optimizing `schbench`. CPU utilization alone cannot tell the agent whether a candidate helped; it must extract a latency objective, preserve the workload parameters, and compare repeated runs. A mixed foreground and batch workload adds another constraint because improving average throughput can still starve the process on the critical path. Runtime counters describe what happened, while the request determines which measurements count as success.

A general-purpose agent can attempt all three steps in one conversation. That approach repeatedly spends model tokens rediscovering scheduler interfaces, parsing noisy output, and repairing generated code. It also places privileged actions next to unconstrained reasoning. A hallucinated command or an invalid scheduler configuration can end the experiment before the agent learns anything useful.

SchedCP treats the translation from request to experiment as a control-plane problem. The agent interprets workload intent and proposes an optimization direction. A stable systems layer owns profiling, policy construction, verification, deployment, and measurement, giving every candidate the same path from source code to evidence.

## Turn the Request into an Experiment

SchedCP first performs goal inference, turning the workload description and baseline measurements into an explicit objective. Policy synthesis then searches the repository for a suitable scheduler or produces a new `sched_ext` policy. Keeping these stages separate prevents a plausible scheduler choice from silently redefining the goal it is supposed to optimize.

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

## Keep Generated Schedulers Behind a Verifier

Three services keep the loop grounded. The workload analysis engine runs the exact target command and returns CPU, memory, scheduler, and application-level measurements. The policy repository records schedulers with their configurations, workload context, and previous results, giving the agent evidence to inspect before it spends time generating code.

When existing policies fall short, the execution verifier controls the synthesis path. The agent submits scheduler source through a dedicated interface; SchedCP compiles it, applies static and dynamic checks, and only then makes it available for an experiment. Measurement happens through the same control plane, so a policy enters the repository with its workload context and observed result rather than an unsupported claim.

The MCP boundary exposes these operations as tools with narrow jobs: `list_schedulers` finds existing policies, `system_monitor` collects measurements, and `create_and_verify_scheduler` admits generated source only after verification. The agent plans across these tools while deployment details remain inside SchedCP. That boundary also lets the benchmark harness and scheduler implementation evolve without changing the reasoning interface.

![SchedCP control-plane design](https://raw.githubusercontent.com/eunomia-bpf/schedcp/master/document/design.png)

## Why `sched_ext` Fits Agentic Optimization

Traditional scheduler experimentation often requires kernel patches, reboot cycles, or out-of-tree modules. Those costs make iterative search painfully slow and increase the damage from a bad candidate. `sched_ext` moves the scheduling policy into verified eBPF while keeping a kernel-managed safety boundary. If a policy exits or fails, the system can return to the normal scheduler instead of leaving the machine with a permanently modified kernel.

This model gives SchedCP the iteration speed an agent needs. It can compile a policy, load it for one benchmark run, collect the result, unload it, and move to the next candidate. The kernel verifier and the SchedCP execution verifier cover different risks: the kernel checks eBPF safety properties, while SchedCP checks whether the generated scheduler and its deployment fit the experiment it is about to run.

## What the Evaluation Shows

The paper evaluates SchedCP with workloads that include Linux kernel builds, `schbench`, and batch-processing tasks. Within that evaluation, the generated or selected policies improve performance by up to 1.79x. Reusing the control plane, policy repository, and structured tools also reduces optimization cost by 13x compared with the paper's naive agentic baseline.

The 1.79x result is the best improvement within the evaluated workloads, and the 13x figure compares against the paper's naive agentic baseline. These measurements support a narrower claim than universal automatic speedup: the structured control plane lets an agent complete the end-to-end search with lower cost while every accepted candidate still passes verification and workload measurement. The repository includes the prompts, workloads, generated schedulers, and benchmark paths needed to inspect that process.

## Trying the Artifact

After building the repository and its `sched_ext` submodules, `autotune` accepts the workload command as the optimization target. For example, the paper artifact includes a Linux build workload and `schbench`:

```bash
./autotune/target/release/autotune cc \
  "make -C workloads/linux-build-bench/linux -j"

./autotune/target/release/autotune cc \
  workloads/basic/schbench/schbench
```

The [SchedCP repository](https://github.com/eunomia-bpf/schedcp) documents the kernel requirements, build steps, MCP tools, and paper workloads. The [paper](https://arxiv.org/abs/2509.01245) provides the architecture and evaluation methodology. Together they make the result inspectable at three levels: the agent's reasoning interface, the verified `sched_ext` policy, and the workload measurement that decides whether a candidate survives.

Scheduler tuning gives agentic OS control a demanding test. The agent must translate intent into a metric, produce a policy the kernel can run, and accept the measured result even when it contradicts the initial plan. SchedCP makes those transitions explicit, leaving semantic choices with the agent and privileged changes with mechanisms that can verify, measure, and roll back each attempt.
