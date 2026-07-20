---
date: 2026-07-10
slug: schedcp-agentic-linux-scheduler
description: Linux scheduler tuning turns workload intent into measured tradeoffs, and the SchedCP paper reports up to 1.79x speedup and 13x lower search cost from a verified sched_ext loop.
---

# Can an AI Agent Tune the Linux Scheduler? Inside SchedCP

A scheduler that makes a Linux kernel build faster can make `schbench` latency worse. Both runs may show busy CPUs, but the operator cares about throughput in one case and wake-up latency in the other, so an AI agent cannot tune the scheduler safely until workload intent becomes part of the control loop.

An AI agent can read a request such as "reduce tail latency without starving the batch jobs" and turn it into an experiment. Giving that agent an unrestricted root shell also lets one hallucinated command, broken configuration, or invalid scheduler end the experiment before it produces useful feedback. Generated kernel policy code still needs to compile, pass safety checks, deploy correctly, and improve the target workload under measurement.

[SchedCP](https://github.com/eunomia-bpf/schedcp) turns that open-ended shell session into a controlled optimization loop. A Model Context Protocol (MCP) server exposes specific observation and scheduler-management operations, while Linux `sched_ext` provides the loadable eBPF policies. The paper [**Towards Agentic OS: An LLM Agent Framework for Linux Schedulers**](https://arxiv.org/abs/2509.01245) studies how this split gives the agent room to reason while a systems control plane retains authority over privileged execution.

<!-- more -->

## Faster According to Which Metric?

Consider optimizing `schbench`. CPU utilization alone cannot tell the agent whether a candidate helped. It must extract a latency objective, preserve the workload parameters, and compare repeated runs. A mixed foreground and batch workload adds another constraint because improving average throughput can still starve the process on the critical path. Runtime counters describe what happened, while the request determines which measurements count as success.

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

When existing policies fall short, the execution verifier controls the synthesis path. The paper describes three gates. The kernel eBPF verifier checks memory safety and termination. Scheduler-specific static analysis looks for logic failures such as starvation and unfairness that the kernel verifier does not cover. Dynamic validation then runs the candidate in a secure micro-VM before a monitored canary deployment. Successful validation issues a signed deployment token, and a circuit breaker can roll back a policy when measured performance degrades.

The MCP boundary exposes these operations as tools with narrow jobs: `list_schedulers` finds existing policies, `system_monitor` collects measurements, and `create_and_verify_scheduler` admits generated source only after verification. The agent plans across these tools while deployment details remain inside SchedCP. That boundary also lets the benchmark harness and scheduler implementation evolve without changing the reasoning interface.

![SchedCP control-plane design](https://raw.githubusercontent.com/eunomia-bpf/schedcp/master/document/design.png)

## Why `sched_ext` Fits Agentic Optimization

Traditional scheduler experimentation often requires kernel patches, reboot cycles, or out-of-tree modules. Those costs make iterative search painfully slow and increase the damage from a bad candidate. `sched_ext` moves the scheduling policy into verified eBPF while keeping a kernel-managed safety boundary. If a policy exits or fails, the system can return to the normal scheduler instead of leaving the machine with a permanently modified kernel.

This model gives SchedCP the iteration speed an agent needs. It can compile a policy, load it for one benchmark run, collect the result, unload it, and move to the next candidate. The kernel verifier and the SchedCP execution verifier cover different risks: the kernel checks eBPF safety properties, while SchedCP checks whether the generated scheduler and its deployment fit the experiment it is about to run.

## What the Evaluation Shows

The paper labels this a preliminary evaluation. It uses two machines, Linux 6.13 and 6.14, with Claude Code running Opus 4. Each case is measured three times and averaged. The results show that the loop can recover from a poor first choice, but they do not establish universal scheduler improvement across machines and workloads.

For a Linux kernel build, the first selected scheduler, `scx_rusty`, produced a 1.63x speedup over EEVDF. Iterative refinement then selected `scx_layered` and reached 1.79x. The more instructive result comes from `schbench`. The first AI configuration, `scx_bpfland`, was worse than EEVDF. After three feedback iterations, the agent selected `scx_rusty`, reaching 2.11x better P99 latency and 1.60x higher throughput than EEVDF.

For eight batch workloads with 39 short tasks and one long task, sched-agent inferred a Longest Job First policy and reduced average latency by 20%. The paper also reports a 13x improvement in scheduler-generation efficiency, reaching 2.5 minutes and a $0.45 synthesis cost per workload.

These measurements support a narrower claim than universal automatic speedup. A structured control plane can make the search cheaper, reject unsafe candidates, and use workload feedback to overturn the agent's initial choice. The repository includes prompts, workloads, generated schedulers, and benchmark paths for inspecting that process, while a broader benchmark remains future work.

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

## References

- [Towards Agentic OS: An LLM Agent Framework for Linux Schedulers](https://arxiv.org/abs/2509.01245)
- [SchedCP repository](https://github.com/eunomia-bpf/schedcp)
- [Linux kernel documentation: Extensible Scheduler Class](https://docs.kernel.org/scheduler/sched-ext.html)
- [Model Context Protocol specification](https://modelcontextprotocol.io/specification/)
- [schbench scheduler benchmark](https://kernel.googlesource.com/pub/scm/linux/kernel/git/mason/schbench/)
