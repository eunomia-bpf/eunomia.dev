---
date: 2026-05-24
slug: measuring-cpu-noise-in-gpu-inference-with-ebpf
description: Quantitative eBPF tracing of CUDA kernel launches, scheduler context switches, and IRQs shows when CPU noise matters for GPU LLM inference and how CPU pinning recovers throughput.
---

# When CPU Noise Slows Down GPU Inference: Measuring Scheduler and IRQ Impact with eBPF

GPU inference often looks like a GPU problem, but the CPU still sits on the critical path. It prepares inputs, launches CUDA kernels, manages synchronization, handles runtime calls, and shares cores with system work, interrupts, and other tenants. If that CPU-side launch path is delayed, the GPU can be left waiting even when the GPU kernels themselves are fast.

This post asks a concrete question: when an LLM inference workload is running on a GPU, how much do Linux CPU scheduling decisions and IRQ handling actually matter?

To answer it, we built an eBPF tracing tool, `cuda_sched_trace`, that records CUDA kernel launches, scheduler context switches, and hard/soft IRQ events with nanosecond timestamps. We then ran Qwen3 0.6B inference under clean and noisy-neighbor conditions: CPU load from `stress-ng`, network load from `iperf3`, disk load from `fio`, a combined heavy-load case, and a mitigation case using CPU pinning and priority adjustment.

The short version: in a clean environment, scheduler and IRQ overhead are small. Under production-like noisy-neighbor conditions, they can become very real. Combined CPU, network, and disk interference reduced throughput by **20.5%**, while simple CPU pinning reduced context switches by **96.3%** and recovered most of the lost throughput.

<!-- more -->

## Why CPU Scheduling Shows Up in GPU Inference

Modern GPU workloads, particularly LLM inference and training, require tight coordination between CPU and GPU execution. The CPU is responsible for:

- preparing input data and kernel parameters
- launching GPU kernels through CUDA APIs
- managing memory transfers and synchronization

An interruption to that CPU-side workflow can delay GPU kernel submission. In the worst case, the GPU has available compute capacity but no new work to execute.

The motivation comes partly from Meta's work on `sched_ext` for AI training optimization, where production issues include "IRQs preempting our important tasks." Network interrupts (`NET_RX`/`NET_TX`) and block device interrupts can matter for large distributed training jobs, and custom scheduling policies can improve AI workload performance by 5-20%.

But the impact is workload-dependent. A single-node LLM inference loop is not the same as distributed training with all-reduce traffic. Before investing in custom scheduling, we wanted measurements that separate scheduler problems from normal application behavior.

The study has four goals:

1. Measure the baseline impact of CPU scheduling on GPU kernel launches.
2. Characterize IRQ interference patterns and their performance cost.
3. Quantify noisy-neighbor impact under CPU, network, disk, and combined load.
4. Evaluate how much CPU pinning and priority adjustment help.

## Tracing the Launch Path

We developed `cuda_sched_trace`, an eBPF-based tracing tool that combines CUDA API uprobes, Linux scheduler tracepoints, and IRQ tracepoints.

### CUDA API Tracing

The tool attaches uprobes to CUDA Driver and Runtime APIs:

```c
// Attach to CUDA Driver API
SEC("uprobe/cuLaunchKernel")
int trace_cuLaunchKernel(struct pt_regs *ctx) {
    // Capture: timestamp, pid, tid, grid/block dimensions, shared memory, stream
    // Mark process as GPU process for scheduler tracking
}

// Attach to CUDA Runtime API
SEC("uprobe/cudaLaunchKernel")
int trace_cudaLaunchKernel(struct pt_regs *ctx) { ... }

SEC("uprobe/cudaDeviceSynchronize")
int trace_cudaDeviceSynchronize_enter(struct pt_regs *ctx) { ... }

SEC("uretprobe/cudaDeviceSynchronize")
int trace_cudaDeviceSynchronize_exit(struct pt_regs *ctx) { ... }
```

### Scheduler Event Tracing

Scheduler activity is captured through `sched_switch`, filtered to GPU-related processes:

```c
SEC("tp_btf/sched_switch")
int BPF_PROG(sched_switch, bool preempt, struct task_struct *prev, struct task_struct *next) {
    // Only track if prev or next is a GPU process
    // Record: timestamp, prev/next pid, off-cpu/on-cpu duration
}
```

### IRQ Tracing

Hard and soft IRQs are tracked through kernel tracepoints:

```c
SEC("tp_btf/irq_handler_entry")
int BPF_PROG(irq_handler_entry, int irq, struct irqaction *action) {
    // Track hard IRQ entry, record IRQ number and handler name
}

SEC("tp_btf/irq_handler_exit")
int BPF_PROG(irq_handler_exit, int irq, struct irqaction *action) {
    // Calculate IRQ duration
}

SEC("tp_btf/softirq_entry")
int BPF_PROG(softirq_entry, unsigned int vec_nr) {
    // Track soft IRQ: TIMER, NET_RX, NET_TX, BLOCK, SCHED, RCU, etc.
}

SEC("tp_btf/softirq_exit")
int BPF_PROG(softirq_exit, unsigned int vec_nr) {
    // Calculate soft IRQ duration
}
```

The data path is straightforward: the GPU application issues CUDA calls; eBPF programs observe CUDA, scheduler, and IRQ events in kernel space; events are sent through a BPF ring buffer; analysis scripts parse the resulting CSV.

```text
┌─────────────────────────────────────────────────────────────────┐
│                         User Space                               │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │ GPU App     │    │ cuda_sched  │    │ Analysis Scripts    │  │
│  │ (qwen3.cu)  │    │ _trace      │    │ (Python)            │  │
│  └──────┬──────┘    └──────┬──────┘    └──────────┬──────────┘  │
│         │                  │                       │             │
│         │ CUDA calls       │ perf_event            │ CSV parsing │
│         ▼                  ▼                       ▼             │
├─────────────────────────────────────────────────────────────────┤
│                         Kernel Space                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │ uprobes     │    │ tracepoints │    │ BPF Ring Buffer     │  │
│  │ (CUDA API)  │    │ (sched,irq) │    │ (Event Queue)       │  │
│  └─────────────┘    └─────────────┘    └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Benchmark and Environment

The benchmark is Qwen3 0.6B LLM inference using `qwen3.cu`.

| Property | Value |
|----------|-------|
| Model | Qwen3-0.6B-FP32 |
| Task | Single-turn Q&A |
| Input | "What is eBPF?" |
| Output | ~30-50 tokens |
| Kernel Pattern | Burst submission (~950 launches per token) |
| GPU Memory | ~3 GB |

This benchmark is useful because it resembles modern LLM inference, mixes compute-bound and memory-bound kernels, shows a clear burst submission pattern, and produces a measurable throughput metric in tokens per second.

| Component | Specification |
|-----------|---------------|
| CPU | 24 cores (specific model TBD) |
| GPU | NVIDIA GPU with CUDA support |
| Memory | Sufficient for model + system |
| OS | Linux 6.15.11-061511-generic |
| Kernel | BTF-enabled for CO-RE eBPF |
| CUDA | Driver API + Runtime API |

We used three interference tools:

| Tool | Purpose | Configuration |
|------|---------|---------------|
| stress-ng | CPU load | `--cpu 0 --cpu-method fft` (all cores) |
| iperf3 | Network I/O | Server + Client, 10 parallel streams, 60s |
| fio | Disk I/O | `randwrite, bs=4k, iodepth=32, 4 jobs` |

The full experiment has six scenarios:

| Scenario | Description | Interference |
|----------|-------------|--------------|
| Baseline | Clean environment | None |
| Noisy CPU | CPU-intensive | stress-ng on all cores |
| Noisy Network | Network I/O | iperf3 localhost loopback |
| Noisy Disk | Disk I/O | fio random write |
| Heavy Load | Combined | CPU + Network + Disk simultaneously |
| Optimized | CPU pinning | stress-ng + taskset -c 0-3 + nice -n -10 |

Data collection follows the same pattern in every run:

```bash
# Start tracing
sudo ./cuda_sched_trace > trace.csv 2> trace.log &
TRACE_PID=$!

# Run benchmark
cd qwen3.cu
/usr/bin/time -v ./runcu Qwen3-0.6B-FP32.gguf -q "What is eBPF?" -r 1

# Stop tracing
sudo kill -SIGINT $TRACE_PID

# Analyze results
python3 analyze_scheduler_impact.py
```

## Analysis Method

The central analysis compares consecutive CUDA kernel launches:

```text
Launch_i -> [interval] -> Launch_i+1

Group A: Launches with NO context switch in interval (normal flow)
Group B: Launches with context switch in interval (preempted)

Preemption Penalty = median(Group B interval) - median(Group A interval)
```

To compare runs of different lengths, scheduler and IRQ counts are normalized per 1,000 kernel launches:

```text
Sched/1K = (Total Context Switches / Total Kernel Launches) x 1000
IRQ/1K = (Total IRQs / Total Kernel Launches) x 1000
```

Performance impact is reported as:

```text
Slowdown % = (Baseline tok/s - Scenario tok/s) / Baseline tok/s x 100
```

## RQ1: Does CPU Scheduler Significantly Impact GPU Performance in Clean Environments?

The first question is whether scheduler preemption matters when the machine is otherwise clean.

**Experiment design**

- Condition: clean system, no artificial interference
- Metrics: context switch frequency, preemption penalty, total runtime impact
- Analysis: launch-pair comparison with and without context switches

### Results

| Metric | Value |
|--------|-------|
| Total Runtime | 79.5 seconds |
| Kernel Launches | 51,464 |
| Context Switches | 592 (7.44 Hz) |
| OFF-CPU Time | 7.88 ms (0.01%) |

Launch-pair analysis shows that almost every consecutive launch pair is unaffected by context switches:

| Group | Count | Percentage | P50 Interval | P90 Interval | P99 Interval |
|-------|-------|------------|--------------|--------------|--------------|
| No Context Switch | 51,401 | 99.9% | 2 us | 4 us | 4 us |
| With Context Switch | 62 | 0.1% | 15.3 ms | 15.5 ms | 5.0 s |

The median preemption penalty is **15.3 ms**. That is large for the affected pairs, but only 62 pairs were affected.

Tail-latency attribution confirms that most outliers are not caused by scheduler preemption:

| Percentile | Total Outliers | With Context Switch | Attribution |
|------------|----------------|---------------------|-------------|
| P95+ | 2,580 | 62 (2.4%) | 97.6% application |
| P99+ | 515 | 62 (12.0%) | 88.0% application |

The total scheduler impact is:

```text
Impact = Affected Pairs x Penalty = 62 x 15ms = 0.93 seconds
Percentage = 0.93 / 79.5 = 1.2%
```

**Finding:** in clean environments, CPU scheduler impact is minimal at **1.2%**. The vast majority of kernel launch pairs, **99.9%**, are unaffected by context switches. Tail latency mostly comes from application behavior such as token-generation boundaries, not scheduler preemption.

## RQ2: What Is the Impact of IRQ Interrupts on GPU Performance?

The second question is whether IRQs directly interfere with the CPU-side launch path.

**Experiment design**

- Condition: clean system with IRQ tracing enabled
- Metrics: IRQ frequency, duration, type distribution
- Analysis: IRQ time as percentage of total runtime

### Results

| Metric | Value |
|--------|-------|
| Total Runtime | 4.99 seconds |
| Kernel Launches | 125,236 |
| Soft IRQs | 653 events |
| Hard IRQs | 0 events |

Soft IRQ type distribution:

| Type | Count | Total Time | Avg Time | Max Time | Percentage |
|------|-------|------------|----------|----------|------------|
| TIMER | 317 | 0.77 ms | 2.4 us | 30.1 us | 49% |
| RCU | 291 | 0.40 ms | 1.4 us | 17.2 us | 45% |
| NET_RX | 30 | 0.13 ms | 4.5 us | 14.0 us | 4.6% |
| SCHED | 15 | 0.07 ms | 4.9 us | 18.9 us | 2.3% |

Total IRQ impact:

```text
Total IRQ Time: 1.38 ms
Percentage of Runtime: 0.0276%
```

There are real reasons to worry about IRQs: direct handler time, cache pollution, CPU pipeline disruption, and delay accumulation on critical paths. But for this local inference workload, actual IRQ impact is small.

The reason is the workload shape. Qwen3 submits about 950 launches in a burst lasting less than 100 us, so IRQs rarely land inside the burst. Most IRQs happen between bursts during CPU compute. TIMER interrupts dominate and have a small cache footprint. There is little network I/O, so `NET_RX` appears only 30 times, and there are no hard IRQs from NVMe or SSD block-device interrupts.

**Finding:** IRQ impact is negligible for local LLM inference at **0.0276%**. This does not mean IRQs never matter. Distributed training with network communication or on-the-fly data loading can see much higher IRQ impact, estimated around **5-20%**.

## RQ3: How Do Noisy Neighbors Affect GPU Performance?

The third question is the most production-relevant one: what happens when the GPU workload shares a machine with other CPU, network, and disk activity?

**Experiment design**

| Scenario | Interference | Purpose |
|----------|--------------|---------|
| Baseline | None | Reference point |
| Noisy CPU | stress-ng (all cores) | CPU contention |
| Noisy Network | iperf3 (10 streams) | Network IRQ |
| Noisy Disk | fio (4 jobs, randwrite) | Block IRQ |
| Heavy Load | All three combined | Production simulation |
| Optimized | CPU stress + taskset + nice | Mitigation test |

### Results

Normalized metrics per 1,000 kernel launches:

| Scenario | Launches | Sched/1K | Soft IRQ/1K | Hard IRQ/1K | IRQ Time (ms) |
|----------|----------|----------|-------------|-------------|---------------|
| Baseline | 56,882 | 22.8 | 5.8 | 0.0 | 0.62 |
| Noisy CPU | 61,184 | **11,932.8** | 6.4 | 0.0 | 0.33 |
| Noisy Network | 154,394 | 6.0 | 2.7 | 0.0 | 0.92 |
| Noisy Disk | 126,670 | 29.3 | 3.9 | **0.1** | 1.03 |
| **Heavy Load** | **99,424** | **6,044.6** | 2.4 | 0.0 | 0.37 |
| Optimized | 108,984 | 445.2 | 2.8 | 0.0 | 0.71 |

Performance impact:

| Scenario | tok/s | Runtime (s) | Slowdown | Context Switch Increase |
|----------|-------|-------------|----------|------------------------|
| Baseline | 54.77 | 3.00 | - | 1x |
| Noisy CPU | 49.93 | 4.15 | **8.8%** | **524x** |
| Noisy Network | 53.23 | 7.22 | 2.8% | 0.26x |
| Noisy Disk | 54.95 | 5.60 | -0.3% | 1.3x |
| **Heavy Load** | **43.56** | **6.97** | **20.5%** | **265x** |
| Optimized | 53.75 | 5.10 | 1.9% | 19.5x |

### Scenario Analysis

**Noisy CPU (`stress-ng`)** causes the most direct scheduling pressure. Context switches increase **524x**, from 22.8 to 11,932.8 per 1,000 launches, and throughput drops by **8.8%**. The mechanism is simple: the CFS scheduler time-slices between the GPU process and `stress-ng` workers.

**Noisy Network (`iperf3`)** behaves differently. Context switches actually decrease, because the network load changes CPU competition patterns, while soft IRQs rise slightly. Throughput drops only **2.8%**. In this local setup, network I/O primarily shows up as IRQ overhead rather than scheduler pressure.

**Noisy Disk (`fio`)** introduces the first hard IRQs, corresponding to block-device interrupts, but context switches remain low and throughput is effectively unchanged at **-0.3%** slowdown. Disk I/O has little impact on this workload.

**Heavy Load (CPU + Network + Disk)** is the worst case. Throughput drops by **20.5%**, and scheduler events rise to 6,044.6 per 1,000 launches, a **265x** increase over baseline. Interestingly, that is only **50.7%** of the context-switch rate in the Noisy CPU case. The interference sources compete with each other, but their combined effect is still worst overall.

Heavy-load soft IRQ breakdown:

| Type | Count | Total Time | Avg Time |
|------|-------|------------|----------|
| RCU | 213 | 217.4 us | 1.0 us |
| TIMER | 17 | 122.9 us | 7.2 us |
| SCHED | 5 | 33.3 us | 6.7 us |

**Finding:** noisy neighbors significantly affect GPU performance. Combined CPU, network, and disk interference causes **20.5%** degradation. The signatures differ by source: CPU contention increases context switches, network I/O affects IRQ overhead, disk I/O introduces block interrupts with little throughput impact here, and combined load is worst due to cumulative effects.

## RQ4: Can CPU Pinning Effectively Mitigate Scheduler Impact?

The fourth question is whether a simple deployment-level mitigation helps before reaching for a custom scheduler.

**Experiment design**

- Baseline: Noisy CPU scenario with `stress-ng` on all cores
- Optimized: same `stress-ng` load, but the GPU process runs with:
  - `taskset -c 0-3` to pin it to cores 0-3
  - `nice -n -10` to give it higher priority

### Results

| Metric | Noisy CPU | Optimized | Improvement |
|--------|-----------|-----------|-------------|
| Sched/1K | 11,932.8 | 445.2 | **96.3% reduction** |
| tok/s | 49.93 | 53.75 | **7.6% improvement** |
| vs. Baseline | 8.8% slower | 1.9% slower | Significant recovery |

CPU pinning and priority adjustment recover most of the lost throughput. But the optimized case still has 445.2 scheduler events per 1,000 launches, compared with 22.8 in the clean baseline. That is still **19.5x** higher than baseline.

Complete elimination is hard because:

1. `stress-ng` workers may still be scheduled on cores 0-3.
2. System daemons and kernel threads cannot be fully excluded by `taskset`.
3. IRQ affinity may still route interrupts to pinned cores.

For stronger isolation, the next steps are kernel-level isolation and IRQ placement:

```bash
# 1. Use isolcpus kernel parameter (boot time)
isolcpus=4-7 nohz_full=4-7

# 2. Bind GPU process to isolated cores
taskset -c 4-7 ./gpu_app

# 3. Bind IRQs away from GPU cores
echo 0-3 > /proc/irq/*/smp_affinity_list

# 4. Use cgroups for CPU isolation
cgcreate -g cpu:gpu_workload
cgset -r cpuset.cpus=4-7 gpu_workload
cgexec -g cpu:gpu_workload ./gpu_app
```

**Finding:** CPU pinning is highly effective. It reduces context switches by **96.3%** and recovers **7.6%** throughput. But full recovery under heavy load requires deeper isolation such as `isolcpus`, `nohz_full`, cpusets, and IRQ affinity management.

## What the Results Mean

The results point to four practical insights.

First, environment matters. Scheduler impact ranges from **1.2%** in a clean environment to **20.5%** under combined heavy load. Optimizing the scheduler on a quiet dedicated server may not be worth the complexity. On a shared host, it can be the difference between stable and degraded inference.

Second, workload shape matters. Qwen3 has bursty kernel submission, roughly 950 launches in less than 100 us per token burst. That shape makes it resilient to many IRQs because interrupts usually occur between bursts. A different workload with continuous network communication, streaming input, or tighter CPU-GPU handoff might behave differently.

Third, interference sources have distinct signatures:

| Interference | Primary Impact | Secondary Impact |
|--------------|----------------|------------------|
| CPU | Context switches | None |
| Network | IRQ overhead | Slight scheduling |
| Disk | Hard IRQs | Minimal |
| Combined | All of above | Worst overall |

Fourth, simple mitigations work, but only up to a point:

- CPU pinning: very effective, **96%** context-switch reduction
- Priority adjustment: helpful but limited
- Full isolation: requires kernel configuration and IRQ affinity management

## Comparison with Meta's sched_ext Findings

Our results differ from Meta's AI training observations because the workload is different.

| Aspect | Meta (AI Training) | Our Study (LLM Inference) |
|--------|-------------------|---------------------------|
| Primary Issue | Network IRQ (NET_RX) | CPU scheduling |
| IRQ Impact | 5-20% | 0.03% (local inference) |
| Optimization | sched_ext layer | taskset + nice |
| Workload | Distributed training | Single-node inference |

The key difference is communication. Distributed training constantly exchanges data through all-reduce, making `NET_RX` a major bottleneck. Local inference has minimal network I/O, so the dominant issue under noise is CPU scheduling rather than network interrupts.

## Limitations

There are several limits to this study:

1. eBPF tracing itself adds **1-5%** overhead.
2. The tool only supports CUDA, not OpenCL or HIP.
3. The trace does not include GPU-side execution timing, so it cannot directly measure actual kernel runtime.
4. IRQ attribution is limited: the trace cannot always identify which process caused a given IRQ.
5. The experiments use a single GPU and do not cover multi-GPU behavior.

## Practical Recommendations

For production deployments:

| Environment | Recommendation | Expected Benefit |
|-------------|----------------|------------------|
| Dedicated Server | No optimization needed | - |
| Shared Server (light) | taskset + nice | 5-10% improvement |
| Shared Server (heavy) | isolcpus + IRQ affinity | 15-20% improvement |
| Kubernetes | CPU limits + nodeSelector | Varies |

The decision tree is simple:

```text
Is GPU workload latency-sensitive?
├── No -> No optimization needed
└── Yes -> Is server shared?
    ├── No -> Monitor only, optimize if needed
    └── Yes -> How heavy is colocated load?
        ├── Light -> taskset + nice
        └── Heavy -> isolcpus + dedicated cores
```

## Conclusion

CPU scheduling and IRQ handling do not always matter for GPU inference, but they matter under the conditions where production systems often run: shared hosts, background load, and noisy neighbors.

The clean baseline shows minimal overhead: **1.2%** scheduler impact and **0.03%** IRQ impact. But combined CPU, network, and disk interference causes **20.5%** throughput degradation. CPU pinning cuts context switches by **96.3%** and recovers most of the lost performance, but not all of it.

The practical lesson is to measure first. Use tracing to identify whether your workload is scheduler-bound, IRQ-sensitive, or mostly application-limited. Then choose the mitigation that matches the signature: CPU pinning for CPU contention, IRQ affinity for interrupt interference, I/O tuning for block-device pressure, and full CPU isolation when the workload is latency-sensitive and colocated load is heavy.

## References

1. Meta Platforms, Inc. "Accelerating AI Training with sched_ext." Linux Plumbers Conference 2025. <https://lpc.events/event/19/contributions/2039/>
2. NVIDIA Corporation. "CUDA Driver API Reference." <https://docs.nvidia.com/cuda/cuda-driver-api/>
3. Linux Kernel Documentation. "BPF Documentation." <https://www.kernel.org/doc/html/latest/bpf/>
4. stress-ng. "A tool to load and stress a computer system." <https://github.com/ColinIanKing/stress-ng>
5. iperf3. "A TCP, UDP, and SCTP network bandwidth measurement tool." <https://github.com/esnet/iperf>
6. fio. "Flexible I/O Tester." <https://github.com/axboe/fio>
