---
date: 2026-05-24
slug: measuring-cpu-noise-in-gpu-inference-with-ebpf
description: 通过 eBPF 追踪 CUDA kernel launch、调度器上下文切换和 IRQ，定量分析 CPU 噪声何时会拖慢 GPU LLM 推理，以及 CPU 绑核能恢复多少吞吐。
---

# CPU 噪声会拖慢 GPU 推理吗：用 eBPF 定量测量调度器与 IRQ 影响

GPU 推理看起来像是 GPU 问题，但 CPU 仍然在关键路径上。CPU 负责准备输入、发起 CUDA kernel launch、管理同步和运行时调用，同时还要和系统任务、中断、其他租户共享核心。如果 CPU 端 launch 路径被打断，GPU 可能会在有算力的情况下等不到新任务。

这篇文章回答一个具体问题：当 LLM 推理跑在 GPU 上时，Linux CPU 调度器和 IRQ 中断到底会带来多少影响？

我们开发了一个基于 eBPF 的追踪工具 `cuda_sched_trace`，用纳秒级时间戳同时记录 CUDA kernel launch、调度器上下文切换、硬中断和软中断事件。然后用 Qwen3 0.6B 推理作为基准，在干净环境和 noisy neighbor 场景下测试：`stress-ng` 制造 CPU 负载，`iperf3` 制造网络负载，`fio` 制造磁盘负载，再加入组合重负载和 CPU 绑核优化场景。

结论很直接：干净环境中，调度器和 IRQ 开销很小；但在类似生产环境的 noisy neighbor 条件下，它们会变得很明显。CPU、网络、磁盘组合干扰让吞吐下降 **20.5%**，而简单的 CPU 绑核和优先级调整能减少 **96.3%** 的上下文切换，并恢复大部分吞吐损失。

<!-- more -->

## 为什么 GPU 推理会受 CPU 调度影响

现代 GPU 工作负载，尤其是 LLM 推理和训练，需要 CPU 和 GPU 紧密协作。CPU 负责：

- 准备输入数据和 kernel 参数
- 通过 CUDA API 发起 GPU kernel
- 管理内存传输和同步

如果这个 CPU 端流程被抢占，就会延迟 GPU kernel 提交。最坏情况下，GPU 有可用算力，但没有新的 work 被提交。

这个问题的动机部分来自 Meta 关于 AI 训练优化的 `sched_ext` 工作。生产环境中常见问题包括“IRQ 抢占重要任务”。网络中断（`NET_RX`/`NET_TX`）和块设备中断会影响大规模分布式训练，自定义调度策略可以让 AI 工作负载提升 5-20%。

但影响大小强依赖工作负载。单节点 LLM 推理不同于带 all-reduce 通信的分布式训练。在投入自定义调度器之前，我们需要先区分：哪些延迟真的是调度器问题，哪些只是应用自身行为。

这项研究有四个目标：

1. 测量 CPU 调度对 GPU kernel launch 的基线影响。
2. 表征 IRQ 干扰模式及其性能成本。
3. 量化 CPU、网络、磁盘和组合 noisy neighbor 的影响。
4. 评估 CPU 绑核和优先级调整的效果。

## 追踪 GPU launch 路径

我们开发了 `cuda_sched_trace`，一个结合 CUDA API uprobe、Linux 调度器 tracepoint 和 IRQ tracepoint 的 eBPF 工具。

### CUDA API 追踪

工具通过 uprobe 附加到 CUDA Driver API 和 Runtime API：

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

### 调度器事件追踪

调度器事件通过 `sched_switch` 捕获，并过滤到 GPU 相关进程：

```c
SEC("tp_btf/sched_switch")
int BPF_PROG(sched_switch, bool preempt, struct task_struct *prev, struct task_struct *next) {
    // Only track if prev or next is a GPU process
    // Record: timestamp, prev/next pid, off-cpu/on-cpu duration
}
```

### IRQ 追踪

硬中断和软中断通过内核 tracepoint 记录：

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

整体数据路径很简单：GPU 应用发起 CUDA 调用；eBPF 程序在内核中观察 CUDA、调度器和 IRQ 事件；事件通过 BPF ring buffer 送到用户态；分析脚本解析 CSV。

```text
┌─────────────────────────────────────────────────────────────────┐
│                         用户空间                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │ GPU 应用    │    │ cuda_sched  │    │ 分析脚本            │  │
│  │ (qwen3.cu)  │    │ _trace      │    │ (Python)            │  │
│  └──────┬──────┘    └──────┬──────┘    └──────────┬──────────┘  │
│         │                  │                       │             │
│         │ CUDA 调用        │ perf_event            │ CSV 解析    │
│         ▼                  ▼                       ▼             │
├─────────────────────────────────────────────────────────────────┤
│                         内核空间                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │ uprobes     │    │ tracepoints │    │ BPF Ring Buffer     │  │
│  │ (CUDA API)  │    │ (sched,irq) │    │ (事件队列)          │  │
│  └─────────────┘    └─────────────┘    └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## 基准和环境

基准测试是基于 `qwen3.cu` 的 Qwen3 0.6B LLM 推理。

| 属性 | 值 |
|------|-----|
| 模型 | Qwen3-0.6B-FP32 |
| 任务 | 单轮问答 |
| 输入 | "What is eBPF?" |
| 输出 | 约 30-50 tokens |
| Kernel 模式 | 批量提交（每个 token 约 950 次 launch） |
| GPU 内存 | 约 3 GB |

这个基准有几个优点：它代表现代 LLM 推理工作负载，混合计算密集型和内存密集型 kernel，有清晰的批量提交模式，并且能用 tok/s 衡量吞吐。

| 组件 | 规格 |
|------|------|
| CPU | 24 核心 |
| GPU | 支持 CUDA 的 NVIDIA GPU |
| 内存 | 足够支持模型 + 系统 |
| 操作系统 | Linux 6.15.11-061511-generic |
| 内核 | 启用 BTF 以支持 CO-RE eBPF |
| CUDA | Driver API + Runtime API |

干扰工具如下：

| 工具 | 用途 | 配置 |
|------|------|------|
| stress-ng | CPU 负载 | `--cpu 0 --cpu-method fft`（所有核心） |
| iperf3 | 网络 I/O | 服务端 + 客户端，10 并行流，60 秒 |
| fio | 磁盘 I/O | `randwrite, bs=4k, iodepth=32, 4 jobs` |

完整实验包含六个场景：

| 场景 | 描述 | 干扰 |
|------|------|------|
| Baseline | 干净环境 | 无 |
| Noisy CPU | CPU 密集型 | stress-ng 在所有核心 |
| Noisy Network | 网络 I/O | iperf3 本地环回 |
| Noisy Disk | 磁盘 I/O | fio 随机写入 |
| Heavy Load | 组合 | CPU + Network + Disk 同时运行 |
| Optimized | CPU 绑核 | stress-ng + taskset -c 0-3 + nice -n -10 |

每次运行的数据采集流程一致：

```bash
# 启动追踪
sudo ./cuda_sched_trace > trace.csv 2> trace.log &
TRACE_PID=$!

# 运行基准测试
cd qwen3.cu
/usr/bin/time -v ./runcu Qwen3-0.6B-FP32.gguf -q "What is eBPF?" -r 1

# 停止追踪
sudo kill -SIGINT $TRACE_PID

# 分析结果
python3 analyze_scheduler_impact.py
```

## 分析方法

核心分析比较连续的 CUDA kernel launch：

```text
Launch_i -> [interval] -> Launch_i+1

Group A: 间隔内无上下文切换的 launch（正常流程）
Group B: 间隔内有上下文切换的 launch（被抢占）

抢占惩罚 = median(Group B 间隔) - median(Group A 间隔)
```

为了比较不同长度的运行，调度器和 IRQ 计数按每 1,000 个 kernel launch 归一化：

```text
Sched/1K = (上下文切换总数 / Kernel Launch 总数) x 1000
IRQ/1K = (IRQ 总数 / Kernel Launch 总数) x 1000
```

性能影响计算为：

```text
性能下降 % = (Baseline tok/s - 场景 tok/s) / Baseline tok/s x 100
```

## RQ1: CPU 调度器在干净环境中是否显著影响 GPU 性能？

第一个问题是：当机器没有人工干扰时，调度器抢占是否真的会影响 GPU 性能。

**实验设计**

- 条件：干净系统，无人工干扰
- 指标：上下文切换频率、抢占惩罚、总运行时间影响
- 分析：对比有/无上下文切换的 launch pair

### 结果

| 指标 | 值 |
|------|-----|
| 总运行时间 | 79.5 秒 |
| Kernel Launch 数 | 51,464 |
| 上下文切换 | 592 (7.44 Hz) |
| OFF-CPU 时间 | 7.88 ms (0.01%) |

Launch pair 分析显示，几乎所有连续 launch pair 都没有受到上下文切换影响：

| 分组 | 数量 | 百分比 | P50 间隔 | P90 间隔 | P99 间隔 |
|------|------|--------|----------|----------|----------|
| 无上下文切换 | 51,401 | 99.9% | 2 us | 4 us | 4 us |
| 有上下文切换 | 62 | 0.1% | 15.3 ms | 15.5 ms | 5.0 s |

有上下文切换的 pair 中，抢占惩罚中位数是 **15.3 ms**。这对被影响的 pair 来说很大，但只有 62 个 pair 被影响。

尾延迟归因也显示，大多数异常值不是调度器抢占造成的：

| 百分位 | 总异常值 | 有上下文切换 | 归因 |
|--------|----------|--------------|------|
| P95+ | 2,580 | 62 (2.4%) | 97.6% 应用问题 |
| P99+ | 515 | 62 (12.0%) | 88.0% 应用问题 |

总调度器影响为：

```text
影响 = 受影响的 pairs x 惩罚 = 62 x 15ms = 0.93 秒
百分比 = 0.93 / 79.5 = 1.2%
```

**发现：** 在干净环境中，CPU 调度器影响很小，约为 **1.2%**。绝大多数 kernel launch pair，即 **99.9%**，不受上下文切换影响。尾延迟主要来自应用行为，例如 token 生成边界，而不是调度器抢占。

## RQ2: IRQ 中断对 GPU 性能的影响有多大？

第二个问题是：IRQ 是否会直接干扰 CPU 端 launch 路径。

**实验设计**

- 条件：启用 IRQ 追踪的干净系统
- 指标：IRQ 频率、持续时间、类型分布
- 分析：IRQ 时间占总运行时间的百分比

### 结果

| 指标 | 值 |
|------|-----|
| 总运行时间 | 4.99 秒 |
| Kernel Launch 数 | 125,236 |
| 软中断 | 653 事件 |
| 硬中断 | 0 事件 |

软中断类型分布：

| 类型 | 次数 | 总时间 | 平均时间 | 最大时间 | 百分比 |
|------|------|--------|----------|----------|--------|
| TIMER | 317 | 0.77 ms | 2.4 us | 30.1 us | 49% |
| RCU | 291 | 0.40 ms | 1.4 us | 17.2 us | 45% |
| NET_RX | 30 | 0.13 ms | 4.5 us | 14.0 us | 4.6% |
| SCHED | 15 | 0.07 ms | 4.9 us | 18.9 us | 2.3% |

总 IRQ 影响：

```text
总 IRQ 时间: 1.38 ms
运行时间占比: 0.0276%
```

从理论上看，IRQ 确实值得担心：直接 handler 执行时间、cache 污染、CPU 流水线扰动，以及关键路径上的延迟累积。但对这个本地推理工作负载来说，实际 IRQ 影响很小。

原因在于工作负载形态。Qwen3 每个 burst 大约提交 950 次 launch，窗口小于 100 us，IRQ 很少落在 burst 内部。大多数 IRQ 发生在 burst 之间，也就是 CPU 计算期间。TIMER 中断占主导，cache footprint 小。网络 I/O 很少，所以 `NET_RX` 只有 30 次；也没有来自 NVMe/SSD 块设备的硬中断。

**发现：** IRQ 对本地 LLM 推理的影响可忽略，只有 **0.0276%**。但这不代表 IRQ 永远不重要。带网络通信或即时数据加载的分布式训练可能有明显更高的 IRQ 影响，估计约 **5-20%**。

## RQ3: Noisy Neighbor 如何影响 GPU 性能？

第三个问题最接近生产环境：当 GPU 工作负载和其他 CPU、网络、磁盘活动共处一台机器时会发生什么。

**实验设计**

| 场景 | 干扰 | 目的 |
|------|------|------|
| Baseline | 无 | 参考点 |
| Noisy CPU | stress-ng（所有核心） | CPU 竞争 |
| Noisy Network | iperf3（10 流） | 网络 IRQ |
| Noisy Disk | fio（4 jobs，randwrite） | 块 IRQ |
| Heavy Load | 三者组合 | 生产环境模拟 |
| Optimized | CPU stress + taskset + nice | 缓解测试 |

### 结果

每 1,000 个 kernel launch 的归一化指标：

| 场景 | Launch 数 | Sched/1K | Soft IRQ/1K | Hard IRQ/1K | IRQ 时间 (ms) |
|------|----------|----------|-------------|-------------|---------------|
| Baseline | 56,882 | 22.8 | 5.8 | 0.0 | 0.62 |
| Noisy CPU | 61,184 | **11,932.8** | 6.4 | 0.0 | 0.33 |
| Noisy Network | 154,394 | 6.0 | 2.7 | 0.0 | 0.92 |
| Noisy Disk | 126,670 | 29.3 | 3.9 | **0.1** | 1.03 |
| **Heavy Load** | **99,424** | **6,044.6** | 2.4 | 0.0 | 0.37 |
| Optimized | 108,984 | 445.2 | 2.8 | 0.0 | 0.71 |

性能影响：

| 场景 | tok/s | 运行时间 (s) | 性能下降 | 上下文切换增加 |
|------|-------|-------------|----------|----------------|
| Baseline | 54.77 | 3.00 | - | 1x |
| Noisy CPU | 49.93 | 4.15 | **8.8%** | **524x** |
| Noisy Network | 53.23 | 7.22 | 2.8% | 0.26x |
| Noisy Disk | 54.95 | 5.60 | -0.3% | 1.3x |
| **Heavy Load** | **43.56** | **6.97** | **20.5%** | **265x** |
| Optimized | 53.75 | 5.10 | 1.9% | 19.5x |

### 按场景分析

**Noisy CPU (`stress-ng`)** 带来最直接的调度压力。上下文切换增加 **524 倍**，从每 1,000 个 launch 22.8 次增加到 11,932.8 次，吞吐下降 **8.8%**。机制很简单：CFS 调度器在 GPU 进程和 `stress-ng` worker 之间分时。

**Noisy Network (`iperf3`)** 的行为不同。上下文切换反而减少，因为网络负载改变了 CPU 竞争模式；软中断略有增加。吞吐只下降 **2.8%**。在这个本地测试中，网络 I/O 主要体现为 IRQ 开销，而不是调度压力。

**Noisy Disk (`fio`)** 首次引入硬中断，对应块设备中断，但上下文切换仍然很低，吞吐几乎不变，性能下降为 **-0.3%**。磁盘 I/O 对这个工作负载影响很小。

**Heavy Load (CPU + Network + Disk)** 是最坏情况。吞吐下降 **20.5%**，调度事件达到每 1,000 个 launch 6,044.6 次，相比 baseline 增加 **265 倍**。有趣的是，这只有 Noisy CPU 场景上下文切换率的 **50.7%**。干扰源之间会相互竞争，但组合效果仍然最差。

Heavy Load 软中断分解：

| 类型 | 次数 | 总时间 | 平均时间 |
|------|------|--------|----------|
| RCU | 213 | 217.4 us | 1.0 us |
| TIMER | 17 | 122.9 us | 7.2 us |
| SCHED | 5 | 33.3 us | 6.7 us |

**发现：** Noisy neighbor 会显著影响 GPU 性能。CPU、网络、磁盘组合干扰导致 **20.5%** 性能下降。不同来源的特征不同：CPU 竞争增加上下文切换，网络 I/O 增加 IRQ 开销，磁盘 I/O 引入块中断但对这里的吞吐影响很小，组合负载由于累积效应最差。

## RQ4: CPU 绑核能否有效缓解调度器影响？

第四个问题是：在使用自定义调度器之前，简单的部署级优化是否有效。

**实验设计**

- 基线：Noisy CPU 场景，即 `stress-ng` 跑满所有核心
- 优化：相同 `stress-ng` 负载，但 GPU 进程使用：
  - `taskset -c 0-3` 绑定到核心 0-3
  - `nice -n -10` 提升优先级

### 结果

| 指标 | Noisy CPU | Optimized | 改善 |
|------|-----------|-----------|------|
| Sched/1K | 11,932.8 | 445.2 | **96.3% 减少** |
| tok/s | 49.93 | 53.75 | **7.6% 提升** |
| vs. Baseline | 8.8% 更慢 | 1.9% 更慢 | 显著恢复 |

CPU 绑核和优先级调整恢复了大部分吞吐损失。但 Optimized 仍有每 1,000 个 launch 445.2 次调度事件，而干净 baseline 是 22.8 次。也就是说，它仍然是 baseline 的 **19.5 倍**。

无法完全消除的原因包括：

1. `stress-ng` worker 仍可能被调度到核心 0-3。
2. 系统守护进程和内核线程无法被 `taskset` 完全排除。
3. IRQ affinity 仍可能把中断路由到绑定核心。

更强隔离需要内核级 CPU 隔离和 IRQ 放置：

```bash
# 1. 使用 isolcpus 内核参数（启动时）
isolcpus=4-7 nohz_full=4-7

# 2. 将 GPU 进程绑定到隔离的核心
taskset -c 4-7 ./gpu_app

# 3. 将 IRQ 绑定到远离 GPU 的核心
echo 0-3 > /proc/irq/*/smp_affinity_list

# 4. 使用 cgroups 进行 CPU 隔离
cgcreate -g cpu:gpu_workload
cgset -r cpuset.cpus=4-7 gpu_workload
cgexec -g cpu:gpu_workload ./gpu_app
```

**发现：** CPU 绑核非常有效。它减少 **96.3%** 的上下文切换，并恢复 **7.6%** 的吞吐。但在重负载下，要完全恢复到 baseline，还需要 `isolcpus`、`nohz_full`、cpuset 和 IRQ affinity 管理。

## 这些结果意味着什么

结果指向四个实践洞察。

第一，环境很重要。调度器影响从干净环境的 **1.2%** 到组合重负载的 **20.5%** 不等。在安静的专用服务器上优化调度器未必值得；但在共享宿主机上，它可能决定推理是否稳定。

第二，工作负载形态很重要。Qwen3 是 bursty kernel submission，每个 token burst 大约在小于 100 us 的窗口内提交 950 次 launch。这种形态对许多 IRQ 有弹性，因为中断通常发生在 burst 之间。换成持续网络通信、流式输入或更紧 CPU-GPU handoff 的工作负载，结果可能不同。

第三，不同干扰源有不同特征：

| 干扰 | 主要影响 | 次要影响 |
|------|----------|----------|
| CPU | 上下文切换 | 无 |
| 网络 | IRQ 开销 | 轻微调度 |
| 磁盘 | 硬中断 | 最小 |
| 组合 | 以上全部 | 最严重 |

第四，简单优化有效，但有上限：

- CPU 绑核：非常有效，减少 **96%** 上下文切换
- 优先级调整：有帮助但有限
- 完全隔离：需要内核配置和 IRQ affinity 管理

## 与 Meta sched_ext 发现的对比

我们的结果和 Meta 在 AI 训练中的观察不同，原因是工作负载不同。

| 方面 | Meta（AI 训练） | 本研究（LLM 推理） |
|------|----------------|-------------------|
| 主要问题 | 网络 IRQ (NET_RX) | CPU 调度 |
| IRQ 影响 | 5-20% | 0.03%（本地推理） |
| 优化方案 | sched_ext layer | taskset + nice |
| 工作负载 | 分布式训练 | 单节点推理 |

关键区别是通信。分布式训练持续通过 all-reduce 交换数据，使 `NET_RX` 成为主要瓶颈。本地推理网络 I/O 很少，因此 noisy neighbor 下的主导问题是 CPU 调度，而不是网络中断。

## 局限性

这项研究有几个限制：

1. eBPF 追踪本身增加 **1-5%** 开销。
2. 工具目前只支持 CUDA，不支持 OpenCL 或 HIP。
3. trace 不包含 GPU 端执行时间，因此无法直接测量实际 kernel runtime。
4. IRQ 归因有限：trace 无法总是识别哪个进程导致了某个 IRQ。
5. 实验使用单 GPU，未覆盖多 GPU 行为。

## 实践建议

生产部署可以参考下面的策略：

| 环境 | 建议 | 预期收益 |
|------|------|----------|
| 专用服务器 | 无需优化 | - |
| 共享服务器（轻负载） | taskset + nice | 5-10% 提升 |
| 共享服务器（重负载） | isolcpus + IRQ 亲和性 | 15-20% 提升 |
| Kubernetes | CPU limits + nodeSelector | 视情况而定 |

决策树如下：

```text
GPU 工作负载对延迟敏感吗？
├── 否 -> 无需优化
└── 是 -> 服务器是共享的吗？
    ├── 否 -> 仅监控，需要时优化
    └── 是 -> 共存负载有多重？
        ├── 轻 -> taskset + nice
        └── 重 -> isolcpus + 专用核心
```

## 结论

CPU 调度和 IRQ 处理并不总是影响 GPU 推理，但它们会在生产系统常见条件下变得重要：共享主机、后台负载和 noisy neighbor。

干净 baseline 显示影响很小：调度器影响 **1.2%**，IRQ 影响 **0.03%**。但 CPU、网络、磁盘组合干扰会导致 **20.5%** 吞吐下降。CPU 绑核能减少 **96.3%** 上下文切换，并恢复大部分性能，但无法完全恢复。

实践经验是先测量。先用 tracing 判断你的工作负载是 scheduler-bound、IRQ-sensitive，还是主要受应用自身限制。然后按特征选择缓解策略：CPU 竞争用 CPU 绑核，IRQ 干扰用 IRQ affinity，块设备压力用 I/O 调度器调优，延迟敏感且共存负载重时再上完整 CPU 隔离。

## 参考文献

1. Meta Platforms, Inc. "Accelerating AI Training with sched_ext." Linux Plumbers Conference 2025. <https://lpc.events/event/19/contributions/2039/>
2. NVIDIA Corporation. "CUDA Driver API Reference." <https://docs.nvidia.com/cuda/cuda-driver-api/>
3. Linux Kernel Documentation. "BPF Documentation." <https://www.kernel.org/doc/html/latest/bpf/>
4. stress-ng. "A tool to load and stress a computer system." <https://github.com/ColinIanKing/stress-ng>
5. iperf3. "A TCP, UDP, and SCTP network bandwidth measurement tool." <https://github.com/esnet/iperf>
6. fio. "Flexible I/O Tester." <https://github.com/axboe/fio>
