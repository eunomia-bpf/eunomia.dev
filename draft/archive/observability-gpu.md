# How can bpftime do for gpu observability

In short:

- From "what can be observed" perspective, what bpftime can do can all be covered by using Neutrino (fine-grain binary instrumentation tools) combined with kernel eBPF. You can use Neutrino to trace GPU events and record the events with timestamp, and then use kernel eBPF to trace the CPU events and record the events with timestamp, then merge them together. This is because the GPU instrumentation part in bpftime is the same as how Neutrino works.
- The programming model of bpftime `can reduce the storage / computation cost` of this cross layer correlation, you can do it in place with in the same probe/tools in bpftime, but not in other tools. Cross layer tracing is necessary for GPU tools.
- Compare to neutrino and other tools: we are more safe, more expressiveness, and (can be) more efficiency, support better policy or trcaing across multiple layers.

What can be interesting:

- Correlate CPU and GPU events (Or other devices) in the same programming model
- Correlate multiple GPU events

## single GPU

### [launchlate](https://github.com/eunomia-bpf/bpftime/tree/master/example/gpu/launchlate) - Kernel Launch Latency Profiler

Measures the time between `cudaLaunchKernel()` on CPU and actual kernel execution on GPU. Reveals hidden queue delays, stream dependencies, and scheduling overhead that make fast kernels slow in production.

**Use case**: A practitioner running multi-GPU training observes sporadic +80-120ms spikes in iteration time every few minutes. GPU utilization (via Prometheus/NVML) averages look fine, but step times show thick tails—training jobs occasionally stall even when the next kernel should launch immediately after collectives. They run Nsight Systems with lightweight sampling and NVTX ranges in the step loop. The timeline confirms gaps between consecutive kernels, but the cause of delayed kernel launches remains unclear: CPU dataloader? CUDA graph launch path? Python GC? Host threads descheduled by cgroup pressure? CUPTI activity tracing adds overhead and drops records under load; Perfetto traces show host thread churn but don't align precisely with device launch times. After two days of runs, they have hypotheses but no certainty. They need to tie device real-timestamps to the kernel-launch intent seen by the runtime, without enabling heavyweight profiling. A teammate suggests bpftime/`launchlate` because it adds host-side hooks to capture enqueue time (when runtime/NCCL/Triton issues kernel launch) and near-device probes to capture time-to-first-instruction (or ready-to-run). Subtracting the two directly measures per-kernel launch latency (host↔driver↔queue↔device wait), continuously and with low overhead. They enable `launchlate` with minimal arguments (process filter + "record kname, stream_id, grid/block, enqueue_ts, start_ts, end_ts") and let it run in production—no rebuild, no Nsight recording. Real-time logged records show: `enqueue_ts` (host-side intent time), `start_ts` (device actual launch), `Δ_launch = start_ts − enqueue_ts`, `run_ns = end_ts − start_ts`, `kname, stream_id, dev_id, grid, block`, optional tags (tenant/job ID, step ID, NVTX scope if present). Plotting Δ_launch over time reveals bursts where Δ_launch spikes from ~80µs to 60-100ms for kernels after all-reduce. Correlating with host metrics shows: (a) cgroup CPU throttling when other tenant CPUs spike; (b) occasional Python GIL pauses in input pipeline threads. Crucially, device runtime stays stable—the stall happens before the first instruction. Queue-depth counters (if enabled) show backpressure in per-context command queues when host threads miss wakeups. Root cause: not "slow kernels" at all, but host-side launch starvation—next kernel's launch intent delayed by CPU throttling + thread scheduling jitter that accumulates behind the runtime's submission queue. Nsight hinted at gaps; bpftime's paired timestamps make the cause unambiguous. They pin launcher threads to isolated CPUs, raise CPU quota for the training pod in cgroup, and move dataloaders to `numa_node=local` to reduce wakeup jitter. Post-rollout, Δ_launch falls back to ~50-120µs even under load; tail iterations vanish. **Results**: p99.9 step time reduced by 8.7%; zero dropped traces; instrumentation stays on for guardrail monitoring (Δ_launch SLO alerts).

**Example output:**

```
12:34:56 Launch Latency Distribution (Stream 7, 5min window):
latency         : count    distribution
0-50us          : 1847     |****************************************
50-100us        : 2134     |*******************************************
100-500us       : 892      |*******************
500us-1ms       : 45       |*
1-10ms          : 8        |
10-100ms        : 127      |***                                        ⚠️
100ms+          : 15       |                                           ⚠️

Total samples: 5068
p50: 68us  |  p99: 95.4ms  |  p99.9: 118ms
[ALERT] 142 samples >10ms (2.8% of traffic)
```

> example story generated by AI.

```c
BPF_MAP_DEF(BPF_MAP_TYPE_ARRAY, launch_time);

// CPU-side uprobe captures launch time
SEC("uprobe/app:cudaLaunchKernel")
int uprobe_launch(struct pt_regs *ctx) {
    u64 ts_cpu = bpf_ktime_get_ns();  // When did CPU request launch?
    bpf_map_update_elem(&launch_time, &key, &ts_cpu, BPF_ANY);
}

// GPU-side gpuprobe captures execution start
SEC("gpuprobe/_Z9vectorAddPKfS0_Pf")
int gpuprobe_exec() {
    u64 ts_gpu = bpf_get_globaltimer();  // When did GPU actually start?
    u64 *ts_cpu = bpf_map_lookup_elem(&launch_time, &key);

    u64 latency = ts_gpu - *ts_cpu;  // How long did kernel wait in queue?
    u32 bin = get_hist_bin(latency);
    // Update histogram...
}
```

### Copy–Compute Overlap Meter

Measures the temporal overlap between memory transfers (copy engine) and kernel execution (compute engine). Reveals copy-engine starvation and inefficient stream utilization that underutilize GPUs despite appearing busy.

**Use case**: An ML infrastructure engineer notices GPU utilization metrics oscillate between 40-90% during training, with kernels appearing to start late whenever large host-to-device (H2D) data bursts occur. Nsight Systems timeline visualizations confirm the pattern—copy and compute operations serialize instead of overlapping—but the tool produces per-run snapshots unsuitable for continuous production monitoring, and doesn't quantify the overlap numerically across thousands of iterations. They need a scalar metric tracking copy-compute overlap in real time to diagnose whether the problem stems from insufficient stream parallelism, staging buffer contention, or copy-engine saturation. A colleague recommends bpftime's overlap meter because it hooks both CPU-side async memory APIs (`cudaMemcpyAsync`) to capture copy intervals and GPU-side kernel entry/exit tracepoints to capture compute intervals, then correlates them per-stream to compute overlap ratios: `(copy ∩ compute) / (copy ∪ compute)`. They deploy the tool with stream-aware interval tracking (recording `ts_start`, `ts_end`, `stream_id`, `kind={copy|kernel}`, `dev_id`) and let it run continuously in their training infrastructure—no code changes, no snapshot profiling sessions. The live metrics dashboard fed by bpftime's ringbuffer output immediately reveals the issue: overlap ratio hovers at 0.2–0.3 on two critical streams during input shard realignment phases, meaning 70-80% of the time copy and compute engines sit idle waiting for each other. Detailed interval logs show copy operations blocking on the previous kernel's completion due to single-stream serialization, and kernels stalling while waiting for staging buffers refilled by synchronous fallback paths. The overlap meter also exposes periodic "copy starvation" windows where compute idles for 15-30ms waiting for delayed H2D transfers because the dataloader's prefetch depth was set to 1. Root cause: inadequate stream parallelism combined with undersized staging buffers that force synchronous fallbacks under load, compounded by shallow prefetching that fails to hide H2D latency. They redesign the data pipeline to use dedicated copy and compute streams with double-buffered staging areas, increase prefetch depth to 3, and add async stream callbacks to trigger next-batch prefetching immediately upon compute launch. Post-deployment, the overlap meter shows sustained ratios >0.8 even during peak shard realignment, and copy-starvation windows disappear entirely. **Results**: Training throughput increased 25%; GPU utilization stabilized at 85-92%; the overlap metric remains enabled as a production SLO (alert if overlap <0.7 for >10 consecutive batches).

**Example output:**

```
Stream  Window  Copy(ms)  Kernel(ms)  Overlap(ms)  Union(ms)  Ratio   Status
------------------------------------------------------------------------------
3       0-100   145.2     178.6       42.1         281.7      0.149   ⚠️ LOW
5       0-100   152.8     181.2       38.6         295.4      0.131   ⚠️ LOW
3       100-200 148.9     175.3       51.2         273.0      0.188   ⚠️ LOW
5       100-200 151.1     179.8       48.7         282.2      0.173   ⚠️ LOW
[ALERT] Streams 3,5: overlap ratio <0.7 for 200 consecutive windows

--- After pipeline redesign with dedicated streams + double buffering ---

3       0-100   156.2     182.1       148.9        189.4      0.786   ✓ OK
5       0-100   159.8     185.6       155.1        190.3      0.815   ✓ OK
3       100-200 154.7     180.2       147.6        187.3      0.788   ✓ OK
5       100-200 158.3     183.9       153.8        188.4      0.816   ✓ OK
[INFO] All streams: sustained overlap >0.78
```

> example story generated by AI.

```c
struct stream_state {
    u64 copy_start;
    u64 copy_end;
    u64 kernel_start;
    u64 kernel_end;
    u64 overlap_ns;      // accumulated overlap time
    u64 union_ns;        // accumulated union time
    u32 window_count;
    u32 dev_id;
};

BPF_MAP_DEF(BPF_MAP_TYPE_HASH, stream_states);  // key: stream_id
BPF_MAP_DEF(BPF_MAP_TYPE_ARRAY, overlap_ratios); // per-stream overlap metrics for export

// Compute overlap in-kernel: (copy ∩ kernel) / (copy ∪ kernel)
static inline void update_overlap(u64 stream_id, struct stream_state *s) {
    // Intersection: max(0, min(end, end) - max(start, start))
    u64 start = s->copy_start > s->kernel_start ? s->copy_start : s->kernel_start;
    u64 end = s->copy_end < s->kernel_end ? s->copy_end : s->kernel_end;
    u64 intersection = (end > start) ? (end - start) : 0;

    // Union: total span from earliest start to latest end
    u64 total_start = s->copy_start < s->kernel_start ? s->copy_start : s->kernel_start;
    u64 total_end = s->copy_end > s->kernel_end ? s->copy_end : s->kernel_end;
    u64 union_time = total_end - total_start;

    s->overlap_ns += intersection;
    s->union_ns += union_time;
    s->window_count++;

    // Export ratio every N windows for monitoring
    if (s->window_count >= 100) {
        u32 ratio = (u32)((s->overlap_ns * 1000) / s->union_ns);  // ratio * 1000
        bpf_map_update_elem(&overlap_ratios, &stream_id, &ratio, BPF_ANY);
        s->overlap_ns = 0;
        s->union_ns = 0;
        s->window_count = 0;
    }
}

// CPU: memcpy async start - capture copy interval begin
SEC("uprobe/libcudart:cudaMemcpyAsync")
int on_memcpy_start(void *ctx) {
    u64 stream_id = ARG_stream(ctx);
    struct stream_state *s = bpf_map_lookup_elem(&stream_states, &stream_id);
    if (!s) {
        struct stream_state new_state = {0};
        new_state.dev_id = ARG_device(ctx);
        bpf_map_update_elem(&stream_states, &stream_id, &new_state, BPF_ANY);
        s = bpf_map_lookup_elem(&stream_states, &stream_id);
    }
    if (s) s->copy_start = bpf_ktime_get_ns();
    return 0;
}

// CPU: memcpy async end (via cudaStreamSynchronize or callback)
SEC("uprobe/libcudart:cudaStreamSynchronize")
int on_memcpy_end(void *ctx) {
    u64 stream_id = ARG_stream(ctx);
    struct stream_state *s = bpf_map_lookup_elem(&stream_states, &stream_id);
    if (s && s->copy_start > 0) {
        s->copy_end = bpf_ktime_get_ns();
        // If kernel interval exists, compute overlap
        if (s->kernel_end > 0 && s->kernel_start > 0) {
            update_overlap(stream_id, s);
        }
    }
    return 0;
}

// GPU: kernel entry - capture compute interval begin
SEC("gputp/kernel_entry")
int k_entry(dev_ctx *gctx) {
    u64 stream_id = gctx->stream;
    struct stream_state *s = bpf_map_lookup_elem(&stream_states, &stream_id);
    if (!s) {
        struct stream_state new_state = {0};
        new_state.dev_id = gctx->dev_id;
        bpf_map_update_elem(&stream_states, &stream_id, &new_state, BPF_ANY);
        s = bpf_map_lookup_elem(&stream_states, &stream_id);
    }
    if (s) s->kernel_start = bpf_get_globaltimer();
    return 0;
}

// GPU: kernel exit - capture compute interval end and compute overlap
SEC("gputp/kernel_exit")
int k_exit(dev_ctx *gctx) {
    u64 stream_id = gctx->stream;
    struct stream_state *s = bpf_map_lookup_elem(&stream_states, &stream_id);
    if (s && s->kernel_start > 0) {
        s->kernel_end = bpf_get_globaltimer();
        // If copy interval exists, compute overlap
        if (s->copy_end > 0 && s->copy_start > 0) {
            update_overlap(stream_id, s);
        }
    }
    return 0;
}

// User-space: read overlap_ratios map periodically for monitoring/alerting
// Ratio is overlap_ns/union_ns * 1000, so divide by 1000 to get 0.0-1.0 range
```

## Multi-GPU

### Pipeline Bubble Detector

Identifies idle gaps (bubbles) between pipeline-parallel stages in multi-GPU training, revealing where GPUs wait for previous stages and why (P2P wait, host barrier, dependency events). Computes handoff time distributions across pipeline stages in production.

**Use case**: A 4-stage pipeline-parallel training job (4 stages × 4 GPUs) shows uneven device utilizations (70%/50%/75%/55%), indicating inefficient pipeline flow. Nsight Systems timeline shows the execution pattern with visible gaps between stages, but the team needs quantitative analysis of bubble distribution across thousands of micro-batches in production to pinpoint root causes, not just visual inspection of single runs. They need to know: which stage transitions create the longest bubbles, and what causes them—P2P memory copy delays, host-side barrier synchronization, or GPU event dependencies? A teammate suggests bpftime's pipeline bubble detector because it tracks per-stage stream activity by hooking kernel entry/exit on each GPU, computes handoff time (last kernel exit on stage i, GPU i → first kernel entry on stage i+1, GPU i+1) with cross-device timestamp alignment, and correlates bubbles with concurrent P2P operations and event waits. They configure the detector with stage-to-stream mappings (or let it infer from NVTX ranges if present) and deploy across all 4 pipeline stages—no code changes, continuous tracking. Within hours, the bubble distribution data reveals the problem: largest bubbles occur at transitions 1→2 (avg 850µs) and 3→4 (avg 920µs), significantly higher than 0→1 (120µs) and 2→3 (180µs). Detailed logs show these bubbles correlate perfectly with P2P copy completion delays—stage 2 and stage 4 GPUs wait for PCIe transfers from previous stages, compounded by cross-device event hazards where `cudaEventSynchronize` on the receiving GPU blocks until the sending GPU's event fires, adding host-side round-trip latency. The detector also reveals that only 1 micro-batch is in-flight at a time, meaning whenever a bubble occurs, the entire pipeline stalls. Root cause: insufficient pipeline depth (1 micro-batch) combined with synchronous P2P handoffs that serialize stages, plus event-based synchronization introducing host latency. They refactor to use device-side event chains (`cudaStreamWaitEvent` instead of host sync), increase in-flight micro-batches from 1 to 4 to hide P2P latency with overlapping computation, and switch to GPUDirect RDMA where available. Post-deployment, bubble detector shows handoff gaps collapsed: 1→2 drops to 180µs, 3→4 to 210µs, and utilization evens out across all stages (72%/71%/74%/73%). **Results**: Training throughput increased 38%; pipeline bubble time reduced by 76%; utilization variance across stages dropped from 25 percentage points to 3; continuous monitoring enabled for pipeline efficiency SLO (alert if any stage handoff >500µs p95).

**Example output:**

```
Pipeline Bubble Analysis (micro-batches 1000-1500, 5min window):
Stage→Stage  Handoff(µs)      Bubble Distribution                
-----------------------------------------------------------------------------------------
0→1          p50: 115         |***                                      ✓ Minimal (optimal)
             p95: 142         |****
             p99: 168         |*****

1→2          p50: 780         |*************************                ⚠️ P2P copy latency
             p95: 912         |******************************           + Event sync overhead
             p99: 1124        |************************************

2→3          p50: 165         |*****                                    ✓ Acceptable
             p95: 198         |******
             p99: 234         |*******

3→4          p50: 845         |***************************              ⚠️ P2P copy latency
             p95: 1043        |**********************************       + Event sync overhead
             p99: 1287        |******************************************

[ALERT] Stage 1→2: p95 bubble 912µs (threshold: 500µs)
[ALERT] Stage 3→4: p95 bubble 1043µs (threshold: 500µs)

GPU Utilization:
  Stage 0 (GPU 0): 70%  |**********************
  Stage 1 (GPU 1): 50%  |***************        ⚠️ Bubble-limited
  Stage 2 (GPU 2): 75%  |***********************
  Stage 3 (GPU 3): 55%  |*****************      ⚠️ Bubble-limited

--- After pipeline optimization (device events + 4 micro-batches) ---

Pipeline Bubble Analysis (micro-batches 2000-2500, 5min window):
Stage→Stage  Handoff(µs)      Bubble Distribution                 
-----------------------------------------------------------------------------------------
0→1          p50: 108         |***                                      ✓ Optimal
             p95: 134         |****
             p99: 156         |*****

1→2          p50: 165         |*****                                    ✓ Fixed
             p95: 189         |******
             p99: 218         |*******

2→3          p50: 148         |*****                                    ✓ Optimal
             p95: 172         |******
             p99: 195         |******

3→4          p50: 187         |******                                   ✓ Fixed
             p95: 224         |*******
             p99: 251         |********

[INFO] All stage handoffs <500µs p95 ✓

GPU Utilization:
  Stage 0 (GPU 0): 72%  |**********************
  Stage 1 (GPU 1): 71%  |**********************
  Stage 2 (GPU 2): 74%  |***********************
  Stage 3 (GPU 3): 73%  |**********************
[INFO] Utilization variance: 3 percentage points (was 25)
```

> example story generated by AI.

```c
// Per-device stage tracking with in-kernel bubble computation
struct stage_state {
    u64 last_exit_time;      // Last kernel exit on this stage
    u64 stream;              // Stage stream ID
    u32 dev_id;
    u32 stage_id;
    u64 micro_batch_id;
};

struct bubble_stats {
    u64 handoff_gap_ns;      // Bubble duration between stages
    u32 from_stage;
    u32 to_stage;
    u64 micro_batch_id;
    u64 p2p_wait_ns;         // Concurrent P2P wait time (if detected)
    u32 event_sync_detected; // cudaEventSynchronize detected
};

BPF_HASH(stage_streams, u32 /*dev_id*/, u64 /*stream*/, 32);
BPF_HASH(stage_last_exit, u32 /*stage_id*/, struct stage_state, 32);
BPF_ARRAY(bubble_histogram, u64 /*count*/, 1024);  // [stage_pair][bucket]
BPF_RINGBUF(bubble_events, 1 << 20);

// Per-GPU clock sync for cross-device timestamp alignment
BPF_HASH(gpu_clock_offset, u32 /*dev_id*/, s64 /*offset_ns*/, 32);

static inline u64 align_gpu_time(u32 from_dev, u32 to_dev, u64 ts_from) {
    s64 *offset = bpf_map_lookup_elem(&gpu_clock_offset, &to_dev);
    if (!offset) return ts_from;
    return ts_from + *offset;  // Adjust to target GPU's clock domain
}

// Track kernel exit on pipeline stage
SEC("gputp/kernel_exit")
int on_stage_exit(dev_ctx *gctx) {
    u64 *stage_stream = bpf_map_lookup_elem(&stage_streams, &gctx->dev_id);
    if (!stage_stream || gctx->stream != *stage_stream) return 0;

    // This is a pipeline stage kernel exit
    u32 stage_id = gctx->dev_id;  // Simple linear mapping; customize as needed
    struct stage_state s = {
        .last_exit_time = bpf_get_globaltimer(),
        .stream = gctx->stream,
        .dev_id = gctx->dev_id,
        .stage_id = stage_id,
        .micro_batch_id = get_current_microbatch_id(gctx)  // From NVTX or counter
    };
    bpf_map_update_elem(&stage_last_exit, &stage_id, &s, BPF_ANY);
    return 0;
}

// Track kernel entry on next stage - compute bubble in-kernel
SEC("gputp/kernel_entry")
int on_next_stage_entry(dev_ctx *gctx) {
    u64 *stage_stream = bpf_map_lookup_elem(&stage_streams, &gctx->dev_id);
    if (!stage_stream || gctx->stream != *stage_stream) return 0;

    u32 this_stage = gctx->dev_id;
    u32 prev_stage = this_stage - 1;
    if (prev_stage >= 32) return 0;  // Out of bounds

    struct stage_state *prev = bpf_map_lookup_elem(&stage_last_exit, &prev_stage);
    if (!prev) return 0;

    // Align timestamps across GPU clock domains
    u64 prev_exit_aligned = align_gpu_time(prev->dev_id, gctx->dev_id, prev->last_exit_time);
    u64 this_entry = bpf_get_globaltimer();

    // Compute bubble (handoff gap) in-kernel
    u64 bubble_ns = this_entry - prev_exit_aligned;

    // Update histogram bucket
    u32 bucket = (u32)(bubble_ns / 50000);  // 50µs buckets
    if (bucket >= 1024) bucket = 1023;
    u32 hist_idx = (prev_stage * 256) + bucket;
    u64 *count = bpf_map_lookup_elem(&bubble_histogram, &hist_idx);
    if (count) __sync_fetch_and_add(count, 1);

    // Check for concurrent P2P or event sync (simplified detection)
    u32 p2p_active = detect_p2p_wait(gctx);
    u32 event_sync = detect_event_sync(gctx);

    // Export detailed stats only for significant bubbles (>500µs)
    if (bubble_ns > 500000) {
        struct bubble_stats b = {
            .handoff_gap_ns = bubble_ns,
            .from_stage = prev_stage,
            .to_stage = this_stage,
            .micro_batch_id = prev->micro_batch_id,
            .p2p_wait_ns = p2p_active ? estimate_p2p_wait(gctx) : 0,
            .event_sync_detected = event_sync
        };
        bpf_ringbuf_output(&bubble_events, &b, sizeof(b), 0);
    }

    return 0;
}

// User-space: read bubble_histogram for percentile computation
// Read bubble_events for root cause correlation (P2P, event sync)
// Compare across stages to identify problematic transitions
```

### NCCL Straggler & Overlap Detector

Identifies which GPU is the straggler in each collective operation (all-reduce, all-gather, broadcast) and measures actual compute-communication overlap per stream and device. Reveals hidden multi-GPU coordination bottlenecks that standard tools can't track continuously in production.

**Use case**: A training job with 8×A100 GPUs shows periodic steps where throughput collapses by 25%, but the team can't pinpoint which rank causes the slowdown. Nsight Systems confirms NCCL kernels appear in the timeline and provides overlap heatmaps in post-analysis, but it's session-oriented—not designed for always-on per-job telemetry with live straggler identification, and multi-process correlation across ranks isn't turnkey. They need continuous monitoring that identifies stragglers in real-time without heavyweight profiling sessions. A teammate suggests bpftime's NCCL detector because it hooks NCCL CPU APIs (`ncclAllReduce`, `ncclAllGather`, `ncclBroadcast`) to capture collective metadata (operation type, byte count, stream, device ID, CPU timestamp), then matches these to GPU-side NCCL kernel execution using per-device clock synchronization. The tool computes per-GPU phase spans (first kernel entry → last kernel exit) and overlap ratios with concurrent non-NCCL compute kernels. They deploy the detector across all 8 ranks with minimal configuration—no code changes, no session-based profiling, just continuous telemetry export. Within minutes, the live dashboard reveals GPU 3 consistently starts collectives 1.8ms late and finishes last in every all-reduce operation. The detector's per-device timing shows GPU 3's NCCL phase begins after all other GPUs have already started their ring-exchange, forcing the entire collective to wait. Detailed interval logs expose the root cause: GPU 3 runs an extra staging dataloader shard that blocks the main compute stream right before collective entry, delaying the NCCL enqueue. The compute-communication overlap metric for GPU 3 shows 0.15 (vs. 0.82 for other GPUs), confirming the dataloader blocks compute during communication setup. Correlating with process affinity maps reveals the staging shard was mistakenly pinned to GPU 3's memory, causing synchronous H2D transfers that serialize with collective preparation. They migrate the dataloader shard to host-pinned memory shared across all GPUs, eliminating the per-device blocking. Post-deployment, all GPUs show synchronized collective entry within 50µs, the straggler latency drops from 1.8ms to negligible, overlap ratios stabilize at 0.78-0.85 across all devices, and throughput recovers fully. **Results**: Training throughput increased 22%; collective completion time reduced by 28%; straggler variance (p99-p50 start latency across GPUs) dropped from 1.8ms to 45µs; instrumentation stays on for continuous straggler detection (alert if any GPU lags >500µs).

**Example output:**

```
Collective: AllReduce #1847 (Stream 5, 4.2GB, fp16)
GPU   Start(ms)  End(ms)    Duration(ms)  Lag(ms)  Overlap  Status
------------------------------------------------------------------------
0     1250.12    1252.45    2.33          0.00     0.847    ✓ OK
1     1250.09    1252.41    2.32          0.00     0.839    ✓ OK
2     1250.15    1252.48    2.33          0.03     0.851    ✓ OK
3     1251.94    1253.76    1.82          1.82     0.152    ⚠️ STRAGGLER
4     1250.11    1252.43    2.32          0.00     0.843    ✓ OK
5     1250.08    1252.39    2.31          0.00     0.856    ✓ OK
6     1250.13    1252.46    2.33          0.01     0.848    ✓ OK
7     1250.10    1252.42    2.32          0.00     0.841    ✓ OK

[ALERT] GPU 3: straggler detected, starts 1.82ms late (p99: 1.94ms over 200 collectives)
[ALERT] GPU 3: overlap ratio 0.152 << 0.7 threshold

--- After dataloader migration to host-pinned shared memory ---

Collective: AllReduce #2156 (Stream 5, 4.2GB, fp16)
GPU   Start(ms)  End(ms)    Duration(ms)  Lag(ms)  Overlap  Status
------------------------------------------------------------------------
0     1450.23    1452.14    1.91          0.00     0.823    ✓ OK
1     1450.21    1452.11    1.90          0.00     0.817    ✓ OK
2     1450.25    1452.16    1.91          0.02     0.829    ✓ OK
3     1450.24    1452.15    1.91          0.01     0.814    ✓ OK
4     1450.22    1452.13    1.91          0.00     0.826    ✓ OK
5     1450.20    1452.10    1.90          0.00     0.835    ✓ OK
6     1450.23    1452.14    1.91          0.00     0.821    ✓ OK
7     1450.22    1452.12    1.90          0.00     0.819    ✓ OK

[INFO] All GPUs synchronized: max lag 20µs (p99: 45µs)
[INFO] Overlap ratios: 0.78-0.84 across all devices
[INFO] Collective completion time: 1.91ms (was 3.76ms with straggler)
```

> example story generated by AI.

```c
// Per-GPU collective statistics tracked in-kernel
struct nccl_stats {
    u64 start_time;      // First NCCL kernel start on this GPU
    u64 end_time;        // Last NCCL kernel end on this GPU
    u64 collective_id;   // Collective operation ID
    u32 dev_id;
    u32 op_type;
    u64 bytes;
    u64 compute_overlap_ns;  // Time spent in non-NCCL kernels during collective
};

BPF_HASH(nccl_pending, u64 /*stream*/, struct nccl_stats, 8192);
BPF_HASH(nccl_active, u64 /*stream*/, struct nccl_stats, 8192);
BPF_ARRAY(gpu_collective_summary, struct nccl_stats, 256);  // [dev_id][collective_id % 256]

// CPU: hook NCCL API to initiate tracking
SEC("uprobe/libnccl:ncclAllReduce")
int on_nccl_allreduce(void *ctx) {
    u64 stream = ARG_stream(ctx);
    static u64 collective_counter = 0;

    struct nccl_stats s = {
        .collective_id = __sync_fetch_and_add(&collective_counter, 1),
        .dev_id = current_dev(),
        .op_type = OP_ALLREDUCE,
        .bytes = ARG_count(ctx) * ARG_dtype_size(ctx),
        .start_time = 0,
        .end_time = 0,
        .compute_overlap_ns = 0
    };
    bpf_map_update_elem(&nccl_pending, &stream, &s, BPF_ANY);
    return 0;
}

// GPU: first NCCL kernel marks collective start
SEC("gputp/kernel_entry")
int on_nccl_entry(dev_ctx *gctx) {
    struct nccl_stats *pending = bpf_map_lookup_elem(&nccl_pending, &gctx->stream);
    if (pending && pending->dev_id == gctx->dev_id) {
        // NCCL collective start
        pending->start_time = bpf_get_globaltimer();
        bpf_map_update_elem(&nccl_active, &gctx->stream, pending, BPF_ANY);
        bpf_map_delete_elem(&nccl_pending, &gctx->stream);
        return 0;
    }

    // Track concurrent non-NCCL compute for overlap calculation
    struct nccl_stats *active = bpf_map_lookup_elem(&nccl_active, &gctx->stream);
    if (active) {
        // This is a non-NCCL kernel running concurrently - mark for overlap tracking
        gctx->is_compute_overlap = 1;
        gctx->overlap_start = bpf_get_globaltimer();
    }
    return 0;
}

// GPU: kernel exit - compute metrics in-kernel
SEC("gputp/kernel_exit")
int on_nccl_exit(dev_ctx *gctx) {
    struct nccl_stats *active = bpf_map_lookup_elem(&nccl_active, &gctx->stream);
    if (!active) return 0;

    u64 now = bpf_get_globaltimer();

    // If this was a concurrent compute kernel, accumulate overlap time
    if (gctx->is_compute_overlap) {
        active->compute_overlap_ns += (now - gctx->overlap_start);
        gctx->is_compute_overlap = 0;
        return 0;
    }

    // Otherwise, this is NCCL kernel exit - update end time
    active->end_time = now;

    // Heuristic: if this looks like the last NCCL kernel, finalize stats
    if (is_last_nccl_kernel(gctx)) {
        // Compute metrics in-kernel
        u64 duration = active->end_time - active->start_time;
        u64 overlap_ratio = (active->compute_overlap_ns * 1000) / duration;  // ratio * 1000

        // Store per-GPU summary for export
        u32 idx = (active->dev_id * 256) + (active->collective_id % 256);
        struct nccl_stats summary = *active;
        bpf_map_update_elem(&gpu_collective_summary, &idx, &summary, BPF_ANY);

        // Only export summary stats, not every event
        if (active->collective_id % 10 == 0) {  // Export every 10th collective
            struct compact_evt {
                u64 collective_id;
                u32 dev_id;
                u64 duration_ns;
                u32 overlap_ratio;  // * 1000
                u64 bytes;
            } evt = {
                .collective_id = active->collective_id,
                .dev_id = active->dev_id,
                .duration_ns = duration,
                .overlap_ratio = (u32)overlap_ratio,
                .bytes = active->bytes
            };
            bpf_ringbuf_output(&events, &evt, sizeof(evt), 0);
        }

        bpf_map_delete_elem(&nccl_active, &gctx->stream);
    }
    return 0;
}

// User-space: read gpu_collective_summary for all GPUs to identify stragglers
// Compare start_time across devices for same collective_id to find lag
// Export only aggregated metrics, not raw events - much lower overhead
```



