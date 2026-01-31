---
date: 2026-01-29
---

# A Taxonomy of GPU Bugs: 19 Defect Classes for CUDA Verification

## Introduction

GPU programming introduces a distinct class of correctness and performance challenges that differ fundamentally from traditional CPU-based systems. The SIMT (Single Instruction, Multiple Threads) execution model, hierarchical memory architecture, and massive parallelism create unique bug patterns that require specialized verification and detection techniques.

Just as eBPF enables safe, verified extension code to run inside the Linux kernel, [bpftime gpu_ext](https://github.com/eunomia-bpf/bpftime) (The [arxiv](https://arxiv.org/abs/2512.12615), formally name [eGPU](https://dl.acm.org/doi/10.1145/3723851.3726984)) bring eBPF to GPUs—allowing user-defined policy code (for observability, scheduling, or resource control) to be injected into GPU drivers and kernels with **static verification guarantees**. Such a GPU extension framework must ensure that policy code cannot introduce crashes, hangs, data races, or unbounded overhead. A critical concern in modern GPU deployments is **performance interference in multi-tenant environments**: contention for shared resources makes execution time unpredictable. "Making Powerful Enemies on NVIDIA GPUs" studies how adversarial kernels can amplify slowdowns, arguing that performance interference is a *system-level safety* property when GPUs are shared. This motivates treating bounded overhead as a correctness property—not merely an optimization goal.

To build a sound GPU extension verifier, we must first understand what can go wrong. This taxonomy identifies the defect classes a verifier must address, drawing lessons from eBPF's success: restrict the programming model, enforce bounded execution, and verify memory safety before loading. We synthesize findings from static verifiers (GPUVerify, GKLEE, ESBMC-GPU), dynamic detectors (Compute Sanitizer, Simulee, CuSan), and empirical bug studies (Wu et al., ScoRD, iGUARD) into 19 defect classes organized along two dimensions: impact type (Safety, Correctness, Performance) and GPU specificity (GPU-specific, GPU-amplified, CPU-shared). Each entry provides concrete examples, documents detection tools, and offers actionable verification strategies.

<!-- more -->


---

## Taxonomy Overview

Each bug class is categorized along two dimensions:

**Impact Type:**
- **Safety** — Program fails to complete safely (crash, hang, isolation failure, deadlock)
- **Correctness** — Program completes but produces wrong results
- **Performance** — Program works correctly but inefficiently

**GPU Specificity:**
- **GPU-specific** — Unique to GPU/SIMT execution model
- **GPU-amplified** — Exists on CPUs but much more severe on GPUs
- **CPU-shared** — Similar on both platforms

| # | Bug Class | Impact | GPU Specificity |
|---|-----------|--------|-----------------|
| 1 | Barrier Divergence | Safety | GPU-specific |
| 2 | Invalid Warp Sync | Safety | GPU-specific |
| 3 | Insufficient Atomic/Sync Scope | Correctness | GPU-specific |
| 4 | Warp-divergence Race | Correctness | GPU-specific |
| 5 | Uncoalesced Memory Access | Performance | GPU-specific |
| 6 | Control-Flow Divergence | Performance | GPU-specific |
| 7 | Bank Conflicts | Performance | GPU-specific |
| 8 | Block-Size Dependence | Correctness | GPU-specific |
| 9 | Launch Config Assumptions | Correctness | GPU-specific |
| 10 | Missing Volatile/Fence | Correctness | GPU-specific |
| 11 | Shared-Memory Data Races | Correctness | GPU-specific |
| 12 | Redundant Barriers | Performance | GPU-specific |
| 13 | Host ↔ Device Async Races | Correctness | GPU-specific |
| 14 | Atomic Contention | Performance | GPU-amplified |
| 15 | Non-Barrier Deadlocks | Safety | GPU-amplified |
| 16 | Kernel Non-Termination | Safety | GPU-amplified |
| 17 | Global-Memory Data Races | Correctness | CPU-shared |
| 18 | Memory Safety | Safety | CPU-shared |
| 19 | Arithmetic Errors | Correctness/Safety | CPU-shared |

---

## Insights from a Taxonomy of GPU Defects

We conducted a comprehensive study of GPU correctness defects by synthesizing findings from empirical bug analyses ([Wu et al.][21], [iGUARD][7]), static verifiers ([GPUVerify][1], [GKLEE][18], [ESBMC-GPU][10]), and runtime detectors ([Compute Sanitizer][3], [Simulee][19], [ScoRD][4]). Our taxonomy identifies 19 distinct classes of GPU programming defects, uncovering fundamental insights into the unique correctness challenges posed by GPU architectures:

**First**, we observe that *control-flow uniformity* is a foundational correctness requirement for GPU kernels. Non-uniform execution across threads—caused by GPU's SIMT execution model—breaks implicit synchronization assumptions and triggers GPU-specific correctness violations, such as barrier divergence, warp synchronization errors, and subtle warp-divergence races. This insight elevates uniformity from a performance concern to a correctness property that GPU verification frameworks must explicitly enforce.

**Second**, GPU's scoped memory synchronization semantics (e.g., block-scoped atomics, missing fences, volatile misuse) create unique correctness hazards rarely encountered on CPU platforms. Our analysis emphasizes that synchronization primitives' scopes must be explicit, conservative, and verifiable at the kernel level. This requirement is critical for correctness given GPU memory model subtleties.

**Third**, performance interference in GPUs—manifested as uncoalesced accesses, atomic contention, redundant barriers, and bank conflicts—must be viewed as a *safety and isolation* concern rather than mere inefficiency. Our taxonomy reveals how adversarial workloads exploit GPU parallelism to amplify performance issues into denial-of-service attacks in multi-tenant environments. Consequently, bounded overhead must be explicitly enforced as a correctness property in GPU extension frameworks.

**Finally**, our study highlights that liveness (deadlocks, infinite loops) and memory safety (out-of-bounds accesses, temporal violations) are system-level concerns uniquely amplified by GPU parallelism. Unlike traditional CPU environments, GPU kernel hangs or memory violations can trigger hardware-level recovery affecting all tenants. Thus, GPU liveness and memory safety must be explicitly recognized as first-class system-level correctness properties in verifier designs.

Together, these insights not only characterize GPU correctness issues more precisely but also inform principled design requirements for GPU kernel extensibility and verification frameworks—moving beyond traditional CPU-centric correctness towards a GPU-aware system correctness definition.

---

## Canonical bug list

### 1) Barrier Divergence at Block Barriers (`__syncthreads`) — Safety, GPU-specific

#### What it is / why it matters
A block-wide barrier requires *all* threads in the block to reach it. If the barrier is placed under a condition that evaluates differently across threads, some threads wait forever → deadlock / kernel hang. This is treated as a first-class defect in GPU kernel verification (e.g., "barrier divergence" in GPUVerify), and is also one of the main CUDA synchronization bug types characterized/targeted by AuCS/Wu. Note that general control-flow divergence is a performance issue, but barrier divergence is the *specific, critical case* where divergent control flow causes threads to reach a barrier non-uniformly, turning a performance issue into a **liveness/correctness failure** (deadlock).

#### Bug example

```cuda
__global__ void k(float* a) {
  if (threadIdx.x < 16) __syncthreads(); // divergent barrier => UB / deadlock
  a[threadIdx.x] = 1.0f;
}
```

#### Seen in / checked by
* GPUVerify: checking divergence is a core goal ("divergence freedom").([Nathan Chong][1])
* Simulee detects **barrier divergence bugs** in real-world code.([zhangyuqun.github.io][19])
* Wu et al.: explicitly defines barrier divergence and places it under improper synchronization.([arXiv][21])
* Tools like Compute Sanitizer `synccheck` report "divergent thread(s) in block"; Oclgrind can also detect barrier divergence (OpenCL).

#### Checking approach
* **Static check (GPUVerify-style):** prove that each barrier is reached by all threads in the relevant scope, often via uniformity reasoning.([Nathan Chong][1])
* **Dynamic check:** synccheck-style runtime validation, and Simulee-style bug finding.([zhangyuqun.github.io][19])

#### Verification strategy
Make this a *hard* verifier rule: policy code must not contain any block-wide barrier primitive (or any helper that can implicitly behave like a block-wide barrier). If you ever allow barriers in policy code, require **warp-/block-uniform control flow** for any path reaching a barrier (uniform predicate analysis), otherwise reject. Simplest and strongest: **forbid `__syncthreads()` inside policies** — this directly eliminates an entire class of GPU hangs.

---

### 2) Invalid Warp Synchronization (`__syncwarp` mask, warp-level barriers) — Safety, GPU-specific

#### What it is / why it matters
Warp-level sync requires correct participation masks. A common failure is calling `__syncwarp(mask)` where not all lanes that reach the barrier are included in `mask`, or where divergence causes only a subset to arrive.

#### Bug example

```cuda
__global__ void k(int* out) {
  int lane = threadIdx.x & 31;
  if (lane < 16) {
    __syncwarp(0xffffffff);  // only 16 lanes arrive, but mask expects all 32
  }
  out[threadIdx.x] = lane;
}
```

#### Seen in / checked by
* Compute Sanitizer `synccheck` explicitly reports "Invalid arguments" and "Divergent thread(s) in warp" classes for these hazards.([NERSC Documentation][8])
* iGUARD discusses how newer CUDA features (e.g., independent thread scheduling + cooperative groups) create new race/sync hazards beyond the classic model.([Aditya K Kamath][7])

#### Checking approach
* Runtime validation via `synccheck`.
* Static analysis to verify mask correctness at each `__syncwarp` callsite.

#### Verification strategy
If policies can ever emit warp-level sync or cooperative-groups barriers, require a *verifiable* mask discipline: e.g., only `__syncwarp(0xffffffff)` (full mask) or masks proven to equal the active mask at the callsite. Otherwise, simplest is: **ban warp sync primitives entirely** inside policies.

---

### 3) Insufficient Atomic/Sync Scope — Correctness, GPU-specific

#### What it is / why it matters
GPU adds *scope* and memory-model subtleties that don't exist on CPUs. **Scoped races** occur when synchronization/atomics are done at an insufficient scope (e.g., using `atomicAdd_block` when `atomicAdd` with device scope is needed). This is a distinct GPU bug class because scope semantics are unique to CUDA's memory model.

#### Bug example

```cuda
// Scoped race: using block-scope atomic when device-scope is needed
__global__ void k(int* counter) {
  atomicAdd_block(counter, 1);  // only block-scope, may race across blocks
}
```

#### Seen in / checked by
* ScoRD introduces *scoped races* due to insufficient scope and argues this is a distinct bug class.([CSA - IISc Bangalore][4])
* iGUARD further targets races introduced by "scoped synchronization" and advanced CUDA features (independent thread scheduling, cooperative groups).([Aditya K Kamath][7])

#### Checking approach
* **Scope verification:** ensure atomics/sync use sufficient scope for the access pattern.
* Require explicit scope annotations and validate against access patterns.

#### Verification strategy
Treat scope as part of the verifier contract: if policies do atomic/synchronizing operations, require the *strongest* allowed scope (or forbid nontrivial scope usage). Practically: ban cross-block shared global updates unless they're done through a small set of "safe" helpers (e.g., per-SM/per-warp buffers → host aggregation). If policies use scoped atomics, require the scope to be explicit and conservative.

---

### 4) Warp-divergence Race — Correctness, GPU-specific

#### What it is / why it matters
A **warp-divergence race** is a GPU-specific phenomenon where **divergence changes which threads are effectively concurrent**, producing racy outcomes that don't map cleanly to CPU assumptions. SIMT execution order + reconvergence can create subtle concurrency patterns. This is one reason "CPU-style race reasoning" doesn't port directly to GPUs. While control-flow divergence is generally a performance issue (serialized execution paths), warp-divergence race is a **correctness** issue where divergence creates unexpected concurrency patterns leading to data races—same root cause, but different failure modes: perf degradation vs. racy/undefined behavior.

#### Bug example

```cuda
__global__ void k(int* A) {
  int lane = threadIdx.x & 31;
  if (lane < 16) A[0] = 1;      // first half writes
  else           A[0] = 2;      // second half writes
  // outcome depends on SIMT execution + reconvergence
}
```

#### Seen in / checked by
* GKLEE explicitly lists "warp-divergence race" among discovered bug classes.([Lingming Zhang][18])
* Simulee stresses CUDA-aware race definitions and discusses GPU-specific race interpretation constraints (e.g., avoiding false positives due to warp lockstep).([zhangyuqun.github.io][19])

#### Checking approach
* **Verifier rule:** treat "lane-divergent side effects" as forbidden unless proven safe.
* Require that any helper with side effects is guarded by a **warp-uniform predicate** or executed only by a designated lane (e.g., lane0). Then the verifier only needs to prove **uniformity** (or single-lane execution), not full SIMT interleavings.

#### Verification strategy
Enforce warp-uniform control flow for policy side effects. If divergence is unavoidable, force "single-lane execution" patterns where only lane0 performs the side effect. This eliminates warp-divergence races by construction.

---

### 5) Uncoalesced / Non‑Coalesceable Global Memory Access Patterns — Performance, GPU-specific

#### What it is / why it matters
Warp memory coalescing is a GPU-specific performance contract. "Uncoalesced" accesses can cause large slowdowns (memory transactions split into many).

#### Bug example

```cuda
__global__ void k(float* a, int stride) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float x = a[tid * stride];   // stride>1 => likely uncoalesced
  a[tid * stride] = x + 1.0f;
}
```

#### Seen in / checked by
* GPUDrano: "detects uncoalesced global memory accesses" and treats them as performance bugs.([GitHub][2], [CAV17][23])
* GKLEE: reports "non-coalesced memory accesses" as performance bugs it finds.([Lingming Zhang][18])
* GPUCheck: detects "non-coalesceable memory accesses."([WebDocs][11])

#### Checking approach
* **Static analysis (GPUDrano/GPUCheck-style):** analyze address expressions in terms of lane-to-address stride; flag when stride exceeds coalescing thresholds.([CAV17][23])

#### Verification strategy
If you want "performance as correctness," this is a flagship rule: restrict policy memory ops to patterns provably coalesced (e.g., affine, lane-linear indexing with small stride), and/or require warp-level aggregation so only one lane performs global updates. Require map operations to use **warp-uniform keys** or **contiguous per-lane indices** (e.g., `base + lane_id`), not random hashes. If policies must do random accesses, restrict them to **lane0 only**, amortizing the uncoalesced behavior to 1 lane/warp.

---

### 6) Control-Flow Divergence (warp branch divergence) — Performance, GPU-specific

#### What it is / why it matters
SIMT divergence serializes paths within a warp, lowering "branch efficiency" and increasing worst-case overhead. This entry focuses on divergence as a **performance** issue. However, divergence is also the root cause of more severe correctness bugs: barrier divergence (deadlock when barriers are in conditional code) and warp-divergence races (unexpected concurrency patterns leading to data races).

#### Bug example

```cuda
__global__ void k(float* out, float* in) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if ((tid & 1) == 0) out[tid] = in[tid] * 2;
  else                out[tid] = in[tid] * 3;  // divergence within warp
}
```

#### Seen in / checked by
* GPUCheck explicitly targets "branch divergence" as a performance problem arising from thread-divergent expressions.([WebDocs][11])
* GKLEE: "divergent warps" as performance bugs.([Lingming Zhang][18])
* Wu et al.: "non-optimal implementation" includes performance loss causes like branch divergence.([arXiv][21])

#### Checking approach
* **Static taint + symbolic reasoning (GPUCheck-style):** identify conditions dependent on thread/lane id, and prove whether divergence is possible.([WebDocs][11])

#### Verification strategy
Divergence is the *core reason* you can treat performance as correctness. Enforce **warp-uniform control flow** for policies (or at least for any code path that triggers side effects / heavy helpers). If you can't prove uniformity, force "single-lane execution" of policy side effects (others become no-ops) to prevent warp amplification. Put a hard cap on the number of helper calls on any path, to bound the "divergence amplification factor."

---

### 7) Shared-Memory Bank Conflicts — Performance, GPU-specific

#### What it is / why it matters
Bank conflicts are a shared-memory–specific performance pathology: accesses serialize when multiple lanes hit the same bank.

#### Bug example

```cuda
__global__ void k(int* out) {
  __shared__ int s[32*32];
  int lane = threadIdx.x & 31;
  // stride hits same bank pattern (illustrative)
  int x = s[lane * 32];
  out[threadIdx.x] = x;
}
```

#### Seen in / checked by
* GKLEE explicitly lists "memory bank conflicts" among detected performance bugs.([Peng Li's Homepage][12])

#### Checking approach
* **Static heuristic:** classify shared-memory index expressions by lane stride and bank mapping; warn if likely conflict.

#### Verification strategy
If policies use shared scratchpads (e.g., per-block staging), either forbid it or enforce a **conflict-free access pattern** (e.g., contiguous per-lane indexing). Most observability policies can avoid shared memory entirely, turning this into a rule: "no shared-memory accesses in policy." Or simply ban shared-memory indexing by untrusted lane-dependent expressions.

---

### 8) Block-Size Dependence — Correctness, GPU-specific

#### What it is / why it matters
Block-size independence is essential for safe block-size tuning. Kernels that implicitly depend on specific `blockDim` values can produce incorrect results or races when launched with different configurations. This is critical for auto-tuning and portability across GPU generations. This entry focuses on **compile-time hardcoded assumptions** within the kernel code itself (e.g., fixed shared memory sizes, hardcoded reduction strides), distinct from runtime launch configuration assumptions about grid dimensions.

#### Bug example

```cuda
__global__ void reduce(float* out, float* in) {
  __shared__ float s[256];
  int tid = threadIdx.x;
  s[tid] = in[blockIdx.x * blockDim.x + tid];
  __syncthreads();
  // Hardcoded reduction assumes exactly 256 threads
  if (tid < 128) s[tid] += s[tid + 128];  // OOB read if blockDim.x < 256
  __syncthreads();                         // incomplete reduction if blockDim.x > 256
  if (tid < 64) s[tid] += s[tid + 64];
  // ... continues with warp-level reduction ...
  if (tid == 0) out[blockIdx.x] = s[0];
}
// Launched with blockDim.x != 256 => wrong results or crash
```

#### Seen in / checked by
* GPUDrano explicitly includes "block-size independence" analysis.([GitHub][2])

#### Checking approach
* **Static analysis (GPUDrano):** analyze kernel code for implicit blockDim dependencies.
* Require explicit declaration of block-size assumptions in kernel metadata.

#### Verification strategy
Policies should not implicitly assume block shapes unless the verifier can guarantee them. If a policy depends on block-level structure, require declaring it (metadata) and validate at attach time. Add verifier rules that forbid hard-coded assumptions about blockDim unless explicitly declared.

---

### 9) Launch Config Assumptions — Correctness, GPU-specific

#### What it is / why it matters
Many CUDA kernels assume certain launch configurations (e.g., single block, specific grid dimensions). Violating these assumptions leads to incorrect results or races that are hard to diagnose. This entry focuses on **runtime launch configuration assumptions** (gridDim, number of blocks), distinct from compile-time hardcoded block-size dependencies within the kernel code.

#### Bug example

```cuda
__global__ void reduce(float* out, float* in, int n) {
  __shared__ float s[256];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + tid;
  s[tid] = (i < n) ? in[i] : 0.0f;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) s[tid] += s[tid + stride];
    __syncthreads();
  }
  if (tid == 0) {
    *out = s[0];  // BUG: assumes gridDim.x == 1, writes final result directly
  }              // if gridDim.x > 1, multiple blocks race on *out
}
// Called with <<<N/256, 256>>> where N > 256 => data race, wrong result
```

#### Seen in / checked by
* Wu et al.'s discussion of detected bugs includes developer responses that kernels "should not be called with more than one block" and suggests adding assertions like `assert(gridDim.x == 1)`.([arXiv][21])

#### Checking approach
* **Contract checking:** encode launch preconditions (gridDim, blockDim assumptions) and enforce them at runtime or statically.
* Add runtime assertions for grid/block dimension assumptions.

#### Verification strategy
If policy code assumes a particular block/warp mapping (e.g., keys use `threadIdx.x` directly), you can end up with correctness or performance regressions when kernels run under different launch configs. If a policy depends on warp- or block-level structure, require declaring it (metadata) and validate at attach time.

---

### 10) Missing Volatile/Fence — Correctness, GPU-specific

#### What it is / why it matters
GPU code often relies on compiler and memory-model subtleties. GKLEE reports a real-world category: forgetting to mark a shared memory variable as `volatile`, producing stale reads/writes due to compiler optimization or caching behavior. This is a GPU-flavored instance of memory visibility/ordering bugs that can be hard to reproduce.([Lingming Zhang][18])

#### Bug example

```cuda
__shared__ int flag;          // should sometimes be volatile / properly fenced
if (tid == 0) flag = 1;
__syncthreads();
while (flag == 0) { }         // may spin if compiler hoists load / visibility issues
```

#### Seen in / checked by
* GKLEE explicitly lists "forgot volatile" as a discovered bug type.([Lingming Zhang][18])
* Simulee and other tools' race detection can surface some of these issues when they manifest as data races.([zhangyuqun.github.io][19])

#### Checking approach
* **Symbolic exploration (GKLEE-style):** explore memory access orderings and detect stale read scenarios.([Lingming Zhang][18])
* **Pattern-based linting:** flag spin-wait loops on shared memory without volatile or fence.

#### Verification strategy
Avoid exposing raw shared/global memory communication to policies; instead provide **helpers with explicit semantics** (e.g., "atomic increment" or "write once" patterns), and verify policies don't implement ad-hoc synchronization loops. Forbid spin-waiting on shared memory in policy code.

---

### 11) Shared-Memory Data Races (`__shared__`) — Correctness, GPU-specific

#### What it is / why it matters
Threads in a block access on-chip shared memory concurrently; missing/incorrect synchronization causes races. This is a classic CUDA bug class (AuCS/Wu).

#### Bug example

```cuda
__global__ void k(int* g) {
  __shared__ int s;
  int t = threadIdx.x;
  if (t == 0) s = 1;
  if (t == 1) s = 2;   // write-write race on s
  __syncthreads();
  g[t] = s;
}
```

#### Seen in / checked by
* GPUVerify explicitly targets **data-race freedom** and defines intra-group / inter-group races.([Nathan Chong][1])
* GKLEE reports finding **races** (and related deadlocks) via symbolic exploration.([Lingming Zhang][18])
* Simulee detects **data race bugs** in real projects and uses a CUDA-aware notion of race.([zhangyuqun.github.io][19])
* Wu et al. classify **data race** under "improper synchronization" as a CUDA-specific root cause.([arXiv][21])
* Compute Sanitizer `racecheck` is a runtime shared-memory hazard detector.([Shinhwei][6])

#### Checking approach
* **Static verifier route (GPUVerify-style):** enforce "race-free under SIMT" by proving that any two potentially concurrent lanes/threads cannot perform conflicting accesses without proper synchronization.([Nathan Chong][1])
* **Dynamic route (Simulee-style):** instrument / simulate memory accesses and flag conflicting pairs; good for bug-finding and regression tests.([zhangyuqun.github.io][19])

#### Verification strategy
If policies have any shared state, require **warp-uniform side effects** or **single-lane side effects** (e.g., lane0 updates) plus explicit atomics. A conservative verifier rule is: policy code cannot write shared memory except via restricted helpers that are race-safe (e.g., per-warp aggregation).

* **Option A – warp-/block-uniform single-writer rules** (e.g., "only lane 0 updates").
* **Option B – atomic-only helpers** for shared objects.
* **Option C – per-thread/per-warp sharding** (each lane updates its own slot).

---

### 12) Redundant Barriers (unnecessary `__syncthreads`) — Performance, GPU-specific

#### What it is / why it matters
A redundant barrier is a performance-pathology class: removing the barrier **does not introduce a race**, so the barrier was unnecessary overhead.

#### Bug example

```cuda
__global__ void k(int* out) {
  __shared__ int s[256];
  int t = threadIdx.x;
  s[t] = t;             // no cross-thread dependence here
  __syncthreads();      // redundant
  out[t] = s[t];
}
```

#### Seen in / checked by
* Wu et al.: defines "redundant barrier function."([arXiv][21])
* Simulee: detects redundant barrier bugs and reports numbers across projects.([zhangyuqun.github.io][19])
* AuCS: repairs synchronization bugs, including redundant barriers.([Shinhwei][6])
* GPURepair tooling also exists to insert/remove barriers to fix races and remove unnecessary ones.([GitHub][17])

#### Checking approach
* **Static/dynamic dependence analysis:** determine whether any read-after-write / write-after-read across threads is protected by the barrier; if not, barrier is removable (Simulee/AuCS angle).([zhangyuqun.github.io][19])

#### Verification strategy
This supports the "performance = safety" story: even "correct" policies can be unacceptable if they introduce barrier overhead. Since policies should avoid barriers entirely, you can convert this into a simpler rule: **"no barriers in policy,"** and separately "policy overhead must be bounded," eliminating this issue by construction. If helpers include barriers internally, you need cost models or architectural restrictions.

---

### 13) Host ↔ Device Asynchronous Data Races (API ordering bugs) — Correctness, GPU-specific

#### What it is / why it matters
CUDA exposes async kernel launches/memcpy/events; host code can race with device work if synchronization is missing. This is a major real-world bug source in heterogeneous programs and is *not* covered by pure kernel-only verifiers.

#### Bug example

```cpp
int* d_data;
cudaMalloc(&d_data, N * sizeof(int));
kernel<<<grid, block>>>(d_data);
// missing cudaDeviceSynchronize() here
int* h_data = (int*)malloc(N * sizeof(int));
cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost);  // race with kernel
```

#### Seen in / checked by
* CuSan is an open-source detector for "data races between (asynchronous) CUDA calls and the host," using Clang/LLVM instrumentation plus ThreadSanitizer.([GitHub][5])

#### Checking approach
* **Dynamic detection (CuSan-style):** instrument host-side CUDA API calls and detect ordering violations at runtime.

#### Verification strategy
If policies interact with host-visible buffers or involve asynchronous map copies, define a strict **lifetime & ordering contract** (e.g., "policy writes are only consumed after a guaranteed sync point"). For testing, integrate CuSan into CI for host-side integration tests of the runtime/loader.

---

### 14) Atomic Contention — Performance, GPU-amplified

#### What it is / why it matters
Heavy atomic contention is a classic "performance bug that behaves like a DoS" under massive parallelism. Even when correctness is preserved, contention on a single address can cause extreme slowdowns (orders of magnitude). With millions of threads, a single hot atomic can serialize execution and cause tail latency explosion.

#### Bug example

```cuda
__global__ void k(int* counter) {
  // All threads atomically increment the same location => extreme contention
  atomicAdd(counter, 1);
}
// Called with <<<1000, 1024>>> => 1M threads contending on one address
```

#### Seen in / checked by
* GPUAtomicContention: an open-source benchmark suite (2025) explicitly measuring atomic performance under contention and across different **memory scopes** (block/device/system) and access patterns.([GitHub][13])

#### Checking approach
* **Budget-based verification:** limit atomic frequency per warp/block.
* **Benchmarking:** use atomic contention benchmarks to calibrate safe budgets.
* **Static analysis:** identify hot atomic targets and warn about contention risk.

#### Verification strategy
Treat "atomic frequency + contention risk" as a verifier-enforced budget: e.g., allow at most one global atomic per warp, or require warp-aggregated updates. For evaluation, you can reuse the open benchmark suite to calibrate "safe budgets" per GPU generation. Consider requiring warp-level reduction before global atomics to reduce contention by 32x.

---

### 15) Non-Barrier Deadlocks — Safety, GPU-amplified

#### What it is / why it matters
Besides barrier divergence (which is specifically about `__syncthreads` under divergent control flow), SIMT lockstep can create deadlocks in other patterns that are unusual on CPUs: spin-waiting, lock contention within a warp, and named-barrier misuse. Warp-specialized kernels often use **named barriers** or structured synchronization patterns between warps/roles (producer/consumer). Bugs include: (a) spin deadlock due to missing signals, (b) unsafe barrier reuse ("recycling") across iterations, (c) races between producers/consumers.

#### Bug example (spin deadlock)

```cuda
__global__ void k(int* flag, int* data) {
  // Block 0 expects Block 1 to set flag, but no global sync exists
  if (blockIdx.x == 0) while (atomicAdd(flag, 0) == 0) { }  // may spin forever
  if (blockIdx.x == 1) { data[0] = 42; /* forgot to set flag */ }
}
```

#### Bug example (named-barrier misuse, sketch)

```cuda
// Producer writes buffer then signals barrier B
// Consumer waits on B then reads buffer
// Bug: consumer waits on wrong barrier instance / reused incorrectly in loop
```

#### Seen in / checked by
* iGUARD notes that lockstep execution can deadlock if threads within a warp use distinct locks.([Aditya K Kamath][7])
* GKLEE reports finding deadlocks via symbolic exploration of GPU kernels.([Lingming Zhang][18])
* ESBMC-GPU models and checks deadlock too.([GitHub][10])
* WEFT verifies **deadlock freedom**, **safe barrier recycling**, and **race freedom** for producer-consumer synchronization (named barriers).([zhangyuqun.github.io][19])

#### Checking approach
* **Protocol verification (WEFT-style):** for specific synchronization patterns, prove deadlock freedom + race freedom + safe reuse. Model barrier instances across loop iterations and prove safe reuse.([zhangyuqun.github.io][19])
* **Symbolic exploration (GKLEE-style):** explore possible interleavings and detect deadlock states.([Lingming Zhang][18])

#### Verification strategy
Ban blocking primitives in policy code (locks, spin loops, waiting on global conditions). Add a verifier rule: **no unbounded loops / no "wait until" patterns**. If you absolutely need synchronization, force "single-lane, nonblocking" patterns and bounded retries. Policies must not interact with named barriers (no waits, no signals). This aligns with the availability story: policies must not create device stalls.

---

### 16) Kernel Non-Termination / Infinite Loops — Safety, GPU-amplified

#### What it is / why it matters
Infinite loops can hang GPU execution. In practice, non-termination is especially dangerous because GPU preemption/recovery can be coarse.

#### Bug example

```cuda
__global__ void k(int* flag) {
  while (*flag == 0) { }  // infinite loop if flag never set
  // or: while (true) { /* missing break */ }
}
```

#### Seen in / checked by
* CL-Vis explicitly calls out infinite loops (together with barrier divergence) as GPU-specific bug types to detect/handle.([Computing and Informatics][9])

#### Checking approach
* **Static bounds analysis:** prove loop termination or enforce compile-time bounded loops.
* **Runtime watchdog:** timeout-based detection (coarse but practical).

#### Verification strategy
This is where "bounded overhead = correctness" is easiest to justify: enforce a **strict instruction/iteration bound** for policy code (like eBPF on CPU). If policies may contain loops, require compile-time bounded loops only, with conservative upper bounds.

---

### 17) Global-Memory Data Races — Correctness, CPU-shared

#### What it is / why it matters
Races on global memory are a fundamental correctness issue. Unlike shared memory (block-local), global memory is accessible by all threads across all blocks, making races harder to reason about. Many GPU race detectors historically focused on shared memory and ignored global-memory races.

#### Bug example

```cuda
__global__ void k(int* g, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  // Multiple threads may write to same location without sync
  if (tid < n) g[tid % 16] += 1;  // race if multiple threads hit same index
}
```

#### Seen in / checked by
* ScoRD explicitly argues that many GPU race detectors focus on shared memory and ignore global-memory races.([CSA - IISc Bangalore][4])
* iGUARD targets races in global memory introduced by advanced CUDA features.([Aditya K Kamath][7])
* GKLEE reports global memory races via symbolic exploration.([Lingming Zhang][18])

#### Checking approach
* **Static verification:** extend race-freedom proofs to global memory accesses.
* **Dynamic detection:** instrument global memory accesses and track conflicting pairs.

#### Verification strategy
If policies can write to global memory (maps, counters, logs), require either: (1) warp-uniform single-writer rules, (2) atomic-only helpers, or (3) per-thread/per-warp sharding. Ban unprotected global writes from policies.

---

### 18) Memory Safety: Out-of-Bounds / Misaligned / Use-After-Free / Use-After-Scope / Uninitialized — Safety, CPU-shared

#### What it is / why it matters
Classic memory safety includes both **spatial** (OOB, misaligned) and **temporal** (UAF, UAS) violations. Temporal bugs exist on GPUs too: pointers can outlive allocations (host frees while kernel still uses, device-side stack frame returns, etc.).

#### Bug example (OOB)

```cuda
__global__ void k(float* a, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  a[tid + 1024] = 0.0f;   // OOB write
}
```

#### Bug example (Use-After-Scope)

```cuda
__device__ int* bad() {
  int local[8];
  return local;          // returns pointer to dead stack frame (UAS)
}
__global__ void k() {
  int* p = bad();
  int x = p[0];          // UAS read
}
```

#### Seen in / checked by
* Compute Sanitizer `memcheck` precisely detects OOB/misaligned accesses (and can detect memory leaks).([NVIDIA Docs][3])
* Oclgrind reports invalid memory accesses in its simulator.([GitHub][16])
* ESBMC-GPU checks pointer safety and array bounds as part of its model checking.([GitHub][10])
* GKLEE's evaluation includes out-of-bounds global memory accesses as error cases.([Lingming Zhang][18])
* Wu et al.: "unauthorized memory access" appears in root-cause characterization.([arXiv][21])
* cuCatch explicitly targets temporal violations using tagging mechanisms and discusses UAF/UAS detection.([d1qx31qr3h6wln.cloudfront.net][20])
* Guardian: PTX-level instrumentation + interception to fence illegal memory accesses under GPU sharing.([arXiv][22])

#### Checking approach
* **Bounds-check instrumentation (Guardian/cuCatch-style):** insert base+bounds checks (or partition-fencing) around loads/stores.([arXiv][22])
* **Temporal tagging + runtime checks (cuCatch-style):** tag allocations and validate before deref.([d1qx31qr3h6wln.cloudfront.net][20])
* **Static verification (ESBMC-GPU):** model checking for pointer safety and array bounds.([GitHub][10])
* **PTX-level instrumentation (Guardian-style):** insert bounds checks and interception to fence illegal accesses.([arXiv][22])
* **Tagging mechanisms (cuCatch-style):** track allocation ownership and validate access rights.([d1qx31qr3h6wln.cloudfront.net][20])

#### Verification strategy
This is the "classic verifier" portion: keep eBPF-like pointer tracking, bounds checks, and restricted helpers. Easiest for policies is to **ban arbitrary pointer dereferences** and force all memory access through safe helpers (maps/ringbuffers). Ideally: policies cannot allocate/free; all policy-visible objects are managed by the extension runtime and remain valid across policy execution (no UAF/UAS by construction). Also add a testing story: run policy-enabled kernels under Compute Sanitizer memcheck in CI for regression.

#### Multi-tenant implications
In spatial sharing (streams/MPS), kernels share a GPU address space. An OOB access by one application can crash other co-running applications (fault isolation issue). Guardian's motivation explicitly calls out this problem and designs PTX-level fencing + interception as a fix.([arXiv][22]) This directly supports the "availability is correctness" story: if policies run in privileged/shared contexts, you must prevent policy code from generating OOB accesses. Either: (a) only allow map helpers (no raw memory), or (b) instrument policy memory ops with bounds checks (Guardian-style PTX rewriting).

#### Bug example (multi-tenant OOB, conceptual)

```cuda
// Tenant A kernel writes OOB and corrupts Tenant B memory in same context.
```

#### Bug example (Uninitialized Memory)

```cuda
__global__ void k(float* out, float* in, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  // 'in' was cudaMalloc'd but never initialized or memset
  out[tid] = in[tid] * 2.0f;  // reading uninitialized memory
}
```

#### Uninitialized Memory — additional notes
Accessing device global memory without initialization leads to nondeterministic behavior. This is a frequent source of heisenbugs because GPU concurrency amplifies nondeterminism. Compute Sanitizer `initcheck` reports cases where device global memory is accessed without being initialized.([NVIDIA Docs][3]) For policies, require explicit initialization semantics (e.g., map lookup returns "not found" unless initialized; forbid reading uninitialized slots).

---

### 19) Arithmetic Errors (overflow, division by zero) — Correctness/Safety, CPU-shared

#### What it is / why it matters
Arithmetic errors can corrupt keys/indices and cascade into memory safety/perf disasters.

#### Bug example

```cuda
__global__ void k(int* out, int* in, int divisor) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  out[tid] = in[tid] / divisor;  // div-by-zero if divisor == 0

  int idx = tid * 1000000;       // overflow for large tid
  out[idx] = 1;                  // corrupted index => OOB
}
```

#### Seen in / checked by
* ESBMC-GPU explicitly lists arithmetic overflow and division-by-zero among the properties it checks for CUDA programs (alongside races/deadlocks/bounds).([GitHub][10])

#### Checking approach
* **Model checking (ESBMC-GPU):** static verification of arithmetic properties.
* **Lightweight runtime checks:** guard div/mod operations.

#### Verification strategy
Optional but reviewer-friendly: add lightweight verifier checks for div-by-zero and dangerous shifts, and constrain pointer arithmetic (already typical in eBPF verifiers). For "perf correctness," overflow in index computations is a common hidden cause of random/uncoalesced patterns.

---

### Summary: Improper Synchronization as a Root-Cause Category (Wu et al.'s Three-Way Taxonomy)

Wu et al.'s empirical study explicitly groups CUDA-specific synchronization issues into three concrete bug types: **data race**, **barrier divergence**, and **redundant barrier functions**. They also highlight that these often manifest as inferior performance and flaky tests. Simulee is used to find these categories in real projects.([arXiv][21])

This is exactly the "verification story" hook: a GPU extension verifier can claim that policy code cannot introduce these synchronization root causes because:
* no barriers allowed,
* warp-uniform side effects enforced,
* bounded helper calls,
* and a restricted memory model for policies.

---


[1]: https://nchong.github.io/papers/oopsla12.pdf "https://nchong.github.io/papers/oopsla12.pdf"
[2]: https://github.com/upenn-acg/gpudrano-static-analysis_v1.0 "https://github.com/upenn-acg/gpudrano-static-analysis_v1.0"
[3]: https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html "https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html"
[4]: https://www.csa.iisc.ac.in/~arkapravab/papers/isca20_ScoRD.pdf "https://www.csa.iisc.ac.in/~arkapravab/papers/isca20_ScoRD.pdf"
[5]: https://github.com/tudasc/cusan "https://github.com/tudasc/cusan"
[6]: https://www.shinhwei.com/cuda-repair.pdf "https://www.shinhwei.com/cuda-repair.pdf"
[7]: https://akkamath.github.io/files/SOSP21_iGUARD.pdf "https://akkamath.github.io/files/SOSP21_iGUARD.pdf"
[8]: https://docs.nersc.gov/tools/debug/compute-sanitizer/ "https://docs.nersc.gov/tools/debug/compute-sanitizer/"
[9]: https://cai.type.sk/content/2019/1/cl-vis-visualization-platform-for-understanding-and-checking-the-opencl-programs/4318.pdf "https://cai.type.sk/content/2019/1/cl-vis-visualization-platform-for-understanding-and-checking-the-opencl-programs/4318.pdf"
[10]: https://github.com/ssvlab/esbmc-gpu "https://github.com/ssvlab/esbmc-gpu"
[11]: https://webdocs.cs.ualberta.ca/~amaral/thesis/TaylorLloydMSc.pdf "https://webdocs.cs.ualberta.ca/~amaral/thesis/TaylorLloydMSc.pdf"
[12]: https://lipeng28.github.io/papers/ppopp12-gklee.pdf "https://lipeng28.github.io/papers/ppopp12-gklee.pdf"
[13]: https://github.com/KIT-OSGroup/GPUAtomicContention "https://github.com/KIT-OSGroup/GPUAtomicContention"
[14]: https://github.com/mc-imperial/gpuverify "https://github.com/mc-imperial/gpuverify"
[15]: https://github.com/yinengy/CUDA-Data-Race-Detector "https://github.com/yinengy/CUDA-Data-Race-Detector"
[16]: https://github.com/jrprice/Oclgrind "https://github.com/jrprice/Oclgrind"
[17]: https://github.com/cs17resch01003/gpurepair "https://github.com/cs17resch01003/gpurepair"
[18]: https://lingming.cs.illinois.edu/publications/icse2020b.pdf "https://lingming.cs.illinois.edu/publications/icse2020b.pdf"
[19]: https://zhangyuqun.github.io/publications/ase2019.pdf "https://zhangyuqun.github.io/publications/ase2019.pdf"
[20]: https://d1qx31qr3h6wln.cloudfront.net/publications/PLDI_2023_cuCatch_2.pdf "https://d1qx31qr3h6wln.cloudfront.net/publications/PLDI_2023_cuCatch_2.pdf"
[21]: https://arxiv.org/pdf/1905.01833 "https://arxiv.org/pdf/1905.01833"
[22]: https://arxiv.org/pdf/2401.09290 "https://arxiv.org/pdf/2401.09290"
[23]: https://www.cis.upenn.edu/~alur/Cav17.pdf "https://www.cis.upenn.edu/~alur/Cav17.pdf"
