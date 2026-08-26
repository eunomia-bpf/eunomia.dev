# 为什么单次 compare-and-swap 会在竞争下丢失原子 min/max 更新？

**简短回答：** compare-and-swap（CAS）只能保证一次 conditional write 是原子的；它不能保证某次尝试失败后，原来的 condition 仍然成立。因此，精确的 shared minimum 或 maximum 需要 retry loop。每次 CAS 失败后，都要把 candidate 与真正赢得竞争的 current value 重新比较。只要 candidate 仍能改善结果就继续尝试；如果忽略失败，即使每次 write 都是 atomic，最终 reduction 仍可能错误。

对竞争激烈的 BPF summary 来说，correctness 只是第一个问题。许多 CPU 或 GPU thread 同时结束时，对一个 shared cache line 执行 CAS loop 可能很昂贵。条件允许时，应保存 per-CPU、per-thread 或 per-block extrema，稍后再 merge，而不是让所有 sample 都竞争一个 global word。

## 一次失败的 CAS 怎样丢掉真正的最小值

假设当前保存的 minimum 是 `100`，两个 worker 分别观察到 sample `80` 和 `50`：

1. 两个 worker 都 load 到 `100`；
2. 第一个 worker 成功把值从 `100` 改成 `80`；
3. 第二个 worker 请求 CAS 把 `100` 改成 `50`；
4. 因为 stored value 已变成 `80`，这次 CAS 失败；
5. 如果第二个 worker 忽略失败，最后报告的 minimum 就是 `80`。

这里没有 torn write。CAS 正确拒绝覆盖一个已经不再匹配 expected value 的值。Bug 在 reduction algorithm：`50` 从未与新的 current minimum 比较。

Maximum 也会发生同样的 race，只需把 inequality 反过来。无论 operation 最终写成 compiler builtin、BPF `cmpxchg` instruction 还是 target-specific atomic primitive，这个结论都成立。一次 attempt 的 atomicity 不等于整个 read-compare-update operation 已完成。

## Loop 必须重新判断，而不是原样重复

正确的 minimum update 需要维持一条规则：return 前，candidate 要么已经 installed，要么已经与一个不大于它的 current value 比较过。使用 GCC memory-model-aware builtins 的普通 C 代码可以写成：

```c
static __always_inline void atomic_min_u64(u64 *ptr, u64 candidate)
{
    u64 observed = __atomic_load_n(ptr, __ATOMIC_RELAXED);

    while (candidate < observed) {
        if (__atomic_compare_exchange_n(ptr, &observed, candidate,
                                        true,
                                        __ATOMIC_RELAXED,
                                        __ATOMIC_RELAXED))
            return;
        /* 失败时，observed 已更新为 current value。 */
    }
}
```

Maximum 使用 `candidate > observed`。GCC 文档说明，`__atomic_compare_exchange_n` 失败时，会把 `*ptr` 的 current contents 写回 `observed`。正是这个行为让 loop 可以做出正确的下一步判断：

- 如果另一个 worker installed 的值仍会被当前 candidate 改善，就使用新的 expected value 继续尝试；
- 如果另一个 worker 已 installed 相等或更 extreme 的值，就停止，因为当前 candidate 已不能改变 reduction。

示例使用 weak compare-exchange，因为 loop 本身已经能容忍 spurious failure；使用 strong compare-exchange 也正确。只有当 extrema 是独立统计量、不承担发布或保护其他 data 的作用时，`RELAXED` ordering 才足够。如果这次 update 还要让另一个 object 对 reader 可见，就必须单独设计 memory-order contract。

Legacy `__sync_val_compare_and_swap` 会返回 attempt 前 memory 中的值。使用它的代码必须检查 return value，并刷新 expectation：

```c
u64 observed = __atomic_load_n(ptr, __ATOMIC_RELAXED);

while (candidate < observed) {
    u64 actual = __sync_val_compare_and_swap(ptr, observed, candidate);

    if (actual == observed)
        break;              /* 本次 attempt installed 了 candidate。 */
    observed = actual;      /* 与 winner 重新比较。 */
}
```

这些 snippet 表达的是 algorithm，而不是承诺所有 compiler、verifier、offload device 或 userspace BPF runtime 都接受相同 source spelling。必须在 exact target 上 build 并实际 load-test exact program。

## Standard BPF atomics 提供什么、不提供什么

Linux BPF instruction-set specification 定义了 32-bit 与 64-bit atomic operations。标准 arithmetic/bitwise 集合包括 add、OR、AND 与 XOR，另外还有 exchange 和 compare-and-exchange；其中没有 native atomic min 或 max instruction。因此，compiler 或 runtime 可能把 min/max 降低为 CAS loop，暴露额外的 target primitive，或者直接拒绝该 source。

对 BPF `cmpxchg`，expected value 放在 `R0` 中；无论 exchange 成功还是失败，旧的 memory value 都会返回到 `R0`。这足以实现“重新判断”规则，但前提是 generated control flow 真正使用了 returned value。

Kernel BPF、userspace BPF runtime 与 GPU BPF backend 未必共享 verifier rules、helper set、atomic width、address space 或 forward-progress guarantee。尤其是，逻辑上没有静态上限的 retry loop，可能无法被 verifier 证明为 bounded。不能把 loop 随意限制为若干次并静默退出，假装问题已经解决：best-effort loop 耗尽 retry 后仍会丢失 extrema。应改用 target 真正支持的 exact alternative，例如 lock、native min/max primitive 或 sharded reduction。

## 优先降低 contention，再优化 CAS loop

Retrying CAS loop 通常被称为 lock-free，因为竞争者中总有人能 progress；但单个 worker 仍可能 retry 多次。一个 global summary 还会让彼此无关的 worker 争用同一 memory location。更好的 layout 往往能直接消除大部分开销：

- 在 kernel BPF 中，`BPF_MAP_TYPE_PERCPU_ARRAY` 会为每个 CPU 提供独立 value，userspace 再 merge per-CPU minima、maxima、sums 与 counts。
- 在面向 GPU 的 runtime 中，可以按 thread block、warp 或 target 支持的其他 shard 保存 summary，再进行第二阶段 reduction。
- 如果 target 对当前 program 和 map value 支持 `bpf_spin_lock`，可以用一把 lock 保护 multi-field summary；但 helper 与 map restrictions 仍然存在，一把 global lock 也可能无法 scale。
- Histogram 可以继续使用 atomic bucket counters，而 extrema 使用 shards，从而保留 distribution，又避免所有 operation 都竞争同一个 min/max word。

Sharding 会改变 read path：consumer 必须 merge 所有 shard，不能把 missing shard 当成 zero。在 update concurrency 很高时，这通常是值得的 trade-off。

## Initialization 与 snapshot 是另外两个 correctness 问题

对于 non-negative 64-bit duration，常见 empty state 是 `min = UINT64_MAX`、`max = 0`、`count = 0`。Consumer 在展示 extrema 前必须检查 `count`；否则 `0` 无法区分“没有 sample”和真实的 zero-duration sample。Signed value 或其他 domain 需要适合其 range 的 sentinel。

分别对 min、max、sum 和 count 使用 atomic operation，也不会让整个 summary 变成 atomic snapshot。Reader 可能看到新的 count 和旧的 maximum，或者相反。如果这些 fields 必须精确描述同一批 sample，就要用 target 支持的 lock 保护完整 value，或使用 versioned snapshot protocol。如果 approximate live telemetry 可以接受，就应明确记录这条 consistency boundary，不能把它暗示成 transaction。

还要决定 sum overflow 如何处理。Exact extrema 并不能让 wrapped sum 或由它计算的 average 变正确。

## 怎样测试修复

Sequential test 无法暴露 lost-update race。测试必须主动制造 contention，并把结果与独立 reduction 对比：

1. 把 minimum 初始化为高于两个 candidate 的值，同时释放两个 worker，多次重复以覆盖不同 CAS ordering。
2. 生成大量 candidate，并发更新 shared summary，再把 min、max、count 与 sum 和可信的 host-side reduction 比较。
3. 包含 duplicate extrema、已经被支配的 candidate、empty state、zero 与 target 支持的最大值。
4. 在每一个声称支持的 execution backend 上运行。Host build 不能证明 verifier acceptance、GPU atomic lowering 或 target memory behavior。
5. 如果 writer active 时也会读取 summary，应测试并记录暂时不一致的 multi-field snapshot 是否允许。

关键 assertion 不能只是“产生了 CAS”。它必须证明每次失败的 attempt 都会与 winner 的值重试，或者证明 candidate 已不能改善结果。

## 参考资料

- [bpftime GPU timing example review：指出丢失的 min/max update](https://github.com/eunomia-bpf/bpftime/pull/517#pullrequestreview-5028606695)
- [GCC `__atomic` builtins：compare-exchange result、refreshed expected value 与 memory order](https://gcc.gnu.org/onlinedocs/gcc/_005f_005fatomic-Builtins.html)
- [GCC legacy `__sync` builtins：compare-and-swap semantics 与 returned old value](https://gcc.gnu.org/onlinedocs/gcc/_005f_005fsync-Builtins.html)
- [Linux BPF instruction-set specification：atomic operations 与 `BPF_CMPXCHG`](https://docs.kernel.org/bpf/standardization/instruction-set.html)
- [Linux array-map 文档：concurrent in-place update、per-CPU array 与 spin lock](https://docs.kernel.org/bpf/map_array.html)
- [OpenTelemetry eBPF metric-label compatibility change](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/pull/3168)
- [OpenTelemetry GenAI request-extraction fix 与 test coverage](https://github.com/open-telemetry/opentelemetry-python-genai/pull/480)
- [Linux BPF 讨论：在 by-value return 中携带 arena pointer](https://lore.kernel.org/bpf/20260825205412.1320099-1-yonghong.song@linux.dev/T/#t)
- [Linux BPF 讨论：dedicated keyring 与 signed loader](https://lore.kernel.org/bpf/20260826164136.1400997-1-daniel@iogearbox.net/T/#t)
- [Linux BPF 讨论：AF_XDP notifier locking](https://lore.kernel.org/bpf/20260826164606.E6D261F000E9@smtp.kernel.org/T/#t)

## 当日社区讨论

今天通过普通可见浏览器检查了全部 6 个批准社区和 15 个 allowlist 频道或公开页面，所有目标均可访问。选题来自过去 24 小时，因此没有使用七天 fallback。姓名、账号、雇主、workspace 与频道身份、message link、精确时间、私有拓扑、原始日志和可搜索回原讨论的措辞均已删除。没有保留原始 transcript，也没有进行任何社交互动。

### Atomic syntax 并不保证 reduction 正确

最强的未解决问题来自一个 timing-distribution example：大量 execution thread 会更新同一个 summary entry。Atomic addition 保护了 total 与 count，但 minimum 和 maximum 都只做一次 compare-and-exchange attempt。检查 public source 后可以确认，失败的 attempt 会被丢弃，因此一个获胜的 intermediate value 可能遮住更 extreme 的 concurrent sample。修复需要上文的 retry-and-re-evaluate invariant，还要针对 target 验证 generated atomic control flow。

讨论也暴露了 scaling 问题。即使 correctness 已修复，一批 thread 同时结束时，一个 shared summary 仍会成为 contention point。Sharded summary 加 merge 很可能比提升单一 location 的 retry throughput 更容易验证，也更便宜。

### Telemetry compatibility 与 credentialed test 需要明确边界

一个 eBPF observability implementation 完成了变更：不再把 service identity 复制到 default metric labels，但继续把它保留为 resource data，并提供 compatibility opt-in。公开[变更](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/pull/3168)记录了由此带来的 dashboard 与 query migration risk。大家关心的不只是 cleaner schema 是否合理，还包括既有 Prometheus consumer 是否有可观察、可回滚的 migration path。

另一个 instrumentation 讨论询问：怎样在不随意共享 live service credential 的情况下验证 provider-specific behavior。相关公开[变更](https://github.com/open-telemetry/opentelemetry-python-genai/pull/480)展示了可持久复用的部分：把 request/response extraction 隔离到 unit test 中，覆盖多个 supported SDK generation 与 sync/async path，把 credentialed call 留给明确受控的 integration layer。复制的 provider response 可以支持 fixture，但必须经过清理、保留 provenance，并与“live integration 已验证”的证据严格区分。

### Kernel 工作集中在 concurrency 与 trust boundary

Public BPF list 的活跃主题包括：verifier 如何支持 value 中返回的 arena pointer、signed program loading 的 dedicated trust store、AF_XDP locking、concurrent object lifetime、private stack、selective module-BTF loading，以及 userspace-probe 的 page boundary。这些是不同 subsystem，但共同的 design pressure 相似：应明确 ownership、lifetime 与 race 后实际观察到的值，不能只依赖 successful common path。

Program-signing 工作还说明，cryptographic verification 只是 loading policy 的一部分。Key selection、sealing、caller-supplied trust store、signature-size bound、tooling 与 end-to-end test 共同构成 acceptance boundary。

### 安静与重复的 surface 仍完成了检查

若干 project help surface 与一个 networking-focused eBPF channel 在 daily window 内没有实质讨论。一个 scheduler deployment thread 收到 follow-up，但其 underlying question 已在前一篇每日 Q&A 中回答，因此没有重复使用。一个 public practitioner forum 最新的非 pinned post 已经是九天前。这些目标被记录为 accessible and quiet 或 duplicate，没有被误写成 zero activity，也没有被用来制造另一个问题。
