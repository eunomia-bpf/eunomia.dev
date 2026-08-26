# Why can a single compare-and-swap lose atomic min/max updates under contention?

**Short answer:** compare-and-swap (CAS) makes one conditional write atomic; it does not guarantee that the condition is still true after a failed attempt. An exact shared minimum or maximum therefore needs a retry loop. After every failed CAS, compare the candidate with the value that actually won the race. Retry only while the candidate still improves that value. If failure is ignored, every individual write may be atomic while the final reduction is still wrong.

For a heavily contended BPF summary, correctness is only the first concern. A CAS loop on one shared cache line can become expensive, especially when many CPU or GPU threads finish together. When possible, keep per-CPU, per-thread, or per-block extrema and merge them later instead of forcing every sample through one global word.

## How one failed CAS loses the true minimum

Suppose the stored minimum is `100`, and two workers observe samples `80` and `50`:

1. both workers load `100`;
2. the first worker successfully changes the value from `100` to `80`;
3. the second worker asks CAS to change `100` to `50`;
4. that CAS fails because the stored value is now `80`; and
5. if the second worker ignores the failure, the reported minimum remains `80`.

There was no torn write. CAS correctly refused to overwrite a value that no longer matched the expected value. The bug is in the reduction algorithm: the `50` sample was never compared with the new current minimum.

The same race applies to a maximum with the inequalities reversed. It also applies whether the operation is spelled as a compiler builtin, a BPF `cmpxchg` instruction, or a target-specific atomic primitive. Atomicity of one attempt is not completion of the whole read-compare-update operation.

## The loop must re-evaluate, not merely repeat

A correct minimum update maintains this rule: before returning, the candidate was either installed, or it was compared with a value that is already no greater. In ordinary C using GCC's memory-model-aware builtins, the shape is:

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
        /* On failure, observed now contains the current value. */
    }
}
```

For maximum, use `candidate > observed`. GCC documents that a failed `__atomic_compare_exchange_n` writes the current contents of `*ptr` back into `observed`. That behavior is what lets the loop make the right next decision:

- if another worker installed a value that the candidate still improves, try again with the new expected value;
- if another worker installed an equal or more extreme value, stop because this candidate can no longer change the reduction.

The example uses weak compare-exchange because the loop already tolerates a spurious failure. Strong compare-exchange is also valid. `RELAXED` ordering is sufficient only when the extrema are independent statistics and do not publish or protect other data. If the update is also meant to make another object visible, its memory-order contract must be designed separately.

Legacy `__sync_val_compare_and_swap` returns the value that was in memory before the attempt. Code using it needs to inspect that return value and refresh its expectation:

```c
u64 observed = __atomic_load_n(ptr, __ATOMIC_RELAXED);

while (candidate < observed) {
    u64 actual = __sync_val_compare_and_swap(ptr, observed, candidate);

    if (actual == observed)
        break;              /* This attempt installed candidate. */
    observed = actual;      /* Re-evaluate against the winner. */
}
```

These snippets describe the algorithm, not a promise that every compiler, verifier, offload device, or userspace BPF runtime accepts the same source spelling. Build and load-test the exact program on the exact target.

## What standard BPF atomics do and do not provide

The Linux BPF instruction-set specification defines 32-bit and 64-bit atomic operations. Its standard arithmetic/bitwise set contains add, OR, AND, and XOR, plus exchange and compare-and-exchange. It does not define a native atomic min or max instruction. A compiler or runtime may therefore lower a min/max operation to a CAS loop, expose an additional target primitive, or reject the source.

For BPF `cmpxchg`, the expected value is supplied in `R0`; the old memory value is returned in `R0` whether the exchange succeeds or fails. That is enough to implement the re-evaluation rule, but only if the generated control flow uses the returned value.

Kernel BPF, a userspace BPF runtime, and a GPU BPF backend do not necessarily share the same verifier rules, helper set, atomic widths, address spaces, or forward-progress guarantees. In particular, a logically unbounded retry loop may not be accepted by a verifier that cannot prove a bound. Do not “fix” that by silently stopping after an arbitrary number of retries: a bounded best-effort loop still loses extrema after exhaustion. Use an exact alternative supported by the target, such as a lock, a native min/max primitive, or sharded reduction.

## Reduce contention before tuning the CAS loop

A retrying CAS loop is lock-free in the common sense that one competing update makes progress, but an individual worker can retry many times. A single global summary also makes unrelated workers contend on the same memory location. Better layouts often remove most of that cost:

- On kernel BPF, a `BPF_MAP_TYPE_PERCPU_ARRAY` gives each CPU its own value. Userspace can merge the per-CPU minima, maxima, sums, and counts.
- On a GPU-oriented runtime, keep summaries per thread block, warp, or another target-supported shard, then perform a second reduction.
- If the target supports `bpf_spin_lock` for the relevant program and map value, one lock can protect a multi-field summary. Its helper and map restrictions still apply, and one global lock may scale poorly.
- A histogram can often keep atomic bucket counters while extrema use shards. This preserves the distribution without routing every operation through one min/max word.

Sharding changes the read path: the consumer must merge every shard and must not mistake a missing shard for zero. It is usually a worthwhile trade when update concurrency is high.

## Initialization and snapshots are separate correctness problems

For non-negative 64-bit durations, a common empty state is `min = UINT64_MAX`, `max = 0`, and `count = 0`. The consumer must check `count` before presenting either extreme; otherwise `0` is ambiguous between “no samples” and a real zero-duration sample. Signed values or other domains need sentinels appropriate to their range.

Atomic min, max, sum, and count operations also do not make the entire summary an atomic snapshot. A reader may observe a new count and an older maximum, or vice versa. If those fields must describe precisely the same sample set, protect the whole value with a supported lock or use a versioned snapshot protocol. If approximate live telemetry is acceptable, document that consistency boundary rather than implying a transaction.

Also decide how sum overflow is handled. Exact extrema do not make a wrapped sum or derived average correct.

## How to test the fix

A sequential test cannot expose the lost-update race. Use tests that create contention and compare the result with an independent reduction:

1. Initialize the minimum above two candidate values, release two workers together, and repeat enough times to exercise both CAS orderings.
2. Generate many candidate values, update the shared summary concurrently, and compare min, max, count, and sum with a trusted host-side reduction.
3. Include duplicate extrema, already-dominated candidates, the empty state, zero, and the largest supported value.
4. Run on every execution backend that claims support. A host build does not prove verifier acceptance, GPU atomic lowering, or target memory behavior.
5. If the summary is read while writers are active, test and document whether a temporarily inconsistent multi-field snapshot is allowed.

The key assertion is not merely that CAS was emitted. It is that every failed attempt either retries against the value that won or proves that the candidate no longer improves the result.

## References

- [bpftime GPU timing example review identifying the lost min/max update](https://github.com/eunomia-bpf/bpftime/pull/517#pullrequestreview-5028606695)
- [GCC `__atomic` builtins: compare-exchange result, refreshed expected value, and memory order](https://gcc.gnu.org/onlinedocs/gcc/_005f_005fatomic-Builtins.html)
- [GCC legacy `__sync` builtins: compare-and-swap semantics and returned old value](https://gcc.gnu.org/onlinedocs/gcc/_005f_005fsync-Builtins.html)
- [Linux BPF instruction-set specification: atomic operations and `BPF_CMPXCHG`](https://docs.kernel.org/bpf/standardization/instruction-set.html)
- [Linux array-map documentation: concurrent in-place updates, per-CPU arrays, and spin locks](https://docs.kernel.org/bpf/map_array.html)
- [OpenTelemetry eBPF metric-label compatibility change](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/pull/3168)
- [OpenTelemetry GenAI request-extraction fix and test coverage](https://github.com/open-telemetry/opentelemetry-python-genai/pull/480)
- [Linux BPF discussion: returning arena pointers by value](https://lore.kernel.org/bpf/20260825205412.1320099-1-yonghong.song@linux.dev/T/#t)
- [Linux BPF discussion: a dedicated keyring and signed loaders](https://lore.kernel.org/bpf/20260826164136.1400997-1-daniel@iogearbox.net/T/#t)
- [Linux BPF discussion: AF_XDP notifier locking](https://lore.kernel.org/bpf/20260826164606.E6D261F000E9@smtp.kernel.org/T/#t)

## Community discussion today

Today's ordinary visible-browser review covered all 6 approved communities and all 15 allowlisted channels or public pages. Every target was accessible. The selected question came from the 24-hour window, so the seven-day fallback was not used. Names, accounts, employers, workspace and channel identities, message links, exact times, private topology, raw logs, and searchable wording have been removed. No raw transcript was retained, and no social interaction was performed.

### Atomic syntax did not guarantee a correct reduction

The strongest unresolved issue was a timing-distribution example in which many execution threads update one summary entry. Atomic addition protected the total and count, but minimum and maximum each made only one compare-and-exchange attempt. Review of the public source showed that a failed attempt was discarded, so a winning intermediate value could hide a more extreme concurrent sample. The correction needs the retry-and-re-evaluate invariant above, plus target-specific validation of the generated atomic control flow.

The discussion also exposed a scaling question. Even after it is correct, one shared summary becomes a contention point when a wave of threads completes together. Sharded summaries followed by a merge are likely to be both easier to validate and cheaper than maximizing retry throughput on one location.

### Telemetry compatibility and credentialed tests need explicit boundaries

An eBPF observability implementation completed a change that stops copying service identity into default metric labels while preserving it as resource data and providing a compatibility opt-in. The public [change](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/pull/3168) documents the resulting dashboard and query migration risk. The broader concern was not whether the cleaner schema is desirable, but whether existing Prometheus consumers have an observable, reversible migration path.

A separate instrumentation discussion asked how contributors can validate provider-specific behavior without casually sharing live service credentials. The related public [change](https://github.com/open-telemetry/opentelemetry-python-genai/pull/480) demonstrates the durable part of the answer: isolate response/request extraction in unit tests, cover multiple supported SDK generations and sync/async paths, and reserve credentialed calls for an explicitly controlled integration layer. A copied provider response can support a fixture, but it must be sanitized, provenance-aware, and kept distinct from proof that the live integration works.

### Kernel work concentrated on concurrency and trust boundaries

The public BPF list was active around verifier support for arena pointers returned inside values, a dedicated trust store for signed program loading, AF_XDP locking, concurrent object lifetime, private stacks, selective module-BTF loading, and userspace-probe page boundaries. These are different subsystems, but the recurring design pressure was similar: make ownership, lifetime, and the exact value observed after a race explicit rather than relying on a successful common path.

Program-signing work also showed that cryptographic verification is only one part of a loading policy. Key selection, sealing, caller-supplied trust stores, signature-size bounds, tooling, and end-to-end tests all belong to the same acceptance boundary.

### Quiet and duplicate surfaces were still checked

Several project help surfaces and one networking-focused eBPF channel had no substantive discussion in the daily window. One scheduler deployment thread received a follow-up, but the underlying question had already been answered in the previous daily Q&A and was not reused. A public practitioner forum's newest non-pinned post was nine days old. These targets were recorded as accessible and quiet or duplicate, not converted into zero activity and not used to manufacture another question.
