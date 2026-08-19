# Why can libbpf load a BPF object slowly on a host with many kernel modules?

**Short answer:** libbpf's kernel-module BTF path is lazy but coarse-grained. It does not read every module BTF for every BPF object. However, once an object needs a type, kfunc, or BTF-based attach target that is not resolved from `vmlinux` BTF, current libbpf iterates the kernel's BTF objects and materializes every module BTF it finds. On a machine with many BTF-enabled modules, that userspace discovery and parsing work can dominate startup even if the object ultimately needs only one module.

Qualifying a tracing section as `SEC("fentry/mymod:foo")` does not currently avoid that cost. The qualifier narrows the later target lookup; it does not tell `load_module_btfs()` which BTF objects to skip.

An RFC patch series proposes an explicit module-name allowlist in `bpf_object_open_opts`. That is the right shape for callers that already know their dependencies, but it is not a released libbpf API at the time of writing. Production code should first measure the exact open/load boundary, keep the current fallback, and adopt the selector only after it lands in the libbpf version actually shipped with the application.

## Where the time goes

Module BTF is not a second copy of all kernel types. Linux can encode a module as split BTF: the module contains its module-specific types and refers back to base `vmlinux` BTF for shared types. This keeps the representation compact, but a loader still has to discover the module BTF, obtain its file descriptor and metadata, parse it against the base BTF, and retain enough state to use it for relocation or target lookup.

Current `load_module_btfs()` follows this broad sequence:

1. Iterate BTF object IDs with `bpf_btf_get_next_id()`.
2. Obtain a file descriptor and metadata for each object.
3. Ignore userspace BTF objects and `vmlinux` itself.
4. Parse every remaining kernel-module BTF using the object's `vmlinux` BTF as its base.
5. Save every parsed module so later CO-RE, kfunc, and attach-target resolution can search it.

The function caches the result inside one `bpf_object`, so it does not repeat the scan for each relocation in that object. The expensive boundary is still all-or-nothing once module lookup is required. This distinction explains why two small objects can have very different startup times: an object resolved entirely from `vmlinux` can avoid the module path, while one module-defined target can pull in the full module set.

The RFC report gives two useful, but not independently reproduced, data points. In one on-demand mobile workload with 93 module BTFs, total loading exceeded 300 ms and module-BTF work accounted for about 69% of it. In a separate scaling test, selecting the needed BTF reduced the reported skeleton open-and-load time from 65.1 ms to 38.7 ms with 300 modules. These numbers show why the path matters; they are not a performance promise for another kernel, object, CPU, storage stack, or page-cache state.

## Prove that module BTF is the bottleneck

Do not infer the cause from the number of loaded modules alone. Map creation, CO-RE relocation, verifier work, BPF token setup, and program loading can all contribute to the same startup interval.

First split the skeleton lifecycle instead of measuring only `open_and_load()`:

```c
struct example_bpf *skel;

start("open");
skel = example_bpf__open();
stop("open");
if (!skel)
        return 1;

start("load");
if (example_bpf__load(skel))
        return 1;
stop("load");
```

Here `start()` and `stop()` stand for a monotonic-clock measurement in the application. Run enough repetitions to report a distribution rather than one sample. Record the exact kernel build, libbpf build, BPF object, module set, privileges, and whether the run is cold or warm. The kernel's libbpf documentation defines open and load as separate lifecycle phases and supports `LIBBPF_LOG_LEVEL=debug` when more loader detail is needed.

Then inventory the BTF files exposed by the running kernel with read-only commands:

```bash
find /sys/kernel/btf -mindepth 1 -maxdepth 1 -type f -printf '%f\n' | sort
find /sys/kernel/btf -mindepth 1 -maxdepth 1 -type f ! -name vmlinux | wc -l
```

The basenames are also the safest starting point for module names in a future selector. Do not derive the allowlist from a package manifest: the runtime set and names are what libbpf sees.

For a convincing comparison, keep the application object and libbpf constant and use matched test boots with different intentional module sets. Do not unload arbitrary production modules merely to run a benchmark. If load time scales with the available module BTF set while verifier and map costs stay stable, and debug output shows that module BTF resolution is entered, the attribution is much stronger.

## What the proposed selector changes

The RFC adds two fields to `bpf_object_open_opts`:

```c
const char **kmod_btf_names;
size_t kmod_btf_names_cnt;
```

When the pointer is absent, the proposal preserves current behavior. When it is present, libbpf copies and deduplicates the requested names, still inspects BTF metadata to learn each object's name, skips parsing names outside the set, and stops iterating after all requested module BTFs have been found. That removes most unrelated parsing and storage, though it does not guarantee constant-time lookup: enumeration still continues until the requested names are encountered.

With the RFC API, a generic loader would look like this:

```c
static const char *needed_kmods[] = {
        "mymod",
};

LIBBPF_OPTS(bpf_object_open_opts, opts,
        .kmod_btf_names = needed_kmods,
        .kmod_btf_names_cnt = 1,
);

obj = bpf_object__open_file("example.bpf.o", &opts);
```

This snippet describes the RFC, not an API available in current released headers. The current upstream `bpf_object_open_opts` ends at `bpf_token_path`; compiling the new initializers against it will fail. A generated skeleton can pass the same open options only when its generator and linked libbpf both support the final API.

The caller must provide the complete dependency set for all programs that will be loaded from the object. Omitting a module that supplies a CO-RE target, kfunc, or BTF attach target can turn a speed improvement into a resolution failure. Include optional autoload programs in that audit, and test both the success path and deliberate omission of each required module.

## Why the section-name qualifier is insufficient

For a BTF tracing target, a module-qualified section tells lookup which module should win when resolving a function. That is useful for correctness, especially if several modules expose similar names. In current libbpf, however, the module BTF collection is populated before the lookup loop applies that qualifier. The search is narrower; the preparation work is not.

This separation is worth preserving conceptually:

- ELF section syntax describes a program and its attachment target.
- CO-RE and extern records describe type and symbol dependencies.
- Loader options describe how userspace should discover and prepare runtime metadata.

Trying to make a section string double as a complete object-level dependency manifest would miss kfunc and relocation dependencies elsewhere in the same object. An explicit loader option can cover the whole object and can be validated independently.

## Safe deployment choices before and after the RFC

Until a selective API is merged and present in your shipped libbpf, the portable behavior is the existing full module-BTF scan whenever module resolution is needed. The practical choices are therefore operational:

- Keep long-lived loader processes and loaded BPF objects alive when the application model permits it, instead of repeatedly paying on-demand startup cost.
- Separate optional programs into different objects when that genuinely removes module dependencies from a latency-sensitive path. Confirm with phase timing; file splitting alone is not proof.
- Build a version-pinned vendor backport only if startup latency is a hard requirement and you can maintain the ABI, selftests, and fallback. Do not expose draft field names as a stable application contract.
- After upstream support exists, feature-gate the selector against the exact libbpf headers and library used at build and runtime. Preserve the default path for unknown dependency sets.

Before enabling selection, add tests for a valid required module, a missing required module, duplicate names, an unrelated loaded module, and a host with no module BTF support. Also preserve diagnostic logging. The RFC's selftests cover these boundaries, while review has already highlighted that a custom test print callback must not swallow unrelated libbpf errors.

Finally, keep the claim narrow. Selective module BTF loading can reduce userspace preparation time. It does not reduce verifier complexity, make a missing BTF target valid, change the kernel's module set, or guarantee a particular end-to-end startup latency.

## References

- [RFC v3 cover letter: selectively loading kernel-module BTFs](https://lore.kernel.org/bpf/20260819090426.267-1-zhaofuyu@vivo.com/)
- [RFC v3 implementation: module selection through `bpf_object_open_opts`](https://lore.kernel.org/bpf/150459b74dba9ea3f0bd133f97039f998e25830ada87b803fcbc6c77b17bcf93@mail.kernel.org/)
- [RFC v3 selftests for valid, missing, duplicate, and skipped module names](https://lore.kernel.org/bpf/5cec664e8218aab8a7304ab755e65410dfa07ff5ef4f19a48ee3ce1baed80aff@mail.kernel.org/)
- [Current libbpf source: `load_module_btfs()`](https://github.com/torvalds/linux/blob/master/tools/lib/bpf/libbpf.c)
- [Current libbpf header: `bpf_object_open_opts`](https://github.com/libbpf/libbpf/blob/master/src/libbpf.h)
- [Linux kernel documentation: split BTF and module BTF](https://docs.kernel.org/bpf/btf.html#btf-base-section)
- [Linux kernel documentation: libbpf lifecycle and logging](https://docs.kernel.org/next/bpf/libbpf/libbpf_overview.html)
- [OpenTelemetry Helm chart proposal: couple the default Prometheus exporter to a scrapeable Service](https://github.com/open-telemetry/opentelemetry-helm-charts/pull/2360)
- [AF_XDP patch series: metadata layout and zero-copy-path fixes](https://lore.kernel.org/bpf/20260819160535.1472459-1-sdf@fomichev.me/)
- [Linux kernel documentation: AF_XDP](https://docs.kernel.org/networking/af_xdp.html)
- [RFC: BPF-driven proactive memory-cgroup reclaim](https://lore.kernel.org/bpf/cover.1787120833.git.zhuhui@kylinos.cn/)
- [RFC: track scalar equality across the low 32 bits in the verifier](https://lore.kernel.org/bpf/5f1b726cd88f2261f70a5aa99f94ea1434c078c1.camel@gmail.com/)

## Community discussion today

Today's ordinary visible-browser review covered all 6 approved communities and all 15 allowlisted channels or public pages. Every target was accessible. The selected question and several public development threads fell inside the 24-hour window, so the seven-day fallback was not used. Names, accounts, employers, community and channel identities, closed-chat message links, exact times, private topology, raw logs, and searchable chat phrasing have been removed. No raw transcript was retained.

### Startup cost is becoming part of the BPF loader contract

The strongest thread treated BPF loading latency as an application property, not merely a development inconvenience. On-demand agents, short-lived tools, and health-critical startup paths may pay metadata-discovery costs before their first event is observed. The proposed module selector is therefore valuable only if its dependency contract is testable: callers need a deterministic way to declare required module BTFs, a clear failure for omissions, and a backward-compatible full-scan mode.

The same discussion exposed an important measurement habit. “Small BPF object” does not imply “small loader workload.” Runtime BTF inventory, CO-RE candidates, kfuncs, and attach targets determine the metadata work. Teams were primarily concerned with separating that cost from verifier time and avoiding optimizations that silently make optional probes disappear.

### Metrics need a consumer lifecycle, not just an enabled endpoint

An observability deployment discussion connected memory growth to a Prometheus exporter that expires metric children while serving scrapes. If no consumer scrapes the endpoint, enabling the exporter can retain series without the expected eviction cycle. A chart change proposes disabling the default exporter unless its managed Service is enabled.

That condition is useful but not equivalent to “someone can scrape this pod.” Pod discovery, annotations, or an externally managed Service can provide a real consumer without the chart-managed Service. The robust configuration model separates three states: exporter enabled, discovery object managed by this chart, and scrape health observed. Upgrade tests should cover all three discovery styles plus the no-consumer case, and memory tests should verify actual series eviction rather than infer it from YAML rendering. The change remains under review, so current users should inspect their effective configuration and scrape path before adopting the proposed default.

### AF_XDP fixes again centered on ABI layout and path parity

The public networking review carried fixes for TX metadata layout across ABIs, honoring metadata in the zero-copy path, and documenting a driver's behavior for oversized frames. Together they show why AF_XDP validation cannot stop after one successful packet path. Userspace descriptors, metadata headroom, copy mode, zero-copy mode, and driver ownership rules form one contract.

A useful regression matrix crosses 32- and 64-bit layout, copy and zero-copy operation, metadata enabled and disabled, and the drivers actually deployed. A fix in the generic path does not prove driver parity, and a working zero-copy driver does not prove that copy mode interprets the same offsets. These were active patches, not a guarantee in an already released kernel.

### BPF policy proposals are moving into memory management and verifier precision

One active proposal would let a sleepable BPF program trigger an asynchronous memory-cgroup reclaim pass based on runtime signals, using semantics modeled on `memory.reclaim`. The key design question is not whether BPF can call reclaim, but where policy feedback, rate limits, target selection, and observability live. An out-of-band reclaim trigger must be evaluated separately from synchronous protection hooks, and one pass must not be mistaken for fulfillment of a requested byte target.

Another verifier discussion addressed relationships that exist only across the low 32 bits of two scalar registers. Losing that relation after zero- or sign-extension can reject compiler-generated programs even when a later branch proves the value range. Preserving more information may improve compiler portability, but state-equivalence and pruning rules must remain sound. The community's concern was therefore precision without accidentally merging states that differ in a safety-relevant way.

The project-focused support and chat surfaces were otherwise quiet, limited to onboarding or automated project notices, or had no substantive technical item in the daily window. The public forum's newest submission was older than the window. No inaccessible target was described as quiet.
