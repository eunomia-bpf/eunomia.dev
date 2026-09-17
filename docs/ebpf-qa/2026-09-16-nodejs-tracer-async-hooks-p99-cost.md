# Why did enabling an eBPF tracer on a Node.js service triple p99 on the hot path when the eBPF probes are cheap?

**Short answer:** almost all of the cost comes from a JavaScript agent injected into the process, not from the eBPF probes themselves. eBPF-based auto-instrumentation tools such as OBI deliver their "zero-code" Node.js support partly by injecting a JS agent through the Node inspector protocol (`SIGUSR1` + `Runtime.evaluate`), and that agent's `async_hooks` `before` hook fires on **every** JS callback to refresh the active request's trace context, signaling the eBPF layer through a `fs.accessSync` sentinel path. That per-async-operation cost lands on the event loop, not in the kernel, so it is amplified on a hot path that runs many callbacks per request. The eBPF probes (a `uv_fs_access` uprobe plus fd-correlation map reads) are cheap; the injected JS agent and its hook are the expensive part.

## Why the eBPF probes are not the main cost

OBI's docs describe Node.js support explicitly as "uses Node.js async hooks to refresh the active request context before async callbacks and to associate outgoing sockets with incoming sockets." That is two very different layers:

1. **The eBPF layer.** An uprobe on `uv_fs_access` decodes the "sentinel" paths the injected agent emits into eBPF events. This is light: it reads a fixed 22-byte path prefix in one controlled uprobe and writes to a small map.
2. **The in-process JS agent** (`fdextractor.js`), injected via the inspector protocol, wraps `net.Socket` prototypes and installs an `async_hooks` `before` hook. That hook runs before every JS callback.

The cost lives in layer 2, because it executes on the JS event loop's hot path. OBI's source states this overhead directly:

```js
// fdextractor.js (abridged)
if (TRACES_ENABLED) {
  // ALS store holds only incomingFd
  const als = new AsyncLocalStorage();

  net.Server.prototype.emit = function (event, ...args) { /* ... */ };

  net.Socket.prototype.write = function (data, ...rest) {
    const doWrite = () => orig.socketWrite.apply(this, [data, ...rest]);
    const store = als.getStore();
    if (store) {
      const outFd = this._handle && this._handle.fd;
      correlate(store.incomingFd, outFd, this);
    }
    return doWrite();
  };

  // Signal the BPF layer before each async callback so it can restore the
  // correct trace context for this request into traces_ctx_v1.
  let ctxActive = false;
  orig.ctxHook = createHook({
    before() {
      const store = als.getStore();
      if (store && store.incomingFd != null && store.incomingFd >= 0) {
        ctxActive = true;
        try { fs.accessSync(`/dev/null/obi-ctx/${pad4(store.incomingFd)}`); } catch (_) {}
      } else if (ctxActive) {
        ctxActive = false;
        try { fs.accessSync('/dev/null/obi-noreqctx'); } catch (_) {}
      }
    },
  });
  orig.ctxHook.enable();
}
```

The `before` hook fires on every callback; the `fs.accessSync` sentinel only issues a synchronous syscall inside a request context, or on the "request -> no-request" transition edge (the comment is explicit about avoiding a syscall on *every* non-request callback, of which there can be many). But the hook itself — plus the wrapped `net` prototype methods — is a per-request fixed cost on the hot path, amplified on a service that runs many callbacks per request.

## How the `fs.accessSync` sentinel moves JS signals into eBPF

`fs.accessSync` is used here because it is safe inside `async_hooks` callbacks: synchronous fs operations do not create `AsyncWrap` objects and therefore do not re-trigger the hook (that invariant is called out in the `fdextractor.js` comment). The sentinel paths are decoded by the eBPF-side `uv_fs_access` uprobe, and OBI's `nodejs.c` sorts them into categories:

```c
// bpf/generictracer/nodejs.c (abridged)
SEC("uprobe/node:uv_fs_access")
int BPF_KPROBE_GUARDED(obi_uv_fs_access, void *loop, void *req, const char *path)
{
    // the obi nodejs agents (fdextractor.js, spanbridge.js) pass signals to
    // the ebpf layer by invoking uv_fs_access() with a fake path. Formats:
    //  1. fd pair correlation (outgoing -> incoming): /dev/null/obi/<fd1><fd2>
    //  2. async context switch (before-hook, before each JS callback):
    //       /dev/null/obi-ctx/<fd>
    //  3. manual span end (spanbridge.js):            /dev/null/obi-span/<json>
    //  4. no request context:                         /dev/null/obi-noreqctx
    ...
}
```

The decisive fd-pair correlation is written to a *separate* eBPF map:

```c
// bpf/generictracer/nodejs.c:441
static __always_inline int handle_fd_correlation(char *buf, const u64 pid_tgid)
{
    ...
    const u64 key = (pid_tgid << 32) | fd2;
    bpf_map_update_elem(&nodejs_fd_map, &key, &fd1, BPF_ANY);
    return 0;
}
```

This `nodejs_fd_map` (a small LRU hash keyed by `(pid_tgid, outgoing fd)` → incoming fd) is what client-span parenting actually reads; it is not `traces_ctx_v1`.

## What `traces_ctx_v1` is and why "turn the sentinel off" is not trivial

`traces_ctx_v1` is the "current request trace context" map OBI maintains; it is **pinned by name** (`LIBBPF_PIN_BY_NAME`) and is part of an OTEP contract (OTEP 4855, "correlating OBI traces to profiles"):

```c
// bpf/shared/obi_ctx.h
struct {
    __uint(type, BPF_MAP_TYPE_LRU_HASH);
    __type(key, u64);
    __type(value, obi_ctx_info_t);
    __uint(max_entries, 1 << 14);
    __uint(pinning, LIBBPF_PIN_BY_NAME);
} traces_ctx_v1 SEC(".maps");
```

It is written eBPF-side when a server span is saved, and refreshed by the `async_hooks` sentinel before each JS callback. Several consumers read it:

- manual span parenting (`handle_node_span` reads `obi_ctx__get`);
- log enrichment (`logenricher.c` reads `obi_ctx__get` to attach the trace id to a log line);
- Go runtime goroutine hand-off (`go_runtime.c` falls back to `traces_ctx_v1` for the current request context when no runtime task is found);
- external trace/profile correlation tooling (the map is pinned, so an external tool can read it via libbpf).

So the map is an **external surface** (pinned + OTEP contract). Gating the sentinel only on "manual spans + log enrichment enabled" could silently break external correlation integrations that read the pinned map. The maintainer's direction is to *separate* per-callback context refresh from fd-pair correlation, auto-enable the refresh for manual spans and log enrichment, and provide an explicit configuration path for profile correlation — rather than a blanket off.

## How to actually profile and mitigate

1. **Profile the event loop first, not the eBPF side.** Compare the per-callback cost of the `before` hook + `fs.accessSync` sentinel against the eBPF uprobe cost, with tracing enabled and disabled. The p99 jump usually comes from the former, not the kernel.
2. **Establish which consumers actually need `traces_ctx_v1`.** If you are not using manual spans, log enrichment, or external trace/profile correlation, then per-callback context refresh is pure cost for you and is the cleanest thing to turn off. But if an external correlation tool reads the pinned map, turning the sentinel off breaks it; go through the "separate + explicit config" path instead.
3. **Gate on the consumer, not globally.** A metrics-only OBI injection injects the same agent but skips the trace-context machinery (`net` prototype wraps + `async_hooks` `before` hook) entirely, so the per-async-operation cost is not paid.
4. **Watch the in-flight performance work.** A draft PR (#3357) improves the Node sentinel's performance, and the "separate per-callback refresh from fd-pair correlation, gate per consumer" direction is the agreed fix.

## The limitation that decides it

The p99-tripling cost is not in the eBPF probes; it is in the per-async-operation cost of the injected JS agent — the `async_hooks` `before` hook (which maintains `traces_ctx_v1` via the sentinel) plus the `fs.accessSync` signaling it issues, all on the event-loop hot path. The eBPF-side fd-correlation map is cheap; what decides p99 is the in-process JS hook cost paid on every callback. "Turning the sentinel off" is not free: the `traces_ctx_v1` it maintains is a pinned, OTEP-contracted external surface read by several consumers, so the right lever is separating the refresh from fd-pair correlation and gating per consumer, not a global off.

## References

- [OBI docs: Trace context association in OBI (Node.js uses async hooks to refresh the active request context before async callbacks)](https://opentelemetry.io/docs/zero-code/obi/context-propagation/)
- [OBI docs: Distributed traces with OBI (Node.js async-hooks compatibility notes)](https://opentelemetry.io/docs/zero-code/obi/distributed-traces/)
- [OBI docs: Trace-log correlation (log enrichment reads the request trace context)](https://opentelemetry.io/docs/zero-code/obi/trace-log-correlation/)
- [OBI source: `bpf/generictracer/nodejs.c` (`obi_uv_fs_access` uprobe decodes the sentinel paths; `handle_fd_correlation` writes `nodejs_fd_map`; `handle_async_switch` refreshes `traces_ctx_v1`)](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/blob/main/bpf/generictracer/nodejs.c)
- [OBI source: `pkg/internal/nodejs/fdextractor.js` (injected agent: `async_hooks` `before` hook + `fs.accessSync` sentinel)](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/blob/main/pkg/internal/nodejs/fdextractor.js)
- [OBI source: `bpf/shared/obi_ctx.h` (`traces_ctx_v1`, `LIBBPF_PIN_BY_NAME`, OTEP 4855)](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/blob/main/bpf/shared/obi_ctx.h)
- [OTEP 4855: correlating OBI traces to profiles](https://github.com/open-telemetry/opentelemetry-specification/pull/4855)
- [Draft PR #3357: improving Node sentinel performance](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/pull/3357)
- [Node.js docs: `async_hooks` (`createHook`, `before` hook; `async_hooks` is marked experimental, with `AsyncLocalStorage` recommended on performance grounds)](https://nodejs.org/api/async_hooks.html)

## Community discussion today

The monitored window was technically dense. A recurring theme was the cost attribution of Node.js auto-instrumentation: one user reported that enabling tracing on a Node service tripled p99 on a hot route while other routes on the same pods barely moved; profiling showed the eBPF probes were cheap (about 4% of process CPU) and that the injected agent's `async_hooks` `before` hook plus its `fs.accessSync` sentinel dominated event-loop CPU (25–55% on HTTP services, 0.9% on queue workers). Maintainers confirmed client-span parenting uses the fd-pair map and pointed out that `traces_ctx_v1` is also read by external trace/profile correlation, so gating the sentinel only on manual spans + log enrichment could silently break that integration; the direction is to separate per-callback context refresh from fd-pair correlation, auto-enable it per consumer, and leave an explicit configuration path for profile correlation; a draft PR on Node sentinel performance is in flight.

A second thread was OBI's config v2 migration: the `migrate` command is currently all-or-nothing (a behavior-preserving conversion that silently dropping fields would make "valid-looking but materially different"), and maintainers lean toward an explicit `--allow-partial` / `--best-effort` mode that migrates what it can, reports the omitted fields, and is not the default; the related OBI Helm-chart v2 support gap (a v1 field still injected into the v2 config schema) is being tracked as a separate issue. A brief follow-up on an OBI Node.js discovery bug (a supervisor-spawned child process sometimes not being instrumented) also appeared.
