# Why can a profile sample link to a trace span that was never sampled, and what has to be enabled before the link exists at all?

Because the link is a pair of identifiers attached to the sample, not a consequence of the trace being
exported. The eBPF profiler reads whichever trace and span ids the interrupted thread currently
publishes, stores them in a per-sample link, and never asks whether that span was sampled. It could not
ask: neither source it reads carries a usable trace-flags byte. So a link means "this thread was inside
this span at the instant of the sample", and a profile may legitimately link to a span the backend never
received.

The link is also absent by default. Two independent opt-ins must both be satisfied, one on the
instrumentation side and one on the profiler side, and there is a third path that exists only for
processes whose SDK publishes a thread-local record. Run the profiler alone and every sample looks like
it ran outside a span; run the instrumenter alone and there is no profile to link to.

## Where the link comes from

1. **The wire format has a slot for it, and a zero that means none.** A Profiles `Sample` carries a
   `link_index` into the shared dictionary's link table, and index zero is the dictionary's mandatory
   zero entry, so the field's default value means "no link exists". A link itself is exactly two
   identifiers, a 16-byte trace id and an 8-byte span id. There is no flags field, no parent, and no
   sampling decision anywhere in it.

2. **First source: a pinned map written by the instrumenter.** OpenTelemetry eBPF Instrumentation
   pins `traces_ctx_v1`, an LRU hash keyed by the packed process-and-thread id with a value of two
   arrays, trace id and span id. It is a general-purpose surface: the same map feeds the
   instrumenter's own log enricher. Its header declares the layout stable for outside readers.

3. **The instrumenter populates it only when somebody reads it.** Keeping that map aligned with the
   active request costs a refresh on every asynchronous context switch of the instrumented runtime, so
   the writers are gated on one BPF constant. The map is filled when the log enricher is enabled, or
   when `populate_trace_context` is set for an outside reader. With no reader the constant stays off,
   the stores compile away, and the pin sits empty. A non-populating run also empties the pin once at
   start, so stale entries from an earlier run do not survive.

4. **The profiler opts in on its side too.** With `obi_process_ctx` off, the profiler sizes that map
   to a single entry and skips pinning it, precisely so the cost of a shared map is not paid for
   nothing. With it on, the profiler first tries to open the existing pin, and only creates and pins
   the map itself when the instrumenter has not. Both sides document the same rule: whoever starts
   first creates the pin.

5. **Second source: a thread-local record the SDK publishes.** The profiler also reads a plain ELF
   thread-local pointer, exported in the dynamic symbol table by the SDK, which points at a record
   holding the trace id, the span id, a validity byte, a trace-flags byte and a short attribute
   payload. The record is built when a request context is attached to a thread and cleared or
   invalidated when it detaches. This path needs no shared map and no instrumenter.

6. **The thread-local record wins.** At the end of unwinding, the profiler stamps the OBI-map ids
   first, then reads the thread context and — when it finds a valid record with at least one non-zero
   id — replaces both ids with the record's. When the record's ids are both zero, meaning the thread
   has no active span, the earlier ids are kept rather than cleared.

7. **What reaches the sample.** The profiler writes a link entry only when both ids are non-zero,
   deduplicating identical pairs into one dictionary entry, then points the sample at it. Samples are
   keyed by stack, attributes and link together, so two samples with the same stack but different
   spans stay separate samples rather than one aggregated row.

## What the link does and does not prove

- **Span ids exist without sampling.** The tracing SDK generates a span id before and independently
  of the sampling decision, specifically so that other components can rely on a unique id even for a
  non-recording span. The thread-local record is written on context attach, which is not the same
  event as a span being sampled. So a link can name a span that was dropped, and the profile is not
  wrong to have it.

- **The trace-flags byte is available and thrown away.** The thread-local record layout defines a
  trace-flags byte between the validity byte and the payload size, aligned so the size field lands on
  a two-byte boundary. The profiler's own C struct for that record names the same offset
  `_reserved`, and no code in the profiler reads trace flags at all. The one place where a sampled
  bit could have been carried is therefore structurally present and functionally unused.

- **A link is a point-in-time observation, not request provenance.** Both sources are keyed by the
  thread that was interrupted. The map entry is deleted when the server span ends and rolled back to
  the parent when a client span ends, and the thread-local pointer is cleared on detach. A sample
  landing after the span ended, or on a warm worker thread that already served another request,
  links to whatever the thread held at that instant.

- **Aggregation degrades as span cardinality rises.** Because the link is part of the sample's
  identity, one hot stack under one span per request becomes one sample per request instead of one
  aggregated sample. That is the intended trade for "which code burned CPU inside this trace", and it
  is also why linked profiles are larger and coarser for aggregate top-N work.

## Verifying and debugging path

1. **Check the instrumenter's side first.** Confirm `populate_trace_context` (or a non-empty log
   enricher) is set, then inspect the pin: an empty map with population on points at the
   instrumenter's protocol or runtime coverage, not at the profiler.

   ```console
   # bpftool map show pinned /sys/fs/bpf/otel/traces_ctx_v1
   ```

   A missing pin with population on means the instrumenter fell back to internal maps, which happens
   when the bpffs directory cannot be created or accessed; its log says so once at startup and
   disables pinning for every map.

2. **Then the profiler's side.** Confirm `obi_process_ctx` is on, otherwise the profiler never reads
   the shared map and every sample is unlinked even though the pin is full.

3. **Then the process's own publishing.** The thread-context loader resolves a TLS symbol in the
   process or its libraries, and stays dormant until both the symbol and the process's published
   schema are present. Three counters separate the cases: a sample whose TLS pointer could not be
   read, a sample whose record could not be read, and a successful read. A process that publishes
   nothing increments none of them.

   ```console
   # the profiler's counters, when its metric endpoint is enabled
   # bpf.threadctx.read_successes
   # bpf.threadctx.errors.read_tls_ptr
   # bpf.threadctx.errors.read_thread_ctx_buf
   ```

4. **Check the attribute payload separately from the ids.** A record whose payload exceeds the
   640-byte cap is truncated and counted, and the sample still carries the ids. A truncated payload
   means the labels are short, not that the link failed.

5. **Separate the two failure shapes.** No link at all means one of the opt-ins is off, or the
   process publishes nothing. A link that names the wrong span means the thread moved between the
   sample and the request you assumed; compare the linked span against the request the thread was
   serving when the sample was taken, not against the request you are debugging.

## Limits

- **The profiles signal is still alpha.** The format, the dictionary layout and the thread-context
  schema version are all marked as in development, and a schema version the profiler does not
  recognize is counted and rejected rather than guessed at. Treat the record layout as a moving
  target when you write your own publisher.

- **The pinned map is the fast path and it has a hard ceiling.** It is an LRU hash preallocated at
  load, so it occupies its full size whether or not anything writes to it, and a busy thread's entry
  can be evicted under load. Read or accept it promptly.

- **Only a bounded handoff list reaches the map.** The instrumenter associates an outgoing request
  with an incoming one through per-runtime mechanisms for a fixed set of runtimes, plus
  language-agnostic thread and socket signals. Work moved onto a queue the instrumenter cannot see
  yields no parent, so the link lands on the wrong span or none at all.

- **Nothing here proves the sampling bit.** Use the link to find the code that ran during a request;
  do not read it as evidence that the corresponding span exists in the trace backend.

## References

- [Profiles data format](https://opentelemetry.io/docs/specs/otel/profiles/data-format/) — the dictionary, the by-reference layout, and the sample-to-link relationship.
- [Profiles protobuf schema](https://github.com/open-telemetry/opentelemetry-proto/blob/main/opentelemetry/proto/profiles/v1development/profiles.proto) — `Sample.link_index` with zero meaning no link, the two-field `Link`, and the sample identity that includes the link.
- [OTEP 4947: thread-level context sharing](https://github.com/open-telemetry/opentelemetry-specification/blob/main/oteps/profiles/4947-thread-ctx.md) — the thread-local record layout including the trace-flags byte, the publication and reading protocols, and the 640-byte recommendation.
- [OTEP 4719: process-level context sharing](https://github.com/open-telemetry/opentelemetry-specification/blob/main/oteps/profiles/4719-process-ctx.md) — where the thread-local schema keys live, and the stated trace-correlation boundary of the process-level mechanism.
- [Instrumentation: correlating profiles to traces](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/blob/main/devdocs/trace-profile-correlation.md) — the shared map specification, key and value, and the `populate_trace_context` opt-in.
- [Instrumentation: trace context and when the map is populated](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/blob/main/devdocs/trace-log-correlation.md) — the cost model, the read-on-demand gate, and the once-only drain when population is off.
- [Instrumentation `pkg/config/ebpf_tracer.go`](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/blob/main/pkg/config/ebpf_tracer.go) — the documented contract of `populate_trace_context`.
- [Instrumentation `pkg/ebpf/tracer_linux.go`](https://github.com/open-telemetry/opentelemetry-ebpf-instrumentation/blob/main/pkg/ebpf/tracer_linux.go) — the bpffs otel directory, the fallback to internal maps, and the warning that external profile correlation stops working.
- [Profiler `tracer/tracer.go`](https://github.com/open-telemetry/opentelemetry-ebpf-profiler/blob/main/tracer/tracer.go) — the shared-map lookup, the single-entry fallback when the opt-in is off, and the create-and-pin path.
- [Profiler interpreter dispatcher](https://github.com/open-telemetry/opentelemetry-ebpf-profiler/blob/main/support/ebpf/interpreter_dispatcher.ebpf.c) — both id sources, their order, and the thread-context override rule, in that file.
- [Profiler `support/ebpf/types.h`](https://github.com/open-telemetry/opentelemetry-ebpf-profiler/blob/main/support/ebpf/types.h) — the on-wire record struct with the reserve byte, and the 640-byte payload cap.
- [Profiler sample reporter](https://github.com/open-telemetry/opentelemetry-ebpf-profiler/blob/main/reporter/internal/pdata/generate.go) — the non-zero-id test and link deduplication in the pdata generator before the sample points at a link.
- [Profiler thread-context loader](https://github.com/open-telemetry/opentelemetry-ebpf-profiler/blob/main/interpreter/threadcontext/threadcontext.go) — the exported TLS symbol lookup and the pending-DTV path.
- [Tracing SDK specification](https://github.com/open-telemetry/opentelemetry-specification/blob/main/specification/trace/sdk.md) — the span id generated independently of the sampling decision, and the recording/sampled reaction table.

## Community discussion today

This run reviewed the two chat workspaces that opted in to the private read-only archive, three
allowlisted channels in total, and read the public bpf mailing list in a visible browser. The archive
was empty in the strict sense: no message in the 24-hour window, and its seven-day window held eight
messages across two channels of one workspace and none in the other. The other four allowlisted chat
workspaces stayed visible-browser-only and could not be read: both Discord invites and both Slack
invites open join or acceptance pages with no channel list, so they are recorded as inaccessible
rather than quiet. A practitioner forum returned a network-policy block page to the same fetch. The
mailing list was readable, with 32 new thread roots in the day's window, and it carried the technical
substance below; the question above comes from the trailing edge of the archive's seven-day window,
where two projects were comparing network traces against continuous profiles, rather than from a
message that asked it outright.

### The eight messages, and what they actually are

Seven of the eight archived messages are one theme: a project's own report that several talks at a
conference covered its eBPF network-observability agent, including a third-party use that combined
network flows with continuous profiles for threat detection, and that two maintainers presented on
embedding the agent in other products. That combination is the interesting part, and it is exactly
the boundary answered above: network flow data says which connection did what, profiles say which
code burned the CPU, and joining them means per-sample span ids rather than a per-host aggregate. The
eighth message is a GenAI instrumentation contributor introducing work on aligning a JavaScript
instrumentation with the GenAI semantic conventions and on adopting newer agent APIs for per-agent
token usage. Neither is a question, so the answer above was reconstructed from the two projects'
documented contracts instead of from the messages.

### Zero-copy AF_XDP without a second allocator per driver

The largest coherent series in the window makes the UMEM a page-pool memory provider instead of
requiring every driver to grow a second receive allocator. The mechanism is worth following because
it moves responsibility rather than adding a path: the core installs the provider at bind through the
queue API, a driver only declares that it supports it and validates chunk size, headroom and
multi-buffer mode, and a refill contract replaces the per-driver NAPI kick — drivers report the end of
each refill and the core publishes the queue's NAPI for provider and socket wakeups. The stack is told
whether it may retain provider buffers; when it may not, it copies them on the pass path, on frame
conversion and on non-XSKMAP redirects, and because frame copies are linear, multi-buffer frames to
devmap and cpumap are dropped rather than handled. The stated numbers on one queue are 21.2 million
packets per second received, 28.9 million transmitted, and a 1.9 percent drop cost for an
`XDP_DROP` program at 30 instructions per packet. Two open ends are named in the letter: aligned
4 KiB chunks only, and generic XDP plus cpumap still cannot feed a provider socket.

### Counting grace periods instead of assuming them

A two-patch change to the multi-program attach path produced the best review arithmetic in the window.
The proposal defers RCU synchronization until a peer is reused; the review reply accepts the idea and
then corrects the placement, because the wait was attached to the commit that runs after every change
rather than to the single swap that needs it. The reply's own table is the argument: for a second
change arriving inside a grace period of the first, an attach-then-attach saves one wait, while
detach-then-detach, replace-then-detach and attach-then-detach each add one. The correction is to
start the grace period in the attach swap only, and to leave the state alone on detach and replace,
because the peer lookup already waited for the old state. The reviewer's framing is the durable part:
the wait protects the previous generation of the entry, so the question to ask of any such series is
which generation it waits for. No boot test was run, only a clean x86-64 build.

### The sockmap iterator rule, now applied

The restriction covered here earlier in the week — only sockmap iterators may update a sockmap from an
iterator context — has now been applied to the BPF tree after a third revision. The final version
grew beyond its title in three ways that show what the underlying defect was: the socket pointer
handed to an iterator program is now marked trusted, which is what lets the update be checked at all;
the destroy helper's callers are narrowed to the TCP and UDP iterators; and a selftest that had
deliberately provoked the deadlock by updating a sockmap from the UNIX-socket iterator had its update
removed. The lesson for anyone writing an iterator that mutates a map is that the permission is a
property of the iterator program, not of the program's target map.

### Errors that used to be silently ignored

A thirteen-patch filesystem series teaches callers of a dentry-resolution helper to handle failure,
including two BPF touchpoints: uprobes now fail at creation when the real inode cannot be resolved,
and a BPF kfunc returns null instead of an inode it never obtained. The stated reason for the error is
a lazy lookup on an overlayfs lower layer, which used to be dropped with a warning. The reason this
belongs in a BPF discussion is the uprobe path: a probe created against an unresolvable real inode is
attached to nothing and fires never, which is indistinguishable from a function that is merely not
called. The series removes the boolean helper that could ignore the error, so the compiler enforces
the new contract. A separate two-patch libbpf fix in the same window has the same shape at a smaller
scale: an unaligned enum was read through a bitfield helper with a zero width, shifting a 64-bit
value by 64, so the dumped value carried the wrong sign.

### Security telemetry that the scheduler can starve

An RFC outside the core list asks a question with a real answer for BPF users: a scheduling policy
loaded with `CAP_SYS_ADMIN` can route security daemons to a low-priority dispatch queue with
microsecond slices and probabilistic skipping, keeping them below the runnable-task watchdog while
their ring-buffer consumers fall behind kernel event production. The reported measurements claim
silent alert drops up to complete loss across 150 trials per sensor, without detaching any probe. The
proposed mitigation is a BPF LSM hook on map creation, validating the struct-ops registration against
an allowlist, plus a tracepoint pair for scheduler load and dispatch. Two premises are worth checking
before adopting it: the slices are the policy's own choice, so the enforcement surface is a policy
allowlist rather than the scheduler core; and the userspace remedy the authors also give — running
ring-buffer consumer threads in a real-time class — depends on how the daemons are deployed, not on the
kernel.

### Two smaller shapes

A sockmap patch for data loss after a redirect drew an automated review that separates introduced from
pre-existing defects: one high-severity leak added by the change, one medium-severity stale cork size,
and a high-severity pre-existing issue the patch merely makes easier to reach. The distinction matters
when answering such a review, because only the introduced defects are the author's to fix in the same
series. Finally, a kallsyms series adds name and address query ioctls for the kernel's symbol table,
which is the kind of interface that lets a tracing tool resolve a kernel address without parsing a
text file and without holding a reference to a module that may unload.