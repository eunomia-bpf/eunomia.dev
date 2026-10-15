## 2026-10-14 eBPF Q&A run-report (eunomia-community-radar)

## Selected candidate

- Slug: `fuse-otel-trace-propagation-via-obi-trace-map`
- Question: how feasible it is to add OpenTelemetry trace propagation to a
  FUSE userspace filesystem daemon by exporting OBI's trace information
  through its trace-context map (`traces_ctx_v1`) and collecting it in the
  FUSE program using the pid and tid.
- Source: the 10-14 snapshot carries eleven messages, all in the two
  opt-in CNCF OpenTelemetry instrumentation channels. The thread this page
  answers is a practitioner asking how feasible it is to use OBI to
  export trace information to `traces_ctx_v1` and read it back from a
  FUSE userspace daemon keyed by pid and tid. The archive carried the
  question plus the maintainers' follow-ups (FUSE can read the caller's
  thread; the thread-group id is reachable through the caller's status
  page under `/proc`; Go and Python-with-asyncio are instrumented while
  C++ is not; the OBI map is populated only when a trace context already
  exists on the serving thread).
- Decisive boundary: kernel-side visibility versus OBI-side storage. The
  kernel fills `fuse_in_header.pid` with the caller thread's TID
  (`task_pid(current)` in `fs/fuse/req.c`), so the daemon always knows
  which thread issued the request; OBI pins `traces_ctx_v1` (an
  LRU-hash keyed by the packed thread-group-and-thread id) that holds the
  W3C trace/span id pair OBI already captured on that thread, but only
  when `populate_trace_context` is on and OBI has a context to store.
  Untracked protocols or uninstrumented runtimes leave the entry absent,
  and the TID-to-TGID resolution is the glue needed to build OBI's key.
- The two remaining threads in the window had already been answered by an
  earlier post or were a review-status update rather than a question: the
  Cilium eBPF discussion cross-post is the 10-13 published entry, and
  the semantic-conventions PR note was a review bump, so this run
  selected the non-duplicative FUSE and OBI trace-propagation question.

## Verification against public primary sources

- kernel `fs/fuse/req.c`: `fuse_args.pid` is set to
  `pid_nr_ns(task_pid(current), fc->pid_ns)` - the caller thread's TID
  in the mount's pid namespace; copied into `fuse_in_header.pid`
  (`include/uapi/linux/fuse.h`) by `fuse_args_to_req` in
  `fs/fuse/dev.c`. The flock path in `fs/fuse/file.c` deliberately uses
  `task_tgid`, which is the contrast proving the generic header field is
  thread-scoped.
- OBI `bpf/shared/obi_ctx.h`: `traces_ctx_v1` is a
  `BPF_MAP_TYPE_LRU_HASH` pinned by name (default `/sys/fs/bpf/`),
  `u64` key with a two-field value - `trace_id` (16 bytes) and `span_id`
  (8 bytes), the W3C Trace Context pair. Entries are written at
  server-span-start keyed by `bpf_get_current_pid_tgid()`
  (thread-group high 32 bits, thread id low 32 bits).
- OBI `pkg/config/ebpf_tracer.go`: population is gated by
  `populate_trace_context` (default off) - the map is only filled when
  something reads it.
- OpenTelemetry OTEP 4855 (opentelemetry-specification): the external
  trace-context map spec OBI implements.
- OBI `SUPPORT_MATRIX.md`: Go 1.17+, Java JDK8+, Node.js 12.17+, Python
  (asyncio) supported; C++ absent - matching the maintainers' answer in
  the thread.

## Privacy + content

- No names/handles/Slack-Discord URLs/timestamps/message-sequence
  numbers/employer/team/internal repo names/IPs/private logs in either
  page. The snapshot carries mention hashes and a PR author handle; none
  of them appear in the article - the discussion author and maintainers
  are referenced only generically ("the maintainers"). Public kernel and
  OBI repository links and the OTEP/W3C spec links appear in
  `## References` / `## 参考` only.
- Content order per the standard: direct answer, mechanism,
  verification/debugging path, limitation, references, community
  discussion. H1s kept free of underscores, backticks, and apostrophes;
  the body's longest backticked token is the 26-character
  `bpf_get_current_pid_tgid()`, well under the 390px mobile overflow
  gate.

## Coverage disclosure

Two opt-in archive channels covered (the two CNCF OpenTelemetry
instrumentation channels) with eleven messages in the 10-14 snapshot.
The candidate was selected from the FUSE-and-OBI trace-propagation
thread. Two threads in the window were skipped: the Cilium eBPF
discussion cross-post, which is the already-published 10-13 entry, and
the semantic-conventions review-status update, which is a status note
rather than a question. Visible-browser-only sources (the eunomia-bpf
and sched-ext Discord servers, the bpf mailing list, and r/eBPF) were
not reviewed this run (no visible-browser session) -
marked uncovered-not-quiet on the page.

## Artifacts

- EN: `docs/ebpf-qa/2026-10-14-fuse-otel-trace-propagation-via-obi-trace-map.md`
- ZH: `docs/ebpf-qa/2026-10-14-fuse-otel-trace-propagation-via-obi-trace-map.zh.md`
- Index links: first items of `docs/ebpf-qa/index.md` (`Latest Answers`)
  and `docs/ebpf-qa/index.zh.md` (`最新回答`).
- Published QA commit on `origin/main`: `90657793`
  (`docs(ebpf-qa): fuse-otel-trace-propagation-via-obi-trace-map (2026-10-14)`),
  containing exactly the four owned paths.
- Receipt: `/workspaces/.agent-state/eunomia-qa/receipt-2026-10-14.json`
  `status=published` (written by the validator only), commit
  `90657793cb8080888e1f57d7e85068f3263ecd09`.

## eBPF Q&A publication follow-up (deploy + live verification)

- QA commit `90657793` (4 paths) landed on `origin/main`; the GitHub
  Pages `Deploy Static App` run `38664992098` went green on that commit
  (`conclusion: success`, 18m36s), so all four routes returned 200 with
  the verbatim EN and ZH H1s and the new slug present in both indexes.
  `cache-control: public, max-age=0, must-revalidate` and
  `cf-cache-status: DYNAMIC` confirm a fresh deploy, not a cached copy.
- The validator's first run completed the whole pipeline in ~2094s
  (content test, build, Chromium render, scoped commit + push) and the
  Pages deploy landed inside its 20-minute public-poll window, so the
  run went straight to receipt `status=published` with
  `branch`/`candidate_paths`/`index_links`/`privacy`/`content_test`/
  `build`/`render`/`commit_push`/`remote_contains_commit`/`public` all
  `ok`. No second re-verify run was needed - unlike 10-13, where the
  first run's public-poll 404'd and a re-verify pass produced the
  published receipt.
- One operational note: nothing new. The 3600s cap covering
  build->render->commit+push->public-poll, the forward-move before
  committing, and the scoped `git commit --only` against a shared
  checkout with a concurrent agent's staged entries are all already
  covered by the 10-12/10-13 durable guidance; no new edge occurred.

This run-log commits separately as the 10-14 same-day artifact.
