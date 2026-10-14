## 2026-10-13 eBPF Q&A run-report (eunomia-community-radar)

## Selected candidate

- Slug: `cilium-ebpf-tracepoint-cpu-only-new-kernel-regression`
- Question: why a tracepoint eBPF program attached through the Cilium eBPF
  Go library fires only on CPU 0 on a new kernel (for example kernel 7.0 /
  Ubuntu 26.04), when the identical program logged system calls system-wide
  on older distributions (Debian 13/14, Ubuntu 24.04).
- Source: the 10-13 snapshot carries four messages, all in the two
  opt-in CNCF OpenTelemetry instrumentation channels. The thread this page
  answers is a practitioner cross-posting an open question from the Cilium
  eBPF Go library's public discussion space: `link.Tracepoint` logged only
  CPU-0-attributed `sys_enter_open` on a new distro, worked on Debian
  13/14 and Ubuntu 24.04, `taskset -c 1,15` produced no `trace_pipe`
  output while `taskset -c 0` logged. The archive carried only the
  cross-post pointer, so the question substance and the maintainers'
  diagnosis are reproduced here in anonymized form from the public
  discussion.
- Decisive boundary: attach side versus delivery side. The library opens
  one tracepoint-typed perf event with `pid == -1, cpu == 0` (correct,
  by design). Healthy kernels still fire that single event on every CPU
  because tracepoint BPF events do not honor a CPU filter. A kernel
  regression on the syscall trace-event path broke that guarantee, so the
  CPU-0-scoped event stopped delivering cross-CPU events. The library is
  not at fault; the regression is fixed upstream in the kernel.
- The two remaining threads in the window echo already-published entries -
  the available-skills dedicated-attribute modeling question (10-11) and
  the model-serving-signals / GenAI metric catalog question (10-09) - so
  this run selected the non-duplicative Cilium tracepoint question.

## Verification against public primary sources

- cilium/ebpf `link/tracepoint.go` (v0.22.0): `Tracepoint` resolves the
  event id under the tracefs `events` tree and calls
  `openTracepointPerfEvent` with the all-threads sentinel
  `perfAllThreads` (`-1`).
- cilium/ebpf `link/perf_event.go` (v0.22.0): when `pid` is the
  all-threads sentinel the local `cpu` stays `0`, so the syscall is a
  `PERF_TYPE_TRACEPOINT` `perf_event_open` with `pid = -1, cpu = 0`.
  One event, scoped to CPU 0.
- iovisor/bcc issue 1490 (closed): BPF tracing events are system-wide and
  do not honor a CPU filter; among tracing events only uprobes support a
  PID filter. One event attached on CPU 0 fired on every CPU.
- Kernel commit `2c2b322acdcc78575b8d6afa64a085cf92e03c12`
  ("tracing: perf: Fix stale head for perf syscall tracing", fixing
  `edca33a56297d`): the per-CPU perf event list head could point at the
  wrong CPU after preemption/migration on the syscall user-space-arg
  path. The upstream fix restores system-wide delivery.
- cilium/ebpf discussion 2126: the practitioner question this page
  answers.

## Privacy + content

- No names/handles/Slack-Discord URLs/timestamps/message-sequence
  numbers/employer/team/internal repo names/IPs/private logs in either
  page; the discussion author and maintainers are referenced only
  generically ("the maintainers"). Public GitHub repository, issue,
  discussion, and kernel-commit links appear in `## References` /
  `## 参考` only.
- Content order per the standard: direct answer, mechanism,
  verification/debugging path, limitation, references, community
  discussion. H1s kept free of underscores, backticks, and apostrophes;
  the body's longest backticked token is the 23-character
  `openTracepointPerfEvent`, well under the 390px gate. Two longer API
  signatures were rewritten in words (the `link.Tracepoint` call form and
  the `PerfEventOpen` syscall form) so no inline code token exceeds the
  mobile width.

## Coverage disclosure

Two opt-in archive channels covered (the two CNCF OpenTelemetry
instrumentation channels) with four messages in the 10-13 snapshot. The
candidate was selected from the Cilium eBPF cross-posted
tracepoint/CPU-0 thread. Two threads in the window were skipped because
they echo already-published entries: the available-skills span-attribute
modeling question (10-11) and the model-serving-signals / GenAI metric
catalog question (10-09). Visible-browser-only sources (the eunomia-bpf
and sched-ext Discord servers, the bpf mailing list, and r/eBPF) were not
reviewed this run (no visible-browser session) -
marked uncovered-not-quiet on the page.

## Artifacts

- EN: `docs/ebpf-qa/2026-10-13-cilium-ebpf-tracepoint-cpu-only-new-kernel-regression.md`
- ZH: `docs/ebpf-qa/2026-10-13-cilium-ebpf-tracepoint-cpu-only-new-kernel-regression.zh.md`
- Index links: first items of `docs/ebpf-qa/index.md` (`Latest Answers`)
  and `docs/ebpf-qa/index.zh.md` (`最新回答`).
- Published QA commit on `origin/main`: `8d810259`
  (`docs(ebpf-qa): cilium-ebpf-tracepoint-cpu-only-new-kernel-regression (2026-10-13)`),
  containing exactly the four owned paths.
- Receipt: `/workspaces/.agent-state/eunomia-qa/receipt-2026-10-13.json`
  `status=published` (written by the validator only), commit
  `8d8102595ba311daa2458a590485b8288e92f400`.

## eBPF Q&A publication follow-up (deploy + live verification)

- QA commit `8d810259` (4 paths) landed on `origin/main`; the GitHub
  Pages `Deploy Static App` run `38485038024` went green on that commit
  (`conclusion: success`, 21m18s), so all four routes returned 200 with
  the expected H1s and the new slug present in both indexes.
  `cache-control: public, max-age=0, must-revalidate` and
  `cf-cache-status: DYNAMIC` confirm a fresh deploy, not a cached copy.
- The two live article pages render the verbatim EN and ZH H1s; the new
  slug is linked from both the EN and ZH index pages (the slug substring
  appears six times in each index page's raw HTML - the link plus its
  surrounding metadata occurrences).
- The validator re-verify pass returned receipt `status=published`, commit
  `8d8102595ba311daa2458a590485b8288e92f400`, with `branch`,
  `candidate_paths`, `index_links`, `privacy`, `remote_contains_commit`,
  and `public` all `ok`, and `content_test` / `build` / `render` /
  `commit_push` `skipped_already_published` (the re-verify path does not
  re-run those because the original commit already passed them).
- One operational note: the first validator run was given a 3600s cap,
  which comfortably covered `npm test:content` + build + Chromium render
  + commit + push + public-poll (it completed in ~1805s, landing
  `8d810259`). Its `public` check then 404'd only because the Pages
  deploy (15-17 minutes) had not finished. Watching the green Pages run
  and re-running the same validator produced the `status=published`
  receipt in ~4s, exactly as the 10-11/10-12 recovery prescribes. No new
  durable lesson - the 10-12 pre-commit cap guidance already covered the
  only edge that could recur here.

This run-log commits separately as the 10-13 same-day artifact.
