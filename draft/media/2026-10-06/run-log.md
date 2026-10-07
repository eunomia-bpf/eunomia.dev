# 2026-10-06 QA 运行日志

## 2026-10-06 eBPF Q&A run-report (eunomia-community-radar)

## Selected candidate

- Slug: `in-kernel-drop-decisions-vs-userspace-escalation-under-ringbuf-load`
- Question: why a drop decision for a redundant packet stays in the kernel
  rather than escalating every retry to user space, and how a busy BPF ring
  buffer should be sized and watched.
- Source: an opt-in archive thread asking how to benchmark kernel-space
  drop latency against user-space context-switch cost under thousands of
  concurrent socket-layer retries per second with a heavily loaded ring
  buffer. The thread was open-ended and attached no public primary source or
  decisive boundary. This is the thread the 10-05 page explicitly deferred
  ("no public primary source or decisive boundary was available, so it
  stayed unpublished"); it is the only 10-06 thread not yet published —
  the other two (OBI k8s-cache env var; the GenAI skill-definitions PR)
  were published on 10-03 and 10-05 respectively.
- On the five on-disk `2026-09-2*` pairs: each is in HEAD and already
  linked in `index.md`, and sits under a concurrent agent's staged `D`
  cleanup, so none of them is the retained candidate. Left untouched.

## Verification against public primary sources

- Kernel ring buffer (`docs.kernel.org/bpf/ringbuf.html`): power-of-2
  shared multi-producer single-consumer buffer with a memory-mappable
  data area; `bpf_ringbuf_reserve()`/`commit()`/`discard()` take a
  compile-time constant size and the reservation fails non-blocking (NULL)
  when the ring has no space left; `bpf_ringbuf_query()` reports
  `BPF_RB_AVAIL_DATA`, `BPF_RB_RING_SIZE`, `BPF_RB_CONS_POS`,
  `BPF_RB_PROD_POS`; `BPF_RB_NO_WAKEUP`/`BPF_RB_FORCE_WAKEUP` control how
  the producer wakes the consumer; each reserved record carries a small
  header with the record length and a busy bit; in NMI context the
  reservation can fail even when the ring is not full because it takes a
  spin lock an interrupt may not acquire.
- Verifier limit: the load-time verifier-complexity budget frames program
  shape as a fixed cost paid once at load, not per packet; BPFConf 2025's
  "Beyond 1M BPF instructions" frames the one-milion-instruction limit as
  that load-time budget, spent on unrolled loops and always-inlined
  helpers.
- The consequence: per-packet cost of the in-kernel dedup path is a
  handful of map operations, so a deterministic bounded-state drop stays
  in the kernel while a full ring degrades observability, not
  datapath correctness.

## Privacy + content

- No names/handles/Slack-Discord URLs/exact timestamps/IPs/credentials in
  either page; anonymized summary only. Public primary-source links in
  `## References` only. Dead references (man7.org `bpf(7)`, libbpf
  readthedocs user guide) were excluded because they no longer resolve.
- Content order per the standard: direct answer, mechanism,
  verification/debugging path, limitation, `## References`, community
  discussion. Mobile-clean rendering via short inline code tokens.

## Coverage disclosure

Two opt-in archive channels covered (8 messages in the 10-06 snapshot;
byte-identical to the 10-05 snapshot, the archive window had not advanced).
Of the three threads in the snapshot, two were already published earlier
(10-03 and 10-05) and the third is this page. Visible-browser-only sources
(Discord, the eunomia-bpf and sched-ext communities, the bpf mailing list,
and r/eBPF) were not reviewed this run (no visible-browser session) —
marked uncovered-not-quiet on the page.

## Artifacts

- EN: `docs/ebpf-qa/2026-10-06-in-kernel-drop-decisions-vs-userspace-escalation-under-ringbuf-load.md`
- ZH: `docs/ebpf-qa/2026-10-06-in-kernel-drop-decisions-vs-userspace-escalation-under-ringbuf-load.zh.md`
- Index links: first items of `docs/ebpf-qa/index.md` (`Latest Answers`) and
  `docs/ebpf-qa/index.zh.md` (`最新回答`).
- Published QA commit on `origin/main`: `beda998135`
  (`docs(ebpf-qa): in-kernel-drop-decisions-vs-userspace-escalation-under-ringbuf-load (2026-10-06)`).
- Receipt: `/workspaces/.agent-state/eunomia-qa/receipt-2026-10-06.json`
  `status=published` (written by the validator only).
- This run-log commits separately as the 10-06 same-day artifact.

## eBPF Q&A publication follow-up (deploy + live verification)

- QA commit `beda998135` (4 paths) landed on `origin/main`; the GitHub
  Pages `Deploy Static App` run 37551105057 went green on that commit, so
  all four routes returned 200 with the expected H1s and the new slug in
  both indexes. `cache-control: public, max-age=0, must-revalidate` and
  `cf-cache-status: DYNAMIC` confirm a fresh deploy, not a cached copy.
- Validator re-run on the already-published commit (re-verify path, no
  re-commit): receipt `status=published`, all checks `ok` (the four content
  gates `skipped_already_published`; `branch`, `candidate_paths`,
  `index_links`, `privacy`, `remote_contains_commit`, `public` `ok`).
