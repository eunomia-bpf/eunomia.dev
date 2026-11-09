# Not Published / Pending Media Ledger

Last checked: 2026-11-09

This file tracks source posts that still need platform work. The latest checked `.github/publisher/posts_queue.txt` is empty; historical rows below are retained as candidate-source tracking, not proof that Medium/Dev.to are still queued.

For full per-platform counts across all configured source files, run `python .github/publisher/media/check_media_ledger.py --show-missing`. The canonical machine-readable ledger uses one JSON file per platform under [`platforms/`](platforms/).

2026-11-09: the `publish-posts` workflow published `docs/blog/posts/claude-code-analysis.md` to DEV
(<https://dev.to/yunwei37/reverse-engineering-claude-codes-ssl-traffic-with-ebpf-5bod>, `2026-11-09T01:04:06Z`)
from a byte-identical copy at `draft/media/2026-11-09/claude-code-analysis/devto.md`, so it is no longer a
pending DEV source; its Medium copy was already confirmed as `1dde03bcc7ef`. The run used the LA 2026-11-08
normal slot, entered in `draft/plan/publishing-queue.zh.md`. `.github/publisher/posts_queue.txt` is empty again
because the workflow pops each entry after a successful publish.

2026-11-08: the `publish-posts` workflow published `docs/blog/posts/ai-lean-proof-checking-cost.md`
to DEV (<https://dev.to/yunwei37/ai-now-writes-lean-proofs-faster-than-lean-can-check-them-51mh>,
`2026-11-08T01:00:47Z`) from `draft/media/2026-11-08/ai-lean-proof-checking-cost/devto.md`, which differs from
the source only in two absolute-URL rewrites, so it is no longer a pending DEV source; its Medium copy is
still unpublished. The run used the LA 2026-11-07 normal slot.

2026-11-07: the `publish-posts` workflow published `docs/blog/posts/actplane.md` to DEV
(<https://dev.to/yunwei37/actplane-pushing-agent-harness-enforcement-down-to-kernel-ebpf-2doh>, `2026-11-07T01:31:46Z`)
from a byte-identical copy at `draft/media/2026-11-07/actplane/devto.md`, so it is no longer a pending DEV source;
its Medium copy was already confirmed as `8077f9164a34`. The run used the LA 2026-11-07 normal slot, entered in
`draft/plan/publishing-queue.zh.md`. `.github/publisher/posts_queue.txt` is empty again because the workflow pops
each entry after a successful publish.

2026-11-06: the `publish-posts` workflow published `docs/blog/posts/schedcp-agentic-os.md`
to DEV (<https://dev.to/yunwei37/can-an-ai-agent-tune-the-linux-scheduler-inside-schedcp-and-sched-agent-4cj0>,
`2026-11-06T10:44:00Z`) from `draft/media/2026-11-06/schedcp-agentic-os/devto.md`, so it is no longer a
pending DEV source; its Medium copy is still unpublished. The run used the LA 2026-11-06 normal slot,
entered in `draft/plan/publishing-queue.zh.md`.

2026-11-04: the scheduled `publish-posts` workflow published `docs/blog/posts/gthulhu-vllm-cpu-pressure.md`
to DEV (<https://dev.to/yunwei37/when-does-cpu-scheduling-help-ai-inference-5did>, `2026-11-05T01:39:39Z`),
so it is no longer a pending DEV source; its Medium copy was published 2026-10-31 and re-verified as the
9-H2 source body. No new queue task was created: the entry that authorized this run was inserted for
LA 2026-11-04 in `draft/plan/publishing-queue.zh.md` and is now checked off.

## Current Publisher Queue

The machine ledger currently reports 41 unmatched Chinese sources on Zhihu and
31 on Juejin (78/109 mapped). These raw counts include paused material,
non-standalone index or reference pages, and duplicate source representations.
The ordered platform actions that remain suitable candidates are maintained
one-by-one in `draft/plan/publishing-queue.zh.md`; 24 Zhihu and 0 Juejin tasks
remain. Each checkbox is one platform task. The normal target is one
publication per day; documented catch-up slots may add another platform, never
a second post on the same platform that day unless it redeems a catch-up gap.

The LinkedIn link shares for tutorials 50 through 54 are complete. There are no
remaining LinkedIn queue tasks. SchedCP, AgentCgroup, CPU noise, Weekly Analysis
reports, and content with confirmed LinkedIn publication records are not
included in this LinkedIn batch.

Medium and DEV have completed the queued BPFix, tutorials 50 through 54,
AgentCgroup, CPU noise, Agent Sandbox, ACRFence, and Runtime Security articles,
and the 2026-11-08 DEV copy of the Lean proof-checking post. Neither platform
has an unfinished queue task. Raw unmatched-source counts do
not authorize additional publication. Medium tutorial 53's flattened
Requirements table was repaired in place on 2026-08-30; no duplicate was created.

## Additional Platform Status

| Platform | Confirmed published | Not published / unresolved | Next action |
| --- | --- | --- | --- |
| X / Twitter | Several historical self-authored posts are confirmed on `@yunwei37`; see `published.md` | The configured/planned `@eaborai` account currently showed `此账号不存在`; no full X archive export was done | Replace or verify `@eaborai` in planning/metadata; paginate/export `@yunwei37` before declaring full X history complete |
| LinkedIn | Normal visible browser checks confirmed the `Yusheng Zheng` / `yunwei37` profile and tutorial 50-54 shares; see `platforms/linkedin.json` | Machine ledger maps 11/124 English sources; no unfinished LinkedIn queue task | Monitor existing posts; unmatched sources are not automatic publication tasks |
| Zhihu | Normal visible browser checks confirmed the fsession tutorial publication on 2026-08-14; machine ledger maps 68/109 Chinese sources | 41 configured Chinese sources remain unmatched; the visible profile was rechecked on 2026-09-23 and `/creator` still redirects to `/signin`, with no `z_c0` auth cookie, so the session is not signed in | Resume the first queued AgentCgroup task after a visible Zhihu session is signed in; recheck the title before submission |
| Juejin | xpu/npu-kernel-driver at <https://juejin.cn/post/7698947426490318888> (2026-10-21) is confirmed public with full public-page QA, published through the visible editor using the LA 2026-10-21 normal slot with no catch-up gap redeemed, alongside xpu/gpu-kernel-driver at <https://juejin.cn/post/7698552541928554511> (2026-10-20) is confirmed public with full public-page QA, published through the visible editor using the LA 2026-10-20 normal slot with no catch-up gap redeemed, alongside xpu/flamegraph at <https://juejin.cn/post/7698287001586040872> (2026-10-19) is confirmed public with full public-page QA, published through the visible editor using the LA 2026-10-19 normal slot with no catch-up gap redeemed, alongside struct_ops at <https://juejin.cn/post/7697622920013709347> (2026-10-18) is confirmed public with full public-page QA, published through the visible editor using the LA 2026-10-18 normal slot with no catch-up gap redeemed, alongside dynptr at <https://juejin.cn/post/7697186290753159202> (2026-10-17) is confirmed public with full public-page QA, published through the visible editor using the LA 2026-10-17 normal slot with no catch-up gap redeemed, alongside bpf_wq at <https://juejin.cn/post/7697094073967116323> (2026-10-16) is confirmed public with full public-page QA, published through the visible editor using the LA 2026-10-16 normal slot with no catch-up gap redeemed, alongside bpf_token at <https://juejin.cn/post/7696759034846445583> (2026-10-15) is confirmed public with full public-page QA, published through the visible editor using the LA 2026-10-15 normal slot with no catch-up gap redeemed, alongside bpf_iters at <https://juejin.cn/post/7696384745624141824> (2026-10-14) is confirmed public with full public-page QA, published through the visible editor using the LA 2026-10-14 normal slot with no catch-up gap redeemed, alongside bpf_arena at <https://juejin.cn/post/7695977967902933026> (2026-10-13) is confirmed public with full public-page QA, published through the visible editor using the LA 2026-10-13 normal slot with no catch-up gap redeemed, alongside cgroup-tutorial at <https://juejin.cn/post/7695667191396040738> (2026-10-12) is confirmed public with full public-page QA, published through the visible editor using the LA 2026-10-12 normal slot with no catch-up gap redeemed, alongside bpftrace-tutorial at <https://juejin.cn/post/7695068144415244323> (2026-10-11) is confirmed public with full public-page QA, published through the visible editor using the LA 2026-10-11 normal slot with no catch-up gap redeemed, alongside 1-helloworld at <https://juejin.cn/post/7694934986782064649> (2026-10-10) is confirmed public with full public-page QA, published through the visible editor using the LA 2026-10-10 normal slot with no catch-up gap redeemed, alongside 17-biopattern at <https://juejin.cn/post/7694534084639195174> (2026-10-09) is confirmed public with full public-page QA, published through the visible editor using the LA 2026-10-09 normal slot with no catch-up gap redeemed, alongside 19-lsm-connect at <https://juejin.cn/post/7694131662616428571> (2026-10-08) is confirmed public with full public-page QA, published through the visible editor using the LA 2026-10-08 normal slot with no catch-up gap redeemed, alongside 20-tc at <https://juejin.cn/post/7693579496968831022> (2026-10-07) is confirmed public with full public-page QA, published through the visible editor using the LA 2026-10-07 normal slot with no catch-up gap redeemed, alongside 21-xdp at <https://juejin.cn/post/7693018382872330278> (2026-10-06) is confirmed public with full public-page QA, published through the visible editor using the LA 2026-10-06 normal slot with no catch-up gap redeemed, alongside 22-android at <https://juejin.cn/post/7692860578532933642> (2026-10-05) is confirmed public with full public-page QA, published through the visible editor using the LA 2026-10-05 normal slot with no catch-up gap redeemed, alongside 29-sockops at <https://juejin.cn/post/7691585964054003758> (2026-10-02) is confirmed public with full public-page QA, published through the visible editor using the LA 2026-10-02 normal slot with no catch-up gap redeemed, alongside 28-detach at <https://juejin.cn/post/7691345821565141019> (2026-10-02) is confirmed public with full public-page QA, published through the visible editor as the second publication of the LA 2026-10-02 slot, redeeming the 2026-09-03 catch-up gap, alongside 27-replace at <https://juejin.cn/post/7691510191482519603> (2026-10-02) is confirmed public with full public-page QA, published through the visible editor as the third publication of the LA 2026-10-02 slot, redeeming the 2026-09-16 catch-up gap and leaving no outstanding catch-up gaps, alongside 26-sudo at <https://juejin.cn/post/7692143474695012402> (2026-10-03) is confirmed public with full public-page QA, published through the visible editor using the LA 2026-10-03 normal slot with no catch-up gap redeemed, alongside 24-hide at <https://juejin.cn/post/7692741078310912027> (2026-10-04) is confirmed public with full public-page QA, published through the visible editor using the LA 2026-10-04 normal slot with no catch-up gap redeemed, alongside 31-goroutine at <https://juejin.cn/post/7691326130511167526> (2026-10-01) is confirmed public with full public-page QA, published through the visible editor using the LA 2026-10-01 normal slot with no catch-up gap redeemed, alongside 32-wallclock-profiler at <https://juejin.cn/post/7691151105851066391> (2026-09-30) is confirmed public with full public-page QA, published through the visible editor using the LA 2026-09-30 normal slot with no catch-up gap redeemed, alongside 33-funclatency at <https://juejin.cn/post/7691081437695482937> (2026-09-29) is confirmed public with full public-page QA, published through the visible editor using the LA 2026-09-29 normal slot with no catch-up gap redeemed, alongside 34-syscall at <https://juejin.cn/post/7690979496298637338> (2026-09-28) is confirmed public with full public-page QA, published through the visible editor using the LA 2026-09-28 normal slot with no catch-up gap redeemed, alongside 35-user-ringbuf at <https://juejin.cn/post/7690813678851970571> (2026-09-27) is confirmed public with full public-page QA, published through the visible editor using the LA 2026-09-27 normal slot with no catch-up gap redeemed, alongside 37-uprobe-rust at <https://juejin.cn/post/7690645355813357078> (2026-09-26) is confirmed public with full public-page QA, published through the visible editor using the LA 2026-09-26 normal slot with no catch-up gap redeemed, alongside 38-btf-uprobe at <https://juejin.cn/post/7689852222630122548> (2026-09-25) is confirmed public with full public-page QA, published through the visible editor using the LA 2026-09-25 normal slot with no catch-up gap redeemed, alongside 40-mysql at <https://juejin.cn/post/7691185608911134758> (2026-09-24) is confirmed public with full public-page QA, published through the visible editor as a gap-funded extra task on the LA 2026-09-24 slot, alongside 41-xdp-tcpdump at <https://juejin.cn/post/7691124509454822408> (2026-09-24) is confirmed public with full public-page QA, published through the visible editor using the LA 2026-09-24 normal slot, alongside 39-nginx at <https://juejin.cn/post/7691334193581959952> (2026-09-24) is confirmed public with full public-page QA, published through the visible editor on the LA 2026-09-24 slot ahead of its targeted LA 2026-09-25 normal slot, alongside 36-kfunc at <https://juejin.cn/post/7688489073967251494> (2026-09-22, review cleared 2026-09-23), and 42-xdp-loadbalancer at <https://juejin.cn/post/7688532185121947658> (2026-09-23), joining Runtime Security, 48-energy, 47-cuda-events, and 46-xdp-test | No known unresolved Juejin publication defect; the machine ledger maps 78/109 sources, with 31 unmatched and 0 open rolling-queue tasks | Monitor [author profile](https://juejin.cn/user/4288563097635144/posts); keep pre-staging the next queue artifact before its slot |
| Medium | Does moving a GPU server's request loop into the kernel save CPU? at <https://medium.com/@yunwei356/does-moving-a-gpu-servers-request-loop-into-the-kernel-save-cpu-d8c070b5bd42> (2026-11-03T14:45:15Z) is the newest confirmed Medium item, published by the scheduled `publish-posts` workflow through the documented Medium API with topics `ebpf`/`gpu`/`kernel` and no `canonical_url`. Its public page carries the same three API-import defects as `medium-5001bb9afab6`: no rendered story title (`h1: 0`, body opens at the Mermaid block), the Mermaid fence kept as raw flowchart code, and the results table flattened into `| ... | — -|` prose rows; both paper links also resolve to <https://medium.com/papers/bpf-gpu-server.pdf> (403) because the source's relative link was sent unchanged. The body itself is complete (~1,976 words, 8 body H3 from the 8 source `##`, full ending, GitHub links intact) | <https://medium.com/@yunwei356/does-moving-a-gpu-servers-request-loop-into-the-kernel-save-cpu-d8c070b5bd42>. `medium-59aff1cc69a5` still renders the pre-restructure 10-H2 Gthulhu body | Repair titles and tables for both API-published stories once a logged-in visible Medium editor session or `MEDIUM_API_KEY` is available; `/p/d8c070b5bd42/edit` and `/p/5001bb9afab6/edit` both still redirect to `/m/signin` |
| DEV Community | Does moving a GPU server's request loop into the kernel save CPU? at <https://dev.to/yunwei37/does-moving-a-gpu-servers-request-loop-into-the-kernel-save-cpu-d72> (2026-11-04T02:30:51Z) is the newest confirmed DEV item, published from `docs/blog/posts/bpf-gpu-server.md` through the documented DEV API as a DEV-only retry. The scheduled 2026-11-03 run published the Medium twin and failed DEV with `422 Tag "io_uring" contains non-alphanumeric or prohibited unicode characters`; because the API reports a per-platform failure inside a `200` response, `publish.py` treated it as success and removed the queue entry. The retry used `PUBLISH_PLATFORMS=devto` with tags `ebpf`/`gpu`/`kernel` and the prepared artifact `draft/media/2026-11-03/bpf-gpu-server/devto.md`. Public API and visible DOM agree: exact title, 8 body H2 matching the 8 source `##`, one 3-row results table, the Mermaid fence rendered as one inline SVG, both paper links absolute and live, and the complete ending. Canonical is DEV self rather than eunomia.dev | <https://dev.to/yunwei37/does-moving-a-gpu-servers-request-loop-into-the-kernel-save-cpu-d72>. The public author API now lists 65 posts; AgentCgroup was confirmed public on 2026-08-27 and every earlier DEV item stays confirmed | No DEV action is queued. Open follow-ups: the qwen3 copy still needs a deliberate canonical decision, and a later API publish should pass `canonical_url` and absolute links so new DEV copies stop self-canonicalizing |
| Reddit | Historical `u/yunwei123` posts are confirmed for eBPF tutorial, GPTtrace, Code-Survey-like discussion, and Wasm-bpf; see `published.md` | No evidence yet that current 2026 flagship posts were submitted to r/eBPF, r/netsec, r/LocalLLaMA, HN, or lobste.rs | Use Reddit only manually for flagship posts; check subreddit fit before posting |
| Xiaohongshu / RedNote | None confirmed | No account URL or published note URL found; current browser search requires login; public search found no clear `eunomia.dev` / `eunomia-bpf` / `bpftime` / `AgentSight` result | Treat as not started; create account and visual-note workflow only after there is image-card/video capacity |

## Pending Verification

These items need follow-up before moving to `published.md` or clearing them:

| Item | Why it needs verification | Suggested check |
| --- | --- | --- |
| Full Zhihu article history | Profile reports 114 articles; normal visible scrolling collected 113 unique links and exact-title matching resolved two tutorial/blog duplicates | Recheck visible title/keyword matches immediately before publishing any remaining Zhihu-missing tutorial; do not use APIs or hidden endpoints |
| Full X history | `from:yunwei37` search recorded visible project hits, but not a complete account export | Use X advanced search or account data export before declaring the ledger complete |
| Full Reddit history | `author:yunwei123` search recorded visible project hits, but not all comments or deleted/crossposted content | Use Reddit user listing and subreddit search for `eunomia.dev`, `eunomia-bpf`, `bpftime`, `AgentSight` |
| Xiaohongshu login/account | Search results were hidden behind login and no account URL is known | Sign in or provide account URL before doing a definitive platform audit |
| Repo-referenced platform URLs | Some links appear in documentation as references; they may be our posts, partner posts, or citations | Open each URL and confirm author/account before marking as confirmed |

## Add New Pending Items

Use this format when adding a post:

```md
| `source/path.md` | Human title | Not queued / queued | Not published / draft / published URL | Not published / draft / published URL | Next concrete action |
```
