# Not Published / Pending Media Ledger

Last checked: 2026-09-15

This file tracks source posts that still need platform work. The latest checked `.github/publisher/posts_queue.txt` is empty; historical rows below are retained as candidate-source tracking, not proof that Medium/Dev.to are still queued.

For full per-platform counts across all configured source files, run `python .github/publisher/media/check_media_ledger.py --show-missing`. The canonical machine-readable ledger uses one JSON file per platform under [`platforms/`](platforms/).

## Current Publisher Queue

The machine ledger currently reports 39 unmatched Chinese sources on Zhihu and
65 on Juejin (42/107 mapped). These raw counts include paused material,
non-standalone index or reference pages, and duplicate source representations.
The ordered platform actions that remain suitable candidates are maintained
one-by-one in `draft/plan/publishing-queue.zh.md`; 24 Zhihu and 36 Juejin tasks
remain. Each checkbox is one platform task. The normal target is one
publication per day; documented catch-up slots may add another platform, never
a second post on the same platform that day.

The LinkedIn link shares for tutorials 50 through 54 are complete. There are no
remaining LinkedIn queue tasks. SchedCP, AgentCgroup, CPU noise, Weekly Analysis
reports, and content with confirmed LinkedIn publication records are not
included in this LinkedIn batch.

Medium and DEV have completed the queued BPFix, tutorials 50 through 54,
AgentCgroup, CPU noise, Agent Sandbox, ACRFence, and Runtime Security articles.
Neither platform has an unfinished queue task. Raw unmatched-source counts do
not authorize additional publication. Medium tutorial 53's flattened
Requirements table was repaired in place on 2026-08-30; no duplicate was created.

## Additional Platform Status

| Platform | Confirmed published | Not published / unresolved | Next action |
| --- | --- | --- | --- |
| X / Twitter | Several historical self-authored posts are confirmed on `@yunwei37`; see `published.md` | The configured/planned `@eaborai` account currently showed `此账号不存在`; no full X archive export was done | Replace or verify `@eaborai` in planning/metadata; paginate/export `@yunwei37` before declaring full X history complete |
| LinkedIn | Normal visible browser checks confirmed the `Yusheng Zheng` / `yunwei37` profile and tutorial 50-54 shares; see `platforms/linkedin.json` | Machine ledger maps 11/124 English sources; no unfinished LinkedIn queue task | Monitor existing posts; unmatched sources are not automatic publication tasks |
| Zhihu | Normal visible browser checks confirmed the fsession tutorial publication on 2026-08-14; machine ledger maps 68/107 Chinese sources | 39 configured Chinese sources remain unmatched; the visible profile was rechecked on 2026-09-15 and `/creator` still redirects to `/signin`, with no `z_c0` auth cookie, so the session is not signed in | Resume the first queued AgentCgroup task after a visible Zhihu session is signed in; recheck the title before submission |
| Juejin | Runtime Security cleared review and is public at <https://juejin.cn/post/7683507165231841295>, 48-energy was published at <https://juejin.cn/post/7684795356321628170> on 2026-09-12, 47-cuda-events at <https://juejin.cn/post/7684547172777443374> on 2026-09-13, and 46-xdp-test at <https://juejin.cn/post/7685094363799207982> on 2026-09-14, all confirmed public; 45-scx-nest was submitted at <https://juejin.cn/spost/7685561582916009994> on 2026-09-15 and is still under review | No known unresolved Juejin publication defect; the machine ledger maps 42/107 sources, with 65 unmatched and 36 open rolling-queue tasks (one of them, 45-scx-nest, submitted and awaiting review); three documented catch-up gaps remain (2026-08-29, 2026-09-01, 2026-09-03) | Recheck 45-scx-nest review status and confirm its public `/post/` URL, then continue with the next queued Juejin task (44-scx-simple) in the next available Juejin daily slot |
| Medium | All currently queued English articles are confirmed in `platforms/medium.json`; tutorial 53 formatting repair passed public QA on 2026-08-30 | No unfinished Medium queue task or known tutorial 53 formatting defect | Monitor existing articles; do not republish completed items |
| DEV Community | AgentCgroup was confirmed public on 2026-08-27; all current DEV queue items are complete | No unfinished DEV queue task | Monitor the confirmed public pages; queue new sources only when separately authorized |
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
