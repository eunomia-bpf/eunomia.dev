# Not Published / Pending Media Ledger

Last checked: 2026-07-20

This file tracks source posts that still need platform work. The latest checked `.github/publisher/posts_queue.txt` is empty; historical rows below are retained as candidate-source tracking, not proof that Medium/Dev.to are still queued.

For full per-platform counts across all configured source files, run `python .github/publisher/media/check_media_ledger.py --show-missing`. The canonical machine-readable ledger uses one JSON file per platform under [`platforms/`](platforms/).

## Current Publisher Queue

| Source | Title | Medium/Dev.to | Zhihu | Juejin | Next action |
| --- | --- | --- | --- | --- | --- |
| `docs/tutorials/50-tcx` | eBPF Tutorial by Example 50: Composable Traffic Control with TCX Links | Confirmed on both | Confirmed missing | Confirmed missing | Publish the unchanged Chinese source on the scheduled Zhihu and Juejin dates |
| `docs/blog/posts/cpu-noise-gpu-inference.md` | When CPU Noise Slows Down GPU Inference: Measuring Scheduler and IRQ Impact with eBPF | Confirmed on both | Confirmed missing | Confirmed missing | Publish the unchanged Chinese source on the scheduled Zhihu and Juejin dates |
| `docs/blog/posts/runtime-security-for-opaque-ai-agents.md` | Runtime Observability and Enforcement for Opaque AI Agents with eBPF: Beyond Sandboxes and Approvals | Confirmed on both | Confirmed | Confirmed missing | Publish the unchanged Chinese source to Juejin on its scheduled date |
| `docs/blog/posts/agent-check-restore-safety.md` | ACRFence: Preventing Semantic Rollback Attacks in Agent Checkpoint-Restore | Confirmed on both | Confirmed | Confirmed missing | Publish the unchanged Chinese source to Juejin on its scheduled date |

## Additional Platform Status

| Platform | Confirmed published | Not published / unresolved | Next action |
| --- | --- | --- | --- |
| X / Twitter | Several historical self-authored posts are confirmed on `@yunwei37`; see `published.md` | The configured/planned `@eaborai` account currently showed `此账号不存在`; no full X archive export was done | Replace or verify `@eaborai` in planning/metadata; paginate/export `@yunwei37` before declaring full X history complete |
| LinkedIn | In-app browser/sidebar confirmed `Yusheng Zheng` / `yunwei37` profile and visible authored posts for ActPlane/AgentSight, ACRFence, GPU observability, agentpprof, and BPFix; see `platforms/linkedin.json` | Current script coverage is 6/122 English target sources; several search-visible posts still lack exact post permalinks | Continue through normal LinkedIn browser UI only; scroll recent activity to backfill exact permalinks and older posts |
| Zhihu | Normal visible scrolling collected 113 unique article links from a profile reporting 114 articles; machine ledger maps 63/105 Chinese sources | 42 configured Chinese sources remain unmatched | Publish only ledger-confirmed gaps and recheck the visible title immediately before submission |
| Juejin | Normal visible pagination covered four pages and 39 unique authored article links; machine ledger maps 25/105 Chinese sources | 80 configured Chinese sources remain unmatched | Publish unchanged Chinese sources through the platform's independent schedule and normal editor |
| Medium | Normal visible scrolling collected 62 authored story links; machine ledger maps 60/122 English sources | 62 configured English sources remain unmatched | Publish confirmed gaps through the platform's independent schedule and normal Medium web editor |
| DEV Community | Profile reports 57 posts and visible scrolling collected 54 unique links; machine ledger maps 47/122 English sources | 75 configured English sources remain unmatched | Publish confirmed gaps through the platform's independent schedule and normal DEV web editor |
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
