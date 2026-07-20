# Not Published / Pending Media Ledger

Last checked: 2026-07-19

This file tracks source posts that still need platform work. The latest checked `.github/publisher/posts_queue.txt` is empty; historical rows below are retained as candidate-source tracking, not proof that Medium/Dev.to are still queued.

For full per-platform counts across all configured source files, run `python .github/publisher/media/check_media_ledger.py --show-missing`. The canonical machine-readable ledger uses one JSON file per platform under [`platforms/`](platforms/).

## Current Publisher Queue

| Source | Title | Medium/Dev.to | Zhihu | Juejin | Next action |
| --- | --- | --- | --- | --- | --- |
| `docs/tutorials/50-tcx` | eBPF Tutorial by Example 50: Composable Traffic Control with TCX Links | Queue empty; ledger verification needed | Not verified on recent Zhihu article page | Not visible on first Juejin article page | Prepare Chinese platform copy; likely publish to both Zhihu and Juejin after site canonical is live |
| `docs/blog/posts/cpu-noise-gpu-inference.md` | When CPU Noise Slows Down GPU Inference: Measuring Scheduler and IRQ Impact with eBPF | Queue empty; ledger verification needed | Not visible on recent Zhihu article page | Not visible on first Juejin article page | Good candidate for Zhihu/Juejin; needs Chinese title and diagrams checked |
| `docs/blog/posts/runtime-security-for-opaque-ai-agents.md` | Runtime Observability and Enforcement for Opaque AI Agents with eBPF: Beyond Sandboxes and Approvals | Queue empty; ledger verification needed | Appears published on Zhihu as `基于 eBPF 的不透明 AI Agent 运行时可观测与执行控制：超越沙箱与审批` | Not visible on first Juejin article page | Verify exact source mapping; publish or schedule for Juejin if not already there |
| `docs/blog/posts/agent-check-restore-safety.md` | ACRFence: Preventing Semantic Rollback Attacks in Agent Checkpoint-Restore | Queue empty; ledger verification needed | Appears published on Zhihu as `ACRFence：防止 AI Agent 检查点恢复中的语义回滚攻击` | Not visible on first Juejin article page | Verify exact source mapping; publish or schedule for Juejin if not already there |

## Additional Platform Status

| Platform | Confirmed published | Not published / unresolved | Next action |
| --- | --- | --- | --- |
| X / Twitter | Several historical self-authored posts are confirmed on `@yunwei37`; see `published.md` | The configured/planned `@eaborai` account currently showed `此账号不存在`; no full X archive export was done | Replace or verify `@eaborai` in planning/metadata; paginate/export `@yunwei37` before declaring full X history complete |
| LinkedIn | In-app browser/sidebar confirmed `Yusheng Zheng` / `yunwei37` profile and visible authored posts for ActPlane/AgentSight, ACRFence, GPU observability, agentpprof, and BPFix; see `platforms/linkedin.json` | Current script coverage is 5/118 English target sources; several search-visible posts still lack exact post permalinks | Continue through normal LinkedIn browser UI only; scroll recent activity to backfill exact permalinks and older posts |
| Reddit | Historical `u/yunwei123` posts are confirmed for eBPF tutorial, GPTtrace, Code-Survey-like discussion, and Wasm-bpf; see `published.md` | No evidence yet that current 2026 flagship posts were submitted to r/eBPF, r/netsec, r/LocalLLaMA, HN, or lobste.rs | Use Reddit only manually for flagship posts; check subreddit fit before posting |
| Xiaohongshu / RedNote | None confirmed | No account URL or published note URL found; current browser search requires login; public search found no clear `eunomia.dev` / `eunomia-bpf` / `bpftime` / `AgentSight` result | Treat as not started; create account and visual-note workflow only after there is image-card/video capacity |

## Pending Verification

These items need follow-up before moving to `published.md` or clearing them:

| Item | Why it needs verification | Suggested check |
| --- | --- | --- |
| Full Zhihu article history | Profile reports 111 articles, but this check only recorded the visible article page snapshot | Paginate or export profile article list before declaring the ledger complete |
| Full Juejin article history | This check only recorded the first article page | Use Juejin article pagination or profile API carefully to capture older posts |
| Full X history | `from:yunwei37` search recorded visible project hits, but not a complete account export | Use X advanced search or account data export before declaring the ledger complete |
| Full Reddit history | `author:yunwei123` search recorded visible project hits, but not all comments or deleted/crossposted content | Use Reddit user listing and subreddit search for `eunomia.dev`, `eunomia-bpf`, `bpftime`, `AgentSight` |
| Xiaohongshu login/account | Search results were hidden behind login and no account URL is known | Sign in or provide account URL before doing a definitive platform audit |
| Repo-referenced platform URLs | Some links appear in documentation as references; they may be our posts, partner posts, or citations | Open each URL and confirm author/account before marking as confirmed |
| Medium/Dev.to published state | Existing queue tracks pending automation, but this check did not inspect Medium/Dev.to accounts | Run publisher dry run or inspect platform dashboards separately |

## Add New Pending Items

Use this format when adding a post:

```md
| `source/path.md` | Human title | Not queued / queued | Not published / draft / published URL | Not published / draft / published URL | Next concrete action |
```
