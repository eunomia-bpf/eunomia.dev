# Not Published / Pending Media Ledger

Last checked: 2026-08-17

This file tracks source posts that still need platform work. The latest checked `.github/publisher/posts_queue.txt` is empty; historical rows below are retained as candidate-source tracking, not proof that Medium/Dev.to are still queued.

For full per-platform counts across all configured source files, run `python .github/publisher/media/check_media_ledger.py --show-missing`. The canonical machine-readable ledger uses one JSON file per platform under [`platforms/`](platforms/).

## Current Publisher Queue

The machine ledger currently reports 39 unmatched Chinese sources on Zhihu and
75 on Juejin. These raw counts include paused material, non-standalone index or
reference pages, and duplicate source representations. The ordered platform
actions that remain suitable candidates are maintained one-by-one in
`draft/plan/publishing-queue.zh.md`; each checkbox is one platform task and the
daily patrol completes at most one.

The rolling queue also includes one LinkedIn link-share task for each remaining
new tutorial from 52 through 54. SchedCP, AgentCgroup, CPU noise, Weekly Analysis
reports, and content with confirmed LinkedIn publication records are not
included in this LinkedIn batch.

Medium and DEV have completed BPFix and tutorials 50 through 52. Each platform still
has two faithful long-form syndication tasks for tutorials 53 and 54. These
tasks preserve the English source title and body and allow only the mechanical
rendering adaptations required by each platform.

## Additional Platform Status

| Platform | Confirmed published | Not published / unresolved | Next action |
| --- | --- | --- | --- |
| X / Twitter | Several historical self-authored posts are confirmed on `@yunwei37`; see `published.md` | The configured/planned `@eaborai` account currently showed `此账号不存在`; no full X archive export was done | Replace or verify `@eaborai` in planning/metadata; paginate/export `@yunwei37` before declaring full X history complete |
| LinkedIn | Normal visible browser checks confirmed the `Yusheng Zheng` / `yunwei37` profile and the AgentSight macOS/Windows update; see `platforms/linkedin.json` | Current script coverage is 8/124 English target sources; several search-visible posts still lack exact post permalinks | Continue through normal LinkedIn browser UI only; scroll recent activity to backfill exact permalinks and older posts |
| Zhihu | Normal visible browser checks confirmed the fsession tutorial publication on 2026-08-14; machine ledger maps 68/107 Chinese sources | 39 configured Chinese sources remain unmatched | Publish only ledger-confirmed gaps and recheck the visible title immediately before submission |
| Juejin | Normal visible pagination covered four pages and 40 unique authored article links; the 2026-08-16 public-page check confirmed tutorial 52; machine ledger maps 32/107 Chinese sources | 75 configured Chinese sources remain unmatched | Publish unchanged Chinese sources through the rolling one-platform-per-day queue and normal editor |
| Medium | Tutorial 52 was published by API and passed full public-page QA on 2026-08-17; machine ledger includes the confirmed URL | Run the ledger checker for current unmatched-source coverage | Publish confirmed gaps through the Medium API, then perform visible public-page QA |
| DEV Community | Tutorial 52 was published by API and passed full public-page QA on 2026-08-17; machine ledger includes the confirmed URL | Run the ledger checker for current unmatched-source coverage | Publish confirmed gaps through the DEV API, then perform visible public-page QA |
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
