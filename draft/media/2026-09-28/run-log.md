## 2026-09-28 eBPF Q&A run-report (eunomia-community-radar)

- Question: "Why do BPF task iterators split between sleepable and non-sleepable programs, and which process views fall on each side?" (EN + ZH).
- Slug: `2026-09-28-task-iterator-sleepable-exe-cmdline`.
- Selection: no retained candidate existed at run start (the 09-27 publication was a no-op and no 09-27/09-28 pair existed on `origin/main`), so the run's duty fell to authoring a fresh pair under the 2026-09-28 run date. The Step 0 snapshot of the two watchlist-opted Slack archives returned 4 messages in the rolling window (`snapshot=ok bytes=2872 messages=4 opt_in_sources=2 result=ok`); one message was a repository-internal bug report (discarded under anonymization) and one was non-technical, leaving two substantive threads — a tracepoint-vs-LSM hook cost gap and a `/proc` interval-polling "near end of useful life" discussion — plus a carried-over `sock:inet_sock_set_state` attribution thread from 09-26. Topic verified absent from the de-dup list.
- Source basis: torvalds/linux master @ 2026-09-28 — `kernel/bpf/task_iter.c` (the three task-family `bpf_iter_reg` blocks all set `.feature = BPF_ITER_RESCHED`; the `task_vma` seq walk's `mmap_read_lock_killable` / contention re-acquire / `task_vma_seq_stop` drop; `bpf_find_vma` trylock + IRQ-work deferred unlock; the `bpf_iter_task_new`/`bpf_iter_task_next`/`bpf_iter_task_destroy` kfunc iterator over an opaque `__u64[3]` handle), `kernel/bpf/bpf_iter.c` (`bpf_seq_read` `can_resched` + `cond_resched` + `MAX_ITER_OBJECTS` cap + the one-directional sleepable attach gate `if (prog->sleepable && !bpf_iter_target_support_resched(tinfo)) return -EINVAL;` + the two `bpf_iter_run_prog` RCU branches), `include/linux/bpf.h` (`BPF_ITER_RESCHED` bit, `struct bpf_iter_reg`), `include/linux/mm_types.h` (`arg_start`/`arg_end` + `arg_lock`, the RCU-marked `struct file __rcu *exe_file`), `include/linux/sched.h` (`TASK_COMM_LEN = 16`, `char comm[TASK_COMM_LEN]`), `kernel/bpf/helpers.c` (`bpf_probe_read_user`/`bpf_probe_read_user_str` prototypes). Residual disclosure on the page: no `bpf_mmap_read_lock`/`_unlock` kfunc exists in this mainline snapshot, so the `task_vma` sleepable coupling is grounded on `mmap_read_lock_killable` plus the sleepable run path instead.
- Anonymization: the 4 reachable messages carried a vendor product/blog URL, a personal name, and one repository-internal report; all removed or reduced to role-based "a vendor" / "a thread" phrasing. No names, handles, workspace/channel/message URLs, timestamps, or private deployment details are reproduced. The vendor and blog names that surfaced as leads in the snapshot are not present in the page or this log.
- Content gate: `npm --prefix app run test:content` 82/82 pass, 0 fail (local pre-flight; the Pages pipeline is the authoritative build).
- Commit A: `bfaa6644c54f47fd26cb2075aeeb2a08281f5607` `docs(ebpf-qa): task-iterator-sleepable-exe-cmdline (2026-09-28)` on `main` (4 paths: EN/ZH pair + both indexes), pushed to `origin/main`; Pages run 36393795298 (`deploy-static-app`) completed `success`.
- Validator re-verify: receipt `/workspaces/.agent-state/eunomia-qa/receipt-2026-09-28.json`, `status=published` (`branch=ok`, `candidate_paths=ok`, `index_links=ok`, `privacy=ok`, `remote_contains_commit=ok`, `public=ok`; content-test/build/render/commit-push checks `skipped_already_published`).
- Live QA: H1 verbatim + body anchors (`BPF_ITER_RESCHED`, `mmap_read_lock_killable`, `cond_resched`, `exe_file`, `arg_start`, `bpf_probe_read_user`, `bpf_iter_task_new`) verified on both routes with a fresh `?cb=` bust and a browser UA; both index hrefs live on `/ebpf-qa/` and `/zh/ebpf-qa/`.
- Run note: this run completes the missed 09-27 publication under the 2026-09-28 run date. The `omp-duty.sh` stale-`--continue` defect was fixed in this session: the duty now resumes a session only when a session file matching today's UTC date prefix exists (`ls "$state/omp-sessions" | grep -q "^$(date -u +%F)"`), so a stale older session can no longer re-inject completed work and make the agent stop with zero tool calls.
- Public URLs:
  - EN: https://eunomia.dev/ebpf-qa/2026-09-28-task-iterator-sleepable-exe-cmdline/
  - ZH: https://eunomia.dev/zh/ebpf-qa/2026-09-28-task-iterator-sleepable-exe-cmdline/


# 2026-09-28 内容巡检运行日志

运行日按 America/Los_Angeles 自然日计算：本日正常额度为 LA 2026-09-28。提交时刻 2026-09-29 07:27 +08 = 16:27 PDT，落在 LA 2026-09-28；本日 LA 前无同日发布，使用正常额度，未核销补发缺口（缺口保持 2 条：2026-09-03、2026-09-16）。

## 队列任务：掘金 34-syscall（LA 09-28 正常额度）

- 任务行：`draft/plan/publishing-queue.zh.md` 第 95 行（`排队`，现翻为 `[x]`）。第 94 行知乎同源文仍 `阻塞`（无 `z_c0` 会话，`/creator` 重定向到 `/signin`），本次未涉及。
- 源文：`docs/tutorials/34-syscall/README.zh.md`；发布稿 `draft/media/2026-09-28/34-syscall/juejin-body.md`（移除源 H1，其余正文逐字保留；6113 字符、9043 字节、3 个 H2、0 个 H3、0 个 H4、16 个围栏 [8 个代码块：3 c、3 bash、1 sh、1 console]、0 张图片、0 个表格、5 条唯一外链；无相对链接）。
- 草稿未预置：新标签打开 `https://juejin.cn/editor/drafts/new`，编辑器自动暂存草稿 7690415131084292134 由本次提交消费、草稿箱保持 4 条；标题经原生 `HTMLInputElement` setter + 冒泡 `input` 事件写入并回读，正文以 base64 分块经 `agent-browser eval --stdin` 注入 CodeMirror，写后回读 6113/6113 字符。
- 分类 `后端`（CDP 鼠标点击选中）；提交标签 `后端`、`Linux`、`性能优化`（原生 setter + CDP 选项点击提交，实际落定 chip 顺序已按 DOM 读回确认为 后端、Linux、性能优化）。

## 提交

- 提交时刻 2026-09-29 07:27 +08 = 16:27 PDT；弹窗「确定并发布」用页内真实指针事件序列（pointerover/enter/mouseover/mouseenter/pointerdown/mousedown/focus/pointerup/mouseup/click），一次即成，返回 `https://juejin.cn/published` 且 `document.title === '发布成功'`。
- 新文 id `7690415131084324902`：暂存 <https://juejin.cn/spost/7690415131084324902>，正式 <https://juejin.cn/post/7690415131084324902>。提交前创作者中心 全部 (65) / 已发布 (65) / 审核中 (0) / 未通过 (0)；提交后 全部 (66) / 已发布 (65) / 审核中 (1) / 未通过 (0)。

## 提交结果与判定

- 提交后 curl 探测不可作为发布信号（掘金 SPA 壳对任意 `/post/<id>`、`/spost/<id>` 路径在审核期即返回 200，沿用既有结论）；判定以登录浏览器正文渲染 + 创作者中心 审核中 计数为准。
- 浏览器轮询（≤175 秒步长）：09:52 +08 创作者中心翻为 已发布 (66) / 审核中 (0)（约 145 分钟审核间隔，长于 35 的 65 分钟与 37/38 的 36–40 分钟），`/post/7690415131084324902` 渲染公开正文，`/spost/` 跳转至 `/post/`。记 `confirmed`（url + staged_url 均记录）。

## 公开页 QA（/post/7690415131084324902，登录浏览器）

- 标题逐字「eBPF 开发实践：使用 eBPF 修改系统调用参数」（title_occurrences==1、正文 h1==0，标题由 title 字段承载）；正文单份。
- 3 个 H2 / 0 个 H3 / 0 个 H4；8 个代码块（`pre code` 语言类统计：3 c、3 bash、1 sh、1 console）；0 张正文图片；0 个表格。
- 5 条唯一外链目标全部被掘金改写为 `link.juejin.cn?target=…`（共 6 个链接：github bpf-developer-tutorial 仓库根 ×2、github src/34-syscall ×1、github eunomia-bpf ×1、eunomia.dev/tutorials/1-helloworld/ ×1、eunomia.dev/zh/tutorials/ ×1），正文内 0 相对链接。
- 无 `审核中` / `文章有更新` / `已被删除` 标记；评论 0（`暂无评论数据` 空态，经两次独立探测确认）。早期计数（09:52 +08 检查点）：2 展现 / 5 阅读 / 0 点赞 / 0 评论 / 0 收藏。

## 台账更新

- `platforms/juejin.json`：新增 `juejin-7690415131084324902`（`confirmed`，置顶至第 54 条），`last_checked` → 2026-09-28；notes 记录审核期、curl 不可信结论、自动暂存草稿消费与 QA 摘要。
- `sources.json`：`last_checked` → 2026-09-28（映射由检查器从平台条目的 `source_path` 派生，无需新增 source 键）。
- `published.md`：`Last checked:` → 2026-09-28；`## Juejin` 表格首行新增该条目；补记 2026-09-28 叙述行。
- `not-published.md`：`Last checked:` → 2026-09-28；掘金未映射 56 → 55（53/108）、滚动队列 26 → 25；掘金状态行置顶 34-syscall（下一位 35-user-ringbuf）；知乎计数不变（40 未映射、68/108）。
- `draft/plan/publishing-queue.zh.md`：更新时间 → 2026-09-28；补发缺口段落补 09-28 巡检说明（正常额度、未核销缺口、缺口保持 2 条）；Ledger 基线按检查器更新（掘金 55，映射 53/108）；剩余队列掘金 26 → 25、总计 50 → 49；第 95 行翻 `[x]` 并附正式地址、QA 摘要与 `confirmed`（53/108）；第 94 行（知乎）保持 `阻塞`。
- `community-feedback.md`：`### 2026-09-28` 新检查点（置于 `### 2026-09-27` 之前）：发布记录、审核间隔 07:27→09:52 +08（约 145 分钟）、早期计数 2 展现 / 5 阅读、35-user-ringbuf 3 / 8、37-uprobe-rust 16 / 14、38-btf-uprobe 9 / 14、41-xdp-tcpdump 4403 / 48 / 1 收藏、全部可见行仍 0 评论、下一检查点在 33-funclatency 额度后。
- 发布稿 `juejin.md`：状态改「已发布」，记录正式地址、审核时间线与 QA 摘要。
- 验证器：`check_media_ledger.py` 通过（Juejin 53/108 映射、55 未发布、54 confirmed；exit 0）。

## 编辑器经验

- 弹窗「确定并发布」面板在 CDP 鼠标点击下保持 `display:none`，仅由页内合成指针事件序列（pointerover/enter/mouseover/mouseenter/pointerdown/mousedown/focus/pointerup/mouseup/click）打开；分类与标签仍复用 2026-09-24/25/26/27 已回写的技能条目（真实 CDP 指针事件选分类、原生 setter + CDP 点击选标签、base64 分块经 `eval --stdin` 注入）。
- chip 激活态首次读回与 Vue 提交存在竞态（首次读回需再读一次确认）；已回读确认 后端、Linux、性能优化 落定。
- 提交新选项时先前已提交的 tag 可能掉落（观察到提交 `后端` 后 `Linux` 一度消失，补回）；最终 DOM 顺序 后端、Linux、性能优化。
- 审核清除继续落在分钟级：本次约 145 分钟（35 为 65 分钟，37/38 为 36–40 分钟）；继续按「浏览器轮询至 审核中 归零」判定，SPA 壳 200 不作为发布信号。
