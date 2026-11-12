# 2026-10-13 内容巡检运行日志

运行日按 America/Los_Angeles 自然日计算：本日正常额度为 LA 2026-10-13。唯一提交时刻为 2026-10-14 07:27 +08（= 2026-10-13 16:27 PDT），落在 LA 2026-10-13。巡检开始时补发缺口为 0 条（10-09 → 10-10 → 10-11 → 10-12 → 10-13 连续巡检，无新增补发缺口），bpf_arena 使用正常额度，未核销任何缺口；本次结束后仍无补发缺口。知乎任务仍 `阻塞`——队列第 14 行 `阻塞` 条款自 2026-09-17 起未恢复（本次未重新探测知乎，亦跳过队列内全部知乎 `排队` 行），故首个可执行的 `排队` 任务为掘金 bpf_arena（队列第 120 行；第 119 行为知乎 bpf_arena 行，`阻塞`）；其后下一可执行任务为第 122 行掘金 bpf_iters（第 121 行为知乎 bpf_iters 行，`阻塞`）。本次未启动任何知乎任务。

## 队列任务：掘金 bpf_arena（LA 10-13 正常额度）

- 任务行：`draft/plan/publishing-queue.zh.md` 第 120 行（`排队`，现翻为 `[x]`）。上一可执行掘金行第 118 行 cgroup-tutorial 已于 10-12 巡检完成；第 119 行为知乎 bpf_arena 行（`阻塞`），跳过；下一可执行掘金行第 122 行为 bpf_iters（LA 2026-10-14 正常额度，第 121 行为知乎 bpf_iters 行，`阻塞`）。
- 源文：`docs/tutorials/features/bpf_arena/README.zh.md`（360 行）；发布稿 `draft/media/2026-10-13/bpf_arena/juejin-body.md`（移除源 H1，其余正文逐字保留；11017 字符、18013 字节、7 个 H2 / 7 个 H3 / 0 个 H4、8 个代码块 [4 c：§BPF Arena 简介两个 / §arena_list.bpf.c / §arena_list.c；3 bash：§编译和执行；1 无语言标签：§编译和执行 预期测试输出]、0 张图片、0 个表格、4 条唯一外链目标 [github eunomia-bpf/bpf-developer-tutorial ×2、eunomia.dev/tutorials/、lwn.net/Articles/961594/]、0 条相对链接）。
- 标题逐字取源 H1「eBPF 实例教程：BPF Arena 零拷贝共享内存」（源 H1 无系列编号，发布稿不另加编号）。
- 草稿未预置：`https://juejin.cn/editor/drafts/new` 新编辑器会话（无预置草稿）；标题经 `input[placeholder*="标题"]` 原生 setter 写入并回读，正文以 base64 分块写入 `window.__b64` 后按 UTF-8 解码注入 CodeMirror，写后回读 11017 字符与本地一致。
- 分类 `后端`（`.category-list .item` 真实 CDP 指针点击）；标签 `Linux`、`后端`、`性能优化`（逐条落定，最终 chip DOM 顺序 `Linux`、`后端`、`性能优化`，沿用 17-biopattern / 19-lsm-connect / 1-helloworld / bpftrace-tutorial / cgroup 教程标签组）。

## 提交

- 提交时刻 2026-10-14 07:27 +08 = 2026-10-13 16:27 PDT；提交前创作者中心基线 全部 82 / 已发布 82 / 审核中 0 / 未通过 0（cgroup 提交后基线）。单次弹窗「确定并发布」返回「发布成功」并落到 `https://juejin.cn/published`。新文 id `7695977967902933026`，URL `https://juejin.cn/post/7695977967902933026`。提交后创作者中心 全部 83 / 已发布 82 / 审核中 1 / 未通过 0。

## 提交结果与判定

- 进入审核间隔：提交后暂存地址 `/spost/7695977967902933026`；审核中窗口内 SPA shell 对 `/post/` URL 返回 HTTP 200，curl 状态码不是公开信号，公开性以登录创作者中心为准；07:30 +08 轮询翻为 全部 83 / 已发布 83 / 审核中 0（审核间隔约 3 分钟，与 26-sudo / 20-tc / bpftrace-tutorial 直接公开一档的快先例一致；对比最长 1-helloworld 约 173 分钟、40 分钟档 38-btf-uprobe），随后规范 `/post/` URL 上线。
- 记 `confirmed`，带 `staged_url`（审核间隔先例：38-btf-uprobe / 17-biopattern / 1-helloworld / cgroup-tutorial；区别于 bpftrace / 26-sudo / 20-tc 直接公开条目无 `staged_url`）。

## 公开页 QA（登录浏览器）

- bpf_arena `/post/7695977967902933026`：标题逐字「eBPF 实例教程：BPF Arena 零拷贝共享内存」（页面级 H1 仅 1 次，正文区域无重复 H1）；单份正文探测（11017 字符 artifact；渲染 DOM 对代码块空白做归一化，页面 `.post-renderer` innerText 读为 12633）；7 个 H2 / 7 个 H3 / 0 个 H4；8 个代码块 [4 c / 3 bash / 1 无语言标签]；0 张图片（正文区域无内容图，页面 15–19 张 `<img>` 全部为站点 chrome：logo / 头像 / `lf-web-assets` UI 贴图，无一落在 `.post-renderer` 正文内）；0 个表格；4 条唯一外链目标全部改写为 `link.juejin.cn?target=` [github eunomia-bpf/bpf-developer-tutorial ×2、eunomia.dev/tutorials/、lwn.net/Articles/961594/]；无 `审核中` / `文章有更新` / `已被删除` / `找不到页面` 标记；评论 0（`暂无评论数据` 空态）。早期计数（07:30 +08 创作者中心放行检查点）：0 展现 / 1 阅读 / 0 点赞 / 0 评论 / 0 收藏；片刻后复核 0 展现 / 3 阅读。

## 台账更新

- `platforms/juejin.json`：置顶新增 `juejin-7695977967902933026` 一条 `confirmed`（newest-first；带 `id`、`status`、`title`、`url`、`staged_url`、`source_path`、`checked_via` 与 notes，审核间隔故带 `staged_url`），`published[]` 70 → 71，`last_checked` → 2026-10-13。
- `sources.json`：`last_checked` → 2026-10-13（映射由检查器从平台条目的 `source_path` 派生，无需新增 source 键）。
- `published.md`：`Last checked:` → 2026-10-13；cgroup-tutorial 叙述段后补记 2026-10-13 bpf_arena 叙述行（审核间隔、07:30 +08 翻为已发布、约 3 分钟、与 26-sudo / 20-tc / bpftrace-tutorial 快先例一致）；Juejin confirmed 表格首行置顶 bpf_arena（newest-first，含 `/post/` 正式地址与 `/spost/` 暂存地址）。
- `not-published.md`：掘金未映射 39 → 38（69/108 → 70/108）、滚动队列 9 → 8（24 Zhihu and 8 Juejin tasks，共 33 → 32）；掘金状态行前置 bpf_arena 子句。
- `draft/plan/publishing-queue.zh.md`：更新时间 → 2026-10-13；补发缺口段落前置 10-13 巡检句（正常额度 1 条、审核间隔 /spost/ 暂存、07:30 +08 翻为已发布、约 3 分钟、与 26-sudo / 20-tc / bpftrace-tutorial 快先例一致）；Ledger 基线按检查器更新（掘金 38，映射 70/108）；剩余队列掘金 9 → 8、总计 33 → 32；第 120 行翻 `[x]` 并附提交时刻、放行时刻、分类/标签、QA 摘要、正式地址与 `confirmed`；知乎任务保持 `阻塞`。
- `community-feedback.md`：`### 2026-10-13` 检查点新增于 `### 2026-10-12` 之上，含发布 + 审核间隔、公开页 QA + 早期计数、跟踪 top ten 移动、前向指针四条（07:30 +08 检查点 0 展现 / 1 阅读 / 0 点赞 / 0 评论 / 0 收藏；片刻后 0 展现 / 3 阅读），前向指针改指 10-14 正常额度掘金 bpf_iters。
- 发布稿 `draft/media/2026-10-13/bpf_arena/juejin.md`：状态改「已确认公开」（post `7695977967902933026`，`confirmed`），记录正式地址、提交时刻、审核间隔放行时刻、QA 摘要；早期计数引用 07:30 +08 检查点。
- 验证器：`check_media_ledger.py` 通过（Juejin 70/108 映射、38 未发布、71 confirmed；exit 0）。

## 编辑器经验

- 正文注入：body 文件 base64 化后分块写入 `window.__b64`，再 `document.querySelector('.CodeMirror').CodeMirror.setValue(new TextDecoder('utf-8').decode(Uint8Array.from(atob(window.__b64), c => c.charCodeAt(0))))`（直接 `atob` 结果为乱码，必须按 UTF-8 解码）。
- 分类 chip（`.category-list .item`）、标签选项（`.byte-select-option`）与「确定并发布」必须用 raw CDP `Input.dispatchMouseEvent`——页内合成 `dispatchEvent` 不会真正选中/发布；提交成功判据为弹窗「发布成功」并落地 `https://juejin.cn/published`。
- 驱动走 `agent-browser-wrapper.sh`（daemon 已挂 CDP 9222）：中文经 `eval --stdin < /tmp/*.js` 走文件（inline shell 嵌中文字符串会撑破解析器）；计数读取沿用 10-12 对 `.essays-container a.link` 逐锚点 `innerText` 正则行配对法。

## 跟踪 top ten 复核（07:30 +08 创作者中心逐行读取）

- 流量累积：[1-helloworld](https://juejin.cn/post/7694934986782064649)（10-10 巡检 10:14 +08 放行）由 10-12 检查点 1129 展现 / 28 阅读累积至 1291 展现 / 32 阅读。
- 未重测：[27-replace](https://juejin.cn/post/7691510191482519603)、[28-detach](https://juejin.cn/post/7691345821565141019)、[29-sockops](https://juejin.cn/post/7691585964054003758)、[31-goroutine](https://juejin.cn/post/7691326130511167526)、[32-wallclock-profiler](https://juejin.cn/post/7691151105851031590)、[33-funclatency](https://juejin.cn/post/7690830871915216922)、[34-syscall](https://juejin.cn/post/7690415131084324902)、[35-user-ringbuf](https://juejin.cn/post/7689742195037110323)、[37-uprobe-rust](https://juejin.cn/post/7689408456146599963) 九条已随两条较新帖（cgroup / bpf_arena）挤入而下移出创作者中心「最新十条」可视窗口（全部 tab / 已发布 tab 均只渲染最新 10 行），故本次未逐行重测其计数；全部被跟踪行仍读 0 评论（`暂无评论数据` 空态持续），无需回复或更正。
- 新文 bpf_arena 在 07:30 +08 放行检查点读 0 展现 / 1 阅读，片刻后 0 展现 / 3 阅读。
- 复核方法：登录创作者中心 `https://juejin.cn/creator/content/article/essays?status=all`，对每行 `.essays-container a.link` 锚点取 `innerText` 并按 `(\d+)\s*展现\s*·\s*(\d+)\s*阅读\s*·…` 逐行配对（行精确，避免跨行错位配对）。


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
