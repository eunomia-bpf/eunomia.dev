# 2026-10-12 内容巡检运行日志

运行日按 America/Los_Angeles 自然日计算：本日正常额度为 LA 2026-10-12。唯一提交时刻为 2026-10-13 07:16 +08（= 2026-10-12 16:16 PDT），落在 LA 2026-10-12。巡检开始时补发缺口为 0 条（10-09 → 10-10 → 10-11 → 10-12 连续巡检，无新增补发缺口），cgroup-tutorial 使用正常额度，未核销任何缺口；本次结束后仍无补发缺口。知乎任务仍 `阻塞`——队列第 14 行 `阻塞` 条款标 `z_c0` 会话自 2026-09-17 起未恢复（本次未重新探测知乎，亦跳过队列内全部知乎 `排队` 行），故首个可执行的 `排队` 任务为掘金 cgroup-tutorial（队列第 118 行）；其后下一可执行任务为第 120 行掘金 bpf_arena（第 119 行为知乎 bpf_arena 行，`阻塞`）。本次未启动任何知乎任务。

## 队列任务：掘金 cgroup-tutorial（LA 10-12 正常额度）

- 任务行：`draft/plan/publishing-queue.zh.md` 第 118 行（`排队`，现翻为 `[x]`）。上一可执行掘金行第 116 行 bpftrace-tutorial 已于 10-11 巡检完成；下一可执行掘金行第 120 行为 bpf_arena（LA 2026-10-13 正常额度，第 119 行为知乎 bpf_arena 行，`阻塞`）。
- 源文：`docs/tutorials/cgroup/README.zh.md`（606 行）；发布稿 `draft/media/2026-10-12/cgroup/juejin-body.md`（移除源 H1，其余正文逐字保留；18184 字符、22850 字节、11 个 H2 / 11 个 H3 / 2 个 H4、11 个代码块 [3 c / 6 bash / 2 无语言标签]、0 张图片、3 个表格 [socket-addr 7 行、device 1 行、sysctl 1 行]、7 条唯一外链目标 [docs.kernel.org/bpf/libbpf/program_types.html、docs.ebpf.io CGROUP_SOCK_ADDR / CGROUP_DEVICE / CGROUP_SYSCTL 各 1、github eunomia-bpf/bpf-developer-tutorial ×2、eunomia.dev/tutorials/]、0 条相对链接）。
- 标题逐字取源 H1「eBPF 实例教程：基于 cgroup 的策略控制」（源 H1 无系列编号，发布稿不另加编号）。
- 草稿未预置：`https://juejin.cn/editor/drafts/new` 新编辑器会话；标题经 `input[placeholder*="标题"]` 原生 `HTMLInputElement` setter 写入并回读，正文以 base64 分块（6000 字符/块）写入 `window.__b64` 后解码注入 CodeMirror，写后回读 18184 字符与本地一致。
- 分类 `后端`（`.category-list .item` 真实 CDP 指针点击）；标签 `Linux`、`后端`、`性能优化`（逐条落定，最终 chip DOM 顺序 `Linux`、`后端`、`性能优化`，沿用 17-biopattern / 19-lsm-connect / 1-helloworld / bpftrace-tutorial 教程标签组）。

## 提交

- 提交时刻 2026-10-13 07:16 +08 = 2026-10-12 16:16 PDT；提交前创作者中心基线 全部 81 / 已发布 81 / 审核中 0 / 未通过 0。单次弹窗「确定并发布」返回「发布成功」并落到 `https://juejin.cn/published`。新文 id `7695667191396040738`，URL `https://juejin.cn/post/7695667191396040738`。提交后创作者中心 全部 82 / 已发布 81 / 审核中 1 / 未通过 0。

## 提交结果与判定

- 进入审核间隔：提交后暂存地址 `/spost/7695667191396040738`；审核中窗口内 SPA shell 对 `/post/` URL 返回 HTTP 200，curl 状态码不是公开信号，公开性以登录创作者中心为准；07:56 +08 轮询翻为 全部 82 / 已发布 82 / 审核中 0（审核间隔约 40 分钟，与 38-btf-uprobe 先例一致；对比直接公开先例 26-sudo / 20-tc / bpftrace 约 3 分钟、最长 1-helloworld 约 173 分钟），随后规范 `/post/` URL 上线。
- 记 `confirmed`，带 `staged_url`（审核间隔先例：38-btf-uprobe / 17-biopattern / 1-helloworld；区别于 bpftrace 直接公开条目无 `staged_url`）。

## 公开页 QA（登录浏览器）

- cgroup-tutorial `/post/7695667191396040738`：标题逐字「eBPF 实例教程：基于 cgroup 的策略控制」（页面级 H1 仅 1 次，正文区域无重复 H1）；单份正文探测（18184 字符 artifact；渲染 DOM 对代码块空白做归一化，页面 innerText 读为 17273）；11 个 H2 / 11 个 H3 / 2 个 H4；11 个代码块 [3 c / 6 bash / 2 无语言标签]；0 张图片（正文区域无内容图）；3 个表格；7 条唯一外链目标全部改写为 `link.juejin.cn?target=`；无 `审核中` / `文章有更新` / `已被删除` 标记；评论 0（`暂无评论数据` 空态）。早期计数（08:07 +08 创作者中心检查点）：2 展现 / 1 阅读 / 0 点赞 / 0 评论 / 0 收藏。

## 台账更新

- `platforms/juejin.json`：置顶新增 `juejin-7695667191396040738` 一条 `confirmed`（newest-first；带 `id`、`status`、`title`、`url`、`staged_url`、`source_path`、`checked_via` 与 notes，审核间隔故带 `staged_url`），`published[]` 69 → 70，`last_checked` → 2026-10-12。
- `sources.json`：`last_checked` → 2026-10-12（映射由检查器从平台条目的 `source_path` 派生，无需新增 source 键）。
- `published.md`：`Last checked:` → 2026-10-12；bpftrace-tutorial 叙述段后补记 2026-10-12 cgroup-tutorial 叙述行（审核间隔、07:56 +08 翻为已发布、约 40 分钟、与 38-btf-uprobe 先例一致）；Juejin confirmed 表格首行置顶 cgroup-tutorial（newest-first，含 `/post/` 正式地址与 `/spost/` 暂存地址）。
- `not-published.md`：掘金未映射 40 → 39（68/108 → 69/108）、滚动队列 10 → 9（24 Zhihu and 9 Juejin tasks，共 34 → 33）；掘金状态行前置 cgroup-tutorial 子句（before bpftrace-tutorial）。
- `draft/plan/publishing-queue.zh.md`：更新时间 → 2026-10-12；补发缺口段落前置 10-12 巡检句（正常额度 1 条、审核间隔 /spost/ 暂存、07:56 +08 翻为已发布、约 40 分钟、与 38-btf-uprobe 先例一致）；Ledger 基线按检查器更新（掘金 39，映射 69/108）；剩余队列掘金 10 → 9、总计 34 → 33；第 118 行翻 `[x]` 并附提交时刻、放行时刻、分类/标签、QA 摘要、正式地址与 `confirmed`；知乎任务保持 `阻塞`。
- `community-feedback.md`：`### 2026-10-12` 检查点新增于 `### 2026-10-11` 之上，含发布 + 审核间隔、公开页 QA + 早期计数、跟踪 top ten 移动、前向指针四条（08:07 +08 检查点 2 展现 / 1 阅读 / 0 点赞 / 0 评论 / 0 收藏），前向指针改指 10-13 正常额度掘金 bpf_arena。
- 发布稿 `draft/media/2026-10-12/cgroup/juejin.md`：状态改「已确认公开」（post `7695667191396040738`，`confirmed`），记录正式地址、提交时刻、审核间隔放行时刻、QA 摘要；早期计数引用 08:07 +08 检查点。
- 验证器：`check_media_ledger.py` 通过（Juejin 69/108 映射、39 未发布、70 confirmed；exit 0）。

## 编辑器经验

- 正文注入：body 文件 base64 化后 6000 字符/块写入 `window.__b64`，再 `document.querySelector('.CodeMirror').CodeMirror.setValue(new TextDecoder('utf-8').decode(Uint8Array.from(atob(window.__b64), c => c.charCodeAt(0))))`（直接 `atob` 结果为乱码，必须按 UTF-8 解码）。
- 分类 chip（`.category-list .item`）、标签选项（`.byte-select-option`）与「确定并发布」必须用 raw CDP `Input.dispatchMouseEvent`——页内合成 `dispatchEvent` 不会真正选中/发布；标签逐条落定：focus 第 0 个 `.publish-popup .byte-select__input`，原生 setter 写 `value` + 冒泡 `input` 事件，等约 1s 后 CDP 真实点击匹配的 `.byte-select-option`；已落定 chip 从 `.publish-popup .byte-select__tag` 回读；提交成功判据为 `document.title === '发布成功'`（落地 `https://juejin.cn/published`）。
- 驱动脚本走 `node`：Python 无 `websocket` 模块而 Node 22 有全局 `WebSocket`，CDP 驱动脚本写 `/tmp/cdp_*.js` 经 `node` 执行（inline `node -e` 嵌中文字符串会撑破 shell 解析器，不再用 inline）。
- 计数读取改进：对 `.essays-container a.link` 每锚点直接 `innerText` 正则匹配 `/(\d+)\s*展现\s*·\s*(\d+)\s*阅读\s*·…/` 逐行读取（行精确，替代 10-11「单用 body.innerText 正则恒空」的上溯法）。

## 跟踪 top ten 复核（08:07 +08 创作者中心逐行读取）

- 上移：[27-replace](https://juejin.cn/post/7691510191482519603) 22→23 阅读、[29-sockops](https://juejin.cn/post/7691585964054003758) 74→75 阅读（1 点赞 / 1 收藏 held）、[31-goroutine](https://juejin.cn/post/7691326130511167526) 24→25 阅读、[32-wallclock-profiler](https://juejin.cn/post/7691151105851031590) 19→21 阅读。
- 流量累积：[1-helloworld](https://juejin.cn/post/7694934986782064649)（10-10 巡检 10:14 +08 放行）由 10-11 检查点 639 展现 / 19 阅读累积至 1129 展现 / 28 阅读。
- 未动：[28-detach](https://juejin.cn/post/7691345821565141019) 38、[33-funclatency](https://juejin.cn/post/7690830871915216922) 12、[34-syscall](https://juejin.cn/post/7690415131084324902) 13、[35-user-ringbuf](https://juejin.cn/post/7689742195037110323) 17、[37-uprobe-rust](https://juejin.cn/post/7689408456146599963) 21。
- 全部被跟踪行仍读 0 评论（`暂无评论数据` 空态持续），新文 cgroup-tutorial 在 08:07 +08 复核时读 2 展现 / 1 阅读 / 0 点赞 / 0 评论 / 0 收藏；无需回复或更正。
- 复核方法：登录创作者中心 `https://juejin.cn/creator/content/article/essays?status=all`，对每行 `.essays-container a.link` 锚点取 `innerText` 并按 `(\d+)\s*展现\s*·\s*(\d+)\s*阅读\s*·…` 逐行配对（行精确，避免跨行错位配对）。


## 2026-10-12 eBPF Q&A run-report (eunomia-community-radar)

## Selected candidate

- Slug: `nodejs-genai-instrumentation-otel-js-contrib-vs-dedicated-repo`
- Question: why Node.js GenAI instrumentation lives in
  `opentelemetry-js-contrib` rather than a dedicated repository, and why
  there is no Node.js equivalent to the Python GenAI instrumentation repo.
- Source: the 10-12 snapshot carries fourteen messages, all in the two
  CNCF OpenTelemetry instrumentation channels
  (`otel-ebpf-instrumentation` / `otel-genai-instrumentation`). The thread
  this page answers: a practitioner asked whether there were plans for a
  Node.js repository similar to the Python GenAI instrumentation repo; the
  reply was that, as of now, there were no known plans for one, and that the
  GenAI instrumentation is landing in `opentelemetry-js-contrib` with the
  auto-generated conventions landing in `opentelemetry-js`, with
  `semantic-conventions-genai` issue 247 the only visible guidance on which
  artifacts to generate and whether they live in separate repos.
- Decisive boundary: the distribution split. The GenAI work separates into a
  language-neutral conventions layer (`semantic-conventions-genai`) that
  federates into the core `opentelemetry-js` client, and an asymmetric
  instrumentation layer - Python has a dedicated `opentelemetry-python-genai`
  repo, while Node.js has no standalone GenAI repo and its OpenAI
  auto-instrumentation ships from the `opentelemetry-js-contrib` monorepo.
  The absence of a Node.js GenAI repo is the expected result of that
  layout, not an access gap.
- This is the Node.js GenAI repository thread that the 10-11 run deferred
  ("no decisive technical boundary this run"). This run supplies that
  boundary: the conventions/instrumentation distribution split, plus the
  Python dedicated-repo asymmetry. It is non-duplicative against the four
  saturated telemetry-data-model threads - 10-05
  (offered-vs-executed blind spot), 10-09 (model-serving-signals), 10-10
  (logger.enabled opt-in/opt-out), and 10-11 (available-skills-as-attribute).

## Verification against public primary sources

- OpenTelemetry semantic-conventions-genai README
  (`github.com/open-telemetry/semantic-conventions-genai`): the
  language-neutral source of truth for GenAI spans, metrics, events, and the
  attribute registry; it extends the core semantic-conventions repo and
  generates per-language reference material, so it has no per-language home
  and carries no per-language instrumentation code.
- OpenTelemetry semantic-conventions-genai issue 247
  (`github.com/open-telemetry/semantic-conventions-genai/issues/247`): the
  discussion of which artifacts should be generated for a language and
  whether they belong in separate repos - the only visible guidance on the
  repo-organization question.
- OpenTelemetry opentelemetry-python-genai
  (`github.com/open-telemetry/opentelemetry-python-genai`): the dedicated
  Python GenAI instrumentation repo, with per-client packages (OpenAI,
  Anthropic, Bedrock, LangChain, Llama-Index, OpenAI Agents, and others)
  built on a shared GenAI helper package - the layout that has no Node.js
  counterpart.
- OpenTelemetry opentelemetry-js-contrib
  (`github.com/open-telemetry/opentelemetry-js-contrib`), specifically the
  `packages/instrumentation-openai` package: the single monorepo for
  community-maintained JS instrumentation, where the OpenAI GenAI
  auto-instrumentation lives as one package among many.
- OpenTelemetry opentelemetry-js release notes (1.42.0): the GenAI semantic
  conventions federate into the shared `semantic-conventions-genai` repo and
  their auto-generated bindings ship in the core `opentelemetry-js`
  client - why the conventions side is not split per language.

## Privacy + content

- No names/handles/Slack-Discord URLs/timestamps/message-sequence
  numbers/employer/team/internal repo names/IPs/private logs in either
  page; anonymized summary only. Public GitHub repository, issue, and doc
  links appear in `## References` / `## 参考` only.
- Content order per the standard: direct answer, mechanism,
  verification/debugging path, limitation, references, community
  discussion. H1s kept free of underscores, backticks, and apostrophes;
  the body's longest backticked token is the 26-character
  `semantic-conventions-genai`, well under the 390px gate. The one 41-
  character token, `@opentelemetry/auto-instrumentations-node`, was
  rewritten in words in the body ("the standard auto-instrumentations
  bundle for Node") and kept in its exact form only in a References item.

## Coverage disclosure

Two opt-in archive channels covered (the two CNCF OpenTelemetry
instrumentation channels) with fourteen messages in the 10-12 snapshot.
The candidate was selected from the Node.js GenAI repository-organization
thread. Three other threads in the window were skipped because they echo
already-published entries: the `model-serving-signals` catalog question
(10-09), the `logger.enabled` opt-in/opt-out question (10-10), and the
available-skills span-attribute modeling question (10-11). Visible-
browser-only sources (the eunomia-bpf and sched-ext Discord servers, the
bpf mailing list, and r/eBPF) were not reviewed this run (no visible-
browser session) - marked uncovered-not-quiet on the page.

## Artifacts

- EN: `docs/ebpf-qa/2026-10-12-nodejs-genai-instrumentation-otel-js-contrib-vs-dedicated-repo.md`
- ZH: `docs/ebpf-qa/2026-10-12-nodejs-genai-instrumentation-otel-js-contrib-vs-dedicated-repo.zh.md`
- Index links: first items of `docs/ebpf-qa/index.md` (`Latest Answers`)
  and `docs/ebpf-qa/index.zh.md` (`最新回答`).
- Published QA commit on `origin/main`: `5c2bf5ca3`
  (`docs(ebpf-qa): nodejs-genai-instrumentation-otel-js-contrib-vs-dedicated-repo (2026-10-12)`),
  containing exactly the four owned paths.
- Receipt: `/workspaces/.agent-state/eunomia-qa/receipt-2026-10-12.json`
  `status=published` (written by the validator only), commit
  `5c2bf5ca360e4851e45e8b0654c4ab9a89db9c53`.

## eBPF Q&A publication follow-up (deploy + live verification)

- QA commit `5c2bf5ca3` (4 paths) landed on `origin/main`; the GitHub
  Pages `Deploy Static App` run `38322559202` went green on that commit
  (`conclusion: success`), so all four routes returned 200 with the
  expected H1s and the new slug present in both indexes.
  `cache-control: public, max-age=0, must-revalidate` and
  `cf-cache-status: DYNAMIC` confirm a fresh deploy, not a cached copy.
- The two live article pages render the verbatim EN and ZH H1s; the new
  slug is linked from both the EN and ZH index pages (the slug substring
  appears six times in each index page's raw HTML - the link plus its
  surrounding metadata occurrences).
- The validator re-verify pass returned receipt `status=published`, commit
  `5c2bf5ca360e4851e235714a66f6c1f4`, with `branch`, `candidate_paths`,
  `index_links`, `privacy`, `remote_contains_commit`, and `public` all
  `ok`, and `content_test` / `build` / `render` / `commit_push`
  `skipped_already_published` (the re-verify path does not re-run those
  because the original commit already passed them).
- Two concrete operational notes hit this run:
  1. The first validator run was capped at 300s, which is shorter than
     the local `npm test:content` + build + Chromium render pass, so it
     was killed *before* `commit_and_push` - nothing landed. This is the
     pre-commit mirror of the 10-11 post-commit timeout. Because
     `origin/main` had advanced to `f0583c838` in the meantime, the
     recovery was to forward-move with `git reset --soft origin/main`
     (preserving the concurrent staged set and worktree), then re-run the
     validator with a cap long enough to finish build + render + commit
     (it completed in ~1800s, landing `5c2bf5ca3`).
  2. That run's `public` check then 404'd only because the Pages deploy
     (15-17 minutes) had not finished. Watching the green Pages run and
     re-running the same validator produced the `status=published`
     receipt, exactly as the 10-11 recovery prescribes.

This run-log commits separately as the 10-12 same-day artifact.
