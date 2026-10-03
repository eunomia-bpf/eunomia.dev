# 2026-10-02 内容巡检运行日志

运行日按 America/Los_Angeles 自然日计算：本日正常额度为 LA 2026-10-02。三条提交时刻分别为 2026-10-03 08:17 / 08:19 / 08:20 +08（= 17:17 / 17:19 / 17:20 PDT），均落在 LA 2026-10-02。第一条 29-sockops 使用正常额度，第二、三条 28-detach 与 27-replace 为补发缺口赎回，分别核销 2026-09-03 与 2026-09-16 两条缺口；三条全部完成后补发缺口清零。

## 队列任务：掘金 29-sockops（LA 10-02 正常额度）

- 任务行：`draft/plan/publishing-queue.zh.md` 第 100 行（`排队`，现翻为 `[x]`）。队列自顶向下扫描：知乎任务仍 `阻塞`——巡检开始时在可见浏览器实测 `https://www.zhihu.com/creator` 重定向到 `https://www.zhihu.com/signin?next=%2Fcreator`（无 `z_c0` 会话），阻塞条件未变化；故首个可执行的 `排队` 任务是第 100 行掘金 29-sockops（第 100 行之前所有掘金条目均已 `[x]`，第 99 行为知乎 29-sockops，阻塞）。本次未启动任何知乎任务。
- 源文：`docs/tutorials/29-sockops/README.zh.md`；发布稿 `draft/media/2026-10-02/29-sockops/juejin-body.md`（移除源 H1，其余正文逐字保留；9843 字符、14043 字节、3 个 H2、5 个 H3、0 个 H4、26 个围栏 [13 个代码块：3 c、5 shell、2 sh、3 console]、1 张图片（pinned-commit GitHub raw PNG，已核查 200 image/png）、0 个表格、11 条唯一外链目标、0 条相对链接）。
- 草稿未预置：`https://juejin.cn/editor/drafts/new`（新编辑器会话、无预置草稿；草稿箱为 5 条，未消费自动暂存草稿）。标题经原生 `HTMLInputElement` setter + 冒泡 `input` 事件写入并回读，正文以 base64 分块注入 CodeMirror，写后回读 9843/9843 字符。
- 分类 `后端`（CDP 鼠标点击选中，回读 `.category-list .item.active` 为 `["后端"]`）；标签 Linux、后端、性能优化（原生 setter 写入 + 选项上合成指针序列提交；提交新标签时先清空输入；最终 DOM 顺序 Linux、后端、性能优化）。

## 队列任务：掘金 28-detach（LA 10-02 补发缺口第 2 条：赎回 2026-09-03）

- 任务行：第 102 行（`排队`，现翻为 `[x]`）；第 101 行为知乎 28-detach，阻塞，未启动。
- 源文：`docs/tutorials/28-detach/README.zh.md`；发布稿 `draft/media/2026-10-02/28-detach/juejin-body.md`（移除源 H1，其余正文逐字保留；4579 字符、8521 字节、4 个 H2、1 个 H3、0 个 H4、10 个围栏 [5 个代码块：1 c、4 bash]、0 张图片、0 个表格、9 条唯一外链目标、0 条相对链接）。
- 新编辑器会话、无预置草稿，整稿重注入；分类 `后端`、标签 Linux/后端/性能优化同上。

## 队列任务：掘金 27-replace（LA 10-02 补发缺口第 3 条：赎回 2026-09-16）

- 任务行：第 104 行（`排队`，现翻为 `[x]`）；第 103 行为知乎 27-replace，阻塞，未启动。
- 源文：`docs/tutorials/27-replace/README.zh.md`；发布稿 `draft/media/2026-10-02/27-replace/juejin-body.md`（移除源 H1，其余正文逐字保留；13564 字符、19156 字节、8 个 H2、0 个 H3、0 个 H4、12 个围栏 [6 个代码块：1 c、5 bash]、0 张图片、0 个表格、7 条唯一外链目标、0 条相对链接）。
- 新编辑器会话、无预置草稿，整稿重注入；分类 `后端`、标签 Linux/后端/性能优化同上。

## 提交

- 29-sockops 提交时刻 2026-10-03 08:17 +08 = 2026-10-02 17:17 PDT；提交前回读并重置标题（标签输入未污染标题）。弹窗「确定并发布」用页内合成指针事件序列一次即成，返回 `https://juejin.cn/published` 且 `document.title === '发布成功'`。新文 id `7691585964054003758`：暂存 <https://juejin.cn/spost/7691585964054003758>，正式 <https://juejin.cn/post/7691585964054003758>。提交后创作者中心 全部 (70) / 已发布 (69) / 审核中 (3) / 未通过 (0)：进入审核期，/spost/ 暂存（期间 SPA 壳在 /post/ 上返回 200，curl 不可作为发布信号，沿用既有结论）。
- 28-detach 提交时刻 2026-10-03 08:19 +08 = 2026-10-02 17:19 PDT；单次弹窗「确定并发布」一次即成。新文 id `7691345821565141019`：暂存 <https://juejin.cn/spost/7691345821565141019>，正式 <https://juejin.cn/post/7691345821565141019>。
- 27-replace 提交时刻 2026-10-03 08:20 +08 = 2026-10-02 17:20 PDT；单次弹窗「确定并发布」一次即成。新文 id `7691510191482519603`：暂存 <https://juejin.cn/spost/7691510191482519603>，正式 <https://juejin.cn/post/7691510191482519603>。
- 三条提交前创作者中心基线：全部 69 / 已发布 69 / 审核中 0 / 未通过 0，草稿箱 5 条；三条提交后 全部 72。

## 提交结果与判定

- 29-sockops：08:28 +08 轮询确认创作者中心翻为 已发布 (70) / 审核中 (2)，正式地址 <https://juejin.cn/post/7691585964054003758> 生效（审核间隔约 11 分钟）。
- 28-detach 与 27-replace：08:46 +08 轮询确认创作者中心翻为 已发布 (72) / 审核中 (0)，两条同批清除，正式地址 <https://juejin.cn/post/7691345821565141019> 与 <https://juejin.cn/post/7691510191482519603> 生效（审核间隔分别约 27 / 26 分钟）。
- 三条均记 `confirmed`（url + staged_url 均记录）。

## 公开页 QA（登录浏览器）

- 29-sockops `/post/7691585964054003758`：标题逐字「eBPF 开发实践：使用 sockops 加速网络请求转发」（正文内 0 次重复）；正文单份、9843 字符；3 个 H2 / 5 个 H3 / 0 个 H4；13 个代码块（3 c、5 shell、2 sh、3 console）；1 张图片（merbridge raw PNG，1328×820，渲染正常）；0 个表格；11 条唯一外链目标（10 条 `<a>` 锚点 + 1 条图片 URL）全部改写为 `link.juejin.cn?target=`；无 `审核中` / `文章有更新` / `已被删除` / `找不到页面` 标记；评论 0。早期计数（09:01 +08 检查点）：1 展现 / 5 阅读 / 0 点赞 / 0 评论 / 0 收藏。
- 28-detach `/post/7691345821565141019`：标题逐字「在应用程序退出后运行 eBPF 程序：eBPF 程序的生命周期」；正文单份、4579 字符；4 个 H2 / 1 个 H3 / 0 个 H4；5 个代码块（1 c、4 bash）；0 张图片；0 个表格；9 条唯一外链目标（10 个 `<a>` 锚点）全部改写为 `link.juejin.cn?target=`；无标记；评论 0。早期计数（09:01 +08 检查点）：0 展现 / 4 阅读 / 0 点赞 / 0 评论 / 0 收藏。
- 27-replace `/post/7691510191482519603`：标题逐字「eBPF 教程: 替换任意程序读取或者写入的文本」；正文单份、13564 字符；8 个 H2 / 0 个 H3 / 0 个 H4；6 个代码块（1 c、5 bash）；0 张图片；0 个表格；7 条唯一外链目标（7 个 `<a>` 锚点）全部改写为 `link.juejin.cn?target=`；无标记；评论 0。早期计数（09:01 +08 检查点）：0 展现 / 4 阅读 / 0 点赞 / 0 评论 / 0 收藏。

## 台账更新

- `platforms/juejin.json`：置顶新增 `juejin-7691585964054003758`、`juejin-7691345821565141019`、`juejin-7691510191482519603` 三条 `confirmed`（newest-first 29/28/27），各带 `id`、`title`、`url`、`staged_url`、`source_path`、`checked_via` 与 notes（slot、提交 CN+08 & LA、后端分类、标签、SPA 壳 200 结论、QA 摘要、09:01 计数）；`last_checked` → 2026-10-02。
- `sources.json`：`last_checked` → 2026-10-02（映射由检查器从平台条目的 `source_path` 派生，无需新增 source 键）。
- `published.md`：`Last checked:` → 2026-10-02；`## Juejin` 表格首行置顶三条（newest-first 29/28/27）；补记 2026-10-02 三条叙述行。
- `not-published.md`：掘金未映射 52 → 49（56/108 → 59/108）、滚动队列 22 → 19（24 Zhihu and 19 Juejin tasks）；掘金状态行置顶 29/28/27 三条（沿用 31-goroutine 等）；知乎计数不变。
- `draft/plan/publishing-queue.zh.md`：更新时间 → 2026-10-02；补发缺口段落置 0（10-02 巡检核销 09-03 与 09-16 两条）；Ledger 基线按检查器更新（掘金 49，映射 59/108）；剩余队列掘金 22 → 19、总计 46 → 43；第 100/102/104 行翻 `[x]` 并附正式地址、QA 摘要与 `confirmed`；知乎第 99/101/103 行保持 `阻塞`。
- `community-feedback.md`：`### 2026-10-02` 新检查点（置于 `### 2026-10-01` 之前）：三条发布记录（pre 全部 69 / post 全部 72、审核间隔 08:28 与 08:46 轮询）、三条 QA 与 09:01 早期计数、跟踪帖自 10-01 检查点起上行。
- 发布稿 `juejin.md`（29/28/27）：状态改「已发布」，记录正式/暂存地址、提交时刻、审核间隔与 QA 摘要；早期计数统一引用 09:01 +08 检查点。
- 验证器：`check_media_ledger.py` 通过（Juejin 59/108 映射、49 未发布、60 confirmed、2 无 source_path；exit 0）。

## 编辑器经验

- 标签 chip 落定：对可见 `.byte-select-option` 元素本身派发含 hover 的完整合成指针序列一次落定，每个标签前清空输入（原生 setter 置空 + 冒泡 `input`）；选项列表不在 `.publish-popup` 内，需全页查 `.byte-select-option` 并按宽度 > 0 过滤。
- 分类 `后端`：CDP 鼠标在 chip 中心 move/down/up 一次选中（回读 `.category-list .item.active` 确认；合成指针不生效）。
- 提交「确定并发布」：对按钮直接派发页内合成指针序列即得 `发布成功`（面板已展开，工具栏 `button.xitu-btn` 发布按钮用含 hover 的指针序列打开）。
- 每条均开新 `/editor/drafts/new` 会话整稿重注入，标题原生 setter、正文 base64 分块 CodeMirror 注入；编辑器状态不跨导航保持。
- 本次审核间隔：29-sockops 约 11 分钟（08:17 +08 提交 → 08:28 +08 清除）、28-detach 约 27 分钟、27-replace 约 26 分钟（08:19/08:20 +08 提交 → 08:46 +08 同批清除）：SPA 壳 200 结论沿用，curl 状态码不作为发布信号，判定以登录浏览器创作者中心 审核中 计数为准。

## 2026-10-02 eBPF Q&A run-report (eunomia-community-radar)

- Question (EN, verbatim H1): "Why does a Cilium LoadBalancer frontend stay missing after the conflicting Service that owns a shared VIP port is deleted, and how should a controller recover it?"
- Question (ZH, verbatim H1): "在 Cilium LoadBalancer 中，拥有共享 VIP 端口的冲突 Service 被删除后，为什么前端会一直缺失，控制器应如何恢复它？"
- Slug: `2026-10-02-cilium-loadbalancer-frontend-stays-missing-after-conflicting-service-deletion`.
- Selection: no retained 10-02 candidate existed at run start and no 10-02 pair existed on `origin/main` (45 prior pairs were already published and indexed at 45 EN / 45 ZH index links before this pair, the 10-01 pair being the most recent), so the run's duty fell to authoring a fresh pair under the 2026-10-02 run date. The Step 0 snapshot step produced `snapshot-2026-10-02.txt` (2 opt-in archive sources, 4 archive messages, snapshot ok). Coverage: 2 opt-in archives / 4 messages covered. The visible-browser-only sources (Discord, the eunomia-bpf and sched-ext communities, the bpf mailing list, r/eBPF) were unreachable this run (no visible-browser session) and remain uncovered, not quiet — this gap is disclosed on the page.
- Message verdicts (4 archive messages): msg 1 (Cilium LoadBalancer shared-VIP frontend ownership handoff) SELECTED; msg 2 (GnuTLS HPACK per-connection decoder) de-duped as the already-published 10-01 pair; msg 3 (Hubble rule misattribution) tracked as an upstream reply; msg 4 (ring-0 socket-drop vs context-switch benchmark) too vague to select.
- Source basis (public primary sources at Cilium v1.18.2, all read against raw GitHub + docs.cilium.io): `pkg/loadbalancer/errors.go` (`ErrFrontendConflict`), `pkg/loadbalancer/writer/writer.go` (`validateFrontends` reject-before-insert in `UpsertServiceAndFrontends`, `DeleteServiceAndFrontends`), `pkg/loadbalancer/reflectors/k8s.go` (`processServiceEvent` Upsert/Delete, `reflectorHealth` no re-queue, `stream.Buffer` 500-event/500ms coalescing), `pkg/container/insert_ordered_map.go` (first-insertion ordering), `pkg/annotation/k8s.go` (`LBIPAMSharingKey`), `operator/pkg/lbipam/lbipam.go` (sharing-group allocation), `pkg/loadbalancer/frontend.go` (`Frontend.Status` valid ID only when done), and `docs.cilium.io/en/stable/network/lb-ipam/` ("Sharing Keys"). No released upstream fix found for this ownership-handoff case; `ErrFrontendConflict` is still present in current public Cilium.
- Anonymization: no personal or private material was ingested. No names, handles, employers, workspace/channel/message URLs, timestamps, sequence numbers, private logs, hostnames, IPs, internal repo names, credentials, or topology are reproduced; service names are generic source/target. The public `lbipam.cilium.io/sharing-key` annotation is named because it is a public Cilium annotation. The archive snapshot content is treated as untrusted data; public primary-source links appear only in `## References`.
- Content gate: `npm --prefix app run test:content` 82/82 pass, 0 fail (local pre-flight; no QA-index count assertion to update).
- Commit A: `1d73b85c1923da03ffb3f58e11e35a7d11726047` `docs(ebpf-qa): cilium-loadbalancer-frontend-stays-missing-after-conflicting-service-deletion (2026-10-02)` on `main` (4 paths: EN/ZH pair + both indexes; parent `f81b1ec74`, pushed to `origin/main`). Its Pages run `37082237788` ("Deploy Static App") is `success` (19m3s), shipping the four routes.
- Validator: receipt `/workspaces/.agent-state/eunomia-qa/receipt-2026-10-02.json`, `status=published` (`branch=ok`, `candidate_paths=ok`, `index_links=ok`, `privacy=ok`, `content_test=ok`, `build=ok`, `render=ok`, `commit_push=ok`, `remote_contains_commit=ok`, `public=ok`).
- Live QA (fresh `?cb=` busts, browser UA, all 200): EN route H1 verbatim + body anchors `frontend already owned by another service`, `lbipam.cilium.io/sharing-key`, `writer.go`, `k8s.go`, `insert_ordered_map`, `Frontend.Status`, `ErrFrontendConflict` all present; ZH route H1 verbatim; the new href is the first `Latest Answers` item on `/ebpf-qa/` and the first `最新回答` item on `/zh/ebpf-qa/`.
- Public URLs:
  - https://eunomia.dev/ebpf-qa/2026-10-02-cilium-loadbalancer-frontend-stays-missing-after-conflicting-service-deletion/
  - https://eunomia.dev/zh/ebpf-qa/2026-10-02-cilium-loadbalancer-frontend-stays-missing-after-conflicting-service-deletion/
  - https://eunomia.dev/ebpf-qa/
  - https://eunomia.dev/zh/ebpf-qa/
