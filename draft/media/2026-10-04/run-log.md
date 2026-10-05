# 2026-10-04 内容巡检运行日志

运行日按 America/Los_Angeles 自然日计算：本日正常额度为 LA 2026-10-04。唯一提交时刻为 2026-10-05 07:41 +08（= 16:41 PDT），落在 LA 2026-10-04。巡检开始时补发缺口为 0 条，24-hide 使用正常额度，未核销任何缺口；本次结束后仍无补发缺口。知乎任务仍 `阻塞`——巡检开始时在可见浏览器复查 `https://www.zhihu.com/creator` 仍重定向到 `https://www.zhihu.com/signin?next=%2Fcreator`（无 `z_c0` 会话），阻塞条件未变化，故首个可执行的 `排队` 任务为掘金 24-hide；本次未启动任何知乎任务。

## 队列任务：掘金 24-hide（LA 10-04 正常额度）

- 任务行：`draft/plan/publishing-queue.zh.md` 第 107 行（`排队`，现翻为 `[x]`）。第 105 行为知乎 26-sudo，`阻塞`，未启动。
- 源文：`docs/tutorials/24-hide/README.zh.md`；发布稿 `draft/media/2026-10-04/24-hide/juejin-body.md`（移除源 H1，其余正文逐字保留；15109 字符、21711 字节、4 个 H2、1 个 H3、0 个 H4、14 个代码块 [9 c、1 bash、1 sh、3 console]、0 张图片、0 个表格、3 条唯一外链目标、0 条相对链接）。
- 草稿未预置：`https://juejin.cn/editor/drafts/new`（新编辑器会话、无预置草稿）；编辑器草稿箱为 5 条，未被本次提交消费。标题经原生 `HTMLInputElement` setter + 冒泡 `input` 事件写入并回读，正文以 base64 分块注入 CodeMirror，写后回读 15109/15109 字符。
- 分类 `后端`（CDP 鼠标在 chip 中心 move 553 169 / down / up 选中，回读 `.category-list .item.active` 为 `["后端"]`；页内合成指针事件不生效）；标签 Linux、后端、性能优化（原生 setter 写入 + 选项上合成指针序列提交；提交新标签时先清空输入；最终 DOM 顺序 Linux、后端、性能优化）。

## 提交

- 提交时刻 2026-10-05 07:41 +08 = 2026-10-04 16:41 PDT；提交前创作者中心基线 全部 73 / 已发布 73 / 审核中 0 / 未通过 0。弹窗「确定并发布」用含 hover 的合成指针事件序列一次即成，返回 `https://juejin.cn/published` 且 `document.title === '发布成功'`。新文 id `7692741078310912027`：进入审核期，`/spost/7692741078310912027` 暂存，创作者中心 审核中 (1)；期间 SPA 壳对 `/post/` 一律回 200，curl 状态码不可作为发布信号，判定以登录态创作者中心为准。

## 提交结果与判定

- 08:35 +08 轮询创作者中心清除为 全部 74 / 已发布 74 / 审核中 0 / 未通过 0：`/post/7692741078310912027` 上线，审核间隔约 54 分钟（07:41 → 08:35 +08）。
- 记 `confirmed`（带 `staged_url` = `<https://juejin.cn/spost/7692741078310912027>`——本次存在审核期暂存，沿用 29-sockops / 28-detach / 27-replace 的 `staged_url` 先例；与 10-03 26-sudo 直接公开、无 `staged_url` 的先例不同）。

## 公开页 QA（登录浏览器）

- 24-hide `/post/7692741078310912027`：标题逐字「eBPF 开发实践：使用 eBPF 隐藏进程或文件信息」；单份正文探测；4 个 H2 / 1 个 H3 / 0 个 H4；14 个代码块 {9 c、1 bash、1 sh、3 console}；0 张图片；0 个表格；3 条唯一外链目标（github bpf-developer-tutorial 仓库根、tree main/src/24-hide、eunomia.dev/zh/tutorials）全部改写为 `link.juejin.cn?target=`；无 `审核中` / `文章有更新` / `已被删除` 标记；评论 0（暂无评论数据 空态）。渲染 innerText 为 14980 字符（稿面 15109，代码块空白归一化所致；结构计数以稿面为准，沿用 27-replace / 26-sudo 记法）。早期计数（08:40 +08 创作者中心检查点）：0 展现 / 1 阅读 / 0 点赞 / 0 评论 / 0 收藏。

## 台账更新

- `platforms/juejin.json`：置顶新增 `juejin-7692741078310912027` 一条 `confirmed`（newest-first；带 `id`、`status`、`title`、`url`、`staged_url`、`source_path`、`checked_via` 与 notes）；`last_checked` → 2026-10-04。
- `sources.json`：`last_checked` → 2026-10-04（映射由检查器从平台条目的 `source_path` 派生，无需新增 source 键）。
- `published.md`：`Last checked:` → 2026-10-04；`## Juejin` 表格首行置顶 24-hide（newest-first，含 `staged_url` 审核期记法）；补记 2026-10-04 叙述行（审核期版式，同 31-goroutine / 29-sockops）。
- `not-published.md`：掘金未映射 48 → 47（60/108 → 61/108）、滚动队列 18 → 17（24 Zhihu and 17 Juejin tasks）；掘金状态行追加 24-hide 子句（alongside 26-sudo 之前）。
- `draft/plan/publishing-queue.zh.md`：更新时间 → 2026-10-04；补发缺口段落追加 10-04 巡检句（正常额度 1 条、审核期 /spost/ 暂存、08:35 +08 清除）；Ledger 基线按检查器更新（掘金 47，映射 61/108）；剩余队列掘金 18 → 17、总计 42 → 41；第 107 行翻 `[x]` 并附正式地址、QA 摘要与 `confirmed`；知乎第 105 行保持 `阻塞`。
- `community-feedback.md`：`### 2026-10-04` 检查点新增发布与 QA 两条（08:40 +08 早期计数 0/1/0/0/0），前向指针改指 10-05 正常额度 22-android。
- 发布稿 `draft/media/2026-10-04/24-hide/juejin.md`：状态改「已发布」，记录正式地址、审核期暂存地址、提交时刻与 QA 摘要；早期计数引用 08:40 +08 检查点。
- 验证器：`check_media_ledger.py` 通过（Juejin 61/108 映射、47 未发布、62 confirmed；exit 0）。

## 编辑器经验

- 与 10-02 / 10-03 一致：分类 `后端` 需 CDP 鼠标在 chip 中心 move/down/up 一次选中（回读 `.category-list .item.active` 确认；页内合成指针不生效）；标签 chip 对可见 `.byte-select-option` 元素派发含 hover 的合成指针序列一次落定，每个标签前清空输入。
- 本次为审核期（区别于 10-03 26-sudo 直接公开）：提交后创作者中心 审核中 (1)、`/spost/` 暂存；08:35 +08 轮询翻为 已发布 (74) / 审核中 (0)，间隔约 54 分钟；审核期间 SPA 壳对 `/post/` 一律回 200，判定以登录浏览器创作者中心 审核中 计数为准。
- 每条开新 `/editor/drafts/new` 会话整稿重注入，标题原生 setter、正文 base64 分块 CodeMirror 注入；编辑器草稿箱 5 条未被本次提交消费（草稿箱计数不变）。

## 2026-10-04 eBPF Q&A run-report (eunomia-community-radar)

## Selected candidate

- Slug: `cilium-hubble-flow-attributed-to-wrong-policy-rule-on-overlap`
- Question: why Hubble attributes a flow to the wrong policy rule when two
  overlapping L3/L4 policy entries match, when the BPF datapath enforces the
  correct entry.
- Source: opt-in archive thread sighting the Hubble misattribution, answered
  in-thread against cilium/cilium#48945 and the fix PR cilium/cilium#49062.
  The thread was first sighted on 10-03 and deferred (fix still in review);
  on 10-04 the fix had landed on main (two commits, 2026-09-29) and the
  upstream issue was closed, making it publishable.
- Other 10-04 archive threads: the LoadBalancer shared-VIP frontend thread
  (re-post of the 10-02 published question), the GnuTLS HPACK thread
  (10-01 published), the OTEL env-var thread (10-03 published), a
  high-frequency socket-layer benchmark request (too thin to publish), and
  an OTel GenAI semantic-conventions PR request (out of scope). All covered
  in the page's "Community discussion today".

## Verification against public primary sources

- `bpf/lib/policy.h` @ v1.20.2: datapath precedence ladder — specific entry
  at `MAX_PRECEDENCE` short-circuits; otherwise higher precedence wins; at
  equal precedence the longer `lpm_prefix_length` entry wins; tie selects the
  specific-identity entry. Per-entry BYTES/PACKETS counters via policy
  accounting.
- `pkg/policy/mapstate.go` @ v1.20.2: inverted comparison on the allow path
  (`idKey.PrefixLength() > aggKey.PrefixLength()` returns the aggregate
  entry) — the opposite of the datapath; deny path unconditionally returns
  the specific entry at equal precedence.
- `main`: corrected by two commits on 2026-09-29 (issue reporter): the
  allow-path comparison flip plus a deny-path follow-up that routes
  same-precedence denies through the prefix comparison. Confirmed present
  in `v1.21.0-pre.3`; absent from stable v1.20.x. PR #49062 (AI-generated,
  declined, closed unmerged) independently flagged the deny-path variant in
  review.

## Privacy + content

- No names/handles/Slack-Discord URLs/exact timestamps/IPs/credentials in
  either page; anonymized summary only. Public primary-source links in
  `## References` only.
- Content order per the standard: direct answer, mechanism,
  verification/debugging path, limitation, `## References`, community
  discussion. Mobile-clean rendering via short inline code tokens; the Go
  comparison sits in a fenced block (internal scroll, no horizontal page
  overflow at 390px).

## Coverage disclosure

Two opt-in archive channels covered (11 messages in
`snapshot-2026-10-04.txt`). Visible-browser-only sources (Discord,
eunomia-bpf and sched-ext communities, bpf mailing list, r/eBPF) were not
reviewed this run (no visible-browser session) — marked uncovered-not-quiet
on the page.

## Artifacts

- EN: `docs/ebpf-qa/2026-10-04-cilium-hubble-flow-attributed-to-wrong-policy-rule-on-overlap.md`
- ZH: `docs/ebpf-qa/2026-10-04-cilium-hubble-flow-attributed-to-wrong-policy-rule-on-overlap.zh.md`
- Index links: first items of `docs/ebpf-qa/index.md` (`Latest Answers`) and
  `docs/ebpf-qa/index.zh.md` (`最新回答`).
- Published QA commit on `origin/main`: `bc19e1c92`
  (`docs(ebpf-qa): cilium-hubble-flow-attributed-to-wrong-policy-rule-on-overlap (2026-10-04)`).
- This run-log commits separately as the 10-04 same-day artifact.
- Receipt: `/workspaces/.agent-state/eunomia-qa/receipt-2026-10-04.json`
  (written by the validator only).
