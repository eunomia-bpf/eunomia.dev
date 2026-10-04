# 2026-10-03 内容巡检运行日志

运行日按 America/Los_Angeles 自然日计算：本日正常额度为 LA 2026-10-03。唯一提交时刻为 2026-10-04 07:41 +08（= 16:41 PDT），落在 LA 2026-10-03。巡检开始时补发缺口为 0 条，26-sudo 使用正常额度，未核销任何缺口；本次结束后仍无补发缺口。知乎任务仍 `阻塞`——巡检开始时在可见浏览器复查 `https://www.zhihu.com/creator` 仍重定向到 `https://www.zhihu.com/signin?next=%2Fcreator`（无 `z_c0` 会话），阻塞条件未变化，故首个可执行的 `排队` 任务为掘金 26-sudo；本次未启动任何知乎任务。

## 队列任务：掘金 26-sudo（LA 10-03 正常额度）

- 任务行：`draft/plan/publishing-queue.zh.md` 第 106 行（`排队`，现翻为 `[x]`）。第 105 行为知乎 26-sudo，`阻塞`，未启动。
- 源文：`docs/tutorials/26-sudo/README.zh.md`；发布稿 `draft/media/2026-10-03/26-sudo/juejin-body.md`（移除源 H1，其余正文逐字保留；8687 字符、12799 字节、7 个 H2、0 个 H3、0 个 H4、3 个代码块 [1 c、2 bash]、0 张图片、0 个表格、6 条唯一外链目标、0 条相对链接）。
- 草稿未预置：`https://juejin.cn/editor/drafts/new`（新编辑器会话、无预置草稿）；编辑器自动暂存草稿 7692058641251221514 由本次提交消费。标题经原生 `HTMLInputElement` setter + 冒泡 `input` 事件写入并回读，正文以 base64 分块注入 CodeMirror，写后回读 8687/8687 字符。
- 分类 `后端`（CDP 鼠标在 chip 中心 move 553 169 / down / up 选中，回读 `.category-list .item.active` 为 `["后端"]`；页内合成指针事件不生效）；标签 Linux、后端、性能优化（原生 setter 写入 + 选项上合成指针序列提交；提交新标签时先清空输入；最终 DOM 顺序 Linux、后端、性能优化）。

## 提交

- 提交时刻 2026-10-04 07:41 +08 = 2026-10-03 16:41 PDT；提交前创作者中心基线 全部 72 / 已发布 72 / 审核中 0 / 未通过 0。弹窗「确定并发布」用含 hover 的合成指针事件序列一次即成，返回 `https://juejin.cn/published` 且 `document.title === '发布成功'`。新文 id `7692143474695012402`：正式 <https://juejin.cn/post/7692143474695012402>（直接公开，无 `/spost/` 暂存间隔）。

## 提交结果与判定

- 07:47 +08 检查点创作者中心已为 全部 73 / 已发布 73 / 审核中 0 / 未通过 0：直接公开、无审核间隔，正式地址自提交起即生效（期间 SPA 壳在 /post/ 上返回 200，curl 不可作为发布信号，判定以登录浏览器创作者中心 审核中 计数为准）。
- 记 `confirmed`（仅 url，无 `staged_url`——本次不存在暂存间隔；沿用 42-xdp-loadbalancer 无 `staged_url` 的先例）。

## 公开页 QA（登录浏览器）

- 26-sudo `/post/7692143474695012402`：标题逐字「eBPF 教程: 文件操纵实现 sudo 权限提升」；单份正文探测（4 个标记各出现 1 次）；7 个 H2 / 0 个 H3 / 0 个 H4；3 个代码块 {1 c、2 bash}；0 张图片；0 个表格；6 条唯一外链目标（6 个 `<a>` 锚点：eunomia.dev/tutorials/、github bpf-developer-tutorial 仓库根、tree main/src/26-sudo、github pathtofile/bad-bpf、man7.org bpf-helpers.7、lwn.net/Articles/695991）全部改写为 `link.juejin.cn?target=`；无 `审核中` / `文章有更新` / `已被删除` / `找不到页面` 标记；评论 0（暂无评论数据 空态）。渲染 innerText 为 8403 字符（稿面 8687，渲染空白归一化所致；结构计数以稿面为准，沿用 27-replace 记法）。早期计数（07:47 +08 检查点）：0 展现 / 1 阅读 / 0 点赞 / 0 评论 / 0 收藏。

## 台账更新

- `platforms/juejin.json`：置顶新增 `juejin-7692143474695012402` 一条 `confirmed`（newest-first；带 `id`、`title`、`url`、`source_path`、`checked_via` 与 notes，无 `staged_url`）；`last_checked` → 2026-10-03。
- `sources.json`：`last_checked` → 2026-10-03（映射由检查器从平台条目的 `source_path` 派生，无需新增 source 键）。
- `published.md`：`Last checked:` → 2026-10-03；`## Juejin` 表格首行置顶 26-sudo（newest-first）；补记 2026-10-03 叙述行。
- `not-published.md`：掘金未映射 49 → 48（59/108 → 60/108）、滚动队列 19 → 18（24 Zhihu and 18 Juejin tasks）；掘金状态行追加 26-sudo 子句。
- `draft/plan/publishing-queue.zh.md`：更新时间 → 2026-10-03；补发缺口段落追加 10-03 巡检句（正常额度 1 条、直接公开无审核间隔）；Ledger 基线按检查器更新（掘金 48，映射 60/108）；剩余队列掘金 19 → 18、总计 43 → 42；第 106 行翻 `[x]` 并附正式地址、QA 摘要与 `confirmed`；知乎第 105 行保持 `阻塞`。
- `community-feedback.md`：`### 2026-10-03` 新检查点（置于 `### 2026-10-02` 之前）：发布记录（pre 全部 72 / post 73/73/0/0、直接公开无审核间隔）、QA 与 07:47 +08 早期计数、跟踪帖自 10-02 检查点起上行。
- 发布稿 `draft/media/2026-10-03/26-sudo/juejin.md`：状态改「已发布」，记录正式地址、提交时刻、无审核间隔与 QA 摘要；早期计数引用 07:47 +08 检查点。
- 验证器：`check_media_ledger.py` 通过（Juejin 60/108 映射、48 未发布、61 confirmed、2 无 source_path；exit 0）。

## 编辑器经验

- 与 10-02 一致：分类 `后端` 需 CDP 鼠标在 chip 中心 move/down/up 一次选中（回读 `.category-list .item.active` 确认；页内合成指针不生效）；标签 chip 对可见 `.byte-select-option` 元素派发含 hover 的合成指针序列一次落定，每个标签前清空输入。
- 本次为直接公开：提交后创作者中心下一次轮询（07:47 +08）即为 已发布 (73) / 审核中 (0)，无 `/spost/` 暂存窗口；沿用 SPA 壳 200 结论，判定以登录浏览器创作者中心 审核中 计数为准。
- 每条开新 `/editor/drafts/new` 会话整稿重注入，标题原生 setter、正文 base64 分块 CodeMirror 注入；编辑器自动暂存草稿 7692058641251221514 由本次提交消费（草稿箱计数不变）。

## 2026-10-03 eBPF Q&A run-report (eunomia-community-radar)

- Question (EN, verbatim H1): "Why does the OBI Kubernetes cache address environment variable get ignored when the Helm chart renders a Config v2 document, and where must the cache address be written instead?"
- Question (ZH, verbatim H1): "为什么在 OBI 的 Config v2 文档下，Helm 图自动注入的 Kubernetes 缓存地址环境变量会被忽略，缓存地址又该写在哪里？"
- Slug: `2026-10-03-obi-k8s-cache-env-var-ignored-in-v2-config`.
- Selection: no retained 10-03 candidate existed at run start and no 10-03 pair existed on `origin/main` (the 10-02 Cilium LoadBalancer pair was the most recent published and indexed pair), so the run's duty fell to authoring a fresh pair under the 2026-10-03 run date. Two opt-in archive sources provided eleven messages, all covered above; the selected thread is a recurring multi-participant exchange whose decisive boundary is the v1-overlay vs v2-single-doc configuration split, resolved by the upstream chart fix (pull request 2440, merged 2026-10-02, shipped in chart 0.14.2) or by hand-writing the v2 field. Distinct from the already-published 08-22 OBI Prometheus-scraping and 09-19 OBI v1→v2 migration pairs. The visible-browser-only sources (Discord, the eunomia-bpf and sched-ext communities, the bpf mailing list, r/eBPF) could not be reviewed in this run (no visible-browser session was available) and are marked uncovered, not quiet — the gap is disclosed on the page and in this report.
- Source basis (public primary sources, all fetched and read this session against raw GitHub): OBI `devdocs/k8s-cache.md` (v1 `attributes.kubernetes.meta_cache_address` / env `OTEL_EBPF_KUBE_META_CACHE_ADDRESS`; "no address → local in-process cache"), OBI v2 `migration.md` (`OTEL_EBPF_*` overlays are not read in v2), OBI v2 `config-v2.md` (field move + rename to `metadata_cache.address` under the Kubernetes enricher block; loader-dependent `${VAR}` substitution), OBI Helm `daemonset.yaml` (injects the env var when `k8sCache.replicas` is set; the rendered v2 configmap carried no field referencing it), and open-telemetry/opentelemetry-helm-charts pull request 2440 "fix(obi): set the k8s cache address in the Config v2 document" (merged 2026-10-02 → chart 0.14.2, renders `metadata_cache.address` into the v2 configmap).
- Anonymization: no personal or private material was ingested. No names, handles, employers, Slack/Discord channel URLs, exact timestamps, private logs, hostnames, IPs, or credentials are reproduced; the community-discussion section is an anonymized paraphrase of the thread. Public primary-source links appear only in `## References`.
- Content gate: `npm --prefix app run test:content` 82/82 pass, 0 fail (local pre-flight before the validator).
- Mobile overflow (run-specific hiccup, content-local fix, no CSS change): the validator's real-Chromium mobile render (390px) flagged a horizontal-overflow regression on this page driven by two long unbreakable inline field paths — the 72-char `extensions.obi.enrich.enrichers.kubernetes.metadata_cache.address` and its 47/54/40-char variants (`enrich.enrichers.kubernetes.metadata_cache.address`, `attributes.kubernetes.meta_cache_address`). The fix is content-local and in-scope (2 files only, both owned by the validator's commit set): the long dotted paths are reworded to keep only the short `metadata_cache.address` leaf / `meta_cache_address` leaf inline and to phrase the parent block in prose ("the `metadata_cache.address` field under the Kubernetes enricher block", "`meta_cache_address` under the `attributes.kubernetes` block"). Fenced `pre` blocks scroll internally via `overflow-x-auto` and do not contribute to document `scrollWidth`, so no global CSS `overflow-wrap` was needed. After the fix, both the EN and ZH routes render `overflow=false` at desktop (1440×900) and mobile (390×844). The validator's scoped commit (4 owned paths) and the live deploy both ship this content-local fix; `app/styles/globals.css` was not touched.
- Validator (Chromium host-lib hiccup, environment fix): the first validator run failed at the render stage because the Playwright Chromium headless shell could not launch — missing host shared libraries (`libnspr4.so` and companions) on the Debian 12 root host. Resolved by `apt-get install -y --no-install-recommends libnspr4 libnss3 libatk1.0-0 libatk-bridge2.0-0 libdbus-1-3 libatspi2.0-0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libxkbcommon0 libasound2 libcairo2 libdrm2 libx11-6 libxcb1 libxext6`. No site code was changed for this; it is a prerequisite for the headless-shell render.
- Commit A: `474f824f42fbb8c82eb902985b112584260c19a2` `docs(ebpf-qa): obi-k8s-cache-env-var-ignored-in-v2-config (2026-10-03)` on `main` (4 paths: EN/ZH pair + both indexes), pushed to `origin/main`. The commit landed cleanly on the first validator attempt after two non-ff rejections: the concurrent Juejin agent advanced `origin/main` (`519ada859 → 4f25fe726 → ff1da2215`) between the validator's scoped commit and its `git push`, so the push was retried onto the fresh tip; the commit content is identical across attempts.
- Validator: receipt `/workspaces/.agent-state/eunomia-qa/receipt-2026-10-03.json`, `status=published` (`branch=ok`, `candidate_paths=ok`, `index_links=ok`, `privacy=ok`, `commit_push`/`remote_contains_commit`/`public=ok`, `content_test`/`build`/`render` recorded on the commit attempt). The final receipt reflects the already-published path: `content_test`/`build`/`render`/`commit_push` = `skipped_already_published`, `remote_contains_commit=ok`, `public=ok`, commit `474f824f42fb`.
- Live QA (fresh `?cb=` busts, browser UA, all 200): EN route H1 verbatim + body anchors `metadata_cache.address`, `OTEL_EBPF_KUBE_META_CACHE_ADDRESS`, `k8sCache.replicas`, "pull request 2440", "chart 0.14.2" all present; ZH route H1 verbatim; the new href is the first `Latest Answers` item on `/ebpf-qa/` and the first `最新回答` item on `/zh/ebpf-qa/`.
- Public URLs:
  - https://eunomia.dev/ebpf-qa/2026-10-03-obi-k8s-cache-env-var-ignored-in-v2-config/
  - https://eunomia.dev/zh/ebpf-qa/2026-10-03-obi-k8s-cache-env-var-ignored-in-v2-config/
  - https://eunomia.dev/ebpf-qa/
  - https://eunomia.dev/zh/ebpf-qa/
