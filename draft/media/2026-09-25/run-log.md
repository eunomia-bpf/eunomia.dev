## 2026-09-25 eBPF Q&A run-report (eunomia-community-radar)

- Question: "Why does a full BPF hash map reject new keys with E2BIG, while updating an existing key still succeed?" (EN + ZH).
- Slug: `2026-09-25-bpf-hash-map-full-update-vs-insert`.
- Selection: no retained candidate and no new community material this run — the 7-day rolling snapshot across both watchlist-opted Slack archives returned zero messages (`snapshot=ok bytes=0 messages=0`; the empty-archive coverage gap is disclosed on both pages), and Discord / bpf@vger / r/eBPF remain visible-browser-only with no browser session available. Fell back to a genuine recurring practitioner boundary of the monitored community ("my map rejects new keys at capacity, yet updating the same key works" / "a key disappeared from my LRU map without a delete"); topic verified absent from the de-dup list (09-23's zeroing-on-delete page covers unlink semantics, not the capacity boundary).
- Source basis: torvalds/linux master @ 2026-09-25 — `kernel/bpf/hashtab.c` (`alloc_htab_elem` E2BIG at freelist exhaustion / `is_map_full` for new keys, the "when map is full and update() is replacing old element" comment, the `extra_elems` swap, `BPF_F_LOCK` in-place `copy_map_value_locked` shortcut), `kernel/bpf/bpf_lru_list.c` (`bpf_common_lru_pop_free` local → global-fetch-with-shrink → steal → ENOMEM), `include/uapi/linux/bpf.h` (the `E2BIG` contract text, `struct bpf_map_info`), `Documentation/bpf/map_hash.rst`, `kernel/bpf/syscall.c`.
- Content gate: `npm --prefix app run test:content` 82/82 pass, 0 fail.
- Commit A: `286ae01beb74` `docs(ebpf-qa): bpf-hash-map-full-update-vs-insert (2026-09-25)` on `main` (4 paths: EN/ZH pair + both indexes), pushed to origin/main; Pages run 36202826984 (`deploy-static-app`) completed `success`.
- Validator re-verify: receipt `/workspaces/.agent-state/eunomia-qa/receipt-2026-09-25.json`, `status=published` (`branch=ok`, `candidate_paths=ok`, `index_links=ok`, `privacy=ok`, `remote_contains_commit=ok`, `public=ok`; content-test/build/render/commit-push checks `skipped_already_published`).
- Live QA: H1 verbatim + body anchors verified on `https://eunomia.dev/ebpf-qa/2026-09-25-bpf-hash-map-full-update-vs-insert/` and `/zh/…` (fresh `?cb=` bust, browser UA); both index hrefs live on `/ebpf-qa/` and `/zh/ebpf-qa/`; fixed spin-lock and LRU-step wording confirmed live, stale wording absent.
- Public URLs:
  - EN: https://eunomia.dev/ebpf-qa/2026-09-25-bpf-hash-map-full-update-vs-insert/
  - ZH: https://eunomia.dev/zh/ebpf-qa/2026-09-25-bpf-hash-map-full-update-vs-insert/

# 2026-09-25 内容巡检运行日志

运行日按 America/Los_Angeles 自然日计算：本日正常额度为 LA 2026-09-25。提交时刻 2026-09-26 07:27 +08 = 16:27 PDT，落在 LA 2026-09-25；本日 LA 前无同日发布，使用正常额度，未核销补发缺口（缺口保持 2 条：2026-09-03、2026-09-16）。

## 队列任务：掘金 38-btf-uprobe（LA 09-25 正常额度）

- 任务行：`draft/plan/publishing-queue.zh.md` 第 89 行（`排队`，现翻为 `[x]`）。第 88 行知乎同源文仍 `阻塞`（无 `z_c0`），本次未涉及。
- 源文：`docs/tutorials/38-btf-uprobe/README.zh.md`；发布稿 `draft/media/2026-09-25/38-btf-uprobe/juejin-body.md`（移除源 H1，其余正文逐字保留；9780 字符编辑器值、14056 字节文件；5 个 H2、2 个 H3、0 个 H4、18 个代码块（5 c、4 sh、9 console）、0 张图片、0 个表格、4 条唯一外链）。
- 标题：借助 eBPF 和 BTF，让用户态也能一次编译、到处运行（源 H1 逐字）。
- 分类 `后端`；提交标签 `Linux`、`后端`、`性能优化`（实际落定 chip 顺序 性能优化/后端/Linux，集合一致，已按实际记录）。

## 提交

- 草稿未预置：新标签打开 `https://juejin.cn/editor/drafts/new`，标题经原生 setter + `input` 事件写入，正文以 base64 分块（18740 b64 字符、4 块）经 `agent-browser eval --stdin` 注入 CodeMirror，写后回读 9780/9780。
- 提交时刻 2026-09-26 07:27 +08 = 16:27 PDT；弹窗「确定并发布」用真实 CDP 指针序列，一次即成，返回 `https://juejin.cn/published` 且 `document.title === '发布成功'`。
- 新文 id `7689065487933833262`：暂存 <https://juejin.cn/spost/7689065487933833262>，正式 <https://juejin.cn/post/7689065487933833262>。

## 提交结果与判定方法更正

- 提交后 07:29 +08 curl 探测 `/post/` 与 `/spost/` 均返回 HTTP 200，但登录浏览器中 `/post/` 仍渲染「找不到页面」、创作者中心仍 审核中 (1)。结论：掘金 SPA 壳对任意 `/post/<id>` 路径在审核期即返回 200，**curl 状态码不是发布信号**；判定以登录浏览器正文渲染 + 创作者中心 审核中 计数为准。
- 浏览器轮询（每 ~90 秒重开 `essays?status=all` 读计数）：07:47–08:06 均为 已发布 (62) / 审核中 (1)；08:07 +08 起翻为 已发布 (63) / 审核中 (0)（约 40 分钟审核，介于 39-nginx 的 45–60 秒与 45-scx-nest 的约 1 小时之间）。
- 08:12 +08 复查：`/post/7689065487933833262` 渲染公开正文，`/spost/` 跳转至 `/post/`。记 `confirmed`（url + staged_url 均记录）。

## 公开页 QA（/post/7689065487933833262，登录浏览器）

- 标题逐字「借助 eBPF 和 BTF，让用户态也能一次编译、到处运行」；正文单份（`在现代 Linux 系统中` 出现 1 次）。
- 5 个 H2 / 2 个 H3 / 0 个 H4；18 个代码块（`pre code` 语言类统计：5 `language-c`、4 `language-sh`、9 `language-console`）；0 张正文图片（`.markdown-body img` 过滤后 0）；0 个表格。
- 4 条唯一外链目标全部被掘金改写为 `link.juejin.cn?target=…`（eunomia.dev/tutorials/ ×2、bpf-developer-tutorial…/38-btf-uprobe ×2、bpftime ×1、bpf-developer-tutorial 仓库根 ×1），`?target=` 内无裸斜杠，正文内 0 相对链接。
- 无 `审核中` / `文章有更新` / `已被删除` 标记；评论 0（`暂无评论数据` 空态）。早期计数 0 展现 / 2 阅读 / 0 点赞 / 0 收藏。

## 台账更新

- `platforms/juejin.json`：新增 `juejin-7689065487933833262`（`confirmed`，第 51 条），`last_checked` → 2026-09-25；notes 记录审核期、curl 不可信结论与 QA 摘要。
- `sources.json`：`last_checked` → 2026-09-25。
- `published.md`：`Last checked:` → 2026-09-25；`## Juejin` 表格首行新增该条目；补记 2026-09-25 叙述行。
- `not-published.md`：`Last checked:` → 2026-09-25；掘金未映射 58 → 57、已映射 49/107 → 50/107；滚动队列 29 → 28；掘金状态行置顶 38-btf-uprobe（下一位 37-uprobe-rust）。
- `draft/plan/publishing-queue.zh.md`：更新时间 → 2026-09-25；补发缺口段落补 09-25 巡检说明（正常额度、未核销缺口、缺口保持 2 条）；Ledger 基线掘金 58 → 57（映射 50/107）；剩余队列掘金 29 → 28、总计 53 → 52；第 89 行翻 `[x]` 并附正式地址、QA 摘要与 `confirmed`（50/107）；第 88 行（知乎）保持 `阻塞`。
- `community-feedback.md`：`### 2026-09-25` 新检查点（置于 `### 2026-09-24` 之前）：发布记录、审核间隔 07:27→08:07 +08、早期计数、39-nginx 增量（1097 展现 / 25 阅读 / 1 点赞 / 1 收藏）、41-xdp-tcpdump 3985 / 36、40-mysql 746 / 24、全部 12 篇跟踪公开文仍 0 评论、无外部回声、下一检查点在 37-uprobe-rust 额度后。
- 发布稿 `juejin.md`：状态改「已发布」，记录正式地址、审核时间线与 QA 摘要；更正其「提交结果判定」条款（curl 200 不可信，以浏览器 + 创作者中心为准）。
- 验证器：`check_media_ledger.py` 通过（Juejin 50/107 映射、57 未发布、51 confirmed）。

## 编辑器经验（已回写技能）

- 掘金 SPA 壳在审核期对任意 `/post/<id>`、`/spost/<id>` 路径返回 HTTP 200，curl 探测不可作为发布判定；以登录浏览器正文渲染与创作者中心 审核中 计数为准（本次 07:27 提交、07:29 curl 200/200 时仍 审核中、浏览器 404；08:07 清除后 `/post/` 渲染、`/spost/` 跳转）。已回写 `.agents/skills/juejin-publisher/SKILL.md` 与 `.github/publisher/media/juejin-skill.md`。
- 长正文注入：`agent-browser eval` 内联参数带反引号/引号会破坏 shell 解析（`unterminated backquote`）；写入 `/tmp/*.js` 后经 `eval --stdin` 管道，配合 base64 分块（4 块 × ~4.7KB）一次写满 9780 字符。
- 发布弹窗 `.publish-popup` 本次含 6 个 `.byte-select__input`（标签/合集/话题各一对，可见 3 个，width-0 隐藏重复 3 个）；index 0 仍为标签。落定 chip 顺序可与输入顺序不同（本次 性能优化/后端/Linux），台账记录实际集合。
- 审核清除不总落在 45–60 秒窗内：本次约 40 分钟；技能已按「浏览器轮询至 审核中 归零」更新，而非按 curl。
