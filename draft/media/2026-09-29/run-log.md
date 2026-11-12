# 2026-09-29 内容巡检运行日志

运行日按 America/Los_Angeles 自然日计算：本日正常额度为 LA 2026-09-29。提交时刻 2026-09-30 08:51 +08 = 17:51 PDT，落在 LA 2026-09-29；本日 LA 前无同日发布，使用正常额度，未核销补发缺口（缺口保持 2 条：2026-09-03、2026-09-16）。

## 队列任务：掘金 33-funclatency（LA 09-29 正常额度）

- 任务行：`draft/plan/publishing-queue.zh.md` 第 96 行（`排队`，现翻为 `[x]`）。第 90/92/94 行知乎同源文仍 `阻塞`（无 `z_c0` 会话，`/creator` 重定向到 `/signin`），本次未涉及；第 97 行（32-wallclock-profiler）未启动。
- 源文：`docs/tutorials/33-funclatency/README.zh.md`（无「原文地址」页脚，仅 1/23/29/35 有）；发布稿 `draft/media/2026-09-29/33-funclatency/juejin-body.md`（移除源 H1，其余正文逐字保留；6136 字符、9254 字节、5 个 H2、3 个 H3、0 个 H4、6 个围栏 [3 个代码块：1 c、2 console]、0 张图片、0 个表格、5 条唯一外链 [github bpf-developer-tutorial 仓库根 ×1、github src/33-funclatency ×1、github bpftime 仓库 ×1、eunomia.dev/tutorials/ ×1、github iovisor/bcc funclatency.c ×1]；无相对链接）。
- 草稿未预置：新标签打开 `https://juejin.cn/editor/drafts/new`（新编辑器会话、无预置草稿）；编辑器自动暂存草稿 7690839722769514505 由本次提交消费、草稿箱保持 4 条；标题经原生 `HTMLInputElement` setter + 冒泡 `input` 事件写入并回读，正文以 base64 分块注入 CodeMirror，写后回读 6136/6136 字符。
- 分类 `后端`（CDP 鼠标点击选中）；提交标签 后端、Linux、性能优化（原生 setter + 选项点击提交，`byte-select__input` 提交顺序 后端 → Linux → 性能优化；提交新标签时先前已提交标签可能掉落，Vue 提交竞态后回读 chip 确认，缺失补回；最终 DOM 顺序 后端、Linux、性能优化）。

## 提交

- 提交时刻 2026-09-30 08:51 +08 = 17:51 PDT；弹窗「确定并发布」用页内合成指针事件序列（pointerover/enter/mouseover/enter → pointerdown/mousedown(+focus) → pointerup/mouseup → click）一次即成，返回 `https://juejin.cn/published` 且 `document.title === '发布成功'`。
- 新文 id `7690830871915216922`：暂存 <https://juejin.cn/spost/7690830871915216922>，正式 <https://juejin.cn/post/7690830871915216922>。提交前创作者中心 全部 (66) / 已发布 (66) / 审核中 (0) / 未通过 (0) [INFERENCE：由 34-syscall 于 09-28 09:52 +08 翻为 已发布 (66) 后至今无同日发布推得]；提交后 全部 (67) / 已发布 (67) / 审核中 (0) / 未通过 (0)。

## 提交结果与判定

- 直接公开，无 审核中 窗口：提交后创作者中心立即为 已发布 (67) / 审核中 (0)，`/spost/` 与 `/post/` 同时可公开访问，无需任何轮询。
- 提交后 curl 探测不可作为发布信号（掘金 SPA 壳对任意 `/post/<id>`、`/spost/<id>` 路径在审核期即返回 200，沿用既有结论）；判定以登录浏览器正文渲染 + 创作者中心 审核中 计数为准。
- 记 `confirmed`（url + staged_url 均记录）。

## 公开页 QA（/post/7690830871915216922，登录浏览器）

- 标题逐字「使用 eBPF 测量函数延迟」（title_occurrences==1）；正文单份。
- 5 个 H2 / 3 个 H3 / 0 个 H4（H2：什么是 eBPF？/为什么函数延迟很重要？/用于函数延迟的 eBPF 内核代码/运行函数延迟工具/结论；H3：代码解释/用户空间函数延迟/内核空间函数延迟）；3 个代码块（`pre code` 语言类统计：1 c、2 console）；0 张正文图片；0 个表格。
- 5 条唯一外链目标全部被掘金改写为 `link.juejin.cn?target=…`（github bpf-developer-tutorial 仓库根 ×1、github src/33-funclatency ×1、github bpftime 仓库 ×1、eunomia.dev/tutorials/ ×1、github iovisor/bcc funclatency.c ×1），正文内 0 相对链接。
- 无 `审核中` / `文章有更新` / `已被删除` 标记；评论 0（`暂无评论数据` 空态）。早期计数（08:51 +08 检查点）：0 展现 / 1 阅读 / 0 点赞 / 0 评论 / 0 收藏。

## 台账更新

- `platforms/juejin.json`：新增 `juejin-7690830871915216922`（`confirmed`，置顶），`last_checked` → 2026-09-29；notes 记录直接公开无审核间隔、08:51 +08 检查点 已发布 (67) / 审核中 (0)、curl 不可信结论、自动暂存草稿消费与 QA 摘要（55 条 confirmed）。
- `sources.json`：`last_checked` → 2026-09-29（映射由检查器从平台条目的 `source_path` 派生，无需新增 source 键）。
- `published.md`：`Last checked:` → 2026-09-29；`## Juejin` 表格首行新增该条目；补记 2026-09-29 叙述行。
- `not-published.md`：`Last checked:` → 2026-09-29；掘金未映射 55 → 54（54/108）、滚动队列 25 → 24（24 Zhihu and 24 Juejin tasks）；掘金状态行置顶 33-funclatency（沿列 34-syscall、35-user-ringbuf、37-uprobe-rust、38-btf-uprobe、41-xdp-tcpdump 不变）；知乎计数不变。
- `draft/plan/publishing-queue.zh.md`：更新时间 → 2026-09-29；补发缺口段落补 09-29 巡检说明（正常额度、未核销缺口、缺口保持 2 条）；Ledger 基线按检查器更新（掘金 54，映射 54/108）；剩余队列掘金 25 → 24、总计 49 → 48；第 96 行翻 `[x]` 并附正式地址、QA 摘要与 `confirmed`（54/108）；第 90/92/94 行（知乎）保持 `阻塞`。
- `community-feedback.md`：`### 2026-09-29` 新检查点（置于 `### 2026-09-28` 之前）：发布记录、直接公开无审核间隔、早期计数 0 展现 / 1 阅读、34-syscall 4 / 6、35-user-ringbuf 6 / 10、37-uprobe-rust 23 / 17、38-btf-uprobe 11 / 15、41-xdp-tcpdump 4463 / 49 / 1 收藏、全部可见行仍 0 评论、下一检查点在 32-wallclock-profiler 额度后。
- 发布稿 `juejin.md`：状态改「已发布」，记录正式/暂存地址、提交时刻与 QA 摘要。
- 验证器：`check_media_ledger.py` 通过（Juejin 54/108 映射、54 未发布、55 confirmed；exit 0）。

## 编辑器经验

- 发布弹窗「确定并发布」面板在普通 CDP 鼠标点击下可能保持 `display:none`（面板零矩形、按钮 0×0）：首次点击未开面板，需对工具栏发布按钮（`button` class `xitu-btn`，右上 ≈ x=896 y=32）重新派发含 hover 的完整指针序列（pointerover/enter/mouseover/enter → pointerdown/mousedown(+focus) → pointerup/mouseup → click）→ 面板展开至 560×801、`确定并发布` 落至 90×30 @(897, 834)；随后对 `确定并发布` 以实时坐标重放页内合成指针序列即得 `发布成功`。已按坐标实时读取重开面板，未硬编码坐标。
- chip 激活态首次读回与 Vue 提交存在竞态（首次读回需再读一次确认）；提交新标签时先前已提交标签可能掉落（观察到提交新标签后 `Linux` 一度消失，补回）；已回读确认 后端、Linux、性能优化 落定，最终 DOM 顺序 后端、Linux、性能优化。
- 本次无审核间隔：判定直接取创作者中心 已发布 (67) / 审核中 (0)。SPA 壳 200 结论沿用：任意 `/post/<id>`、`/spost/<id>` 在审核期即返回 200，curl 状态码不作为发布信号，判定以登录浏览器 + 创作者中心 审核中 计数为准。

## 2026-09-29 eBPF Q&A run-report (eunomia-community-radar)

- Question: "Why does BPF hash map key iteration return keys in an order that looks random instead of insertion order?" (EN + ZH).
- Slug: `2026-09-29-bpf-hash-map-key-iteration-order-looks-random`.
- Selection: no retained candidate existed at run start and no 09-29 pair existed on `origin/main` (4 prior pairs 09-24/09-25/09-26/09-28 were already published and indexed at 43 EN / 43 ZH index links), so the run's duty fell to authoring a fresh pair under the 2026-09-29 run date. The Step 0 snapshot step found no 2026-09-29 snapshot file in the private workdir (`snapshot-2026-09-29.*` absent; only 09-12/09-24/09-25 snapshots on disk) and the browser-only allowlist communities had no visible-browser session, so 0 watchlist-opted-in archives were reachable; the answer was grounded in public primary sources instead, disclosed honestly in the page and here. Topic verified absent from the de-dup list.
- Source basis: torvalds/linux master @ 2026-09-29 — `kernel/bpf/hashtab.c` (`htab_map_get_next_key` bucket walk with `NULL` key → first nonempty bucket, `jhash2`/`jhash` via `htab_map_hash`, `htab->n_buckets = roundup_pow_of_two(max_entries)`, `htab->hashrnd = get_random_u32()` unless `BPF_F_ZERO_SEED`, `hlist_nulls_add_head_rcu` chain-head insertion), `include/linux/jhash.h` (jhash/jhash2 seed mixing), `include/uapi/linux/bpf.h` (`BPF_F_ZERO_SEED` test-only flag), `include/linux/bpf_types.h` (binding table: `BPF_MAP_TYPE_HASH`/`PERCPU_HASH`/`LRU_HASH`/`LRU_PERCPU_HASH`/`HASH_OF_MAPS` → the five `htab*` ops blocks sharing `htab_map_get_next_key`; only `BPF_MAP_TYPE_RHASH` → `rhtab_map_ops`/`rhashtable_next_key`), libbpf `src/libbpf.h` (`bpf_map__get_next_key`: `NULL` key = first key, `-ENOENT` on exhaustion).
- Anonymization: no 09-29 community messages were reachable, so no personal or private material was ingested. No names, handles, employers, workspace/channel/message URLs, timestamps, sequence numbers, private logs, hostnames, IPs, internal repo names, credentials, or topology are reproduced. Public primary-source links appear only in `## References`.
- Content gate: `npm --prefix app run test:content` 82/82 pass, 0 fail (local pre-flight; the Pages pipeline is the authoritative build).
- Commit A: `3cbb085dee1487f42baa6008016b560bafadf2a5` `docs(ebpf-qa): bpf-hash-map-key-iteration-order-looks-random (2026-09-29)` on `main` (4 paths: EN/ZH pair + both indexes; parent `9955d1901`, pushed to `origin/main`), Pages run `36791564287` completed `success`.
- Validator re-verify: receipt `/workspaces/.agent-state/eunomia-qa/receipt-2026-09-29.json`, `status=published` (`branch=ok`, `candidate_paths=ok`, `index_links=ok`, `privacy=ok`, `remote_contains_commit=ok`, `public=ok`).
- Live QA: H1 verbatim + body anchors (`BPF_F_ZERO_SEED`, `get_random_u32`, `htab_map_get_next_key`, `jhash`, `hlist_nulls_add_head_rcu`, `rhashtable_next_key`) verified on both routes with a fresh `?cb=` bust and a browser UA; both index hrefs live on `/ebpf-qa/` and `/zh/ebpf-qa/`.
- Run note: this run completes the 09-29 publication. The coverage gap (no 09-29 snapshot; browser-only communities unavailable) is disclosed honestly in the page's "Community discussion today" / "当日社区讨论" section and here; the answer is grounded in public primary sources, and nothing was invented to fill the gap.
- Public URLs:
  - EN: https://eunomia.dev/ebpf-qa/2026-09-29-bpf-hash-map-key-iteration-order-looks-random/
  - ZH: https://eunomia.dev/zh/ebpf-qa/2026-09-29-bpf-hash-map-key-iteration-order-looks-random/
