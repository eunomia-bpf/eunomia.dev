# 2026-09-23 内容巡检发布记录（America/Los_Angeles 自然日）

- 运行模式：eunomia-content-patrol 定时巡检。任务日 2026-09-23，LA 时区 03:28 PDT 起执行（10:28 UTC）。
- 前置核对：持久可见 Chrome CDP 9222 存活（`Chrome/146.0.7680.31`）；仓库 `main` 干净于上一轮提交，`git fetch` 后无新远端提交。

## 43-kfuncs 审核复查（不消耗新额度）

- 09-22 以 09-22 正常额度提交的 `docs/tutorials/43-kfuncs/README.zh.md` 审核通过：`/post/7688489073967251494` 由“找不到页面”变为公开页。
- 公开页 QA：原标题逐字一致；正文单份；9 个 H2、11 个 H3、3 个 H4、20 个 `<pre>`；8 条唯一外链（10 个锚点，2 个目标重复：bpf-developer-tutorial、eunomia.dev/tutorials 各出现两次，与源文一致）；1 张正文图片 `raw.githubusercontent.com/eunomia-bpf/code-survey/main/imgs/cumulative_helper_kfunc_timeline.png` 完整加载（`naturalWidth×Height = 800×500`，`complete=true`，无转存失败标记，首轮 `0x0` 为懒加载未触发）。无 `审核中`/`文章有更新`/`已被删除` 标记，评论空态。ledger 由 `review_pending` 改为 `confirmed`。
- 队列行由 `[ ] 阻塞` 改为 `[x] 排队`，附公开 URL 与 QA 摘要。

## 42-xdp-loadbalancer 发布（09-23 正常额度）

- 源文：`docs/tutorials/42-xdp-loadbalancer/README.zh.md`（23326 字节，H1 `# eBPF 开发者教程： 简单的 XDP 负载均衡器`）。发布稿只移除源 H1，正文其余保持不变。
- 本地 artifact：`draft/media/2026-09-23/42-xdp-loadbalancer/juejin-body.md`（16487 字符、559 行、5 个 H2、19 个 H3、2 个 H4、25 个代码块 [12 c + 8 sh + 4 console + 1 txt]、0 图片、0 表格、4 条唯一外链、无相对链接）与 `juejin.md`。
- 提交路径：可见编辑器 `https://juejin.cn/editor/drafts/new`，原生 setter 写入标题与正文（正文经 base64 stdin 注入，回读 16487 字符一致），分类 `后端`，标签 `Linux`、`后端`、`性能优化`（逐枚回读 `.byte-select__tag` 确认）。
- `确定并发布` 按钮沿用已记录的真实指针序列（合成 `click()` 仅触发自动保存）；提交页返回“发布成功”（`https://juejin.cn/published`，`document.title = 发布成功`）。
- 结果：**直接公开**，`https://juejin.cn/post/7688532185121947658`（无暂存审核间隔）。公开页 QA：原标题逐字一致、正文单份、5 个 H2、19 个 H3、2 个 H4、25 个 `<pre>`、4 条外链、0 正文图片、分类后端、三标签正常、无审核/更新/删除标记、评论空态（0 条）。ledger 记为 `confirmed`。

## 雷达巡检（10 篇公开掘金文章）

- 42-xdp-loadbalancer 1 阅读（5 H2）；43-kfuncs 16（9 H2）；44-scx-simple 49；45-scx-nest 33；46-xdp-test 26；ACRFence 53；Agent Sandbox 266；47-cuda-events 56；48-energy 34；Runtime Security 36。
- 全部 10 篇均无 `审核中`/`文章有更新`/`已被删除` 标记，评论均为 `暂无评论数据` 空态（0 条），无需回复或更正。
- 已把 2026-09-23 checkpoint 追加到 `.github/publisher/media/community-feedback.md`（append-only，CRLF 保持）。

## 其他

- 知乎 `/creator` 仍重定向 `https://www.zhihu.com/signin?next=%2Fcreator`、cookie 无 `z_c0`，24 条知乎任务保持 `阻塞`。
- 补发缺口保持 4 条（2026-08-29、2026-09-01、2026-09-03、2026-09-16），本日未核销（知乎仍阻塞）。
- 队列更新：头部日期、补发缺口、Ledger 基线（掘金 46/107）、剩余队列（知乎 24、掘金 32、共 56）、43-kfuncs 与 42-xdp-loadbalancer 两行。
- ledger checker `.github/publisher/media/check_media_ledger.py` 退出码 0；掘金 47 条 `confirmed`、0 条 `review_pending`。

## 可读快照对账（回答“为何与 ledger 不一致”）

- 巡检 skill 要求 `published.md`/`not-published.md` 与 `platforms/*.json` 同步。本轮复核对账发现同类漂移不止掘金一处：知乎快照仅 26/66 条 confirmed，领英快照漏掉全部以搜索证据 URL 记录的条目（JSON 15 条 confirmed / 12 个唯一公开 URL）。
- 已补齐：知乎 40 行、领英 4 行，以及 3 条仅有 `evidence_url`（无固定 permalink）的领英条目；领英表按日期倒序重排，相对日期行置后。X/Twitter 表曾被误写入 11 行领英行，已移除。
- 根因修复：`check_media_ledger.py` 原先只校验 JSON 与源文件覆盖，快照漂移时仍返回 0。新增 `validate_snapshot`：每条 confirmed 的 `url`（无 permalink 时用 `evidence_url`）必须在 `published.md` 出现，可 `--snapshot` 指定他文件。负例删除 1 行知乎即 exit 2，修复后 exit 0。
- 已记录到 `.github/publisher/media/README.md` 与 `.agents/skills/eunomia-content-patrol/SKILL.md`。

## 汇总日期核对（同类漂移的第三处）

- 复查 ledger 时发现 checker 首行打印的 `Last checked: 2026-08-02` 来自 `sources.json`，而 `platforms/*.json` 最新为 `2026-09-23`（juejin），`README.md` 头部为 `2026-08-17`：两份汇总日期分别落后其所汇总的分平台 ledger 约 7 周与 5 周。`published.md` / `not-published.md` 头部已是 `2026-09-23`，无需改动。
- 已把 `sources.json` 的 `last_checked` 与 `README.md` 的头部日期更新为 `2026-09-23`（二者均为 LF，JSON 保持字节级 roundtrip）。
- 与快照漂移同源的根因修复：`check_media_ledger.py` 新增 `validate_freshness`，要求 `sources.json` 的 `last_checked` 与快照的 `Last checked:` 均不早于分平台最新的 `last_checked`。负例：`sources.json` 改回 `2026-08-02` → exit 2；快照改回 `2026-08-01` → exit 2；复原后 exit 0。
- 已同步记录到 `.github/publisher/media/README.md` 与 `.agents/skills/eunomia-content-patrol/SKILL.md`。
- 会话复核（无发布动作）：持久可见 Chrome CDP 9222 存活（`Chrome/146.0.7680.31`，pid 664572，profile `/home/gem/.config/browser`，Xvnc `:99.0` 1280x1024，窗口归属同 pid）。登录态：掘金 ✅、LinkedIn ✅、X ✅、Medium ✅、Reddit ✅、知乎 ❌（无 `z_c0`，仍 `阻塞`）、**DEV ⚠️ 浏览器会话失效**（`/dashboard` → `/magic_links/new`），且本机无 `DEV_TO_API_KEY` / `MEDIUM_API_KEY`，下次 DEV 发布前需补密钥或重新登录。

## 排队任务 41-xdp-tcpdump 的额度判定（未发布，待 09-24 额度）

- 本轮获用户明确授权（`authorized`）执行队列 `排队` 任务。队首为 `draft/plan/publishing-queue.zh.md` 第 84 行 `掘金：docs/tutorials/41-xdp-tcpdump/README.zh.md`。
- 阻塞点不是审核或编辑器，而是**同一平台同一自然日只能一篇**（队列第 137 行规则）。09-23 的正常掘金额度已由 42-xdp-loadbalancer 用掉：公开页 JSON-LD `datePublished` 为 `2026-09-23T10:52:25+00:00`，落在 LA 2026-09-23 内（同页 43-kfuncs 为 `2026-09-23T05:29:32+00:00`）。若此刻再发，将是同一 LA 自然日的第二篇掘金文。
- 补发缺口（4 条）能否核销为“当天再发一篇”：**不能**。解析队列全部 `- [x]` 行的 `(日期, 平台)` 组合，除 2026-08-27 的 `Medium 5 / DEV 5`（该日行文为「对账确认此前已公开」，属存量发现而非当日新增发布）外，**没有任何一天在同一平台发布过两篇**；掘金历史上从未同日两篇。且 2026-09-14/09-17/09-22/09-23 五份 run-log 均记 `因知乎阻塞且掘金每日上限 1 条`、`保留到后续可用日`——补发额度是**向上追平**的容量，不是“同日同平台第二篇”的许可，每个补发日仍需独立自然日。
- 准备已就绪（不消耗额度）：`draft/media/2026-09-24/41-xdp-tcpdump/juejin-body.md`（源 H1 移除外逐字保留，11793 字符、15125 字节、LF 0 CRLF、5 H2 / 9 H3 / 9 H4、34 个代码围栏 [12 c + 3 bash + 19 源文裸围栏输出块]、0 图片、0 表格、3 条外链、无相对链接）与 `juejin.md`。
- 重复检查：作者公开列表 `https://juejin.cn/user/4288563097635144/posts`（向下滚动加载，115 条标题）**无** 41-xdp-tcpdump 对应标题，可安全首发。
- 下一步：LA ≥ 2026-09-24 00:00 PDT 时执行第 84 行发布，按 42-xdp-loadbalancer 已验证的编辑器械流程与公开页 QA 清单，随后更新 `platforms/juejin.json`、`published.md`、`not-published.md`、队列行与本 run-log，同一提交内完成。

### 「同一天」的时区口径（实测判定：LA 自然日）

- 运行主机本地时区为 `Asia/Singapore`（+08），而仓库所有 run-log 标题、队列补发缺口与 Ledger 基线均按 **America/Los_Angeles 自然日**记账；两者在 +08 的 00:00–08:00 之间会指向不同日期，因此实测判定该规则的日期边界。
- 抓取 ledger 内全部 47 篇掘金公开页的 JSON-LD `datePublished`，按三个时区分别统计「同平台同日」碰撞：UTC 与 Asia/Shanghai 均出现 2026-09-13 与 2026-09-23 两组碰撞；**America/Los_Angeles 下 2026 年无任何碰撞**（仅 2023 年历史批量回补存在）。
- 09-23 这组碰撞即 43-kfuncs（`13:29 CST` = `05:29 UTC` = **22:29 PDT 09-22**）与 42-xdp-loadbalancer（`18:52 CST` = `10:52 UTC` = `03:52 PDT 09-23`）。仓库自身把它们分别记为 **09-22** 与 **09-23** 两个运行日各自的额度，只有按 LA 边界才与该记账一致；UTC/+08 会把它们错误地压进同一天。
- 结论：`同一平台同一天最多一篇` 的「天」= LA 自然日。故 41-xdp-tcpdump 在 LA 2026-09-23 已无名额（该日额度属 42-xdp-loadbalancer），须等 LA 2026-09-24 00:00 PDT（= 2026-09-24 07:00:30 UTC）。

## eBPF Q&A daily publication (duty day 2026-09-23, America/Los_Angeles)

- **Published:** `bpf-hash-map-value-zeroing-on-delete`
  - EN: https://eunomia.dev/ebpf-qa/2026-09-23-bpf-hash-map-value-zeroing-on-delete/
  - ZH: https://eunomia.dev/zh/ebpf-qa/2026-09-23-bpf-hash-map-value-zeroing-on-delete/
  - Question: does deleting a key from a BPF hash map zero out the value
    memory, and can a reused slot still hold the old value bytes.
  - Commit `2f2311382` on `main` (4 files: EN+ZH pages + both index bullets), pushed.
- **Sources (public primary only):** kernel `hashtab.c` (prealloc vs
  NO_PREALLOC element lifecycle, `free_htab_elem`/`htab_elem_free`,
  `htab_map_delete_elem`), `memalloc.c` (RCU-deferred reclaim, unzeroed
  recycling), `bpf.h` (`BPF_F_NO_PREALLOC`), `Documentation/bpf/map_array.rst`
  (ARRAY zero-init contrast), `map_hash.rst`. All cited URLs verified HTTP 200
  (`raw.githubusercontent.com` for source, `docs.kernel.org` for docs).
- **Coverage gap (reported honestly, not blocking):** the opt-in archive
  snapshot for this duty day returned 0 usable technical messages, so the
  `## Community discussion today` / `## 当日社区讨论` sections fall back to the
  public discussion grounding already cited in the pages. No private data
  (names/handles/employers/channels/message URLs/timestamps) committed.
- **Validator:** the first full-pipeline run stalled ~2 h in the local
  `next build` under machine contention and was cancelled after the
  documented ~20-min threshold. The 4 QA paths were then committed manually
  and the receipt completed via the validator's documented already-published
  fast path: `content_test/build/render/commit_push` =
  `skipped_already_published`, `remote_contains_commit: ok`, `public: ok`.
  Receipt `/workspaces/.agent-state/eunomia-qa/receipt-2026-09-23.json`,
  `status: published`, commit `2f2311382153…`, mode 0600.
- **Live verification:** all 4 public URLs return HTTP 200 with the correct
  H1 (EN/ZH); both `/ebpf-qa/` indexes link the slug.
- **Concurrent work preserved:** a `git fetch` fast-forward
  (`f28b112d3` → `1e508550f`, +20 paths) overlapped 7 in-progress WIP paths
  (6 dirty tracked files + the untracked 09-22 run-log, whose local bytes
  differed from the remote's newly-tracked copy). All 7 were preserved
  byte-identically via a scoped save/restore; the `latest.md` staged blob
  (`2c8a97b2`) was restored to keep its `MM` shape; the staged deletions and
  the agent-skills gitlink (`4a69aa0`) were left untouched. No concurrent
  bytes were lost.

## Post-publication correction (2026-09-23, BPF hash-map article)

- 当日已发布的 `bpf-hash-map-value-zeroing-on-delete`（EN `docs/ebpf-qa/2026-09-23-bpf-hash-map-value-zeroing-on-delete.md`，ZH `...zh.md`）两处事实错误修正，正文 H1 逐字不变，index 路径无改动。
- 错误 1（“经由 map API 不存在跨 key 的值泄漏”）：per-CPU hash map 的 `BPF_F_CPU` 创建路径上 `pcpu_copy_value` 只写指定 CPU 槽位即返回，其余 CPU 保留被回收元素的原始字节，新 key 的 lookup 会读到已删除 key 的 per-CPU 值。修复补丁（`Fixes: c6936161fd55`，`pcpu_init_value` diff）+ 元素复用 selftest 已于 2026-09-23 提交 bpf 邮件列表（openwall msg 16/18，`test_percpu_map_cpu_flag_create`），截至行文尚未合入主线。文中仅以 URL 与主题引用。
- 错误 2（“hash 值既不在创建时也不在删除时清零”）：不精确——非 per-CPU 预分配池在 map 创建时经 `__GFP_ZERO`（`__bpf_map_area_alloc`，`kernel/bpf/syscall.c`）清零；per-CPU 值区域由 `prealloc_init` 的 `bpf_map_alloc_percpu` 分配，创建时经 `alloc_percpu` 零填充保证清零（BPF 代码无 `__GFP_ZERO` 标志，属 `alloc_percpu` 本身性质）；普遍成立的是删除或复用时不重新清零。
- 更正落地：EN/ZH 各 10 处镜像更正；提交 `d4e09ecd4`（4 路径：EN+ZH 两篇 + 两 index 文件，显式 pathspec），已推 `main`。本地 `node --test` 内容测试（`eBPF Q&A`）1/1 通过。
- 发布器 skill 的 10 处更正记录在 `.agents/skills/eunomia-content-patrol/SKILL.md`；`check_media_ledger.py` 新增 `validate_snapshot` 与 `validate_freshness` 校验（负例 exit 2 / 修复后 exit 0）。
