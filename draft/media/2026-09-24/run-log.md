# 2026-09-24 内容巡检运行日志

运行日按 America/Los_Angeles 自然日计算：本日正常额度为 LA 2026-09-24。09-23 的掘金额度已被 42-xdp-loadbalancer 占用，同日不得再发同平台一篇，因此 41-xdp-tcpdump 顺延至 LA 09-24 的 00:00 PDT 之后提交（实际提交时刻 2026-09-24 07:10 UTC / 00:10 PDT，信息流显示北京时间 2026-09-24 15:10）。

## 队列任务：掘金 41-xdp-tcpdump

- 任务行：`draft/plan/publishing-queue.zh.md` 第 84 行，原状态 `排队`。
- 源文：`docs/tutorials/41-xdp-tcpdump/README.zh.md`；发布稿 `draft/media/2026-09-24/41-xdp-tcpdump/juejin-body.md`（移除源 H1，其余正文逐字保留；11793 字符）。
- 标题：eBPF 示例教程：使用 XDP 捕获 TCP 信息（源 H1 逐字，保留全角冒号）。
- 分类 `后端`；标签 `Linux`、`后端`、`性能优化`。
- 提交结果：可见编辑器提交页返回“发布成功”。文章先落在暂存地址 <https://juejin.cn/spost/7688905828694294566>（创作者中心显示为唯一 `审核中` 条目），同会话内审核通过，正式地址 <https://juejin.cn/post/7688905828694294566> 生效，`/spost/` 随后 404。
- 公开页 QA：标题逐字、正文单份、5 个 H2、9 个 H3、9 个 H4、17 个代码块（12 个 `c`、3 个 `bash`、2 个源文裸围栏示例输出块）、0 张正文图片、0 个表格、3 条唯一外链目标、无 `审核中`/`文章有更新`/`已被删除` 标记、评论区为空。作者主页将其列为最新一篇。

## 台账更新

- `platforms/juejin.json`：新增 `juejin-7688905828694294566`（`confirmed`），`last_checked` → 2026-09-24，已发布条目 47 → 48。
- `published.md`：`Last checked:` → 2026-09-24；`## Juejin` 表格首行新增该条目；补记 2026-09-24 叙述行。
- `not-published.md`：`Last checked:` → 2026-09-24；掘金未映射 61 → 60、已映射 46/107 → 47/107；滚动队列 32 → 31；掘金状态行置顶该发布。
- `draft/plan/publishing-queue.zh.md`：更新时间 → 2026-09-24；Ledger 基线掘金 63 → 62、映射 46/107 → 47/107；剩余队列掘金 32 → 31、总计 56 → 55；第 84 行翻为 `[x]` 并附正式地址与 QA 摘要；补发缺口保持 4 条（2026-08-29、2026-09-01、2026-09-03、2026-09-16）。

## 编辑器经验（已回写技能）

- 分类 chip：合成 hover/pointer 事件序列本次仍未生效（`.category-list .item.active` 返回空），必须使用真实 CDP `page.mouse` 移动/按下/抬起；一次即成。
- 标签输入：`.publish-popup .byte-select__input` index 0 需用原生 value setter + 冒泡 `input` 事件写入，再点击文档级可见 `.byte-select-option`；CLI `keyboard type` 会把文本泄漏到标题/正文。
- 提交按钮：`确定并发布` 只需真实 CDP 指针序列（move → move → down → up），本次一次即成，返回 `https://juejin.cn/published` 且 `document.title === '发布成功'`。
- 审核不是稳定态：本次 `/spost/` 暂存仅维持数分钟，同会话内即换为 `/post/`。`/post/` 返回 200 且无 `审核中` 标记即可直接记 `confirmed`，不必按 09-22 的 `/spost/` 惯例先记 `review_pending`。

## 社会雷达（41-xdp-tcpdump 发布后）

- 观察时刻 2026-09-24 01:05 PDT（08:05 UTC）。创作者中心显示 已发布 (60) / 审核中 (0) / 未通过 (0)。
- 新文 <https://juejin.cn/post/7688905828694294566> 公开，标题逐字、无审核/更新/删除标记；早期计数 1886 展现 / 13 阅读 / 0 评论。
- 跟踪中的 10 篇公开掘金文章全部保持 `暂无评论数据` 空评论状态，无需回复或更正。阅读数领先者：44-scx-simple 51、47-cuda-events 59、ACRFence 54。
- 外部回声：精确标题的网页搜索只返回官方源页、GitHub 源码与新发布掘金页，尚无转载或引用。无需行动。
- 下一检查点：40-mysql 发布后复查是否出现首条非空评论。

## 队列任务：掘金 40-mysql（LA 2026-09-24 补发缺口额度）

- 任务行：`draft/plan/publishing-queue.zh.md` 第 86 行（`排队`）。第 85 行知乎同源文仍 `阻塞`（无 `z_c0`）。
- 当日正常额度已由 41-xdp-tcpdump 使用，本次经用户指示按补发缺口额度发布，作为当日同平台第二条；发布后核销最早缺口 2026-08-29，缺口由 4 条减为 3 条（2026-09-01、2026-09-03、2026-09-16）。
- 发布稿 `draft/media/2026-09-24/40-mysql/juejin-body.md`：源文 `docs/tutorials/40-mysql/README.zh.md` 移除源 H1，其余正文逐字保留；3237 字符、6071 字节、5 个 H2、4 个 H3、0 个 H4、3 个代码块（1 bt、1 bash、1 console）、0 张图片、0 个表格、3 条唯一外链。
- 提交：草稿 id `7688990935071096882` 已预置（标题、正文、分类、标签全部服务端持久化），提交时刻 2026-09-24 18:44 +08 = 10:44 UTC = 03:44 PDT；弹窗「确定并发布」用真实 CDP 指针序列，一次即成，返回 `https://juejin.cn/published` 且 `document.title === '发布成功'`。
- 提交结果：`/post/7689030007914545198` 返回 200，`/spost/7689030007914545198` 与草稿 `/spost/7688990935071096882` 均 404 —— 直接公开、无审核等待，记 `confirmed`。创作者中心 已发布 (61) / 审核中 (0) / 未通过 (0)。
- 公开页 QA：标题逐字、正文单份、5 个 H2、4 个 H3、0 个 H4、3 个代码块（`language-bt`/`language-bash`/`language-console`）、0 张正文图片（页面上 5 个 `img` 均为头像与 `xitu_juejin_web` 图标）、0 个表格、3 条唯一外链目标（均被掘金改写为 `link.juejin.cn?target=`）、无 `审核中`/`文章有更新`/`已被删除` 标记、评论 0。
- 已查重：`platforms/juejin.json` 原无 40-mysql 条目，源文此前未在掘金发布过。

## 社会雷达（40-mysql 发布后）

- 观察时刻 2026-09-24 20:47 +08（05:47 PDT）。创作者中心显示 已发布 (61) / 审核中 (0) / 未通过 (0)，较发布前 60 增加 1。
- 新文 <https://juejin.cn/post/7689030007914545198> 公开，标题逐字、无审核/更新/删除标记；早期计数 2 展现 / 1 阅读 / 0 点赞 / 0 评论 / 0 收藏。
- 同日 <https://juejin.cn/post/7688905828694294566>（41-xdp-tcpdump）计数由 01:05 PDT 的 1886 展现 / 13 阅读 增至 3425 展现 / 18 阅读，仍 0 评论。
- 当前列出的 10 篇公开掘金文章全部保持 0 评论，`暂无评论数据` 空评论状态延续，无需回复或更正。阅读数领先者：47-cuda-events 59、44-scx-simple 51（1 收藏）、45-scx-nest 35、48-energy 35。
- 外部回声：尚无转载或引用。下一检查点：39-nginx 发布后复查是否出现首条非空评论。

## 编辑器经验补充（创作者中心读取）

- 创作者中心按 URL 直接进入，勿点 SPA 导航：`https://juejin.cn/creator/content/article/all` 的 `.byte-tab-pane` 为空，且会叠加 `选择你感兴趣的技术方向` 引导弹窗与 `当前操作失败` 提示，页面文本只剩背景框架，看起来像空账号。
- 直接加载 `https://juejin.cn/creator/content/article/essays?status=all`，数秒后计数（`全部 (N)`/`已发布 (N)`/`审核中 (N)`/`未通过 (N)`）与逐篇 `展现 / 阅读 / 点赞 / 评论 / 收藏` 均出现在 `document.body.innerText`。已回写 `.agents/skills/juejin-publisher/SKILL.md` 与 `.github/publisher/media/juejin-skill.md`。

## 更正：已发布 EN+ZH Q&A 的 BPF hash map 值清零事实修复

- 对象：已发布双语 Q&A 对 `docs/ebpf-qa/2026-09-23-bpf-hash-map-value-zeroing-on-delete(.zh).md`（`/ebpf-qa/…/` 与 `/zh/ebpf-qa/…/`）。
- 缺陷一（自相矛盾）：原文在描述了 `BPF_F_CPU` 创建路径的旧值残留后，仍笼统否认存在任何 API 可见的逻辑泄漏。更正：该创建路径本身即 API 可见的跨 key 泄漏——对未指定 CPU 以 `BPF_F_CPU|cpu<<32` 执行 `bpf_map_lookup_elem_flags` / lookup-batch，可读到前任元素在这些 CPU 上的 per-CPU 值，直至 2026-09-23 提交至 bpf 邮件列表的清零补丁合入（截至 2026-09-24 主线抓取尚未合入）；普通 HASH 全值更新与 BPF 程序创建路径（经 `pcpu_init_value` 清零非当前 CPU 槽位）无跨 key API 泄漏。
- 缺陷二（LRU 擦除 + 复零过度声明）：「删除/驱逐前写零是唯一可靠擦除方式」暗示不存在的用户态驱逐前钩子；「删除或复用时无路径复零」过度声明——BPF 程序 per-CPU 创建路径会通过 `pcpu_init_value` 清零其他 CPU 的槽位（用户态 `BPF_F_CPU` 创建路径不会，即该 bug）。现措辞：显式 `delete` 前写零仍是唯一可靠擦除；自动 LRU 驱逐无用户态保证时点，被驱逐值的字节可能存活至后续分配覆写；创建后唯一的其他复零点即 `pcpu_init_value` 的非当前 CPU 清零。
- 引用：清零补丁与自测按 URL + 主题引用（linux-kernel 邮件列表 2026-09-23 第 16、18 号，`[PATCH bpf v2 1/2] bpf: Zero-fill other CPUs when BPF_F_CPU creates a per-cpu hash element` 及其自测）；合入声明限定为「截至 2026-09-24 主线抓取尚未合入」，不使用「已修复」措辞。
- 提交：`b81bcae86`（仅该 EN+ZH 对，18 增 18 删；index 两页无改动，H1 未变）；验证器 re-verify 分支通过（receipt `…/eunomia-qa/receipt-2026-09-23.json`，commit `b81bcae86`，status `published`）；内容测试 82/82 通过；Pages 部署成功（含 `b81bcae86` 的运行成功）；线上 EN 页确认更正锚点（`bpf_map_lookup_elem_flags`、`previous occupant`、`cross-key value leak`、合入限定措辞）与 ZH 页确认对应锚点（「经由 map API 唯一能触发的跨 key 值泄漏」等），双 H1 逐字在位。
## 队列任务：掘金 39-nginx（LA 2026-09-24 补发缺口额度，用户 "public" 指令提前发布）

- 任务行：`draft/plan/publishing-queue.zh.md` 第 87 行（`排队`，现翻为 `[x]`）。
- 发布稿：`draft/media/2026-09-24/39-nginx/juejin-body.md`（原 `draft/media/2026-09-25/` 按 LA 发布日移入本日目录）：源文 `docs/tutorials/39-nginx/README.zh.md` 移除源 H1，其余正文逐字保留；4838 字符、7904 字节、6 个 H2、5 个 H3、0 个 H4、3 个代码块（2 bt、1 console）、0 张图片、0 个表格、4 条唯一外链。
- 排期：本任务原占 LA 2026-09-25 正常额度；09-24 正常额度已被 41-xdp-tcpdump 使用、补发额度已被 40-mysql 使用。经用户 "public" 指令以补发缺口额度提前发布（当日同平台第三条，仅用于核销补发）；发布后核销缺口 2026-09-01，缺口由 3 条减为 2 条（2026-09-03、2026-09-16），LA 2026-09-25 正常额度仍留给 38-btf-uprobe。
- 提交：草稿 id `7689029864910929970` 已预置（标题、正文、分类 `后端`、标签 `Linux`/`后端`/`性能优化` 全部服务端持久化），提交时刻 2026-09-24 21:55 +08 = 13:55 UTC = 06:55 PDT（JSON-LD `datePublished` 2026-09-24T13:55:37+00:00）；弹窗「确定并发布」用真实 CDP 指针序列，单次弹窗提交一次即成，返回「发布成功」。
- 提交结果：审核期 `/post/7689030180350394419` 与 `/spost/7689030180350394419` 均 404；约 45–60 秒后 `/post/` 转 200 而 `/spost/` 404，确认公开、无审核等待，记 `confirmed`。创作者中心 已发布 (62) / 审核中 (0) / 未通过 (0)。
- 公开页 QA：标题逐字、正文单份、6 个 H2、5 个 H3、0 个 H4、3 个代码块（2 bt、1 console）、0 张内容图片、0 个表格、4 条唯一外链目标（均被掘金改写为 `link.juejin.cn?target=`）、无 `审核中`/`文章有更新`/`已被删除` 标记、评论 0。

## 台账更新

- `platforms/juejin.json`：新增 `juejin-7689030180350394419`（`confirmed`，第 50 条），`last_checked` → 2026-09-24。
- `published.md`：`## Juejin` 表格首行新增 39-nginx 条目；补记 2026-09-24 叙述行。
- `not-published.md`：掘金未映射 59 → 58、已映射 48/107 → 49/107；滚动队列 30 → 29（下一位 38-btf-uprobe）；补发缺口 3 → 2；掘金状态行置顶 39-nginx。
- `draft/plan/publishing-queue.zh.md`：补发缺口 3 → 2；剩余队列掘金 30 → 29、总计 54 → 53；第 87 行翻为 `[x]` 并附正式地址、QA 摘要与缺口核销；第 15 行补 39-nginx 段落。
- `community-feedback.md`：`### 2026-09-24` 追加 39-nginx 发布条目与早期计数条目；「10 篇当前列出的公开掘金文章」更新为 11 篇。
- 发布稿 `juejin.md`：状态改「已发布」；注明预置草稿已于 2026-09-24 21:55 +08 提交、勿再提交。

## 社会雷达（39-nginx 发布后）

- 观察时刻 2026-09-25 07:34 +08（16:34 PDT）。创作者中心显示 全部 (62) / 已发布 (62) / 审核中 (0) / 未通过 (0)。
- 新文 <https://juejin.cn/post/7689030180350394419> 公开，标题逐字、无审核/更新/删除标记；早期计数 0 展现 / 7 阅读 / 0 评论 / 0 收藏。
- 同日 <https://juejin.cn/post/7688905828694294566>（41-xdp-tcpdump）计数由 40-mysql 检查点的 3425 展现 / 18 阅读 增至 3657 展现 / 25 阅读；<https://juejin.cn/post/7689030007914545198>（40-mysql）为 3 展现 / 3 阅读。
- 当前列出的 11 篇公开掘金文章全部保持 `暂无评论数据` 空评论状态，无需回复或更正。
- 外部回声：尚无转载或引用。下一检查点：38-btf-uprobe 额度使用后复查是否出现首条非空评论。

## eBPF 每日 Q&A（bpf-lpm-trie-lookup-prefixlen-caps-match）

- 观察时刻 2026-09-24 17:52 PDT。当日问题：BPF LPM-trie 查找 key 的 `prefixlen` 如何决定最长前缀匹配的胜负（`limit = min(node->prefixlen, key->prefixlen)`，`kernel/bpf/lpm_trie.c`）。
- 覆盖缺口（如实记录）：两个 watchlist 选中的 Slack 存档本次不可访问——只读快照读取器拒绝覆盖已存在的 0 字节快照文件，Step 0 快照返回 `output_exists`，未读到任何存档内容；allowlist 里的 Discord 频道与公开邮件列表仅限 visible-browser，且本次无可用浏览器会话。因此本次为**回退选择**：问题取自被监控的 eBPF 开发社区中反复出现的从业者边界，完全依据公开一手资料（内核 BPF 文档 + 上游 `lpm_trie.c` + `uapi bpf.h`），而非任何 thread；页内已如实标注该缺口。
- 发布：Commit A `364886a8e`（4 条路径：EN 页、ZH 镜像、两个 index），Pages run 36074436282 部署成功。
- 验证：校验器 re-verify 分支 rc=0，receipt `receipt-2026-09-24.json` status=published；本地 headless-chromium render 步骤在该运行环境失败（环境原因，非内容问题），故以 re-verify 分支跳过本地渲染/提交。
- 线上核验（EN+ZH 逐字 H1 + 内容锚点）：`192.168.0.5`、`min(node->prefixlen, key->prefixlen)`、`LPM_TREE_NODE_FLAG_IM`、`max_prefixlen`、`output_exists`（EN）；`当日社区讨论`（ZH）——均命中；两个 index 路由上的 href 均在线。
- 内容测试：`npm --prefix app run test:content` 82/82 通过（~14 min）。
