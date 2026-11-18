# 2026-11-18 内容巡检运行日志

运行日按 +08 计（系统日期 2026-11-18，UTC `2026-11-18T00:00Z` 起，LA 2026-11-17 16:00 PST）。本次巡检完成**一条 DEV 发布**；队列里 23 条知乎任务仍是平台级 `阻塞`。补发缺口 0。

## 任务选择

- 队列顶部 23 条 `排队` 任务全部是知乎；可见 Chrome 上 `https://www.zhihu.com/creator` 仍落 `/signin?next=%2Fcreator`，cookie 只有 `_zap`/`d_c0`/`_xsrf` 等匿名标识，无 `z_c0`，属平台级 `阻塞`，未消耗额度。
- 掘金仍**掉登录**：`https://juejin.cn/creator/home` 落回 `https://juejin.cn/`，导航为 登录/注册。队列中掘金任务为 0，无影响。
- Medium 仍无 `MEDIUM_API_KEY`；可见 Chrome `/me/stories` 本次落在 Cloudflare `Just a moment...` 验证页（Ray `a5c2c5da2b56eb2e`），并非登录页。
- X `@yunwei37` 已登录（1,753 posts）；LinkedIn `/in/yunwei37` 仍落 `/authwall`。
- 按"历史回补默认只处理 tutorial"取教程序列中最早的 DEV 缺口 `11-bootstrap`：`platforms/devto.json` 的明细行和 `confirmed_source_urls` 都没有它，Medium 副本早已 `confirmed`。发布前用 DEV 公开 API 复查账号 77 篇、精确标题 0 篇重复；站点原文 `/tutorials/11-bootstrap/` 与 `/zh/tutorials/11-bootstrap/` 均 200。

## 准备稿

- `draft/media/2026-11-18/11-bootstrap/devto.md` 与源文只有三处差异，每处都是英文源文缺陷，中文 `README.zh.md` 都干净：
  1. `README.md:125` 把 `if (start_ts)` 与 `duration_ns = bpf_ktime_get_ns() - *start_ts;` 挤成一行（中文 `README.zh.md:123-124` 是两行）。
  2. `README.md:269` 在代码行尾带内联 ` ``` `，提前闭合 `c` 围栏。渲染时其后 9 行代码与整段 `bootstrap.h` 块变成正文段落；站点原文 `/tutorials/11-bootstrap/` 的第 7 个 `pre` 里就能看到 `e->duration_ns = duration_ns;``` `。准备稿删掉这处内联围栏，块数恢复为源文的 19。
  3. `README.md:617` 在 `Compile and Run` 输出末行多出 `".`。
- 结构（准备稿）：1 个 H1（精确标题）、7 个正文 H2、2 个 H3、38 行首围栏 = 19 个代码块（`c` x16、`shell` x2、`console` x1）、0 图片、0 表格、0 相对链接。

## 发布事实

- 路径：`.github/publisher/posts_queue.txt`（1 条，字段 `path`/`tags`）-> commit `53331791e68` 推送 `main` -> `gh workflow run publish-posts.yml -f platforms=devto -f publish_count=1 -f draft_only=false`。run 44210208198 在 `2026-11-18T00:44:18Z` 公开：<https://dev.to/yunwei37/ebpf-tutorial-by-example-11-develop-user-space-programs-with-libbpf-and-trace-exec-and-exit-2mci>，article id 5060707，tag `ebpf`/`linux`/`tutorial`/`kernel`，reading_time 11，0 反应 0 评论。workflow 随后自行推送 `a6bc74e9110` 清空队列。
- 正文对账：公开 `body_markdown` 与准备稿去首个 H1 后**逐字相同**（22,532 字符）。
- 结构对账（可见页 DOM）：1 h1（精确标题）、7 个正文 H2、2 个 H3、19 个 `pre`（`highlight c` x16、`highlight shell` x2、`highlight console` x1）、0 处泄漏的反引号、0 表格、0 正文图片、结尾完整（含 `eunomia.dev/tutorials/11-bootstrap` 链接）、tag 芯片 4 个、0 评论。
- 覆盖：DEV 71 -> 72/128，`check_media_ledger.py` 退出 0（92 条 confirmed，0 条 confirmed-without-source_path）。
- 额度：发布时刻 `2026-11-18T00:44:18Z` = LA `2026-11-17 16:44 PST`，用掉 **LA 11-17 正常额度**（LA 11-17 当日此前无其它发布）；补发缺口仍 0。

## 观测与异常

- 运行延迟本次未复现：dispatch 后 12 秒起 job，整条 25 秒（11-17 那次排队 21 分钟）。
- 定时运行仍不接管空队列：`2026-11-17T16:34:19Z` 的定时 run 44127277298 读到空 `posts_queue.txt` 后输出 `No posts in queue to publish` 并成功退出，与 11-10..17 的先例一致。
- 列表滞后复现：发布后数分钟，公开作者 API 仍只返回 77 篇，`5060707` 不在列表里，而 per-article 端点与公开页都已 200。沿用 11-12 的结论：不要用列表判断发布结果。
- X 侧两条新帖有独立进展：`2122736534206181608`（Lean 证明检查成本）7 小时 939 浏览 / 12 赞 / 2 转帖 / 7 书签，唯一回复（`@difficultyang`）建议让 agent 别写检查代价高的证明，属观点类，暂不需回复；`2122011756633665955`（bpf-gpu-server）从 885 浏览 / 29 赞升到 941 / 30。
- HN 无新回声：`search_by_date` 最新的 eunomia 故事仍是 50385391（AgentSight，2026-11-05，2 分 0 评论）；11-14..17 的 "eunomia" 评论命中都是 "Euromaxxed"/"Europoor" 之类的模糊匹配。

## 遗留

- canonical 仍为 DEV 自身（连续第十六次）。根因仍在上游 `media-publisher`：`publish-multi` 从 `title/content/tags/is_draft/platforms` 重建 payload，`canonical_url` 到不了 DEV 分支。
- `docs/tutorials/11-bootstrap/README.md:125`、`:269`、`:617` 三处缺陷仍在源文里；阅读页 `/tutorials/11-bootstrap/` 当前就是这么渲染的。修需回改上游 `bpf-developer-tutorial`。

## 下一次具体动作

1. 恢复知乎登录后，从 `53-egress-pacer` 起继续处理 23 条知乎任务（掘金恢复登录后可作备用发布通道）。
2. 若 DEV 继续日更：下一候选 `12-profile`（DEV 缺失，`13-tcpconnlat`/`14-tcpstates`/`15-javagc` 同样缺）。