# 2026-11-17 内容巡检运行日志

运行日按 +08 计（系统日期 2026-11-17，UTC `2026-11-17T00:00Z` 起，LA 2026-11-16 16:00 PST）。本次巡检完成**一条 DEV 发布**；队列里 23 条知乎任务仍是平台级 `阻塞`。补发缺口 0。

## 任务选择

- 队列顶部 23 条 `排队` 任务全部是知乎；可见 Chrome 上 `https://www.zhihu.com/creator` 仍落 `/signin?next=%2Fcreator`，cookie 只有 `d_c0`/`_zap` 等匿名标识，无 `z_c0`，属平台级 `阻塞`，未消耗额度。
- 掘金仍**掉登录**：`https://juejin.cn/creator/home` 落回 `https://juejin.cn/`，导航为 登录/注册。队列中掘金任务为 0，无影响。
- Medium 仍无 `MEDIUM_API_KEY`，可见 Chrome `/me/stories` 显示 Sign in；shell 也没有 `DEV_TO_API_KEY`，因此仍走 secret 支撑的 `publish-posts.yml`。
- X `@yunwei37` 已登录（1,752 posts）；LinkedIn `/in/yunwei37` 仍落 `/authwall`。
- 按"历史回补默认只处理 tutorial"取教程序列中最早的 DEV 缺口 `8-exitsnoop`：`platforms/devto.json` 的明细行和 `confirmed_source_urls` 都没有它，Medium 副本 `73291d5e3a50` 早已 `confirmed`。发布前用 DEV 公开 API 复查账号 77 篇、精确标题 0 篇重复；站点原文 `/tutorials/8-exitsnoop/` 与 `/zh/tutorials/8-exitsnoop/` 均 200。

## 准备稿

- `draft/media/2026-11-17/8-exitsnoop/devto.md` 与源文只有一行差异：源文 `Run:` console 块的表头是 `TIME     PID     PPID    EXIT_CODE  DURATION_NS  COMM".`，准备稿删掉多余的 `".`；中文源文 `README.zh.md` 同一行是干净的 `COMM`。其余逐字相同（6,989 字节）。
- 结构：1 个 H1（精确标题）、4 个正文 H2（`Ring Buffer`、`exitsnoop`、`Compile and Run`、`Summary`）、1 个 H3、5 个围栏（`c` x2、`shell` x1、`console` x2）、0 图片、0 表格、无 HTML 注释、无相对链接。
- 源文缺陷（未改同步源，只修准备稿）：`docs/tutorials/8-exitsnoop/README.md:140` 的 `COMM".` 来自上游 `bpf-developer-tutorial`，与 tutorial 4 的围栏粘连同属同步源未回改的遗留。

## 发布事实

- 路径：`.github/publisher/posts_queue.txt`（1 条，字段 `path`/`tags`）-> commit `75085bfacdd` 推送 `main` -> `gh workflow run publish-posts.yml -f platforms=devto -f publish_count=1 -f draft_only=false`。run 43986441112 在 `2026-11-17T00:38:36Z` 公开：<https://dev.to/yunwei37/ebpf-tutorial-by-example-8-monitoring-process-exit-events-output-with-ring-buffer-5ea2>，article id 5052614，tag `ebpf`/`linux`/`tutorial`/`kernel`，reading_time 4，0 反应 0 评论。workflow 随后自行推送 `37785955a0e` 清空队列。
- 正文对账：公开 `body_markdown` 与准备稿去首个 H1 后**逐字相同**（6,901 字符）。
- 结构对账（可见页 DOM）：1 h1（精确标题）、4 个正文 H2、1 H3、5 个 `pre`（`highlight c` x2、`highlight shell` x1、`highlight console` x2）、0 表格、0 正文图片、953 词、结尾完整（含 `bpf-developer-tutorial` 链接）、tag 芯片 4 个、0 评论。
- 覆盖：DEV 70 -> 71/128，`check_media_ledger.py` 退出 0（90 条 confirmed，0 条 confirmed-without-source_path）。
- 额度：发布时刻 `2026-11-17T00:38:36Z` = LA `2026-11-16 16:38 PST`，用掉 **LA 11-16 正常额度**（LA 11-16 当日此前无其它发布）；补发缺口仍 0。

## 观测与异常

- 运行延迟复现：dispatch `00:17:37Z`，job 到 `00:38:26Z` 才启动（排队约 21 分钟），整条 21 分 22 秒。同期 bpftime `queued` 稳定 39-40 条、本仓库另 2 条 queued。11-14 固定的 `runs-on: ubuntu-24.04` 没有挡住这段排队；11-15 的 17 秒与 11-16 的 62 秒都是短窗口。
- 列表滞后：发布 20 分钟后公开作者 API 仍只返回 77 篇、`5052614` 不在列表里，而 per-article 端点与公开页都是 200。沿用 11-12 的结论：不要用列表判断发布结果。
- X 分享（bpf-gpu-server）从 ~8 小时的 300 浏览 / 12 赞 / 3 书签 / 1 回复升到 885 / 29 / 10 / 1；唯一回复仍只是复述帖子的结论，无需回复。
- Medium 的 tutorial 8 旧副本本次可读（无 Cloudflare 拦截），标题为 `… Exit Events, Print Output with Ring Buffer`，比源文多一个 `Print`，是 2023 年导入的旧缺陷；修需登录 Medium 编辑器。

## 遗留

- canonical 仍为 DEV 自身（连续第十五次）。根因仍在上游 `media-publisher`：`publish-multi` 从 `title/content/tags/is_draft/platforms` 重建 payload，`canonical_url` 到不了 DEV 分支；已并入 `platforms/devto.json` 的同一条 `pending_verification`。
- `docs/tutorials/8-exitsnoop/README.md:140` 的 `COMM".` 与 tutorial 4 的 `README.md:107-109` 围栏粘连都仍未回改上游 `bpf-developer-tutorial`。
- 队列记录口径：`9-runqlat`/`10-hardirqs`/`16-memleak` 没有明细行但已在 `confirmed_source_urls`，`check_media_ledger.py` 按覆盖计算已计入；下一 DEV 缺口是 `11-bootstrap`。

## 下一次具体动作

1. 恢复知乎登录后，从 `53-egress-pacer` 起继续处理 23 条知乎任务（掘金恢复登录后可作备用发布通道）。
2. 若 DEV 继续日更：下一候选 `11-bootstrap`（DEV 缺失），按教程序号继续推进。