# 2026-11-16 内容巡检运行日志

运行日按 +08 计（系统日期 2026-11-16，UTC `2026-11-16T00:00Z` 起，LA 2026-11-15 16:00 PST）。本次巡检完成**一条 DEV 发布**；队列里 23 条知乎任务仍是平台级 `阻塞`。补发缺口 0。

## 任务选择

- 队列顶部 23 条 `排队` 任务全部是知乎；可见 Chrome 上 `https://www.zhihu.com/creator` 仍落 `/signin?next=%2Fcreator`，cookie 只有匿名标识（`_zap`/`d_c0`/`_xsrf`/`Hm_*`/`gdxidpyhxdE` 等），无 `z_c0`，属平台级 `阻塞`，未消耗额度。
- 掘金仍**掉登录**：`https://juejin.cn/creator/home` 首屏是登出态的创作者中心骨架（导航里有 `登录`/`注册`），settle 后落回 `https://juejin.cn/`，cookie 只有 `_tea_utm_cache_2608`/`s_v_web_id`/`__tea_cookie_tokens_2608`，无账号态。队列中掘金任务为 0，无待办。
- Medium 仍无 `MEDIUM_API_KEY`，可见 Chrome `/me/stories` 显示 Sign in；shell 也没有 `DEV_TO_API_KEY`，因此仍走 secret 支撑的 `publish-posts.yml`。
- X `@yunwei37` 已登录（显示 1,752 posts，bio 含 eunomia.dev）；LinkedIn `/in/yunwei37` 仍落 `/authwall`，本次只读到 X。
- 按"历史回补默认只处理 tutorial"取教程序列中最早的 DEV 缺口 `7-execsnoop`：`platforms/devto.json` 的明细行和 `confirmed_source_urls` 都没有它（上一版队列的下一步也是它），Medium 副本 `cc6213d19c76` 早已 `confirmed`。发布前用 DEV 公开 API 复查账号 74 篇、精确标题 0 篇重复；站点原文 `/tutorials/7-execsnoop/` 与 `/zh/tutorials/7-execsnoop/` 均 200。

## 准备稿

- `draft/media/2026-11-16/7-execsnoop/devto.md` 是源文 `docs/tutorials/7-execsnoop/README.md` 的**逐字节副本**（`diff -u` 无输出，5,103 字节，sha256 相同）。源文本身干净：无图片、无表格、无相对链接，2 个外链已是绝对地址，6 个围栏自带 `c`/`shell`/`console` 标签。
- 结构：1 个 H1（精确标题）、3 个正文 H2（`perf buffer`、`execsnoop`、`Summary`）、0 个 H3/H4、6 个围栏、0 图片、0 表格。
- 源文缺点（未改，逐字节同步）：最后一句以多余的 `."` 结尾，上游 `bpf-developer-tutorial` 同样如此；`README.zh.md` 侧未复核。

## 发布事实

- 路径：`.github/publisher/posts_queue.txt`（1 条，字段 `path`/`tags`）-> commit `ddf2bab1fa5` 推送 `main` -> `gh workflow run publish-posts.yml -f platforms=devto -f publish_count=1 -f draft_only=false`。run 43781972664 在 `2026-11-16T00:43:45Z` 公开：<https://dev.to/yunwei37/ebpf-tutorial-by-example-7-capturing-process-execution-output-with-perf-event-array-5d6c>，article id 5044058，tag `ebpf`/`linux`/`tutorial`/`kernel`，reading_time 3，0 反应 0 评论。workflow 随后自行推送 `c6cd16c` 清空队列。
- 正文对账：公开 `body_markdown` 与准备稿去首个 H1 后**逐字相同**（5,013 字符）。
- 结构对账（可见页 DOM）：1 h1（精确标题）、3 个正文 H2、0 H3、6 个 `pre`（`highlight c` x3、`highlight shell` x2、`highlight console` x1，没有围栏丢语言）、0 表格、0 正文图片、645 词、结尾完整（含 `bpf-developer-tutorial` 链接）、tag 芯片 4 个、0 评论。
- 覆盖：DEV 69 -> 70/128，`check_media_ledger.py` 退出 0（88 条 confirmed，0 条 confirmed-without-source_path）。
- 额度：发布时刻 `2026-11-16T00:43:45Z` = LA `2026-11-15 16:43 PST`，用掉 **LA 11-15 正常额度**（LA 11-15 当日此前无其它发布）；补发缺口仍 0。

## 观测与异常

- DEV 账号列表（公开 API，77 篇）里 `eBPF Tutorial: Tracing CUDA GPU Operations` 有两份公开副本：`2523222`（2025-05-25，`/…-20kp`）和 ledger 记录的 `3135152`（2025-12-30，`/…-45eb`），两者都 200。保留较新的作映射，未删除、未重发。另有 8 篇 2024 年随笔在 `docs/` 下找不到同名源（`Prompt Engineering is Dead…`、`Where Does Innovation Thrive` 等），符合"legacy `docs/blogs/` 不进默认源集"的规则，未入账。
- 运行延迟未复现：11-14 固定 `runs-on: ubuntu-24.04` 后，11-15 是 17 秒；本次 run 43781972664 在 `00:42:34Z` 创建、`00:43:36Z` 才起 job（约 62 秒排队）、`00:43:49Z` 完成，整条 1 分 15 秒。
- 首次进入 DEV 页面时可见 Chrome 是登出态（`/dashboard` 与 `/edit` 都跳 `/magic_links/new`）；发布与公开页 QA 不依赖它，编辑器修复路径本次仍不可用。

## 遗留

- canonical 仍为 DEV 自身（连续第十四次）。根因仍在上游 `media-publisher`：`publish-multi` 从 `title/content/tags/is_draft/platforms` 重建 payload，`canonical_url` 到不了 DEV 分支；已并入 `platforms/devto.json` 的同一条 `pending_verification`。
- tutorial 4 的源文围栏粘连（`docs/tutorials/4-opensnoop/README.md:107-109`）未回改上游 `bpf-developer-tutorial`。

## 下一次具体动作

1. 恢复知乎登录后，从 `53-egress-pacer` 起继续处理 23 条知乎任务（掘金已恢复登录，可作备用发布通道）。
2. 若 DEV 继续日更：下一候选 `8-exitsnoop`（DEV 缺失），按教程序号继续推进。