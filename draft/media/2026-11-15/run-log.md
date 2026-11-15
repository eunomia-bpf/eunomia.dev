# 2026-11-15 内容巡检运行日志

运行日按 +08 计（系统日期 2026-11-15，UTC `2026-11-15T00:00Z` 起，LA 2026-11-14 16:00 PST）。本次巡检完成**一条 DEV 发布**；队列里 23 条知乎任务仍是平台级 `阻塞`。补发缺口 0。

## 任务选择

- 队列顶部 23 条 `排队` 任务全部是知乎；可见 Chrome 上 `https://www.zhihu.com/creator` 仍落 `/signin?next=%2Fcreator`，cookie 只有匿名标识（`_zap`/`d_c0`/`_xsrf` 等），无 `z_c0`，属平台级 `阻塞`，未消耗额度。
- 掘金仍**掉登录**：`https://juejin.cn/creator/home` 落到 `https://juejin.cn/` 并显示 登录/注册（11-13 还是已登录，11-14 起如此）。队列中掘金任务为 0，无影响。
- Medium 无 `MEDIUM_API_KEY`，可见 Chrome `/me/stories` 显示 Sign in；shell 也没有 `DEV_TO_API_KEY`，因此仍走 secret 支撑的 `publish-posts.yml`。
- 按"历史回补默认只处理 tutorial"取教程序列中最早的 DEV 缺口 `6-sigsnoop`。发布前用 DEV 公开 API 复查账号 75 篇、精确标题 0 篇重复；站点原文 `/tutorials/6-sigsnoop/` 与 `/zh/tutorials/6-sigsnoop/` 均 200 且 H1 一致。

## 队列记录缺陷（已修）

- 上一版队列把下一个候选写成 `5-uprobe-bashreadline`（"Medium 已确认，DEV 缺失"）。实际它的 DEV 副本 2024-01-20 就已在线（article 1736310，`...-capturing-readline-function-calls-with-uprobe-ad7`，URL 200），并且早就在 `platforms/devto.json` 的 `confirmed_source_urls` 里，只是没有对应的 `published` 明细行，所以肉眼按明细行找缺口时会把它当成缺失。
- 本次把判定基准改成 `confirmed_source_urls ∪ published[*].source_path`（也就是 `check_media_ledger.py` 用的口径），据此跳过 5，取 `6-sigsnoop`。

## 准备稿

- `draft/media/2026-11-15/6-sigsnoop/devto.md` 是源文 `docs/tutorials/6-sigsnoop/README.md` 的**逐字节副本**（`diff -u` 无输出，4895 字节）。源文本身干净：无图片、无表格、无相对链接，3 个外链已是绝对地址，5 个围栏自带 `c`/`shell`/`console` 标签，无行尾空白和制表符。
- 结构：1 个 H1（精确标题）、2 个正文 H2（`sigsnoop`、`Summary`，本来就是 H2，满足"body 最高级为 H2"）、0 个 H3/H4、5 个围栏、0 图片、0 表格、652 词。

## 发布事实

- 路径：`.github/publisher/posts_queue.txt`（1 条，字段 `path`/`tags`）-> commit `a2c451c7c7` 推送 `main` -> `gh workflow run publish-posts.yml -f platforms=devto -f publish_count=1 -f draft_only=false`。run 43657764558 在 `2026-11-15T00:21:37Z` 公开：<https://dev.to/yunwei37/ebpf-tutorial-by-example-6-capturing-signal-sending-and-store-state-with-hash-maps-12h9>，article id 5038589，tag `ebpf`/`linux`/`tutorial`/`kernel`，0 反应 0 评论。workflow 随后自行推送 `49dc97fbb5` 清空队列。
- 正文对账：公开 `body_markdown` 与准备稿去首个 H1 后**逐字相同**（4,807 字符）。
- 结构对账（可见页 DOM）：1 h1（精确标题）、2 个正文 H2、0 H3、5 个 `pre`（`pre.highlight c` x2、`highlight shell` x1、`highlight console` x2，没有围栏丢语言）、0 表格、0 正文图片、625 词、结尾完整（含 `bpf-developer-tutorial` 与 `eunomia.dev/tutorials/` 两个链接）、0 评论。
- 覆盖：DEV 68 -> 69/128，`check_media_ledger.py` 退出 0（86 条 confirmed，0 条 confirmed-without-source_path）。
- 额度：发布时刻 `2026-11-15T00:21:37Z` = LA `2026-11-14 16:21 PST`，用掉 **LA 11-14 正常额度**（LA 11-14 当日此前无其它发布）；补发缺口仍 0。

## 运行延迟

- 11-14 把 `publish-posts.yml` 的 `runs-on` 固定为 `ubuntu-24.04` 后，本次 dispatch 到取得 runner 只用 17 秒、整条 run 17 秒完成（11-14 那次是 7 分钟）。单次快跑不足以说明组织级积压消失，只记录为"本次未复现"。

## 遗留

- canonical 仍为 DEV 自身（连续第十三次）。根因仍在上游 `media-publisher`：`publish-multi` 从 `title/content/tags/is_draft/platforms` 重建 payload，`canonical_url` 到不了 DEV 分支；已并入 `platforms/devto.json` 的同一条 `pending_verification`。
- tutorial 4 的源文围栏粘连（`docs/tutorials/4-opensnoop/README.md:107-109`）未回改上游 `bpf-developer-tutorial`；下一次同步前它仍会出现在每一份新的 syndication 里。

## 下一次具体动作

1. 恢复知乎登录后，从 `53-egress-pacer` 起继续处理 23 条知乎任务；恢复掘金登录（当前无任务）。
2. 若 DEV 继续日更：下一候选 `7-execsnoop`（DEV 缺失），按教程序号继续推进。

## 站点部署

- 本次只改文档与账本，但仍推送到 `main`，所以触发了 `Deploy Static App`：run 43658878929（head `592f83a4f5`）在 `2026-11-15T01:03Z` `success`，13 个步骤全绿（`Verify static export` 用了约 23 分钟，是 `npm ci` + `npm run verify` 的常态耗时）。部署后 `https://eunomia.dev/tutorials/6-sigsnoop/` 与 `/zh/tutorials/6-sigsnoop/` 均 200，正文各含 5 处 `sigsnoop`。
- 补记 run-log 的这次小改动又触发 run 43661414816（head `a8c978fdb42`），`2026-11-15T01:38Z` `success`。两次都已经 `Deploy to GitHub Pages` 完成，线上 `https://eunomia.dev/tutorials/6-sigsnoop/` 仍是 200。（教训：当天最后一次 `main` 推送尽量把所有文档改动合并进同一个 commit，避免为纯账本变动重复付 40 分钟的 `npm verify` + build。）