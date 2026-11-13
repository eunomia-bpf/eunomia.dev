# 2026-11-13 内容巡检运行日志

运行日按 +08 计（系统日期 2026-11-13，UTC `2026-11-13T00:00Z` 起，LA 2026-11-12 16:00 PST）。本次巡检完成**一条 DEV 发布**；队列里 23 条知乎任务仍是平台级 `阻塞`。补发缺口 0。

## 任务选择

- 队列顶部 23 条 `排队` 任务全部是知乎。可见 Chrome 打开 `https://www.zhihu.com/creator` 仍落 `https://www.zhihu.com/signin?next=%2Fcreator`，没有 `z_c0`，属平台级 `阻塞`，未消耗任何额度。
- Medium：shell 仍无 `MEDIUM_API_KEY`（`GET /api/articles/me/all` 用空 key 返回 401）。与 11-11/12 记录的"编辑器未登录"不同，本次可见 Chrome 的 `medium.com/me/stories` 是**已登录**状态（有 Write、Drafts/Published 标签、user options menu），浏览器会话本身可用，只是没有走编辑器发布。
- 掘金：可见 Chrome 已登录，`https://juejin.cn/creator` 落 `/creator/home`；队列中掘金任务为 0，没有待办。
- 按"历史回补默认只处理 tutorial"取教程序列中最早的 DEV 缺口 `3-fentry-unlink`（Medium `cb6a374da48b` 自 2023 年已 `confirmed`，DEV 缺失）。发布前用 DEV 公开 API 复查账号 73 篇、精确标题 0 篇重复。

## 发布事实

- 准备稿 `draft/media/2026-11-13/3-fentry-unlink/devto.md` 与源文只差一处：源文正文的 `[example 2-kprobe-unlink](../2-kprobe-unlink/README.md)` 是站内相对路径，同步到 DEV 会变成 DEV 域下的死链，准备稿改写为 `https://eunomia.dev/tutorials/2-kprobe-unlink/`（HTTP 200）。其余逐字节相同（6,735 字节）。
- 路径：`.github/publisher/posts_queue.txt`（1 条，字段 `path`/`tags`）-> commit `6aea5f95d6` 推送 `main` -> `gh workflow run publish-posts.yml -f platforms=devto -f publish_count=1 -f draft_only=false`。run 43335255543 在 `2026-11-13T00:13:10Z` 公开：<https://dev.to/yunwei37/ebpf-tutorial-by-example-3-monitoring-unlink-system-calls-with-fentry-k28>，article id 5026194，tag `ebpf`/`linux`/`tutorial`/`kernel`，reading_time 4，0 反应 0 评论。workflow 随后自行推送 `95b9c816b4` 清空队列。
- 正文对账：公开 `body_markdown` 与准备稿去首个 H1 后**逐字相同**（6,678 字符；只有结尾换行差一个，`splitlines()` 完全相等）。
- 结构对账：1 h1（精确标题）、3 个正文 H2（= 源文 3 个 `##`）、1 H3、0 H4、9 个 `pre`（18 个围栏，源文自带 `c` x1 / `console` x7 / `shell` x1，DEV 渲染为 `pre.highlight.c`/`console`/`shell`，无围栏丢语言）、0 表格、0 正文图片、939 词、结尾完整。
- 外链：`github.com/eunomia-bpf/eunomia-bpf`、`github.com/eunomia-bpf/bpf-developer-tutorial`、`eunomia.dev/tutorials/`、`eunomia.dev/tutorials/2-kprobe-unlink/` 均 200；站点原文 <https://eunomia.dev/tutorials/3-fentry-unlink/> 200 且 H1 一致。
- 覆盖：DEV 66 -> 67/128，`check_media_ledger.py` 退出 0（82 条 confirmed，0 条 confirmed-without-source_path）。

## 同次修正

- tutorial 1 DEV 副本的历史记录里有 3 处写成 `...-hello-world-framework-and-development-process-2kmb`（多一个 `process`）。该 slug 是 404；公开 API 与 `platforms/devto.json` 的正式地址是 `...-hello-world-framework-and-development-2kmb`（200）。已修正 `draft/plan/publishing-queue.zh.md`、`draft/media/2026-11-11/run-log.md` 与 `.github/publisher/media/community-feedback.md`；ledger、`published.md`、`not-published.md` 原本就正确。11-11 记录的"发布后 404"因此有一次误判风险：当时确实读取了 404，但后续复核引用的是错 slug，本次以正确 slug 复核为 200。

## 遗留

- canonical 仍为 DEV 自身（连续第十次）。根因仍在上游 `media-publisher`：`publish-multi` 从 `title/content/tags/is_draft/platforms` 重建 payload，`canonical_url` 到不了 DEV 分支；已并入 `platforms/devto.json` 的同一条 `pending_verification`。
- DEV 作者列表继续滞后：本次按账号列表读到 73 篇（比 11-12 的 71 篇多 2 篇），新文在发布后数分钟内仍可能不在列表里；列表计数不能当发布证据。
- tutorial 1/2 的 2023 年 Medium 旧副本格式缺陷未复查（需要登录编辑器逐个修），本次未改变其 open 状态。

## 下一次具体动作

1. 恢复 Chrome 配置里的知乎登录后，从 `53-egress-pacer` 起继续处理 23 条知乎任务。
2. 若 DEV 继续日更：下一候选 `4-opensnoop`（DEV 缺失），之后 `5-...` 序列继续按教程序号推进。