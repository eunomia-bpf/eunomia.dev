# 2026-11-12 内容巡检运行日志

运行日按 +08 计（系统日期 2026-11-12，UTC `2026-11-12T00:00Z` 起，LA 2026-11-11 16:00 PST）。本次巡检完成**一条 DEV 发布**；队列里 23 条知乎任务仍是平台级 `阻塞`，Medium 无 API key 且公开副本被 Cloudflare 拦。补发缺口 0。

## 任务选择

- 队列顶部 23 条 `排队` 任务全部是知乎，可见 Chrome 打开 `/creator` 仍跳 `/signin?next=%2Fcreator`（cookie 只有 `_zap`/`d_c0`/`_xsrf`/`SESSIONID` 等匿名或挂载态标识，无 `z_c0`），属平台级 `阻塞`。
- Medium 双重不可用：shell 无 `MEDIUM_API_KEY`；`medium.com/me/stories/public` 未登录，且 tutorial 2 旧副本本次直接返回 Cloudflare `403` + `Just a moment...`。
- 按“历史回补默认只处理 tutorial”取教程序列中最早的 DEV 缺口：`2-kprobe-unlink`（Medium `bbdd8c020dbb` 已 `confirmed`，DEV 无）。发布前用 DEV 公开 API 复查账号 71 篇、精确标题 0 篇重复。

## 发布事实

- 准备稿 `draft/media/2026-11-12/2-kprobe-unlink/devto.md` 是源文逐字节副本：源文无图片、正文链接已是绝对地址、8 个无语言围栏保持原样，无需任何改写。
- 路径：`.github/publisher/posts_queue.txt`（1 条，字段 `path`）-> commit `d3f3e2d1a6` 推送 `main` -> `gh workflow run publish-posts.yml -f platforms=devto -f publish_count=1 -f draft_only=false`。run 43126676054 在 `2026-11-12T00:21:38Z` 公开：<https://dev.to/yunwei37/ebpf-tutorial-by-example-2-monitoring-unlink-system-calls-with-kprobe-259l>，article id 5019807，tag `ebpf`/`linux`/`tutorial`/`kernel`，reading_time 7，0 反应 0 评论。workflow 随后自行推送 `9728456ac1` 清空队列。
- 正文对账：公开 `body_markdown` 与准备稿去首个 H1 后**逐字相同**（10949 字符，无围栏改写）。
- 结构对账：1 h1、3 个正文 H2（= 源文 3 个 `##`）、0 H3、8 个 `pre`（16 个围栏）、1 个 10 项有序列表、0 表格、0 图片、1575 词、结尾完整。无语言围栏未被 DEV 重新标注，但仍着色为 `highlight c` x4 / `console` x2 / `shell` x2，未丢语言。
- 外链：`github.com/eunomia-bpf/eunomia-bpf`、`github.com/eunomia-bpf/bpf-developer-tutorial`、`eunomia.dev/tutorials/` 均 200；站点原文 <https://eunomia.dev/tutorials/2-kprobe-unlink/> 200 且 H1 一致。
- 覆盖：DEV 65 -> 66/128，`check_media_ledger.py` 退出 0（81 条 confirmed，0 条 confirmed-without-source_path）。

## 遗留

- canonical 仍为 DEV 自身（连续第九次）。根因仍在上游 `media-publisher`：`publish-multi` 从 `title/content/tags/is_draft/platforms` 重建 payload，`canonical_url` 到不了 DEV 分支；已并入 `platforms/devto.json` 的同一条 `pending_verification`，不另开新条目。
- DEV 作者列表继续滞后：发布 10 分钟后 `/api/articles?username=yunwei37` 仍返回 71 篇且不含新文，而按路径 API 与可见页已可读。列表计数不能当发布证据。
- tutorial 2 的 2023 Medium 副本本次无法复查（Cloudflare `403`），既有格式缺陷状态未验证。

## 下一次具体动作

1. 恢复 Chrome 配置里的知乎登录后，从 `53-egress-pacer` 起继续处理 23 条知乎任务。
2. 若 DEV 继续日更：下一候选 `3-fentry-unlink`（Medium `cb6a374da48b` 已确认、DEV 缺失），之后 `4-opensnoop`。