# 2026-11-10 内容巡检运行日志

运行日按 +08 计（系统日期 2026-11-10，UTC `2026-11-10T01:00Z` 起，LA 2026-11-09 16:00 PST）。本次巡检完成**一条 DEV 发布**，并补记 3 条被漏记的旧 DEV 副本；队列里 23 条知乎任务仍是平台级 `阻塞`，Medium 无 API key 且编辑器未登录，补发缺口 0。

## 任务选择

- 队列顶部 23 条 `排队` 任务全部是知乎，因可见 Chrome 未登录而处于平台级 `阻塞`（`/creator` -> `/signin`），不能卡住其它平台。
- Medium 路径双重不可用：shell 无 `MEDIUM_API_KEY`，可见编辑器登出。2026 年英文 Blog 已无 Medium/DEV 缺口，按"历史回补默认只处理 tutorial"转向教程序列。
- 教程序列按编号取最早者：`0-introduce`（Medium 已 `confirmed`，DEV 缺失）。`1-helloworld` 作为备选一并检查过，两者都是无图、无相对链接的干净源。
- 发布前用 DEV 公开 API 复查：账号 69 篇、精确标题 `eBPF Tutorial by Example 0: Introduction to Core Concepts and Tools` 0 篇重复。

## 发布事实

- 准备稿 `draft/media/2026-11-10/0-introduce/devto.md` 是源文逐字节副本，唯一改动是删掉最后一句末尾多出来的闭合 `".`（`docs/tutorials/0-introduce/README.md` 与上游 `bpf-developer-tutorial/src/0-introduce/README.md` 都仍带着它）。tutorial 源文由上游仓库同步进本仓库，因此没有在本仓库改源。
- 路径：`.github/publisher/posts_queue.txt`（1 条，字段 `path`）-> commit `f43587cf2f` 推送 -> `gh workflow run publish-posts.yml -f platforms=devto -f publish_count=1 -f draft_only=false`。
- 第一次触发（run 42732618639）读到空队列：workflow 从 `origin/main` 取 `posts_queue.txt`，而队列条目当时只在本地。推送后重跑 run 42732728828 于 `2026-11-10T01:01:58Z` 公开：<https://dev.to/yunwei37/ebpf-tutorial-by-example-0-introduction-to-core-concepts-and-tools-1j0f>，article id 5007697，tag `ebpf`/`linux`/`tutorial`/`kernel`，reading_time 9，0 反应 0 评论。
- 正文对账：公开 `body_markdown` 与准备稿去首个 H1 后**逐字相同**（15453 字符，无围栏重标注、无实体改写）。
- 结构对账：1 h1、7 个正文 H2（= 源文 7 个 `##`）、10 H3、0 个 `pre`（源文本就没有围栏）、0 表格、0 图片、2097 词、结尾完整；正文外链（tutorial 仓库、eunomia.dev/tutorials、ebpf.io、docs.ebpf.io、bpftime、bcc、libbpf-bootstrap、eunomia-bpf）全部可点。
- 覆盖：DEV 60 -> 64/128（其中 3 条为下节补记），ledger `confirmed` 14 -> 18 条。

## 补记的 ledger 缺口

`cgroup`、`features/bpf_token`、`features/dynptr` 三篇的 DEV 副本自 2026-02-17/02-24/03-17 起就公开在线，但只有 Medium 副本入账，DEV 侧八个月来一直按"未发布"计数。本次用公开 API（title/tags/body 长度）与可见页（h1、H2 数、`pre` 数、0 图片、结尾完整）复查后补入 `platforms/devto.json`（`devto-1k2d`/`devto-3b5i`/`devto-cj2`）并在 `published.md` 加行，`check_media_ledger.py` 现报 64/128、0 confirmed-without-source_path。

## canonical 缺陷（连续第七次）

DEV 副本自指，站点原文 <https://eunomia.dev/tutorials/0-introduce/> 在线且渲染正常。根因仍在上游 `media-publisher` 的两个端点（`canonical_url` 在 `/api/publish-multi` 与 `/api/publish/dev` 都会被丢掉），本仓库脚本单改无效；只写进 `platforms/devto.json` 的 `pending_verification`，不再重复排查。

## 会话状态

可见 Chrome 在知乎 `/creator`、掘金 `/creator/home`、Medium、DEV 编辑器均登出；shell 无 `MEDIUM_API_KEY`/`DEV_TO_API_KEY`。secret 支撑的 `publish-posts` 仍是唯一可用发布路径。23 条知乎 `排队` 任务保持平台级 `阻塞`。

## 下一次具体动作

1. 恢复 Chrome 配置里的知乎登录后，从 `53-egress-pacer` 起继续处理 23 条知乎任务。
2. 若 DEV 继续日更：下一个候选是 `1-helloworld`（Medium 已确认、DEV 缺失，源文干净）；再往后按编号 `2-kprobe-unlink`、`3-fentry-unlink`、`4-opensnoop` 依次取用。
3. 上游 `bpf-developer-tutorial` 的 `src/0-introduce/README.md` 结尾多出的 `".` 建议在上游修掉，否则下次同步会把它带回本仓库。