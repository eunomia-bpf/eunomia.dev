# 2026-11-11 内容巡检运行日志

运行日按 +08 计（系统日期 2026-11-11，UTC `2026-11-11T00:00Z` 起，LA 2026-11-10 16:00 PST）。本次巡检完成**一条 DEV 发布**；队列里 23 条知乎任务仍是平台级 `阻塞`，Medium 无 API key 且编辑器未登录。补发缺口 0。

## 任务选择

- 队列顶部 23 条 `排队` 任务全部是知乎，可见 Chrome 打开 `/creator` 仍跳 `/signin`，属平台级 `阻塞`，不能卡住其它平台。
- Medium 路径双重不可用：shell 无 `MEDIUM_API_KEY`，可见 `/me/stories` 未登录；2026 年英文 Blog 的 Medium/DEV 副本已无缺口，按"历史回补默认只处理 tutorial"转向教程序列。
- 教程序列按编号取最早者：`1-helloworld`（Medium `33d4220022a4` 已 `confirmed`，DEV 缺失）。发布前用 DEV 公开 API 复查账号 71 篇、精确标题 0 篇重复。
- 定时运行不会自行排队：`2026-11-10T15:18:17Z` 的 run 42842937432 读到空 `posts_queue.txt`、无发布却成功结束，因此按 2026-11-10 的先例手工入队后 dispatch。

## 发布事实

- 准备稿 `draft/media/2026-11-11/1-helloworld/devto.md` 与源文只有一处差异：源文（与上游 `bpf-developer-tutorial/src/1-helloworld/README.md` 一致）把 "…publishing it to Github Packages." 与 "- GitHub Actions, used for automating build, test, and release processes" 挤在同一行，渲染成一个连写条目；准备稿拆成两条 `- `。源文由上游同步进本仓库，本次未改源。
- 路径：`.github/publisher/posts_queue.txt`（1 条，字段 `path`）-> commit `b506fe766c` 推送 `main` -> `gh workflow run publish-posts.yml -f platforms=devto -f publish_count=1 -f draft_only=false`。run 42929758642 在 `2026-11-11T00:17:57Z` 公开：<https://dev.to/yunwei37/ebpf-tutorial-by-example-1-hello-world-framework-and-development-2kmb>，article id 5013503，tag `ebpf`/`linux`/`tutorial`/`kernel`，0 反应 0 评论。workflow 随后自行推送 `397f06da97` 清空队列。
- 正文对账：公开 `body_markdown` 与准备稿去首个 H1 后**逐字相同**（15630 字符；10 个无语言围栏保持无标注，未被重新标注）。
- 结构对账：1 h1、7 个正文 H2（= 源文 7 个 `##`）、1 H3、10 个 `pre`（20 个围栏）、0 表格、0 图片、2233 词、结尾完整；正文外链（releases、4 个 starter template、GitHub template 文档、tutorial 仓库、eunomia.dev/tutorials）全部可点。
- 覆盖：DEV 64 -> 65/128，`check_media_ledger.py` 退出 0（79 条 confirmed，0 条 confirmed-without-source_path）。

## 公开页 404（已自愈）

- 00:20Z 首次打开公开页时页面完整：1 h1、7 正文 H2、1 H3、10 `pre`、0 图 0 表、2233 词、结尾完整，4 组外链全部 200，正是上面对账的依据。
- 约 00:24Z 起同一 URL 开始返回 `404: Page Not Found`，持续到约 01:44Z。特征：整段 404 都来自同一条 Fastly 负缓存（每次请求 `x-request-id` 相同，`age` 从 71 s 一路涨到 4612 s），随机 query string、`Cache-Control: no-cache`、爬虫 UA、换边缘节点都消不掉；期间其它边缘节点的 `200` 与它并存。
- 同时段 `/api/articles/5013503`、`/api/articles?tag=ebpf`、作者列表、RSS、可见 profile 卡片都一直能列出该文。`/api/articles/yunwei37/<slug>` 与页面共享同一条 404、又在同一刻恢复，因此当时「按 id 可取、按路径不可取」指向的是缓存而不是未列出/审核态——这一点我最初判断错了，记录里的 `pending_verification` 已按实测更正。
- 01:46Z 起该 URL、按路径 API、RSS 全部 200；01:49-01:52Z 连续 26 次请求全 200，独立代理 `r.jina.ai` 也是 200。可见 Chrome 重新做整页 DOM 复核，数字与 00:20Z 首次 QA 完全一致（1 h1、7 正文 H2、1 H3、10 `pre`、0 表 0 图、2233 词、结尾完整、tag `ebpf`/`linux`/`tutorial`/`kernel`、无 unlisted 提示、canonical 仍是 DEV 自身）。
- 结论：发布与页面都成立，`platforms/devto.json` 的 `devto-2kmb` 保持 `confirmed`，`checked_via` 记为 `..._before_and_after_url_recovery`；`pending_verification` 里那条待登录复核已撤销。没有重发，也没有改 slug。

## canonical 缺陷（连续第八次）

DEV 副本自指，站点原文 <https://eunomia.dev/tutorials/1-helloworld/> 在线（1 h1 / 7 H2 / 10 `pre` / 2452 词）。根因仍在上游 `media-publisher`：`publish-multi` 从 `title/content/tags/is_draft/platforms` 重建 payload，`canonical_url` 在到达 DEV 分支前就被丢掉，本仓库脚本单改无效；只写进 `platforms/devto.json` 的 `pending_verification`。

## 旧副本缺陷（只记录，未修）

tutorial 1 的 Medium 副本（2023-08-15）仍在线且 `canonical` 正确指向 eunomia.dev 原文，但正文有四处旧导入缺陷：`###` 被压成 `##`（H2/H3 层级丢失）、代码围栏无语言标注、`./ecli run package.json` 被吞成 `eclirunpackage.json`、以及与源文相同的连写条目。修需登录 Medium 编辑器，当前会话登出。

## 会话状态

可见 Chrome 在知乎 `/creator`、掘金、Medium、X、DEV 编辑器均登出；shell 无 `MEDIUM_API_KEY`/`DEV_TO_API_KEY`。secret 支撑的 `publish-posts` 仍是唯一可用发布路径；23 条知乎 `排队` 任务保持平台级 `阻塞`。

## 下一次具体动作

0. article 5013503 已自愈并完成整页复核，无需再查；若下次遇到同类 404，先看 `x-request-id`/`age` 是否单调，并至少等满 90 分钟再判断（本次负缓存存活约 80 分钟）。

1. 恢复 Chrome 配置里的知乎登录后，从 `53-egress-pacer` 起继续处理 23 条知乎任务。
2. 若 DEV 继续日更：下一候选 `2-kprobe-unlink`（Medium 已确认、DEV 缺失），之后按编号 `3-fentry-unlink`、`4-opensnoop` 依次取用。
3. 上游 `bpf-developer-tutorial` 的 `src/1-helloworld/README.md` 第 182 行合并条目建议在上游拆开，否则每次同步都会把连写带回本仓库。