# 2026-11-27 内容巡检运行日志

运行日按 +08 计（UTC `2026-11-27T00:00Z` 起，LA 2026-11-26 16:00 PST）。本次巡检完成**一条 DEV 发布**；队列里 23 条知乎任务仍是平台级 `阻塞`。补发缺口 0。

## 运行前状态

- CDP 9222 直接可用（`Chrome/146.0.7680.31`，supervisor 日志停在 11-26 的 `Chromium ready on 127.0.0.1:9222, pid=39153`），无需再打 `runuser` 软链。
- omp 自带的 headless 浏览器 daemon 起不来（`hub ps` 里 `omp.browser.headless: failed exit=21`），attach 到 9222 时首次 `browser.open` 又超时；改用 `app: { cdp_url }` 并在 `tab.run` 内用 `page.evaluate` 后全部正常。直接 `browser.open` 不带 `cdp_url` 会失败。
- 本地 `main` 落后 `origin/main` 9 个提交（含 `1054560` patrol prompt 改动、`e02a8f7`/`26f74b9` 公司页），rebase 后再动手。
- 仓库根目录的 `core.*` 与 `profile.html` 仍未跟踪，未纳入提交。
- `check_media_ledger.py` 改动前退出 0（DEV 80/128，108 条 confirmed），改动后退出 0（DEV 81/128，110 条 confirmed）。

## 任务选择

- 队列顶部 23 条 `排队` 任务全部是知乎；可见 Chrome `/creator` 仍落 `/signin?next=%2Fcreator`，平台级 `阻塞`，未消耗额度。
- 掘金 `/creator/home` 这次渲染出创作者中心但**仍未登录**（导航为 登录/注册，粉丝/展现/阅读全 0）；Medium `/me/stories` 仍 `Sign up`/`Sign in` 且无草稿，`MEDIUM_API_KEY` 仍不存在；LinkedIn `/in/yunwei37` 落 `/authwall`；DEV web 会话也掉登录，secret-backed workflow 仍是唯一 DEV 通道。X `@yunwei37` 已登录（1,766 posts）。
- 按"历史回补默认只处理 tutorial"取教程序列中最早的 DEV 缺口 `24-hide`（`devto.json` 明细行与 `confirmed_source_urls` 都没有它）。发布前用 DEV 公开 API 复查账号 86 篇、精确标题 0 篇、标题含 `Hiding` 0 篇；站点 `/tutorials/24-hide/` 与 `/zh/tutorials/24-hide/` 均 200。

## 准备稿

`draft/media/2026-11-27/24-hide/devto.md` 只修两处英文源文缺陷：

- `README.md:368` 把 `and then print the corresponding message.` 重复成独立一段（上一段 `:366` 已以同一句结尾）；中文 `README.zh.md:364` 只说一次。线上站点同样连续出现两次。
- `:384` 把仓库链接写成 `https: [//github.com/...`，站点显示为 `Full source code: https: //github.com/...`（协议与 URL 被拆开）；中文 `README.zh.md:380` 是干净的 `<https://...>`。

其余逐字节保留：14 个围栏（9 `c`、1 `bash`、1 `sh`、3 `console`）全部带语言标签、0 图片、0 表格、0 相对链接、0 脚注标记。上游 `bpf-developer-tutorial` 的 `src/24-hide/README.md` 与本地英文源逐字节相同（21,738 字节），两处缺陷同样在上游。

结构（准备稿）：1 个 H1（精确标题）、4 个 H2、1 个 H3、0 个 H4、14 个代码块。

## 发布事实

- 路径：`.github/publisher/posts_queue.txt`（1 条，字段 `path`/`tags`）-> commit `44cc0a35384` 推送 `main` -> `gh workflow run publish-posts.yml -f platforms=devto -f publish_count=1 -f draft_only=false`。run 45915263748 在 `2026-11-27T00:14:08Z` 公开：<https://dev.to/yunwei37/ebpf-practical-tutorial-hiding-process-or-file-information-2f6n>，article id 5128109，tag `ebpf`/`linux`/`tutorial`/`kernel`，reading_time 12，0 反应 0 评论。workflow 随后自行推送 `1fd5cf7` 清空队列。
- 正文对账：公开 `body_markdown` 与准备稿去首个 H1 后**逐字相同**（21,551 字符；仅首尾空行差异，`trim()` 后完全一致）。
- 结构对账（可见页 DOM）：1 个精确标题 h1、4 个正文 H2（`Background Knowledge and Implementation Mechanism`、`User-Style eBPF Programming`、`Compile and Run`、`Summary`）、1 个 H3、0 个 H4、14 个 `pre`（9 `c` + 2 `shell` + 3 `console`；DEV 把 `bash` 与 `sh` 渲染为 `shell` 并保留标签）、0 正文图片、0 表格、0 泄漏围栏、4 个 tag 芯片、`Top comments (0)`、结尾完整（`The original link of this article: https://eunomia.dev/tutorials/24-hide`）、修好的 `Full source code:` 段落只有一个可用链接。
- 额度：发布时刻 `2026-11-27T00:14:08Z` = LA `2026-11-26 16:14 PST`，用掉 **LA 11-26 正常额度**；补发缺口仍 0。

## 观测

- 运行延迟：dispatch 后约 2 分钟起 job，job 本身 12 秒。
- 列表滞后：本次未复现——检查时公开作者 API 返回 88 篇，同时包含 `5128109` 与前一天 `5120170`（11-23..11-26 连续四天滞后）。仍沿用"per-article 端点与公开页才是发布证据"。
- X `2122736534206181608`（Lean 证明检查成本）1,869 浏览 / 18 赞 / 4 转帖 / 8 书签 / 2 回复（上一快照 1,858/18/4/8/2）；两条回复仍是观点与感叹，不需回。`2122011756633665955`（bpf-gpu-server）1,007 浏览 / 30 赞 / 9 书签 / 1 回复（1,006/30/9/1），唯一回复仍是 `@sunilmehta_si` 11-16 的复述。账号 1,766 posts。
- 站点原文：`/tutorials/24-hide/` 本身带上面两处缺陷（`and then print the corresponding message.` 连续出现、`Full source code: https: //github.com/...`），中文页无此问题。三处已知缺陷仍未修：`/tutorials/19-lsm-connect/` 的 `16843009".` 与内联围栏、`/tutorials/17-biopattern/` 重复的 `compatibility issues`、`/tutorials/15-javagc/` 的 `problem.In Linux`。
- HN 无新回声：`search_by_date` 里带 `eunomia` 的最新故事仍是 50385391（AgentSight，2026-11-05），11-14..11-25 的命中均为无关故事。

## 台账同步

`platforms/devto.json` 新增 `devto-2f6n` 行与 `confirmed_source_urls` 映射（id 用 slug 尾段，避免与既有 `devto-*` 冲突），`last_checked` 提到 2026-11-27，`browser_observations` 追加一条，`pending_verification` 的自 canonical 计数改为二十五并纳入 `devto-2f6n`，同时把 24-hide 的两处英文源缺陷并入英文源缺陷清单；`sources.json` 同步日期；`published.md`（LF）加行、刷新日期并补 11-27 的列表观察；`not-published.md`（CRLF）加 11-27 段并更正"Medium 副本早已确认"的错误——Medium 的 11 篇 feed 里没有该源，Medium 仍因无 `MEDIUM_API_KEY` 且会话掉登录而不可发布。`check_media_ledger.py` 退出 0。

## 遗留

- canonical 仍为 DEV 自身（连续第二十五次）。根因仍在上游 `media-publisher`：`publish-multi` 从 `title/content/tags/is_draft/platforms` 重建 payload，`canonical_url` 到不了 DEV 分支。
- 站点原文缺陷同前：19/17/15 未修，24-hide 的两处也未修。`docs/tutorials/**` 由上游同步，本地无该仓库，按 0/1/3/4/8/11/13/14/15/17/19/20/22 的先例只修准备稿。
- omp headless 浏览器 daemon 起不来，只能 attach 已经运行的 CDP 9222；若 supervisor 再停，浏览器工作会整体不可用。

## 下一次具体动作

1. 恢复知乎登录后，从 `53-egress-pacer` 起继续处理 23 条知乎任务（掘金恢复登录后可作备用发布通道）。
2. 若 DEV 继续日更：下一候选 `28-detach`（`devto.json` 无该行，DEV 账号 88 篇里也没有副本；剩余未映射英文 tutorial 还有 `29-sockops`/`37-uprobe-rust`/`40-mysql`/`49-hid`/`bpftrace-tutorial`，`18-further-reading` 是目录页）。发布前同样核对上游英文源文的围栏/标点缺陷，并对照中文 `README.zh.md`。
3. 下次巡检开工先探 CDP 9222；attach 用 `app: { cdp_url: "http://127.0.0.1:9222" }`，并在 `tab.run` 里用 `page.evaluate`。