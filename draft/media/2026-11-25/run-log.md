# 2026-11-25 内容巡检运行日志

运行日按 +08 计（UTC `2026-11-25T00:00Z` 起，LA 2026-11-24 16:00 PST）。本次巡检完成**一条 DEV 发布**；队列里 23 条知乎任务仍是平台级 `阻塞`。补发缺口 0。

## 运行前状态

- CDP 9222 正常（`Chrome/146.0.7680.31`），无需恢复；本轮浏览器操作全部通过 `app.cdp_url: http://127.0.0.1:9222` 接管持久可见 Chrome（omp 自带 browser daemon 仍未起）。
- 本地 `main` 落后 `origin/main` 3 个提交，rebase 后再动手；推送前再次 fetch 确认无新提交。
- 仓库根目录 core dump（14 个 `core.*`）与 `profile.html` 仍未跟踪，未纳入提交。
- `check_media_ledger.py` 改动前退出 0，改动后仍退出 0（DEV 79/128，106 条 confirmed）。

## 任务选择

- 队列顶部 23 条 `排队` 任务全部是知乎；可见 Chrome `/creator` 仍落 `/signin?next=%2Fcreator`，平台级 `阻塞`，未消耗额度。
- 掘金 `/creator/home` 仍掉登录（落回 `https://juejin.cn/`，导航为 登录/注册）；Medium `/me/stories` 仍 `Sign up`/`Sign in` 且无 `MEDIUM_API_KEY`；LinkedIn `/in/yunwei37` 落注册页；DEV web 会话也掉登录（`/dashboard` 落 `/magic_links/new`），secret-backed workflow 仍是唯一 DEV 通道。
- 按"历史回补默认只处理 tutorial"取教程序列中最早的 DEV 缺口 `20-tc`：`devto.json` 明细行与 `confirmed_source_urls` 都没有它（`18-further-reading` 是目录页，`0-introduce` 与 `3-fentry-unlink` 早有公开副本并已在明细行里）。发布前用 DEV 公开 API 复查账号 84 篇、精确标题 0 篇、`Example 20` 前缀 0 篇、标题含 `Traffic Control` 0 篇；站点 `/tutorials/20-tc/` 与 `/zh/tutorials/20-tc/` 均 200。

## 准备稿

`draft/media/2026-11-25/20-tc/devto.md` 是 `docs/tutorials/20-tc/README.md` 的**逐字节副本**（6,238 字节，sha256 相同）。该英文源文无围栏/标点缺陷、无图片、无表格、无相对链接，唯一的链接是绝对地址，6 个围栏自带 `c`/`console`/`shell` 标签，因此无需修补；上游 `bpf-developer-tutorial` 的 `src/20-tc/README.md` 与本地文件 `diff` 为空。

结构（准备稿）：1 个 H1（精确标题）、5 个 H2、0 个 H3、12 行围栏 = 6 个代码块（`c` x2、`console` x3、`shell` x1）、0 图片、0 表格。

## 发布事实

- 路径：`.github/publisher/posts_queue.txt`（1 条，字段 `path`/`tags`）-> commit `aa9cf7cbafb` 推送 `main` -> `gh workflow run publish-posts.yml -f platforms=devto -f publish_count=1 -f draft_only=false`。run 45516590397 在 `2026-11-25T00:24:40Z` 公开：<https://dev.to/yunwei37/ebpf-tutorial-by-example-20-tc-traffic-control-224e>，article id 5112060，tag `ebpf`/`linux`/`tutorial`/`kernel`，reading_time 4，0 反应 0 评论。workflow 随后自行推送 `01b513b` 清空队列。
- 正文对账：公开 `body_markdown` 与准备稿去首个 H1 后**逐字相同**（6,185 / 6,185 字符）。
- 结构对账（可见页 DOM）：1 个精确标题 h1、5 个正文 H2（`Overview of tc`、`Writing eBPF Programs`、`Compilation and Execution`、`Summary`、`References`）、0 个 H3、6 个 `pre`（`highlight c` x2、`highlight console` x3、`highlight shell` x1，语言标签全部保留）、0 泄漏反引号、0 图片、0 表格、正文 5,977 字符、结尾完整（`The original link of this article: …/20-tc`）、4 个 tag 芯片、`Top comments (0)`。
- 额度：发布时刻 `2026-11-25T00:24:40Z` = LA `2026-11-24 16:24 PST`，用掉 **LA 11-24 正常额度**；补发缺口仍 0。

## 观测

- 运行延迟：dispatch 后 run 排队约 7 分钟才起 job（11-22、11-23、11-24 都在 20 秒内起 job），job 本身 21 秒。
- 列表滞后：发布后数分钟，公开作者 API 返回 84 篇，既不含 `5112060` 也不含 `5104605`，而 per-article 端点与公开页均 200。继续沿用"不要用列表判断发布结果"。
- X `2122736534206181608`（Lean 证明检查成本）1,847 浏览 / 18 赞 / 4 转帖 / 8 书签 / 2 回复（上一快照 1,828/18/4/8/2）；两条回复仍是观点与感叹，不需回。`2122011756633665955`（bpf-gpu-server）1,002 浏览 / 30 赞 / 9 书签 / 1 回复（997/30/9/1），唯一回复仍是 `@sunilmehta_si` 11-16 的复述。
- 站点原文：`/tutorials/20-tc/` 本身干净（5 H2、6 个代码块、语言标签保留）。三处已知缺陷仍未修：`/tutorials/19-lsm-connect/` 仍渲染 `16843009".``` ` 并把两行 `wget` 输出挤出代码块；`/tutorials/17-biopattern/` 仍在同一段重复 `compatibility issues` 两次；`/tutorials/15-javagc/` 仍渲染 `problem.In Linux` 与 `Summary.Through`。
- HN 无新回声：`search_by_date` 最新的 eunomia 故事仍是 50385391（AgentSight，2026-11-05，2 分 0 评论）；精确标题搜索无 tutorial 20 副本。DEV 11-09..11-25 的副本全部 0 反应 0 评论。
- 台账同步：`platforms/devto.json` 新增 `devto-224e` 行与 `confirmed_source_urls` 映射，`last_checked` 提到 2026-11-25，`pending_verification` 的自 canonical 条目计数改为二十二并加入 `devto-224e`；`sources.json` 同步日期；`published.md`（LF）加行、刷新日期并补 11-24/11-25 的列表滞后观察；`not-published.md`（CRLF）加 11-25 段并替换 DEV 状态行。`check_media_ledger.py` 退出 0。

## 遗留

- canonical 仍为 DEV 自身（连续第二十三次）。根因仍在上游 `media-publisher`：`publish-multi` 从 `title/content/tags/is_draft/platforms` 重建 payload，`canonical_url` 到不了 DEV 分支。
- 站点原文缺陷同前三条（19/17/15）。`docs/tutorials/**` 由上游同步，本地无该仓库，按 0/1/3/4/8/11/13/14/15/17/19 的先例只修准备稿。

## 下一次具体动作

1. 恢复知乎登录后，从 `53-egress-pacer` 起继续处理 23 条知乎任务（掘金恢复登录后可作备用发布通道）。
2. 若 DEV 继续日更：下一候选 `22-android`（`devto.json` 无该行，DEV 账号 84 篇里也没有副本；剩余未映射英文 tutorial 还有 `24-hide`/`28-detach`/`29-sockops`/`37-uprobe-rust`/`40-mysql`/`49-hid`/`bpftrace-tutorial`，`18-further-reading` 是目录页）。发布前同样核对上游英文源文的围栏/标点缺陷，并对照中文 `README.zh.md`。