# 2026-11-24 内容巡检运行日志

运行日按 +08 计（UTC `2026-11-24T00:00Z` 起，LA 2026-11-23 16:00 PST）。本次巡检完成**一条 DEV 发布**；队列里 23 条知乎任务仍是平台级 `阻塞`。补发缺口 0。

## 运行前状态

- CDP 9222 正常（`Chrome/146.0.7680.31`），无需恢复。注意：omp 自带 browser daemon 起不来（`Shared browser daemon unavailable`），本轮全部浏览器操作都通过 `app.cdp_url: http://127.0.0.1:9222` 接管持久可见 Chrome。
- 本地 `main` 落后 `origin/main` 3 个提交（daily data brief / weekly org report / ebpf-qa probe），rebase 后再动手；推送时又赶上 `3fa10147885`（agentsight docs sync），再 rebase 一次后推送。
- `check_media_ledger.py` 在本轮改动前退出 0；改动后仍退出 0（DEV 78/128，104 条 confirmed）。
- 仓库根目录 core dump 与 `profile.html` 仍未跟踪，未纳入提交。

## 任务选择

- 队列顶部 23 条 `排队` 任务全部是知乎；可见 Chrome `/creator` 仍落 `/signin?next=%2Fcreator`（匿名 cookie、无 `z_c0`），平台级 `阻塞`，未消耗额度。
- 掘金 `/creator/home` 仍掉登录（落回 `https://juejin.cn/`，导航为 登录/注册）；Medium `/me/stories` 仍 `Sign up`/`Sign In` 且 Stories 为空，shell 与 `.env` 都没有 `MEDIUM_API_KEY`；LinkedIn `/in/yunwei37` 仍落 `/authwall`；DEV 自身 web 会话也掉登录（`/dashboard` 落 `/magic_links/new`），因此 secret-backed workflow 仍是唯一 DEV 通道。
- 按"历史回补默认只处理 tutorial"取教程序列中最早的 DEV 缺口 `19-lsm-connect`：`devto.json` 明细行与 `confirmed_source_urls` 都没有它（`18-further-reading` 是目录页）。发布前用 DEV 公开 API 复查账号 84 篇、精确标题 0 篇、`Tutorial by Example 19` 前缀 0 篇、标题含 `LSM` 0 篇；站点 `/tutorials/19-lsm-connect/` 与 `/zh/tutorials/19-lsm-connect/` 均 200。

## 准备稿

`draft/media/2026-11-24/19-lsm-connect/devto.md` 与源文只差一行：`README.md:154` 在 `lsm: blocking 16843009` 后多出收尾引号 `".` 和内联围栏，提前关闭 `console` 块，使其后两行 `wget-7061` 输出落到正文；准备稿删掉多余引号与围栏（中文 `README.zh.md:153-156` 干净）。文件其余部分逐字节相同；上游 `bpf-developer-tutorial` 的 `src/19-lsm-connect/README.md` 与本地英文源 `diff` 为空。

结构（准备稿）：1 个 H1（精确标题）、7 个 H2、0 个 H3、18 行围栏 = 9 个代码块（`console` x6、`conf` x1、`C` x1、`shell` x1）、0 图片、0 表格、0 相对链接。

## 发布事实

- 路径：`.github/publisher/posts_queue.txt`（1 条，字段 `path`/`tags`）-> commit `2f4cccd3692` 推送 `main` -> `gh workflow run publish-posts.yml -f platforms=devto -f publish_count=1 -f draft_only=false`。run 45320596815 在 `2026-11-24T03:36:01Z` 公开：<https://dev.to/yunwei37/ebpf-tutorial-by-example-19-security-detection-and-defense-using-lsm-321l>，article id 5104605，tag `ebpf`/`linux`/`tutorial`/`kernel`，reading_time 5，0 反应 0 评论。workflow 随后自行推送 `7e061fcd1d7` 清空队列。
- 正文对账：公开 `body_markdown` 与准备稿去首个 H1 后**逐字相同**（8,000 / 8,000 字符，本轮无截断差）。
- 结构对账（可见页 DOM）：1 h1（精确标题；另有 DEV 站点自身的 "Want to make DEV your full-time job?" 促销 h1）、7 个正文 H2、0 个 H3、9 个 `pre`（`highlight console` x6、`highlight conf`、`highlight c`、`highlight shell`，语言标签全部保留）、0 处泄漏反引号、0 图片、0 表格、正文 7,618 字符、结尾完整（`The original link of this article: …/19-lsm-connect`）、4 个 tag 芯片、`Top comments (0)`。
- 修饰复核：修补处公开页显示 `lsm: blocking 16843009`（无 `".`），两行 `wget-7061` 完整位于 `console` 块内。
- 额度：发布时刻 `2026-11-24T03:36:01Z` = LA `2026-11-23 19:36 PST`，用掉 **LA 11-23 正常额度**（LA 11-23 当日此前无其它发布；11-23 那次发布用的是 LA 11-22 额度）；补发缺口仍 0。

## 观测

- 运行延迟：dispatch 后约 20 秒起 job，整条 21 秒（11-23 那次 28 秒）。
- 列表滞后：发布后数分钟，公开作者 API 返回 84 篇，**不含**新发布的 `5104605`，却含 `5096382` 与 `5090916`；per-article 端点与公开页均 200。继续沿用"不要用列表判断发布结果"。
- X `2122736534206181608`（Lean 证明检查成本）1,828 浏览 / 18 赞 / 4 转帖 / 8 书签 / 2 回复（上一快照 1,817/18/4/8/2）；两条回复仍是 `@difficultyang` 的观点与 `@sk_196883` 的感叹，不需回。`2122011756633665955`（bpf-gpu-server）997 浏览 / 30 赞 / 9 书签 / 1 回复（991/30/9/1），唯一回复仍是 `@sunilmehta_si` 11-16 的复述。
- 站点原文（三处都仍未修）：live `/tutorials/19-lsm-connect/` 仍渲染 `lsm: blocking 16843009".``` ` 并把两行 `wget` 输出挤出代码块；`/tutorials/17-biopattern/` 仍渲染 `} counters SEC(".maps").` 与重复两遍的 "This change may cause compatibility issues…"；`/tutorials/15-javagc/` 仍渲染 `problem.In Linux` 与 `Summary.Through this introductory eBPF tutorial …`。
- HN 无新回声：`search_by_date` 最新的 eunomia 故事仍是 50385391（AgentSight，2026-11-05，2 分 0 评论）；精确标题 `Security Detection and Defense using LSM` 与 `lsm-connect` 均 0 命中。DEV 11-09..11-24 的副本全部 0 反应 0 评论。
- 台账同步：`platforms/devto.json` 新增 `devto-321l` 行与 `confirmed_source_urls` 映射，`last_checked` 提到 2026-11-24；`sources.json` 同步日期；`published.md`（LF）加行并刷新日期；`not-published.md`（CRLF）加 11-24 段，并把滞后的 DEV 状态行（仍写 11-16 的 tutorial 7）更新为本次发布。`check_media_ledger.py` 退出 0。

## 遗留

- canonical 仍为 DEV 自身（连续第二十二次）。根因仍在上游 `media-publisher`：`publish-multi` 从 `title/content/tags/is_draft/platforms` 重建 payload，`canonical_url` 到不了 DEV 分支。
- 站点原文缺陷同上三处。上游 `eunomia-bpf/bpf-developer-tutorial` 的 `src/19-lsm-connect/README.md` 需同一处修补；`docs/tutorials/**` 由上游同步，本地无该仓库，按 0/1/3/4/8/11/13/14/15/17 的先例只修准备稿。

## 下一次具体动作

1. 恢复知乎登录后，从 `53-egress-pacer` 起继续处理 23 条知乎任务（掘金恢复登录后可作备用发布通道）。
2. 若 DEV 继续日更：下一候选 `20-tc`（`devto.json` 无该行，DEV 账号 84 篇里也没有副本；`18-further-reading` 是目录页）。发布前同样核对上游英文源文的围栏/标点缺陷，并对照中文 `README.zh.md`。