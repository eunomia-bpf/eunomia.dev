# 2026-11-23 内容巡检运行日志

运行日按 +08 计（UTC `2026-11-23T00:00Z` 起，LA 2026-11-22 16:00 PST）。本次巡检完成**一条 DEV 发布**；队列里 23 条知乎任务仍是平台级 `阻塞`。补发缺口 0。

## 运行前状态

- CDP 9222 正常（`Chrome/146.0.7680.31`），无需恢复；supervisor 日志末次记录为 2026-11-20 08:03 的就绪行。
- 本地 `main` 落后 `origin/main` 44 个提交（daily data brief / SEO / 报告索引），已 rebase 后再动手。
- `check_media_ledger.py` 在本轮改动前退出 0。
- 仓库根目录 13 个 `core.*` core dump 与 `profile.html` 仍未跟踪，未纳入提交。

## 任务选择

- 队列顶部 23 条 `排队` 任务全部是知乎；可见 Chrome 上 `https://www.zhihu.com/creator` 落 `https://www.zhihu.com/signin?next=%2Fcreator`，cookie 只有 `_zap`/`d_c0`/`__snaker__id`/`_xsrf`/`Hm_lvt_*`/`HMACCOUNT`/`BEC`/`gdxidpyhxdE` 等匿名标识，无 `z_c0`，属平台级 `阻塞`，未消耗额度。
- 掘金仍**掉登录**：`https://juejin.cn/creator/home` 落回 `https://juejin.cn/`，只剩 `_tea_utm_cache_2608`/`s_v_web_id`/`__tea_cookie_tokens_2608`。队列中掘金任务为 0，无影响。
- Medium 仍无 `MEDIUM_API_KEY`（shell 与 `.env` 都没有）；可见 Chrome `/me/stories` 为未登录的 `Sign up`/`Sign in`，Stories 为空。
- X `@yunwei37` 已登录（1,758 posts）；LinkedIn `/in/yunwei37` 仍落 `/authwall`。
- 按"历史回补默认只处理 tutorial"取教程序列中最早的 DEV 缺口 `17-biopattern`：`platforms/devto.json` 的明细行与 `confirmed_source_urls` 都没有它，Medium 副本早已 `confirmed`。发布前用 DEV 公开 API 复查账号 82 篇、精确标题 0 篇、`Tutorial by Example 17` 前缀 0 篇、标题含 `Disk I/O` 0 篇；站点原文 `/tutorials/17-biopattern/` 与 `/zh/tutorials/17-biopattern/` 均 200。

## 准备稿

`draft/media/2026-11-23/17-biopattern/devto.md` 与源文只有四处差异，全部是英文源文缺陷，中文 `README.zh.md` 四处都干净：

1. `:145` 的 `counters` map 片段以 `.` 结尾；准备稿改为 `;`（中文 `:145` 有分号）。
2. `:206` 把 "This change may cause compatibility issues with eBPF programs on different versions of the kernel." 在同一段里重复两遍；准备稿只留一遍（中文 `:203` 只说一次）。
3. `:202-204` 把 `Statistics update` 下四条子项中的三条退回父级列表；准备稿恢复嵌套（中文 `:198-201` 四条都嵌套）。
4. `:195`/`:197`/`:203`/`:226`/`:263`/`:265`/`:322` 七处把行内代码跨度闭合在括号之后（`` `(filter_dev` is `true` ) ``）；准备稿改成 ``(`filter_dev` is `true`)``（中文 `:196` 等处干净）。

结构（准备稿）：1 个 H1（精确标题）、4 个正文 H2、1 个 H3、22 行首围栏 = 11 个代码块（`bash` x2、`c` x8、`console` x1）、0 图片、0 表格、0 处残留内联围栏。

## 发布事实

- 路径：`.github/publisher/posts_queue.txt`（1 条，字段 `path`/`tags`）-> commit `14c208fb31d` 推送 `main` -> `gh workflow run publish-posts.yml -f platforms=devto -f publish_count=1 -f draft_only=false`。run 45087813444 在 `2026-11-23T00:17:04Z` 公开：<https://dev.to/yunwei37/ebpf-tutorial-by-example-17-count-randomsequential-disk-io-11e7>，article id 5096382，tag `ebpf`/`linux`/`tutorial`/`kernel`，reading_time 9，0 反应 0 评论。workflow 随后自行推送 `372f723406d` 清空队列。
- 正文对账：公开 `body_markdown` 与准备稿去首个 H1 后逐字相同，只差 DEV 截掉的结尾换行（16,768 / 16,769 字符）。
- 结构对账（可见页 DOM）：1 h1（精确标题）、4 个正文 H2（`Random/Sequential Disk I/O`、`Biopattern`、`eBPF Biopattern Implementation Principles`、`Summary`）、1 个 H3（`User State Code`）、11 个 `pre`（`highlight shell` x2、`highlight c` x8、`highlight console` x1）、0 处泄漏的反引号、0 图片、0 表格、正文 16,247 字符、结尾完整、tag 芯片 4 个、`Top comments (0)`。
- 四处修补在公开页一一复核：map 片段带分号、重复句只剩一遍、四条子项为嵌套 `li`、七处行内代码跨度渲染为 `(filter_dev is true)` 等无字面反引号。
- 覆盖：DEV 76 -> 77/128，`check_media_ledger.py` 退出 0（102 条 confirmed，0 条 confirmed-without-source_path）。
- 额度：发布时刻 `2026-11-23T00:17:04Z` = LA `2026-11-22 16:17 PST`，用掉 **LA 11-22 正常额度**（LA 11-22 当日此前无其它发布）；补发缺口仍 0。

## 观测

- 运行延迟：`workflow_dispatch` 后约 2 分钟起 job，整条 28 秒（11-22 那次 21 秒）。
- 列表滞后复现：发布后约 20 分钟，公开作者 API 返回 82 篇，**既不含** `5096382` 也**不含**前一天的 `5090916`，而两者的 per-article 端点与公开页均 200。11-22 那次"列表已包含新发布"只维持一天，继续沿用"不要用列表判断发布结果"。
- X `2122736534206181608`（Lean 证明检查成本）1,817 浏览 / 18 赞 / 4 转帖 / 8 书签 / 2 回复（上一快照 1,812/18/4/8/2）；两条回复仍是 `@difficultyang` 的观点与 `@sk_196883` 的感叹，不需回。`2122011756633665955`（bpf-gpu-server）991 浏览 / 30 赞 / 9 书签 / 1 回复（990/30/9/1），唯一回复仍是 `@sunilmehta_si` 11-16 的复述。
- 站点原文：live `/tutorials/15-javagc/` 仍 5 个正文 H2（缺 `Summary`），与 11-22 记录一致；`/tutorials/17-biopattern/` 仍 1 H1 / 4 H2 / 1 H3 / 11 `pre`，四处源文缺陷均在渲染里可见。
- HN 无新回声：`search_by_date` 最新的 eunomia 故事仍是 50385391（AgentSight，2026-11-05，2 分 0 评论）；`Count Random/Sequential Disk I/O` 精确标题搜索 0 命中。DEV 11-09..11-23 的副本全部 0 反应 0 评论。

## 遗留

- canonical 仍为 DEV 自身（连续第二十一次）。根因仍在上游 `media-publisher`：`publish-multi` 从 `title/content/tags/is_draft/platforms` 重建 payload，`canonical_url` 到不了 DEV 分支。
- 站点原文 `https://eunomia.dev/tutorials/17-biopattern/` 仍渲染上述四处源文缺陷。上游 `eunomia-bpf/bpf-developer-tutorial` 的 `src/17-biopattern/README.md` 需同一处修补；`docs/tutorials/**` 由上游同步，本地无该仓库，按 0/1/3/4/8/11/13/14/15 的先例只修准备稿。

## 下一次具体动作

1. 恢复知乎登录后，从 `53-egress-pacer` 起继续处理 23 条知乎任务（掘金恢复登录后可作备用发布通道）。
2. 若 DEV 继续日更：下一候选 `19-lsm-connect`（按精确标题在 DEV 账号 82 篇里没有副本；`18-further-reading` 是目录页）。发布前同样要核对上游英文源文的围栏/标点缺陷，并同步核对中文源文。