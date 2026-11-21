# 2026-11-21 内容巡检运行日志

运行日按 +08 计（UTC `2026-11-21T00:00Z` 起，LA 2026-11-20 16:00 PST）。本次巡检完成**一条 DEV 发布**；队列里 23 条知乎任务仍是平台级 `阻塞`。补发缺口 0。

## 运行前状态

- CDP 9222 正常（`Chrome/146.0.7680.31`），无需恢复；本次 `browser.open()` 仍走 `app: { cdp_url: "http://127.0.0.1:9222" }` 依附可见 Chrome。
- 本地 `main` 落后 `origin/main` 4 个提交（含 `323f34bb489` 的 `bpf-developer-tutorial` 同步），已 rebase；同步后 `docs/tutorials/14-tcpstates/README.md` 未变，六处缺陷仍在。
- `check_media_ledger.py` 在本轮改动前退出 0（DEV 74/128、96 条 confirmed）。

## 任务选择

- 队列顶部 23 条 `排队` 任务全部是知乎；可见 Chrome 上 `https://www.zhihu.com/creator` 落 `https://www.zhihu.com/signin?next=%2Fcreator`，cookie 只有 `_zap`/`_d_c0`/`_xsrf`/`HMACCOUNT` 等匿名标识，无 `z_c0`，属平台级 `阻塞`，未消耗额度。
- 掘金仍**掉登录**：`https://juejin.cn/creator/home` 落回 `https://juejin.cn/`，只有 `_tea_utm_cache_2608`/`s_v_web_id`/`__tea_cookie_tokens_2608`。队列中掘金任务为 0，无影响。
- Medium 仍无 `MEDIUM_API_KEY`；可见 Chrome `/me/stories` 只有 `_ga`/`g_state`/`_dd_s`，页面为未登录的 `Sign up`/`Sign in`。
- X `@yunwei37` 已登录（1,757 posts）；LinkedIn `/in/yunwei37` 仍落 `/authwall`。
- 按"历史回补默认只处理 tutorial"取教程序列中最早的 DEV 缺口 `14-tcpstates`：`platforms/devto.json` 的明细行与 `confirmed_source_urls` 都没有它，Medium 副本早已 `confirmed`。发布前用 DEV 公开 API 复查账号 80 篇、精确标题 0 篇、`Tutorial by Example 14` 前缀 0 篇；站点原文 `/tutorials/14-tcpstates/` 与 `/zh/tutorials/14-tcpstates/` 均 200。

## 准备稿

`draft/media/2026-11-21/14-tcpstates/devto.md` 与源文只有六处差异，全部是英文源文的围栏/标点缺陷，中文 `README.zh.md` 六处都干净：

1. `:49-50` 把 `dports` map 后面的内联围栏写成 ```` ```__type(key, __u16); ````，`c` 块提前闭合，`dports` 整块渲染成正文；准备稿补回 `__type(key, __u16);` 一行（`README.zh.md:44-49` 是完整 map）。
2. `:176` 段落开头是多余的闭引反引号（`` handle_event` is a callback ``），准备稿改为开引。
3. `:252` 把独立的 `Instructions:` 挤进上一句 `…analyze the RTT data.`，准备稿断回两段。
4. `:303-304` 用 `".` / `"` 把两行 `tcpstates` 输出粘起，准备稿还原为两行。
5. `:351-352` 把 `Built with eunomia-bpf framework.".` 与闭合围栏粘成一行，围栏里漏出 `See https://…`；准备稿还原为 `framework.` 与独立 `See https://…` 行。
6. `:387` 把 `256 -> 511` 与 `512 -> 1023` 两行 histogram 粘成一行，准备稿拆开。

结构（准备稿）：1 个 H1（精确标题）、5 个正文 H2、4 个 H3、18 行首围栏 = 9 个代码块（`sh`、`c` x4、`console` x3、`shell` x1）、0 图片、0 表格、0 处残留内联围栏。

## 发布事实

- 路径：`.github/publisher/posts_queue.txt`（1 条，字段 `path`/`tags`）-> commit `ba808ed769b` 推送 `main` -> `gh workflow run publish-posts.yml -f platforms=devto -f publish_count=1 -f draft_only=false`。run 44843640872 在 `2026-11-21T00:25:03Z` 公开：<https://dev.to/yunwei37/ebpf-tutorial-by-example-14-recording-tcp-connection-status-and-tcp-rtt-4ggl>，article id 5084990，tag `ebpf`/`linux`/`tutorial`/`kernel`，reading_time 11，0 反应 0 评论。workflow 随后自行推送 `62a1ea316fa` 清空队列。
- 正文对账：公开 `body_markdown` 与准备稿去首个 H1 后**逐字相同**（21,856 字符）。
- 结构对账（可见页 DOM，整页滚动）：1 h1（精确标题）、5 个正文 H2（`tcprtt and tcpstates`、`tcpstate eBPF code`、`User-Space Processing for tcpstate`、`Compilation and Execution`、`Summary`）、4 个 H3（`Define BPF Maps`、`Trace TCP Connection State Changes`、`Update Timestamps`、`tcprtt kernel eBPF code`）、9 个 `pre`（`highlight shell` x2、`highlight c` x4、`highlight console` x3）、0 处泄漏的反引号、0 图片、0 表格、正文 21,352 字符、结尾完整（含 `eunomia.dev/tutorials/14-tcpstates` 尾注）、tag 芯片 4 个、0 评论。
- 六处修补在公开页一一复核：`dports` 的 `__type(key` 行 3 条齐全、`handle_event` 段无游离反引号、`data.` 与 `Instructions:` 已分段、`225.794".` 消失、`See https://…` 在 help 块内、`512 -> 1023` 为独立一行。
- 外链：准备稿 5 个外链目标（`bpf-developer-tutorial/tree/main/src/14-tcpstates`、bcc 的 `tcpstates_example.txt`、`tcprtt.py`、`libbpf-tools/tcpstates.bpf.c`、教程仓库）全部 200。
- 覆盖：DEV 74 -> 75/128，`check_media_ledger.py` 退出 0（98 条 confirmed，0 条 confirmed-without-source_path）。
- 额度：发布时刻 `2026-11-21T00:25:03Z` = LA `2026-11-20 16:25 PST`，用掉 **LA 11-20 正常额度**（LA 11-20 当日此前无其它发布）；补发缺口仍 0。

## 观测

- 运行延迟：dispatch 后 job 于 `00:24:52Z` 启动，整条约 30 秒（11-20 那次 19 秒）。
- 列表滞后复现：发布后公开作者 API 返回 81 篇但**不含** `5084990`，而 per-article 端点与公开页已 200。再次确认"不要用列表判断发布结果"。
- X `2122736534206181608`（Lean 证明检查成本）1,801 浏览 / 18 赞 / 4 转帖 / 8 书签 / 2 回复（上一快照 1,785/18/4/8/2）；两条回复仍是 `@difficultyang` 的观点与 `@sk_196883` 的感叹，不需回。`2122011756633665955`（bpf-gpu-server）987 浏览 / 30 赞 / 10 书签 / 1 回复（983/30/10/1），唯一回复仍是 `@sunilmehta_si` 11-16 的复述。
- HN 无新回声：`search_by_date` 最新的 eunomia 故事仍是 50385391（AgentSight，2026-11-05，2 分 0 评论）；tutorial 14 精确标题搜索 0 命中。DEV 11-09..11-21 的副本全部 0 反应 0 评论（含昨日 tutorial 13）。

## 遗留

- canonical 仍为 DEV 自身（连续第十九次）。根因仍在上游 `media-publisher`：`publish-multi` 从 `title/content/tags/is_draft/platforms` 重建 payload，`canonical_url` 到不了 DEV 分支。
- 站点原文 `https://eunomia.dev/tutorials/14-tcpstates/` 仍渲染全部六处源文缺陷（`} dports SEC(".maps");` 成正文并漏出 2 个围栏、`225.794".`、`framework."`、`|512 -> 1023` 粘连、`handle_event` 段缺开引反引号、`data.Instructions:` 粘连）。上游 `eunomia-bpf/bpf-developer-tutorial` 的 `src/14-tcpstates/README.md` 与本地英文源文 `diff` 为空（410 行），两边需同一处上游修补；`docs/tutorials/**` 由上游同步，本地无该仓库，按 0/1/3/4/8/11/13 的先例只修准备稿。
- `install-agent-clis.sh` 的 `runuser` PATH 缺口仍未回改镜像；本次浏览器已就绪，未触及。
- 仓库根目录累计 13 个 `core.*` core dump 与一个 `profile.html`，均未跟踪；本次未纳入提交。

## 下一次具体动作

1. 恢复知乎登录后，从 `53-egress-pacer` 起继续处理 23 条知乎任务（掘金恢复登录后可作备用发布通道）。
2. 若 DEV 继续日更：下一候选 `15-javagc`（按精确标题在 DEV 账号 81 篇里没有副本；`16-memleak` 已在 `confirmed_source_urls` 里算覆盖）。发布前同样要核对上游英文源文的围栏/标点缺陷，并同步核对中文源文。