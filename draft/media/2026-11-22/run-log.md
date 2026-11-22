# 2026-11-22 内容巡检运行日志

运行日按 +08 计（UTC `2026-11-22T00:00Z` 起，LA 2026-11-21 16:00 PST）。本次巡检完成**一条 DEV 发布**；队列里 23 条知乎任务仍是平台级 `阻塞`。补发缺口 0。

## 运行前状态

- CDP 9222 正常（`Chrome/146.0.7680.31`），无需恢复；浏览器操作依附可见 Chrome。
- 本地 `main` 落后 `origin/main` 6 个提交（agent-skills 子模块指针、`docs/ebpf-qa/` 两篇、daily data brief），已 rebase；同步后 `docs/tutorials/15-javagc/README.md` 未变，四处缺陷仍在。
- `check_media_ledger.py` 在本轮改动前退出 0（DEV 75/128、98 条 confirmed）。
- 仓库根目录 13 个 `core.*` core dump 与 `profile.html` 仍未跟踪，未纳入提交。

## 任务选择

- 队列顶部 23 条 `排队` 任务全部是知乎；可见 Chrome 上 `https://www.zhihu.com/creator` 落 `https://www.zhihu.com/signin?next=%2Fcreator`，cookie 只有 `_zap`/`d_c0`/`_xsrf`/`HMACCOUNT`/`captcha_session_v2` 等匿名标识，无 `z_c0`，属平台级 `阻塞`，未消耗额度。
- 掘金仍**掉登录**：`https://juejin.cn/creator/home` 落回 `https://juejin.cn/`，只有 `_tea_utm_cache_2608`/`s_v_web_id`/`__tea_cookie_tokens_2608`。队列中掘金任务为 0，无影响。
- Medium 仍无 `MEDIUM_API_KEY`（shell 与 `.env` 都没有）；可见 Chrome `/me/stories` 页面为未登录的 `Sign up`/`Sign in`，Stories 为空。
- X `@yunwei37` 已登录（1,758 posts）；LinkedIn `/in/yunwei37` 仍落 `/authwall`。
- 按"历史回补默认只处理 tutorial"取教程序列中最早的 DEV 缺口 `15-javagc`：`platforms/devto.json` 的明细行与 `confirmed_source_urls` 都没有它，Medium 副本早已 `confirmed`。发布前用 DEV 公开 API 复查账号 82 篇、精确标题 0 篇、`Tutorial by Example 15` 前缀 0 篇、标题含 `Java`/`GC` 0 篇；站点原文 `/tutorials/15-javagc/` 与 `/zh/tutorials/15-javagc/` 均 200。

## 准备稿

`draft/media/2026-11-22/15-javagc/devto.md` 与源文只有四处差异，全部是英文源文缺陷，中文 `README.zh.md` 四处都干净：

1. `:47` 把 `…solve the problem.` 与 `In Linux, USDT …` 挤成一段；准备稿断回两段（`README.zh.md:46` 同样是两段）。
2. `:105` 的 `perf_map` 定义写成 `__uint(type, BPF_MAP_TYPE_PERF_EVENT_ARRAY)`，少了分号；准备稿补回（中文 `:105` 有）。
3. `:185` 把 ``…and `gc_end`, respectively.`` 与 ``Finally, we have a `LICENSE` array…`` 挤成一段；准备稿拆成两段（中文 `:185-187` 是两段）。
4. `:317` 丢了 `## Summary` 标题，正文渲染成 `Summary.Through this introductory eBPF tutorial …`；准备稿还原为 `## Summary` + 正文（中文 `:319` 是 `## 总结`）。

结构（准备稿）：1 个 H1（精确标题）、6 个正文 H2、3 个 H3、20 行首围栏 = 10 个代码块（`bash` x3、`c` x4、`shell` x2、`console` x1）、0 图片、0 表格、0 处残留内联围栏；六个外链目标全部 200。

## 发布事实

- 路径：`.github/publisher/posts_queue.txt`（1 条，字段 `path`/`tags`）-> commit `a5a5360964a` 推送 `main` -> `gh workflow run publish-posts.yml -f platforms=devto -f publish_count=1 -f draft_only=false`。run 44965605572 在 `2026-11-22T00:14:19Z` 公开：<https://dev.to/yunwei37/ebpf-tutorial-by-example-15-capturing-user-space-java-gc-duration-using-usdt-4fd4>，article id 5090916，tag `ebpf`/`linux`/`tutorial`/`kernel`，reading_time 9，0 反应 0 评论。workflow 随后自行推送 `bca850b92b6` 清空队列。
- 正文对账：公开 `body_markdown` 与准备稿去首个 H1 后**逐字相同**（16,143 字符）。
- 结构对账（可见页 DOM，整页滚动）：1 h1（精确标题）、6 个正文 H2（`Introduction to USDT`、`Introduction to Java GC`、`eBPF Implementation Mechanism`、`Installing Dependencies`、`Compiling and Running`、`Summary`）、3 个 H3、10 个 `pre`（`highlight shell` x5、`highlight c` x4、`highlight console` x1）、0 处泄漏的反引号、0 图片、0 表格、正文 15,865 字符、结尾完整、tag 芯片 4 个、0 评论。
- 四处修补在公开页一一复核：`problem.In Linux`、`respectively.Finally`、`Summary.Through` 均不再粘连，`PERF_EVENT_ARRAY` 带分号，`Summary` 恢复为独立 H2（站点原文仍缺）。
- 覆盖：DEV 75 -> 76/128，`check_media_ledger.py` 退出 0（100 条 confirmed，0 条 confirmed-without-source_path）。
- 额度：发布时刻 `2026-11-22T00:14:19Z` = LA `2026-11-21 16:14 PST`，用掉 **LA 11-21 正常额度**（LA 11-21 当日此前无其它发布）；补发缺口仍 0。

## 观测

- 运行延迟：`workflow_dispatch` 后约 2 分钟起 job、整条 21 秒（11-21 那次 22 秒）。
- 列表滞后未复现：发布后约 5 分钟，公开作者 API 即返回 83 篇且**含** `5090916`。仍沿用"不要用列表判断发布结果"的结论。
- X `2122736534206181608`（Lean 证明检查成本）1,812 浏览 / 18 赞 / 4 转帖 / 8 书签 / 2 回复（上一快照 1,801/18/4/8/2）；两条回复仍是 `@difficultyang` 的观点与 `@sk_196883` 的感叹，不需回。`2122011756633665955`（bpf-gpu-server）990 浏览 / 30 赞 / 9 书签 / 1 回复（987/30/10/1），唯一回复仍是 `@sunilmehta_si` 11-16 的复述。
- HN 无新回声：`search_by_date` 最新的 eunomia 故事仍是 50385391（AgentSight，2026-11-05，2 分 0 评论）；tutorial 15 精确标题与 `eBPF USDT Java GC` 搜索均 0 命中。DEV 11-09..11-22 的副本全部 0 反应 0 评论。

## 遗留

- canonical 仍为 DEV 自身（连续第二十次）。根因仍在上游 `media-publisher`：`publish-multi` 从 `title/content/tags/is_draft/platforms` 重建 payload，`canonical_url` 到不了 DEV 分支。
- 站点原文 `https://eunomia.dev/tutorials/15-javagc/` 仍渲染四处源文缺陷，且目录只有 5 个 H2（缺 `Summary`）。上游 `eunomia-bpf/bpf-developer-tutorial` 的 `src/15-javagc/README.md` 需同一处修补；`docs/tutorials/**` 由上游同步，本地无该仓库，按 0/1/3/4/8/11/13/14 的先例只修准备稿。
- `install-agent-clis.sh` 的 `runuser` PATH 缺口仍未回改镜像；本次浏览器已就绪，未触及。

## 下一次具体动作

1. 恢复知乎登录后，从 `53-egress-pacer` 起继续处理 23 条知乎任务（掘金恢复登录后可作备用发布通道）。
2. 若 DEV 继续日更：下一候选 `17-biopattern`（按精确标题在 DEV 账号 83 篇里没有副本；`16-memleak` 已在 `confirmed_source_urls` 里算覆盖）。发布前同样要核对上游英文源文的围栏/标点缺陷，并同步核对中文源文。