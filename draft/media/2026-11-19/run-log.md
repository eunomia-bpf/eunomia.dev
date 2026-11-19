# 2026-11-19 内容巡检运行日志

运行日按 +08 计（系统日期 2026-11-19，UTC `2026-11-19T00:00Z` 起，LA 2026-11-18 16:00 PST）。本次巡检完成**一条 DEV 发布**；队列里 23 条知乎任务仍是平台级 `阻塞`。补发缺口 0。

## 运行前阻塞（已修，非本次内容操作）

- 开工时 CDP 9222 无响应（`curl` 返回 7），`/var/log/gem/browser-supervisor.log` 只有一行 `waiting for config /var/run/gem/browser-supervisor.json`，`/var/run/gem/` 里没有该配置，`supervisord`、`Xvnc :99.0`、`/opt/browser/chrome` 全未启动 —— 不是本 skill 记录的 crash-loop（`rc=-5` / crashpad 目录 root 属主），而是 PID 1 之后容器启动就没走完。
- 根因：`/opt/gem/gem.sh` 的 pre-services hook（PID 257）卡在 `until bash /run/coder-agent-config/install-agent-clis.sh` 重试循环里。该脚本以 root 运行时会 `exec runuser -u gem`，但 hook 导出的 `PATH` 是 `/home/gem/.local/bin:/usr/local/bin:/usr/bin:/bin`，`runuser` 只在 `/usr/sbin`（`/sbin` 为同一文件的另一链接），于是每次都 `runuser: not found` 失败，重试 10 秒一轮，`write_browser_supervisor_config` 和 `start_supervisord` 永远到不了。
- 修复：`ln -sfn /usr/sbin/runuser /usr/local/bin/runuser`（只加一个 PATH 内符号链接，未改脚本、未动 profile）。hook 下一轮成功，`browser-supervisor.json` 写入，`supervisord` 起 `tigervnc` / `openbox` / `agent-browser` / `mcp-server-browser`，CDP 9222 恢复（`Chrome/146.0.7680.31`）。
- 说明：`install-agent-clis.sh` 自身仍会打印 `dsh needs a newer Node.js than this image provides; skipped`（`npm ETARGET @deepseek-ai/dsh@*`），但脚本以 0 退出，不再阻塞启动。

## 任务选择

- 队列顶部 23 条 `排队` 任务全部是知乎；可见 Chrome 上 `https://www.zhihu.com/creator` 落登录页，cookie 只有 `_zap`/`d_c0`/`_snaker__id`/`_xsrf`/`HMACCOUNT` 等匿名标识，无 `z_c0`，属平台级 `阻塞`，未消耗额度。
- 掘金仍**掉登录**：`https://juejin.cn/creator/home` 落回 `https://juejin.cn/`，导航为 登录/注册。队列中掘金任务为 0，无影响。
- Medium 仍无 `MEDIUM_API_KEY`；可见 Chrome `/me/stories` 本次是**未登录**的 `Sign up`/`Sign in` 与空草稿列表（11-18 那次是 Cloudflare 验证页）。
- X `@yunwei37` 已登录（1,755 posts）；LinkedIn `/in/yunwei37` 仍落 `/authwall`。
- 按"历史回补默认只处理 tutorial"取教程序列中最早的 DEV 缺口 `12-profile`：`platforms/devto.json` 的明细行和 `confirmed_source_urls` 都没有它，Medium 副本早已 `confirmed`。发布前用 DEV 公开 API 复查账号 79 篇、精确标题 0 篇重复；站点原文 `/tutorials/12-profile/` 与 `/zh/tutorials/12-profile/` 均 200，结构一致（1 H1、2 H2、3 H3、15 `pre`）。

## 准备稿

- `draft/media/2026-11-19/12-profile/devto.md` 是源文**逐字节副本**（18,575 字节，sha256 `227592390c9c…` 相同）。该英文源文没有围栏缺陷、没有相对链接、没有图片和表格，是这轮回补里第一份不需要修源的 tutorial（4/8/11 都带英文缺陷）。
- 结构（准备稿）：1 个 H1（精确标题）、2 个正文 H2、3 个 H3、30 行首围栏 = 15 个代码块（`c` x11、`console` x2、`rust` x2）、0 图片、0 表格、0 相对链接。

## 发布事实

- 路径：`.github/publisher/posts_queue.txt`（1 条，字段 `path`/`tags`）-> commit `7e6a53f415b` 推送 `main` -> `gh workflow run publish-posts.yml -f platforms=devto -f publish_count=1 -f draft_only=false`。run 44429067976 在 `2026-11-19T00:27:07Z` 公开：<https://dev.to/yunwei37/ebpf-tutorial-by-example-12-using-ebpf-program-profile-for-performance-analysis-4im2>，article id 5069111，tag `ebpf`/`linux`/`tutorial`/`kernel`，reading_time 9，0 反应 0 评论。workflow 随后自行推送 `b2a4e96e9f7` 清空队列。
- 正文对账：公开 `body_markdown` 与准备稿去首个 H1 后**逐字相同**（18,490 字符）。
- 结构对账（可见页 DOM，整页滚动）：1 h1（精确标题）、2 个正文 H2（`eBPF Tool: profile Performance Analysis Example`、`Implementation Principle`）、3 个 H3（`Kernel Space Part`、`User Mode Section (Rust Implementation)`、`Summary`）、15 个 `pre`（`highlight c` x11、`highlight console` x2、`highlight rust` x2）、0 处泄漏的反引号、0 表格、0 正文图片、结尾完整（含 `eunomia.dev/tutorials/12-profile` 链接）、tag 芯片 4 个、0 评论。5 个正文外链目标（`bpf-developer-tutorial/tree/main/src/12-profile`、Cargo Book、教程仓库、`/tutorials/`、原文）全部 200。
- 覆盖：DEV 72 -> 73/128，`check_media_ledger.py` 退出 0（94 条 confirmed，0 条 confirmed-without-source_path）。
- 额度：发布时刻 `2026-11-19T00:27:07Z` = LA `2026-11-18 16:27 PST`，用掉 **LA 11-18 正常额度**（LA 11-18 当日此前无其它发布）；补发缺口仍 0。

## 观测

- 运行延迟未复现：dispatch 后约 20 秒起 job，整条 18 秒（11-17 那次排队 21 分钟）。
- 列表滞后复现：发布后数分钟，公开作者 API 返回 79 篇且不含 `5069111`，而 per-article 端点与公开页都已 200。沿用 11-12 的结论：不要用列表判断发布结果。
- X `2122736534206181608`（Lean 证明检查成本）升到 1,728 浏览 / 18 赞 / 4 转帖 / 8 书签 / 2 回复（11-18 快照为 939 / 12 / 2 / 7 / 1）。新增回复 `@sk_196883` 是"怎么大家都在写证明"的感叹，`@difficultyang` 仍是重申本文论点，均不需回复。
- X `2122011756633665955`（bpf-gpu-server）973 浏览 / 30 赞 / 10 书签 / 1 回复；该回复 `@sunilmehta_si` 日期为 Nov 16，说明 11-18 快照里"no new reply"是记录错误，不是新增。内容复述结论，不需回复。
- HN 无新回声：`search_by_date` 最新的 eunomia 故事仍是 50385391（AgentSight，2026-11-05，2 分 0 评论）；11-14..18 的 "eunomia" 评论命中都是他词模糊匹配。

## 遗留

- canonical 仍为 DEV 自身（连续第十七次）。根因仍在上游 `media-publisher`：`publish-multi` 从 `title/content/tags/is_draft/platforms` 重建 payload，`canonical_url` 到不了 DEV 分支。
- `install-agent-clis.sh` 的 `runuser` PATH 缺口没有回改镜像：本次只在 `/usr/local/bin` 加了符号链接。若换容器，浏览器栈会同样卡在 pre-services hook；`browser-ctl.sh` 的 `SUPERVISOR_CONF=/opt/gem/supervisord/browser.conf` 也指向一个不存在的文件，属同一套过时脚本，未改。
- 仓库根目录累计 13 个 `core.*` core dump 与一个 `profile.html`，均未跟踪；本次未纳入提交。

## 下一次具体动作

1. 恢复知乎登录后，从 `53-egress-pacer` 起继续处理 23 条知乎任务（掘金恢复登录后可作备用发布通道）。
2. 若 DEV 继续日更：下一候选 `13-tcpconnlat`（`14-tcpstates`/`15-javagc` 同样缺）。