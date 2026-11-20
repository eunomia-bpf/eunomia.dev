# 2026-11-20 内容巡检运行日志

运行日按 +08 计（系统日期 2026-11-20，UTC `2026-11-20T00:00Z` 起，LA 2026-11-19 16:00 PST）。本次巡检完成**一条 DEV 发布**；队列里 23 条知乎任务仍是平台级 `阻塞`。补发缺口 0。

## 运行前阻塞（已修，非本次内容操作）

- 开工时 CDP 9222 无响应（`curl` 返回 7），`/var/log/gem/browser-supervisor.log` 只有一行 `waiting for config /var/run/gem/browser-supervisor.json`，`supervisord`、`Xvnc :99.0` 与 `/opt/browser/chrome` 都未启动。
- 与 2026-11-19 同一根因：`/opt/gem/gem.sh` 的 pre-services hook（PID 257）仍卡在 `until bash /run/coder-agent-config/install-agent-clis.sh` 重试循环里，该脚本 `exec runuser -u gem` 时 hook 的 `PATH` 是 `/home/gem/.local/bin:/usr/local/bin:/usr/bin:/bin`，不含 `/usr/sbin`。
- 修复：`ln -sfn /usr/sbin/runuser /usr/local/bin/runuser`。hook 下一轮成功，`browser-supervisor.json` 写入，`tigervnc` / `openbox` / `agent-browser` 起，CDP 9222 恢复（`Chrome/146.0.7680.31`）。
- 说明：符号链接只存在于本次容器实例；镜像里的 `install-agent-clis.sh` 未改，换容器会同样卡住。`browser-ctl.sh` 的 `SUPERVISOR_CONF=/opt/gem/supervisord/browser.conf` 也仍指向不存在的文件，属同一套过时脚本。
- 浏览器恢复后 `browser.open()` 默认的 headless 守护进程仍报 `Shared browser daemon unavailable`，改用 `app: { cdp_url: "http://127.0.0.1:9222" }` 依附已运行的可见 Chrome 即可正常操作。

## 任务选择

- 队列顶部 23 条 `排队` 任务全部是知乎；可见 Chrome 上 `https://www.zhihu.com/creator` 落 `https://www.zhihu.com/signin?next=%2Fcreator`，cookie 只有 `_zap`/`d_c0`/`_xsrf`/`HMACCOUNT` 等匿名标识，无 `z_c0`，属平台级 `阻塞`，未消耗额度。
- 掘金仍**掉登录**：`https://juejin.cn/creator/home` 落回 `https://juejin.cn/`，导航为 登录/注册。队列中掘金任务为 0，无影响。
- Medium 仍无 `MEDIUM_API_KEY`；可见 Chrome `/me/stories` 为未登录的 `Sign up`/`Sign in` 与空草稿列表。
- X `@yunwei37` 已登录（1,755 posts）；LinkedIn `/in/yunwei37` 仍落 `/authwall`。
- 按"历史回补默认只处理 tutorial"取教程序列中最早的 DEV 缺口 `13-tcpconnlat`：`platforms/devto.json` 的明细行和 `confirmed_source_urls` 都没有它，Medium 副本早已 `confirmed`。发布前用 DEV 公开 API 复查账号 80 篇、精确标题 0 篇重复、`eBPF Tutorial by Example 13` 前缀 0 篇；站点原文 `/tutorials/13-tcpconnlat/` 与 `/zh/tutorials/13-tcpconnlat/` 均 200，结构一致（1 H1、5 H2、3 H3、10 `pre`）。

## 准备稿

- `draft/media/2026-11-20/13-tcpconnlat/devto.md` 与源文只有四处差异：
  1. `README.md:34` 的 `![tcpconnlat1](tcpconnlat1.png)` 是站内相对路径，改写成 `https://eunomia.dev/_content-assets/docs/tutorials/13-tcpconnlat/tcpconnlat1.png`（实测 200，`image/png`，135,015 字节）。
  2. `:209` 的注释开头是 `/*".`（多出引号与句点），还原为 `/**`；中文 `README.zh.md:208` 是干净的。
  3. `:544` 把编译输出挤成 `BPF      .output/tcpconnlat.bpf.o".GEN-SKEL .output/tcpconnlat.skel.h` 一行，拆成两行；中文 `README.zh.md:573-574` 本来就是两行。
  4. `:573` 的 `> The original link of this article:` 指向旧 slug `/tutorials/13-tcpconnect`，改为 `/tutorials/13-tcpconnlat`。旧 slug 仍返回 200 的跳转壳，canonical 指向真页，但正文里引用它会让读者多一跳。
- 三处文本缺陷都在上游 `eunomia-bpf/bpf-developer-tutorial` 的 `src/13-tcpconnlat/README.md` 里同样存在（实测 raw 内容第 209、544 行一致）。
- 结构（准备稿）：1 个 H1（精确标题）、5 个正文 H2、3 个 H3、20 行首围栏 = 10 个代码块（`c` x9、`console` x1）、1 张图片（已绝对化）、0 表格。

## 发布事实

- 路径：`.github/publisher/posts_queue.txt`（1 条，字段 `path`/`tags`）-> commit `ac1acc55346` 推送 `main` -> `gh workflow run publish-posts.yml -f platforms=devto -f publish_count=1 -f draft_only=false`。run 44644797186 在 `2026-11-20T00:18:40Z` 公开：<https://dev.to/yunwei37/ebpf-tutorial-by-example-13-statistics-of-tcp-connection-delay-with-libbpf-45kc>，article id 5077727，tag `ebpf`/`linux`/`tutorial`/`kernel`，reading_time 13，0 反应 0 评论。workflow 随后自行推送 `58bb6079cac` 清空队列。
- 正文对账：公开 `body_markdown` 与准备稿去首个 H1 后**逐字相同**（24,645 字符）。
- 结构对账（可见页 DOM，整页滚动）：1 h1（精确标题）、5 个正文 H2（`Background`、`Overview of tcpconnlat Tool`、`eBPF Implementation of tcpconnlat`、`Compilation and Execution`、`Summary`）、3 个 H3（`TCP Connection Principle`、`Analysis of tcp_v4_connect function`、`Kernel Code`）、10 个 `pre`（`highlight c` x9、`highlight console` x1）、0 处泄漏的反引号、1 张 800x801 图片已加载且 alt 保留、0 表格、3,049 词、结尾完整（含改好的 `eunomia.dev/tutorials/13-tcpconnlat`）、tag 芯片 4 个、0 评论。5 个正文外链目标（`bpf-developer-tutorial/tree/main/src/13-tcpconnlat`、`elixir.bootlin.com` 的 `tcp_ipv4.c`、bcc 的 `tcpconnlat.c`、教程仓库、原文）全部 200。
- 覆盖：DEV 73 -> 74/128，`check_media_ledger.py` 退出 0（96 条 confirmed，0 条 confirmed-without-source_path）。
- 额度：发布时刻 `2026-11-20T00:18:40Z` = LA `2026-11-19 16:18 PST`，用掉 **LA 11-19 正常额度**（LA 11-19 当日此前无其它发布）；补发缺口仍 0。

## 观测

- 运行延迟：dispatch 后 9 秒起 job、整条 12 秒，是这一个月里最快的一次（11-17 那次排队 21 分钟）。
- 列表滞后未复现：本次公开作者 API 在发布后即返回 80 篇且包含 `5077727`。仍沿用 11-12 的结论：不要用列表判断发布结果（11-17..19 三次都滞后）。
- X `2122736534206181608`（Lean 证明检查成本）1,785 浏览 / 18 赞 / 4 转帖 / 8 书签 / 2 回复（上一快照 1,728/18/4/8/2），两条回复仍不需回。`2122011756633665955`（bpf-gpu-server）983 浏览 / 30 赞 / 10 书签 / 1 回复，唯一回复仍是 `@sunilmehta_si` 11-16 的复述。
- HN 无新回声：`search_by_date` 最新的 eunomia 故事仍是 50385391（AgentSight，2026-11-05，2 分 0 评论）；11-16..19 的 "eunomia" 评论命中都是他词模糊匹配。tutorial 13 的精确标题搜索只返回站点原文、教程索引和一个 GitHub fork 镜像，无独立讨论。

## 遗留

- canonical 仍为 DEV 自身（连续第十八次）。根因仍在上游 `media-publisher`：`publish-multi` 从 `title/content/tags/is_draft/platforms` 重建 payload，`canonical_url` 到不了 DEV 分支。
- 站点原文 `https://eunomia.dev/tutorials/13-tcpconnlat/` 仍渲染 `/*".* This unhashes` 与 `.output/tcpconnlat.bpf.o".GEN-SKEL` 两处源文缺陷（准备稿已修）。`docs/tutorials/**` 由上游同步，本地无该仓库，按 0/1/3/4/8/11 的先例只修准备稿。
- `install-agent-clis.sh` 的 `runuser` PATH 缺口没有回改镜像：本次仍只在 `/usr/local/bin` 加了符号链接。
- 仓库根目录累计 13 个 `core.*` core dump 与一个 `profile.html`，均未跟踪；本次未纳入提交。

## 下一次具体动作

1. 恢复知乎登录后，从 `53-egress-pacer` 起继续处理 23 条知乎任务（掘金恢复登录后可作备用发布通道）。
2. 若 DEV 继续日更：下一候选 `14-tcpstates`（`15-javagc` 同样缺）。