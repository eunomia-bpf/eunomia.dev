# 2026-11-14 内容巡检运行日志

运行日按 +08 计（系统日期 2026-11-14，UTC `2026-11-14T00:00Z` 起，LA 2026-11-13 16:00 PST）。本次巡检完成**一条 DEV 发布**；队列里 23 条知乎任务仍是平台级 `阻塞`。补发缺口 0。

## 运行前修复：CDP 9222 无响应

- 现象：`curl http://127.0.0.1:9222/json/version` 返回 rc=7（连接被拒），`/tmp/.X11-unix` 为空，`/var/run/gem/browser-supervisor.json` 不存在。
- 根因不是既有的 crashpad 记录：PID 1（`/opt/gem/gem.sh`）卡在 `until bash /run/coder-agent-config/install-agent-clis.sh` 重试循环里。pre-services hook 的 `PATH` 为 `/home/gem/.local/bin:/usr/local/bin:/usr/bin:/bin`，缺 `/usr/sbin`，因此 root 分支的 `exec runuser` 以 rc=127 失败，supervisord 从未启动，浏览器监督进程也就没写状态文件。
- 修复（只增不删）：`/usr/local/bin/runuser` -> `/usr/sbin/runuser` 符号链接；`sudo -n find /home/gem -user root -exec chown gem:gem {} +`（5,430 个文件，含 root 拥有的 `/home/gem/.npm`、`/home/gem/.local/state`）；`sudo -n chown -R gem:gem /home/gem/.config/chromium`。
- 结果：`/var/log/gem/browser-supervisor.log` 出现 `Chromium ready on 127.0.0.1:9222, pid=20524`，`/json/version` 返回 `Chrome/146.0.7680.31`。未重启 PID 1（容器入口，杀掉会重建整个运行时）。

## 任务选择

- 队列顶部 23 条 `排队` 任务全部是知乎，可见 Chrome 上 `https://www.zhihu.com/creator` 仍落 `/signin?next=%2Fcreator`，属平台级 `阻塞`，未消耗额度。
- 掘金本次已**掉登录**：`https://juejin.cn/creator/home` 落到 `https://juejin.cn/` 并显示 登录/注册（11-13 还是已登录）。队列中掘金任务为 0，无影响。
- 按"历史回补默认只处理 tutorial"取教程序列中最早的 DEV 缺口 `4-opensnoop`（Medium 副本自 2023 年已 `confirmed`，`platforms/devto.json` 无对应 `source_path`）。发布前用 DEV 公开 API 复查账号 74 篇、精确标题 0 篇重复，可见 `https://dev.to/yunwei37` 最新为 tutorial 3/2/1/0，无 tutorial 4。
- shell 仍无 `DEV_TO_API_KEY`/`MEDIUM_API_KEY`，走 secret 支撑的 `publish-posts.yml`。

## 准备稿的唯一差异：源文围栏粘连

- `docs/tutorials/4-opensnoop/README.md:107-109` 把 `trace_pipe` 命令和第一行输出粘成一行（`$ sudo cat .../trace_pipe".\-3840345 [010] ...`），多出一个 `".\`。中文版 `README.zh.md` 已是正确的三行形式。
- `docs/tutorials/**` 由 `bpf-developer-tutorial` 同步而来（本 checkout 无该仓库），按 0/1/3 的既有做法**只修准备稿**：`draft/media/2026-11-14/4-opensnoop/devto.md` 恰好一处 hunk 还原三行，`diff -u` 除该 hunk 外无差异。
- 结构：1 个 H1（精确标题）、3 个正文 H2、0 个 H3/H4、12 个围栏（6 个 `pre`）、0 图片、0 表格、约 1,172 词；外链全部绝对（`github.com/eunomia-bpf/eunomia-bpf`、`.../bpf-developer-tutorial/tree/main/src/4-opensnoop`、`github.com/eunomia-bpf/bpf-developer-tutorial`、`eunomia.dev/tutorials/`）；站点原文 <https://eunomia.dev/tutorials/4-opensnoop/> 200 且 H1 一致。

## 发布事实

- 路径：`.github/publisher/posts_queue.txt`（1 条，字段 `path`/`tags`）-> commit `9c07f5809f` 推送 `main` -> `gh workflow run publish-posts.yml -f platforms=devto -f publish_count=1 -f draft_only=false`。run 43568653176 在 `2026-11-14T07:02:12Z` 公开：<https://dev.to/yunwei37/ebpf-tutorial-by-example-4-capturing-opening-files-and-filter-with-global-variables-39bh>，article id 5034069，tag `ebpf`/`linux`/`tutorial`/`kernel`，0 反应 0 评论。workflow 随后自行推送 `79a07620e2` 清空队列。
- 正文对账：公开 `body_markdown` 与准备稿去首个 H1 后**逐字相同**（8,100 字符）；粘连的 `trace_pipe".\` 在公开正文里已不存在，命令与首行输出是分开的三行。
- 结构对账（可见页 DOM）：1 h1（精确标题）、3 个正文 H2（Capturing the System Call Collection of Process Opening Files in eBPF / How Global Variables Work in eBPF / Summary）、0 H3、6 个 `pre`、0 表格、0 正文图片、1,157 词、结尾完整（"…or website https://eunomia.dev/tutorials/ for more examples and a complete tutorial."）。
- 覆盖：DEV 67 -> 68/128，`check_media_ledger.py` 退出 0（84 条 confirmed，0 条 confirmed-without-source_path）。
- 额度：发布时刻 `2026-11-14T07:02:12Z` = LA `2026-11-13 23:02 PST`，用掉 **LA 11-13 正常额度**；补发缺口仍 0。

## 运行延迟：`ubuntu-latest` 组织级积压

- 首个 dispatch（run 43534543886，`2026-11-14T00:29:38Z`）排队约 6 小时未获 runner。同期证据：同仓库 `app-static-pages.yml` 中 `runs-on: ubuntu-24.04` 的 deploy job 在 `created 06:14:30` 后数十秒内 `started`；bpftime 同 `ubuntu-latest` 标签 `queued` 峰值 268、`in_progress` 稳定 5-10；GitHub status 页 "All Systems Operational"、无未解决 incident；仓库无 self-hosted runner（`total_count: 0`）。
- 处置：把 `.github/workflows/publish-posts.yml` 的 `runs-on` 从 `ubuntu-latest` 改为 `ubuntu-24.04`（commit `5462d96717`），重新 dispatch（run 43568653176）后 7 分钟取得 runner，1 分钟完成发布。前两次 dispatch（43534543886、43552763549、43565716716）均已 cancel，队列条目未受影响——`posts_queue.txt` 只有在 workflow 成功后才由 bot 清空。
- 后续观察（同一 org 级积压仍在）：记录推送触发的两次 `Deploy Static App`（run 43568645362 / 43569726614）在 `07:14Z` 仍处 `pending`/`in_progress` 且 job 的 `startedAt` 为空，即同样没拿到 runner；同时 `eunomia-bpf/bpftime` 的 `queued` 仍是 **277**，`bpf-developer-tutorial` 只有 4。可见积压不是 `ubuntu-latest` 标签独有，`ubuntu-24.04` 也会被同一批排队作业拖住，只是本次发布窗口里恰好有空位。结论按此收窄：pin `ubuntu-24.04` 是当时唯一可用的摆渡方案，不是对积压的根治；根治要看 bpftime 侧是否在无上限地堆排队 run。本站 deploy 为纯文档变更，不阻塞本次交付。收尾结果：`Deploy Static App` run 43569726614 在 `07:15:37Z` 被 concurrency 组取消（它不是最后一次推送；`cancel-in-progress` 对 push 事件为 false，但新 run 43570250202 于 `07:15:36Z` 创建后一秒旧 run 即转 cancelled，属组内正常让位），最新 run 43570250202 载 `3c96254915` 于 `07:15:36Z` 入队、`07:22Z` 取得 runner、`07:44:14Z` 全部 13 步 success（含 `Deploy to GitHub Pages` 与 Discord 通知）；`https://eunomia.dev/tutorials/4-opensnoop/` 仍 200。
- 代价：轮询把 GitHub core API 打到过 403 rate limit（`rate_limit` 随后显示 4,945/5,000 剩余，说明是短窗口突发），此后改为 20-30 分钟一次的稀疏检查加 DEV 公开 API 判定发布。

## 遗留

- canonical 仍为 DEV 自身（连续第十二次）。根因仍在上游 `media-publisher`：`publish-multi` 从 `title/content/tags/is_draft/platforms` 重建 payload，`canonical_url` 到不了 DEV 分支；已并入 `platforms/devto.json` 的同一条 `pending_verification`。
- tutorial 4 的源文围栏粘连未回改上游 `bpf-developer-tutorial`；下一次同步前它仍会出现在每一份新的 syndication 里。

## 下一次具体动作

1. 恢复知乎登录后，从 `53-egress-pacer` 起继续处理 23 条知乎任务；恢复掘金登录（当前无任务）。
2. 若 DEV 继续日更：下一候选 `5-uprobe-bashreadline`（DEV 缺失），按教程序号继续推进。