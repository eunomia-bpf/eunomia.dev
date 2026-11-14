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

## 遗留

- canonical 仍为 DEV 自身（连续第十一次）。根因仍在上游 `media-publisher`：`publish-multi` 从 `title/content/tags/is_draft/platforms` 重建 payload，`canonical_url` 到不了 DEV 分支；已并入 `platforms/devto.json` 的同一条 `pending_verification`。
- tutorial 4 的源文围栏粘连未回改上游 `bpf-developer-tutorial`；下一次同步前它仍会出现在每一份新的 syndication 里。

## 下一次具体动作

1. 恢复知乎登录后，从 `53-egress-pacer` 起继续处理 23 条知乎任务；恢复掘金登录（当前无任务）。
2. 若 DEV 继续日更：下一候选 `5-uprobe-bashreadline`（DEV 缺失），按教程序号继续推进。