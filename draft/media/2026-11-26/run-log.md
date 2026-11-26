# 2026-11-26 内容巡检运行日志

运行日按 +08 计（UTC `2026-11-26T00:00Z` 起，LA 2026-11-25 16:00 PST）。本次巡检完成**一条 DEV 发布**；队列里 23 条知乎任务仍是平台级 `阻塞`。补发缺口 0。

## 运行前状态

- CDP 9222 无响应（`curl` 返回 7），修复后为 `Chrome/146.0.7680.31`。根因是已知镜像缺陷：PID 1 的 pre-services hook 在 `install-agent-clis.sh` 里调用 `runuser`，而 hook 的 `PATH` 不含 `/usr/sbin`，supervisor 一直停在 `waiting for config`。修法沿用 11-19/11-20 先例，只做一条软链：`sudo ln -sfn /usr/sbin/runuser /usr/local/bin/runuser`；未删除或重建 profile，登录状态保留（OWNERSHIP 不变）。修复后 supervisor 写入配置、起 Xvnc :99.0 与 Chromium，日志出现 `Chromium ready on 127.0.0.1:9222, pid=39153`。
- 本地 `main` 落后 `origin/main` 4 个提交（`61633537f85`、`58b7a677f26`、`93afeba1869`、`67e7dedbfa2`），rebase 后再动手。
- 仓库根目录 14 个 `core.*`（各约 1.3 TB 稀疏 dump）与 `profile.html` 仍未跟踪，未纳入提交。
- `check_media_ledger.py` 改动前退出 0（DEV 79/128，106 条 confirmed），改动后退出 0（DEV 80/128，108 条 confirmed）。

## 任务选择

- 队列顶部 23 条 `排队` 任务全部是知乎；可见 Chrome `/creator` 仍落 `/signin?next=%2Fcreator`，平台级 `阻塞`，未消耗额度。
- 掘金 `/creator/home` 仍掉登录（落回 `https://juejin.cn/`，导航为 登录/注册）；Medium `/me/stories` 仍 `Sign up`/`Sign in` 且无 `MEDIUM_API_KEY`；LinkedIn `/in/yunwei37` 落 `/authwall`；DEV web 会话也掉登录（`/dashboard` 落 `/magic_links/new`），secret-backed workflow 仍是唯一 DEV 通道。
- 按"历史回补默认只处理 tutorial"取教程序列中最早的 DEV 缺口 `22-android`：`devto.json` 明细行与 `confirmed_source_urls` 都没有它。发布前用 DEV 公开 API 复查账号 86 篇、精确标题 0 篇、`Tutorial by Example 22` 前缀 0 篇、标题含 `Android` 0 篇；站点 `/tutorials/22-android/` 与 `/zh/tutorials/22-android/` 均 200。

## 准备稿

`draft/media/2026-11-26/22-android/devto.md` 只修一处英文源文缺陷：`README.md:44` 把两条列表项粘成 `... /data/eadb/deb.tar.gz`+ `rm -r /data/local/tmp/assets`，`rm` 那条从未成为独立条目（中文 `README.zh.md:42-43` 两条分开）。其余为逐字节保留：15 个绝对外链、0 图片、0 表格、0 相对链接、8 个围栏全部带 `console` 标签。上游 `bpf-developer-tutorial` 的 `src/22-android/README.md` 与本地英文源 `diff` 为空（12,652 字节），缺陷同样在上游。

结构（准备稿）：1 个 H1（精确标题）、7 个 H2、1 个 H3、2 个 H4、4 个代码块（全部 `console`）、0 图片、0 表格。

## 发布事实

- 路径：`.github/publisher/posts_queue.txt`（1 条，字段 `path`/`tags`）-> commit `4dd62d5776c` 推送 `main` -> `gh workflow run publish-posts.yml -f platforms=devto -f publish_count=1 -f draft_only=false`。run 45729677157 在 `2026-11-26T00:34:36Z` 公开：<https://dev.to/yunwei37/ebpf-tutorial-by-example-using-ebpf-programs-on-android-cj2>，article id 5120170，tag `ebpf`/`linux`/`tutorial`/`kernel`，reading_time 7，0 反应 0 评论。workflow 随后自行推送 `1672c47` 清空队列。
- 正文对账：公开 `body_markdown` 与准备稿去首个 H1 后**逐字相同**（12,595 / 12,595 字符）。
- 结构对账（可见页 DOM）：1 个精确标题 h1、7 个正文 H2（`Background`、`Test Environment`、`Environment Setup[^SeeFlowerX]`、`Build Tools`、`Results`、`Summary`、`Reference`）、1 个 H3、2 个 H4、4 个 `pre`（全部 `highlight console`，语言标签保留）、修好的 `rm -r` 已成独立列表项、0 泄漏反引号、0 图片、0 表格、4 个 tag 芯片、`Top comments (0)`、结尾完整（`The original link of this article: https://eunomia.dev/tutorials/22-android`）。
- 脚注：3 个 `[^...]` 标记在 DEV 上按字面渲染；线上站点 `/tutorials/22-android/` 同样按字面渲染（该源文没有脚注定义），因此不是平台新增缺陷，未改写。
- 额度：发布时刻 `2026-11-26T00:34:36Z` = LA `2026-11-25 16:34 PST`，用掉 **LA 11-25 正常额度**；补发缺口仍 0。

## 观测

- 运行延迟：dispatch 后约 1 分钟起 job（11-25 那次排队约 7 分钟），job 本身 21 秒。
- 列表滞后：发布后数分钟，公开作者 API 返回 86 篇，不含 `5120170` 但已重新包含前一天的 `5112060`，而 per-article 端点与公开页均 200（未登录 profile 页报 87 篇）。继续沿用"不要用列表判断发布结果"。
- X `2122736534206181608`（Lean 证明检查成本）1,858 浏览 / 18 赞 / 4 转帖 / 8 书签 / 2 回复（上一快照 1,847/18/4/8/2）；两条回复仍是观点与感叹，不需回。`2122011756633665955`（bpf-gpu-server）1,006 浏览 / 30 赞 / 9 书签 / 1 回复（1,002/30/9/1），唯一回复仍是 `@sunilmehta_si` 11-16 的复述。账号 1,764 posts（11-23 为 1,758）。
- 站点原文：`/tutorials/22-android/` 本身只带上面那一处列表粘连缺陷。三处已知缺陷仍未修：`/tutorials/19-lsm-connect/` 仍把 `16843009".` 与内联围栏混进 trace 块、`/tutorials/17-biopattern/` 仍在同一段重复 `compatibility issues` 两次、`/tutorials/15-javagc/` 仍渲染 `problem.In Linux`。
- HN 无新回声：`search_by_date` 里带 `eunomia` 的最新故事仍是 50385391（AgentSight，2026-11-05），其余命中是 11-24/11-25 的无关故事；精确标题搜索无 tutorial 22 副本。

## 台账同步

`platforms/devto.json` 新增 `devto-android` 行与 `confirmed_source_urls` 映射（id 用 `devto-android`，因为 `devto-cj2` 已被 2026-02-17 的 BPF dynamic pointers 副本占用），`last_checked` 提到 2026-11-26，`browser_observations` 追加一条，`pending_verification` 的自 canonical 计数改为二十四并纳入 `devto-android`，同时把 22-android 的列表粘连缺陷并入英文源缺陷清单；`sources.json` 同步日期；`published.md`（LF）加行、刷新日期并补 11-26 的列表滞后观察；`not-published.md`（CRLF）加 11-26 段。`check_media_ledger.py` 退出 0。

## 遗留

- canonical 仍为 DEV 自身（连续第二十四次）。根因仍在上游 `media-publisher`：`publish-multi` 从 `title/content/tags/is_draft/platforms` 重建 payload，`canonical_url` 到不了 DEV 分支。
- 站点原文缺陷同前：19/17/15 未修，22-android 的列表粘连也未修。`docs/tutorials/**` 由上游同步，本地无该仓库，按 0/1/3/4/8/11/13/14/15/17/19/20 的先例只修准备稿。
- 浏览器修复（`runuser` 软链）只是运行期补丁，镜像缺陷未回改，换容器会复现。

## 下一次具体动作

1. 恢复知乎登录后，从 `53-egress-pacer` 起继续处理 23 条知乎任务（掘金恢复登录后可作备用发布通道）。
2. 若 DEV 继续日更：下一候选 `24-hide`（`devto.json` 无该行，DEV 账号 86 篇里也没有副本；剩余未映射英文 tutorial 还有 `28-detach`/`29-sockops`/`37-uprobe-rust`/`40-mysql`/`49-hid`/`bpftrace-tutorial`，`18-further-reading` 是目录页）。发布前同样核对上游英文源文的围栏/标点缺陷，并对照中文 `README.zh.md`。
3. 下次巡检开工先探 CDP 9222；若 supervisor 又停在 `waiting for config`，先确认 `/usr/local/bin/runuser` 软链是否随容器重建丢失。