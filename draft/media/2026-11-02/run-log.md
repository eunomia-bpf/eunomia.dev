# 2026-11-02 内容巡检运行日志

运行日按 America/Los_Angeles 自然日计算：LA 2026-11-02（系统日期 2026-11-03 +08）。本次巡检**未产生新发布**：队列中唯一可发布的 `bpf-gpu-server`（Medium + DEV）已明确排入 LA 2026-11-09，24 条知乎任务仍被平台级登录 `阻塞`，补发缺口为 0（LA 11-01 已有确认发布）。因此本次交付物是**台账修复**：一条已公开但未入账的 DEV 发布。

## 关键发现：DEV 上已有一篇未入账的公开文章

- <https://dev.to/yunwei37/can-a-language-model-run-in-linux-ebpf-55ip> 于 `2026-11-01T07:55:12Z`（LA 2026-11-01 00:55 PDT）公开，源 `docs/blog/posts/qwen3-ebpf-in-kernel-inference.md`。
- 2026-11-01 那次提交（`c953647c4e`）只记录了 Medium 副本，DEV 副本从未进入 `devto.json` / `published.md` / `not-published.md`。DEV 队列行原本写「Medium + DEV」的只有 `bpf-gpu-server`，qwen3 行只写 Medium，所以这是纯粹的台账漏记，不是漏发布。
- 复核路径（两条独立证据）：公开 DEV API `GET /api/articles/yunwei37/...-55ip` 返回 `published_at 2026-11-01T07:55:12Z`、`canonical_url` 自指、tags `ebpf,llm,kernel,inference`、0 评论 0 反应；可见浏览器公开页 DOM 与源码逐项对齐——精确 H1、5 个正文 H2 与源 5 个 `##` 一一对应、12 个内联代码 span 与源完全相同、无遗漏尾段。
- 渲染差异（对后续有参考价值）：DEV 把源里的 ` ```mermaid ` 围栏渲染成内联 SVG，`pre` 数量为 0；Medium 同一篇文章只留原始 flowchart 源码。DEV 的 canonical 指向自身，不是 eunomia.dev。

## 公开页复核（持久可见 Chrome，CDP 9222）

- Medium `5001bb9afab6`（qwen3）：**两个缺陷仍在**。`h1: 0`，正文从 Mermaid 段落直接开始；图仍是原始代码。`article:published_time` 仍为 `2026-11-01T07:55:13.046Z`。编辑器路径不可达：`/p/5001bb9afab6/edit` 与 `/me/stats` 均被 Cloudflare `Just a moment...` 拦住（curl 与可见浏览器都命中同一挑战页），修复仍需要用户在持久 profile 重新登录 Medium。
- Medium `59aff1cc69a5`（gthulhu）：**结构漂移仍在**，活页面渲染 10 个 H2 + 11 个 H3，仍保留 `## What is Gthulhu?` 与旧的评级标题，而源在 `0481cdb4d1` 之后是 9 H2 / 11 H3。同时 `article:published_time` 已从 `2026-11-01T04:01:15Z` 变为 `2026-11-02T04:36:11Z`，即 11-01 那次编辑器 pass 确实触发了一次重新发布，但没有改变标题结构——此前提交里「live story 现在显示 9 H2」的说法不成立，已按实测改正。canonical 仍正确指向 <https://eunomia.dev/blog/2026/10/17/gthulhu-vllm-cpu-pressure/>。

## 会话与平台基线

- 掘金：可见会话可用，创作者中心基线 `全部 91 / 已发布 91 / 审核中 0 / 未通过 0`，草稿箱 5，与 10-21 发布以来一致，没有新的待审条目。
- 跟踪 top ten（对比 11-01 读数，全部单调上升，点赞未变，评论全为 0）：npu-kernel-driver 5125→5146 展现 / 108→113 阅读；gpu-kernel-driver 5547→5558 / 83→85；flamegraph 647→652 / 52→53；struct_ops 23→26 / 24→24；dynptr 2519→2537 / 66→68；bpf_wq 25→26 / 31→32；bpf_token 1081→1093 / 71→71；bpf_iters 963→967 / 54→55；bpf_arena 2661→2666 / 86→86；cgroup 21→22 / 31→31。
- 知乎：`https://www.zhihu.com/creator` 仍落到 `https://www.zhihu.com/signin?next=%2Fcreator`，cookie 有 `d_c0`/`_zap`/`JOID`/`crystal`/`SESSIONID`/`osd`/`_xsrf`/`__zse_ck`（另有 `BEC`、`gdxidpyhxdE`），**没有 `z_c0`**，恢复条件未变。
- 凭据：本 shell 中 `MEDIUM_API_KEY` 与 `DEV_TO_API_KEY` 均不存在；DEV 的公开 API 读路径无需 key，这是本次发现漏记文章的手段。

## 外部回声（首次记录到 Hacker News）

- `https://news.ycombinator.com/item?id=49854228`：qwen3 文章被第三方 `RohanAdwankar` 于 2026-09-26 提交（7 分 / 3 评论）。两条正面评论，一条是 `snaveen` 2026-09-29 引用 Pangram 历史链接称「100% AI generated」——这是目前唯一一条内容层面的外部批评，未回复。
- `https://news.ycombinator.com/item?id=50138075`：`bpf-gpu-server` 被第三方 `thatslast` 于 2026-10-18 提交（1 分 / 0 评论）。该文正是 11-09 排期的 Medium+DEV 源，HN 上已有条目不影响那次交叉发布，但后续若考虑 HN 需要先看这条。
- 检索口径：`hn.algolia.com` `restrictSearchableAttributes=url` 命中 5 条 eunomia.dev URL，其余 3 条为旧 tutorial 与 GPU observability 文章，均为第三方提交。

## 台账更新

- `platforms/devto.json`：新增 `devto-55ip`（`confirmed`，`date 2026-11-01`，含 QA 与渲染差异记录）、`confirmed_source_urls` 补一行、`browser_observations` 追加 11-02 的 API+DOM 复核、`last_checked` → 2026-11-02。
- `platforms/hackernews.json`：`last_checked` → 2026-11-02，`third_party_mentions` 记入两条第三方提交（含分数、评论与 Pangram 批评），`pending_verification` 改写为「仍无项目账号自身的确认提交」与「回应 Pangram 需明确授权」。
- `platforms/medium.json`：`last_checked` → 2026-11-02；`medium-5001bb9afab6` 补记两个缺陷与编辑器拦截仍在；`medium-59aff1cc69a5` 补记结构漂移仍在并把 `published_time` 变更写实；新增 11-02 `browser_observations`（`partially_blocked`）。
- `published.md`（LF）：`Last checked:` → 2026-11-02；DEV 段新增 qwen3 行、映射数 53 → 54 / 127；gthulhu 行补入 11-02 复读结论（10 H2 仍在、published_time 已变）。
- `not-published.md`（CRLF）：`Last checked:` → 2026-11-02；DEV 状态行改为最新确认项为 `55ip` 并说明 canonical 自指、公开 API 可复核；Medium 行补入标题修复仍 open 的实测。
- `sources.json`：`last_checked` → 2026-11-02，不滞后于平台台账。
- 验证器：`python3 .github/publisher/media/check_media_ledger.py --show-missing` **exit 0**；DEV 覆盖 54/127、61 条 confirmed，Medium 仍是 67/127。

## 后续

- Medium 标题修复（`5001bb9afab6`）与 canonical（<https://eunomia.dev/blog/2026/09/24/qwen3-ebpf-in-kernel-inference>，已实测 200）仍待用户在持久可见 Chrome 重新登录后完成；届时应清理残留的字面 `## `、用格式菜单转成标题块。
- gthulhu 的 Medium 副本与 9-H2 源之间的结构差异维持「不重发、不静默重同步」，等一次有意识的内容更新。
- DEV `55ip` 的 canonical 目前是自指；若要把权重压回 eunomia.dev，需要在 DEV 编辑器里改 canonical（当前编辑器登出）。
- 下一次检查点：复读掘金 top ten、复查两条最新 Medium story 的 response/echo、`bpf-gpu-server` 按 LA 2026-11-09 正常额度执行 Medium+DEV。