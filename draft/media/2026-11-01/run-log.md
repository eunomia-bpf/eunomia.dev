# 2026-11-01 内容巡检运行日志

运行日按 America/Los_Angeles 自然日计算：本日正常额度为 LA 2026-11-01。本次巡检**未产生任何发布**，因此没有需要归一化的发布时刻。巡检开始时补发缺口 0 条，结束时仍为 0 条（沿用 10-22～10-30 的记账口径：掘金/Medium/DEV 无可执行行，知乎按平台级登录 `阻塞` 整体挂起，不计入缺口）。

## 为什么今天没有发布

- LA 2026-11-01 的 Medium 正常额度已在上游被消费：`c953647c4e`（docs(media): publish qwen3 to Medium, queue bpf-gpu-server for 2026-11-09）把 `docs/blog/posts/qwen3-ebpf-in-kernel-inference.md` 经 Medium API 发出（`medium-5001bb9afab6`，`date: 2026-11-01`，`confirmed`）。同一条提交把唯一的另一个 Medium/DEV 队列项 `docs/blog/posts/bpf-gpu-server.md` 明确排入 **2026-11-09（下周一）** 的正常额度。
- 于是本环境内不存在「今天可执行」的队列行：掘金 0 条、Medium 0 条、DEV 0 条，剩余 24 个平台任务全部是知乎且被平台级登录 `阻塞` 收住。按 No-Filler Rule，不为填额度而提前发布 11-09 已排期的内容。
- DEV 侧独立复核亦不可执行：`dev.to/new` 只渲染登录墙，本 shell 中 `DEV_TO_API_KEY` 不存在（`MEDIUM_API_KEY` 同样不存在，与队列中「本地 `.env` 已恢复」的记录不一致——那次 API 发布确实成功过，但对本次巡检的 shell 不成立，后续运行需自行探测，不继承任一结论）。
- 因此本次的价值在核验与台账修复，而非发布。

## 公开页 QA（持久可见 Chrome，CDP 9222，完整滚动）

- <https://medium.com/@yunwei356/can-a-language-model-run-in-linux-ebpf-5001bb9afab6>：正文与源一致——渲染 5 个正文标题，逐一对应源 `docs/blog/posts/qwen3-ebpf-in-kernel-inference.md` 的 5 个 `##`；1 个 `pre`；4 个正文外链（源的 3 个目标 + Medium 自身会员推广图链接）；约 1,410 词；尾段与源逐字一致。JSON-LD `dateCreated`/`dateModified` = `2026-11-01T07:55:13Z`；`link[rel=canonical]` 指向 Medium 自身；response 区为空（0 responses）。
- 渲染位置口径：Medium 把源的 `##` 渲染成 `h3`（不是 `h2`），可见标题才是 `h1`。因此「`h2: 0`」不是缺陷，缺失的 `h1` 才是。

## 本次发现的两个真实缺陷

1. **该 story 没有可见标题**（`h1: 0`，页面无 `graf--title` 块）。成因：Medium 没有独立标题输入框，可见标题是正文第一个带 `data-testid="editorTitleParagraph"` 的块（在 `59aff1cc69a5` 编辑页确认其形态为 `h3.graf--h3.graf--leading.graf--title`）；API 的 `title` 字段只影响列表与 SEO，不进正文。本次修复未完成：在 `/p/5001bb9afab6/edit` 拆出新的首个空块并输入标题文本后，`## ` 快捷键未把该块转换为标题块（字面 `## ` 残留并已自动保存），随后 medium.com 开始返回 Cloudflare「Performing security verification」拦截页、`/p/5001bb9afab6/edit` 重定向到 `/m/signin?redirect=…&loginType=default`，编辑器不可达。**修复处于 open 状态**，需要可见 profile 重新登录 Medium；届时应清理残留的字面 `## `，并用该块的格式菜单转成标题块，而不是再试 `## ` 快捷键。
2. **Mermaid 图只剩源码**。源有 1 个 ` ```mermaid ` 围栏（第 21–35 行），Medium 不渲染 Mermaid，公开页只有 1 个 `pre` 装 flowchart 原文。已把「Medium 不渲染 Mermaid」「API 发布必须自带可见 H1」两条写进 `medium-publisher/references/platform-preferences.md`；如该图承担论证，后续改为导出图片再上传。

## 一个被纠正的误报（本次最重要的复核）

- 10-31 之后本次首先怀疑「eunomia.dev 博客文章页整站 404」：`/blog/2026-10-17/gthulhu-vllm-cpu-pressure/` 等 URL 返回 404、H1 为 `This page does not exist`，而 `/`、`/blog/`、`/tutorials/50-tcx/`、`/papers/` 为 200，`sitemap.xml`（884 loc）仍列出它们，`/blog/posts/<slug>/` 却返回 200——一度被判断为导出/路由不匹配。
- 复核结论：**误报，站点无缺陷**。真实路由是斜杠日期形式，`https://eunomia.dev/blog/2026/10/17/gthulhu-vllm-cpu-pressure` 与 `/blog/2026/09/24/qwen3-ebpf-in-kernel-inference` 均返回 200、`<title>` 正确、`rel=canonical` 自指；从 sitemap 的 112 条日期型 blog URL 中抽 14 条（随机 10 条 + 两条被跟踪文章的中英 4 份），全部 200。真正 404 的是连字符日期形式 `/blog/2026-10-17/<slug>/`，而站点从未使用它——`grep` 在 `docs/`、`app/`、`draft/` 与台账中都找不到连字符形式的 blog 链接（唯一一处在本次巡检自己写的 `community-feedback.md` 块里）。误读根因：终端/工具显示会把路径段 `/2026-10-17/` 折叠成 `2026-10-17`，照屏幕重敲 URL 就悄悄改了路由。`/blog/posts/<slug>/` 的 200 也不是第二种路由，而是 `app/scripts/generate-legacy-redirects.mjs` 按设计生成的 meta-refresh 跳转桩。`/blog/2025/`、`/blog/2026/` 确实 404（没有导出按年归档页）。
- 已把「报告页面失效前先逐字节核对 URL 形态」写进 `eunomia-social-radar/SKILL.md` 第 1 步；`/blog/posts/gthulhu…/` 的 meta-refresh 桩自指 canonical 指向带斜杠的规范页，属正常。

## 其他发现

- **署名/同步漂移**：`0481cdb4d1`（2026-11-01）把 `docs/blog/posts/gthulhu-vllm-cpu-pressure.md` 改为 9 H2 / 11 H3（grep 已核实），而 10-31 的 Medium 拷贝 `medium-59aff1cc69a5` 仍镜像旧的 10 H2 结构。决定：不重复发布、不静默重同步，只记录漂移，留待一次有意识的内容更新 pass。
- 会话状态：掘金创作者中心登录可用，基线不变（全部 91 / 已发布 91 / 审核中 0，草稿箱 5），跟踪 top ten 增量单调向上，无待回复/待转发；知乎仍无登录（`/creator` → `/signin?next=%2Fcreator`，无 `z_c0`）；LinkedIn/X 登出但公开页可读，无新评论；Medium 可见会话在本次运行后期失效（Cloudflare 拦截 + `/m/signin` 重定向），下一次 Medium 编辑需用户先重新登录。
- 杂项：一次按键误落到 `https://x.com/compose/post`，留下草稿文本 `guage model run in Linux eBPF? `；该标签页已关闭，**未发布任何内容**，如再看到该草稿可直接清空。

## 台账更新

- `published.md`（LF）：`Last checked:` → 2026-11-01；Medium 表新增 2026-11-01 的 `medium-5001bb9afab6` 行（含 QA 证据、两个渲染缺陷、canonical 留空的原因）；同步刷新三处过期口径（Medium 映射 66 → 67、LinkedIn 11/124 → 11/127、DEV 53/124 → 53/127，均以 `check_media_ledger.py` 当前输出为准）。
- `not-published.md`（CRLF，字节级编辑后复查 `CRLF kept: True, lone LF: 0`）：Medium 状态行改为最新确认项为 `5001bb9afab6`，并记下标题修复 open 与 Gthulhu 结构漂移。
- `platforms/medium.json`：`medium-5001bb9afab6` 的 `notes` 补入 QA 结果、两个缺陷、canonical 待设值（`<https://eunomia.dev/blog/2026/09/24/qwen3-ebpf-in-kernel-inference>`，已实测 200）与 open 的编辑器修复；JSON 重新校验通过（8 条，`last_checked` 2026-11-01）。
- `community-feedback.md`：radar 的 `### 2026-11-01` 块保留，父任务事后更正其两处事实——(a)「dev.to 因父任务发布中而未观察」→ 实为**没有发布在途**，额度已被上游消费、`bpf-gpu-server` 排 11-09；(b)「可见 Medium 会话已登录」→ 运行末期已失效，需重新登录。整条「博客 404」缺陷块改写为误报复核结论，其下游两句（canonical 指向 404、gthulhu.org 是唯一可读副本）一并作废。
- 技能：`eunomia-social-radar/SKILL.md` 增加 URL 逐字节核验规则；`medium-publisher/references/platform-preferences.md` 增加 Mermaid 不渲染、API 发布需自带 H1、Cloudflare 拦截会连带登出编辑器三点。
- `sources.json`：`last_checked` 保持 2026-11-01，不滞后于平台台账。
- 验证器：`python3 .github/publisher/media/check_media_ledger.py` **exit 0**（此前 `ERROR: confirmed entry medium/medium-5001bb9afab6 is missing from the snapshot` 已消除）。

## 后续

- 待办（Medium，需要会话）：在持久可见 Chrome 重新登录 Medium 后完成 `5001bb9afab6` 的标题修复（清理残留 `## `、用格式菜单转标题块）、设置 canonical、重跑公开页完整滚动 QA；`medium.json` 的 notes 与 `published.md` 行随之从 open 改为 done。
- 知乎 24 条 `排队` 仍被登录 `阻塞` 收住；恢复条件不变——用户需在该可见 Chrome profile 完成知乎登录且 `/creator` 落在创作中心，之后从 AgentCgroup 知乎任务继续。
- 下一次检查点：复读掘金 top ten、复查两条最新 Medium story 的 response/highlight，并按新规则逐字节核对 canonical URL 形态；`bpf-gpu-server` 的 Medium+DEV 任务按排在 LA 2026-11-09 的正常额度执行。
