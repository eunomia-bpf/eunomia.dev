# 2026-10-31 内容巡检运行日志

运行日按 America/Los_Angeles 自然日计算：本日正常额度为 LA 2026-10-31。唯一发布时刻为公开页 `article:published_time` `2026-11-01T04:01:15.346Z` = 2026-10-31 21:01 PDT（= 2026-11-01 12:01 +08），落在 LA 2026-10-31。巡检开始时补发缺口 0 条，本次使用正常额度 1 条、未核销任何缺口，结束后仍为 0 条。

## 队列任务：Medium gthulhu-vllm-cpu-pressure（LA 10-31 正常额度）

- 队列状态：队列顶部 `Medium 0 条`（掘金 0 条、剩余 24 条全部为知乎，其中 23 条 `排队` 被平台级登录 `阻塞` 收住）。按队列第 19 行「后续新增英文长文继续交叉排入」的口径，本次把该源补入队列并当次完成，回填为 `[x]`（`draft/plan/publishing-queue.zh.md` 第 134 行）。
- 源文：`docs/blog/posts/gthulhu-vllm-cpu-pressure.md`，规范地址 <https://eunomia.dev/blog/2026/10/17/gthulhu-vllm-cpu-pressure/>。
- 发布稿：`draft/media/2026-10-31/gthulhu-vllm-cpu-pressure/medium-body.html`（语义 HTML，1 个 H1 / 10 个 H2 / 11 个 H3 / 3 个 figure / 7 个 pre / 15 个链接，24445 字节），由同目录 `build_medium_html.py` 从源 Markdown 生成；Medium 的 Markdown 路径此前有 parser error 2012 先例，故直接使用语义 HTML。
- 发布路径：`MEDIUM_API_KEY` 在本环境不存在，改用持久可见 Chrome（CDP 9222）的 Medium 编辑器。正文用 `Input.insertText` 粘贴到空正文块，标题取 H1（Medium 标题框 43/100 字符），preview subtitle 取源第二段首 135 字符，topics 通过下拉的 `ArrowDown` + `Enter` 选定 `Linux` / `Ebpf` / `AI`。

## 偏离与修复（本次唯一异常）

- 粘贴把源中单个 bash 代码块（`taskset` 两行命令，中间一个空行）拆成相邻两个 `PRE` 块；后续用 caret + Backspace 合并时又产生 `processtaskset` 连写与重复文本，`execCommand('insertHTML', '<pre>…</pre>')` 重建仍留下 `pre` 嵌套在 `span.pre--content` 中。
- 最终修复方式：把该块内容整体选中后以 `Input.insertText` 重写 5 行原文，保留 `auto (Bash)` 标签与单一 `pre`；重载编辑器后确认持久化。此后把编辑器 205 个块与 `medium-body.html` 逐块对比，文本 0 处差异，结构（源 h1→Medium 标题、h2→H3、h3→H4）0 处差异。

## 公开页 QA（登录浏览器，完整滚动）

- <https://medium.com/@yunwei356/when-does-cpu-scheduling-help-ai-inference-59aff1cc69a5>：`<h1>` 逐字为「When Does CPU Scheduling Help AI Inference?」，正文 10 H2 / 11 H3、7 个代码块（11 / 1 / 7 / 5 / 5 / 20 / 10 行，`taskset` 块恢复为 5 行原文）、3 个带 figcaption 的 figure（Gthulhu workflow、Decode throughput comparison、TTFT comparison）、5 张图片全部 `naturalWidth > 0`、15 个正文链接、3170 词，滚动到底后尾段为源尾注 `Originally published at eunomia.dev on October 17, 2026.`。
- `<link rel=canonical>` 指向 <https://eunomia.dev/blog/2026/10/17/gthulhu-vllm-cpu-pressure/>；JSON-LD `dateCreated` 2026-10-17T00:00:28Z、`dateModified` 2026-11-01T04:01:15Z；页面 topics 为 Linux / Ebpf / AI。
- 规范页同步复核：`eunomia.dev` 原文 H1、10 个 H2、7 个代码块、3 张图片（1672 / 900 / 760 自然宽）全部正常。
- 该 story 的 response 区仍为空（0 responses），无待回复评论。

## 台账更新

- `platforms/medium.json`：置顶新增 `medium-59aff1cc69a5`（`confirmed`，date 2026-10-31，带 `canonical_url`、`source_path`、`tags`、`checked_via` 与 notes）；`last_checked` → 2026-11-01。
- `published.md`（LF）：`Last checked:` → 2026-11-01；Medium 表新增 2026-10-31 行。
- `not-published.md`（CRLF，字节级编辑后复查仍为 CRLF）：`Last checked:` → 2026-11-01；Medium 状态行改为本次已确认、队列无未完成任务。
- `sources.json`：`last_checked` → 2026-11-01（检查器要求不滞后于平台台账）。
- `draft/plan/publishing-queue.zh.md`：更新时间 → 2026-11-01；顶部新增已核定行；发布队列追加本次 `[x]` 行；补发缺口段追加 10-31 句。
- `community-feedback.md`：新增 `### 2026-10-31` 巡检块（发布、会话状态、规范页复核、下一检查点）。
- 验证器：`check_media_ledger.py` exit 0；Medium 映射 65/127 → 66/127、confirmed 70 → 71，`published.md` 无缺失 URL，汇总日期不滞后。

## 后续

- 掘金队列仍为 0 条，24 条知乎 `排队` 仍被登录 `阻塞` 收住；恢复条件不变——用户需在持久可见 Chrome 中完成知乎登录，且 `/creator` 落在创作中心而非 `/signin`，之后从 AgentCgroup 知乎任务继续。
- 下一次检查点：复读掘金 top ten，并观察本条 Medium story 是否产生 response 或外部引用。