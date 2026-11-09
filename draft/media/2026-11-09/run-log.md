# 2026-11-09 内容巡检运行日志

运行日按 +08 计（系统日期 2026-11-09，UTC `2026-11-09T01:00Z` 起，LA 2026-11-08 16:00 PST）。本次巡检完成**一条 DEV 发布**；队列里 24 条知乎任务仍是平台级 `阻塞`，Medium 无 API key 且编辑器未登录，补发缺口 0。

## 任务选择

- 队列顶部 24 条 `排队` 任务全部是知乎，因可见 Chrome 未登录而处于平台级 `阻塞`，不能卡住其它平台。
- Medium 路径双重不可用：shell 无 `MEDIUM_API_KEY`，可见编辑器登出。
- 唯一可执行的日更任务因此是"Medium 已有副本、DEV 缺失"的英文长文。逐条核对 2026 年英文长文后，只剩 `docs/blog/posts/claude-code-analysis.md`（2026-02-13）：Medium 已 `confirmed`（`1dde03bcc7ef`），DEV 缺失。`agent-work-unit`（Weekly Analysis）、`agentpprof-semantic-flamegraph`（暂不排队）按队列规则排除。
- 发布前用 DEV 作者 API 复查：账号 67 篇、精确标题 `Reverse Engineering Claude Code's SSL Traffic with eBPF` 0 篇重复。

## 发布事实

- 准备稿 `draft/media/2026-11-09/claude-code-analysis/devto.md` 是源文**逐字节副本**：无图片，正文唯一链接（AgentSight GitHub）与 arXiv 链接已是绝对地址，无需任何改写。
- 路径：`.github/publisher/posts_queue.txt`（1 条，字段 `path` 而非 `artifact`）-> `gh workflow run publish-posts.yml -f platforms=devto -f publish_count=1 -f draft_only=false`。
- run 42543519003 于 `2026-11-09T01:04:06Z` 公开：<https://dev.to/yunwei37/reverse-engineering-claude-codes-ssl-traffic-with-ebpf-5bod>，article id 5001004，tag `ebpf`/`ai`/`linux`/`security`，reading_time 20，0 反应 0 评论；workflow 的 `Published post and updated queue [skip ci]` 提交（`0c6b30141d`）清空了队列。
- 正文对账：公开 `body_markdown`（38501 字符）与准备稿去 frontmatter/首个 H1 后（38398 字符）逐字相同，差 14 处全部是 DEV 的围栏重标注——10 个空语言围栏改 `plaintext`，2 个请求抓包块改 `http`。
- 结构对账：1 h1、12 个正文 H2（= 源文 12 个 `##`）、64 H3、6 H4、50 个渲染 `pre`（100 个围栏）、7 张表格共 45 行（源文 52 行减去 7 行分隔线）、0 图片、4696 词、结尾完整，两个外链目标（AgentSight GitHub、arXiv `2508.02736`）均存在。`<!-- more -->` 注释未泄漏为可见文本，`/bun-vfs$$/` 保持原样。
- 覆盖：DEV 59 -> 60/128，ledger 14 条 `confirmed`（73 条 confirmed entries 含 source-set 展开）。

## canonical 缺陷（连续第六次）

DEV 副本自指，站点原文 <https://eunomia.dev/blog/2026/02/13/reverse-engineering-claude-codes-ssl-traffic-with-ebpf/> 在线。根因仍在上游：`media-publisher` 的 `pages/api/publish-multi.ts` 只从请求体取 `title/content/tags/is_draft/platforms` 重建 payload，`pages/api/publish/[media].ts` 的 DEV 分支只发 `title/body_markdown/tags/published`，`canonical_url` 在任何一层都会被丢掉。修复需改该仓库端点，本仓库脚本单改无效；已把六次条目写入 `platforms/devto.json` 的 `pending_verification`。

## 会话状态

可见 Chrome 在知乎 `/creator`、掘金 `/creator/home`、Medium、DEV 编辑器均登出；shell 无 `MEDIUM_API_KEY`/`DEV_TO_API_KEY`。secret 支撑的 `publish-posts` 仍是唯一可用发布路径。24 条知乎 `排队` 任务保持平台级 `阻塞`。

## 下一次具体动作

1. 恢复 Chrome 配置里的知乎登录后，从 `53-egress-pacer` 起继续处理 24 条知乎任务。
2. 复查 Medium `5001bb9afab6` 的缺标题缺陷，以及 SchedCP / Lean 两篇仍缺 Medium 副本的英文长文；Medium 需 `MEDIUM_API_KEY` 或可见编辑器登录。
3. 若 DEV 继续日更，下一个候选是 Medium 已确认而 DEV 缺失的英文长文；2026 年队列已清空，需先有新源文或重新评估更早的 tutorial。