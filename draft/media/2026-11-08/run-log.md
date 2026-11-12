# 2026-11-08 内容巡检运行日志

运行日按 +08 计（系统日期 2026-11-08，UTC `2026-11-08T01:00Z` 起，LA 2026-11-07 17:00 PST）。本次巡检完成**一条 DEV 发布**，并修好一处源文缺陷；队列里 24 条知乎任务仍是平台级 `阻塞`，补发缺口 0。

## 发布事实

- 候选：`docs/blog/posts/ai-lean-proof-checking-cost.md`（2026-11-06 英文长文，上游在上一轮巡检之后合入），Medium 与 DEV 均无副本，按"后续新增英文长文继续交叉排入"补入。发布前用 DEV 作者 API 复查：账号 68 篇、精确标题 0 篇重复。
- 源文修复：`95d9e1b9eb` 的编辑改写删掉了图片行，却保留了讲解"Figure 1"、蓝点和红线的段落，站点与中文版都不再渲染这张图，而 `docs/blog/posts/imgs/ai-lean-checking-fig1.png` 仍在仓库内且线上 200。已在英文第 19 段之后、中文第 19 段之后各补回原 alt 文本的图片行（`e709995832`）。
- 准备稿 `draft/media/2026-11-08/ai-lean-proof-checking-cost/devto.md` 与源文只差两处**必需的改写**：图片与 ProofServe PDF 从相对路径改成 `https://eunomia.dev/_content-assets/...` 绝对地址。其余逐字相同。
- 路径：`.github/publisher/posts_queue.txt`（1 条）-> `gh workflow run publish-posts.yml -f platforms=devto, ... -f publish_count=1 -f draft_only=false`。
- run 42432328544 于 `2026-11-08T01:00:47Z` 公开：<https://dev.to/yunwei37/ai-now-writes-lean-proofs-faster-than-lean-can-check-them-51mh>，article id 4996302，tag `ai`/`performance`/`programming`/`opensource`，reading_time 6，0 反应 0 评论；workflow 的 `Published post and updated queue [skip ci]` 提交（`5061b85d70`）清空了队列。
- 正文对账：公开 `body_markdown` 与准备稿去 frontmatter 与首个 H1 后**逐字相同**（8211 字符，无围栏改写），这与 ActPlane 那次的 3 处 `plaintext` 标注差异不同。
- 结构对账：1 h1、7 个正文 H2（= 源文 7 个 `##`）、1 个渲染 `pre`、0 表格、5 个外链目标（Cerebras、SemiAnalysis、ProofServe 仓库、v0.1.0 release、ProofServe PDF）、图片 800x503 且 alt 文本保留、结尾完整。
- 覆盖：DEV 58 -> 59/128，ledger 13 条 `confirmed`。

## 站点部署缺口（已修复）

源文图片修复的 push（`e709995832`）触发的 Deploy Static App run 42432319860 被 `Check current production revision` 判为 stale 并**跳过构建**：发布 workflow 的 bot 提交 `5061b85d70` 已把 `main` 推进到该 SHA 之后。`[skip ci]` 只作用于 CI 触发，不阻止下一次 push 的比较基准更新，因此修复后的页面在线上仍是旧 HTML。已用 `gh workflow run app-static-pages.yml --ref main`（run 42433232871）补跑，并核对线上 `fig1` 引用。

## 会话状态（延续上一轮）

- 可见 Chrome 在知乎 `/creator`、掘金 `/creator/home`、Medium、DEV 编辑器全部登出；shell 中 `MEDIUM_API_KEY` / `DEV_TO_API_KEY` 仍缺失。secret 支撑的 `publish-posts` 是唯一可用发布路径，DEV 公开页仍可匿名阅读，故 QA 不受影响。
- 24 条知乎 `排队` 任务保持平台级 `阻塞`。

## canonical 缺陷根因（新证据）

连续第 5 次自指。此前记录为"`publish.py` 不发 `canonical_url`"，本轮读到了更上游的原因：`media-publisher` 的 `pages/api/publish-multi.ts` 从请求体只取 `title, content, tags, is_draft, platforms` 五个字段重建 payload，`canonical_url` 会被丢弃；`pages/api/publish/[media].ts` 的 DEV 分支也只发 `title/body_markdown/tags/published`。因此修复必须在 `media-publisher` 端点内加字段并透传到 `article.canonical_url`，单改本仓库脚本无效。

## 外部回声

Algolia 按日期扫（`created_at_i > 2026-11-05`）：`eunomia` story 仍只有 50385391（AgentSight，2 分 0 评论），窗口内 0 条新评论，`dev.to/yunwei37` 0 条。DEV 近期 6 篇全部 0 反应 0 评论。无需回复。

## 下一步

1. 在 `media-publisher` 的 `publish-multi` 与 DEV 分支加入 `canonical_url`，下一篇发布时验证。
2. 恢复持久 Chrome 登录态（知乎、掘金、Medium、DEV），从 `53-egress-pacer` 继续 24 条知乎任务。
3. Medium `5001bb9afab6` 缺标题/裸 Mermaid 缺陷仍未复查；SchedCP 的 Medium 副本仍缺。