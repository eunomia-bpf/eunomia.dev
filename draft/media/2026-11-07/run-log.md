# 2026-11-07 内容巡检运行日志

运行日按 +08 计（系统日期 2026-11-07，UTC `2026-11-07T01:31Z` 至 `01:45Z`，LA 2026-11-06 17:31–17:45 PST）。本次巡检完成**一条 DEV 发布**，队列里 24 条知乎任务仍是平台级 `阻塞`，补发缺口 0。

## 发布事实

- `docs/blog/posts/actplane.md`（2026-05-31 英文长文）此前只有 Medium 副本（`8077f9164a34`），DEV 缺失，按"后续新增英文长文继续交叉排入"补入队列。
- 准备稿 `draft/media/2026-11-07/actplane/devto.md` 是源文**逐字节副本**：源文无图片、无相对链接、外链均为绝对地址，因此不需要任何改写。此前的准备步骤总会改写相对图片地址，本文是首个零改写准备稿。
- 路径：`.github/publisher/posts_queue.txt`（1 条）-> `gh workflow run publish-posts.yml -f platforms=devto -f publish_count=1 -f draft_only=false`。
- run 42324529677 于 `2026-11-07T01:31:46Z` 公开：<https://dev.to/yunwei37/actplane-pushing-agent-harness-enforcement-down-to-kernel-ebpf-2doh>，article id 4991286，tag `ebpf`/`linux`/`ai`/`security`，reading_time 16，0 反应 0 评论；workflow 自身的 `Published post and updated queue [skip ci]` 提交（`3b4300a..68a18ec`）清空了队列。
- 正文对账：公开 `body_markdown` 与准备稿去 frontmatter 与首个 H1 后逐字相同，唯一差异是 DEV 把 12 个无语言围栏中的 3 个标注成 `plaintext`（`norm()` 归一后 `SequenceMatcher` 零差异）。
- 结构对账：1 h1、11 个正文 H2（= 源文 11 个 `##`）、6 个渲染 `pre`（12 个围栏）、1 个 5 行表格、13 个外链目标（ActPlane 仓库、arXiv 2606.25189、AgentSight、aya、E2B、Daytona、Tetragon、两个 ACM DOI）、0 张图片、References 尾段完整。
- 覆盖：DEV 57 -> 58/127。

## 档期偏差（需下次修正）

11-06 的 run 比定时晚约 10 小时（`2026-11-06T10:44:00Z` 而非约 `00:00Z`），已在 LA 11-06 02:44 PST 用掉 LA 11-06 额度；本条落在 LA 11-06 17:31 PST，因此实际是 **LA 11-06 的第二条 DEV 发布**，队列里按 +08 运行日（11-07）单独记档并在条目中写明。补发缺口不变（11-05 无任务）。下次若仍在 LA 当日发第二条同平台,应优先换平台或推迟到 LA 换日。

## 阻塞与会话状态

- 知乎 `/creator` -> `/signin`，掘金 `/creator/home` -> 站点根页，Medium 与 DEV 编辑器全部登出；可见 Chrome 的持久会话整体失效。
- `MEDIUM_API_KEY` / `DEV_TO_API_KEY` 在 shell 中缺失，因此只有 secret 支撑的 `publish-posts` 可用；Medium 编辑器阻塞时无法补齐 SchedCP 的 Medium 副本。
- 24 条知乎 `排队` 任务保持平台级 `阻塞`，未尝试发布。

## 外部回声

Algolia 按日期扫（`created_at_i > 2026-11-01`）：`eunomia` story 22 条，最新仍是 50385391（AgentSight，2 分 0 评论）；窗口内只有评论 50383008；`dev.to/yunwei37` 评论 0 条。qwen3 49854228（7 分 3 评论）与 bpf-gpu-server 50138075（1 分 0 评论）不变。无需回复。

## 下一步

1. `publish.py` 的 `publish-multi` 仍不发 `canonical_url`，本轮是**连续第四次**自指；`media-publisher` 会拒绝未知字段，需扩展端点或在该脚本里直连 DEV `/api/articles`，并在下一篇上验证。
2. 恢复持久 Chrome 的登录态（知乎、掘金、Medium、DEV 编辑器），然后从 `53-egress-pacer` 继续 24 条知乎任务。
3. Medium `5001bb9afab6` 的缺标题/裸 Mermaid 缺陷仍未复查；会话恢复后一并处理，并补齐 SchedCP 的 Medium 副本。