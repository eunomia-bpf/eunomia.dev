# 2026-09-17 eBPF 每日问答发布（America/Los_Angeles 自然日）

- 运行模式：eunomia-qa 定时问答巡检（eunomia-community-radar 路由）。任务日 2026-09-17，LA 时区 16:02 PDT 起执行。
- 候选来源：当日 25 条 opt-in Slack 归档消息（技术密度高，含 4 条实质线程）。选定问题——「GenAI 智能体 span 是否应把每次工具执行嵌套在请求它的模型调用下面，还是把模型调用与工具执行都作为同一个智能体 span 的兄弟节点？」，slug `genai-agent-invoke-agent-chat-execute-tool-sibling-tree`。与既往 32 篇无重复（既有 GenAI 篇覆盖属性稳定性、OpenInference 共存、评估证据引用，均不涉及 span 树的父子/兄弟形状）。
- 公开一手来源（全部 200 校验 + 本人逐一复核决定性引文）：
  - OTEL GenAI 语义约定（已迁至 `open-telemetry/semantic-conventions-genai` 仓库）：`docs/gen-ai/gen-ai-agent-spans.md`——`invoke_agent` INTERNAL = "GenAI agent invocation within the same process"，Span kind `INTERNAL`；`invoke_agent` CLIENT 示例点名 OpenAI Assistants API 与 AWS Bedrock Agents；plan span 节："the LLM call that generates the plan SHOULD be a child of the plan span, and **the tool or task spans produced from the plan are typically sibling operations under the same `invoke_agent` span**"（决定性引文）。
  - 同仓库 `docs/gen-ai/gen-ai-spans.md`：`chat` span kind `CLIENT`（同进程内 MAY `INTERNAL`）；`execute_tool` span kind `INTERNAL`，span 名 `execute_tool {gen_ai.tool.name}`，`gen_ai.tool.name` 为 Required。
  - `gen_ai.tool.call.id`（Recommended）："The tool call identifier"，用于把模型请求与兑现它的工具关联——即兄弟节点靠属性而非树结构关联。
  - 已发布文档 `https://opentelemetry.io/docs/specs/semconv/gen-ai/`（200）；W3C Trace Context 规范（因果一跳携带 `traceparent` 的依据）。
  - OpenAI Agents API 可观测性文档（公开 beta 引文："Tracing is enabled by default for new sessions. The public beta API does not expose tracing configuration or external trace exporters."）。
- H1 处理：初版 H1 含下划线/反引号（`invoke_agent`、`execute_tool`），按发布标准改写为普通词（"agent span" / "tool execution" / "model call"），避免站点 H1 流水线剥离下划线导致 live-content 校验失败与额外重部署；正文保留反引号代码。
- 隐私核对：页面仅引用公开文档与规范引文，无人员名、工作区/频道/消息链接、时间戳、私有日志或可检索回原帖的措辞；社区讨论部分角色化（使用者/维护者），去部署细节。
- 提交与推送：commit `2d029c8d0 docs(ebpf-qa): genai-agent-invoke-agent-chat-execute-tool-sibling-tree (2026-09-17)`。本地 `content.test.ts` 既有无关用例在本机竞争下会卡（已记录前飞故障），按 prompt 的已发布复核路径：先自行提交四个 scoped 路径（4 files, 108 insertions）并推送，再跑校验器 `skipped_already_published` 复核路径（跳过本地 build/content-test，直接验证线上页面）。
- 推送时远端已前进到 `b8f55130f`（agent-skills gitlink + seo-data + research CXL 文档，与 4 个 docs 路径不相交）；用 `git reset --soft origin/main`（只移动指针、不动工作树）+ 重新提交 + 推送。`2d029c8d0` 已在 `origin/main`。
- 校验器：`receipt-2026-09-17.json` `status: published`（0600），全部检查 ok（`remote_contains_commit: ok`、`public: ok`、`privacy: ok`、`index_links: ok`）；`content_test`/`build`/`render`/`commit_push` 走已发布复核路径。
- 线上核验：EN/ZH 页面 200，H1 命中；两个 `/ebpf-qa/` 与 `/zh/ebpf-qa/` 索引均链接 slug。
- 并发文件保持原样（未暂存、字节不变）：`.agents/sources/agent-skills`（gitlink）、`app/lib/site-config.generated.ts`、`app/next.config.mjs`、`draft/media/2026-09-13/run-log.md`，以及远端前进带来的 `.github/seo-data/*`、`docs/research/*` 工作树滞后状态。
- 原始快照临时目录（chmod 700，`snapshot=ok bytes=10767 messages=25`，`/tmp/qa-snap.QdbayL`）已删除；未提交任何快照原文。
- 当日社区讨论（四个实质线程，均已匿名化进页面）：① GenAI 智能体 span 形状（本页问题）；② 托管 harness（OpenAI Agents API）的可观测性锁定——CLIENT 版 `invoke_agent` 的树边界；③ OBI 配置 v1→v2 迁移的全有或全无行为与维护者 `--allow-partial` 方向 + Helm chart 缺口；④ Node.js 哨兵性能承接（草稿 PR #3357、排除后 pod 悬挂、按消费方 gate 的未决边界）。
