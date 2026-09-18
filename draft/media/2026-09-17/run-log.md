# 2026-09-17 内容巡检

- 运行模式：定时巡检（eunomia-content-patrol），LA 自然日 2026-09-17。
- 调用子技能：eunomia-social-radar、juejin-publisher。
- 异常恢复：持久可见 Chrome 自 2026-09-16 00:52 PDT 起以 `rc=-5`（SIGTRAP）崩溃重启循环，共 128105 次失败重启，CDP 9222 不可用。根因是用户 HOME 内 `~/.config/chromium` 及 profile 内 108 个条目被 root 拥有，crashpad 无法创建 `Crash Reports/new` 而 abort；仅修正归属（不改动、不删除 profile）后，09-18 07:11 恢复 `Chromium ready on 127.0.0.1:9222`，Juejin 登录态保持。根因与修复步骤已写入 `.agents/skills/eunomia-content-patrol/SKILL.md` 的“Visible Browser Recovery”。
- 异常恢复：同一次 root 事故（2026-09-15 20:47-20:50）还使仓库 416 个条目（含 `.git`、`draft/`）归属 root，导致本地无法 `git add`/提交；另删除了 10 个受版本控制的符号链接/子模块条目（`AGENTS.md`、`agent.md`、`.claude/skills`、`.agents/sources/agent-skills`、`.github/seo-skills`、`docs/tutorials/third_party/libbpf` 及 4 个 `vmlinux.h`）。本次修正归属并以 `git checkout --` 还原这些已跟踪条目，未提交内容改动。
- 发布：掘金 44-scx-simple 教程，2026-09-17 使用 09-17 正常额度发布并确认公开：<https://juejin.cn/post/7686408837754142770>。提交时先进入“审核中”，同日复查审核通过、`/post/` 解析为公开页后完成公开页 QA；ledger 记为 `confirmed`。
- 已用 artifact：`draft/media/2026-09-16/44-scx-simple/juejin-body.md`（11651 字符）与 `juejin.md` 记录。
- 监测发现（16:20-16:27 PDT）：45-scx-nest 28 阅读 / 6 个 H2；46-xdp-test 23 / 6；ACRFence 48 / 10；Agent Sandbox 257 / 单份正文结构稳定；47-cuda-events 49 / 13；48-energy 32 / 14；Runtime Security 32 / 10。七篇均无审核/更新/删除标记、0 评论（`暂无评论数据`），无回复或更正待办。注意：掘金公开页阅读计数每次加载自增 1，故读数包含巡检自身访问，不与上一基线严格可比。`https://eunomia.dev/zh/tutorials/45-scx-nest/` 返回 200。
- 阻塞：知乎 `/creator` 仍跳转 `/signin`，可见会话无 `z_c0`，24 条知乎任务继续阻塞（恢复条件同队列第 14 行）。
- 补发缺口：09-16 因浏览器故障无发布窗口，新增 1 条补发额度，现共 4 条（2026-08-29、2026-09-01、2026-09-03、2026-09-16），因知乎阻塞且掘金每日上限未核销。
- 下一步：下一个 Juejin LA 自然日额度处理队列第 82 行 `docs/tutorials/43-kfuncs/README.zh.md`。

## eBPF 每日问答发布（2026-09-17 LA 额度）

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
