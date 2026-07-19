# 项目推广素材库（草案）

本文维护 AgentSight 和 ActPlane 在项目首页、README、知乎 AI Works、项目广场、社区帖、短介绍等渠道可复用的公开文案。这里只放已经适合公开传播的项目定位、能力、使用场景和链接，不放客户信息、商业策略、未公开合作或私有评估材料。

## AgentSight

### 知乎 AI Works 字段

- 项目名称：AgentSight
- 项目简介：零插桩 AI Agent 行为观测
- 项目类型：行业应用
- 项目版本：0.2.60
- 应用作品链接：https://github.com/eunomia-bpf/agentsight
- GitHub 链接：https://github.com/eunomia-bpf/agentsight

### 项目说明

AgentSight 是 AI Agent 时代的系统级观测工具。现在的 coding agent 不只是在聊天窗口里生成文本，它会调用 shell、修改文件、启动子进程、访问网络、运行测试，甚至跨多个工具链完成任务。如果只看 SDK trace、应用日志或网关记录，很多真正的系统副作用会消失在视野之外。

AgentSight 像面向 AI Agent 的 perf、top 和 strace：不要求修改 agent 代码，不要求接入 SDK，也不要求把模型流量改走代理。它用 eBPF 和 TLS 边界追踪，在应用之外关联 LLM 调用、进程树、文件操作、网络连接、token 使用和工具意图，让你看到一个 agent 到底对机器做了什么，以及这些动作是由哪段上下文触发的。

适合这些场景：
- 监控 Claude Code、Codex、Gemini CLI、OpenCode、OpenClaw 等本地或容器化 agent。
- 审计闭源 CLI 或第三方 agent，不依赖它们主动暴露日志。
- 排查 prompt injection 触发的文件、进程和网络副作用。
- 发现 token 成本异常、重复失败循环和长时间运行的 agent session。
- 为安全评审、合规留痕和事故复盘提供系统边界证据。
- 将捕获到的 LLM 调用导出为 OpenTelemetry GenAI span，接入现有观测栈。
- 使用 agentpprof 生成语义 flamegraph，分析 prompt、工具调用和系统效果之间的关系。

入口：
- 在线演示：https://agentsight.us
- GitHub：https://github.com/eunomia-bpf/agentsight
- 文档：https://eunomia.dev/zh/agentsight/
- 论文：https://doi.org/10.1145/3766882.3767169

## ActPlane

### 知乎 AI Works 字段

- 项目名称：ActPlane
- 项目简介：内核级 Agent 信息流控制
- 项目类型：行业应用
- 项目版本：0.0.9
- 应用作品链接：https://github.com/eunomia-bpf/ActPlane
- GitHub 链接：https://github.com/eunomia-bpf/ActPlane

### 项目说明

ActPlane 是面向 AI Agent 的内核级策略执行引擎。Prompt 约束、工具审批和沙箱都很重要，但它们经常停在 agent 的某一层入口：agent 可以 shell out，可以生成脚本，可以通过子进程、SDK、包管理器或容器里的命令产生副作用。ActPlane 把约束下沉到操作系统边界，用 eBPF 和 BPF-LSM 在内核态观察并强制执行 exec、文件、网络和数据流规则。

它的核心定位不是再加一层普通 allowlist，而是把信息流控制变成 agent harness 的一部分。策略可以沿进程树、fork/exec、文件读写和网络连接传播标签，表达“从秘密文件派生出来的数据不能外发”“review 子 agent 必须保持只读”“修改源码后必须先跑测试再提交”“生产数据库只能通过 migration 工具访问”这类跨事件、跨进程、带上下文的规则。

适合这些场景：
- 安全：阻止 agent 通过 shell、Python 脚本或子进程绕过 git、文件和网络策略。
- 合规：把 agent 的运行约束写成 policy-as-code，生成可审查、可复现的控制证据。
- 可靠执行：让“提交前跑测试”“改 schema 后重新生成代码”等工程流程变成可执行规则。
- 数据保护：追踪由 .env、密钥文件、生产数据或下载内容派生出的文件和进程，限制外发和扩散。
- 多 agent 协作：给 review、audit、build、release 等不同角色的 agent 分配不同运行域和权限边界。
- 反馈闭环：当策略被触发时，不只返回 Permission denied，而是把可读原因反馈给 agent，让它换一条合规路径继续完成任务。

入口：
- 文档：https://eunomia.dev/actplane/
- GitHub：https://github.com/eunomia-bpf/ActPlane
- 论文：https://arxiv.org/abs/2606.25189
- 安装：cargo install actplane
