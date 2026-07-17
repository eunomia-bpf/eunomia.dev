---
date: 2026-07-15
slug: ebpf-ai-agent-policy-enforcement
description: AI agent 安全规则已写在 CLAUDE.md 和 AGENTS.md 里，但无人执行。对 2116 条声明的实证研究显示 64% 是 policy，大多数需要结合上下文的内核级 eBPF 执行控制。
---

# 2116 条 CLAUDE.md 和 AGENTS.md 规则：AI Agent 安全策略写了却没人执行

一个 AI coding agent 执行 `git commit`，内核没有看到任何异常：一个熟悉的进程在写熟悉的文件。仓库的 CLAUDE.md 里写着 "Run the full test suite before committing"，而 agent 在上次测试之后又改了源码。这条规则来自论文数据集中的真实语句，今天没有任何一层在执行它。

[ActPlane 论文](https://arxiv.org/abs/2606.25189)量化了这道裂缝。论文度量了开发者在 CLAUDE.md 和 AGENTS.md 里实际写下的 policy，分类了执行这些 policy 需要什么条件，并验证了 OS 级 enforcement 加上 semantic feedback 能否真正生效。论文在 intro 中的全景概括："64% of statements are policies, 83% involve system actions, and 74% depend on context that cannot be pre-defined statically."

<!-- more -->

## 开发者在 CLAUDE.md 和 AGENTS.md 里到底写了什么

讨论 AI agent 安全时，大多数人从威胁模型或攻击面入手。ActPlane 换了一个出发点：开发者已经告诉 agent 该做什么、不该做什么了，那执行这些指令需要什么？

论文调研了 64 个含 CLAUDE.md 和 AGENTS.md 的热门仓库（中位数 20K GitHub star，快照 2026-05-23），覆盖 84 份 instruction file 和 2116 条独立 statement。与此前只在文件或章节标题粒度做分析的研究不同，ActPlane 对每一条 statement 独立分类。论文提出三个问题：instruction file 主要是行为 policy 还是描述性 context？哪些 policy 需要 OS 级 enforcement，需要什么类型的检查？将这些 policy 实例化为具体可执行规则需要什么 context？

Statement 的提取经过两遍 LLM agent 辅助流水线，为每条 statement 记录源行范围和四个标签：content type、topic、enforcement level、context requirement。验证脚本确认了完整的源覆盖和逐字 span 匹配，并由独立的 Claude 和 Codex agent 交叉检查。最后，100 条分层抽样的 statement 经人工标注者独立审核，确认标签正确。

在这 2116 条 statement 中，64% 是 policy：它们要求、禁止或约束某个具体 agent 行为。其余 36% 是 descriptive context，例如架构说明或项目背景。各仓库的 policy 密度差异很大，从 0% 到 97% 不等，70.1% 的仓库 policy 数量多于 description。按行统计会遗漏这个比例，因为 description 平均 6.8 行而 policy 平均 3.6 行，按行数只显示 49% 是 policy，这正是 statement 级分析的价值所在。

为了解 policy 在各关注领域的分布，论文将每条 statement 分配到改编自先前 instruction file 研究的 12 个主题类别中，应用于 statement 粒度而非文件粒度。Development Process 和 Implementation Details 以 87% 和 85% 的比例主导 policy 版图；Architecture 以描述性内容为主，policy 仅占 23%，因为目录布局和设计摘要构成了这些章节的主体。

![64 个含 CLAUDE.md 或 AGENTS.md 的仓库中每个仓库的 policy 比例](imgs/actplane-empirical_rq1_policy_fraction.png)

数据集中的五条真实 statement 展示了 enforcement 需求的差异幅度：

| Statement | Enforcement level | Context |
|---|---|---|
| S4: "Never push to main directly." | per-event | self-contained |
| S5: "Never modify upstream source code." | per-event | project |
| S6: "Run the full test suite before committing." | cross-event | project |
| S7: "Data read from .env must not reach the network." | cross-event | project |
| S8: "Do not update dependencies without approval." | per-event | task |

## 大多数规则系统可观测，最难的是 cross-event

论文把每条 policy 分类到 enforcement waterfall 的第一个匹配层级：semantic-only 涵盖推理、沟通或输出风格；content 涵盖文件内容上的谓词；per-event 涵盖单个命令、文件访问或网络连接；cross-event 涵盖依赖跨操作的时间顺序或数据来源的 policy。Content、per-event 和 cross-event 的并集叫做 system-observable。

1361 条 policy 中只有 17% 是 semantic-only，其余 83% 是 system-observable，即内核级 monitor 原则上可以评估：38% 需要 content inspection，29% 匹配单个 OS event，16% 需要 cross-event 状态。Cross-event policy 集中在 Development Process 领域，占全部 cross-event policy 的 39.5%。

![enforcement waterfall：semantic-only、content、per-event 和 cross-event 在 1361 条 policy 上的分布](imgs/actplane-empirical_waterfall_enforcement.png)

这些 cross-event policy 遵循四种反复出现的模式。Temporal ordering 约束排序："提交前先跑测试"要求一个事件发生在另一个之后，而不是仅仅在更早的某个时间点发生过。Cross-file consistency 链接跨工件的变更："行为变更时同步更新文档"将源码编辑和文档更新耦合在一起。Multi-step workflow 执行带有验证门的发布 checklist，每个步骤必须完成后才能进入下一步。Conditional trigger 耦合操作："改了 spec 就必须同步 SDK"只在前置条件满足时触发。这些 policy 都无法从单个 event 判断：enforcement 必须记录运行了什么、以什么顺序、以及从那时起发生了什么变化。此类 policy 很普遍：81% 的仓库至少包含一条 cross-event policy，43% 的仓库横跨全部四个 enforcement 档位。

Context 依赖使 enforcement 挑战叠加。1127 条 system-observable policy 中，只有 26.4% 是 self-contained。多数（64.2%）需要 project context："the test suite"或"upstream source"必须对着具体仓库解析后才能变成可执行规则。即便是 per-event policy，如 S5 "Never modify upstream source code"，也需要先解析哪些路径构成"upstream source"，才能让文件写入检查生效。另有 9.4% 需要 task context，例如"unless explicitly requested"或"without approval"。

两种困难相互叠加：需要跨事件追踪状态的 policy 也正是那些很少指定编写规则所需的具体命令和路径的 policy。Cross-event policy 的 context 依赖高达 95%（77% project，19% task），而 content policy 为 58%。"提交前跑测试"听起来简单，但 enforcement engine 需要知道要监视哪条测试命令、哪些源码目录算"相关编辑"、以及测试是通过了还是仅仅运行了。

![context waterfall：self-contained、project context 和 task context 在 1127 条 system-observable policy 上的分布](imgs/actplane-empirical_waterfall_context.png)

一组固定的静态规则只能覆盖 self-contained 的部分。实例化其余 policy 需要先读取仓库、解释当前任务，然后才能运行任何检查。

## 为什么现有层执行不了这些规则

Prompt 指令依赖模型自身的遵从能力，但容易受 prompt injection 攻击，且在长上下文窗口中和用户的任务 prompt 争夺注意力。独立的 agent 或 LLM guard 可以在运行时检查 prompt、响应或行为轨迹，但这些检查本质上是概率性的。

Tool-call guardrail 和应用级 IFC 系统在 harness 边界确定性地拦截，但它们只能观察经过 harness 中介的请求，看不到工具开始执行之后的系统级效果。间接 subprocess、shell-out 或编译出来的二进制文件都能绕过 tool 边界。比如 agent 写了一个包含 `subprocess.run(["git", "push"])` 的 Python 脚本然后执行它：tool-call 层看到的是"run python script.py"，而不是脚本里的 `git push`。

seccomp、AppArmor、Landlock、Tetragon 等 OS 机制控制的是 resource access 而非开发者所描述的 action。它们要求静态预写策略，报错也只有一句令 agent 困惑的不透明 EPERM，不解释违反了哪条规则，也不说明如何恢复。

论文把这些层串在一起的核心洞察是：大多数规则需要存在于 agent 处的项目或任务 context，因此 agent 自身必须能将 policy 转化为具体规则；然而许多 policy 定义事件顺序或数据流，对 tool-call guardrail 不可见，因此规则必须足够具体以进行确定性 OS 级 enforcement。弥合这个差距正是 ActPlane 要解决的问题。

这些发现确立了两项设计要求。第一，policy 规范必须 agent 可编写且 OS 可执行：agent 需要用最少的专业知识将自然语言 policy 转化为具体规则，并且需要 semantic feedback 来理解违规并恢复。第二，enforcement 必须安全、隔离且高效：agent 编写的 policy 不能削弱更高权限设置的约束，不能影响其他 agent 的 policy，也不能拖慢 agent 的正常工作负载。

## ActPlane 的设计：agent 写规则，内核来执行

![ActPlane 总览：离任务最近的 agent 编写具体 policy DSL，由内核编译并执行](imgs/actplane-illustration.png)

每条 ActPlane 规则由五个部分组成：标识治理对象的 source、target operation（如 exec、write、connect）、effect、可选的 temporal gate、以及用于 semantic feedback 的 reason 字符串。论文自己的贯穿示例可以让这些组件具体化：

```
kill exec "git" "commit" unless after exec "go" "test" exits 0
```

这条规则会终止任何 `git commit`，除非 `go test` 在最近一次相关源码编辑之后成功退出过。这里省略的 reason 字段会在规则触发时向 agent 提供结构化的解释。

三种 effect 对应了 instruction 与 constraint 的区分。Block 是 pre-operation 同步拒绝，没有 TOCTOU 窗口：内核在系统调用执行之前拦截它，agent 可以改道重试。Kill 在操作开始后终止进程，防止 agent 在被 block 后切换到其他通道。Notify 只传递引导信息而不阻止操作。Constraint 使用 block 或 kill，instruction 使用 notify。

Temporal gate 让规则表达顺序关系而不仅是时间点谓词。`after ... since ...` 结构编码了一个事件必须发生在另一个事件之后：测试必须在最近一次编辑之后运行过，而不是在更早的某个时间点运行过即可。`exits N` 限定符区分成功退出和失败退出。Lineage gate 检查进程祖先关系，允许规则将操作限制在特定的进程树中。

Information-flow label 沿 fork、exec、read、write、connect 传播，且是 monotonic 的：一旦进程读取了带标签的对象，标签就不可移除。当进程读取 `.env` 时，它获得该文件的 source label。如果它之后尝试 connect 到外部 endpoint，匹配该 label 的规则就会触发并阻断连接。研究中的 S7（"Data read from .env must not reach the network"）就是这样变成可执行的 cross-event 规则的。

Policy authority 依靠 temporal trust boundary。Agent 启动前加载的规则是 higher-authority，对 agent 不可写。Agent 及其子 agent 可以在 child domain 中添加新规则或收窄现有规则，但不能削弱、移除或禁用继承的约束。运行时 delta 通过 ring buffer 到达内核，经完全在内核中的 authority checker 校验每一项变更是否符合 domain 层级，通过后才会激活。信任计算基础（TCB）由内核 enforcement engine 和 higher-authority policy 组成，该边界以下的一切都是不可信执行。由于 authority checker 完全在内核空间运行，被攻陷的用户空间 agent 无法在其 domain 层级允许的范围之外修改 active rule set。

由于 label 是 monotonic 的，长时间运行的会话有 over-tainting 风险：经过大量读取后，进程可能累积过多 label，导致后续每个操作都触发规则。在典型的编码会话中，一个进程可能读取几十个配置和源文件；没有缓解措施的话，每次读取都会添加 label，读取足够多之后每次 write 或 connect 都会匹配某条规则。ActPlane 的缓解方式是在 spawn 新子进程时清除继承的 label，将 taint 累积限制在每个进程的生命周期内而非整个会话。

607 条 policy 的数据集运用了大多数 DSL 特性，验证了语言的表达力。Effect 偏向观测：66% 的子句是 notify，29% 是 block，仅 5% 是 kill，反映出大多数 policy 监控而非阻止。Hook 集中在代码执行（60% exec）和文件变更（37% write），网络和清理操作各不到 1%。Cross-event 特性使用广泛：28% 的 policy 使用 `after/since` temporal gate，214 条使用 `unless` 编码例外。

实现规模紧凑。Userspace compiler 和 runner 约 3.2K 行 Rust 代码，eBPF enforcement engine 约 1.8K 行 BPF C 代码。BPF-LSM hook 处理 pre-operation 决策（block），tracepoint 处理观测和 post-operation 终止（kill）。Label 以 64-bit bitmask 存储在 per-object BPF map 中，传播归结为一次按位 OR。引擎支持最多 128 条并发规则，而数据集中观测到的最大仓库有 66 条 policy。关于部署架构和机制细节的更深入介绍，见[ActPlane：把 Agent Harness Enforcement 下沉到内核 eBPF](https://eunomia.dev/zh/blog/2026/05/31/actplane-pushing-agent-harness-enforcement-down-to-kernel-ebpf/)。

## 这条路走得通吗：评估结果

Policy 翻译可以覆盖整个数据集。一个 Codex agent 在首次或二次尝试中为全部 607 条 OS-enforceable policy 编译了 ActPlane 规则，607 条里只有 2 条需要语法重试。翻译使用 7 个 subagent（4 个并行）在 34 分钟内完成，将 607 条 policy 扩展为 1283 条规则行。成本约 $0.028/条，使用 1.7M 输入 token 和 177K 输出，相比之下人工编写约 $11/条。大多数 policy 结构简单：74% 最多有两个 enforcement 子句，第 95 百分位也在 152 token 以下。

上下文 enforcement 解决的 violation 远多于任何基线。在 decision-compliance benchmark（190 条 trace、38 条源自实证研究的规则）上，ActPlane 的 Decision Compliance Rate 为 75.8%：

| System | DCR |
|---|---|
| ActPlane | 75.8% |
| ActPlane-opaque（无反馈） | 53.7% |
| FIDES（tool 级 IFC） | 48.9% |
| Prompt-filter | 48.4% |
| Tool-regex | 45.3% |

基准为每条规则生成五条 trace：两条合规 trace 用于压力测试误报（一条演练接近边界的允许操作，一条执行技术上合规但类似违规的操作），三条 violation trace 反映部署中编码 agent 的执行路径类别。Direct violation 通过 tool call 触发被禁止的操作；script violation 将编写和执行拆分到子进程中；hidden violation 将副作用嵌入中性入口点后面的辅助工件中。

差距集中在 violation trace 上：ActPlane 正确解决 114 条中的 86 条，基线为 27 到 44 条，即 2.0 到 3.2 倍的改进。所有系统在合规轨迹和直接违规上表现相当；ActPlane 的优势在 script 和 hidden trace 上展开，因为被禁止的效果发生在子进程内或中性入口点后面，tool-call interception 观察不到。

Semantic feedback 是合规与盲目重试的分水岭。完整 ActPlane 产生的正确 violation-trace outcome 是无反馈引擎的三倍，86 对 27。恢复率也讲了同样的故事：semantic feedback 下 97.7%，无反馈下 31.4%。当 agent 知道操作为何被拒、需要修复什么状态时，它会改道完成任务；只收到一句空洞的拒绝时，它会通过替代路径反复重试同一个被禁止的操作。

开销可以放进日常开发工作流。论文在 policy 已加载但无规则触发的 no-hit 配置下，测量两个工作负载的端到端 overhead。第一个工作负载是重放 68 个 tool action 和 20 个 Bash 子进程的 agent trace 套件，第二个是 Linux kernel build（`defconfig` + `vmlinux`，`make -j24`）。32 条 active rule 下，ActPlane 在 agent trace 上增加 1.9%，在 kernel build 上增加 6.5%；100 条规则时仍低于 8.4%。

微基准测试定位了每系统调用开销的分布。Fork 和 exec 的绝对增加量最大，跨规则数量范围为 3 到 69 微秒，但相对于 49 和 248 微秒的原生延迟仍然适度。文件和网络操作保持轻量：open 在 100 条规则时达到 13.4 微秒，write 和 connect 保持亚微秒级。整个 tool call 的系统调用序列累积的 ActPlane 开销比单次 2 到 10 秒的 LLM 推理轮次小五到六个数量级。Policy 更新传播迅速：通过用户空间 ring buffer 提交的单规则热重载平均 26.3 微秒到达内核排空路径，立即的 exec violation 在 p50 176.4 微秒内被检测到（包含进程启动和事件传递）。

ActPlane 的优势在第二个模型上复现。DeepSeek-Pro V4 端到端复现保持系统排名不变，ActPlane 以 77.4% DCR 居首，两个模型设置之间的 per-cell 一致性产生 Cohen's kappa 0.822。

翻译质量同时驱动检测率和恢复率，因为规则过窄会遗漏违规，规则过宽会匹配合规操作。为衡量可改进性，论文将每条假阴性 trace 的证据和纠正反馈提供给翻译 agent，让其修订规则一次。用修订后的规则重新运行 28 条假阴性 trace，恢复 26 条（93%），表明该 DSL 支持迭代完善。

真实编码任务在合成 trace 之外确认了同样的模式。在 OctoBench 的 21 个任务子集上（61 条 OS-enforceable 规则，覆盖 7 个仓库），ActPlane 的 user-query reward 比无 enforcement 基线提高 9.9 分，implementation/test reward 提高 9.7 分。收益超出了合规类型检查的范围，表明 OS 级 enforcement 加 semantic feedback 不仅帮助 agent 遵守规则，还帮助它们更有效地完成任务。

外部安全基准验证了论文自身数据集之外的覆盖面。在 361 个 OpenAgentSafety 个人助理任务中，ActPlane 以 higher-authority 规则预加载 agent 生成的安全 policy，阻止了 74% 的 baseline-unsafe 行为（106 起 unsafe outcome 中拦截 78 起）。28 起未拦截案例分为三类：聊天或语义伤害（不安全行为是没有 OS 可观测工件的消息）、不安全文件内容（不在 ActPlane 的主要覆盖范围内）、以及服务端工件（效果是服务容器内的 WebDAV 上传或数据库变更，当前 hook 集观察不到）。

[ActPlane 源码](https://github.com/eunomia-bpf/ActPlane)已在 GitHub 开源。仓库中的 `policies/` 目录包含全部 64 个仓库的 607 条翻译规则，可以作为编写自己 instruction file 的起点。

## 常见问题

### eBPF 足以解决 AI agent 安全问题吗？

eBPF 对 OS event（文件写入、进程启动、网络连接等）提供确定性 enforcement，覆盖了 83% 的 system-observable policy。但剩余 17% 的 semantic-only policy（涵盖推理、沟通风格或输出质量）在内核之外，需要 harness 层和其他控制来处理。Content policy（需要读取文件内容的 38%）更适合由 linter 和静态分析器处理，不在 eBPF 的 per-event 和 cross-event enforcement 范围内。任务意图、policy authority、内容语义和 isolation 都是内核 enforcement 点周围的层需要承担的职责。

### Behavioral baseline 能替代 policy 吗？

Behavioral baseline 回答的是"这是否异常"，通过标记偏离历史行为的操作来工作。Policy 回答的是"在当前任务下这是否被允许"。一次常规 `git commit` 按 baseline 标准可以完全正常，同时违反一条项目规则；一个陌生的部署 endpoint 可以触发异常告警，但如果用户明确要求了它，操作就是合法的。Detection 和 authorization 是两个不同的问题。成熟系统会同时使用两者：anomaly signal 可以提议候选规则，再经 authority-aware review 后变为 enforced policy。

### AgentSight 会执行 ActPlane policy 吗？

不会。[AgentSight](https://eunomia.dev/zh/blog/2025/08/26/agentsight-keeping-your-ai-agents-under-control-with-ebpf-powered-system-observability/) 是系统级 profiler 和 monitor，捕捉 agent 在运行时的行为。ActPlane 是 enforcement engine，拦截或重定向违反已加载 policy 的操作。两者是互补角色：AgentSight 的运行时证据可以为 policy review 和优化提供输入，但 observation 和 enforcement 的职责保持分离。一个组织可以先用 AgentSight 发现 agent 频繁写入生产配置文件，然后编写对应的 ActPlane 规则来阻止该模式并附上禁止原因。

### ActPlane 仍然覆盖不了什么？

论文的 threat model 对边界做了明确声明。Semantically equivalent operation 可以绕过基于字符串的匹配：agent 可以使用自制 Git client 而非调用 `exec git`，但底层的 `connect` 和 `write` 系统调用仍然可见。协议边界后面的服务端效果（如服务容器内的 WebDAV 上传或数据库变更）也逃脱了当前 hook 集。文件内容语义、kernel compromise、CAP_BPF compromise 和 side channel 均不在覆盖范围内。17% 的 semantic-only policy（涵盖推理质量、沟通风格或输出格式）需要 harness 层而非内核 enforcement 来处理。

数据集本身是这篇论文最独特的贡献。在 ActPlane 之前，没有人度量过开发者到底要求 agent 遵守什么规则，也没有人分析过这些规则在 enforcement 需求上的分布。这些规则大多数已经写在成千上万仓库的 CLAUDE.md 和 AGENTS.md 里，缺失的是一个能读懂项目 context、理解当前任务、把自然语言 policy 编译成具体 kernel 级规则的 enforcement 层。[ActPlane 仓库](https://github.com/eunomia-bpf/ActPlane)包含完整实现，将内核 enforcement 与隔离、身份和内容控制并置的三层安全模型见[基于 eBPF 的不透明 AI Agent 运行时可观测与执行控制](https://eunomia.dev/zh/blog/2026/05/25/runtime-security-for-ai-agents/)。
