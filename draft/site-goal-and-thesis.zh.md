# eunomia.dev 网站目标与核心 thesis（草案）

> 状态：内部草案，不发布。最后更新：2026-07-19。
> 边界：本文只记录公开网站可以承载的目标、叙事、信息架构和转化原则。客户策略、定价、pipeline、融资、竞争情报、个人约束、内部商业模型和内部交付模型不写入本仓库，应放在私有 strategy repo。

## 一句话

eunomia.dev 是 Eunomia 的公开可信档案馆、技术作品集、开源入口和高意图合作入口。它的任务不是把所有访客都导向一个 SaaS funnel，而是让目标读者在想到 AI agent observability、runtime enforcement、eBPF infrastructure、systems safety、GPU/runtime research 时，能把 Eunomia 和维护者的判断力、工程可信度、研究 taste 联系起来。

## 核心 thesis

AI agent 的真实行为不能只从 SDK、MCP proxy、sandbox、HTTP 日志或 agent 自报事件里理解。真正影响安全、可靠性和成本的行为发生在更低一层：进程、文件、网络、exec、syscall、runtime queue、GPU launch 和 checkpoint/restore 边界。

Eunomia 的核心 thesis 是：

> AI agent 时代需要一个系统级观测与执行平面：在应用和 sandbox 之下看见 agent 实际做了什么，并把可审计的规则变成可执行的运行时约束。

这个 thesis 应该贯穿首页、Products、AgentSight、ActPlane、bpftime、blog、paper 和 llms.txt，但每个页面只承担自己那一段，不把所有战略信息塞进首页。

## 网站的四个目标

### 1. 建立公开可信度

网站首先要证明 Eunomia 不是空口叙事。可信度来自开源代码、论文、教程、真实 trace、benchmark、architecture diagram、limitations、activity reports 和长期维护记录。

对应页面和资产：

- GitHub org 与各项目 repo
- bpf-developer-tutorial
- bpftime docs 与 OSDI 论文
- AgentSight / ActPlane docs、论文和 demo
- blog / papers / reports
- llms.txt 与 llms-full.txt

### 2. 组织三支柱叙事

网站需要让访客快速看懂 Eunomia 不是一堆松散项目，而是围绕同一个系统层 thesis 展开。

三支柱：

1. **Agent Observability & Harness（主叙事）**：AgentSight 负责零插桩观测，ActPlane 负责系统层 policy enforcement，相关研究负责解释为什么 SDK、proxy 和 sandbox 之外还需要系统边界。
2. **eBPF 教育与 runtime infrastructure（信任底盘）**：教程、eunomia-bpf、bpftime、llvmbpf、wasm-bpf 证明维护者懂底层机制，也带来长期搜索流量和开发者信任。
3. **GPU / systems research（前沿差异化）**：GPU tracing、runtime extension、scheduling 和 OS 研究证明 Eunomia 的 thesis 不局限在传统 Linux observability，而是在探索 runtime 行为可观测、可扩展、可约束的更大空间。

### 3. 承接不同意图的访客

网站不应该只有一个转化动作。不同访客的当前意图不同，应该被送到不同的下一步。

高意图访客：

- 目标：发起 scoped pilot、enterprise support、design-partner 或生产集成讨论。
- 入口：Products、Services / Enterprise Support、Agent runtime infrastructure 页面。
- CTA 语言：Discuss a scoped pilot、Plan a production integration、Talk about agent runtime support。

开发者访客：

- 目标：读 docs、跑 tutorial、star repo、试 demo、安装工具、打开 issue 或 discussion。
- 入口：GitHub、Tutorials、Project docs、AgentSight / ActPlane docs、bpftime docs。
- CTA 语言：Read docs、Try the live demo、Install、Star on GitHub。

研究和媒体访客：

- 目标：理解论文贡献、引用标准 URL、找到图、数据、术语和项目关系。
- 入口：Papers、Blog、Reports、llms.txt、llms-full.txt。
- CTA 语言：Read the paper、See the artifact、Use this citation。

长期关注者：

- 目标：RSS、newsletter 或平台关注，保持低摩擦连接。
- 入口：Blog、RSS、GitHub Discussions、X / LinkedIn / 中文平台。
- CTA 语言：Follow updates、Subscribe via RSS、Join discussions。

### 4. 支持内容和分发复利

eunomia.dev 是 canonical archive，不是所有平台传播的终点。站外平台应该服务原生讨论和信任建设，站内保留最完整、最稳定、最可引用版本。

内容比例建议保持教育/分享内容多于产品叙事。教育内容负责长期受众、术语所有权和开源信任；产品叙事站在这些资产上承接高意图需求。

## 转化目标

网站的主转化不是“注册 SaaS”，而是“高质量问题进入对话”。

更合适的北极星转化：

> 有真实 agent/runtime/eBPF 场景的团队，愿意和 Eunomia 讨论一个 scoped pilot、architecture review、production integration 或 enterprise support。

这不排斥未来产品化，也不把合作形态限定成传统人力交付。公开网站只需要表达“我们能把复杂系统问题 scope 清楚、观测清楚、验证清楚、交付清楚”；更具体的内部商业和交付方式不写在本仓库。

## 定位边界：开源系统工程，不是单一产品公司

Eunomia 的统一公开定位是：一个由维护者主导的开源系统工程团队（maintainer-led open-source systems engineering effort）。它围绕 AI agent、runtime、eBPF 和 GPU/system 边界，把反复出现的系统问题沉淀为可验证的基础设施、研究成果和工程实践。

产品是这个体系里的承载形式之一，不是全部。AgentSight、ActPlane、bpftime、教程、论文、报告、demo 和服务入口共同证明一件事：Eunomia 能把复杂系统里的真实行为看清楚，把约束和评估落到可执行、可复现、可维护的工程边界上。

公开表达里应区分四种相互支撑的角色：

1. **开源项目和产品**：把重复出现的问题沉淀成可试用、可集成、可维护的工具，也是能力边界和工程质量的公开证据。
2. **论文和技术报告**：解释为什么这些问题值得做、现有方法漏在哪里、我们的机制和数据支持什么结论。
3. **内容平台**：持续展示判断力、研究 taste、工程边界感和教育能力，不把每篇内容都写成外链漏斗。
4. **服务和合作入口**：承接真实团队的 agent/runtime/eBPF 问题，进入 scoped pilot、architecture review、POC、benchmark、policy design、integration 或 production hardening。

这四种角色不是要求所有读者依次走完的转化漏斗。读者可以只学习、复现、使用开源项目或参与讨论；只有带着真实工程问题的人，才需要进入具体合作。

eunomia.dev 使用 Eunomia 的机构级语气，负责保存稳定、可引用、可验证的公开资产；个人账号使用维护者语气，负责观点、研究判断和工程复盘；组织号和项目号负责正式发布、版本与支持承诺。三者都应邀请高质量技术对话，但不混淆个人判断与组织承诺，也不把每次表达都导向注册或销售。

## 信息架构原则

### 首页

首页承担“看懂 Eunomia 是什么”的任务，不承担所有转化任务。

当前约束：

- 暂不改首页 hero 的 CTA 权重，GitHub 主入口保持现状。
- 首页可以逐步增强 thesis 表达，但不变成 SaaS landing page。
- 首页项目区应减少“平铺项目列表”的感觉，强化三支柱分组。

建议方向：

- 首屏保留开源信任信号，同时更明确写出系统层 observability / enforcement thesis。
- 三支柱区块比单纯项目卡片更重要。
- AgentSight、ActPlane、bpftime 的关系要清楚：agent 产品线是主叙事，bpftime 是 runtime foundation，tutorial/docs 是信任底盘。

### Products

Products 是高意图访客的主要路径，不应只是项目目录。

建议顺序：

1. AI Agent Observability & Enforcement
2. bpftime
3. Services / Enterprise Support

Products 页面应该回答：

- 适合谁：AI infra、AgentOps、Platform/SRE、runtime/security teams。
- 解决什么：agent 行为不可见、SDK 插桩不完整、sandbox 之外缺少执行证据、runtime 集成和性能验证困难。
- 交付什么：architecture review、POC、benchmark、policy design、integration code、production hardening。
- 下一步是什么：讨论 scoped pilot 或阅读 docs / demo。

### Services / Enterprise Support

Services 页面应该从“服务列表”升级为“合作入口”。

核心文案方向：

> Bring a real agent/runtime problem; we scope it, instrument it, benchmark it, and harden it.

页面应避免传统咨询公司的泛化表达，重点强调：

- 固定范围
- 可交付物
- 可测量结果
- 开源 core 不锁定
- 对复杂环境的系统工程能力

### Agent 产品页

AgentSight / ActPlane / agent runtime infrastructure 页面应形成三段路径：

- 低意图：Read docs / GitHub
- 中意图：Try live demo / Install
- 高意图：Discuss a scoped pilot

`agentsight.us` live demo 是重要资产，应在 Agent 相关页面和文章里更显眼，但不要让 demo 取代 docs 和论文证据。

### Blog 和文章页

Blog 主要服务解释、证据和长期发现，不应每篇都强卖。

建议：

- Agent 相关文章文末轻 CTA：Try AgentSight、Read the enforcement guide、Discuss a scoped pilot。
- 教程文章文末轻 CTA：Read the tutorial、Star on GitHub、Open discussion。
- 主题聚合页按三支柱组织：AI Agent Observability & Harness、eBPF Infrastructure、GPU & Systems Research。

## 近期小改动清单

第一轮小改动应低风险、不改现有 URL、不移动 route ownership。

可以做：

1. 新增本文档，作为网站目标和 thesis 的对齐点。
2. Products 下拉顺序调整为 Agent 产品线、bpftime、Services。
3. Services 页面统一 CTA 文案为 Discuss a scoped pilot。
4. Agent runtime infrastructure 页面显式加入 live demo / docs / pilot 三段路径。
5. Agent 相关文章文末增加轻量 CTA，优先从高价值文章开始。
6. blog 聚合页后续按三支柱做主题入口。

暂不做：

1. 不改变首页 hero CTA 权重。
2. 不移动 `/bpftime/`、`/tutorials/`、`/blog/`、`/agentsight/`、`/actplane/` 等既有 URL。
3. 不把网站改成 SaaS landing page。
4. 不把内部商业模型、客户策略或定价写进 public repo。

## 判断标准

一个页面或改动是否符合本 thesis，可以用四个问题检查：

1. 它是否让访客更快理解“系统层观测与执行平面”这个主线？
2. 它是否用证据、机制和边界建立信任，而不是只堆形容词？
3. 它是否把不同意图的访客送到合适的下一步？
4. 它是否保留开源、论文、教程这些长期信任资产，而不是牺牲它们换短期转化？

如果答案不是“是”，就先不要改 UI 或导航。
