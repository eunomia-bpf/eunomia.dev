# eunomia.dev SEO/GEO 分析与执行计划

> 状态：内部计划文档，不发布。最后更新：2026-07-16。
> 数据来源：站内基建审计（app/ 代码级）+ 16 组关键词 SERP 实测 + 竞品内容调研。

## 一、现状分析

### 1.1 站内基建：底子好，有明确的洞

做得好的部分不用动：每页都有集中生成的 title/description/canonical/OG/Twitter 标签（`app/components/SeoHead.tsx`），description 有 frontmatter → 正文摘录 → 标题的三级兜底；JSON-LD 齐全（Organization + BreadcrumbList + Article/TechArticle + SoftwareSourceCode）；中英 hreflang 双向完整（head 和 sitemap 两处）；sitemap/RSS 双语言、lastmod 取自 git；老 MkDocs URL 有 meta-refresh + canonical 跳转桩；404/search 正确 noindex；trailing slash 一致。

按影响排序的问题：

| 优先级 | 问题 | 位置 | 影响 |
|---|---|---|---|
| P0 | og:image 全站共用一张 SVG，社交平台（Twitter/LinkedIn/Slack）不渲染 SVG，所有分享链接无预览图 | `app/lib/seo.ts:15-19`、`app/scripts/generate-static-metadata.ts:34-74` | 每一次社媒分享的点击率 |
| P0 | 转载无 canonical：dev.to/Medium 转载页没有指回 eunomia.dev，publish 链路（`.github/publisher/publish.py` → media-publisher.vercel.app）payload 里没有 canonical_url 字段 | `.github/publisher/publish.py:197-203` | 旗舰内容的排名被自己的转载吃掉 |
| P1 | sitemap 含 25 个 noindex 的旧 `/blogs/*` URL，矛盾信号 | `generate-static-metadata.ts:76-133` + `page-factories.tsx:130` | 爬虫信任度 |
| P1 | blog 无 tag/分类/分页落地页，51 篇文章一张平铺列表，没有可索引的主题聚合页 | `app/components/BlogListing.tsx` | 主题权重聚合（topic cluster）缺失 |
| P1 | `llms-full.txt` 名不副实：只是加长索引，指向 GitHub raw markdown，不含正文；llms.txt/llms-full.txt 均手工维护，无生成脚本、无 CI 校验 | `app/public/llms*.txt` | GEO：AI 引擎无法零抓取作答；staleness 风险 |
| P2 | 无 FAQPage/HowTo 结构化数据（AI 答案引擎最爱抽取的两类） | `SeoHead.tsx` | GEO 富结果 |
| P2 | 图片 alt 不强制、无 lazy loading；标题无可点击锚点；redirect 桩缺 robots noindex | `render.ts` / `sanitize.ts` / `generate-legacy-redirects.mjs` | 小分项 |
| P2 | 43/51 篇英文 blog 缺手写 description（自动摘录兜底，质量参差）；个别 H1 带 `**` 星号、英文文件用中文标题 | `docs/blog/posts/` | 摘要质量 |

### 1.2 站外可见度：赢老词，输新词，输给自己的转载

- **赢的**：老 eBPF 教程词很能打（"eBPF tutorial libbpf" 排 #2/#5/#7；"eBPF LSM tutorial security" 的 `/tutorials/19-lsm-connect/` 排 #4）；自造品牌词 ActPlane 全覆盖。
- **输给自己的**："AI agent observability eBPF" 的 SERP 上是 arXiv、dev.to、Medium、ResearchGate，eunomia.dev 原文完全不出现；连搜 "AgentSight" 品牌词，原文也只排 #7，落后于 arXiv/ACM/Medium/GitHub repo。已核实 dev.to 转载页无 rel=canonical、无 "originally published at" 声明。
- **完全缺席的高价值词**："AI agent runtime security"（Noma/Zenity/Palo Alto/Okta/Microsoft 占据）、"how to monitor Claude Code"（SigNoz/Jellyfish 占据，而 AgentSight 就是干这个的）、"AI agent sandbox security"、"MCP security monitoring"、"prompt injection detection runtime"（Sysdig/Palo Alto）、"OpenClaw security"（CrowdStrike/Zenity/SentinelOne，零 eBPF 角度）、"Falco vs Tetragon"。
- **bpftime**：GitHub #1、arXiv #2、Medium #3，eunomia.dev 文档页不上榜。

### 1.3 GEO：观点赢了，链接输了

AI 搜索的答案已经大量复述 AgentSight 的叙事（boundary tracing、semantic gap、<3% overhead 原样出现在生成答案里），但引用全部指向 arXiv/dev.to/GitHub，不指向 eunomia.dev。llms.txt 本身质量不错（品牌消歧、项目卡片、引用信息齐全），短板在 llms-full.txt 不含正文和维护无自动化。

### 1.4 竞品版图与我们的独占位

两个阵营：安全厂商（ARMO/Zenity/CrowdStrike/Sysdig/Palo Alto/Wiz/Noma）写 "runtime security / MCP security / OpenClaw"，观测厂商（Langfuse/LangSmith/Datadog/Arize/SigNoz）用 SDK/OTel 插桩做 "agent tracing"。**没有任何一家做免插桩的内核级 agent 观测，也没有任何一家做内核级 enforcement**。ARMO 的文章明确承认 semantic gap 存在但说 eBPF 解决不了，这正是我们论文正面回答的问题。独占位就三句话：管不了改不了的第三方 agent（Claude Code/OpenClaw/Gemini CLI）也能观测；intent 和 syscall 在内核边界关联；不止检测，还能执行（ActPlane in-kernel IFC）。

### 1.5 品牌定位修正（基于资产盘点）

对外定位是 **AI Agent Observability / Harness 平台**，不是"安全厂商"：安全（enforcement）是平台能力之一，叙事主轴是"看得见、管得住、跑得快"。资产盘点（repo 内 + workspace + 公开足迹三路扫描）显示品牌实际站在三根支柱上，网站叙事和首页应该按这个结构组织：

1. **Agent Observability & Harness**（主叙事）：AgentSight（含 `top` live 模式、agent-session crate、agentpprof 语义火焰图）+ ActPlane + 一批论文（AgentCgroup、ACRFence、RewardGuard、Sandlock、AgentCap、Fork/Explore/Commit）。
2. **eBPF 教育与基础设施**（流量基本盘）：bpf-developer-tutorial（4.2k star）、bpftime、llvmbpf、wasm-bpf。这是现在搜索排名最好的资产，负责把流量引向支柱 1。
3. **GPU / 系统研究**（前沿差异化）：bpftime GPU offload、gpu_ext、gPerf、eInfer/ProfInfer、HetGPU/GPUOS、CUPTI/NVBit 教程。支撑 "eBPF 从内核延伸到 GPU" 的长线故事。

当前网站问题：首页和 blog 把三根支柱平铺混排，访客看不出主叙事。改进方向：首页头部一句定位陈述 + 三支柱分区；blog 聚合页按支柱组织（对应第二节批次 B 的 tag 落地页，正好一柱一页）。

## 二、站内修复计划（分三批，每批一个 PR）

**批次 A（本周，最高杠杆，不新增页面）**
1. og:image 换成构建期生成的 PNG（至少一张站级默认图；有余力按 section 出 3-4 张变体）。
2. publish 链路加 canonical：**整条搁置**（2026-07-16 owner 决定 media-publisher 先不改）。核实结论备档：media-publisher 的 publish-multi 端点只接受 title/content/tags/is_draft/platforms，未知字段被静默丢弃，所以只改 publish.py 是死代码（首次尝试已从 PR #115 撤出）。将来重启时的正确顺序：先改 media-publisher 服务端转发 canonical，再给 publish.py 加最小传参（URL 推导读构建产物，不复刻 slug 逻辑）。在此之前 canonical 全靠第 3 条的手动补 + 新转载前先在后台设置。
3. **手动补历史转载的 canonical**（不用等代码）：dev.to/Medium 后台把 AgentSight 等已转载文章逐篇设 canonical 指回 eunomia.dev。这是零成本、收益最大的单个动作。
4. sitemap 剔除 noindex 的 `/blogs/*`。

**批次 B（两周内，内容基建）**
5. blog tag 落地页：先手工建 3 个主题聚合页（AI Agent Security / eBPF Tutorials / GPU & Systems），IA 配置进 mkdocs.yaml，不动任何既有 URL。
6. llms.txt/llms-full.txt 生成化：并入 `generate:content-artifacts`，llms-full.txt 嵌入各页正文或高质量摘要，CI 加校验。
7. 给 8-10 篇高价值旧文补手写 description（AgentSight 论文文最优先，它现在连 description 都没有）；修掉 H1 里的 `**` 和英文文件的中文标题。

**批次 C（一个月内，锦上添花）**
8. FAQPage JSON-LD（教程和带 FAQ 段的 blog）、HowTo（教程）。
9. 图片 alt 校验进 CI、lazy loading、heading autolink、redirect 桩加 noindex。

**批次 D（存量 blog 修复，来自 2026-07-16 全站审计）**
10. ~~依附第三方的两篇 GPU 深度文（iaprof-analysis、nvidia-open-driver-analysis）加自家框架段 + "What This Means for eBPF-Based GPU Observability" 收尾节 + description~~（已完成，待进 PR）。
11. 两篇近重复的 GPU profiling survey（gpu-profile-tool-impl ↔ gpu-profile-tools-analysis，相隔 10 天同关键词自蚕食）：一篇定位"全景综述"、一篇定位"实现内幕"，显式互链；顺带修 analysis 篇的 ZH 章节顺序倒置和 "other co-processor" 语病。
12. runtime-security-for-opaque-ai-agents：H1 从 100 字符压到 ~70（如 "Runtime Security for AI Agents with eBPF: Beyond Sandboxes and Approvals"），description 429→155；与 actplane.md 的"三层约束"论证收敛为总览文独占、机制文回链。
13. ACRFence 文标题去自造词前置（改 "Preventing Semantic Rollback Attacks in AI Agent Checkpoint/Restore" 类，代号进正文）；agentpprof/actplane 的超长 description（380c/326c）压到 150-160。
14. legacy 卫生批：英文文件实为中文的两篇（how-to-write-rust-in-wasm、lmp-eunomia）、EN 文件中文 H1（how-to-write-c-in-wasm）、H1 带字面 `**` 的两篇（cxlmemtest、osdi-sosp-obser-debug）、coolbpf zh 缺 `<!-- more -->`；全部旧文系统性缺 description（除最近 8 篇），按流量优先级分批补。
15. NVIDIA 篇正文存量 52 个无空格 em dash 清理（tie-back pass 按"不动正文"约束未处理）。

## 三、Blog 写作计划（资产优先，双轨制）

写作规范：全部走 `tech-blog-writer`（新写）+ `blog-writing-style`（审校）两个 skill；写作一律用 claude-opus-4-6[1m]（Agent 工具钉不了版本，用 `.claude/agents/prose-writer.md` 或无头 CLI `claude -p --model 'claude-opus-4-6[1m]'`）；中英双语同结构；description 150-160 字符；title ≤60 字符关键词前置；每篇 2-3 条站内互链 + 指向 GitHub repo。**草稿一律先落 `draft/blog/`，人工审核后再移入 `docs/blog/posts/` 走 PR 发布。**

**体裁原则（2026-07-16 定）：blog 与产品用法解耦。** blog 只承载论点、数据、设计决策与战报；安装、命令、操作步骤类内容一律进产品 docs（AgentSight 教程写在 agentsight repo 的 docs 里，构建时同步到站内），blog 里只留一条"上手看这里"的链接。教程型选题（BTF/兼容性/TLS 抓包等 GitHub issue 高频主题）相应改道 docs/FAQ 页，不占 blog 位。数据文必须用真实测量数据，先出数再动笔。

### 轨道 A：资产变现（优先——已经做完的工作，只差一篇 blog）

资产盘点来自三路扫描：repo 内（agentsight/actplane/bpftime 同步文档、draft 存量）、workspace 项目、公开足迹（arXiv author 页、GitHub org、讲座记录、个人网站）。按价值排序：

| # | 资产 | 来源 | 工作标题 | 状态 |
|---|---|---|---|---|
| A1 | AgentSight `top` live 模式 | agentsight/docs/usage.md | ~~blog 已砍（2026-07-16）~~ CLI 用法进 agentsight docs | 已砍 |
| A2 | bpftime GPU offload | bpftime/example/gpu + draft/eBPF-for-GPU | ~~取消~~ 与已发布 gpu-observability-challenges（2025-10-14）重复；增量 benchmark 数字并入旧文更新 | 已取消 |
| A3 | Claude Code 免插桩监控（AgentSight 核心场景 = SEO 需求 #1） | 真实 trace（ai-agent-trace-backups 或新采会话） | 改为数据文："追踪真实 Claude Code 会话，hooks 看不到的有多少"；教程部分迁 agentsight docs quickstart | 挂起，等真实测量 |
| A4 | gpu_ext（GPU 驱动里的 eBPF struct_ops，arXiv 2512.12615 + LPC'25 talk，最高 4.8x） | ~/workspace/gpu/gpu_ext + arXiv | Making the GPU Driver Programmable with eBPF | 第二批 |
| A5 | gPerf（on-/off-GPU 归因 profiler，1770 行论文稿） | draft/eBPF-for-GPU/gpuprofile.md | gPerf: Finding the Host-Side Stalls GPU Util Bars Hide | 第二批 |
| A6 | eInfer/ProfInfer（分布式 LLM 推理逐请求追踪，eBPF'25） | 论文 | Tracing One LLM Request Across CPU, GPU and Nodes | 第二批 |
| A7 | agent-session crate（Claude/Codex/Gemini 会话统一 IR） | agentsight/agent-session | One IR for Every Coding-Agent Log | 第二批 |
| A8 | ActPlane rule language + cookbook（政策写法实操） | actplane/docs | Writing Information-Flow Policies for AI Agents | 第二批 |
| A9 | multikernel 现成英文草稿 | ~/workspace/multikernel | Rethinking Multikernel Architecture | 移植即可，最低成本 |
| A10 | bpf-benchmark / KOperation、MVVM+Wharf、wbpf+uXDP、tutorial 4.2k star 里程碑 | 各 repo | 打包成主题合集文 | 第三批 |

**需要等外部时间点的资产**：清单与原因见私有战略库（research-pipeline）。

### 轨道 B：SEO 需求缺口（原 12 篇清单，资产轨覆盖后剩余部分）

按需求 × 胜率排序，前六：Securing OpenClaw（时效最强）、Falco vs Tetragon vs AgentSight、AI Agent Runtime Security 实践指南（FAQ 体）、Detect Prompt Injection at the Kernel、eBPF LSM→agent 教程桥接、MCP Security 内核监控。其后：sandbox 对比、无 SDK 观测对比、semantic flamegraph、bpftime 词权重回收。A3 已覆盖原清单 #1，第 12 篇（回应 ARMO）已发布在 main。

### 选题来源补充：GitHub issues（2026-07-16 已扫描六个 repo）

高频提问 = 已验证的搜索需求。首轮扫描结果，按频率 × 跨 repo 广度排序：

| 主题 | 证据（示例 issue） | 建议文章 | 体裁 |
|---|---|---|---|
| vmlinux.h / BTF 找不到（最高频） | tutorial#88/#118/#81/#106、eunomia-bpf#361 | Fixing "vmlinux BTF not found": BTF, CO-RE and BPF Tokens Explained | FAQ |
| 内核/发行版/架构兼容性（跨所有 repo，tutorial#94 是维护者认可的 docs gap） | tutorial#94/#86/#74、bpftime#357/#361/#396、ActPlane#20/#9 | eBPF Compatibility Guide: Kernels, Distros, Arch | 指南 |
| HTTP/TLS 抓包示例跑不起来 | tutorial#154(10 评论)/#162/#126/#107、bpftime#187 | Capturing HTTP/TLS Traffic with eBPF: Making sslsniff Work | 教程+FAQ |
| bpftime GPU/CUDA 使用（高评论簇，与 A2 呼应） | bpftime#496(14c)/#552(11c)/#543/#459 | 并入 A2 或独立 CUDA tracing 教程 | 教程 |
| bpftime 作为库使用 + 非 root 运行 | bpftime#421/#348/#435、#353/#564/#406 | Embedding bpftime as a Library, Running Non-Root | 指南 |

另有一批 issue 可以直接用现有文章回链解决（零成本宣传）：bpftime 原理类 → bpftime/userspace-ebpf/llvmbpf 文；ActPlane 政策类 → policy-plane 文；wasm-bpf 构建类 → wasm-bpf 文。回链动作可以随下一次 issue triage 顺手做。每月重扫一次。

| # | 工作标题 | 目标搜索意图 | 体裁 | 凭什么赢 |
|---|---|---|---|---|
| 1 | ~~Monitor Claude Code 教程~~ 改道：教程进 agentsight docs quickstart，blog 位由 A3 数据文承接 | how to monitor Claude Code | docs + 数据文 | 全网无免插桩方案，AgentSight 独占；搜索词由 docs 页 + 数据文合力覆盖 |
| 2 | Securing OpenClaw: eBPF Runtime Monitoring Guide | OpenClaw security | 教程+FAQ | 2026 最热 agent 安全话题，零 eBPF 角度竞品 |
| 3 | Falco vs Tetragon vs AgentSight for AI Agents | Falco vs Tetragon | 对比 | 热对比词，无人从 agent 角度框 |
| 4 | AI Agent Runtime Security with eBPF: A Practical Guide | AI agent runtime security | FAQ 指南 | 全是商业厂商，缺 OSS+论文背书 |
| 5 | Detect Prompt Injection at the Kernel with eBPF | prompt injection detection runtime | 教程+benchmark | 论文里有真实检测数据 |
| 6 | eBPF LSM Tutorial: Enforcing AI Agent Policy | eBPF LSM tutorial | 教程 | 已排 #4 的 LSM 教程权重桥接到 agent 垂直 |
| 7 | MCP Security: Monitoring Tool Calls with eBPF | MCP security monitoring | 教程+FAQ | arXiv/Wiz 占据，无内核角度 |
| 8 | Sandboxing AI Agents: gVisor vs microVM vs eBPF | AI agent sandbox security | 对比+benchmark | ActPlane 1.9-8.4% overhead 实测数据 |
| 9 | AI Agent Observability Without SDKs | AI agent observability tool | 对比 | Langfuse/LangSmith 做不了免插桩 |
| 10 | Semantic Flamegraphs for AI Agents with eBPF | AI agent profiling | 教程 | agentpprof 独有 feature，无人竞争 |
| 11 | bpftime: Userspace eBPF for AI Agent Observability | bpftime / userspace eBPF | 教程 | 把 bpftime 词的所有权从 GitHub/arXiv 拉回站内 |
| 12 | What 2,116 CLAUDE.md Rules Reveal About AI Agent Safety（slug 不变：ebpf-ai-agent-policy-enforcement） | AI agent safety / CLAUDE.md rules | 论文数据文 | 2026-07-16 重写：去 ARMO 依附、按 arXiv 版术语（policies 而非 directives）、以实证数据为主体、配论文图 |

草稿统一放 `draft/blog/`，成稿移入 `docs/blog/posts/`（带 slug + date），再进转载队列。

## 四、发布节奏计划（为什么不能一次全上，怎么分批）

一次性上 10+ 篇的问题：Google 对突然的发布脉冲不会立刻给权重，新站点/低频站点的抓取配额有限；一次全上等于所有文章在同一个抓取周期里互相竞争，且后续几个月没有"持续活跃"信号。SEO 回报来自稳定节奏而不是总量。

**发布管线（draft → 审核 → PR → 站内 → 转载）：**

1. 草稿在 `draft/blog/` 生成（claude-opus-4-6[1m] + 两个写作 skill），双语一对。
2. 人工审核草稿：技术事实、数字、口径（尤其论文是否在审）。
3. 通过后移入 `docs/blog/posts/`（定 slug + date + description），从 main 拉分支走非 draft PR（CLAUDE.md 工作流），跑 `npm run verify`。
4. PR 合并即站内发布；1-2 周后加入 `.github/publisher/posts_queue.txt` 转载（前提：canonical 改造已合并）。

**节奏设计（每周 1 篇站内 + 滞后转载）：**

- **站内发布**：每周固定发 1 篇（双语一对），资产轨（A1-A3）先行，SEO 轨穿插。OpenClaw 这篇有时效性，最迟两周内排上。
- **转载（dev.to/Medium）滞后 1-2 周**：等 Google 已经索引了 eunomia.dev 原文、且转载带 canonical 之后再进 `posts_queue.txt`。现有 CI（`.github/workflows/publish-posts.yml`，每周二 cron，每次 2 篇）刚好匹配这个节奏，不用改频率。**前置条件：批次 A 第 2 项 canonical 改造合并前，队列保持为空，一篇都不要转载。**
- **队列用法**：每周站内发布后，把上上周的文章加进 `posts_queue.txt`（`{"path": "docs/blog/posts/xxx.md", "tags": [...]}`），保持队列里始终 1-2 篇。
- **度量**：上线 Google Search Console（如果还没验证），每两周看一次目标词排名和抓取统计；每篇文章上线 4 周后复查目标词，不进前 20 的回炉改 title/description。

12 篇按此节奏约一个季度发完，正好覆盖一个完整的 SEO 生效周期。

## 五、网站整体观感改进

1. **社交卡片**（= 批次 A 第 1 项）：这是"观感"对外的第一面，分享无图的观感损失比站内任何样式问题都大。
2. **blog 列表页**：51 篇平铺列表改为"置顶 3-4 篇旗舰 + 按主题分组"，配合 tag 落地页；每篇卡片显示 description 摘要而不是只有标题和日期。
3. **首页叙事**：当前首页偏项目罗列，建议头部直接给 "AI Agent Observability & Enforcement, built on eBPF" 的定位陈述 + AgentSight/ActPlane/bpftime 三卡片 + 一张 semantic flamegraph 截图（现成资产，视觉冲击强）。文案改动走 mkdocs.yaml/内容层，不动路由。
4. **文章页**：heading autolink（方便分享定位）、代码块 copy 按钮（如果还没有）、文末统一的 "Try AgentSight / Star on GitHub" CTA 组件。
5. **agentsight.us live demo**：从首页和 AgentSight 相关文章显眼位置链过去，live demo 是比截图强一个量级的观感资产。

## 六、整体宣传计划

**双引擎目标（2026-07-16 定）**：教育/分享内容与产品叙事内容按 7:3 到 8:2 配比。教育是目标本身和复利资产（受众、信任、术语所有权、贡献者），不只是转化手段；产品叙事站在它上面做转化。指标体系与商业目标见私有战略库。

**平台矩阵（分层进入，一源多发，正本永远在本站）**：已有（本站/GitHub/X/dev.to+Medium 队列）→ 第一扩展（知乎、LinkedIn，半自动）→ 场合性（HN/Reddit/lobste.rs，永远人工，只投旗舰）→ 观察（掘金、公众号）→ 暂缓（小红书、B站/YouTube，等图卡/视频产能）。每个新平台小样本试 4-6 周看数据定去留；转载状态记录在 `.github/publisher/ledger.md`。

1. **先修 canonical 再谈一切**（历史转载手动补；publish 链路搁置期间新转载一律平台后台手动设 canonical），否则宣传做得越多，权重漏得越多。
2. **每篇旗舰文的发布 checklist**：站内上线 → X/Twitter（@eaborai）+ 个人账号双发 → 有数据/有反直觉结论的投 Hacker News（美东工作日上午）和 lobste.rs → r/eBPF、r/netsec、r/LocalLLaMA 按主题选发 → 1-2 周后 dev.to/Medium 转载（带 canonical）→ 中文版同步发掘金/知乎专栏。
3. **借力节点**：论文 accept/camera-ready、KubeCon 2026（draft 里已有材料）、eBPF Summit 等 talk 是天然的流量脉冲，对应周的 blog 排期让位给配套文章（talk 的文字版是最容易的高质量内容）。
4. **社区背书**：ebpf.io 的 landscape/newsletter、ISovalent/Cilium 社区 newsletter 接受项目投稿；awesome-agent-runtime-security 已收录 AgentSight/ActPlane（eBPF 组），保持条目信息最新；学术侧 arXiv + Google Scholar 主页 + 论文致谢页统一指回 eunomia.dev。
5. **GitHub 门面**：agentsight/ActPlane 两个 repo 的 README 首屏加 eunomia.dev 对应文章链接（GitHub 是现在排名最好的自有资产，把它的流量导回站内）。
6. **一致性**：所有对外物料（论文、README、转载、talk slides）统一用 eunomia.dev 的 URL 作为规范链接，不要再让 arXiv/GitHub 当事实上的主页。
7. **内容类型 × 渠道矩阵**（第 2 条 checklist 是旗舰文流程，其他类型按此分流）：

| 内容类型 | 主渠道 | 次渠道 | 纪律 |
|---|---|---|---|
| 论文/数据文（旗舰） | HN + X 双账号 + 站内 | 1-2 周后 dev.to/Medium（手动 canonical）、知乎/掘金 | HN 每月最多 1-2 投，只投有反直觉结论的 |
| 版本发布 | GitHub Release + X | ebpf.io newsletter 投稿、r/eBPF | Release notes 链接站内文 |
| 教程/FAQ（docs） | 站内 docs + Google 自然流量 | 相关 issue 回链（零成本、精准） | 不投社交，靠搜索长尾 |
| 客户音域（solution brief/白皮书） | 直发目标客户 + LinkedIn | 客户对话附件 | 不做公开推广，服务转化 |
| talk 文字版 | 站内 + 会议官方渠道 | X 线程 | 会后一周内发 |
| 中文内容 | 知乎专栏 + 公众号（待定） | 掘金 | 与英文同结构，独立排期 |

## 七、风险与边界

- 转载 canonical 是平台方功能，Medium 导入工具偶尔丢 canonical，发完抽查页面源码。
- 不改任何既有 URL/slug/路由（CLAUDE.md 红线）；tag 页是纯新增。
- OpenClaw/竞品动态变化快，第 2、3 篇动笔前重查一次 SERP。
- 发布队列的 PUBLISH_PASSWORD/Vercel 服务是外部依赖，canonical 改造要先在 draft_only 模式验证一轮。
