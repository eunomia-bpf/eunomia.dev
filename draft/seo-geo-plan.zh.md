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

## 二、站内修复计划（分三批，每批一个 PR）

**批次 A（本周，最高杠杆，不新增页面）**
1. og:image 换成构建期生成的 PNG（至少一张站级默认图；有余力按 section 出 3-4 张变体）。
2. publish 链路加 canonical：`publish.py` payload 增加 `canonical_url` 字段（由 post 路径推导 eunomia.dev URL），media-publisher Vercel 服务把它转发为 dev.to 的 `canonical_url` 和 Medium 的 `canonicalUrl`。
3. **手动补历史转载的 canonical**（不用等代码）：dev.to/Medium 后台把 AgentSight 等已转载文章逐篇设 canonical 指回 eunomia.dev。这是零成本、收益最大的单个动作。
4. sitemap 剔除 noindex 的 `/blogs/*`。

**批次 B（两周内，内容基建）**
5. blog tag 落地页：先手工建 3 个主题聚合页（AI Agent Security / eBPF Tutorials / GPU & Systems），IA 配置进 mkdocs.yaml，不动任何既有 URL。
6. llms.txt/llms-full.txt 生成化：并入 `generate:content-artifacts`，llms-full.txt 嵌入各页正文或高质量摘要，CI 加校验。
7. 给 8-10 篇高价值旧文补手写 description（AgentSight 论文文最优先，它现在连 description 都没有）；修掉 H1 里的 `**` 和英文文件的中文标题。

**批次 C（一个月内，锦上添花）**
8. FAQPage JSON-LD（教程和带 FAQ 段的 blog）、HowTo（教程）。
9. 图片 alt 校验进 CI、lazy loading、heading autolink、redirect 桩加 noindex。

## 三、Blog 写作计划（12 篇，按需求 × 胜率排序）

写作规范：全部走 `tech-blog-writer`（新写）+ `blog-writing-style`（审校）两个 skill；写作一律用 claude-opus-4-6[1m]（Agent 工具钉不了版本，用 `.claude/agents/prose-writer.md` 或无头 CLI `claude -p --model 'claude-opus-4-6[1m]'`）；中英双语同结构；description 150-160 字符；title ≤60 字符关键词前置；每篇 2-3 条站内互链 + 指向 GitHub repo。

| # | 工作标题 | 目标搜索意图 | 体裁 | 凭什么赢 |
|---|---|---|---|---|
| 1 | Monitor Claude Code with eBPF: Full Runtime Visibility | how to monitor Claude Code | 教程 | 全网无免插桩方案，AgentSight 独占 |
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
| 12 | （已在写）eBPF AI Agent Enforcement Needs a Contextual Policy Plane | eBPF AI agent enforcement | 反驳/解读 | 正面回应 ARMO，本分支已完成初稿 |

草稿统一放 `draft/blog/`，成稿移入 `docs/blog/posts/`（带 slug + date），再进转载队列。

## 四、发布节奏计划（为什么不能一次全上，怎么分批）

一次性上 10+ 篇的问题：Google 对突然的发布脉冲不会立刻给权重，新站点/低频站点的抓取配额有限；一次全上等于所有文章在同一个抓取周期里互相竞争，且后续几个月没有"持续活跃"信号。SEO 回报来自稳定节奏而不是总量。

**节奏设计（每周 1 篇站内 + 滞后转载）：**

- **站内发布**：每周固定发 1 篇（双语一对），按第三节的优先级顺序。第 1、2 篇（Claude Code 监控、OpenClaw）在批次 A 修复合并后立刻上，OpenClaw 这篇有时效性，最迟两周内。
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

1. **先修 canonical 再谈一切**（历史转载手动补 + publish 链路改造），否则宣传做得越多，权重漏得越多。
2. **每篇旗舰文的发布 checklist**：站内上线 → X/Twitter（@eaborai）+ 个人账号双发 → 有数据/有反直觉结论的投 Hacker News（美东工作日上午）和 lobste.rs → r/eBPF、r/netsec、r/LocalLLaMA 按主题选发 → 1-2 周后 dev.to/Medium 转载（带 canonical）→ 中文版同步发掘金/知乎专栏。
3. **借力节点**：论文 accept/camera-ready、KubeCon 2026（draft 里已有材料）、eBPF Summit 等 talk 是天然的流量脉冲，对应周的 blog 排期让位给配套文章（talk 的文字版是最容易的高质量内容）。
4. **社区背书**：ARMO 文章已把 AgentSight 当研究引用，回应文（第 12 篇）发布后可以礼貌地 @ 他们；ebpf.io 的 landscape/newsletter、ISovalent/Cilium 社区 newsletter 都接受项目投稿。
5. **GitHub 门面**：agentsight/ActPlane 两个 repo 的 README 首屏加 eunomia.dev 对应文章链接（GitHub 是现在排名最好的自有资产，把它的流量导回站内）。
6. **一致性**：所有对外物料（论文、README、转载、talk slides）统一用 eunomia.dev 的 URL 作为规范链接，不要再让 arXiv/GitHub 当事实上的主页。

## 七、风险与边界

- 转载 canonical 是平台方功能，Medium 导入工具偶尔丢 canonical，发完抽查页面源码。
- 不改任何既有 URL/slug/路由（CLAUDE.md 红线）；tag 页是纯新增。
- OpenClaw/竞品动态变化快，第 2、3 篇动笔前重查一次 SERP。
- 发布队列的 PUBLISH_PASSWORD/Vercel 服务是外部依赖，canonical 改造要先在 draft_only 模式验证一轮。
