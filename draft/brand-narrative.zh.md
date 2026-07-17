# eunomia-bpf 品牌叙事（草案，2026-07-16）

本文档是全站长文（blog、docs 落地页、README、转载、演讲）的对齐基准。所有对外内容的定位措辞、术语、宣称边界以此为准；修改品牌口径先改这里。配套执行见 `draft/seo-geo-plan.zh.md`（发布节奏、选题轨道）与 `.claude/skills/seo-geo/SKILL.md`（写法规则）。

## 风格（先于一切内容决策）

**把证据放在形容词的位置上。** 五个特征：

1. **一切都被测量过**：数字带条件和出处（75.8% DCR、1.9% 开销、2116 条 statement），不写 "significantly"。
2. **给机制，不给形容词**：说得出 hook 名字才算说了机制（BPF-LSM pre-operation、byte-pattern 定位 BoringSSL、bitmask 按位 OR）。
3. **公开说边界**：limitations 是信用（"on covered hooks"、106 起漏 28 起、over-tainting 缓解）。这是与营销型厂商的气质分界线。
4. **老师本能**：品牌靠教程教出来的（tutorial 4.2k star），文章倾向教会读者而非说服读者。
5. **系统层的固执**：所有项目一个立场，真相在下面那层（below the sandbox and below HTTP）。

参照系：Brendan Gregg、Cilium/Isovalent 技术文、fly.io/tailscale 工程博客、系统论文 limitations 节。反面清单：blazing fast、revolutionize、emoji 轰炸、渐变大字报 SaaS 风。

**"专业" = 把这个风格执行得更严，不是换企业腔。** 我们的客户是做安全评审的平台/安全工程师：对他们，带 hook 名的架构图 + 诚实的 limitations 比 marketing PDF 更专业。企业腔会毁掉差异化。视觉气质同理：终端/内核美学（monospace、架构图、火焰图、真实 trace 截图），不是 SaaS 渐变风。

## 一句话

eunomia-bpf 是 AI agent 时代的系统级观测与执行平面：用 eBPF 在内核层看见 agent 的全部行为，并把写下的规则变成可执行的约束，零插桩、不改一行 agent 代码。

英文版（对外统一用语）：The system-level observability and enforcement plane for AI agents: see everything an agent does and enforce the rules you already wrote, with zero instrumentation, below the sandbox and below HTTP.

## 三支柱（与 seo-geo-plan §1.5 对齐）

1. **Agent Observability & Harness（主叙事）**：AgentSight（零插桩观测）、ActPlane（内核级 policy enforcement）、agentpprof（semantic flamegraph）。差异化：内核级真相，SDK/MCP 代理和 sandbox 都看不到的 subprocess、文件、非 HTTP 系统调用，我们看得到、管得住。
2. **eBPF 教育与生态基本盘（信任来源）**：bpf-developer-tutorial（4.2k star）、eunomia-bpf、bpftime。这是流量和社区信任的底盘，也是"我们真的懂内核"的证明。
3. **系统研究前沿（可信度来源）**：GPU（bpftime GPU、gpu_ext、gPerf）、调度（SchedCP）、OS 论文。论文是宣称的背书，每个关键数字都能指到 arXiv。

## 证据点（宣称必须挂证据）

- 实证：ActPlane 论文 2116 条 statement 研究、DCR 75.8%、开销 1.9%-8.4%（arXiv:2606.25189）
- 独占能力：闭源二进制（Claude Code）的 TLS 级追踪；BPF-LSM 的 pre-operation 拦截 + semantic feedback；eBPF 进 GPU kernel
- 社区：tutorial 4.2k star、awesome-agent-runtime-security 榜单 eBPF 组收录 AgentSight 与 ActPlane

## 语言资产（GEO 术语，全站统一写法）

zero-instrumentation / 零插桩；agent harness；policy（论文口径，不用 directive）；system-observable；cross-event；semantic feedback；temporal trust boundary；"below the sandbox and below HTTP"；kernel-level truth。同一概念全站一个写法，中文里成熟术语保留英文。

## 宣称边界（禁区）

- 不自称 sandbox / 算力 / SDK runtime 厂商；不宣称"我们是 execution layer"（做 ride-along 定义文，依据见私有战略库）。
- 不用未测量的数字；带范围的宣称保留限定语（"on covered hooks"、"up to"）。
- 在审论文不称录用；preprint 口径明示。
- 不以单一厂商文章为立论框架（seo-geo skill 第三方框架卫生条款）。

## 两种音域（2026-07-16 补充：面向客户要更专业）

一个事实底座（本文档的证据点与术语表），两种对外音域，不混用：

- **开发者音域**：blog、教程、论文解读。第一人称、有场景、有代码，建立技术信任。现有内容全部属于此类，周更。
- **客户音域**（新增轨道）：面向评估者与采购者的专业呈现。语气精确克制，每个宣称带引用与限定语，结构服务快速评估（问题、盲区、架构、证据、部署要求、FAQ），可导出 PDF。挂在站内 `/products/` IA 之下（只新增页面，不动任何现有 URL，配置进 mkdocs.yaml）。

客户音域的内容清单与节奏：

| 内容 | 说明 | 节奏 |
|---|---|---|
| Solution brief x3 | 对应三个 ICP：平台团队 agent 审计管控（ActPlane）、agent 基建嵌入式观测（AgentSight）、企业合规留痕；2-3 页（兼作客户对话敲门材料，客户策略见私有战略库） | 随产品版本更新 |
| 技术白皮书 x1 | Runtime Observability and Enforcement for AI Agents：整合升格三篇支柱 blog（runtime-security 总览、ActPlane 机制文、2116 研究文） | 季度更新 |
| 评估者 FAQ 页 | 安全评审问题：所需权限、数据流向、内核版本要求、开销上限、故障模式、TCB 边界 | 随产品版本更新 |
| Case study | 有 design partner 后才写，不编造 | 每个 partner 一篇 |

## 长文管理与推送节奏

- 唯一正本：`docs/blog/posts/`（中英对）；草稿与策略文档在 `draft/`；选题来自 seo-geo-plan 轨道 A/B 与用户对话高频问题。
- 体裁边界：blog 只装论点/数据/设计决策/战报，教程进产品 docs（体裁原则见 seo-geo-plan §三）。
- 定期推送：`publish-posts.yml` 每周二 cron 从 `posts_queue.txt` 转载 dev.to/Medium；恢复入队的前提是历史转载 canonical 手动补齐且新转载发布时在平台后台设置 canonical（media-publisher 暂不改）。站内节奏 1 篇/周。
- 品牌一致性检查：新长文发布前对照本文档的一句话、术语表和禁区各过一遍。

## 待定（需要 owner 拍板）

- 中文品牌一句话是否需要更口语的变体（用于知乎/公众号等渠道）
- newsletter/邮件订阅是否纳入"定期推送"（目前只有 RSS + 转载）
- 三支柱在首页 IA 上的呈现顺序
